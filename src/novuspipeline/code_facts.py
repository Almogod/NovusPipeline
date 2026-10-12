"""
code_facts.py — NovusPipeline: static facts extracted from one Python module

Pure AST analysis (no workspace knowledge) feeding the codebase index:

  imports      static, lazy (inside functions) and dynamic imports
               (`importlib.import_module("x")`, `__import__("x")`, relative forms
               with `package=__name__`, `runpy.run_module`), plus the attributes
               used from each imported module (`mod.attr`, `getattr(mod, "attr")`)
  unresolved   dynamic imports whose target is computed at runtime; an f-string
               with a constant prefix (`f"plugins.{name}"`) keeps that prefix so
               every module under it can be treated as reachable
  path loads   `spec_from_file_location` / `SourceFileLoader` / `runpy.run_path`
               with a literal path
  defs / refs  top-level definitions and every identifier the module mentions
               (names, attributes, identifier-like strings), for dead-code checks
  exports      a literal `__all__`, for resolving `from x import *`
  shapes       rename-insensitive structural hash of each function (clones)
"""

import ast
import copy
import hashlib
import re
from typing import Any

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,59}$")
_PATH_LOADERS = {"spec_from_file_location": 1, "SourceFileLoader": 1, "run_path": 0}


def dotted(node: ast.AST) -> str | None:
    """`a.b.c` for Name/Attribute chains, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _lazy_node_ids(tree: ast.Module) -> set[int]:
    """ids of nodes executed only when a function runs (not at import time)."""
    lazy: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for child in ast.walk(node):
                if child is not node:
                    lazy.add(id(child))
    return lazy


def _resolve_relative(name: str, base_parts: list[str]) -> str:
    level = len(name) - len(name.lstrip("."))
    rest = name[level:]
    base = base_parts[:len(base_parts) - (level - 1)] if level > 1 else base_parts
    return ".".join(base + ([rest] if rest else []))


def extract_imports(tree: ast.Module, module: str, is_package: bool
                    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(facts, unresolved dynamic imports, literal path loads). See module docstring."""
    package = module.split(".") if is_package else module.split(".")[:-1]
    lazy_ids = _lazy_node_ids(tree)
    facts: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    path_loads: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    # `from pkg import mod` may bind a submodule or a plain symbol; which one is
    # only known against the module map, so record it as a candidate ("exact").
    from_aliases: dict[str, str] = {}
    import_module_names = {"importlib.import_module", "__import__"}
    run_module_names = {"runpy.run_module"}

    for node in ast.walk(tree):
        lazy = id(node) in lazy_ids
        if isinstance(node, ast.Import):
            for a in node.names:
                facts.append({"module": a.name, "names": [], "line": node.lineno, "lazy": lazy, "dynamic": False})
                # `import a.b` binds `a`, so only plain or aliased imports map a local name to exactly this module.
                if a.asname or "." not in a.name:
                    aliases[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom):
            target = _resolve_relative("." * node.level + (node.module or ""), package) if node.level \
                else (node.module or "")
            if not target:
                continue
            names = ["*"] if any(a.name == "*" for a in node.names) else [a.name for a in node.names]
            facts.append({"module": target, "names": names, "line": node.lineno, "lazy": lazy, "dynamic": False})
            for a in node.names:
                if a.name == "*":
                    continue
                from_aliases[a.asname or a.name] = f"{target}.{a.name}"
                if target == "importlib" and a.name == "import_module":
                    import_module_names.add(a.asname or a.name)
                if target == "runpy" and a.name == "run_module":
                    run_module_names.add(a.asname or a.name)

    for local, mod in list(aliases.items()):
        if mod == "importlib":
            import_module_names.add(f"{local}.import_module")
        if mod == "runpy":
            run_module_names.add(f"{local}.run_module")

    def dynamic_target(call: ast.Call) -> tuple[str | None, dict[str, Any] | None]:
        """(resolved module, None) or (None, unresolved record) for an import_module-style call."""
        arg = call.args[0]
        pkg = call.args[1] if len(call.args) > 1 else next((k.value for k in call.keywords if k.arg == "package"), None)
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            name = arg.value
            if not name.startswith("."):
                return name, None
            if isinstance(pkg, ast.Constant) and isinstance(pkg.value, str):
                return _resolve_relative(name, pkg.value.split(".")), None
            if isinstance(pkg, ast.Name) and pkg.id in ("__name__", "__package__"):
                return _resolve_relative(name, module.split(".") if pkg.id == "__name__" else package), None
            return None, {"line": call.lineno, "expr": ast.unparse(call)[:100], "prefix": None}
        prefix = None
        if isinstance(arg, ast.JoinedStr) and arg.values and isinstance(arg.values[0], ast.Constant):
            prefix = str(arg.values[0].value).rstrip(".") or None
        return None, {"line": call.lineno, "expr": ast.unparse(call)[:100], "prefix": prefix}

    # `m = importlib.import_module("x")` binds `m` to module x, like `import x as m`.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Call) and node.value.args \
                and (dotted(node.value.func) or "") in import_module_names:
            resolved, _ = dynamic_target(node.value)
            if resolved:
                aliases[node.targets[0].id] = resolved

    used: dict[str, set[str]] = {}
    candidate_used: dict[str, set[str]] = {}

    def note_use(base: str, attr: str) -> None:
        if base in aliases:
            used.setdefault(aliases[base], set()).add(attr)
        elif base in from_aliases:
            candidate_used.setdefault(from_aliases[base], set()).add(attr)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            note_use(node.value.id, node.attr)
        elif isinstance(node, ast.Call):
            target = dotted(node.func) or ""
            short = target.rsplit(".", 1)[-1]
            lazy = id(node) in lazy_ids
            if target == "getattr" and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) \
                    and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                note_use(node.args[0].id, node.args[1].value)
            elif target in import_module_names or target in run_module_names:
                if not node.args:
                    continue
                resolved, problem = dynamic_target(node)
                if resolved:
                    facts.append({"module": resolved, "names": [], "line": node.lineno, "lazy": lazy, "dynamic": True})
                else:
                    unresolved.append(problem)
            elif short in _PATH_LOADERS:
                idx = _PATH_LOADERS[short]
                if len(node.args) > idx and isinstance(node.args[idx], ast.Constant) \
                        and isinstance(node.args[idx].value, str):
                    path_loads.append({"line": node.lineno, "path": node.args[idx].value})
                elif short == "run_path" or "spec_from_file_location" in target or "SourceFileLoader" in target:
                    unresolved.append({"line": node.lineno, "expr": ast.unparse(node)[:100], "prefix": None})

    for fact in facts:
        if not fact["names"] and fact["module"] in used:
            fact["names"] = sorted(used[fact["module"]])
    facts.extend({"module": m, "names": sorted(attrs), "line": 0, "lazy": True, "dynamic": False, "exact": True}
                 for m, attrs in candidate_used.items())
    return facts, unresolved, path_loads


def extract_defs(tree: ast.Module) -> list[dict[str, Any]]:
    return [{"name": n.name, "line": n.lineno, "kind": "class" if isinstance(n, ast.ClassDef) else "function",
             "decorated": bool(n.decorator_list)}
            for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def extract_refs(tree: ast.Module) -> list[str]:
    """Every identifier the module mentions; identifier-like strings count (registries, getattr)."""
    refs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Store):
            refs.add(node.id)
        elif isinstance(node, ast.Attribute):
            refs.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and _IDENTIFIER.match(node.value):
            refs.add(node.value)
        elif isinstance(node, ast.alias):
            refs.add((node.asname or node.name).split(".")[0])
            refs.add(node.name.split(".")[-1])
    return sorted(refs)


def extract_exports(tree: ast.Module) -> list[str] | None:
    """A literal `__all__`, else None."""
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets) \
                    and isinstance(node.value, (ast.List, ast.Tuple)):
                return [e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return None


class _Normalize(ast.NodeTransformer):
    """Drops names, annotations, decorators and docstrings: what is left is the logic's shape."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.Name(id="_", ctx=node.ctx)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.arg(arg="_", annotation=None)

    def _visit_def(self, node):
        node.name, node.returns, node.decorator_list = "_", None, []
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
        self.generic_visit(node)
        return node

    visit_FunctionDef = _visit_def
    visit_AsyncFunctionDef = _visit_def


def function_shape(node: ast.AST) -> tuple[str, int]:
    """(rename-insensitive structural hash, AST node count) of a function definition."""
    size = sum(1 for _ in ast.walk(node))
    normalized = _Normalize().visit(copy.deepcopy(node))
    return hashlib.sha1(ast.dump(normalized, include_attributes=False).encode()).hexdigest()[:16], size
