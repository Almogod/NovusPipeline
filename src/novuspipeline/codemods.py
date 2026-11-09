"""
codemods.py — NovusPipeline: scope-aware, formatting-preserving migrations (libcst)

The regex rewrites in `modernizer.py` cover a few classic smells. This module is
the extensible engine for *library and language migrations*: each `Migration`
detects uses of removed or deprecated APIs and, where a behavior-identical
rewrite exists, fixes them while keeping comments, whitespace and formatting.

A migration is a small `libcst` transformer:

    class MyMigration(Migration):
        id = "MIG-XYZ"; title = "..."; severity = "HIGH"; min_python = "3.12"
        def leave_Attribute(self, original, updated): ...
            self.hit(original, fixable=True)       # records a finding
            return new_node if self.fix else updated

`self.hit()` is called for every detection; the transformer only rewrites when
`self.fix` is true and the hit was marked fixable. Migrations that cannot be
fixed safely stay detection-only (they still appear in audits and CI scans).

Add your own by subclassing `Migration`; it registers itself.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, ClassVar

try:
    import libcst as cst
    from libcst.metadata import MetadataWrapper, PositionProvider
except ImportError:  # libcst is a core dependency; stay importable if it is missing
    cst = None  # type: ignore[assignment]

REGISTRY: dict[str, type[Migration]] = {}


@dataclass
class Hit:
    line: int
    fixable: bool
    detail: str = ""


if cst is not None:

    class Migration(cst.CSTTransformer):
        """Base class: subclass, set the class attributes and override `leave_*` / `visit_*`."""

        METADATA_DEPENDENCIES = (PositionProvider,)

        id: ClassVar[str] = ""
        title: ClassVar[str] = ""
        severity: ClassVar[str] = "MEDIUM"
        min_python: ClassVar[str | None] = None   # earliest Python version in which a covered API is removed
        guideline: ClassVar[str] = ""
        rag_query: ClassVar[str] = "Python Library Modernization"
        pack: ClassVar[str] = "stdlib"            # group for --pack selection (stdlib, pydantic, sqlalchemy, ...)
        fixable: ClassVar[bool] = True            # False: detection only
        trigger: ClassVar[str] = ""               # regex that must match the source for any hit (speed)

        @classmethod
        def relevant(cls, code: str) -> bool:
            """Cheap pre-check before parsing: can this migration find anything in `code`?"""
            if not cls.trigger:
                return True
            if "_trigger_re" not in cls.__dict__:
                cls._trigger_re = re.compile(cls.trigger, re.MULTILINE)
            return cls._trigger_re.search(code) is not None

        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)
            if cls.id:
                REGISTRY[cls.id] = cls

        def __init__(self, fix: bool = False) -> None:
            super().__init__()
            self.fix = fix
            self.hits: list[Hit] = []

        def hit(self, node: cst.CSTNode, fixable: bool = True, detail: str = "") -> None:
            pos = self.get_metadata(PositionProvider, node)
            self.hits.append(Hit(pos.start.line, fixable, detail))

        # Compatibility shims keep old APIs on purpose (`if sys.version_info < (3,):`, `if
        # hasattr(base64, "decodebytes"):`, `except AttributeError:` fallbacks). Hits inside
        # them are dropped and the subtree is left exactly as written.
        def _guard_open(self) -> None:
            self._guards = [*getattr(self, "_guards", []), len(self.hits)]

        def _guard_close(self, original: cst.CSTNode) -> cst.CSTNode:
            mark = self._guards.pop()
            del self.hits[mark:]
            return original

        def visit_If(self, node: cst.If) -> None:
            if _is_compat_test(node.test):
                self._guard_open()

        def leave_If(self, original: cst.If, updated: cst.If) -> cst.BaseStatement:
            return self._guard_close(original) if _is_compat_test(original.test) else updated

        def visit_ExceptHandler(self, node: cst.ExceptHandler) -> None:
            if _catches(node, _FALLBACK_ERRORS):
                self._guard_open()

        def leave_ExceptHandler(self, original: cst.ExceptHandler,
                                updated: cst.ExceptHandler) -> cst.ExceptHandler:
            return self._guard_close(original) if _catches(original, _FALLBACK_ERRORS) else updated

        def finish(self, module: cst.Module) -> cst.Module:
            """Hook for whole-module edits after the visit (e.g. adding an import)."""
            return module

else:  # pragma: no cover
    class Migration:  # type: ignore[no-redef]
        id = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dotted(node: Any) -> str | None:
    """'a.b.c' for a Name/Attribute chain, else None."""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr.value}" if base else None
    return None


_COMPAT_TEST = re.compile(
    r"\b(?:version_info|hexversion|PY2|PY3|PY3K|PY\d\d|IS_PY\d\w*|PYTHON\d\w*|hasattr|sys\.modules|"
    r"VERSION|__version__|\w*_V[12])\b")
_FALLBACK_ERRORS = {"ImportError", "ModuleNotFoundError", "AttributeError", "NameError"}
_EMPTY = cst.Module([]) if cst is not None else None


def _is_compat_test(test: cst.BaseExpression) -> bool:
    return _COMPAT_TEST.search(_EMPTY.code_for_node(test)) is not None


def _catches(handler: cst.ExceptHandler, names: set[str]) -> bool:
    if handler.type is None:
        return False
    types = [e.value for e in handler.type.elements] if isinstance(handler.type, cst.Tuple) else [handler.type]
    return any((_dotted(t) or "").split(".")[-1] in names for t in types)


def _top_level_imports(module: cst.Module) -> set[str]:
    """Names bound by plain `import x` / `import x.y` statements directly in the module body."""
    bound: set[str] = set()
    for stmt in module.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for small in stmt.body:
                if isinstance(small, cst.Import):
                    for alias in small.names:
                        if alias.asname is None:
                            dotted = _dotted(alias.name)
                            if dotted:
                                bound.add(dotted)
    return bound


def _insert_after_import(module: cst.Module, existing: str, new_stmt: str) -> cst.Module:
    """Adds `new_stmt` right after the top-level `import <existing>` line; no-op if absent."""
    body = list(module.body)
    for i, stmt in enumerate(body):
        if isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) == 1 \
                and isinstance(stmt.body[0], cst.Import):
            names = [a for a in stmt.body[0].names if a.asname is None]
            if any(_dotted(a.name) == existing for a in names):
                body.insert(i + 1, cst.parse_statement(new_stmt).with_changes(leading_lines=()))
                return module.with_changes(body=body)
    return module


def _dotted_trigger(dotted_names: list[str], methods: list[str] = ()) -> re.Pattern:
    """
    Regex that matches source which may use one of `dotted_names`: `mod.attr` as written, or
    `attr` imported with `from mod import ... attr` (also parenthesized), or `.method`.
    """
    parts = set()
    for d in dotted_names:
        mod, _, attr = d.rpartition(".")
        parts.add(re.escape(".".join(d.split(".")[-2:])) + r"\b")
        if mod:
            parts.add(r"^\s*from\s+" + re.escape(mod) + r"\s+import\s*(?:\([^)]*|[^\n]*)\b" + re.escape(attr) + r"\b")
        parts.add(r"^\s*import\s+[^\n]*\b" + re.escape(d) + r"\b")
    parts |= {r"\." + re.escape(m) + r"\b" for m in methods}
    return re.compile("|".join(sorted(parts)), re.MULTILINE)


def _import_alias_names(node: cst.ImportFrom) -> list[cst.ImportAlias] | None:
    return None if isinstance(node.names, cst.ImportStar) else list(node.names)


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------

_UNITTEST_ALIASES = {
    "assertEquals": "assertEqual", "assertNotEquals": "assertNotEqual", "assert_": "assertTrue",
    "assertAlmostEquals": "assertAlmostEqual", "assertNotAlmostEquals": "assertNotAlmostEqual",
    "assertRegexpMatches": "assertRegex", "assertNotRegexpMatches": "assertNotRegex",
    "assertRaisesRegexp": "assertRaisesRegex", "assertItemsEqual": "assertCountEqual",
    "failUnless": "assertTrue", "failIf": "assertFalse", "failUnlessEqual": "assertEqual",
    "failIfEqual": "assertNotEqual", "failUnlessAlmostEqual": "assertAlmostEqual",
    "failIfAlmostEqual": "assertNotAlmostEqual", "failUnlessRaises": "assertRaises",
}


class UnittestAliases(Migration):
    id = "MIG-UNITTEST-ALIASES"
    pack = "unittest"
    trigger = r"\bself\.(?:" + "|".join(_UNITTEST_ALIASES) + r")\b"
    title = "Deprecated unittest assertion aliases (removed in Python 3.12)"
    severity = "HIGH"
    min_python = "3.12"
    guideline = ("`assertEquals`, `assertRegexpMatches`, `failUnless` and friends were deprecated aliases and "
                 "are removed in Python 3.12. The replacements take the same arguments.")

    def __init__(self, fix: bool = False) -> None:
        super().__init__(fix)
        self._in_test_class: list[bool] = []
        self._defined: list[set[str]] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        is_test = any((_dotted(b.value) or "").split(".")[-1].endswith("TestCase") for b in node.bases)
        defined = {s.name.value for s in node.body.body if isinstance(s, cst.FunctionDef)}
        self._in_test_class.append(is_test)
        self._defined.append(defined)

    def leave_ClassDef(self, original: cst.ClassDef, updated: cst.ClassDef) -> cst.ClassDef:
        self._in_test_class.pop()
        self._defined.pop()
        return updated

    def leave_Attribute(self, original: cst.Attribute, updated: cst.Attribute) -> cst.BaseExpression:
        name = original.attr.value
        if name not in _UNITTEST_ALIASES or not self._in_test_class or not self._in_test_class[-1]:
            return updated
        if not (isinstance(original.value, cst.Name) and original.value.value == "self"):
            return updated
        if name in self._defined[-1]:  # the class defines its own method with that name
            return updated
        self.hit(original, detail=f"{name} -> {_UNITTEST_ALIASES[name]}")
        if self.fix:
            return updated.with_changes(attr=cst.Name(_UNITTEST_ALIASES[name]))
        return updated


_COLLECTIONS_ABC = {
    "Awaitable", "Coroutine", "AsyncIterable", "AsyncIterator", "AsyncGenerator", "Hashable", "Iterable",
    "Iterator", "Generator", "Reversible", "Sized", "Container", "Callable", "Collection", "Set", "MutableSet",
    "Mapping", "MutableMapping", "MappingView", "KeysView", "ItemsView", "ValuesView", "Sequence",
    "MutableSequence", "ByteString",
}


class CollectionsAbc(Migration):
    id = "MIG-COLLECTIONS-ABC"
    trigger = r"^\s*from\s+collections\s+import\b|\bcollections\.(?:" + "|".join(sorted(_COLLECTIONS_ABC)) + r")\b"
    title = "Abstract base classes imported from `collections` (removed in Python 3.10)"
    severity = "HIGH"
    min_python = "3.10"
    guideline = "`collections.Mapping`, `collections.Callable` etc. moved to `collections.abc` and the aliases are gone in 3.10."

    def __init__(self, fix: bool = False) -> None:
        super().__init__(fix)
        self._needs_abc_import = False
        self._top_imports: set[str] = set()

    def visit_Module(self, node: cst.Module) -> None:
        self._top_imports = _top_level_imports(node)

    def leave_SimpleStatementLine(self, original: cst.SimpleStatementLine,
                                  updated: cst.SimpleStatementLine) -> Any:
        imports = [s for s in original.body if isinstance(s, cst.ImportFrom)
                   and s.relative == () and _dotted(s.module) == "collections" and _import_alias_names(s)]
        if not imports:
            return updated
        stmt = imports[0]
        names = _import_alias_names(stmt) or []
        moved = [a for a in names if isinstance(a.name, cst.Name) and a.name.value in _COLLECTIONS_ABC]
        if not moved:
            return updated
        single = len(original.body) == 1
        self.hit(stmt, fixable=single)
        if not (self.fix and single):
            return updated
        kept = [a for a in names if a not in moved]
        abc_stmt = cst.SimpleStatementLine(body=[stmt.with_changes(
            module=cst.Attribute(value=cst.Name("collections"), attr=cst.Name("abc")),
            names=[a.with_changes(comma=cst.MaybeSentinel.DEFAULT) for a in moved])],
            leading_lines=original.leading_lines)
        if not kept:
            return abc_stmt
        kept_fixed = [*kept[:-1], kept[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)]
        kept_stmt = original.with_changes(body=[stmt.with_changes(names=kept_fixed)], leading_lines=())
        return cst.FlattenSentinel([abc_stmt, kept_stmt])

    def leave_Attribute(self, original: cst.Attribute, updated: cst.Attribute) -> cst.BaseExpression:
        if not (isinstance(original.value, cst.Name) and original.value.value == "collections"
                and original.attr.value in _COLLECTIONS_ABC):
            return updated
        fixable = "collections" in self._top_imports or "collections.abc" in self._top_imports
        self.hit(original, fixable=fixable)
        if self.fix and fixable:
            self._needs_abc_import = "collections.abc" not in self._top_imports
            return updated.with_changes(value=cst.Attribute(value=cst.Name("collections"), attr=cst.Name("abc")))
        return updated

    def finish(self, module: cst.Module) -> cst.Module:
        if self._needs_abc_import and "collections.abc." in module.code:
            return _insert_after_import(module, "collections", "import collections.abc")
        return module


def _aliases(module: cst.Module) -> dict[str, str]:
    """Local name -> dotted origin, from every `import a.b as c` / `from a import b as c` in the module."""
    out: dict[str, str] = {}

    class _V(cst.CSTVisitor):
        def visit_Import(self, node: cst.Import) -> None:
            for a in node.names:
                dotted = _dotted(a.name)
                if not dotted:
                    continue
                if a.asname and isinstance(a.asname.name, cst.Name):
                    out[a.asname.name.value] = dotted
                else:
                    root = dotted.split(".")[0]
                    out.setdefault(root, root)

        def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
            if node.relative or isinstance(node.names, cst.ImportStar) or node.module is None:
                return
            mod = _dotted(node.module)
            for a in node.names:
                if isinstance(a.name, cst.Name):
                    local = a.asname.name.value if a.asname and isinstance(a.asname.name, cst.Name) else a.name.value
                    out[local] = f"{mod}.{a.name.value}"

    module.visit(_V())
    return out


def _resolve(dotted: str | None, aliases: dict[str, str]) -> str | None:
    if not dotted:
        return None
    head, _, rest = dotted.partition(".")
    if head not in aliases:
        return None
    return aliases[head] + ("." + rest if rest else "")


def _names_used(module: cst.Module) -> set[str]:
    """Every Name in the module outside import statements (used to drop imports that became unused)."""
    used: set[str] = set()

    class _V(cst.CSTVisitor):
        def visit_Import(self, node: cst.Import) -> bool:
            return False

        def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
            return False

        def visit_Name(self, node: cst.Name) -> None:
            used.add(node.value)

    module.visit(_V())
    return used


def _drop_import(module: cst.Module, dotted: str) -> cst.Module:
    """Removes a top-level `import <dotted>` line that imports nothing else."""
    body = [s for s in module.body
            if not (isinstance(s, cst.SimpleStatementLine) and len(s.body) == 1 and isinstance(s.body[0], cst.Import)
                    and len(s.body[0].names) == 1 and s.body[0].names[0].asname is None
                    and _dotted(s.body[0].names[0].name) == dotted)]
    return module.with_changes(body=body)


class _ApiRenames(Migration):
    """
    Table-driven, behavior-identical renames: `renames` maps an old dotted API to its new one
    (`"pipes.quote": "shlex.quote"`), `module_renames` maps an old module to its new name.
    Handles `mod.attr` (when `mod` is imported at top level; the new module is imported if
    needed) and `from mod import attr [as x]` (rewritten keeping the local name).
    """

    renames: ClassVar[dict[str, str]] = {}
    module_renames: ClassVar[dict[str, str]] = {}
    drop_unused: ClassVar[tuple[str, ...]] = ()   # removed modules whose import is deleted once unused

    def __init__(self, fix: bool = False) -> None:
        super().__init__(fix)
        self._top: set[str] = set()
        self._add_imports: dict[str, str] = {}   # new module -> module whose import line it follows

    @classmethod
    def relevant(cls, code: str) -> bool:
        if "_trigger_re" not in cls.__dict__:
            cls._trigger_re = _dotted_trigger([*cls.renames, *cls.module_renames])
        return cls._trigger_re.search(code) is not None

    def visit_Module(self, node: cst.Module) -> None:
        self._top = _top_level_imports(node)

    def _bound(self, dotted: str) -> str | None:
        """The imported module prefix that makes `dotted` reachable, if any."""
        parts = dotted.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in self._top:
                return prefix
        return None

    def leave_Attribute(self, original: cst.Attribute, updated: cst.Attribute) -> cst.BaseExpression:
        dotted = _dotted(original)
        if dotted is None:
            return updated
        if dotted in self.renames:
            new = self.renames[dotted]
            bound = self._bound(dotted)
            self.hit(original, fixable=bound is not None, detail=f"{dotted} -> {new}")
            if self.fix and bound is not None:
                needed = new.rsplit(".", 1)[0] if "." in new else new
                if needed not in self._top and self._bound(new) is None:
                    self._add_imports.setdefault(needed, bound)
                return cst.parse_expression(new)
        elif dotted in self.module_renames and dotted in self._top:
            self.hit(original, detail=f"{dotted} -> {self.module_renames[dotted]}")
            if self.fix:
                return cst.parse_expression(self.module_renames[dotted])
        return updated

    def leave_Import(self, original: cst.Import, updated: cst.Import) -> cst.Import:
        names = list(updated.names)
        changed = False
        for i, a in enumerate(original.names):
            dotted = _dotted(a.name)
            if dotted in self.module_renames:
                self.hit(original, detail=f"{dotted} -> {self.module_renames[dotted]}")
                if self.fix:
                    names[i] = names[i].with_changes(name=cst.parse_expression(self.module_renames[dotted]))
                    changed = True
        return updated.with_changes(names=names) if changed else updated

    def leave_SimpleStatementLine(self, original: cst.SimpleStatementLine,
                                  updated: cst.SimpleStatementLine) -> Any:
        stmt = original.body[0] if original.body else None
        if not isinstance(stmt, cst.ImportFrom) or stmt.relative or stmt.module is None \
                or isinstance(stmt.names, cst.ImportStar):
            return updated
        mod = _dotted(stmt.module)
        single = len(original.body) == 1
        if mod in self.module_renames:  # from old_module import x  ->  from new_module import x
            self.hit(stmt, fixable=single, detail=f"{mod} -> {self.module_renames[mod]}")
            if self.fix and single:
                return updated.with_changes(body=[updated.body[0].with_changes(
                    module=cst.parse_expression(self.module_renames[mod]))])
            return updated
        groups: dict[str, list[cst.ImportAlias]] = {}
        kept: list[cst.ImportAlias] = []
        for a in stmt.names:
            name = a.name.value if isinstance(a.name, cst.Name) else None
            full = f"{mod}.{name}"
            target = self.renames.get(full) or self.module_renames.get(full)
            if name is None or target is None:
                kept.append(a)
                continue
            self.hit(stmt, fixable=single, detail=f"{full} -> {target}")
            new_mod, _, new_name = target.rpartition(".")
            local = a.asname.name.value if a.asname and isinstance(a.asname.name, cst.Name) else name
            alias = cst.ImportAlias(name=cst.Name(new_name),
                                    asname=None if local == new_name else cst.AsName(name=cst.Name(local)))
            groups.setdefault(new_mod, []).append(alias)
        if not groups or not (self.fix and single):
            return updated
        lines = [cst.SimpleStatementLine(
            body=[cst.ImportFrom(module=cst.parse_expression(m), names=aliases)],
            leading_lines=original.leading_lines if i == 0 else ())
            for i, (m, aliases) in enumerate(groups.items())]
        if kept:
            kept = [*kept[:-1], kept[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)]
            lines.append(original.with_changes(body=[stmt.with_changes(names=kept)], leading_lines=()))
        return cst.FlattenSentinel(lines)

    def finish(self, module: cst.Module) -> cst.Module:
        code = module.code
        for needed, after in self._add_imports.items():
            if re.search(rf"\b{re.escape(needed)}\b", code.replace(f"import {after}", "")):
                module = _insert_after_import(module, after, f"import {needed}")
        used = _names_used(module)
        for mod in self.drop_unused:
            if mod in self._top and mod.split(".")[0] not in used:
                module = _drop_import(module, mod)
        return module


class RenamedStdlibApis(_ApiRenames):
    id = "MIG-RENAMED-APIS"
    title = "Standard-library API removed in a recent Python, with an identical replacement"
    severity = "HIGH"
    min_python = "3.9"
    guideline = ("Each of these was an alias (or a moved copy) of the replacement, which takes the same arguments "
                 "and behaves the same, so the rewrite is mechanical. Removed in: `base64.encodestring` 3.9, "
                 "`cgi.parse_qs` 3.8, `xml.etree.cElementTree` 3.9, `asyncio.Task.current_task` 3.9, "
                 "`imp` / `pipes` / `SafeConfigParser` 3.12-3.13, `logging.warn` 3.13.")
    renames: ClassVar[dict[str, str]] = {
        "base64.encodestring": "base64.encodebytes",
        "base64.decodestring": "base64.decodebytes",
        "logging.warn": "logging.warning",
        "threading.currentThread": "threading.current_thread",
        "threading.activeCount": "threading.active_count",
        "configparser.SafeConfigParser": "configparser.ConfigParser",
        "cgi.parse_qs": "urllib.parse.parse_qs",
        "cgi.parse_qsl": "urllib.parse.parse_qsl",
        "imp.reload": "importlib.reload",
        "pipes.quote": "shlex.quote",
        "asyncio.Task.current_task": "asyncio.current_task",
        "asyncio.Task.all_tasks": "asyncio.all_tasks",
        "os.errno": "errno",
    }
    module_renames: ClassVar[dict[str, str]] = {"xml.etree.cElementTree": "xml.etree.ElementTree"}
    drop_unused = ("imp", "pipes", "cgi")


class CgiEscape(Migration):
    id = "MIG-CGI-ESCAPE"
    trigger = r"\bcgi\.escape\b"
    title = "`cgi.escape` (removed in Python 3.8)"
    severity = "HIGH"
    min_python = "3.8"
    guideline = ("`cgi.escape(s)` -> `html.escape(s, quote=False)`. `html.escape` quotes `\"` and `'` by default, "
                 "`cgi.escape` did not, so `quote=False` keeps the output identical.")

    def __init__(self, fix: bool = False) -> None:
        super().__init__(fix)
        self._bound = False
        self._need_html = False
        self._top: set[str] = set()

    def visit_Module(self, node: cst.Module) -> None:
        self._top = _top_level_imports(node)
        self._bound = "cgi" in self._top

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.BaseExpression:
        if _dotted(original.func) != "cgi.escape":
            return updated
        # Only the quote=False form is identical: with quote=True, html.escape also escapes `'`.
        args = list(updated.args)
        quote_false = len(args) == 2 and not args[1].star and isinstance(args[1].value, cst.Name) \
            and args[1].value.value == "False" and (args[1].keyword is None or args[1].keyword.value == "quote")
        ok = self._bound and len(args) in (1, 2) and not args[0].star and args[0].keyword is None \
            and (len(args) == 1 or quote_false)
        self.hit(original, fixable=ok, detail="" if ok else "quote=True: html.escape also escapes ', review")
        if not (self.fix and ok):
            return updated
        args = [args[0].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
                cst.Arg(keyword=cst.Name("quote"), value=cst.Name("False"),
                        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""),
                                              whitespace_after=cst.SimpleWhitespace("")))]
        self._need_html = "html" not in self._top
        return updated.with_changes(func=cst.Attribute(value=cst.Name("html"), attr=cst.Name("escape")), args=args)

    def finish(self, module: cst.Module) -> cst.Module:
        if self._need_html and "html.escape(" in module.code:
            module = _insert_after_import(module, "cgi", "import html")
        if self.fix and self._bound and "cgi" not in _names_used(module):
            module = _drop_import(module, "cgi")  # cgi itself is removed in 3.13
        return module


class _DetectOnly(Migration):
    """
    Detection of APIs that have no behavior-identical replacement. `targets` are dotted names,
    matched through import aliases (`from inspect import getargspec` is found too); `methods`
    are attribute names matched on any receiver, but only in files importing `requires_import`.
    """

    fixable: ClassVar[bool] = False
    targets: ClassVar[dict[str, str]] = {}
    methods: ClassVar[dict[str, str]] = {}
    requires_import: ClassVar[str] = ""

    def __init__(self, fix: bool = False) -> None:
        super().__init__(fix)
        self._aliases: dict[str, str] = {}
        self._attr_names: set[int] = set()
        self._active = True

    @classmethod
    def relevant(cls, code: str) -> bool:
        if cls.requires_import and not re.search(rf"^\s*(?:import|from)\s+{re.escape(cls.requires_import)}\b",
                                                 code, re.MULTILINE):
            return False
        if "_trigger_re" not in cls.__dict__:
            cls._trigger_re = _dotted_trigger(list(cls.targets), list(cls.methods))
        return cls._trigger_re.search(code) is not None

    def visit_Module(self, node: cst.Module) -> None:
        self._aliases = _aliases(node)
        if self.requires_import:
            self._active = any(v == self.requires_import or v.startswith(self.requires_import + ".")
                               for v in self._aliases.values())

    def visit_Attribute(self, node: cst.Attribute) -> None:
        self._attr_names.add(id(node.attr))

    # Names the module defines itself (stubs like `class Extra: ...`, `Extra = None`) are not uses.
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self._attr_names.add(id(node.name))

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._attr_names.add(id(node.name))

    def visit_AssignTarget(self, node: cst.AssignTarget) -> None:
        if isinstance(node.target, cst.Name):
            self._attr_names.add(id(node.target))

    def leave_Attribute(self, original: cst.Attribute, updated: cst.BaseExpression) -> cst.BaseExpression:
        if not self._active:
            return updated
        resolved = _resolve(_dotted(original), self._aliases)
        if resolved in self.targets:
            self.hit(original, fixable=False, detail=self.targets[resolved])
        elif original.attr.value in self.methods and resolved is None:
            self.hit(original, fixable=False, detail=self.methods[original.attr.value])
        return updated

    def leave_Name(self, original: cst.Name, updated: cst.Name) -> cst.Name:
        if self._active and id(original) not in self._attr_names:
            resolved = self._aliases.get(original.value)
            if resolved in self.targets and resolved != original.value:
                self.hit(original, fixable=False, detail=self.targets[resolved])
        return updated


class RemovedStdlibApis(_DetectOnly):
    id = "MIG-REMOVED-APIS"
    title = "Standard-library API removed in a recent Python (manual migration)"
    severity = "HIGH"
    min_python = "3.8"
    guideline = "These APIs no longer exist; the replacement changes arguments or semantics, so migrate by hand."
    targets: ClassVar[dict[str, str]] = {
        "inspect.getargspec": "use inspect.getfullargspec or inspect.signature (removed in 3.11)",
        "inspect.formatargspec": "use inspect.signature (removed in 3.11)",
        "time.clock": "use time.perf_counter or time.process_time (removed in 3.8)",
        "asyncio.coroutine": "use `async def` (removed in 3.11)",
        "string.letters": "use string.ascii_letters",
        "sys.maxint": "use sys.maxsize",
        "fractions.gcd": "use math.gcd (always non-negative; fractions.gcd took the sign of b; removed in 3.9)",
        "platform.linux_distribution": "use the `distro` package or platform.freedesktop_os_release (removed in 3.8)",
        "platform.dist": "use the `distro` package (removed in 3.8)",
        "unittest.makeSuite": "use unittest.TestLoader().loadTestsFromTestCase (removed in 3.13)",
        "unittest.findTestCases": "use unittest.TestLoader().loadTestsFromModule (removed in 3.13)",
        "unittest.getTestCaseNames": "use unittest.TestLoader().getTestCaseNames (removed in 3.13)",
        "locale.format": "use locale.format_string (removed in 3.12)",
        "locale.resetlocale": "use locale.setlocale(locale.LC_ALL, '') (removed in 3.13)",
        "ssl.wrap_socket": "use ssl.SSLContext.wrap_socket (removed in 3.12)",
        "ssl.match_hostname": "SSLContext does hostname checking (removed in 3.12)",
        "gettext.lgettext": "use gettext.gettext (removed in 3.11)",
        "binascii.b2a_hqx": "binhex encoding was removed in 3.11",
        "imp.load_source": "use importlib.util.spec_from_file_location (imp removed in 3.12)",
        "imp.find_module": "use importlib.util.find_spec (imp removed in 3.12)",
        "imp.load_module": "use importlib.import_module (imp removed in 3.12)",
        "typing.io": "import from typing directly (removed in 3.13)",
        "typing.re": "import from typing directly (removed in 3.13)",
        "ast.Num": "use ast.Constant and check type(node.value) (removed in 3.14)",
        "ast.Str": "use ast.Constant with a str value (removed in 3.14)",
        "ast.Bytes": "use ast.Constant with a bytes value (removed in 3.14)",
        "ast.NameConstant": "use ast.Constant (removed in 3.14)",
        "ast.Ellipsis": "use ast.Constant with value ... (removed in 3.14)",
    }


class NaiveUtcNow(_DetectOnly):
    id = "MIG-DATETIME-UTCNOW"
    title = "Naive UTC datetimes (`utcnow` / `utcfromtimestamp`, deprecated in Python 3.12)"
    severity = "MEDIUM"
    min_python = "3.12"
    guideline = ("`datetime.utcnow()` returns a naive datetime. `datetime.now(timezone.utc)` is timezone-aware, "
                 "which changes comparison and serialization behavior, so it is not rewritten automatically.")
    targets: ClassVar[dict[str, str]] = {
        "datetime.datetime.utcnow": "use datetime.now(timezone.utc) after checking naive/aware comparisons",
        "datetime.datetime.utcfromtimestamp": "use datetime.fromtimestamp(ts, timezone.utc)",
    }


_DEAD_MODULES = {
    # PEP 594 (removed in 3.13) and earlier removals
    **dict.fromkeys(("aifc", "audioop", "cgi", "cgitb", "chunk", "crypt", "imghdr", "mailcap", "msilib", "nis", "nntplib", "ossaudiodev", "pipes", "sndhdr", "spwd", "sunau", "telnetlib", "uu", "xdrlib", "lib2to3"), "removed in Python 3.13 (PEP 594)"),
    "asynchat": "removed in Python 3.12; use asyncio",
    "asyncore": "removed in Python 3.12; use asyncio",
    "smtpd": "removed in Python 3.12; use aiosmtpd",
    "distutils": "removed in Python 3.12; use setuptools / sysconfig / shutil",
    "imp": "removed in Python 3.12; use importlib",
    "binhex": "removed in Python 3.11",
    "formatter": "removed in Python 3.10",
    "parser": "removed in Python 3.10; use ast",
    "symbol": "removed in Python 3.10",
    "macpath": "removed in Python 3.8",
    "dummy_threading": "removed in Python 3.9; use threading",
    "_dummy_thread": "removed in Python 3.9",
    "pkg_resources": "deprecated by setuptools; use importlib.metadata / importlib.resources",
}


class RemovedModules(Migration):
    id = "MIG-REMOVED-MODULES"
    trigger = r"^\s*(?:import|from)\s+(?:[\w.]+\s*,\s*)*(?:" + "|".join(_DEAD_MODULES) + r")\b"
    title = "Import of a standard-library module removed from recent Python"
    severity = "CRITICAL"
    min_python = "3.10"
    fixable: ClassVar[bool] = False
    guideline = ("These modules no longer exist, so the import fails at runtime on new Python versions. Imports "
                 "inside `try: ... except ImportError` are treated as compatibility shims and not reported.")

    def __init__(self, fix: bool = False) -> None:
        super().__init__(fix)
        self._guarded = 0

    @staticmethod
    def _catches_import_error(node: cst.Try) -> bool:
        for h in node.handlers:
            names = [] if h.type is None else (
                [_dotted(e.value) for e in h.type.elements] if isinstance(h.type, cst.Tuple) else [_dotted(h.type)])
            if h.type is None or {"ImportError", "ModuleNotFoundError", "Exception"} & set(filter(None, names)):
                return True
        return False

    def visit_Try(self, node: cst.Try) -> None:
        self._guarded += self._catches_import_error(node)

    def leave_Try(self, original: cst.Try, updated: cst.Try) -> cst.Try:
        self._guarded -= self._catches_import_error(original)
        return updated

    def _check(self, node: cst.CSTNode, dotted: str | None) -> None:
        if not dotted or self._guarded:
            return
        root = dotted.split(".")[0]
        if root in _DEAD_MODULES:
            self.hit(node, fixable=False, detail=f"`{root}` {_DEAD_MODULES[root]}")

    def leave_Import(self, original: cst.Import, updated: cst.Import) -> cst.Import:
        for a in original.names:
            self._check(original, _dotted(a.name))
        return updated

    def leave_ImportFrom(self, original: cst.ImportFrom, updated: cst.ImportFrom) -> cst.ImportFrom:
        if not original.relative and original.module is not None:
            self._check(original, _dotted(original.module))
        return updated


class PydanticV1(_DetectOnly):
    id = "MIG-PYDANTIC-V2"
    title = "Pydantic v1 API (deprecated since Pydantic 2)"
    severity = "MEDIUM"
    pack = "pydantic"
    requires_import = "pydantic"
    guideline = ("Pydantic 2 keeps the v1 names as deprecated shims with subtly different validation; migrate with "
                 "the replacements below and run your tests (see the official migration guide, or `bump-pydantic`).")
    targets: ClassVar[dict[str, str]] = {
        "pydantic.validator": "use @field_validator (mode/signature differ)",
        "pydantic.root_validator": "use @model_validator(mode='before'|'after')",
        "pydantic.BaseSettings": "moved to the pydantic-settings package",
        "pydantic.parse_obj_as": "use TypeAdapter(T).validate_python",
        "pydantic.Extra": "use model_config = ConfigDict(extra=...)",
    }
    methods: ClassVar[dict[str, str]] = {
        "parse_obj": "use Model.model_validate", "parse_raw": "use Model.model_validate_json",
        "parse_file": "read the file and use model_validate_json", "update_forward_refs": "use model_rebuild",
        "schema_json": "use json.dumps(Model.model_json_schema())", "__fields__": "use model_fields",
    }


class SqlAlchemy2(_DetectOnly):
    id = "MIG-SQLALCHEMY-2"
    title = "SQLAlchemy 1.x API (removed or legacy in SQLAlchemy 2.0)"
    severity = "MEDIUM"
    pack = "sqlalchemy"
    requires_import = "sqlalchemy"
    guideline = ("SQLAlchemy 2.0 removed implicit autocommit, `engine.execute()` and string SQL without `text()`, "
                 "and moved `declarative_base` to `sqlalchemy.orm`. Turn on `SQLALCHEMY_WARN_20=1` under 1.4 to "
                 "find runtime-only cases.")
    targets: ClassVar[dict[str, str]] = {
        "sqlalchemy.ext.declarative.declarative_base": "use sqlalchemy.orm.declarative_base or DeclarativeBase",
        "sqlalchemy.orm.mapper": "use registry.map_imperatively",
        "sqlalchemy.ext.declarative.declared_attr": "use sqlalchemy.orm.declared_attr",
        "sqlalchemy.orm.relation": "use sqlalchemy.orm.relationship",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def available() -> bool:
    return cst is not None


def _version(v: str | None) -> tuple[int, ...]:
    return tuple(int(x) for x in str(v).split(".") if x.isdigit()) if v else ()


def _select(only: list[str] | None = None, ignore: list[str] | None = None, packs: list[str] | None = None,
            target_python: str | None = None) -> list[type[Migration]]:
    """
    Migrations to run. `only`/`ignore` are migration ids, `packs` limits to those packs, and
    `target_python` ("3.12") keeps only stdlib migrations whose API is gone by that version
    (library packs have no Python version and always apply).
    """
    target = _version(target_python)
    return [m for mid, m in REGISTRY.items()
            if (not only or mid in only) and not (ignore and mid in ignore)
            and (not packs or m.pack in packs)
            and not (target and m.min_python and _version(m.min_python) > target)]


def settings(root: str) -> dict[str, Any]:
    """
    Project selection from `.novus/config.toml`:

        [migrations]
        target_python = "3.12"            # only APIs removed by 3.12
        ignore = ["MIG-DATETIME-UTCNOW"]
        packs = ["stdlib", "unittest"]    # default: all
    """
    import tomllib

    try:
        with open(os.path.join(root, ".novus", "config.toml"), "rb") as f:
            cfg = tomllib.load(f).get("migrations", {})
    except (OSError, ValueError):
        return {}
    out: dict[str, Any] = {}
    for key in ("only", "ignore", "packs"):
        if isinstance(cfg.get(key), list):
            out[key] = [str(x) for x in cfg[key]]
    if cfg.get("target_python"):
        out["target_python"] = str(cfg["target_python"])
    return out


def _run(wrapper: MetadataWrapper, migration: type[Migration], fix: bool) -> tuple[str, list[Hit]]:
    """Runs one migration over an already-parsed module (a wrapper can be visited many times)."""
    t = migration(fix=fix)
    new_module = wrapper.visit(t)
    if fix:
        new_module = t.finish(new_module)
    return (new_module.code if fix else ""), t.hits


def _wrap(code: str) -> MetadataWrapper:
    return MetadataWrapper(cst.parse_module(code), unsafe_skip_copy=True)


def scan(code: str, file_path: str = "", **selection: Any) -> list[dict[str, Any]]:
    """Findings (same shape as the rule engine's) for one source; [] if it is not valid Python 3."""
    if cst is None or (file_path and not file_path.lower().endswith((".py", ".pyw"))):
        return []
    migrations = [m for m in _select(**selection) if m.relevant(code)]
    if not migrations:
        return []
    try:
        wrapper = _wrap(code)
    except Exception:  # not valid Python 3 (e.g. Python 2 source): nothing to migrate yet
        return []
    lines = code.split("\n")
    target = _version(selection.get("target_python"))
    findings: list[dict[str, Any]] = []
    for m in migrations:
        try:
            _, hits = _run(wrapper, m, fix=False)
        except Exception:  # a migration bug must never break an audit
            continue
        for h in hits:
            removed = re.search(r"removed in (?:Python )?(\d+\.\d+)", h.detail)
            if target and removed and _version(removed.group(1)) > target:
                continue  # this API still exists in the target version
            findings.append({
                "smell_id": m.id, "name": m.title + (f": {h.detail}" if h.detail else ""),
                "line_number": h.line, "line_content": lines[h.line - 1].strip() if 0 < h.line <= len(lines) else "",
                "category": "python", "severity": m.severity, "rag_query": m.rag_query,
                "file_path": file_path, "source": "migration", "fix": "safe" if h.fixable else None,
                "guideline": m.guideline, "min_python": m.min_python,
            })
    return sorted(findings, key=lambda f: (f["line_number"], f["smell_id"]))


def apply(code: str, **selection: Any) -> tuple[str, list[str]]:
    """Applies every safe fix; returns (new_code, notes). Unparseable input is returned unchanged."""
    if cst is None:
        return code, []
    notes: list[str] = []
    current = code
    wrapper = None
    for m in _select(**selection):
        if not m.relevant(current):
            continue
        try:
            wrapper = wrapper or _wrap(current)
            new, hits = _run(wrapper, m, fix=True)
            new_wrapper = _wrap(new) if new != current else wrapper  # the result must still be valid Python
        except Exception:
            continue
        fixed = sum(1 for h in hits if h.fixable)
        manual = len(hits) - fixed
        if fixed and new != current:
            current, wrapper = new, new_wrapper
            notes.append(f"{m.title}: rewrote {fixed} use(s) ({m.id}).")
        if manual:
            notes.append(f"Manual review: {manual} use(s) of {m.title} ({m.id}) need a manual change.")
    return current, notes


def catalog() -> list[dict[str, Any]]:
    """Every registered migration, for listings."""
    return [{"id": m.id, "title": m.title, "severity": m.severity, "min_python": m.min_python,
             "guideline": m.guideline,
             "fixable": m.fixable, "pack": m.pack}
            for m in REGISTRY.values()]
