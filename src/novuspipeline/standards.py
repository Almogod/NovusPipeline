"""
standards.py — NovusPipeline: your modernization standards (rule packs)

Built-in rules encode general good practice. Industries, companies and individuals
have their own: "money is never a float", "every HTTP call has a timeout", "no
`yaml.load`", "this internal client replaced that one". A rule pack is a TOML file
of such rules. Every rule can carry:

  detection    calls (dotted targets, import aliases resolved), imports, a regex
               pattern (matched with strings and comments blanked), and
               `require_keyword` (flag calls that lack e.g. `timeout=`)
  guideline    markdown explaining the standard: shown in audits, returned by
               `query_rag_guidelines`, and given to the LLM as a requirement
  fix          replace_call / replace_import / add_keyword / replacement (regex),
               applied by the modernizer and, like every change, verified by the
               tests in the sandbox
  enforce      "warn" (report; LLM proposals may not add violations) or "block"
               (the pipeline does not commit a file that still violates it)
  scope        `paths` / `exclude` globs relative to the workspace

Packs are loaded from three places; a later rule with the same id replaces an earlier one:

  user       <data dir>/standards/*.toml  every project of this user (an individual's standards)
  project    <workspace>/.novus/standards/*.toml  checked in, shared by the team
  built-in   novuspipeline/standard_packs/*.toml  examples; installed into a project with
             `install_standard_pack` (copied, so the team can edit them)

A pack may also tune Ruff: `[ruff] select = [...]` adds rule families/codes,
`ignore = [...]` drops codes (e.g. when a pack rule supersedes them with a fix).
"""

import ast
import contextlib
import fnmatch
import glob
import json
import os
import re
import threading
from typing import Any

from .code_facts import dotted
from .modernizer import mask_strings_and_comments

SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
ENFORCEMENT = ("warn", "block")
CUSTOM_PACK = "custom.toml"  # the file `add_modernization_rule` manages in each scope
PROJECT_DIR = os.path.join(".novus", "standards")
BUILTIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standard_packs")
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_LIST_FIELDS = ("calls", "imports", "paths", "exclude")
_STR_FIELDS = ("title", "guideline", "category", "severity", "enforce", "pattern", "require_keyword",
               "replace_call", "replace_import", "add_keyword", "replacement")
_RULE_FIELDS = {"id", *_LIST_FIELDS, *_STR_FIELDS}
_SCRIPT_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


def user_dir() -> str:
    from .ingest_rag import data_dir
    return os.path.join(data_dir(), "standards")


def scope_dir(workspace: str, scope: str) -> str:
    return user_dir() if scope == "user" else os.path.join(workspace, PROJECT_DIR)


def builtin_packs() -> dict[str, str]:
    return {os.path.splitext(os.path.basename(p))[0]: p for p in sorted(glob.glob(os.path.join(BUILTIN_DIR, "*.toml")))}


def _read_toml(path: str) -> dict[str, Any]:
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def validate_rule(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """(normalized rule, problems). A rule with problems is not loaded."""
    problems = [f"unknown field `{k}`" for k in raw if k not in _RULE_FIELDS]
    rule: dict[str, Any] = {"id": str(raw.get("id", "")).strip()}
    if not _ID_RE.match(rule["id"]):
        problems.append(f"invalid id `{rule['id']}` (letters, digits, `-`, `_`, `.`; starts with a letter)")
    for key in _LIST_FIELDS:
        value = raw.get(key, [])
        value = [value] if isinstance(value, str) else value
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            problems.append(f"`{key}` must be a list of strings")
            value = []
        rule[key] = [v.strip() for v in value if v.strip()]
    for key in _STR_FIELDS:
        value = raw.get(key, "")
        if not isinstance(value, str):
            problems.append(f"`{key}` must be a string")
            value = ""
        rule[key] = value.strip() if key != "guideline" else value.strip("\n")
    rule["title"] = rule["title"] or rule["id"]
    rule["category"] = (rule["category"] or "python").lower()
    rule["severity"] = (rule["severity"] or "MEDIUM").upper()
    rule["enforce"] = (rule["enforce"] or "warn").lower()
    if rule["severity"] not in SEVERITIES:
        problems.append(f"severity must be one of {', '.join(SEVERITIES)}")
    if rule["enforce"] not in ENFORCEMENT:
        problems.append("enforce must be `warn` or `block`")
    if not (rule["calls"] or rule["imports"] or rule["pattern"]):
        problems.append("needs at least one of `calls`, `imports`, `pattern`")
    if rule["pattern"]:
        try:
            re.compile(rule["pattern"])
        except re.error as e:
            problems.append(f"invalid `pattern`: {e}")
    if rule["require_keyword"] and not rule["calls"]:
        problems.append("`require_keyword` needs `calls`")
    if rule["add_keyword"] and not rule["require_keyword"]:
        problems.append("`add_keyword` needs `require_keyword`")
    if rule["add_keyword"]:
        try:
            ast.parse(f"f({rule['add_keyword']})")
            if not rule["add_keyword"].startswith(rule["require_keyword"] + "="):
                problems.append(f"`add_keyword` must look like `{rule['require_keyword']}=<value>`")
        except SyntaxError:
            problems.append("`add_keyword` must be a keyword argument such as `timeout=30`")
    if rule["replace_call"] and not rule["calls"]:
        problems.append("`replace_call` needs `calls`")
    if rule["replace_call"] and not re.fullmatch(r"[A-Za-z_][\w.]*", rule["replace_call"]):
        problems.append("`replace_call` must be a dotted name such as `yaml.safe_load`")
    if rule["replace_import"] and not rule["imports"]:
        problems.append("`replace_import` needs `imports`")
    if rule["replacement"] and not rule["pattern"]:
        problems.append("`replacement` needs `pattern`")
    return (None if problems else rule), problems


class Standards:
    """The merged, validated rules active for one workspace."""

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self.rules: list[dict[str, Any]] = []
        self.packs: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.ruff_select: list[str] = []
        self.ruff_ignore: list[str] = []
        by_id: dict[str, dict[str, Any]] = {}
        for scope in ("user", "project"):
            for path in sorted(glob.glob(os.path.join(scope_dir(workspace, scope), "*.toml"))):
                self._load_pack(path, scope, by_id)
        self.rules = list(by_id.values())
        self._embeddings: list[list[float]] | None = None

    def _load_pack(self, path: str, scope: str, by_id: dict[str, dict[str, Any]]) -> None:
        shown = os.path.relpath(path, self.workspace) if scope == "project" else path
        try:
            data = _read_toml(path)
        except Exception as e:
            self.errors.append(f"`{shown}`: not valid TOML ({e})")
            return
        meta = data.get("pack") or {}
        name = str(meta.get("name") or os.path.splitext(os.path.basename(path))[0])
        loaded = 0
        for i, raw in enumerate(data.get("rule") or []):
            if not isinstance(raw, dict):
                self.errors.append(f"`{shown}` rule #{i + 1}: must be a table")
                continue
            rule, problems = validate_rule(raw)
            if rule is None:
                self.errors.append(f"`{shown}` rule `{raw.get('id', '#' + str(i + 1))}`: {'; '.join(problems)}")
                continue
            rule.update(pack=name, scope=scope, file=path)
            by_id[rule["id"]] = rule
            loaded += 1
        ruff = data.get("ruff") or {}
        self.ruff_select += [str(c) for c in ruff.get("select", [])]
        self.ruff_ignore += [str(c) for c in ruff.get("ignore", [])]
        self.packs.append({"name": name, "description": str(meta.get("description", "")), "scope": scope,
                           "path": shown, "rules": loaded})

    # ---------------------------------------------------------------- matching

    def applies(self, rule: dict[str, Any], rel_path: str) -> bool:
        rel = rel_path.replace(os.sep, "/")
        if rule["paths"] and not any(fnmatch.fnmatch(rel, g) for g in rule["paths"]):
            return False
        if any(fnmatch.fnmatch(rel, g) for g in rule["exclude"]):
            return False
        is_script = rel.lower().endswith(_SCRIPT_EXTENSIONS)
        if is_script:  # only regex rules for TypeScript/JavaScript, and only those meant for them
            return bool(rule["pattern"]) and rule["category"] in ("typescript", "javascript")
        return rel.lower().endswith((".py", ".pyw"))

    @staticmethod
    def _aliases(tree: ast.Module) -> dict[str, str]:
        """Local name -> what it refers to, from imports (`import yaml as y`, `from yaml import load`)."""
        out: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        out[a.asname] = a.name
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for a in node.names:
                    if a.name != "*":
                        out[a.asname or a.name] = f"{node.module}.{a.name}"
        return out

    @classmethod
    def _call_target(cls, call: ast.Call, aliases: dict[str, str]) -> str | None:
        target = dotted(call.func)
        if target is None:
            return None
        head, _, rest = target.partition(".")
        return aliases.get(head, head) + ("." + rest if rest else "")

    def _violations(self, rule: dict[str, Any], tree: ast.Module | None,
                    masked_lines: list[str]) -> list[tuple[int, Any]]:
        """[(line, node or None)] for one rule."""
        hits: list[tuple[int, Any]] = []
        if tree is not None and rule["calls"]:
            aliases = self._aliases(tree)
            wanted = set(rule["calls"])
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and self._call_target(node, aliases) in wanted:
                    if rule["require_keyword"] and (
                            any(k.arg in (rule["require_keyword"], None) for k in node.keywords)):
                        continue  # has the keyword (or **kwargs that may carry it)
                    hits.append((node.lineno, node))
        if tree is not None and rule["imports"]:
            banned = set(rule["imports"])
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    names = [node.module]
                if any(n in banned or any(n.startswith(b + ".") for b in banned) for n in names):
                    hits.append((node.lineno, node))
        if rule["pattern"]:
            rx = re.compile(rule["pattern"])
            hits += [(i, None) for i, line in enumerate(masked_lines, 1) if rx.search(line)]
        unique = {(line, id(node)): (line, node) for line, node in hits}
        return sorted(unique.values(), key=lambda hit: hit[0])

    @staticmethod
    def _parse(code: str, rel_path: str) -> tuple[ast.Module | None, list[str]]:
        if rel_path.lower().endswith(_SCRIPT_EXTENSIONS):
            return None, code.split("\n")
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError):
            tree = None
        return tree, mask_strings_and_comments(code).split("\n")

    def scan(self, code: str, rel_path: str) -> list[dict[str, Any]]:
        """Findings in the same shape as the rule engine's, with source "standard"."""
        rules = [r for r in self.rules if self.applies(r, rel_path)]
        if not rules:
            return []
        tree, masked = self._parse(code, rel_path)
        lines = code.split("\n")
        findings = []
        for rule in rules:
            for line, _ in self._violations(rule, tree, masked):
                findings.append({
                    "smell_id": rule["id"], "name": rule["title"], "line_number": line,
                    "line_content": lines[line - 1].strip() if 0 < line <= len(lines) else "",
                    "category": rule["category"], "severity": rule["severity"], "rag_query": rule["title"],
                    "file_path": rel_path, "source": "standard", "pack": rule["pack"], "enforce": rule["enforce"],
                    "fix": "safe" if self._fixable(rule) else None, "guideline": rule["guideline"],
                })
        return sorted(findings, key=lambda f: (f["line_number"], f["smell_id"]))

    @staticmethod
    def _fixable(rule: dict[str, Any]) -> bool:
        return bool(rule["replace_call"] or rule["replace_import"] or rule["add_keyword"] or rule["replacement"])

    def blocking(self, code: str, rel_path: str) -> list[dict[str, Any]]:
        return [f for f in self.scan(code, rel_path) if f["enforce"] == "block"]

    # ---------------------------------------------------------------- fixes

    def apply_fixes(self, code: str, rel_path: str) -> tuple[str, list[str]]:
        """Applies each fixable rule in turn; a rule whose fix would not parse is skipped."""
        notes: list[str] = []
        for rule in self.rules:
            if not self._fixable(rule) or not self.applies(rule, rel_path):
                continue
            tree, masked = self._parse(code, rel_path)
            if tree is None:
                break
            hits = self._violations(rule, tree, masked)
            if not hits:
                continue
            new = _fix_rule(rule, code, hits, masked)
            if new == code:
                continue
            try:
                ast.parse(new)
            except SyntaxError:
                notes.append(f"Skipped standard {rule['id']} fix: the result would not parse.")
                continue
            code = new
            notes.append(f"Applied standard {rule['id']} ({rule['pack']}): {rule['title']} ({len(hits)}x).")
        return code, notes

    # ---------------------------------------------------------------- guidelines

    def search(self, query: str, n: int, category: str = "") -> list[tuple[float, dict[str, Any]]]:
        """Rules whose guideline matches `query`, scored like the RAG store (cosine of TF-IDF vectors)."""
        from .ingest_rag import TFIDFEmbeddingFunction
        rules = [r for r in self.rules if not category or r["category"] == category]
        if not rules:
            return []
        ef = TFIDFEmbeddingFunction()
        q = ef._embed_single(query, {})
        scored = []
        for r in rules:
            v = ef._embed_single(f"{r['title']} {r['id']} {r['guideline']} {' '.join(r['calls'] + r['imports'])}", {})
            scored.append((sum(a * b for a, b in zip(q, v)), r))
        return sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])[:n]

    def get(self, rule_id: str) -> dict[str, Any] | None:
        return next((r for r in self.rules if r["id"] == rule_id), None)

    def prompt_block(self, findings: list[dict[str, Any]]) -> str:
        """Standards the LLM must follow for this file (violated ones first, then the rest briefly)."""
        if not self.rules:
            return ""
        violated = {f["smell_id"] for f in findings if f.get("source") == "standard"}
        lines = ["Organization modernization standards (the result must comply):"]
        for r in sorted(self.rules, key=lambda r: (r["id"] not in violated, r["id"]))[:20]:
            summary = " ".join(r["guideline"].split())[:300]
            lines.append(f"- [{r['id']}{', violated in this file' if r['id'] in violated else ''}] "
                         f"{r['title']}: {summary}")
        return "\n".join(lines)


def _line_offsets(code: str):
    lines = code.splitlines(keepends=True) or [""]
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(lineno: int, col_bytes: int) -> int:
        # AST columns are UTF-8 byte offsets; convert to a character index.
        return starts[lineno - 1] + len(lines[lineno - 1].encode("utf-8")[:col_bytes].decode("utf-8", "replace"))
    return lines, starts, offset


def _ensure_import(code: str, tree: ast.Module, module: str) -> str:
    """Adds `import <module>` after the existing top-level imports unless the name is already bound."""
    bound = {(a.asname or a.name).split(".")[0] for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
             for a in n.names}
    bound |= {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    if module.split(".")[0] in bound:
        return code
    after = 0
    for i, node in enumerate(tree.body):
        is_docstring = i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) \
            and isinstance(node.value.value, str)
        if is_docstring or isinstance(node, (ast.Import, ast.ImportFrom)):
            after = node.end_lineno
        else:
            break
    lines = code.split("\n")
    lines.insert(after, f"import {module}")
    return "\n".join(lines)


def _fix_rule(rule: dict[str, Any], code: str, hits: list[tuple[int, Any]],
              masked: list[str]) -> str:
    lines, starts, offset = _line_offsets(code)
    edits: list[tuple[int, int, str]] = []
    needs_import = None
    for line, node in hits:
        if isinstance(node, ast.Call) and rule["add_keyword"]:
            close = offset(node.end_lineno, node.end_col_offset) - 1  # the closing parenthesis
            inside = code[offset(node.func.end_lineno, node.func.end_col_offset):close]
            stripped = inside.strip()
            sep = "" if stripped in ("(", "") else " " if stripped.endswith(",") else ", "
            edits.append((close, close, sep + rule["add_keyword"]))
        elif isinstance(node, ast.Call) and rule["replace_call"]:
            f = node.func
            edits.append((offset(f.lineno, f.col_offset), offset(f.end_lineno, f.end_col_offset), rule["replace_call"]))
            if "." in rule["replace_call"]:
                needs_import = rule["replace_call"].rsplit(".", 1)[0]
        elif isinstance(node, ast.Import) and rule["replace_import"]:
            start, end = offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset)
            parts = []
            for a in node.names:
                banned = a.name in rule["imports"]
                if banned:
                    local = a.asname or a.name
                    parts.append(rule["replace_import"] + (f" as {local}" if local != rule["replace_import"] else ""))
                else:
                    parts.append(a.name + (f" as {a.asname}" if a.asname else ""))
            edits.append((start, end, "import " + ", ".join(parts)))
        elif isinstance(node, ast.ImportFrom) and rule["replace_import"] and node.module in rule["imports"]:
            start = offset(node.lineno, node.col_offset)
            text = code[start:offset(node.end_lineno, node.end_col_offset)]
            edits.append((start, start + len(text), re.sub(r"^from\s+[\w.]+", f"from {rule['replace_import']}", text)))
        elif node is None and rule["replacement"]:
            rx = re.compile(rule["pattern"])
            original = lines[line - 1]
            body = original.rstrip("\r\n")
            if rx.search(masked[line - 1]) and masked[line - 1] == body:  # no strings/comments on the line
                edits.append((starts[line - 1], starts[line - 1] + len(body), rx.sub(rule["replacement"], body)))
    for start, end, text in sorted(set(edits), reverse=True):
        code = code[:start] + text + code[end:]
    if needs_import:
        with contextlib.suppress(SyntaxError):
            code = _ensure_import(code, ast.parse(code), needs_import)
    return code


# ---------------------------------------------------------------- loading & editing

_CACHE: dict[str, tuple[str, Standards]] = {}
_CACHE_LOCK = threading.Lock()


def signature(workspace: str) -> str:
    parts = []
    for scope in ("user", "project"):
        for path in sorted(glob.glob(os.path.join(scope_dir(workspace, scope), "*.toml"))):
            with contextlib.suppress(OSError):
                parts.append(f"{path}:{os.path.getmtime(path)}:{os.path.getsize(path)}")
    return "|".join(parts)


def load(workspace: str) -> Standards:
    """Active standards for `workspace`, reloaded whenever a pack file changes."""
    key, sig = os.path.normcase(os.path.abspath(workspace)), signature(workspace)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is None or cached[0] != sig:
            cached = (sig, Standards(workspace))
            _CACHE[key] = cached
        return cached[1]


def _toml_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    text = str(value)
    if "\n" in text and "'''" not in text:
        return "'''\n" + text + "\n'''"
    return json.dumps(text, ensure_ascii=False)  # JSON string escapes are valid TOML basic strings


def _write_pack(path: str, data: dict[str, Any]) -> None:
    out = ["# Managed by NovusPipeline's add_modernization_rule; hand edits are kept as long as this file stays",
           "# valid TOML (comments are not preserved). Rules you write by hand can live in other *.toml files here.",
           ""]
    pack = data.get("pack") or {}
    if pack:
        out.append("[pack]")
        out += [f"{k} = {_toml_value(v)}" for k, v in pack.items()]
        out.append("")
    ruff = data.get("ruff") or {}
    if ruff:
        out.append("[ruff]")
        out += [f"{k} = {_toml_value(v)}" for k, v in ruff.items()]
        out.append("")
    for rule in data.get("rule") or []:
        out.append("[[rule]]")
        out += [f"{k} = {_toml_value(v)}" for k, v in rule.items() if v not in ("", [], None)]
        out.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))
    os.replace(tmp, path)


def upsert_rule(workspace: str, raw: dict[str, Any], scope: str = "project") -> tuple[str | None, list[str]]:
    """Adds or replaces a rule in the scope's custom pack. Returns (pack path, problems)."""
    rule, problems = validate_rule(raw)
    if rule is None:
        return None, problems
    path = os.path.join(scope_dir(workspace, scope), CUSTOM_PACK)
    owner = Standards(workspace).get(rule["id"])
    if owner is not None and os.path.abspath(owner["file"]) != os.path.abspath(path):
        return None, [f"`{rule['id']}` is defined in `{owner['file']}`; edit that file, or pick another id"]
    data = _read_toml(path) if os.path.exists(path) else {"pack": {"name": f"{scope}-custom",
                                                                  "description": f"Rules added via MCP ({scope})"}}
    stored = {k: rule[k] for k in ("id", "title", "severity", "enforce", "category", *_LIST_FIELDS,
                                   "pattern", "require_keyword", "replace_call", "replace_import", "add_keyword",
                                   "replacement", "guideline")}
    rules = [r for r in data.get("rule") or [] if r.get("id") != rule["id"]] + [stored]
    data["rule"] = rules
    _write_pack(path, data)
    return path, []


def remove_rule(workspace: str, rule_id: str, scope: str = "project") -> tuple[bool, str]:
    path = os.path.join(scope_dir(workspace, scope), CUSTOM_PACK)
    if not os.path.exists(path):
        owner = Standards(workspace).get(rule_id)
        return False, (f"`{rule_id}` is defined in `{owner['file']}`; edit that file" if owner
                       else f"No rule `{rule_id}` in the {scope} standards")
    data = _read_toml(path)
    rules = data.get("rule") or []
    kept = [r for r in rules if r.get("id") != rule_id]
    if len(kept) == len(rules):
        owner = Standards(workspace).get(rule_id)
        return False, (f"`{rule_id}` is defined in `{owner['file']}`; edit that file" if owner
                       else f"No rule `{rule_id}` in `{path}`")
    data["rule"] = kept
    _write_pack(path, data)
    return True, path


def install_pack(workspace: str, name: str, scope: str = "project") -> tuple[str | None, str]:
    packs = builtin_packs()
    if name not in packs:
        return None, f"Unknown pack `{name}`. Built-in packs: {', '.join(packs) or 'none'}"
    target = os.path.join(scope_dir(workspace, scope), f"{name}.toml")
    if os.path.exists(target):
        return None, f"`{target}` already exists; edit it, or delete it to reinstall"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(packs[name], encoding="utf-8") as src, open(target, "w", encoding="utf-8", newline="\n") as dst:
        dst.write(src.read())
    return target, ""
