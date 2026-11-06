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
        min_python: ClassVar[str | None] = None   # Python version in which the old API is removed
        guideline: ClassVar[str] = ""
        rag_query: ClassVar[str] = "Python Library Modernization"

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
        if self._needs_abc_import:
            return _insert_after_import(module, "collections", "import collections.abc")
        return module


class Base64Legacy(Migration):
    id = "MIG-BASE64"
    title = "`base64.encodestring` / `decodestring` (removed in Python 3.9)"
    severity = "HIGH"
    min_python = "3.9"
    guideline = "`base64.encodestring` and `decodestring` were renamed `encodebytes` and `decodebytes` with identical behavior."
    _MAP: ClassVar[dict[str, str]] = {"encodestring": "encodebytes", "decodestring": "decodebytes"}

    def __init__(self, fix: bool = False) -> None:
        super().__init__(fix)
        self._bound = False

    def visit_Module(self, node: cst.Module) -> None:
        self._bound = "base64" in _top_level_imports(node)

    def leave_Attribute(self, original: cst.Attribute, updated: cst.Attribute) -> cst.BaseExpression:
        if isinstance(original.value, cst.Name) and original.value.value == "base64" \
                and original.attr.value in self._MAP:
            self.hit(original, fixable=self._bound)
            if self.fix and self._bound:
                return updated.with_changes(attr=cst.Name(self._MAP[original.attr.value]))
        return updated


class LoggingWarn(Migration):
    id = "MIG-LOGGING-WARN"
    title = "`logging.warn` (deprecated since 3.3, removed in Python 3.13)"
    severity = "MEDIUM"
    min_python = "3.13"
    guideline = "`logging.warn(...)` is an alias of `logging.warning(...)`; the alias is removed in 3.13."

    def __init__(self, fix: bool = False) -> None:
        super().__init__(fix)
        self._bound = False

    def visit_Module(self, node: cst.Module) -> None:
        self._bound = "logging" in _top_level_imports(node)

    def leave_Attribute(self, original: cst.Attribute, updated: cst.Attribute) -> cst.BaseExpression:
        if isinstance(original.value, cst.Name) and original.value.value == "logging" \
                and original.attr.value == "warn":
            self.hit(original, fixable=self._bound)
            if self.fix and self._bound:
                return updated.with_changes(attr=cst.Name("warning"))
        return updated


class _DetectOnly(Migration):
    """Detection of dotted-name uses that have no behavior-identical replacement."""

    targets: ClassVar[dict[str, str]] = {}

    def leave_Attribute(self, original: cst.Attribute, updated: cst.BaseExpression) -> cst.BaseExpression:
        dotted = _dotted(original)
        if dotted in self.targets:
            self.hit(original, fixable=False, detail=self.targets[dotted])
        return updated


class RemovedStdlibApis(_DetectOnly):
    id = "MIG-REMOVED-APIS"
    title = "Standard-library API removed in a recent Python (manual migration)"
    severity = "HIGH"
    min_python = "3.11"
    guideline = "These APIs no longer exist; the replacement changes arguments or semantics, so migrate by hand."
    targets: ClassVar[dict[str, str]] = {
        "inspect.getargspec": "use inspect.getfullargspec or inspect.signature (removed in 3.11)",
        "time.clock": "use time.perf_counter or time.process_time (removed in 3.8)",
        "asyncio.coroutine": "use `async def` (removed in 3.11)",
        "os.errno": "use the errno module",
        "string.letters": "use string.ascii_letters",
        "sys.maxint": "use sys.maxsize",
        "array.array.tostring": "use tobytes",
        "unittest.makeSuite": "use unittest.TestLoader().loadTestsFromTestCase (removed in 3.13)",
        "locale.format": "use locale.format_string (removed in 3.12)",
        "ssl.wrap_socket": "use ssl.SSLContext.wrap_socket (removed in 3.12)",
        "imp.load_source": "use importlib (module removed in 3.12)",
    }


class NaiveUtcNow(_DetectOnly):
    id = "MIG-DATETIME-UTCNOW"
    title = "Naive UTC datetimes (`utcnow` / `utcfromtimestamp`, deprecated in Python 3.12)"
    severity = "MEDIUM"
    min_python = "3.12"
    guideline = ("`datetime.utcnow()` returns a naive datetime. `datetime.now(timezone.utc)` is timezone-aware, "
                 "which changes comparison and serialization behavior, so it is not rewritten automatically.")
    targets: ClassVar[dict[str, str]] = {
        "datetime.utcnow": "use datetime.now(timezone.utc) after checking naive/aware comparisons",
        "datetime.datetime.utcnow": "use datetime.datetime.now(datetime.timezone.utc)",
        "datetime.utcfromtimestamp": "use datetime.fromtimestamp(ts, timezone.utc)",
        "datetime.datetime.utcfromtimestamp": "use datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def available() -> bool:
    return cst is not None


def _select(only: list[str] | None, ignore: list[str] | None) -> list[type[Migration]]:
    return [m for mid, m in REGISTRY.items()
            if (not only or mid in only) and not (ignore and mid in ignore)]


def _run(code: str, migration: type[Migration], fix: bool) -> tuple[str, list[Hit]]:
    module = cst.parse_module(code)
    wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
    t = migration(fix=fix)
    new_module = wrapper.visit(t)
    new_module = t.finish(new_module) if fix else new_module
    return new_module.code, t.hits


def scan(code: str, file_path: str = "", only: list[str] | None = None,
         ignore: list[str] | None = None) -> list[dict[str, Any]]:
    """Findings (same shape as the rule engine's) for one source; [] if it is not valid Python 3."""
    if cst is None or (file_path and not file_path.lower().endswith((".py", ".pyw"))):
        return []
    try:
        cst.parse_module(code)
    except Exception:  # not valid Python 3 (e.g. Python 2 source): nothing to migrate yet
        return []
    lines = code.split("\n")
    findings: list[dict[str, Any]] = []
    for m in _select(only, ignore):
        try:
            _, hits = _run(code, m, fix=False)
        except Exception:  # a migration bug must never break an audit
            continue
        for h in hits:
            findings.append({
                "smell_id": m.id, "name": m.title + (f": {h.detail}" if h.detail else ""),
                "line_number": h.line, "line_content": lines[h.line - 1].strip() if 0 < h.line <= len(lines) else "",
                "category": "python", "severity": m.severity, "rag_query": m.rag_query,
                "file_path": file_path, "source": "migration", "fix": "safe" if h.fixable else None,
                "guideline": m.guideline, "min_python": m.min_python,
            })
    return sorted(findings, key=lambda f: (f["line_number"], f["smell_id"]))


def apply(code: str, only: list[str] | None = None, ignore: list[str] | None = None) -> tuple[str, list[str]]:
    """Applies every safe fix; returns (new_code, notes). Unparseable input is returned unchanged."""
    if cst is None:
        return code, []
    notes: list[str] = []
    current = code
    for m in _select(only, ignore):
        try:
            new, hits = _run(current, m, fix=True)
            cst.parse_module(new)  # the result must still be valid Python
        except Exception:
            continue
        fixed = sum(1 for h in hits if h.fixable)
        manual = len(hits) - fixed
        if fixed and new != current:
            current = new
            notes.append(f"{m.title}: rewrote {fixed} use(s) ({m.id}).")
        if manual:
            notes.append(f"Manual review: {manual} use(s) of {m.title} ({m.id}) need a manual change.")
    return current, notes


def catalog() -> list[dict[str, Any]]:
    """Every registered migration, for listings."""
    return [{"id": m.id, "title": m.title, "severity": m.severity, "min_python": m.min_python,
             "guideline": m.guideline,
             "fixable": _DetectOnly not in m.__mro__}
            for m in REGISTRY.values()]
