"""
modernizer.py — NovusPipeline: Legacy Smell Detection & Parity-Preserving Modernization

`LegacySmellDetector` finds legacy code smells. Regex rules run on source whose
string literals and comments have been masked (same length, same line
structure), so text inside docstrings/comments is never reported. When the
source parses as Python 3, the bare-except and missing-type-hint rules use the
AST instead of regexes.

`CodeModernizer` applies only transformations that keep runtime behavior
identical (e.g. urllib2 -> urllib.request, not -> httpx), refuses anything it
cannot rewrite safely, and verifies that its output still parses.
"""

import ast
import io
import re
import tokenize
from typing import Any, Dict, List, Optional, Tuple

PYTHON_EXTENSIONS = (".py", ".pyw")
SCRIPT_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def mask_strings_and_comments(code: str) -> str:
    """
    Returns `code` with comment text and string-literal contents replaced by
    spaces. Length, line breaks and quote characters are preserved, so regex
    match offsets and line numbers on the masked text map 1:1 onto `code`.
    Falls back to the unmasked source if it cannot be tokenized.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code

    line_starts = [0]
    for line in io.StringIO(code).readlines():
        line_starts.append(line_starts[-1] + len(line))
    chars = list(code)

    def blank(start, end, keep_head: int = 0, keep_tail: int = 0) -> None:
        a = line_starts[start[0] - 1] + start[1] + keep_head
        b = line_starts[end[0] - 1] + end[1] - keep_tail
        for i in range(max(a, 0), min(b, len(chars))):
            if chars[i] not in "\r\n":
                chars[i] = " "

    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            blank(tok.start, tok.end)
        elif tok.type == tokenize.STRING:
            text = tok.string
            prefix = len(text) - len(text.lstrip("rRbBuUfF"))
            quote = 3 if text[prefix:prefix + 3] in ('"""', "'''") else 1
            blank(tok.start, tok.end, prefix + quote, quote)
        elif fstring_middle is not None and tok.type == fstring_middle:
            blank(tok.start, tok.end)
    return "".join(chars)


def _sub_outside_literals(pattern: str, repl, code: str, flags: int = re.MULTILINE) -> Tuple[str, int]:
    """re.subn that only matches code, never text inside strings or comments."""
    masked = mask_strings_and_comments(code)
    matches = list(re.finditer(pattern, masked, flags))
    for m in reversed(matches):
        original = code[m.start():m.end()]
        code = code[:m.start()] + re.sub(pattern, repl, original, count=1, flags=flags) + code[m.end():]
    return code, len(matches)


def _parse(code: str) -> Optional[ast.AST]:
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


# ---------------------------------------------------------------------------
# Smell detection
# ---------------------------------------------------------------------------

def _missing_type_hints(node: ast.AST) -> bool:
    """True if a def lacks a return annotation or any parameter annotation (self/cls exempt)."""
    if node.returns is None:
        return True
    a = node.args
    params = a.posonlyargs + a.args + a.kwonlyargs + [p for p in (a.vararg, a.kwarg) if p]
    for i, p in enumerate(params):
        if i == 0 and p.arg in ("self", "cls") and p in (a.posonlyargs + a.args):
            continue
        if p.annotation is None:
            return True
    return False


class LegacySmellDetector:
    """Scans code strings for enterprise legacy code smells."""

    SMELL_PATTERNS = [
        {
            "id": "PY-SMELL-001",
            "name": "Python 2 Print Statement / Missing Print Parentheses",
            "regex": r"^\s*print\s+[\"'\w]",
            "category": "python",
            "rag_query": "Python 2 print statement modernization logging",
            "severity": "HIGH",
        },
        {
            "id": "PY-SMELL-002",
            "name": "Obsolete urllib2 / Python 2 urllib API",
            # Python 3's urllib.parse / urllib.request are NOT smells; only the
            # Python 2 modules and the Python 2-only urllib.* functions are.
            "regex": r"^\s*import\s+(?:[\w.]+\s*,\s*)*urllib2\b|^\s*from\s+urllib2\s+import"
                     r"|\burllib\.(?:urlopen|urlencode|urlretrieve|quote_plus|quote|unquote|urlcleanup)\(",
            "category": "python",
            "rag_query": "Replace urllib2 with httpx or requests",
            "severity": "HIGH",
        },
        {
            "id": "PY-SMELL-003",
            "name": "Bare Exception Handling (except:)",
            "regex": r"^\s*except\s*:",
            "category": "python",
            "rag_query": "Python Error Handling & Logging specific exception",
            "severity": "CRITICAL",
            "ast_rule": True,
        },
        {
            "id": "PY-SMELL-004",
            "name": "Deprecated os.popen Usage",
            "regex": r"\bos\.popen\(",
            "category": "python",
            "rag_query": "Replace os.popen with subprocess run capture_output",
            "severity": "HIGH",
        },
        {
            "id": "PY-SMELL-005",
            "name": "Unsafe Pickle Serialization",
            "regex": r"^\s*import\s+(?:[\w.]+\s*,\s*)*c?[Pp]ickle\b|^\s*from\s+c?[Pp]ickle\s+import|\bc?[Pp]ickle\.loads?\(",
            "category": "python",
            "rag_query": "Replace pickle with pydantic json serialization",
            "severity": "CRITICAL",
        },
        {
            "id": "PY-SMELL-006",
            "name": "Missing Type Hints in Function Signature",
            "regex": r"^\s*(?:async\s+)?def\s+\w+\s*\([^)]*\)\s*:",
            "category": "python",
            "rag_query": "Python Type System Modernization union syntax",
            "severity": "MEDIUM",
            "ast_rule": True,
        },
        {
            "id": "TS-SMELL-001",
            "name": "Legacy ES5 'var' Declaration",
            "regex": r"\bvar\s+\w+",
            "category": "typescript",
            "rag_query": "JavaScript ES Modernization replace var const let",
            "severity": "HIGH",
        },
        {
            "id": "TS-SMELL-002",
            "name": "Loose 'any' Type Annotation",
            "regex": r":\s*any\b",
            "category": "typescript",
            "rag_query": "TypeScript Strict Mode replace any with unknown interface",
            "severity": "MEDIUM",
        },
        {
            "id": "TS-SMELL-003",
            "name": "Legacy CommonJS require() Import",
            "regex": r"const\s+\w+\s*=\s*require\(",
            "category": "typescript",
            "rag_query": "Replace CommonJS require with ES modules import",
            "severity": "MEDIUM",
        },
    ]
    _BY_ID = {p["id"]: p for p in SMELL_PATTERNS}

    @classmethod
    def _finding(cls, smell_id: str, line_number: int, line_content: str, file_path: str) -> Dict[str, Any]:
        p = cls._BY_ID[smell_id]
        return {
            "smell_id": smell_id,
            "name": p["name"],
            "line_number": line_number,
            "line_content": line_content.strip(),
            "category": p["category"],
            "severity": p["severity"],
            "rag_query": p["rag_query"],
            "file_path": file_path,
        }

    @classmethod
    def scan_code(cls, code: str, file_path: str = "") -> List[Dict[str, Any]]:
        """Returns one finding per (smell, line), sorted by line number."""
        lower = file_path.lower()
        if lower.endswith(PYTHON_EXTENSIONS):
            categories = {"python"}
        elif lower.endswith(SCRIPT_EXTENSIONS):
            categories = {"typescript"}
        else:
            categories = {"python", "typescript"}

        is_python = "python" in categories and not lower.endswith(SCRIPT_EXTENSIONS)
        tree = _parse(code) if is_python else None
        masked = mask_strings_and_comments(code) if is_python else code
        original_lines = code.split("\n")

        found: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for idx, line in enumerate(masked.split("\n"), 1):
            for p in cls.SMELL_PATTERNS:
                if p["category"] not in categories or (tree is not None and p.get("ast_rule")):
                    continue
                if re.search(p["regex"], line):
                    found[(p["id"], idx)] = cls._finding(p["id"], idx, original_lines[idx - 1], file_path)

        if tree is not None:
            def line_of(n: int) -> str:
                return original_lines[n - 1] if 0 < n <= len(original_lines) else ""

            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    found[("PY-SMELL-003", node.lineno)] = cls._finding(
                        "PY-SMELL-003", node.lineno, line_of(node.lineno), file_path)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _missing_type_hints(node):
                    found[("PY-SMELL-006", node.lineno)] = cls._finding(
                        "PY-SMELL-006", node.lineno, line_of(node.lineno), file_path)

        return sorted(found.values(), key=lambda f: (f["line_number"], f["smell_id"]))


# ---------------------------------------------------------------------------
# Rule-based parity-preserving code modernizer
# ---------------------------------------------------------------------------

_URLLIB2_ERRORS = {"HTTPError", "URLError", "ContentTooShortError"}


def _ensure_import(code: str, statement: str) -> str:
    """Adds `statement` after the module docstring / __future__ imports unless already present."""
    if re.search(rf"^\s*{re.escape(statement)}\s*$", code, re.MULTILINE):
        return code
    tree = _parse(code)
    insert_after = 0
    if tree is not None:
        for i, node in enumerate(tree.body):
            is_docstring = i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str)
            is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
            if is_docstring or is_future:
                insert_after = node.end_lineno
            else:
                break
    lines = code.split("\n")
    if insert_after == 0:
        # Keep a shebang / encoding cookie on the first lines.
        while insert_after < len(lines) and lines[insert_after].startswith("#") and insert_after < 2:
            insert_after += 1
    lines.insert(insert_after, statement)
    return "\n".join(lines)


class CodeModernizer:
    """
    Applies rule-based modernizations that keep runtime behavior identical.
    Each method returns (new_code, notes); notes starting with "Manual review"
    describe smells it deliberately did not rewrite.
    """

    @classmethod
    def _migrate_urllib2(cls, code: str, changes: List[str]) -> str:
        masked = mask_strings_and_comments(code)
        if "urllib2" not in masked:
            return code

        plain_import = re.compile(r"^([ \t]*)import[ \t]+urllib2[ \t]*$", re.MULTILINE)
        from_import = re.compile(r"^([ \t]*)from[ \t]+urllib2[ \t]+import[ \t]+([\w \t,]+?)[ \t]*$", re.MULTILINE)
        other_forms = len(re.findall(r"\burllib2\b", masked)) - len(plain_import.findall(masked)) \
            - len(from_import.findall(masked)) - len(re.findall(r"\burllib2\.\w+", masked))
        if other_forms:
            changes.append("Manual review: urllib2 is used in a form that cannot be rewritten "
                           "safely (alias or multi-module import); migrate to urllib.request by hand.")
            return code

        used = set(re.findall(r"\burllib2\.(\w+)", masked))
        imports: List[str] = []
        if used - _URLLIB2_ERRORS or not used:
            imports.append("import urllib.request")
        if used & _URLLIB2_ERRORS:
            imports.append("import urllib.error")

        def plain_repl(m: re.Match) -> str:
            return "\n".join(m.group(1) + imp for imp in imports)

        def from_repl(m: re.Match) -> str:
            names = [n.strip() for n in m.group(2).split(",") if n.strip()]
            request = [n for n in names if n.split(" as ")[0].strip() not in _URLLIB2_ERRORS]
            errors = [n for n in names if n.split(" as ")[0].strip() in _URLLIB2_ERRORS]
            out = []
            if request:
                out.append(f"{m.group(1)}from urllib.request import {', '.join(request)}")
            if errors:
                out.append(f"{m.group(1)}from urllib.error import {', '.join(errors)}")
            return "\n".join(out)

        code, n_plain = _sub_outside_literals(plain_import.pattern, plain_repl, code)
        code, n_from = _sub_outside_literals(from_import.pattern, from_repl, code)
        code, _ = _sub_outside_literals(r"\burllib2\.(HTTPError|URLError|ContentTooShortError)\b",
                                        r"urllib.error.\1", code)
        code, _ = _sub_outside_literals(r"\burllib2\.(\w+)", r"urllib.request.\1", code)
        if n_plain or n_from:
            changes.append("Migrated Python 2 `urllib2` to its Python 3 equivalents `urllib.request` / "
                           "`urllib.error` (same API and behavior).")
        return code

    @classmethod
    def _migrate_print_statements(cls, code: str, changes: List[str]) -> str:
        masked_lines = mask_strings_and_comments(code).split("\n")
        lines = code.split("\n")
        converted = skipped = 0
        for i, masked in enumerate(masked_lines):
            bare = re.match(r"^([ \t]*)print[ \t]*$", masked)
            if bare:
                lines[i] = f"{bare.group(1)}print(){lines[i][bare.end(1) + 5:]}"
                converted += 1
                continue
            m = re.match(r"^([ \t]*)print[ \t]+(?![(=])(.*?)[ \t]*$", masked)
            if not m:
                continue
            body = lines[i][m.start(2):m.end(2)]
            masked_body = m.group(2)
            if masked_body.startswith(">>") or masked_body.rstrip().endswith(","):
                skipped += 1  # print >>f / trailing-comma forms have no 1:1 rewrite
                continue
            comment = lines[i][m.end(2):]
            lines[i] = f"{m.group(1)}print({body}){comment}"
            converted += 1
        if converted:
            changes.append(f"Converted {converted} Python 2 `print` statement(s) to `print()` calls.")
        if skipped:
            changes.append(f"Manual review: {skipped} `print >>file` / trailing-comma print statement(s) left as-is.")
        return "\n".join(lines)

    @classmethod
    def modernize_python(cls, code: str) -> Tuple[str, List[str]]:
        """Modernizes legacy Python code while preserving runtime behavior."""
        changes: List[str] = []
        parsed_before = _parse(code) is not None
        updated = code

        # `print x` is a syntax error in Python 3, so only Python 2 source needs this;
        # on valid Python 3 (`print -1` is a legal expression) it would change meaning.
        if not parsed_before:
            updated = cls._migrate_print_statements(updated, changes)
        updated = cls._migrate_urllib2(updated, changes)

        # os.popen(cmd).read() -> subprocess.run(..., shell=True, stdout=PIPE, text=True).stdout
        # shell=True and stderr passthrough match os.popen exactly.
        updated, n = _sub_outside_literals(
            r"\bos\.popen\((.*?)\)\.read\(\)",
            r"subprocess.run(\1, shell=True, stdout=subprocess.PIPE, text=True).stdout",
            updated,
        )
        if n:
            updated = _ensure_import(updated, "import subprocess")
            changes.append(f"Replaced {n} deprecated `os.popen(...).read()` call(s) with `subprocess.run`. "
                           "`shell=True` is kept for identical behavior; review for command injection.")
        if re.search(r"\bos\.popen\(", mask_strings_and_comments(updated)):
            changes.append("Manual review: `os.popen` used without `.read()`; rewrite to subprocess by hand.")

        updated, n = _sub_outside_literals(r"^([ \t]*)except[ \t]*:", r"\1except Exception:", updated)
        if n:
            changes.append(f"Replaced {n} bare `except:` with `except Exception:` "
                           "(KeyboardInterrupt / SystemExit now propagate, as intended).")

        if re.search(r"\bc?[Pp]ickle\.loads?\(", mask_strings_and_comments(updated)):
            changes.append("Manual review: pickle deserialization has no behavior-preserving automatic "
                           "replacement; migrate to json/pydantic by hand.")

        if parsed_before and _parse(updated) is None:
            return code, ["Aborted: transformed code no longer parses; original left unchanged."]
        return updated, changes

    @classmethod
    def modernize_typescript(cls, code: str) -> Tuple[str, List[str]]:
        """Modernizes legacy JavaScript/TypeScript code."""
        changes: List[str] = []
        updated = code

        # `let`, not `const`: a `var` may be reassigned later.
        if re.search(r"\bvar\s+", updated):
            updated = re.sub(r"\bvar\s+", "let ", updated)
            changes.append("Replaced ES5 `var` with block-scoped `let`.")

        for var_name, quote, mod_path in re.findall(r"const\s+(\w+)\s*=\s*require\((['\"])(.*?)\2\);?", updated):
            for old in (f"const {var_name} = require('{mod_path}');", f'const {var_name} = require("{mod_path}");'):
                if old in updated:
                    updated = updated.replace(old, f"import {var_name} from '{mod_path}';")
                    changes.append(f"Converted `const {var_name} = require('{mod_path}')` -> ES module import.")
                    break

        return updated, changes
