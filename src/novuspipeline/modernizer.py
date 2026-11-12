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

One rewrite is a deliberate exception: a bare `except:` becomes
`except Exception:`, which stops swallowing KeyboardInterrupt / SystemExit. It
is applied only where nothing in the `try` block shows the code relies on
catching those (no await/yield, no exit calls, no re-raise); a handler that
re-raises becomes the exactly equivalent `except BaseException:`, and anything
else is left for manual review.

JavaScript/TypeScript has no parser here, so its rewrites are limited to `var`
declarations that a conservative scan can show behave the same as `let`.
"""

import ast
import io
import re
import tokenize
from typing import Any, ClassVar

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


# A `/` after one of these starts a regex literal; after anything else it is division.
# Operators such as `+` are left out on purpose: reading a division as a regex would hide
# the identifiers after it, and hidden identifiers are what makes a rewrite unsafe.
_REGEX_PRECEDERS = set("(,=:[!&|?{;")
_REGEX_KEYWORDS = {"return", "typeof", "instanceof", "in", "of", "new", "delete", "void", "throw", "case", "do",
                   "else", "yield", "await"}


def mask_script_literals(code: str) -> str | None:
    """
    JavaScript/TypeScript counterpart of `mask_strings_and_comments`: comments, string and
    template text and regex bodies become spaces (same length, same lines). `${...}`
    expressions inside template literals stay visible, because they are code. Returns None
    when a literal is unterminated, i.e. the scan cannot be trusted.
    """
    out = list(code)
    n = len(code)

    def blank(a: int, b: int) -> None:
        for k in range(a, min(b, n)):
            if out[k] not in "\r\n":
                out[k] = " "

    def template(start: int) -> int | None:
        """Blanks template text from `start`; returns the index after the closing backtick or `${`."""
        j = start
        while j < n:
            if code[j] == "\\":
                j += 2
            elif code[j] == "`":
                blank(start, j)
                return j + 1
            elif code.startswith("${", j):
                blank(start, j)
                open_templates.append(braces)
                return j + 2
            else:
                j += 1
        return None

    open_templates: list[int] = []  # brace depth at each `${` still open
    braces = 0
    prev, prev_word = "", ""        # last significant character / identifier, to tell regex from division
    i = 0
    while i < n:
        c = code[i]
        if c in " \t\r\n":
            i += 1
        elif code.startswith("//", i):
            j = code.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif code.startswith("/*", i):
            j = code.find("*/", i + 2)
            if j < 0:
                return None
            blank(i, j + 2)
            i = j + 2
        elif c in "\"'":
            j = i + 1
            while j < n and code[j] != c:
                if code[j] == "\n":
                    return None
                j += 2 if code[j] == "\\" else 1
            if j >= n:
                return None
            blank(i + 1, j)
            i, prev, prev_word = j + 1, c, ""
        elif c == "`":
            j = template(i + 1)
            if j is None:
                return None
            i, prev, prev_word = j, "`", ""
        elif c == "/" and (not prev or prev in _REGEX_PRECEDERS or prev_word in _REGEX_KEYWORDS):
            j, in_class = i + 1, False
            while j < n and code[j] != "\n" and (in_class or code[j] != "/"):
                if code[j] == "\\":
                    j += 1
                elif code[j] == "[":
                    in_class = True
                elif code[j] == "]":
                    in_class = False
                j += 1
            if j < n and code[j] == "/":
                blank(i + 1, j)
                i, prev, prev_word = j + 1, "/", ""
            else:  # no closing slash on this line: an operator after all
                i, prev, prev_word = i + 1, "/", ""
        elif c.isalnum() or c in "_$":
            j = i
            while j < n and (code[j].isalnum() or code[j] in "_$"):
                j += 1
            i, prev, prev_word = j, "a", code[i:j]
        else:
            if c == "{":
                braces += 1
            elif c == "}":
                if open_templates and open_templates[-1] == braces:
                    open_templates.pop()
                    j = template(i + 1)
                    if j is None:
                        return None
                    i, prev, prev_word = j, "`", ""
                    continue
                braces -= 1
            i, prev, prev_word = i + 1, c, ""
    return None if open_templates else "".join(out)


def _sub_outside_literals(pattern: str, repl, code: str, flags: int = re.MULTILINE) -> tuple[str, int]:
    """re.subn that only matches code, never text inside strings or comments."""
    masked = mask_strings_and_comments(code)
    matches = list(re.finditer(pattern, masked, flags))
    for m in reversed(matches):
        original = code[m.start():m.end()]
        code = code[:m.start()] + re.sub(pattern, repl, original, count=1, flags=flags) + code[m.end():]
    return code, len(matches)


def _parse(code: str) -> ast.AST | None:
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def top_level_names(code: str, include_imports: bool = True) -> set | None:
    """
    Names a module defines (and, with `include_imports`, imports) at top level — what
    `from mod import x` can see. None if unparseable. Use include_imports=False for the
    module's own definitions: dropping an obsolete `import urllib2` is not an API break.
    """
    tree = _parse(code)
    if tree is None:
        return None
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if include_imports:
                names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        else:  # names defined under module-level `if` / `try`
            names.update(n.name for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
            names.update(n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    return names


def missing_names(new_code: str, required: list[str]) -> list[str]:
    """Names in `required` (e.g. symbols other modules import) that `new_code` no longer provides."""
    names = top_level_names(new_code)
    if names is None or "*" in names:
        return []
    return sorted(n for n in required if n not in names)


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

    SMELL_PATTERNS: ClassVar[list[dict[str, Any]]] = [
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
    _BY_ID: ClassVar[dict[str, dict[str, Any]]] = {p["id"]: p for p in SMELL_PATTERNS}

    @classmethod
    def _finding(cls, smell_id: str, line_number: int, line_content: str, file_path: str) -> dict[str, Any]:
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
    def scan_code(cls, code: str, file_path: str = "") -> list[dict[str, Any]]:
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
        # Scripts: text inside strings and comments is not code either ("var" in a sentence).
        masked = mask_strings_and_comments(code) if is_python else (mask_script_literals(code) or code)
        original_lines = code.split("\n")

        found: dict[tuple[str, int], dict[str, Any]] = {}
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
# Names whose presence in a `try` block means a bare `except:` may be there to catch something
# `except Exception:` would let through (argparse's parse_args exits with SystemExit).
_NON_EXCEPTION_SIGNS = {"SystemExit", "KeyboardInterrupt", "GeneratorExit", "BaseException", "BaseExceptionGroup",
                        "CancelledError", "exit", "quit", "parse_args", "parse_known_args", "parse_intermixed_args"}
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _walk_same_scope(body: list[ast.stmt]):
    """Nodes that run as part of `body` itself (nested function and class bodies are skipped)."""
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def _relies_on_base_exceptions(body: list[ast.stmt]) -> bool:
    for node in _walk_same_scope(body):
        if isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith, ast.Yield, ast.YieldFrom)):
            return True  # cancellation and generator close arrive as BaseException subclasses
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        if name in _NON_EXCEPTION_SIGNS:
            return True
    return False


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
    def _migrate_urllib2(cls, code: str, changes: list[str]) -> str:
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
        imports: list[str] = []
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
    def _migrate_print_statements(cls, code: str, changes: list[str]) -> str:
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
    def _narrow_bare_excepts(cls, code: str, changes: list[str]) -> str:
        """
        `except:` -> `except Exception:` where the `try` block gives no sign of relying on
        catching KeyboardInterrupt / SystemExit / GeneratorExit / CancelledError, and
        -> `except BaseException:` (identical behavior) where the handler re-raises.
        """
        bare_line = re.compile(r"^([ \t]*)except[ \t]*:")
        tree = _parse(code)
        if tree is None:
            # Not valid Python 3 (yet): no AST to judge each handler by, so judge the whole file.
            masked = mask_strings_and_comments(code)
            if not re.search(bare_line.pattern, masked, re.MULTILINE):
                return code
            names = "|".join(sorted(_NON_EXCEPTION_SIGNS | {"await", "yield"}))
            if re.search(rf"\b(?:{names})\b|^[ \t]*raise[ \t]*$", masked, re.MULTILINE):
                changes.append("Manual review: bare `except:` left as-is; the file exits, re-raises or awaits, "
                               "so narrowing it could change behavior.")
                return code
            code, n = _sub_outside_literals(bare_line.pattern, r"\1except Exception:", code)
            changes.append(f"Replaced {n} bare `except:` with `except Exception:` "
                           "(KeyboardInterrupt / SystemExit are no longer swallowed).")
            return code

        lines = code.split("\n")
        narrowed = reraising = skipped = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                if handler.type is not None or not bare_line.match(lines[handler.lineno - 1]):
                    continue
                if any(isinstance(n, ast.Raise) and n.exc is None for n in _walk_same_scope(handler.body)):
                    replacement, reraising = "BaseException", reraising + 1
                elif _relies_on_base_exceptions(node.body) or _relies_on_base_exceptions(handler.body):
                    skipped += 1
                    continue
                else:
                    replacement, narrowed = "Exception", narrowed + 1
                lines[handler.lineno - 1] = bare_line.sub(rf"\1except {replacement}:", lines[handler.lineno - 1], count=1)
        if narrowed:
            changes.append(f"Replaced {narrowed} bare `except:` with `except Exception:` "
                           "(KeyboardInterrupt / SystemExit are no longer swallowed).")
        if reraising:
            changes.append(f"Replaced {reraising} re-raising bare `except:` with the equivalent `except BaseException:`.")
        if skipped:
            changes.append(f"Manual review: {skipped} bare `except:` left as-is; the guarded code awaits, yields or "
                           "exits, so it may rely on catching cancellation or exit.")
        return "\n".join(lines)

    @classmethod
    def modernize_python(cls, code: str) -> tuple[str, list[str]]:
        """Modernizes legacy Python code while preserving runtime behavior."""
        changes: list[str] = []
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

        updated = cls._narrow_bare_excepts(updated, changes)

        if re.search(r"\bc?[Pp]ickle\.loads?\(", mask_strings_and_comments(updated)):
            changes.append("Manual review: pickle deserialization has no behavior-preserving automatic "
                           "replacement; migrate to json/pydantic by hand.")

        if _parse(updated) is not None:
            from . import ruff_backend
            updated, fixed = ruff_backend.apply_safe_fixes(updated, "module.py")
            if fixed:
                changes.append(f"Applied Ruff safe fixes (behavior-preserving): {', '.join(fixed)}.")

        if parsed_before and _parse(updated) is None:
            return code, ["Aborted: transformed code no longer parses; original left unchanged."]
        return updated, changes

    @classmethod
    def modernize_typescript(cls, code: str, file_path: str = "") -> tuple[str, list[str]]:
        """
        Modernizes legacy JavaScript/TypeScript code. Without a parser, only `var`
        declarations that `_let_equivalent_vars` can show behave the same as `let` are
        rewritten; `require()` is reported, never rewritten (an `import` is hoisted, needs
        an ES module, and binds a default export differently).
        """
        changes: list[str] = []
        masked = mask_script_literals(code)
        if masked is None:
            return code, ["Manual review: the file has an unterminated string, template or comment; nothing was rewritten."]

        total = len(re.findall(r"\bvar\b", masked))
        if total:
            reason = ""
            if file_path.lower().endswith((".jsx", ".tsx")):
                reason = "JSX text cannot be told apart from code without a parser"
            elif re.search(r"\b(?:eval|with)\s*\(", masked):
                reason = "the file uses `eval` or `with`, which can reach any variable"
            positions = [] if reason else _let_equivalent_vars(masked, file_path)
            updated = code
            for p in sorted(positions, reverse=True):
                updated = updated[:p] + "let" + updated[p + 3:]  # `let`, not `const`: it may be reassigned
            code = updated
            if positions:
                changes.append(f"Replaced {len(positions)} ES5 `var` declaration(s) with block-scoped `let` "
                               "(each declared once in a function body or module, and never used before its "
                               "declaration or outside that body).")
            if total > len(positions):
                changes.append(f"Manual review: {total - len(positions)} `var` declaration(s) left as-is: "
                               + (reason or "`let` cannot be shown to behave the same (hoisting, redeclaration, a "
                                            "loop or other block, or a script-level global)") + ".")

        requires = len(re.findall(r"\b(?:const|let|var)\s+[\w$]+\s*=\s*require\(", masked))
        if requires:
            changes.append(f"Manual review: {requires} CommonJS `require()` import(s); converting to `import` changes "
                           "module semantics, so migrate the module as a whole.")
        return code, changes


def _let_equivalent_vars(masked: str, file_path: str = "") -> list[int]:
    """
    Offsets of `var` keywords (in JS/TS source masked by `mask_script_literals`) that can
    become `let` without changing behavior. A declaration qualifies only if all hold:
      - it is a statement of its own declaring one plain name, ending in `;`
      - the name is declared nowhere else, and its first mention in the file is the
        declaration (so no hoisted use, no parameter or import of the same name)
      - the name is not used in its own initializer (`var a = a || {}`)
      - it sits directly in a function body, or at the top level of a module: not in an
        `if`/loop/`try` block (a `let` there is scoped to the block and, in a loop, is a
        new binding per iteration) and not at the top level of a classic script, where
        `var` creates a global
      - every mention is inside that function body
      - no named function could read it early: a `function f() {}` declaration is hoisted
        and may run before the `var` line, where `let` throws instead of giving undefined
    """
    pairs: list[tuple[int, int]] = []
    stack: list[int] = []
    for m in re.finditer(r"[{}]", masked):
        if m.group() == "{":
            stack.append(m.start())
        elif not stack:
            return []
        else:
            pairs.append((stack.pop(), m.start()))
    if stack:
        return []

    def is_function_body(open_brace: int) -> bool:
        head = masked[:open_brace].rstrip()
        if head.endswith("=>"):
            return True
        if not head.endswith(")"):
            return False  # `else {`, `try {`, `do {`, a typed TS signature, ...: not provably a function
        depth = 0
        for k in range(len(head) - 1, -1, -1):
            depth += (head[k] == ")") - (head[k] == "(")
            if depth == 0:
                return re.search(r"\b(?:if|for|while|switch|catch|with|await)\s*$", head[:k]) is None
        return False

    def innermost(position: int) -> tuple[int, int]:
        inside = [(a, b) for a, b in pairs if a < position < b]
        return max(inside, key=lambda p: p[0]) if inside else (-1, len(masked))

    named_functions = [m.start() for m in re.finditer(r"\bfunction\b\s*\*?\s*[\w$]+\s*\(", masked)]

    is_module = file_path.lower().endswith((".mjs", ".cjs", ".mts", ".cts")) or re.search(
        r"^\s*(?:import|export)\b|\brequire\s*\(|\bmodule\.exports\b|\bexports\.", masked, re.MULTILINE) is not None

    safe: list[int] = []
    for m in re.finditer(r"^[ \t]*var[ \t]+([A-Za-z_$][\w$]*)", masked, re.MULTILINE):
        name, start, name_end = m.group(1), m.start() + m.group().index("var"), m.end()
        before = masked[:start].rstrip()
        if before and before[-1] not in ";{}":
            continue  # `for (var ...`, `export var`, a braceless `if`/`else`, a `case` label, ...
        # The statement must end with `;`, with no second declarator and no line break at depth 0.
        depth, end = 0, None
        for k in range(name_end, len(masked)):
            ch = masked[k]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth < 0:
                    break
            elif depth == 0 and ch in ",\n":
                break
            elif depth == 0 and ch == ";":
                end = k
                break
        if end is None:
            continue
        mentions = [x.start() for x in re.finditer(rf"(?<![\w$]){re.escape(name)}(?![\w$])", masked)]
        declared = len(re.findall(rf"\b(?:var|let|const|function|class)\s+{re.escape(name)}(?![\w$])", masked))
        if declared != 1 or mentions[0] != name_end - len(name) or any(name_end <= x <= end for x in mentions):
            continue
        lo, hi = innermost(start)
        if not (is_function_body(lo) if lo >= 0 else is_module):
            continue
        if any(not lo < x < hi for x in mentions):
            continue
        nested = any(innermost(x) != (lo, hi) for x in mentions)
        if nested and any(lo < f < hi for f in named_functions):
            continue
        safe.append(start)
    return safe
