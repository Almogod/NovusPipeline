"""
ruff_backend.py — NovusPipeline: Ruff as the broad detection and safe-fix engine

The built-in rule engine (modernizer.py) covers a handful of legacy smells and
is the GNN's training teacher, so it stays small and stable. Ruff adds hundreds
of maintained rules (pyupgrade, bandit, bugbear, pyflakes, ...) and marks each
auto-fix as safe or unsafe; only safe fixes are ever applied automatically.

Ruff runs with --isolated (the analyzed project's own Ruff config is ignored)
so results are consistent for every workspace. If the Ruff binary cannot be
found, every function returns empty results and `available()` reports why.
"""

import json
import os
import re
import subprocess
from typing import Any

# Rule families scanned. Chosen for modernization value and a low false-positive rate.
DETECT_SELECT = ["UP", "S", "B", "F", "PTH", "C4", "PIE", "SIM", "PERF", "E9", "PLE"]
# Codes skipped: already covered by the built-in rules (which the GNN learns), or noisy.
IGNORE = [
    "E722",                      # bare except            -> PY-SMELL-003
    "S301", "S403",              # pickle                 -> PY-SMELL-005
    "S605",                      # os.popen with shell    -> PY-SMELL-004
    "S101",                      # assert used: fires on every test
    "S311",                      # random module: rarely a real issue
    "S603", "S607",              # every subprocess call
]
# Families whose *safe* fixes may be applied automatically. F401 (unused import) is
# excluded on purpose: removing an import can drop a name other modules re-import.
FIX_SELECT = ["UP", "C4", "PIE", "SIM"]

_SEVERITY = {"S": "HIGH", "B": "HIGH", "F": "MEDIUM", "UP": "MEDIUM", "E9": "CRITICAL", "PLE": "HIGH"}
_HIGH_F = {"F821", "F811", "F822", "F823"}  # undefined / redefined names are likely bugs
_CATEGORY = {"S": "security", "UP": "python", "F": "python", "B": "python", "E9": "python", "PLE": "python"}
_RAG_QUERY = {
    "security": "Input validation injection prevention secrets security",
    "python": "Python Library Modernization",
    "clean_code": "API Interface Design clean code simplification",
}
_TIMEOUT = 120


def _family(code: str) -> str:
    for prefix in sorted(_SEVERITY.keys() | {"PTH", "C4", "PIE", "SIM", "PERF"}, key=len, reverse=True):
        if code.startswith(prefix):
            return prefix
    return code.rstrip("0123456789")


def _ruff_bin() -> str | None:
    try:
        from ruff.__main__ import find_ruff_bin
        return find_ruff_bin()
    except (ImportError, FileNotFoundError):
        return None


def available() -> tuple[bool, str]:
    binary = _ruff_bin()
    return (True, binary) if binary else (False, 'Ruff not installed (pip install ruff)')


def severity(code: str) -> str:
    if code in _HIGH_F:
        return "HIGH"
    return _SEVERITY.get(_family(code), "LOW")


def _finding(d: dict[str, Any], file_path: str, lines: list[str]) -> dict[str, Any]:
    # Parse failures carry no rule code (older Ruff) or a non-rule label such as
    # "invalid-syntax" (Ruff 0.12+). For legacy code that usually means "not valid
    # Python 3" (e.g. Python 2 syntax), which is critical for a migration.
    is_rule = bool(re.fullmatch(r"[A-Z]+[0-9]+", d.get("code") or ""))
    code = d["code"] if is_rule else "SYNTAX"
    row = d["location"]["row"]
    category = _CATEGORY.get(_family(code), "clean_code") if is_rule else "python"
    return {
        "smell_id": f"RUFF-{code}",
        "name": d["message"].split(". ")[0] if is_rule else f"Not valid Python 3: {d['message']}",
        "line_number": row,
        "line_content": lines[row - 1].strip() if 0 < row <= len(lines) else "",
        "category": category,
        "severity": severity(code) if is_rule else "CRITICAL",
        "rag_query": _RAG_QUERY[category],
        "file_path": file_path,
        "fix": (d.get("fix") or {}).get("applicability"),
        "source": "ruff",
        "url": d.get("url"),
    }


def _run(args: list[str], stdin: str | None = None, cwd: str | None = None) -> subprocess.CompletedProcess | None:
    binary = _ruff_bin()
    if binary is None:
        return None
    try:
        return subprocess.run([binary, *args], input=stdin, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=cwd, timeout=_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return None


_CODE_RE = re.compile(r"^[A-Z]+[0-9]*$")


def _check_args(extend_select: list[str] | None = None, extend_ignore: list[str] | None = None) -> list[str]:
    """Ruff arguments; `extend_*` come from modernization standards (rule packs)."""
    select = DETECT_SELECT + [c for c in extend_select or [] if _CODE_RE.match(c)]
    ignore = IGNORE + [c for c in extend_ignore or [] if _CODE_RE.match(c)]
    return ["check", "--isolated", "--no-cache", "--exit-zero", "--output-format", "json",
            "--select", ",".join(dict.fromkeys(select)), "--ignore", ",".join(dict.fromkeys(ignore))]


def scan_code(code: str, file_path: str = "x.py", extend_select: list[str] | None = None,
              extend_ignore: list[str] | None = None) -> list[dict[str, Any]]:
    """Ruff findings for one in-memory Python source (empty if Ruff is unavailable)."""
    if not file_path.lower().endswith((".py", ".pyw")):
        return []
    res = _run([*_check_args(extend_select, extend_ignore), "--stdin-filename", os.path.basename(file_path) or "x.py", "-"], stdin=code)
    if res is None or not res.stdout.strip():
        return []
    try:
        diagnostics = json.loads(res.stdout)
    except ValueError:
        return []
    lines = code.split("\n")
    return [_finding(d, file_path, lines) for d in diagnostics]


def scan_files(root: str, rel_paths: list[str], extend_select: list[str] | None = None,
               extend_ignore: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Ruff findings for many workspace files in one process: {rel_path: findings}."""
    out: dict[str, list[dict[str, Any]]] = {p: [] for p in rel_paths}
    if not rel_paths:
        return out
    res = None
    for start in range(0, len(rel_paths), 400):  # keep command lines under OS limits
        batch = rel_paths[start:start + 400]
        res = _run([*_check_args(extend_select, extend_ignore), "--", *batch], cwd=root)
        if res is None or not res.stdout.strip():
            continue
        try:
            diagnostics = json.loads(res.stdout)
        except ValueError:
            continue
        cache: dict[str, list[str]] = {}
        for d in diagnostics:
            rel = os.path.relpath(d["filename"], root).replace(os.sep, "/")
            if rel not in out:
                continue
            if rel not in cache:
                try:
                    with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as f:
                        cache[rel] = f.read().split("\n")
                except OSError:
                    cache[rel] = []
            out[rel].append(_finding(d, rel, cache[rel]))
    return out


def apply_safe_fixes(code: str, file_path: str = "x.py") -> tuple[str, list[str]]:
    """Applies Ruff's *safe* fixes for FIX_SELECT; returns (new_code, list of fixed rule codes)."""
    if not file_path.lower().endswith((".py", ".pyw")):
        return code, []
    before = {f["smell_id"] for f in scan_code(code, file_path) if f["fix"] == "safe"
              and _family(f["smell_id"][5:]) in FIX_SELECT}
    if not before:
        return code, []
    res = _run(["check", "--isolated", "--no-cache", "--exit-zero", "--fix", "--quiet",
                "--select", ",".join(FIX_SELECT), "--stdin-filename", os.path.basename(file_path) or "x.py", "-"],
               stdin=code)
    if res is None or res.returncode not in (0, 1) or not res.stdout:
        return code, []
    fixed = res.stdout
    remaining = {f["smell_id"] for f in scan_code(fixed, file_path)}
    return fixed, sorted(s for s in before if s not in remaining)
