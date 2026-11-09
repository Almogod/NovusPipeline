"""
cli.py — NovusPipeline from the command line, CI and pre-commit (no MCP client needed)

    novuspipeline scan [PATH ...]      report legacy code, deprecated APIs, security and standards findings
    novuspipeline fix  [PATH ...]      apply the behavior-preserving rewrites (optionally test-gated)
    novuspipeline campaign [PATH ...]  modernize the codebase as verified per-file commits on a branch
    novuspipeline migrations           list the available library/language migrations
    novuspipeline serve [...]          the MCP server (also what plain `novuspipeline` starts)

Everything runs locally and uses the same engines as the MCP tools: built-in rules, libcst
migrations, Ruff and your rule packs (`.novus/standards/*.toml`).
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
from typing import Any

from . import __version__, codemods, ruff_backend, standards
from .codebase_context import MAX_FILE_BYTES, SKIP_DIRS
from .modernizer import CodeModernizer, LegacySmellDetector

SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
SARIF_LEVEL = {"CRITICAL": "error", "HIGH": "error", "MEDIUM": "warning", "LOW": "note"}
COMMANDS = {"scan", "fix", "campaign", "migrations", "serve"}


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

def discover(root: str, paths: list[str]) -> list[str]:
    """Python files (relative to `root`, '/'-separated) under `paths`, skipping venvs, caches and build dirs."""
    found: list[str] = []
    for p in paths or ["."]:
        full = os.path.abspath(p if os.path.isabs(p) else os.path.join(root, p))
        if os.path.isfile(full):
            candidates = [full] if full.endswith((".py", ".pyw")) else []
        else:
            candidates = []
            for dirpath, dirnames, filenames in os.walk(full):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
                candidates += [os.path.join(dirpath, n) for n in sorted(filenames) if n.endswith((".py", ".pyw"))]
        for c in candidates:
            try:
                if os.path.getsize(c) <= MAX_FILE_BYTES:
                    found.append(os.path.relpath(c, root).replace(os.sep, "/"))
            except OSError:
                continue
    return sorted(set(found))


def _read(path: str) -> tuple[str, str]:
    """(text with \\n newlines, original newline style)."""
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8", errors="replace")
    newline = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), newline


def _write(path: str, text: str, newline: str) -> None:
    with open(path, "wb") as f:
        f.write(text.replace("\n", newline).encode("utf-8"))


def _git_dirty(root: str, rel: str) -> bool:
    try:
        res = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=root, capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return res.returncode == 0 and bool(res.stdout.strip())


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

def migration_selection(root: str, args: argparse.Namespace) -> dict[str, Any]:
    """`.novus/config.toml` [migrations], overridden by command-line flags."""
    sel = codemods.settings(root)
    for key, flag in (("only", "migration"), ("ignore", "skip"), ("packs", "pack")):
        if getattr(args, flag, None):
            sel[key] = getattr(args, flag)
    if getattr(args, "target_python", None):
        sel["target_python"] = args.target_python
    return sel


def collect_findings(root: str, files: list[str], use_ruff: bool = True, use_standards: bool = True,
                     migrations: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Findings for `files`, each tagged with `file_path` (relative) and `source`."""
    active = standards.load(root) if use_standards else None
    ruff = ruff_backend.scan_files(root, files, active.ruff_select if active else None,
                                   active.ruff_ignore if active else None) if use_ruff else {}
    out: list[dict[str, Any]] = []
    for rel in files:
        code, _ = _read(os.path.join(root, rel))
        for f in LegacySmellDetector.scan_code(code, rel):
            out.append({**f, "source": "rules", "fix": None})
        out += codemods.scan(code, rel, **(migrations or {}))
        out += [{**f, "file_path": rel} for f in ruff.get(rel, [])]
        if active:
            out += [{**f, "file_path": rel} for f in active.scan(code, rel)]
    return sorted(out, key=lambda f: (f["file_path"], f["line_number"], f["smell_id"]))


def to_sarif(findings: list[dict[str, Any]]) -> dict[str, Any]:
    rules: dict[str, dict[str, Any]] = {}
    results = []
    for f in findings:
        rules.setdefault(f["smell_id"], {
            "id": f["smell_id"], "name": f["smell_id"].replace("-", " ").title().replace(" ", ""),
            "shortDescription": {"text": f["name"]},
            "properties": {"severity": f["severity"], "source": f.get("source", "")},
        })
        results.append({
            "ruleId": f["smell_id"], "level": SARIF_LEVEL.get(f["severity"], "warning"),
            "message": {"text": f["name"]},
            "locations": [{"physicalLocation": {
                "artifactLocation": {"uri": f["file_path"]},
                "region": {"startLine": max(1, f["line_number"])}}}],
        })
    return {"version": "2.1.0", "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "runs": [{"tool": {"driver": {"name": "NovusPipeline", "version": __version__,
                                          "informationUri": "https://github.com/Almogod/NovusPipeline",
                                          "rules": list(rules.values())}}, "results": results}]}


def _format_text(findings: list[dict[str, Any]], files: int) -> str:
    lines = [f"{f['file_path']}:{f['line_number']}: [{f['severity']}] {f['smell_id']} {f['name']}"
             + (" (safe fix)" if f.get("fix") == "safe" else "") for f in findings]
    by_sev = {s: sum(1 for f in findings if f["severity"] == s) for s in reversed(SEVERITIES)}
    safe = sum(1 for f in findings if f.get("fix") == "safe")
    summary = ", ".join(f"{n} {s.lower()}" for s, n in by_sev.items() if n) or "none"
    lines.append(f"\n{len(findings)} finding(s) in {files} file(s): {summary}."
                 + (f" {safe} can be fixed automatically: run `novuspipeline fix`." if safe else ""))
    return "\n".join(lines)


def cmd_scan(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.workspace)
    files = discover(root, args.paths)
    findings = collect_findings(root, files, use_ruff=not args.no_ruff, use_standards=not args.no_standards,
                                migrations=migration_selection(root, args))
    floor = SEVERITIES.index(args.min_severity)
    findings = [f for f in findings if SEVERITIES.index(f["severity"]) >= floor]
    if args.format == "json":
        print(json.dumps({"files": len(files), "findings": findings}, indent=2))
    elif args.format == "sarif":
        print(json.dumps(to_sarif(findings), indent=2))
    else:
        print(_format_text(findings, len(files)))
    if args.fail_on != "never":
        threshold = SEVERITIES.index(args.fail_on.upper())
        if any(SEVERITIES.index(f["severity"]) >= threshold for f in findings):
            return 1
    return 0


# ---------------------------------------------------------------------------
# fix
# ---------------------------------------------------------------------------

def modernize_text(code: str, rel: str, active: standards.Standards | None,
                   migrations: dict[str, Any] | None = None) -> tuple[str, list[str]]:
    new, notes = CodeModernizer.modernize_python(code, migrations)
    if active is not None:
        new, std_notes = active.apply_fixes(new, rel)
        notes += std_notes
    return new, notes


def cmd_fix(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.workspace)
    files = discover(root, args.paths)
    active = None if args.no_standards else standards.load(root)
    selection = migration_selection(root, args)
    dry = args.diff or args.check
    originals: dict[str, tuple[str, str]] = {}
    changed: list[str] = []
    skipped: list[str] = []

    for rel in files:
        path = os.path.join(root, rel)
        code, newline = _read(path)
        new, notes = modernize_text(code, rel, active, selection)
        if new == code:
            continue
        if not dry and not args.force and _git_dirty(root, rel):
            skipped.append(rel)
            continue
        changed.append(rel)
        if args.diff:
            sys.stdout.writelines(difflib.unified_diff(code.splitlines(True), new.splitlines(True),
                                                       f"a/{rel}", f"b/{rel}"))
        if not dry:
            originals[rel] = (code, newline)
            _write(path, new, newline)
        if not args.diff:
            print(f"{'would fix' if dry else 'fixed'} {rel}")
            for n in notes:
                print(f"  - {n}")

    for rel in skipped:
        print(f"skipped {rel}: uncommitted changes (commit or stash them, or pass --force)", file=sys.stderr)

    if originals and args.test_cmd:
        from . import server  # heavy import, only needed for the sandboxed test runner
        server.WORKSPACE_ROOT = root
        code_, output = server._execute_test_command(args.test_cmd)
        if code_ != 0:
            for rel, (text, newline) in originals.items():
                _write(os.path.join(root, rel), text, newline)
            baseline, _ = server._execute_test_command(args.test_cmd) if code_ is not None else (None, "")
            cause = ("the test command could not run" if code_ is None
                     else "they fail without the changes too, so fix the tests first" if baseline != 0
                     else "the changes broke them; try smaller batches (e.g. one directory) or `novuspipeline campaign`")
            print(f"\nTests failed (exit {code_}); all {len(originals)} file(s) restored: {cause}.\n{output}",
                  file=sys.stderr)
            return 2
        print(f"\nTests passed in the sandbox for {len(originals)} changed file(s).")

    print(f"\n{len(changed)} file(s) {'would change' if dry else 'changed'}, {len(files) - len(changed) - len(skipped)} "
          f"already modern" + (f", {len(skipped)} skipped." if skipped else "."))
    return 1 if (args.check and changed) else 0


# ---------------------------------------------------------------------------
# migrations / serve / entry point
# ---------------------------------------------------------------------------

def cmd_campaign(args: argparse.Namespace) -> int:
    from . import campaign

    root = os.path.abspath(args.workspace)
    if args.status:
        ledger = campaign.load_ledger(root, args.branch)
        print(campaign.format_summary(ledger, []) if ledger["files"] else f"No campaign on '{args.branch}' yet.")
        return 0
    res = campaign.run(root, args.branch, args.paths, args.test_cmd or "", args.max_files, args.retry_failed,
                       progress=print)
    print("\n" + (res.get("summary") or ""))
    if res.get("pr_draft"):
        print(f"\nPR description draft: {res['pr_draft']}")
    if not res["ok"]:
        print(f"\nError: {res['error']}", file=sys.stderr)
        return 2
    return 0


def cmd_migrations(_args: argparse.Namespace) -> int:
    for m in codemods.catalog():
        fix = "auto-fix" if m["fixable"] else "detect-only"
        py = f"py{m['min_python']}" if m["min_python"] else "-"
        print(f"{m['id']:<22} {m['pack']:<11} {py:<7} {m['severity']:<8} {fix:<11} {m['title']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="novuspipeline", description="Modernize legacy Python codebases safely.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("paths", nargs="*", help="files or directories (default: the workspace)")
        p.add_argument("--workspace", default=".", help="project root (default: current directory)")
        p.add_argument("--no-standards", action="store_true", help="ignore .novus/standards rule packs")
        p.add_argument("--target-python", metavar="X.Y",
                       help="only migrations for APIs removed by this Python version (library packs always apply)")
        p.add_argument("--pack", action="append", metavar="NAME", help="only these migration packs (repeatable)")
        p.add_argument("--migration", action="append", metavar="ID", help="only these migrations (repeatable)")
        p.add_argument("--skip", action="append", metavar="ID", help="skip these migrations (repeatable)")

    scan = sub.add_parser("scan", help="report findings", description="Report findings; non-zero exit for CI.")
    common(scan)
    scan.add_argument("--format", choices=["text", "json", "sarif"], default="text")
    scan.add_argument("--fail-on", choices=[*[s.lower() for s in SEVERITIES], "never"], default="never",
                      type=str.lower, help="exit 1 if any finding is at least this severe (default: never)")
    scan.add_argument("--min-severity", choices=SEVERITIES, default="LOW", type=str.upper)
    scan.add_argument("--no-ruff", action="store_true")
    scan.set_defaults(func=cmd_scan)

    fix = sub.add_parser("fix", help="apply behavior-preserving rewrites",
                         description="Apply safe rewrites in place. Files with uncommitted changes are skipped.")
    common(fix)
    fix.add_argument("--diff", action="store_true", help="print a unified diff, change nothing")
    fix.add_argument("--check", action="store_true", help="change nothing; exit 1 if anything would change")
    fix.add_argument("--force", action="store_true", help="also rewrite files with uncommitted changes")
    fix.add_argument("--test-cmd", help="sandboxed test command (e.g. 'python -m pytest'); all changes are "
                                        "reverted if it fails")
    fix.set_defaults(func=cmd_fix)

    camp = sub.add_parser("campaign", help="modernize the codebase as verified per-file commits on a branch",
                          description="Modernize every file (dependencies first) in a disposable worktree; each file "
                                      "is committed on BRANCH only if its tests pass. Resumable.")
    camp.add_argument("paths", nargs="*", help="limit to these files or directories (relative to the workspace)")
    camp.add_argument("--workspace", default=".", help="project root (default: current directory)")
    camp.add_argument("--branch", default="novus-campaign", help="branch receiving the commits (default: novus-campaign)")
    camp.add_argument("--test-cmd", help="test command for every file (default: the tests that import it)")
    camp.add_argument("--max-files", type=int, help="attempt at most this many changed files in this run")
    camp.add_argument("--retry-failed", action="store_true", help="retry files whose tests failed before")
    camp.add_argument("--status", action="store_true", help="show the campaign's progress and exit")
    camp.set_defaults(func=cmd_campaign)

    sub.add_parser("migrations", help="list library/language migrations").set_defaults(func=cmd_migrations)
    sub.add_parser("serve", help="run the MCP server (see `novuspipeline serve --help`)", add_help=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        from .server import main as serve_main
        serve_main(argv[1:])
        return
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help", "--version")):
        from .server import main as serve_main  # plain `novuspipeline [--workspace ...]` is the MCP server
        serve_main(argv)
        return
    args = build_parser().parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
