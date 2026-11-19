"""
Guided terminal mode: `novuspipeline` in a terminal (or `novuspipeline start`) asks for a project
folder, scans it, and fixes everything that can be fixed automatically, test-gated, on a new
branch. `novuspipeline fix PATH` runs the same flow without questions (for scripts and CI).
"""
import contextlib
import os
import sys

from . import __version__, fixer

# ---------------------------------------------------------------------------- terminal output

def _enable_ansi() -> bool:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return False
    if os.name == "nt":  # legacy consoles need virtual-terminal processing switched on
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            return False
    return True


class _Out:
    def __init__(self) -> None:
        self.color = _enable_ansi()

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def bold(self, t: str) -> str: return self._c("1", t)
    def dim(self, t: str) -> str: return self._c("2", t)
    def green(self, t: str) -> str: return self._c("32", t)
    def red(self, t: str) -> str: return self._c("31", t)
    def yellow(self, t: str) -> str: return self._c("33", t)
    def cyan(self, t: str) -> str: return self._c("36", t)

    def line(self, text: str = "") -> None:
        print(text, flush=True)

    def status(self, text: str) -> None:
        """A progress line that the next status or line overwrites (plain lines when not a terminal)."""
        if sys.stdout.isatty():
            width = max(20, _terminal_width() - 1)
            print("\r" + text[:width].ljust(width), end="", flush=True)

    def end_status(self) -> None:
        if sys.stdout.isatty():
            print("\r" + " " * max(20, _terminal_width() - 1) + "\r", end="", flush=True)


def _terminal_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


_ICONS = {fixer.FIXED: "✔", fixer.UNCHANGED: "·", fixer.DIRTY: "-"}


# ---------------------------------------------------------------------------- input

def _ask(prompt: str) -> str | None:
    try:
        return input(prompt)
    except EOFError:
        return None


def _confirm(out: _Out, question: str, default: bool) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    answer = _ask(f"{question} {out.dim(hint)} ")
    if answer is None:
        return False
    answer = answer.strip().lower()
    return default if not answer else answer in ("y", "yes")


def clean_path(raw: str) -> str:
    """A pasted path: drag-and-drop and "Copy as path" add quotes; PowerShell drag-and-drop adds `& `."""
    path = raw.strip()
    if path.startswith("& "):
        path = path[2:].strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in "\"'":
        path = path[1:-1]
    return os.path.abspath(os.path.expanduser(path)) if path else ""


def _ask_folder(out: _Out) -> str | None:
    here = os.getcwd()
    while True:
        raw = _ask(f"{out.bold('Project folder')} {out.dim('(paste a path, Enter for ' + here + ')')}: ")
        if raw is None:
            return None
        path = clean_path(raw) or here
        if os.path.isdir(path):
            return path
        out.line(out.red(f"  '{path}' is not a folder. Try again, or press Ctrl+C to quit."))


# ---------------------------------------------------------------------------- the flow

def run(srv, path: str | None = None, assume_yes: bool = False, branch: str = "", test_command: str = "",
        use_llm: bool = False, merge: bool = False, include_tests: bool = False,
        allow_untested: bool = False) -> int:
    """
    Scan a project and fix it. `path=None` asks for the folder. `assume_yes` answers every
    question with its safe default ("yes" to fixing on a branch, "no" to anything that changes
    the working copy unless `merge` is set). Returns a process exit code.
    """
    out = _Out()
    interactive = not assume_yes and sys.stdin is not None and sys.stdin.isatty()
    try:
        return _run(out, srv, path, interactive, branch, test_command, use_llm, merge, include_tests, allow_untested)
    except KeyboardInterrupt:
        out.end_status()
        out.line(out.yellow("\nStopped. Commits already made stay on the branch; your working copy is unchanged."))
        return 130


def _run(out: _Out, srv, path, interactive, branch, test_command, use_llm, merge, include_tests,
         allow_untested) -> int:
    out.line(out.bold(out.cyan(f"NovusPipeline {__version__}")) + out.dim("  modernize legacy code, test-gated"))
    out.line()
    from . import policy
    active, problem = policy.current_or_none()
    if active is None:
        out.line(out.red(f"Error: {problem}"))
        return 2
    denial = active.tool_denial("run_autonomous_modernization_pipeline", "write")
    if denial or (allow_untested and not active.allow_untested) or (use_llm and not active.allow_llm):
        out.line(out.red("Error: " + (denial or "organization policy does not allow "
                                      + ("--allow-untested" if allow_untested and not active.allow_untested
                                         else "--llm") + ".")))
        return 2

    if path is None:
        if not interactive:
            out.line(out.red("No project folder given. Run `novuspipeline fix PATH`."))
            return 2
        path = _ask_folder(out)
        if path is None:
            return 1
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        out.line(out.red(f"'{root}' is not a folder."))
        return 2
    srv.WORKSPACE_ROOT = root
    out.line(f"Project: {out.bold(root)}")

    # 1. git: fixes are commits on a branch
    check = fixer.preflight(root)
    if not check.ok:
        out.line(out.yellow(f"Note: {check.problem}"))
        if not check.can_initialize:
            return 1
        question = "Create a git repository here and commit the current files as a starting point?"
        if not (interactive and _confirm(out, question, default=False)):
            out.line("Nothing was changed. Run `git init` and commit your files, then start again.")
            return 1
        try:
            sha = fixer.initialize_repo(root)
        except Exception as e:
            out.line(out.red(f"Could not set up git: {e}"))
            return 1
        out.line(out.green(f"  Committed the current files as {sha[:8]}."))
        check = fixer.preflight(root)
        if not check.ok:
            out.line(out.red(check.problem))
            return 1

    # 2. scan
    out.line()
    out.line(out.bold("Scanning"))
    out.status("  indexing the codebase (first run can take a minute)...")
    with contextlib.suppress(Exception):  # the index adds context; fixing works without it
        srv.index_codebase(False)
    files = fixer.workspace_files(root, include_tests)
    candidates = fixer.find_candidates(
        srv, files, progress=lambda i, n, rel: out.status(f"  [{i}/{n}] checking {rel}"))
    out.end_status()
    skipped_dirty = [c for c in candidates if c.path in check.dirty]
    candidates = [c for c in candidates if c.path not in check.dirty]
    out.line(f"  {len(files)} source file(s) checked, {out.bold(str(len(candidates)))} with automatic fixes"
             + (out.yellow(f", {len(skipped_dirty)} skipped (uncommitted changes)") if skipped_dirty else ""))

    if not candidates:
        out.line()
        out.line(out.green("Nothing to fix automatically.") + " For findings that need a person, run "
                 "`novuspipeline overview` and `novuspipeline audit <file>`.")
        _list_dirty(out, skipped_dirty)
        return 0

    for c in candidates[:15]:
        out.line(f"  {out.cyan('•')} {c.path} {out.dim(_short(c.changes))}")
    if len(candidates) > 15:
        out.line(out.dim(f"  ... and {len(candidates) - 15} more"))

    # 3. confirm the plan
    branch = branch or fixer.default_branch_name()
    out.line()
    out.line(f"Each file is rewritten in a separate checkout, verified with your tests in a sandbox, and committed "
             f"to {out.bold(branch)} only if they pass. Your files and your current branch are not changed.")
    if interactive and not _confirm(out, "Start fixing?", default=True):
        out.line("Nothing was changed.")
        return 0

    # 4. fix
    out.line()
    out.line(out.bold("Fixing"))
    results: dict[str, fixer.FileResult] = {}
    pending = candidates
    for attempt in range(1, fixer.MAX_PASSES + 1):
        if attempt > 1:
            # Tests that import several legacy modules can fail because of a file fixed later in the
            # previous pass. Retrying on top of the fixes committed since then resolves that order.
            out.line(out.dim(f"  Retrying {len(pending)} file(s) on top of the fixes committed so far"))
        for i, c in enumerate(pending, 1):
            out.status(f"  [{i}/{len(pending)}] {c.path}: rewriting and running tests...")
            result = fixer.fix_file(srv, c, branch, test_command, use_llm, allow_untested)
            results[c.path] = result
            out.end_status()
            out.line(f"  [{i}/{len(pending)}] {_icon(out, result.outcome)} {c.path}  "
                     f"{out.dim(fixer.OUTCOME_LABELS[result.outcome] + f' ({result.seconds}s)')}")
        fixed_now = sum(1 for c in pending if results[c.path].outcome == fixer.FIXED)
        pending = [c for c in pending if results[c.path].outcome == fixer.TESTS_FAILED]
        if not pending or not fixed_now:
            break

    baseline_fails = False
    if any(r.outcome == fixer.TESTS_FAILED for r in results.values()):
        out.status("  checking whether your tests pass without any change...")
        baseline_fails = fixer.baseline_fails(srv, test_command)
        out.end_status()
    return _summary(out, root, branch, list(results.values()), skipped_dirty, check, interactive, merge,
                    baseline_fails)


def _summary(out, root, branch, results, skipped_dirty, check, interactive, merge, baseline_fails) -> int:
    fixed = [r for r in results if r.outcome == fixer.FIXED]
    problems = [r for r in results if r.outcome not in (fixer.FIXED, fixer.UNCHANGED)]
    out.line()
    out.line(out.bold("Summary"))
    out.line(f"  {out.green(str(len(fixed)))} file(s) fixed and committed to {out.bold(branch)}"
             if fixed else "  No file was committed.")
    by_outcome: dict[str, list] = {}
    for r in problems:
        by_outcome.setdefault(r.outcome, []).append(r)
    for outcome, rows in by_outcome.items():
        out.line(f"  {out.yellow(str(len(rows)))} {fixer.OUTCOME_LABELS[outcome]}: "
                 + ", ".join(r.path for r in rows[:5]) + (" ..." if len(rows) > 5 else ""))
    _list_dirty(out, skipped_dirty)

    if fixer.NO_TESTS in by_outcome:
        out.line(out.dim("  Changes are only committed when tests prove them. Add tests (or pass --test-command) "
                         "and run again."))
    if fixer.TESTS_FAILED in by_outcome:
        first = by_outcome[fixer.TESTS_FAILED][0]
        if baseline_fails:
            out.line(out.yellow("  Your tests already fail without any change, so these rewrites cannot be verified. "
                                "Run `novuspipeline test python -m pytest` (or your test command) to see the "
                                "failures, fix them, and run again."))
        out.line(out.dim(f"  To see why, run `novuspipeline modernize {first.path} --branch {branch}`."))

    if not fixed:
        return 1 if problems else 0

    out.line()
    out.line(out.bold("Next"))
    out.line(f"  Review:  git log --stat {check.current_branch or 'HEAD'}..{branch}")
    out.line(f"           git diff {check.current_branch or 'HEAD'}...{branch}")
    if check.current_branch:
        do_merge = merge or (interactive and _confirm(
            out, f"Bring the fixes into `{check.current_branch}` now (fast-forward)?", default=False))
        if do_merge:
            ok, message = fixer.fast_forward(root, branch)
            out.line(("  " + out.green(message)) if ok else ("  " + out.yellow(message)))
        else:
            out.line(f"  Merge:   git merge {branch}")
    out.line(f"  PR:      git push origin {branch}   (a PR description draft is in .novus_index/prs/)")
    return 0 if not problems else 1


def _list_dirty(out: _Out, skipped) -> None:
    if skipped:
        out.line(out.dim("  Commit or stash these to include them: " + ", ".join(c.path for c in skipped[:8])
                         + (" ..." if len(skipped) > 8 else "")))


def _icon(out: _Out, outcome: str) -> str:
    icon = _ICONS.get(outcome, "✖")
    return out.green(icon) if outcome == fixer.FIXED else out.yellow(icon) if outcome == fixer.DIRTY \
        else out.dim(icon) if outcome == fixer.UNCHANGED else out.red(icon)


def _short(changes: list[str], limit: int = 70) -> str:
    if not changes:
        return ""
    text = changes[0].split(" (")[0].rstrip(".")
    more = f" +{len(changes) - 1} more" if len(changes) > 1 else ""
    return (text[:limit - 1] + "…" if len(text) > limit else text) + more
