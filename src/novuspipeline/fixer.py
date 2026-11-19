"""
Fix a whole codebase: find every file the rule engine (plus Ruff, your standards and learned
renames) can rewrite automatically, then run each through the test-gated pipeline onto one
branch. This is the logic behind `novuspipeline fix` and the interactive mode; it holds no UI.

Nothing here weakens the pipeline's guarantees: every file is rewritten in a disposable
worktree, verified with the project's tests in the sandbox, and committed only if they pass.
The working copy is never written, except by `fast_forward`, which the caller must confirm.
"""
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from . import codebase_context

# Outcome of one file, in the order a summary lists them.
FIXED = "fixed"
TESTS_FAILED = "tests_failed"
NO_TESTS = "no_tests"
BLOCKED = "blocked"
API_BREAK = "api_break"
DIRTY = "uncommitted"
UNCHANGED = "unchanged"
ERROR = "error"

OUTCOME_LABELS = {
    FIXED: "fixed and committed",
    TESTS_FAILED: "not committed: tests failed after the rewrite",
    NO_TESTS: "not committed: no tests to verify the change",
    BLOCKED: "not committed: still violates a blocking standard",
    API_BREAK: "not committed: would remove a symbol other modules import",
    DIRTY: "skipped: file has uncommitted changes",
    UNCHANGED: "nothing to change",
    ERROR: "error",
}

_SCRIPT_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
MAX_PASSES = 3  # first pass plus retries of files whose tests failed, while retries still fix something
_NO_TESTS_RAN = re.compile(r"NO TESTS RAN|no tests ran|collected 0 items", re.IGNORECASE)


@dataclass
class Candidate:
    path: str           # workspace-relative, forward slashes
    changes: list[str]  # what the automatic rewrite would do


@dataclass
class FileResult:
    path: str
    outcome: str
    changes: list[str] = field(default_factory=list)
    commit: str = ""
    detail: str = ""    # the pipeline's full report
    seconds: float = 0.0


@dataclass
class Preflight:
    ok: bool
    problem: str = ""          # why the workspace cannot be fixed yet
    can_initialize: bool = False  # the problem is "not a repository / no commits", fixable by `initialize_repo`
    current_branch: str = ""
    dirty: set[str] = field(default_factory=set)


def preflight(root: str) -> Preflight:
    """Checks that `root` is a git repository with a commit to branch from."""
    import git

    try:
        repo = git.Repo(root)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        try:  # a subfolder of a repository: the pipeline works from the repository root
            outer = git.Repo(root, search_parent_directories=True).working_tree_dir
        except (git.InvalidGitRepositoryError, git.NoSuchPathError):
            outer = None
        if outer:
            return Preflight(False, f"this folder is inside the git repository `{outer}`. "
                                    "Open that folder instead (the repository root).")
        return Preflight(False, "this folder is not a git repository. Fixes are committed on a branch, "
                                "so the project needs git.", can_initialize=True)
    if not repo.head.is_valid():
        return Preflight(False, "the repository has no commits yet, so there is nothing to branch from.",
                         can_initialize=True)
    branch = "" if repo.head.is_detached else repo.active_branch.name
    dirty = set()
    for line in repo.git.status("--porcelain", "-uno").splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        dirty.add(_relative_to(root, os.path.join(repo.working_tree_dir, path)))
    return Preflight(True, current_branch=branch, dirty=dirty)


def initialize_repo(root: str) -> str:
    """`git init` (if needed) and a first commit of the folder as it is, honoring .gitignore. Returns the sha."""
    import git

    try:
        repo = git.Repo(root)
    except git.InvalidGitRepositoryError:
        repo = git.Repo.init(root)
    repo.git.add("-A")
    try:
        repo.git.commit("-m", "Snapshot before NovusPipeline modernization")
    except git.GitCommandError as e:
        if "user.name" in str(e) or "user.email" in str(e) or "identity" in str(e).lower():
            raise RuntimeError("git does not know who you are yet. Run `git config --global user.name \"Your Name\"` "
                               "and `git config --global user.email you@example.com`, then try again.") from e
        raise
    return repo.head.commit.hexsha


def default_branch_name() -> str:
    name = time.strftime("novus/modernize-%Y%m%d-%H%M%S")
    from . import policy
    active, _ = policy.current_or_none()
    if active is not None and active.branch_denial(name):
        name = active.branch_prefix + name.removeprefix("novus/")
    return name


def workspace_files(root: str, include_tests: bool = False) -> list[str]:
    """Python and JavaScript/TypeScript files, skipping the same directories the codebase index skips."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in codebase_context.SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if not (name.endswith(".py") or name.lower().endswith(_SCRIPT_EXTENSIONS)):
                continue
            full = os.path.join(dirpath, name)
            try:
                if os.path.getsize(full) > codebase_context.MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            rel = _relative_to(root, full)
            if not include_tests and codebase_context._is_test_file(rel):
                continue  # tests are the safety net; rewriting them would let a change verify itself
            found.append(rel)
    return found


def find_candidates(srv, files: list[str] | None = None,
                    progress: Callable[[int, int, str], None] | None = None) -> list[Candidate]:
    """Files whose automatic rewrite changes something. Computed in memory; nothing is written."""
    root = srv.WORKSPACE_ROOT
    files = workspace_files(root) if files is None else files
    candidates = []
    for i, rel in enumerate(files, 1):
        if progress:
            progress(i, len(files), rel)
        try:
            original, _ = srv._read_source(os.path.join(root, rel))
            new, notes, error = srv._modernized_source(os.path.join(root, rel), original)
        except Exception:
            continue  # unreadable or unparseable: the audit reports it, the batch skips it
        if not error and new is not None and new != original:
            candidates.append(Candidate(rel, [n for n in notes if n]))
    return candidates


def classify(result: str) -> str:
    text = result.lstrip()
    if text.startswith("🎉"):
        return FIXED
    if text.startswith("Pipeline finished: nothing"):
        return UNCHANGED
    if "has uncommitted changes" in text:
        return DIRTY
    if text.startswith("❌ Verification: NONE") or "no tests to verify" in text \
            or (text.startswith("❌ Autonomous") and _NO_TESTS_RAN.search(text)):
        return NO_TESTS
    if text.startswith("❌ Autonomous"):
        return TESTS_FAILED
    if "violates blocking standards" in text:
        return BLOCKED
    if "would no longer provide" in text:
        return API_BREAK
    return ERROR


def fix_file(srv, candidate: Candidate, branch: str, test_command: str = "", use_llm: bool = False,
             allow_untested: bool = False) -> FileResult:
    start = time.perf_counter()
    try:
        report = srv.run_autonomous_modernization_pipeline(candidate.path, test_command, branch, use_llm,
                                                           allow_untested=allow_untested)
    except Exception as e:  # the pipeline reports its own errors; this is a last resort
        report = f"Error: {type(e).__name__}: {e}"
    outcome = classify(report)
    sha = re.search(r"committed `([0-9a-f]+)`", report)
    return FileResult(candidate.path, outcome, candidate.changes, sha.group(1) if sha else "", report,
                      round(time.perf_counter() - start, 1))


def baseline_fails(srv, test_command: str = "") -> bool:
    """Whether the project's tests fail with no change at all (run in the sandbox on a copy)."""
    command = test_command.strip() or srv._default_test_command(None)
    if not command:
        return False
    report = srv.run_local_tests(command)
    codes = re.findall(r"^Exit Code: (\S+)", report, flags=re.MULTILINE)
    return bool(codes) and codes[-1] != "0" and not _NO_TESTS_RAN.search(report)


def fast_forward(root: str, branch: str) -> tuple[bool, str]:
    """Moves the checked-out branch to `branch` if that is a fast-forward. Changes the working copy."""
    import git

    repo = git.Repo(root)
    try:
        repo.git.merge("--ff-only", branch)
        return True, f"Your branch now includes the fixes from `{branch}`."
    except git.GitCommandError as e:
        return False, (f"Could not fast-forward ({e.stderr.strip() or e}). Merge `{branch}` yourself, "
                       f"e.g. `git merge {branch}`.")


def _relative_to(root: str, full: str) -> str:
    return os.path.relpath(full, root).replace(os.sep, "/")
