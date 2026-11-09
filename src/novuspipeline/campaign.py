"""
campaign.py — NovusPipeline: modernize a whole codebase as a stack of verified, per-file commits

`run_autonomous_modernization_pipeline` handles one file. A campaign walks every Python file
(or those under `paths`) in dependency order (modules before the modules that import them),
and for each one:

  1. rewrites it with the same engines as the pipeline (rules, migrations, Ruff safe fixes,
     learned renames, your standards),
  2. refuses changes that drop symbols other modules import, or leave a blocking standard,
  3. runs the tests that import the file (or `test_command`) in the sandbox, and
  4. commits that file alone on `branch` if they pass; otherwise it restores the file and
     re-runs the same tests on the unchanged code to tell "this change broke the tests"
     apart from "these tests were already failing".

Everything happens in one disposable git worktree, so the working copy and checked-out
branch are never touched. Progress is saved after every file in
`.novus_index/campaigns/<branch>.json`, so an interrupted or partial campaign (`max_files`)
resumes where it stopped, and each commit can be reviewed, reverted or bisected on its own.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from typing import Any

from . import codebase_context, sandbox

LEDGER_DIR = os.path.join(".novus_index", "campaigns")
FINAL = {"committed", "unchanged", "api-break", "blocked"}      # not retried on resume
RETRYABLE = {"tests-failed", "baseline-failing", "error"}      # retried with retry_failed=True


def ledger_path(root: str, branch: str) -> str:
    return os.path.join(root, LEDGER_DIR, re.sub(r"[^\w.-]", "_", branch) + ".json")


def load_ledger(root: str, branch: str) -> dict[str, Any]:
    try:
        with open(ledger_path(root, branch), encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("files"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {"branch": branch, "files": {}}


def _save_ledger(root: str, ledger: dict[str, Any]) -> None:
    path = ledger_path(root, ledger["branch"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
    os.replace(tmp, path)


def dependency_order(forward: dict[str, dict[str, Any]], files: list[str]) -> list[str]:
    """`files` with each file's (transitive) imports first; cycles keep a stable order."""
    wanted = set(files)
    seen: set[str] = set()
    order: list[str] = []
    for start in sorted(files):
        if start in seen:
            continue
        stack = [(start, iter(sorted(forward.get(start, {}))))]
        seen.add(start)
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                if node in wanted:
                    order.append(node)
            elif child not in seen and child in forward:
                seen.add(child)
                stack.append((child, iter(sorted(forward.get(child, {})))))
    return order


def _in_scope(rel: str, paths: list[str]) -> bool:
    if not paths:
        return True
    for p in paths:
        p = p.replace(os.sep, "/").strip("/")
        if p in ("", ".") or rel == p or rel.startswith(p + "/"):
            return True
    return False


def run(root: str, branch: str = "novus-campaign", paths: list[str] | None = None, test_command: str = "",
        max_files: int | None = None, retry_failed: bool = False,
        progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """
    Runs (or resumes) a campaign; returns {"ok", "error", "branch", "results": [...], "ledger", "summary"}.
    `max_files` caps how many files with changes are attempted in this run.
    """
    import git

    from . import server  # shares the pipeline's modernizer, test runner and standards

    say = progress or (lambda _msg: None)
    root = os.path.abspath(root)
    if os.path.normcase(server.WORKSPACE_ROOT) != os.path.normcase(root):
        server.WORKSPACE_ROOT = root

    def fail(msg: str) -> dict[str, Any]:
        return {"ok": False, "error": msg, "branch": branch, "results": [], "ledger": ledger_path(root, branch)}

    try:
        repo = git.Repo(root)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return fail("a campaign needs a git repository (it commits each file on a branch)")
    if not repo.head.is_valid():
        return fail("the repository has no commits yet")
    try:
        repo.git.check_ref_format("--branch", branch)
    except git.GitCommandError:
        return fail(f"'{branch}' is not a valid branch name")
    if not repo.head.is_detached and repo.active_branch.name == branch:
        return fail(f"'{branch}' is your checked-out branch; use a separate branch for the campaign")

    index = codebase_context.get_index(root)
    candidates = [rel for rel in index.files if _in_scope(rel, paths or [])]
    files = dependency_order(index._forward(), candidates)
    ledger = load_ledger(root, branch)
    start_point = branch if branch in [b.name for b in repo.branches] else "HEAD"
    say(f"Campaign on '{branch}' (from {start_point}): {len(files)} Python file(s) in scope.")

    results: list[dict[str, Any]] = []
    attempted = 0
    aborted = ""
    wt_root = tempfile.mkdtemp(prefix="novus-campaign-")
    wt = os.path.join(wt_root, "wt")
    try:
        repo.git.worktree("add", "--detach", wt, start_point)
        sandbox.link_dependency_dirs(root, wt)
        wt_repo = git.Repo(wt)

        for rel in files:
            wt_file = os.path.join(wt, rel)
            if not os.path.isfile(wt_file):
                continue
            original, newline = server._read_source(wt_file)
            digest = hashlib.sha1(original.encode("utf-8")).hexdigest()
            prev = ledger["files"].get(rel, {})
            if prev.get("source_sha1") == digest and (
                    prev.get("status") in FINAL or (prev.get("status") in RETRYABLE and not retry_failed)):
                continue  # already handled, and the file has not changed since
            new, notes, error = server._modernized_source(rel, original)
            entry: dict[str, Any] = {"path": rel, "notes": notes, "source_sha1": digest,
                                     "at": time.strftime("%Y-%m-%dT%H:%M:%S")}

            if error or new is None:
                continue
            if new == original:
                entry["status"] = "unchanged"
            elif server._standards().blocking(new, rel):
                entry["status"] = "blocked"
                entry["reason"] = "still violates a blocking standard after the automatic fixes"
            else:
                if max_files is not None and attempted >= max_files:
                    break
                attempted += 1
                ctx = index.file_context(rel) if rel in index.files else None
                broken = server._broken_api(new, ctx["api_surface"]) if ctx else []
                if broken:
                    entry["status"] = "api-break"
                    entry["reason"] = f"would remove {', '.join(broken)}, which other modules import"
                else:
                    command = test_command.strip() or server._default_test_command(ctx)
                    entry["test_command"] = command
                    say(f"[{attempted}] {rel}: testing with `{command}` ...")
                    server._write_source(wt_file, new, newline)
                    code, output = server._execute_test_command(command, workspace=wt, disposable=True)
                    if code == 0:
                        msg = f"refactor(novus): modernize {rel}\n\n" + "\n".join(f"- {n}" for n in notes)
                        wt_repo.git.add("--", rel)
                        wt_repo.git.commit("-m", msg, "--", rel)
                        sha = wt_repo.head.commit.hexsha
                        digest = hashlib.sha1(new.encode("utf-8")).hexdigest()  # what the branch now holds
                        wt_repo.git.branch("-f", branch, sha)
                        entry.update(status="committed", sha=sha, source_sha1=digest)
                    else:
                        server._write_source(wt_file, original, newline)
                        if code is None:
                            entry.update(status="error", reason=output[-2000:])
                            aborted = f"the test command could not run: {output.splitlines()[0] if output else ''}"
                        else:
                            base_code, _ = server._execute_test_command(command, workspace=wt, disposable=True)
                            entry["status"] = "baseline-failing" if base_code != 0 else "tests-failed"
                            entry["reason"] = ("these tests fail without the change too" if base_code != 0
                                               else f"tests failed with the change (exit {code})")
                            entry["output"] = output[-2000:]
            ledger["files"][rel] = entry
            _save_ledger(root, ledger)
            results.append(entry)
            if entry["status"] != "unchanged":
                say(f"  {rel}: {entry['status']}" + (f" ({entry.get('reason')})" if entry.get("reason") else ""))
            if aborted:
                break
    except Exception as e:  # report, keep the ledger; the worktree is always cleaned up
        aborted = f"{type(e).__name__}: {e}"
    finally:
        sandbox.unlink_dependency_dirs(wt)
        with contextlib.suppress(git.GitCommandError):
            repo.git.worktree("remove", "--force", wt)
        shutil.rmtree(wt_root, ignore_errors=True)
        with contextlib.suppress(git.GitCommandError):
            repo.git.worktree("prune")

    remaining = sum(1 for rel in files if rel not in ledger["files"])
    summary = format_summary(ledger, results, aborted, remaining=remaining)
    pr_path = server._write_pr_draft(branch, f"Modernization campaign ({branch})", summary)
    return {"ok": not aborted, "error": aborted, "branch": branch, "results": results,
            "ledger": ledger_path(root, branch), "pr_draft": pr_path, "summary": summary}


def format_summary(ledger: dict[str, Any], results: list[dict[str, Any]], aborted: str = "",
                   remaining: int = 0) -> str:
    entries = list(ledger["files"].values())
    count = {s: sum(1 for e in entries if e["status"] == s) for s in
             ("committed", "tests-failed", "baseline-failing", "api-break", "blocked", "error", "unchanged")}
    lines = [f"### Campaign `{ledger['branch']}`",
             f"{count['committed']} file(s) committed, each verified by its tests; "
             f"{count['tests-failed']} rejected because tests failed with the change; "
             f"{count['baseline-failing']} skipped because their tests already fail; "
             f"{count['api-break'] + count['blocked']} need a manual change; {count['unchanged']} already modern."]
    if aborted:
        lines.append(f"\n**Stopped early**: {aborted}")
    if remaining:
        lines.append(f"\n{remaining} file(s) not processed yet: run the campaign again to continue.")
    sections = [("committed", "Committed"), ("tests-failed", "Rejected: tests failed with the change"),
                ("baseline-failing", "Skipped: tests already failing"), ("api-break", "Needs review: API change"),
                ("blocked", "Needs review: blocking standard"), ("error", "Errors")]
    for status, title in sections:
        rows = [e for e in entries if e["status"] == status]
        if not rows:
            continue
        lines.append(f"\n#### {title} ({len(rows)})")
        for e in rows:
            what = "; ".join(n for n in e.get("notes", []) if not n.startswith("Manual review"))[:300]
            extra = f" `{e['sha'][:8]}`" if e.get("sha") else ""
            lines.append(f"- `{e['path']}`{extra}: {e.get('reason') or what or 'modernized'}")
    if results:
        lines.append(f"\n_This run processed {len(results)} file(s)._")
    return "\n".join(lines)
