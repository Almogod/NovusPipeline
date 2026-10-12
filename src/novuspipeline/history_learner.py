"""
history_learner.py — NovusPipeline: learning smells and fixes from commit history

Rule engines only know what someone wrote down. A team's history shows what it
actually fixes: if `self.assertEquals(...)` became `self.assertEqual(...)` in
seven separate commits, every remaining `assertEquals` is a smell *this team*
cares about, with a known fix. This module mines that knowledge:

  mine     for each non-merge commit, compare every changed function before and
           after: calls that disappeared vs calls that appeared. Focused changes
           (1-3 calls each way) become candidate replacements `old -> new`;
           a pair counts as argument-preserving when a removed and an added call
           had identical arguments. Module-level import swaps are mined too.
  learn    a pair seen in at least `min_support` distinct commits becomes a
           learned smell, with support, consistency (share of commits removing
           `old` that used `new` instead) and example commits as evidence. A
           single large rename commit therefore never creates a rule. A real
           phase-out also means the team stops *writing* `old`: pairs whose `old`
           was still newly introduced after the last replacement (more often than
           it was replaced) are incidental churn of a ubiquitous call, not a smell.
           (On Inkscape's history, real migrations had 0 such introductions, while
           `self.assertEqual -> self._assertPath` had 50+; introductions *during* a
           migration are normal, from work that was already in flight.)
  detect   remaining calls to `old` in the current code are findings.
  fix      only when every observed example preserved the arguments (a pure
           rename) and `new` needs no import the file lacks; anything else is
           reported as a suggestion. Fixes are applied by the pipeline, which still
           runs the tests in the sandbox and rejects API breaks.

Mining is incremental: commits already seen are skipped, so it can run in the
background after each index refresh and keep learning as the team commits.
State lives in <workspace>/.novus_index/learned.json.
"""

import ast
import contextlib
import hashlib
import json
import os
import threading
import time
from typing import Any

from .code_facts import dotted

STATE_VERSION = 2
MAX_CHANGES_PER_SIDE = 3          # larger diffs are refactors, not focused replacements
DEFAULT_MIN_SUPPORT = 3
DEFAULT_MIN_CONSISTENCY = 0.6
# Applying a fix automatically needs near-unanimous history: on Inkscape's history a
# 0.70-consistency pair (`selected.items` -> `svg.selected.items`) was only usually right.
AUTO_FIX_MIN_SUPPORT = 2
AUTO_FIX_MIN_CONSISTENCY = 0.9
_PARSE_CACHE_SIZE = 512
MAX_INTRODUCTIONS_KEPT = 50   # newest introduction times kept per call/import
_IGNORED_CALLS = {"print", "len", "str", "int", "list", "dict", "set", "tuple", "isinstance", "range", "super"}


def _calls(node: ast.AST) -> list[tuple[str, str]]:
    """(dotted call target, argument fingerprint) for every call under `node`."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            target = dotted(n.func)
            if target and target not in _IGNORED_CALLS:
                args = ast.dump(ast.Module(body=[ast.Expr(ast.Tuple(elts=list(n.args), ctx=ast.Load()))], type_ignores=[]),
                                include_attributes=False) + "|" + ";".join(ast.dump(k, include_attributes=False)
                                                                           for k in n.keywords)
                out.append((target, args))
    return out


def _functions(tree: ast.Module) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{item.name}"] = item
    return out


def _module_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


def diff_pairs(before: str, after: str) -> list[dict[str, Any]]:
    """Focused replacements between two versions of one file."""
    try:
        old_tree, new_tree = ast.parse(before), ast.parse(after)
    except (SyntaxError, ValueError, RecursionError):
        return []
    return _pairs_from_trees(old_tree, new_tree)


def _introduced(old_tree: ast.Module | None, new_tree: ast.Module) -> set[str]:
    """`call:<target>` / `import:<module>` keys newly used in `new_tree` (all of them for a new file)."""
    old_fns = _functions(old_tree) if old_tree is not None else {}
    keys: set[str] = set()
    for qualname, fn in _functions(new_tree).items():
        before = {t for t, _ in _calls(old_fns[qualname])} if qualname in old_fns else set()
        keys.update(f"call:{t}" for t, _ in _calls(fn) if t not in before)
    old_imports = _module_imports(old_tree) if old_tree is not None else set()
    keys.update(f"import:{m}" for m in _module_imports(new_tree) - old_imports)
    return keys


def _pairs_from_trees(old_tree: ast.Module, new_tree: ast.Module) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    old_fns, new_fns = _functions(old_tree), _functions(new_tree)
    for qualname in old_fns.keys() & new_fns.keys():
        b_calls, a_calls = _calls(old_fns[qualname]), _calls(new_fns[qualname])
        removed = {t for t, _ in b_calls} - {t for t, _ in a_calls}
        added = {t for t, _ in a_calls} - {t for t, _ in b_calls}
        if not (1 <= len(removed) <= MAX_CHANGES_PER_SIDE and 1 <= len(added) <= MAX_CHANGES_PER_SIDE):
            continue
        for old in removed:
            old_args = {a for t, a in b_calls if t == old}
            for new in added:
                preserved = bool(old_args & {a for t, a in a_calls if t == new})
                if preserved or (len(removed) == 1 and len(added) == 1):
                    pairs.append({"kind": "call", "old": old, "new": new, "args_preserved": preserved,
                                  "qualname": qualname})
    removed_imports = _module_imports(old_tree) - _module_imports(new_tree)
    added_imports = _module_imports(new_tree) - _module_imports(old_tree)
    if 1 <= len(removed_imports) <= 2 and 1 <= len(added_imports) <= 2:
        for old in removed_imports:
            for new in added_imports:
                pairs.append({"kind": "import", "old": old, "new": new, "args_preserved": False, "qualname": "<module>"})
    return pairs


class HistoryLearner:
    def __init__(self, root: str, index_dir: str = ".novus_index") -> None:
        self.root = os.path.abspath(root)
        self.path = os.path.join(self.root, index_dir, "learned.json")
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state: dict[str, Any] = {"version": STATE_VERSION, "seen": [], "observations": [], "removals": {},
                                      "introductions": {}, "mined_commits": 0, "updated_at": None}
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") == STATE_VERSION:
                self.state = data
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ mining

    def refresh(self, max_new_commits: int = 2000) -> dict[str, Any]:
        """
        Mines commits not seen before (newest first, up to `max_new_commits`). One
        `git log --raw` call lists every modified .py blob; blobs are read through git's
        persistent object reader and each blob is parsed once (consecutive commits share
        them: one commit's "after" is the next one's "before").
        """
        import git

        with self._lock:
            start = time.perf_counter()
            try:
                repo = git.Repo(self.root)
                repo.head.commit  # noqa: B018 - raises on an empty repository
            except (git.InvalidGitRepositoryError, git.NoSuchPathError, ValueError):
                return {"mined": 0, "error": "not a git repository with commits"}
            seen = set(self.state["seen"])
            raw = repo.git.log("--no-merges", "--raw", "--no-abbrev", "-M",
                               f"--max-count={max_new_commits + len(seen)}",
                               "--format=%x00%H%x1f%ct%x1f%s", "HEAD", "--", "*.py")
            cache: dict[str, ast.Module | None] = {}

            def parsed(sha: str) -> ast.Module | None:
                if sha not in cache:
                    if len(cache) >= _PARSE_CACHE_SIZE:
                        cache.pop(next(iter(cache)))
                    try:
                        source = repo.odb.stream(bytes.fromhex(sha)).read().decode("utf-8", errors="replace")
                        cache[sha] = ast.parse(source)
                    except (SyntaxError, ValueError, RecursionError, KeyError, OSError):
                        cache[sha] = None
                return cache[sha]

            mined = 0
            for record in raw.split("\x00"):
                if not record.strip():
                    continue
                header, *changes = record.strip("\n").split("\n")
                sha, timestamp, subject = ([*header.split("\x1f"), "", ""])[:3]
                if sha in seen:
                    continue
                if mined >= max_new_commits:
                    break
                seen.add(sha)
                mined += 1
                for line in changes:
                    # ":100644 100644 <old blob> <new blob> M\t<path>" or "... R087\t<old path>\t<new path>"
                    if not line.startswith(":") or "\t" not in line:
                        continue
                    meta, *paths = line[1:].split("\t")
                    parts = meta.split()
                    path = paths[-1]
                    if len(parts) < 5 or parts[4][0] not in "AMR" or not path.endswith(".py") or parts[2] == parts[3]:
                        continue
                    after = parsed(parts[3])
                    before = parsed(parts[2]) if parts[4][0] != "A" else None
                    if after is None:
                        continue
                    for key in _introduced(before, after):
                        times = self.state["introductions"].setdefault(key, [])
                        times.append(int(timestamp or 0))
                        if len(times) > MAX_INTRODUCTIONS_KEPT:
                            times.sort()
                            del times[0]
                    if before is None:
                        continue
                    for pair in _pairs_from_trees(before, after):
                        self.state["observations"].append({
                            **pair, "commit": sha[:10], "file": path, "message": subject[:80],
                            "time": int(timestamp or 0)})
                        key = f"{pair['kind']}:{pair['old']}"
                        commits = self.state["removals"].setdefault(key, [])
                        if sha[:10] not in commits:
                            commits.append(sha[:10])
            self.state["seen"] = sorted(seen)
            self.state["mined_commits"] = len(seen)
            self.state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if mined:
                self._save()
            return {"mined": mined, "total_commits": len(seen), "observations": len(self.state["observations"]),
                    "seconds": round(time.perf_counter() - start, 2)}

    def refresh_in_background(self, max_new_commits: int = 500) -> None:
        """Keeps learning without blocking tool calls; at most one mining thread per workspace."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._safe_refresh, args=(max_new_commits,), daemon=True)
        self._thread.start()

    def _safe_refresh(self, max_new_commits: int) -> None:
        # Background learning must never take the server down.
        with contextlib.suppress(Exception):
            self.refresh(max_new_commits)

    # ------------------------------------------------------------------ learned rules

    def rules(self, min_support: int = DEFAULT_MIN_SUPPORT,
              min_consistency: float = DEFAULT_MIN_CONSISTENCY) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for obs in list(self.state["observations"]):  # snapshot: a background refresh may be appending
            if obs["old"] != obs["new"]:
                groups.setdefault((obs["kind"], obs["old"], obs["new"]), []).append(obs)
        rules = []
        for (kind, old, new), observations in groups.items():
            commits = sorted({o["commit"] for o in observations})
            removal_commits = len(self.state["removals"].get(f"{kind}:{old}", [])) or len(commits)
            consistency = len(commits) / removal_commits
            if len(commits) < min_support or consistency < min_consistency:
                continue
            last_replacement = max(o["time"] for o in observations)
            still_adopted = sum(1 for t in self.state["introductions"].get(f"{kind}:{old}", []) if t > last_replacement)
            if still_adopted > len(commits):
                continue
            preserved = all(o["args_preserved"] for o in observations)
            examples, seen_commits = [], set()
            for o in sorted(observations, key=lambda o: -o["time"]):
                if o["commit"] not in seen_commits:
                    seen_commits.add(o["commit"])
                    examples.append({k: o[k] for k in ("commit", "message", "file", "qualname")})
            rules.append({
                "id": "LEARNED-" + hashlib.sha1(f"{kind}:{old}->{new}".encode()).hexdigest()[:6],
                "kind": kind, "old": old, "new": new, "support": len(commits), "consistency": round(consistency, 2),
                "args_preserved": preserved, "still_adopted": still_adopted, "examples": examples[:3],
            })
        # When one `old` has several learned replacements, keep the best-supported.
        best: dict[tuple[str, str], dict[str, Any]] = {}
        for r in sorted(rules, key=lambda r: (-r["support"], -r["consistency"])):
            best.setdefault((r["kind"], r["old"]), r)
        return sorted(best.values(), key=lambda r: (-r["support"], r["old"]))

    # ------------------------------------------------------------------ detection & fixes

    @staticmethod
    def _occurrences(tree: ast.Module, rule: dict[str, Any]) -> list[ast.AST]:
        if rule["kind"] == "call":
            return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and dotted(n.func) == rule["old"]]
        return [n for n in tree.body
                if (isinstance(n, ast.Import) and any(a.name == rule["old"] for a in n.names))
                or (isinstance(n, ast.ImportFrom) and n.module == rule["old"] and not n.level)]

    def scan(self, code: str, file_path: str, rules: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError):
            return []
        lines = code.split("\n")
        findings = []
        for rule in rules if rules is not None else self.rules():
            for node in self._occurrences(tree, rule):
                evidence = ", ".join(f"{e['commit'][:7]} \"{e['message']}\"" for e in rule["examples"][:2])
                findings.append({
                    "smell_id": rule["id"],
                    "name": f"Team replaced `{rule['old']}` with `{rule['new']}` in {rule['support']} commits",
                    "line_number": node.lineno,
                    "line_content": lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                    "category": "python", "severity": "MEDIUM",
                    "rag_query": "Python Library Modernization", "file_path": file_path, "source": "learned",
                    "fix": "safe" if self.auto_fixable(rule, tree) else None, "evidence": evidence,
                })
        return findings

    @staticmethod
    def auto_fixable(rule: dict[str, Any], tree: ast.Module | None = None) -> bool:
        """Pure renames with near-unanimous history, needing no import the file lacks."""
        if (rule["kind"] != "call" or not rule["args_preserved"] or rule["support"] < AUTO_FIX_MIN_SUPPORT
                or rule["consistency"] < AUTO_FIX_MIN_CONSISTENCY):
            return False
        old_root, new_root = rule["old"].split(".")[0], rule["new"].split(".")[0]
        if old_root == new_root:
            return True
        if tree is None:
            return False
        bound = {(a.asname or a.name).split(".")[0] for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                 for a in n.names}
        bound |= {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        return new_root in bound

    def apply_fixes(self, code: str, rules: list[dict[str, Any]] | None = None) -> tuple[str, list[str]]:
        """Rewrites call targets for auto-fixable learned rules; returns (new code, notes)."""
        original = code
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError):
            return code, []
        lines = code.splitlines(keepends=True)
        line_starts = [0]
        for line in lines:
            line_starts.append(line_starts[-1] + len(line))

        def offset(lineno: int, col_bytes: int) -> int:
            # AST columns are UTF-8 byte offsets; convert to a character index.
            return line_starts[lineno - 1] + len(lines[lineno - 1].encode("utf-8")[:col_bytes].decode("utf-8", "replace"))

        edits: list[tuple[int, int, str]] = []
        notes = []
        for rule in rules if rules is not None else self.rules():
            if not self.auto_fixable(rule, tree):
                continue
            nodes = self._occurrences(tree, rule)
            for node in nodes:
                f = node.func
                edits.append((offset(f.lineno, f.col_offset), offset(f.end_lineno, f.end_col_offset), rule["new"]))
            if nodes:
                notes.append(f"Applied learned fix {rule['id']}: `{rule['old']}` -> `{rule['new']}` "
                             f"({len(nodes)}x; seen in {rule['support']} past commits, e.g. "
                             f"{rule['examples'][0]['commit'][:7]} \"{rule['examples'][0]['message']}\").")
        for start, end, text in sorted(edits, reverse=True):
            code = code[:start] + text + code[end:]
        try:
            ast.parse(code)
        except SyntaxError:
            return original, []
        return code, notes


_LEARNERS: dict[str, HistoryLearner] = {}
_LEARNERS_LOCK = threading.Lock()


def get_learner(root: str) -> HistoryLearner:
    key = os.path.normcase(os.path.abspath(root))
    with _LEARNERS_LOCK:
        return _LEARNERS.setdefault(key, HistoryLearner(root))
