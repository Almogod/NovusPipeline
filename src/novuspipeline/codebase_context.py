"""
codebase_context.py — NovusPipeline: Codebase Context Index

A persistent, incrementally refreshed index of every Python file in the
workspace, so tools can reason about a file in the context of its codebase:

  findings     built-in rules + Ruff (hundreds of maintained rules) per file
  GNN          per-function smell probabilities and structural embeddings
  import graph static, lazy and dynamic imports (importlib / __import__ /
               getattr / star imports / file loaders); who imports a module,
               which of its symbols they use (its API surface), related tests
  context      codebase-level smells no single-file linter can see: import
  smells       cycles, dead code, duplicated logic, untested hotspots, god
               modules, deprecated internal APIs, unresolvable dynamic imports

Files are re-analyzed only when their content changes (or the GNN changes).
Stored at <workspace>/.novus_index/index.json. Team settings are read from
<workspace>/.novus/config.toml, e.g.

    [deprecated]
    "mypkg.utils.old_helper" = "use mypkg.utils.new_helper"
"""

import ast
import hashlib
import json
import math
import os
import textwrap
import threading
import time
from collections.abc import Iterable
from typing import Any

from . import gnn_model, history_learner, ruff_backend, standards
from .code_facts import dotted, extract_defs, extract_exports, extract_imports, extract_refs, function_shape
from .modernizer import LegacySmellDetector

INDEX_VERSION = 4  # bump when stored record format changes; older indexes are rebuilt
INDEX_DIRNAME = ".novus_index"
CONFIG_PATH = os.path.join(".novus", "config.toml")
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", ".env", "__pycache__", "node_modules",
    ".chroma_db", ".gnn_model", INDEX_DIRNAME, "build", "dist", ".tox", ".nox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages", ".eggs",
}
MAX_FILE_BYTES = 1_000_000
MAX_FILES = 5000
SEVERITY_WEIGHT = {"CRITICAL": 3.0, "HIGH": 2.0, "MEDIUM": 1.0, "LOW": 0.25}
SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
# PY-SMELL-002 is about import statements, so it is attributed at file level only.
FUNCTION_LEVEL_SMELLS = [s for s in gnn_model.SMELL_LABELS if s != "PY-SMELL-002"]
CLONE_MIN_NODES = 40          # smaller functions share shapes by coincidence
GOD_MODULE = {"fan_in": 5, "fan_out": 8, "lines": 400}

CONTEXT_SMELLS = {
    "CTX-CYCLE": ("Import cycle", "MEDIUM"),
    "CTX-DEAD": ("Possibly unused definition", "LOW"),
    "CTX-CLONE": ("Duplicated logic", "MEDIUM"),
    "CTX-UNTESTED": ("Risky module without tests", "MEDIUM"),
    "CTX-GOD": ("God module (too many dependencies both ways)", "LOW"),
    "CTX-DEPRECATED": ("Uses a deprecated internal API", "MEDIUM"),
    "CTX-DYNAMIC": ("Unresolvable dynamic import (import graph may be incomplete)", "LOW"),
}


def _module_name(rel_path: str) -> str:
    parts = rel_path[:-3].split("/") if rel_path.endswith(".py") else rel_path.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_test_file(rel_path: str) -> bool:
    base = os.path.basename(rel_path)
    dirs = rel_path.split("/")[:-1]
    return base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py" \
        or any(d in ("tests", "test") for d in dirs)


def _functions(tree: ast.Module, code: str) -> list[dict[str, Any]]:
    """Top-level functions and methods of top-level classes, with standalone source and shape."""
    out = []

    def add(node: ast.AST, qualname: str) -> None:
        segment = ast.get_source_segment(code, node, padded=True)
        if segment:
            shape, size = function_shape(node)
            out.append({"qualname": qualname, "lineno": node.lineno, "end_lineno": node.end_lineno,
                        "shape": shape, "size": size, "source": textwrap.dedent(segment)})

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, node.name)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(item, f"{node.name}.{item.name}")
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _read_toml(path: str) -> dict[str, Any]:
    try:
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, ValueError, ImportError):
        return {}


def _strongly_connected(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's SCCs (iterative); returns components with more than one node."""
    index, low, on_stack, stack, out, counter = {}, {}, set(), [], [], [0]
    for root in graph:
        if root in index:
            continue
        work = [(root, iter(sorted(graph.get(root, ()))))]
        index[root] = low[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter[0]
                    counter[0] += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(graph.get(child, ())))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    out.append(sorted(component))
    return out


class CodebaseIndex:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.index_path = os.path.join(self.root, INDEX_DIRNAME, "index.json")
        self.files: dict[str, dict[str, Any]] = {}
        self.meta: dict[str, Any] = {}
        self.gnn_error: str | None = None
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------ build

    def _load(self) -> None:
        try:
            with open(self.index_path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") == INDEX_VERSION:
                self.files, self.meta = data["files"], data.get("meta", {})
        except (OSError, ValueError, KeyError):
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        tmp = self.index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": INDEX_VERSION, "meta": self.meta, "files": self.files}, f)
        os.replace(tmp, self.index_path)

    def _workspace_files(self) -> Iterable[str]:
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for name in sorted(filenames):
                if name.endswith(".py"):
                    full = os.path.join(dirpath, name)
                    try:
                        too_big = os.path.getsize(full) > MAX_FILE_BYTES
                    except OSError:
                        continue
                    if not too_big:
                        yield os.path.relpath(full, self.root).replace(os.sep, "/")
                        count += 1
                        if count >= MAX_FILES:
                            return

    def _gnn_signature(self) -> str | None:
        status = gnn_model.get_gnn_status()
        if not (status["checkpoint_exists"] and status["torch_available"]):
            return None
        return str(status["metadata"].get("trained_at")) + str(os.path.getmtime(gnn_model.CHECKPOINT_PATH))

    def _config_signature(self) -> str:
        parts = []
        for rel in (CONFIG_PATH, "pyproject.toml"):
            try:
                parts.append(f"{rel}:{os.path.getmtime(os.path.join(self.root, rel))}")
            except OSError:
                parts.append(f"{rel}:-")
        return "|".join(parts) + f"|ruff:{ruff_backend.available()[0]}|standards:{standards.signature(self.root)}"

    def _read_config(self) -> None:
        cfg = _read_toml(os.path.join(self.root, CONFIG_PATH))
        self.meta["deprecated"] = {str(k): str(v) for k, v in (cfg.get("deprecated") or {}).items()}
        project = _read_toml(os.path.join(self.root, "pyproject.toml")).get("project", {})
        targets = list((project.get("scripts") or {}).values()) + list((project.get("gui-scripts") or {}).values())
        for group in (project.get("entry-points") or {}).values():
            targets.extend(group.values())
        self.meta["entry_points"] = sorted({str(t).split("[")[0].strip() for t in targets})

    def _analyze_file(self, rel: str, code: str, sha: str) -> dict[str, Any]:
        record: dict[str, Any] = {
            "path": rel, "module": _module_name(rel), "sha1": sha, "is_test": _is_test_file(rel),
            "is_package": rel.endswith("__init__.py"), "parse_error": None, "lines": code.count("\n") + 1,
            "imports": [], "dynamic_unresolved": [], "path_loads": [], "defs": [], "refs": [], "exports": None, "calls": {},
            "functions": [], "file_probs": None, "file_embedding": None,
            "rule_findings": [{"smell_id": f["smell_id"], "line": f["line_number"], "severity": f["severity"],
                               "name": f["name"], "source": "rules"}
                              for f in LegacySmellDetector.scan_code(code, rel)]
            + [{"smell_id": f["smell_id"], "line": f["line_number"], "severity": f["severity"], "name": f["name"],
                "source": "standard", "fix": f["fix"]} for f in standards.load(self.root).scan(code, rel)],
        }
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
            record["parse_error"] = f"{type(e).__name__}: {e}"
            return record
        record["imports"], record["dynamic_unresolved"], record["path_loads"] = \
            extract_imports(tree, record["module"], record["is_package"])
        record["defs"] = extract_defs(tree)
        record["refs"] = extract_refs(tree)
        record["exports"] = extract_exports(tree)
        for node in ast.walk(tree):  # call target -> lines, for matching rules learned from history
            if isinstance(node, ast.Call):
                target = dotted(node.func)
                if target and "." in target:
                    record["calls"].setdefault(target, []).append(node.lineno)
        record["functions"] = _functions(tree, code)
        record["_source"] = code
        return record

    def _run_gnn(self, records: list[dict[str, Any]]) -> None:
        """Fills file/function GNN fields in place, in one batched pass."""
        jobs: list[tuple[dict[str, Any], dict[str, Any] | None, str]] = []
        for rec in records:
            if rec.get("_source") is not None:
                jobs.append((rec, None, rec["_source"]))
                jobs.extend((rec, fn, fn["source"]) for fn in rec["functions"])
        if not jobs:
            return
        try:
            results = gnn_model.analyze_many([code for _, _, code in jobs])
            thresholds = gnn_model.get_thresholds()
            self.gnn_error = None
        except RuntimeError as e:
            self.gnn_error = str(e)
            return
        for (rec, fn, _), res in zip(jobs, results):
            if res is None:
                continue
            probs = {k: round(v, 4) for k, v in res["probs"].items()}
            embedding = [round(x, 4) for x in res["embedding"]]
            if fn is None:
                rec["file_probs"], rec["file_embedding"] = probs, embedding
            else:
                fn["probs"], fn["embedding"] = probs, embedding
                fn["flagged"] = [s for s in FUNCTION_LEVEL_SMELLS if probs[s] > thresholds[s]]

    def refresh(self, force: bool = False) -> dict[str, Any]:
        """Re-analyzes new/changed files (all files if `force`, or if the GNN, config or Ruff changed)."""
        with self._lock:
            start = time.perf_counter()
            gnn_sig = self._gnn_signature()
            config_sig = self._config_signature()
            rebuild_all = force or gnn_sig != self.meta.get("gnn_signature") \
                or config_sig != self.meta.get("config_signature")
            if rebuild_all:
                self._read_config()

            seen, changed = set(), []
            for rel in self._workspace_files():
                seen.add(rel)
                full = os.path.join(self.root, rel)
                try:
                    with open(full, "rb") as f:
                        raw = f.read()
                except OSError:
                    continue
                sha = hashlib.sha1(raw).hexdigest()
                if rebuild_all or self.files.get(rel, {}).get("sha1") != sha:
                    changed.append(self._analyze_file(rel, raw.decode("utf-8", errors="replace"), sha))

            removed = [rel for rel in self.files if rel not in seen]
            for rel in removed:
                del self.files[rel]

            active = standards.load(self.root)
            ruff = ruff_backend.scan_files(self.root, [rec["path"] for rec in changed],
                                           active.ruff_select, active.ruff_ignore)
            for rec in changed:
                rec["rule_findings"] += [{"smell_id": f["smell_id"], "line": f["line_number"], "severity": f["severity"],
                                          "name": f["name"], "source": "ruff", "fix": f["fix"]}
                                         for f in ruff.get(rec["path"], [])]

            if gnn_sig is not None:
                self._run_gnn(changed)
            else:
                self.gnn_error = gnn_model.get_gnn_status().get("torch_error") or \
                    "No trained GNN checkpoint (run train_gnn.py)."
            for rec in changed:
                rec.pop("_source", None)
                for fn in rec["functions"]:
                    fn.pop("source", None)
                self.files[rec["path"]] = rec

            if changed or removed or rebuild_all:
                self._cache.clear()
                self.meta.update({"gnn_signature": gnn_sig, "config_signature": config_sig,
                                  "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                self._save()
            if os.environ.get("NOVUS_LEARN_FROM_HISTORY", "1") != "0":
                history_learner.get_learner(self.root).refresh_in_background()
            return {"files": len(self.files), "reanalyzed": len(changed), "removed": len(removed),
                    "functions": sum(len(r["functions"]) for r in self.files.values()),
                    "gnn": self.gnn_error is None and gnn_sig is not None,
                    "ruff": ruff_backend.available()[0],
                    "seconds": round(time.perf_counter() - start, 2)}

    # ---------------------------------------------------------------- import graph

    def _module_map(self) -> dict[str, str]:
        """Module name -> path, including `src.`-less aliases for src layouts."""
        if "modules" not in self._cache:
            mapping: dict[str, str] = {}
            for rel, rec in self.files.items():
                mapping.setdefault(rec["module"], rel)
                if rec["module"].startswith("src."):
                    mapping.setdefault(rec["module"][4:], rel)
            self._cache["modules"] = mapping
        return self._cache["modules"]

    def _public_names(self, rel: str) -> list[str]:
        rec = self.files[rel]
        if rec.get("exports") is not None:
            return list(rec["exports"])
        return [d["name"] for d in rec.get("defs", []) if not d["name"].startswith("_")]

    def _resolve(self, rec: dict[str, Any], import_time_only: bool = False) -> dict[str, set]:
        """Workspace paths this file imports -> symbols it uses from each."""
        modules = self._module_map()
        out: dict[str, set] = {}
        for fact in rec["imports"]:
            if import_time_only and (fact.get("lazy") or fact.get("dynamic") or fact.get("exact")):
                continue
            target, names = fact["module"], fact["names"]
            if fact.get("exact"):  # attributes of a `from pkg import mod` candidate
                path = modules.get(target)
                if path and path != rec["path"]:
                    out.setdefault(path, set()).update(names)
                continue
            if names == ["*"]:
                path = modules.get(target)
                if path and path != rec["path"]:
                    out.setdefault(path, set()).update(self._public_names(path))
                continue
            for name in names:
                sub = modules.get(f"{target}.{name}")
                if sub and sub != rec["path"]:
                    out.setdefault(sub, set())
            plain = [n for n in names if f"{target}.{n}" not in modules]
            parts = target.split(".")
            for i in range(len(parts), 0, -1):
                path = modules.get(".".join(parts[:i]))
                if path:
                    if path != rec["path"]:
                        out.setdefault(path, set()).update(plain if i == len(parts) else [])
                    break
        if not import_time_only:
            base = os.path.dirname(rec["path"])
            for load in rec.get("path_loads", []):
                for candidate in (os.path.normpath(os.path.join(base, load["path"])), os.path.normpath(load["path"])):
                    rel = candidate.replace(os.sep, "/")
                    if rel in self.files and rel != rec["path"]:
                        out.setdefault(rel, set())
                        break
        return out

    def _forward(self) -> dict[str, dict[str, set]]:
        """Resolved import graph (static + lazy + dynamic), cached until the next change."""
        if "forward" not in self._cache:
            self._cache["forward"] = {rel: self._resolve(rec) for rel, rec in self.files.items()}
        return self._cache["forward"]

    def _risks(self) -> dict[str, tuple[float, dict[str, dict[str, str]]]]:
        self.learned_smells()  # drops cached risks when background mining learned something new
        if "risks" not in self._cache:
            self._cache["risks"] = {rel: self.file_risk(rec) for rel, rec in self.files.items()}
        return self._cache["risks"]

    def learned_smells(self) -> dict[str, list[dict[str, Any]]]:
        """
        Remaining uses of calls/imports this team has repeatedly replaced in its own
        history (see history_learner). Recomputed whenever mining has learned more.
        """
        learner = history_learner.get_learner(self.root)
        key = (learner.state.get("mined_commits"), len(learner.state.get("observations", [])))
        if self._cache.get("learned_key") == key and "learned" in self._cache:
            return self._cache["learned"]
        rules = learner.rules()
        out: dict[str, list[dict[str, Any]]] = {}
        for rel, rec in self.files.items():
            for rule in rules:
                if rule["kind"] == "call":
                    lines = rec.get("calls", {}).get(rule["old"], [])
                else:
                    lines = [f["line"] for f in rec["imports"] if f["module"] == rule["old"] and not f.get("dynamic")
                             and not f.get("exact")]
                if lines:
                    out.setdefault(rel, []).append({
                        "smell_id": rule["id"], "severity": "MEDIUM", "line": lines[0], "count": len(lines),
                        "name": f"`{rule['old']}` -> `{rule['new']}` (team history)", "old": rule["old"],
                        "new": rule["new"], "support": rule["support"], "consistency": rule["consistency"],
                        "auto_fix": history_learner.HistoryLearner.auto_fixable(rule) or None})
        self._cache.pop("risks", None)
        self._cache.update({"learned": out, "learned_key": key})
        return out

    def normalize(self, path: str) -> str | None:
        full = path if os.path.isabs(path) else os.path.join(self.root, path)
        try:
            rel = os.path.relpath(os.path.abspath(full), self.root).replace(os.sep, "/")
        except ValueError:  # different drive on Windows
            return None
        return rel if rel in self.files else None

    def dependencies(self, rel: str) -> dict[str, list[str]]:
        return {p: sorted(n) for p, n in self._forward()[rel].items()}

    def dependents(self, rel: str) -> dict[str, list[str]]:
        return {other: sorted(uses[rel]) for other, uses in self._forward().items() if rel in uses}

    def related_tests(self, rel: str) -> list[str]:
        tests = {p for p in self.dependents(rel) if self.files[p]["is_test"]}
        stem = os.path.basename(rel)[:-3]
        tests.update(p for p, r in self.files.items()
                     if r["is_test"] and os.path.basename(p) in (f"test_{stem}.py", f"{stem}_test.py"))
        return sorted(tests)

    def api_surface(self, rel: str) -> list[str]:
        names = set()
        for used in self.dependents(rel).values():
            names.update(used)
        return sorted(names)

    # ---------------------------------------------------------------- similarity

    def _centered(self, kind: str) -> dict[tuple[str, str], list[float]]:
        """
        Embeddings minus their mean over the index. GNN embeddings are
        post-ReLU/max-pool, so raw vectors share a large common component and
        every pair scores ~0.97 cosine; centering makes similarity discriminative.
        """
        key = f"centered_{kind}"
        if key not in self._cache:
            vectors: dict[tuple[str, str], list[float]] = {}
            for p, r in self.files.items():
                if kind == "function":
                    vectors.update({(p, f["qualname"]): f["embedding"] for f in r["functions"] if f.get("embedding")})
                elif r.get("file_embedding"):
                    vectors[(p, "")] = r["file_embedding"]
            if vectors:
                dim = len(next(iter(vectors.values())))
                mean = [sum(v[i] for v in vectors.values()) / len(vectors) for i in range(dim)]
                vectors = {k: [x - m for x, m in zip(v, mean)] for k, v in vectors.items()}
            self._cache[key] = vectors
        return self._cache[key]

    def similar(self, rel: str, qualname: str = "", n: int = 5) -> list[dict[str, Any]]:
        """Most structurally similar functions (or files, if `qualname` is empty) elsewhere."""
        vectors = self._centered("function" if qualname else "file")
        query = vectors.get((rel, qualname))
        if query is None:
            return []
        flags = {(p, f["qualname"]): (set(f.get("flagged", [])), f["lineno"])
                 for p, r in self.files.items() for f in r["functions"]}
        scored = []
        for (p, q), vec in vectors.items():
            if (p, q) == (rel, qualname):
                continue
            row: dict[str, Any] = {"path": p, "similarity": round(_cosine(query, vec), 3)}
            if qualname:
                row.update({"qualname": q, "lineno": flags[(p, q)][1],
                            "shared_smells": sorted(flags[(rel, qualname)][0] & flags[(p, q)][0])})
            scored.append(row)
        return sorted(scored, key=lambda s: -s["similarity"])[:n]

    # ---------------------------------------------------------------- codebase-level smells

    def context_smells(self) -> dict[str, list[dict[str, Any]]]:
        """{path: [codebase-level smells]}; see CONTEXT_SMELLS."""
        if "context" in self._cache:
            return self._cache["context"]
        out: dict[str, list[dict[str, Any]]] = {rel: [] for rel in self.files}

        def add(rel: str, smell_id: str, detail: str, line: int | None = None) -> None:
            name, severity = CONTEXT_SMELLS[smell_id]
            out[rel].append({"smell_id": smell_id, "name": name, "severity": severity, "line": line,
                             "detail": detail, "source": "context"})

        forward = self._forward()
        fan_in: dict[str, int] = {}
        for uses in forward.values():
            for target in uses:
                fan_in[target] = fan_in.get(target, 0) + 1

        # Import cycles: import-time edges only (lazy imports inside functions are the usual cycle fix).
        import_time = {rel: set(self._resolve(rec, import_time_only=True)) for rel, rec in self.files.items()}
        for component in _strongly_connected(import_time):
            for rel in component:
                add(rel, "CTX-CYCLE", "imports form a cycle: " + " -> ".join(f"`{p}`" for p in component)
                    + f" -> `{component[0]}`")

        # Dead code: a top-level definition no file mentions anywhere (names, attributes, strings).
        mentioned: set[str] = set()
        for rec in self.files.values():
            mentioned.update(rec.get("refs", []))
            for fact in rec.get("imports", []):
                mentioned.update(n for n in fact["names"] if n != "*")
        entry_names = {ep.split(":")[-1].split(".")[-1] for ep in self.meta.get("entry_points", [])}
        dynamic_prefixes = [u["prefix"] for rec in self.files.values() for u in rec.get("dynamic_unresolved", [])
                            if u.get("prefix")]
        for rel, rec in self.files.items():
            if rec["is_test"] or rec["parse_error"] or os.path.basename(rel) in ("__init__.py", "__main__.py", "setup.py"):
                continue
            if any(rec["module"] == p or rec["module"].startswith(p + ".") for p in dynamic_prefixes):
                continue  # loaded by a plugin-style dynamic import
            exported = set(rec.get("exports") or [])
            for d in rec.get("defs", []):
                name = d["name"]
                if (d["decorated"] or name.startswith("__") or name in exported or name in entry_names
                        or name in ("main", "cli") or name in mentioned):
                    continue
                add(rel, "CTX-DEAD", f"`{name}` ({d['kind']}) is never referenced in the workspace "
                    "(static check: dynamic access such as getattr with a computed name is not visible)", d["line"])

        # Duplicated logic: identical rename-insensitive structure.
        shapes: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for rel, rec in self.files.items():
            for fn in rec.get("functions", []):
                if fn.get("shape") and fn.get("size", 0) >= CLONE_MIN_NODES:
                    shapes.setdefault(fn["shape"], []).append((rel, fn))
        for group in shapes.values():
            if len(group) < 2:
                continue
            for rel, fn in group:
                others = [f"`{p}::{f['qualname']}` (L{f['lineno']})" for p, f in group if (p, f["qualname"]) != (rel, fn["qualname"])]
                add(rel, "CTX-CLONE", f"`{fn['qualname']}` has the same structure as {', '.join(others[:3])}"
                    + (f" (+{len(others) - 3} more)" if len(others) > 3 else ""), fn["lineno"])

        # Untested hotspots, god modules, deprecated APIs, unresolvable dynamic imports.
        deprecated = self.meta.get("deprecated", {})
        modules = self._module_map()
        for rel, rec in self.files.items():
            if not rec["is_test"]:
                severe = [f for f in rec["rule_findings"] if f.get("severity") in ("HIGH", "CRITICAL")]
                if fan_in.get(rel, 0) >= 1 and (severe or fan_in.get(rel, 0) >= 3) and not self.related_tests(rel):
                    add(rel, "CTX-UNTESTED", f"imported by {fan_in.get(rel, 0)} module(s)"
                        + (f", {len(severe)} high-severity finding(s)" if severe else "") + ", and no test imports it")
                fan_out = len(forward.get(rel, {}))
                if fan_in.get(rel, 0) >= GOD_MODULE["fan_in"] and fan_out >= GOD_MODULE["fan_out"] \
                        and rec.get("lines", 0) >= GOD_MODULE["lines"]:
                    add(rel, "CTX-GOD", f"imported by {fan_in[rel]} modules, imports {fan_out}, {rec['lines']} lines; "
                        "consider splitting responsibilities")
            for fact in rec.get("imports", []):
                hits = [fact["module"]] if fact["module"] in deprecated else []
                hits += [f"{fact['module']}.{n}" for n in fact["names"] if f"{fact['module']}.{n}" in deprecated]
                for hit in hits:
                    add(rel, "CTX-DEPRECATED", f"uses `{hit}`: {deprecated[hit]}", fact.get("line") or None)
            for u in rec.get("dynamic_unresolved", []):
                reachable = [m for m in modules if u.get("prefix") and m.startswith(u["prefix"] + ".")]
                add(rel, "CTX-DYNAMIC", f"`{u['expr']}`" + (f" (treating {len(reachable)} module(s) under "
                                                            f"`{u['prefix']}` as reachable)" if u.get("prefix") else ""),
                    u["line"])

        self._cache["context"] = out
        return out

    # ---------------------------------------------------------------- risk & queries

    def file_risk(self, rec: dict[str, Any]) -> tuple[float, dict[str, dict[str, str]]]:
        """
        Severity-weighted score; each smell id counts once, from rules/Ruff, the GNN, or the
        codebase-level analysis. The GNN signal is the file-level prediction OR any
        function-level flag (per-function predictions catch a single slip in a large file).
        Returns (score, {smell_id: {"source", "severity", "name"}}).
        """
        details: dict[str, dict[str, str]] = {}
        for f in rec["rule_findings"]:
            details.setdefault(f["smell_id"], {"source": f.get("source", "rules"),
                                               "severity": f.get("severity", "MEDIUM"), "name": f.get("name", "")})
        if rec.get("file_probs"):
            thresholds = gnn_model.get_thresholds()
            function_flags = {s for fn in rec["functions"] for s in fn.get("flagged", [])}
            for smell_id, p in rec["file_probs"].items():
                if p > thresholds.get(smell_id, 1) or smell_id in function_flags:
                    if smell_id in details:
                        details[smell_id]["source"] = "rules+gnn"
                    else:
                        pattern = LegacySmellDetector._BY_ID[smell_id]
                        details[smell_id] = {"source": "gnn", "severity": pattern["severity"], "name": pattern["name"]}
        for smell in self.learned_smells().get(rec["path"], []):
            details.setdefault(smell["smell_id"], {"source": "learned", "severity": smell["severity"],
                                                   "name": smell["name"]})
        for smell in self.context_smells().get(rec["path"], []):
            details.setdefault(smell["smell_id"], {"source": "context", "severity": smell["severity"],
                                                   "name": smell["name"]})
        score = sum(SEVERITY_WEIGHT.get(d["severity"], 1.0) for d in details.values())
        return score, details

    def hotspots(self, n: int = 10) -> list[dict[str, Any]]:
        fan_in: dict[str, int] = {}
        for uses in self._forward().values():
            for target in uses:
                fan_in[target] = fan_in.get(target, 0) + 1
        rows = []
        for rel, rec in self.files.items():
            score, details = self._risks()[rel]
            if score:
                rows.append({"path": rel, "risk": score, "smells": details, "dependents": fan_in.get(rel, 0),
                             "is_test": rec["is_test"]})
        return sorted(rows, key=lambda r: (-r["risk"], -r["dependents"], r["path"]))[:n]

    def file_context(self, rel: str) -> dict[str, Any]:
        rec = self.files[rel]
        functions = [{"qualname": f["qualname"], "lineno": f["lineno"], "end_lineno": f["end_lineno"],
                      "gnn": f.get("flagged", []),
                      "rules": sorted({r["smell_id"] for r in rec["rule_findings"]
                                       if f["lineno"] <= r["line"] <= f["end_lineno"]})}
                     for f in rec["functions"]]
        risk, details = self._risks()[rel]
        own = {s for s, d in details.items() if d["source"] != "context"}
        follow_ups = sorted({other for other, (_, other_details) in self._risks().items()
                             if other != rel and own & {s for s, d in other_details.items() if d["source"] != "context"}
                             and not self.files[other]["is_test"]})
        return {
            "path": rel, "module": rec["module"], "is_test": rec["is_test"], "parse_error": rec["parse_error"],
            "risk": risk, "smells": details,
            "functions_with_smells": [f for f in functions if f["gnn"] or f["rules"]],
            "context_smells": self.context_smells().get(rel, []),
            "learned_smells": self.learned_smells().get(rel, []),
            "dependencies": self.dependencies(rel),
            "dependents": self.dependents(rel),
            "api_surface": self.api_surface(rel),
            "related_tests": self.related_tests(rel),
            "same_smell_elsewhere": follow_ups,
            "dynamic_unresolved": rec.get("dynamic_unresolved", []),
            "gnn_available": rec.get("file_probs") is not None,
        }


_INDEXES: dict[str, CodebaseIndex] = {}
_INDEXES_LOCK = threading.Lock()


def get_index(root: str, refresh: bool = True) -> CodebaseIndex:
    """Process-wide index per workspace root, refreshed incrementally on each call."""
    key = os.path.normcase(os.path.abspath(root))
    with _INDEXES_LOCK:
        index = _INDEXES.setdefault(key, CodebaseIndex(root))
    if refresh:
        index.refresh()
    return index


def summarize_smells(details: dict[str, dict[str, str]], limit: int = 6) -> str:
    """`ID [source]` list, most severe first, capped."""
    ordered = sorted(details.items(), key=lambda kv: (SEVERITY_ORDER.get(kv[1]["severity"], 9), kv[0]))
    text = ", ".join(f"{sid} [{d['source']}]" for sid, d in ordered[:limit])
    return text + (f" (+{len(ordered) - limit} more)" if len(ordered) > limit else "")


def format_file_context(ctx: dict[str, Any], limit: int = 8) -> str:
    """Markdown summary of `CodebaseIndex.file_context`, shared by several MCP tools."""
    def more(items: list[Any]) -> str:
        return f" (+{len(items) - limit} more)" if len(items) > limit else ""

    lines = [f"### 🧭 Codebase Context for `{ctx['path']}` (module `{ctx['module']}`)"]
    deps = ctx["dependents"]
    if deps:
        lines.append(f"- **Imported by {len(deps)} module(s)** (blast radius):{more(list(deps))}")
        for path, names in list(deps.items())[:limit]:
            lines.append(f"  - `{path}`" + (f" uses `{', '.join(names)}`" if names else ""))
    else:
        lines.append("- **Imported by**: no other workspace module")
    if ctx["api_surface"]:
        lines.append(f"- **Public API used elsewhere** (keep names/signatures stable): "
                     f"`{', '.join(ctx['api_surface'][:limit * 2])}`")
    tests = ctx["related_tests"]
    lines.append(f"- **Related tests**: {', '.join(f'`{t}`' for t in tests[:limit]) or 'none found'}{more(tests)}")
    if ctx["dependencies"]:
        lines.append(f"- **Depends on**: {', '.join(f'`{p}`' for p in list(ctx['dependencies'])[:limit])}"
                     f"{more(list(ctx['dependencies']))}")
    if ctx["context_smells"]:
        lines.append("- **Codebase-level smells**:")
        for s in ctx["context_smells"][:limit]:
            where = f" (L{s['line']})" if s.get("line") else ""
            lines.append(f"  - [{s['severity']}] `{s['smell_id']}` {s['name']}{where}: {s['detail']}")
        if len(ctx["context_smells"]) > limit:
            lines.append(f"  - (+{len(ctx['context_smells']) - limit} more)")
    if ctx.get("learned_smells"):
        lines.append("- **Learned from this repository's history** (replacements the team made repeatedly):")
        for s in ctx["learned_smells"][:limit]:
            fix = "auto-fixable" if s["auto_fix"] else "suggestion"
            lines.append(f"  - `{s['smell_id']}` {s['name']}: {s['count']}x from L{s['line']} "
                         f"({s['support']} commits, consistency {s['consistency']:.0%}; {fix})")
    hot = ctx["functions_with_smells"]
    if hot:
        lines.append("- **Smells localized to functions** (GNN per-function / rules + Ruff):")
        for f in hot[:limit]:
            parts = []
            if f["rules"]:
                parts.append("rules: " + ", ".join(f["rules"][:6]) + (" ..." if len(f["rules"]) > 6 else ""))
            if f["gnn"]:
                parts.append("GNN: " + ", ".join(f["gnn"]))
            lines.append(f"  - `{f['qualname']}` (L{f['lineno']}-{f['end_lineno']}): {'; '.join(parts)}")
    if ctx["same_smell_elsewhere"]:
        others = ctx["same_smell_elsewhere"]
        lines.append(f"- **Same smells in other files** (follow-up candidates): "
                     f"{', '.join(f'`{p}`' for p in others[:limit])}{more(others)}")
    if not ctx["gnn_available"]:
        lines.append("- _GNN signals unavailable for this file (no checkpoint/torch, or unparseable source)._")
    return "\n".join(lines)
