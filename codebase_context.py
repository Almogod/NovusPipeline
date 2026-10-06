"""
codebase_context.py — NovusPipeline Phase 6: Codebase Context Index

Builds a persistent, incrementally refreshed index of every Python file in the
workspace so tools can reason about a file *in the context of its codebase*:

  - GNN structure:  per-function smell probabilities (localizes the file-level
                    GNN to functions) and graph embeddings (structural
                    similarity search: "where else does this pattern occur?")
  - rule findings:  `LegacySmellDetector` results per file
  - import graph:   who imports a module, which tests cover it, and which of
                    its symbols other modules actually use (its API surface,
                    which a modernization must not break)

Files are re-analyzed only when their content hash changes, or when the GNN
checkpoint changes. The index is stored at <workspace>/.novus_index/index.json.
The GNN is optional: without torch or a checkpoint, the index still provides
the import graph and rule findings.
"""

import ast
import hashlib
import json
import math
import os
import textwrap
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gnn_model
from modernizer import LegacySmellDetector

INDEX_VERSION = 2  # bump when stored record format changes; older indexes are rebuilt
INDEX_DIRNAME = ".novus_index"
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", ".env", "__pycache__", "node_modules",
    ".chroma_db", ".gnn_model", INDEX_DIRNAME, "build", "dist", ".tox", ".nox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages", ".eggs",
}
MAX_FILE_BYTES = 1_000_000
MAX_FILES = 5000
SEVERITY_WEIGHT = {"CRITICAL": 3.0, "HIGH": 2.0, "MEDIUM": 1.0}
# PY-SMELL-002 is about import statements, so it is attributed at file level only.
FUNCTION_LEVEL_SMELLS = [s for s in gnn_model.SMELL_LABELS if s != "PY-SMELL-002"]


def _module_name(rel_path: str) -> str:
    parts = rel_path[:-3].split("/") if rel_path.endswith(".py") else rel_path.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_test_file(rel_path: str) -> bool:
    base = os.path.basename(rel_path)
    dirs = rel_path.split("/")[:-1]
    return base.startswith("test_") or base.endswith("_test.py") or any(d in ("tests", "test") for d in dirs)


def _functions(tree: ast.Module, code: str) -> List[Dict[str, Any]]:
    """Top-level functions and methods of top-level classes, with standalone source."""
    out = []

    def add(node: ast.AST, qualname: str) -> None:
        segment = ast.get_source_segment(code, node, padded=True)
        if segment:
            out.append({"qualname": qualname, "lineno": node.lineno, "end_lineno": node.end_lineno,
                        "source": textwrap.dedent(segment)})

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, node.name)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(item, f"{node.name}.{item.name}")
    return out


def _imports(tree: ast.Module, module: str, is_package: bool) -> List[Dict[str, Any]]:
    """
    Import facts with relative imports resolved to absolute module names:
    [{"module": "pkg.mod", "names": [symbols used from it]}]. For `import m as x`
    the names are the attributes accessed as `x.<attr>`.
    """
    package = module.split(".") if is_package else module.split(".")[:-1]
    facts: List[Dict[str, Any]] = []
    aliases: Dict[str, str] = {}
    # `from pkg import mod` may bind a submodule or a plain symbol; which one is
    # only known against the module map, so record it as a candidate ("exact").
    from_aliases: Dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                facts.append({"module": a.name, "names": []})
                # `import a.b` binds `a`, so only plain or aliased imports map a
                # local name to exactly this module.
                if a.asname or "." not in a.name:
                    aliases[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - (node.level - 1)] if node.level > 1 else package
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if target:
                facts.append({"module": target, "names": [a.name for a in node.names if a.name != "*"]})
                for a in node.names:
                    if a.name != "*":
                        from_aliases[a.asname or a.name] = f"{target}.{a.name}"

    used: Dict[str, set] = {}
    candidate_used: Dict[str, set] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in aliases:
                used.setdefault(aliases[node.value.id], set()).add(node.attr)
            elif node.value.id in from_aliases:
                candidate_used.setdefault(from_aliases[node.value.id], set()).add(node.attr)
    for fact in facts:
        if not fact["names"] and fact["module"] in used:
            fact["names"] = sorted(used[fact["module"]])
    facts.extend({"module": m, "names": sorted(attrs), "exact": True} for m, attrs in candidate_used.items())
    return facts


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class CodebaseIndex:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.index_path = os.path.join(self.root, INDEX_DIRNAME, "index.json")
        self.files: Dict[str, Dict[str, Any]] = {}
        self.meta: Dict[str, Any] = {}
        self.gnn_error: Optional[str] = None
        self._lock = threading.Lock()
        self._cache: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------ build

    def _load(self) -> None:
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
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

    def _gnn_signature(self) -> Optional[str]:
        status = gnn_model.get_gnn_status()
        if not (status["checkpoint_exists"] and status["torch_available"]):
            return None
        return str(status["metadata"].get("trained_at")) + str(os.path.getmtime(gnn_model.CHECKPOINT_PATH))

    def _analyze_file(self, rel: str, code: str, sha: str) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "path": rel, "module": _module_name(rel), "sha1": sha, "is_test": _is_test_file(rel),
            "is_package": rel.endswith("__init__.py"), "parse_error": None, "imports": [],
            "functions": [], "file_probs": None, "file_embedding": None,
            "rule_findings": [{"smell_id": f["smell_id"], "line": f["line_number"]}
                              for f in LegacySmellDetector.scan_code(code, rel)],
        }
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
            record["parse_error"] = f"{type(e).__name__}: {e}"
            return record
        record["imports"] = _imports(tree, record["module"], record["is_package"])
        record["functions"] = _functions(tree, code)
        record["_source"] = code
        return record

    def _run_gnn(self, records: List[Dict[str, Any]]) -> None:
        """Fills file/function GNN fields in place, in one batched pass."""
        jobs: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]], str]] = []
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

    def refresh(self, force: bool = False) -> Dict[str, Any]:
        """Re-analyzes new/changed files (all files if `force` or the GNN changed)."""
        with self._lock:
            start = time.perf_counter()
            gnn_sig = self._gnn_signature()
            rebuild_all = force or gnn_sig != self.meta.get("gnn_signature")

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
                self.meta.update({"gnn_signature": gnn_sig, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                self._save()
            return {"files": len(self.files), "reanalyzed": len(changed), "removed": len(removed),
                    "functions": sum(len(r["functions"]) for r in self.files.values()),
                    "gnn": self.gnn_error is None and gnn_sig is not None,
                    "seconds": round(time.perf_counter() - start, 2)}

    # ---------------------------------------------------------------- queries

    def _module_map(self) -> Dict[str, str]:
        """Module name -> path, including `src.`-less aliases for src layouts."""
        mapping: Dict[str, str] = {}
        for rel, rec in self.files.items():
            mapping.setdefault(rec["module"], rel)
            if rec["module"].startswith("src."):
                mapping.setdefault(rec["module"][4:], rel)
        return mapping

    def _resolve(self, rec: Dict[str, Any]) -> Dict[str, set]:
        """Workspace paths this file imports -> symbols it uses from each."""
        modules = self._module_map()
        out: Dict[str, set] = {}
        for fact in rec["imports"]:
            target, names = fact["module"], fact["names"]
            if fact.get("exact"):  # attributes of a `from pkg import mod` candidate
                path = modules.get(target)
                if path and path != rec["path"]:
                    out.setdefault(path, set()).update(names)
                continue
            for name in names:
                sub = modules.get(f"{target}.{name}")
                if sub:
                    out.setdefault(sub, set())
            plain = [n for n in names if f"{target}.{n}" not in modules]
            parts = target.split(".")
            for i in range(len(parts), 0, -1):
                path = modules.get(".".join(parts[:i]))
                if path:
                    if path != rec["path"]:
                        out.setdefault(path, set()).update(plain if i == len(parts) else [])
                    break
        return out

    def _forward(self) -> Dict[str, Dict[str, set]]:
        """Resolved import graph for all files, cached until the next change."""
        if "forward" not in self._cache:
            self._cache["forward"] = {rel: self._resolve(rec) for rel, rec in self.files.items()}
        return self._cache["forward"]

    def _risks(self) -> Dict[str, Tuple[float, Dict[str, str]]]:
        if "risks" not in self._cache:
            self._cache["risks"] = {rel: self.file_risk(rec) for rel, rec in self.files.items()}
        return self._cache["risks"]

    def normalize(self, path: str) -> Optional[str]:
        full = path if os.path.isabs(path) else os.path.join(self.root, path)
        try:
            rel = os.path.relpath(os.path.abspath(full), self.root).replace(os.sep, "/")
        except ValueError:  # different drive on Windows
            return None
        return rel if rel in self.files else None

    def dependencies(self, rel: str) -> Dict[str, List[str]]:
        return {p: sorted(n) for p, n in self._forward()[rel].items()}

    def dependents(self, rel: str) -> Dict[str, List[str]]:
        return {other: sorted(uses[rel]) for other, uses in self._forward().items() if rel in uses}

    def related_tests(self, rel: str) -> List[str]:
        tests = {p for p in self.dependents(rel) if self.files[p]["is_test"]}
        stem = os.path.basename(rel)[:-3]
        tests.update(p for p, r in self.files.items()
                     if r["is_test"] and os.path.basename(p) in (f"test_{stem}.py", f"{stem}_test.py"))
        return sorted(tests)

    def api_surface(self, rel: str) -> List[str]:
        names = set()
        for used in self.dependents(rel).values():
            names.update(used)
        return sorted(names)

    def _centered(self, kind: str) -> Dict[Tuple[str, str], List[float]]:
        """
        Embeddings minus their mean over the index. GNN embeddings are
        post-ReLU/max-pool, so raw vectors share a large common component and
        every pair scores ~0.97 cosine; centering makes similarity discriminative.
        """
        key = f"centered_{kind}"
        if key not in self._cache:
            vectors: Dict[Tuple[str, str], List[float]] = {}
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

    def similar(self, rel: str, qualname: str = "", n: int = 5) -> List[Dict[str, Any]]:
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
            row: Dict[str, Any] = {"path": p, "similarity": round(_cosine(query, vec), 3)}
            if qualname:
                row.update({"qualname": q, "lineno": flags[(p, q)][1],
                            "shared_smells": sorted(flags[(rel, qualname)][0] & flags[(p, q)][0])})
            scored.append(row)
        return sorted(scored, key=lambda s: -s["similarity"])[:n]

    def file_risk(self, rec: Dict[str, Any]) -> Tuple[float, Dict[str, str]]:
        """
        Severity-weighted smell score; each smell counts once, from rules or GNN. The GNN
        signal is the file-level prediction OR any function-level flag: the file-level model
        under-detects a single slip in a large file, which per-function predictions catch.
        """
        score, sources = 0.0, {}
        rule_ids = {f["smell_id"] for f in rec["rule_findings"]}
        thresholds = gnn_model.get_thresholds() if rec.get("file_probs") else {}
        function_flags = {s for fn in rec["functions"] for s in fn.get("flagged", [])}
        for smell_id in sorted(rule_ids | set(rec.get("file_probs") or {})):
            in_rules = smell_id in rule_ids
            in_gnn = bool(rec.get("file_probs")) and (
                rec["file_probs"].get(smell_id, 0) > thresholds.get(smell_id, 1) or smell_id in function_flags)
            if in_rules or in_gnn:
                severity = LegacySmellDetector._BY_ID[smell_id]["severity"]
                score += SEVERITY_WEIGHT[severity]
                sources[smell_id] = "rules+gnn" if in_rules and in_gnn else "rules" if in_rules else "gnn"
        return score, sources

    def hotspots(self, n: int = 10) -> List[Dict[str, Any]]:
        fan_in: Dict[str, int] = {}
        for uses in self._forward().values():
            for target in uses:
                fan_in[target] = fan_in.get(target, 0) + 1
        rows = []
        for rel, rec in self.files.items():
            score, sources = self._risks()[rel]
            if score:
                rows.append({"path": rel, "risk": score, "smells": sources, "dependents": fan_in.get(rel, 0),
                             "is_test": rec["is_test"]})
        return sorted(rows, key=lambda r: (-r["risk"], -r["dependents"], r["path"]))[:n]

    def file_context(self, rel: str) -> Dict[str, Any]:
        rec = self.files[rel]
        functions = [{"qualname": f["qualname"], "lineno": f["lineno"], "end_lineno": f["end_lineno"],
                      "gnn": f.get("flagged", []),
                      "rules": sorted({r["smell_id"] for r in rec["rule_findings"]
                                       if f["lineno"] <= r["line"] <= f["end_lineno"]})}
                     for f in rec["functions"]]
        risk, sources = self._risks()[rel]
        follow_ups = [(other, smell_id) for smell_id in sources
                      for other, (_, other_sources) in self._risks().items()
                      if other != rel and smell_id in other_sources]
        return {
            "path": rel, "module": rec["module"], "is_test": rec["is_test"], "parse_error": rec["parse_error"],
            "risk": risk, "smells": sources,
            "functions_with_smells": [f for f in functions if f["gnn"] or f["rules"]],
            "dependencies": self.dependencies(rel),
            "dependents": self.dependents(rel),
            "api_surface": self.api_surface(rel),
            "related_tests": self.related_tests(rel),
            "same_smell_elsewhere": sorted({p for p, _ in follow_ups}),
            "gnn_available": rec.get("file_probs") is not None,
        }


_INDEXES: Dict[str, CodebaseIndex] = {}
_INDEXES_LOCK = threading.Lock()


def get_index(root: str, refresh: bool = True) -> CodebaseIndex:
    """Process-wide index per workspace root, refreshed incrementally on each call."""
    key = os.path.normcase(os.path.abspath(root))
    with _INDEXES_LOCK:
        index = _INDEXES.setdefault(key, CodebaseIndex(root))
    if refresh:
        index.refresh()
    return index


def format_file_context(ctx: Dict[str, Any], limit: int = 8) -> str:
    """Markdown summary of `CodebaseIndex.file_context`, shared by several MCP tools."""
    def more(items: List[Any]) -> str:
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
    hot = ctx["functions_with_smells"]
    if hot:
        lines.append("- **Smells localized to functions** (GNN per-function / rule engine):")
        for f in hot[:limit]:
            parts = []
            if f["rules"]:
                parts.append("rules: " + ", ".join(f["rules"]))
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
