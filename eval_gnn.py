"""
eval_gnn.py — NovusPipeline: GNN Stress-Test & Evaluation Harness

Evaluates the trained SmellGNN (single model or ensemble) beyond its own splits:
  1. In-distribution   fresh template corpus (unseen seed)
  2. Contrast suite    smelly snippet vs. structurally-identical clean lookalike
  3. Real code (OOD)   held-out third-party packages (never trained on: whole
                       packages are assigned to the test split), file- and
                       function-level, with bootstrap 95% confidence intervals;
                       plus this repository's own files
  4. Invariance        meaning-preserving rewrites (rename parameters, reorder
                       definitions, add docstrings) must not change predictions
  5. Needle-in-haystack one smell injected into increasingly large clean files
  6. Scale/robustness  latency vs. graph size, deep nesting, long expressions

Ground truth is the rule engine the GNN distills (`LegacySmellDetector`), so
these numbers measure agreement with it on unseen code.

Usage:
    python eval_gnn.py              # writes .gnn_model/eval_report.json
"""

import ast
import glob
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import gnn_model
import train_gnn
from code_graph import build_graph_from_code

LABELS = gnn_model.SMELL_LABELS
BOOTSTRAP_ROUNDS = 1000


def teacher_labels(code: str) -> Dict[str, int]:
    return dict(zip(LABELS, map(int, train_gnn.teacher_labels(code))))


def thresholds() -> Dict[str, float]:
    return gnn_model.get_thresholds()


def predictions(codes: List[str]) -> List[Optional[Dict[str, int]]]:
    """Thresholded ensemble predictions, batched; None where no graph could be built."""
    th = thresholds()
    out = []
    for res in gnn_model.analyze_many(codes):
        out.append(None if res is None else {l: int(res["probs"][l] > th[l]) for l in LABELS})
    return out


def prf(tp: int, fp: int, fn: int, negatives: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "support": tp + fn, "false_positive_rate": round(fp / negatives, 3) if negatives else 0.0}


def _counts(pairs: List[Tuple[Dict[str, int], Dict[str, int]]], label: str) -> Tuple[int, int, int, int]:
    tp = sum(1 for p, t in pairs if p[label] and t[label])
    fp = sum(1 for p, t in pairs if p[label] and not t[label])
    fn = sum(1 for p, t in pairs if not p[label] and t[label])
    return tp, fp, fn, sum(1 for _, t in pairs if not t[label])


def _macro(result: Dict) -> Optional[float]:
    # F1 is undefined for labels with no positives; those are judged by false_positive_rate.
    supported = [l for l in LABELS if result[l]["support"]]
    return round(sum(result[l]["f1"] for l in supported) / len(supported), 3) if supported else None


def score(samples: List[Tuple[str, Dict[str, int]]], bootstrap: bool = False) -> Dict:
    preds = predictions([code for code, _ in samples])
    pairs = [(p, t) for p, (_, t) in zip(preds, samples) if p is not None]
    result: Dict = {label: prf(*_counts(pairs, label)) for label in LABELS}
    result["macro_f1"] = _macro(result)
    result["samples"] = len(pairs)
    if bootstrap and pairs:
        rng = random.Random(0)
        stats: Dict[str, List[float]] = {l: [] for l in LABELS + ["macro_f1"]}
        for _ in range(BOOTSTRAP_ROUNDS):
            sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            r = {label: prf(*_counts(sample, label)) for label in LABELS}
            for l in LABELS:
                if r[l]["support"]:
                    stats[l].append(r[l]["f1"])
            m = _macro(r)
            if m is not None:
                stats["macro_f1"].append(m)
        result["ci95_f1"] = {k: [round(sorted(v)[int(0.025 * len(v))], 3), round(sorted(v)[int(0.975 * len(v)) - 1], 3)]
                             for k, v in stats.items() if len(v) >= 20}
    return result


# ---------------------------------------------------------------------------
# 1. In-distribution
# ---------------------------------------------------------------------------

def eval_in_distribution(n: int = 300) -> Dict:
    corpus = train_gnn.generate_corpus(num_samples=n, seed=987654)
    samples = [(code, dict(zip(LABELS, map(int, label)))) for code, label in corpus]
    return score(samples)


# ---------------------------------------------------------------------------
# 2. Contrast suite: (smelly, clean lookalike, label under test)
# ---------------------------------------------------------------------------

CONTRAST_CASES = [
    ("PY-SMELL-002",
     "import urllib2\ndef fetch(url: str) -> bytes:\n    return urllib2.urlopen(url).read()\n",
     "import httpx\ndef fetch(url: str) -> bytes:\n    return httpx.get(url).read()\n"),
    ("PY-SMELL-002",
     "from urllib2 import urlopen\ndef fetch(url: str) -> bytes:\n    return urlopen(url).read()\n",
     "from httpx import get\ndef fetch(url: str) -> bytes:\n    return get(url).read()\n"),
    ("PY-SMELL-003",
     "def risky(d: list) -> object:\n    try:\n        return d[0]\n    except:\n        return None\n",
     "def risky(d: list) -> object:\n    try:\n        return d[0]\n    except IndexError:\n        return None\n"),
    ("PY-SMELL-003",
     "class Repo:\n    def load(self, p: str) -> str:\n        try:\n            return open(p).read()\n        except:\n            return ''\n",
     "class Repo:\n    def load(self, p: str) -> str:\n        try:\n            return open(p).read()\n        except OSError as e:\n            return str(e)\n"),
    ("PY-SMELL-004",
     "import os\ndef run(cmd: str) -> str:\n    return os.popen(cmd).read()\n",
     "import os\ndef run(fd: int) -> str:\n    return os.fdopen(fd).read()\n"),
    ("PY-SMELL-005",
     "import pickle\ndef load(b: bytes) -> object:\n    return pickle.loads(b)\n",
     "import json\ndef load(b: bytes) -> object:\n    return json.loads(b)\n"),
    ("PY-SMELL-005",
     "import pickle\ndef save(o: object, p: str) -> None:\n    with open(p, 'wb') as f:\n        f.write(pickle.dumps(o))\n",
     "import json\ndef save(o: object, p: str) -> None:\n    with open(p, 'wb') as f:\n        f.write(json.dumps(o))\n"),
    ("PY-SMELL-006",
     "def total(items):\n    return sum(items)\n",
     "def total(items: list[int]) -> int:\n    return sum(items)\n"),
]


def eval_contrast() -> Dict:
    th = thresholds()
    per_label: Dict[str, List[int]] = {label: [] for label in LABELS}
    cases = []
    for label, smelly, clean in CONTRAST_CASES:
        assert teacher_labels(smelly)[label] == 1 and teacher_labels(clean)[label] == 0, label
        p_smelly = gnn_model.predict_smells(smelly)[label]
        p_clean = gnn_model.predict_smells(clean)[label]
        ok = int(p_smelly > th[label] and p_clean <= th[label])
        per_label[label].append(ok)
        cases.append({"label": label, "p_smelly": round(p_smelly, 3), "p_clean": round(p_clean, 3),
                      "separated": bool(ok),
                      "identical_graphs": build_graph_from_code(smelly) == build_graph_from_code(clean)})
    summary = {label: round(sum(v) / len(v), 3) for label, v in per_label.items() if v}
    return {"contrast_accuracy": summary, "cases": cases}


# ---------------------------------------------------------------------------
# 3. Real code (OOD): third-party packages (training uses the stdlib only),
#    plus this repository's own Python files
# ---------------------------------------------------------------------------

_EXTERNAL_CACHE: Dict[str, list] = {}


def external_chunks(chunks_per_package: int = 40) -> List[str]:
    """Raw (unaugmented) chunks from the held-out test packages, cached for the run."""
    if "chunks" not in _EXTERNAL_CACHE:
        _EXTERNAL_CACHE["chunks"] = [c for c, _, _ in train_gnn.package_corpus("test", chunks_per_package, 2024, False, 0)]
    return _EXTERNAL_CACHE["chunks"]


def eval_external_code() -> Dict:
    report = score([(c, teacher_labels(c)) for c in external_chunks()], bootstrap=True)
    report["packages"] = [p for p in train_gnn.list_packages(train_gnn.site_packages_root())
                          if train_gnn.package_split(p) == "test"]
    return report


def eval_external_functions(max_functions: int = 3000) -> Dict:
    """Per-function agreement on held-out packages: what the codebase index's localization relies on."""
    samples = []
    for code in external_chunks():
        for fn in train_gnn._function_nodes(ast.parse(code)):
            fn_code = ast.unparse(fn)
            samples.append((fn_code, teacher_labels(fn_code)))
            if len(samples) >= max_functions:
                return score(samples, bootstrap=True)
    return score(samples, bootstrap=True)


class _RenameParameters(ast.NodeTransformer):
    """Renames every function's parameters (and their uses inside it), except self/cls."""

    def visit_FunctionDef(self, node):
        a = node.args
        params = [p for p in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]
                  if p.arg not in ("self", "cls")]
        mapping = {p.arg: f"p{i}_renamed" for i, p in enumerate(params)}
        for p in params:
            p.arg = mapping[p.arg]
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in mapping:
                sub.id = mapping[sub.id]
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef


def eval_invariance(max_samples: int = 400) -> Dict:
    """Fraction of held-out chunks whose predicted smell set survives each meaning-preserving rewrite."""
    rng = random.Random(7)
    chunks = external_chunks()[:max_samples]

    def rename(code: str) -> str:
        return ast.unparse(_RenameParameters().visit(ast.parse(code)))

    def reorder(code: str) -> str:
        tree = ast.parse(code)
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        rest = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        rng.shuffle(rest)
        tree.body = imports + rest
        return ast.unparse(tree)

    def docstrings(code: str) -> str:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                node.body.insert(0, ast.Expr(ast.Constant("Added docstring: except: pickle.loads urllib2")))
        return ast.unparse(tree)

    base = predictions(chunks)
    report = {}
    for name, rewrite in (("rename_parameters", rename), ("reorder_definitions", reorder),
                          ("add_docstrings", docstrings)):
        rewritten = []
        for code in chunks:
            try:
                rewritten.append(rewrite(code))
            except (SyntaxError, ValueError, RecursionError):
                rewritten.append(code)
        after = predictions(rewritten)
        valid = [(b, a) for b, a in zip(base, after) if b is not None and a is not None]
        report[name] = round(sum(b == a for b, a in valid) / len(valid), 4) if valid else None
    report["samples"] = len(chunks)
    return report


def eval_real_code() -> Dict:
    root = os.path.dirname(os.path.abspath(__file__))
    files = sorted(p for p in glob.glob(os.path.join(root, "*.py")))
    samples, per_file = [], []
    th = thresholds()
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        truth = teacher_labels(code)
        probs = gnn_model.predict_smells(code)
        samples.append((code, truth))
        per_file.append({
            "file": os.path.basename(path),
            "nodes": build_graph_from_code(code)["num_nodes"],
            "teacher": [l for l in LABELS if truth[l]],
            "gnn": [l for l in LABELS if probs[l] > th[l]],
        })
    return {"metrics": score(samples), "files": per_file}


# ---------------------------------------------------------------------------
# 4. Needle in a haystack: does one smell survive pooling over large graphs?
# ---------------------------------------------------------------------------

_CLEAN_FN = "def add_{n}(x: int, y: int) -> int:\n    total = x\n    for i in range(y):\n        total += i\n    return total\n"
_NEEDLES = {
    "PY-SMELL-003": "def risky(d: list) -> object:\n    try:\n        return d[0]\n    except:\n        return None\n",
    "PY-SMELL-005": "import pickle\ndef load(b: bytes) -> object:\n    return pickle.loads(b)\n",
}


def eval_needle(sizes=(0, 5, 20, 50, 100, 250)) -> Dict:
    out: Dict[str, List[Dict]] = {}
    for label, needle in _NEEDLES.items():
        rows = []
        for n in sizes:
            fns = [_CLEAN_FN.format(n=i) for i in range(n)]
            fns.insert(n // 2, needle)
            code = "\n".join(fns)
            rows.append({"clean_functions": n,
                         "nodes": build_graph_from_code(code)["num_nodes"],
                         "p": round(gnn_model.predict_smells(code)[label], 3)})
        out[label] = rows
    return out


# ---------------------------------------------------------------------------
# 5. Scale & robustness
# ---------------------------------------------------------------------------

def eval_scale() -> Dict:
    latency = []
    for n in (10, 100, 1000, 3000):
        code = "\n".join(_CLEAN_FN.format(n=i) for i in range(n))
        t0 = time.perf_counter()
        nodes = build_graph_from_code(code)["num_nodes"]
        t1 = time.perf_counter()
        gnn_model.predict_smells(code)
        t2 = time.perf_counter()
        latency.append({"functions": n, "nodes": nodes,
                        "graph_ms": round((t1 - t0) * 1000, 1),
                        "total_ms": round((t2 - t0) * 1000, 1)})

    stress = {
        "nested_ifs_depth_90": "def f(x: int) -> int:\n" + "".join(
            "    " * (i + 1) + "if x:\n" for i in range(90)) + "    " * 91 + "return x\n",
        "binop_chain_2000": "x = " + " + ".join(["1"] * 2000) + "\n",
        "binop_chain_10000": "x = " + " + ".join(["1"] * 10000) + "\n",
        "empty_file": "",
    }
    robustness = {}
    for name, code in stress.items():
        try:
            gnn_model.predict_smells(code)
            robustness[name] = "ok"
        except Exception as e:  # report, never crash the harness
            robustness[name] = f"{type(e).__name__}: {str(e)[:80]}"
    return {"latency": latency, "robustness": robustness}


def main() -> None:
    if not gnn_model.get_gnn_status()["checkpoint_exists"]:
        sys.exit("No trained checkpoint. Run `python train_gnn.py` first.")

    report = {
        "checkpoint_metadata": gnn_model.get_gnn_status()["metadata"],
        "thresholds": thresholds(),
        "in_distribution": eval_in_distribution(),
        "contrast": eval_contrast(),
        "external_code": eval_external_code(),
        "external_functions": eval_external_functions(),
        "invariance": eval_invariance(),
        "real_code": eval_real_code(),
        "needle": eval_needle(),
        "scale": eval_scale(),
    }

    out_path = os.path.join(os.path.dirname(gnn_model.CHECKPOINT_PATH), "eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "checkpoint_metadata"}, indent=2))
    print(f"\n[NovusPipeline GNN] Full report written to {out_path}")


if __name__ == "__main__":
    main()
