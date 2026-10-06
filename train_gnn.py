"""
train_gnn.py — NovusPipeline Phase 5: GNN Smell Classifier Training

Trains SmellGNN (gnn_model.py) by distilling `modernizer.LegacySmellDetector`:
a synthetic corpus is generated from templates, labeled by the rule-based
detector, and the GNN learns to reproduce those labels from the AST graph.

The corpus is built to remove the shortcuts the first version learned:
  - hard negatives: clean code structurally identical to each smell but using
    a different API (requests vs urllib2, yaml vs pickle, `except KeyError:`
    vs bare `except:`), forcing the model to use identifier/field features
  - typed and untyped variants of every template, so "missing type hints"
    does not simply co-occur with every other smell
  - smells placed in methods as well as top-level functions
  - file sizes from 1 to ~40 functions, closer to real source files

Usage:
    python train_gnn.py                       # train with defaults, save checkpoint
    python train_gnn.py --conv gcn --seed 7   # override architecture / seed
    python train_gnn.py --sweep 42 7 123      # compare seeds without saving
"""

import argparse
import ast
import hashlib
import math
import os
import random
import statistics
import sysconfig
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from code_graph import build_graph_from_code
from gnn_model import SMELL_LABELS, GNNConfig, build_model, graph_dict_to_pyg_data, save_checkpoint
from modernizer import LegacySmellDetector

# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------

def teacher_labels(code: str) -> List[float]:
    """
    LegacySmellDetector labels for `code` — the exact rule engine the server
    runs, so GNN/rule agreement in `analyze_legacy_codebase` is meaningful. The
    detector masks strings/comments and uses the AST for 003/006 itself.
    """
    found = {f["smell_id"] for f in LegacySmellDetector.scan_code(code, "teacher.py")}
    return [1.0 if label in found else 0.0 for label in SMELL_LABELS]


# ---------------------------------------------------------------------------
# Templates: (imports, function body). Placeholders:
#   {fn} function name, {a}/{b} parameter names, {ta}/{tb} optional parameter
#   annotations, {rt} optional return annotation.
# The evaluation contrast suite (eval_gnn.py) deliberately uses *different*
# lookalike APIs (httpx, json, os.fdopen, IndexError/OSError) than the hard
# negatives below, so it measures generalization rather than memorization.
# ---------------------------------------------------------------------------

Template = Tuple[List[str], str]

SMELLY_TEMPLATES: Dict[str, List[Template]] = {
    "PY-SMELL-002": [
        (["import urllib2"], "def {fn}({a}{ta}){rt}:\n    return urllib2.urlopen({a}).read()\n"),
        (["from urllib2 import urlopen"], "def {fn}({a}{ta}){rt}:\n    resp = urlopen({a})\n    return resp.read()\n"),
        (["import urllib2"], "def {fn}({a}{ta}, {b}{tb}){rt}:\n    req = urllib2.Request({a}, data={b})\n    return urllib2.urlopen(req).read()\n"),
    ],
    "PY-SMELL-003": [
        ([], "def {fn}({a}{ta}){rt}:\n    try:\n        return {a}[0]\n    except:\n        return None\n"),
        ([], "def {fn}({a}{ta}){rt}:\n    try:\n        {b} = int({a})\n    except:\n        {b} = 0\n    return {b}\n"),
        ([], "def {fn}({a}{ta}){rt}:\n    for item in {a}:\n        try:\n            item.close()\n        except:\n            continue\n    return None\n"),
    ],
    "PY-SMELL-004": [
        (["import os"], "def {fn}({a}{ta}){rt}:\n    return os.popen({a}).read()\n"),
        (["import os"], "def {fn}({a}{ta}){rt}:\n    {b} = os.popen('ls ' + {a})\n    return {b}.readlines()\n"),
    ],
    "PY-SMELL-005": [
        (["import pickle"], "def {fn}({a}{ta}){rt}:\n    with open({a}, 'rb') as fh:\n        return pickle.load(fh)\n"),
        (["import pickle"], "def {fn}({a}{ta}, {b}{tb}){rt}:\n    with open({b}, 'wb') as fh:\n        fh.write(pickle.dumps({a}))\n"),
        (["import pickle"], "def {fn}({a}{ta}){rt}:\n    return pickle.loads({a})\n"),
    ],
}

HARD_NEGATIVE_TEMPLATES: List[Template] = [
    (["import requests"], "def {fn}({a}{ta}){rt}:\n    return requests.get({a}).json()\n"),
    (["from requests import get"], "def {fn}({a}{ta}){rt}:\n    resp = get({a})\n    return resp.json()\n"),
    (["import requests"], "def {fn}({a}{ta}, {b}{tb}){rt}:\n    req = requests.Request({a}, data={b})\n    return requests.Session().send(req).json()\n"),
    ([], "def {fn}({a}{ta}){rt}:\n    try:\n        return {a}[0]\n    except KeyError:\n        return None\n"),
    ([], "def {fn}({a}{ta}){rt}:\n    try:\n        {b} = int({a})\n    except (TypeError, ValueError):\n        {b} = 0\n    return {b}\n"),
    ([], "def {fn}({a}{ta}){rt}:\n    for item in {a}:\n        try:\n            item.close()\n        except Exception as exc:\n            continue\n    return None\n"),
    (["import os"], "def {fn}({a}{ta}){rt}:\n    return os.getenv({a}).strip()\n"),
    (["import os"], "def {fn}({a}{ta}){rt}:\n    {b} = os.listdir('/' + {a})\n    return {b}.copy()\n"),
    (["import subprocess"], "def {fn}({a}{ta}){rt}:\n    return subprocess.run({a}, capture_output=True).stdout\n"),
    (["import yaml"], "def {fn}({a}{ta}){rt}:\n    with open({a}, 'rb') as fh:\n        return yaml.safe_load(fh)\n"),
    (["import zlib"], "def {fn}({a}{ta}, {b}{tb}){rt}:\n    with open({b}, 'wb') as fh:\n        fh.write(zlib.compress({a}))\n"),
    (["import tomllib"], "def {fn}({a}{ta}){rt}:\n    return tomllib.loads({a})\n"),
]

CLEAN_TEMPLATES: List[Template] = [
    ([], "def {fn}({a}{ta}, {b}{tb}){rt}:\n    return {a} + {b}\n"),
    ([], "def {fn}({a}{ta}){rt}:\n    total = 0\n    for item in {a}:\n        total += item\n    return total\n"),
    ([], "def {fn}({a}{ta}){rt}:\n    return {a} % 2 == 0\n"),
    ([], "def {fn}({a}{ta}, {b}{tb}){rt}:\n    if {b} == 0:\n        return None\n    return {a} / {b}\n"),
    ([], "def {fn}({a}{ta}){rt}:\n    return [v for v in {a} if v > 0]\n"),
    ([], "def {fn}({a}{ta}){rt}:\n    {b} = {{}}\n    for key, value in {a}.items():\n        {b}[value] = key\n    return {b}\n"),
    ([], "def {fn}({a}{ta}){rt}:\n    while {a} > 1:\n        {a} = {a} // 2\n    return {a}\n"),
]

PARAM_NAMES = ["data", "path", "url", "value", "payload", "item_list", "src", "raw",
               "target", "buf", "cfg", "record", "blob", "cmd", "query", "node"]
FUNC_VERBS = ["load", "fetch", "parse", "run", "read", "build", "process", "handle",
              "compute", "get", "make", "sync", "resolve", "collect"]
PARAM_TYPES = [": str", ": bytes", ": int", ": list", ": dict", ": object"]
RETURN_TYPES = [" -> object", " -> str", " -> int | None", " -> list", " -> None", " -> bytes"]


def _render(template: Template, rng: random.Random, n: int, typed: bool, as_method: bool) -> Tuple[List[str], str]:
    imports, body = template
    a, b = rng.sample(PARAM_NAMES, 2)
    source = body.format(
        fn=f"{rng.choice(FUNC_VERBS)}_{n}",
        a=a, b=b,
        ta=rng.choice(PARAM_TYPES) if typed else "",
        tb=rng.choice(PARAM_TYPES) if typed else "",
        rt=rng.choice(RETURN_TYPES) if typed else "",
    )
    if as_method:
        source = source.replace("(", "(self, ", 1)
        source = f"class Service{n}:\n" + "".join("    " + line + "\n" for line in source.splitlines())
    return imports, source


def generate_corpus(num_samples: int, seed: int) -> List[Tuple[str, List[float]]]:
    """Returns [(source_code, multi_hot_label_vector), ...], labeled by the teacher."""
    rng = random.Random(seed)
    smell_ids = list(SMELLY_TEMPLATES.keys())
    samples = []

    for i in range(num_samples):
        picks: List[Template] = []
        num_smells = rng.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0]
        for sid in rng.sample(smell_ids, k=num_smells):
            picks.append(rng.choice(SMELLY_TEMPLATES[sid]))
        for _ in range(rng.choices([0, 1, 2, 3], weights=[0.25, 0.35, 0.25, 0.15])[0]):
            picks.append(rng.choice(HARD_NEGATIVE_TEMPLATES))
        # Log-uniform filler count (0..~80) so graph sizes span tiny to real-file scale.
        for _ in range(int(math.exp(rng.uniform(0, math.log(81)))) - 1):
            picks.append(rng.choice(CLEAN_TEMPLATES))
        if not picks:
            picks.append(rng.choice(CLEAN_TEMPLATES))
        rng.shuffle(picks)

        # Typing discipline is a per-file property in real code (fully typed,
        # mostly typed, mixed, untyped), not an independent coin flip per function.
        typed_rate = rng.choices([1.0, 0.9, 0.5, 0.0], weights=[0.45, 0.2, 0.2, 0.15])[0]
        imports: List[str] = []
        bodies: List[str] = []
        for j, template in enumerate(picks):
            imps, src = _render(template, rng, i * 100 + j,
                                typed=rng.random() < typed_rate, as_method=rng.random() < 0.25)
            imports.extend(imp for imp in imps if imp not in imports)
            bodies.append(src)

        code = "\n".join(imports) + ("\n\n" if imports else "") + "\n\n".join(bodies)
        samples.append((code, teacher_labels(code)))
    return samples


class _Counterfactual(ast.NodeTransformer):
    """
    Rewrites real code so the same module appears both with and without a
    smell: add/remove return annotations (PY-SMELL-006) and make excepts
    bare/typed (PY-SMELL-003). Labels are recomputed by the teacher afterwards,
    so the rewrite only has to produce valid code, not exact labels.
    """

    # Tried and rejected: an "annotate every def but one" / "exactly one bare
    # except" variant. It raised external 006 recall 0.88 -> 0.96 but its
    # false-positive rate 2% -> 15% (see phases.md, Phase 5 evaluation).
    def __init__(self, rng: random.Random) -> None:
        self.annotate = rng.random() < 0.5
        self.strip = not self.annotate and rng.random() < 0.3
        self.bare = rng.random() < 0.25
        self.unbare = not self.bare and rng.random() < 0.5

    def _visit_def(self, node):
        self.generic_visit(node)
        a = node.args
        params = a.posonlyargs + a.args + a.kwonlyargs + [p for p in (a.vararg, a.kwarg) if p]
        if params and params[0].arg in ("self", "cls"):
            params = params[1:]
        if self.annotate:
            if node.returns is None:
                node.returns = ast.Name(id="object", ctx=ast.Load())
            for p in params:
                if p.annotation is None:
                    p.annotation = ast.Name(id="object", ctx=ast.Load())
        elif self.strip:
            node.returns = None
            for p in params:
                p.annotation = None
        return node

    visit_FunctionDef = _visit_def
    visit_AsyncFunctionDef = _visit_def

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.ExceptHandler:
        self.generic_visit(node)
        if self.bare and node.type is not None:
            node.type, node.name = None, None
        elif self.unbare and node.type is None:
            node.type = ast.Name(id="Exception", ctx=ast.Load())
        return node


def _flatten_oversized(stmts: List[ast.stmt], max_nodes: int) -> List[ast.stmt]:
    """Statements that fit in a chunk; oversized classes contribute their methods instead."""
    out: List[ast.stmt] = []
    for stmt in stmts:
        size = sum(1 for _ in ast.walk(stmt))
        if size <= max_nodes:
            out.append(stmt)
        elif isinstance(stmt, ast.ClassDef):
            out.extend(_flatten_oversized(stmt.body, max_nodes))
    return out


def _function_nodes(module: ast.Module) -> List[ast.AST]:
    """Top-level functions and methods of top-level classes (what the codebase index localizes to)."""
    out: List[ast.AST] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(node)
        elif isinstance(node, ast.ClassDef):
            out.extend(n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return out


def real_code_corpus(root: str, num_samples: int, seed: int, max_nodes: int = 2500,
                     augment: bool = False, functions_per_chunk: int = 0,
                     exclude_dirs: Tuple[str, ...] = ("test", "tests", "site-packages", "lib2to3", "idlelib")
                     ) -> List[Tuple[str, List[float], str]]:
    """
    Teacher-labeled chunks of real Python source under `root`. Each file's
    top-level statements are grouped into chunks of at most ~`max_nodes` AST
    nodes, so samples look like real modules (docstrings, decorators, nested
    functions, f-strings) at bounded size. Files are visited in a seeded order
    for reproducibility. With `augment`, each chunk is counterfactually
    rewritten (see `_Counterfactual`) before labeling. With
    `functions_per_chunk`, up to that many individual functions per chunk are
    also emitted as standalone samples, matching how the codebase index runs
    the GNN per function. Only chunks count toward `num_samples`.

    Each returned sample carries its source file as a third element so callers
    can split train/test by file rather than by chunk.
    """
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in exclude_dirs and not d.startswith("."))
        paths.extend(os.path.join(dirpath, f) for f in sorted(filenames) if f.endswith(".py"))
    rng = random.Random(seed)
    rng.shuffle(paths)

    samples: List[Tuple[str, List[float], str]] = []
    chunks = 0
    for path in paths:
        if chunks >= num_samples:
            break
        try:
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (SyntaxError, UnicodeDecodeError, ValueError, RecursionError, OSError):
            continue

        chunk: List[ast.stmt] = []
        size = 0
        for stmt in _flatten_oversized(tree.body, max_nodes) + [None]:
            stmt_size = sum(1 for _ in ast.walk(stmt)) if stmt is not None else 0
            if chunk and (stmt is None or size + stmt_size > max_nodes):
                try:
                    module = ast.Module(body=chunk, type_ignores=[])
                    if augment:
                        module = ast.fix_missing_locations(_Counterfactual(rng).visit(module))
                    code = ast.unparse(module)
                    build_graph_from_code(code)
                    samples.append((code, teacher_labels(code), path))
                    chunks += 1
                    functions = _function_nodes(module)
                    for fn in rng.sample(functions, k=min(functions_per_chunk, len(functions))):
                        fn_code = ast.unparse(fn)
                        build_graph_from_code(fn_code)
                        samples.append((fn_code, teacher_labels(fn_code), path))
                except (SyntaxError, ValueError, RecursionError):
                    pass
                chunk, size = [], 0
            if stmt is not None:
                chunk.append(stmt)
                size += stmt_size
            if chunks >= num_samples:
                break
    return samples


def stdlib_root() -> str:
    return sysconfig.get_paths()["stdlib"]


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def _to_dataset(samples):
    return [graph_dict_to_pyg_data(build_graph_from_code(code), y=label) for code, label in samples]


def _predict(model, loader):
    import torch

    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for batch in loader:
            probs.append(torch.sigmoid(model(batch, batch.batch)))
            targets.append(batch.y)
    return torch.cat(probs), torch.cat(targets)


def _f1(pred, target) -> float:
    tp = float((pred * target).sum())
    fp = float((pred * (1 - target)).sum())
    fn = float(((1 - pred) * target).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def calibrate_thresholds(probs, targets) -> Dict[str, float]:
    """Per-label threshold maximizing F1 on the validation split (ties -> closest to 0.5)."""
    thresholds = {}
    for i, label in enumerate(SMELL_LABELS):
        candidates = [round(0.05 * k, 2) for k in range(1, 20)]
        best = max(candidates, key=lambda t: (_f1((probs[:, i] > t).float(), targets[:, i]), -abs(t - 0.5)))
        thresholds[label] = best
    return thresholds


def evaluate_split(probs, targets, thresholds: Dict[str, float]) -> Dict[str, object]:
    """F1 per label; labels with no positives in the split get None (F1 undefined) and
    are excluded from macro-F1, with their false-positive rate reported instead."""
    preds = (probs > probs.new_tensor([thresholds[l] for l in SMELL_LABELS])).float()
    per_label: Dict[str, Optional[float]] = {}
    fp_rate: Dict[str, float] = {}
    for i, label in enumerate(SMELL_LABELS):
        positives = float(targets[:, i].sum())
        negatives = len(targets) - positives
        per_label[label] = round(_f1(preds[:, i], targets[:, i]), 4) if positives else None
        fp_rate[label] = round(float((preds[:, i] * (1 - targets[:, i])).sum()) / negatives, 4) if negatives else 0.0
    supported = [f1 for f1 in per_label.values() if f1 is not None]
    return {
        "macro_f1": round(sum(supported) / len(supported), 4) if supported else None,
        "per_label_f1": per_label,
        "false_positive_rate": fp_rate,
        "exact_match": round(float((preds == targets).all(dim=1).float().mean()), 4),
        "samples": len(targets),
    }


def build_datasets(seed: int, num_samples: int = 2000, real_samples: int = 2000) -> Dict[str, list]:
    """Seeded train/val/test (+ real-code-only test) PyG datasets. Independent of model config,
    so a sweep can build them once per seed and reuse them across architectures."""
    synthetic = generate_corpus(num_samples, seed)
    random.Random(seed).shuffle(synthetic)
    n_train, n_val = int(0.7 * len(synthetic)), int(0.15 * len(synthetic))
    splits = {"train": synthetic[:n_train], "val": synthetic[n_train:n_train + n_val],
              "test": synthetic[n_train + n_val:]}

    # Real chunks are split by source file (not by chunk) so near-duplicate
    # chunks of one module never straddle train and test.
    real_test: List[Tuple[str, List[float]]] = []
    if real_samples:
        for code, label, path in real_code_corpus(stdlib_root(), real_samples, seed, augment=True,
                                                  functions_per_chunk=3):
            bucket = int(hashlib.md5(os.path.basename(path).encode()).hexdigest(), 16) % 100
            split = "val" if bucket < 15 else "test" if bucket < 30 else "train"
            splits[split].append((code, label))
            if split == "test":
                real_test.append((code, label))

    return {
        "train": _to_dataset(splits["train"]),
        "val": _to_dataset(splits["val"]),
        "test": _to_dataset(splits["test"]),
        "test_real": _to_dataset(real_test),
    }


def run_training(config: GNNConfig, seed: int, num_samples: int = 2000, real_samples: int = 2000,
                 max_epochs: int = 80, patience: int = 10, save: bool = True,
                 verbose: bool = True, datasets: Optional[Dict[str, list]] = None) -> Dict[str, object]:
    import torch
    from torch_geometric.loader import DataLoader

    random.seed(seed)
    torch.manual_seed(seed)
    log = print if verbose else (lambda *a, **k: None)

    datasets = datasets or build_datasets(seed, num_samples, real_samples)
    train_set, val_set = datasets["train"], datasets["val"]
    test_set, real_test_set = datasets["test"], datasets["test_real"]

    label_rate = torch.cat([d.y for d in train_set]).mean(dim=0)
    log(f"[NovusPipeline GNN] {len(train_set)}/{len(val_set)}/{len(test_set)} train/val/test graphs, "
        f"label rates {dict(zip(SMELL_LABELS, [round(float(r), 2) for r in label_rate]))}")

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_weight = ((1 - label_rate) / label_rate.clamp(min=1e-3)).clamp(1.0, 10.0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch, batch.batch), batch.y)
            loss.backward()
            optimizer.step()

        val_probs, val_targets = _predict(model, val_loader)
        val_loss = float(torch.nn.functional.binary_cross_entropy(val_probs.clamp(1e-6, 1 - 1e-6), val_targets))
        if val_loss < best_loss - 1e-4:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if epoch % 5 == 0:
            log(f"[NovusPipeline GNN] epoch {epoch:3d}  val BCE {val_loss:.4f}  (best {best_loss:.4f})  "
                f"{(time.perf_counter() - start) / epoch:.1f}s/epoch")
        if bad_epochs >= patience:
            log(f"[NovusPipeline GNN] early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    val_probs, val_targets = _predict(model, val_loader)
    thresholds = calibrate_thresholds(val_probs, val_targets)
    test_probs, test_targets = _predict(model, test_loader)
    metrics = {
        "seed": seed,
        "epochs_trained": epoch,
        "num_samples": num_samples,
        "real_samples": real_samples,
        "val": evaluate_split(val_probs, val_targets, thresholds),
        "test": evaluate_split(test_probs, test_targets, thresholds),
    }
    if real_test_set:
        real_probs, real_targets = _predict(model, DataLoader(real_test_set, batch_size=64))
        metrics["test_real"] = evaluate_split(real_probs, real_targets, thresholds)
    if save:
        save_checkpoint(model, config, thresholds, metrics)
        log(f"[NovusPipeline GNN] saved checkpoint; thresholds {thresholds}")
    log(f"[NovusPipeline GNN] test metrics: {metrics['test']}")
    return metrics


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conv", choices=["gcn", "sage", "sage_max", "gin"])
    parser.add_argument("--layers", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--no-tokens", action="store_true", help="ablation: drop identifier features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=2000, help="synthetic template samples")
    parser.add_argument("--real-samples", type=int, default=2000,
                        help="teacher-labeled chunks of real code from the Python stdlib (0 to disable)")
    parser.add_argument("--sweep", type=int, nargs="+", metavar="SEED",
                        help="train once per seed without saving; report mean/std macro-F1 on "
                             "held-out real-code files (the model-selection metric)")
    args = parser.parse_args(argv)

    config = GNNConfig()
    overrides = {k: v for k, v in {"conv": args.conv, "num_layers": args.layers,
                                     "hidden_dim": args.hidden}.items() if v is not None}
    if args.no_tokens:
        overrides["use_tokens"] = False
    config = replace(config, **overrides)

    if args.sweep:
        runs = [run_training(config, s, args.samples, args.real_samples, save=False, verbose=False)
                for s in args.sweep]
        split = "test_real" if "test_real" in runs[0] else "test"
        scores = [r[split]["macro_f1"] for r in runs]
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"[NovusPipeline GNN] {config} -> {split} macro-F1 {statistics.mean(scores):.4f} +/- {std:.4f} "
              f"{scores}; mixed test {[r['test']['macro_f1'] for r in runs]}; "
              f"{split} per-label {[r[split]['per_label_f1'] for r in runs]}; "
              f"{split} FP-rate {[r[split]['false_positive_rate'] for r in runs]}")
        return

    run_training(config, args.seed, args.samples, args.real_samples)


if __name__ == "__main__":
    main()
