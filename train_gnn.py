"""
train_gnn.py — NovusPipeline Phase 5: GNN Smell Classifier Training

Trains the SmellGNN (gnn_model.py) to predict structural code smells from AST
graphs alone, using `modernizer.LegacySmellDetector` as a weak-supervision
teacher: a synthetic corpus of small Python snippets is generated from
templates, each snippet is labeled by running the existing rule-based
detector over its source text, and the GNN is trained to reproduce those
labels from graph structure only (no text/token features). This distills the
regex/AST rule engine into a model that can generalize to structural
variations the regex patterns don't literally match.

Usage:
    python train_gnn.py
"""

import random
from typing import Dict, List, Tuple

from code_graph import build_graph_from_code
from gnn_model import SMELL_LABELS, build_model, graph_dict_to_pyg_data, save_checkpoint
from modernizer import LegacySmellDetector

random.seed(42)

# ---------------------------------------------------------------------------
# Synthetic snippet templates, one pool per smell label plus a clean pool.
# Each template is a self-contained, syntactically valid Python function
# (required: build_graph_from_code uses ast.parse). Multiple templates are
# combined into one synthetic "file" per sample so the model sees varied
# multi-label combinations and varied surrounding structure, not just one
# smell in isolation.
# ---------------------------------------------------------------------------

CLEAN_TEMPLATES = [
    "def add_{n}(x: int, y: int) -> int:\n    return x + y\n",
    "def total_{n}(items: list[int]) -> int:\n    total = 0\n    for item in items:\n        total += item\n    return total\n",
    "def is_even_{n}(x: int) -> bool:\n    return x % 2 == 0\n",
    "class Counter_{n}:\n    def __init__(self) -> None:\n        self.value = 0\n\n    def increment(self) -> int:\n        self.value += 1\n        return self.value\n",
    "def safe_divide_{n}(a: float, b: float) -> float | None:\n    if b == 0:\n        return None\n    return a / b\n",
    "def filter_positive_{n}(values: list[int]) -> list[int]:\n    return [v for v in values if v > 0]\n",
]

SMELLY_TEMPLATES: Dict[str, List[str]] = {
    "PY-SMELL-002": [
        "import urllib2\ndef fetch_{n}(url):\n    return urllib2.urlopen(url).read()\n",
        "from urllib2 import urlopen\ndef download_{n}(url):\n    return urlopen(url).read()\n",
    ],
    "PY-SMELL-003": [
        "def risky_{n}(data):\n    try:\n        return data[0]\n    except:\n        return None\n",
        "def parse_{n}(raw):\n    try:\n        value = int(raw)\n    except:\n        value = 0\n    return value\n",
        "def load_{n}(path):\n    try:\n        with open(path) as f:\n            return f.read()\n    except:\n        pass\n",
    ],
    "PY-SMELL-004": [
        "import os\ndef run_cmd_{n}(cmd):\n    return os.popen(cmd).read()\n",
        "import os\ndef list_files_{n}(path):\n    return os.popen('ls ' + path).read()\n",
    ],
    "PY-SMELL-005": [
        "import pickle\ndef load_data_{n}(path):\n    with open(path, 'rb') as f:\n        return pickle.loads(f.read())\n",
        "import pickle\ndef save_data_{n}(obj, path):\n    with open(path, 'wb') as f:\n        f.write(pickle.dumps(obj))\n",
    ],
    "PY-SMELL-006": [
        "def process_{n}(data):\n    return data\n",
        "def handler_{n}(request, context):\n    return {{'ok': True}}\n",
    ],
}


def _instantiate(template: str, n: int) -> str:
    return template.replace("{n}", str(n)).format(n=n) if "{{" in template else template.replace("{n}", str(n))


def generate_corpus(num_samples: int = 600) -> List[Tuple[str, List[float]]]:
    """Returns [(source_code, multi_hot_label_vector), ...]."""
    samples: List[Tuple[str, List[float]]] = []
    all_smell_ids = list(SMELLY_TEMPLATES.keys())

    for i in range(num_samples):
        pieces: List[str] = []

        # Randomly decide how many distinct smells this synthetic file has
        # (0 = pure-clean sample, up to 3 combined smells).
        num_smells = random.choices([0, 1, 2, 3], weights=[0.2, 0.4, 0.3, 0.1])[0]
        chosen_smells = random.sample(all_smell_ids, k=min(num_smells, len(all_smell_ids)))

        for smell_id in chosen_smells:
            template = random.choice(SMELLY_TEMPLATES[smell_id])
            pieces.append(_instantiate(template, i * 10 + len(pieces)))

        # Add 0-2 clean filler functions for structural diversity.
        for _ in range(random.randint(0, 2)):
            template = random.choice(CLEAN_TEMPLATES)
            pieces.append(_instantiate(template, i * 10 + len(pieces)))

        if not pieces:
            template = random.choice(CLEAN_TEMPLATES)
            pieces.append(_instantiate(template, i * 10))

        source = "\n".join(pieces)

        # Weak-supervision labels: ask the existing rule-based detector what
        # it thinks is present, rather than trusting our own template bookkeeping.
        findings = LegacySmellDetector.scan_code(source, "synthetic.py")
        present_ids = {f["smell_id"] for f in findings}
        label = [1.0 if sid in present_ids else 0.0 for sid in SMELL_LABELS]

        samples.append((source, label))

    return samples


def train() -> None:
    import torch
    from torch.optim import Adam
    from torch.nn import BCEWithLogitsLoss
    from torch_geometric.loader import DataLoader

    print("[NovusPipeline GNN] Generating synthetic training corpus...")
    raw_samples = generate_corpus(num_samples=600)

    dataset = []
    skipped = 0
    for source, label in raw_samples:
        try:
            graph = build_graph_from_code(source)
        except SyntaxError:
            skipped += 1
            continue
        dataset.append(graph_dict_to_pyg_data(graph, y=label))

    random.shuffle(dataset)
    split = int(len(dataset) * 0.8)
    train_set, val_set = dataset[:split], dataset[split:]
    print(f"[NovusPipeline GNN] Dataset: {len(dataset)} graphs ({skipped} skipped on parse error), "
          f"train={len(train_set)} val={len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = BCEWithLogitsLoss()

    epochs = 30
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch.node_type_ids, batch.node_features, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        if epoch % 5 == 0 or epoch == epochs:
            avg_loss = total_loss / max(len(train_set), 1)
            print(f"[NovusPipeline GNN] Epoch {epoch}/{epochs} - train BCE loss: {avg_loss:.4f}")

    metrics = evaluate(model, val_loader, len(val_set))
    save_checkpoint(model, metrics)
    print(f"[NovusPipeline GNN] Saved checkpoint + metadata. Val metrics: {metrics}")


def evaluate(model, val_loader, num_val: int) -> Dict[str, float]:
    import torch

    model.eval()
    correct_per_label = [0] * len(SMELL_LABELS)
    exact_match = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch.node_type_ids, batch.node_features, batch.edge_index, batch.batch)
            preds = (torch.sigmoid(logits) > 0.5).float()
            total += preds.size(0)
            exact_match += int((preds == batch.y).all(dim=1).sum().item())
            for i in range(len(SMELL_LABELS)):
                correct_per_label[i] += int((preds[:, i] == batch.y[:, i]).sum().item())

    total = max(total, 1)
    per_label_acc = {label: round(correct_per_label[i] / total, 4) for i, label in enumerate(SMELL_LABELS)}
    return {
        "val_samples": num_val,
        "exact_match_accuracy": round(exact_match / total, 4),
        "per_label_accuracy": per_label_acc,
    }


if __name__ == "__main__":
    train()
