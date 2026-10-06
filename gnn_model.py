"""
gnn_model.py — NovusPipeline Phase 5: GNN Structural Code-Smell Classifier

A graph neural network over the AST graphs from `code_graph.py` that predicts,
per Python file, which smells from `modernizer.LegacySmellDetector` are
present. Node inputs are learned embeddings of (AST type, parent field,
identifier bucket) plus scalar position features; graph readout is max +
attention pooling, so one smelly node is not averaged away in a large file.

The pooled graph vector (`encode_graph`) is the intended "structure" input
for a future graph+transformer hybrid encoder.

The architecture is described by `GNNConfig`, which is saved into the
checkpoint metadata, so a checkpoint always reloads with the architecture it
was trained with.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from code_graph import (
    FIELD_NAMES,
    NODE_TYPES,
    NUM_SCALAR_FEATURES,
    NUM_TOKEN_BUCKETS,
    build_graph_from_code,
)

# PY-SMELL-001 (Python 2 `print x`) is excluded: it is invalid Python 3, so it
# can never appear in an ast.parse()-able graph.
SMELL_LABELS: List[str] = [
    "PY-SMELL-002",  # Obsolete urllib/urllib2 import
    "PY-SMELL-003",  # Bare except clause
    "PY-SMELL-004",  # Deprecated os.popen
    "PY-SMELL-005",  # Unsafe pickle serialization
    "PY-SMELL-006",  # Missing type hints
]


@dataclass
class GNNConfig:
    conv: str = "sage"            # "gcn" | "sage" | "sage_max" | "gin"
    num_layers: int = 3
    hidden_dim: int = 64
    type_embed_dim: int = 32
    field_embed_dim: int = 16
    token_embed_dim: int = 32
    dropout: float = 0.1
    use_tokens: bool = True       # ablation switch: False reproduces the type-only model


_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gnn_model")
CHECKPOINT_PATH = os.path.join(_MODEL_DIR, "smell_gnn.pt")
METADATA_PATH = os.path.join(_MODEL_DIR, "metadata.json")

_MODEL = None
_MODEL_MTIME: Optional[float] = None
_LOAD_ERROR: Optional[str] = None


def _require_torch():
    try:
        import torch
        import torch_geometric  # noqa: F401
        return torch
    except ImportError as e:
        raise RuntimeError(
            "GNN features require 'torch' and 'torch_geometric' to be installed "
            f"in this environment. Original import error: {e}"
        )


def build_model(config: Optional[GNNConfig] = None):
    """Constructs a fresh, untrained SmellGNN for `config`."""
    config = config or GNNConfig()
    torch = _require_torch()
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_max_pool
    from torch_geometric.nn.aggr import AttentionalAggregation

    def make_conv(in_dim: int, out_dim: int):
        if config.conv == "gcn":
            return GCNConv(in_dim, out_dim)
        if config.conv == "sage":
            return SAGEConv(in_dim, out_dim)
        if config.conv == "sage_max":
            # Max aggregation: one distinctive child (e.g. a `returns` annotation)
            # is not diluted by a function's many body statements.
            return SAGEConv(in_dim, out_dim, aggr="max")
        if config.conv == "gin":
            return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
        raise ValueError(f"Unknown conv type '{config.conv}'.")

    class SmellGNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.type_embedding = nn.Embedding(len(NODE_TYPES), config.type_embed_dim)
            self.field_embedding = nn.Embedding(len(FIELD_NAMES), config.field_embed_dim)
            in_dim = config.type_embed_dim + config.field_embed_dim + NUM_SCALAR_FEATURES
            if config.use_tokens:
                self.token_embedding = nn.Embedding(NUM_TOKEN_BUCKETS, config.token_embed_dim, padding_idx=0)
                in_dim += config.token_embed_dim

            self.convs = nn.ModuleList(
                [make_conv(in_dim if i == 0 else config.hidden_dim, config.hidden_dim)
                 for i in range(config.num_layers)]
            )
            self.dropout = nn.Dropout(config.dropout)
            self.attention_pool = AttentionalAggregation(gate_nn=nn.Linear(config.hidden_dim, 1))
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, len(SMELL_LABELS)),
            )

        def encode(self, data, batch):
            parts = [
                self.type_embedding(data.node_type_ids),
                self.field_embedding(data.node_field_ids),
                data.node_features,
            ]
            if config.use_tokens:
                parts.append(self.token_embedding(data.node_token_ids))
            x = torch.cat(parts, dim=-1)
            for i, conv in enumerate(self.convs):
                h = F.relu(conv(x, data.edge_index))
                x = self.dropout(h) + (x if i > 0 else 0)
            return torch.cat([global_max_pool(x, batch), self.attention_pool(x, batch)], dim=-1)

        def forward(self, data, batch):
            return self.classifier(self.encode(data, batch))  # logits [num_graphs, num_labels]

    return SmellGNN()


def graph_dict_to_pyg_data(graph: Dict[str, Any], y: Optional[List[float]] = None):
    """Converts the plain-dict graph from code_graph.py into a PyG Data object."""
    torch = _require_torch()
    from torch_geometric.data import Data

    data = Data(
        node_type_ids=torch.tensor(graph["node_type_ids"], dtype=torch.long),
        node_field_ids=torch.tensor(graph["node_field_ids"], dtype=torch.long),
        node_token_ids=torch.tensor(graph["node_token_ids"], dtype=torch.long),
        node_features=torch.tensor(graph["node_features"], dtype=torch.float),
        edge_index=torch.tensor(graph["edge_index"], dtype=torch.long).t().contiguous(),
        num_nodes=graph["num_nodes"],
    )
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    return data


def _read_metadata() -> Dict[str, Any]:
    if not os.path.exists(METADATA_PATH):
        return {}
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_thresholds() -> Dict[str, float]:
    """Per-label decision thresholds calibrated on the validation split at training time."""
    saved = _read_metadata().get("thresholds") or {}
    return {label: float(saved.get(label, 0.5)) for label in SMELL_LABELS}


def get_gnn_status() -> Dict[str, Any]:
    """Returns checkpoint/metadata status without requiring torch to be importable."""
    try:
        _require_torch()
        torch_available, torch_error = True, None
    except RuntimeError as e:
        torch_available, torch_error = False, str(e)

    return {
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_exists": os.path.exists(CHECKPOINT_PATH),
        "labels": SMELL_LABELS,
        "metadata": _read_metadata(),
        "torch_available": torch_available,
        "torch_error": torch_error,
        "load_error": _LOAD_ERROR,
    }


def _load_model():
    global _MODEL, _MODEL_MTIME, _LOAD_ERROR

    if not os.path.exists(CHECKPOINT_PATH):
        _MODEL = None
        _LOAD_ERROR = f"No trained checkpoint at '{CHECKPOINT_PATH}'. Run `train_gnn.py` first."
        return None

    # Reload if train_gnn.py replaced the checkpoint while the server was running.
    mtime = os.path.getmtime(CHECKPOINT_PATH)
    if _MODEL is not None and mtime == _MODEL_MTIME:
        return _MODEL

    try:
        torch = _require_torch()
        config = GNNConfig(**_read_metadata().get("config", {}))
        model = build_model(config)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
        model.eval()
        _MODEL, _MODEL_MTIME, _LOAD_ERROR = model, mtime, None
        return model
    except Exception as e:
        _MODEL = None
        _LOAD_ERROR = f"Failed to load GNN checkpoint: {e}"
        return None


def _run(code: str, encode_only: bool):
    model = _load_model()
    if model is None:
        raise RuntimeError(_LOAD_ERROR or "GNN model unavailable.")
    torch = _require_torch()
    data = graph_dict_to_pyg_data(build_graph_from_code(code))
    batch = torch.zeros(data.num_nodes, dtype=torch.long)
    with torch.no_grad():
        return model.encode(data, batch) if encode_only else torch.sigmoid(model(data, batch))


def predict_smells(code: str) -> Dict[str, float]:
    """
    Returns {smell_id: probability}. Compare against `get_thresholds()`, not 0.5.

    Raises RuntimeError if torch/the checkpoint are unavailable, SyntaxError for
    invalid Python 3, ValueError for pathologically nested or oversized source.
    """
    probs = _run(code, encode_only=False).squeeze(0).tolist()
    return dict(zip(SMELL_LABELS, probs))


def encode_graph(code: str) -> List[float]:
    """Pooled graph embedding (2 * hidden_dim floats), for downstream fusion models."""
    return _run(code, encode_only=True).squeeze(0).tolist()


def analyze_many(codes: List[str], batch_size: int = 64) -> List[Optional[Dict[str, Any]]]:
    """
    Batched smell probabilities + graph embeddings, one forward pass per batch.
    Returns, per input, {"probs": {smell_id: p}, "embedding": [...]}, or None if
    that input could not be turned into a graph (syntax error, too large, ...).
    Raises RuntimeError if the model itself is unavailable.
    """
    model = _load_model()
    if model is None:
        raise RuntimeError(_LOAD_ERROR or "GNN model unavailable.")
    torch = _require_torch()
    from torch_geometric.data import Batch

    graphs: List[Any] = []
    for code in codes:
        try:
            graphs.append(graph_dict_to_pyg_data(build_graph_from_code(code)))
        except (SyntaxError, ValueError):
            graphs.append(None)

    results: List[Optional[Dict[str, Any]]] = [None] * len(codes)
    valid = [i for i, g in enumerate(graphs) if g is not None]
    with torch.no_grad():
        for start in range(0, len(valid), batch_size):
            idx = valid[start:start + batch_size]
            batch = Batch.from_data_list([graphs[i] for i in idx])
            embeddings = model.encode(batch, batch.batch)
            probs = torch.sigmoid(model.classifier(embeddings))
            for row, i in enumerate(idx):
                results[i] = {
                    "probs": dict(zip(SMELL_LABELS, probs[row].tolist())),
                    "embedding": embeddings[row].tolist(),
                }
    return results


def save_checkpoint(model, config: GNNConfig, thresholds: Dict[str, float], metrics: Dict[str, Any]) -> None:
    torch = _require_torch()
    os.makedirs(_MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    metadata = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "labels": SMELL_LABELS,
        "config": asdict(config),
        "thresholds": thresholds,
        **metrics,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
