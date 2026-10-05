"""
gnn_model.py — NovusPipeline Phase 5: GNN Structural Code-Smell Classifier

Defines a small Graph Convolutional Network that consumes the AST graphs
produced by `code_graph.py` and predicts, per Python source file, which
structural code smells (from `modernizer.LegacySmellDetector`) are present.

This is a *structural* signal: the GNN never sees raw text/tokens, only the
typed AST graph shape, so it complements (rather than duplicates) the
regex-based detector in `modernizer.py` and the keyword-based RAG retrieval
in `server.py`. It is also the natural "graph half" of a future graph+
transformer hybrid encoder, since it already produces a fixed-size pooled
graph embedding before the final classifier head.

Only Python files are supported: the graph is built from `ast.parse`, which
requires syntactically valid Python 3 source.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from code_graph import NODE_TYPES, NUM_SCALAR_FEATURES, build_graph_from_code

# Smells detectable purely from a valid-Python-3 AST (excludes PY-SMELL-001,
# the Python-2 `print x` statement, which is invalid Python 3 syntax and
# therefore can never appear in an ast.parse()-able graph).
SMELL_LABELS: List[str] = [
    "PY-SMELL-002",  # Obsolete urllib/urllib2 import
    "PY-SMELL-003",  # Bare except:
    "PY-SMELL-004",  # Deprecated os.popen
    "PY-SMELL-005",  # Unsafe pickle serialization
    "PY-SMELL-006",  # Missing type hints
]

TYPE_EMBED_DIM = 32
HIDDEN_DIM = 64

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gnn_model")
CHECKPOINT_PATH = os.path.join(_MODEL_DIR, "smell_gnn.pt")
METADATA_PATH = os.path.join(_MODEL_DIR, "metadata.json")

_MODEL = None
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


def build_model():
    """Constructs a fresh, untrained SmellGNN. Import-guarded on torch."""
    torch = _require_torch()
    import torch.nn as nn
    from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

    class SmellGNN(nn.Module):
        def __init__(
            self,
            num_node_types: int = len(NODE_TYPES),
            type_embed_dim: int = TYPE_EMBED_DIM,
            scalar_feat_dim: int = NUM_SCALAR_FEATURES,
            hidden_dim: int = HIDDEN_DIM,
            num_labels: int = len(SMELL_LABELS),
        ) -> None:
            super().__init__()
            self.type_embedding = nn.Embedding(num_node_types, type_embed_dim)
            in_dim = type_embed_dim + scalar_feat_dim
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_labels),
            )
            self.global_mean_pool = staticmethod(global_mean_pool)
            self.global_max_pool = staticmethod(global_max_pool)

        def forward(self, node_type_ids, node_features, edge_index, batch):
            import torch.nn.functional as F

            type_emb = self.type_embedding(node_type_ids)
            x = torch.cat([type_emb, node_features], dim=-1)
            x = F.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv3(x, edge_index))

            pooled = torch.cat(
                [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1
            )
            return self.classifier(pooled)  # logits, shape [batch, num_labels]

    return SmellGNN()


def graph_dict_to_pyg_data(graph: Dict[str, Any], y: Optional[List[float]] = None):
    """Converts the plain-dict graph from code_graph.py into a PyG Data object."""
    torch = _require_torch()
    from torch_geometric.data import Data

    node_type_ids = torch.tensor(graph["node_type_ids"], dtype=torch.long)
    node_features = torch.tensor(graph["node_features"], dtype=torch.float)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long).t().contiguous()

    data = Data(
        node_type_ids=node_type_ids,
        node_features=node_features,
        edge_index=edge_index,
        num_nodes=graph["num_nodes"],
    )
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    return data


def get_gnn_status() -> Dict[str, Any]:
    """Returns checkpoint/metadata status without requiring torch to be importable."""
    exists = os.path.exists(CHECKPOINT_PATH)
    metadata: Dict[str, Any] = {}
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    try:
        _require_torch()
        torch_available = True
        torch_error = None
    except RuntimeError as e:
        torch_available = False
        torch_error = str(e)

    return {
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_exists": exists,
        "labels": SMELL_LABELS,
        "metadata": metadata,
        "torch_available": torch_available,
        "torch_error": torch_error,
        "load_error": _LOAD_ERROR,
    }


def _load_model():
    global _MODEL, _LOAD_ERROR
    if _MODEL is not None:
        return _MODEL

    if not os.path.exists(CHECKPOINT_PATH):
        _LOAD_ERROR = f"No trained checkpoint at '{CHECKPOINT_PATH}'. Run `train_gnn.py` first."
        return None

    try:
        torch = _require_torch()
        model = build_model()
        state = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _MODEL = model
        return model
    except Exception as e:
        _LOAD_ERROR = str(e)
        return None


def predict_smells(code: str) -> Dict[str, float]:
    """
    Builds the AST graph for `code` and returns {smell_id: probability} from
    the trained GNN. Raises RuntimeError with a clear, user-facing message if
    torch/the checkpoint are unavailable, or SyntaxError if `code` is not
    valid Python 3 (mirrors ast.parse's own behavior).
    """
    model = _load_model()
    if model is None:
        raise RuntimeError(_LOAD_ERROR or "GNN model unavailable.")

    torch = _require_torch()
    graph = build_graph_from_code(code)
    data = graph_dict_to_pyg_data(graph)
    batch = torch.zeros(data.num_nodes, dtype=torch.long)

    with torch.no_grad():
        logits = model(data.node_type_ids, data.node_features, data.edge_index, batch)
        probs = torch.sigmoid(logits).squeeze(0).tolist()

    return dict(zip(SMELL_LABELS, probs))


def save_checkpoint(model, metrics: Dict[str, Any]) -> None:
    torch = _require_torch()
    os.makedirs(_MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    metadata = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "labels": SMELL_LABELS,
        "node_type_vocab_size": len(NODE_TYPES),
        "hidden_dim": HIDDEN_DIM,
        "type_embed_dim": TYPE_EMBED_DIM,
        **metrics,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
