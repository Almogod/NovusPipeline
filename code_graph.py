"""
code_graph.py — NovusPipeline Phase 5: AST Graph Construction

Converts Python source code into a typed graph representation suitable for a
Graph Neural Network: nodes = AST nodes (typed by grammar production, e.g.
`Try`, `ExceptHandler`, `Call`), edges = parent-child structural edges plus
next-sibling control-flow edges within the same block.

Deliberately has zero dependency on torch/PyTorch Geometric so graph
construction can be built, tested, and reused independently of which ML
framework consumes it (`gnn_model.py` converts the plain-dict output here
into a `torch_geometric.data.Data` object).
"""

import ast
from typing import Any, Dict, List, Tuple

# Fixed vocabulary of common Python AST node types. Anything not in this list
# (rare/version-specific nodes) falls back to the UNK bucket so the model has
# a stable, bounded input space regardless of Python version quirks.
NODE_TYPES: List[str] = [
    "Module", "FunctionDef", "AsyncFunctionDef", "ClassDef", "Return",
    "Delete", "Assign", "AugAssign", "AnnAssign", "For", "AsyncFor",
    "While", "If", "With", "AsyncWith", "Raise", "Try", "ExceptHandler",
    "Assert", "Import", "ImportFrom", "Global", "Nonlocal", "Expr", "Pass",
    "Break", "Continue", "BoolOp", "BinOp", "UnaryOp", "Lambda", "IfExp",
    "Dict", "Set", "ListComp", "SetComp", "DictComp", "GeneratorExp",
    "Await", "Yield", "YieldFrom", "Compare", "Call", "Constant",
    "Attribute", "Subscript", "Starred", "Name", "List", "Tuple", "Slice",
    "arguments", "arg", "keyword", "withitem", "comprehension", "Load",
    "Store", "Del", "And", "Or", "Add", "Sub", "Mult", "Div", "Not", "Eq",
    "NotEq", "Lt", "Gt", "In", "NotIn", "Is", "IsNot",
    "UNK",
]
NODE_TYPE_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(NODE_TYPES)}
UNK_INDEX = NODE_TYPE_TO_INDEX["UNK"]
NUM_SCALAR_FEATURES = 4  # [normalized_depth, normalized_num_children, normalized_sibling_index, is_leaf]


def _node_type_index(node: ast.AST) -> int:
    return NODE_TYPE_TO_INDEX.get(type(node).__name__, UNK_INDEX)


def build_graph_from_code(code: str) -> Dict[str, Any]:
    """
    Parses `code` and returns a plain-dict graph:
      {
        "node_type_ids": List[int]          length N, index into NODE_TYPES
        "node_features": List[List[float]]  length N, each NUM_SCALAR_FEATURES floats
        "edge_index": List[Tuple[int, int]] directed edges (both structural + reverse)
        "num_nodes": int
      }

    Raises SyntaxError if `code` is not parseable Python — callers should
    handle this the same way `ast.parse` callers elsewhere in this project do.
    """
    tree = ast.parse(code)

    node_type_ids: List[int] = []
    node_features: List[List[float]] = []
    edges: List[Tuple[int, int]] = []

    # Assign each AST node a stable integer id via BFS/DFS traversal, tracking
    # parent, depth, and sibling position for the scalar feature vector.
    node_ids: Dict[int, int] = {}

    def visit(node: ast.AST, parent_id: int, depth: int, sibling_index: int, num_siblings: int) -> None:
        this_id = len(node_type_ids)
        node_ids[id(node)] = this_id
        node_type_ids.append(_node_type_index(node))

        children = list(ast.iter_child_nodes(node))
        is_leaf = 1.0 if not children else 0.0
        node_features.append([
            min(depth / 20.0, 1.0),
            min(len(children) / 10.0, 1.0),
            min(sibling_index / 10.0, 1.0) if num_siblings > 1 else 0.0,
            is_leaf,
        ])

        if parent_id >= 0:
            edges.append((parent_id, this_id))
            edges.append((this_id, parent_id))

        for i, child in enumerate(children):
            visit(child, this_id, depth + 1, i, len(children))

        # Sequential control-flow edges between consecutive siblings in the
        # same block (captures statement order beyond pure tree structure).
        for i in range(len(children) - 1):
            a = node_ids[id(children[i])]
            b = node_ids[id(children[i + 1])]
            edges.append((a, b))

    visit(tree, parent_id=-1, depth=0, sibling_index=0, num_siblings=1)

    if not edges:
        # Single-node graph (e.g. empty module): add a self-loop so GNN
        # message passing has at least one edge to operate on.
        edges.append((0, 0))

    return {
        "node_type_ids": node_type_ids,
        "node_features": node_features,
        "edge_index": edges,
        "num_nodes": len(node_type_ids),
    }
