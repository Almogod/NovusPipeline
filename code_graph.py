"""
code_graph.py — NovusPipeline Phase 5: AST Graph Construction

Converts Python source code into a typed graph for the GNN. Each node is an
AST node carrying three categorical features plus a few scalars:

  - node type     (`Try`, `ExceptHandler`, `Call`, ...)
  - parent field  which slot of its parent it fills (`returns`, `annotation`,
                  `type`, `body`, ...) — makes "except with no type" and
                  "def with no return annotation" explicit in the graph
  - identifier    hashed bucket of the node's name (`Name.id`, `Attribute.attr`,
                  `alias.name`, `ImportFrom.module`) — required because several
                  project smells are defined by API names (`urllib2`, `os.popen`,
                  `pickle`) that are otherwise structurally identical to clean code

Edges: parent<->child structural edges plus next-sibling edges in each field.

No dependency on torch, so graph construction is testable on its own.
"""

import ast
import hashlib
from typing import Any, Dict, List, Optional, Tuple

NODE_TYPES: List[str] = [
    "Module", "FunctionDef", "AsyncFunctionDef", "ClassDef", "Return",
    "Delete", "Assign", "AugAssign", "AnnAssign", "For", "AsyncFor",
    "While", "If", "With", "AsyncWith", "Raise", "Try", "TryStar",
    "ExceptHandler", "Assert", "Import", "ImportFrom", "alias", "Global",
    "Nonlocal", "Expr", "Pass", "Break", "Continue", "BoolOp", "BinOp",
    "UnaryOp", "Lambda", "IfExp", "Dict", "Set", "ListComp", "SetComp",
    "DictComp", "GeneratorExp", "Await", "Yield", "YieldFrom", "Compare",
    "Call", "Constant", "JoinedStr", "FormattedValue", "NamedExpr",
    "Attribute", "Subscript", "Starred", "Name", "List", "Tuple", "Slice",
    "arguments", "arg", "keyword", "withitem", "comprehension", "Match",
    "match_case", "Load", "Store", "Del", "And", "Or", "Add", "Sub", "Mult",
    "Div", "Mod", "Not", "USub", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE",
    "In", "NotIn", "Is", "IsNot",
    "UNK",
]
NODE_TYPE_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(NODE_TYPES)}
UNK_INDEX = NODE_TYPE_TO_INDEX["UNK"]

FIELD_NAMES: List[str] = [
    "<root>", "body", "orelse", "finalbody", "handlers", "test", "target",
    "targets", "value", "values", "iter", "args", "posonlyargs", "kwonlyargs",
    "vararg", "kwarg", "defaults", "kw_defaults", "returns", "annotation",
    "decorator_list", "type", "func", "keywords", "elts", "keys", "ops",
    "comparators", "left", "right", "op", "operand", "slice", "ctx", "items",
    "context_expr", "optional_vars", "generators", "ifs", "elt", "key", "exc",
    "cause", "msg", "names", "bases", "lower", "upper", "step", "format_spec",
    "subject", "cases", "pattern", "guard",
    "UNK",
]
FIELD_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FIELD_NAMES)}
FIELD_UNK_INDEX = FIELD_TO_INDEX["UNK"]

# Bucket 0 = "node has no identifier"; 1..N-1 = md5-hashed identifiers. md5
# (not builtin hash()) so buckets are stable across processes.
NUM_TOKEN_BUCKETS = 4096
NUM_SCALAR_FEATURES = 4  # [depth, num_children, sibling_index, is_leaf], each scaled to [0, 1]

# Above this many nodes we refuse rather than make an MCP call take tens of
# seconds (~2.5s per 100k nodes on CPU).
MAX_NODES = 150_000


def identifier_bucket(identifier: str) -> int:
    digest = hashlib.md5(identifier.lower().encode("utf-8")).hexdigest()
    return 1 + int(digest, 16) % (NUM_TOKEN_BUCKETS - 1)


def _node_identifier(node: ast.AST) -> Optional[str]:
    # User-chosen function/class/parameter names are deliberately excluded:
    # they carry no smell signal and would invite overfitting.
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.alias):
        return node.name
    if isinstance(node, ast.ImportFrom):
        return node.module
    return None


def _children_with_fields(node: ast.AST) -> List[Tuple[ast.AST, str]]:
    out: List[Tuple[ast.AST, str]] = []
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            out.append((value, field))
        elif isinstance(value, list):
            out.extend((item, field) for item in value if isinstance(item, ast.AST))
    return out


def build_graph_from_code(code: str) -> Dict[str, Any]:
    """
    Parses `code` and returns a plain-dict graph:
      {
        "node_type_ids":  List[int]               index into NODE_TYPES
        "node_field_ids": List[int]               index into FIELD_NAMES
        "node_token_ids": List[int]               identifier bucket, 0 = none
        "node_features":  List[List[float]]       NUM_SCALAR_FEATURES each
        "edge_index":     List[Tuple[int, int]]   directed edges
        "num_nodes":      int
      }

    Raises SyntaxError for invalid Python 3, and ValueError for source that is
    too deeply nested for the Python parser or exceeds MAX_NODES.
    """
    try:
        tree = ast.parse(code)
    except (RecursionError, MemoryError) as e:
        raise ValueError(f"Source is too deeply nested to parse ({type(e).__name__}).") from e

    node_type_ids: List[int] = []
    node_field_ids: List[int] = []
    node_token_ids: List[int] = []
    node_features: List[List[float]] = []
    edges: List[Tuple[int, int]] = []
    # (parent_id, field) -> child ids in source order, for sibling edges
    field_children: Dict[Tuple[int, str], List[int]] = {}

    # Explicit stack instead of recursion: long expression chains (e.g. a
    # 2000-term `a + b + ...`) parse fine but exceed Python's recursion limit.
    stack: List[Tuple[ast.AST, int, str, int, int, int]] = [(tree, -1, "<root>", 0, 0, 1)]
    while stack:
        node, parent_id, field, depth, sibling_index, num_siblings = stack.pop()
        this_id = len(node_type_ids)
        if this_id >= MAX_NODES:
            raise ValueError(f"Source exceeds the {MAX_NODES}-node graph limit.")

        children = _children_with_fields(node)
        identifier = _node_identifier(node)

        node_type_ids.append(NODE_TYPE_TO_INDEX.get(type(node).__name__, UNK_INDEX))
        node_field_ids.append(FIELD_TO_INDEX.get(field, FIELD_UNK_INDEX))
        node_token_ids.append(identifier_bucket(identifier) if identifier else 0)
        node_features.append([
            min(depth / 20.0, 1.0),
            min(len(children) / 10.0, 1.0),
            min(sibling_index / 10.0, 1.0) if num_siblings > 1 else 0.0,
            0.0 if children else 1.0,
        ])

        if parent_id >= 0:
            edges.append((parent_id, this_id))
            edges.append((this_id, parent_id))
            field_children.setdefault((parent_id, field), []).append(this_id)

        for i in range(len(children) - 1, -1, -1):
            child, child_field = children[i]
            stack.append((child, this_id, child_field, depth + 1, i, len(children)))

    for ids in field_children.values():
        for a, b in zip(ids, ids[1:]):
            edges.append((a, b))

    if not edges:
        edges.append((0, 0))

    return {
        "node_type_ids": node_type_ids,
        "node_field_ids": node_field_ids,
        "node_token_ids": node_token_ids,
        "node_features": node_features,
        "edge_index": edges,
        "num_nodes": len(node_type_ids),
    }
