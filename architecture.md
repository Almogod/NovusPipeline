# NovusPipeline Architecture Blueprint

- **Server Protocol**: Model Context Protocol (MCP) via Stdio transport (`fastmcp` 4.x). Nothing but JSON-RPC may be written to stdout.
- **Runtime**: Python 3.11+.
- **Workspace**: the analyzed codebase is `NOVUS_WORKSPACE_ROOT` (default: server working directory). Tool-owned state (guideline DB, GNN checkpoint) lives next to the server; per-codebase state (`.novus_index/`, PR drafts, `reports/`) lives in the workspace.
- **Rule Engine**: `modernizer.py`, string/comment-masked regexes plus AST rules for detection; behavior-preserving rewrites only.
- **Vector DB**: ChromaDB with an offline TF-IDF embedder (`ingest_rag.py`), versioned so stored and query embeddings always match.
- **Structural Code Intelligence**: PyTorch Geometric GraphSAGE (`gnn_model.py`) over AST graphs with type/field/identifier node features (`code_graph.py`), distilled from the rule engine on synthetic + real stdlib code (`train_gnn.py`), stress-tested by `eval_gnn.py`. Exposes pooled graph embeddings for future graph+transformer fusion.
- **Codebase Context**: `codebase_context.py` combines per-function GNN predictions and embeddings with a resolved import graph (dependents, API surface, related tests, structural similarity, hotspots); feeds audits, LLM prompts, test selection and PR descriptions.
- **Git Integration**: GitPython plus the git CLI (hooks honored); validated branch names; only explicitly modernized files are committed.
- **Verification Engine**: Subprocess execution of allowlisted test-runner command shapes only; exit code decides success.
