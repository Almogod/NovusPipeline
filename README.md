# NovusPipeline

A local MCP server that modernizes legacy codebases: it detects legacy code smells (rule engine + a graph neural network), retrieves modernization guidelines (RAG), applies behavior-preserving rewrites, verifies them with the project's own tests, and drafts a scoped git PR. A codebase index gives every tool context about the file it is changing: who imports it, which of its symbols they use, which tests cover it, and where the same smells occur elsewhere.

## Setup

```shell
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate elsewhere)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python ingest_rag.py              # seed the guideline vector DB
python train_gnn.py               # train the GNN smell classifier (~10 min on CPU)
python eval_gnn.py                # optional: stress-test report -> .gnn_model/eval_report.json
```

Register the server with your MCP client using `mcp_config_snippet.json`. Set `NOVUS_WORKSPACE_ROOT` to the codebase you want to modernize (defaults to the server's working directory). `NOVUS_PYTHON` overrides the interpreter used to run tests; by default the workspace's own `.venv`/`venv` is used.

## Tools

| Area | Tools |
|---|---|
| Codebase context | `index_codebase`, `get_codebase_overview`, `get_file_context`, `find_similar_code` |
| Audit | `analyze_legacy_codebase`, `analyze_code_structure_gnn`, `read_legacy_file` |
| Modernize | `apply_code_modernization`, `generate_llm_modernization_proposal`, `run_autonomous_modernization_pipeline` |
| Verify & ship | `run_local_tests`, `create_git_migration_pr`, `format_modernization_report`, `finalize_git_migration_pr` |
| Guidelines (RAG) | `query_rag_guidelines`, `search_rag_by_id`, `ingest_rag_document`, `get_rag_stats`, `reset_rag_database` |
| Status | `get_gnn_model_status`, `get_local_llm_status` |

A typical session: `get_codebase_overview` to pick a hotspot, `get_file_context` to see its blast radius, then `run_autonomous_modernization_pipeline` on it. The pipeline runs the tests that import the file, rolls back on failure or on a broken public API, commits only that file, and lists follow-up files with the same smells.

`run_local_tests` only runs test-runner commands (`python -m pytest|unittest`, `pytest`, `npm test`, `npm run test[:x]`, `node --test`, `npx jest|vitest|mocha`, `cargo test`, `go test`, `mvn test`, `gradle test`).

## Development

```shell
python -m unittest test_server -v
```

Tests that touch git or write files run in a throwaway repository; the suite never changes this repository's branches or working tree. See `phases.md` for the change history and the GNN evaluation results, and `architecture.md` / `rules.md` for design and refactoring constraints.
