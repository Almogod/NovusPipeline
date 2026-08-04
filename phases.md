# NovusPipeline Implementation Phases

## Phase 1: Local MCP Scaffolding (COMPLETED)
- [x] Initialize Python virtual environment (`.venv`) and install `fastmcp`, `pydantic`, `gitpython`, `chromadb`.
- [x] Implement core `server.py` exposing:
  - `read_legacy_file`
  - `query_rag_guidelines`
  - `run_local_tests`
  - `create_git_migration_pr`
- [x] Implement path traversal security boundaries (`is_path_in_workspace`).
- [x] Implement `test_server.py` unit test suite (All 6 tests passing).
- [x] Create `mcp_config_snippet.json` for Antigravity IDE MCP server registration.

## Phase 2: RAG Vector Indexing (COMPLETED)
- [x] Configure persistent local ChromaDB vector database in `.chroma_db`.
- [x] Implement `ingest_rag.py` with offline zero-dependency embedding function (`LocalSimpleEmbeddingFunction`).
- [x] Populate `novus_guidelines` collection with enterprise refactoring handbooks (Python modernization, TypeScript standards, security rules, and clean code guidelines).
- [x] Integrate vector search into `query_rag_guidelines` tool in `server.py`.
- [x] Verify vector retrieval and unit tests in `test_server.py` (6/6 tests passing).

## Phase 3: Autonomous Modernization Loop (Next Phase)
- Implement continuous refactoring pipeline with test-driven validation loops.

## Phase 4: Git PR & Modernization Reporting
- Format comprehensive summary reports and manage PR creation workflows.
