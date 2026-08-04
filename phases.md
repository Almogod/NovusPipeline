# NovusPipeline Implementation Phases

## Phase 1: Local MCP Scaffolding (COMPLETED)
- [x] Initialize Python virtual environment (`.venv`) and install `fastmcp`, `pydantic`, `gitpython`, `chromadb`.
- [x] Implement core `server.py` exposing:
  - `read_legacy_file`
  - `query_rag_guidelines`
  - `run_local_tests`
  - `create_git_migration_pr`
- [x] Implement path traversal security boundaries (`is_path_in_workspace`).
- [x] Implement `test_server.py` unit test suite.
- [x] Create `mcp_config_snippet.json` for Antigravity IDE MCP server registration.

## Phase 2 & 2+: Enterprise RAG Vector Pipeline & Server Robustness (COMPLETED)
- [x] Configure persistent local ChromaDB vector database in `.chroma_db`.
- [x] Implement `ingest_rag.py` with offline zero-dependency TF-IDF weighted embedding (`TFIDFEmbeddingFunction`).
- [x] Populate `novus_guidelines` collection with 14 enterprise refactoring handbooks across `python`, `typescript`, `security`, and `clean_code`.
- [x] Implement category filtering (`python`, `typescript`, `security`, `clean_code`), `n_results` control, and relevance scoring.
- [x] Implement `search_rag_by_id` tool for direct retrieval of guidelines by document ID (e.g. `py-001`).
- [x] Implement `get_rag_stats` diagnostic tool for collection metrics and status reporting.
- [x] Implement `reset_rag_database` management tool to trigger programmatically controlled DB re-seeding.
- [x] Harden file reading with max 5MB size limit (`read_legacy_file`).
- [x] Harden sandbox command execution with shell injection prevention (`run_local_tests`).

## Phase 3: Autonomous Modernization Loop & Local Model Integration (COMPLETED)
- [x] Implement static code smell detection engine (`modernizer.py` / `LegacySmellDetector`).
- [x] Implement rule-based parity-preserving code transformation engine (`CodeModernizer`).
- [x] Integrate fine-tuned local Unsloth LLM model adapter (`local_llm.py`):
  - Model Path: `C:\Users\Hp\.unsloth\studio\outputs\unsloth_Qwen3.5-2B_1785882774`
  - Base Model: `unsloth/Qwen3.5-2B`
  - Chat template prompt formatter (`apply_chat_template`)
- [x] Expose `get_local_llm_status` MCP tool for local fine-tuned LLM verification.
- [x] Expose `generate_llm_modernization_proposal` MCP tool for RAG-guided local LLM refactoring proposals.
- [x] Expose `analyze_legacy_codebase` MCP tool for code smell auditing & RAG guideline matching.
- [x] Expose `apply_code_modernization` MCP tool with automated `.bak` safety backup snapshots.
- [x] Expose `run_autonomous_modernization_pipeline` MCP tool combining:
  1. Legacy Audit & Smell Detection
  2. RAG Guideline Retrieval
  3. Parity-Preserving Code Modernization (Local LLM / Rule engine)
  4. Sandboxed Verification Test Execution
  5. Automatic Rollback on Failure / Git PR Draft Creation on Success!
- [x] Expand unit test suite to **23/23 tests passing** in `test_server.py`.

## Phase 4: Git PR & Modernization Reporting (Next Phase)
- Format comprehensive summary reports and manage PR creation workflows.
