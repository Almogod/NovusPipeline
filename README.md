# NovusPipeline

<!-- mcp-name: io.github.almogod/novuspipeline -->

An MCP server that modernizes legacy Python codebases safely. It finds legacy code smells (rule engine + a graph neural network over the AST), knows the codebase around each file (who imports it, which of its symbols they use, which tests cover it), applies behavior-preserving rewrites, verifies them with your own tests, and commits only the files it changed. Everything runs locally: your code never leaves your machine.

## Install

Requires Python 3.11+. The server runs on your machine and is launched by your MCP client.

| Install | What you get | Size |
|---|---|---|
| `uvx novuspipeline` / `pip install novuspipeline` | Rule engine, codebase index (import graph, dependents, related tests), RAG guidelines, test-gated pipeline, git PR drafts | small |
| `novuspipeline[gnn]` | + structural GNN: smell cross-check, per-function localization, `find_similar_code` | + torch (CPU) |
| `novuspipeline[llm]` | + local LLM proposals (`use_llm`), validated and test-gated | + torch, transformers; NVIDIA GPU with ~4 GB free VRAM recommended |

On Linux, install CPU torch first to avoid a multi-GB CUDA download: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Connect your MCP client

Claude Desktop (`claude_desktop_config.json`), Cursor (`.cursor/mcp.json`) and similar clients:

```json
{
  "mcpServers": {
    "novuspipeline": {
      "command": "uvx",
      "args": ["--from", "novuspipeline[gnn]", "novuspipeline", "--workspace", "/absolute/path/to/your/project"]
    }
  }
}
```

Claude Code: `claude mcp add novuspipeline -- uvx --from "novuspipeline[gnn]" novuspipeline --workspace /path/to/project`

Then ask your agent to use the `modernization_workflow` prompt, or call `get_codebase_overview` to start.

## Settings

| Option | Purpose |
|---|---|
| `--workspace DIR` / `NOVUS_WORKSPACE_ROOT` | The codebase to modernize (default: current directory) |
| `--transport http --port 8000` | Serve over HTTP instead of stdio. No authentication: keep it on `127.0.0.1` |
| `NOVUS_PYTHON` | Interpreter for running your tests (default: the workspace's `.venv`/`venv`, else the server's) |
| `NOVUS_DATA_DIR` | Where the guideline database lives (default: your user data directory) |
| `NOVUS_GNN_DIR` | Use a retrained GNN checkpoint instead of the bundled one |
| `NOVUS_LLM_BASE_MODEL` | Hugging Face model for LLM proposals (default `unsloth/Qwen3.5-2B`, downloaded on first use) |
| `NOVUS_LLM_PATH` | Optional LoRA adapter directory |
| `NOVUS_LLM_PYTHON` | Run the LLM worker in a separate (GPU) environment |
| `NOVUS_LLM_OFFLINE=1` | Never download models; use the local Hugging Face cache only |

## Tools

| Area | Tools |
|---|---|
| Codebase context | `index_codebase`, `get_codebase_overview`, `get_file_context`, `find_similar_code` |
| Audit | `analyze_legacy_codebase`, `analyze_code_structure_gnn`, `read_legacy_file` |
| Modernize | `apply_code_modernization`, `generate_llm_modernization_proposal`, `run_autonomous_modernization_pipeline` |
| Verify & ship | `run_local_tests`, `create_git_migration_pr`, `format_modernization_report`, `finalize_git_migration_pr` |
| Guidelines (RAG) | `query_rag_guidelines`, `search_rag_by_id`, `ingest_rag_document`, `get_rag_stats`, `reset_rag_database` |
| Status | `get_gnn_model_status`, `get_local_llm_status` |

Tools are annotated read-only or destructive so clients can ask before changing anything. The pipeline runs the tests that import the target file, rolls back on failure or if a symbol other modules import would disappear, and commits only that file on a new branch.

**Safety notes.** `run_local_tests` only accepts test-runner commands (`python -m pytest|unittest`, `pytest`, `npm test`, `npm run test[:x]`, `node --test`, `npx jest|vitest|mocha`, `cargo test`, `go test`, `mvn test`, `gradle test`), but your test suite itself runs with your permissions, as it would in your terminal. Only point the server at code you trust to run.

## Development

```shell
git clone https://github.com/Almogod/NovusPipeline && cd NovusPipeline
python -m venv .venv && .venv/Scripts/activate        # source .venv/bin/activate on macOS/Linux
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[gnn]"
python -m unittest discover -s tests -v                # git/file tests run in throwaway repos

python -m novuspipeline.train_gnn --device auto        # retrain the GNN (GPU recommended)
python -m novuspipeline.eval_gnn                       # stress test on held-out packages
python -m novuspipeline.eval_llm                       # base model vs. LoRA adapter A/B
```

`phases.md` records the design history and evaluation results; `architecture.md` and `rules.md` describe the design and refactoring constraints. MIT licensed.
