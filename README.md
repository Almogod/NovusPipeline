# NovusPipeline

[![PyPI](https://img.shields.io/pypi/v/novuspipeline)](https://pypi.org/project/novuspipeline/)
[![Python](https://img.shields.io/pypi/pyversions/novuspipeline)](https://pypi.org/project/novuspipeline/)
[![CI](https://github.com/Almogod/NovusPipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Almogod/NovusPipeline/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<!-- mcp-name: io.github.almogod/novuspipeline -->

An MCP server that modernizes legacy Python codebases safely. It finds legacy code smells (built-in rules, [Ruff](https://docs.astral.sh/ruff/), a graph neural network over the AST, codebase-wide analysis, and smells it learns from your own git history), knows the codebase around each file (who imports it, which of its symbols they use, which tests cover it), applies behavior-preserving rewrites, verifies them with your own tests in a sandbox, and commits only the files it changed, on a branch. Everything runs locally: your code never leaves your machine.

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
| `NOVUS_LEARN_FROM_HISTORY=0` | Don't mine git history in the background after indexing |
| `NOVUS_SANDBOX=docker`, `NOVUS_SANDBOX_IMAGE` | Run tests in a container with no network (the image must contain your test dependencies) |
| `NOVUS_SANDBOX_MEMORY_MB`, `NOVUS_SANDBOX_MAX_PROCESSES` | Sandbox limits for test runs (default 4096 MB, 256 processes) |
| `NOVUS_SANDBOX_ENV_ALLOW=A,B` | Environment variables your tests need (all others are hidden from them) |

Per-project settings go in `.novus/config.toml`:

```toml
[deprecated]   # internal APIs to migrate away from; every use becomes a CTX-DEPRECATED finding
"myapp.db.legacy_query" = "use myapp.db.query"
"myapp.utils" = "split into myapp.text and myapp.io"
```

Entry points declared in `pyproject.toml` are never reported as dead code.

## Your standards (industry, team or personal rules)

Modernizing to "best practice" is not enough when your industry or company has its own rules. Before modernizing, add them as **rule packs**: every audit reports violations, the pipeline applies their fixes, the LLM is told to follow them (and proposals that add violations are rejected), and rules with `enforce = "block"` stop the pipeline from committing a file that still violates them.

Rule packs are TOML files, loaded from:

| Location | Applies to |
|---|---|
| `.novus/standards/*.toml` in the project | everyone working on the project (commit it) |
| `standards/*.toml` in your user data directory | all your projects (personal standards) |

Example `.novus/standards/payments.toml`:

```toml
[pack]
name = "acme-payments"
description = "Payment service standards"

[ruff]                       # optional: tune Ruff for this codebase
select = ["DTZ"]             # also check timezone-naive datetimes
ignore = ["S113"]            # superseded by PAY-TIMEOUT below, which also fixes it

[[rule]]
id = "PAY-TIMEOUT"
title = "Outbound HTTP calls need a timeout"
severity = "MEDIUM"
calls = ["requests.get", "requests.post"]   # dotted, as imported; aliases are resolved
require_keyword = "timeout"                 # only flag calls without timeout=
add_keyword = "timeout=10"                  # automatic fix
guideline = "A stalled provider must not hold a payment worker forever."

[[rule]]
id = "PAY-NO-PICKLE"
title = "No pickle in the payment service"
severity = "HIGH"
imports = ["pickle"]
paths = ["src/payments/**"]
enforce = "block"                           # pipeline refuses to commit while this remains
guideline = "Serialize with JSON or protobuf; pickle executes code on load."

[[rule]]
id = "PAY-LEGACY-CLIENT"
title = "Use the new ledger client"
calls = ["ledger_v1.post_entry"]
replace_call = "ledger.post_entry"          # automatic fix; the import is added if missing
guideline = "ledger_v1 is frozen; ledger.post_entry has the same signature."
```

| Field | Meaning |
|---|---|
| `calls`, `imports`, `pattern` | detection (at least one); `pattern` is a regex matched with strings and comments blanked out |
| `require_keyword` | with `calls`: only calls missing this keyword argument |
| `replace_call`, `replace_import`, `add_keyword`, `replacement` | automatic fix (regex replacement for `pattern`); fixes go through the same test-gated pipeline as everything else |
| `severity` | `CRITICAL`, `HIGH`, `MEDIUM` (default), `LOW` |
| `enforce` | `warn` (default) or `block` |
| `paths`, `exclude` | glob scope, relative to the project |
| `category` | guideline category (default `python`; `typescript`/`javascript` lets `pattern` rules check TS/JS files) |
| `guideline` | markdown shown in audits, returned by `query_rag_guidelines`, and given to the LLM |

You can also do all of this through your assistant:

- `add_modernization_rule` creates or updates a rule (in `.novus/standards/custom.toml`, or `scope="user"`) and reports how many files violate it.
- `list_modernization_standards` shows every active rule, where it comes from, and problems in pack files.
- `remove_modernization_rule` deletes a rule added with `add_modernization_rule`.
- `install_standard_pack` copies a built-in example pack into the project for you to edit: `secure-defaults` (HTTP timeouts, safe YAML, TLS verification, `shell=True`) or `python-312-ready` (APIs removed in Python 3.12).

**Guidelines only.** To add reference material without detection (a style guide, an industry regulation summary), use `ingest_rag_document` with any category, e.g. `fintech` or `healthcare`. Ingesting an existing id, including a built-in one such as `py-002`, replaces it; `remove_rag_document` deletes one.

## What it finds

| Source | Examples |
|---|---|
| Your standards | whatever your rule packs define (see above) |
| Built-in legacy rules | Python 2 print and `urllib2`, bare `except`, `os.popen`, unsafe `pickle`, missing type hints (plus ES5 `var`, `any`, `require()` in TypeScript/JavaScript) |
| Ruff | pyupgrade, bugbear, security (bandit), pathlib, comprehension and simplification rules; safe fixes are applied automatically |
| Structural GNN (`[gnn]`) | the same smells from AST shape, localized to functions; structurally similar code |
| Codebase-wide | import cycles (import-time only), dead code, duplicated functions (rename-insensitive), untested hotspots, god modules, uses of APIs listed in `[deprecated]`, unresolvable dynamic imports |
| Learned from your history | calls and imports your team replaced the same way in several separate commits, with the commits as evidence |

Dynamic imports are part of the import graph: `importlib.import_module("pkg.x")`, `__import__`, relative imports with `package=__name__`, `runpy`, spec loaders from file paths, and `getattr(module, "name")`. A dynamic import that can't be resolved is reported (`CTX-DYNAMIC`), and modules under its literal prefix are treated as reachable rather than dead.

**Learned smells.** `learn_from_history` (also run in the background after indexing) compares each changed function before and after every commit. A replacement such as `client.fetch_json(...)` → `client.get_json(...)` becomes a rule when it shows up in at least 3 separate commits, when the team usually chose that replacement for the old call, and when nobody kept writing the old call after the last replacement (this filters out churn of everyday calls). Remaining uses are reported in audits, file context and the overview. Pure renames with consistent history are applied automatically by the pipeline, and like every other change they must pass your tests first. In a time-split check on Inkscape's extensions repository, both auto-fixable rules learned from the history before mid-2019 were later fully applied by its developers, and 4 of the 5 rules with remaining uses were later acted on.

## Tools

| Area | Tools |
|---|---|
| Codebase context | `index_codebase`, `get_codebase_overview`, `get_file_context`, `find_similar_code`, `learn_from_history` |
| Audit | `analyze_legacy_codebase`, `analyze_code_structure_gnn`, `read_legacy_file` |
| Modernize | `apply_code_modernization`, `generate_llm_modernization_proposal`, `run_autonomous_modernization_pipeline` |
| Verify & ship | `run_local_tests`, `create_git_migration_pr`, `format_modernization_report`, `finalize_git_migration_pr` |
| Standards | `list_modernization_standards`, `add_modernization_rule`, `remove_modernization_rule`, `install_standard_pack` |
| Guidelines (RAG) | `query_rag_guidelines`, `search_rag_by_id`, `ingest_rag_document`, `remove_rag_document`, `get_rag_stats`, `reset_rag_database` |
| Status | `get_gnn_model_status`, `get_local_llm_status` |

`run_autonomous_modernization_pipeline` never touches your working copy. It modernizes the file in a disposable git worktree, runs the tests that import the file in the sandbox, rejects the change if the tests fail or if a symbol other modules import would disappear, and commits only that file to a branch (`branch_name`, default `auto-modernization-branch`; repeated calls add commits to it). It refuses to run on a file with uncommitted changes, or with your checked-out branch as the target. A PR description draft is written to `.novus_index/prs/`.

### Sandbox

Test commands must be test runners (`python -m pytest|unittest`, `pytest`, `npm test`, `npm run test[:x]`, `node --test`, `npx jest|vitest|mocha`, `cargo test`, `go test`, `mvn test`, `gradle test`), and they never run in your working tree:

| Protection | Windows | Linux | macOS |
|---|---|---|---|
| Runs on a temporary copy of the project (or the pipeline's disposable worktree); your files can't be modified | ✓ | ✓ | ✓ |
| Environment scrubbed (no tokens/API keys); home, app-data and temp point into the sandbox, so `~/.ssh`, `~/.aws` etc. are not found through them | ✓ | ✓ | ✓ |
| Memory and process-count limits; the whole process tree is killed on timeout | Job Object | rlimits + process group | process group (memory limit not enforced) |
| No network | with `NOVUS_SANDBOX=docker` | ✓ when user namespaces are available, else Docker | with Docker |

Every test result says which protections were active. Without Docker the sandbox is not a VM: tests still run as your user and can read files outside the project by absolute path. Use `NOVUS_SANDBOX=docker` for code you don't trust.

### Approve once

Tools carry MCP annotations (read-only, destructive, idempotent). Apart from the six destructive tools, every tool only reads, or writes only to `.novus_index/`, a sandbox, or a branch other than the one you have checked out, so you can approve them once (`add_modernization_rule` and `install_standard_pack` also write rule files under `.novus/standards/`). In Claude Code (`.claude/settings.json`), this keeps asking only for the tools that change your working copy, your checked-out branch, or delete guidelines or rules:

```json
{
  "permissions": {
    "allow": ["mcp__novuspipeline"],
    "ask": [
      "mcp__novuspipeline__apply_code_modernization",
      "mcp__novuspipeline__create_git_migration_pr",
      "mcp__novuspipeline__finalize_git_migration_pr",
      "mcp__novuspipeline__reset_rag_database",
      "mcp__novuspipeline__remove_rag_document",
      "mcp__novuspipeline__remove_modernization_rule"
    ]
  }
}
```

Other clients have an equivalent "always allow" option per tool.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks and design rules, [docs/architecture.md](docs/architecture.md) for how the pieces fit together, and [docs/design-history.md](docs/design-history.md) for design decisions and evaluation results. Report vulnerabilities as described in [SECURITY.md](SECURITY.md); changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

MIT, see [LICENSE](LICENSE).
