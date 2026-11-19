<div align="center">

# 🛠️ NovusPipeline

**Modernize legacy Python codebases safely, through your AI assistant.**

An MCP server that audits, rewrites, tests and commits, on a branch, and only if your tests pass.

[![PyPI](https://img.shields.io/pypi/v/novuspipeline?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/novuspipeline/)
[![Python](https://img.shields.io/pypi/pyversions/novuspipeline?logo=python&logoColor=white)](https://pypi.org/project/novuspipeline/)
[![CI](https://github.com/Almogod/NovusPipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Almogod/NovusPipeline/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[Install](#install) · [Connect](#connect-your-mcp-client) · [Standards](#your-standards-industry-team-or-personal-rules) · [What it finds](#what-it-finds) · [Tools](#tools) · [Sandbox](#sandbox) · [Organizations](#-for-organizations)

</div>

<!-- mcp-name: io.github.almogod/novuspipeline -->

## ✨ Overview

NovusPipeline finds legacy code smells (built-in rules, [Ruff](https://docs.astral.sh/ruff/), a graph neural network over the AST, codebase-wide analysis, and smells it learns from your own git history), knows the codebase around each file (who imports it, which of its symbols they use, which tests cover it), applies behavior-preserving rewrites, verifies them with your own tests in a sandbox, and commits only the files it changed, on a branch.

| | |
|---|---|
| 🔍 **Context-aware** | Import graph, dependents, used symbols and related tests for every file |
| 🧪 **Test-gated** | A change that fails your tests, or removes a symbol others import, is rejected |
| 🌿 **Non-invasive** | Works in a disposable git worktree and commits only the changed file to a branch |
| 📐 **Your rules** | Industry, team or personal standards as TOML rule packs, with `block` enforcement |
| 🔒 **Local** | Everything runs on your machine: your code never leaves it |
| 🏢 **Governed** | Org-wide policy, SSO/token auth with roles, and a tamper-evident audit log ([details](docs/enterprise.md)) |

## 📦 Install

Requires Python 3.11+. The server runs on your machine and is launched by your MCP client.

| Install | What you get | Size |
|---|---|---|
| `uvx novuspipeline` / `pip install novuspipeline` | Rule engine, codebase index (import graph, dependents, related tests), RAG guidelines, test-gated pipeline, git PR drafts | small |
| `novuspipeline[gnn]` | + structural GNN: smell cross-check, per-function localization, `find_similar_code` | + torch (CPU) |
| `novuspipeline[llm]` | + local LLM proposals (`use_llm`), validated and test-gated | + torch, transformers; NVIDIA GPU with ~4 GB free VRAM recommended |

On Linux, install CPU torch first to avoid a multi-GB CUDA download: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## 🔌 Connect your MCP client

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

## ⌨️ Fix a project from the terminal

No AI assistant or API key needed. Run it, paste your project folder, and it fixes everything it can:

```text
$ uvx novuspipeline
NovusPipeline 0.1.0  modernize legacy code, test-gated

Project folder (paste a path, Enter for the current folder): C:\code\billing

Scanning
  48 source file(s) checked, 6 with automatic fixes
  • app/net.py Migrated Python 2 `urllib2` to its Python 3 equivalents +1 more
  ...
Each file is rewritten in a separate checkout, verified with your tests in a sandbox, and
committed to novus/modernize-20261118-164652 only if they pass. Start fixing? [Y/n]

Fixing
  [1/6] ✔ app/net.py  fixed and committed (1.6s)
  [2/6] ✖ app/maths.py  not committed: tests failed after the rewrite (2.2s)
  ...
Bring the fixes into `main` now (fast-forward)? [y/N]
```

- Your files and your checked-out branch are untouched until you choose to bring the fixes in. Every fix is a separate commit on a new `novus/modernize-<time>` branch, ready to review, merge or push as a PR.
- A file is committed only when your tests pass with the change. Files whose tests failed get one more try on top of the other fixes, since tests often import several legacy modules at once. A file no test covers is not committed: add tests, pass `--test-command`, or pass `--allow-untested` to commit after a compile/import check and review the diff yourself.
- Test files are not rewritten, because they are the safety net (`--include-tests` to change that). Files with uncommitted changes are skipped.
- If the folder is not a git repository yet, it offers to create one and commit the current files as the starting point.

Without questions, for scripts and CI: `novuspipeline fix /path/to/project --yes` (add `--merge` to fast-forward your branch to the fixes, `--branch NAME` to choose the branch).

### Individual commands

Every MCP tool is also a terminal command:

```sh
novuspipeline overview                      # files ranked by modernization risk
novuspipeline audit src/db.py               # findings for one file
novuspipeline context src/db.py             # dependents, used symbols, related tests
novuspipeline modernize src/db.py           # one file: rewrite + test + commit on a branch
novuspipeline standards install secure-defaults
novuspipeline test python -m pytest         # run tests in the sandbox
novuspipeline --workspace /path/to/project overview
```

Also: `index`, `gnn`, `similar`, `history`, `guidelines search|show|stats`, `status`, and `apply` (rewrites in place with a `.bak`; asks first, `--yes` to skip). `novuspipeline <command> --help` lists each command's options. The exit code is non-zero when a command fails, or when `test` runs failing tests, so it works in CI.

`novuspipeline` with no command opens the guided mode only in a terminal. MCP clients launch it without one and get the MCP server, exactly as before (`novuspipeline serve` forces the server).

## ⚙️ Settings

| Option | Purpose |
|---|---|
| `--workspace DIR` / `NOVUS_WORKSPACE_ROOT` | The codebase to modernize (default: current directory) |
| `--transport http --port 8000` | Serve over HTTP instead of stdio. Any address but `127.0.0.1` needs authentication ([enterprise guide](docs/enterprise.md)) |
| `NOVUS_AUTH_TOKENS_FILE`, `NOVUS_AUTH_TOKEN`, `NOVUS_OIDC_*` | HTTP authentication: hashed API tokens with roles, one shared token, or your identity provider |
| `NOVUS_POLICY_FILE` | Additional organization policy (the system-wide one is in the site config directory) |
| `NOVUS_AUDIT_LOG` | Append a hash-chained JSON line per tool call to this file |
| `NOVUS_LOG_LEVEL`, `NOVUS_LOG_FORMAT=json` | Server logs on stderr (default `CRITICAL`, plain text) |
| `NOVUS_PYTHON` | Interpreter for running your tests (default: the workspace's `.venv`/`venv`, else the server's) |
| `NOVUS_DATA_DIR` | Where the guideline database lives (default: your user data directory) |
| `NOVUS_GNN_DIR` | Use a retrained GNN checkpoint instead of the bundled one |
| `NOVUS_LLM_BASE_MODEL` | Hugging Face model for LLM proposals (default `unsloth/Qwen3.5-2B`, downloaded on first use) |
| `NOVUS_LLM_PATH` | Optional LoRA adapter directory |
| `NOVUS_LLM_PYTHON` | Run the LLM worker in a separate (GPU) environment |
| `NOVUS_LLM_OFFLINE=1` | Never download models; use the local Hugging Face cache only |
| `NOVUS_BASELINE=0` | Turn off the built-in security baseline (blocks `verify=False`, binding `0.0.0.0`, `raise` without `from` in `except`) |
| `NOVUS_LEARN_FROM_HISTORY=0` | Don't mine git history in the background after indexing |
| `NOVUS_SANDBOX=docker`, `NOVUS_SANDBOX_IMAGE` | Run tests in a container with no network (the image must contain your test dependencies) |
| `NOVUS_SANDBOX_MEMORY_MB`, `NOVUS_SANDBOX_MAX_PROCESSES` | Sandbox limits for test runs (default 4096 MB, 256 processes) |
| `NOVUS_SANDBOX_ENV_ALLOW=A,B` | Environment variables your tests need (all others are hidden from them) |
| `NOVUS_SANDBOX_LOW_INTEGRITY=0` | Windows: let tests write outside the sandbox (for suites that write into the real `node_modules`) |

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

## 🔎 What it finds

| Source | Examples |
|---|---|
| Your standards | whatever your rule packs define (see above) |
| Built-in legacy rules | Python 2 print and `urllib2`, bare `except`, `os.popen`, unsafe `pickle`, missing type hints (plus ES5 `var`, `any`, `require()` in TypeScript/JavaScript) |
| Ruff | pyupgrade, bugbear, security (bandit), pathlib, comprehension and simplification rules; safe fixes are applied automatically |
| Structural GNN (`[gnn]`) | the same smells from AST shape, localized to functions; structurally similar code |
| Codebase-wide | import cycles (import-time only), dead code, duplicated functions (rename-insensitive), untested hotspots, god modules, uses of APIs listed in `[deprecated]`, unresolvable dynamic imports |
| Learned from your history | calls and imports your team replaced the same way in several separate commits, with the commits as evidence |

**What gets rewritten automatically.** Rewrites keep behavior: `urllib2` to `urllib.request`, Python 2 `print`, `os.popen(...).read()` to `subprocess.run`, Ruff's safe fixes, your standards' fixes and learned renames. Two cases need more care:

- A bare `except:` becomes `except Exception:`, which no longer swallows `KeyboardInterrupt` and `SystemExit`. That is the point of the fix, but it is a behavior change, so it is skipped where the guarded code awaits, yields, exits or parses arguments (it may rely on catching cancellation or exit) and reported for review instead. A handler that re-raises becomes the exactly equivalent `except BaseException:`.
- JavaScript/TypeScript is analyzed without a parser. `var` becomes `let` only for a declaration made once, directly in a function body or module, and never used before it or outside that body; everything else, and every `require()`, is reported for review. The codebase index, the GNN and the imported-symbol check cover Python only, and a script file is verified with `npm test` (or the `test_command` you pass), not with the Python tests.

Dynamic imports are part of the import graph: `importlib.import_module("pkg.x")`, `__import__`, relative imports with `package=__name__`, `runpy`, spec loaders from file paths, and `getattr(module, "name")`. A dynamic import that can't be resolved is reported (`CTX-DYNAMIC`), and modules under its literal prefix are treated as reachable rather than dead.

**Learned smells.** `learn_from_history` (also run in the background after indexing) compares each changed function before and after every commit. A replacement such as `client.fetch_json(...)` → `client.get_json(...)` becomes a rule when it shows up in at least 3 separate commits, when the team usually chose that replacement for the old call, and when nobody kept writing the old call after the last replacement (this filters out churn of everyday calls). Remaining uses are reported in audits, file context and the overview. Pure renames with consistent history are applied automatically by the pipeline, and like every other change they must pass your tests first. In a time-split check on Inkscape's extensions repository, both auto-fixable rules learned from the history before mid-2019 were later fully applied by its developers, and 4 of the 5 rules with remaining uses were later acted on.

## 🧰 Tools

| Area | Tools |
|---|---|
| Codebase context | `index_codebase`, `get_codebase_overview`, `get_file_context`, `find_similar_code`, `learn_from_history` |
| Audit | `analyze_legacy_codebase`, `analyze_code_structure_gnn`, `read_legacy_file` |
| Modernize | `apply_code_modernization`, `generate_llm_modernization_proposal`, `run_autonomous_modernization_pipeline` |
| Verify & ship | `run_local_tests`, `create_git_migration_pr`, `format_modernization_report`, `finalize_git_migration_pr` |
| Standards | `list_modernization_standards`, `add_modernization_rule`, `remove_modernization_rule`, `install_standard_pack` |
| Guidelines (RAG) | `query_rag_guidelines`, `search_rag_by_id`, `ingest_rag_document`, `remove_rag_document`, `get_rag_stats`, `reset_rag_database` |
| Status | `get_gnn_model_status`, `get_local_llm_status` |

`run_autonomous_modernization_pipeline` never touches your working copy. It modernizes the file in a disposable git worktree, runs the tests that import the file in the sandbox, rejects the change if the tests fail or if a symbol other modules import would disappear, and commits only that file to a branch (`branch_name`, default `auto-modernization-branch`; repeated calls add commits to it). A run in which zero tests executed is not a pass: nothing is committed unless you pass `allow_untested=true`, and then only after the file compiles and imports in the sandbox; every result starts with a `Verification:` line saying which. It refuses to run on a file with uncommitted changes, or with your checked-out branch as the target. A PR description draft is written to `.novus_index/prs/`.

### 🛡️ Sandbox

Test commands must be test runners (`python -m pytest|unittest`, `pytest`, `npm test`, `npm run test[:x]`, `node --test`, `npx jest|vitest|mocha`, `cargo test`, `go test`, `mvn test`, `gradle test`), and they never run in your working tree. Paths in a test command must be relative to the project (no absolute paths, `~` or `..`), and options that start another program (`go test -exec`, `npm --script-shell`, Maven plugin goals) are refused:

| Protection | Windows | Linux | macOS |
|---|---|---|---|
| Runs on a temporary copy of the project (or the pipeline's disposable worktree), with no access to the server's stdin | ✓ | ✓ | ✓ |
| Cannot write outside the sandbox (your files, including the linked `node_modules`, are read-only to the tests) | ✓ low integrity level | with Docker | with Docker |
| Environment scrubbed (no tokens/API keys); home, app-data and temp point into the sandbox, so `~/.ssh`, `~/.aws` etc. are not found through them | ✓ | ✓ | ✓ |
| Memory and process-count limits; the whole process tree is killed on timeout | Job Object | rlimits + process group | process group (memory limit not enforced) |
| No network | with `NOVUS_SANDBOX=docker` | ✓ when user namespaces are available, else Docker | with Docker |

Every test result says which protections were active. Without Docker the sandbox is not a VM: tests still run as your user and can read files outside the project by absolute path; on Linux and macOS they can also write to them, and on Windows and macOS they have network access. Use `NOVUS_SANDBOX=docker` for code you don't trust.

The file tools do not read or write anything under `.git`, and the HTTP transport rejects requests whose `Host` or `Origin` header is not local, so a web page in your browser cannot drive the server. Serving any other address requires authentication (see [For organizations](#-for-organizations)).

### ✅ Approve once

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

## 🏢 For organizations

NovusPipeline can be rolled out to many developers or run as a shared service. The
[enterprise guide](docs/enterprise.md) covers it:

- **Organization policy**: a system-wide `policy.toml` that projects and users cannot loosen. Use it to disable tools, forbid untested commits or the LLM, require the Docker sandbox, enforce rule packs projects cannot override, restrict which branches get commits, and force an audit log. An invalid policy fails closed.
- **Authentication and roles** over HTTP: hashed API tokens (`novuspipeline token create`) or OIDC/SSO, with `viewer`, `developer` and `admin` roles mapped onto the tools' annotations. The server refuses to serve a non-loopback address without authentication.
- **Audit log**: one JSON line per tool call from any surface, with the caller, arguments (long values hashed), outcome and duration, hash-chained (`novuspipeline audit-log verify`).
- **Operations**: a container image (non-root, `/healthz`, JSON logs), no network egress in the core install, air-gapped installs, and releases with an SBOM and signed provenance.

`novuspipeline policy` shows the effective policy, authentication and audit settings.

## 👩‍💻 Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks and design rules, [docs/architecture.md](docs/architecture.md) for how the pieces fit together, and [docs/design-history.md](docs/design-history.md) for design decisions and evaluation results. Report vulnerabilities as described in [SECURITY.md](SECURITY.md); changes are listed in [CHANGELOG.md](CHANGELOG.md).

## 📄 License

MIT, see [LICENSE](LICENSE).
