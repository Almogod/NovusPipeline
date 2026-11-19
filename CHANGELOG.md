# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added (organizations)
- **Organization policy** (`policy.toml` in the site config directory, plus `NOVUS_POLICY_FILE`; most
  restrictive wins): `disabled_tools`, `allow_destructive_tools`, `allow_untested`, `allow_llm`, `llm_offline`,
  `require_docker_sandbox`, `required_packs` (loaded last, so projects cannot override or weaken their rules),
  `branch_prefix`, `audit_log` and `require_auth`. Enforced for MCP calls, terminal commands and guided mode.
  An invalid policy fails closed. `novuspipeline policy` shows the effective settings.
- **HTTP authentication and roles**: hashed API tokens with expiry (`NOVUS_AUTH_TOKENS_FILE`,
  `novuspipeline token create`), a shared token (`NOVUS_AUTH_TOKEN`), and OIDC/JWT (`NOVUS_OIDC_JWKS_URI`,
  `NOVUS_OIDC_ISSUER`, `NOVUS_OIDC_AUDIENCE`). Roles `viewer`/`developer`/`admin` (scopes `novus:read`,
  `novus:write`, `novus:admin`) follow the tools' annotations; `tools/list` only shows what the caller may call.
- **Audit log** (`NOVUS_AUDIT_LOG` or the policy's `audit_log`): a hash-chained JSON line per tool call with
  caller, transport, arguments (long values hashed), outcome and duration, including refused calls; safe for
  several processes; `novuspipeline audit-log verify` checks the chain.
- `GET /healthz` for load balancers and orchestrators; `NOVUS_LOG_LEVEL` and `NOVUS_LOG_FORMAT=json`.
- Container image (`Dockerfile`: non-root, health check, JSON logs), published to GHCR on release.
- Releases carry a CycloneDX SBOM and signed build provenance; CI runs CodeQL, `pip-audit` and a container
  smoke test. `docs/enterprise.md` is the deployment guide.

### Changed (organizations)
- **The HTTP transport refuses to serve a non-loopback address without authentication** (it used to warn).
  Configure authentication, or pass `--allow-unauthenticated` when a gateway in front authenticates.
- ChromaDB anonymized telemetry is explicitly disabled.

### Changed
- **A run where zero tests executed is no longer a pass.** `run_autonomous_modernization_pipeline` used to
  commit when a project without tests exited 0. It now refuses, unless `allow_untested=true`, and even then
  only after a built-in floor check (the file must compile and import in the sandbox; an import that also
  fails on the original file is reported as inconclusive). Every result, PR draft and report starts with a
  `Verification:` line saying how many tests ran, or that none did.

### Added
- Built-in `baseline` security pack, active with no configuration: it blocks commits of files containing
  `verify=False` (S501), binding all interfaces (S104) and `raise` inside `except` without a cause (B904).
  Install a copy with `install_standard_pack baseline` to edit it, set `NOVUS_BASELINE=0` to turn it off, or
  mark a reviewed line `# noqa: S104`. Packs can now list Ruff codes under `[ruff] block`.
- Guided terminal mode: `novuspipeline` in a terminal asks for a project folder, then fixes every
  file it can, each test-gated and committed to a new `novus/modernize-<time>` branch; files whose
  tests failed are retried on top of the other fixes. `novuspipeline fix PATH --yes` does the same
  without questions (`--merge` fast-forwards your branch to the fixes, `--allow-untested` passes
  through to the pipeline).
- Command line interface: `overview`, `audit`, `context`, `modernize`, `apply`, `test`, `standards`,
  `guidelines`, `history`, `index`, `gnn`, `similar`, `status`, backed by the same functions as the MCP
  tools. Launched by an MCP client (no terminal), `novuspipeline` still starts the MCP server.
- `.novus_index/` contains its own `.gitignore`, so it no longer shows up in `git status`.

### Security
- Test commands may no longer contain absolute paths, `~` or `..`: they pointed
  out of the sandbox copy at real files (`pytest --basetemp=<dir>` empties that
  directory). Options that start another program are refused (`go test -exec`,
  `-toolexec`, `npm --script-shell`, `cargo --config`, `node -e`, Maven
  `plugin:goal` invocations).
- Sandboxed tests no longer inherit the server's stdin, which is the MCP channel
  under the stdio transport.
- Windows: sandboxed tests run with a low integrity level and cannot write
  outside the sandbox, including through the linked `node_modules`
  (`NOVUS_SANDBOX_LOW_INTEGRITY=0` to disable).
- pytest is detected without importing it from the project directory, which ran
  a `pytest.py` in the workspace outside the sandbox.
- The file tools refuse paths under `.git`.
- The HTTP transport validates `Host` and `Origin` headers on loopback addresses
  (DNS rebinding).
- GNN checkpoints load with `weights_only=True`; the LLM worker log moved from
  the shared temp directory to the user data directory.

### Changed
- A bare `except:` is narrowed to `except Exception:` only where the `try` block
  does not await, yield, exit or parse arguments; a re-raising handler becomes
  `except BaseException:`; other cases are reported for manual review.
- JavaScript/TypeScript: strings, comments, templates and regex literals are
  masked before detection. `var` becomes `let` only where that is provably
  equivalent, and `require()` is no longer rewritten to `import`.
- The pipeline verifies a JavaScript/TypeScript file with `npm test` (or
  `test_command`) and stops when there is neither, instead of running the
  Python tests.

## [0.1.0]

First public release.

### Added
- MCP server (`novuspipeline`, stdio or HTTP) with 27 annotated tools and a
  `modernization_workflow` prompt.
- Detection from five sources: built-in legacy rules, Ruff, a structural GNN
  (optional `[gnn]` extra, checkpoint bundled), codebase-level analysis (import
  cycles, dead code, clones, untested hotspots, god modules, deprecated internal
  APIs, unresolved dynamic imports), and smells learned from the repository's
  git history.
- Modernization standards: TOML rule packs (personal and per project) with
  detection, guidelines, automatic fixes and `warn`/`block` enforcement;
  built-in example packs `secure-defaults` and `python-312-ready`.
- Codebase index: import graph including dynamic imports, dependents and the
  API they use, related tests, per-function findings, risk ranking.
- Rewrites: behavior-preserving built-in rewrites, Ruff safe fixes, learned
  renames, standards fixes, and optional local-LLM proposals (`[llm]` extra)
  behind a validation gate.
- Sandbox for test runs: temporary project copy, scrubbed environment,
  memory/process limits, process-tree kill on timeout, network isolation on
  Linux or with Docker.
- Pipeline that modernizes one file in a disposable git worktree, verifies it
  with the related tests, and commits it to a branch without touching the
  working copy; PR description drafts and Markdown reports.
- Guideline store (ChromaDB, offline TF-IDF embeddings) with custom categories.
