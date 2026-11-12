# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
