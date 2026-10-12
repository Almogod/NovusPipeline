# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
