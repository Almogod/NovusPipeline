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

## Phase 4: Git PR & Modernization Reporting (COMPLETED)
- [x] Implement enterprise Markdown report artifact generator (`reporter.py` / `ModernizationReporter`).
- [x] Expose `format_modernization_report` MCP tool saving structured audit reports under `reports/modernization_report_<branch>.md`.
- [x] Expose `finalize_git_migration_pr` MCP tool managing complete draft PR metadata, commit verification, and `gh` CLI command instructions.
- [x] Expand comprehensive unit test suite to **26/26 tests passing** cleanly in `test_server.py`.

## Phase 5: GNN Structural Code-Smell Classifier (COMPLETED)
- [x] `code_graph.py`: Python source -> AST graph. Each node carries its AST
  type, the parent field it fills (`returns`, `annotation`, `type`, ...) and a
  hashed identifier (`Name.id`, `Attribute.attr`, `alias.name`,
  `ImportFrom.module`; md5 buckets, stable across processes). Edges are
  parent<->child plus next-sibling. Iterative traversal (no recursion limit);
  `ValueError` above 150k nodes or for unparseably deep source. No torch dependency.
- [x] `gnn_model.py`: `SmellGNN`, 3-layer GraphSAGE with residuals, max +
  attention pooling, multi-label head over `PY-SMELL-002..006`. Architecture
  is a `GNNConfig` stored in checkpoint metadata, so checkpoints always reload
  with the architecture they were trained with. Per-label thresholds are
  calibrated on validation data. `encode_graph()` exposes the pooled graph
  vector for a future graph+transformer encoder.
- [x] `train_gnn.py`: distills `LegacySmellDetector` (run on source with
  strings/comments stripped) into the GNN. Corpus = 2000 synthetic template
  files (with structurally identical hard negatives, typed/untyped variants,
  smells in methods, 1-80 functions per file) + ~820 chunks of real Python
  stdlib code with counterfactual rewrites (add/remove return annotations,
  bare/typed excepts). Real code is split train/val/test by file. Fully
  seeded, early stopping, `--sweep` for multi-seed comparisons.
- [x] `eval_gnn.py`: stress-test harness (in-distribution, smelly-vs-lookalike
  contrast suite, third-party code never trained on, needle-in-haystack,
  latency/robustness). Writes `.gnn_model/eval_report.json`.
- [x] MCP tools `get_gnn_model_status` and `analyze_code_structure_gnn`;
  `analyze_legacy_codebase` appends a GNN cross-check that can never break the
  rule-based audit (any GNN error is reported inline).
- [x] Pinned `requirements.txt`. Unit tests: 38 (graph features, recursion
  fix, identifier-collision guard, lookalike separation, audit resilience).

### Phase 5 evaluation (shipped checkpoint, seed 42)
The first version saw only AST node *types*. Its 0.91 in-distribution F1 hid
that `import urllib2`/`pickle`/`os.popen` code and clean lookalikes
(`json`, `httpx`) produced byte-identical graphs, so 3 of 5 smells were
undetectable by construction.

| Stress test | original (types only, synthetic) | + field/identifier features (synthetic) | shipped (+ real stdlib code, counterfactuals) |
|---|---|---|---|
| Smelly vs. structural lookalike (8 pairs) | 0/8 separated | 7/8 | 8/8 |
| Third-party code, never trained on (500 chunks), macro-F1 | not measured | 0.39 | 0.73* |
| `PY-SMELL-006` false-positive rate on third-party code | not measured | 30% | 2% |
| This repo's own files agree with teacher | not comparable (raw teacher) | 5/10 | 10/10 |
| One pickle smell in an 8k-node file | p = 0.001 | p = 1.0 | p = 1.0 |
| 2000-term expression | RecursionError | ok | ok |
| Latency (CPU) | | | ~1 s / 32k nodes, ~3 s / 96k nodes |

\* Bare except and pickle score P = R = 1.0. The only `PY-SMELL-002` positives
in third-party code are rule-engine false positives (see below), which the
GNN correctly declines to copy. Missing type hints (006): precision 0.985,
recall 0.88.

Architecture sweep (3 seeds each, F1 on the well-supported labels 003/006 of
held-out stdlib files): GraphSAGE 0.988 (selected; also fastest),
SAGE-max 0.983, GCN 0.974, GIN 0.940. Ablation without identifier features:
F1 = 0 on all of 002/004/005, confirming the identifier features are required.

Rejected: an "annotate every def but one" counterfactual raised 006 recall
0.88 -> 0.96 but its false-positive rate 2% -> 15%; for an advisory signal
next to a rule engine, false alarms cost more than misses.

### Retrained after the Phase 6 rule-engine fixes (current checkpoint)
The teacher changed (see Phase 6), so the GNN was retrained on its labels, now
also with standalone real functions (3 per stdlib chunk) so per-function
predictions are trained for rather than extrapolated. Both rows below are
judged by the fixed rule engine:

| | previous checkpoint | current checkpoint |
|---|---|---|
| Third-party code (500 chunks), macro-F1 | 0.858 | 0.980 |
| 006 file level: precision / recall / false-positive rate | 0.985 / 0.84 / 2.2% | 1.0 / 0.89 / 0% |
| 006 per function (1500 functions): precision / recall / FP rate | 0.84 / 0.95 / 19.7% | 0.997 / 0.97 / 0.3% |
| Contrast pairs, needle-in-haystack | 8/8, pass | 8/8, pass |

### Known GNN limitations
- File-level 006 still misses ~11% of files where one function in an
  otherwise-typed module lacks hints. The codebase index compensates by
  also using per-function predictions (recall 0.97) in its risk scores.
- Real-code support for 002/004/005 is tiny (a handful of positives in
  third-party and held-out stdlib code), so those labels are validated mainly
  by the synthetic and contrast suites.
- Labels are only as good as the teacher: the GNN learns `LegacySmellDetector`.

## Phase 6: Project Hardening & Codebase-Aware GNN Integration (COMPLETED)

### Codebase context index (`codebase_context.py`)
- [x] Persistent index of every Python file (`.novus_index/index.json`,
  gitignored), refreshed incrementally by content hash; full rebuild when the
  GNN checkpoint changes. Skips venvs, `node_modules`, build/cache dirs.
- [x] Per-function GNN smell predictions (localizes the file-level model) and
  graph embeddings, from one batched forward pass (`gnn_model.analyze_many`).
- [x] Import graph with relative imports, `import x as y` attribute tracking,
  `from pkg import submodule`, and `src/` layouts: dependents, dependencies,
  related tests, and API surface (symbols other modules actually use).
- [x] Structural similarity search over mean-centered GNN embeddings (raw
  post-ReLU embeddings all scored ~0.97 cosine). On this repo it surfaced the
  TF-IDF tokenizer duplicated in `server.py` and `ingest_rag.py`, now removed.
- [x] Hotspot ranking: severity-weighted smells (rule engine and/or GNN) with
  blast radius (number of importing modules).
- [x] New MCP tools: `index_codebase`, `get_codebase_overview`,
  `get_file_context`, `find_similar_code`.
- [x] Context feeds existing tools: `analyze_legacy_codebase` appends it;
  `generate_llm_modernization_proposal` puts dependents/API surface into the
  LLM prompt; the pipeline picks the tests that import the file, rolls back if
  the change removes an imported symbol, and lists blast radius and follow-up
  files in the PR; `apply_code_modernization` warns on API breaks.

### Fixes
- [x] Test sandbox allowed `python -c` / `node -e` / any `npm run`: now an
  allowlist of test-runner command shapes; `.cmd`/`.bat` runners reject cmd.exe
  metacharacters; `python` resolves to the workspace venv (`NOVUS_PYTHON`
  overrides); output truncated; 600 s timeout.
- [x] Pipeline treated any output containing "OK" as a pass: now exit code 0.
- [x] Git tool committed the whole workspace (`git add -A`) and used the raw
  branch name in a file path: now validates branch names
  (`git check-ref-format`), stages/commits only the given files (pipeline passes
  the modernized file), excludes `.bak`/PR drafts/index by default, and commits
  via the git CLI so hooks run.
- [x] Path check: resolves symlinks, case-insensitive on Windows, rejects
  other drives.
- [x] Rule engine: masks strings/comments; 002 no longer flags `urllib.parse` /
  `urllib3`; AST-based 003/006 (no duplicate findings, multi-line signatures,
  unannotated parameters); per-language rule sets; no crash on deep nesting.
- [x] Code modernizer was not behavior-preserving: `urllib2` -> `httpx` left
  `urllib2.*` calls behind and changed semantics; now -> `urllib.request` /
  `urllib.error` (same API). `os.popen` -> `subprocess.run(shell=True)` with the
  import added (was `shell=False`, which breaks any command with arguments).
  Bare-except rewrite no longer touches string literals. Removed the
  `from __future__ import annotations` rule (changes runtime annotation
  semantics; was inserted above docstrings). Added Python 2 `print`
  conversion. TS `var` -> `let` (not `const`, which breaks reassignment).
  Output that no longer parses is rejected.
- [x] RAG: one shared TF-IDF embedder; camelCase tokenization fixed; query
  expansion (implemented but never called) now used; relevance score now real
  cosine similarity (was `1 - squared L2`); collections carry an embedding
  version and are re-embedded in place (keeping custom docs) when it changes;
  progress output moved off stdout (it corrupted the MCP stdio channel); the
  guideline DB lives with the tool, not inside the analyzed workspace.
- [x] `NOVUS_WORKSPACE_ROOT` points the server at any codebase.
- [x] Local LLM: model cached per process (was reloaded on every call);
  greedy decoding without conflicting sampling args.
- [x] `analyze_legacy_codebase` skipped the GNN when rules found nothing;
  findings are now grouped per smell (one RAG lookup per smell, not per line).
- [x] Tests ran real git operations on this repository (switched branches,
  auto-committed, deleted `.gnn_model/`): all git/pipeline/file-writing tests
  now run in a throwaway repository. 74 tests.

### Remaining limitations
- RAG retrieval is lexical TF-IDF over 14 guidelines; some queries rank a
  generic guideline above the specific one.
- TS/JS rules are regex-only without string masking; `require` -> `import`
  uses a default import, which differs from CommonJS for some modules.
- The import graph does not see dynamic imports (`importlib`, `__import__`).
- Branches `test-auto-pipeline-branch`, `test-final-pr-branch` and
  `test-modernization-branch` were created in this repo by the old tests; they
  are left in place (`git branch -D <name>` to remove).
