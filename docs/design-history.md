# NovusPipeline Design History

_A chronological record of design decisions and evaluation results. File names and counts reflect the state at each phase; see the README and docs/architecture.md for the current design._

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

## Phase 7: Robust GNN Training, Fine-Tuned Adapter Testing, MCP Hardening (COMPLETED)

### GNN training protocol
- [x] Real training data from installed third-party packages, split by whole
  package (93 train / 22 val / 19 test, hashed on package name) and capped at
  25 chunks per package; stdlib split by file. Only training data is
  counterfactually augmented; evaluation splits are raw real code.
  Exact-duplicate samples between training and evaluation are removed.
- [x] Model selection on validation real code only, counting labels with at
  least 10 validation positives (002/004/005 have 0-1 in real code, so their F1
  is a coin flip). The Phase 5 sweep had selected on a test split; fixed.
- [x] GPU training (`--device`), LR-on-plateau, gradient clipping, early
  stopping; seed ensembles (`--ensemble N`); bootstrap 95% CIs and an
  invariance suite (rename parameters / reorder definitions / add docstrings)
  in `eval_gnn.py`.
- [x] Sweep (2 seeds each, validation real-code F1 on bare-except + type hints):
  SAGE 4x128 0.9996 (selected; zero variance, 0% false positives), SAGE 3x128
  0.9992, SAGE 4x64 0.9992, SAGE 3x64 0.9990, GCN 3x128 0.9958.
- [x] A 5-seed ensemble tied the single model on validation (0.9996); by the
  rule fixed beforehand (ship an ensemble only if it beats the single model on
  validation) the single model ships, at 1/5 of the inference cost.

Final model vs previous checkpoint, same harness, 19 never-seen packages:

| | previous (SAGE 3x64) | final (SAGE 4x128) |
|---|---|---|
| File-level macro-F1 (95% CI) | 0.953 (0.911-0.965) | 0.998 (0.994-1.0) |
| 006 file level, recall / precision | 0.755 / 1.0 | 0.994 / 0.994 |
| 006 per function (1,574 positives), recall / precision | 0.956 / 0.999 | 0.999 / 1.0 |
| Bare except (6-7 positives) | F1 1.0 | F1 1.0 |
| Invariance (rename / reorder / docstrings) | 100% / 100% / 99.7% | 100% / 99.7% / 100% |
| Latency | ~0.3 s / 3.2k nodes, ~1.2 s / 32k | same |

Real-code support for 002/004/005 remains 0-2 positives, so those labels are
validated by the synthetic and contrast suites (all pass).

### Fine-tuned LoRA adapter (`eval_llm.py`)
- [x] The adapter had never been applied. It was trained on the multimodal
  `Qwen3_5ForConditionalGeneration` (weights under `model.language_model.layers.*`);
  the old loader used the wrong class and lacked torch/peft, and a text-only
  load leaves PEFT's LoRA layers at their zero init because no saved key
  matches. `llm_worker.py` remaps the keys, verifies 192/192 tensors loaded
  with non-zero LoRA B, and refuses to start otherwise.
- [x] A/B on 8 legacy snippets (same weights, greedy, adapter on/off; parity
  measured by executing original and rewrite in an isolated subprocess):

  | | adapter | base Qwen3.5-2B | rule engine |
  |---|---|---|---|
  | parseable code | 2/8 | 8/8 | 8/8 |
  | passes validation | 25% | 100% | 100% |
  | smells resolved | 25% | 94% | 27% |
  | behavior parity | 15% | 95% | 100% |
  | speed (RTX 4050) | 6.4 tok/s | 8.5 tok/s | instant |

  The adapter (30 steps, batch 2, ~10 examples) never learned to stop: it
  drops the requested code fence, invents functions, loops, and hits the
  token cap. Recommendation: run the base model (`NOVUS_LLM_ADAPTER=0`) until
  the adapter is retrained on a few hundred legacy->modern pairs with EOS.
- [x] Even the base model changed behavior once (`except:` ->
  `except ValueError:`), which only tests can catch; `use_llm` stays opt-in and
  test-gated.

### LLM integration
- [x] LLM runs in a separate worker process (`llm_worker.py`, own GPU venv,
  `requirements-llm.txt`), JSON lines over stdio; restarted on crash, killed on
  timeout; stderr logged to `%TEMP%/novus_llm_worker.log`. Text-only model
  class (no vision tower): 3.55 GB VRAM, ~25 s load. LoRA merged for inference
  unless `NOVUS_LLM_MERGE=0`.
- [x] Validation gate for proposals: parses, no new rule-engine smells, keeps
  imported symbols, adds no new functions/classes or `__main__` block.
- [x] `run_autonomous_modernization_pipeline(use_llm=True)` applies only
  validated proposals, else falls back to the rule engine; tests gate both.

### MCP hardening
- [x] All 21 tools carry MCP annotations (read-only / destructive /
  idempotent), run in worker threads with per-tool timeouts, and raise
  `ToolError` (isError) instead of returning "Error: ..." as success.
- [x] Workspace-mutating tools are serialized by a lock.
- [x] `modernization_workflow` MCP prompt; `get_local_llm_status(probe=true)`
  reports device, VRAM, adapter verification.
- [x] Tests: 84 (stub-worker protocol: crash/restart, hang/timeout, valid /
  API-breaking / hallucinated proposals; LLM pipeline both ways; real stdio
  server with annotations, isError and concurrent calls).

## Phase 8: Public Distribution (IN PROGRESS)
- [x] Python package `novuspipeline` (`src/` layout, relative imports), console
  script `novuspipeline [--workspace DIR] [--transport stdio|http]`, `python -m
  novuspipeline`. Light core install; extras `[gnn]` and `[llm]`.
- [x] No machine-specific paths: GNN checkpoint (1.2 MB) bundled in the wheel;
  guideline DB in the user data dir, seeded automatically on first use; LLM
  defaults to the base model (`NOVUS_LLM_BASE_MODEL`), adapter optional
  (`NOVUS_LLM_PATH`), worker interpreter auto-detected.
- [x] Verified from a built wheel in a fresh venv: core install (no torch) serves
  all 21 tools over stdio, auto-seeds RAG, indexes a project, degrades GNN/LLM
  features with install hints; `[gnn]` install loads the bundled checkpoint.
- [x] Tests isolated from the user's data dir; CI matrix (Linux/Windows/macOS x
  Python 3.11-3.13) and a wheel smoke test; release workflow with PyPI trusted
  publishing.
- [ ] Owner decisions before release: confirm the MIT license (placeholder), set
  up the PyPI trusted publisher, push and tag `v0.1.0`, then register with the
  MCP Registry (`mcp-publisher init` / `publish`; README carries the
  `mcp-name` marker).

## Phase 9: Broader Detection, True Sandbox, Learning From History (COMPLETED)

### Ruff as detection and safe-fix backend (`ruff_backend.py`)
- [x] Ruff (dependency `ruff>=0.6`) scans every indexed file and every audit:
  pyupgrade, bandit, bugbear, pyflakes, pathlib, comprehensions, simplify,
  perflint, syntax errors. Noisy rules for legacy code are ignored (bare-except
  is already a built-in rule; `assert`, subprocess and random warnings).
- [x] Only Ruff's safe fixes (UP, C4, PIE, SIM) are applied by the modernizer,
  after the built-in rewrites; LLM proposals are rejected if they introduce new
  HIGH/CRITICAL Ruff findings (e.g. undefined names, F821).

### Codebase-level smells (`codebase_context.py`, `code_facts.py`)
- [x] `CTX-CYCLE` (import-time cycles only; lazy imports inside functions don't
  count), `CTX-DEAD` (unreferenced top-level definitions; decorated, exported,
  entry points, dunder and `main` are exempt), `CTX-CLONE` (rename-insensitive
  structural hash, >= 40 AST nodes), `CTX-UNTESTED` (risky, depended-on files
  with no related tests), `CTX-GOD` (fan-in, fan-out and size together),
  `CTX-DEPRECATED` (`.novus/config.toml` `[deprecated]`), `CTX-DYNAMIC`.
- [x] Risk score is severity-weighted across rules, Ruff, GNN, codebase-level
  and learned smells; each smell id counts once.
- [x] On this repository: found the one truly unused function (`encode_graph`)
  and the two modules with no tests; no dead-code false positives on the
  decorator-registered MCP tools.

### Dynamic imports
- [x] Resolved into the import graph: `importlib.import_module` (absolute and
  relative with `package=__name__`/`__package__`), `__import__`, `runpy`,
  spec loaders from file paths, aliases bound from `import_module`, and
  `getattr(module, "literal")` as symbol usage; star imports honor `__all__`.
- [x] Unresolvable targets (e.g. f-strings) are reported as `CTX-DYNAMIC`, and
  modules under their literal prefix are treated as reachable (no false dead code).

### True sandbox (`sandbox.py`) and worktree pipeline
- [x] Every test run: temporary project copy, allowlisted environment, home/temp
  redirected, memory/process limits, whole-tree kill on timeout (Windows Job
  Object created suspended so no child escapes; POSIX rlimits + process group),
  network isolation on Linux via `unshare -rn`, Docker opt-in. Verified on
  Windows: workspace writes don't reach the user's files, secrets are invisible,
  300 MB limit raises MemoryError, timeout kills grandchildren, linked
  `node_modules` survive cleanup.
- [x] `run_autonomous_modernization_pipeline` is no longer destructive: it works
  in a disposable worktree and only adds commits to a branch that isn't checked
  out. It refuses files with uncommitted changes. PR drafts moved to
  `.novus_index/prs/`. Together with the MCP annotations, clients can approve
  every non-destructive tool once (README: "Approve once").

### Learning smells from commit history (`history_learner.py`)
- [x] Mines focused call/import replacements per changed function; rules need
  >= 3 separate commits, >= 60% consistency, and no adoption of the old call
  after the last replacement. Evidence (commits, messages) is attached to every
  finding. Auto-fix only for argument-preserving renames with >= 90%
  consistency that need no new import; applied by the test-gated pipeline.
- [x] Incremental (`.novus_index/learned.json`), runs in the background after
  each index refresh (`NOVUS_LEARN_FROM_HISTORY=0` disables), and on demand via
  the new `learn_from_history` tool (22 tools now).
- [x] Mining speed: one `git log --raw -M` plus a blob parse cache; the Inkscape
  extensions repository (1553 Python-touching commits) mines in ~3 minutes
  (was 8.5 minutes with per-commit diffs), re-runs in < 1 s.
- [x] Time-split evaluation on Inkscape extensions: learned from the history up
  to June 2019, then checked against what the developers did afterwards.
  | Learned rule (before T) | Uses at T | Uses at HEAD | Auto-fix |
  |---|---|---|---|
  | `self.unittouu` -> `self.svg.unittouu` | 8 | 0 | yes |
  | `self.xpathSingle` -> `self.svg.getElement` | 2 | 0 | yes |
  | `self.selected.items` -> `self.svg.selected.items` | 3 | 0 | no (70% consistency) |
  | `simpletransform.parseTransform` -> `Transform` | 5 | 3 | no (arguments change) |
  | `simplestyle.parseStyle` -> `inkex.parseStyle` | 1 | 1 | no |
  | import `lxml` -> `inkex.elements` | 70 files | 24 of 473 files | no |
  Both auto-fixable rules were later fully applied by the team; 4 of the 5 call
  rules with remaining uses were later acted on. The adoption filter removed the
  one false positive (`self.assertEqual` -> `self._assertPath`, a test-helper
  swap in 4 commits while `assertEqual` gained 600 new uses).
- [x] Tests: 121 (Ruff, dynamic imports, context smells, sandbox protections,
  worktree pipeline incl. CRLF and branch extension, learner mining, filters,
  incrementality, findings in all tools, learned fix applied on a branch).

### Known limitations
- Learned rules cover call targets and imports, matched by dotted name (no type
  inference): `self.x.load` and `other.load` are different calls.
- Without Docker, Windows and macOS test runs have no network isolation, and
  the sandbox does not stop tests from reading files by absolute path.
- Clone detection is per function; near-duplicates with small edits are not found.

## Phase 10: Organization & Industry Standards (COMPLETED)
- [x] Rule packs (`standards.py`): TOML files of rules with detection (calls with
  import aliases resolved, `require_keyword`, imports, masked regex), guideline,
  automatic fix (`replace_call` adds the needed import, `replace_import` keeps
  references working via `import new as old`, `add_keyword`, regex
  `replacement`), severity, `enforce = warn|block`, and path scope. Invalid rules
  are reported with the reason and skipped; a broken pack never breaks the server.
- [x] Scopes: personal (user data dir) and project (`.novus/standards/`, shared
  via git); built-in example packs `secure-defaults` and `python-312-ready`
  installed by copying. Packs can add or ignore Ruff codes (e.g. a rule that
  supersedes `S113` with a fix).
- [x] Wired into every stage: audits (standards listed first, with their
  guideline and enforcement), index and risk scores (source `standard`; the
  index rebuilds when a pack changes), `query_rag_guidelines` /
  `search_rag_by_id` (standards ranked with the same TF-IDF scoring), LLM prompt
  (standards the file violates first), LLM validation (no new violations),
  rule-based rewrites (standards' fixes applied last), and the pipeline (no
  commit while a `block` rule is violated; prepared changes are listed).
- [x] New tools: `list_modernization_standards`, `add_modernization_rule`,
  `remove_modernization_rule`, `install_standard_pack`, `remove_rag_document`
  (27 tools). `ingest_rag_document` accepts any category slug (e.g. `fintech`).
- [x] Tests: 138 (+17: validation messages, alias resolution, keyword and
  `**kwargs` handling, string/comment masking, import and regex fixes, path
  scope, user scope, broken packs, hand-written packs protected, audit/overview/
  RAG integration, Ruff superseding, apply, pipeline block and commit, LLM gate).
- Limits: matching is by dotted name after alias resolution (no type
  inference); `pattern` rules are line-based; fixes are syntactic and rely on
  the tests for behavior.
