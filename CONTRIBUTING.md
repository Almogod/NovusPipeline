# Contributing

## Setup

```shell
git clone https://github.com/Almogod/NovusPipeline && cd NovusPipeline
python -m venv .venv && .venv/Scripts/activate        # source .venv/bin/activate on macOS/Linux
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux/Windows: avoids CUDA wheels
pip install -e ".[gnn]" build twine ruff
```

## Checks (CI runs the same)

```shell
ruff check src tests
python -m unittest discover -s tests -v    # git and file tests run in throwaway repositories
python -m build && twine check dist/*
```

All three must pass before merging. Tests must not touch this repository, the
user's data directory or the network; use `WorkspaceTestCase` in
`tests/test_server.py` for anything that needs a workspace.

## Design rules

These constraints are what makes the tool safe to point at production code;
changes that weaken them need a design discussion first.

- **Behavior preservation**: automatic rewrites must keep behavior. Anything
  that might not (LLM output, learned or standard fixes) is validated and
  verified by the project's tests before it is committed.
- **API stability**: never remove or rename a symbol other modules import; the
  pipeline rejects changes that would.
- **Scoped, isolated changes**: the pipeline works in a disposable worktree and
  commits only the file it modernized; it never edits the working copy, the
  checked-out branch, or pushes.
- **Sandboxed execution**: project code (tests) only runs through `sandbox.run`.
- **Workspace boundary**: file paths are resolved and checked against the workspace.
- **Clean stdio**: under the stdio transport, stdout is the JSON-RPC channel;
  log to stderr.
- **Honest output**: tools report what was actually done and which protections
  were active; failures raise MCP errors instead of returning success text.

## GNN and LLM

- `python -m novuspipeline.train_gnn --device auto` retrains the GNN;
  `python -m novuspipeline.eval_gnn` stress-tests a checkpoint. Record results
  in `docs/design-history.md` when replacing the bundled checkpoint.
- `python -m novuspipeline.eval_llm` compares the base LLM with a LoRA adapter.
  The LLM worker can run from a separate GPU environment: create a venv with
  CUDA torch and `pip install -e ".[llm]"`, then set `NOVUS_LLM_PYTHON` to its
  interpreter (a `.venv-llm` in the repository root is found automatically).

## Releases

1. Update `__version__` in `src/novuspipeline/__init__.py` and `CHANGELOG.md`.
2. Tag `vX.Y.Z` and publish a GitHub release; the `publish` workflow builds,
   checks that the tag matches the version, and uploads to PyPI with trusted
   publishing.
