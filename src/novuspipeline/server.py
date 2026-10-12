import logging
import os
import shutil
import sys

# Ensure all logging is routed strictly to stderr at CRITICAL level BEFORE importing fastmcp
logging.basicConfig(level=logging.CRITICAL, stream=sys.stderr)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.CRITICAL)
logging.root.addHandler(stderr_handler)

os.environ["FASTMCP_SHOW_SERVER_BANNER"] = "0"
os.environ["FASTMCP_LOG_LEVEL"] = "CRITICAL"

import contextlib
import functools
import re
import shlex
import subprocess
import threading
from collections import Counter

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import codebase_context, gnn_model, history_learner, local_llm, ruff_backend, sandbox, standards
from .ingest_rag import CHROMA_DIR, COLLECTION_NAME, ensure_seeded, get_collection
from .ingest_rag import similarity_from_distance as ingest_rag_similarity
from .modernizer import CodeModernizer, LegacySmellDetector, missing_names
from .reporter import ModernizationReporter

# ---------------------------------------------------------------------------
# Server Initialization
# ---------------------------------------------------------------------------

mcp = FastMCP("NovusPipeline")

# Tools that modify the workspace (files, git, test runs) are serialized: they run in
# worker threads, and two concurrent pipeline runs must never interleave git operations.
_WORKSPACE_LOCK = threading.RLock()
_TOOL_KINDS = {
    "read": {"read_only_hint": True, "destructive_hint": False, "idempotent_hint": True},
    "write": {"read_only_hint": False, "destructive_hint": False, "idempotent_hint": True},
    "destructive": {"read_only_hint": False, "destructive_hint": True, "idempotent_hint": False},
}


def _tool(kind: str, timeout: float = 900.0):
    """
    Registers a tool with MCP annotations, a timeout, and thread execution. Results starting
    with "Error" are raised as ToolError so clients see isError=True. The undecorated function
    is returned unchanged, so tools can call each other and tests can call them directly.
    """
    def register(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if kind == "read":
                result = fn(*args, **kwargs)
            else:
                with _WORKSPACE_LOCK:
                    result = fn(*args, **kwargs)
            if isinstance(result, str) and result.startswith("Error"):
                raise ToolError(result)
            return result

        mcp.tool(wrapper, name=fn.__name__, timeout=timeout, run_in_thread=True,
                 annotations=ToolAnnotations(open_world_hint=False, **_TOOL_KINDS[kind]))
        return fn
    return register

# The codebase being modernized. Defaults to the server's working directory;
# set NOVUS_WORKSPACE_ROOT to point the server at another project.
WORKSPACE_ROOT = os.path.abspath(os.environ.get("NOVUS_WORKSPACE_ROOT") or os.getcwd())
MAX_READ_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB safety limit
TEST_TIMEOUT_SECONDS = 600
MAX_TOOL_OUTPUT_CHARS = 20_000

QUERY_EXPANSIONS = {
    "async": ["asyncio", "httpx", "aiofiles", "taskgroup", "cancellederror", "concurrency"],
    "blocking": ["asyncio", "httpx", "time.sleep", "thread", "io"],
    "type": ["type hints", "mypy", "pyright", "pydantic", "strict", "union"],
    "security": ["sanitization", "path traversal", "pydantic", "zod", "injection", "whitelist"],
    "react": ["hooks", "usestate", "useeffect", "tanstack", "zustand", "context"],
    "typescript": ["strict", "tsconfig", "zod", "interface", "type guard"],
    "python2": ["print", "urllib2", "optparse", "subprocess", "future"],
}


def is_path_in_workspace(target_path: str) -> bool:
    """
    True if `target_path` resolves inside the workspace. Symlinks are resolved
    (a link inside the workspace pointing outside is rejected) and comparison is
    case-insensitive where the filesystem is (Windows).
    """
    root = os.path.normcase(os.path.realpath(WORKSPACE_ROOT))
    target = os.path.normcase(os.path.realpath(target_path))
    try:
        return os.path.commonpath([root, target]) == root
    except ValueError:  # different drives on Windows
        return False


def _resolve_workspace_path(file_path: str) -> str:
    """Absolute path for a workspace-relative or absolute `file_path` (not yet validated)."""
    return os.path.abspath(file_path if os.path.isabs(file_path) else os.path.join(WORKSPACE_ROOT, file_path))


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"[... {len(text) - limit} earlier characters truncated ...]\n" + text[-limit:]


def expand_query(query: str) -> str:
    """Expand natural language queries with domain tech synonyms for higher recall."""
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    expanded = set(tokens)
    for key, synonyms in QUERY_EXPANSIONS.items():
        if key in tokens:
            expanded.update(synonyms)
    return " ".join(expanded)


# ---------------------------------------------------------------------------
# Core Tools (Phase 1 & Phase 2+)
# ---------------------------------------------------------------------------

@_tool("read")
def read_legacy_file(file_path: str) -> str:
    """
    Accepts a file path string; returns raw legacy code string.
    Enforces security path traversal protection and max file size limits.
    """
    try:
        full_path = _resolve_workspace_path(file_path)

        if not is_path_in_workspace(full_path):
            return f"Error: Path '{file_path}' is outside authorized project workspace."

        if not os.path.isfile(full_path):
            return f"Error: File '{file_path}' does not exist."

        file_size = os.path.getsize(full_path)
        if file_size > MAX_READ_FILE_SIZE_BYTES:
            return (
                f"Error: File '{file_path}' is {file_size / (1024*1024):.2f}MB, "
                f"exceeding maximum allowed size limit of {MAX_READ_FILE_SIZE_BYTES / (1024*1024):.0f}MB."
            )

        with open(full_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file '{file_path}': {e!s}"


def _standards() -> "standards.Standards":
    return standards.load(WORKSPACE_ROOT)


def _rel(path: str) -> str:
    full = path if os.path.isabs(path) else os.path.join(WORKSPACE_ROOT, path)
    return os.path.relpath(full, WORKSPACE_ROOT).replace(os.sep, "/")


def _format_standard(rule: dict, score: float | None = None) -> str:
    relevance = f" (relevance: {round(score, 3)})" if score is not None else ""
    enforce = "blocks commits" if rule["enforce"] == "block" else "warning"
    return (f"### [standard: {rule['pack']}] `{rule['id']}` {rule['title']}{relevance}\n"
            f"_Severity {rule['severity']}, {enforce}; from `{rule['file']}`._\n\n{rule['guideline']}")


@_tool("read")
def query_rag_guidelines(query: str, category: str = "", n_results: int = 3) -> str:
    """
    Query the guidelines: the RAG vector database (built-in and ingested documents) plus
    the active modernization standards (rule packs) of this workspace and user.

    Args:
        query:     Natural language query or code snippet to match against guidelines.
        category:  Optional filter, e.g. 'python', 'typescript', 'security', 'clean_code',
                   or a custom category used by ingested documents or rule packs.
        n_results: Number of top results to return (default 3, max 10).
    """
    try:
        n_results = max(1, min(n_results, 10))
        expanded = f"{query} {expand_query(query)}"
        try:
            std_hits = _standards().search(expanded, n_results, category.strip().lower())
        except Exception:
            std_hits = []

        try:
            collection = ensure_seeded()
            where_filter = {"category": {"$eq": category.strip()}} if category.strip() else None
            results = collection.query(
                query_texts=[f"{query} {expand_query(query)}"],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

            scored = [(score, _format_standard(rule, score)) for score, rule in std_hits]
            for doc, meta, dist in zip(docs, metas, dists):
                score = round(ingest_rag_similarity(dist), 3) if dist is not None else 0.0
                title = meta.get("title", "Guideline")
                cat = meta.get("category", "")
                scored.append((score, f"### [{cat}] {title} (relevance: {score})\n\n{doc}"))
            if scored:
                header = f"## RAG Guidelines - Query: `{query}`"
                if category.strip():
                    header += f" | Category: `{category.strip()}`"
                sections = [text for _, text in sorted(scored, key=lambda t: -t[0])[:n_results]]
                return header + "\n\n" + "\n\n---\n\n".join(sections)
        except Exception:
            pass  # vector store unavailable: fall through to the built-in rules
        if std_hits:
            return f"## Organization Standards - Query: `{query}`\n\n" + "\n\n---\n\n".join(
                _format_standard(rule, score) for score, rule in std_hits)

        # Fallback when the vector store cannot be used
        default_rules = [
            "### Novus Rule #1: Explicit Typing & Modern Constructs",
            "- Always add type hints (Python 3.10+ syntax / TypeScript strict mode).",
            "- Replace obsolete libraries (e.g. urllib2 -> httpx/requests, ES5 var -> const/let).",
            "",
            "### Novus Rule #2: Security & Error Handling",
            "- Enforce strict parameter validation (Pydantic / Zod models).",
            "- Do not suppress raw exceptions without logging or structural recovery.",
            "",
            "### Novus Rule #3: Logical & Test Parity",
            "- Code transformations must strictly preserve logical parity.",
            "- Ensure deterministic behavior and full backward compatibility in function interfaces."
        ]
        return f"Query: '{query}'\nMatched Compliance Guidelines (fallback):\n" + "\n".join(default_rules)
    except Exception as e:
        return f"Error querying RAG guidelines: {e!s}"


@_tool("read")
def search_rag_by_id(document_id: str) -> str:
    """
    Retrieve a specific modernization guideline document directly by its ID.

    Args:
        document_id: Unique document ID (e.g. 'py-001', 'sec-001') or a standard's rule id.
    """
    try:
        if not document_id.strip():
            return "Error: document_id cannot be empty."
        rule = _standards().get(document_id.strip())
        if rule is not None:
            return _format_standard(rule)

        collection = ensure_seeded()

        res = collection.get(ids=[document_id.strip()], include=["documents", "metadatas"])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])

        if not docs:
            return f"Error: Document with ID '{document_id}' was not found in the RAG database."

        doc = docs[0]
        meta = metas[0] if metas else {}
        title = meta.get("title", "Guideline")
        cat = meta.get("category", "")

        return f"## RAG Guideline `{document_id}`: [{cat}] {title}\n\n{doc}"
    except Exception as e:
        return f"Error searching RAG document by ID '{document_id}': {e!s}"


@_tool("read")
def get_rag_stats() -> str:
    """
    Returns diagnostic statistics and status overview of the local RAG vector database.
    """
    try:
        chroma_dir = CHROMA_DIR
        try:
            collection = ensure_seeded()
            count = collection.count()
            all_res = collection.get(include=["metadatas"])
            metas = all_res.get("metadatas", [])
            categories = Counter(m.get("category", "unknown") for m in metas)

            cat_str = "\n".join(f"  - `{cat}`: {num} documents" for cat, num in categories.items())
            active = _standards()
            std_str = (f"- **Organization Standards**: {len(active.rules)} rule(s) from {len(active.packs)} pack(s) "
                       f"(see `list_modernization_standards`)\n")
            return (
                f"## NovusPipeline RAG Database Overview\n\n"
                f"- **Status**: Operational\n"
                f"- **Storage Path**: `{chroma_dir}`\n"
                f"- **Collection Name**: `{COLLECTION_NAME}`\n"
                f"- **Total Guidelines**: {count}\n"
                f"- **Category Breakdown**:\n{cat_str}\n"
                + std_str
            )
        except Exception as e:
            return f"RAG Status: Collection '{COLLECTION_NAME}' error: {e!s}"
    except Exception as e:
        return f"Error fetching RAG stats: {e!s}"


@_tool("write")
def ingest_rag_document(document_id: str, title: str, category: str, content: str) -> str:
    """
    Dynamically ingest a custom document into the RAG vector database.

    Args:
        document_id: Unique identifier for this document (e.g. 'py-custom-001').
        title:       Human-readable title for the guideline.
        category:    Category tag: 'python', 'typescript', 'security', 'clean_code', or your own
                     (e.g. 'fintech', 'healthcare'); lowercase letters, digits, '_' and '-'.
        content:     Full text content of the guideline or document to embed and store.

    Re-ingesting an existing id (including a built-in one such as 'py-002') replaces it, which
    is how guidelines are modified. Guidelines advise; to detect, fix or block violations,
    add a rule with `add_modernization_rule`.
    """
    try:
        if not document_id.strip():
            return "Error: document_id cannot be empty."
        if not content.strip():
            return "Error: content cannot be empty."
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", category.strip()):
            return (f"Error: category must be a lowercase slug (letters, digits, '_', '-'), e.g. 'python' or "
                    f"'fintech'. Got: '{category}'.")

        collection = get_collection(create=True)
        collection.upsert(
            documents=[content.strip()],
            metadatas=[{"title": title.strip(), "category": category.strip()}],
            ids=[document_id.strip()]
        )
        return (
            f"Successfully ingested document '{document_id}' into the RAG database.\n"
            f"Title: {title}\n"
            f"Category: {category}\n"
            f"Content length: {len(content)} characters."
        )
    except Exception as e:
        return f"Error ingesting document into RAG: {e!s}"


@_tool("destructive")
def remove_rag_document(document_id: str) -> str:
    """
    Deletes one guideline from the RAG vector database (built-in ones come back with
    `reset_rag_database`). Rules from rule packs are removed with `remove_modernization_rule`.

    Args:
        document_id: Id of the document to delete (e.g. 'py-custom-001').
    """
    try:
        collection = ensure_seeded()
        if not collection.get(ids=[document_id.strip()])["ids"]:
            return f"Error: Document with ID '{document_id}' was not found in the RAG database."
        collection.delete(ids=[document_id.strip()])
        return f"Removed document '{document_id}' from the RAG database."
    except Exception as e:
        return f"Error removing RAG document: {e}"


@_tool("destructive")
def reset_rag_database() -> str:
    """
    Programmatically resets and re-seeds the local RAG database with core enterprise guidelines.
    """
    try:
        from .ingest_rag import seed_rag_database
        seed_rag_database(reset=True)
        return "Successfully reset and re-seeded the RAG vector database."
    except Exception as e:
        return f"Error resetting RAG database: {e!s}"


ALLOWED_TEST_COMMANDS = (
    "pytest [args]",
    "python -m pytest|unittest [args]",
    "python --version",
    "node --test [args]",
    "npm test [args]  /  npm run test[:name] [args]",
    "npx jest|vitest|mocha [args]",
    "cargo test | go test | mvn test | gradle test | ./gradlew test  [args]",
)
_SHELL_CHAINING = (";", "&&", "||", "`", "$(")
# Windows runs .cmd/.bat files through cmd.exe, which interprets these even
# though we never pass shell=True.
_BATCH_METACHARACTERS = set("&|<>^%!\"")


def _validate_test_command(args: list[str]) -> str | None:
    """Error message if `args` is not an allowed test-runner invocation, else None."""
    exe = os.path.basename(args[0]).lower()
    for suffix in (".exe", ".cmd", ".bat"):
        exe = exe[:-len(suffix)] if exe.endswith(suffix) else exe
    rest = args[1:]
    # `python -c ...` / `node -e ...` would turn the "test runner" into arbitrary code execution.
    if exe in ("python", "python3", "py"):
        ok = (len(rest) >= 2 and rest[0] == "-m" and rest[1] in ("pytest", "unittest")) or rest in (["--version"], ["-V"])
    elif exe == "pytest":
        ok = True
    elif exe == "node":
        ok = bool(rest) and rest[0] == "--test"
    elif exe == "npm":
        ok = bool(rest) and (rest[0] in ("test", "t") or
                             (len(rest) >= 2 and rest[0] == "run" and re.fullmatch(r"test(:[\w.-]+)?", rest[1]) is not None))
    elif exe == "npx":
        ok = bool(rest) and rest[0] in ("jest", "vitest", "mocha")
    elif exe in ("cargo", "go"):
        ok = bool(rest) and rest[0] == "test"
    elif exe in ("mvn", "gradle", "gradlew"):
        ok = "test" in rest and not any(a.startswith("exec") for a in rest)
    else:
        ok = False
    if not ok:
        return f"Error: '{' '.join(args)}' is not an allowed test command. Allowed: {'; '.join(ALLOWED_TEST_COMMANDS)}."
    return None


def _python_interpreter() -> str:
    """NOVUS_PYTHON, else the workspace's own virtualenv, else this server's interpreter."""
    if os.environ.get("NOVUS_PYTHON"):
        return os.environ["NOVUS_PYTHON"]
    for venv in (".venv", "venv"):
        for rel in (("Scripts", "python.exe"), ("bin", "python")):
            candidate = os.path.join(WORKSPACE_ROOT, venv, *rel)
            if os.path.isfile(candidate):
                return candidate
    return sys.executable


def _execute_test_command(command: str, workspace: str | None = None,
                          disposable: bool = False) -> tuple[int | None, str]:
    """
    Runs an allowed test command in the sandbox (see sandbox.py) against `workspace`
    (default: the workspace root, copied first; pass disposable=True for a checkout that
    may be used directly, such as the pipeline's git worktree). Returns (exit code, or
    None if it could not run, and the output text including the isolation summary).
    """
    workspace = workspace or WORKSPACE_ROOT
    command = command.strip()
    if not command:
        return None, "Error: Empty command specified."
    for token in _SHELL_CHAINING:
        if token in command:
            return None, f"Error: Command contains dangerous shell chaining character '{token}'."
    try:
        args = shlex.split(command, posix=os.name != "nt")
    except ValueError as e:
        return None, f"Error: Could not parse command '{command}': {e}"
    if os.name == "nt":
        args = [a[1:-1] if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'" else a for a in args]

    error = _validate_test_command(args)
    if error:
        return None, error

    if os.path.basename(args[0]).lower().split(".")[0] in ("python", "python3", "py"):
        args[0] = _python_interpreter()
    else:
        resolved = shutil.which(args[0], path=os.pathsep.join([WORKSPACE_ROOT, os.environ.get("PATH", "")]))
        if resolved is None:
            return None, f"Error: '{args[0]}' was not found on PATH."
        if resolved.lower().endswith((".cmd", ".bat")) and any(set(a) & _BATCH_METACHARACTERS for a in args[1:]):
            return None, "Error: Arguments to a .cmd/.bat runner may not contain shell metacharacters (& | < > ^ % ! \")."
        args[0] = resolved

    try:
        res = sandbox.run(args, workspace, TEST_TIMEOUT_SECONDS, copy=not disposable)
    except OSError as e:
        return None, f"Error executing test command '{command}': {e}"
    isolation = "Sandbox: " + "; ".join(res.isolation)
    if res.timed_out:
        return None, (f"Error: Command '{command}' timed out after {TEST_TIMEOUT_SECONDS} seconds "
                      f"(process tree killed).\n{isolation}\n{_truncate(res.output)}")
    output = ["=== OUTPUT ===", res.output] if res.output else []
    output += [isolation, f"\nExit Code: {res.exit_code}"]
    return res.exit_code, _truncate("\n".join(output))


@_tool("write")
def run_local_tests(command: str) -> str:
    """
    Runs a test-suite command in a sandbox and returns its console output and exit code.
    The sandbox runs against a temporary copy of the project (your files cannot be
    modified) with a scrubbed environment (no tokens or keys; home/temp redirected) and
    memory/process limits; the whole process tree is killed on timeout. The output states
    which protections were active. Only test-runner invocations are allowed (e.g.
    `python -m pytest`, `python -m unittest discover`, `npm test`, `cargo test`); `python`
    resolves to the workspace virtualenv when one exists.
    """
    return _execute_test_command(command)[1]


def _pr_metadata_path(branch_name: str) -> str:
    # Inside the tool-owned index dir (never staged), so runs leave no stray files in the project.
    # Branch names may contain "/" (feature/x); keep the draft file a flat name.
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", branch_name)
    return os.path.join(WORKSPACE_ROOT, codebase_context.INDEX_DIRNAME, "prs", f"{safe}.md")


@_tool("destructive")
def create_git_migration_pr(branch_name: str, commit_message: str, pr_title: str, pr_description: str,
                            files: list[str] | None = None) -> str:
    """
    Creates/checks out `branch_name`, stages changes, commits, and writes draft PR metadata.

    Args:
        branch_name:    Branch to create or check out (validated with `git check-ref-format`).
        commit_message: Commit message.
        pr_title:       Draft PR title.
        pr_description: Draft PR body (Markdown).
        files:          Workspace paths to stage. If omitted, all changes are staged except
                        `.bak` modernization snapshots.
    """
    try:
        import git
        repo = git.Repo(WORKSPACE_ROOT)

        try:
            repo.git.check_ref_format("--branch", branch_name)
        except git.GitCommandError:
            return f"Error: '{branch_name}' is not a valid git branch name."

        to_stage: list[str] = []
        for path in files or []:
            full = _resolve_workspace_path(path)
            if not is_path_in_workspace(full):
                return f"Error: Path '{path}' is outside authorized project workspace."
            to_stage.append(os.path.relpath(full, repo.working_tree_dir))

        current_branch = repo.active_branch.name if not repo.head.is_detached else repo.head.commit.hexsha[:8]

        if branch_name not in [b.name for b in repo.branches]:
            repo.create_head(branch_name).checkout()
            status_msg = f"Created and checked out new branch '{branch_name}' (from '{current_branch}')."
        else:
            repo.branches[branch_name].checkout()
            status_msg = f"Checked out existing branch '{branch_name}'."

        if to_stage:
            repo.git.add("--", *to_stage)
        else:
            # Never commit this tool's own artifacts into the user's project.
            repo.git.add("-A", "--", ".", ":(exclude)*.bak", ":(exclude).novus_pr_*.md",
                         f":(exclude){codebase_context.INDEX_DIRNAME}")

        staged = repo.git.diff("--cached", "--name-only", "--", *to_stage) if to_stage \
            else repo.git.diff("--cached", "--name-only")
        if staged.strip():
            # Commit through the git CLI so the repository's hooks and signing config apply.
            # With explicit files, `-- <paths>` keeps any unrelated pre-staged changes out of the commit.
            repo.git.commit("-m", commit_message, *(["--", *to_stage] if to_stage else []))
            commit_info = f"Committed staged changes: {repo.head.commit.hexsha[:8]}"
        else:
            commit_info = "No staged changes to commit."

        pr_file_path = _write_pr_draft(branch_name, pr_title, pr_description)
        return f"{status_msg}\n{commit_info}\nDraft PR metadata written to '{pr_file_path}'."
    except Exception as e:
        return f"Error performing Git modernization operations: {e!s}"


def _write_pr_draft(branch_name: str, pr_title: str, pr_description: str) -> str:
    path = _pr_metadata_path(branch_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Draft PR: {pr_title}\n\n## Target Branch\n`{branch_name}`\n\n## Description\n{pr_description}\n\n"
                "---\n*Generated automatically by NovusPipeline Git Modernization Tool*\n")
    return path


# ---------------------------------------------------------------------------
# Phase 3 & Local LLM Integration Tools
# ---------------------------------------------------------------------------

@_tool("read", timeout=900)
def get_local_llm_status(probe: bool = False) -> str:
    """
    Status of the fine-tuned local LLM (LoRA adapter on unsloth/Qwen3.5-2B), served by a
    separate worker process. With `probe`, starts the worker and loads the model (can take a
    minute) to report the real device, load time and VRAM use.

    Args:
        probe: Load the model now instead of only reporting configuration.
    """
    try:
        info = local_llm.get_model_info(probe=probe)
        if info["adapter_path"] and not info["adapter_found"]:
            status_str = f"Adapter not found at NOVUS_LLM_PATH ({info['adapter_path']})"
        elif info.get("load_error"):
            status_str = f"Load failed: {info['load_error']}"
        elif probe:
            status_str = "Loaded and ready"
        elif info["worker_python"] is None:
            status_str = ('No LLM runtime (pip install "novuspipeline[llm]" or set NOVUS_LLM_PYTHON); '
                          "rule-based fallback in use")
        else:
            status_str = "Configured (loads on first use; call with probe=true to verify)"
        adapter = (f"`{info['adapter_path']}`" + (f" (LoRA r={info.get('lora_rank')}, alpha={info.get('lora_alpha')})"
                                                  if info.get("lora_rank") else "")
                   if info["adapter_path"] else "none (base model only)")
        lines = [
            "## Integrated Local LLM Status\n",
            f"- **Model**: `{info['model_name']}`",
            f"- **Base Model**: `{info['base_model']}`",
            f"- **Adapter**: {adapter}",
            f"- **Worker Interpreter**: `{info['worker_python']}` (log: `{info['worker_log']}`)",
            f"- **Status**: {status_str}",
        ]
        worker = info.get("worker") or {}
        if worker.get("loaded"):
            lines.append(f"- **Runtime**: `{worker.get('model_class')}` on {worker.get('device')} "
                         f"({worker.get('gpu', 'CPU')}), loaded in {worker.get('load_seconds')}s, "
                         f"{worker.get('vram_allocated_gb', 0)} GB VRAM")
        return "\n".join(lines)
    except Exception as e:
        return f"Error getting local LLM status: {e!s}"


def _codebase_context(file_path: str) -> tuple[dict | None, str]:
    """(file context from the codebase index, or None with a reason). Never raises."""
    try:
        index = codebase_context.get_index(WORKSPACE_ROOT)
        rel = index.normalize(_resolve_workspace_path(file_path))
        if rel is None:
            return None, "not an indexed Python file in this workspace"
        return index.file_context(rel), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _broken_api(new_code: str, api_surface: list[str]) -> list[str]:
    """Symbols other modules import from this file that `new_code` no longer provides."""
    return missing_names(new_code, api_surface)


@_tool("read", timeout=1800)
def generate_llm_modernization_proposal(file_path: str, rag_query: str = "") -> str:
    """
    Generates a modernized code proposal for a legacy file using the integrated fine-tuned
    Unsloth Qwen 3.5 2B local model, conditioned on retrieved RAG guidelines and on codebase
    context (which symbols other modules depend on, where the smells are, related tests).

    Args:
        file_path: Relative or absolute path to legacy source file.
        rag_query: Optional search query to customize RAG guideline retrieval.
    """
    try:
        code = read_legacy_file(file_path)
        if code.startswith("Error:"):
            return code

        smells = sorted({f["smell_id"] for f in LegacySmellDetector.scan_code(code, file_path)})
        search_term = rag_query.strip() or " ".join(
            [LegacySmellDetector._BY_ID[s]["rag_query"] for s in smells[:3]] or
            [f"Modernize legacy code {os.path.basename(file_path)}"])
        active, rel = _standards(), _rel(file_path)
        std_findings = active.scan(code, rel)
        rag_guidelines = "\n\n".join(filter(None, [active.prompt_block(std_findings),
                                                    query_rag_guidelines(search_term, n_results=2)]))

        ctx, _ = _codebase_context(file_path)
        context_md = codebase_context.format_file_context(ctx) if ctx else ""
        p = local_llm.propose_modernization(code, rag_guidelines, context_md,
                                            required_names=ctx["api_surface"] if ctx else [],
                                            standards_scan=lambda c: active.scan(c, rel))
        v = p["validation"]
        verdict = ("✅ Passes validation (parses, no new smells or standard violations, imported symbols kept)"
                   if v["valid"]
                   else "❌ Rejected: " + "; ".join(v["issues"]))
        details = [f"- **Engine**: `{p['engine']}`" + (f" (fallback reason: {p['error']})" if p["error"] else ""),
                   f"- **Validation**: {verdict}",
                   f"- **Smells resolved**: {', '.join(v.get('resolved', [])) or 'none'}"]
        details += [f"- ⚠️ {w}" for w in v.get("warnings", [])]
        if p["stats"]:
            s = p["stats"]
            details.append(f"- **Generation**: {s.get('new_tokens')} tokens in {s.get('seconds')}s "
                           f"({s.get('tokens_per_second')} tok/s)")

        return (
            f"## LLM Modernization Proposal for `{file_path}`\n"
            f"**Model**: `{local_llm.model_name()}`\n\n"
            f"### Retrieved Guidelines\n{rag_guidelines}\n\n"
            + (f"{context_md}\n\n" if context_md else "")
            + "### Proposal\n" + "\n".join(details) + "\n\n"
            + f"```python\n{p['code'] or ''}```\n\n"
            + (f"### Model Notes\n{p['raw']}" if p["engine"] == "rule-based" else "")
        )
    except Exception as e:
        return f"Error generating LLM modernization proposal for '{file_path}': {e!s}"


@_tool("read")
def analyze_legacy_codebase(file_path: str) -> str:
    """
    Audits a file for legacy code smells, cross-references RAG guidelines, cross-checks
    with the structural GNN, and adds codebase context (who depends on this file, which of
    its symbols they use, related tests, which functions hold the smells, and where else
    in the codebase the same smells occur).

    Args:
        file_path: Target relative or absolute file path to analyze.
    """
    try:
        code = read_legacy_file(file_path)
        if code.startswith("Error:"):
            return code

        findings = LegacySmellDetector.scan_code(code, file_path)
        ruff_findings = ruff_backend.scan_code(code, file_path, *_ruff_settings())
        learned = history_learner.get_learner(WORKSPACE_ROOT).scan(code, file_path) if file_path.endswith(".py") else []
        active = _standards()
        standard_findings = active.scan(code, _rel(file_path))
        report = [f"## Modernization Audit Report for `{file_path}`\n"]

        if findings or ruff_findings or learned or standard_findings:
            guideline_cache: dict = {}

            def guideline(f: dict) -> str:
                key = (f["rag_query"], f["category"])
                if key not in guideline_cache:
                    guideline_cache[key] = query_rag_guidelines(f["rag_query"], category=f["category"], n_results=1)
                return guideline_cache[key]

            by_smell: dict = {}
            by_severity = lambda f: codebase_context.SEVERITY_ORDER[f["severity"]]  # noqa: E731
            for f in sorted(standard_findings, key=by_severity) + findings + sorted(ruff_findings, key=by_severity) \
                    + learned:
                by_smell.setdefault(f["smell_id"], []).append(f)
            safe = sum(1 for f in ruff_findings + learned + standard_findings if f["fix"] == "safe")
            total = len(findings) + len(ruff_findings) + len(learned) + len(standard_findings)
            blocking = sum(1 for f in standard_findings if f["enforce"] == "block")
            report.append(f"Found **{total}** finding(s) of **{len(by_smell)}** kind(s) "
                          f"({len(standard_findings)} from your modernization standards"
                          + (f", {blocking} of them blocking" if blocking else "") + ", "
                          f"{len(findings)} built-in legacy rules, {len(ruff_findings)} Ruff, "
                          f"{len(learned)} learned from this repository's history"
                          + (f"; {safe} with a safe auto-fix" if safe else "") + "):\n")
            for smell_id, items in by_smell.items():
                first = items[0]
                fix = {"safe": " — safe auto-fix available", "unsafe": " — fix available, needs review"}.get(
                    first.get("fix") or "", "")
                report.append(f"### [{first['severity']}] `{smell_id}`: {first['name']} ({len(items)}x){fix}")
                shown = items[:10]
                report.append("```code\n" + "\n".join(f"L{f['line_number']}: {f['line_content']}" for f in shown)
                              + ("\n..." if len(items) > len(shown) else "") + "\n```")
                if first.get("url"):
                    report.append(f"Rule docs: {first['url']}")
                if first.get("source") == "standard":  # the organization's own rule is the guideline
                    enforce = ("**blocking**: the pipeline will not commit this file while it remains"
                               if first["enforce"] == "block" else "warning")
                    report.append(f"**Standard** `{first['pack']}` ({enforce}):\n{first['guideline']}\n\n---\n")
                    continue
                if first.get("source") == "learned":  # the team's own commits are the guideline
                    report.append(f"**Evidence from history**: {first['evidence']}\n\n---\n")
                    continue
                report.append(f"**Recommended Guideline**:\n{guideline(first)}\n\n---\n")
        else:
            report.append("✅ No legacy code smells detected by your standards, the built-in rules or Ruff.\n")

        if file_path.endswith(".py"):
            # The GNN is advisory: any failure here must not lose the rule-based audit above.
            try:
                probs = gnn_model.predict_smells(code)
                thresholds = gnn_model.get_thresholds()
                rule_based_ids = {f["smell_id"] for f in findings}
                report.append("### 🧠 Structural GNN Cross-Check")
                report.append(
                    "Graph Neural Network prediction from the AST graph (node types, fields and "
                    "identifiers; strings and comments are not seen), for comparison against the "
                    "rule-based findings above:\n"
                )
                for smell_id, prob in probs.items():
                    predicted = prob > thresholds[smell_id]
                    flag = "⚠️ Likely Present" if predicted else "OK"
                    agreement = "agrees" if predicted == (smell_id in rule_based_ids) else "DISAGREES"
                    report.append(
                        f"- `{smell_id}`: {prob:.3f} (threshold {thresholds[smell_id]:.2f}; "
                        f"{flag}, rule-engine {agreement})"
                    )
            except Exception as e:
                report.append(f"### 🧠 Structural GNN Cross-Check\nUnavailable: {type(e).__name__}: {e}")

            ctx, reason = _codebase_context(file_path)
            report.append("\n" + (codebase_context.format_file_context(ctx) if ctx
                                  else f"### 🧭 Codebase Context\nUnavailable: {reason}"))

        return "\n".join(report)
    except Exception as e:
        return f"Error analyzing legacy codebase for '{file_path}': {e!s}"


def _read_source(path: str) -> tuple[str, str]:
    """(text with "\\n" newlines, the file's newline style), so rewrites keep CRLF files CRLF."""
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        raw = f.read()
    return raw.replace("\r\n", "\n"), ("\r\n" if "\r\n" in raw else "\n")


def _write_source(path: str, text: str, newline: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\r\n", "\n").replace("\n", newline))


def _ruff_settings() -> tuple[list[str], list[str]]:
    active = _standards()
    return active.ruff_select, active.ruff_ignore


def _modernized_source(path: str, original: str, modernized_code: str = ""
                       ) -> tuple[str | None, list[str], str | None]:
    """(new code, change notes, error) for a file's content; nothing is written."""
    if modernized_code.strip():
        return modernized_code, ["Applied LLM / user-supplied modernized code."], None
    if path.endswith(".py"):
        new, notes = CodeModernizer.modernize_python(original)
        # Pure renames the team made consistently in its own history; still test-gated by the caller.
        new, learned_notes = history_learner.get_learner(WORKSPACE_ROOT).apply_fixes(new)
        # The organization's standards last, so their fixes win.
        new, standard_notes = _standards().apply_fixes(new, _rel(path))
        return new, notes + learned_notes + standard_notes, None
    if path.endswith((".ts", ".tsx", ".js", ".jsx")):
        new, notes = CodeModernizer.modernize_typescript(original)
        return new, notes, None
    return None, [], "Error: Automated modernization rules not implemented for this file extension"


@_tool("destructive")
def apply_code_modernization(file_path: str, modernized_code: str = "") -> str:
    """
    Phase 3 Tool: Applies modernization changes to a file safely. Creates a backup snapshot (.bak)
    before writing edits. If modernized_code is omitted, auto-applies rule-based refactorings.

    Args:
        file_path:        Target file path within workspace.
        modernized_code:  Optional custom refactored code string. If empty, uses automated rules.
    """
    try:
        full_path = _resolve_workspace_path(file_path)

        if not is_path_in_workspace(full_path):
            return f"Error: Path '{file_path}' is outside authorized project workspace."

        if not os.path.isfile(full_path):
            return f"Error: File '{file_path}' does not exist."

        original_code, newline = _read_source(full_path)

        new_code, applied_changes, error = _modernized_source(full_path, original_code, modernized_code)
        if error:
            return f"{error} ('{file_path}')"

        notes = "\n".join(f"  - {c}" for c in applied_changes)
        if new_code == original_code:
            return f"No applicable automated transformations for '{file_path}'; file left unchanged.\n{notes}"

        warning = ""
        if full_path.endswith(".py"):
            ctx, _ = _codebase_context(file_path)
            broken = _broken_api(new_code, ctx["api_surface"]) if ctx else []
            if broken:
                warning = (f"\n⚠️ WARNING: other modules import {', '.join(f'`{n}`' for n in broken)} from this "
                           f"file, which the new code no longer defines.")

        shutil.copyfile(full_path, full_path + ".bak")
        _write_source(full_path, new_code, newline)

        return (
            f"Successfully updated '{file_path}'.\n"
            f"Backup created at '{file_path}.bak'.\n"
            f"Applied Transformations:\n{notes}{warning}"
        )
    except Exception as e:
        return f"Error applying modernization to '{file_path}': {e!s}"


_PYTEST_AVAILABLE: dict = {}


def _default_test_command(ctx: dict | None) -> str:
    """Tests that import the file (from the import graph) if any, else the whole suite."""
    python = _python_interpreter()
    if python not in _PYTEST_AVAILABLE:
        try:
            _PYTEST_AVAILABLE[python] = subprocess.run(
                [python, "-c", "import pytest"], cwd=WORKSPACE_ROOT, capture_output=True, timeout=60
            ).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _PYTEST_AVAILABLE[python] = False
    tests = (ctx or {}).get("related_tests", [])
    if _PYTEST_AVAILABLE[python]:
        return "python -m pytest -q " + " ".join(shlex.quote(t) for t in tests) if tests else "python -m pytest -q"
    if tests:
        return "python -m unittest " + " ".join(t[:-3].replace("/", ".") for t in tests)
    return "python -m unittest discover"


@_tool("write", timeout=3600)
def run_autonomous_modernization_pipeline(
    file_path: str,
    test_command: str = "",
    branch_name: str = "auto-modernization-branch",
    use_llm: bool = False,
) -> str:
    """
    Codebase-aware, sandboxed modernization of one file. Everything happens in a disposable
    git worktree, so your working copy and current branch are never modified; the only
    lasting effect of a successful run is a commit on `branch_name`.
      1. Audits the file (your standards + rules + Ruff + GNN) and loads its codebase context.
      2. Rewrites it (rule engine, learned and standard fixes, or a validated LLM proposal
         with `use_llm`, which is given your standards and may not add violations).
      3. Rejects the change if it removes symbols other modules import, or if the file
         still violates a standard with `enforce = "block"`.
      4. Runs verification tests in the sandbox: `test_command`, or by default the tests
         that import this file (falling back to the whole suite).
      5. Exit code 0 -> commits only this file on `branch_name` (created at HEAD, or extended
         if it exists) and drafts a PR with blast radius and follow-up files.

    Args:
        file_path:    Target legacy file to modernize (must have no uncommitted changes).
        test_command: Optional test command; empty selects related tests automatically.
        branch_name:  Branch that receives the commit (default 'auto-modernization-branch').
        use_llm:      Try the local LLM first; its proposal is used only if it passes
                      validation (parses, no new smells, imported symbols kept). Tests gate both.
    """
    import tempfile

    import git

    full_path = _resolve_workspace_path(file_path)
    if not is_path_in_workspace(full_path):
        return f"Error: Path '{file_path}' is outside authorized project workspace."
    try:
        repo = git.Repo(WORKSPACE_ROOT)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return "Error: the pipeline needs a git repository (it commits the result on a branch)."
    try:
        repo.git.check_ref_format("--branch", branch_name)
    except git.GitCommandError:
        return f"Error: '{branch_name}' is not a valid git branch name."
    if not repo.head.is_detached and repo.active_branch.name == branch_name:
        return f"Error: '{branch_name}' is your checked-out branch; choose a separate branch for the pipeline."
    rel = os.path.relpath(full_path, repo.working_tree_dir).replace(os.sep, "/")
    if repo.git.status("--porcelain", "--", rel).strip():
        return (f"Error: '{file_path}' has uncommitted changes. Commit or stash them first: the pipeline works "
                "from committed code in a separate worktree so it never touches your working copy.")

    ctx, _ = _codebase_context(file_path)
    audit_res = analyze_legacy_codebase(file_path)
    start_point = branch_name if branch_name in [b.name for b in repo.branches] else "HEAD"
    wt_root = tempfile.mkdtemp(prefix="novus-worktree-")
    wt = os.path.join(wt_root, "wt")
    try:
        repo.git.worktree("add", "--detach", wt, start_point)
        wt_file = os.path.join(wt, rel)
        if not os.path.isfile(wt_file):
            return f"Error: '{rel}' does not exist at {start_point}."
        original, newline = _read_source(wt_file)

        engine_note = "Engine: rule-based modernizer."
        llm_code = ""
        if use_llm and rel.endswith(".py"):
            smells = sorted({x["smell_id"] for x in LegacySmellDetector.scan_code(original, rel)})
            active = _standards()
            rag = "\n\n".join(filter(None, [
                active.prompt_block(active.scan(original, rel)),
                query_rag_guidelines(" ".join(LegacySmellDetector._BY_ID[s]["rag_query"] for s in smells[:3])
                                     or "modernize legacy python", n_results=2)]))
            p = local_llm.propose_modernization(original, rag,
                                                codebase_context.format_file_context(ctx) if ctx else "",
                                                required_names=ctx["api_surface"] if ctx else [],
                                                standards_scan=lambda c: active.scan(c, rel))
            if p["engine"] != "rule-based" and p["validation"]["valid"] and p["code"] and p["code"] != original:
                llm_code, std_notes = active.apply_fixes(p["code"], rel)
                engine_note = (f"Engine: `{p['engine']}` (validated; resolved "
                               f"{', '.join(p['validation']['resolved']) or 'no rule-engine smells'}).")
            else:
                reason = p["error"] or "; ".join(p["validation"]["issues"]) or "no change proposed"
                engine_note = f"Engine: rule-based modernizer (LLM proposal not used: {reason})."

        new_code, notes, error = _modernized_source(rel, original, llm_code)
        if error:
            return f"Pipeline Aborted during modernization phase:\n{error} ('{file_path}')"
        if llm_code:
            notes += std_notes
        changes = "\n".join(f"  - {c}" for c in notes)
        blocking = _standards().blocking(new_code, rel)
        if blocking:
            listed = "\n".join(f"  - L{f['line_number']} `{f['smell_id']}` {f['name']}: "
                               f"{f['line_content']}" for f in blocking[:20])
            return (f"❌ Pipeline stopped: after modernization `{file_path}` still violates blocking standards, so "
                    f"nothing was committed. These need a manual change (or `use_llm=true`):\n{listed}\n\n"
                    f"Automated changes that were prepared:\n{changes or '  - none'}")
        if new_code == original:
            return f"Pipeline finished: nothing to modernize automatically in '{file_path}'.\n{changes}"
        broken = _broken_api(new_code, ctx["api_surface"]) if ctx and rel.endswith(".py") else []
        if broken:
            return (f"❌ Pipeline stopped: the modernized file would no longer provide "
                    f"{', '.join(f'`{n}`' for n in broken)}, which other modules import. Nothing was changed.")
        _write_source(wt_file, new_code, newline)

        sandbox.link_dependency_dirs(WORKSPACE_ROOT, wt)
        command = test_command.strip() or _default_test_command(ctx)
        exit_code, test_res = _execute_test_command(command, workspace=wt, disposable=True)
        if exit_code != 0:
            reason = "could not be run" if exit_code is None else f"failed (exit code {exit_code})"
            return (f"❌ Autonomous Modernization Pipeline: verification `{command}` {reason}. "
                    f"The change was discarded; your working copy was never modified.\n\nTest Output:\n{test_res}")

        wt_repo = git.Repo(wt)
        wt_repo.git.add("--", rel)
        wt_repo.git.commit("-m", f"refactor: autonomous modernization of {os.path.basename(rel)}", "--", rel)
        sha = wt_repo.head.commit.hexsha
        wt_repo.git.branch("-f", branch_name, sha)

        impact = codebase_context.format_file_context(ctx) if ctx else "_Codebase context unavailable._"
        follow_ups = (ctx or {}).get("same_smell_elsewhere", [])
        pr_path = _write_pr_draft(
            branch_name, f"Autonomous Modernization: {os.path.basename(rel)}",
            f"### Modernization Audit\n{audit_res}\n\n### Applied Changes\n{engine_note}\n{changes}\n\n"
            f"### Test Verification\nPassed (exit code 0): `{command}`\n\n### Impact\n{impact}")
        follow_up_md = ("\n\n5. Follow-up candidates (same smells elsewhere):\n"
                        + "\n".join(f"   - `{p}`" for p in follow_ups[:10])) if follow_ups else ""
        return (
            f"🎉 Autonomous Modernization Pipeline Completed Successfully!\n\n"
            f"1. Audit Findings & RAG Match:\n{audit_res[:300]}...\n\n"
            f"2. Transformations Applied ({engine_note}):\n{changes}\n\n"
            f"3. Verification (`{command}`):\n{test_res}\n\n"
            f"4. Git: committed `{sha[:8]}` on `{branch_name}` (from {start_point}) in an isolated worktree; "
            f"your working copy and current branch are unchanged. Draft PR: '{pr_path}'.{follow_up_md}"
        )
    except Exception as e:
        return f"Error executing autonomous modernization pipeline for '{file_path}': {type(e).__name__}: {e}"
    finally:
        sandbox.unlink_dependency_dirs(wt)  # before removal: never delete through the node_modules link
        with contextlib.suppress(git.GitCommandError):
            repo.git.worktree("remove", "--force", wt)
        shutil.rmtree(wt_root, ignore_errors=True)
        with contextlib.suppress(git.GitCommandError):
            repo.git.worktree("prune")


# ---------------------------------------------------------------------------
# Phase 4: Git PR & Modernization Reporting Tools
# ---------------------------------------------------------------------------

@_tool("write", timeout=1800)
def format_modernization_report(
    file_path: str,
    branch_name: str = "auto-modernization-branch",
    test_command: str = ""
) -> str:
    """
    Phase 4 Tool: Compiles audit findings, codebase impact, test logs, and Git PR
    metadata into a Markdown report saved under 'reports/'.

    Args:
        file_path:    Target modernized source file path.
        branch_name:  Git branch associated with this migration.
        test_command: Test command to verify the build; empty selects the tests that import the file.
    """
    try:
        audit_res = analyze_legacy_codebase(file_path)
        ctx, _reason = _codebase_context(file_path)
        command = test_command.strip() or _default_test_command(ctx)
        exit_code, test_res = _execute_test_command(command)
        mod_details = (f"Rule-engine / LLM modernization of `{file_path}`. "
                       f"Verification `{command}` exit code: {exit_code}.")

        report_md = ModernizationReporter.generate_report(
            file_path=file_path,
            audit_summary=audit_res,
            modernization_details=mod_details,
            test_output=test_res,
            branch_name=branch_name,
            pr_draft_file=os.path.basename(_pr_metadata_path(branch_name)),
        )

        saved_file = ModernizationReporter.save_report_artifact(WORKSPACE_ROOT, branch_name, report_md)

        return (
            f"Successfully generated and saved Modernization Report artifact.\n"
            f"Report Path: `{saved_file}`\n\n"
            f"Report Summary Preview:\n{report_md[:400]}..."
        )
    except Exception as e:
        return f"Error formatting modernization report for '{file_path}': {e!s}"


@_tool("destructive")
def finalize_git_migration_pr(
    branch_name: str = "auto-modernization-branch",
    base_branch: str = "main",
    pr_title: str = "Refactor: Enterprise Codebase Modernization",
    pr_description: str = "Automated modernization pipeline run complete."
) -> str:
    """
    Phase 4 Tool: Finalizes Git PR metadata, verifies staged repository state,
    creates draft PR markdown files, and formats full Pull Request workflow instructions.

    Args:
        branch_name:    Target feature/migration branch name.
        base_branch:    Target base branch to merge into (default 'main').
        pr_title:       Custom PR title.
        pr_description: Custom PR summary description.
    """
    try:
        git_res = create_git_migration_pr(
            branch_name=branch_name,
            commit_message=f"refactor: finalize enterprise modernization on {branch_name}",
            pr_title=pr_title,
            pr_description=pr_description
        )
        if git_res.startswith("Error"):
            return git_res

        pr_file_path = _pr_metadata_path(branch_name)
        report_file = ModernizationReporter.report_path(WORKSPACE_ROOT, branch_name)

        return (
            f"## 🏁 NovusPipeline Git PR Finalization\n\n"
            f"{git_res}\n\n"
            f"### PR Artifact Checklist\n"
            f"- **Base Branch**: `{base_branch}`\n"
            f"- **Target Branch**: `{branch_name}`\n"
            f"- **PR Summary Metadata**: `{pr_file_path}`\n"
            f"- **Full Modernization Report**: `{report_file}`\n\n"
            f"### GitHub / GitLab Command\n"
            f"```shell\n"
            f"git push origin {shlex.quote(branch_name)}\n"
            f"gh pr create --title {shlex.quote(pr_title)} --body-file {shlex.quote(pr_file_path)} "
            f"--base {shlex.quote(base_branch)}\n"
            f"```"
        )
    except Exception as e:
        return f"Error finalizing Git migration PR for branch '{branch_name}': {e!s}"


# ---------------------------------------------------------------------------
# Phase 5: GNN Structural Code-Smell Classifier
# ---------------------------------------------------------------------------

@_tool("read")
def get_gnn_model_status() -> str:
    """
    Phase 5 Tool: Returns status/metadata for the trained Graph Neural Network
    structural code-smell classifier (trained via `train_gnn.py`).
    """
    try:
        status = gnn_model.get_gnn_status()
        meta = status.get("metadata", {})
        lines = [
            "## GNN Structural Smell Classifier Status\n",
            f"- **Checkpoint Path**: `{status['checkpoint_path']}`",
            f"- **Checkpoint Trained**: {status['checkpoint_exists']}",
            f"- **Torch Available**: {status['torch_available']}",
            f"- **Supported Labels**: {', '.join(status['labels'])}",
        ]
        if status.get("torch_error"):
            lines.append(f"- **Torch Error**: {status['torch_error']}")
        if status.get("load_error"):
            lines.append(f"- **Load Error**: {status['load_error']}")
        if meta:
            test = meta.get("test", {})
            thresholds = meta.get("thresholds", {})
            lines.append(f"- **Trained At**: {meta.get('trained_at', 'unknown')} (seed {meta.get('seed', 'n/a')})")
            lines.append(f"- **Architecture**: `{meta.get('config', {})}`")
            lines.append(f"- **Held-out Test Macro-F1**: {test.get('macro_f1', 'n/a')} "
                         f"(exact match {test.get('exact_match', 'n/a')})")
            for label, f1 in test.get("per_label_f1", {}).items():
                lines.append(f"  - `{label}`: F1 {f1}, threshold {thresholds.get(label, 0.5)}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error getting GNN model status: {e!s}"


@_tool("read")
def analyze_code_structure_gnn(file_path: str) -> str:
    """
    Phase 5 Tool: Runs the trained Graph Neural Network over a Python file's AST
    graph to predict structural code smells, independent of the regex/text-based
    detector in `analyze_legacy_codebase`. Python files only (requires a
    syntactically valid AST).

    Args:
        file_path: Target relative or absolute Python file path to analyze.
    """
    try:
        if not file_path.endswith(".py"):
            return f"Error: GNN structural analysis currently only supports Python ('.py') files, got '{file_path}'."

        code = read_legacy_file(file_path)
        if code.startswith("Error:"):
            return code

        probs = gnn_model.predict_smells(code)
        thresholds = gnn_model.get_thresholds()

        lines = [f"## GNN Structural Analysis for `{file_path}`\n"]
        for smell_id, prob in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
            flag = "⚠️ Likely Present" if prob > thresholds[smell_id] else "OK"
            lines.append(f"- `{smell_id}`: **{prob:.3f}** (threshold {thresholds[smell_id]:.2f}; {flag})")
        lines.append(
            "\n_Predicted from the AST graph (node types, fields and identifiers; strings and comments "
            "are not seen). File-level only, no line localization. Treat as a complementary signal to "
            "`analyze_legacy_codebase`, not a replacement._"
        )
        return "\n".join(lines)
    except SyntaxError as e:
        return f"Error: '{file_path}' is not valid Python 3 syntax, cannot build AST graph: {e!s}"
    except ValueError as e:
        return f"Error: '{file_path}' cannot be analyzed by the GNN: {e!s}"
    except RuntimeError as e:
        return f"Error: GNN model unavailable: {e!s}"
    except Exception as e:
        return f"Error running GNN structural analysis on '{file_path}': {e!s}"


# ---------------------------------------------------------------------------
# Phase 6: Codebase Context (GNN + import graph)
# ---------------------------------------------------------------------------

@_tool("write")
def index_codebase(force_rebuild: bool = False) -> str:
    """
    Builds or incrementally refreshes the codebase context index: every Python file's
    import graph, rule-engine findings, and per-function GNN smell predictions and
    structural embeddings. Other tools refresh it automatically; call this to force a
    full rebuild (e.g. after retraining the GNN) or to see indexing stats.

    Args:
        force_rebuild: Re-analyze every file instead of only new/changed ones.
    """
    try:
        index = codebase_context.get_index(WORKSPACE_ROOT, refresh=False)
        stats = index.refresh(force=force_rebuild)
        gnn = "enabled" if stats["gnn"] else f"unavailable ({index.gnn_error})"
        return (
            f"## Codebase Index for `{WORKSPACE_ROOT}`\n\n"
            f"- **Python files**: {stats['files']} ({stats['reanalyzed']} re-analyzed, {stats['removed']} removed)\n"
            f"- **Functions indexed**: {stats['functions']}\n"
            f"- **GNN signals**: {gnn}\n"
            f"- **Time**: {stats['seconds']}s\n"
            f"- **Stored at**: `{index.index_path}`"
        )
    except Exception as e:
        return f"Error indexing codebase: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Modernization standards (rule packs)
# ---------------------------------------------------------------------------

@_tool("read")
def list_modernization_standards() -> str:
    """
    Lists the modernization standards that apply to this workspace: rule packs from your user
    data directory (personal) and from `.novus/standards/` (project, shared by the team), every
    rule with its detection, automatic fix and enforcement, problems in pack files, and the
    built-in example packs you can install with `install_standard_pack`.
    """
    try:
        active = _standards()
        lines = [f"## Modernization Standards: {len(active.rules)} rule(s) from {len(active.packs)} pack(s)\n"]
        for pack in active.packs:
            lines.append(f"- **{pack['name']}** ({pack['scope']}, `{pack['path']}`): {pack['rules']} rule(s)"
                         + (f" - {pack['description']}" if pack["description"] else ""))
        if active.rules:
            lines.append("\n### Rules")
        for r in sorted(active.rules, key=lambda r: (codebase_context.SEVERITY_ORDER[r["severity"]], r["id"])):
            detect = "; ".join(filter(None, [
                f"calls {', '.join(r['calls'])}" + (f" without `{r['require_keyword']}=`" if r["require_keyword"] else "")
                if r["calls"] else "",
                f"imports {', '.join(r['imports'])}" if r["imports"] else "",
                f"pattern `{r['pattern']}`" if r["pattern"] else ""]))
            fix = ("replace call with `" + r["replace_call"] + "`" if r["replace_call"] else
                   "replace import with `" + r["replace_import"] + "`" if r["replace_import"] else
                   "add `" + r["add_keyword"] + "`" if r["add_keyword"] else
                   "regex replacement" if r["replacement"] else "none (manual or LLM)")
            scope = (f"; only {', '.join(r['paths'])}" if r["paths"] else "") + \
                (f"; except {', '.join(r['exclude'])}" if r["exclude"] else "")
            lines.append(f"- [{r['severity']}] `{r['id']}` {r['title']} ({r['enforce']}, pack {r['pack']}): "
                         f"detects {detect}; auto-fix: {fix}{scope}")
        if active.ruff_select or active.ruff_ignore:
            lines.append(f"\n**Ruff tuning**: select {active.ruff_select or '-'}, ignore {active.ruff_ignore or '-'}")
        if active.errors:
            lines.append("\n### ⚠️ Problems (these rules are not active)")
            lines += [f"- {e}" for e in active.errors]
        installed = {p["name"] for p in active.packs}
        available = [n for n in standards.builtin_packs() if n not in installed]
        if available:
            lines.append(f"\n**Built-in example packs** (install with `install_standard_pack`): "
                         f"{', '.join(f'`{n}`' for n in available)}")
        if not active.rules:
            lines.append("\nNo standards yet: add rules with `add_modernization_rule`, write a pack in "
                         "`.novus/standards/*.toml`, or install a built-in pack.")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing modernization standards: {type(e).__name__}: {e}"


@_tool("write")
def add_modernization_rule(
    rule_id: str,
    title: str,
    guideline: str,
    severity: str = "MEDIUM",
    calls: list[str] | None = None,
    imports: list[str] | None = None,
    pattern: str = "",
    require_keyword: str = "",
    replace_call: str = "",
    replace_import: str = "",
    add_keyword: str = "",
    replacement: str = "",
    paths: list[str] | None = None,
    exclude: list[str] | None = None,
    enforce: str = "warn",
    category: str = "python",
    scope: str = "project",
) -> str:
    """
    Adds (or replaces) a modernization rule, so audits, the LLM and the pipeline hold code to
    your industry's or organization's standard. Stored in `.novus/standards/custom.toml`
    (scope 'project', shared via git) or in your user data directory (scope 'user', applies to
    all your projects). Takes effect immediately.

    Detection needs at least one of `calls`, `imports`, `pattern`. Examples:
      - every HTTP call has a timeout: calls=["requests.get", "requests.post"],
        require_keyword="timeout", add_keyword="timeout=30"
      - no pickle in the payments service: imports=["pickle"], paths=["src/payments/**"],
        enforce="block"
      - internal API migration: calls=["legacy_db.query"], replace_call="db.query"

    Args:
        rule_id:         Unique id, e.g. 'ACME-001' (letters, digits, '-', '_', '.').
        title:           One-line name of the standard.
        guideline:       Markdown explaining the standard and the preferred alternative; shown in
                         audits, returned by query_rag_guidelines, and given to the LLM.
        severity:        CRITICAL, HIGH, MEDIUM or LOW.
        calls:           Dotted call targets as imported, e.g. ["yaml.load"]; aliases are resolved.
        imports:         Modules that must not be imported (submodules included).
        pattern:         Regex matched per line, with strings and comments blanked out.
        require_keyword: With `calls`: only flag calls missing this keyword argument.
        replace_call:    Auto-fix: new dotted call target (an import is added if needed).
        replace_import:  Auto-fix: module that replaces the banned import.
        add_keyword:     Auto-fix for `require_keyword`, e.g. "timeout=30".
        replacement:     Auto-fix: regex replacement for `pattern` (lines with strings/comments are skipped).
        paths:           Only apply to files matching these globs (e.g. ["src/payments/**"]).
        exclude:         Skip files matching these globs (e.g. ["tests/**"]).
        enforce:         'warn' (report; LLM proposals may not add violations) or 'block' (the
                         pipeline will not commit a file that still violates the rule).
        category:        Guideline category for query_rag_guidelines filters (default 'python';
                         'typescript'/'javascript' makes `pattern` rules apply to TS/JS files).
        scope:           'project' (default) or 'user'.
    """
    try:
        if scope not in ("project", "user"):
            return "Error: scope must be 'project' or 'user'."
        raw = {"id": rule_id, "title": title, "guideline": guideline, "severity": severity, "calls": calls or [],
               "imports": imports or [], "pattern": pattern, "require_keyword": require_keyword,
               "replace_call": replace_call, "replace_import": replace_import, "add_keyword": add_keyword,
               "replacement": replacement, "paths": paths or [], "exclude": exclude or [], "enforce": enforce,
               "category": category}
        path, problems = standards.upsert_rule(WORKSPACE_ROOT, raw, scope)
        if problems:
            return "Error: rule not saved: " + "; ".join(problems)
        try:
            index = codebase_context.get_index(WORKSPACE_ROOT, refresh=False)
            index.refresh()  # the standards signature changed: findings are recomputed
            hits = sum(1 for rec in index.files.values()
                       if any(f["smell_id"] == rule_id.strip() for f in rec["rule_findings"]))
            usage = f"\nCurrently violated in {hits} indexed file(s)."
        except Exception:
            usage = ""
        return f"Saved rule `{rule_id.strip()}` to `{path}` ({scope} standards).{usage}\n\n" + \
            _format_standard(_standards().get(rule_id.strip()))
    except Exception as e:
        return f"Error adding modernization rule: {type(e).__name__}: {e}"


@_tool("destructive")
def remove_modernization_rule(rule_id: str, scope: str = "project") -> str:
    """
    Removes a rule added with `add_modernization_rule` (from the scope's `custom.toml`). Rules
    in other pack files are edited in those files.

    Args:
        rule_id: Id of the rule to remove.
        scope:   'project' (default) or 'user'.
    """
    try:
        if scope not in ("project", "user"):
            return "Error: scope must be 'project' or 'user'."
        ok, detail = standards.remove_rule(WORKSPACE_ROOT, rule_id.strip(), scope)
        return f"Removed rule `{rule_id}` from `{detail}`." if ok else f"Error: {detail}"
    except Exception as e:
        return f"Error removing modernization rule: {type(e).__name__}: {e}"


@_tool("write")
def install_standard_pack(name: str, scope: str = "project") -> str:
    """
    Installs a built-in example rule pack (see `list_modernization_standards`) by copying it to
    `.novus/standards/<name>.toml` (or the user standards directory), where it can be edited.

    Args:
        name:  Built-in pack name, e.g. 'secure-defaults' or 'python-312-ready'.
        scope: 'project' (default) or 'user'.
    """
    try:
        if scope not in ("project", "user"):
            return "Error: scope must be 'project' or 'user'."
        target, error = standards.install_pack(WORKSPACE_ROOT, name.strip(), scope)
        if error:
            return f"Error: {error}"
        active = _standards()
        count = sum(1 for r in active.rules if os.path.abspath(r["file"]) == os.path.abspath(target))
        return f"Installed `{name}` ({count} rules) at `{target}`. Edit that file to adapt it to your standards."
    except Exception as e:
        return f"Error installing standard pack: {type(e).__name__}: {e}"


@_tool("write", timeout=3600)
def learn_from_history(max_commits: int = 2000, min_support: int = history_learner.DEFAULT_MIN_SUPPORT) -> str:
    """
    Mines this repository's git history for smells the team has already been fixing:
    calls and imports that were repeatedly replaced the same way across separate commits
    (e.g. `self.assertEquals` -> `self.assertEqual` in 7 commits). Each learned rule comes
    with its evidence (support, consistency, example commits). Remaining uses then show up
    in audits, file context and the overview; pure renames with near-unanimous history are
    applied by the modernization pipeline, which still runs the tests in the sandbox.
    Mining is incremental and also runs in the background after each index refresh; call
    this to mine now. Only writes `.novus_index/learned.json`.

    Args:
        max_commits: Maximum number of not-yet-mined commits to analyze (newest first).
        min_support: Minimum number of distinct commits that must show a replacement (at least 2).
    """
    try:
        learner = history_learner.get_learner(WORKSPACE_ROOT)
        stats = learner.refresh(max_new_commits=max(1, min(max_commits, 20000)))
        if stats.get("error"):
            return f"Error: cannot learn from history: {stats['error']}"
        rules = learner.rules(min_support=max(2, min_support))
        lines = [f"## Learned From History: {stats['total_commits']} commit(s) mined "
                 f"({stats['mined']} new, {stats['seconds']}s), {len(rules)} rule(s)\n"]
        if not rules:
            lines.append("No replacement was made consistently in enough separate commits yet.")
        for rule in rules[:50]:
            fix = "auto-fixable (pure rename)" if learner.auto_fixable(rule) else "suggestion only"
            lines.append(f"- `{rule['id']}` [{rule['kind']}] `{rule['old']}` -> `{rule['new']}`: "
                         f"{rule['support']} commits, consistency {rule['consistency']:.0%}, {fix}")
            for e in rule["examples"][:2]:
                lines.append(f"  - {e['commit'][:7]} \"{e['message']}\" (`{e['file']}`, {e['qualname']})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error learning from history: {type(e).__name__}: {e}"


@_tool("read")
def get_codebase_overview(top_n: int = 10) -> str:
    """
    Codebase-wide modernization overview: files ranked by risk (built-in rules + Ruff + GNN +
    codebase-level smells, severity-weighted) with their blast radius, codebase-level smells
    (import cycles, dead code, duplicated logic, untested hotspots, god modules, deprecated
    APIs, unresolvable dynamic imports), the most depended-on modules, and the smell
    distribution. Use it to decide what to modernize first.

    Args:
        top_n: Number of hotspot files / hub modules to list (default 10, max 50).
    """
    try:
        top_n = max(1, min(top_n, 50))
        index = codebase_context.get_index(WORKSPACE_ROOT)
        files = index.files
        if not files:
            return f"No Python files found under `{WORKSPACE_ROOT}`."

        lines = [f"## Codebase Overview: {len(files)} Python files, "
                 f"{sum(len(r['functions']) for r in files.values())} functions\n"]
        if index.gnn_error:
            lines.append(f"_GNN signals unavailable: {index.gnn_error}_")
        ruff_ok, ruff_note = ruff_backend.available()
        if not ruff_ok:
            lines.append(f"_Ruff findings unavailable: {ruff_note}_")

        lines.append("\n### 🔥 Modernization Hotspots (risk = severity-weighted smells; [source] of each signal)")
        for h in index.hotspots(top_n):
            lines.append(f"- `{h['path']}` risk **{h['risk']:.1f}**, imported by {h['dependents']} module(s)"
                         f"{' (test)' if h['is_test'] else ''}: {codebase_context.summarize_smells(h['smells'])}")

        context = index.context_smells()
        by_kind: dict = {}
        for rel, smells in context.items():
            for s in smells:
                by_kind.setdefault(s["smell_id"], []).append((rel, s))
        if by_kind:
            lines.append("\n### 🏗️ Codebase-Level Smells (need the whole codebase to see)")
            for smell_id in codebase_context.CONTEXT_SMELLS:
                items = by_kind.get(smell_id, [])
                if not items:
                    continue
                name, severity = codebase_context.CONTEXT_SMELLS[smell_id]
                lines.append(f"- [{severity}] `{smell_id}` {name}: {len(items)} occurrence(s)")
                for rel, s in items[:3]:
                    lines.append(f"  - `{rel}`{' L' + str(s['line']) if s.get('line') else ''}: {s['detail']}")

        learned = index.learned_smells()
        rules = history_learner.get_learner(WORKSPACE_ROOT).rules()
        if rules:
            lines.append("\n### 🎓 Learned From This Repository's History (replacements the team made repeatedly)")
            for rule in rules[:top_n]:
                affected = [rel for rel, smells in learned.items() if any(s["smell_id"] == rule["id"] for s in smells)]
                fix = "auto-fixable" if history_learner.HistoryLearner.auto_fixable(rule) else "suggestion"
                lines.append(f"- `{rule['id']}` `{rule['old']}` -> `{rule['new']}`: {rule['support']} commits, "
                             f"consistency {rule['consistency']:.0%}, {fix}; still used in {len(affected)} file(s)"
                             + (f", e.g. `{affected[0]}`" if affected else ""))

        fan_in = Counter()
        for rel in files:
            fan_in.update(index.dependencies(rel).keys())
        lines.append("\n### 🕸️ Most Depended-On Modules (changes here have the widest blast radius)")
        for rel, count in fan_in.most_common(top_n):
            lines.append(f"- `{rel}`: imported by {count} module(s)")

        distribution, names = Counter(), {}
        for rec in files.values():
            distribution.update({f["smell_id"] for f in rec["rule_findings"]})
            names.update({f["smell_id"]: f.get("name", "") for f in rec["rule_findings"]})
        if distribution:
            lines.append("\n### 📊 Files Affected per Finding (built-in rules + Ruff)")
            for smell_id, count in distribution.most_common(top_n * 2):
                lines.append(f"- `{smell_id}` {names[smell_id]}: {count} file(s)")

        broken = [rel for rel, rec in files.items() if rec["parse_error"]]
        if broken:
            lines.append(f"\n### ⚠️ Not valid Python 3 ({len(broken)} file(s), rule engine only)")
            lines.extend(f"- `{rel}`" for rel in broken[:top_n])
        return "\n".join(lines)
    except Exception as e:
        return f"Error building codebase overview: {type(e).__name__}: {e}"


@_tool("read")
def get_file_context(file_path: str) -> str:
    """
    Everything the codebase knows about one Python file: which modules import it and which of
    its symbols they use (keep those stable), related tests, its own dependencies, which
    functions hold smells (GNN per-function + rule engine), and other files with the same smells.

    Args:
        file_path: Workspace-relative or absolute path of a Python file.
    """
    full = _resolve_workspace_path(file_path)
    if not is_path_in_workspace(full):
        return f"Error: Path '{file_path}' is outside authorized project workspace."
    ctx, reason = _codebase_context(file_path)
    if ctx is None:
        return f"Error: No codebase context for '{file_path}': {reason}."
    return codebase_context.format_file_context(ctx, limit=25)


@_tool("read")
def find_similar_code(file_path: str, function_name: str = "", n_results: int = 5) -> str:
    """
    Finds structurally similar code elsewhere in the codebase using GNN graph embeddings:
    functions similar to `function_name` in `file_path` (use `Class.method` for methods), or
    files similar to the whole file if no function is given. Useful to locate other copies of
    a legacy pattern after fixing one, or duplicated logic to consolidate.

    Args:
        file_path:     Workspace-relative or absolute path of a Python file.
        function_name: Optional function/method qualname within the file.
        n_results:     Number of matches (default 5, max 25).
    """
    try:
        full = _resolve_workspace_path(file_path)
        if not is_path_in_workspace(full):
            return f"Error: Path '{file_path}' is outside authorized project workspace."
        index = codebase_context.get_index(WORKSPACE_ROOT)
        rel = index.normalize(full)
        if rel is None:
            return f"Error: '{file_path}' is not an indexed Python file."
        if index.gnn_error:
            return f"Error: structural similarity needs the GNN: {index.gnn_error}"
        if function_name and not any(f["qualname"] == function_name for f in index.files[rel]["functions"]):
            available = ", ".join(f["qualname"] for f in index.files[rel]["functions"][:30])
            return f"Error: no function `{function_name}` in `{rel}`. Available: {available}"

        matches = index.similar(rel, function_name, max(1, min(n_results, 25)))
        target = f"`{rel}::{function_name}`" if function_name else f"`{rel}`"
        if not matches:
            return f"No structurally comparable code found for {target}."
        lines = [f"## Code structurally similar to {target}",
                 "_Cosine similarity of mean-centered GNN embeddings; ~1.0 = near-duplicate structure._\n"]
        for m in matches:
            where = f"`{m['path']}::{m['qualname']}` (L{m['lineno']})" if function_name else f"`{m['path']}`"
            shared = f" — shares smells: {', '.join(m['shared_smells'])}" if m.get("shared_smells") else ""
            lines.append(f"- {where}: **{m['similarity']:.3f}**{shared}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error finding similar code: {type(e).__name__}: {e}"


@mcp.prompt()
def modernization_workflow(file_path: str = "") -> str:
    """Step-by-step, codebase-aware modernization workflow for an agent using these tools."""
    target = f"`{file_path}`" if file_path else "the highest-risk file"
    return (
        f"Modernize {target} in this codebase safely:\n"
        "0. Call `list_modernization_standards`. If the user has industry or organization standards that "
        "are not there yet, add them first with `add_modernization_rule` (or `install_standard_pack`), so "
        "every later step holds the code to them.\n"
        "1. If no file is given, call `get_codebase_overview` and pick the top hotspot that is not a test.\n"
        "2. Call `get_file_context` to see its dependents, the public API they use (must not change), "
        "related tests, and which functions hold the smells.\n"
        "3. Call `analyze_legacy_codebase` for the findings (standards, rules, Ruff, learned) and guidelines.\n"
        "4. Optionally call `generate_llm_modernization_proposal`; only proposals marked "
        "'Passes validation' are safe to apply.\n"
        "5. Call `run_autonomous_modernization_pipeline` (set `use_llm` to try the LLM first). It runs the "
        "related tests, rolls back on failure, API breakage or remaining blocking-standard violations, and "
        "commits only that file.\n"
        "6. Review the follow-up candidates it lists and repeat for them; use `find_similar_code` to find "
        "other copies of the same pattern."
    )


def main(argv: list[str] | None = None) -> None:
    """Console entry point: `novuspipeline [--workspace DIR] [--transport stdio|http]`."""
    import argparse

    from . import __version__

    global WORKSPACE_ROOT
    parser = argparse.ArgumentParser(prog="novuspipeline", description="NovusPipeline MCP server")
    parser.add_argument("--workspace", help="codebase to modernize (default: NOVUS_WORKSPACE_ROOT or current directory)")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                        help="stdio for MCP clients that launch the server (default); http to serve over the network")
    parser.add_argument("--host", default="127.0.0.1", help="http only; the server has no authentication, so keep it local")
    parser.add_argument("--port", type=int, default=8000, help="http only")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    if args.workspace:
        WORKSPACE_ROOT = os.path.abspath(args.workspace)
    if not os.path.isdir(WORKSPACE_ROOT):
        parser.error(f"workspace '{WORKSPACE_ROOT}' is not a directory")

    if args.transport == "http":
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            print(f"WARNING: serving on {args.host} without authentication; anyone who can reach it can read "
                  f"and modify {WORKSPACE_ROOT}.", file=sys.stderr)
        mcp.run(transport="http", host=args.host, port=args.port, show_banner=False, log_level="CRITICAL")
    else:
        mcp.run(transport="stdio", show_banner=False, log_level="CRITICAL")


if __name__ == "__main__":
    main()
