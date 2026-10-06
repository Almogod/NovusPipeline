import os
import sys
import logging
import shutil

# Ensure all logging is routed strictly to stderr at CRITICAL level BEFORE importing fastmcp
logging.basicConfig(level=logging.CRITICAL, stream=sys.stderr)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.CRITICAL)
logging.root.addHandler(stderr_handler)

os.environ["FASTMCP_SHOW_SERVER_BANNER"] = "0"
os.environ["FASTMCP_LOG_LEVEL"] = "CRITICAL"

import re
import shlex
import subprocess
from collections import Counter
from typing import List, Optional, Tuple
from fastmcp import FastMCP

from modernizer import LegacySmellDetector, CodeModernizer
import codebase_context
import local_llm
import gnn_model
from ingest_rag import CHROMA_DIR, COLLECTION_NAME, get_collection, similarity_from_distance as ingest_rag_similarity
from reporter import ModernizationReporter

# ---------------------------------------------------------------------------
# Server Initialization
# ---------------------------------------------------------------------------

mcp = FastMCP("NovusPipeline")

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

@mcp.tool()
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

        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@mcp.tool()
def query_rag_guidelines(query: str, category: str = "", n_results: int = 3) -> str:
    """
    Query the RAG vector database for modernization and clean-code guidelines.

    Args:
        query:     Natural language query or code snippet to match against guidelines.
        category:  Optional filter - one of 'python', 'typescript', 'security', 'clean_code'.
                   Leave empty to search across all categories.
        n_results: Number of top results to return (default 3, max 10).
    """
    try:
        n_results = max(1, min(n_results, 10))

        if os.path.exists(CHROMA_DIR):
            try:
                collection = get_collection()
                if collection is None:
                    raise LookupError("collection missing")

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

                if docs:
                    header = f"## RAG Guidelines - Query: `{query}`"
                    if category.strip():
                        header += f" | Category: `{category.strip()}`"
                    sections = []
                    for doc, meta, dist in zip(docs, metas, dists):
                        score = round(ingest_rag_similarity(dist), 3) if dist is not None else 0.0
                        title = meta.get("title", "Guideline")
                        cat = meta.get("category", "")
                        sections.append(f"### [{cat}] {title} (relevance: {score})\n\n{doc}")
                    return header + "\n\n" + "\n\n---\n\n".join(sections)
            except Exception:
                pass

        # Fallback when vector DB is not initialized
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
        return f"Error querying RAG guidelines: {str(e)}"


@mcp.tool()
def search_rag_by_id(document_id: str) -> str:
    """
    Retrieve a specific modernization guideline document directly by its ID.

    Args:
        document_id: Unique document ID (e.g. 'py-001', 'ts-002', 'sec-001', 'clean-001').
    """
    try:
        if not document_id.strip():
            return "Error: document_id cannot be empty."

        collection = get_collection() if os.path.exists(CHROMA_DIR) else None
        if collection is None:
            return "Error: RAG database is not initialized. Run ingest_rag.py first."

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
        return f"Error searching RAG document by ID '{document_id}': {str(e)}"


@mcp.tool()
def get_rag_stats() -> str:
    """
    Returns diagnostic statistics and status overview of the local RAG vector database.
    """
    try:
        chroma_dir = CHROMA_DIR
        if not os.path.exists(chroma_dir):
            return "RAG Status: Database not initialized at '.chroma_db'."

        try:
            collection = get_collection()
            if collection is None:
                raise LookupError("collection does not exist; run ingest_rag.py")
            count = collection.count()
            all_res = collection.get(include=["metadatas"])
            metas = all_res.get("metadatas", [])
            categories = Counter(m.get("category", "unknown") for m in metas)

            cat_str = "\n".join(f"  - `{cat}`: {num} documents" for cat, num in categories.items())
            return (
                f"## NovusPipeline RAG Database Overview\n\n"
                f"- **Status**: Operational\n"
                f"- **Storage Path**: `{chroma_dir}`\n"
                f"- **Collection Name**: `{COLLECTION_NAME}`\n"
                f"- **Total Guidelines**: {count}\n"
                f"- **Category Breakdown**:\n{cat_str}"
            )
        except Exception as e:
            return f"RAG Status: Collection '{COLLECTION_NAME}' error: {str(e)}"
    except Exception as e:
        return f"Error fetching RAG stats: {str(e)}"


@mcp.tool()
def ingest_rag_document(document_id: str, title: str, category: str, content: str) -> str:
    """
    Dynamically ingest a custom document into the RAG vector database.

    Args:
        document_id: Unique identifier for this document (e.g. 'py-custom-001').
        title:       Human-readable title for the guideline.
        category:    Category tag - one of 'python', 'typescript', 'security', 'clean_code'.
        content:     Full text content of the guideline or document to embed and store.
    """
    try:
        if not document_id.strip():
            return "Error: document_id cannot be empty."
        if not content.strip():
            return "Error: content cannot be empty."
        valid_categories = {"python", "typescript", "security", "clean_code"}
        if category.strip() not in valid_categories:
            return f"Error: category must be one of {sorted(valid_categories)}. Got: '{category}'."

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
        return f"Error ingesting document into RAG: {str(e)}"


@mcp.tool()
def reset_rag_database() -> str:
    """
    Programmatically resets and re-seeds the local RAG database with core enterprise guidelines.
    """
    try:
        from ingest_rag import seed_rag_database
        seed_rag_database(reset=True)
        return "Successfully reset and re-seeded the RAG vector database."
    except Exception as e:
        return f"Error resetting RAG database: {str(e)}"


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


def _validate_test_command(args: List[str]) -> Optional[str]:
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


def _execute_test_command(command: str) -> Tuple[Optional[int], str]:
    """Runs an allowed test command; returns (exit code or None if not run, output text)."""
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
        res = subprocess.run(args, cwd=WORKSPACE_ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=TEST_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return None, f"Error: Command '{command}' timed out after {TEST_TIMEOUT_SECONDS} seconds."
    except OSError as e:
        return None, f"Error executing test command '{command}': {e}"

    output = []
    if res.stdout:
        output += ["=== STDOUT ===", res.stdout]
    if res.stderr:
        output += ["=== STDERR ===", res.stderr]
    output.append(f"\nExit Code: {res.returncode}")
    return res.returncode, _truncate("\n".join(output))


@mcp.tool()
def run_local_tests(command: str) -> str:
    """
    Runs a test-suite command inside the workspace and returns its console output
    and exit code. Only test-runner invocations are allowed (e.g. `python -m pytest`,
    `python -m unittest discover`, `npm test`, `cargo test`); `python` resolves to
    the workspace virtualenv when one exists.
    """
    return _execute_test_command(command)[1]


def _pr_metadata_path(branch_name: str) -> str:
    # Branch names may contain "/" (feature/x); keep the draft file a flat name in the workspace root.
    return os.path.join(WORKSPACE_ROOT, f".novus_pr_{re.sub(r'[^A-Za-z0-9._-]', '_', branch_name)}.md")


@mcp.tool()
def create_git_migration_pr(branch_name: str, commit_message: str, pr_title: str, pr_description: str,
                            files: Optional[List[str]] = None) -> str:
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

        to_stage: List[str] = []
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

        pr_metadata = f"""# Draft PR: {pr_title}

## Target Branch
`{branch_name}`

## Description
{pr_description}

---
*Generated automatically by NovusPipeline Git Modernization Tool*
"""
        pr_file_path = _pr_metadata_path(branch_name)
        with open(pr_file_path, "w", encoding="utf-8") as f:
            f.write(pr_metadata)

        return f"{status_msg}\n{commit_info}\nDraft PR metadata written to '{pr_file_path}'."
    except Exception as e:
        return f"Error performing Git modernization operations: {str(e)}"


# ---------------------------------------------------------------------------
# Phase 3 & Local LLM Integration Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_local_llm_status() -> str:
    """
    Returns the status and metadata for the integrated fine-tuned local model:
    Path: C:\\Users\\Hp\\.unsloth\\studio\\outputs\\unsloth_Qwen3.5-2B_1785882774
    """
    try:
        info = local_llm.get_model_info()
        status_str = "Operational (Configured)" if info["exists"] else "Path Not Found"
        return (
            f"## Integrated Local LLM Status\n\n"
            f"- **Model Path**: `{info['model_path']}`\n"
            f"- **Model Name**: `{info['model_name']}`\n"
            f"- **Base Model**: `{info['base_model']}`\n"
            f"- **Adapter Config Found**: {info['has_adapter_config']}\n"
            f"- **Status**: {status_str}"
        )
    except Exception as e:
        return f"Error getting local LLM status: {str(e)}"


def _codebase_context(file_path: str) -> Tuple[Optional[dict], str]:
    """(file context from the codebase index, or None with a reason). Never raises."""
    try:
        index = codebase_context.get_index(WORKSPACE_ROOT)
        rel = index.normalize(_resolve_workspace_path(file_path))
        if rel is None:
            return None, "not an indexed Python file in this workspace"
        return index.file_context(rel), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _top_level_names(code: str) -> Optional[set]:
    """Names a module defines or imports at top level (what `from mod import x` can see)."""
    import ast
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        else:  # e.g. names defined under `if`/`try` at module level
            names.update(n.name for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
            names.update(n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    return names


def _broken_api(new_code: str, api_surface: List[str]) -> List[str]:
    """Symbols other modules import from this file that `new_code` no longer provides."""
    names = _top_level_names(new_code)
    if names is None or "*" in names:
        return []
    return sorted(n for n in api_surface if n not in names)


@mcp.tool()
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
        rag_guidelines = query_rag_guidelines(search_term, n_results=2)

        ctx, _ = _codebase_context(file_path)
        context_md = codebase_context.format_file_context(ctx) if ctx else ""
        proposal = local_llm.generate_llm_modernization(code, rag_guidelines, codebase_context=context_md)

        return (
            f"## LLM Modernization Proposal for `{file_path}`\n"
            f"**Model**: `unsloth_Qwen3.5-2B_1785882774`\n\n"
            f"### Retrieved Guidelines\n{rag_guidelines}\n\n"
            + (f"{context_md}\n\n" if context_md else "")
            + f"### Proposed Refactoring\n{proposal}"
        )
    except Exception as e:
        return f"Error generating LLM modernization proposal for '{file_path}': {str(e)}"


@mcp.tool()
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
        report = [f"## Modernization Audit Report for `{file_path}`\n"]

        if findings:
            by_smell: dict = {}
            for f in findings:
                by_smell.setdefault(f["smell_id"], []).append(f)
            report.append(f"Found **{len(findings)}** code smell(s) of **{len(by_smell)}** kind(s):\n")
            for smell_id, items in by_smell.items():
                first = items[0]
                report.append(f"### [{first['severity']}] `{smell_id}`: {first['name']} ({len(items)}x)")
                shown = items[:10]
                report.append("```code\n" + "\n".join(f"L{f['line_number']}: {f['line_content']}" for f in shown)
                              + ("\n..." if len(items) > len(shown) else "") + "\n```")
                rag_res = query_rag_guidelines(first["rag_query"], category=first["category"], n_results=1)
                report.append(f"**Recommended Guideline**:\n{rag_res}\n\n---\n")
        else:
            report.append("✅ No rule-based legacy code smells detected.\n")

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
        return f"Error analyzing legacy codebase for '{file_path}': {str(e)}"


@mcp.tool()
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

        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            original_code = f.read()

        if modernized_code.strip():
            new_code = modernized_code
            applied_changes = ["Applied LLM / user-supplied modernized code."]
        elif full_path.endswith(".py"):
            new_code, applied_changes = CodeModernizer.modernize_python(original_code)
        elif full_path.endswith((".ts", ".tsx", ".js", ".jsx")):
            new_code, applied_changes = CodeModernizer.modernize_typescript(original_code)
        else:
            return f"Error: Automated modernization rules not implemented for file extension of '{file_path}'."

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
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_code)

        return (
            f"Successfully updated '{file_path}'.\n"
            f"Backup created at '{file_path}.bak'.\n"
            f"Applied Transformations:\n{notes}{warning}"
        )
    except Exception as e:
        return f"Error applying modernization to '{file_path}': {str(e)}"


_PYTEST_AVAILABLE: dict = {}


def _default_test_command(ctx: Optional[dict]) -> str:
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


def _restore(full_path: str) -> str:
    backup = full_path + ".bak"
    if os.path.exists(backup):
        shutil.copyfile(backup, full_path)
        os.remove(backup)
        return "Rolled back from backup snapshot."
    return "Backup snapshot file not found for rollback."


@mcp.tool()
def run_autonomous_modernization_pipeline(
    file_path: str,
    test_command: str = "",
    branch_name: str = "auto-modernization-branch"
) -> str:
    """
    Executes the complete, codebase-aware autonomous refactoring loop:
      1. Audits the file (rules + GNN) and loads its codebase context from the index.
      2. Applies parity-preserving modernizations (with a .bak safety snapshot).
      3. Rolls back if the change removes symbols other modules import.
      4. Runs verification tests: `test_command`, or by default the tests that import
         this file (falling back to the whole suite).
      5. Exit code 0 -> commits only this file on `branch_name` and drafts a PR that lists
         the blast radius and follow-up files with the same smells. Otherwise rolls back.

    Args:
        file_path:    Target legacy file to modernize.
        test_command: Optional test command; empty selects related tests automatically.
        branch_name:  Target Git branch for PR creation (default 'auto-modernization-branch').
    """
    full_path = _resolve_workspace_path(file_path)
    if not is_path_in_workspace(full_path):
        return f"Error: Path '{file_path}' is outside authorized project workspace."

    try:
        ctx, _ = _codebase_context(file_path)
        audit_res = analyze_legacy_codebase(file_path)

        mod_res = apply_code_modernization(file_path)
        if mod_res.startswith("Error"):
            return f"Pipeline Aborted during modernization phase:\n{mod_res}"
        if mod_res.startswith("No applicable"):
            return f"Pipeline finished: nothing to modernize automatically.\n{mod_res}"

        if ctx and full_path.endswith(".py"):
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                broken = _broken_api(f.read(), ctx["api_surface"])
            if broken:
                return (f"❌ Pipeline rolled back: the modernized file no longer provides "
                        f"{', '.join(f'`{n}`' for n in broken)}, which other modules import.\n{_restore(full_path)}")

        command = test_command.strip() or _default_test_command(ctx)
        exit_code, test_res = _execute_test_command(command)

        if exit_code != 0:
            reason = "could not be run" if exit_code is None else f"failed (exit code {exit_code})"
            return (f"❌ Autonomous Modernization Pipeline: verification `{command}` {reason}.\n"
                    f"{_restore(full_path)}\n\nTest Output:\n{test_res}")

        impact = codebase_context.format_file_context(ctx) if ctx else "_Codebase context unavailable._"
        follow_ups = (ctx or {}).get("same_smell_elsewhere", [])
        pr_res = create_git_migration_pr(
            branch_name=branch_name,
            commit_message=f"refactor: autonomous modernization of {os.path.basename(file_path)}",
            pr_title=f"Autonomous Modernization: {os.path.basename(file_path)}",
            pr_description=(f"### Modernization Audit\n{audit_res}\n\n### Applied Changes\n{mod_res}\n\n"
                            f"### Test Verification\nPassed (exit code 0): `{command}`\n\n### Impact\n{impact}"),
            files=[file_path],
        )
        if os.path.exists(full_path + ".bak"):
            os.remove(full_path + ".bak")

        follow_up_md = ("\n\n5. Follow-up candidates (same smells elsewhere):\n"
                        + "\n".join(f"   - `{p}`" for p in follow_ups[:10])) if follow_ups else ""
        return (
            f"🎉 Autonomous Modernization Pipeline Completed Successfully!\n\n"
            f"1. Audit Findings & RAG Match:\n{audit_res[:300]}...\n\n"
            f"2. Transformations Applied:\n{mod_res}\n\n"
            f"3. Verification (`{command}`):\n{test_res}\n\n"
            f"4. Git Status:\n{pr_res}{follow_up_md}"
        )
    except Exception as e:
        _restore(full_path)
        return f"Error executing autonomous modernization pipeline for '{file_path}': {str(e)}"


# ---------------------------------------------------------------------------
# Phase 4: Git PR & Modernization Reporting Tools
# ---------------------------------------------------------------------------

@mcp.tool()
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
        ctx, reason = _codebase_context(file_path)
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
        return f"Error formatting modernization report for '{file_path}': {str(e)}"


@mcp.tool()
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
        return f"Error finalizing Git migration PR for branch '{branch_name}': {str(e)}"


# ---------------------------------------------------------------------------
# Phase 5: GNN Structural Code-Smell Classifier
# ---------------------------------------------------------------------------

@mcp.tool()
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
        return f"Error getting GNN model status: {str(e)}"


@mcp.tool()
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
        return f"Error: '{file_path}' is not valid Python 3 syntax, cannot build AST graph: {str(e)}"
    except ValueError as e:
        return f"Error: '{file_path}' cannot be analyzed by the GNN: {str(e)}"
    except RuntimeError as e:
        return f"Error: GNN model unavailable: {str(e)}"
    except Exception as e:
        return f"Error running GNN structural analysis on '{file_path}': {str(e)}"


# ---------------------------------------------------------------------------
# Phase 6: Codebase Context (GNN + import graph)
# ---------------------------------------------------------------------------

@mcp.tool()
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


@mcp.tool()
def get_codebase_overview(top_n: int = 10) -> str:
    """
    Codebase-wide modernization overview: files ranked by smell risk (rule engine + GNN,
    severity-weighted) with their blast radius (number of importing modules), the most
    depended-on modules, and the smell distribution. Use it to decide what to modernize first.

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
            lines.append(f"_GNN signals unavailable: {index.gnn_error}_\n")

        lines.append("### 🔥 Modernization Hotspots (risk = severity-weighted smells; source of each signal)")
        for h in index.hotspots(top_n):
            smells = ", ".join(f"{s} [{src}]" for s, src in h["smells"].items())
            lines.append(f"- `{h['path']}` risk **{h['risk']:.0f}**, imported by {h['dependents']} module(s)"
                         f"{' (test)' if h['is_test'] else ''}: {smells}")

        fan_in = Counter()
        for rel in files:
            fan_in.update(index.dependencies(rel).keys())
        lines.append("\n### 🕸️ Most Depended-On Modules (changes here have the widest blast radius)")
        for rel, count in fan_in.most_common(top_n):
            lines.append(f"- `{rel}`: imported by {count} module(s)")

        distribution = Counter()
        for rec in files.values():
            distribution.update({f["smell_id"] for f in rec["rule_findings"]})
        if distribution:
            lines.append("\n### 📊 Files Affected per Smell (rule engine)")
            for smell_id, count in distribution.most_common():
                lines.append(f"- `{smell_id}` {LegacySmellDetector._BY_ID[smell_id]['name']}: {count} file(s)")

        broken = [rel for rel, rec in files.items() if rec["parse_error"]]
        if broken:
            lines.append(f"\n### ⚠️ Not valid Python 3 ({len(broken)} file(s), rule engine only)")
            lines.extend(f"- `{rel}`" for rel in broken[:top_n])
        return "\n".join(lines)
    except Exception as e:
        return f"Error building codebase overview: {type(e).__name__}: {e}"


@mcp.tool()
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


@mcp.tool()
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


if __name__ == "__main__":
    os.environ["FASTMCP_SHOW_SERVER_BANNER"] = "0"
    os.environ["FASTMCP_LOG_LEVEL"] = "CRITICAL"
    mcp.run(transport="stdio", show_banner=False, log_level="CRITICAL")
