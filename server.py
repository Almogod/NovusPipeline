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

import math
import re
import subprocess
from collections import Counter
from fastmcp import FastMCP
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from modernizer import LegacySmellDetector, CodeModernizer

# ---------------------------------------------------------------------------
# Server Initialization
# ---------------------------------------------------------------------------

mcp = FastMCP("NovusPipeline")

WORKSPACE_ROOT = os.path.abspath(os.getcwd())
COLLECTION_NAME = "novus_guidelines"
EMBEDDING_DIM = 256
MAX_READ_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB safety limit

_TECH_STOP_WORDS = {
    "the", "a", "an", "is", "it", "in", "of", "to", "and", "or", "for",
    "on", "with", "be", "by", "at", "as", "this", "that", "from", "not",
    "use", "are", "was", "were", "has", "have", "do", "does",
    "its", "their", "if", "else", "then", "all", "each", "any"
}

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
    """Ensure the path stays strictly within the configured workspace directory."""
    abs_target = os.path.abspath(target_path)
    return abs_target.startswith(WORKSPACE_ROOT)


def expand_query(query: str) -> str:
    """Expand natural language queries with domain tech synonyms for higher recall."""
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    expanded = set(tokens)
    for key, synonyms in QUERY_EXPANSIONS.items():
        if key in tokens:
            expanded.update(synonyms)
    return " ".join(expanded)


# ---------------------------------------------------------------------------
# TF-IDF Offline Embedding Function (256-dim, zero network calls)
# ---------------------------------------------------------------------------

class TFIDFEmbeddingFunction(EmbeddingFunction):
    """
    Offline TF-IDF weighted embedding (256-dim).
    Uses log-scaled TF, corpus-level IDF, and bigram neighborhood signal spreading.
    Zero network calls or external dependencies.
    """

    def __init__(self) -> None:
        super().__init__()

    def name(self) -> str:
        return "tfidf_256"

    def _tokenize(self, text: str) -> list[str]:
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text.lower())
        tokens = re.findall(r"[a-z0-9]+", text)
        return [t for t in tokens if len(t) > 1 and t not in _TECH_STOP_WORDS]

    def _embed_single(self, doc: str, idf: dict[str, float]) -> list[float]:
        tokens = self._tokenize(doc)
        if not tokens:
            return [0.0] * EMBEDDING_DIM
        tf = Counter(tokens)
        total = len(tokens)
        vec = [0.0] * EMBEDDING_DIM
        for token, count in tf.items():
            weight = math.log(1 + count / total) * idf.get(token, 1.0)
            idx = hash(token) % EMBEDDING_DIM
            vec[idx] += weight
            vec[(idx + 1) % EMBEDDING_DIM] += weight * 0.3
            vec[(idx - 1) % EMBEDDING_DIM] += weight * 0.3
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def __call__(self, input: Documents) -> Embeddings:
        tokenized = [self._tokenize(doc) for doc in input]
        N = len(input)
        df: dict[str, int] = {}
        for tokens in tokenized:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in df}
        return [self._embed_single(doc, idf) for doc in input]


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
        full_path = (
            os.path.abspath(os.path.join(WORKSPACE_ROOT, file_path))
            if not os.path.isabs(file_path)
            else os.path.abspath(file_path)
        )

        if not is_path_in_workspace(full_path):
            return f"Error: Path '{file_path}' is outside the authorized project workspace."

        if not os.path.exists(full_path):
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
        chroma_dir = os.path.join(WORKSPACE_ROOT, ".chroma_db")

        if os.path.exists(chroma_dir):
            try:
                import chromadb
                client = chromadb.PersistentClient(path=chroma_dir)
                ef = TFIDFEmbeddingFunction()
                collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

                where_filter = {"category": {"$eq": category.strip()}} if category.strip() else None

                results = collection.query(
                    query_texts=[query],
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
                        score = round(max(0.0, 1.0 - abs(dist)), 3) if dist is not None else 0.0
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

        import chromadb
        chroma_dir = os.path.join(WORKSPACE_ROOT, ".chroma_db")
        if not os.path.exists(chroma_dir):
            return f"Error: RAG database is not initialized. Run ingest_rag.py first."

        client = chromadb.PersistentClient(path=chroma_dir)
        ef = TFIDFEmbeddingFunction()
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

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
        chroma_dir = os.path.join(WORKSPACE_ROOT, ".chroma_db")
        if not os.path.exists(chroma_dir):
            return "RAG Status: Database not initialized at '.chroma_db'."

        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        ef = TFIDFEmbeddingFunction()

        try:
            collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
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

        import chromadb
        chroma_dir = os.path.join(WORKSPACE_ROOT, ".chroma_db")
        client = chromadb.PersistentClient(path=chroma_dir)
        ef = TFIDFEmbeddingFunction()

        try:
            collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
        except Exception:
            collection = client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
                metadata={"description": "NovusPipeline Enterprise Modernization Guidelines"}
            )

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


@mcp.tool()
def run_local_tests(command: str) -> str:
    """
    Accepts a test suite terminal command (e.g., 'pytest', 'npm test');
    executes inside workspace sandbox and returns standard out / error console text.
    """
    try:
        command_clean = command.strip()
        if not command_clean:
            return "Error: Empty command specified."

        dangerous_chars = [";", "&&", "||", "`", "$("]
        for char in dangerous_chars:
            if char in command_clean:
                return f"Error: Command contains dangerous shell chaining character '{char}'."

        parts = command_clean.split()
        allowed_executables = ["pytest", "python", "npm", "node", "unittest", "cargo", "go", "mvn", "gradle"]
        executable_basename = os.path.basename(parts[0]).lower().replace(".exe", "").replace(".cmd", "")

        if executable_basename not in allowed_executables:
            return (
                f"Error: Command '{parts[0]}' is not in the allowed local verification tools whitelist "
                f"({', '.join(allowed_executables)})."
            )

        res = subprocess.run(
            parts,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=120
        )

        output = []
        if res.stdout:
            output.append("=== STDOUT ===")
            output.append(res.stdout)
        if res.stderr:
            output.append("=== STDERR ===")
            output.append(res.stderr)
        output.append(f"\nExit Code: {res.returncode}")

        return "\n".join(output) if output else f"Executed '{command}' with Exit Code: {res.returncode}"
    except subprocess.TimeoutExpired:
        return f"Error: Command '{command}' timed out after 120 seconds."
    except Exception as e:
        return f"Error executing test command '{command}': {str(e)}"


@mcp.tool()
def create_git_migration_pr(branch_name: str, commit_message: str, pr_title: str, pr_description: str) -> str:
    """
    Accepts branch name, commit message, and PR details;
    triggers local Git branch creation, stages workspace changes, commits, and formats draft PR metadata.
    """
    try:
        import git
        repo = git.Repo(WORKSPACE_ROOT)

        current_branch = repo.active_branch.name

        if branch_name not in [b.name for b in repo.branches]:
            new_branch = repo.create_head(branch_name)
            new_branch.checkout()
            status_msg = f"Created and checked out new branch '{branch_name}' (from '{current_branch}')."
        else:
            repo.branches[branch_name].checkout()
            status_msg = f"Checked out existing branch '{branch_name}'."

        repo.git.add(A=True)

        if repo.is_dirty(index=True) or repo.untracked_files:
            commit = repo.index.commit(commit_message)
            commit_info = f"Committed staged changes: {commit.hexsha[:8]}"
        else:
            commit_info = "No unstaged/staged changes detected to commit."

        pr_metadata = f"""# Draft PR: {pr_title}

## Target Branch
`{branch_name}`

## Description
{pr_description}

---
*Generated automatically by NovusPipeline Git Modernization Tool*
"""
        pr_file_path = os.path.join(WORKSPACE_ROOT, f".novus_pr_{branch_name}.md")
        with open(pr_file_path, "w", encoding="utf-8") as f:
            f.write(pr_metadata)

        return f"{status_msg}\n{commit_info}\nDraft PR metadata written to '{pr_file_path}'."
    except Exception as e:
        return f"Error performing Git modernization operations: {str(e)}"


# ---------------------------------------------------------------------------
# Phase 3: Autonomous Refactoring Loop Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_legacy_codebase(file_path: str) -> str:
    """
    Phase 3 Tool: Audits legacy code for enterprise code smells and cross-references
    matching guidelines from the RAG database to generate a Modernization Strategy.

    Args:
        file_path: Target relative or absolute file path to analyze.
    """
    try:
        code = read_legacy_file(file_path)
        if code.startswith("Error:"):
            return code

        findings = LegacySmellDetector.scan_code(code, file_path)

        if not findings:
            return f"## Modernization Audit Report for `{file_path}`\n\n✅ No legacy code smells detected. File satisfies active enterprise standards."

        report = [f"## Modernization Audit Report for `{file_path}`\n"]
        report.append(f"Found **{len(findings)}** code smell(s):\n")

        for f in findings:
            smell_id = f["smell_id"]
            name = f["name"]
            line = f["line_number"]
            severity = f["severity"]
            query = f["rag_query"]

            report.append(f"### [{severity}] `{smell_id}`: {name} (Line {line})")
            report.append(f"```code\n{f['line_content']}\n```")

            # Fetch matching guideline from RAG
            rag_res = query_rag_guidelines(query, category=f["category"], n_results=1)
            report.append(f"**Recommended Guideline**:\n{rag_res}\n\n---\n")

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
        full_path = (
            os.path.abspath(os.path.join(WORKSPACE_ROOT, file_path))
            if not os.path.isabs(file_path)
            else os.path.abspath(file_path)
        )

        if not is_path_in_workspace(full_path):
            return f"Error: Path '{file_path}' is outside authorized project workspace."

        if not os.path.exists(full_path):
            return f"Error: File '{file_path}' does not exist."

        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            original_code = f.read()

        # Create backup snapshot
        backup_path = full_path + ".bak"
        shutil.copyfile(full_path, backup_path)

        if modernized_code.strip():
            new_code = modernized_code
            applied_changes = ["Applied user-supplied modernized code."]
        else:
            if full_path.endswith(".py"):
                new_code, applied_changes = CodeModernizer.modernize_python(original_code)
            elif full_path.endswith((".ts", ".js")):
                new_code, applied_changes = CodeModernizer.modernize_typescript(original_code)
            else:
                return f"Error: Automated modernization rules not implemented for file extension of '{file_path}'."

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_code)

        changes_summary = "\n".join(f"  - {c}" for c in applied_changes)
        return (
            f"Successfully updated '{file_path}'.\n"
            f"Backup created at '{file_path}.bak'.\n"
            f"Applied Transformations:\n{changes_summary}"
        )
    except Exception as e:
        return f"Error applying modernization to '{file_path}': {str(e)}"


@mcp.tool()
def run_autonomous_modernization_pipeline(
    file_path: str,
    test_command: str = "python -m unittest test_server.py",
    branch_name: str = "auto-modernization-branch"
) -> str:
    """
    Phase 3 Tool: Executes the complete autonomous refactoring loop:
      1. Scans file for legacy smells & queries RAG guidelines.
      2. Applies parity-preserving modernizations (with .bak safety snapshot).
      3. Executes verification test suite.
      4. If tests PASS -> Stages, commits, and creates draft PR.
      5. If tests FAIL -> Automatically rolls back file from backup snapshot!

    Args:
        file_path:    Target legacy file to modernize.
        test_command: Sandboxed test command to verify behavior (default 'python -m unittest test_server.py').
        branch_name:  Target Git branch for PR creation (default 'auto-modernization-branch').
    """
    full_path = (
        os.path.abspath(os.path.join(WORKSPACE_ROOT, file_path))
        if not os.path.isabs(file_path)
        else os.path.abspath(file_path)
    )
    backup_path = full_path + ".bak"

    try:
        # Step 1: Audit
        audit_res = analyze_legacy_codebase(file_path)

        # Step 2: Apply modernization
        mod_res = apply_code_modernization(file_path)
        if mod_res.startswith("Error"):
            return f"Pipeline Aborted during modernization phase:\n{mod_res}"

        # Step 3: Run Verification Tests
        test_res = run_local_tests(test_command)

        if "Exit Code: 0" in test_res or "OK" in test_res:
            # Step 4: Verification Passed -> Git PR
            pr_res = create_git_migration_pr(
                branch_name=branch_name,
                commit_message=f"refactor: autonomous modernization of {os.path.basename(file_path)}",
                pr_title=f"Autonomous Modernization: {os.path.basename(file_path)}",
                pr_description=f"### Modernization Audit\n{audit_res}\n\n### Applied Changes\n{mod_res}\n\n### Test Verification\nPassed: `{test_command}`"
            )

            # Cleanup backup on success
            if os.path.exists(backup_path):
                os.remove(backup_path)

            return (
                f"🎉 Autonomous Modernization Pipeline Completed Successfully!\n\n"
                f"1. Audit Findings & RAG Match:\n{audit_res[:300]}...\n\n"
                f"2. Transformations Applied:\n{mod_res}\n\n"
                f"3. Verification Results:\n{test_res}\n\n"
                f"4. Git Status:\n{pr_res}"
            )
        else:
            # Step 5: Verification Failed -> Automatic Rollback!
            if os.path.exists(backup_path):
                shutil.copyfile(backup_path, full_path)
                os.remove(backup_path)
                rollback_status = f"Rolled back '{file_path}' from backup snapshot."
            else:
                rollback_status = "Backup snapshot file not found for rollback."

            return (
                f"❌ Autonomous Modernization Pipeline Failed Verification Tests!\n"
                f"{rollback_status}\n\n"
                f"Test Output:\n{test_res}"
            )
    except Exception as e:
        if os.path.exists(backup_path):
            try:
                shutil.copyfile(backup_path, full_path)
                os.remove(backup_path)
            except Exception:
                pass
        return f"Error executing autonomous modernization pipeline for '{file_path}': {str(e)}"


if __name__ == "__main__":
    os.environ["FASTMCP_SHOW_SERVER_BANNER"] = "0"
    os.environ["FASTMCP_LOG_LEVEL"] = "CRITICAL"
    mcp.run(transport="stdio", show_banner=False, log_level="CRITICAL")
