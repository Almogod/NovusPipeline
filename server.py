import os
import subprocess
from fastmcp import FastMCP

# Initialize FastMCP Server for NovusPipeline
mcp = FastMCP("NovusPipeline")

# Workspace root configuration
WORKSPACE_ROOT = os.path.abspath(os.getcwd())

def is_path_in_workspace(target_path: str) -> bool:
    """Ensure the path stays strictly within the configured workspace directory."""
    abs_target = os.path.abspath(target_path)
    return abs_target.startswith(WORKSPACE_ROOT)


@mcp.tool()
def read_legacy_file(file_path: str) -> str:
    """
    🛠️ Accepts a file path string; returns raw legacy code string.
    Ensures security path restraints to workspace bounds.
    """
    try:
        full_path = os.path.abspath(os.path.join(WORKSPACE_ROOT, file_path)) if not os.path.isabs(file_path) else os.path.abspath(file_path)
        
        if not is_path_in_workspace(full_path):
            return f"Error: Path '{file_path}' is outside the authorized project workspace."
        
        if not os.path.exists(full_path):
            return f"Error: File '{file_path}' does not exist."
            
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@mcp.tool()
def query_rag_guidelines(query: str) -> str:
    """
    📚 Accepts a code snippet query string; returns relevant modernization
    and clean-code guidelines from the vector database or knowledge system.
    """
    try:
        chroma_dir = os.path.join(WORKSPACE_ROOT, ".chroma_db")
        if os.path.exists(chroma_dir):
            try:
                import chromadb
                client = chromadb.PersistentClient(path=chroma_dir)
                collection = client.get_or_create_collection(name="novus_guidelines")
                results = collection.query(query_texts=[query], n_results=3)
                
                if results and results.get("documents") and results["documents"][0]:
                    docs = results["documents"][0]
                    return "\n\n---\n\n".join(docs)
            except Exception:
                pass

        default_rules = [
            "### Novus Rule #1: Explicit Typing & Modern Constructs",
            "- Always add type hints (Python 3.10+ syntax / TypeScript strict mode).",
            "- Replace obsolete libraries (e.g. `urllib2` -> `httpx`/`requests`, ES5 `var` -> `const`/`let`).",
            "",
            "### Novus Rule #2: Security & Error Handling",
            "- Enforce strict parameter validation (Pydantic / Zod models).",
            "- Do not suppress raw exceptions without logging or structural recovery.",
            "",
            "### Novus Rule #3: Logical & Test Parity",
            "- Code transformations must strictly preserve logical parity.",
            "- Ensure deterministic behavior and full backward compatibility in function interfaces."
        ]
        return f"Query: '{query}'\nMatched Compliance Guidelines:\n" + "\n".join(default_rules)
    except Exception as e:
        return f"Error querying RAG guidelines: {str(e)}"


@mcp.tool()
def run_local_tests(command: str) -> str:
    """
    ⚡ Accepts a test suite terminal command (e.g., 'pytest', 'npm test');
    executes inside workspace sandbox and returns standard out / error console text.
    """
    try:
        parts = command.strip().split()
        if not parts:
            return "Error: Empty command specified."
            
        allowed_executables = ["pytest", "python", "npm", "node", "unittest", "cargo", "go", "mvn", "gradle"]
        executable_basename = os.path.basename(parts[0]).lower().replace(".exe", "").replace(".cmd", "")
        
        if executable_basename not in allowed_executables:
            return f"Error: Command '{parts[0]}' is not in the allowed local verification tools whitelist ({', '.join(allowed_executables)})."
            
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
    🌿 Accepts branch name, commit message, and PR details;
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


if __name__ == "__main__":
    print("Starting NovusPipeline FastMCP Server...")
    mcp.run()
