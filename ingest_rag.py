"""
ingest_rag.py — NovusPipeline Phase 2+: RAG Vector Pipeline Seeder

Populates a persistent local ChromaDB database with enterprise modernization
guidelines using an offline TF-IDF-weighted embedding function for accurate
keyword-driven semantic retrieval without any external API calls.

Usage:
    python ingest_rag.py
"""

import os
import math
import re
from collections import Counter

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

WORKSPACE_ROOT = os.path.abspath(os.getcwd())
CHROMA_DIR = os.path.join(WORKSPACE_ROOT, ".chroma_db")
COLLECTION_NAME = "novus_guidelines"
EMBEDDING_DIM = 256

_TECH_STOP_WORDS = {
    "the", "a", "an", "is", "it", "in", "of", "to", "and", "or", "for",
    "on", "with", "be", "by", "at", "as", "this", "that", "from", "not",
    "use", "are", "was", "were", "has", "have", "do", "does",
    "its", "their", "if", "else", "then", "all", "each", "any"
}


class TFIDFEmbeddingFunction(EmbeddingFunction):
    """
    Offline TF-IDF weighted embedding (256-dim). Zero network calls.
    Uses log-scaled TF, corpus-level IDF, and bigram neighborhood spreading.
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
# Enterprise Modernization Guideline Corpus (14 guidelines, 4 categories)
# ---------------------------------------------------------------------------

GUIDELINES = [
    # ---- Python ----
    {
        "id": "py-001",
        "title": "Python Type System Modernization",
        "category": "python",
        "content": """### Python Type System Modernization
- Always add type hints to function signatures using Python 3.10+ union syntax (`str | None` not `Optional[str]`).
- Enable `from __future__ import annotations` for forward references in Python 3.8-3.9 codebases.
- Use `TypeVar`, `Generic`, and `Protocol` for polymorphic code rather than loose `Any`.
- Run `mypy --strict` or `pyright` in CI to catch type regressions automatically.
- Replace bare `dict` and `list` with typed `dict[str, Any]` or dataclasses for structured payloads."""
    },
    {
        "id": "py-002",
        "title": "Python Library Modernization",
        "category": "python",
        "content": """### Python Library Modernization
- Replace `urllib`/`urllib2` with `httpx` (async-first) or `requests` (sync).
- Replace `optparse` with `argparse` or `typer` for CLI interfaces.
- Replace deprecated `os.popen`, `commands` with `subprocess.run` using `capture_output=True`.
- Replace `pickle` for IPC with `json`, `msgpack`, or `pydantic` serialization.
- Replace legacy `ConfigParser` with `pydantic-settings` with `.env` file support.
- Replace `print x` (Python 2) with `print(x)` and add structured logging with the `logging` module."""
    },
    {
        "id": "py-003",
        "title": "Python Async & Concurrency",
        "category": "python",
        "content": """### Python Async & Concurrency
- Convert blocking network I/O to async with `asyncio` + `httpx.AsyncClient` or `aiofiles`.
- Avoid `time.sleep` in async context; use `await asyncio.sleep(n)` instead.
- Use `asyncio.TaskGroup` (Python 3.11+) or `asyncio.gather` for parallel tasks.
- Replace thread-unsafe module-level globals with `contextvars.ContextVar`.
- Always handle `asyncio.CancelledError` by re-raising after cleanup; never swallow it.
- Use `async with asyncio.timeout(n)` (Python 3.11+) instead of `asyncio.wait_for` where possible."""
    },
    {
        "id": "py-004",
        "title": "Python Error Handling & Logging",
        "category": "python",
        "content": """### Python Error Handling & Logging
- Never use bare `except:` or `except Exception: pass` - always log or re-raise.
- Use structured logging (`structlog` or standard `logging` with JSON formatters) not bare `print`.
- Catch the most specific exception type; avoid catching `BaseException` unless intentional.
- Add context to error messages: `raise ValueError(f"Invalid config: {config!r}") from original_exc`.
- Use `contextlib.suppress(SpecificError)` for truly ignorable errors to signal intent clearly."""
    },
    # ---- TypeScript / JavaScript ----
    {
        "id": "ts-001",
        "title": "TypeScript Strict Mode & Type Safety",
        "category": "typescript",
        "content": """### TypeScript Strict Mode & Type Safety
- Enable `"strict": true` in `tsconfig.json` - this catches `null`, `undefined`, and implicit `any`.
- Replace `any` with precise types: interfaces, union types, or `unknown` + type guard.
- Use `satisfies` operator (TS 4.9+) for validated object literals without losing inference.
- Avoid type assertions (`as Type`) except at system boundaries; prefer type guards (`typeof`, `instanceof`).
- Model API responses with `zod` schemas and infer TypeScript types: `type User = z.infer<typeof UserSchema>`."""
    },
    {
        "id": "ts-002",
        "title": "JavaScript / TypeScript ES Modernization",
        "category": "typescript",
        "content": """### JavaScript / TypeScript ES Modernization
- Replace `var` with `const` (immutable) or `let` (mutable); never use `var`.
- Replace callback-style async with `async/await` and Promises.
- Replace CommonJS `require()` with ES modules `import`/`export`.
- Replace `_.forEach`, `_.map` Lodash patterns with native array methods (`map`, `filter`, `reduce`).
- Replace string concatenation with template literals.
- Replace `arguments` object with rest parameters `(...args: string[])`.
- Use optional chaining `?.` and nullish coalescing `??` instead of verbose null guards."""
    },
    {
        "id": "ts-003",
        "title": "Frontend React / Component Modernization",
        "category": "typescript",
        "content": """### Frontend React & Component Modernization
- Replace class components with functional components using React Hooks (`useState`, `useEffect`).
- Replace lifecycle methods (`componentDidMount`) with `useEffect` using correct dependency arrays.
- Avoid `useEffect` for data fetching; prefer TanStack Query or SWR for server state.
- Replace prop drilling with React Context or Zustand for shared state.
- Memoize expensive computations with `useMemo`; memoize callbacks with `useCallback`.
- Add explicit TypeScript prop types to all components: `interface Props { ... }`."""
    },
    # ---- Security ----
    {
        "id": "sec-001",
        "title": "Path Security & File Access",
        "category": "security",
        "content": """### Path Security & File Access
- Always resolve and anchor file paths: `os.path.abspath(path).startswith(WORKSPACE_ROOT)`.
- Never pass raw user input to `open()`, `subprocess`, or `os.system` without sanitization.
- Use `pathlib.Path` for cross-platform path manipulation to avoid OS separator issues.
- Validate file extensions against an explicit allowlist before reading or executing.
- Set restrictive file permissions (`0o600` for secrets, `0o644` for read-only configs)."""
    },
    {
        "id": "sec-002",
        "title": "Input Validation & Injection Prevention",
        "category": "security",
        "content": """### Input Validation & Injection Prevention
- Validate all external inputs with `Pydantic` v2 (Python) or `Zod` (TypeScript) schemas at system boundaries.
- Never build SQL queries via string concatenation - always use parameterized queries or ORM query builders.
- Avoid `eval()`, `exec()`, `Function()`, or dynamic code execution with untrusted input.
- Sanitize HTML output to prevent XSS; use a library like `bleach` (Python) or `DOMPurify` (JS).
- Use command whitelists (`allowed_executables`) when executing subprocesses; never `shell=True` with user data."""
    },
    {
        "id": "sec-003",
        "title": "Authentication & Secrets Management",
        "category": "security",
        "content": """### Authentication & Secrets Management
- Never hardcode secrets, API keys, or credentials in source code - use environment variables or a secrets manager.
- Load secrets from `.env` files with `python-dotenv` (Python) or `dotenv` (Node) - never commit `.env` to Git.
- Use short-lived tokens (JWT with expiry) over long-lived API keys when possible.
- Hash passwords with `bcrypt` or `argon2`; never use `md5` or `sha1` for password storage.
- Rotate secrets regularly and audit access logs; use tools like HashiCorp Vault for production."""
    },
    # ---- Clean Code ----
    {
        "id": "clean-001",
        "title": "Logical Parity & Test-Driven Refactoring",
        "category": "clean_code",
        "content": """### Logical Parity & Test-Driven Refactoring
- Transformations must preserve complete functional behavior, return types, and side effects.
- Write or update unit tests before AND after code modernization (Red - Green - Refactor).
- Never suppress raw exceptions without structured logging and explicit intent documentation.
- Use mutation testing (`mutmut`, `stryker`) to validate test coverage quality, not just line coverage.
- Snapshot test output of pure functions to catch unintended behavioral changes during refactoring."""
    },
    {
        "id": "clean-002",
        "title": "API & Interface Design Principles",
        "category": "clean_code",
        "content": """### API & Interface Design Principles
- Follow the Single Responsibility Principle: each function/class/module should have one reason to change.
- Prefer explicit over implicit: function parameters over module-level globals.
- Avoid stringly-typed interfaces; use enums, literal types, or dataclasses for structured parameters.
- Design APIs to be additive: new optional parameters with defaults, never remove or rename existing ones.
- Document public APIs with docstrings including parameter types, return values, and example usage."""
    },
    {
        "id": "clean-003",
        "title": "Performance & Algorithmic Complexity",
        "category": "clean_code",
        "content": """### Performance & Algorithmic Complexity
- Replace O(n^2) nested loops over collections with O(n) hash map lookups where possible.
- Avoid repeated attribute access in hot loops: cache `obj.attr` to a local variable.
- Use generators and lazy evaluation for large data pipelines instead of materializing full lists.
- Profile before optimizing: use `cProfile` (Python) or `Clinic`/`perf` (Node.js) to identify bottlenecks.
- Batch database queries to avoid N+1 query problems; use `.select_related()`/`.prefetch_related()` in ORMs."""
    },
    {
        "id": "clean-004",
        "title": "Dependency Management & Versioning",
        "category": "clean_code",
        "content": """### Dependency Management & Versioning
- Pin all direct dependencies to exact or constrained versions in lock files (`poetry.lock`, `package-lock.json`).
- Audit dependencies for known CVEs: `pip-audit` (Python), `npm audit` (Node), `cargo audit` (Rust).
- Remove unused dependencies; track with `deptry` (Python) or `depcheck` (Node).
- Separate runtime vs dev dependencies explicitly (`pyproject.toml [dev]`, `devDependencies`).
- Use Renovate or Dependabot for automated, PR-driven dependency updates."""
    },
]


def _safe_print(text: str) -> None:
    """Print text safely on Windows terminals with limited encoding."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def seed_rag_database(reset: bool = True) -> None:
    """
    Ingest all enterprise guidelines into local persistent ChromaDB instance.

    Args:
        reset: If True, drops and recreates the collection for a clean slate.
    """
    _safe_print(f"[NovusPipeline RAG] Initializing ChromaDB at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = TFIDFEmbeddingFunction()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            _safe_print(f"[NovusPipeline RAG] Dropped existing '{COLLECTION_NAME}' collection.")
        except Exception:
            pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={
            "description": "NovusPipeline Enterprise Modernization Guidelines",
            "version": "2.0",
            "embedding_dim": str(EMBEDDING_DIM),
        }
    )

    documents = [g["content"] for g in GUIDELINES]
    metadatas = [{"title": g["title"], "category": g["category"]} for g in GUIDELINES]
    ids = [g["id"] for g in GUIDELINES]

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    categories = sorted(set(g["category"] for g in GUIDELINES))
    _safe_print(f"[NovusPipeline RAG] Ingested {len(documents)} guidelines.")
    _safe_print(f"[NovusPipeline RAG] Categories: {', '.join(categories)}")
    _safe_print(f"[NovusPipeline RAG] IDs: {', '.join(ids)}")
    _safe_print("[NovusPipeline RAG] Done. Run ingest_rag.py again to refresh.")


def verify_retrieval(query: str = "async python blocking io", n: int = 3) -> None:
    """Quick sanity check - query the freshly seeded collection."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = TFIDFEmbeddingFunction()
    col = client.get_collection(COLLECTION_NAME, embedding_function=ef)
    results = col.query(query_texts=[query], n_results=n, include=["documents", "metadatas", "distances"])
    _safe_print(f"\n[NovusPipeline RAG] Retrieval check (query='{query}'):")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ), 1):
        score = round(1.0 - dist, 3) if dist is not None else "N/A"
        preview = doc[:120].encode("ascii", errors="replace").decode("ascii")
        _safe_print(f"\n  [{i}] {meta['title']} (category={meta['category']}, score={score})")
        _safe_print(f"      {preview}...")


if __name__ == "__main__":
    seed_rag_database(reset=True)
    verify_retrieval("async python blocking io")
    verify_retrieval("typescript var const arrow function")
    verify_retrieval("path traversal sql injection input validation")
