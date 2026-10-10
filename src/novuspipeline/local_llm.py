r"""
local_llm.py — NovusPipeline Local LLM Modernization Engine (client side)

Talks to `llm_worker.py`, a separate process that hosts a local LLM (by default
the base model `unsloth/Qwen3.5-2B`, optionally with a LoRA adapter). The
worker runs under its own interpreter so GPU dependencies and ~4.5 GB of
weights stay out of the MCP server process.

Configuration (environment):
  NOVUS_LLM_BASE_MODEL  Hugging Face model id (default unsloth/Qwen3.5-2B)
  NOVUS_LLM_PATH        optional LoRA adapter directory
  NOVUS_LLM_PYTHON      worker interpreter; default: a dev `.venv-llm` next to the
                        repo, else this interpreter if `transformers` + `torch` are
                        installed (pip install "novuspipeline[llm]")

Every proposal is validated before anyone may apply it: it must parse, must not
introduce rule-engine smells, and must keep every symbol other modules import.
Without a worker (or with NOVUS_FAST_TEST=1) the rule-based modernizer is used.
"""

import importlib.util
import json
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_MODEL = "unsloth/Qwen3.5-2B"
WORKER_SCRIPT = os.path.join(_HERE, "llm_worker.py")


def adapter_path() -> str:
    return os.environ.get("NOVUS_LLM_PATH", "")


def model_name() -> str:
    path = adapter_path()
    return os.path.basename(path.rstrip("\\/")) if path else os.environ.get("NOVUS_LLM_BASE_MODEL", DEFAULT_BASE_MODEL)
# Worker stderr (load progress, tracebacks) — kept for diagnosing load/OOM failures.
WORKER_LOG = os.path.join(tempfile.gettempdir(), "novus_llm_worker.log")
LOAD_TIMEOUT_SECONDS = 600
GENERATE_TIMEOUT_SECONDS = 900

SYSTEM_PROMPT = (
    "You are NovusPipeline, an autonomous code modernization AI. Refactor the legacy code to "
    "comply with the guidelines (modern libraries, explicit type hints, no bare except, secure "
    "APIs) while strictly preserving its behavior. Never rename or remove functions, classes or "
    "variables that other modules depend on. Reply with the complete modernized file in a single "
    "```python code block, then at most three short bullet points describing the changes."
)


def llm_python() -> Optional[str]:
    """Worker interpreter: NOVUS_LLM_PYTHON, a dev .venv-llm, this interpreter with [llm] installed, else None."""
    if os.environ.get("NOVUS_LLM_PYTHON"):
        return os.environ["NOVUS_LLM_PYTHON"]
    repo_root = os.path.dirname(os.path.dirname(_HERE))  # src/novuspipeline -> repo
    for rel in ((".venv-llm", "Scripts", "python.exe"), (".venv-llm", "bin", "python")):
        candidate = os.path.join(repo_root, *rel)
        if os.path.isfile(candidate):
            return candidate
    if importlib.util.find_spec("transformers") and importlib.util.find_spec("torch"):
        return sys.executable
    return None


class LLMWorker:
    """One persistent worker process; thread-safe; restarted if it dies, killed if it hangs."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lines: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 0

    def _start(self) -> None:
        python = llm_python()
        if python is None:
            raise RuntimeError('No LLM interpreter: pip install "novuspipeline[llm]" or set NOVUS_LLM_PYTHON.')
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        with open(WORKER_LOG, "a", encoding="utf-8") as log:
            self._proc = subprocess.Popen([python, "-u", WORKER_SCRIPT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=log, text=True, encoding="utf-8", env=env)
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self._proc, self._lines), daemon=True).start()

    @staticmethod
    def _pump(proc: subprocess.Popen, lines: "queue.Queue[str]") -> None:
        for line in proc.stdout:
            lines.put(line)
        lines.put("")  # EOF marker: worker exited

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def request(self, op: str, timeout: float, **payload: Any) -> Dict[str, Any]:
        with self._lock:
            if not self.running:
                self._start()
            self._next_id += 1
            req_id = self._next_id
            self._proc.stdin.write(json.dumps({"id": req_id, "op": op, **payload}) + "\n")
            self._proc.stdin.flush()
            while True:
                try:
                    line = self._lines.get(timeout=timeout)
                except queue.Empty:
                    self.stop()
                    raise TimeoutError(f"LLM worker did not answer '{op}' within {timeout:.0f}s (worker restarted).")
                if line == "":
                    self._proc = None
                    raise RuntimeError("LLM worker exited unexpectedly (out of memory?).")
                resp = json.loads(line)
                if resp.get("id") == req_id:
                    return resp

    def stop(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except OSError:
                pass
            self._proc = None


_WORKER = LLMWorker()


def get_model_info(probe: bool = False) -> Dict[str, Any]:
    """Static configuration; with `probe`, starts the worker and loads the model to report device/VRAM."""
    adapter = adapter_path()
    info: Dict[str, Any] = {
        "adapter_path": adapter or None,
        "adapter_found": bool(adapter) and os.path.exists(os.path.join(adapter, "adapter_config.json")),
        "model_name": model_name(),
        "base_model": os.environ.get("NOVUS_LLM_BASE_MODEL", DEFAULT_BASE_MODEL),
        "worker_python": llm_python(),
        "worker_running": _WORKER.running,
        "worker_log": WORKER_LOG,
    }
    if info["adapter_found"]:
        try:
            with open(os.path.join(adapter, "adapter_config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            info.update(base_model=cfg.get("base_model_name_or_path", info["base_model"]),
                        lora_rank=cfg.get("r"), lora_alpha=cfg.get("lora_alpha"),
                        training_method=cfg.get("unsloth_training_method"))
        except (OSError, ValueError):
            pass
    if probe:
        try:
            resp = _WORKER.request("load", LOAD_TIMEOUT_SECONDS)
            info["worker"] = resp.get("status")
            info["load_error"] = resp.get("error")
        except Exception as e:
            info["load_error"] = f"{type(e).__name__}: {e}"
        info["worker_running"] = _WORKER.running
    return info


def generate_modernization_messages(legacy_code: str, rag_guidelines: str, codebase_context: str = "") -> List[Dict[str, str]]:
    context_block = f"\nCodebase context (dependents, public API, smell locations):\n{codebase_context}\n" \
        if codebase_context else ""
    user = (f"Modernization guidelines:\n{rag_guidelines}\n{context_block}\n"
            f"Legacy code to refactor:\n```python\n{legacy_code}\n```")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def generate_modernization_prompt(legacy_code: str, rag_guidelines: str, codebase_context: str = "") -> str:
    """Plain-text rendering of the chat messages (the worker applies the real chat template)."""
    return "\n\n".join(f"<|{m['role']}|>\n{m['content']}"
                       for m in generate_modernization_messages(legacy_code, rag_guidelines, codebase_context))


def extract_code(text: str) -> Optional[str]:
    """The model's code: the longest fenced block that parses, else the whole reply if it parses."""
    import ast
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.DOTALL)
    for block in sorted(blocks, key=len, reverse=True) + [text]:
        try:
            ast.parse(block)
            return block.rstrip() + "\n"
        except (SyntaxError, ValueError):
            continue
    return None


def _defined_callables(code: str) -> set:
    """Top-level function and class names (module constants/aliases are allowed to be added)."""
    import ast
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return set()
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def validate_proposal(original: str, proposed: Optional[str], required_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Gate for applying a proposal: parses, introduces no smells, keeps imported symbols."""
    from .modernizer import LegacySmellDetector, missing_names, top_level_names

    if proposed is None:
        return {"valid": False, "issues": ["No parseable Python code in the model output."]}
    before = Counter(f["smell_id"] for f in LegacySmellDetector.scan_code(original, "x.py"))
    after = Counter(f["smell_id"] for f in LegacySmellDetector.scan_code(proposed, "x.py"))
    issues: List[str] = []
    introduced = sorted(s for s in after if after[s] > before.get(s, 0))
    if introduced:
        issues.append(f"Introduces smells: {', '.join(introduced)}")
    broken = missing_names(proposed, required_names or [])
    if broken:
        issues.append(f"Removes symbols other modules import: {', '.join(broken)}")
    # A modernization rewrites existing code; new functions/classes or a __main__ block
    # are hallucinated scope (seen with the over-trained adapter), not a refactor.
    old_defs = top_level_names(original, include_imports=False) or set()
    added = sorted(_defined_callables(proposed) - _defined_callables(original))
    if added:
        issues.append(f"Adds top-level definitions not in the original: {', '.join(added[:8])}")
    if re.search(r"^if\s+__name__\s*==", proposed, re.MULTILINE) and not re.search(r"^if\s+__name__\s*==", original, re.MULTILINE):
        issues.append("Adds an `if __name__ == '__main__':` block")
    old_public = {n for n in old_defs if not n.startswith("_")}
    dropped = sorted(old_public - (top_level_names(proposed) or set()))
    return {
        "valid": not issues,
        "issues": issues,
        "warnings": [f"Drops public top-level names: {', '.join(dropped)}"] if dropped else [],
        "smells_before": dict(before), "smells_after": dict(after),
        "resolved": sorted(s for s in before if after.get(s, 0) < before[s]),
    }


def _max_new_tokens(code: str) -> int:
    # A rewrite is about as long as its input (~3.5 chars/token for code), plus room for notes.
    return max(256, min(4096, int(len(code) / 3.5 * 1.3) + 192))


def propose_modernization(legacy_code: str, rag_guidelines: str, codebase_context: str = "",
                          required_names: Optional[List[str]] = None, adapter: bool = True) -> Dict[str, Any]:
    """
    Structured LLM proposal: {"engine", "code", "raw", "validation", "stats", "error"}.
    Falls back to the rule-based modernizer when no worker is available.
    """
    from .modernizer import CodeModernizer

    def rule_based(reason: str) -> Dict[str, Any]:
        code, changes = CodeModernizer.modernize_python(legacy_code)
        return {"engine": "rule-based", "code": code, "raw": "\n".join(f"- {c}" for c in changes),
                "validation": validate_proposal(legacy_code, code, required_names), "stats": {}, "error": reason}

    if os.environ.get("NOVUS_FAST_TEST") == "1":
        return rule_based("NOVUS_FAST_TEST=1")
    try:
        resp = _WORKER.request("generate", GENERATE_TIMEOUT_SECONDS + LOAD_TIMEOUT_SECONDS,
                               messages=generate_modernization_messages(legacy_code, rag_guidelines, codebase_context),
                               max_new_tokens=_max_new_tokens(legacy_code), adapter=adapter)
    except Exception as e:
        logging.warning(f"Local LLM unavailable, using rule-based fallback: {e}")
        return rule_based(f"{type(e).__name__}: {e}")
    if not resp.get("ok"):
        return rule_based(resp.get("error", "worker error"))
    code = extract_code(resp["text"])
    return {"engine": f"{model_name()}{'' if adapter else ' (adapter disabled)'}", "code": code, "raw": resp["text"],
            "validation": validate_proposal(legacy_code, code, required_names), "stats": resp.get("stats", {}),
            "error": None}
