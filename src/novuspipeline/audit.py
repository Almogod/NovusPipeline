"""
Audit log: one JSON object per line for every tool call (MCP or terminal), including calls
refused by authorization or organization policy.

Enabled by the policy's `audit_log` (which cannot be overridden) or NOVUS_AUDIT_LOG. Each
record carries `prev`, the SHA-256 of the previous line, so deleting or editing a record
breaks the chain; `novuspipeline audit-log verify` checks it. Appends take an OS file lock,
so several server processes can share one log.

Arguments are recorded, except that long values (file contents, proposed code, PR bodies)
are replaced by their SHA-256 and length. Results are not recorded, except the first line
of a write or destructive tool's result (e.g. the pipeline's `Verification:` line) and the
first line of an error.

    {"ts": "2026-11-18T16:46:52.123Z", "id": "…", "event": "tool_call", "tool": "run_autonomous_modernization_pipeline",
     "kind": "write", "outcome": "ok", "principal": "ci-bot", "auth": "token", "transport": "mcp-http",
     "workspace": "/srv/repos/billing", "args": {"file_path": "app/db.py", …}, "summary": "Verification: …",
     "duration_ms": 8123, "version": "0.1.0", "host": "build-7", "pid": 4242, "prev": "…"}
"""
from __future__ import annotations

import contextvars
import datetime as _dt
import getpass
import hashlib
import json
import os
import socket
import threading
import uuid
from dataclasses import dataclass
from typing import Any

ENV_AUDIT_LOG = "NOVUS_AUDIT_LOG"
GENESIS = "0" * 64
MAX_INLINE_CHARS = 200
FAILURE_PREFIXES = ("Error", "❌", "Pipeline Aborted")


@dataclass(frozen=True)
class Caller:
    principal: str
    auth: str  # "local" (stdio / terminal: the OS user), "token", "oidc"
    scopes: tuple[str, ...] = ()


# Set per request by the MCP middleware; unset means a local caller (the OS user).
CALLER: contextvars.ContextVar[Caller | None] = contextvars.ContextVar("novus_caller", default=None)
# "cli" until the server starts; then "mcp-stdio" or "mcp-http".
TRANSPORT = "cli"


def local_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def current_caller() -> Caller:
    return CALLER.get() or Caller(local_user(), "local")


def log_path() -> str:
    """The configured audit log, or "" when auditing is off. Raises PolicyError on a bad policy."""
    from . import policy
    configured = policy.current().audit_log or os.environ.get(ENV_AUDIT_LOG, "").strip()
    return os.path.abspath(os.path.expanduser(configured)) if configured else ""


def _redact(value: Any) -> Any:
    if isinstance(value, bool | int | float) or value is None:
        return value
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    if len(text) <= MAX_INLINE_CHARS:
        return value if isinstance(value, str | list | dict) else text
    return {"sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(), "chars": len(text)}


def _first_line(text: str, limit: int = 300) -> str:
    for line in text.splitlines():
        if line.strip():
            line = line.strip()
            return line if len(line) <= limit else line[:limit] + "…"
    return ""


# ------------------------------------------------------------------ locked append

if os.name == "nt":
    import msvcrt

    def _lock(f) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)  # retries for ~10 s, then raises

    def _unlock(f) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def _unlock(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _last_line(f) -> bytes:
    f.seek(0, os.SEEK_END)
    end = f.tell()
    chunk = b""
    pos = end
    while pos > 0:
        step = min(65536, pos)
        pos -= step
        f.seek(pos)
        chunk = f.read(step) + chunk
        stripped = chunk.rstrip(b"\r\n")
        nl = stripped.rfind(b"\n")
        if nl >= 0:
            return stripped[nl + 1:]
    return chunk.rstrip(b"\r\n")


_THREAD_LOCK = threading.Lock()


def _append(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
    with _THREAD_LOCK, os.fdopen(fd, "r+b") as f:
        _lock(f)
        try:
            last = _last_line(f)
            record["prev"] = hashlib.sha256(last).hexdigest() if last else GENESIS
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
            f.seek(0, os.SEEK_END)
            f.write(line + b"\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            _unlock(f)


def check_writable(path: str) -> str | None:
    """Why `path` cannot be used as the audit log, or None."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
        os.close(fd)
    except OSError as e:
        return f"audit log `{path}` is not writable: {e}"
    return None


# ------------------------------------------------------------------ public API

def record(tool: str, kind: str, args: dict[str, Any], outcome: str, *, workspace: str = "",
           result: str = "", detail: str = "", duration_ms: int | None = None,
           caller: Caller | None = None, event: str = "tool_call") -> None:
    """Appends one record when auditing is on. Raises OSError when the log cannot be written."""
    path = log_path()
    if not path:
        return
    from . import __version__
    caller = caller or current_caller()
    entry: dict[str, Any] = {
        "ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "id": uuid.uuid4().hex,
        "event": event,
        "tool": tool,
        "kind": kind,
        "outcome": outcome,
        "principal": caller.principal,
        "auth": caller.auth,
        "transport": TRANSPORT,
        "workspace": workspace,
        "args": {k: _redact(v) for k, v in args.items()},
    }
    if caller.scopes:
        entry["scopes"] = list(caller.scopes)
    summary = detail or (_first_line(result) if result and (kind != "read" or outcome != "ok") else "")
    if summary:
        entry["summary"] = summary
    if duration_ms is not None:
        entry["duration_ms"] = duration_ms
    entry.update(version=__version__, host=socket.gethostname(), pid=os.getpid())
    _append(path, entry)


def outcome_of(result: Any) -> str:
    return "error" if isinstance(result, str) and result.lstrip().startswith(FAILURE_PREFIXES) else "ok"


def verify(path: str) -> tuple[bool, str]:
    """Checks the hash chain of the log at `path`: (intact, message)."""
    try:
        with open(path, "rb") as f:
            lines = [ln.rstrip(b"\r\n") for ln in f if ln.strip()]
    except OSError as e:
        return False, f"cannot read `{path}`: {e}"
    prev = GENESIS
    for number, line in enumerate(lines, 1):
        try:
            entry = json.loads(line)
        except ValueError:
            return False, f"line {number}: not valid JSON"
        if entry.get("prev") != prev:
            return False, (f"line {number}: chain broken (a record before it was edited, removed or inserted); "
                           f"{number - 1} record(s) before it verified")
        prev = hashlib.sha256(line).hexdigest()
    return True, f"{len(lines)} record(s), chain intact"
