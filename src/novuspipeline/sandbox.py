"""
sandbox.py — NovusPipeline: isolated execution of test commands

Test suites run arbitrary project code, so they never run in the user's working
tree with the user's environment. Every run gets:

  workspace copy   a temporary copy of the project (or a git worktree the caller
                   already created); the user's files cannot be modified
  scrubbed env     an allowlist of variables; HOME / USERPROFILE / APPDATA / TEMP
                   point into the sandbox, so ~/.ssh, ~/.aws, tokens and API keys
                   are not visible (extend with NOVUS_SANDBOX_ENV_ALLOW=A,B)
  limits           memory and process-count limits, and the whole process tree is
                   killed on timeout (Windows Job Object; POSIX rlimits + process group)
  no stdin         stdin is the null device: under the stdio transport the server's
                   own stdin is the MCP channel, which tests must never read from
  write-up block   Windows: the run gets a low integrity level, so it can write only
                   inside the sandbox; the user's files, including ones reached by
                   absolute path or through the node_modules link, are read-only to it
                   (NOVUS_SANDBOX_LOW_INTEGRITY=0 turns this off)
  network          blocked on Linux when user namespaces allow (`unshare -n`);
                   not isolated on Windows/macOS without a container
  docker           NOVUS_SANDBOX=docker runs inside `docker run --network none`
                   with the image in NOVUS_SANDBOX_IMAGE (which must contain the
                   project's test dependencies)

`isolation` in every result says exactly which protections were active.
"""

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

MEMORY_LIMIT_MB = int(os.environ.get("NOVUS_SANDBOX_MEMORY_MB", "4096"))
PROCESS_LIMIT = int(os.environ.get("NOVUS_SANDBOX_MAX_PROCESSES", "256"))
COPY_IGNORE = {".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__", ".novus_index",
               ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox", "build", "dist", ".eggs"}
# Directories linked (not copied) into the sandbox because tests need them and they are large.
LINK_DIRS = ("node_modules",)
ENV_ALLOW = {
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "OS", "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ", "TERM",
    "PYTHONIOENCODING", "PYTHONUTF8", "CI",
}


@dataclass
class SandboxResult:
    exit_code: int | None
    output: str
    isolation: list[str] = field(default_factory=list)
    seconds: float = 0.0
    timed_out: bool = False


def _scrubbed_env(home: str) -> dict[str, str]:
    allow = ENV_ALLOW | {v.strip() for v in os.environ.get("NOVUS_SANDBOX_ENV_ALLOW", "").split(",") if v.strip()}
    env = {k: v for k, v in os.environ.items() if k.upper() in allow}
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
        env[name] = home
    tmp = os.path.join(home, "tmp")
    os.makedirs(tmp, exist_ok=True)
    env.update(TEMP=tmp, TMP=tmp, TMPDIR=tmp, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    return env


def _link_dir(src: str, dst: str) -> bool:
    try:
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(src, dst)  # junctions need no admin rights
        else:
            os.symlink(src, dst, target_is_directory=True)
        return True
    except (OSError, ImportError, AttributeError):
        return False


def link_dependency_dirs(src: str, dst: str) -> None:
    for name in LINK_DIRS:
        if os.path.isdir(os.path.join(src, name)) and not os.path.exists(os.path.join(dst, name)):
            _link_dir(os.path.join(src, name), os.path.join(dst, name))


def unlink_dependency_dirs(dst: str) -> None:
    """
    Removes the links (never their targets) before the sandbox is deleted: a recursive
    delete that followed a junction would wipe the user's real node_modules.
    """
    for name in LINK_DIRS:
        path = os.path.join(dst, name)
        try:
            if os.path.islink(path):
                os.unlink(path)
            elif os.name == "nt" and os.path.isdir(path) and getattr(os.path, "isjunction", lambda p: False)(path):
                os.rmdir(path)  # removes the junction itself, not the directory it points to
        except OSError:
            pass


def copy_workspace(src: str, dst: str) -> None:
    """Copies the project for a test run, skipping VCS/venv/cache dirs; links heavy dependency dirs."""
    def ignore(directory: str, names: list[str]) -> list[str]:
        return [n for n in names if n in COPY_IGNORE]

    shutil.copytree(src, dst, ignore=ignore, symlinks=True)
    link_dependency_dirs(src, dst)


# --------------------------------------------------------------------------- Windows job objects

def _windows_job(memory_mb: int, max_processes: int):
    """A Job Object that limits memory/process count and kills everything when closed."""
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                    "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObject failed")
    info = EXTENDED_LIMIT()
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS, JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x8, 0x200, 0x2000
    info.BasicLimitInformation.LimitFlags = (JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_JOB_MEMORY
                                             | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    info.BasicLimitInformation.ActiveProcessLimit = max_processes
    info.JobMemoryLimit = memory_mb * 1024 * 1024
    JobObjectExtendedLimitInformation = 9
    if not kernel32.SetInformationJobObject(wintypes.HANDLE(job), JobObjectExtendedLimitInformation,
                                            ctypes.byref(info), ctypes.sizeof(info)):
        kernel32.CloseHandle(wintypes.HANDLE(job))
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
    return job


def _windows_assign_to_job(proc: subprocess.Popen, job) -> None:
    """Assigns a CREATE_SUSPENDED process to the job before it runs (no escape window)."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(int(proc._handle))):
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")


def _windows_resume(proc: subprocess.Popen) -> None:
    import ctypes
    from ctypes import wintypes
    ctypes.WinDLL("ntdll").NtResumeProcess(wintypes.HANDLE(int(proc._handle)))


# Runs the real command as a child of a process whose token was lowered. The child is
# created *by* a low-integrity token, so the process object itself is low too; lowering a
# running process only changes its token, and its own children (multiprocessing) could then
# not open it.
_TRAMPOLINE = "import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:]))"


def low_integrity_enabled() -> bool:
    return os.name == "nt" and os.environ.get("NOVUS_SANDBOX_LOW_INTEGRITY", "1") != "0"


def allow_sandbox_writes(path: str) -> bool:
    """
    Windows: labels `path` low integrity, inherited by everything created inside it
    afterwards, so a low-integrity test run can write there (and nowhere else). Call it on
    an empty directory before filling it. Returns False where this does not apply or fails.
    """
    if not low_integrity_enabled():
        return False
    icacls = os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32", "icacls.exe")
    try:
        return subprocess.run([icacls, path, "/setintegritylevel", "(OI)(CI)L"], stdin=subprocess.DEVNULL,
                              capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _windows_lower_integrity(proc: subprocess.Popen) -> None:
    """Drops a CREATE_SUSPENDED process to low integrity: no writes to medium-integrity objects (the user's files)."""
    import ctypes
    from ctypes import wintypes

    class SID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi32.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    advapi32.GetLengthSid.argtypes = [ctypes.c_void_p]
    advapi32.GetLengthSid.restype = wintypes.DWORD
    advapi32.SetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    TOKEN_QUERY, TOKEN_ADJUST_DEFAULT, TokenIntegrityLevel, SE_GROUP_INTEGRITY = 0x0008, 0x0080, 25, 0x20
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(wintypes.HANDLE(int(proc._handle)), TOKEN_QUERY | TOKEN_ADJUST_DEFAULT,
                                     ctypes.byref(token)):
        raise OSError(ctypes.get_last_error(), "OpenProcessToken failed")
    sid = ctypes.c_void_p()
    try:
        if not advapi32.ConvertStringSidToSidW("S-1-16-4096", ctypes.byref(sid)):  # low mandatory level
            raise OSError(ctypes.get_last_error(), "ConvertStringSidToSid failed")
        label = SID_AND_ATTRIBUTES(sid, SE_GROUP_INTEGRITY)
        if not advapi32.SetTokenInformation(token, TokenIntegrityLevel, ctypes.byref(label),
                                            ctypes.sizeof(label) + advapi32.GetLengthSid(sid)):
            raise OSError(ctypes.get_last_error(), "SetTokenInformation failed")
    finally:
        if sid:
            kernel32.LocalFree(sid)
        kernel32.CloseHandle(token)


def _windows_close_job(job) -> None:
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.TerminateJobObject(wintypes.HANDLE(job), 1)
    kernel32.CloseHandle(wintypes.HANDLE(job))


# --------------------------------------------------------------------------- POSIX limits

def _posix_preexec(memory_mb: int, max_processes: int):
    def apply() -> None:
        import resource
        os.setsid()
        for limit, value in ((resource.RLIMIT_AS, memory_mb * 1024 * 1024), (resource.RLIMIT_NPROC, max_processes)):
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(limit, (value, value))
    return apply


def _linux_netns_prefix() -> list[str]:
    """`unshare -rn` (new user + network namespace: no network) when the kernel allows it."""
    if not sys.platform.startswith("linux") or not shutil.which("unshare"):
        return []
    try:
        ok = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    return ["unshare", "-rn"] if ok else []


# --------------------------------------------------------------------------- docker

def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True,
                              timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _run_docker(args: list[str], workdir: str, timeout: int, image: str) -> SandboxResult:
    name = f"novus-sandbox-{os.getpid()}-{int(time.time() * 1000)}"
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--memory", f"{MEMORY_LIMIT_MB}m",
           "--pids-limit", str(PROCESS_LIMIT), "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "-v", f"{workdir}:/work", "-w", "/work", image, *args]
    start = time.perf_counter()
    try:
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=timeout)
        out = (res.stdout or "") + (("\n" + res.stderr) if res.stderr else "")
        return SandboxResult(res.returncode, out, [f"docker container ({image}), no network, memory/pids limits"],
                             time.perf_counter() - start)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return SandboxResult(None, "", ["docker"], time.perf_counter() - start, timed_out=True)


# --------------------------------------------------------------------------- entry point

def run(args: list[str], workspace: str, timeout: int, copy: bool = True) -> SandboxResult:
    """
    Runs `args` against `workspace` in isolation. With copy=True (default) the project is
    copied first; pass copy=False when `workspace` is already a disposable checkout.
    """
    root = tempfile.mkdtemp(prefix="novus-sandbox-")
    try:
        workdir = os.path.join(root, "work") if copy else workspace
        isolation = []
        # Before anything is created inside: the copy, home and temp inherit the label. A
        # disposable checkout was labelled by its creator; labelling it again covers new files.
        low = allow_sandbox_writes(root) and (copy or allow_sandbox_writes(workspace))
        if copy:
            copy_workspace(workspace, workdir)
            isolation.append("project copy (your files cannot be modified)")
        else:
            isolation.append("disposable checkout")

        mode = os.environ.get("NOVUS_SANDBOX", "auto").lower()
        if mode == "docker" or (mode == "auto" and os.environ.get("NOVUS_SANDBOX_IMAGE") and docker_available()):
            image = os.environ.get("NOVUS_SANDBOX_IMAGE", f"python:{sys.version_info.major}.{sys.version_info.minor}-slim")
            result = _run_docker(args, workdir, timeout, image)
            result.isolation = isolation + result.isolation
            return result

        home = os.path.join(root, "home")
        os.makedirs(home, exist_ok=True)
        env = _scrubbed_env(home)
        # The sandbox's sources must win over an editable install of the same package, or
        # tests would import the user's original tree instead of the code being verified.
        env["PYTHONPATH"] = os.pathsep.join(p for p in (workdir, os.path.join(workdir, "src")) if os.path.isdir(p))
        isolation.append(f"scrubbed environment ({len(env)} allowlisted variables; home/temp redirected)")
        kwargs: dict = {"cwd": workdir, "env": env, "stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE,
                        "stderr": subprocess.STDOUT, "text": True, "encoding": "utf-8", "errors": "replace"}
        job = None
        if os.name == "nt":
            kwargs["creationflags"] = 0x00000004  # CREATE_SUSPENDED until the job and integrity level are set
            try:
                job = _windows_job(MEMORY_LIMIT_MB, PROCESS_LIMIT)
            except OSError:
                job = None
            if low:
                args = [sys.executable, "-I", "-S", "-c", _TRAMPOLINE, *args]
        else:
            kwargs["preexec_fn"] = _posix_preexec(MEMORY_LIMIT_MB, PROCESS_LIMIT)
            isolation.append(f"rlimits: {MEMORY_LIMIT_MB} MB address space, {PROCESS_LIMIT} processes; own process group")
            prefix = _linux_netns_prefix()
            args = prefix + args
            isolation.append("network: blocked (network namespace)" if prefix else "network: not isolated")

        start = time.perf_counter()
        proc = subprocess.Popen(args, **kwargs)
        try:
            if os.name == "nt":
                try:  # the process is suspended: nothing of it has run yet
                    if job is not None:
                        _windows_assign_to_job(proc, job)
                    if low:
                        _windows_lower_integrity(proc)
                except OSError as e:
                    proc.kill()  # never let it run with less isolation than was set up for
                    proc.communicate()
                    raise OSError(f"could not apply the sandbox limits ({e}); NOVUS_SANDBOX_LOW_INTEGRITY=0 runs "
                                  "tests without the low integrity level") from e
                if job is not None:
                    isolation.append(f"job object: {MEMORY_LIMIT_MB} MB memory, {PROCESS_LIMIT} processes, "
                                     "tree killed on exit")
                isolation.append("low integrity level (cannot modify files outside the sandbox)" if low
                                 else "runs with your user's write access")
                isolation.append("network: not isolated (needs a container on Windows)")
                _windows_resume(proc)
            try:
                out, _ = proc.communicate(timeout=timeout)
                return SandboxResult(proc.returncode, out or "", isolation, time.perf_counter() - start)
            except subprocess.TimeoutExpired:
                if job is not None:
                    _windows_close_job(job)
                    job = None
                elif os.name != "nt":
                    import signal
                    with contextlib.suppress(OSError):
                        os.killpg(proc.pid, signal.SIGKILL)
                proc.kill()
                out, _ = proc.communicate()
                return SandboxResult(None, out or "", isolation, time.perf_counter() - start, timed_out=True)
        finally:
            if job is not None:
                _windows_close_job(job)
    finally:
        if copy:
            unlink_dependency_dirs(os.path.join(root, "work"))
        shutil.rmtree(root, ignore_errors=True)
