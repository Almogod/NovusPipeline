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


def _windows_start_in_job(proc: subprocess.Popen, job) -> None:
    """Assigns a CREATE_SUSPENDED process to the job, then resumes it (no escape window)."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    handle = wintypes.HANDLE(int(proc._handle))
    if not kernel32.AssignProcessToJobObject(wintypes.HANDLE(job), handle):
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
    ntdll.NtResumeProcess(handle)


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
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
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
        kwargs: dict = {"cwd": workdir, "env": env, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True,
                            "encoding": "utf-8", "errors": "replace"}
        job = None
        if os.name == "nt":
            try:
                job = _windows_job(MEMORY_LIMIT_MB, PROCESS_LIMIT)
                kwargs["creationflags"] = 0x00000004  # CREATE_SUSPENDED until assigned to the job
                isolation.append(f"job object: {MEMORY_LIMIT_MB} MB memory, {PROCESS_LIMIT} processes, tree killed on exit")
            except OSError:
                job = None
            isolation.append("network: not isolated (needs a container on Windows)")
        else:
            kwargs["preexec_fn"] = _posix_preexec(MEMORY_LIMIT_MB, PROCESS_LIMIT)
            isolation.append(f"rlimits: {MEMORY_LIMIT_MB} MB address space, {PROCESS_LIMIT} processes; own process group")
            prefix = _linux_netns_prefix()
            args = prefix + args
            isolation.append("network: blocked (network namespace)" if prefix else "network: not isolated")

        start = time.perf_counter()
        proc = subprocess.Popen(args, **kwargs)
        try:
            if job is not None:
                _windows_start_in_job(proc, job)
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
