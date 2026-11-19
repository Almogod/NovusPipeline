"""
Organization policy: settings an administrator applies to every NovusPipeline run on a
machine or in a container image, which a project, a user setting or an environment
variable cannot loosen.

Sources, merged so that the most restrictive value wins:

  system   <site config dir>/policy.toml, e.g. /etc/xdg/novuspipeline/policy.toml (Linux),
           C:\\ProgramData\\novuspipeline\\policy.toml (Windows),
           /Library/Application Support/novuspipeline/policy.toml (macOS)
  explicit NOVUS_POLICY_FILE (adds restrictions; it cannot lift the system file's)

    disabled_tools          = ["apply_code_modernization"]   # tools nobody may call
    allow_destructive_tools = false    # also disables every tool annotated destructive
    allow_untested          = false    # the pipeline never commits a file no test ran against
    allow_llm               = false    # no local LLM proposals
    llm_offline             = true     # the LLM never downloads models
    require_docker_sandbox  = true     # tests only run in a no-network container
    required_packs          = ["baseline", "secure-defaults", "/etc/acme/payments.toml"]
    branch_prefix           = "novus/" # commits only go to branches with this prefix
    audit_log               = "/var/log/novuspipeline/audit.jsonl"
    require_auth            = true     # the HTTP transport needs authentication, even on loopback

A policy file that cannot be read, or that has an unknown key or a wrong type, makes every
tool refuse to run (fail closed) instead of being ignored.

This is a guardrail for managed deployments (a shared server, a container, a managed
workstation): it binds this installation, not a user who installs another copy.
"""
from __future__ import annotations

import os
import threading
import tomllib
from dataclasses import dataclass, field, replace

ENV_POLICY_FILE = "NOVUS_POLICY_FILE"


class PolicyError(RuntimeError):
    """The policy file exists but is not valid; callers must refuse to run."""


@dataclass(frozen=True)
class Policy:
    sources: tuple[str, ...] = ()
    disabled_tools: frozenset[str] = frozenset()
    allow_destructive_tools: bool = True
    allow_untested: bool = True
    allow_llm: bool = True
    llm_offline: bool = False
    require_docker_sandbox: bool = False
    required_packs: tuple[str, ...] = ()
    branch_prefix: str = ""
    audit_log: str = ""
    require_auth: bool = False
    _set: frozenset[str] = field(default=frozenset(), repr=False, compare=False)

    def tool_denial(self, name: str, kind: str) -> str | None:
        """Why the policy forbids calling tool `name` (annotated `kind`), or None."""
        if name in self.disabled_tools:
            return f"`{name}` is disabled by organization policy."
        if kind == "destructive" and not self.allow_destructive_tools:
            return f"`{name}` changes your working copy or deletes data; organization policy disables such tools."
        return None

    def branch_denial(self, branch: str) -> str | None:
        if self.branch_prefix and not branch.startswith(self.branch_prefix):
            return f"organization policy only allows commits to branches starting with `{self.branch_prefix}`."
        return None

    def describe(self) -> str:
        if not self.sources:
            return "Organization policy: none (no policy file; defaults apply)."
        lines = ["Organization policy: " + ", ".join(f"`{s}`" for s in self.sources)]
        for key, (_, default) in _FIELDS.items():
            value = getattr(self, key)
            if value != default:
                shown = sorted(value) if isinstance(value, frozenset | tuple) else value
                lines.append(f"  {key} = {shown!r}")
        if len(lines) == 1:
            lines.append("  (the file sets nothing that differs from the defaults)")
        return "\n".join(lines)


# key -> (type, default). Booleans named allow_* merge with AND, require_*/llm_offline with OR.
_FIELDS: dict[str, tuple[type, object]] = {
    "disabled_tools": (list, frozenset()),
    "allow_destructive_tools": (bool, True),
    "allow_untested": (bool, True),
    "allow_llm": (bool, True),
    "llm_offline": (bool, False),
    "require_docker_sandbox": (bool, False),
    "required_packs": (list, ()),
    "branch_prefix": (str, ""),
    "audit_log": (str, ""),
    "require_auth": (bool, False),
}


def system_policy_path() -> str:
    from platformdirs import site_config_dir
    return os.path.join(site_config_dir("novuspipeline", appauthor=False), "policy.toml")


def _parse(path: str) -> Policy:
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise PolicyError(f"cannot read organization policy `{path}`: {e}") from e
    unknown = sorted(set(data) - set(_FIELDS))
    if unknown:
        raise PolicyError(f"organization policy `{path}` has unknown setting(s): {', '.join(unknown)}")
    values: dict[str, object] = {}
    for key, raw in data.items():
        kind, _ = _FIELDS[key]
        if not isinstance(raw, kind) or (kind is list and not all(isinstance(v, str) for v in raw)):
            expected = "a list of strings" if kind is list else kind.__name__
            raise PolicyError(f"organization policy `{path}`: `{key}` must be {expected}")
        values[key] = raw
    if "disabled_tools" in values:
        values["disabled_tools"] = frozenset(values["disabled_tools"])
    if "required_packs" in values:
        values["required_packs"] = tuple(values["required_packs"])
        _check_packs(path, values["required_packs"])
    if values.get("audit_log"):
        values["audit_log"] = os.path.abspath(os.path.expanduser(values["audit_log"]))
    return Policy(sources=(path,), _set=frozenset(values), **values)


def _check_packs(path: str, packs: tuple[str, ...]) -> None:
    from .standards import builtin_packs
    builtin = builtin_packs()
    for pack in packs:
        if pack in builtin:
            continue
        if not os.path.isabs(pack):
            raise PolicyError(f"organization policy `{path}`: required pack `{pack}` is neither a built-in pack "
                              f"({', '.join(sorted(builtin))}) nor an absolute path to a .toml file")
        if not os.path.isfile(pack):
            raise PolicyError(f"organization policy `{path}`: required pack file `{pack}` does not exist")


def _merge(strict: Policy, extra: Policy) -> Policy:
    """`strict` (the system policy) with `extra`'s additional restrictions."""
    def pick(key: str):
        a, b = getattr(strict, key), getattr(extra, key)
        if key not in extra._set:
            return a
        if key not in strict._set:
            return b
        if key.startswith("allow_"):
            return a and b
        if key.startswith("require_") or key == "llm_offline":
            return a or b
        if key == "disabled_tools":
            return a | b
        if key == "required_packs":
            return a + tuple(p for p in b if p not in a)
        return a  # strings: the system policy's value stands
    merged = {key: pick(key) for key in _FIELDS}
    return replace(strict, sources=strict.sources + extra.sources, _set=strict._set | extra._set, **merged)


_CACHE: tuple[tuple, Policy] | None = None
_LOCK = threading.Lock()


def _stat(path: str) -> tuple | None:
    try:
        st = os.stat(path)
        return (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def current() -> Policy:
    """The effective policy, re-read when a policy file changes. Raises PolicyError when invalid."""
    global _CACHE
    paths = [system_policy_path()]
    explicit = os.environ.get(ENV_POLICY_FILE, "").strip()
    if explicit:
        explicit = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isfile(explicit):
            raise PolicyError(f"{ENV_POLICY_FILE} points to `{explicit}`, which does not exist")
        paths.append(explicit)
    key = tuple((p, _stat(p)) for p in paths)
    with _LOCK:
        if _CACHE is not None and _CACHE[0] == key:
            return _CACHE[1]
    result = Policy()
    for path, st in key:
        if st is not None:
            result = _merge(result, _parse(path)) if result.sources else _parse(path)
    with _LOCK:
        _CACHE = (key, result)
    return result


def current_or_none() -> tuple[Policy | None, str]:
    """(policy, "") or (None, error message): for callers that report instead of raising."""
    try:
        return current(), ""
    except PolicyError as e:
        return None, str(e)
