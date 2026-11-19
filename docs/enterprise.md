# Deploying NovusPipeline in an organization

This guide is for the platform, security and developer-tooling teams who roll NovusPipeline
out to many developers or run it as a shared service. It covers authentication and roles,
the organization policy, the audit log, the container image, network egress and supply-chain
verification. The [README](../README.md) covers day-to-day use.

## Deployment models

| Model | How it runs | Identity | Use it for |
|---|---|---|---|
| **Per developer (stdio)** | Each MCP client launches `novuspipeline` on the developer's machine | The OS user | The default; nothing to operate |
| **Managed workstation** | As above, plus a system-wide policy file distributed by MDM / configuration management | The OS user | Enforcing organization rules on every developer machine |
| **Shared server (HTTP)** | One server per repository (container or VM), clients connect over HTTP | Bearer tokens or OIDC, with roles | CI bots, central modernization campaigns, agents without local checkouts |

One server process serves **one workspace** (repository). To serve several repositories,
run one process (container) per repository behind a reverse proxy, with a route each.

## Authentication and roles (HTTP transport)

Stdio and terminal use are local: the caller is the OS user, with every role. Over HTTP,
every request must carry `Authorization: Bearer <token>`. The server **refuses to start**
on any address other than loopback without authentication (`--allow-unauthenticated`
overrides this, only for a gateway that authenticates in front of it). The policy setting
`require_auth = true` also requires authentication on loopback.

### Roles

Roles follow the tools' MCP annotations, so new tools are classified automatically.

| Role | Scope | Can call |
|---|---|---|
| `viewer` | `novus:read` | Read-only tools: overview, audits, file context, guideline search, status |
| `developer` | `novus:write` | + tools that write only to `.novus_index/`, a sandbox, rule files or a branch other than the checked-out one: the autonomous pipeline, test runs, indexing, standards and guideline edits |
| `admin` | `novus:admin` | + destructive tools: `apply_code_modernization`, `create_git_migration_pr`, `finalize_git_migration_pr`, `reset_rag_database`, `remove_rag_document`, `remove_modernization_rule` |

A caller only sees the tools their role allows (`tools/list` is filtered). A call
outside their role is refused and recorded in the audit log as `denied`.

### API tokens

```sh
export NOVUS_AUTH_TOKENS_FILE=/etc/novuspipeline/tokens.toml
novuspipeline token create --name ci-bot --role developer --expires 2027-06-30 --append "$NOVUS_AUTH_TOKENS_FILE"
```

The token is printed once. The file stores only its SHA-256 hash; comparison is
constant-time; expired tokens are rejected. Revoke a token by deleting its entry and
restarting the server.

```toml
[[token]]
name = "ci-bot"            # recorded as the principal in the audit log
sha256 = "9f86d081884c7d65…"
role = "developer"
expires = 2027-06-30       # optional; valid through the end of that day (UTC)
```

`NOVUS_AUTH_TOKEN=<token>` is a single shared admin token for small setups (at least 24
characters). Prefer the tokens file: it gives each caller its own identity in the audit log.

### OIDC / SSO

Tokens issued by your identity provider (Entra ID, Okta, Auth0, Keycloak, Google, …):

| Variable | Value |
|---|---|
| `NOVUS_OIDC_JWKS_URI` | The provider's JWKS endpoint (https) |
| `NOVUS_OIDC_ISSUER` | Expected `iss` |
| `NOVUS_OIDC_AUDIENCE` | Expected `aud` (the API you registered for NovusPipeline) |
| `NOVUS_OIDC_ALGORITHM` | Signing algorithm, default `RS256` |

All three of the JWKS URI, issuer and audience are required; without an audience check a
token issued for another application would be accepted. Grant the scopes `novus:read`,
`novus:write` or `novus:admin` through your provider's app roles or scope configuration.
The audited principal is `preferred_username`, else `email`, else `sub`.

API tokens and OIDC can be enabled together (for example OIDC for people, tokens for CI).

### TLS

The server speaks plain HTTP. Off loopback, terminate TLS in front of it (a reverse proxy,
ingress controller or service mesh), or bearer tokens cross the network in the clear.

## Organization policy

A TOML file that a project, a user setting or an environment variable cannot loosen.

| Location | Platform |
|---|---|
| `/etc/xdg/novuspipeline/policy.toml` | Linux (and the container image) |
| `C:\ProgramData\novuspipeline\policy.toml` | Windows |
| `/Library/Application Support/novuspipeline/policy.toml` | macOS |
| `NOVUS_POLICY_FILE` | Any: adds restrictions to the system file, cannot lift them |

```toml
# Tools nobody may call (hidden from tools/list and refused everywhere, including the terminal)
disabled_tools = ["reset_rag_database"]
# Also disable every destructive tool (the working copy, the checked-out branch, deletions)
allow_destructive_tools = false
# The pipeline never commits a file that no test ran against
allow_untested = false
# No local LLM proposals, or no model downloads
allow_llm = true
llm_offline = true
# Tests only run in a container with no network; fail instead of falling back
require_docker_sandbox = true
# Rule packs that are always active and that projects cannot override or weaken:
# built-in names (baseline, secure-defaults, python-312-ready) or absolute paths
required_packs = ["baseline", "secure-defaults", "/etc/xdg/novuspipeline/packs/acme.toml"]
# Commits only go to branches with this prefix (match your branch protection rules)
branch_prefix = "novus/"
# Audit log location; the environment cannot redirect or turn it off
audit_log = "/var/log/novuspipeline/audit.jsonl"
# The HTTP transport needs authentication even on loopback
require_auth = true
```

When both files exist, the most restrictive value wins: `allow_*` settings combine with
AND, `require_*` and `llm_offline` with OR, `disabled_tools` and `required_packs` are
combined, and the system file's `branch_prefix` and `audit_log` stand.

**Fail closed.** A policy file that cannot be parsed, has an unknown key (a typo such as
`allow_untestd`) or a value of the wrong type makes every tool refuse to run, the server
refuse to start, and `/healthz` report 503. It is never silently ignored.

`novuspipeline policy` prints the effective policy, its sources, the authentication
methods and the audit log, and exits non-zero when any of them is invalid. Run it after
rolling out a change.

Ruff codes that a required pack lists under `[ruff] block` can still be suppressed on a
single reviewed line with Ruff's `# noqa: <code>` (for example `# noqa: S104`); look for
new `noqa` comments in code review.

**Scope of enforcement.** The policy binds this installation: every MCP call, terminal
command and guided run. It is a guardrail for managed machines and shared servers, not a
boundary against a user with shell access who installs another copy. On developer machines,
pair it with your usual software controls.

## Audit log

Enable it with the policy's `audit_log` (enforced) or `NOVUS_AUDIT_LOG`. Every tool call
from any surface (MCP stdio, MCP HTTP, terminal commands, guided mode) appends one JSON
line, including calls refused by role or policy:

```json
{"ts":"2026-11-18T16:46:52.123Z","id":"5c0f…","event":"tool_call","tool":"run_autonomous_modernization_pipeline",
 "kind":"write","outcome":"ok","principal":"ci-bot","auth":"token","transport":"mcp-http",
 "workspace":"/workspace","args":{"file_path":"app/db.py","branch_name":"novus/db","use_llm":false},
 "scopes":["novus:read","novus:write"],"summary":"Verification: ✅ 12 test(s) executed and passed",
 "duration_ms":8123,"version":"0.1.0","host":"novus-billing-7c9","pid":7,"prev":"a3f1…"}
```

| Field | Meaning |
|---|---|
| `outcome` | `ok`, `error` (the tool reported a failure or raised) or `denied` (role or policy) |
| `principal`, `auth` | Who: token name, OIDC user, or the OS user for `local` |
| `transport` | `mcp-stdio`, `mcp-http` or `cli` |
| `args` | Arguments; values over 200 characters (file contents, proposed code, PR bodies) are replaced by `{"sha256", "chars"}` |
| `summary` | First line of a write/destructive tool's result, or of an error; read tools' results (which contain code) are never logged |
| `prev` | SHA-256 of the previous line: the hash chain |

Calls a tool makes to other tools internally are part of the one record.

**Tamper evidence.** `novuspipeline audit-log verify [PATH]` checks the chain and reports
the first record that was edited, removed or inserted. The chain detects changes inside the
file; to protect against someone deleting or rewriting the whole file, ship it off the host
(Fluent Bit, Vector, the Splunk or Datadog agent tailing the JSONL file) and keep it on
storage the service account can only append to.

Appends take an OS file lock, so several server processes may share one log. The file is
created with mode 0600. The server refuses to start if the configured log is not writable,
so a deployment is never silently unaudited. To rotate, move the file away (not
`copytruncate`): the next record creates a new file, whose chain starts again.

## Container image

```sh
docker run -d --name novus-billing -p 127.0.0.1:8000:8000 \
  -v /srv/repos/billing:/workspace \
  -v /etc/novuspipeline:/etc/xdg/novuspipeline:ro \
  -v /var/log/novuspipeline:/var/log/novuspipeline \
  -e NOVUS_AUTH_TOKENS_FILE=/etc/xdg/novuspipeline/tokens.toml \
  ghcr.io/almogod/novuspipeline:0.1
```

- Runs as an unprivileged user (uid 10001); the workspace must be writable by it for
  commits to the pipeline branch.
- Logs are JSON on stderr (`NOVUS_LOG_FORMAT=json`, `NOVUS_LOG_LEVEL=INFO`).
- `GET /healthz` is the liveness and readiness probe. It is unauthenticated and returns only
  status, version and whether auth and a policy are active: `200` when ready, `503` when the
  policy is invalid or the workspace is missing.
- Model downloads are off (`NOVUS_LLM_OFFLINE=1`). Build with `--build-arg EXTRAS=gnn` for
  the structural GNN (CPU torch).
- Git commits are authored as `NovusPipeline <novuspipeline@localhost>`; set
  `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`, `GIT_COMMITTER_NAME` and `GIT_COMMITTER_EMAIL` to change it.
- Inside the container, tests run in the process sandbox (temporary copy, scrubbed
  environment, limits; no network when the runtime allows user namespaces). The container
  is the isolation boundary: do not mount the Docker socket into it, and do not set
  `require_docker_sandbox` for a containerized server (there is no Docker inside it, so
  every test run would be refused). Restrict the container's own network instead, for
  example with a Kubernetes NetworkPolicy that denies egress.

Kubernetes: run one Deployment per repository with the repository on a PersistentVolume,
the policy and tokens file from a ConfigMap and a Secret mounted at
`/etc/xdg/novuspipeline`, `httpGet: {path: /healthz, port: 8000}` probes, and TLS at the
ingress.

## Network egress and air-gapped installs

With the core install nothing leaves the machine: Ruff, the rule engine, the GNN
checkpoint (bundled) and the guideline store (offline TF-IDF embeddings, ChromaDB
telemetry disabled) all run locally. The only outbound connections are:

| Connection | When | Turn off |
|---|---|---|
| Hugging Face model download | First use of the optional LLM (`[llm]` extra) | `llm_offline = true` (policy) or `NOVUS_LLM_OFFLINE=1`, or `allow_llm = false` |
| Your OIDC provider's JWKS endpoint | OIDC is configured | Do not configure OIDC |
| Whatever your tests do | Test runs | `require_docker_sandbox = true` (no network) |

Air-gapped install: on a connected machine `pip download "novuspipeline[gnn]" -d wheels`
(add `--index-url https://download.pytorch.org/whl/cpu` for torch on Linux), copy
`wheels/` across, then `pip install --no-index --find-links wheels "novuspipeline[gnn]"`.
For the LLM, copy a populated Hugging Face cache and set `NOVUS_LLM_OFFLINE=1`.

## Verifying releases

Every release ships:

- PyPI packages with PyPI's trusted-publishing attestations.
- The wheel and sdist on the GitHub release with signed build provenance:
  `gh attestation verify novuspipeline-X.Y.Z-py3-none-any.whl --repo Almogod/NovusPipeline`.
- A CycloneDX SBOM of a core install (`novuspipeline-vX.Y.Z.cdx.json`) on the GitHub release.
- The container image with provenance and SBOM attestations:
  `gh attestation verify oci://ghcr.io/almogod/novuspipeline:X.Y.Z --repo Almogod/NovusPipeline`.

CI runs CodeQL (Python, JavaScript/TypeScript, workflows), `pip-audit` against the
resolved dependencies, the test suite on Linux, Windows and macOS with Python 3.11–3.13,
and a container smoke test that checks the image refuses to start without authentication.

## Limitations

- One workspace per server process; no multi-tenancy inside a process.
- No built-in rate limiting or TLS: put a reverse proxy in front of a shared server.
- Token changes take effect on restart.
- Without Docker, the test sandbox is not a VM (see [SECURITY.md](../SECURITY.md)).
