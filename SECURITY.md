# Security Policy

## Supported versions

Security fixes are released for the latest minor version on PyPI.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
**Security → Report a vulnerability** on
[github.com/Almogod/NovusPipeline](https://github.com/Almogod/NovusPipeline/security/advisories/new).
Do not open a public issue. You should receive a response within 7 days; please
include the version, your platform, and steps to reproduce.

## Security model

NovusPipeline runs locally, launched by your MCP client, with your user's
permissions. Its protections, and their limits:

- **The workspace is untrusted content.** Nothing from it is executed outside the
  sandbox, and what the tools return (file contents, findings) may contain text
  written to steer your assistant. The limits below are what holds if it does.
- **File access** is confined to the configured workspace; paths outside it, and
  anything under `.git` (hooks and config run code; config can hold credentials),
  are refused.
- **Test commands** must match an allowlist of test runners, may only name paths
  relative to the project, may not carry options that start another program,
  and run in a sandbox: a temporary copy of the project (or a disposable git
  worktree), a scrubbed environment with home/temp redirected, no stdin, memory
  and process limits, and the process tree killed on timeout. On Windows the run
  has a low integrity level and cannot write outside the sandbox. Without Docker
  (`NOVUS_SANDBOX=docker`), the sandbox is not a VM: tests run as your user, can
  read files by absolute path, can write to them on Linux and macOS, and have
  network access on Windows and macOS. Only modernize code you would run in
  your own terminal, or use Docker.
- **Git** commands run in the workspace with your git configuration. Do not
  point the server at a directory whose `.git` folder came from someone else
  (for example an unpacked archive): git itself runs programs named in
  `.git/config`.
- **Git**: the autonomous pipeline never modifies the working copy or the
  checked-out branch and never pushes. Tools that do change the working copy
  are annotated as destructive so clients ask before running them.
- **HTTP transport** authenticates callers with hashed API tokens or OIDC
  (`NOVUS_AUTH_TOKENS_FILE`, `NOVUS_AUTH_TOKEN`, `NOVUS_OIDC_*`) and limits each
  to its role (`viewer`, `developer`, `admin`, following the tools' MCP
  annotations). Without authentication it only serves loopback addresses, where
  it validates `Host` and `Origin` headers against DNS rebinding; any other
  address is refused unless `--allow-unauthenticated` is passed for a gateway
  that authenticates. It speaks plain HTTP: terminate TLS in front of it.
- **Organization policy** (`policy.toml` in the site config directory) binds
  this installation: it cannot be loosened by projects, user settings or
  environment variables, and an invalid policy makes every tool refuse to run.
  It is not a boundary against a user with shell access who installs another
  copy.
- **Audit log**: when configured, every tool call (including refused ones) is
  appended as a hash-chained JSON line; `novuspipeline audit-log verify` detects
  edited, removed or inserted records. Ship the log off the host to protect it
  against wholesale deletion.
- **Models**: the GNN checkpoint ships in the package. The optional LLM is
  downloaded from Hugging Face on first use unless `NOVUS_LLM_OFFLINE=1` (or the
  policy's `llm_offline`); no code is sent to any service, and ChromaDB telemetry
  is disabled.

Deployment guidance for organizations is in [docs/enterprise.md](docs/enterprise.md).
