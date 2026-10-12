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

- **File access** is confined to the configured workspace; paths outside it are refused.
- **Test commands** must match an allowlist of test runners and run in a sandbox:
  a temporary copy of the project (or a disposable git worktree), a scrubbed
  environment with home/temp redirected, memory and process limits, and the
  process tree killed on timeout. Without Docker (`NOVUS_SANDBOX=docker`), the
  sandbox is not a VM: tests run as your user, can read files by absolute path,
  and have network access on Windows and macOS. Only modernize code you would
  run in your own terminal, or use Docker.
- **Git**: the autonomous pipeline never modifies the working copy or the
  checked-out branch and never pushes. Tools that do change the working copy
  are annotated as destructive so clients ask before running them.
- **HTTP transport** has no authentication. Keep it on `127.0.0.1` (the default);
  the server warns when bound to another address.
- **Models**: the GNN checkpoint ships in the package. The optional LLM is
  downloaded from Hugging Face on first use unless `NOVUS_LLM_OFFLINE=1`; no
  code is sent to any service.
