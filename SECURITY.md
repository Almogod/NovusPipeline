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
- **HTTP transport** has no authentication. Keep it on `127.0.0.1` (the default);
  the server warns when bound to another address. On a loopback address it
  validates `Host` and `Origin` headers, which stops DNS-rebinding requests from
  a web page; bound to any other address it does not.
- **Models**: the GNN checkpoint ships in the package. The optional LLM is
  downloaded from Hugging Face on first use unless `NOVUS_LLM_OFFLINE=1`; no
  code is sent to any service.
