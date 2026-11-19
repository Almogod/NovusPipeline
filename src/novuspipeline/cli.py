"""
Command line interface: the same tools the MCP server offers, runnable from a terminal.

    novuspipeline overview            codebase-wide risk overview
    novuspipeline audit app/db.py     findings for one file
    novuspipeline modernize app/db.py rewrite, test, commit on a branch (test-gated)
    novuspipeline                     in a terminal: guided mode (paste a folder, it fixes it)
                                      launched by an MCP client: the MCP server
    novuspipeline fix PATH --yes      fix a whole project without questions

Every command calls the same function the matching MCP tool does, so a result here is the
result an AI assistant would get. Importing the server (and with it the ML stack) is
deferred until a command actually runs, so `--help` and `--version` stay instant.
"""
import argparse
import os
import re
import sys
import warnings

from . import __version__

# Result prefixes the tools use to report that they did not do what was asked.
_FAILURE_PREFIXES = ("Error", "❌", "Pipeline Aborted")

_EPILOG = """\
With no command: in a terminal, the guided mode (`start`); launched by an MCP client (no
terminal), the MCP server over stdio, so existing MCP client configs keep working.
Run `novuspipeline <command> --help` for a command's options."""


def _workspace_parent() -> argparse.ArgumentParser:
    """`--workspace` is accepted before or after the command."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--workspace", default=argparse.SUPPRESS,
                        help="codebase to work on (default: NOVUS_WORKSPACE_ROOT or the current directory)")
    return parent


def _server_parent(suppress: bool) -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)

    def default(value):
        return argparse.SUPPRESS if suppress else value

    parent.add_argument("--transport", choices=["stdio", "http"], default=default("stdio"),
                        help="stdio for MCP clients that launch the server (default); http to serve locally")
    parent.add_argument("--host", default=default("127.0.0.1"),
                        help="http only; any address but loopback needs authentication (docs/enterprise.md)")
    parent.add_argument("--port", type=int, default=default(8000), help="http only")
    parent.add_argument("--allow-unauthenticated", action="store_true", default=default(False),
                        help="http only: serve a non-loopback address without authentication "
                             "(only behind a gateway that authenticates)")
    return parent


def build_parser() -> argparse.ArgumentParser:
    ws = _workspace_parent()
    parser = argparse.ArgumentParser(
        prog="novuspipeline", epilog=_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Modernize legacy codebases safely: audit, rewrite, test in a sandbox, commit on a branch.",
        parents=[_server_parent(suppress=False)])
    parser.add_argument("--workspace", default=None,
                        help="codebase to work on (default: NOVUS_WORKSPACE_ROOT or the current directory)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_text, description=help_text, parents=[ws])

    add("start", "guided mode: paste a project folder, then it scans and fixes it (the default in a terminal)")

    p = add("fix", "fix every file that can be fixed automatically, test-gated, on a new branch")
    p.add_argument("path", nargs="?", default=None, help="project folder (asked for when omitted in a terminal)")
    p.add_argument("-y", "--yes", action="store_true", help="no questions: fix on a branch, never touch your files")
    p.add_argument("--branch", default="", help="branch for the fixes (default: a new novus/modernize-<time>)")
    p.add_argument("--test-command", default="", metavar="CMD",
                   help="verification command for every file (default: the tests that import each file)")
    p.add_argument("--llm", action="store_true", help="try the local LLM first (needs the [llm] extra)")
    p.add_argument("--merge", action="store_true", help="fast-forward your current branch to the fixes at the end")
    p.add_argument("--include-tests", action="store_true", help="also rewrite test files (off: tests are the safety net)")
    p.add_argument("--allow-untested", action="store_true",
                   help="commit files no test covers, after a compile/import check (review the diff yourself)")

    sub.add_parser("serve", help="start the MCP server (the default when launched by an MCP client)",
                   description="Start the MCP server.", parents=[ws, _server_parent(suppress=True)])

    p = add("overview", "files ranked by modernization risk, codebase-level smells, hotspots")
    p.add_argument("--top", type=int, default=10, metavar="N", help="files to list (default 10)")

    p = add("index", "build or refresh the codebase index")
    p.add_argument("--force", action="store_true", help="rebuild from scratch")

    p = add("audit", "findings for one file (your standards, rules, Ruff, GNN) with guidelines")
    p.add_argument("file")

    p = add("context", "who imports a file, which of its symbols they use, which tests cover it")
    p.add_argument("file")

    p = add("gnn", "structural smell prediction for a Python file (needs the [gnn] extra)")
    p.add_argument("file")

    p = add("similar", "structurally similar code elsewhere (needs the [gnn] extra)")
    p.add_argument("file")
    p.add_argument("--function", default="", metavar="NAME", help="function to compare (Class.method for methods)")
    p.add_argument("-n", "--results", type=int, default=5, metavar="N")

    p = add("modernize", "rewrite a file in a disposable worktree, test it, commit on a branch only if tests pass")
    p.add_argument("file")
    p.add_argument("--test-command", default="", metavar="CMD",
                   help="verification command (default: the tests that import the file)")
    p.add_argument("--branch", default="auto-modernization-branch", help="branch that receives the commit")
    p.add_argument("--llm", action="store_true", help="try the local LLM first (needs the [llm] extra)")

    p = add("apply", "rewrite a file IN PLACE in your working copy (writes a .bak); prefer `modernize`")
    p.add_argument("file")
    p.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")

    p = add("test", "run a test command in the sandbox")
    p.add_argument("test_command", nargs=argparse.REMAINDER, metavar="command",
                   help="everything after `test` is the command, e.g. `test python -m pytest tests/`")

    p = add("history", "learn modernization rules from this repository's git history")
    p.add_argument("--max-commits", type=int, default=2000, metavar="N")
    p.add_argument("--min-support", type=int, default=None, metavar="N", help="separate commits needed (default 3)")

    p = add("standards", "list, install or remove modernization rule packs")
    ssub = p.add_subparsers(dest="standards_command", metavar="<action>", required=True)
    ssub.add_parser("list", help="show every active rule and the built-in packs", parents=[ws])
    q = ssub.add_parser("install", help="copy a built-in pack into the project", parents=[ws])
    q.add_argument("name", help="secure-defaults or python-312-ready")
    q.add_argument("--user", action="store_true", help="install for all your projects")
    q = ssub.add_parser("remove", help="remove a rule added with add_modernization_rule", parents=[ws])
    q.add_argument("rule_id")
    q.add_argument("--user", action="store_true")

    p = add("guidelines", "search the guideline database")
    gsub = p.add_subparsers(dest="guidelines_command", metavar="<action>", required=True)
    q = gsub.add_parser("search", help="find guidelines matching a query", parents=[ws])
    q.add_argument("query", nargs="+")
    q.add_argument("--category", default="")
    q.add_argument("-n", "--results", type=int, default=3, metavar="N")
    q = gsub.add_parser("show", help="show one guideline by id (e.g. py-001)", parents=[ws])
    q.add_argument("document_id")
    gsub.add_parser("stats", help="database statistics", parents=[ws])

    p = add("status", "GNN checkpoint and local LLM status")
    p.add_argument("--probe", action="store_true", help="load the LLM to report device and VRAM (slow)")

    sub.add_parser("policy", help="show the effective organization policy, authentication and audit settings",
                   description="Show the effective organization policy, authentication and audit settings.")

    p = sub.add_parser("token", help="create API tokens for the HTTP transport",
                       description="Create API tokens for the HTTP transport (NOVUS_AUTH_TOKENS_FILE).")
    tsub = p.add_subparsers(dest="token_command", metavar="<action>", required=True)
    q = tsub.add_parser("create", help="generate a token and the entry to add to the tokens file")
    q.add_argument("--name", required=True, help="who or what uses the token (recorded in the audit log)")
    q.add_argument("--role", choices=["viewer", "developer", "admin"], default="developer",
                   help="viewer: read-only tools; developer: + branch commits and test runs (default); "
                        "admin: + destructive tools")
    q.add_argument("--expires", default="", metavar="YYYY-MM-DD", help="last day the token is valid")
    q.add_argument("--append", default="", metavar="FILE", help="append the entry to this tokens file")

    p = sub.add_parser("audit-log", help="check the audit log's hash chain",
                       description="Check that no audit record was edited, removed or inserted.")
    asub = p.add_subparsers(dest="audit_command", metavar="<action>", required=True)
    q = asub.add_parser("verify", help="verify the hash chain")
    q.add_argument("path", nargs="?", default="", help="log file (default: the configured audit log)")
    return parser


def _confirm(prompt: str) -> bool:
    if not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _dispatch(srv, args: argparse.Namespace) -> str:
    """Call the tool that backs `args.command`. `srv` is the server module."""
    cmd = args.command
    if cmd == "overview":
        return srv.get_codebase_overview(args.top)
    if cmd == "index":
        return srv.index_codebase(args.force)
    if cmd == "audit":
        return srv.analyze_legacy_codebase(args.file)
    if cmd == "context":
        return srv.get_file_context(args.file)
    if cmd == "gnn":
        return srv.analyze_code_structure_gnn(args.file)
    if cmd == "similar":
        return srv.find_similar_code(args.file, args.function, args.results)
    if cmd == "modernize":
        return srv.run_autonomous_modernization_pipeline(args.file, args.test_command, args.branch, args.llm)
    if cmd == "apply":
        if not args.yes and not _confirm(f"Rewrite {args.file} in place (a .bak copy is kept)?"):
            return "Error: not applied. Confirm at the prompt, or pass --yes (required when not run interactively)."
        return srv.apply_code_modernization(args.file)
    if cmd == "test":
        if not args.test_command:
            return "Error: give a test command, e.g. `novuspipeline test python -m pytest`."
        return srv.run_local_tests(" ".join(args.test_command))
    if cmd == "history":
        kwargs = {} if args.min_support is None else {"min_support": args.min_support}
        return srv.learn_from_history(args.max_commits, **kwargs)
    if cmd == "standards":
        action = args.standards_command
        scope = "user" if getattr(args, "user", False) else "project"
        if action == "list":
            return srv.list_modernization_standards()
        if action == "install":
            return srv.install_standard_pack(args.name, scope)
        return srv.remove_modernization_rule(args.rule_id, scope)
    if cmd == "guidelines":
        action = args.guidelines_command
        if action == "search":
            return srv.query_rag_guidelines(" ".join(args.query), args.category, args.results)
        if action == "show":
            return srv.search_rag_by_id(args.document_id)
        return srv.get_rag_stats()
    if cmd == "status":
        return f"{srv.get_gnn_model_status()}\n\n{srv.get_local_llm_status(args.probe)}"
    raise AssertionError(f"unhandled command {cmd!r}")


def _use_utf8_output() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)  # torch.jit deprecation noise from the GNN
    # Results contain symbols that a legacy Windows code page cannot encode.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def _set_workspace(server, workspace: str | None, parser: argparse.ArgumentParser) -> None:
    if workspace:
        server.WORKSPACE_ROOT = os.path.abspath(workspace)
    if not os.path.isdir(server.WORKSPACE_ROOT):
        parser.error(f"workspace '{server.WORKSPACE_ROOT}' is not a directory")


def _exit_code(command: str, result: str) -> int:
    """0 on success. `test` also fails when the tests did (the tool reports `Exit Code: N` at the end)."""
    if result.lstrip().startswith(_FAILURE_PREFIXES):
        return 1
    if command == "test":
        codes = re.findall(r"^Exit Code: (\S+)", result, flags=re.MULTILINE)
        if codes and codes[-1] != "0":
            return 1
    return 0


def _in_terminal() -> bool:
    """A person at a terminal. MCP clients launch the server with pipes, never a terminal."""
    return bool(sys.stdin and sys.stdin.isatty() and sys.stdout and sys.stdout.isatty())


def _token_create(args: argparse.Namespace) -> int:
    import datetime

    from . import auth
    if args.expires:
        try:
            datetime.date.fromisoformat(args.expires)
        except ValueError:
            print("Error: --expires must be a date, e.g. 2027-06-30", file=sys.stderr)
            return 2
    token = auth.new_token()
    entry = auth.token_entry_toml(args.name, token, args.role, args.expires)
    if args.append:
        path = os.path.abspath(os.path.expanduser(args.append))
        new = not os.path.exists(path)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(("" if new else "\n") + entry + "\n")
        print(f"Added `{args.name}` ({args.role}) to {path}. The server picks it up on restart.", file=sys.stderr)
    else:
        print("Add this entry to the file NOVUS_AUTH_TOKENS_FILE points to:\n", file=sys.stderr)
        print(entry)
    print(f"\nToken for `{args.name}` (shown once; only its hash is stored):\n  {token}\n"
          f"Clients send it as `Authorization: Bearer <token>`.", file=sys.stderr)
    return 0


def _policy_report() -> int:
    from . import audit, auth, policy
    lines, failed = [], False
    try:
        lines.append(policy.current().describe())
    except policy.PolicyError as e:
        lines.append(f"Error: {e}\n  Every tool refuses to run until the policy file is fixed.")
        failed = True
    lines.append(f"  (system policy location: {policy.system_policy_path()})")
    try:
        _, methods = auth.build_verifier()
        lines.append("HTTP authentication: " + ("; ".join(methods) if methods else
                                                "none (only loopback addresses can be served)"))
    except auth.AuthConfigError as e:
        lines.append(f"HTTP authentication: Error: {e}")
        failed = True
    if not failed:
        log = audit.log_path()
        problem = audit.check_writable(log) if log else None
        lines.append(f"Audit log: {log}" + (f" ({problem})" if problem else "") if log else
                     f"Audit log: off (set {audit.ENV_AUDIT_LOG} or the policy's audit_log)")
        failed = bool(problem)
    print("\n".join(lines))
    return 1 if failed else 0


def _audit_verify(path: str) -> int:
    from . import audit, policy
    if not path:
        try:
            path = audit.log_path()
        except policy.PolicyError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        if not path:
            print(f"Error: no audit log configured; pass a path or set {audit.ENV_AUDIT_LOG}.", file=sys.stderr)
            return 1
    ok, message = audit.verify(path)
    print(f"{path}: {message}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Administration commands: no workspace, no ML stack.
    if args.command == "token":
        return _token_create(args)
    if args.command == "policy":
        return _policy_report()
    if args.command == "audit-log":
        return _audit_verify(args.path)
    guided = args.command == "start" or (args.command is None and args.transport == "stdio" and _in_terminal())

    from . import server  # heavy: FastMCP, ChromaDB

    if guided or args.command == "fix":
        from . import interactive
        _use_utf8_output()
        if args.command == "fix":
            path = args.path or args.workspace
            if path is None and not _in_terminal():
                path = os.getcwd()
            return interactive.run(server, path, assume_yes=args.yes, branch=args.branch,
                                   test_command=args.test_command, use_llm=args.llm, merge=args.merge,
                                   include_tests=args.include_tests, allow_untested=args.allow_untested)
        return interactive.run(server, args.workspace)

    _set_workspace(server, args.workspace, parser)

    if args.command in (None, "serve"):
        server.run_server(args.transport, args.host, args.port, args.allow_unauthenticated)
        return 0

    _use_utf8_output()
    try:
        result = _dispatch(server, args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    print(result)
    return _exit_code(args.command, result)


if __name__ == "__main__":
    sys.exit(main())
