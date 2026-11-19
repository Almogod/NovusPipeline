"""Organization policy, audit log, authentication and role-based access."""
import asyncio
import datetime
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from novuspipeline import audit, auth, cli, interactive, policy, standards
from test_server import LEGACY_MODULE, WorkspaceTestCase, server  # also applies the test environment settings


class PolicyTestCase(WorkspaceTestCase):
    """A workspace with no system policy and an explicit policy file the test controls."""

    def setUp(self) -> None:
        super().setUp()
        self.admin = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.admin.cleanup)
        self.system_policy = os.path.join(self.admin.name, "system-policy.toml")
        self.policy_file = os.path.join(self.admin.name, "policy.toml")
        for patcher in (mock.patch.object(policy, "system_policy_path", return_value=self.system_policy),
                        mock.patch.dict(os.environ, {"NOVUS_AUDIT_LOG": ""})):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop(policy.ENV_POLICY_FILE, None)
        self.addCleanup(os.environ.pop, policy.ENV_POLICY_FILE, None)
        policy._CACHE = None
        self.addCleanup(setattr, policy, "_CACHE", None)

    def set_policy(self, text: str, path: str = "") -> None:
        path = path or self.policy_file
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        if path == self.policy_file:
            os.environ[policy.ENV_POLICY_FILE] = path
        policy._CACHE = None

    def audit_records(self, path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


class TestPolicy(PolicyTestCase):

    def test_no_policy_file_means_defaults(self):
        active = policy.current()
        self.assertEqual(active.sources, ())
        self.assertIsNone(active.tool_denial("reset_rag_database", "destructive"))

    def test_disabled_and_destructive_tools_are_refused(self):
        self.set_policy('disabled_tools = ["get_rag_stats"]\nallow_destructive_tools = false\n')
        self.assertIn("disabled by organization policy", server.get_rag_stats())
        self.assertTrue(server.reset_rag_database().startswith("Error:"))
        self.assertIn("temp workspace", server.read_legacy_file("README.md"))  # everything else still works

    def test_invalid_policy_fails_closed(self):
        self.set_policy('allow_untestd = false\n')  # typo: must not be silently ignored
        res = server.read_legacy_file("README.md")
        self.assertTrue(res.startswith("Error: refused"), res)
        self.assertIn("allow_untestd", res)
        self.set_policy('allow_llm = "no"\n')
        self.assertIn("must be bool", server.read_legacy_file("README.md"))

    def test_missing_explicit_policy_file_fails_closed(self):
        os.environ[policy.ENV_POLICY_FILE] = os.path.join(self.admin.name, "missing.toml")
        self.assertTrue(server.read_legacy_file("README.md").startswith("Error: refused"))

    def test_explicit_policy_cannot_loosen_the_system_policy(self):
        self.set_policy('allow_untested = false\nbranch_prefix = "novus/"\ndisabled_tools = ["a"]\n',
                        self.system_policy)
        self.set_policy('allow_untested = true\nrequire_docker_sandbox = true\nbranch_prefix = ""\n'
                        'disabled_tools = ["b"]\n')
        active = policy.current()
        self.assertFalse(active.allow_untested)
        self.assertTrue(active.require_docker_sandbox)
        self.assertEqual(active.branch_prefix, "novus/")
        self.assertEqual(active.disabled_tools, {"a", "b"})
        self.assertEqual(len(active.sources), 2)

    def test_pipeline_honors_allow_untested_llm_and_branch_prefix(self):
        self.write("calc.py", "def add(a, b):\n    return a + b\n")
        self.commit_all()
        self.set_policy('allow_untested = false\nallow_llm = false\nbranch_prefix = "novus/"\n')
        res = server.run_autonomous_modernization_pipeline("calc.py", branch_name="novus/x", allow_untested=True)
        self.assertIn("does not allow committing untested files", res)
        res = server.run_autonomous_modernization_pipeline("calc.py", branch_name="novus/x", use_llm=True)
        self.assertIn("does not allow LLM", res)
        res = server.run_autonomous_modernization_pipeline("calc.py", branch_name="feature")
        self.assertIn("branches starting with `novus/`", res)
        self.assertNotIn("feature", [b.name for b in self.repo.branches])
        self.assertIn("does not allow LLM", server.generate_llm_modernization_proposal("calc.py"))

    def test_guided_mode_refuses_what_policy_forbids_and_uses_the_branch_prefix(self):
        self.set_policy('allow_untested = false\nbranch_prefix = "ai/"\n')
        out = io.StringIO()
        with redirect_stdout(out):
            code = interactive.run(server, self.root, assume_yes=True, allow_untested=True)
        self.assertEqual(code, 2)
        self.assertIn("--allow-untested", out.getvalue())
        from novuspipeline import fixer
        self.assertTrue(fixer.default_branch_name().startswith("ai/modernize-"))

    def test_required_pack_cannot_be_overridden_by_the_project(self):
        self.write(".novus/standards/weak.toml",
                   '[[rule]]\nid = "SEC-NO-PICKLE"\ntitle = "weakened"\nimports = ["nothing_real"]\n')
        pack = os.path.join(self.admin.name, "org.toml")
        with open(pack, "w", encoding="utf-8") as f:
            f.write('[pack]\nname = "acme"\n[[rule]]\nid = "SEC-NO-PICKLE"\ntitle = "No pickle"\n'
                    'imports = ["pickle"]\nenforce = "block"\n')
        self.set_policy(f"required_packs = [{json.dumps(pack)}]\n")
        active = standards.load(self.root)
        rule = next(r for r in active.rules if r["id"] == "SEC-NO-PICKLE")
        self.assertEqual((rule["scope"], rule["enforce"], rule["imports"]), ("policy", "block", ["pickle"]))
        self.assertTrue(active.scan("import pickle\n", "app.py"))

    def test_required_pack_must_exist(self):
        self.set_policy('required_packs = ["no-such-pack"]\n')
        with self.assertRaises(policy.PolicyError):
            policy.current()

    def test_docker_sandbox_required_but_unavailable_refuses_to_run_tests(self):
        self.set_policy("require_docker_sandbox = true\n")
        with mock.patch("novuspipeline.sandbox.docker_available", return_value=False):
            res = server.run_local_tests("python -m unittest discover")
        self.assertTrue(res.startswith("Error: organization policy requires the Docker sandbox"), res)


class TestAuditLog(PolicyTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.log = os.path.join(self.admin.name, "logs", "audit.jsonl")

    def test_tool_calls_are_recorded_with_redacted_arguments_and_a_hash_chain(self):
        self.set_policy(f"audit_log = {json.dumps(self.log)}\n")
        server.read_legacy_file("README.md")
        server.apply_code_modernization("missing.py", modernized_code="x = 1\n" * 100)
        records = self.audit_records(self.log)
        self.assertEqual([r["tool"] for r in records], ["read_legacy_file", "apply_code_modernization"])
        first, second = records
        self.assertEqual((first["outcome"], first["kind"], first["auth"], first["transport"]), ("ok", "read", "local", "cli"))
        self.assertNotIn("summary", first)  # a read tool's result (file contents) is never logged
        self.assertEqual(second["outcome"], "error")
        self.assertEqual(set(second["args"]["modernized_code"]), {"sha256", "chars"})
        self.assertEqual(first["prev"], audit.GENESIS)
        ok, message = audit.verify(self.log)
        self.assertTrue(ok, message)

    def test_policy_audit_log_cannot_be_redirected_by_the_environment(self):
        self.set_policy(f"audit_log = {json.dumps(self.log)}\n")
        with mock.patch.dict(os.environ, {"NOVUS_AUDIT_LOG": os.path.join(self.admin.name, "elsewhere.jsonl")}):
            self.assertEqual(audit.log_path(), self.log)

    def test_nested_tool_calls_are_one_record(self):
        os.environ["NOVUS_AUDIT_LOG"] = self.log
        self.write("legacy.py", LEGACY_MODULE)
        server.analyze_legacy_codebase("legacy.py")  # calls other tools internally
        self.assertEqual([r["tool"] for r in self.audit_records(self.log)], ["analyze_legacy_codebase"])

    def test_denials_are_recorded(self):
        self.set_policy(f'audit_log = {json.dumps(self.log)}\ndisabled_tools = ["get_rag_stats"]\n')
        server.get_rag_stats()
        (record,) = self.audit_records(self.log)
        self.assertEqual(record["outcome"], "denied")
        self.assertIn("disabled", record["summary"])

    def test_tampering_breaks_the_chain(self):
        os.environ["NOVUS_AUDIT_LOG"] = self.log
        for _ in range(3):
            server.get_rag_stats()
        with open(self.log, encoding="utf-8") as f:
            lines = f.readlines()
        self.assertTrue(audit.verify(self.log)[0])
        with open(self.log, "w", encoding="utf-8") as f:
            f.writelines([lines[0], lines[2]])  # record 2 removed
        ok, message = audit.verify(self.log)
        self.assertFalse(ok)
        self.assertIn("line 2", message)

    def test_cli_verifies_the_configured_log(self):
        os.environ["NOVUS_AUDIT_LOG"] = self.log
        server.get_rag_stats()
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["audit-log", "verify"]), 0)
        self.assertIn("1 record(s), chain intact", out.getvalue())


class TestTokens(unittest.TestCase):

    def verify(self, verifier, token):
        return asyncio.run(verifier.verify_token(token))

    def test_hashed_tokens_roles_and_expiry(self):
        good, old = auth.new_token(), auth.new_token()
        yesterday = datetime.date.today() - datetime.timedelta(days=2)
        verifier = auth.HashedTokenVerifier([auth.TokenEntry("ci", auth.hash_token(good), "developer"),
                                             auth.TokenEntry("old", auth.hash_token(old), "admin", yesterday)])
        access = self.verify(verifier, good)
        self.assertEqual(access.client_id, "ci")
        self.assertTrue(auth.allows(access.scopes, "write"))
        self.assertFalse(auth.allows(access.scopes, "destructive"))
        self.assertIsNone(self.verify(verifier, old))
        self.assertIsNone(self.verify(verifier, good + "x"))

    def test_tokens_file_from_the_cli_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tokens.toml")
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                self.assertEqual(cli.main(["token", "create", "--name", "alice", "--role", "viewer",
                                           "--expires", "2099-01-01", "--append", path]), 0)
                cli.main(["token", "create", "--name", "bob", "--append", path])
            token = next(w for w in err.getvalue().split() if w.startswith("novus_"))
            entries = auth.load_tokens_file(path)
            self.assertEqual([(e.name, e.role) for e in entries], [("alice", "viewer"), ("bob", "developer")])
            with open(path, encoding="utf-8") as f:
                self.assertNotIn(token, f.read())  # only hashes are stored
            access = self.verify(auth.HashedTokenVerifier(entries), token)
            self.assertEqual(access.client_id, "alice")

    def test_bad_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tokens.toml")
            with open(path, "w", encoding="utf-8") as f:
                f.write('[[token]]\nname = "x"\nsha256 = "abc"\nrole = "viewer"\n')
            with self.assertRaises(auth.AuthConfigError):
                auth.load_tokens_file(path)
        with mock.patch.dict(os.environ, {auth.ENV_SHARED_TOKEN: "short"}), self.assertRaises(auth.AuthConfigError):
            auth.build_verifier()
        with mock.patch.dict(os.environ, {auth.ENV_OIDC_JWKS: "https://idp.example/jwks"}), \
                self.assertRaises(auth.AuthConfigError) as raised:
            auth.build_verifier()
        self.assertIn(auth.ENV_OIDC_AUDIENCE, str(raised.exception))


class TestStartup(PolicyTestCase):

    def setUp(self) -> None:
        super().setUp()
        for key in (auth.ENV_TOKENS_FILE, auth.ENV_SHARED_TOKEN, auth.ENV_OIDC_JWKS, auth.ENV_OIDC_ISSUER,
                    auth.ENV_OIDC_AUDIENCE):
            patcher = mock.patch.dict(os.environ, {key: ""})
            patcher.start()
            self.addCleanup(patcher.stop)
        for attr, value in (("auth", None), ("enabled", False)):
            target = server.mcp if attr == "auth" else server._ACCESS
            patcher = mock.patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(setattr, audit, "TRANSPORT", "cli")

    def test_non_loopback_without_auth_refuses_to_start(self):
        with self.assertRaises(server.StartupError):
            server.prepare_server("http", "0.0.0.0")
        notes = server.prepare_server("http", "0.0.0.0", allow_unauthenticated=True)
        self.assertTrue(any("without authentication" in n for n in notes))
        server.prepare_server("http", "127.0.0.1")
        server.prepare_server("stdio")

    def test_policy_can_require_auth_even_on_loopback(self):
        self.set_policy("require_auth = true\n")
        with self.assertRaises(server.StartupError):
            server.prepare_server("http", "127.0.0.1")
        server.prepare_server("stdio")  # local, the OS user

    def test_unwritable_audit_log_refuses_to_start(self):
        os.environ["NOVUS_AUDIT_LOG"] = self.root  # a directory
        with self.assertRaises(server.StartupError):
            server.prepare_server("stdio")

    def test_shared_token_enables_auth(self):
        os.environ[auth.ENV_SHARED_TOKEN] = auth.new_token()
        notes = server.prepare_server("http", "0.0.0.0")
        self.assertIsNotNone(server.mcp.auth)
        self.assertTrue(server._ACCESS.enabled)
        self.assertTrue(any("shared token" in n for n in notes))

    def test_health_endpoint(self):
        response = asyncio.run(server._healthz(None))
        body = json.loads(response.body)
        self.assertEqual((response.status_code, body["status"]), (200, "ok"))
        self.assertNotIn(self.root, response.body.decode())
        self.set_policy("nonsense = 1\n")
        self.assertEqual(asyncio.run(server._healthz(None)).status_code, 503)


class TestRoleBasedAccess(PolicyTestCase):
    """The middleware against an in-process MCP client, with the request's token patched in."""

    def call(self, role: str | None, action):
        from fastmcp import Client
        from fastmcp.server.auth import AccessToken
        token = None if role is None else AccessToken(token="t", client_id=f"{role}-bot", scopes=list(auth.ROLES[role]),
                                                      claims={"novus_auth": "token"})

        async def run():
            async with Client(server.mcp) as client:
                return await action(client)
        with mock.patch.object(server._ACCESS, "enabled", True), \
                mock.patch("fastmcp.server.dependencies.get_access_token", return_value=token):
            return asyncio.run(run())

    def test_tools_are_listed_by_role(self):
        names = lambda c: c.list_tools()  # noqa: E731
        viewer = {t.name for t in self.call("viewer", names)}
        developer = {t.name for t in self.call("developer", names)}
        admin = {t.name for t in self.call("admin", names)}
        self.assertIn("read_legacy_file", viewer)
        self.assertNotIn("run_local_tests", viewer)
        self.assertIn("run_autonomous_modernization_pipeline", developer)
        self.assertNotIn("apply_code_modernization", developer)
        self.assertIn("apply_code_modernization", admin)
        self.assertEqual(admin, set(server.TOOL_KINDS))

    def test_calls_outside_the_role_are_refused_and_audited(self):
        log = os.path.join(self.admin.name, "audit.jsonl")
        os.environ["NOVUS_AUDIT_LOG"] = log
        with self.assertRaises(Exception) as raised:
            self.call("viewer", lambda c: c.call_tool("reset_rag_database", {}))
        self.assertIn("not permitted", str(raised.exception))
        self.call("viewer", lambda c: c.call_tool("get_rag_stats", {}))
        denied, allowed = self.audit_records(log)
        self.assertEqual((denied["outcome"], denied["principal"], denied["tool"]), ("denied", "viewer-bot", "reset_rag_database"))
        self.assertEqual((allowed["outcome"], allowed["principal"], allowed["auth"]), ("ok", "viewer-bot", "token"))

    def test_policy_disabled_tools_are_hidden(self):
        self.set_policy('disabled_tools = ["get_rag_stats"]\n')
        names = {t.name for t in self.call("admin", lambda c: c.list_tools())}
        self.assertNotIn("get_rag_stats", names)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestHttpEndToEnd(WorkspaceTestCase):
    """A real server process: bearer tokens over HTTP, 401 without one, roles enforced."""

    def test_token_authentication_over_http(self):
        import httpx2 as httpx
        from fastmcp import Client
        from fastmcp.client.auth import BearerAuth

        viewer = auth.new_token()
        tokens = os.path.join(self.root, "..", f"tokens-{os.getpid()}.toml")
        with open(tokens, "w", encoding="utf-8") as f:
            f.write(auth.token_entry_toml("viewer-bot", viewer, "viewer") + "\n")
        self.addCleanup(os.remove, tokens)
        port = _free_port()
        env = dict(os.environ, NOVUS_AUTH_TOKENS_FILE=tokens, NOVUS_POLICY_FILE="", NOVUS_AUDIT_LOG="")
        proc = subprocess.Popen([sys.executable, "-m", "novuspipeline", "serve", "--workspace", self.root,
                                 "--transport", "http", "--host", "127.0.0.1", "--port", str(port)],
                                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.addCleanup(proc.wait, 10)
        self.addCleanup(proc.kill)
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 90
        while True:
            try:
                health = httpx.get(f"{base}/healthz", timeout=2)
                break
            except httpx.TransportError:
                if proc.poll() is not None or time.time() > deadline:
                    self.fail(f"server did not start: {proc.stderr.read().decode(errors='replace')}")
                time.sleep(0.5)
        self.assertEqual(health.json()["auth"], True)
        self.assertEqual(httpx.post(f"{base}/mcp", json={}, timeout=5).status_code, 401)

        async def session(token):
            async with Client(f"{base}/mcp", auth=BearerAuth(token)) as client:
                names = {t.name for t in await client.list_tools()}
                stats = await client.call_tool("get_rag_stats", {})
                try:
                    await client.call_tool("run_local_tests", {"command": "python -m unittest"})
                    refused = ""
                except Exception as e:
                    refused = str(e)
                return names, stats, refused
        names, stats, refused = asyncio.run(session(viewer))
        self.assertIn("get_rag_stats", names)
        self.assertNotIn("run_local_tests", names)
        self.assertFalse(stats.is_error)
        self.assertIn("not permitted", refused)
        try:
            asyncio.run(session(auth.new_token()))
            rejected = ""
        except Exception as e:  # the client raises on the 401 response (checked directly above)
            rejected = f"{type(e).__name__}: {e}"
        self.assertTrue(rejected, "an unknown token was accepted")


if __name__ == "__main__":
    unittest.main()
