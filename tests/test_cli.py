"""The command line interface: argument handling, exit codes, and compatibility with MCP client configs."""
import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from novuspipeline import cli, server
from test_server import LEGACY_MODULE, WorkspaceTestCase  # also applies the test environment settings


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = cli.main(list(argv))
        except SystemExit as exit_:  # argparse errors and --version
            code = exit_.code if isinstance(exit_.code, int) else 1
    return code, out.getvalue(), err.getvalue()


class TestParser(unittest.TestCase):

    def test_no_command_means_serve_with_existing_server_options(self):
        # MCP client configs launch `novuspipeline --workspace X` and `--transport http --port N`.
        args = cli.build_parser().parse_args(["--workspace", "/w", "--transport", "http", "--port", "9000"])
        self.assertIsNone(args.command)
        self.assertEqual((args.workspace, args.transport, args.port), ("/w", "http", 9000))

    def test_workspace_works_before_and_after_the_command(self):
        parser = cli.build_parser()
        self.assertEqual(parser.parse_args(["--workspace", "/a", "audit", "f.py"]).workspace, "/a")
        self.assertEqual(parser.parse_args(["audit", "f.py", "--workspace", "/b"]).workspace, "/b")
        self.assertIsNone(parser.parse_args(["audit", "f.py"]).workspace)

    def test_serve_subcommand_accepts_server_options(self):
        args = cli.build_parser().parse_args(["serve", "--transport", "http", "--port", "1234"])
        self.assertEqual((args.command, args.transport, args.port), ("serve", "http", 1234))
        self.assertEqual(cli.build_parser().parse_args(["serve"]).transport, "stdio")

    def test_test_command_keeps_its_own_options(self):
        args = cli.build_parser().parse_args(["test", "python", "-m", "pytest", "-x"])
        self.assertEqual(args.test_command, ["python", "-m", "pytest", "-x"])

    def test_version_does_not_load_the_server(self):
        code, out, _ = run_cli("--version")
        self.assertEqual(code, 0)
        self.assertIn("novuspipeline", out)

    def test_exit_codes(self):
        self.assertEqual(cli._exit_code("audit", "## Report"), 0)
        self.assertEqual(cli._exit_code("audit", "Error: File 'x' does not exist."), 1)
        self.assertEqual(cli._exit_code("modernize", "❌ Pipeline stopped: nothing was committed"), 1)
        self.assertEqual(cli._exit_code("test", "OK\n\nExit Code: 0"), 0)
        self.assertEqual(cli._exit_code("test", "FAILED\n\nExit Code: 1"), 1)
        # an "Exit Code:" inside the tests' own output must not decide the result
        self.assertEqual(cli._exit_code("test", "Exit Code: 3 printed by a test\n...\nExit Code: 0"), 0)


class TestCommands(WorkspaceTestCase):

    def test_audit_prints_findings_and_exits_zero(self):
        self.write("legacy.py", LEGACY_MODULE)
        code, out, _ = run_cli("audit", "legacy.py")
        self.assertEqual(code, 0, out)
        self.assertIn("Modernization Audit Report", out)

    def test_missing_file_exits_nonzero(self):
        code, out, _ = run_cli("audit", "missing.py")
        self.assertEqual(code, 1)
        self.assertIn("does not exist", out)

    def test_workspace_option_is_applied(self):
        with mock.patch.object(server, "WORKSPACE_ROOT", self.root):
            other = os.path.join(self.root, "sub")
            os.makedirs(other)
            with open(os.path.join(other, "m.py"), "w", encoding="utf-8") as f:
                f.write(LEGACY_MODULE)
            code, out, _ = run_cli("audit", "m.py", "--workspace", other)
            self.assertEqual(code, 0, out)
            self.assertEqual(server.WORKSPACE_ROOT, os.path.abspath(other))

    def test_workspace_must_be_a_directory(self):
        code, _, err = run_cli("overview", "--workspace", os.path.join(self.root, "nope"))
        self.assertEqual(code, 2)
        self.assertIn("not a directory", err)

    def test_apply_needs_confirmation_when_not_interactive(self):
        self.write("legacy.py", LEGACY_MODULE)
        with mock.patch.object(cli, "_confirm", return_value=False):
            code, out, _ = run_cli("apply", "legacy.py")
        self.assertEqual(code, 1)
        self.assertIn("not applied", out)
        self.assertEqual(self.read("legacy.py"), LEGACY_MODULE)

    def test_apply_with_yes_rewrites_the_file(self):
        self.write("legacy.py", LEGACY_MODULE)
        code, out, _ = run_cli("apply", "legacy.py", "--yes")
        self.assertEqual(code, 0, out)
        self.assertNotEqual(self.read("legacy.py"), LEGACY_MODULE)
        self.assertTrue(os.path.exists(os.path.join(self.root, "legacy.py.bak")))

    def test_standards_list(self):
        code, out, _ = run_cli("standards", "list")
        self.assertEqual(code, 0, out)
        self.assertIn("Modernization Standards", out)

    def test_test_command_failure_sets_exit_code(self):
        self.write("test_fail.py", "import unittest\n\nclass T(unittest.TestCase):\n    def test_x(self):\n        self.assertTrue(False)\n")
        code, out, _ = run_cli("test", "python", "-m", "unittest", "test_fail")
        self.assertEqual(code, 1, out)


if __name__ == "__main__":
    unittest.main()
