"""Sandboxed test execution and the worktree-isolated pipeline."""
import os
import sys
import time
import unittest
from unittest import mock

from novuspipeline import sandbox, server
from novuspipeline.server import run_autonomous_modernization_pipeline, run_local_tests
from test_server import LEGACY_MODULE, WorkspaceTestCase  # also applies the test environment settings


class TestSandboxedTestRuns(WorkspaceTestCase):

    def run_test_file(self, body: str) -> str:
        self.write("test_probe.py", "import os, unittest\n\nclass T(unittest.TestCase):\n" + body)
        return run_local_tests("python -m unittest test_probe")

    def test_tests_cannot_modify_the_workspace(self):
        self.write("data.txt", "original")
        out = self.run_test_file("    def test_write(self):\n        open('data.txt', 'w').write('HACKED')\n")
        self.assertIn("Exit Code: 0", out)
        self.assertEqual(self.read("data.txt"), "original")
        self.assertIn("project copy", out)

    def test_secrets_are_not_visible_and_home_is_redirected(self):
        with mock.patch.dict(os.environ, {"NOVUS_TEST_SECRET_TOKEN": "sk-live-123"}):
            out = self.run_test_file(
                "    def test_env(self):\n        self.assertIsNone(os.environ.get('NOVUS_TEST_SECRET_TOKEN'))\n"
                "        self.assertNotEqual(os.path.expanduser('~'), " + repr(os.path.expanduser("~")) + ")\n")
        self.assertIn("Exit Code: 0", out, out)
        self.assertIn("scrubbed environment", out)

    @unittest.skipIf(sys.platform == "darwin", "macOS does not enforce RLIMIT_AS")
    def test_memory_limit_is_enforced(self):
        with mock.patch.object(sandbox, "MEMORY_LIMIT_MB", 300):
            out = self.run_test_file("    def test_alloc(self):\n        b = bytearray(800 * 1024 * 1024)\n")
        self.assertNotIn("Exit Code: 0", out)

    def test_timeout_kills_the_run(self):
        with mock.patch.object(server, "TEST_TIMEOUT_SECONDS", 3):
            start = time.time()
            out = self.run_test_file("    def test_hang(self):\n        import time\n        time.sleep(60)\n")
        self.assertIn("timed out", out)
        self.assertLess(time.time() - start, 30)

    def test_linked_node_modules_survive_cleanup(self):
        self.write("node_modules/dep/index.js", "module.exports = 1;\n")
        result = sandbox.run([sys.executable, "-c", "import os; print(os.path.isdir('node_modules'))"], self.root, 60)
        self.assertIn("True", result.output)
        self.assertTrue(os.path.exists(os.path.join(self.root, "node_modules", "dep", "index.js")))


class TestWorktreePipeline(WorkspaceTestCase):

    def test_refuses_file_with_uncommitted_changes(self):
        self.write("calc.py", LEGACY_MODULE)
        self.commit_all()
        self.write("calc.py", LEGACY_MODULE + "\n# local edit\n")
        res = run_autonomous_modernization_pipeline("calc.py", test_command="python --version")
        self.assertIn("uncommitted changes", res)
        self.assertIn("# local edit", self.read("calc.py"))

    def test_refuses_current_branch_as_target(self):
        self.write("calc.py", LEGACY_MODULE)
        self.commit_all()
        self.assertIn("checked-out branch", run_autonomous_modernization_pipeline("calc.py", branch_name="main"))

    def test_preserves_crlf_line_endings(self):
        with open(os.path.join(self.root, "calc.py"), "w", encoding="utf-8", newline="") as f:
            f.write(LEGACY_MODULE.replace("\n", "\r\n"))
        self.repo.git.config("core.autocrlf", "false")
        self.commit_all()
        res = run_autonomous_modernization_pipeline("calc.py", test_command="python --version", branch_name="crlf")
        self.assertIn("Completed Successfully", res, res)
        committed = self.repo.git.show("crlf:calc.py", strip_newline_in_stdout=False)
        self.assertIn("import urllib.request", committed)
        self.assertNotIn("\n", committed.replace("\r\n", ""))  # every newline is still CRLF

    def test_extends_existing_branch(self):
        self.write("calc.py", LEGACY_MODULE)
        self.write("other.py", "def g():\n    try:\n        pass\n    except:\n        pass\n")
        self.commit_all()
        for name in ("calc.py", "other.py"):
            res = run_autonomous_modernization_pipeline(name, test_command="python --version", branch_name="batch")
            self.assertIn("Completed Successfully", res, res)
        log = self.repo.git.log("--format=%s", "batch").splitlines()
        self.assertEqual(log[:2], ["refactor: autonomous modernization of other.py",
                                   "refactor: autonomous modernization of calc.py"])


if __name__ == "__main__":
    unittest.main()
