import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest

from novuspipeline import cli

LEGACY = "import logging\n\n\ndef f():\n    logging.warn('x')\n"
FIXED = "import logging\n\n\ndef f():\n    logging.warning('x')\n"


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            cli.main(list(argv))
            code = 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
    return code, out.getvalue(), err.getvalue()


def git(root: str, *args: str) -> None:
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=root, check=True,
                   capture_output=True)


class CliCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.app = os.path.join(self.root, "app.py")
        with open(self.app, "w", newline="") as f:
            f.write(LEGACY)

    def read(self) -> str:
        with open(self.app, newline="") as f:
            return f.read()


class TestScan(CliCase):
    def test_text_report_and_exit_codes(self):
        code, out, _ = run("scan", "--workspace", self.root, "--no-ruff")
        self.assertEqual(code, 0)
        self.assertIn("app.py:5: [MEDIUM] MIG-LOGGING-WARN", out)
        self.assertEqual(run("scan", "--workspace", self.root, "--no-ruff", "--fail-on", "high")[0], 0)
        self.assertEqual(run("scan", "--workspace", self.root, "--no-ruff", "--fail-on", "medium")[0], 1)

    def test_json_and_sarif(self):
        data = json.loads(run("scan", "--workspace", self.root, "--no-ruff", "--format", "json")[1])
        self.assertEqual(data["files"], 1)
        self.assertIn("MIG-LOGGING-WARN", {f["smell_id"] for f in data["findings"]})
        sarif = json.loads(run("scan", "--workspace", self.root, "--no-ruff", "--format", "sarif")[1])
        result = next(r for r in sarif["runs"][0]["results"] if r["ruleId"] == "MIG-LOGGING-WARN")
        self.assertEqual(result["locations"][0]["physicalLocation"]["region"]["startLine"], 5)
        self.assertEqual(sarif["version"], "2.1.0")

    def test_skips_virtualenvs(self):
        os.makedirs(os.path.join(self.root, "node_modules"))
        with open(os.path.join(self.root, "node_modules", "x.py"), "w") as f:
            f.write(LEGACY)
        self.assertEqual(cli.discover(self.root, []), ["app.py"])

    def test_min_severity_filters(self):
        _, out, _ = run("scan", "--workspace", self.root, "--no-ruff", "--min-severity", "high")
        self.assertNotIn("MIG-LOGGING-WARN", out)


class TestFix(CliCase):
    def test_diff_and_check_do_not_write(self):
        _, out, _ = run("fix", "--workspace", self.root, "--diff")
        self.assertIn("+    logging.warning('x')", out)
        self.assertEqual(self.read(), LEGACY)
        self.assertEqual(run("fix", "--workspace", self.root, "--check")[0], 1)
        self.assertEqual(self.read(), LEGACY)

    def test_fix_writes_and_preserves_crlf(self):
        with open(self.app, "w", newline="") as f:
            f.write(LEGACY.replace("\n", "\r\n"))
        self.assertEqual(run("fix", "--workspace", self.root)[0], 0)
        self.assertEqual(self.read(), FIXED.replace("\n", "\r\n"))
        self.assertEqual(run("fix", "--workspace", self.root, "--check")[0], 0)

    def test_files_with_uncommitted_changes_are_skipped(self):
        git(self.root, "init", "-q")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-qm", "init")
        with open(self.app, "a") as f:
            f.write("x = 1\n")
        _, _, err = run("fix", "--workspace", self.root)
        self.assertIn("uncommitted", err)
        self.assertNotIn("warning(", self.read())
        run("fix", "--workspace", self.root, "--force")
        self.assertIn("logging.warning('x')", self.read())

    def test_migrations_listing(self):
        code, out, _ = run("migrations")
        self.assertEqual(code, 0)
        self.assertIn("MIG-UNITTEST-ALIASES", out)


if __name__ == "__main__":
    unittest.main()
