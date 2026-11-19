"""Whole-project fixing (`novuspipeline fix`) and the guided terminal mode."""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import git

from novuspipeline import cli, fixer, interactive, server
from test_server import WorkspaceTestCase  # also applies the test environment settings

NET = "import urllib2\n\ndef fetch(u):\n    return urllib2.urlopen(u).read()\n"
MATHS = "def ratio(a, b):\n    try:\n        return a / b\n    except:\n        return 0\n"
# Imports both modules, so testing the first rewrite fails until the other is fixed too.
TESTS = ("import unittest\nfrom app import net, maths\n\nclass T(unittest.TestCase):\n"
         "    def test_fetch(self):\n        self.assertTrue(callable(net.fetch))\n"
         "    def test_ratio(self):\n        self.assertEqual(maths.ratio(1, 0), 0)\n")
GIT_IDENTITY = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}


def run_quiet(fn, *args, **kwargs):
    out = io.StringIO()
    with redirect_stdout(out):
        code = fn(*args, **kwargs)
    return code, out.getvalue()


class TestHelpers(unittest.TestCase):

    def test_clean_path_strips_what_pasting_adds(self):
        here = os.path.abspath("proj")
        self.assertEqual(interactive.clean_path('"proj"'), here)
        self.assertEqual(interactive.clean_path("'proj'  "), here)
        self.assertEqual(interactive.clean_path("& 'proj'"), here)  # PowerShell drag-and-drop
        self.assertEqual(interactive.clean_path("   "), "")

    def test_classify_pipeline_reports(self):
        self.assertEqual(fixer.classify("🎉 Autonomous Modernization Pipeline Completed"), fixer.FIXED)
        self.assertEqual(fixer.classify("Pipeline finished: nothing to modernize automatically"), fixer.UNCHANGED)
        self.assertEqual(fixer.classify("❌ Autonomous Modernization Pipeline: verification `x` failed (exit code 1)"),
                         fixer.TESTS_FAILED)
        self.assertEqual(fixer.classify("❌ Autonomous Modernization Pipeline: verification `x` failed\nNO TESTS RAN"),
                         fixer.NO_TESTS)
        self.assertEqual(fixer.classify("❌ Pipeline stopped: no tests to verify `a.js` with."), fixer.NO_TESTS)
        self.assertEqual(fixer.classify("❌ Verification: NONE. `python -m pytest -q` executed 0 tests"), fixer.NO_TESTS)
        self.assertEqual(fixer.classify("❌ Pipeline stopped: after modernization `a` still violates blocking "
                                        "standards"), fixer.BLOCKED)
        self.assertEqual(fixer.classify("❌ Pipeline stopped: the modernized file would no longer provide `x`"),
                         fixer.API_BREAK)
        self.assertEqual(fixer.classify("Error: 'a.py' has uncommitted changes."), fixer.DIRTY)
        self.assertEqual(fixer.classify("Error executing pipeline: boom"), fixer.ERROR)


class TestEntryPoint(unittest.TestCase):

    def test_bare_command_in_a_terminal_opens_the_guided_mode(self):
        with mock.patch.object(cli, "_in_terminal", return_value=True), \
                mock.patch.object(interactive, "run", return_value=0) as run, \
                mock.patch.object(server, "run_server") as serve:
            self.assertEqual(cli.main([]), 0)
        run.assert_called_once()
        serve.assert_not_called()

    def test_bare_command_from_an_mcp_client_serves(self):
        with mock.patch.object(cli, "_in_terminal", return_value=False), \
                mock.patch.object(interactive, "run") as run, \
                mock.patch.object(server, "run_server") as serve, \
                mock.patch.object(server, "WORKSPACE_ROOT", os.getcwd()):
            cli.main(["--workspace", os.getcwd()])
        serve.assert_called_once_with("stdio", "127.0.0.1", 8000, False)
        run.assert_not_called()

    def test_http_transport_always_serves(self):
        with mock.patch.object(cli, "_in_terminal", return_value=True), \
                mock.patch.object(interactive, "run") as run, \
                mock.patch.object(server, "run_server") as serve, \
                mock.patch.object(server, "WORKSPACE_ROOT", os.getcwd()):
            cli.main(["--transport", "http"])
        serve.assert_called_once()
        run.assert_not_called()


class TestFixProject(WorkspaceTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.write("app/__init__.py", "")
        self.write("app/net.py", NET)
        self.write("app/maths.py", MATHS)
        self.write("app/clean.py", "def clean(x: str) -> str:\n    return x.strip()\n")
        self.write("test_app.py", TESTS)
        self.commit_all()

    def branches(self) -> set:
        return {b.name for b in self.repo.branches}

    def test_candidates_skip_clean_files_and_tests(self):
        found = [c.path for c in fixer.find_candidates(server)]
        self.assertEqual(found, ["app/maths.py", "app/net.py"])

    def test_fix_commits_every_file_on_a_branch_and_leaves_the_working_copy(self):
        code, out = run_quiet(interactive.run, server, self.root, assume_yes=True, branch="novus/fixes")
        self.assertEqual(code, 0, out)
        # maths.py fails on the first pass (net.py still imports urllib2) and passes on the retry
        self.assertIn("Retrying 1 file(s)", out)
        self.assertIn("2 file(s) fixed", out)
        self.assertIn("novus/fixes", self.branches())
        self.assertEqual(self.read("app/maths.py"), MATHS)
        self.assertEqual(self.repo.active_branch.name, "main")
        fixed = self.repo.git.show("novus/fixes:app/maths.py")
        self.assertIn("except Exception:", fixed)
        self.assertNotIn("??", self.repo.git.status("--porcelain"))  # .novus_index ignores itself

    def test_merge_fast_forwards_the_current_branch(self):
        code, out = run_quiet(interactive.run, server, self.root, assume_yes=True, branch="novus/fixes", merge=True)
        self.assertEqual(code, 0, out)
        self.assertIn("except Exception:", self.read("app/maths.py"))

    def test_files_with_uncommitted_changes_are_skipped(self):
        self.write("app/maths.py", MATHS + "\n# local edit\n")
        _, out = run_quiet(interactive.run, server, self.root, assume_yes=True, branch="novus/fixes")
        self.assertIn("1 skipped (uncommitted changes)", out)
        self.assertIn("# local edit", self.read("app/maths.py"))

    def test_guided_mode_asks_for_the_folder_and_confirmation(self):
        answers = iter([f'"{self.root}"', "", "n"])  # pasted path in quotes, Enter = start, no merge
        with mock.patch.object(interactive, "_ask", side_effect=lambda _: next(answers)), \
                mock.patch("sys.stdin") as stdin:
            stdin.isatty.return_value = True
            code, out = run_quiet(interactive.run, server, None)
        self.assertEqual(code, 0, out)
        self.assertIn("2 file(s) fixed", out)
        self.assertIn("git merge novus/modernize-", out)
        self.assertEqual(self.read("app/maths.py"), MATHS)

    def test_declining_changes_nothing(self):
        answers = iter(["n"])
        with mock.patch.object(interactive, "_ask", side_effect=lambda _: next(answers)), \
                mock.patch("sys.stdin") as stdin:
            stdin.isatty.return_value = True
            code, out = run_quiet(interactive.run, server, self.root)
        self.assertEqual(code, 0)
        self.assertIn("Nothing was changed", out)
        self.assertEqual(self.branches(), {"main"})


class TestNotARepository(unittest.TestCase):

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        with open(os.path.join(self.root, "m.py"), "w", encoding="utf-8") as f:
            f.write(MATHS)
        patcher = mock.patch.object(server, "WORKSPACE_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_without_confirmation_nothing_is_created(self):
        code, out = run_quiet(interactive.run, server, self.root, assume_yes=True)
        self.assertEqual(code, 1)
        self.assertIn("not a git repository", out)
        self.assertFalse(os.path.exists(os.path.join(self.root, ".git")))

    def test_guided_mode_can_initialize_git(self):
        answers = iter(["y", "n"])  # create the repository; then decline fixing
        with mock.patch.dict(os.environ, GIT_IDENTITY), \
                mock.patch.object(interactive, "_ask", side_effect=lambda _: next(answers)), \
                mock.patch("sys.stdin") as stdin:
            stdin.isatty.return_value = True
            code, out = run_quiet(interactive.run, server, self.root)
        self.assertEqual(code, 0, out)
        repo = git.Repo(self.root)
        self.assertTrue(repo.head.is_valid())
        self.assertIn("m.py", repo.git.ls_files())


if __name__ == "__main__":
    unittest.main()
