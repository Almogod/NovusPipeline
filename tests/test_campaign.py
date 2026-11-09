import json

from novuspipeline import campaign, server
from test_server import WorkspaceTestCase  # also applies the test environment settings

TESTS = "python -m unittest discover -s tests -t ."
UTIL = "import threading\n\n\ndef count():\n    return threading.activeCount()\n"
APP = "import threading\n\nfrom pkg.util import count\n\n\ndef main():\n    return threading.currentThread().name, count()\n"
FRAGILE = "import threading\n\n\ndef name():\n    return threading.currentThread.__name__\n"
TEST = ("import unittest\n\nfrom pkg.app import main\nfrom pkg.fragile import name\n\n\n"
        "class T(unittest.TestCase):\n    def test_main(self):\n        self.assertEqual(main()[0], 'MainThread')\n\n"
        "    def test_name(self):\n        self.assertEqual(name(), 'currentThread')\n")


class TestDependencyOrder(WorkspaceTestCase):
    def test_imports_come_first_and_cycles_terminate(self):
        forward = {"a.py": {"b.py": set()}, "b.py": {"c.py": set()}, "c.py": {"a.py": set()}, "d.py": {}}
        order = campaign.dependency_order(forward, ["a.py", "b.py", "c.py", "d.py"])
        self.assertEqual(sorted(order), ["a.py", "b.py", "c.py", "d.py"])
        self.assertLess(order.index("c.py"), order.index("b.py"))


class TestCampaign(WorkspaceTestCase):
    def setUp(self):
        super().setUp()
        self.write(".gitignore", ".novus_index/\n")
        self.write("pkg/__init__.py", "")
        self.write("pkg/util.py", UTIL)
        self.write("pkg/app.py", APP)
        self.write("pkg/fragile.py", FRAGILE)
        self.write("tests/__init__.py", "")
        self.write("tests/test_app.py", TEST)
        self.commit_all()

    def test_commits_verified_files_rejects_breaking_ones_and_resumes(self):
        first = campaign.run(self.root, "modern", test_command=TESTS, max_files=1)
        self.assertTrue(first["ok"], first["error"])
        self.assertEqual([r["path"] for r in first["results"] if r["status"] == "committed"], ["pkg/util.py"])

        second = campaign.run(self.root, "modern", test_command=TESTS)
        status = {r["path"]: r["status"] for r in second["results"]}
        self.assertEqual(status["pkg/app.py"], "committed")
        self.assertEqual(status["pkg/fragile.py"], "tests-failed")  # __name__ is observable: the gate catches it
        self.assertNotIn("pkg/util.py", status)  # resumed, not redone

        log = self.repo.git.log("--format=%s", "modern").splitlines()
        self.assertEqual(log[:2], ["refactor(novus): modernize pkg/app.py", "refactor(novus): modernize pkg/util.py"])
        self.assertEqual(self.committed_files("modern"), {"pkg/app.py"})
        self.assertIn("threading.active_count()", self.repo.git.show("modern:pkg/util.py"))
        # The working copy and the checked-out branch are untouched.
        self.assertEqual(self.read("pkg/util.py"), UTIL)
        self.assertEqual(self.repo.active_branch.name, "main")
        self.assertEqual(self.repo.git.worktree("list").count("\n"), 0)

        with open(campaign.ledger_path(self.root, "modern"), encoding="utf-8") as f:
            ledger = json.load(f)
        self.assertEqual(ledger["files"]["pkg/fragile.py"]["status"], "tests-failed")
        self.assertIn("Rejected: tests failed", second["summary"])

    def test_already_failing_tests_are_not_blamed_on_the_change(self):
        self.write("tests/test_app.py", TEST + "\n    def test_broken(self):\n        self.fail('pre-existing')\n")
        self.commit_all("broken test")
        res = campaign.run(self.root, "modern", paths=["pkg/util.py"], test_command=TESTS)
        self.assertEqual([r["status"] for r in res["results"]], ["baseline-failing"])
        self.assertNotIn("modern", [b.name for b in self.repo.branches])

    def test_refuses_checked_out_branch_and_bad_commands(self):
        self.assertIn("checked-out", campaign.run(self.root, "main")["error"])
        res = campaign.run(self.root, "modern", test_command="python -c 'print(1)'")
        self.assertFalse(res["ok"])
        self.assertIn("could not run", res["error"])

    def test_mcp_tool(self):
        out = server.run_modernization_campaign("modern", "pkg/util.py", TESTS)
        self.assertIn("1 file(s) committed", out)
