"""Smells and fixes learned from the repository's own commit history."""
import unittest

from novuspipeline import history_learner
from novuspipeline.history_learner import HistoryLearner, diff_pairs
from novuspipeline.server import (
    analyze_legacy_codebase,
    get_codebase_overview,
    get_file_context,
    learn_from_history,
    run_autonomous_modernization_pipeline,
)
from test_server import WorkspaceTestCase  # also applies the test environment settings

CLIENT = "class client:\n    @staticmethod\n    def get_json(n):\n        return n + 1\n"


def test_module(n: int, call: str = "client.fetch_json") -> str:
    # A team-specific API rename that no built-in rule or Ruff knows about.
    return (f"import unittest\n\nfrom api import client\n\n\nclass T{n}(unittest.TestCase):\n"
            f"    def test_a(self):\n        self.assertEqual({call}({n}), {n + 1})\n\n"
            f"    def test_b(self):\n        self.assertTrue({n})\n")


class TestDiffPairs(unittest.TestCase):

    def test_argument_preserving_rename(self):
        pairs = diff_pairs("def f(x):\n    return old.load(x, 1)\n", "def f(x):\n    return new.load(x, 1)\n")
        self.assertEqual([(p["old"], p["new"], p["args_preserved"]) for p in pairs], [("old.load", "new.load", True)])

    def test_large_rewrites_are_not_focused_replacements(self):
        before = "def f():\n" + "".join(f"    a.c{i}()\n" for i in range(5))
        after = "def f():\n" + "".join(f"    b.d{i}()\n" for i in range(5))
        self.assertEqual(diff_pairs(before, after), [])

    def test_unparseable_versions_are_skipped(self):
        self.assertEqual(diff_pairs("def f(:\n", "def f():\n    a.b()\n"), [])


class LearnerTestCase(WorkspaceTestCase):

    def setUp(self) -> None:
        super().setUp()
        history_learner._LEARNERS.clear()
        self.addCleanup(history_learner._LEARNERS.clear)

    def commit(self, files: dict, message: str) -> None:
        for rel, content in files.items():
            self.write(rel, content)
        self.repo.git.add("-A")
        # One minute apart: the learner orders events by commit time, which has 1 s resolution.
        self._clock = getattr(self, "_clock", 1_700_000_000) + 60
        date = f"{self._clock} +0000"
        self.repo.git.commit("-m", message, env={"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date})

    def build_history(self) -> None:
        """Four test modules; the team replaces fetch_json in three separate commits."""
        self.commit({"api.py": CLIENT, **{f"test_m{i}.py": test_module(i) for i in range(4)}}, "add tests")
        for i in range(3):
            self.commit({f"test_m{i}.py": test_module(i, "client.get_json")}, f"tests: use get_json in m{i}")


class TestLearning(LearnerTestCase):

    def test_learns_rule_with_evidence(self):
        self.build_history()
        out = learn_from_history()
        self.assertIn("`client.fetch_json` -> `client.get_json`: 3 commits, consistency 100%", out)
        self.assertIn("auto-fixable (pure rename)", out)
        self.assertIn("tests: use get_json in m", out)

    def test_one_large_commit_does_not_create_a_rule(self):
        self.commit({f"test_m{i}.py": test_module(i) for i in range(4)}, "add tests")
        self.commit({f"test_m{i}.py": test_module(i, "client.get_json") for i in range(4)}, "mass rename")
        self.assertIn("0 rule(s)", learn_from_history())

    def test_argument_changes_are_suggestions_only(self):
        for i in range(3):
            self.commit({f"m{i}.py": "def f(x):\n    return store.load(x)\n"}, f"add m{i}")
            self.commit({f"m{i}.py": "def f(x):\n    return store.load_v2(x, strict=True)\n"}, f"migrate m{i}")
        out = learn_from_history()
        self.assertIn("`store.load` -> `store.load_v2`", out)
        self.assertIn("suggestion only", out)
        learner = history_learner.get_learner(self.root)
        code = "def g(y):\n    return store.load(y)\n"
        self.assertEqual(learner.apply_fixes(code), (code, []))

    def test_inconsistent_history_is_not_learned(self):
        self.commit({f"m{i}.py": "def f(x):\n    return util.parse(x)\n" for i in range(4)}, "add")
        for i, new in enumerate(["a.parse", "b.parse", "c.parse", "d.parse"]):
            self.commit({f"m{i}.py": f"def f(x):\n    return {new}(x)\n"}, f"change {i}")
        self.assertNotIn("util.parse", learn_from_history(min_support=2))

    def test_calls_still_being_adopted_are_not_smells(self):
        # A few commits swap out a ubiquitous call, but the team keeps writing new uses of it.
        self.commit({f"m{i}.py": "def f(x):\n    return log.info(x)\n" for i in range(3)}, "add")
        for i in range(3):
            self.commit({f"m{i}.py": "def f(x):\n    return log.debug(x)\n"}, f"quieter {i}")
        self.assertIn("`log.info` -> `log.debug`", learn_from_history())
        for i in range(5):
            self.commit({f"n{i}.py": f"def g{i}(x):\n    return log.info(x)\n"}, f"new feature {i}")
        self.assertNotIn("log.info", learn_from_history())

    def test_mining_is_incremental(self):
        self.build_history()
        learner = history_learner.get_learner(self.root)
        first = learner.refresh()
        self.assertEqual(first["mined"], first["total_commits"])
        self.assertEqual(learner.refresh()["mined"], 0)
        self.commit({"test_m3.py": test_module(3, "client.get_json")}, "last one")
        self.assertEqual(learner.refresh()["mined"], 1)
        # State survives a restart.
        self.assertEqual(HistoryLearner(self.root).state["mined_commits"], first["total_commits"] + 1)

    def test_not_a_repository_with_history(self):
        self.assertEqual(HistoryLearner(self.root + "/missing").refresh()["mined"], 0)


class TestLearnedSmellsInTools(LearnerTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.build_history()
        learn_from_history()

    def test_audit_reports_remaining_use_with_evidence(self):
        out = analyze_legacy_codebase("test_m3.py")
        self.assertIn("Team replaced `client.fetch_json` with `client.get_json` in 3 commits", out)
        self.assertIn("Evidence from history", out)
        self.assertIn("1 learned from this repository's history", out)
        self.assertIn("Learned from this repository's history", out)  # codebase context section

    def test_file_context_and_overview(self):
        self.assertIn("(team history)", get_file_context("test_m3.py"))
        overview = get_codebase_overview()
        self.assertIn("Learned From This Repository's History", overview)
        self.assertIn("still used in 1 file(s), e.g. `test_m3.py`", overview)

    def test_pipeline_applies_learned_fix_on_branch(self):
        res = run_autonomous_modernization_pipeline("test_m3.py", test_command="python -m unittest test_m3",
                                                    branch_name="learned")
        self.assertIn("Completed Successfully", res, res)
        self.assertIn("Applied learned fix", res)
        committed = self.repo.git.show("learned:test_m3.py")
        self.assertIn("client.get_json(3)", committed)
        self.assertNotIn("fetch_json", committed)
        self.assertIn("fetch_json", self.read("test_m3.py"))  # working copy untouched

    def test_fix_keeps_code_valid_and_non_ascii_offsets(self):
        learner = history_learner.get_learner(self.root)
        code = "def test(self):\n    x = 'é'; client.fetch_json(x, 'é')\n"
        new, notes = learner.apply_fixes(code)
        self.assertEqual(new, "def test(self):\n    x = 'é'; client.get_json(x, 'é')\n")
        self.assertEqual(len(notes), 1)


if __name__ == "__main__":
    unittest.main()
