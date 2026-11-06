import unittest

from novuspipeline import codemods
from novuspipeline.modernizer import CodeModernizer


def rewrite(src: str, only: list[str] | None = None) -> str:
    return codemods.apply(src, only=only)[0]


class TestUnittestAliases(unittest.TestCase):
    ID = ("MIG-UNITTEST-ALIASES",)

    def test_renames_in_testcase_and_keeps_formatting(self):
        src = ("import unittest\n\nclass T(unittest.TestCase):\n    def test_a(self):\n"
               "        self.assertEquals(1,   1)  # why\n        self.assertRegexpMatches('a', 'a')\n")
        out = rewrite(src, list(self.ID))
        self.assertIn("self.assertEqual(1,   1)  # why", out)
        self.assertIn("self.assertRegex('a', 'a')", out)

    def test_ignores_non_test_classes_and_own_definitions(self):
        src = ("import unittest\n\nclass Plain:\n    def f(self):\n        self.assertEquals(1)\n\n"
               "class T(unittest.TestCase):\n    def assertEquals(self, a):\n        pass\n"
               "    def test_a(self):\n        self.assertEquals(1)\n")
        self.assertEqual(rewrite(src, list(self.ID)), src)


class TestCollectionsAbc(unittest.TestCase):
    ID = ("MIG-COLLECTIONS-ABC",)

    def test_from_import_is_split(self):
        out = rewrite("from collections import OrderedDict, Mapping  # c\n", list(self.ID))
        self.assertEqual(out, "from collections.abc import Mapping\nfrom collections import OrderedDict  # c\n")

    def test_from_import_fully_moved(self):
        self.assertEqual(rewrite("from collections import Mapping, Callable\n", list(self.ID)),
                         "from collections.abc import Mapping, Callable\n")

    def test_attribute_use_adds_import(self):
        out = rewrite("import collections\n\nx = isinstance(1, collections.Mapping)\n", list(self.ID))
        self.assertEqual(out, "import collections\nimport collections.abc\n\nx = isinstance(1, collections.abc.Mapping)\n")

    def test_attribute_without_top_level_import_is_reported_not_rewritten(self):
        src = "def f(collections):\n    return collections.Mapping\n"
        self.assertEqual(rewrite(src, list(self.ID)), src)
        self.assertEqual([f["fix"] for f in codemods.scan(src, "a.py", only=list(self.ID))], [None])

    def test_ordereddict_is_not_an_abc(self):
        src = "import collections\nx = collections.OrderedDict()\n"
        self.assertEqual(codemods.scan(src, "a.py"), [])


class TestSmallMigrations(unittest.TestCase):
    def test_base64_and_logging(self):
        src = "import base64\nimport logging\nbase64.encodestring(b'x')\nlogging.warn('x')\n"
        self.assertEqual(rewrite(src), "import base64\nimport logging\nbase64.encodebytes(b'x')\nlogging.warning('x')\n")

    def test_logger_instances_are_not_touched(self):
        src = "import logging\nlog = logging.getLogger()\nlog.warn('x')\n"
        self.assertEqual(rewrite(src), src)

    def test_detect_only_apis_are_reported_with_advice(self):
        found = codemods.scan("import inspect\ninspect.getargspec(f)\nimport datetime\ndatetime.datetime.utcnow()\n", "a.py")
        self.assertEqual({f["smell_id"] for f in found}, {"MIG-REMOVED-APIS", "MIG-DATETIME-UTCNOW"})
        self.assertTrue(all(f["fix"] is None for f in found))


class TestRobustness(unittest.TestCase):
    def test_unparseable_python_is_left_alone(self):
        src = "print 'hello'\nself.assertEquals(1)\n"
        self.assertEqual(codemods.apply(src), (src, []))
        self.assertEqual(codemods.scan(src, "a.py"), [])

    def test_non_python_files_are_skipped(self):
        self.assertEqual(codemods.scan("self.assertEquals(1)", "a.js"), [])

    def test_idempotent(self):
        src = "import logging\nlogging.warn('x')\n"
        once = rewrite(src)
        self.assertEqual(rewrite(once), once)

    def test_modernizer_applies_migrations(self):
        out, notes = CodeModernizer.modernize_python("import logging\nlogging.warn('x')\n")
        self.assertIn("logging.warning('x')", out)
        self.assertTrue(any("MIG-LOGGING-WARN" in n for n in notes))

    def test_custom_migration_registers_itself(self):
        class Demo(codemods.Migration):
            id = "MIG-TEST-DEMO"
            title = "demo"

        try:
            self.assertIn("MIG-TEST-DEMO", codemods.REGISTRY)
        finally:
            codemods.REGISTRY.pop("MIG-TEST-DEMO", None)


if __name__ == "__main__":
    unittest.main()
