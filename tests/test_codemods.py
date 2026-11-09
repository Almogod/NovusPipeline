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


class TestRenamedApis(unittest.TestCase):
    ID = ("MIG-RENAMED-APIS",)

    def test_cross_module_rename_adds_import_and_drops_removed_module(self):
        out = rewrite("import imp\n\nimp.reload(m)\n", list(self.ID))
        self.assertEqual(out, "import importlib\n\nimportlib.reload(m)\n")

    def test_removed_module_import_kept_while_still_used(self):
        out = rewrite("import imp\nimp.reload(m)\nimp.load_source('a', 'b')\n", list(self.ID))
        self.assertEqual(out, "import imp\nimport importlib\nimportlib.reload(m)\nimp.load_source('a', 'b')\n")

    def test_from_import_keeps_local_name(self):
        out = rewrite("from configparser import SafeConfigParser, RawConfigParser\np = SafeConfigParser()\n",
                      list(self.ID))
        self.assertEqual(out, "from configparser import ConfigParser as SafeConfigParser\n"
                              "from configparser import RawConfigParser\np = SafeConfigParser()\n")

    def test_module_rename_with_and_without_alias(self):
        self.assertEqual(rewrite("import xml.etree.cElementTree as ET\n", list(self.ID)),
                         "import xml.etree.ElementTree as ET\n")
        self.assertEqual(rewrite("import xml.etree.cElementTree\nxml.etree.cElementTree.parse(f)\n", list(self.ID)),
                         "import xml.etree.ElementTree\nxml.etree.ElementTree.parse(f)\n")
        self.assertEqual(rewrite("from xml.etree.cElementTree import parse\n", list(self.ID)),
                         "from xml.etree.ElementTree import parse\n")

    def test_nested_attribute_path(self):
        self.assertEqual(rewrite("import asyncio\nasyncio.Task.current_task()\n", list(self.ID)),
                         "import asyncio\nasyncio.current_task()\n")

    def test_unbound_name_is_not_rewritten(self):
        src = "def f(logging):\n    logging.warn('x')\n"
        self.assertEqual(rewrite(src, list(self.ID)), src)


class TestCgiEscape(unittest.TestCase):
    ID = ("MIG-CGI-ESCAPE",)

    def test_default_quote_is_preserved(self):
        self.assertEqual(rewrite("import cgi\ny = cgi.escape(x)\n", list(self.ID)),
                         "import html\ny = html.escape(x, quote=False)\n")

    def test_quote_true_is_detect_only(self):
        src = "import cgi\ny = cgi.escape(x, True)\n"
        self.assertEqual(rewrite(src, list(self.ID)), src)
        self.assertEqual([f["fix"] for f in codemods.scan(src, "a.py", only=list(self.ID))], [None])


class TestDetection(unittest.TestCase):
    def test_removed_modules_skip_import_guards(self):
        src = "import distutils.core\ntry:\n    import telnetlib\nexcept ImportError:\n    telnetlib = None\n"
        found = codemods.scan(src, "a.py", only=["MIG-REMOVED-MODULES"])
        self.assertEqual([f["line_number"] for f in found], [1])

    def test_aliases_are_resolved(self):
        found = codemods.scan("from inspect import getargspec as gas\ngas(f)\n", "a.py", only=["MIG-REMOVED-APIS"])
        self.assertEqual([f["line_number"] for f in found], [1, 2])

    def test_attribute_names_are_not_confused_with_imports(self):
        src = "from inspect import getargspec\nself.getargspec()\n"
        found = codemods.scan(src, "a.py", only=["MIG-REMOVED-APIS"])
        self.assertEqual([f["line_number"] for f in found], [1])

    def test_library_rules_need_the_library(self):
        self.assertEqual(codemods.scan("x.parse_obj(1)\n", "a.py", only=["MIG-PYDANTIC-V2"]), [])
        found = codemods.scan("import pydantic\nM.parse_obj(1)\n", "a.py", only=["MIG-PYDANTIC-V2"])
        self.assertEqual([f["line_number"] for f in found], [2])

    def test_naive_utcnow_through_from_import(self):
        found = codemods.scan("from datetime import datetime\ndatetime.utcnow()\n", "a.py",
                              only=["MIG-DATETIME-UTCNOW"])
        self.assertEqual([f["line_number"] for f in found], [2])


class TestCompatibilityShims(unittest.TestCase):
    """Code that keeps an old API on purpose for old versions is neither reported nor rewritten."""

    def assertUntouched(self, src: str) -> None:
        self.assertEqual(codemods.scan(src, "a.py"), [])
        self.assertEqual(rewrite(src), src)

    def test_version_checks(self):
        self.assertUntouched("import sys\nif sys.version_info >= (3, 3):\n    import importlib\nelse:\n"
                             "    import imp\n    imp.reload(m)\n")

    def test_hasattr_checks(self):
        self.assertUntouched("import base64\nif hasattr(base64, 'decodebytes'):\n    f = base64.decodebytes\n"
                             "else:\n    f = base64.decodestring\n")

    def test_fallback_handlers(self):
        self.assertUntouched("import asyncio\ntry:\n    t = asyncio.current_task\nexcept AttributeError:\n"
                             "    t = asyncio.Task.current_task\n")
        self.assertUntouched("try:\n    from collections.abc import Mapping\nexcept ImportError:\n"
                             "    from collections import Mapping\n")

    def test_local_stubs_are_not_uses(self):
        src = "from pydantic import Extra\n\nclass Extra:\n    pass\n\nExtra = None\n"
        self.assertEqual([f["line_number"] for f in codemods.scan(src, "a.py")], [1])  # the import, not the stubs

    def test_code_outside_the_shim_is_still_migrated(self):
        src = ("import base64, sys\nif sys.version_info < (3,):\n    x = base64.encodestring(b'')\n"
               "y = base64.encodestring(b'')\n")
        self.assertEqual([f["line_number"] for f in codemods.scan(src, "a.py")], [4])
        self.assertEqual(rewrite(src), src.replace("y = base64.encodestring", "y = base64.encodebytes"))

    def test_no_import_added_for_suppressed_rewrites(self):
        src = "import imp, sys\nif sys.version_info < (3,):\n    imp.reload(m)\n"
        self.assertEqual(rewrite(src), src)


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
        self.assertTrue(any("MIG-RENAMED-APIS" in n for n in notes))

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
