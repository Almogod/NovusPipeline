"""Ruff backend, dynamic-import resolution and codebase-level smells."""
import unittest

from novuspipeline import codebase_context, local_llm, ruff_backend
from novuspipeline.modernizer import CodeModernizer
from novuspipeline.server import analyze_legacy_codebase, get_codebase_overview
from test_server import LEGACY_MODULE, WorkspaceTestCase  # also applies the test environment settings

RUFF = ruff_backend.available()[0]


@unittest.skipUnless(RUFF, "Ruff not installed")
class TestRuffBackend(unittest.TestCase):

    def test_scan_maps_severity_category_and_skips_builtin_overlaps(self):
        code = "import pickle\n\nclass C(object):\n    pass\n\ndef f(s):\n    try:\n        return eval(s)\n    except:\n        return None\n"
        ids = {f["smell_id"]: f for f in ruff_backend.scan_code(code, "m.py")}
        self.assertEqual((ids["RUFF-S307"]["severity"], ids["RUFF-S307"]["category"]), ("HIGH", "security"))
        self.assertEqual(ids["RUFF-UP004"]["fix"], "safe")
        for overlap in ("RUFF-E722", "RUFF-S403", "RUFF-S301"):  # the built-in rules own these
            self.assertNotIn(overlap, ids)

    def test_python2_syntax_is_reported_as_critical(self):
        ids = {f["smell_id"]: f["severity"] for f in ruff_backend.scan_code('print "x"\n', "m.py")}
        self.assertEqual(ids.get("RUFF-SYNTAX"), "CRITICAL")

    def test_only_safe_fixes_are_applied(self):
        code = 'class C(object):\n    pass\n\ns = "%s" % 1\n'
        fixed, codes = ruff_backend.apply_safe_fixes(code, "m.py")
        self.assertIn("class C:", fixed)
        self.assertIn('"%s" % 1', fixed)  # UP031 is an unsafe fix: left alone
        self.assertEqual(codes, ["RUFF-UP004"])

    def test_modernizer_applies_ruff_safe_fixes(self):
        out, notes = CodeModernizer.modernize_python("class C(object):\n    pass\n")
        self.assertIn("class C:", out)
        self.assertTrue(any("Ruff safe fixes" in n for n in notes), notes)

    def test_llm_validation_rejects_new_undefined_names(self):
        v = local_llm.validate_proposal("def f(x):\n    return x\n", "def f(x: int) -> int:\n    return y\n", ["f"])
        self.assertFalse(v["valid"])
        self.assertIn("RUFF-F821", " ".join(v["issues"]))


class TestDynamicImports(WorkspaceTestCase):

    def index(self) -> codebase_context.CodebaseIndex:
        idx = codebase_context.CodebaseIndex(self.root)
        idx.refresh()
        return idx

    def test_importlib_and_dunder_import_create_edges(self):
        self.write("pkg/__init__.py", "")
        self.write("pkg/plugin.py", "def run() -> int:\n    return 1\n")
        self.write("pkg/sub.py", "X = 1\n")
        self.write("pkg/loader.py", "import importlib\n\ndef load():\n    m = importlib.import_module('pkg.plugin')\n"
                                    "    s = importlib.import_module('.sub', __package__)\n    return getattr(m, 'run')()\n")
        self.write("legacy.py", "def f():\n    return __import__('pkg.sub')\n")
        idx = self.index()
        self.assertEqual(idx.dependents("pkg/plugin.py"), {"pkg/loader.py": ["run"]})  # getattr counts as use
        self.assertEqual(set(idx.dependents("pkg/sub.py")), {"pkg/loader.py", "legacy.py"})

    def test_from_import_alias_of_import_module(self):
        self.write("tools.py", "def helper() -> int:\n    return 1\n")
        self.write("app.py", "from importlib import import_module as im\n\ntools = im('tools')\nprint(tools.helper())\n")
        self.assertIn("app.py", self.index().dependents("tools.py"))

    def test_star_import_resolves_exports(self):
        self.write("lib.py", "__all__ = ['a']\n\ndef a() -> int:\n    return 1\n\ndef _b() -> int:\n    return 2\n")
        self.write("app.py", "from lib import *\n")
        self.assertEqual(self.index().dependents("lib.py"), {"app.py": ["a"]})

    def test_unresolvable_dynamic_import_is_reported_and_keeps_plugins_alive(self):
        self.write("plugins/__init__.py", "")
        self.write("plugins/alpha.py", "def activate() -> None:\n    pass\n")
        self.write("host.py", "import importlib\n\ndef load(name: str):\n    return importlib.import_module(f'plugins.{name}')\n")
        smells = self.index().context_smells()
        dynamic = [s for s in smells["host.py"] if s["smell_id"] == "CTX-DYNAMIC"]
        self.assertEqual(len(dynamic), 1)
        self.assertIn("plugins", dynamic[0]["detail"])
        self.assertFalse(any(s["smell_id"] == "CTX-DEAD" for s in smells["plugins/alpha.py"]))


class TestContextSmells(WorkspaceTestCase):

    def smells(self) -> dict:
        idx = codebase_context.CodebaseIndex(self.root)
        idx.refresh()
        return {rel: {s["smell_id"]: s for s in items} for rel, items in idx.context_smells().items()}

    def test_import_time_cycle_detected_but_lazy_import_is_not(self):
        self.write("a.py", "import b\n\ndef fa() -> int:\n    return b.fb()\n")
        self.write("b.py", "import a\n\ndef fb() -> int:\n    return 1\n")
        self.write("c.py", "def fc() -> int:\n    import d\n    return d.fd()\n")
        self.write("d.py", "import c\n\ndef fd() -> int:\n    return c.fc()\n")
        smells = self.smells()
        self.assertIn("CTX-CYCLE", smells["a.py"])
        self.assertIn("CTX-CYCLE", smells["b.py"])
        self.assertNotIn("CTX-CYCLE", smells["c.py"])  # the lazy import inside fc() breaks the cycle

    def test_dead_code_precision(self):
        self.write("pyproject.toml", '[project]\nname = "x"\n[project.scripts]\nx = "lib:entry"\n')
        self.write("lib.py", "import functools\n\ndef unused() -> int:\n    return 1\n\ndef used() -> int:\n    return 2\n\n"
                             "def by_string() -> int:\n    return 3\n\n@functools.lru_cache\ndef decorated() -> int:\n"
                             "    return 4\n\ndef entry() -> None:\n    pass\n")
        self.write("app.py", "from lib import used\n\nHANDLERS = {'go': 'by_string'}\nused()\n")
        idx = codebase_context.CodebaseIndex(self.root)
        idx.refresh()
        dead = {s["detail"].split("`")[1] for s in idx.context_smells()["lib.py"] if s["smell_id"] == "CTX-DEAD"}
        self.assertEqual(dead, {"unused"})

    def test_structural_clone_detected_despite_renames(self):
        body = ("def {name}({a}: list) -> dict:\n    {out} = {{}}\n    for {item} in {a}:\n        if {item} is None:\n"
                "            continue\n        key = str({item}).strip().lower()\n        {out}[key] = {out}.get(key, 0) + 1\n"
                "    return {out}\n")
        self.write("one.py", body.format(name="count_words", a="words", out="counts", item="w"))
        self.write("two.py", body.format(name="tally", a="rows", out="totals", item="r"))
        self.write("three.py", "def other(x: int) -> int:\n    return x * 2\n")
        smells = self.smells()
        self.assertIn("CTX-CLONE", smells["one.py"])
        self.assertIn("two.py::tally", smells["one.py"]["CTX-CLONE"]["detail"])
        self.assertNotIn("CTX-CLONE", smells["three.py"])

    def test_untested_hotspot_and_deprecated_api(self):
        self.write(".novus/config.toml", '[deprecated]\n"core.old_api" = "use core.new_api"\n')
        self.write("core.py", "def old_api() -> int:\n    return 1\n\ndef new_api() -> int:\n    return 2\n")
        for i in range(3):
            self.write(f"user{i}.py", f"from core import {'old_api' if i == 0 else 'new_api'}\n")
        smells = self.smells()
        self.assertIn("CTX-UNTESTED", smells["core.py"])
        self.assertIn("use core.new_api", smells["user0.py"]["CTX-DEPRECATED"]["detail"])
        self.assertNotIn("CTX-DEPRECATED", smells["user1.py"])
        self.write("tests/test_core.py", "from core import new_api\n")
        self.assertNotIn("CTX-UNTESTED", self.smells()["core.py"])

    def test_overview_and_audit_surface_new_layers(self):
        self.write("lib.py", LEGACY_MODULE + "\nclass Old(object):\n    pass\n")
        self.write("app.py", "from lib import fetch\nfetch('u')\n")
        overview = get_codebase_overview()
        self.assertIn("Codebase-Level Smells", overview)
        if RUFF:
            self.assertIn("RUFF-", overview)
            report = analyze_legacy_codebase("lib.py")
            self.assertIn("RUFF-UP004", report)
            self.assertIn("safe auto-fix available", report)


if __name__ == "__main__":
    unittest.main()
