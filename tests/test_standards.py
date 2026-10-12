"""Organization / industry modernization standards (rule packs)."""
import os
import unittest

from novuspipeline import local_llm, standards
from novuspipeline.server import (
    add_modernization_rule,
    analyze_legacy_codebase,
    apply_code_modernization,
    get_codebase_overview,
    ingest_rag_document,
    install_standard_pack,
    list_modernization_standards,
    query_rag_guidelines,
    remove_modernization_rule,
    remove_rag_document,
    run_autonomous_modernization_pipeline,
    search_rag_by_id,
)
from test_server import LEGACY_MODULE, WorkspaceTestCase  # also applies the test environment settings

HTTP_MODULE = (
    "import requests\nfrom yaml import load\n\n\n"
    "def fetch(url):\n    return requests.get(url)\n\n\n"
    "def config(stream):\n    return load(stream)\n"
)


class TestRuleValidation(unittest.TestCase):

    def test_valid_rule_is_normalized(self):
        rule, problems = standards.validate_rule({"id": "ACME-1", "calls": "float", "severity": "high"})
        self.assertEqual(problems, [])
        self.assertEqual((rule["calls"], rule["severity"], rule["enforce"]), (["float"], "HIGH", "warn"))

    def test_problems_are_reported(self):
        cases = [
            ({"id": "1bad", "calls": ["x"]}, "invalid id"),
            ({"id": "A", "title": "no detection"}, "needs at least one"),
            ({"id": "A", "pattern": "("}, "invalid `pattern`"),
            ({"id": "A", "calls": ["x"], "add_keyword": "timeout=1"}, "needs `require_keyword`"),
            ({"id": "A", "calls": ["x"], "require_keyword": "timeout", "add_keyword": "retries=2"}, "must look like"),
            ({"id": "A", "calls": ["x"], "severity": "URGENT"}, "severity"),
            ({"id": "A", "calls": ["x"], "replace_call": "not valid()"}, "dotted name"),
            ({"id": "A", "calls": ["x"], "typo_field": 1}, "unknown field"),
        ]
        for raw, expected in cases:
            rule, problems = standards.validate_rule(raw)
            self.assertIsNone(rule, raw)
            self.assertIn(expected, "; ".join(problems), raw)


class StandardsTestCase(WorkspaceTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.addCleanup(standards._CACHE.clear)

    def rule(self, **kwargs) -> str:
        out = add_modernization_rule(**{"rule_id": "ACME-1", "title": "t", "guideline": "g", **kwargs})
        self.assertIn("Saved rule", out, out)
        return out


class TestDetectionAndFixes(StandardsTestCase):

    def test_builtin_pack_detects_and_fixes(self):
        self.assertIn("Installed `secure-defaults`", install_standard_pack("secure-defaults"))
        active = standards.load(self.root)
        found = {(f["smell_id"], f["line_number"]) for f in active.scan(HTTP_MODULE, "svc.py")}
        self.assertEqual(found, {("SEC-TIMEOUT", 6), ("SEC-YAML-LOAD", 10)})  # alias `load` resolved
        fixed, notes = active.apply_fixes(HTTP_MODULE, "svc.py")
        self.assertIn("requests.get(url, timeout=30)", fixed)
        self.assertIn("yaml.safe_load(stream)", fixed)
        self.assertIn("import yaml\n", fixed)  # the fix adds the import it needs
        self.assertEqual(active.scan(fixed, "svc.py"), [])
        self.assertEqual(len(notes), 2)

    def test_keyword_present_or_forwarded_is_not_flagged(self):
        install_standard_pack("secure-defaults")
        code = "import requests\n\ndef f(u, **kw):\n    requests.get(u, timeout=3)\n    requests.get(u, **kw)\n"
        self.assertEqual(standards.load(self.root).scan(code, "a.py"), [])

    def test_patterns_ignore_strings_and_comments(self):
        install_standard_pack("secure-defaults")
        code = "import requests\nMSG = 'never verify=False'  # verify=False\nrequests.get('u', timeout=1, verify=False)\n"
        found = [(f["smell_id"], f["line_number"]) for f in standards.load(self.root).scan(code, "a.py")]
        self.assertEqual(found, [("SEC-TLS-VERIFY", 3)])

    def test_import_replacement_keeps_references_working(self):
        self.rule(imports=["simplejson"], replace_import="json")
        code = "import simplejson\nfrom simplejson import dumps\n\nx = simplejson.loads('1')\n"
        fixed, _ = standards.load(self.root).apply_fixes(code, "a.py")
        self.assertEqual(fixed, "import json as simplejson\nfrom json import dumps\n\nx = simplejson.loads('1')\n")

    def test_regex_replacement_and_path_scope(self):
        self.rule(pattern=r"\bfloat\(", replacement="Decimal(", paths=["payments/**"])
        code = "from decimal import Decimal\n\ndef total(x):\n    return float(x)\n"
        active = standards.load(self.root)
        self.assertEqual(active.scan(code, "reports/x.py"), [])
        fixed, _ = active.apply_fixes(code, "payments/ledger.py")
        self.assertIn("return Decimal(x)", fixed)

    def test_user_scope_applies_to_every_project(self):
        self.rule(rule_id="ME-1", calls=["print"], scope="user")
        other = standards.Standards(os.path.join(self.root, "elsewhere"))
        self.assertIsNotNone(other.get("ME-1"))
        self.assertEqual(other.get("ME-1")["scope"], "user")
        remove_modernization_rule("ME-1", scope="user")

    def test_broken_pack_is_reported_not_fatal(self):
        self.write(".novus/standards/team.toml", "[[rule]]\nid = 'T-1'\ntitle = 'no detection'\n")
        self.write(".novus/standards/broken.toml", "this is not toml [")
        out = list_modernization_standards()
        self.assertIn("needs at least one of", out)
        self.assertIn("not valid TOML", out)

    def test_rules_in_hand_written_packs_are_not_overwritten(self):
        self.write(".novus/standards/team.toml", "[[rule]]\nid = 'T-1'\ncalls = ['x.y']\n")
        self.assertIn("edit that file", add_modernization_rule("T-1", "t", "g", calls=["a.b"]))
        self.assertIn("edit that file", remove_modernization_rule("T-1"))

    def test_add_list_remove_roundtrip(self):
        out = self.rule(calls=["os.getcwdu"], replace_call="os.getcwd", severity="LOW",
                        guideline='Use "os.getcwd()"; quotes\nand newlines survive.')
        self.assertIn("ACME-1", list_modernization_standards())
        self.assertIn('quotes\nand newlines survive', standards.load(self.root).get("ACME-1")["guideline"])
        self.assertIn("ACME-1", out)
        self.assertIn("Removed rule", remove_modernization_rule("ACME-1"))
        self.assertIsNone(standards.load(self.root).get("ACME-1"))


class TestStandardsInTools(StandardsTestCase):

    def test_audit_and_overview_show_standards(self):
        install_standard_pack("secure-defaults")
        self.write("svc.py", HTTP_MODULE)
        audit = analyze_legacy_codebase("svc.py")
        self.assertIn("from your modernization standards", audit)
        self.assertIn("`SEC-TIMEOUT`", audit)
        self.assertIn("**Standard** `secure-defaults`", audit)
        self.assertNotIn("RUFF-S113", audit)  # superseded by the pack's [ruff] ignore
        self.assertIn("SEC-TIMEOUT [standard]", get_codebase_overview())

    def test_rag_returns_standards_and_custom_categories(self):
        install_standard_pack("secure-defaults")
        self.assertIn("[standard: secure-defaults] `SEC-TIMEOUT`", query_rag_guidelines("http request timeout hangs"))
        self.assertIn("HTTP requests must set a timeout", search_rag_by_id("SEC-TIMEOUT"))
        self.assertIn("Successfully ingested", ingest_rag_document(
            "fin-001", "Money handling", "fintech", "Represent money as Decimal, never float."))
        self.assertIn("Money handling", query_rag_guidelines("money decimal float", category="fintech"))
        self.assertIn("Removed document", remove_rag_document("fin-001"))
        self.assertIn("not found", search_rag_by_id("fin-001"))

    def test_apply_uses_standard_fixes(self):
        self.rule(calls=["os.getcwdu"], replace_call="os.getcwd")
        self.write("a.py", "import os\n\n\ndef here() -> str:\n    return os.getcwdu()\n")
        out = apply_code_modernization("a.py")
        self.assertIn("Applied standard ACME-1", out)
        self.assertIn("os.getcwd()", self.read("a.py"))

    def test_pipeline_blocks_on_remaining_violation(self):
        self.rule(rule_id="ACME-NO-PICKLE", imports=["pickle"], enforce="block", severity="HIGH")
        self.write("calc.py", "import pickle\n" + LEGACY_MODULE)
        self.commit_all()
        res = run_autonomous_modernization_pipeline("calc.py", test_command="python --version", branch_name="std")
        self.assertIn("still violates blocking standards", res)
        self.assertIn("urllib.request", res)  # the prepared changes are still listed
        self.assertIn("ACME-NO-PICKLE", res)
        self.assertNotIn("std", [b.name for b in self.repo.branches])

    def test_pipeline_commits_fixed_file(self):
        self.rule(calls=["os.getcwdu"], replace_call="os.getcwd", enforce="block")
        self.write("a.py", "import os\n\n\ndef here() -> str:\n    return os.getcwdu()\n")
        self.commit_all()
        res = run_autonomous_modernization_pipeline("a.py", test_command="python --version", branch_name="std")
        self.assertIn("Completed Successfully", res, res)
        self.assertIn("os.getcwd()", self.repo.git.show("std:a.py"))

    def test_llm_gate_rejects_new_violations(self):
        install_standard_pack("secure-defaults")
        active = standards.load(self.root)
        v = local_llm.validate_proposal("import requests\n\ndef f(u):\n    return u\n",
                                        "import requests\n\ndef f(u):\n    return requests.get(u)\n",
                                        standards_scan=lambda c: active.scan(c, "a.py"))
        self.assertFalse(v["valid"])
        self.assertIn("Violates organization standards: SEC-TIMEOUT", "; ".join(v["issues"]))
        self.assertIn("SEC-TIMEOUT, violated in this file",
                      active.prompt_block(active.scan("import requests\nrequests.get('u')\n", "a.py")))


if __name__ == "__main__":
    unittest.main()
