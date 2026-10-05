import os
import unittest
from server import (
    read_legacy_file,
    query_rag_guidelines,
    search_rag_by_id,
    get_rag_stats,
    ingest_rag_document,
    reset_rag_database,
    run_local_tests,
    create_git_migration_pr,
    analyze_legacy_codebase,
    apply_code_modernization,
    run_autonomous_modernization_pipeline,
    get_local_llm_status,
    generate_llm_modernization_proposal,
    format_modernization_report,
    finalize_git_migration_pr,
    is_path_in_workspace,
    stable_token_index,
    get_gnn_model_status,
    analyze_code_structure_gnn,
)
from modernizer import LegacySmellDetector, CodeModernizer
import local_llm
from code_graph import build_graph_from_code, NODE_TYPES
from reporter import ModernizationReporter


class TestNovusPipelineServerPhase3(unittest.TestCase):

    def test_read_legacy_file_valid(self):
        res = read_legacy_file("README.md")
        self.assertIn("# NovusPipeline", res)

    def test_read_legacy_file_security_path_traversal(self):
        res = read_legacy_file("../../Windows/System32/drivers/etc/hosts")
        self.assertIn("outside authorized project workspace", res)

    def test_is_path_in_workspace_rejects_sibling_directory_prefix(self):
        # A sibling directory that merely shares the workspace root as a string
        # prefix (e.g. "...\NovusPipelineEVIL") must NOT be treated as in-workspace.
        workspace_root = os.path.abspath(os.getcwd())
        sibling_path = workspace_root + "EVIL" + os.sep + "secret.txt"
        self.assertFalse(is_path_in_workspace(sibling_path))

    def test_is_path_in_workspace_accepts_real_subpath_and_root(self):
        workspace_root = os.path.abspath(os.getcwd())
        self.assertTrue(is_path_in_workspace(os.path.join(workspace_root, "server.py")))
        self.assertTrue(is_path_in_workspace(workspace_root))

    def test_query_rag_guidelines_basic(self):
        res = query_rag_guidelines("refactor python 2 print statement")
        self.assertTrue(
            "Guidelines" in res or "Rules" in res or "Python" in res,
            f"Unexpected response: {res[:200]}"
        )

    def test_query_rag_guidelines_query_expansion(self):
        res = query_rag_guidelines("async blocking io", category="python")
        self.assertTrue("Python" in res or "Guidelines" in res)

    def test_query_rag_guidelines_category_filter(self):
        res = query_rag_guidelines("path traversal file access", category="security", n_results=2)
        self.assertIsInstance(res, str)
        self.assertGreater(len(res), 10)

    def test_query_rag_guidelines_n_results(self):
        res = query_rag_guidelines("python typing modernization", n_results=1)
        self.assertIsInstance(res, str)
        self.assertGreater(len(res), 10)

    def test_search_rag_by_id_valid(self):
        res = search_rag_by_id("py-001")
        self.assertIn("Python Type System Modernization", res)
        self.assertIn("py-001", res)

    def test_search_rag_by_id_not_found(self):
        res = search_rag_by_id("nonexistent-id-999")
        self.assertIn("Error", res)
        self.assertIn("was not found", res)

    def test_stable_token_index_deterministic_across_processes(self):
        # Must not rely on Python's salted builtin hash() - ingestion (ingest_rag.py)
        # and querying (server.py) run as separate processes against the same
        # persistent ChromaDB collection, so the token->dimension mapping must be
        # identical every time regardless of PYTHONHASHSEED.
        import subprocess
        import sys

        script = (
            "import hashlib; "
            "from server import stable_token_index; "
            "print(stable_token_index('python', 256))"
        )
        results = set()
        for _ in range(2):
            out = subprocess.run(
                [sys.executable, "-c", script],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
            )
            results.add(out.stdout.strip())
        self.assertEqual(len(results), 1, f"stable_token_index varied across processes: {results}")

    def test_get_rag_stats(self):
        res = get_rag_stats()
        self.assertIn("Total Guidelines", res)
        self.assertIn("Operational", res)

    def test_ingest_rag_document_valid(self):
        res = ingest_rag_document(
            document_id="test-doc-unit-001",
            title="Test Guideline",
            category="python",
            content="Always use f-strings instead of % formatting for readability."
        )
        self.assertIn("Successfully ingested", res)
        self.assertIn("test-doc-unit-001", res)

    def test_ingest_rag_document_invalid_category(self):
        res = ingest_rag_document(
            document_id="test-doc-unit-002",
            title="Test Rule",
            category="ruby",
            content="Some content."
        )
        self.assertIn("Error", res)
        self.assertIn("category", res)

    def test_reset_rag_database(self):
        res = reset_rag_database()
        self.assertIn("Successfully reset", res)

    def test_run_local_tests_allowed_tool(self):
        res = run_local_tests("python --version")
        self.assertIn("Python 3", res)

    def test_run_local_tests_blocked_tool(self):
        res = run_local_tests("powershell Get-Process")
        self.assertIn("not in the allowed local verification tools whitelist", res)

    def test_run_local_tests_command_injection_prevention(self):
        res = run_local_tests("python --version; echo hacked")
        self.assertIn("dangerous shell chaining character", res)

    def test_create_git_migration_pr(self):
        lock_file = os.path.join(os.getcwd(), ".git", "index.lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception:
                pass

        res = create_git_migration_pr(
            branch_name="test-modernization-branch",
            commit_message="test: Phase 3 autonomous pipeline",
            pr_title="Phase 3 Autonomous Modernization PR",
            pr_description="Autonomous loop: analyze -> RAG -> modernize -> test -> PR."
        )
        self.assertTrue(
            "Checked out" in res or "Created and checked out" in res or "Draft PR metadata written" in res,
            f"Unexpected return: {res}"
        )
        pr_file = ".novus_pr_test-modernization-branch.md"
        self.assertTrue(os.path.exists(pr_file), f"PR file not found: {pr_file}")

    # -----------------------------------------------------------------------
    # Phase 3 Tests
    # -----------------------------------------------------------------------

    def test_legacy_smell_detector(self):
        code = "import urllib2\ntry:\n    pass\nexcept:\n    pass"
        findings = LegacySmellDetector.scan_code(code, "sample.py")
        self.assertGreaterEqual(len(findings), 2)
        smell_ids = [f["smell_id"] for f in findings]
        self.assertIn("PY-SMELL-002", smell_ids)
        self.assertIn("PY-SMELL-003", smell_ids)

    def test_code_modernizer_python(self):
        code = "import urllib2\ndef run():\n    try:\n        pass\n    except:\n        pass"
        mod_code, changes = CodeModernizer.modernize_python(code)
        self.assertIn("import httpx", mod_code)
        self.assertIn("except Exception as e:", mod_code)
        self.assertIn("from __future__ import annotations", mod_code)

    def test_analyze_legacy_codebase(self):
        test_file = "sample_legacy.py"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("import urllib2\nvar_x = 10\ntry:\n    pass\nexcept:\n    pass")

        try:
            report = analyze_legacy_codebase(test_file)
            self.assertIn("Modernization Audit Report", report)
            self.assertIn("PY-SMELL-002", report)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_apply_code_modernization(self):
        test_file = "sample_mod.py"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("import urllib2\ntry:\n    pass\nexcept:\n    pass")

        try:
            res = apply_code_modernization(test_file)
            self.assertIn("Successfully updated", res)
            self.assertIn(".bak", res)
            
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("import httpx", content)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(test_file + ".bak"):
                os.remove(test_file + ".bak")

    def test_run_autonomous_modernization_pipeline(self):
        test_file = "sample_pipe.py"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("import urllib2\ntry:\n    pass\nexcept:\n    pass")

        try:
            res = run_autonomous_modernization_pipeline(
                file_path=test_file,
                test_command="python --version",
                branch_name="test-auto-pipeline-branch"
            )
            self.assertIn("Autonomous Modernization Pipeline Completed Successfully", res)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(test_file + ".bak"):
                os.remove(test_file + ".bak")

    def test_get_local_llm_status(self):
        res = get_local_llm_status()
        self.assertIn("Integrated Local LLM Status", res)
        self.assertIn("unsloth_Qwen3.5-2B_1785882774", res)

    def test_generate_llm_modernization_proposal(self):
        test_file = "sample_proposal.py"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("import urllib2\ndef run():\n    pass")

        try:
            res = generate_llm_modernization_proposal(test_file, rag_query="httpx urllib2")
            self.assertIn("LLM Modernization Proposal", res)
            self.assertIn("unsloth_Qwen3.5-2B_1785882774", res)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_modernization_reporter(self):
        report = ModernizationReporter.generate_report(
            file_path="README.md",
            audit_summary="Audit summary test",
            modernization_details="Modernization details test",
            test_output="Exit Code: 0",
            branch_name="test-report-branch"
        )
        self.assertIn("NovusPipeline Codebase Modernization Report", report)
        self.assertIn("test-report-branch", report)

    def test_format_modernization_report(self):
        res = format_modernization_report("README.md", branch_name="test-report-branch-tool", test_command="python --version")
        self.assertIn("Successfully generated and saved Modernization Report artifact", res)

    # -----------------------------------------------------------------------
    # Phase 5 Tests: AST Graph Construction & GNN Structural Classifier
    # -----------------------------------------------------------------------

    def test_build_graph_from_code_structure(self):
        code = "import urllib2\ntry:\n    pass\nexcept:\n    pass\n"
        graph = build_graph_from_code(code)
        self.assertGreater(graph["num_nodes"], 0)
        self.assertEqual(len(graph["node_type_ids"]), graph["num_nodes"])
        self.assertEqual(len(graph["node_features"]), graph["num_nodes"])
        self.assertTrue(all(0 <= t < len(NODE_TYPES) for t in graph["node_type_ids"]))
        self.assertGreater(len(graph["edge_index"]), 0)

    def test_build_graph_from_code_invalid_syntax_raises(self):
        with self.assertRaises(SyntaxError):
            build_graph_from_code("def broken(:\n")

    def test_get_gnn_model_status(self):
        res = get_gnn_model_status()
        self.assertIn("GNN Structural Smell Classifier Status", res)
        self.assertIn("Torch Available", res)

    def test_analyze_code_structure_gnn_rejects_non_python(self):
        res = analyze_code_structure_gnn("rules.md")
        self.assertIn("Error", res)
        self.assertIn("Python", res)

    def test_analyze_code_structure_gnn_on_smelly_file(self):
        test_file = "sample_gnn_smelly.py"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("import pickle\ndef load(path):\n    try:\n        with open(path, 'rb') as f:\n            return pickle.loads(f.read())\n    except:\n        return None\n")

        try:
            res = analyze_code_structure_gnn(test_file)
            if "GNN model unavailable" in res:
                self.skipTest("GNN checkpoint not trained in this environment; run train_gnn.py first.")
            self.assertIn("GNN Structural Analysis", res)
            self.assertIn("PY-SMELL-003", res)
            self.assertIn("PY-SMELL-005", res)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_finalize_git_migration_pr(self):
        lock_file = os.path.join(os.getcwd(), ".git", "index.lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception:
                pass

        res = finalize_git_migration_pr(
            branch_name="test-final-pr-branch",
            base_branch="main",
            pr_title="Finalized PR Modernization",
            pr_description="Full Phase 4 PR summary."
        )
        self.assertIn("NovusPipeline Git PR Finalization", res)
        self.assertIn("gh pr create", res)


if __name__ == "__main__":
    unittest.main()
