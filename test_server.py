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
)


class TestNovusPipelineServerRobustness(unittest.TestCase):

    def test_read_legacy_file_valid(self):
        res = read_legacy_file("README.md")
        self.assertIn("# NovusPipeline", res)

    def test_read_legacy_file_security_path_traversal(self):
        res = read_legacy_file("../../Windows/System32/drivers/etc/hosts")
        self.assertIn("outside the authorized project workspace", res)

    def test_query_rag_guidelines_basic(self):
        res = query_rag_guidelines("refactor python 2 print statement")
        self.assertTrue(
            "Guidelines" in res or "Rules" in res or "Python" in res,
            f"Unexpected response: {res[:200]}"
        )

    def test_query_rag_guidelines_query_expansion(self):
        res = query_rag_guidelines("async blocking Network I/O")
        self.assertIn("Python Async & Concurrency", res)

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
            commit_message="test: Phase 2+ robust RAG pipeline",
            pr_title="Phase 2+ Robust RAG Enhancement PR",
            pr_description="Query expansion, BM25 TF-IDF hybrid, stats, and search by ID."
        )
        self.assertTrue(
            "Checked out" in res or "Created and checked out" in res or "Draft PR metadata written" in res,
            f"Unexpected return: {res}"
        )
        pr_file = ".novus_pr_test-modernization-branch.md"
        self.assertTrue(os.path.exists(pr_file), f"PR file not found: {pr_file}")


if __name__ == "__main__":
    unittest.main()
