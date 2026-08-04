import os
import unittest
from server import read_legacy_file, query_rag_guidelines, run_local_tests, create_git_migration_pr

class TestNovusPipelineServer(unittest.TestCase):

    def test_read_legacy_file_valid(self):
        res = read_legacy_file("README.md")
        self.assertIn("# NovusPipeline", res)

    def test_read_legacy_file_security_path_traversal(self):
        res = read_legacy_file("../../Windows/System32/drivers/etc/hosts")
        self.assertIn("outside the authorized project workspace", res)

    def test_query_rag_guidelines(self):
        res = query_rag_guidelines("refactor python 2 print statement")
        self.assertIn("Compliance Guidelines", res)
        self.assertIn("Explicit Typing", res)

    def test_run_local_tests_allowed_tool(self):
        res = run_local_tests("python --version")
        self.assertIn("Python 3", res)

    def test_run_local_tests_blocked_tool(self):
        res = run_local_tests("powershell Get-Process")
        self.assertIn("not in the allowed local verification tools whitelist", res)

    def test_create_git_migration_pr(self):
        lock_file = os.path.join(os.getcwd(), ".git", "index.lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception:
                pass

        res = create_git_migration_pr(
            branch_name="test-modernization-branch",
            commit_message="test: add phase 1 scaffold",
            pr_title="Phase 1 Scaffolding PR",
            pr_description="Scaffold FastMCP server for legacy modernization."
        )
        self.assertTrue(
            "Checked out" in res or "Created and checked out" in res or "Draft PR metadata written" in res,
            f"Unexpected return: {res}"
        )
        self.assertTrue(os.path.exists(".novus_pr_test-modernization-branch.md"))

if __name__ == "__main__":
    unittest.main()
