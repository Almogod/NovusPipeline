import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("NOVUS_FAST_TEST", "1")  # never load the multi-GB local LLM in tests

import git

import codebase_context
import gnn_model
import ingest_rag
import local_llm
import server
from code_graph import NODE_TYPES, build_graph_from_code, identifier_bucket
from modernizer import CodeModernizer, LegacySmellDetector, mask_strings_and_comments
from reporter import ModernizationReporter
from server import (
    analyze_code_structure_gnn,
    analyze_legacy_codebase,
    apply_code_modernization,
    create_git_migration_pr,
    finalize_git_migration_pr,
    find_similar_code,
    format_modernization_report,
    generate_llm_modernization_proposal,
    get_codebase_overview,
    get_file_context,
    get_gnn_model_status,
    get_local_llm_status,
    get_rag_stats,
    index_codebase,
    ingest_rag_document,
    is_path_in_workspace,
    query_rag_guidelines,
    read_legacy_file,
    reset_rag_database,
    run_autonomous_modernization_pipeline,
    run_local_tests,
    search_rag_by_id,
)


def gnn_ready() -> bool:
    status = gnn_model.get_gnn_status()
    return status["checkpoint_exists"] and status["torch_available"]


def smell_ids(code: str, path: str = "x.py") -> list:
    return [(f["smell_id"], f["line_number"]) for f in LegacySmellDetector.scan_code(code, path)]


class WorkspaceTestCase(unittest.TestCase):
    """
    Runs against a throwaway git repository with `server.WORKSPACE_ROOT` patched
    to it, so tests never create branches, commits or files in this repository.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = os.path.realpath(self._tmp.name)
        patcher = mock.patch.object(server, "WORKSPACE_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

        self.repo = git.Repo.init(self.root, initial_branch="main")
        with self.repo.config_writer() as cfg:  # local to the throwaway repo
            cfg.set_value("user", "name", "Novus Test")
            cfg.set_value("user", "email", "novus-test@example.invalid")
            cfg.set_value("commit", "gpgsign", "false")
        self.write("README.md", "# temp workspace\n")
        self.repo.git.add("-A")
        self.repo.git.commit("-m", "init")

    def write(self, rel: str, content: str) -> str:
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def read(self, rel: str) -> str:
        with open(os.path.join(self.root, rel), "r", encoding="utf-8") as f:
            return f.read()

    def commit_all(self, message: str = "fixture") -> None:
        self.repo.git.add("-A")
        self.repo.git.commit("-m", message)

    def committed_files(self, ref: str = "HEAD") -> set:
        return set(self.repo.git.show("--name-only", "--format=", ref).split())


# ---------------------------------------------------------------------------
# Files, paths, RAG
# ---------------------------------------------------------------------------

class TestFilesAndPaths(unittest.TestCase):

    def test_read_legacy_file_valid(self):
        self.assertIn("# NovusPipeline", read_legacy_file("README.md"))

    def test_read_legacy_file_security_path_traversal(self):
        self.assertIn("outside authorized project workspace", read_legacy_file("../../Windows/System32/drivers/etc/hosts"))

    def test_is_path_in_workspace_rejects_sibling_directory_prefix(self):
        # "...\NovusPipelineEVIL" shares the workspace root as a string prefix only.
        self.assertFalse(is_path_in_workspace(server.WORKSPACE_ROOT + "EVIL" + os.sep + "secret.txt"))

    def test_is_path_in_workspace_accepts_real_subpath_and_root(self):
        self.assertTrue(is_path_in_workspace(os.path.join(server.WORKSPACE_ROOT, "server.py")))
        self.assertTrue(is_path_in_workspace(server.WORKSPACE_ROOT))

    @unittest.skipUnless(os.name == "nt", "Windows paths are case-insensitive")
    def test_is_path_in_workspace_case_insensitive_on_windows(self):
        self.assertTrue(is_path_in_workspace(os.path.join(server.WORKSPACE_ROOT.upper(), "server.py")))


class TestRag(unittest.TestCase):

    def test_query_rag_guidelines_basic(self):
        res = query_rag_guidelines("refactor python 2 print statement")
        self.assertTrue("Guidelines" in res or "Rules" in res or "Python" in res, res[:200])

    def test_query_rag_guidelines_uses_query_expansion(self):
        # "async" expands to asyncio/taskgroup/..., which only the async guideline contains.
        res = query_rag_guidelines("async", n_results=1)
        self.assertIn("Async", res)

    def test_query_rag_guidelines_category_filter(self):
        res = query_rag_guidelines("path traversal file access", category="security", n_results=2)
        self.assertIn("[security]", res)
        self.assertNotIn("[python]", res)

    def test_query_rag_guidelines_n_results(self):
        self.assertGreater(len(query_rag_guidelines("python typing modernization", n_results=1)), 10)

    def test_search_rag_by_id_valid(self):
        res = search_rag_by_id("py-001")
        self.assertIn("Python Type System Modernization", res)

    def test_search_rag_by_id_not_found(self):
        res = search_rag_by_id("nonexistent-id-999")
        self.assertIn("was not found", res)

    def test_get_rag_stats(self):
        res = get_rag_stats()
        self.assertIn("Total Guidelines", res)
        self.assertIn("Operational", res)

    def test_ingest_rag_document_valid(self):
        res = ingest_rag_document("test-doc-unit-001", "Test Guideline", "python",
                                  "Always use f-strings instead of % formatting for readability.")
        self.assertIn("Successfully ingested", res)

    def test_ingest_rag_document_invalid_category(self):
        res = ingest_rag_document("test-doc-unit-002", "Test Rule", "ruby", "Some content.")
        self.assertIn("category", res)

    def test_reset_rag_database_keeps_stdout_clean(self):
        # Under MCP stdio, stdout is the JSON-RPC channel; progress must go to stderr.
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            res = reset_rag_database()
        self.assertIn("Successfully reset", res)
        self.assertEqual(out.getvalue(), "")

    def test_tokenizer_splits_camel_case(self):
        self.assertEqual(ingest_rag.TFIDFEmbeddingFunction()._tokenize("TaskGroup"), ["task", "group"])

    def test_stable_token_index_deterministic_across_processes(self):
        script = "from ingest_rag import stable_token_index; print(stable_token_index('python', 256))"
        outputs = {subprocess.run([sys.executable, "-c", script], cwd=os.path.dirname(os.path.abspath(__file__)),
                                  capture_output=True, text=True).stdout.strip() for _ in range(2)}
        self.assertEqual(len(outputs), 1, outputs)

    def test_stale_embedding_version_is_reembedded_keeping_documents(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
                mock.patch.object(ingest_rag, "CHROMA_DIR", tmp), mock.patch.dict(ingest_rag._CLIENTS, clear=True):
            client = ingest_rag.get_client()
            old = client.create_collection(ingest_rag.COLLECTION_NAME, embedding_function=ingest_rag.TFIDFEmbeddingFunction(),
                                           metadata={"embedding_version": "tfidf_256_v1"})
            old.upsert(ids=["custom-1"], documents=["custom guideline text"], metadatas=[{"title": "c", "category": "python"}])
            migrated = ingest_rag.get_collection()
            self.assertEqual(migrated.metadata["embedding_version"], ingest_rag.EMBEDDING_VERSION)
            self.assertEqual(migrated.get(ids=["custom-1"])["documents"], ["custom guideline text"])


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

class TestLegacySmellDetector(unittest.TestCase):

    def test_detects_urllib2_and_bare_except(self):
        ids = {s for s, _ in smell_ids("import urllib2\ntry:\n    pass\nexcept:\n    pass")}
        self.assertTrue({"PY-SMELL-002", "PY-SMELL-003"} <= ids)

    def test_modern_urllib_is_not_a_smell(self):
        self.assertEqual(smell_ids("import urllib.parse\nimport urllib3\nfrom urllib.request import urlopen\n"), [])

    def test_python2_urllib_api_is_a_smell(self):
        self.assertIn(("PY-SMELL-002", 1), smell_ids("x = urllib.quote(s)\n"))

    def test_strings_and_comments_are_ignored(self):
        code = 'x = 1  # never use bare except: here\ndef f() -> None:\n    """avoid pickle.loads(x)"""\n'
        self.assertEqual(smell_ids(code), [])

    def test_mask_preserves_length_and_lines(self):
        code = 's = "abc"  # comment\nt = f"{s} x"\n'
        masked = mask_strings_and_comments(code)
        self.assertEqual(len(masked), len(code))
        self.assertEqual(masked.count("\n"), code.count("\n"))

    def test_bare_except_reported_once(self):
        self.assertEqual(smell_ids("try:\n    pass\nexcept:\n    pass\n"), [("PY-SMELL-003", 3)])

    def test_missing_type_hints_cases(self):
        self.assertIn(("PY-SMELL-006", 1), smell_ids("def f(x=dict()):\n    return x\n"))
        self.assertIn(("PY-SMELL-006", 1), smell_ids("def f(\n    a,\n    b,\n):\n    return a\n"))
        self.assertIn(("PY-SMELL-006", 1), smell_ids("def f(a) -> int:\n    return a\n"))
        self.assertEqual(smell_ids("class C:\n    def m(self, a: int) -> int:\n        return a\n"), [])
        self.assertEqual(smell_ids("def f(a: int, *xs: int, **kw: str) -> int:\n    return a\n"), [])

    def test_python2_source_falls_back_to_regexes(self):
        ids = {s for s, _ in smell_ids('print "hi"\ntry:\n    x\nexcept:\n    pass\n')}
        self.assertEqual(ids, {"PY-SMELL-001", "PY-SMELL-003"})

    def test_typescript_files_get_only_typescript_rules(self):
        ids = {s for s, _ in smell_ids("var x = 1;\nimport urllib2\n", "a.ts")}
        self.assertEqual(ids, {"TS-SMELL-001"})

    def test_pathologically_nested_source_does_not_crash(self):
        self.assertIsInstance(LegacySmellDetector.scan_code("x = " + "+".join(["1"] * 20000), "x.py"), list)


class TestCodeModernizer(unittest.TestCase):

    LEGACY = (
        '"""Doc mentioning except: and urllib2."""\n'
        "import os\nimport urllib2\n\n"
        "def fetch(url):\n    try:\n        return urllib2.urlopen(url).read()\n"
        "    except urllib2.HTTPError as e:\n        return None\n"
        "    except:\n        msg = \"except: stays\"\n        return msg\n\n"
        "def run(cmd):\n    return os.popen(cmd).read()\n"
    )

    def test_modernize_python_is_behavior_preserving(self):
        out, notes = CodeModernizer.modernize_python(self.LEGACY)
        self.assertIn("import urllib.request", out)
        self.assertIn("urllib.request.urlopen(url)", out)
        self.assertIn("except urllib.error.HTTPError as e:", out)
        self.assertNotIn("urllib2", out.split('"""')[2])  # code, not the docstring
        self.assertIn("subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True).stdout", out)
        self.assertIn("import subprocess", out)
        self.assertIn("except Exception:", out)
        self.assertIn('msg = "except: stays"', out)
        self.assertTrue(out.startswith('"""Doc'), "docstring must stay first")
        self.assertNotIn("from __future__", out)
        compile(out, "out.py", "exec")
        self.assertEqual({s for s, _ in smell_ids(out)} & {"PY-SMELL-002", "PY-SMELL-003", "PY-SMELL-004"}, set())

    def test_aliased_urllib2_is_left_for_manual_review(self):
        code = "import urllib2 as u\nu.urlopen(x)\n"
        out, notes = CodeModernizer.modernize_python(code)
        self.assertEqual(out, code)
        self.assertTrue(any(n.startswith("Manual review") for n in notes))

    def test_python2_print_statements_converted(self):
        out, _ = CodeModernizer.modernize_python('print "a", 1  # c\nprint\nprint >>f, "x"\n')
        self.assertEqual(out.split("\n")[:3], ['print("a", 1)  # c', "print()", 'print >>f, "x"'])

    def test_valid_python3_print_expression_untouched(self):
        code = "print -1\n"  # legal (if odd) Python 3; rewriting would change meaning
        self.assertEqual(CodeModernizer.modernize_python(code)[0], code)

    def test_modernize_typescript_uses_let(self):
        out, _ = CodeModernizer.modernize_typescript("var x = 1;\nx = 2;\n")
        self.assertEqual(out, "let x = 1;\nx = 2;\n")


# ---------------------------------------------------------------------------
# Test runner sandbox
# ---------------------------------------------------------------------------

class TestRunLocalTests(unittest.TestCase):

    def test_allowed_tool(self):
        self.assertIn("Python 3", run_local_tests("python --version"))

    def test_blocked_tool(self):
        self.assertIn("is not an allowed test command", run_local_tests("powershell Get-Process"))

    def test_command_injection_prevention(self):
        self.assertIn("dangerous shell chaining character", run_local_tests("python --version; echo hacked"))

    def test_interpreter_one_liners_blocked(self):
        for cmd in ('python -c "import os"', "node -e 1", "npm run build", "python evil.py", "go run main.go"):
            self.assertIn("is not an allowed test command", run_local_tests(cmd), cmd)

    def test_test_runner_shapes_allowed(self):
        for args in (["python", "-m", "pytest", "-q"], ["python", "-m", "unittest", "discover"], ["npm", "test"],
                     ["npm", "run", "test:unit"], ["cargo", "test"], ["go", "test", "./..."], ["node", "--test"]):
            self.assertIsNone(server._validate_test_command(args), args)


# ---------------------------------------------------------------------------
# Git tools (throwaway repository)
# ---------------------------------------------------------------------------

class TestGitTools(WorkspaceTestCase):

    def test_create_git_migration_pr_creates_branch_and_draft(self):
        self.write("a.py", "x = 1\n")
        res = create_git_migration_pr("test-modernization-branch", "test: commit", "Title", "Body")
        self.assertIn("Created and checked out new branch", res)
        self.assertEqual(self.repo.active_branch.name, "test-modernization-branch")
        self.assertIn("a.py", self.committed_files())
        self.assertTrue(os.path.exists(os.path.join(self.root, ".novus_pr_test-modernization-branch.md")))

    def test_invalid_branch_names_rejected(self):
        for name in ("../evil", "bad..name", "has space"):
            self.assertIn("not a valid git branch name", create_git_migration_pr(name, "m", "t", "d"), name)

    def test_files_argument_commits_only_those_files(self):
        self.write("target.py", "x = 1\n")
        self.write("other.py", "y = 1\n")
        self.commit_all()
        self.write("target.py", "x = 2\n")
        self.write("other.py", "y = 2\n")
        self.write("unrelated_staged.py", "z = 1\n")
        self.repo.git.add("unrelated_staged.py")

        create_git_migration_pr("only-target", "refactor: target", "t", "d", files=["target.py"])
        self.assertEqual(self.committed_files(), {"target.py"})
        self.assertIn("other.py", self.repo.git.diff("--name-only"))

    def test_default_staging_excludes_tool_artifacts(self):
        self.write("a.py", "x = 1\n")
        self.write("a.py.bak", "x = 0\n")
        self.write(".novus_index/index.json", "{}")
        self.write(".novus_pr_old.md", "draft")
        create_git_migration_pr("artifacts", "m", "t", "d")
        self.assertEqual(self.committed_files(), {"a.py"})

    def test_finalize_git_migration_pr_quotes_command(self):
        self.write("a.py", "x = 1\n")
        res = finalize_git_migration_pr("feature/final", "main", 'Fix "quoted" title', "Summary.")
        self.assertIn("NovusPipeline Git PR Finalization", res)
        self.assertIn("gh pr create --title 'Fix \"quoted\" title'", res)
        self.assertTrue(os.path.exists(os.path.join(self.root, ".novus_pr_feature_final.md")))


# ---------------------------------------------------------------------------
# Modernization tools & pipeline (throwaway repository)
# ---------------------------------------------------------------------------

LEGACY_MODULE = "import urllib2\n\ndef fetch(url):\n    try:\n        return urllib2.urlopen(url)\n    except:\n        return None\n"


class TestModernizationTools(WorkspaceTestCase):

    def test_analyze_legacy_codebase(self):
        self.write("sample_legacy.py", "import urllib2\nvar_x = 10\ntry:\n    pass\nexcept:\n    pass")
        report = analyze_legacy_codebase("sample_legacy.py")
        self.assertIn("Modernization Audit Report", report)
        self.assertIn("PY-SMELL-002", report)
        self.assertIn("Codebase Context", report)

    def test_analyze_runs_gnn_cross_check_even_without_rule_findings(self):
        self.write("clean.py", "def add(a: int, b: int) -> int:\n    return a + b\n")
        report = analyze_legacy_codebase("clean.py")
        self.assertIn("No rule-based legacy code smells", report)
        self.assertIn("Structural GNN Cross-Check", report)

    def test_analyze_legacy_codebase_survives_gnn_failure(self):
        self.write("sample_gnn_failure.py", "try:\n    pass\nexcept:\n    pass\n")
        with mock.patch.object(gnn_model, "predict_smells", side_effect=ValueError("too large")):
            report = analyze_legacy_codebase("sample_gnn_failure.py")
        self.assertIn("PY-SMELL-003", report)
        self.assertIn("Structural GNN Cross-Check\nUnavailable: ValueError", report)

    def test_apply_code_modernization(self):
        self.write("sample_mod.py", "import urllib2\ntry:\n    urllib2.urlopen(u)\nexcept:\n    pass")
        res = apply_code_modernization("sample_mod.py")
        self.assertIn("Successfully updated", res)
        self.assertIn("import urllib.request", self.read("sample_mod.py"))
        self.assertTrue(os.path.exists(os.path.join(self.root, "sample_mod.py.bak")))

    def test_apply_code_modernization_noop_writes_nothing(self):
        self.write("clean.py", "x: int = 1\n")
        res = apply_code_modernization("clean.py")
        self.assertIn("No applicable automated transformations", res)
        self.assertFalse(os.path.exists(os.path.join(self.root, "clean.py.bak")))

    def test_apply_code_modernization_warns_when_breaking_imported_api(self):
        self.write("lib.py", "def helper() -> int:\n    return 1\n")
        self.write("app.py", "from lib import helper\n")
        res = apply_code_modernization("lib.py", modernized_code="def renamed() -> int:\n    return 1\n")
        self.assertIn("WARNING", res)
        self.assertIn("`helper`", res)

    def test_generate_llm_modernization_proposal_includes_context(self):
        self.write("lib.py", LEGACY_MODULE)
        self.write("app.py", "from lib import fetch\n")
        res = generate_llm_modernization_proposal("lib.py", rag_query="httpx urllib2")
        self.assertIn("LLM Modernization Proposal", res)
        self.assertIn("Public API used elsewhere", res)
        self.assertIn("fetch", res)

    def test_format_modernization_report(self):
        self.write("lib.py", LEGACY_MODULE)
        res = format_modernization_report("lib.py", branch_name="feature/report", test_command="python --version")
        self.assertIn("Successfully generated and saved Modernization Report artifact", res)
        self.assertTrue(os.path.exists(os.path.join(self.root, "reports", "modernization_report_feature_report.md")))


class TestAutonomousPipeline(WorkspaceTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.write("calc.py", LEGACY_MODULE)
        self.write("other_legacy.py", "def g():\n    try:\n        pass\n    except:\n        pass\n")
        self.commit_all()

    def test_success_commits_only_the_file_and_uses_related_tests(self):
        self.write("test_calc.py", "import unittest\nimport calc\n\nclass T(unittest.TestCase):\n"
                                   "    def test_import(self):\n        self.assertTrue(hasattr(calc, 'fetch'))\n")
        self.commit_all("add tests")
        res = run_autonomous_modernization_pipeline("calc.py", branch_name="test-auto-pipeline-branch")
        self.assertIn("Completed Successfully", res, res)
        self.assertIn("test_calc", res)  # related test selected from the import graph
        self.assertEqual(self.committed_files(), {"calc.py"})
        self.assertIn("other_legacy.py", res)  # same smell elsewhere -> follow-up candidate
        self.assertFalse(os.path.exists(os.path.join(self.root, "calc.py.bak")))

    def test_failure_detected_by_exit_code_not_output_text(self):
        # Output contains "OK" but the run fails: the old substring check would have committed.
        self.write("test_calc.py", "import unittest\n\nclass T(unittest.TestCase):\n"
                                   "    def test_fail(self):\n        self.fail('OK')\n")
        self.commit_all("add failing test")
        head = self.repo.head.commit.hexsha
        res = run_autonomous_modernization_pipeline("calc.py", test_command="python -m unittest test_calc")
        self.assertIn("failed (exit code 1)", res)
        self.assertEqual(self.read("calc.py"), LEGACY_MODULE)
        self.assertEqual(self.repo.head.commit.hexsha, head)
        self.assertEqual(self.repo.active_branch.name, "main")

    def test_outside_workspace_rejected(self):
        self.assertIn("outside authorized", run_autonomous_modernization_pipeline("../../etc/passwd"))


# ---------------------------------------------------------------------------
# Codebase context index (throwaway workspace)
# ---------------------------------------------------------------------------

class TestCodebaseContext(WorkspaceTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.write("pkg/__init__.py", "")
        self.write("pkg/core.py", "def compute(x: int) -> int:\n    return x * 2\n\nCONSTANT = 3\n")
        self.write("pkg/service.py", "from .core import compute\nfrom . import core\n\n"
                                     "def run(v: int) -> int:\n    return compute(v) + core.CONSTANT\n")
        self.write("app.py", "import pkg.service as svc\n\ndef main() -> int:\n    return svc.run(1)\n")
        self.write("tests/test_core.py", "from pkg.core import compute\n")
        self.write("broken_py2.py", 'print "legacy"\n')
        self.write(".venv/lib/ignored.py", "import os\n")
        self.index = codebase_context.CodebaseIndex(self.root)
        self.index.refresh()

    def test_skip_dirs_and_parse_errors(self):
        self.assertNotIn(".venv/lib/ignored.py", self.index.files)
        rec = self.index.files["broken_py2.py"]
        self.assertIsNotNone(rec["parse_error"])
        self.assertEqual([f["smell_id"] for f in rec["rule_findings"]], ["PY-SMELL-001"])

    def test_import_graph_resolves_relative_and_aliased_imports(self):
        self.assertEqual(self.index.dependents("pkg/core.py"),
                         {"pkg/service.py": ["CONSTANT", "compute"], "tests/test_core.py": ["compute"]})
        self.assertEqual(self.index.dependents("pkg/service.py"), {"app.py": ["run"]})
        self.assertEqual(self.index.api_surface("pkg/core.py"), ["CONSTANT", "compute"])
        self.assertEqual(self.index.related_tests("pkg/core.py"), ["tests/test_core.py"])

    def test_incremental_refresh_and_persistence(self):
        self.assertEqual(self.index.refresh()["reanalyzed"], 0)
        self.write("app.py", "def main() -> int:\n    return 0\n")
        os.remove(os.path.join(self.root, "broken_py2.py"))
        stats = self.index.refresh()
        self.assertEqual((stats["reanalyzed"], stats["removed"]), (1, 1))
        self.assertEqual(self.index.dependents("pkg/service.py"), {})
        reloaded = codebase_context.CodebaseIndex(self.root)
        self.assertEqual(set(reloaded.files), set(self.index.files))

    @unittest.skipUnless(gnn_ready(), "GNN checkpoint not trained in this environment")
    def test_similarity_finds_structural_duplicate(self):
        dup = ("def load(path: str) -> list:\n    out = []\n    for line in open(path):\n"
               "        if line.strip():\n            out.append(line.split(','))\n    return out\n")
        self.write("a.py", dup)
        self.write("b.py", dup.replace("load", "read_rows").replace("out", "rows"))
        self.write("c.py", "class K:\n    def m(self) -> None:\n        raise ValueError('x')\n")
        self.index.refresh()
        top = self.index.similar("a.py", "load", n=1)[0]
        self.assertEqual((top["path"], top["qualname"]), ("b.py", "read_rows"))
        self.assertGreater(top["similarity"], 0.9)

    @unittest.skipUnless(gnn_ready(), "GNN checkpoint not trained in this environment")
    def test_function_level_localization(self):
        self.write("mixed.py", "def ok(a: int) -> int:\n    return a\n\n"
                               "def bad(a: int) -> int:\n    try:\n        return a\n    except:\n        return 0\n")
        self.index.refresh()
        hot = {f["qualname"]: f for f in self.index.file_context("mixed.py")["functions_with_smells"]}
        self.assertIn("bad", hot)
        self.assertNotIn("ok", hot)
        self.assertIn("PY-SMELL-003", hot["bad"]["rules"])


class TestContextTools(WorkspaceTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.write("lib.py", LEGACY_MODULE)
        self.write("app.py", "from lib import fetch\n\ndef main() -> None:\n    fetch('u')\n")

    def test_index_codebase(self):
        self.assertIn("Python files**: 2", index_codebase(force_rebuild=True))

    def test_get_codebase_overview(self):
        res = get_codebase_overview()
        self.assertIn("Modernization Hotspots", res)
        self.assertIn("`lib.py`", res)
        self.assertIn("imported by 1 module(s)", res)

    def test_get_file_context(self):
        res = get_file_context("lib.py")
        self.assertIn("Imported by 1 module(s)", res)
        self.assertIn("`app.py` uses `fetch`", res)
        self.assertIn("outside authorized", get_file_context("../outside.py"))

    @unittest.skipUnless(gnn_ready(), "GNN checkpoint not trained in this environment")
    def test_find_similar_code(self):
        self.assertIn("structurally similar", find_similar_code("lib.py", "fetch"))
        self.assertIn("no function `nope`", find_similar_code("lib.py", "nope"))


# ---------------------------------------------------------------------------
# LLM, reporting, GNN
# ---------------------------------------------------------------------------

class TestLlmAndReporting(unittest.TestCase):

    def test_get_local_llm_status(self):
        res = get_local_llm_status()
        self.assertIn("Integrated Local LLM Status", res)
        self.assertIn("unsloth_Qwen3.5-2B_1785882774", res)

    def test_prompt_includes_codebase_context(self):
        prompt = local_llm.generate_modernization_prompt("x = 1", "rules", codebase_context="Imported by app.py")
        self.assertIn("Imported by app.py", prompt)

    def test_modernization_reporter(self):
        report = ModernizationReporter.generate_report("README.md", "Audit", "Details", "Exit Code: 0", "test-report-branch")
        self.assertIn("NovusPipeline Codebase Modernization Report", report)
        self.assertIn("test-report-branch", report)


class TestGnn(unittest.TestCase):

    def test_build_graph_from_code_structure(self):
        graph = build_graph_from_code("import urllib2\ntry:\n    pass\nexcept:\n    pass\n")
        for key in ("node_type_ids", "node_field_ids", "node_token_ids", "node_features"):
            self.assertEqual(len(graph[key]), graph["num_nodes"], key)
        self.assertTrue(all(0 <= t < len(NODE_TYPES) for t in graph["node_type_ids"]))

    def test_build_graph_from_code_invalid_syntax_raises(self):
        with self.assertRaises(SyntaxError):
            build_graph_from_code("def broken(:\n")

    def test_build_graph_from_code_handles_long_expression_chain(self):
        self.assertGreater(build_graph_from_code("x = " + " + ".join(["1"] * 2000) + "\n")["num_nodes"], 2000)

    def test_identifier_features_distinguish_smell_apis(self):
        smelly = build_graph_from_code("import pickle\ndef f(b):\n    return pickle.loads(b)\n")
        clean = build_graph_from_code("import json\ndef f(b):\n    return json.loads(b)\n")
        self.assertEqual(smelly["node_type_ids"], clean["node_type_ids"])
        self.assertNotEqual(smelly["node_token_ids"], clean["node_token_ids"])
        api_names = ["urllib2", "urllib", "urlopen", "pickle", "loads", "load", "dumps",
                     "os", "popen", "json", "httpx", "requests", "subprocess"]
        self.assertEqual(len({identifier_bucket(n) for n in api_names}), len(api_names))

    def test_get_gnn_model_status(self):
        res = get_gnn_model_status()
        self.assertIn("GNN Structural Smell Classifier Status", res)
        self.assertIn("Torch Available", res)

    def test_analyze_code_structure_gnn_rejects_non_python(self):
        self.assertIn("Python", analyze_code_structure_gnn("rules.md"))

    @unittest.skipUnless(gnn_ready(), "GNN checkpoint not trained in this environment")
    def test_gnn_separates_smell_from_structural_lookalike(self):
        thresholds = gnn_model.get_thresholds()
        p_smelly = gnn_model.predict_smells("import pickle\ndef load(b: bytes) -> object:\n    return pickle.loads(b)\n")
        p_clean = gnn_model.predict_smells("import json\ndef load(b: bytes) -> object:\n    return json.loads(b)\n")
        self.assertGreater(p_smelly["PY-SMELL-005"], thresholds["PY-SMELL-005"])
        self.assertLessEqual(p_clean["PY-SMELL-005"], thresholds["PY-SMELL-005"])

    @unittest.skipUnless(gnn_ready(), "GNN checkpoint not trained in this environment")
    def test_analyze_many_matches_single_predictions(self):
        codes = ["def f(x):\n    return x\n", "def broken(:\n", "import pickle\npickle.loads(b)\n"]
        batch = gnn_model.analyze_many(codes)
        self.assertIsNone(batch[1])
        for code, res in ((codes[0], batch[0]), (codes[2], batch[2])):
            single = gnn_model.predict_smells(code)
            for k in single:
                self.assertAlmostEqual(single[k], res["probs"][k], places=5)


if __name__ == "__main__":
    unittest.main()
