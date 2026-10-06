"""
reporter.py — NovusPipeline Phase 4: Git PR & Modernization Reporting Module

Formats enterprise-grade modernization audit reports, diff summaries,
and GitHub/GitLab pull request markdown artifacts.
"""

import os
import re
import time
from typing import Dict, Any, List


class ModernizationReporter:
    """Generates structured Markdown reports for modernized codebases."""

    @classmethod
    def generate_report(
        cls,
        file_path: str,
        audit_summary: str,
        modernization_details: str,
        test_output: str,
        branch_name: str,
        model_used: str = "unsloth_Qwen3.5-2B_1785882774",
        pr_draft_file: str = "",
    ) -> str:
        """Constructs an enterprise-grade Markdown report artifact."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        file_name = os.path.basename(file_path)

        report = f"""# 🚀 NovusPipeline Codebase Modernization Report

**Target File**: `{file_path}`  
**Git Branch**: `{branch_name}`  
**Generated At**: `{timestamp}`  
**LLM Engine**: `{model_used}`  
**RAG Store**: `ChromaDB persistent vector store (.chroma_db)`  

---

## Executive Summary
This report documents the automated modernization lifecycle executed by NovusPipeline for `{file_name}`.
The process performed static code smell detection, retrieved matching compliance handbooks from the local RAG vector store, generated parity-preserving code modernizations, verified changes in a sandboxed test environment, and prepared draft PR metadata.

---

## 1. Modernization Audit & Smell Detection
{audit_summary}

---

## 2. Code Modernization & Transformation Summary
{modernization_details}

---

## 3. Sandboxed Verification Test Output
```console
{test_output.strip()}
```

---

## 4. Git Pull Request Metadata
- **Branch**: `{branch_name}`
- **Commit Status**: see `git log {branch_name}` (this report does not commit)
- **PR Title**: `refactor({file_name}): automated enterprise modernization via NovusPipeline`
- **PR Draft File**: `{pr_draft_file or f".novus_pr_{branch_name}.md"}`

---
*Report generated automatically by NovusPipeline FastMCP Modernization Server.*
"""
        return report

    @classmethod
    def report_path(cls, workspace_root: str, branch_name: str) -> str:
        # "feature/x" must not turn into a nested (or escaping) path.
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", branch_name)
        return os.path.join(workspace_root, "reports", f"modernization_report_{safe}.md")

    @classmethod
    def save_report_artifact(cls, workspace_root: str, branch_name: str, report_content: str) -> str:
        """Saves report content to workspace reports directory."""
        report_file = cls.report_path(workspace_root, branch_name)
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        return report_file
