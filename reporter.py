"""
reporter.py — NovusPipeline Phase 4: Git PR & Modernization Reporting Module

Formats enterprise-grade modernization audit reports, diff summaries,
and GitHub/GitLab pull request markdown artifacts.
"""

import os
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
        model_used: str = "unsloth_Qwen3.5-2B_1785882774"
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
- **Commit Status**: Staged & Committed to `{branch_name}`
- **PR Title**: `refactor({file_name}): automated enterprise modernization via NovusPipeline`
- **PR Draft File**: `.novus_pr_{branch_name}.md`

---
*Report generated automatically by NovusPipeline FastMCP Modernization Server.*
"""
        return report

    @classmethod
    def save_report_artifact(cls, workspace_root: str, branch_name: str, report_content: str) -> str:
        """Saves report content to workspace reports directory."""
        reports_dir = os.path.join(workspace_root, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_file = os.path.join(reports_dir, f"modernization_report_{branch_name}.md")

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        return report_file
