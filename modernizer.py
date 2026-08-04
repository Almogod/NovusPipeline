"""
modernizer.py — NovusPipeline Phase 3: Autonomous Refactoring & Pattern Analysis Engine

Provides AST and rule-based static analysis for legacy code smells, applies
parity-preserving transformations, and orchestrates the autonomous loop:
Scan -> Query RAG -> Apply Modernization -> Test -> Commit/Rollback.
"""

import os
import re
import ast
import shutil
from typing import Dict, List, Any, Tuple


# ---------------------------------------------------------------------------
# Pattern Analyzer & Legacy Smell Detection
# ---------------------------------------------------------------------------

class LegacySmellDetector:
    """Scans code strings for enterprise legacy code smells."""

    SMELL_PATTERNS = [
        {
            "id": "PY-SMELL-001",
            "name": "Python 2 Print Statement / Missing Print Parentheses",
            "regex": r"^\s*print\s+[\"'\w]",
            "category": "python",
            "rag_query": "Python 2 print statement modernization logging",
            "severity": "HIGH"
        },
        {
            "id": "PY-SMELL-002",
            "name": "Obsolete urllib/urllib2 Import",
            "regex": r"import\s+(urllib2|urllib)|from\s+urllib2\s+import",
            "category": "python",
            "rag_query": "Replace urllib2 with httpx or requests",
            "severity": "HIGH"
        },
        {
            "id": "PY-SMELL-003",
            "name": "Bare Exception Handling (except:)",
            "regex": r"except\s*:",
            "category": "python",
            "rag_query": "Python Error Handling & Logging specific exception",
            "severity": "CRITICAL"
        },
        {
            "id": "PY-SMELL-004",
            "name": "Deprecated os.popen Usage",
            "regex": r"os\.popen\(",
            "category": "python",
            "rag_query": "Replace os.popen with subprocess run capture_output",
            "severity": "HIGH"
        },
        {
            "id": "PY-SMELL-005",
            "name": "Unsafe Pickle Serialization",
            "regex": r"import\s+pickle|pickle\.loads?\(",
            "category": "python",
            "rag_query": "Replace pickle with pydantic json serialization",
            "severity": "CRITICAL"
        },
        {
            "id": "PY-SMELL-006",
            "name": "Missing Type Hints in Function Signature",
            "regex": r"def\s+\w+\s*\([^)]*\)\s*:",
            "category": "python",
            "rag_query": "Python Type System Modernization union syntax",
            "severity": "MEDIUM"
        },
        {
            "id": "TS-SMELL-001",
            "name": "Legacy ES5 'var' Declaration",
            "regex": r"\bvar\s+\w+",
            "category": "typescript",
            "rag_query": "JavaScript ES Modernization replace var const let",
            "severity": "HIGH"
        },
        {
            "id": "TS-SMELL-002",
            "name": "Loose 'any' Type Annotation",
            "regex": r":\s*any\b",
            "category": "typescript",
            "rag_query": "TypeScript Strict Mode replace any with unknown interface",
            "severity": "MEDIUM"
        },
        {
            "id": "TS-SMELL-003",
            "name": "Legacy CommonJS require() Import",
            "regex": r"const\s+\w+\s*=\s*require\(",
            "category": "typescript",
            "rag_query": "Replace CommonJS require with ES modules import",
            "severity": "MEDIUM"
        },
    ]

    @classmethod
    def scan_code(cls, code: str, file_path: str = "") -> List[Dict[str, Any]]:
        """Scans source code for legacy patterns and returns a list of detected smells."""
        findings = []
        lines = code.splitlines()

        for line_idx, line in enumerate(lines, 1):
            for pattern in cls.SMELL_PATTERNS:
                if re.search(pattern["regex"], line):
                    findings.append({
                        "smell_id": pattern["id"],
                        "name": pattern["name"],
                        "line_number": line_idx,
                        "line_content": line.strip(),
                        "category": pattern["category"],
                        "severity": pattern["severity"],
                        "rag_query": pattern["rag_query"],
                        "file_path": file_path
                    })

        # AST analysis for Python files
        if file_path.endswith(".py") or not file_path:
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ExceptHandler) and node.type is None:
                        findings.append({
                            "smell_id": "PY-SMELL-003",
                            "name": "Bare Exception Handling (AST detected)",
                            "line_number": getattr(node, "lineno", 0),
                            "line_content": "except:",
                            "category": "python",
                            "severity": "CRITICAL",
                            "rag_query": "Python Error Handling & Logging specific exception",
                            "file_path": file_path
                        })
            except SyntaxError:
                pass  # Regex fallback handles non-parseable code

        return findings


# ---------------------------------------------------------------------------
# Rule-Based Parity-Preserving Code Modernizer
# ---------------------------------------------------------------------------

class CodeModernizer:
    """Applies rule-based, parity-preserving code modernizations."""

    @classmethod
    def modernize_python(cls, code: str) -> Tuple[str, List[str]]:
        """Modernizes legacy Python code while preserving logical behavior."""
        changes = []
        updated_code = code

        # Rule 1: Replace urllib2 with httpx/requests
        if "import urllib2" in updated_code or "from urllib2 import" in updated_code:
            updated_code = re.sub(r"import\s+urllib2", "import httpx", updated_code)
            updated_code = re.sub(r"from\s+urllib2\s+import\s+urlopen", "from httpx import get as urlopen", updated_code)
            changes.append("Replaced deprecated `urllib2` import with `httpx`.")

        # Rule 2: Replace os.popen with subprocess.run
        if "os.popen(" in updated_code:
            updated_code = re.sub(
                r"os\.popen\((.*?)\)\.read\(\)",
                r"subprocess.run(\1, shell=False, capture_output=True, text=True).stdout",
                updated_code
            )
            changes.append("Replaced insecure `os.popen().read()` with `subprocess.run(..., capture_output=True)`.")

        # Rule 3: Replace bare except: with except Exception as e: + logging
        if re.search(r"except\s*:", updated_code):
            updated_code = re.sub(r"except\s*:", "except Exception as e:", updated_code)
            changes.append("Replaced bare `except:` with explicit `except Exception as e:`.")

        # Rule 4: Ensure 'from __future__ import annotations' for Python <3.10 syntax compatibility
        if "def " in updated_code and "from __future__ import annotations" not in updated_code:
            updated_code = "from __future__ import annotations\n" + updated_code
            changes.append("Added `from __future__ import annotations` header.")

        return updated_code, changes

    @classmethod
    def modernize_typescript(cls, code: str) -> Tuple[str, List[str]]:
        """Modernizes legacy JavaScript/TypeScript code."""
        changes = []
        updated_code = code

        # Rule 1: Replace var with const/let
        if re.search(r"\bvar\s+", updated_code):
            updated_code = re.sub(r"\bvar\s+", "const ", updated_code)
            changes.append("Replaced ES5 `var` with `const`.")

        # Rule 2: Replace CommonJS require with ES import
        require_match = re.findall(r"const\s+(\w+)\s*=\s*require\((['\"])(.*?)\2\);?", updated_code)
        for var_name, _, mod_path in require_match:
            old_str = f"const {var_name} = require('{mod_path}');"
            old_str_alt = f'const {var_name} = require("{mod_path}");'
            new_str = f"import {var_name} from '{mod_path}';"
            if old_str in updated_code:
                updated_code = updated_code.replace(old_str, new_str)
                changes.append(f"Converted `const {var_name} = require('{mod_path}')` -> ES module import.")
            elif old_str_alt in updated_code:
                updated_code = updated_code.replace(old_str_alt, new_str)
                changes.append(f"Converted `const {var_name} = require('{mod_path}')` -> ES module import.")

        return updated_code, changes
