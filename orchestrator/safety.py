"""Safety layer for code execution and changes."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from orchestrator.config import BLOCKED_PATHS, PROTECTED_FILES, REQUIRE_APPROVAL


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""
    passed: bool
    errors: List[str]
    warnings: List[str]
    blocked_files: List[str]


class SafetyChecker:
    """Performs safety checks before code execution."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.blocked_patterns = self._compile_patterns(BLOCKED_PATHS)
        self.protected_files = set(PROTECTED_FILES)

    def _compile_patterns(self, patterns: List[str]) -> List[str]:
        """Compile glob patterns for matching."""
        return patterns

    def _matches_pattern(self, file_path: Path, pattern: str) -> bool:
        """Check if file matches a glob pattern."""
        import fnmatch
        rel_path = file_path.relative_to(self.project_root) if file_path.is_absolute() else file_path
        return fnmatch.fnmatch(str(rel_path), pattern) or fnmatch.fnmatch(file_path.name, pattern)

    def check_file_access(self, file_path: Path) -> SafetyCheckResult:
        """Check if file access is allowed."""
        errors = []
        warnings = []
        blocked = []

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if self._matches_pattern(file_path, pattern):
                blocked.append(str(file_path))
                errors.append(f"Blocked path pattern: {pattern}")

        # Check protected files
        if file_path.name in self.protected_files:
            blocked.append(str(file_path))
            errors.append(f"Protected file: {file_path.name} requires explicit approval")

        # Check for sensitive content
        if file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                if self._contains_secrets(content):
                    warnings.append("File may contain secrets or credentials")
            except Exception:
                pass

        return SafetyCheckResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            blocked_files=blocked,
        )

    def _contains_secrets(self, content: str) -> bool:
        """Check for potential secrets in content."""
        secret_patterns = [
            "api_key", "apikey", "secret", "password", "token",
            "private_key", "access_key", "auth_token", "bearer",
        ]
        content_lower = content.lower()
        return any(pattern in content_lower for pattern in secret_patterns)

    def check_python_syntax(self, code: str) -> SafetyCheckResult:
        """Check Python syntax validity."""
        errors = []
        warnings = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e.msg} at line {e.lineno}")
        except Exception as e:
            errors.append(f"Parse error: {e}")

        return SafetyCheckResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            blocked_files=[],
        )

    def check_imports(self, code: str) -> SafetyCheckResult:
        """Check imports for dangerous modules."""
        errors = []
        warnings = []

        dangerous_modules = {
            "subprocess", "os.system", "eval", "exec", "compile",
            "importlib", "pkgutil", "runpy", "code", "codeop",
            "ctypes", "cffi", "ffi", "mmap", "resource",
        }

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in dangerous_modules:
                            warnings.append(f"Potentially dangerous import: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split(".")[0] in dangerous_modules:
                        warnings.append(f"Potentially dangerous import: {node.module}")
        except Exception:
            pass

        return SafetyCheckResult(
            passed=True,  # Warnings don't block
            errors=errors,
            warnings=warnings,
            blocked_files=[],
        )

    def run_all_checks(self, file_path: Path, code: Optional[str] = None) -> SafetyCheckResult:
        """Run all safety checks."""
        all_errors = []
        all_warnings = []
        all_blocked = []

        # File access check
        access_result = self.check_file_access(file_path)
        all_errors.extend(access_result.errors)
        all_warnings.extend(access_result.warnings)
        all_blocked.extend(access_result.blocked_files)

        if code:
            # Syntax check
            syntax_result = self.check_python_syntax(code)
            all_errors.extend(syntax_result.errors)
            all_warnings.extend(syntax_result.warnings)

            # Import check
            import_result = self.check_imports(code)
            all_warnings.extend(import_result.warnings)

        return SafetyCheckResult(
            passed=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            blocked_files=all_blocked,
        )


class ApprovalManager:
    """Manages human approval for code changes."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}

    def request_approval(
        self,
        task_id: str,
        action: str,
        files: List[Path],
        description: str,
    ) -> str:
        """Request approval for an action."""
        approval_id = f"approval_{task_id}_{len(self.pending_approvals)}"
        self.pending_approvals[approval_id] = {
            "task_id": task_id,
            "action": action,
            "files": [str(f) for f in files],
            "description": description,
            "status": "pending",
        }
        return approval_id

    def approve(self, approval_id: str, notes: str = "") -> bool:
        """Approve a pending request."""
        if approval_id in self.pending_approvals:
            self.pending_approvals[approval_id]["status"] = "approved"
            self.pending_approvals[approval_id]["notes"] = notes
            return True
        return False

    def reject(self, approval_id: str, notes: str = "") -> bool:
        """Reject a pending request."""
        if approval_id in self.pending_approvals:
            self.pending_approvals[approval_id]["status"] = "rejected"
            self.pending_approvals[approval_id]["notes"] = notes
            return True
        return False

    def get_pending(self) -> List[Dict[str, Any]]:
        """Get all pending approvals."""
        return [
            {"id": k, **v}
            for k, v in self.pending_approvals.items()
            if v["status"] == "pending"
        ]

    def is_approved(self, approval_id: str) -> bool:
        """Check if approval was granted."""
        return self.pending_approvals.get(approval_id, {}).get("status") == "approved"


def run_safety_check(file_path: str, code: Optional[str] = None, project_root: Optional[Path] = None) -> SafetyCheckResult:
    """Convenience function to run safety checks."""
    root = project_root or Path.cwd()
    checker = SafetyChecker(root)
    path = Path(file_path)
    return checker.run_all_checks(path, code)


if __name__ == "__main__":
    # CLI for standalone safety check
    import argparse

    parser = argparse.ArgumentParser(description="Run safety checks")
    parser.add_argument("file", help="File to check")
    parser.add_argument("--code", help="Code string to check (instead of file)")
    args = parser.parse_args()

    result = run_safety_check(args.file, args.code)

    print(json.dumps({
        "passed": result.passed,
        "errors": result.errors,
        "warnings": result.warnings,
        "blocked_files": result.blocked_files,
    }, indent=2))

    sys.exit(0 if result.passed else 1)