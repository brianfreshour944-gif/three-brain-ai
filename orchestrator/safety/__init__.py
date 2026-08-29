"""Safety package for three-brain AI system."""

from orchestrator.safety.syntax_check import check_syntax, check_directory_syntax
from orchestrator.safety.import_check import check_imports, check_directory_imports, DEFAULT_BLOCKED_IMPORTS
from orchestrator.safety.secret_check import check_secrets, check_directory_secrets, scan_secrets, scan_file, scan_directory
from orchestrator.safety.scope_check import check_scope, check_directory_scope, ScopeChecker, DEFAULT_PROTECTED_PATHS, DEFAULT_PROTECTED_PATTERNS
from orchestrator.safety.test_runner import run_safety_checks, run_all_tests, run_pytest, TestRunner, TestStatus, TestResult, TestSuiteResult

__all__ = [
    # Syntax
    "check_syntax",
    "check_directory_syntax",
    
    # Imports
    "check_imports",
    "check_directory_imports",
    "DEFAULT_BLOCKED_IMPORTS",
    
    # Secrets
    "check_secrets",
    "check_directory_secrets",
    "scan_secrets",
    "scan_file",
    "scan_directory",
    
    # Scope
    "check_scope",
    "check_directory_scope",
    "ScopeChecker",
    "DEFAULT_PROTECTED_PATHS",
    "DEFAULT_PROTECTED_PATTERNS",
    
    # Tests
    "run_safety_checks",
    "run_all_tests",
    "run_pytest",
    "TestRunner",
    "TestStatus",
    "TestResult",
    "TestSuiteResult",
]