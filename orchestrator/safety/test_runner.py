"""Test runner for safety system."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: TestStatus
    message: str = ""
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResult:
    """Result of a test suite."""
    name: str
    results: List[TestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)
    
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)
    
    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
    
    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)
    
    @property
    def total(self) -> int:
        return len(self.results)
    
    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total * 100


class TestRunner:
    """Runs safety tests and collects results."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.suites: List[TestSuiteResult] = []
    
    def run_suite(self, name: str, tests: List[callable]) -> TestSuiteResult:
        """Run a suite of tests."""
        import time
        suite = TestSuiteResult(name=name)
        suite.start_time = time.time()
        
        for test_fn in tests:
            test_name = test_fn.__name__
            start = time.time()
            try:
                result = test_fn()
                duration = (time.time() - start) * 1000
                
                if isinstance(result, tuple):
                    status, message = result
                elif isinstance(result, bool):
                    status = TestStatus.PASSED if result else TestStatus.FAILED
                    message = ""
                else:
                    status = TestStatus.PASSED
                    message = str(result)
                
                suite.results.append(TestResult(
                    name=test_name,
                    status=status,
                    message=message,
                    duration_ms=duration,
                ))
            except Exception as e:
                duration = (time.time() - start) * 1000
                suite.results.append(TestResult(
                    name=test_name,
                    status=TestStatus.ERROR,
                    message=str(e),
                    duration_ms=duration,
                ))
        
        suite.end_time = time.time()
        self.suites.append(suite)
        return suite
    
    async def run_suite_async(self, name: str, tests: List[callable]) -> TestSuiteResult:
        """Run a suite of tests asynchronously (CPU-bound tests in thread pool)."""
        import time
        suite = TestSuiteResult(name=name)
        suite.start_time = time.time()
        
        loop = asyncio.get_event_loop()
        
        for test_fn in tests:
            test_name = test_fn.__name__
            start = time.time()
            try:
                # Run CPU-bound test in thread pool
                result = await loop.run_in_executor(None, test_fn)
                duration = (time.time() - start) * 1000
                
                if isinstance(result, tuple):
                    status, message = result
                elif isinstance(result, bool):
                    status = TestStatus.PASSED if result else TestStatus.FAILED
                    message = ""
                else:
                    status = TestStatus.PASSED
                    message = str(result)
                
                suite.results.append(TestResult(
                    name=test_name,
                    status=status,
                    message=message,
                    duration_ms=duration,
                ))
            except Exception as e:
                duration = (time.time() - start) * 1000
                suite.results.append(TestResult(
                    name=test_name,
                    status=TestStatus.ERROR,
                    message=str(e),
                    duration_ms=duration,
                ))
        
        suite.end_time = time.time()
        self.suites.append(suite)
        return suite
    
    def run_all(self) -> Dict[str, Any]:
        """Run all registered test suites."""
        return {
            "suites": [
                {
                    "name": s.name,
                    "passed": s.passed,
                    "failed": s.failed,
                    "skipped": s.skipped,
                    "errors": s.errors,
                    "total": s.total,
                    "success_rate": s.success_rate,
                    "duration_ms": s.duration_ms,
                    "tests": [
                        {
                            "name": r.name,
                            "status": r.status.value,
                            "message": r.message,
                            "duration_ms": r.duration_ms,
                        }
                        for r in s.results
                    ]
                }
                for s in self.suites
            ],
            "summary": self._summary(),
        }
    
    async def run_all_async(self) -> Dict[str, Any]:
        """Run all registered test suites asynchronously."""
        return {
            "suites": [
                {
                    "name": s.name,
                    "passed": s.passed,
                    "failed": s.failed,
                    "skipped": s.skipped,
                    "errors": s.errors,
                    "total": s.total,
                    "success_rate": s.success_rate,
                    "duration_ms": s.duration_ms,
                    "tests": [
                        {
                            "name": r.name,
                            "status": r.status.value,
                            "message": r.message,
                            "duration_ms": r.duration_ms,
                        }
                        for r in s.results
                    ]
                }
                for s in self.suites
            ],
            "summary": self._summary(),
        }
    
    def _summary(self) -> Dict[str, Any]:
        total_passed = sum(s.passed for s in self.suites)
        total_failed = sum(s.failed for s in self.suites)
        total_skipped = sum(s.skipped for s in self.suites)
        total_errors = sum(s.errors for s in self.suites)
        total_tests = sum(s.total for s in self.suites)
        
        return {
            "total": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "errors": total_errors,
            "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
        }


def run_pytest(project_root: Path, args: Optional[List[str]] = None) -> tuple[bool, str]:
    """Run pytest on the project."""
    if args is None:
        args = ["-v", "--tb=short"]
    
    try:
        cmd = [sys.executable, "-m", "pytest"] + args
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Test timeout (300s)"
    except Exception as e:
        return False, str(e)


async def run_pytest_async(project_root: Path, args: Optional[List[str]] = None) -> tuple[bool, str]:
    """Run pytest on the project asynchronously."""
    if args is None:
        args = ["-v", "--tb=short"]
    
    loop = asyncio.get_event_loop()
    try:
        cmd = [sys.executable, "-m", "pytest"] + args
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            return process.returncode == 0, (stdout + stderr).decode()
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return False, "Test timeout (300s)"
    except Exception as e:
        return False, str(e)


def run_safety_checks(project_root: Path) -> tuple[bool, Dict[str, Any]]:
    """Run all safety checks on the project (synchronous)."""
    from orchestrator.safety import (
        check_directory_syntax,
        check_directory_imports,
        check_directory_secrets,
        check_directory_scope,
    )
    
    results = {}
    all_passed = True
    
    # Syntax check
    valid, errors = check_directory_syntax(project_root)
    results["syntax"] = {"passed": valid, "errors": errors}
    if not valid:
        all_passed = False
    
    # Import check
    valid, errors = check_directory_imports(project_root)
    results["imports"] = {"passed": valid, "errors": errors}
    if not valid:
        all_passed = False
    
    # Secret check
    valid, errors = check_directory_secrets(project_root)
    results["secrets"] = {"passed": valid, "errors": errors}
    if not valid:
        all_passed = False
    
    # Scope check
    valid, errors = check_directory_scope(project_root)
    results["scope"] = {"passed": valid, "errors": errors}
    if not valid:
        all_passed = False
    
    return all_passed, results


async def run_safety_checks_async(project_root: Path) -> tuple[bool, Dict[str, Any]]:
    """Run all safety checks on the project asynchronously (in thread pool)."""
    from orchestrator.safety import (
        check_directory_syntax,
        check_directory_imports,
        check_directory_secrets,
        check_directory_scope,
    )
    
    loop = asyncio.get_event_loop()
    
    # Run all checks concurrently in thread pool
    syntax_task = loop.run_in_executor(None, check_directory_syntax, project_root)
    imports_task = loop.run_in_executor(None, check_directory_imports, project_root)
    secrets_task = loop.run_in_executor(None, check_directory_secrets, project_root)
    scope_task = loop.run_in_executor(None, check_directory_scope, project_root)
    
    results = {}
    all_passed = True
    
    # Wait for all checks
    syntax_valid, syntax_errors = await syntax_task
    results["syntax"] = {"passed": syntax_valid, "errors": syntax_errors}
    if not syntax_valid:
        all_passed = False
    
    imports_valid, imports_errors = await imports_task
    results["imports"] = {"passed": imports_valid, "errors": imports_errors}
    if not imports_valid:
        all_passed = False
    
    secrets_valid, secrets_errors = await secrets_task
    results["secrets"] = {"passed": secrets_valid, "errors": secrets_errors}
    if not secrets_valid:
        all_passed = False
    
    scope_valid, scope_errors = await scope_task
    results["scope"] = {"passed": scope_valid, "errors": scope_errors}
    if not scope_valid:
        all_passed = False
    
    return all_passed, results


def run_all_tests(project_root: Path) -> Dict[str, Any]:
    """Run all tests including pytest and safety checks."""
    runner = TestRunner(project_root)
    
    # Safety checks
    safety_passed, safety_results = run_safety_checks(project_root)
    
    # Pytest
    pytest_passed, pytest_output = run_pytest(project_root)
    
    return {
        "safety": safety_results,
        "pytest": {
            "passed": pytest_passed,
            "output": pytest_output,
        },
        "overall_passed": safety_passed and pytest_passed,
    }


async def run_all_tests_async(project_root: Path) -> Dict[str, Any]:
    """Run all tests including pytest and safety checks asynchronously."""
    runner = TestRunner(project_root)
    
    # Safety checks (async)
    safety_passed, safety_results = await run_safety_checks_async(project_root)
    
    # Pytest (async)
    pytest_passed, pytest_output = await run_pytest_async(project_root)
    
    return {
        "safety": safety_results,
        "pytest": {
            "passed": pytest_passed,
            "output": pytest_output,
        },
        "overall_passed": safety_passed and pytest_passed,
    }