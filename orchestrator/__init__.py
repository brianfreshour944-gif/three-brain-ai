"""Orchestrator package for three-brain AI system."""

from orchestrator.config import *
from orchestrator.main import ThreeBrainOrchestrator, OrchestrationResult, create_orchestrator
from orchestrator.context_manager import ContextManager, TaskContext
from orchestrator.memory import MemoryManager, ProjectMemory, get_memory_manager
from orchestrator.task_store import Task, TaskStore, AgentResult, get_task_store
from orchestrator.router import TaskClassifier, TaskComplexity, TaskClassification, create_classifier, classify_task
from orchestrator.safety import *
from orchestrator.approval_gate import ApprovalGate, ApprovalRequest, ApprovalStatus, create_approval_gate
from orchestrator.workflow import WorkflowEngine, WorkflowOrchestrator, WorkflowState, WorkflowStage, WorkflowStep, create_workflow_engine, create_workflow_orchestrator
from orchestrator.jetbrains import ImplementationPlan, ImplementationStep, PlanGenerator, create_implementation_plan, export_plan_for_jetbrains, export_plan_json

__all__ = [
    # Config
    "TASKS_DIR",
    "MEMORY_DIR",
    "LOGS_DIR",
    "MINISTRAL_BASE_URL",
    "DEEPSEEK_BASE_URL",
    "MINISTRAL_MODEL",
    "DEEPSEEK_MODEL",
    "STRATEGIST_MODEL",
    "BUILDER_TEMP",
    "ANALYST_TEMP",
    "STRATEGIST_MODEL",
    "BUILDER_TEMP",
    "ANALYST_TEMP",
    "STRATEGIST_TEMP",
    "REQUIRE_APPROVAL",
    "BLOCKED_PATHS",
    "PROTECTED_FILES",

    # Main
    "ThreeBrainOrchestrator",
    "OrchestrationResult",
    "create_orchestrator",

    # Context
    "ContextManager",
    "TaskContext",

    # Memory
    "MemoryManager",
    "ProjectMemory",
    "get_memory_manager",

    # Tasks
    "Task",
    "TaskStore",
    "AgentResult",
    "get_task_store",

    # Router
    "TaskClassifier",
    "TaskComplexity",
    "TaskClassification",
    "create_classifier",
    "classify_task",

    # Safety
    "check_syntax",
    "check_directory_syntax",
    "check_imports",
    "check_directory_imports",
    "DEFAULT_BLOCKED_IMPORTS",
    "check_secrets",
    "check_directory_secrets",
    "scan_secrets",
    "scan_file",
    "scan_directory",
    "check_scope",
    "check_directory_scope",
    "ScopeChecker",
    "DEFAULT_PROTECTED_PATHS",
    "DEFAULT_PROTECTED_PATTERNS",
    "run_safety_checks",
    "run_all_tests",
    "run_pytest",
    "TestRunner",
    "TestStatus",
    "TestResult",
    "TestSuiteResult",

    # Approval Gate
    "ApprovalGate",
    "ApprovalRequest",
    "ApprovalStatus",
    "create_approval_gate",

    # Workflow
    "WorkflowEngine",
    "WorkflowOrchestrator",
    "WorkflowState",
    "WorkflowStage",
    "WorkflowStep",
    "create_workflow_engine",
    "create_workflow_orchestrator",

    # JetBrains
    "ImplementationPlan",
    "ImplementationStep",
    "PlanGenerator",
    "create_implementation_plan",
    "export_plan_for_jetbrains",
    "export_plan_json",
]