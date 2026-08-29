"""Orchestrator package for three-brain AI system."""

from orchestrator.config import *
from orchestrator.main import ThreeBrainOrchestrator, OrchestrationResult, create_orchestrator
from orchestrator.context_manager import ContextManager, TaskContext
from orchestrator.memory import MemoryManager, ProjectMemory, get_memory_manager
from orchestrator.task_store import Task, TaskStore, AgentResult, get_task_store
from orchestrator.router import TaskClassifier, TaskComplexity, TaskClassification, create_classifier, classify_task

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
]