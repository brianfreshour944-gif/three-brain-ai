"""Workflow engine for three-brain AI system."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable
from uuid import uuid4

from orchestrator.task_store import Task, get_task_store
from orchestrator.memory import get_memory_manager

logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Workflow stage enumeration."""
    INITIALIZED = "initialized"
    CLASSIFYING = "classifying"
    ROUND1_PARALLEL = "round1_parallel"
    ROUND2_REFINEMENT = "round2_refinement"
    RED_TEAM_REVIEW = "red_team_review"
    APPROVAL_PENDING = "approval_pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISIONS_REQUESTED = "revisions_requested"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowStep(Enum):
    """Individual workflow step."""
    CLASSIFY = "classify"
    ROUND1_BUILDER = "round1_builder"
    ROUND1_ANALYST = "round1_analyst"
    ROUND1_STRATEGIST = "round1_strategist"
    ROUND2_BUILDER = "round2_builder"
    ROUND2_ANALYST = "round2_analyst"
    ROUND2_STRATEGIST = "round2_strategist"
    RED_TEAM = "red_team"
    APPROVAL = "approval"
    EXECUTE = "execute"
    VERIFY = "verify"


@dataclass
class WorkflowState:
    """Current state of a workflow."""
    task_id: str
    stage: WorkflowStage = WorkflowStage.INITIALIZED
    current_step: Optional[WorkflowStep] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    step_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def update_stage(self, stage: WorkflowStage, step: Optional[WorkflowStep] = None):
        self.stage = stage
        if step:
            self.current_step = step
        self.updated_at = datetime.now().isoformat()
    
    def add_step_result(self, step: WorkflowStep, result: Any):
        self.step_results[step.value] = result
        self.updated_at = datetime.now().isoformat()
    
    def complete(self, success: bool = True, error: Optional[str] = None):
        self.completed_at = datetime.now().isoformat()
        if not success:
            self.stage = WorkflowStage.FAILED
            self.error = error
        else:
            self.stage = WorkflowStage.COMPLETED


class WorkflowEngine:
    """Manages workflow execution for tasks."""
    
    def __init__(
        self,
        project_root: Path,
        approval_gate: Optional[Any] = None,
        enable_red_team: bool = True,
    ):
        self.project_root = Path(project_root)
        self.approval_gate = approval_gate
        self.enable_red_team = enable_red_team
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.task_store = get_task_store()
        self.memory = get_memory_manager()
    
    def create_workflow(self, task_id: str) -> WorkflowState:
        """Create a new workflow state for a task."""
        workflow = WorkflowState(task_id=task_id)
        self.active_workflows[task_id] = workflow
        return workflow
    
    def get_workflow(self, task_id: str) -> Optional[WorkflowState]:
        return self.active_workflows.get(task_id)
    
    def remove_workflow(self, task_id: str):
        self.active_workflows.pop(task_id, None)
    
    async def execute_workflow(
        self,
        task_id: str,
        task_description: str,
        user_notes: str = "",
        relevant_files: Optional[List[Path]] = None,
        file_contents: Optional[Dict[str, str]] = None,
        auto_approve: bool = False,
        orchestrator: Any = None,
    ) -> Dict[str, Any]:
        """Execute the full workflow for a task."""
        workflow = self.create_workflow(task_id)
        
        try:
            # Stage 1: Classify
            workflow.update_stage(WorkflowStage.CLASSIFYING, WorkflowStep.CLASSIFY)
            
            # The orchestrator handles classification internally
            if orchestrator:
                result = await orchestrator.run_task(
                    task_description="",
                    user_notes="",
                    relevant_files=[],
                    file_contents={},
                    auto_approve=True,
                )
                
                # The orchestrator handles all stages internally
                # We just track the final result
                
            workflow.complete(success=True)
            return {"status": "completed", "task_id": task_id}
            
        except Exception as e:
            logger.error(f"Workflow failed for task {task_id}: {e}")
            workflow.complete(success=False, error=str(e))
            raise
    
    def get_workflow_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow status."""
        workflow = self.active_workflows.get(task_id)
        if not workflow:
            return None
        
        return {
            "task_id": workflow.task_id,
            "stage": workflow.stage.value,
            "current_step": workflow.current_step.value if workflow.current_step else None,
            "started_at": workflow.started_at,
            "updated_at": workflow.updated_at,
            "completed_at": workflow.completed_at,
            "error": workflow.error,
        }


class WorkflowOrchestrator:
    """High-level workflow orchestrator that coordinates all components."""
    
    def __init__(
        self,
        project_root: Path,
        approval_gate: Optional[Any] = None,
        enable_red_team: bool = True,
    ):
        self.project_root = Path(project_root)
        self.workflow_engine = WorkflowEngine(project_root, approval_gate, enable_red_team)
    
    async def run_task_workflow(
        self,
        task_description: str,
        user_notes: str = "",
        relevant_files: Optional[List[Path]] = None,
        file_contents: Optional[Dict[str, str]] = None,
        auto_approve: bool = False,
        orchestrator: Any = None,
    ) -> Dict[str, Any]:
        """Run complete task workflow."""
        task_id = str(uuid4())[:8]
        
        # Create task in store
        from orchestrator.task_store import get_task_store
        task_store = get_task_store()
        task = task_store.create(
            description=task_description,
            project_root=str(project_root) if project_root else str(Path.cwd()),
            context={"user_notes": user_notes},
        )
        
        # Execute workflow
        result = await self.workflow_engine.execute_workflow(
            task_id=task.id,
            task_description=task_description,
            user_notes=user_notes,
            relevant_files=relevant_files,
            file_contents=file_contents,
            auto_approve=auto_approve,
            orchestrator=orchestrator,
        )
        
        return result
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.workflow_engine.get_workflow_status(task_id)
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        return [
            self.workflow_engine.get_workflow_status(tid)
            for tid in self.workflow_engine.active_workflows.keys()
        ]


def create_workflow_engine(project_root: Path, **kwargs) -> WorkflowEngine:
    """Factory to create workflow engine."""
    return WorkflowEngine(project_root=Path(project_root), **kwargs)


def create_workflow_orchestrator(project_root: Path, **kwargs) -> WorkflowOrchestrator:
    """Factory to create workflow orchestrator."""
    return WorkflowOrchestrator(project_root=Path(project_root), **kwargs)