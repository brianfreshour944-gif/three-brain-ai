"""Task storage and management."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.config import TASKS_DIR


@dataclass
class AgentResult:
    """Result from a single agent."""
    agent_name: str
    content: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Task:
    """A task in the system."""
    id: str
    description: str
    status: str = "pending"  # pending, in_progress, awaiting_approval, approved, rejected, completed, failed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    project_root: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Round 1 results (parallel independent analysis)
    round1_builder: Optional[str] = None
    round1_analyst: Optional[str] = None
    round1_strategist: Optional[str] = None
    
    # Round 2 results (sequential refinement)
    round2_builder: Optional[str] = None
    round2_analyst: Optional[str] = None
    round2_strategist: Optional[str] = None
    
    # Legacy single-pass results (for backward compatibility)
    builder_result: Optional[AgentResult] = None
    analyst_result: Optional[AgentResult] = None
    strategist_result: Optional[AgentResult] = None
    red_team_result: Optional[AgentResult] = None
    
    final_plan: Optional[str] = None
    approval_status: Optional[str] = None  # approved, rejected, revisions_requested
    approval_notes: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        data = asdict(self)
        # Convert AgentResult objects to dicts
        for key in ["builder_result", "analyst_result", "strategist_result", "red_team_result"]:
            if data[key] is not None:
                data[key] = asdict(data[key])
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create Task from dict."""
        # Convert dicts back to AgentResult
        for key in ["builder_result", "analyst_result", "strategist_result", "red_team_result"]:
            if data.get(key) is not None:
                data[key] = AgentResult(**data[key])
        return cls(**data)


class TaskStore:
    """Manages task persistence."""

    def __init__(self, tasks_dir: Optional[Path] = None):
        self.tasks_dir = tasks_dir or Path(TASKS_DIR)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: Dict[str, Task] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all tasks from disk."""
        for task_file in self.tasks_dir.glob("*.json"):
            try:
                with open(task_file, "r") as f:
                    data = json.load(f)
                task = Task.from_dict(data)
                self._tasks[task.id] = task
            except Exception:
                pass

    def _save(self, task: Task) -> None:
        """Save task to disk."""
        task.updated_at = datetime.now().isoformat()
        task_file = self.tasks_dir / f"{task.id}.json"
        with open(task_file, "w") as f:
            json.dump(task.to_dict(), f, indent=2)

    def create(self, description: str, project_root: str = "", context: Optional[Dict[str, Any]] = None) -> Task:
        """Create a new task."""
        task = Task(
            id=str(uuid.uuid4())[:8],
            description=description,
            project_root=project_root,
            context=context or {},
        )
        self._tasks[task.id] = task
        self._save(task)
        return task

    def get(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def update(self, task: Task) -> None:
        """Update task."""
        self._tasks[task.id] = task
        self._save(task)

    def delete(self, task_id: str) -> bool:
        """Delete task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            task_file = self.tasks_dir / f"{task_id}.json"
            if task_file.exists():
                task_file.unlink()
            return True
        return False

    def list_all(self, status: Optional[str] = None) -> List[Task]:
        """List all tasks, optionally filtered by status."""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def get_latest(self) -> Optional[Task]:
        """Get the most recent task."""
        tasks = self.list_all()
        return tasks[0] if tasks else None


def get_task_store() -> TaskStore:
    """Get singleton task store."""
    return TaskStore()