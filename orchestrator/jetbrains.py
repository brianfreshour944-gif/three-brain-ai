"""JetBrains integration for three-brain AI system."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.task_store import Task, get_task_store
from orchestrator.memory import get_memory_manager

logger = logging.getLogger(__name__)


@dataclass
class ImplementationStep:
    """A single implementation step in the plan."""
    order: int
    title: str
    description: str
    file_path: Optional[str] = None
    action: str = "create"  # create, modify, delete, test
    priority: int = 1  # 1=high, 2=medium, 3=low
    estimated_effort: str = "unknown"  # small, medium, large
    dependencies: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    test_command: Optional[str] = None


@dataclass
class ImplementationPlan:
    """Complete implementation plan for JetBrains AI/opencode."""
    task_id: str
    task_description: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    steps: List[ImplementationStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_markdown(self) -> str:
        """Export as markdown for JetBrains AI/opencode consumption."""
        lines = [
            f"# FINAL IMPLEMENTATION PLAN",
            f"",
            f"**Task ID:** {self.task_id}",
            f"**Task:** {self.task_description}",
            f"**Created:** {self.created_at}",
            f"**Updated:** {self.updated_at}",
            f"",
            f"---",
            f"",
        ]
        
        if self.metadata:
            lines.append("## Metadata")
            for k, v in self.metadata.items():
                lines.append(f"- **{k}:** {v}")
            lines.append("")
        
        lines.append("## Implementation Steps")
        lines.append("")
        
        for step in self.steps:
            lines.append(f"### Step {step.order}: {step.title}")
            lines.append(f"")
            lines.append(f"**Description:** {step.description}")
            if step.file_path:
                lines.append(f"**File:** `{step.file_path}`")
            lines.append(f"**Action:** {step.action}")
            lines.append(f"**Priority:** {step.priority} ({'High' if step.priority == 1 else 'Medium' if step.priority == 2 else 'Low'})")
            lines.append(f"**Effort:** {step.estimated_effort}")
            
            if step.dependencies:
                lines.append(f"**Dependencies:** {', '.join(step.dependencies)}")
            
            if step.acceptance_criteria:
                lines.append("**Acceptance Criteria:**")
                for ac in step.acceptance_criteria:
                    lines.append(f"- [ ] {ac}")
            
            if step.test_command:
                lines.append(f"**Test Command:**")
                lines.append(f"```bash")
                lines.append(f"{step.test_command}")
                lines.append(f"```")
            
            lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("## Usage with JetBrains AI / opencode")
        lines.append("")
        lines.append("1. Copy this plan into JetBrains AI Assistant or opencode")
        lines.append("2. Execute steps in order (respecting dependencies)")
        lines.append("3. Mark each step as complete when done")
        lines.append("4. Run test commands to verify")
        lines.append("")
        lines.append("```bash")
        lines.append("# Example: Run all steps")
        lines.append("# opencode run -p FINAL_IMPLEMENTATION_PLAN.md")
        lines.append("```")
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export as JSON for programmatic consumption."""
        import dataclasses
        def serialize(obj):
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            elif isinstance(obj, (list, tuple)):
                return [serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return obj
        
        import dataclasses
        return json.dumps(serialize(self), indent=2)
    
    def save_markdown(self, file_path: Path) -> None:
        """Save plan as markdown file."""
        file_path.write_text(self.to_markdown())
    
    def save_json(self, file_path: Path) -> None:
        """Save plan as JSON file."""
        file_path.write_text(self.to_json())


class PlanGenerator:
    """Generates implementation plans from task results."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.task_store = get_task_store()
        self.memory = get_memory_manager()
    
    def generate_from_task(self, task: Task) -> ImplementationPlan:
        """Generate implementation plan from a completed task."""
        # Extract final plan from task
        final_plan = task.final_plan or ""
        
        plan = ImplementationPlan(
            task_id=task.id,
            task_description=task.description,
            metadata={
                "task_status": task.status,
                "approval_status": task.approval_status,
                "round1_builder": bool(task.round1_builder),
                "round1_analyst": bool(task.round1_analyst),
                "round1_strategist": bool(task.round1_strategist),
                "round2_builder": bool(task.round2_builder),
                "round2_analyst": bool(task.round2_analyst),
                "round2_strategist": bool(task.round2_strategist),
                "has_red_team": bool(task.red_team_result),
            }
        )
        
        # Parse final plan into steps
        steps = self._parse_plan_into_steps(final_plan)
        plan.steps = steps
        
        return plan
    
    def _parse_plan_into_steps(self, plan_text: str) -> List[ImplementationStep]:
        """Parse markdown plan text into structured steps."""
        steps = []
        lines = plan_text.split("\n")
        current_step = None
        step_order = 0
        
        for line in lines:
            # Detect step headers
            if line.startswith("### Step ") or line.startswith("## Step ") or line.startswith("### "):
                if current_step:
                    steps.append(current_step)
                step_order += 1
                current_step = ImplementationStep(
                    order=step_order,
                    title=line.replace("### Step ", "").replace("### ", "").replace("## Step ", "").strip(),
                    description="",
                )
            elif current_step and line.strip():
                if line.startswith("**Description:**"):
                    current_step.description = line.replace("**Description:**", "").strip()
                elif line.startswith("**File:**") or line.startswith("**File:**"):
                    current_step.file_path = line.replace("**File:**", "").replace("`", "").strip()
                elif line.startswith("**Action:**"):
                    current_step.action = line.replace("**Action:**", "").strip()
                elif line.startswith("**Priority:**"):
                    current_step.priority = self._parse_priority(line)
                elif line.startswith("**Effort:**"):
                    current_step.estimated_effort = line.replace("**Effort:**", "").strip()
                elif line.startswith("**Dependencies:**"):
                    deps = line.replace("**Dependencies:**", "").strip()
                    current_step.dependencies = [d.strip() for d in deps.split(",")]
                elif line.startswith("- [ ] ") or line.startswith("- [x] "):
                    current_step.acceptance_criteria.append(line.replace("- [ ] ", "").replace("- [x] ", "").strip())
                elif line.startswith("**Test Command:**"):
                    # Next line should be the command
                    pass
                else:
                    if current_step.description:
                        current_step.description += "\n" + line.strip()
                    else:
                        current_step.description = line.strip()
        
        if current_step:
            steps.append(current_step)
        
        # Ensure steps are ordered
        for i, step in enumerate(steps):
            step.order = i + 1
        
        return steps if steps else [ImplementationStep(
            order=1,
            title="Implement task",
            description="Implement the task as described in the final plan",
            action="implement",
        )]
    
    def _parse_priority(self, line: str) -> int:
        if "High" in line or "1" in line:
            return 1
        elif "Medium" in line or "2" in line:
            return 2
        elif "Low" in line or "3" in line:
            return 3
        return 1


def create_implementation_plan(task: Task) -> ImplementationPlan:
    """Generate implementation plan from task."""
    generator = PlanGenerator(Path.cwd())
    return generator.generate_from_task(task)


def export_plan_for_jetbrains(task: Task, output_dir: Optional[Path] = None) -> Path:
    """Export implementation plan as markdown for JetBrains AI/opencode."""
    plan = create_implementation_plan(task)
    
    if output_dir is None:
        output_dir = Path.cwd() / "plans"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"FINAL_IMPLEMENTATION_PLAN_{task.id}.md"
    plan.save_markdown(file_path)
    
    logger.info(f"Saved implementation plan to {file_path}")
    return file_path


def export_plan_json(task: Task, output_dir: Optional[Path] = None) -> Path:
    """Export implementation plan as JSON."""
    plan = create_implementation_plan(task)
    
    if output_dir is None:
        output_dir = Path.cwd() / "plans"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"FINAL_IMPLEMENTATION_PLAN_{task.id}.json"
    plan.save_json(file_path)
    
    logger.info(f"Saved implementation plan JSON to {file_path}")
    return file_path