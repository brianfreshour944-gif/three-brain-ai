"""Main orchestrator for the three-brain system."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import MinistralAgent, DeepSeekAgent, OpenRouterAgent, RedTeamAgent, create_all_agents
from agents.llm_client import LLMClient, create_ministral_client, create_deepseek_client, create_openrouter_client
from orchestrator.config import REQUIRE_APPROVAL
from orchestrator.context_manager import ContextManager, TaskContext
from orchestrator.context_system import create_context_builder
from orchestrator.memory import get_memory_manager
from orchestrator.task_store import Task, TaskStore, get_task_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Result of full orchestration."""
    task: Task
    builder_output: str
    analyst_output: str
    strategist_output: str
    red_team_output: Optional[str]
    round1_builder: str
    round1_analyst: str
    round1_strategist: str
    round2_builder: str
    round2_analyst: str
    round2_strategist: str
    final_plan: str
    requires_approval: bool


class ThreeBrainOrchestrator:
    """Orchestrates the three-brain AI system with two-round collaborative workflow."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        agents: Optional[Dict[str, Any]] = None,
        enable_red_team: bool = True,
    ):
        self.project_root = project_root or Path.cwd()
        self.context_manager = ContextManager(self.project_root)
        self.memory = get_memory_manager()
        self.task_store = get_task_store()
        self.enable_red_team = enable_red_team

        # Initialize agents
        if agents:
            self.builder = agents.get("builder")
            self.analyst = agents.get("analyst")
            self.strategist = agents.get("strategist")
            self.red_team = agents.get("red_team")
        else:
            all_agents = create_all_agents()
            self.builder = all_agents["builder"]
            self.analyst = all_agents["analyst"]
            self.strategist = all_agents["strategist"]
            self.red_team = RedTeamAgent() if enable_red_team else None

    async def check_llm_health(self) -> Dict[str, bool]:
        """Check health of all LLM endpoints."""
        results = {}

        # Check Ministral
        try:
            async with create_ministral_client() as client:
                results["ministral"] = await client.health_check()
        except Exception:
            results["ministral"] = False

        # Check DeepSeek
        try:
            async with create_deepseek_client() as client:
                results["deepseek"] = await client.health_check()
        except Exception:
            results["deepseek"] = False

        # Check OpenRouter
        try:
            async with create_openrouter_client() as client:
                results["openrouter"] = await client.health_check()
        except Exception:
            results["openrouter"] = False

        return results

    async def run_task(
        self,
        task_description: str,
        user_notes: str = "",
        relevant_files: Optional[List[Path]] = None,
        file_contents: Optional[Dict[str, str]] = None,
        auto_approve: bool = False,
    ) -> OrchestrationResult:
        """Run a full task through the two-round three-brain pipeline."""

        # Create task
        task = self.task_store.create(
            description=task_description,
            project_root=str(self.project_root),
            context={"user_notes": user_notes},
        )
        task.status = "in_progress"
        self.task_store.update(task)

        logger.info(f"Starting task {task.id}: {task_description}")

        try:
            # Build context using ContextBuilder with Repomix
            context_builder = create_context_builder(self.project_root)
            
            # Create model-specific contexts (local models have 4096 limit)
            context_builder_model = context_builder.build_context(
                task_description=task_description,
                user_notes=user_notes,
                relevant_files=relevant_files or [],
                file_contents=file_contents or {},
                target_model="deepseek",  # Use 4096 limit for local models
                use_repomix=True,
                reserve_tokens=3000,  # Reserve 3000 for completion (2048 max + buffer)
            )
            context_strategist = context_builder.build_context(
                task_description=task_description,
                user_notes=user_notes,
                relevant_files=relevant_files or [],
                file_contents=file_contents or {},
                target_model="strategist",  # Use 200K limit for cloud model
                use_repomix=True,
            )

            # ============================================================
            # ROUND 1: Parallel Independent Analysis
            # ============================================================
            logger.info("=== ROUND 1: Parallel Independent Analysis ===")
            task.status = "round1_parallel"
            self.task_store.update(task)

            round1_results = await self._run_round1_parallel(task_description, context_builder_model, context_strategist)
            task.round1_builder = round1_results["builder"]
            task.round1_analyst = round1_results["analyst"]
            task.round1_strategist = round1_results["strategist"]
            self.task_store.update(task)

            # ============================================================
            # ROUND 2: Sequential Refinement
            # ============================================================
            logger.info("=== ROUND 2: Sequential Refinement ===")
            task.status = "round2_refinement"
            self.task_store.update(task)

            round2_results = await self._run_round2_sequential(
                task_description, context_builder_model, context_strategist, round1_results
            )
            task.round2_builder = round2_results["builder"]
            task.round2_analyst = round2_results["analyst"]
            task.round2_strategist = round2_results["strategist"]
            self.task_store.update(task)

            # ============================================================
            # RED TEAM (Optional)
            # ============================================================
            red_team_output = None
            if self.enable_red_team and self.red_team:
                logger.info("=== RED TEAM SECURITY REVIEW ===")
                task.status = "red_team"
                self.task_store.update(task)

                red_team_context = self._build_red_team_context(
                    context_builder_model, round1_results, round2_results
                )
                red_team_response = await self.red_team.process(
                    "Perform security review of the proposed implementation",
                    red_team_context
                )
                task.red_team_result = red_team_response
                red_team_output = red_team_response.content
                self.task_store.update(task)

            # Compile final plan
            final_plan = self._compile_final_plan(task, round1_results, round2_results)
            task.final_plan = final_plan
            task.status = "awaiting_approval" if REQUIRE_APPROVAL and not auto_approve else "approved"
            self.task_store.update(task)

            # Handle approval
            if REQUIRE_APPROVAL and not auto_approve:
                logger.info("Awaiting human approval...")

            logger.info(f"Task {task.id} completed successfully")
            return OrchestrationResult(
                task=task,
                builder_output=round2_results["builder"],
                analyst_output=round2_results["analyst"],
                strategist_output=round2_results["strategist"],
                red_team_output=red_team_output,
                round1_builder=round1_results["builder"],
                round1_analyst=round1_results["analyst"],
                round1_strategist=round1_results["strategist"],
                round2_builder=round2_results["builder"],
                round2_analyst=round2_results["analyst"],
                round2_strategist=round2_results["strategist"],
                final_plan=final_plan,
                requires_approval=REQUIRE_APPROVAL and not auto_approve,
            )

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.status = "failed"
            task.error = str(e)
            self.task_store.update(task)
            raise

    async def _run_round1_parallel(
        self,
        task_description: str,
        context_model: str,
        context_strategist: str,
    ) -> Dict[str, str]:
        """Run Round 1: All three agents analyze independently in parallel."""

        async def run_builder():
            logger.info("Round 1: Builder starting...")
            response = await self.builder.process(task_description, context_model)
            return ("builder", response.content)

        async def run_analyst():
            logger.info("Round 1: Analyst starting...")
            response = await self.analyst.process(
                "Analyze this task independently. Identify requirements, risks, and potential approaches.",
                context_model
            )
            return ("analyst", response.content)

        async def run_strategist():
            logger.info("Round 1: Strategist starting...")
            response = await self.strategist.process(
                "Analyze this task from an architectural perspective. Identify key decisions, trade-offs, and long-term implications.",
                context_strategist
            )
            return ("strategist", response.content)

        # Run all three in parallel
        results = await asyncio.gather(
            run_builder(),
            run_analyst(),
            run_strategist(),
        )

        return {role: content for role, content in results}

    async def _run_round2_sequential(
        self,
        task_description: str,
        context_model: str,
        context_strategist: str,
        round1_results: Dict[str, str],
    ) -> Dict[str, str]:
        """Run Round 2: Sequential refinement with full context."""

        # Build context with all Round 1 outputs
        round1_context = context_model + "\n\n" + "=" * 60 + "\n"
        round1_context += "=== ROUND 1: INDEPENDENT ANALYSES ===\n"
        round1_context += "\n--- BUILDER (Round 1) ---\n" + round1_results["builder"]
        round1_context += "\n--- ANALYST (Round 1) ---\n" + round1_results["analyst"]
        round1_context += "\n--- STRATEGIST (Round 1) ---\n" + round1_results["strategist"]

        # Round 2A: Builder creates final plan incorporating all Round 1 feedback
        logger.info("Round 2A: Builder creating final plan...")
        builder_context = round1_context + "\n\n=== ROUND 2: REFINEMENT ===\n"
        builder_context += "\nCreate a comprehensive final implementation plan incorporating all Round 1 analyses."
        round2_builder = await self.builder.process(
            "Create the final implementation plan based on all Round 1 analyses.",
            builder_context
        )

        # Round 2B: Analyst attacks the Builder's final plan
        logger.info("Round 2B: Analyst reviewing final plan...")
        analyst_context = builder_context + "\n\n=== BUILDER FINAL PLAN ===\n" + round2_builder.content
        round2_analyst = await self.analyst.process(
            "Critique the Builder's final plan. Find flaws, gaps, and improvements.",
            analyst_context
        )

        # Round 2C: Strategist synthesizes into final contract
        logger.info("Round 2C: Strategist creating implementation contract...")
        # Strategist uses the larger context window
        strategist_context = context_strategist + "\n\n" + "=" * 60 + "\n"
        strategist_context += "=== ROUND 1: INDEPENDENT ANALYSES ===\n"
        strategist_context += "\n--- BUILDER (Round 1) ---\n" + round1_results["builder"]
        strategist_context += "\n--- ANALYST (Round 1) ---\n" + round1_results["analyst"]
        strategist_context += "\n--- STRATEGIST (Round 1) ---\n" + round1_results["strategist"]
        strategist_context += "\n\n=== ROUND 2: REFINEMENT ===\n"
        strategist_context += "\n--- BUILDER (Round 2) ---\n" + round2_builder.content
        strategist_context += "\n--- ANALYST (Round 2) ---\n" + round2_analyst.content
        strategist_context += "\n\nSynthesize into a final implementation contract. Resolve all issues."
        round2_strategist = await self.strategist.process(
            "Synthesize into a final implementation contract. Resolve all issues.",
            strategist_context
        )

        return {
            "builder": round2_builder.content,
            "analyst": round2_analyst.content,
            "strategist": round2_strategist.content,
        }

    def _build_red_team_context(
        self,
        context: str,
        round1_results: Dict[str, str],
        round2_results: Dict[str, str],
    ) -> str:
        """Build context for Red Team security review."""
        parts = [
            context,
            "\n\n" + "=" * 60 + "\n",
            "=== ROUND 1: INDEPENDENT ANALYSES ===\n",
            "\n--- BUILDER (Round 1) ---\n" + round1_results["builder"],
            "\n--- ANALYST (Round 1) ---\n" + round1_results["analyst"],
            "\n--- STRATEGIST (Round 1) ---\n" + round1_results["strategist"],
            "\n\n" + "=" * 60 + "\n",
            "=== ROUND 2: REFINEMENT ===\n",
            "\n--- BUILDER (Round 2) ---\n" + round2_results["builder"],
            "\n--- ANALYST (Round 2) ---\n" + round2_results["analyst"],
            "\n--- STRATEGIST (Round 2) ---\n" + round2_results["strategist"],
        ]
        return "\n".join(parts)

    def _compile_final_plan(
        self,
        task: Task,
        round1_results: Dict[str, str],
        round2_results: Dict[str, str],
    ) -> str:
        """Compile the final implementation plan from all rounds."""
        parts = [
            "# FINAL IMPLEMENTATION PLAN",
            f"\n**Task:** {task.description}",
            f"**Task ID:** {task.id}",
            f"**Generated:** {task.updated_at}",
            "\n---\n",
        ]

        # Round 1 Summary
        parts.append("## ROUND 1: INDEPENDENT ANALYSES")
        parts.append("\n### Builder (Round 1)")
        parts.append(round1_results["builder"])
        parts.append("\n### Analyst (Round 1)")
        parts.append(round1_results["analyst"])
        parts.append("\n### Strategist (Round 1)")
        parts.append(round1_results["strategist"])

        # Round 2 Summary
        parts.append("\n\n## ROUND 2: SEQUENTIAL REFINEMENT")
        parts.append("\n### Builder (Round 2) - Final Plan")
        parts.append(round2_results["builder"])
        parts.append("\n### Analyst (Round 2) - Critique")
        parts.append(round2_results["analyst"])
        parts.append("\n### Strategist (Round 2) - Implementation Contract")
        parts.append(round2_results["strategist"])

        # Red Team
        if task.red_team_result:
            parts.append("\n\n## RED TEAM SECURITY REVIEW")
            parts.append(task.red_team_result.content)

        parts.append("\n---\n")
        parts.append("## NEXT STEPS")
        parts.append("1. Review this plan")
        parts.append("2. Approve or request revisions")
        parts.append("3. Implement in JetBrains/IDE")
        parts.append("4. Run tests and validate")

        return "\n".join(parts)

    def approve_task(self, task_id: str, notes: str = "") -> Task:
        """Approve a task for implementation."""
        task = self.task_store.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        task.approval_status = "approved"
        task.approval_notes = notes
        task.status = "approved"
        self.task_store.update(task)

        # Record decision in memory
        self.memory.add_decision(
            decision=f"Approved task {task_id}: {task.description[:50]}",
            rationale=notes or "Human approved",
            context=task.description,
        )

        return task

    def reject_task(self, task_id: str, notes: str) -> Task:
        """Reject a task."""
        task = self.task_store.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        task.approval_status = "rejected"
        task.approval_notes = notes
        task.status = "rejected"
        self.task_store.update(task)

        self.memory.add_decision(
            decision=f"Rejected task {task_id}: {task.description[:50]}",
            rationale=notes,
            context=task.description,
        )

        return task

    def request_revisions(self, task_id: str, notes: str) -> Task:
        """Request revisions to a task."""
        task = self.task_store.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        task.approval_status = "revisions_requested"
        task.approval_notes = notes
        task.status = "in_progress"
        self.task_store.update(task)

        return task

    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get task status."""
        return self.task_store.get(task_id)

    def list_tasks(self, status: Optional[str] = None) -> List[Task]:
        """List tasks."""
        return self.task_store.list_all(status)


def create_orchestrator(project_root: Optional[Path] = None, **kwargs) -> ThreeBrainOrchestrator:
    """Factory to create orchestrator."""
    return ThreeBrainOrchestrator(project_root=project_root, **kwargs)