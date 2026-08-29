"""Router for directing tasks to appropriate agents."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from agents import MinistralAgent, DeepSeekAgent, OpenRouterAgent, RedTeamAgent


class AgentRole(Enum):
    BUILDER = "builder"
    ANALYST = "analyst"
    STRATEGIST = "strategist"
    RED_TEAM = "red_team"


class TaskRouter:
    """Routes tasks to appropriate agents based on type."""

    def __init__(self):
        self.agents = {
            AgentRole.BUILDER: MinistralAgent(),
            AgentRole.ANALYST: DeepSeekAgent(),
            AgentRole.STRATEGIST: OpenRouterAgent(),
            AgentRole.RED_TEAM: RedTeamAgent(),
        }

    def get_agent(self, role: AgentRole):
        """Get agent by role."""
        return self.agents.get(role)

    def get_pipeline_for_task(self, task_type: str) -> List[AgentRole]:
        """Determine agent pipeline for task type."""
        pipelines = {
            "feature": [AgentRole.BUILDER, AgentRole.ANALYST, AgentRole.STRATEGIST, AgentRole.RED_TEAM],
            "bugfix": [AgentRole.BUILDER, AgentRole.ANALYST, AgentRole.STRATEGIST],
            "refactor": [AgentRole.BUILDER, AgentRole.ANALYST, AgentRole.STRATEGIST, AgentRole.RED_TEAM],
            "architecture": [AgentRole.BUILDER, AgentRole.ANALYST, AgentRole.STRATEGIST],
            "security": [AgentRole.RED_TEAM, AgentRole.STRATEGIST],
            "review": [AgentRole.ANALYST, AgentRole.STRATEGIST],
            "default": [AgentRole.BUILDER, AgentRole.ANALYST, AgentRole.STRATEGIST, AgentRole.RED_TEAM],
        }
        return pipelines.get(task_type, pipelines["default"])

    def detect_task_type(self, description: str) -> str:
        """Detect task type from description."""
        desc = description.lower()

        if any(kw in desc for kw in ["security", "vulnerability", "auth", "permission", "secret", "credential"]):
            return "security"
        elif any(kw in desc for kw in ["bug", "fix", "error", "crash", "fail", "issue"]):
            return "bugfix"
        elif any(kw in desc for kw in ["refactor", "restructure", "reorganize", "cleanup"]):
            return "refactor"
        elif any(kw in desc for kw in ["architect", "design", "structure", "pattern"]):
            return "architecture"
        elif any(kw in desc for kw in ["review", "audit", "analyze", "evaluate"]):
            return "review"
        elif any(kw in desc for kw in ["feature", "add", "implement", "create", "build", "new"]):
            return "feature"
        return "default"