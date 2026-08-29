"""Three-Brain AI Orchestrator - Local LLM orchestration for AI engineering."""

__version__ = "0.1.0"
__author__ = "Three-Brain AI"

from orchestrator import ThreeBrainOrchestrator, create_orchestrator
from agents import MinistralAgent, DeepSeekAgent, OpenRouterAgent, RedTeamAgent

__all__ = [
    "ThreeBrainOrchestrator",
    "create_orchestrator",
    "MinistralAgent",
    "DeepSeekAgent",
    "OpenRouterAgent",
    "RedTeamAgent",
]