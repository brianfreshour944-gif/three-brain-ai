"""Prompt templates for all agents."""

from prompts.builder import BUILDER_PROMPT
from prompts.analyst import ANALYST_PROMPT
from prompts.strategist import STRATEGIST_PROMPT
from prompts.red_team import RED_TEAM_PROMPT

__all__ = [
    "BUILDER_PROMPT",
    "ANALYST_PROMPT",
    "STRATEGIST_PROMPT",
    "RED_TEAM_PROMPT",
]