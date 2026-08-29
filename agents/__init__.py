"""Agent interfaces for the three-brain system."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agents.llm_client import LLMClient, create_ministral_client, create_deepseek_client, create_openrouter_client

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from an agent."""
    agent_name: str
    content: str
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, name: str, system_prompt: str, client: LLMClient):
        self.name = name
        self.system_prompt = system_prompt
        self.client = client

    @abstractmethod
    async def process(self, task: str, context: str = "") -> AgentResponse:
        """Process a task and return response."""
        pass

    def _build_messages(self, task: str, context: str = "") -> List[Dict[str, str]]:
        """Build message list for the LLM."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})

        messages.append({"role": "user", "content": task})
        return messages


class MinistralAgent(BaseAgent):
    """Ministral - The Builder agent."""

    def __init__(self, client: Optional[LLMClient] = None):
        from prompts.builder import BUILDER_PROMPT
        super().__init__(
            name="Ministral (Builder)",
            system_prompt=BUILDER_PROMPT,
            client=client or create_ministral_client(),
        )

    async def process(self, task: str, context: str = "") -> AgentResponse:
        """Generate a solution/implementation plan."""
        logger.info(f"[{self.name}] Processing task")
        messages = self._build_messages(task, context)

        async with self.client as client:
            response = await client.chat_completion(messages)

        content = response.choices[0].message.content
        return AgentResponse(
            agent_name=self.name,
            content=content,
            confidence=0.85,
            metadata={"model": self.client.config.model}
        )


class DeepSeekAgent(BaseAgent):
    """DeepSeek - The Analyst agent."""

    def __init__(self, client: Optional[LLMClient] = None):
        from prompts.analyst import ANALYST_PROMPT
        super().__init__(
            name="DeepSeek (Analyst)",
            system_prompt=ANALYST_PROMPT,
            client=client or create_deepseek_client(),
        )

    async def process(self, task: str, context: str = "") -> AgentResponse:
        """Analyze and critique a solution."""
        logger.info(f"[{self.name}] Analyzing task")
        messages = self._build_messages(task, context)

        async with self.client as client:
            response = await client.chat_completion(messages)

        content = response.choices[0].message.content
        return AgentResponse(
            agent_name=self.name,
            content=content,
            confidence=0.9,
            metadata={"model": self.client.config.model}
        )


class OpenRouterAgent(BaseAgent):
    """OpenRouter - The Strategist agent."""

    def __init__(self, client: Optional[LLMClient] = None):
        from prompts.strategist import STRATEGIST_PROMPT
        super().__init__(
            name="OpenRouter (Strategist)",
            system_prompt=STRATEGIST_PROMPT,
            client=client or create_openrouter_client(),
        )

    async def process(self, task: str, context: str = "") -> AgentResponse:
        """Provide strategic architectural review."""
        logger.info(f"[{self.name}] Strategizing task")
        messages = self._build_messages(task, context)

        async with self.client as client:
            response = await client.chat_completion(messages)

        content = response.choices[0].message.content
        return AgentResponse(
            agent_name=self.name,
            content=content,
            confidence=0.95,
            metadata={"model": self.client.config.model}
        )


class RedTeamAgent(BaseAgent):
    """Red Team - Security/Adversarial review agent."""

    def __init__(self, client: Optional[LLMClient] = None):
        from prompts.red_team import RED_TEAM_PROMPT
        super().__init__(
            name="Red Team (Security)",
            system_prompt=RED_TEAM_PROMPT,
            client=client or create_openrouter_client(),
        )

    async def process(self, task: str, context: str = "") -> AgentResponse:
        """Perform security/adversarial review."""
        logger.info(f"[{self.name}] Performing security review")
        messages = self._build_messages(task, context)

        async with self.client as client:
            response = await client.chat_completion(messages)

        content = response.choices[0].message.content
        return AgentResponse(
            agent_name=self.name,
            content=content,
            confidence=0.9,
            metadata={"model": self.client.config.model}
        )


def create_all_agents() -> Dict[str, BaseAgent]:
    """Factory to create all three agents."""
    return {
        "builder": MinistralAgent(),
        "analyst": DeepSeekAgent(),
        "strategist": OpenRouterAgent(),
    }