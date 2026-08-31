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


async def _kaggle_backends_available() -> bool:
    """Check if Ministral/DeepSeek (via the FRP tunnel to Kaggle) are
    actually reachable right now. Used to auto-decide whether to use the
    full 3-brain committee or fall back to OpenRouter-only, with no
    manual prompt needed each session.
    """
    from agents.llm_client import create_ministral_client, create_deepseek_client
    try:
        async with create_ministral_client() as client:
            ministral_ok = await client.health_check()
    except Exception:
        ministral_ok = False
    try:
        async with create_deepseek_client() as client:
            deepseek_ok = await client.health_check()
    except Exception:
        deepseek_ok = False
    return ministral_ok and deepseek_ok


async def create_all_agents_auto() -> Dict[str, BaseAgent]:
    """Auto-detect whether the Kaggle-backed local models are reachable.
    If yes: full 3-brain committee (Builder=Ministral, Analyst=DeepSeek,
    Strategist=OpenRouter). If no: OpenRouter-only fallback for all three
    roles, so tasks can still run even when Kaggle is not started.
    """
    if await _kaggle_backends_available():
        logger.info("Kaggle tunnel detected — using full 3-brain committee.")
        return create_all_agents()
    else:
        logger.warning(
            "Kaggle tunnel not reachable (Ministral/DeepSeek offline) — "
            "falling back to OpenRouter-only for all roles. Start your "
            "Kaggle session to restore the free local models."
        )
        return {
            "builder": OpenRouterAgent(),
            "analyst": OpenRouterAgent(),
            "strategist": OpenRouterAgent(),
        }


def create_all_agents() -> Dict[str, BaseAgent]:
    """Factory to create all three agents (always uses Kaggle-backed models
    regardless of reachability -- use create_all_agents_auto() for the
    auto-detecting version instead, unless you specifically want this
    unconditional behavior).
    """
    return {
        "builder": MinistralAgent(),
        "analyst": DeepSeekAgent(),
        "strategist": OpenRouterAgent(),
    }