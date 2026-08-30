"""LLM Client for OpenAI-compatible endpoints (llama.cpp, OpenRouter)."""

from __future__ import annotations

import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.3
    max_tokens: int = 4096
    top_p: float = 0.9
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    message: Message
    finish_reason: str = "stop"
    index: int = 0


class UsageDetails(BaseModel):
    """Token usage details - handles OpenRouter's nested objects."""
    cached_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    video_tokens: Optional[int] = None
    # Allow extra fields for future compatibility
    class Config:
        extra = "allow"


class CostDetails(BaseModel):
    """Cost details from OpenRouter."""
    upstream_inference_cost: Optional[float] = None
    upstream_completions_cost: Optional[float] = None
    # Allow extra fields
    class Config:
        extra = "allow"


class Usage(BaseModel):
    """Token usage with optional nested detail objects."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[UsageDetails] = None
    completion_tokens_details: Optional[UsageDetails] = None
    cost_details: Optional[CostDetails] = None
    # Allow extra fields for future compatibility
    class Config:
        extra = "allow"


class ChatCompletionResponse(BaseModel):
    choices: List[ChatCompletionChoice]
    model: str
    usage: Optional[Usage] = None
    # Some providers might return extra fields
    class Config:
        extra = "allow"


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    default_temperature: float = 0.3
    default_max_tokens: int = 4096
    timeout: int = 120


class ChatCompletionChoice(BaseModel):
    message: Message
    finish_reason: str = "stop"
    index: int = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.3
    max_tokens: int = 4096
    top_p: float = 0.9
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    message: Message
    finish_reason: str = "stop"
    index: int = 0


class LLMClient:
    """Client for OpenAI-compatible APIs (llama.cpp, OpenRouter)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/"),
            timeout=httpx.Timeout(self.config.timeout, connect=10.0),
            headers=headers,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def health_check(self) -> bool:
        """Check if server is reachable."""
        if not self._client:
            return False
        try:
            resp = await self._client.get("models", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> ChatCompletionResponse:
        """Call chat completions endpoint."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.default_temperature,
            "max_tokens": max_tokens or self.config.default_max_tokens,
            "top_p": 0.9,
            "stream": stream,
        }

        try:
            response = await self._client.post(
                "chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return ChatCompletionResponse(**data)
        except httpx.TimeoutException:
            raise RuntimeError("Request timed out")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Request failed: {e}")

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Stream chat completion."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.default_temperature,
            "max_tokens": max_tokens or self.config.default_max_tokens,
            "top_p": 0.9,
            "stream": True,
        }

        async with self._client.stream(
            "POST",
            "chat/completions",
            json=payload,
            timeout=httpx.Timeout(self.config.timeout, connect=10.0),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    yield data


def create_ministral_client() -> LLMClient:
    """Create client for local Ministral (Builder)."""
    base_url = os.getenv("MINISTRAL_BASE_URL", "http://localhost:8001/v1")
    # Ensure V1 API prefix for llama.cpp OpenAI-compatible endpoint
    if not base_url.endswith('/v1'):
        base_url = base_url.rstrip('/') + '/v1'
    return LLMClient(LLMConfig(
        base_url=base_url,
        api_key=os.getenv("MINISTRAL_API_KEY", "dummy"),
        model=os.getenv("MINISTRAL_MODEL", "ministral"),
        default_temperature=float(os.getenv("BUILDER_TEMP", "0.3")),
        default_max_tokens=int(os.getenv("MINISTRAL_MAX_TOKENS", "2048")),
    ))


def create_deepseek_client() -> LLMClient:
    """Create client for local DeepSeek (Analyst)."""
    base_url = os.getenv("DEEPSEEK_BASE_URL", "http://localhost:8002/v1")
    # Ensure V1 API prefix for llama.cpp OpenAI-compatible endpoint
    if not base_url.endswith('/v1'):
        base_url = base_url.rstrip('/') + '/v1'
    return LLMClient(LLMConfig(
        base_url=base_url,
        api_key=os.getenv("DEEPSEEK_API_KEY", "dummy"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek"),
        default_temperature=float(os.getenv("ANALYST_TEMP", "0.2")),
        default_max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "2048")),
    ))


def create_openrouter_client() -> LLMClient:
    """Create client for OpenRouter (Strategist)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError("OPENROUTER_API_KEY not set in environment")

    return LLMClient(LLMConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=os.getenv("STRATEGIST_MODEL", "anthropic/claude-3.5-sonnet"),
        default_temperature=float(os.getenv("STRATEGIST_TEMP", "0.1")),
        default_max_tokens=int(os.getenv("STRATEGIST_MAX_TOKENS", "8192")),
    ))