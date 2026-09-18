"""LLM Client for OpenAI-compatible endpoints (llama.cpp, OpenRouter)."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: Optional[str] = None


class UsageDetails(BaseModel):
    """Token usage details - handles OpenRouter's nested objects."""
    cached_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    video_tokens: Optional[int] = None

    class Config:
        extra = "allow"


class CostDetails(BaseModel):
    """Cost details from OpenRouter."""
    upstream_inference_cost: Optional[float] = None
    upstream_completions_cost: Optional[float] = None

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

    class Config:
        extra = "allow"


class ChatCompletionChoice(BaseModel):
    message: Message
    finish_reason: str = "stop"
    index: int = 0


class ChatCompletionResponse(BaseModel):
    choices: List[ChatCompletionChoice]
    model: str
    usage: Optional[Usage] = None

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


class LLMClientPool:
    """Singleton pool of HTTP clients per endpoint/model combination."""

    _instances: Dict[str, "LLMClientPool"] = {}
    _locks: Dict[str, asyncio.Lock] = {}
    _global_lock = asyncio.Lock()

    def __new__(cls, config: LLMConfig):
        key = f"{config.base_url.rstrip('/')}:{config.model}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, config: LLMConfig):
        if hasattr(self, '_initialized'):
            return
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._init_lock = asyncio.Lock()
        self._initialized = True

    @classmethod
    def _get_lock(cls, key: str) -> asyncio.Lock:
        if key not in cls._locks:
            cls._locks[key] = asyncio.Lock()
        return cls._locks[key]

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create shared HTTP client with connection pooling."""
        if self._client is not None and not self._client.is_closed:
            return self._client

        async with self._init_lock:
            if self._client is not None and not self._client.is_closed:
                return self._client

            headers = {"Content-Type": "application/json"}
            if self.config.api_key and self.config.api_key != "dummy":
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            limits = httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=300.0,
            )

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"),
                timeout=httpx.Timeout(self.config.timeout, connect=10.0),
                headers=headers,
                limits=limits,
                http2=self._try_http2(),
                follow_redirects=True,
            )
            logger.debug(
                f"Created new pooled client for {self.config.base_url}:{self.config.model}"
            )
            return self._client

    @staticmethod
    def _try_http2() -> bool:
        """Try to enable HTTP/2, return False if httpx[http2] not installed."""
        try:
            import h2  # noqa: F401
            return True
        except ImportError:
            logger.debug("HTTP/2 not available (h2 not installed), falling back to HTTP/1.1")
            return False

    async def close(self):
        """Close the pooled client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @classmethod
    async def close_all(cls):
        """Close all pooled clients."""
        for instance in cls._instances.values():
            await instance.close()
        cls._instances.clear()


class LLMClient:
    """Client for OpenAI-compatible APIs using shared connection pool."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._pool = LLMClientPool(config)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass  # Pool manages lifecycle

    async def health_check(self) -> bool:
        """Check if server is reachable."""
        try:
            client = await self._pool.get_client()
            resp = await client.get("models", timeout=5.0)
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
        """Call chat completions endpoint using pooled connection."""
        client = await self._pool.get_client()

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": (
                temperature
                if temperature is not None
                else self.config.default_temperature
            ),
            "max_tokens": max_tokens or self.config.default_max_tokens,
            "top_p": 0.9,
            "stream": stream,
        }

        try:
            response = await client.post(
                "chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(f"Provider error: {data.get('error')}")
            return ChatCompletionResponse(**data)
        except httpx.TimeoutException as e:
            raise RuntimeError("Request timed out") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Request failed: {e}") from e

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Stream chat completion using pooled connection."""
        client = await self._pool.get_client()

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": (
                temperature
                if temperature is not None
                else self.config.default_temperature
            ),
            "max_tokens": max_tokens or self.config.default_max_tokens,
            "top_p": 0.9,
            "stream": True,
        }

        async with client.stream(
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
    api_key = os.getenv("MINISTRAL_API_KEY", "dummy")
    if api_key == "dummy":
        logger.warning(
            "MINISTRAL_API_KEY is not set (resolved to 'dummy') — the Ministral "
            "endpoint is UNAUTHENTICATED. Requests carry no Authorization header, "
            "so if this endpoint is exposed via Oracle's public ports (FRP tunnel) "
            "anyone can call it. Set MINISTRAL_API_KEY in your environment."
        )
    return LLMClient(LLMConfig(
        base_url=base_url,
        api_key=api_key,
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
    api_key = os.getenv("DEEPSEEK_API_KEY", "dummy")
    if api_key == "dummy":
        logger.warning(
            "DEEPSEEK_API_KEY is not set (resolved to 'dummy') — the DeepSeek "
            "endpoint is UNAUTHENTICATED. Requests carry no Authorization header, "
            "so if this endpoint is exposed via Oracle's public ports (FRP tunnel) "
            "anyone can call it. Set DEEPSEEK_API_KEY in your environment."
        )
    return LLMClient(LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek"),
        default_temperature=float(os.getenv("ANALYST_TEMP", "0.2")),
        default_max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "2048")),
    ))


class FallbackLLMClient:
    """Wraps a primary LLMClient and falls back to a secondary client on provider failure.

    Tracks which backend actually served the most recent call via `last_used`
    ("primary" or "fallback"), so callers (see OpenRouterAgent.process) can
    report accurately instead of always claiming the primary model was used.
    """

    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback
        self.last_used = "primary"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def config(self):
        if self.last_used == "fallback" and self.fallback is not None:
            return self.fallback.config
        return self.primary.config

    async def health_check(self) -> bool:
        return await self.primary.health_check()

    async def chat_completion(self, *args, **kwargs):
        try:
            result = await self.primary.chat_completion(*args, **kwargs)
            self.last_used = "primary"
            return result
        except Exception as e:
            if self.fallback is None:
                raise
            primary_model = self.primary.config.model
            fallback_model = self.fallback.config.model if self.fallback else "N/A"
            logger.warning(
                f"Primary Strategist '{primary_model}' failed ({e.__class__.__name__}: {e}); "
                f"falling back to '{fallback_model}'."
            )
            self.last_used = "fallback"
            result = await self.fallback.chat_completion(*args, **kwargs)
            await _notify_discord(
                f":warning: **Strategist fallback triggered**\n"
                f"Primary model `{primary_model}` failed: `{e.__class__.__name__}`\n"
                f"Fallback: `{fallback_model}` — succeeded"
            )
            return result

    async def chat_completion_stream(self, *args, **kwargs):
        try:
            async for chunk in self.primary.chat_completion_stream(*args, **kwargs):
                yield chunk
            self.last_used = "primary"
        except Exception as e:
            if self.fallback is None:
                raise
            primary_model = self.primary.config.model
            fallback_model = self.fallback.config.model if self.fallback else "N/A"
            logger.warning(
                f"Primary Strategist streaming '{primary_model}' failed ({e.__class__.__name__}: {e}); "
                f"falling back to '{fallback_model}'."
            )
            self.last_used = "fallback"
            await _notify_discord(
                f":warning: **Strategist fallback triggered (streaming)**\n"
                f"Primary `{primary_model}`: `{e.__class__.__name__}`\n"
                f"Fallback: `{fallback_model}`"
            )
            async for chunk in self.fallback.chat_completion_stream(*args, **kwargs):
                yield chunk


def create_openrouter_client():
    """Create client for Strategist: OpenRouter primary, optional local fallback."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError("OPENROUTER_API_KEY not set in environment")

    primary = LLMClient(LLMConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=os.getenv("STRATEGIST_MODEL", "openrouter/nemotron-3-ultra-free"),
        default_temperature=float(os.getenv("STRATEGIST_TEMP", "0.1")),
        default_max_tokens=int(os.getenv("STRATEGIST_MAX_TOKENS", "8192")),
    ))

    fallback_base_url = os.getenv("STRATEGIST_FALLBACK_BASE_URL")
    fallback = None
    if fallback_base_url:
        fb_url = fallback_base_url if fallback_base_url.endswith('/v1') else fallback_base_url.rstrip('/') + '/v1'
        fallback = LLMClient(LLMConfig(
            base_url=fb_url,
            api_key=os.getenv("STRATEGIST_FALLBACK_API_KEY", "dummy"),
            model=os.getenv("STRATEGIST_FALLBACK_MODEL", "falcon3:10b"),
            default_temperature=float(os.getenv("STRATEGIST_TEMP", "0.1")),
            default_max_tokens=int(os.getenv("STRATEGIST_FALLBACK_MAX_TOKENS", os.getenv("STRATEGIST_MAX_TOKENS", "8192"))),
            timeout=180,
        ))

    return FallbackLLMClient(primary, fallback)


async def _notify_discord(message: str) -> None:
    """Send a Discord notification if DISCORD_WEBHOOK_URL is set."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json={"content": message}, timeout=10.0)
    except Exception as e:
        logger.warning(f"Discord notification failed: {e}")


def create_openrouter_client_with_fallback() -> FallbackLLMClient:
    """Create OpenRouter Strategist with OpenRouter-to-OpenRouter fallback (GLM-4 Flash).

    Unlike create_openrouter_client() which uses STRATEGIST_FALLBACK_BASE_URL for
    a local model fallback, this always uses OpenRouter as both primary and fallback,
    so it works even when no local model server is running.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError("OPENROUTER_API_KEY not set in environment")

    primary = LLMClient(LLMConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=os.getenv("STRATEGIST_MODEL", "openrouter/nemotron-3-ultra-free"),
        default_temperature=float(os.getenv("STRATEGIST_TEMP", "0.1")),
        default_max_tokens=int(os.getenv("STRATEGIST_MAX_TOKENS", "8192")),
    ))

    fallback_model = os.getenv("STRATEGIST_FALLBACK_MODEL", "openrouter/z-ai/glm-5.3-flash")
    fallback = LLMClient(LLMConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=fallback_model,
        default_temperature=primary.config.default_temperature,
        default_max_tokens=primary.config.default_max_tokens,
        timeout=primary.config.timeout,
    ))

    logger.info(
        f"OpenRouter Strategist client created: "
        f"primary='{primary.config.model}', fallback='{fallback.config.model}'"
    )
    return FallbackLLMClient(primary=primary, fallback=fallback)
