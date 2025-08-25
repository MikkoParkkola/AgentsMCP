"""
agentsmcp.routing.client
------------------------
Async HTTP client for OpenRouter.ai.

Uses `aiohttp` – a proven high‑performance HTTP library for async Python.
The client centralises:

* Authentication (env variable or keyword)
* Common error handling
* JSON payload helpers
* Request timeouts (configurable)

The client exposes a minimal interface:

    async def chat(self, model: str, messages: list[Message], **kwargs) -> ChatResponse

where `Message` and `ChatResponse` are tiny Pydantic‑style DTOs (but plain
dataclasses for zero‑overhead in this example).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional dependency handling
try:
    import aiohttp
    from aiohttp import ClientError, ClientResponse
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    ClientError = Exception
    ClientResponse = None

# --------------------------------------------------------------------------- #
@dataclass
class Message:
    role: str
    content: str


@dataclass
class ChatResponse:
    id: str
    model: str
    usage: Dict[str, int]
    choices: List[Dict[str, Any]]
    created_at: int


# --------------------------------------------------------------------------- #
class OpenRouterError(RuntimeError):
    """Base error for all OpenRouter client failures."""


class OpenRouterClient:
    """
    Async wrapper around OpenRouter.ai's chat endpoint.

    Parameters
    ----------
    api_key:
        API key to use – pulled from ``OPENROUTER_API_KEY`` by default.
    timeout:
        Timeout in seconds for each HTTP call.
    user_agent:
        Optional custom User‑Agent string.

    Notes
    -----
    * Instantiation is cheap – the internal ``ClientSession`` is shared
      across all calls.  The session is closed via ``close()`` or context
      manager (`async with`).
    * All errors are wrapped in :class:`OpenRouterError`.  Clients can
      catch that and react independently.
    """

    _DEFAULT_TIMEOUT = 30

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout: int | float = _DEFAULT_TIMEOUT,
        user_agent: str = "AgentsMCP/1.0",
    ) -> None:
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for OpenRouter client")
            
        self.api_key: str = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set - client may not work")

        self.timeout = timeout
        self.user_agent: str = user_agent
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": self.user_agent,
                    "Accept": "application/json",
                },
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def chat(
        self,
        *,
        model: str,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1,
        stop: Optional[List[str]] = None,
    ) -> ChatResponse:
        """
        Send a chat request to OpenRouter.ai.

        Parameters
        ----------
        model:
            Fully‑qualified model identifier (``openai/gpt-4`` etc.).
        messages:
            Chat history – a list of :class:`Message`.
        temperature, max_tokens, top_p, stop:
            LLM‑specific parameters.

        Returns
        -------
        ChatResponse:
            Parsed response body.

        Raises
        ------
        OpenRouterError:
            If an HTTP error occurs or the payload is invalid.
        """
        if not HAS_AIOHTTP:
            raise OpenRouterError("aiohttp is required for HTTP requests")

        session = await self._get_session()
        url = "https://openrouter.ai/api/v1/chat/completions"

        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if stop:
            payload["stop"] = stop

        try:
            async with session.post(url, json=payload) as resp:
                return await self._handle_response(resp)
        except (ClientError, asyncio.TimeoutError) as exc:
            logger.exception("HTTP request to OpenRouter failed")
            raise OpenRouterError("Network or timeout") from exc

    async def _handle_response(
        self, resp: ClientResponse
    ) -> ChatResponse:
        """Validate and parse the API response."""
        if resp.status != 200:
            err_body = await resp.text()
            logger.error(
                "OpenRouter returned %s: %s", resp.status, err_body
            )
            raise OpenRouterError(f"HTTP {resp.status}: {err_body}")

        data = await resp.json()
        # Basic field validation – keep it lightweight.
        try:
            return ChatResponse(
                id=data["id"],
                model=data["model"],
                usage=data.get("usage", {}),
                choices=data.get("choices", []),
                created_at=data.get("created", int(time.time())),
            )
        except KeyError as exc:
            raise OpenRouterError("Malformed response body") from exc


class MockOpenRouterClient:
    """Mock client for testing and development when aiohttp is not available."""
    
    def __init__(self, **kwargs):
        logger.info("Using mock OpenRouter client - no real API calls will be made")
        
    async def close(self):
        pass
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def chat(self, *, model: str, messages: List[Message], **kwargs) -> ChatResponse:
        """Mock chat response."""
        await asyncio.sleep(0.1)  # Simulate network delay
        
        return ChatResponse(
            id="mock-response-id",
            model=model,
            usage={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
            choices=[{
                "message": {
                    "role": "assistant",
                    "content": f"Mock response from {model}: {messages[-1].content[:50]}..."
                },
                "finish_reason": "stop"
            }],
            created_at=int(time.time())
        )


# Factory function
def create_client(**kwargs) -> OpenRouterClient:
    """Create appropriate client based on available dependencies."""
    if HAS_AIOHTTP:
        return OpenRouterClient(**kwargs)
    else:
        return MockOpenRouterClient(**kwargs)