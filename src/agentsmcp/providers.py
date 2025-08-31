"""Provider Model Discovery system.

Implements provider-agnostic model discovery across OpenAI, OpenRouter, and Ollama.

Backlog items covered:
- B1: Types and error classes (no HTTP behavior)
- B2: OpenAI adapter
- B3: OpenRouter adapter
- B4: Ollama adapter
- B5: Facade with unified error handling and minimal logging

Note: B6 adds an agent hook in BaseAgent (in a separate file).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import httpx
import structlog

from .config import ProviderType, ProviderConfig


logger = structlog.get_logger(__name__)

# Simple in-process cache for expensive model listings
_MODELS_CACHE: Dict[Tuple[str, str, str], List["Model"]] = {}


# =========================
# Errors (B1)
# =========================


class ProviderError(Exception):
    """Base error for provider operations."""


class ProviderAuthError(ProviderError):
    """Authentication/authorization error when calling a provider."""


class ProviderNetworkError(ProviderError):
    """Network connectivity or timeout error."""


class ProviderProtocolError(ProviderError):
    """Unexpected response shape, status, or protocol violation."""


# =========================
# Types (B1)
# =========================


@dataclass(frozen=True)
class Model:
    """Normalized model descriptor across providers.

    Fields are intentionally minimal; additional metadata can be added later.
    """

    id: str
    provider: ProviderType
    name: Optional[str] = None
    context_length: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "provider": self.provider.value,
            "name": self.name or self.id,
            "context_length": self.context_length,
        }


# =========================
# Provider adapters (B2–B4)
# =========================


def _bearer_headers(api_key: Optional[str]) -> dict:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def openai_list_models(config: ProviderConfig) -> List[Model]:
    """List models from OpenAI using /v1/models (B2).

    - Uses bearer auth with the provided api_key
    - Respects custom api_base when provided; defaults to https://api.openai.com/v1
    - Maps error conditions to Provider* errors
    """
    base = (config.api_base.rstrip("/")) if config.api_base else "https://api.openai.com/v1"
    url = f"{base}/models"
    headers = {"Accept": "application/json", **_bearer_headers(config.api_key)}

    if not config.api_key:
        raise ProviderAuthError("Missing OpenAI API key")

    try:
        logger.debug("openai.list_models.request", url=url)
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, headers=headers)
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:  # type: ignore[attr-defined]
        raise ProviderNetworkError(f"OpenAI network error: {e}") from e
    except Exception as e:  # pragma: no cover - unexpected
        raise ProviderError(f"OpenAI request failed: {e}") from e

    if resp.status_code in (401, 403):
        raise ProviderAuthError(f"OpenAI auth error: HTTP {resp.status_code}")
    if resp.status_code >= 500:
        raise ProviderNetworkError(f"OpenAI server error: HTTP {resp.status_code}")
    if resp.status_code >= 400:
        raise ProviderProtocolError(
            f"OpenAI protocol error: HTTP {resp.status_code} body={resp.text[:200]}"
        )

    try:
        data = resp.json()
        items = data.get("data", [])
        if not isinstance(items, list):
            raise ValueError("Response 'data' is not a list")
    except Exception as e:
        raise ProviderProtocolError(f"OpenAI response parse error: {e}") from e

    models = [
        Model(id=m.get("id", ""), provider=ProviderType.OPENAI, name=m.get("id"))
        for m in items
        if m.get("id")
    ]
    return models


def openrouter_list_models(config: ProviderConfig) -> List[Model]:
    """List models from OpenRouter using /models (B3).

    - Bearer auth with provided api_key
    - Respects custom api_base; defaults to https://openrouter.ai/api/v1
    - Normalizes to Model objects
    - Maps errors accordingly
    """
    base = (config.api_base.rstrip("/")) if config.api_base else "https://openrouter.ai/api/v1"
    url = f"{base}/models"
    headers = {"Accept": "application/json", **_bearer_headers(config.api_key)}

    if not config.api_key:
        raise ProviderAuthError("Missing OpenRouter API key")

    try:
        logger.debug("openrouter.list_models.request", url=url)
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, headers=headers)
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:  # type: ignore[attr-defined]
        raise ProviderNetworkError(f"OpenRouter network error: {e}") from e
    except Exception as e:  # pragma: no cover - unexpected
        raise ProviderError(f"OpenRouter request failed: {e}") from e

    if resp.status_code in (401, 403):
        raise ProviderAuthError(f"OpenRouter auth error: HTTP {resp.status_code}")
    if resp.status_code >= 500:
        raise ProviderNetworkError(
            f"OpenRouter server error: HTTP {resp.status_code}"
        )
    if resp.status_code >= 400:
        raise ProviderProtocolError(
            f"OpenRouter protocol error: HTTP {resp.status_code} body={resp.text[:200]}"
        )

    # Typical shape: { "data": [ {"id": "openai/gpt-4o", "name": "GPT-4o", ...}, ... ] }
    try:
        data = resp.json()
        items = data.get("data")
        if items is None:
            # Older docs also show top-level list; tolerate both
            items = data if isinstance(data, list) else []
        if not isinstance(items, list):
            raise ValueError("Response data is not a list")
    except Exception as e:
        raise ProviderProtocolError(f"OpenRouter response parse error: {e}") from e

    models = []
    for m in items:
        mid = m.get("id") or m.get("name")
        if not mid:
            continue
        models.append(
            Model(
                id=str(mid),
                provider=ProviderType.OPENROUTER,
                name=m.get("name") or str(mid),
            )
        )
    return models


def ollama_list_models(config: ProviderConfig) -> List[Model]:
    """List models from local Ollama daemon using /api/tags (B4).

    No API key is required for localhost. If the daemon is not running or cannot
    be reached, raise ProviderNetworkError.
    """
    base = (config.api_base.rstrip("/")) if config.api_base else "http://localhost:11434"
    url = f"{base}/api/tags"
    headers = {"Accept": "application/json"}

    try:
        logger.debug("ollama.list_models.request", url=url)
        with httpx.Client(timeout=5) as client:
            resp = client.get(url, headers=headers)
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:  # type: ignore[attr-defined]
        raise ProviderNetworkError(
            "Ollama daemon not reachable. Is it running on localhost:11434?"
        ) from e
    except Exception as e:  # pragma: no cover - unexpected
        raise ProviderError(f"Ollama request failed: {e}") from e

    if resp.status_code >= 500:
        raise ProviderNetworkError(f"Ollama server error: HTTP {resp.status_code}")
    if resp.status_code >= 400:
        raise ProviderProtocolError(
            f"Ollama protocol error: HTTP {resp.status_code} body={resp.text[:200]}"
        )

    try:
        data = resp.json()
        items = data.get("models", [])
        if not isinstance(items, list):
            raise ValueError("Response 'models' is not a list")
    except Exception as e:
        raise ProviderProtocolError(f"Ollama response parse error: {e}") from e

    models = []
    for m in items:
        # Prefer the human-facing tag name; fall back to internal model string
        mid = m.get("name") or m.get("model")
        if not mid:
            continue
        models.append(
            Model(id=str(mid), provider=ProviderType.OLLAMA, name=str(mid))
        )
    return models


def ollama_turbo_list_models(config: ProviderConfig) -> List[Model]:
    """List models from cloud Ollama Turbo service using /api/tags.

    - Requires API key for cloud access via OLLAMA_TURBO_API_KEY env var
    - Uses https://ollama.com as default base URL
    - Same API spec as local Ollama but with authentication
    """
    base = (config.api_base.rstrip("/")) if config.api_base else "https://ollama.com"
    url = f"{base}/api/tags"
    headers = {"Accept": "application/json", **_bearer_headers(config.api_key)}

    if not config.api_key:
        raise ProviderAuthError("Missing Ollama Turbo API key (set OLLAMA_TURBO_API_KEY)")

    try:
        logger.debug("ollama_turbo.list_models.request", url=url)
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, headers=headers)
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:  # type: ignore[attr-defined]
        raise ProviderNetworkError(f"Ollama Turbo network error: {e}") from e
    except Exception as e:  # pragma: no cover - unexpected
        raise ProviderError(f"Ollama Turbo request failed: {e}") from e

    if resp.status_code in (401, 403):
        raise ProviderAuthError(f"Ollama Turbo auth error: HTTP {resp.status_code}")
    if resp.status_code >= 500:
        raise ProviderNetworkError(f"Ollama Turbo server error: HTTP {resp.status_code}")
    if resp.status_code >= 400:
        raise ProviderProtocolError(
            f"Ollama Turbo protocol error: HTTP {resp.status_code} body={resp.text[:200]}"
        )

    try:
        data = resp.json()
        items = data.get("models", [])
        if not isinstance(items, list):
            raise ValueError("Response 'models' is not a list")
    except Exception as e:
        raise ProviderProtocolError(f"Ollama Turbo response parse error: {e}") from e

    models = []
    for m in items:
        # Prefer the human-facing tag name; fall back to internal model string
        mid = m.get("name") or m.get("model")
        if not mid:
            continue
        models.append(
            Model(id=str(mid), provider=ProviderType.OLLAMA_TURBO, name=str(mid))
        )
    return models


# =========================
# Facade (B5)
# =========================


def list_models(provider: ProviderType, config: ProviderConfig) -> List[Model]:
    """Facade that routes to per-provider implementations with unified errors (B5).

    Adds lightweight in-process caching to avoid repeated network calls during
    a single session. Cache key excludes secrets: uses provider, api_base and a
    short hash suffix of the api_key if present.
    """
    logger.info("providers.list_models", provider=provider.value)

    # Build cache key (provider, api_base, api_key_hash8)
    import hashlib
    base = (config.api_base or "").strip()
    key_hash = (
        hashlib.sha256(config.api_key.encode()).hexdigest()[:8]
        if config.api_key else ""
    )
    cache_key = (provider.value, base, key_hash)

    if cache_key in _MODELS_CACHE:
        return _MODELS_CACHE[cache_key]

    try:
        if provider == ProviderType.OPENAI:
            models = openai_list_models(config)
        elif provider == ProviderType.OPENROUTER:
            models = openrouter_list_models(config)
        elif provider == ProviderType.OLLAMA:
            models = ollama_list_models(config)
        elif provider == ProviderType.OLLAMA_TURBO:
            models = ollama_turbo_list_models(config)
        else:
            raise ProviderProtocolError(f"Unsupported provider: {provider}")

        # Cache successful results
        _MODELS_CACHE[cache_key] = models
        return models

    except ProviderError:
        raise
    except httpx.HTTPError as e:
        raise ProviderNetworkError(str(e)) from e
    except Exception as e:  # pragma: no cover - unexpected
        raise ProviderError(str(e)) from e


__all__ = [
    "ProviderError",
    "ProviderAuthError",
    "ProviderNetworkError",
    "ProviderProtocolError",
    "Model",
    "ProviderType",
    "ProviderConfig",
    "openai_list_models",
    "openrouter_list_models",
    "ollama_list_models",
    "ollama_turbo_list_models",
    "list_models",
]
