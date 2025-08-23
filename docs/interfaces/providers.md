# Interfaces: Providers & Models

Source: `src/agentsmcp/providers.py`

## Shared Types

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any

class ProviderType(str, Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"

@dataclass
class ProviderConfig:
    api_key: Optional[str]  # may be None for Ollama localhost
    api_base: Optional[str] # e.g. https://api.openai.com/v1, http://localhost:11434

@dataclass
class Model:
    id: str                 # canonical model id returned by provider
    provider: ProviderType
    display_name: Optional[str] = None
    context_window: Optional[int] = None
    aliases: Optional[List[str]] = None
    raw: Optional[Dict[str, Any]] = None  # original payload for debugging
```

## Provider Adapters

```python
# Top-level facade

def list_models(provider: ProviderType, config: ProviderConfig) -> List[Model]:
    """Return available models for provider.
    Raises:
      ProviderAuthError: invalid/missing key when required
      ProviderNetworkError: host unreachable / non-2xx
      ProviderProtocolError: unexpected payload shape
    """

# Provider-specific adapters

def openai_list_models(config: ProviderConfig) -> List[Model]:
    """GET {api_base or https://api.openai.com/v1}/models with Bearer token.
    Filters/normalizes into Model[].
    """

def openrouter_list_models(config: ProviderConfig) -> List[Model]:
    """GET {api_base or https://openrouter.ai/api}/models with Bearer token.
    Normalizes into Model[].
    """

def ollama_list_models(config: ProviderConfig) -> List[Model]:
    """GET {api_base or http://localhost:11434}/api/tags with no auth for localhost.
    Normalizes into Model[].
    """
```

## Errors

```python
class ProviderError(Exception): ...
class ProviderAuthError(ProviderError): ...
class ProviderNetworkError(ProviderError): ...
class ProviderProtocolError(ProviderError): ...
```

## Integration Hook

```python
# In Agent base (reference)

def discover_models(self, provider: ProviderType) -> List[Model]:
    """Uses Config.providers[provider] to invoke list_models()."""
```

Notes:
- Respect configured `api_base` if provided; otherwise use provider defaults.
- Do not hard-fail when Ollama not running; return clear ProviderNetworkError(message).

