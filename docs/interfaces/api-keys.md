# Interfaces: Provider API Keys & Validation

Source: `src/agentsmcp/providers.py` (validation helpers) and `src/agentsmcp/commands/chat.py` (UX prompts)

## Types

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ValidationStatus(str, Enum):
    OK = "ok"
    MISSING = "missing"
    INVALID = "invalid"
    NETWORK = "network"

@dataclass
class ValidationResult:
    status: ValidationStatus
    message: Optional[str] = None  # user-facing
    detail: Optional[str] = None   # logs/debug
```

## Functions

```python
from .providers import ProviderType, ProviderConfig

def validate_provider_config(provider: ProviderType, cfg: ProviderConfig) -> ValidationResult:
    """Lightweight probe: e.g., GET /models (OpenAI/OpenRouter) or GET /api/tags (Ollama).
    Returns a ValidationResult; never raises.
    """

def prompt_for_api_key(provider: ProviderType) -> Optional[str]:
    """Interactive prompt in CLI; returns key or None if cancelled."""

def persist_provider_api_key(provider: ProviderType, key: str, config_path: str) -> None:
    """Writes key to agentsmcp.yaml under providers.{provider}.api_key.
    Non-destructive merge; creates section if missing.
    """
```

## Behavior

- Missing/invalid keys must not crash the chat; show actionable banner and allow switching provider.
- Persist only with explicit confirmation; mask key in UI.

