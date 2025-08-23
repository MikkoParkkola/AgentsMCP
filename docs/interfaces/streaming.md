# Interfaces: Streaming Responses

Source: `src/agentsmcp/commands/chat.py` (runtime) and provider clients.

## Types

```python
from dataclasses import dataclass
from typing import Iterator, Optional, Dict, Any, List

@dataclass
class Chunk:
    delta_text: str
    finish_reason: Optional[str] = None  # e.g., "stop", "length"
    provider_meta: Optional[Dict[str, Any]] = None

@dataclass
class ChatRequest:
    provider: ProviderType
    model: str
    messages: List[Message]
    temperature: float
    api_base: Optional[str]
```

## Functions

```python
# Unified streaming interface (provider-agnostic)

def generate_stream(req: ChatRequest) -> Iterator[Chunk]:
    """Yield chunks as they arrive; final chunk carries finish_reason."""
```

## Behavior

- `/stream on|off` toggles `SessionSettings.stream`.
- When streaming is on, UI renders tokens incrementally; buffer and finalize into a single message on completion.
- When streaming off, fall back to single-shot generation.

