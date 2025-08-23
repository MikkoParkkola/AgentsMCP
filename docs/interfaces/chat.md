# Interfaces: Chat Runtime & Commands

Source: `src/agentsmcp/commands/chat.py`

## Core Session Types

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
from .providers import ProviderType

@dataclass
class Message:
    role: str      # "system" | "user" | "assistant" | "tool"
    content: str
    meta: Optional[Dict[str, Any]] = None

@dataclass
class SessionSettings:
    provider: ProviderType
    model: str
    temperature: float = 0.7
    api_base: Optional[str] = None
    context_budget_pct: Optional[int] = 70  # percent of model context
    max_tokens_out: Optional[int] = None
    stream: bool = False

class ChatSession:
    messages: List[Message]
    settings: SessionSettings
    # mutations
    def set_provider(self, p: ProviderType) -> None: ...
    def set_model(self, model: str) -> None: ...
    def set_api_base(self, url: Optional[str]) -> None: ...
    def set_temperature(self, t: float) -> None: ...
    def set_context_budget_pct(self, pct: Optional[int]) -> None: ...
    def set_stream(self, enabled: bool) -> None: ...
```

## Command Registration

```python
CommandHandler = Callable[[ChatSession, str], None]

# Register a slash command (without the leading slash)

def register_command(name: str, handler: CommandHandler, help: str) -> None: ...
```

## Built-in Commands Interfaces

```python
# /models [provider?]

def cmd_models(session: ChatSession, args: str) -> None:
    """List available models; selecting one sets the session model (and provider if specified)."""

# /provider [name?]

def cmd_provider(session: ChatSession, args: str) -> None:
    """Show or set provider; if missing key, prompt and optionally persist."""

# /api_base [url|clear]

def cmd_api_base(session: ChatSession, args: str) -> None: ...

# /model [id]

def cmd_model(session: ChatSession, args: str) -> None: ...

# /context [percent|off]

def cmd_context(session: ChatSession, args: str) -> None: ...

# /stream [on|off]

def cmd_stream(session: ChatSession, args: str) -> None: ...
```

## Rendering / UI Hooks

- Model list UI exposes: `on_filter(text: str) -> List[Model]`, `on_select(model_id: str) -> None`.
- Errors surface as toast/banner with actionable guidance; do not crash session.

