# Interfaces: Context Window Management

Source: `src/agentsmcp/commands/chat.py` (runtime trimming)

## Types

```python
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class Tokenizer:
    encode: Callable[[str], List[int]]

@dataclass
class TrimStats:
    original_messages: int
    trimmed_messages: int
    original_tokens: int
    kept_tokens: int
```

## Functions

```python
from .chat import Message

def estimate_tokens(msg: Message, tokenizer: Tokenizer) -> int:
    """Compute approximate token count for a message."""

def trim_history(messages: List[Message], budget_tokens: int, tokenizer: Tokenizer) -> List[Message]:
    """Return a pruned message list that fits within budget.
    Strategy: drop oldest non-system messages first; always keep latest user + assistant turns.
    Deterministic and side-effect free.
    """
```

## Behavior

- Use a conservative over-estimate to avoid overflow.
- Always preserve the most recent exchange; preserve system message if present.
- Provide optional stats in logs for observability.

