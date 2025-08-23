# Interfaces: Delegation Workflows (Spec)

Sources: `src/agentsmcp/tools/mcp_tool.py` (existing), future `src/agentsmcp/tools/delegate.py`

## Types

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
from .providers import ProviderType

@dataclass
class DelegationTask:
    title: str
    description: str
    provider: Optional[ProviderType] = None
    model: Optional[str] = None
    params: Optional[Dict[str, Any]] = None  # task-specific payload

@dataclass
class DelegationResult:
    status: str            # "ok" | "error" | "cancelled"
    output: Optional[str]  # human-readable summary
    data: Optional[Dict[str, Any]] = None
```

## Command Pattern

```python
# /delegate <title>\n<free-form description>

def cmd_delegate(session: ChatSession, args: str) -> None:
    """Parses a task, optionally selects provider/model, asks for confirmation, then executes."""
```

## MCP Delegation

```python
# Existing generic tool wrapper

def mcp_call(server: str, tool: str, params: Dict[str, Any]) -> Dict[str, Any]: ...

# Proposed convenience wrapper

def delegate_to_mcp(server: str, tool: str, task: DelegationTask) -> DelegationResult: ...
```

## Guardrails

- Allowlist MCP servers per agent; refuse calls outside the allowlist.
- Display a confirmation prompt before executing external actions.
- Emit an audit log entry with timestamp, tool, and parameters.

