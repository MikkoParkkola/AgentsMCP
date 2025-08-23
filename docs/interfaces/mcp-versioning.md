# Interfaces: MCP Version Negotiation & Tool Down-Conversion

Source: `src/agentsmcp/mcp/server.py`

## Types

```python
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    raw: Dict[str, Any]         # full provider schema (for latest clients)
```

## Functions

```python
LATEST_VERSION = "1.0.0"  # example placeholder

def negotiate_version(client_version: str | None) -> str:
    """Return a server-supported version given a client requested version.
    If None, assume latest. Must never raise; default to LATEST_VERSION.
    """

def downconvert_tools(tools: List[Tool], client_version: str) -> List[Tool]:
    """Strip/reshape tool fields for older clients.
    Rules:
      - Keep: name, description, parameters
      - Drop/flatten fields unknown to target version
    """
```

## Behavior

- On connection, log: client version, negotiated version.
- For latest clients: pass-through `raw` payloads.
- For older clients: serve stripped `Tool` objects per `downconvert_tools` result.

