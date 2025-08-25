"""
agentsmcp.memory
~~~~~~~~~~~~~~~~

Central public API for the agents‑mcp memory subsystem.

The package exposes a simple facade that the rest of the application can use:

* ``MemoryProvider`` – abstract base class for any persistence backend.
* ``RedisProvider`` – fully supported Redis implementation (default).
* ``PostgreSQLProvider`` – optional PostgreSQL implementation.
* ``serialize_context`` / ``deserialize_context`` – JSON‑based context I/O helpers.
* ``get_memory_health`` – expose a health snapshot (used by the web UI).

Any additional providers or utilities should be added here to keep the public
API stable and easy to discover for downstream modules.
"""

from .providers import MemoryProvider, RedisProvider, PostgreSQLProvider, InMemoryProvider
from .serializers import serialize_context, deserialize_context
from .health import get_memory_health

__all__ = [
    "MemoryProvider",
    "RedisProvider",
    "PostgreSQLProvider", 
    "InMemoryProvider",
    "serialize_context",
    "deserialize_context",
    "get_memory_health",
]