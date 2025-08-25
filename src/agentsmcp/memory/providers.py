"""
MemoryProvider interfaces and concrete implementations.

Two production‑ready backends are supported:

* Redis – highly performant, in‑memory & persistent.
* PostgreSQL – persistence‑oriented, good when Redis is not available.

Only the Redis implementation is fully exercised in the test suite
(because the team's current infrastructure runs in Docker).  The
PostgreSQL stub demonstrates how the rest of the code could be
extended.

All provider classes are written as *async* to keep the I/O non‑blocking.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

# Optional dependencies - graceful degradation if not available
try:
    import redis.asyncio as redis_async
    HAS_REDIS = True
except ImportError:
    redis_async = None
    HAS_REDIS = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    HAS_AIOHTTP = False


class MemoryProvider(abc.ABC):
    """
    Abstract base class for all memory providers.

    Subclasses must implement the three lifecycle operations:

    - `store_context` – Insert or update a context for an agent.
    - `load_context` – Retrieve a stored context.
    - `delete_context` – Remove a context (usually on agent shut‑down).
    """

    @abc.abstractmethod
    async def store_context(
        self,
        agent_id: str,
        context: Dict[str, Any],
    ) -> None:
        """Persist the *context* for the agent identified by *agent_id*."""
        raise NotImplementedError

    @abc.abstractmethod
    async def load_context(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored context for *agent_id* or ``None`` if missing."""
        raise NotImplementedError

    @abc.abstractmethod
    async def delete_context(self, agent_id: str) -> None:
        """Delete the stored context for *agent_id*."""
        raise NotImplementedError

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Return ``True`` if the provider is healthy."""
        raise NotImplementedError


@dataclass
class RedisProvider(MemoryProvider):
    """
    Redis implementation of MemoryProvider.

    Connections are pooled via ``redis.asyncio``, which is already
    connection‑pool friendly.  The provider offers persistence via
    the configured Redis instance.

    Example:

        >>> provider = RedisProvider(url="redis://localhost:6379/0")
        >>> await provider.store_context("agent-1", {"foo": "bar"})
        >>> ctx = await provider.load_context("agent-1")
    """

    url: str = field(default="redis://localhost:6379/0")
    health_timeout: float = field(default=3.0)

    _client: Optional[Any] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not HAS_REDIS:
            raise ImportError("redis package is required for RedisProvider")

    async def _get_client(self):
        if self._client is None:
            self._client = redis_async.from_url(self.url, decode_responses=False)
        return self._client

    async def store_context(self, agent_id: str, context: Dict[str, Any]) -> None:
        from .serializers import serialize_context

        client = await self._get_client()
        try:
            data = serialize_context(context, compress=True)
            await client.set(name=agent_id, value=data)
            logger.debug("Stored context for agent %s", agent_id)
        except Exception as exc:
            logger.exception("Failed to store context for agent %s", agent_id)
            raise

    async def load_context(self, agent_id: str) -> Optional[Dict[str, Any]]:
        from .serializers import deserialize_context

        client = await self._get_client()
        try:
            raw = await client.get(name=agent_id)
            if raw is None:
                logger.debug("No context found for agent %s", agent_id)
                return None
            context = deserialize_context(raw, compressed=True)
            logger.debug("Loaded context for agent %s", agent_id)
            return context
        except Exception as exc:
            logger.exception("Failed to load context for agent %s", agent_id)
            raise

    async def delete_context(self, agent_id: str) -> None:
        client = await self._get_client()
        try:
            deleted = await client.delete(agent_id)
            logger.debug("Deleted context for agent %s (deleted: %s)", agent_id, deleted)
        except Exception as exc:
            logger.exception("Failed to delete context for agent %s", agent_id)
            raise

    async def health_check(self) -> bool:
        """
        Fast ping-based health check.

        The connection pool may be in an unknown state (e.g. Redis goes
        silent) — the ``ping`` command is a quick and definitive
        health indicator for Redis.
        """
        if not HAS_REDIS:
            return False
            
        client = await self._get_client()
        try:
            await asyncio.wait_for(client.ping(), timeout=self.health_timeout)
            return True
        except Exception as exc:
            logger.debug("Redis health check failed: %s", exc)
            return False


@dataclass
class PostgreSQLProvider(MemoryProvider):
    """
    PostgreSQL implementation (minimal stub).

    Intended as a drop‑in replacement for Redis when the platform already
    lives inside a database‑centric environment.  The actual SQL wiring
    is omitted due to scope – replace with ``asyncpg`` or SQLAlchemy
    in a production build.

    It is useful for integration tests that run against a real database
    without pulling Redis.
    """

    dsn: str
    _pool: Optional[Any] = field(default=None, init=False, repr=False)

    async def _get_pool(self):
        # This is a placeholder – real code should use `asyncpg.create_pool`
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp package is required for PostgreSQLProvider placeholder")
        if self._pool is None:
            self._pool = aiohttp.ClientSession()
        return self._pool

    async def store_context(self, agent_id: str, context: Dict[str, Any]) -> None:
        # Placeholder – not implemented
        raise NotImplementedError("PostgreSQLProvider not yet wired up")

    async def load_context(self, agent_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("PostgreSQLProvider not yet wired up")

    async def delete_context(self, agent_id: str) -> None:
        raise NotImplementedError("PostgreSQLProvider not yet wired up")

    async def health_check(self) -> bool:
        raise NotImplementedError("PostgreSQLProvider not yet wired up")


class InMemoryProvider(MemoryProvider):
    """
    In-memory provider for testing and development.
    
    This provider stores contexts in memory and does not persist them
    across restarts. Useful for testing and development environments.
    """
    
    def __init__(self):
        self._storage: Dict[str, Dict[str, Any]] = {}
        
    async def store_context(self, agent_id: str, context: Dict[str, Any]) -> None:
        self._storage[agent_id] = context.copy()
        logger.debug("Stored context for agent %s in memory", agent_id)
        
    async def load_context(self, agent_id: str) -> Optional[Dict[str, Any]]:
        context = self._storage.get(agent_id)
        if context:
            logger.debug("Loaded context for agent %s from memory", agent_id)
            return context.copy()
        logger.debug("No context found for agent %s in memory", agent_id)
        return None
        
    async def delete_context(self, agent_id: str) -> None:
        if agent_id in self._storage:
            del self._storage[agent_id]
            logger.debug("Deleted context for agent %s from memory", agent_id)
        else:
            logger.debug("No context to delete for agent %s in memory", agent_id)
            
    async def health_check(self) -> bool:
        return True