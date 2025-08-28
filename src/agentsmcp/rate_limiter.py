"""
Lightweight, env-gated rate limiter and circuit breaker for provider calls.

Defaults to NO-OP unless AGENTSMCP_RATE_LIMIT_ENABLED=1 is set in the env.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Dict


class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = max(0.0, rate)  # tokens per second
        self.capacity = max(1, capacity)
        self.tokens = float(capacity)
        self.updated_at = time.perf_counter()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        # Fast path if disabled
        if self.rate <= 0:
            return
        async with self._lock:
            now = time.perf_counter()
            elapsed = now - self.updated_at
            self.updated_at = now
            # Refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            # If not enough tokens, wait until next token is available
            if self.tokens < 1.0:
                # time to next token
                to_wait = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(max(0.0, to_wait))
                # After sleeping, set tokens to 0 (consume below)
                self.tokens = max(0.0, self.tokens + to_wait * self.rate)
            # Consume 1 token
            self.tokens = max(0.0, self.tokens - 1.0)


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_after: float = 10.0):
        self.failure_threshold = max(1, failure_threshold)
        self.reset_after = max(1.0, reset_after)
        self.failures = 0
        self.opened_at: float | None = None
        self._lock = asyncio.Lock()

    async def before(self) -> None:
        async with self._lock:
            if self.opened_at is not None:
                # half-open after reset_after seconds
                if (time.perf_counter() - self.opened_at) < self.reset_after:
                    # still open
                    raise RuntimeError("circuit_open")
                # half-open: allow one trial
                self.opened_at = None
                self.failures = 0

    async def record(self, ok: bool) -> None:
        async with self._lock:
            if ok:
                self.failures = 0
                return
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = time.perf_counter()


class ProviderGuards:
    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._breakers: Dict[str, CircuitBreaker] = {}
        # Defaults; conservative. Override via env if needed.
        # e.g., AGENTSMCP_RATE_LIMIT_OPENAI_RATE=3, AGENTSMCP_RATE_LIMIT_OPENAI_CAP=6
        self._defaults = {
            "openai": (3.0, 6),
            "openrouter": (2.0, 4),
            "ollama": (0.0, 1),  # local; unlimited
            "claude": (2.0, 4),
            "codex": (2.0, 4),
        }

    def _env_rate(self, name: str, rate: float) -> float:
        return float(os.getenv(f"AGENTSMCP_RATE_LIMIT_{name.upper()}_RATE", rate))

    def _env_cap(self, name: str, cap: int) -> int:
        return int(os.getenv(f"AGENTSMCP_RATE_LIMIT_{name.upper()}_CAP", cap))

    def bucket_for(self, provider: str) -> TokenBucket:
        if provider not in self._buckets:
            rate, cap = self._defaults.get(provider, (1.0, 2))
            self._buckets[provider] = TokenBucket(self._env_rate(provider, rate), self._env_cap(provider, cap))
        return self._buckets[provider]

    def breaker_for(self, provider: str) -> CircuitBreaker:
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker()
        return self._breakers[provider]


_guards = ProviderGuards()


async def guard_provider_call(provider: str, coro):
    """Wrap a provider call with (optional) rate-limit + circuit-breaker.

    Disabled unless AGENTSMCP_RATE_LIMIT_ENABLED=1.
    """
    if os.getenv("AGENTSMCP_RATE_LIMIT_ENABLED") != "1":
        return await coro
    bucket = _guards.bucket_for(provider)
    breaker = _guards.breaker_for(provider)
    await breaker.before()
    await bucket.acquire()
    try:
        result = await coro
        await breaker.record(True)
        return result
    except Exception:
        await breaker.record(False)
        raise


__all__ = ["guard_provider_call"]

