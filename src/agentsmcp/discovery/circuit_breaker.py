"""Circuit breaker implementation for AgentsMCP discovery system.

Provides production-ready circuit breaker with thread safety, state tracking,
and support for both synchronous and asynchronous operations.
"""

from __future__ import annotations

import time
import threading
import logging
from collections import deque
from typing import Callable, TypeVar, Generic, Optional, Deque, Any, Awaitable, Union

from .exceptions import ServiceUnavailableError, NetworkError

T = TypeVar("T")
R = TypeVar("R")

logger = logging.getLogger("agentsmcp.circuit_breaker")


def _run_callable(fn: Callable[..., R], *args: Any, **kwargs: Any) -> R:
    """Helper for synchronous callables."""
    return fn(*args, **kwargs)


async def _run_async_callable(fn: Callable[..., Awaitable[R]], *args: Any, **kwargs: Any) -> R:
    """Helper for asynchronous callables."""
    return await fn(*args, **kwargs)


class CircuitBreaker(Generic[T]):
    """Thread-safe circuit breaker implementing the classic three-state model.
    
    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Circuit is tripped, calls are rejected immediately  
    - HALF_OPEN: Testing if service has recovered
    
    Parameters:
    - failure_threshold: Number of consecutive failures required to open circuit
    - recovery_timeout: Seconds to stay open before allowing a trial request
    - expected_exceptions: Exception types that count as failures
    - name: Identifier used in logs and metrics
    - half_open_success_threshold: Successful calls needed to close from half-open
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exceptions: tuple[type[BaseException], ...] = (NetworkError, ),
        name: str = "circuit_breaker",
        half_open_success_threshold: int = 1,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._expected_exceptions = expected_exceptions
        self._name = name
        self._half_open_success_threshold = half_open_success_threshold

        # State tracking
        self._lock = threading.RLock()
        self._state: str = "CLOSED"                 # CLOSED | OPEN | HALF_OPEN
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None
        self._half_open_successes = 0

        # Metrics - sliding window of recent failure timestamps
        self._failure_timestamps: Deque[float] = deque(maxlen=100)
        self._last_success_latency_ms: Optional[float] = None

    def call(self, fn: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """Synchronous wrapper. If the breaker is OPEN, raises ServiceUnavailableError."""
        return self._execute(_run_callable, fn, *args, **kwargs)

    async def acall(self, fn: Callable[..., Awaitable[R]], *args: Any, **kwargs: Any) -> R:
        """Asynchronous wrapper."""
        return await self._execute(_run_async_callable, fn, *args, **kwargs)   # type: ignore[arg-type]

    def _execute(
        self,
        runner: Callable[[Callable[..., Any], Any, Any], Any],
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Core execution logic with state management."""
        with self._lock:
            self._maybe_reset()
            if self._state == "OPEN":
                logger.warning(
                    "%s is OPEN – rejecting call to %s",
                    self._name,
                    getattr(fn, "__name__", repr(fn)),
                )
                raise ServiceUnavailableError(
                    f"Circuit breaker {self._name} is OPEN",
                    payload={"function": getattr(fn, "__name__", str(fn))},
                )

        # Execute the function outside the lock to avoid blocking other threads
        start_time = time.time()
        try:
            result = runner(fn, *args, **kwargs)
        except Exception as exc:
            # Decide whether the exception is a failure that should trip the breaker
            if isinstance(exc, self._expected_exceptions):
                self._record_failure(exc)
            raise   # re-raise the original error

        # Success path - record latency and possibly transition from HALF_OPEN → CLOSED
        elapsed_ms = (time.time() - start_time) * 1000
        self._last_success_latency_ms = elapsed_ms
        self._record_success()
        return result

    def _maybe_reset(self) -> None:
        """Move from OPEN → HALF_OPEN if timeout elapsed."""
        if self._state == "OPEN" and self._opened_at is not None:
            elapsed = time.time() - self._opened_at
            if elapsed >= self._recovery_timeout:
                logger.info("%s timeout elapsed (%.2fs); moving to HALF_OPEN", self._name, elapsed)
                self._state = "HALF_OPEN"
                self._half_open_successes = 0

    def _record_failure(self, exc: BaseException) -> None:
        """Record a failure and potentially trip the circuit."""
        with self._lock:
            self._consecutive_failures += 1
            self._failure_timestamps.append(time.time())
            logger.error(
                "%s failure %d/%d – %s",
                self._name,
                self._consecutive_failures,
                self._failure_threshold,
                exc,
            )
            if self._consecutive_failures >= self._failure_threshold:
                self._trip()

    def _record_success(self) -> None:
        """Record a success and potentially close the circuit."""
        with self._lock:
            if self._state == "HALF_OPEN":
                self._half_open_successes += 1
                if self._half_open_successes >= self._half_open_success_threshold:
                    self._reset()
            else:
                # In CLOSED state, clear the failure counter on any success
                self._consecutive_failures = 0

    def _trip(self) -> None:
        """Trip the circuit to OPEN state."""
        self._state = "OPEN"
        self._opened_at = time.time()
        logger.warning("%s TRIPPED – moving to OPEN state", self._name)

    def _reset(self) -> None:
        """Reset the circuit to CLOSED state."""
        logger.info("%s reset – moving to CLOSED", self._name)
        self._state = "CLOSED"
        self._opened_at = None
        self._consecutive_failures = 0
        self._half_open_successes = 0

    def status(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the breaker state."""
        with self._lock:
            return {
                "name": self._name,
                "state": self._state,
                "consecutive_failures": self._consecutive_failures,
                "opened_at": self._opened_at,
                "failure_rate_last_min": self._failure_rate(seconds=60),
                "last_success_latency_ms": self._last_success_latency_ms,
                "failure_threshold": self._failure_threshold,
                "recovery_timeout": self._recovery_timeout,
            }

    def _failure_rate(self, *, seconds: int = 60) -> float:
        """Calculate failure count per second over the recent window."""
        if seconds <= 0:
            return 0.0
        
        cutoff = time.time() - seconds
        recent_failures = [t for t in self._failure_timestamps if t >= cutoff]
        return len(recent_failures) / seconds

    def force_open(self) -> None:
        """Force the circuit open (useful for testing or manual intervention)."""
        with self._lock:
            logger.warning("%s forced to OPEN state", self._name)
            self._state = "OPEN"
            self._opened_at = time.time()
    
    def force_closed(self) -> None:
        """Force the circuit closed (useful for testing or manual recovery)."""
        with self._lock:
            logger.info("%s forced to CLOSED state", self._name)
            self._reset()