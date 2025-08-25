"""Retry decorator with exponential backoff for AgentsMCP discovery system.

Provides production-ready retry logic with exponential backoff, jitter,
and support for both synchronous and asynchronous operations.
"""

from __future__ import annotations

import time
import random
import asyncio
import functools
import logging
from typing import Callable, TypeVar, ParamSpec, Any, Awaitable, Union, cast

from .exceptions import NetworkError

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger("agentsmcp.retry")


def exponential_backoff(
    *,
    attempts: int = 5,
    base_delay: float = 0.1,
    max_delay: float = 5.0,
    jitter: bool = True,
    retry_on: tuple[type[BaseException], ...] = (NetworkError,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that retries a function with exponential back-off.

    Parameters:
    - attempts: Maximum number of attempts (including the first try)
    - base_delay: Initial delay in seconds between retries
    - max_delay: Maximum delay in seconds (caps exponential growth)
    - jitter: Whether to add random variation to delay times
    - retry_on: Tuple of exception types that should trigger a retry

    Example:
    >>> @exponential_backoff(attempts=3, base_delay=0.2)
    ... def fragile_operation():
    ...     # May raise NetworkError
    ...     pass
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            delay = base_delay
            last_exception = None
            
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)          # type: ignore[no-any-return]
                except retry_on as exc:
                    last_exception = exc
                    if attempt == attempts:
                        logger.error(
                            "Retry exhausted after %d attempts for %s: %s",
                            attempts,
                            func.__name__,
                            exc,
                        )
                        raise
                    
                    actual_delay = delay
                    if jitter:
                        actual_delay *= random.uniform(0.8, 1.2)
                    
                    logger.warning(
                        "Retry %d/%d for %s after %.2fs – %s",
                        attempt,
                        attempts,
                        func.__name__,
                        actual_delay,
                        exc,
                    )
                    
                    _sleep(actual_delay)
                    delay = min(delay * 2, max_delay)
                except Exception:
                    # Non-retryable exception - re-raise immediately
                    raise
            
            # This should never be reached, but handle it gracefully
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Retry logic failed unexpectedly for {func.__name__}")

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            delay = base_delay
            last_exception = None
            
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)   # type: ignore[no-any-return]
                except retry_on as exc:
                    last_exception = exc
                    if attempt == attempts:
                        logger.error(
                            "Async retry exhausted after %d attempts for %s: %s",
                            attempts,
                            func.__name__,
                            exc,
                        )
                        raise
                    
                    actual_delay = delay
                    if jitter:
                        actual_delay *= random.uniform(0.8, 1.2)
                    
                    logger.warning(
                        "Async retry %d/%d for %s after %.2fs – %s",
                        attempt,
                        attempts,
                        func.__name__,
                        actual_delay,
                        exc,
                    )
                    
                    await _async_sleep(actual_delay)
                    delay = min(delay * 2, max_delay)
                except Exception:
                    # Non-retryable exception - re-raise immediately
                    raise
            
            # This should never be reached, but handle it gracefully
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Async retry logic failed unexpectedly for {func.__name__}")

        # Decide which wrapper to return based on the callable type
        if asyncio.iscoroutinefunction(func):
            return cast(Callable[P, R], async_wrapper)
        else:
            return cast(Callable[P, R], sync_wrapper)

    return decorator


def _sleep(seconds: float) -> None:
    """Blocking sleep (used only in sync wrapper)."""
    time.sleep(seconds)


async def _async_sleep(seconds: float) -> None:
    """Non-blocking sleep for async wrapper."""
    await asyncio.sleep(seconds)


def linear_backoff(
    *,
    attempts: int = 3,
    delay: float = 1.0,
    jitter: bool = True,
    retry_on: tuple[type[BaseException], ...] = (NetworkError,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that retries a function with linear (fixed) delay.
    
    Similar to exponential_backoff but uses a constant delay between attempts.
    Useful for cases where exponential growth is not desired.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception = None
            
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)          # type: ignore[no-any-return]
                except retry_on as exc:
                    last_exception = exc
                    if attempt == attempts:
                        logger.error(
                            "Linear retry exhausted after %d attempts for %s: %s",
                            attempts,
                            func.__name__,
                            exc,
                        )
                        raise
                    
                    actual_delay = delay
                    if jitter:
                        actual_delay *= random.uniform(0.8, 1.2)
                    
                    logger.warning(
                        "Linear retry %d/%d for %s after %.2fs – %s",
                        attempt,
                        attempts,
                        func.__name__,
                        actual_delay,
                        exc,
                    )
                    
                    _sleep(actual_delay)
                except Exception:
                    # Non-retryable exception - re-raise immediately
                    raise
            
            # This should never be reached, but handle it gracefully
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Linear retry logic failed unexpectedly for {func.__name__}")

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception = None
            
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)   # type: ignore[no-any-return]
                except retry_on as exc:
                    last_exception = exc
                    if attempt == attempts:
                        logger.error(
                            "Async linear retry exhausted after %d attempts for %s: %s",
                            attempts,
                            func.__name__,
                            exc,
                        )
                        raise
                    
                    actual_delay = delay
                    if jitter:
                        actual_delay *= random.uniform(0.8, 1.2)
                    
                    logger.warning(
                        "Async linear retry %d/%d for %s after %.2fs – %s",
                        attempt,
                        attempts,
                        func.__name__,
                        actual_delay,
                        exc,
                    )
                    
                    await _async_sleep(actual_delay)
                except Exception:
                    # Non-retryable exception - re-raise immediately  
                    raise
            
            # This should never be reached, but handle it gracefully
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Async linear retry logic failed unexpectedly for {func.__name__}")

        # Decide which wrapper to return based on the callable type
        if asyncio.iscoroutinefunction(func):
            return cast(Callable[P, R], async_wrapper)
        else:
            return cast(Callable[P, R], sync_wrapper)

    return decorator