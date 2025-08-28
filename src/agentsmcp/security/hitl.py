"""
Decorator-based Human-In-The-Loop (HITL) enforcement for critical operations.

Usage:

    from agentsmcp.security.hitl import hitl_required

    @hitl_required(operation="delete_project", risk_level="high")
    async def delete_project(project_id: str):
        ...

Configuration:
- Which operations require approval is controlled via env var
  `AGENTSMCP_HITL_REQUIRED_OPS` (comma-separated) or by passing
  `require=True/False` to the decorator. If `require` is None, uses config.
- Timeout and default decision set in `Approvals` settings.

Security:
- Approval requires a cryptographically signed token issued via the web UI.
- Tokens are short-lived, one-time-use and include a nonce to prevent replay.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import os
import uuid
from typing import Any, Awaitable, Callable, Optional

from .approvals import ApprovalRequest, approvals_manager, get_hitl_settings


def _operation_requires_hitl(operation: str, risk_level: str, explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return explicit
    env = os.getenv("AGENTSMCP_HITL_REQUIRED_OPS", "")
    ops = {s.strip() for s in env.split(",") if s.strip()}
    # If unspecified, require for high risk by default when enabled
    if not ops:
        return get_hitl_settings().enabled and risk_level.lower() == "high"
    return operation in ops


def hitl_required(
    operation: Optional[str] = None,
    *,
    risk_level: str = "high",
    priority: int = 0,
    require: Optional[bool] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Awaitable[Any]]]:
    """Decorator to enforce HITL approval before running the function.

    Works with sync and async functions; always exposed as async wrapper.
    The wrapped function will only execute after an approval decision.
    On rejection, raises `PermissionError`. On timeout, follows default action.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
        op_name = operation or func.__name__

        async def _awaitable_call(*args: Any, **kwargs: Any) -> Any:
            requires = _operation_requires_hitl(op_name, risk_level, require)
            if not requires:
                # Fast path – no overhead for non-critical ops
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return await asyncio.get_running_loop().run_in_executor(None, functools.partial(func, *args, **kwargs))

            # Create approval request and wait for decision
            request_id = str(uuid.uuid4())
            user = kwargs.get("_current_user", "system")  # optional context
            req = ApprovalRequest(
                id=request_id,
                operation=op_name,
                risk_level=risk_level,
                requested_by=str(user),
                payload={"args": repr(args), "kwargs": {k: ("***" if k.lower() in {"password", "token", "secret"} else repr(v)) for k, v in kwargs.items() if not k.startswith("_")}},
                priority=priority,
            )
            await approvals_manager.submit(req)

            decision = await approvals_manager.wait_for_decision(request_id)

            if decision.decision != "approved":
                raise PermissionError(f"HITL decision denied for {op_name}: {decision.decision}")

            # Verify token to guard against tampering/replay
            if not decision.approval_token:
                raise PermissionError("missing_approval_token")
            approvals_manager.verify_token(decision.approval_token)

            # Execute operation
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return await asyncio.get_running_loop().run_in_executor(None, functools.partial(func, *args, **kwargs))

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:  # public wrapper
            return await _awaitable_call(*args, **kwargs)

        # Preserve original signature for FastAPI/DI compatibility
        try:
            async_wrapper.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
        except Exception:
            pass

        return async_wrapper

    return decorator


# Convenience helper to initialize/stop the manager from app lifecycle
async def start_hitl() -> None:
    await approvals_manager.start()


async def stop_hitl() -> None:
    await approvals_manager.stop()
