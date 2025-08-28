"""
Core approval workflow for Human-In-The-Loop (HITL) operations.

This module provides:
- In-memory approval queue with priority handling
- Async wait/notify for decisions with timeout + default action
- Audit trail writer (JSONL) with safe, append-only writes
- Cryptographic approval token issue/verify helpers (JWT-based)
- Replay protection via nonce + one-time-use token registry

Thread-safety: Uses asyncio primitives; safe for single-process FastAPI.
For multi-process deployments, replace the in-memory store with Redis or DB.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

try:
    from pydantic import BaseModel, Field
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore
except Exception:  # pragma: no cover
    # Minimal fallbacks to avoid import-time crashes in minimal installs
    class BaseModel:  # type: ignore
        pass

    class BaseSettings:  # type: ignore
        pass

    def Field(*args, **kwargs):  # type: ignore
        return None

try:
    from jose import jwt, JWTError
except Exception:  # pragma: no cover
    jwt = None  # type: ignore
    JWTError = Exception  # type: ignore


class HitlSettings(BaseSettings):
    """HITL configuration, loaded from env vars.

    Env prefix: AGENTSMCP_HITL_
    """

    enabled: bool = True
    # Default timeout for approvals in seconds
    approval_timeout_seconds: int = 120
    # Default action on timeout: "deny" or "allow"
    timeout_default_action: str = "deny"
    # JWT secret and algorithm for approval tokens
    approval_token_secret: str = Field("change-hitl-secret", env="APPROVAL_TOKEN_SECRET")
    approval_token_algorithm: str = Field("HS256", env="APPROVAL_TOKEN_ALGORITHM")
    # Audit log path (JSONL)
    audit_log_path: str = Field("./build/hitl_audit.log", env="AUDIT_LOG_PATH")
    # Replay cache max size
    replay_cache_max: int = 2048
    # Escalation: when remaining time ratio falls below this, escalate priority
    escalation_threshold_ratio: float = 0.5
    # How much to increase priority on escalation
    escalation_increment: int = 5

    # Pydantic v2 settings configuration
    model_config = SettingsConfigDict(env_prefix="AGENTSMCP_HITL_", env_file=".env")

# Lazy settings to avoid import-time side effects
_HITL_SETTINGS: HitlSettings | None = None

def get_hitl_settings() -> HitlSettings:
    global _HITL_SETTINGS
    if _HITL_SETTINGS is None:
        _HITL_SETTINGS = HitlSettings()
    return _HITL_SETTINGS


class ApprovalRequest(BaseModel):
    """Represents a pending approval request."""

    id: str
    operation: str
    risk_level: str = "high"
    requested_by: str = "system"
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    priority: int = 0
    # SLA deadline (absolute time) used for priority queuing
    deadline_ts: float = Field(
        default_factory=lambda: time.time() + get_hitl_settings().approval_timeout_seconds
    )


class ApprovalDecision(BaseModel):
    """Decision outcome for an approval request."""

    request_id: str
    decision: str  # "approved" | "rejected" | "timeout"
    decided_by: str = "system"
    decided_at: datetime = Field(default_factory=datetime.utcnow)
    reason: Optional[str] = None
    approval_token: Optional[str] = None  # JWT for verification in executor


@dataclass(order=True)
class _QueueItem:
    sort_key: Tuple[int, float] = field(init=False, repr=False)
    priority: int
    deadline_ts: float
    req: ApprovalRequest = field(compare=False)

    def __post_init__(self):
        # Higher priority first, earlier deadline first; heapq is min-heap so invert priority
        self.sort_key = (-self.priority, self.deadline_ts)


class AuditLogger:
    """Append-only JSONL audit logger with periodic fsync for safety."""

    def __init__(self, path: str):
        self._path = path
        self._lock = asyncio.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    async def write(self, event: Dict[str, Any]) -> None:
        line = json.dumps({"ts": datetime.utcnow().isoformat(), **event}, separators=(",", ":")) + "\n"
        async with self._lock:
            # Use standard IO in thread to avoid blocking loop if needed
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._append_sync, line)

    def _append_sync(self, line: str) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)
            try:
                f.flush()
                os.fsync(f.fileno())  # ensure durability
            except Exception:
                pass


class ApprovalsManager:
    """Manages approval requests, queueing, decisions, tokens and auditing."""

    def __init__(self) -> None:
        self._queue: list[_QueueItem] = []
        self._queue_lock = asyncio.Lock()
        self._waiters: Dict[str, asyncio.Future[ApprovalDecision]] = {}
        self._audit = AuditLogger(get_hitl_settings().audit_log_path)
        self._replay_used_nonces: Dict[str, float] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._escalation_task: Optional[asyncio.Task] = None
        self._escalated: set[str] = set()

    # ------------------------- Public API ---------------------------------
    async def start(self) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._replay_cache_gc())
        if self._escalation_task is None:
            self._escalation_task = asyncio.create_task(self._escalation_watchdog())

    async def stop(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except Exception:
                pass
            self._cleanup_task = None
        if self._escalation_task:
            self._escalation_task.cancel()
            try:
                await self._escalation_task
            except Exception:
                pass
            self._escalation_task = None

    async def submit(self, req: ApprovalRequest) -> None:
        """Submit a new approval request to the priority queue."""
        async with self._queue_lock:
            heapq.heappush(self._queue, _QueueItem(priority=req.priority, deadline_ts=req.deadline_ts, req=req))
        await self._audit.write({"event": "hitl_request_created", "request": req.model_dump()})

    async def list_pending(self, limit: int = 100) -> list[ApprovalRequest]:
        async with self._queue_lock:
            return [item.req for item in sorted(self._queue)[:limit]]

    async def pop_next(self) -> Optional[ApprovalRequest]:
        async with self._queue_lock:
            if not self._queue:
                return None
            return heapq.heappop(self._queue).req

    async def decide(self, request_id: str, decision: str, decided_by: str, reason: Optional[str] = None) -> ApprovalDecision:
        """Record a decision and notify waiter if any. Issues a signed approval token on approval."""
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be 'approved' or 'rejected'")
        token = None
        if decision == "approved":
            token = self.issue_token(request_id=request_id, decided_by=decided_by)
        dec = ApprovalDecision(
            request_id=request_id,
            decision=decision,
            decided_by=decided_by,
            reason=reason,
            approval_token=token,
        )
        await self._audit.write({"event": "hitl_decision", **dec.model_dump()})
        fut = self._waiters.get(request_id)
        if fut and not fut.done():
            fut.set_result(dec)
        # Clear from escalated set to avoid memory growth
        self._escalated.discard(request_id)
        return dec

    async def wait_for_decision(self, request_id: str, timeout: Optional[int] = None) -> ApprovalDecision:
        """Await a decision or timeout with default action."""
        to = timeout or get_hitl_settings().approval_timeout_seconds
        fut = asyncio.get_running_loop().create_future()
        self._waiters[request_id] = fut
        try:
            return await asyncio.wait_for(fut, timeout=to)
        except asyncio.TimeoutError:
            default = get_hitl_settings().timeout_default_action
            dec = ApprovalDecision(
                request_id=request_id,
                decision="approved" if default == "allow" else "rejected",
                decided_by="system-timeout",
                reason=f"timeout_{default}",
                approval_token=(self.issue_token(request_id=request_id, decided_by="system-timeout") if default == "allow" else None),
            )
            await self._audit.write({"event": "hitl_timeout", **dec.model_dump()})
            return dec
        finally:
            self._waiters.pop(request_id, None)

    # ------------------------ Token handling ------------------------------
    def issue_token(self, request_id: str, decided_by: str) -> str:
        """Issue a short-lived approval token with nonce for replay protection."""
        if jwt is None:
            raise RuntimeError("python-jose is required for approval tokens")
        nonce = f"{request_id}:{int(time.time()*1000)}"
        payload = {
            "sub": request_id,
            "type": "approval",
            "nonce": nonce,
            "decided_by": decided_by,
            "exp": int(time.time()) + 300,  # 5 minutes validity
        }
        hs = get_hitl_settings()
        token = jwt.encode(payload, hs.approval_token_secret, algorithm=hs.approval_token_algorithm)
        # Pre-register nonce as unused with expiry
        self._replay_used_nonces.setdefault(nonce, 0.0)
        return token

    def verify_token(self, token: str) -> Dict[str, Any]:
        if jwt is None:
            raise RuntimeError("python-jose is required for approval tokens")
        try:
            hs = get_hitl_settings()
            payload = jwt.decode(token, hs.approval_token_secret, algorithms=[hs.approval_token_algorithm])
        except JWTError as exc:  # pragma: no cover
            raise ValueError("invalid_approval_token") from exc
        if payload.get("type") != "approval":
            raise ValueError("invalid_approval_token_type")
        nonce = payload.get("nonce")
        if not nonce:
            raise ValueError("missing_nonce")
        # One-time use check
        if self._replay_used_nonces.get(nonce, 0) == 1:
            raise ValueError("replay_detected")
        # Mark as used
        self._replay_used_nonces[nonce] = 1
        return payload

    async def _replay_cache_gc(self) -> None:
        """Periodic cleanup to keep replay cache bounded."""
        while True:
            await asyncio.sleep(60)
            if len(self._replay_used_nonces) > get_hitl_settings().replay_cache_max:
                # Remove oldest half (approx) by timestamp embedded in nonce
                items = list(self._replay_used_nonces.items())
                items.sort(key=lambda kv: int(kv[0].split(":")[-1]))
                for k, _ in items[: len(items) // 2]:
                    self._replay_used_nonces.pop(k, None)

    async def _escalation_watchdog(self) -> None:
        """Periodically scan the queue and escalate nearing-deadline items.

        Escalation increases the priority and re-queues the item to the heap,
        and emits an audit event. Each request is escalated at most once.
        """
        interval = 5.0
        while True:
            await asyncio.sleep(interval)
            now = time.time()
            threshold_ratio = max(0.0, min(1.0, get_hitl_settings().escalation_threshold_ratio))
            async with self._queue_lock:
                if not self._queue:
                    continue
                # Rebuild the heap if we escalate any items
                changed = False
                new_heap: list[_QueueItem] = []
                while self._queue:
                    item = heapq.heappop(self._queue)
                    remaining = item.deadline_ts - now
                    total = max(1e-6, item.req.deadline_ts - item.req.created_at.timestamp())
                    remaining_ratio = remaining / total if total > 0 else 0.0
                    if remaining_ratio <= threshold_ratio and item.req.id not in self._escalated:
                        # Escalate once
                        old_priority = item.priority
                        item.priority = old_priority + get_hitl_settings().escalation_increment
                        item.sort_key = (-item.priority, item.deadline_ts)
                        self._escalated.add(item.req.id)
                        changed = True
                        # Audit event
                        try:
                            # Use non-async write via create_task to avoid blocking lock
                            asyncio.create_task(self._audit.write({
                                "event": "hitl_escalation",
                                "request_id": item.req.id,
                                "operation": item.req.operation,
                                "old_priority": old_priority,
                                "new_priority": item.priority,
                                "deadline_ts": item.deadline_ts,
                            }))
                        except Exception:
                            pass
                    new_heap.append(item)
                # Restore heap ordering
                for it in new_heap:
                    heapq.heappush(self._queue, it)


# Singleton instance for app-wide usage
approvals_manager = ApprovalsManager()
