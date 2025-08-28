"""
FastAPI router providing a minimal but secure HITL Approval UI and API.

Features:
- Queue listing with priorities
- Approve/Reject endpoints with JWT-based auth and RBAC (admin/operator)
- Rate limiting on decision endpoints (if SlowAPI installed)
- CSRF-light approach: all actions via JSON API with Bearer token
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    import fastapi
    from fastapi import Depends, Request, HTTPException, status
    from fastapi.responses import HTMLResponse
    from fastapi.templating import Jinja2Templates
except Exception as exc:  # pragma: no cover
    raise ImportError("FastAPI and templates are required for approval UI") from exc

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
except Exception:  # pragma: no cover
    Limiter = None  # type: ignore
    get_remote_address = None  # type: ignore

from agentsmcp.security.approvals import (
    ApprovalDecision,
    ApprovalsManager,
    approvals_manager,
)

# We import the auth dependency from the main server module to re-use JWT logic
from agentsmcp.web.server import get_current_user


templates = Jinja2Templates(directory="templates")


def _parse_csv_env(name: str) -> set[str]:
    return {x.strip() for x in os.getenv(name, "").split(",") if x.strip()}


ADMIN_USERS = _parse_csv_env("AGENTSMCP_HITL_ADMIN_USERS") or {"admin"}
OPERATOR_USERS = _parse_csv_env("AGENTSMCP_HITL_OPERATOR_USERS") or {"admin", "operator"}


def _require_role(user: str, required: str) -> None:
    if required == "admin" and user not in ADMIN_USERS:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    if required == "operator" and user not in OPERATOR_USERS and user not in ADMIN_USERS:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Operator role required")


router = fastapi.APIRouter(prefix="/hitl", tags=["hitl"])
_limiter = Limiter(key_func=get_remote_address) if (Limiter and get_remote_address) else None


@router.get("/", response_class=HTMLResponse)
async def approval_page(request: Request) -> HTMLResponse:
    # Serve shell page; client-side JS performs auth and data fetching
    return templates.TemplateResponse("approval.html", {"request": request, "user": "anonymous"})


@router.get("/queue", response_model=List[Dict[str, Any]])
async def get_queue(user: str = Depends(get_current_user)) -> List[Dict[str, Any]]:
    _require_role(user, "operator")
    pending = await approvals_manager.list_pending(limit=200)
    return [p.model_dump() for p in pending]



_RATE_LIMIT = int(os.getenv("AGENTSMCP_HITL_DECISION_RATELIMIT_PER_MIN", "30"))
_rl_state: dict[tuple[str, int], int] = {}


def _rate_limit_decisions(user: str) -> None:
    """Very small in-memory sliding window limiter: _RATE_LIMIT decisions/min per user."""
    from time import time
    minute = int(time() // 60)
    key = (user, minute)
    count = _rl_state.get(key, 0) + 1
    _rl_state[key] = count
    # prune old buckets occasionally
    if len(_rl_state) > 1000:
        for k in list(_rl_state.keys()):
            if k[1] < minute:
                _rl_state.pop(k, None)
    if count > _RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# Apply SlowAPI limiter if available in addition to local limiter
if _limiter:
    decide_decorator = _limiter.limit("30/minute")
else:
    def decide_decorator(fn):
        return fn

@router.post("/decide", response_model=Dict[str, Any])
@decide_decorator
async def decide(payload: Dict[str, Any], user: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Approve or reject a specific request.

    Payload: {"request_id": str, "decision": "approved"|"rejected", "reason": str}
    """
    _require_role(user, "operator")
    request_id = str(payload.get("request_id", "")).strip()
    decision = str(payload.get("decision", "")).strip()
    reason = str(payload.get("reason", "")).strip() or None
    if not request_id or decision not in {"approved", "rejected"}:
        raise HTTPException(status_code=400, detail="Invalid payload")
    _rate_limit_decisions(user)
    dec: ApprovalDecision = await approvals_manager.decide(
        request_id=request_id, decision=decision, decided_by=user, reason=reason
    )
    return {"status": "ok", "decision": dec.model_dump()}
