"""
agentsmcp.discovery.agent_service

Agent Registration & Health‑Monitoring Service (AD2) for AgentsMCP.

Implements:
    • Secure agent registration (JWT‑validated bootstrap token)
    • Periodic heartbeat handling
    • Automatic deregistration on heartbeat timeout (TTL)
    • Agent discovery & load‑token issuance APIs
    • In‑memory registry with TTL cleanup
    • JSON‑Schema validation via Pydantic models
    • Integration with the global ``config`` and ``logging`` modules
    • Comprehensive type hints & docstrings

The module is deliberately self‑contained – importing it into the main
FastAPI application (or mounting it as a sub‑router) is enough to get a
fully‑functional AD2 service.

Dependencies (add to ``requirements.txt``):
    fastapi
    pydantic
    python‑jose[cryptography]
    uvicorn
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import uuid
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.background import BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, Json, ValidationError, field_validator
from jose import JWTError, jwt

# --------------------------------------------------------------------------- #
# Config & logger – assumed to exist in the existing code base
# --------------------------------------------------------------------------- #
try:
    # The project already ships a ``config`` module exposing a ``config`` object.
    # The example below shows the expected attributes – adapt if your config differs.
    from ..config import Config
    
    # Create default config instance
    _default_config = Config()
    
    # Extract JWT settings (these would need to be added to Config class)
    class _Settings:
        JWT_SECRET_KEY: str = getattr(_default_config, 'jwt_secret_key', 'super-secret-key-change-in-production')
        JWT_ALGORITHM: str = getattr(_default_config, 'jwt_algorithm', 'HS256')
        JWT_BOOTSTRAP_AUDIENCE: str = getattr(_default_config, 'jwt_bootstrap_audience', 'agentsmcp-bootstrap')
        JWT_LOAD_AUDIENCE: str = getattr(_default_config, 'jwt_load_audience', 'agentsmcp-load')
        JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = getattr(_default_config, 'jwt_access_token_expire_minutes', 60)
        HEARTBEAT_INTERVAL_SECONDS: int = getattr(_default_config, 'heartbeat_interval_seconds', 30)
        HEARTBEAT_TTL_SECONDS: int = getattr(_default_config, 'heartbeat_ttl_seconds', 120)
        LOG_LEVEL: str = getattr(_default_config, 'log_level', 'INFO')

    settings = _Settings()
    
except Exception as exc:  # pragma: no cover
    # Fallback minimal settings for isolated testing or if the real config is not
    # available at import time.
    class _DummySettings:
        JWT_SECRET_KEY: str = "super-secret"
        JWT_ALGORITHM: str = "HS256"
        JWT_BOOTSTRAP_AUDIENCE: str = "agentsmcp-bootstrap"
        JWT_LOAD_AUDIENCE: str = "agentsmcp-load"
        JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
        HEARTBEAT_INTERVAL_SECONDS: int = 30
        HEARTBEAT_TTL_SECONDS: int = 120
        LOG_LEVEL: str = "INFO"

    settings = _DummySettings()  # type: ignore

log = logging.getLogger("agentsmcp.discovery.agent_service")
log.setLevel(settings.LOG_LEVEL)

# --------------------------------------------------------------------------- #
# Pydantic models – define the JSON schema for request / response bodies
# --------------------------------------------------------------------------- #

class AgentRegisterRequest(BaseModel):
    """
    Payload sent by an agent when it wants to join the MCP.

    Attributes
    ----------
    agent_id: str
        Unique identifier chosen by the agent (e.g. hostname‑UUID). Must be a
        non‑empty string.
    capabilities: List[str]
        A list of capability identifiers the agent supports (e.g. ["cpu", "gpu"]).
    bootstrap_token: str
        JWT signed by the MCP bootstrap authority. The token is verified
        before registration is accepted.
    """
    agent_id: str = Field(..., min_length=1, description="Unique agent identifier")
    capabilities: List[str] = Field(
        default_factory=list,
        description="Capability identifiers supported by the agent",
    )
    bootstrap_token: str = Field(..., description="Bootstrap JWT token")

    @field_validator("agent_id")
    @classmethod
    def _strip(cls, v: str) -> str:  # noqa: D401
        """Strip surrounding whitespace."""
        return v.strip()


class AgentRegisterResponse(BaseModel):
    """Response returned after successful registration."""
    agent_id: str
    registered_at: _dt.datetime
    message: str = "registered"


class HeartbeatRequest(BaseModel):
    """Payload posted by an agent to indicate it is alive."""
    agent_id: str = Field(..., description="Identifier of the sending agent")
    heartbeat_token: str = Field(..., description="Short‑lived JWT issued at registration")


class HeartbeatResponse(BaseModel):
    """Acknowledgement returned for a heartbeat."""
    agent_id: str
    next_expected_heartbeat: _dt.datetime


class LoadTokenRequest(BaseModel):
    """Request a short‑lived load‑token for the given agent."""
    requested_scopes: List[str] = Field(
        default_factory=list,
        description="Scopes that the token should grant (application‑specific)",
    )
    ttl_seconds: Optional[int] = Field(
        None,
        description="Custom TTL for the token; defaults to configured expire minutes",
    )


class LoadTokenResponse(BaseModel):
    """Token returned to the caller."""
    access_token: str
    token_type: str = "bearer"
    expires_at: _dt.datetime


class AgentInfo(BaseModel):
    """Internal representation of a registered agent."""
    agent_id: str
    capabilities: List[str]
    registered_at: _dt.datetime
    last_heartbeat: _dt.datetime
    heartbeat_token: str  # JWT issued at registration, used for heartbeat auth


# --------------------------------------------------------------------------- #
# Registry & background cleanup
# --------------------------------------------------------------------------- #

class AgentRegistry:
    """
    Simple in‑memory registry for agents with TTL‑based eviction.

    The registry is deliberately thread‑safe for the typical async event loop
    scenario – all mutations happen under the same event loop, so a simple
    ``dict`` guarded by ``asyncio.Lock`` is sufficient.
    """

    def __init__(self, ttl_seconds: int) -> None:
        self._agents: Dict[str, AgentInfo] = {}
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()

    async def register(
        self,
        agent_id: str,
        capabilities: List[str],
        heartbeat_token: str,
        now: Optional[_dt.datetime] = None,
    ) -> AgentInfo:
        """
        Register (or re‑register) an agent.

        If the *agent_id* already exists it is refreshed – the registration
        timestamp is **not** changed, only the heartbeat details.
        """
        now = now or _dt.datetime.utcnow()
        async with self._lock:
            if agent_id in self._agents:
                # Refresh heartbeat information only
                info = self._agents[agent_id]
                info.last_heartbeat = now
                info.heartbeat_token = heartbeat_token
                info.capabilities = capabilities
                log.debug("Refreshed registration for agent %s", agent_id)
                return info

            info = AgentInfo(
                agent_id=agent_id,
                capabilities=capabilities,
                registered_at=now,
                last_heartbeat=now,
                heartbeat_token=heartbeat_token,
            )
            self._agents[agent_id] = info
            log.info("Registered new agent %s", agent_id)
            return info

    async def heartbeat(self, agent_id: str, token: str, now: Optional[_dt.datetime] = None) -> AgentInfo:
        """
        Record a heartbeat for *agent_id* after verifying the supplied token.

        Raises:
            KeyError: If the agent is not known.
            HTTPException(401): If the token does not match the stored heartbeat token.
        """
        now = now or _dt.datetime.utcnow()
        async with self._lock:
            info = self._agents[agent_id]  # may raise KeyError → 404 later
            if token != info.heartbeat_token:
                log.warning("Invalid heartbeat token for agent %s", agent_id)
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid heartbeat token")

            info.last_heartbeat = now
            log.debug("Heartbeat received from agent %s", agent_id)
            return info

    async def deregister(self, agent_id: str) -> None:
        """Remove an agent from the registry, silently ignoring missing entries."""
        async with self._lock:
            removed = self._agents.pop(agent_id, None)
            if removed:
                log.info("Deregistered agent %s", agent_id)

    async def get_all(self) -> List[AgentInfo]:
        """Return a snapshot list of currently registered agents."""
        async with self._lock:
            return list(self._agents.values())

    async def cleanup_expired(self) -> int:
        """Remove agents whose last heartbeat is older than TTL. Return number removed."""
        now = _dt.datetime.utcnow()
        cutoff = now - _dt.timedelta(seconds=self._ttl_seconds)

        async with self._lock:
            stale_ids = [aid for aid, info in self._agents.items() if info.last_heartbeat < cutoff]
            for aid in stale_ids:
                self._agents.pop(aid, None)
                log.info("Agent %s timed out and was deregistered (last heartbeat %s)", aid, cutoff)

            return len(stale_ids)


# --------------------------------------------------------------------------- #
# JWT handling helpers
# --------------------------------------------------------------------------- #

def _decode_jwt(token: str, audience: str) -> dict:
    """
    Decode a JWT and verify its signature, expiry, and audience.

    Raises:
        HTTPException(401) on any JWT validation problem.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            audience=audience,
        )
        return payload
    except JWTError as exc:
        log.warning("JWT validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        ) from exc


def create_load_token(agent_id: str, scopes: List[str], ttl_seconds: Optional[int] = None) -> str:
    """
    Issue a short‑lived load‑token for *agent_id*.

    The token contains:
        - ``sub``: the agent identifier,
        - ``scopes``: list supplied by the caller,
        - ``aud``: configured load audience,
        - ``exp``: expiry timestamp.
    """
    expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    exp_delta = _dt.timedelta(seconds=ttl_seconds) if ttl_seconds else _dt.timedelta(minutes=expire_minutes)
    now = _dt.datetime.utcnow()
    to_encode = {
        "sub": agent_id,
        "scopes": scopes,
        "aud": settings.JWT_LOAD_AUDIENCE,
        "iat": now,
        "exp": now + exp_delta,
    }
    token = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    log.debug("Issued load token for agent %s (exp %s)", agent_id, now + exp_delta)
    return token


def create_heartbeat_token(agent_id: str) -> str:
    """
    Create a very short‑lived JWT used for authenticating heartbeat messages.

    The token lives only for the configured heartbeat interval plus a safety
    margin (30 seconds).  It is issued at registration time and refreshed on
    each successful heartbeat (optional – current implementation re‑uses the
    same token for the lifetime of the registration).
    """
    now = _dt.datetime.utcnow()
    ttl = settings.HEARTBEAT_INTERVAL_SECONDS + 30
    payload = {
        "sub": agent_id,
        "aud": settings.JWT_LOAD_AUDIENCE,
        "iat": now,
        "exp": now + _dt.timedelta(seconds=ttl),
    }
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token


# --------------------------------------------------------------------------- #
# FastAPI app (can be mounted or run directly)
# --------------------------------------------------------------------------- #

def create_app() -> FastAPI:
    """
    Construct the FastAPI application for AD2.
    The function makes the service testable – unit‑tests can import ``app = create_app()``.
    """
    app = FastAPI(
        title="AgentsMCP – Agent Registration & Health Monitoring (AD2)",
        version="1.0.0",
        description="Provides agent registration, heartbeat monitoring, discovery and load‑token issuance.",
    )

    # Registry is stored on the app state to keep a singleton across requests.
    app.state.registry = AgentRegistry(ttl_seconds=settings.HEARTBEAT_TTL_SECONDS)

    # ------------------------------------------------------------------- #
    # Startup / shutdown handlers – background TTL cleanup task
    # ------------------------------------------------------------------- #

    @app.on_event("startup")
    async def _startup_event() -> None:
        """
        Spin up the background coroutine that periodically removes timed‑out agents.
        """
        async def _cleaner():
            log.info("Agent TTL cleaner started (interval=%ds)", settings.HEARTBEAT_INTERVAL_SECONDS)
            while True:
                await asyncio.sleep(settings.HEARTBEAT_INTERVAL_SECONDS)
                removed = await app.state.registry.cleanup_expired()
                if removed:
                    log.info("Cleaner removed %d stale agents", removed)

        # ``create_task`` registers the coroutine with the running loop.
        app.state._cleaner_task = asyncio.create_task(_cleaner())

    @app.on_event("shutdown")
    async def _shutdown_event() -> None:
        """
        Cancel the cleaner task cleanly.
        """
        cleaner: asyncio.Task = getattr(app.state, "_cleaner_task", None)  # type: ignore
        if cleaner:
            cleaner.cancel()
            try:
                await cleaner
            except asyncio.CancelledError:
                pass
            log.info("Agent TTL cleaner stopped")

    # ------------------------------------------------------------------- #
    # Dependency to fetch the singleton registry
    # ------------------------------------------------------------------- #

    def get_registry(request: Request) -> AgentRegistry:
        return request.app.state.registry

    # ------------------------------------------------------------------- #
    # 1️⃣  Agent registration
    # ------------------------------------------------------------------- #

    @app.post(
        "/agents/register",
        response_model=AgentRegisterResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Register a new agent",
        tags=["registration"],
    )
    async def register_agent(
        payload: AgentRegisterRequest,
        registry: AgentRegistry = Depends(get_registry),
    ) -> AgentRegisterResponse:
        """
        Register an agent after verifying its *bootstrap_token*.

        The bootstrap token must contain:
            - ``sub``: the agent identifier (must match ``payload.agent_id``)
            - ``aud``: ``settings.JWT_BOOTSTRAP_AUDIENCE``
        """
        # -----------------------------------------------------------------
        # Verify bootstrap JWT
        # -----------------------------------------------------------------
        boot_payload = _decode_jwt(payload.bootstrap_token, audience=settings.JWT_BOOTSTRAP_AUDIENCE)

        token_sub = boot_payload.get("sub")
        if token_sub != payload.agent_id:
            log.warning(
                "Bootstrap token sub (%s) does not match claimed agent_id (%s)",
                token_sub,
                payload.agent_id,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bootstrap token subject mismatch",
            )

        # -----------------------------------------------------------------
        # Issue a short‑lived heartbeat token that the agent will use
        # -----------------------------------------------------------------
        heartbeat_tok = create_heartbeat_token(payload.agent_id)

        # -----------------------------------------------------------------
        # Store / refresh registration
        # -----------------------------------------------------------------
        info = await registry.register(
            agent_id=payload.agent_id,
            capabilities=payload.capabilities,
            heartbeat_token=heartbeat_tok,
        )

        # -----------------------------------------------------------------
        # Return registration details
        # -----------------------------------------------------------------
        return AgentRegisterResponse(
            agent_id=info.agent_id,
            registered_at=info.registered_at,
            message="registered",
        )

    # ------------------------------------------------------------------- #
    # 2️⃣  Heartbeat endpoint
    # ------------------------------------------------------------------- #

    @app.post(
        "/agents/heartbeat",
        response_model=HeartbeatResponse,
        summary="Report agent health",
        tags=["heartbeat"],
    )
    async def heartbeat(
        payload: HeartbeatRequest,
        registry: AgentRegistry = Depends(get_registry),
    ) -> HeartbeatResponse:
        """
        Record a heartbeat from an agent. The request must contain the
        ``heartbeat_token`` that was provided at registration time.
        """
        try:
            info = await registry.heartbeat(payload.agent_id, payload.heartbeat_token)
        except KeyError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not registered")

        next_expected = _dt.datetime.utcnow() + _dt.timedelta(seconds=settings.HEARTBEAT_INTERVAL_SECONDS)
        return HeartbeatResponse(agent_id=info.agent_id, next_expected_heartbeat=next_expected)

    # ------------------------------------------------------------------- #
    # 3️⃣  Deregistration endpoint (manual)
    # ------------------------------------------------------------------- #

    @app.delete(
        "/agents/{agent_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        summary="Deregister an agent",
        tags=["registration"],
    )
    async def deregister_agent(
        agent_id: str,
        registry: AgentRegistry = Depends(get_registry),
    ) -> JSONResponse:
        """
        Remove an agent from the registry.  The operation is idempotent.
        """
        await registry.deregister(agent_id)
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)

    # ------------------------------------------------------------------- #
    # 4️⃣  Discovery API – list all alive agents
    # ------------------------------------------------------------------- #

    class AgentDiscoveryResponse(BaseModel):
        """Schema returned by ``GET /agents``."""
        agents: List[AgentInfo]

    @app.get(
        "/agents",
        response_model=AgentDiscoveryResponse,
        summary="Discover currently registered agents",
        tags=["discovery"],
    )
    async def discover_agents(registry: AgentRegistry = Depends(get_registry)) -> AgentDiscoveryResponse:
        """
        Returns a snapshot of all agents that have not timed out.
        """
        agents = await registry.get_all()
        return AgentDiscoveryResponse(agents=agents)

    # ------------------------------------------------------------------- #
    # 5️⃣  Load‑token issuance API
    # ------------------------------------------------------------------- #

    @app.post(
        "/agents/{agent_id}/load-token",
        response_model=LoadTokenResponse,
        summary="Issue a short‑lived load token for an agent",
        tags=["tokens"],
    )
    async def issue_load_token(
        agent_id: str,
        request: LoadTokenRequest,
        registry: AgentRegistry = Depends(get_registry),
    ) -> LoadTokenResponse:
        """
        Issue a JWT that can be used by the caller to perform privileged
        operations on behalf of the *agent*.  The token is signed with the
        same secret used by the MCP and includes the requested scopes.
        """
        # Verify the agent exists – otherwise we may leak tokens for unknown IDs.
        current_agents = await registry.get_all()
        if not any(ai.agent_id == agent_id for ai in current_agents):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

        token = create_load_token(
            agent_id=agent_id,
            scopes=request.requested_scopes,
            ttl_seconds=request.ttl_seconds,
        )
        payload = jwt.get_unverified_claims(token)  # safe because we just created it
        expires_at = _dt.datetime.fromtimestamp(payload["exp"], tz=_dt.timezone.utc)

        return LoadTokenResponse(access_token=token, expires_at=expires_at)

    # ------------------------------------------------------------------- #
    # 6️⃣  Exception handlers (optional but nice for production)
    # ------------------------------------------------------------------- #

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        """Return a 422 response with the pydantic error details."""
        log.warning("Request validation error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Pass through HTTPException but log it."""
        log.info("HTTPException %s: %s", exc.status_code, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    return app


# --------------------------------------------------------------------------- #
# When the module is executed directly (e.g. ``python -m agentsmcp.discovery.agent_service``)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agentsmcp.discovery.agent_service:create_app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False,
    )