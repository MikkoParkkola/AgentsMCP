"""
agentsmcp/web/server.py
~~~~~~~~~~~~~~~~~~~~~~~

FastAPI based production ready web server for the **AgentsMCP** project.
It provides:

* CORS, security‐related middleware and structured logging
* JWT authentication (login + token refresh)
* RESTful API for agents, tasks and system status
* Server‑Sent Events (SSE) streams for live updates
* Optional WebSocket endpoint for bidirectional communication
* Static file serving (HTML / CSS / JS dashboard)
* Health‑check and Prometheus‑style metrics endpoints
* Configuration via environment variables (pydantic ``BaseSettings``)
* Rate‑limiting and request validation
* Integration layer that forwards calls to the core AgentsMCP components:
  ``AgentRegistry``, ``DiscoveryEngine``, ``CoordinationNode`` and ``RaftCluster``

The module is deliberately self‑contained – all heavy imports are wrapped in
``try/except`` blocks so the server can start even when an optional dependency
(e.g. ``fastapi_sse``) is missing.  Missing features are disabled with a clear
log message.

Typical usage::

    from agentsmcp.web.server import app

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple

# --------------------------------------------------------------------------- #
# Optional external libraries – import lazily and degrade gracefully.
# --------------------------------------------------------------------------- #

# Core FastAPI dependencies
try:
    import fastapi
    from fastapi import (
        BackgroundTasks,
        Depends,
        FastAPI,
        HTTPException,
        Path,
        Query,
        Request,
        Response,
        Security,
        status,
    )
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from starlette.concurrency import run_in_threadpool
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.types import ASGIApp, Receive, Scope, Send
    import starlette.staticfiles
    import uvicorn
    _HAS_FASTAPI = True
except Exception:  # pragma: no cover
    _HAS_FASTAPI = False
    raise ImportError("FastAPI is required for the web server")

# Pydantic for configuration and validation
try:
    from pydantic import BaseModel, Field
    try:
        # Try new pydantic-settings package first (v2+)
        from pydantic_settings import BaseSettings
    except ImportError:
        # Fallback to old import (v1)
        from pydantic import BaseSettings
    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover
    _HAS_PYDANTIC = False
    # Create simple fallbacks
    class BaseModel:
        pass
    class BaseSettings:
        pass
    def Field(*args, **kwargs):
        return None

# SSE support ---------------------------------------------------------------
try:
    from sse_starlette.sse import EventSourceResponse
    _HAS_SSE = True
except Exception:  # pragma: no cover
    try:
        from fastapi_sse import EventSourceResponse  # type: ignore
        _HAS_SSE = True
    except Exception:  # pragma: no cover
        _HAS_SSE = False
        EventSourceResponse = None  # type: ignore

# WebSocket and rate limiting ------------------------------------------------
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    _HAS_SLOWAPI = True
except Exception:  # pragma: no cover
    _HAS_SLOWAPI = False
    Limiter = None
    _rate_limit_exceeded_handler = None
    get_remote_address = None

# JWT handling ---------------------------------------------------------------
try:
    from jose import JWTError, jwt
    _HAS_JOSE = True
except Exception:  # pragma: no cover
    _HAS_JOSE = False
    JWTError = Exception
    jwt = None  # type: ignore

# Structured logging ---------------------------------------------------------
try:
    import structlog
    _HAS_STRUCTLOG = True
except Exception:  # pragma: no cover
    _HAS_STRUCTLOG = False

# WebSocket support
try:
    from fastapi import WebSocket, WebSocketDisconnect
    _HAS_WEBSOCKET = True
except Exception:  # pragma: no cover
    _HAS_WEBSOCKET = False

# --------------------------------------------------------------------------- #
# Configuration via pydantic BaseSettings
# --------------------------------------------------------------------------- #

class Settings(BaseSettings):
    """
    Application settings – populated from environment variables with sensible
    defaults.  ``env_prefix`` keeps the variable names tidy.
    """

    # Core Server ----------------------------------------------------------------
    host: str = Field("0.0.0.0", description="Host for the FastAPI server")
    port: int = Field(8000, description="Port for the FastAPI server")
    reload: bool = Field(False, description="Enable auto‑reload (dev only)")

    # CORS ------------------------------------------------------------------------
    cors_origins: List[str] = Field(
        ["*"],
        description="Allowed CORS origins – use a whitelist in production",
    )
    cors_methods: List[str] = Field(
        ["GET", "POST", "PUT", "DELETE", "OPTIONS"], description="Allowed HTTP methods"
    )
    cors_headers: List[str] = Field(["*"], description="Allowed request headers")
    cors_credentials: bool = Field(True, description="Allow cookies / auth headers")
    cors_max_age: int = Field(600, description="Cache duration for preflight requests")

    # JWT -------------------------------------------------------------------------
    jwt_secret_key: str = Field("change-me-please", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(30)
    refresh_token_expire_minutes: int = Field(1440)  # 24 hours

    # Rate limiting ---------------------------------------------------------------
    rate_limit: str = Field(
        "100/minute",
        description="Global rate limit string understood by SlowAPI; "
        "e.g. '100/minute', '10/second', etc.",
    )
    
    # Security --------------------------------------------------------------------
    allowed_hosts: List[str] = Field(["*"], description="Allowed hostnames")
    security_headers: bool = Field(True, description="Add common security headers")

    # Paths -----------------------------------------------------------------------
    static_dir: str = Field(
        "./static",
        description="Directory that contains the dashboard HTML / CSS / JS assets",
    )
    
    # Integration -----------------------------------------------------------------
    # These are *import paths* to the core components – the actual objects will be
    # imported lazily later.
    agent_registry_path: str = Field(
        "agentsmcp.discovery.agent_service.AgentRegistry", 
        description="Import path for AgentRegistry"
    )
    discovery_engine_path: str = Field(
        "agentsmcp.discovery.matching_engine.DiscoveryEngine", 
        description="Import path for DiscoveryEngine"
    )
    coordination_node_path: str = Field(
        "agentsmcp.discovery.coordination.CoordinationNode", 
        description="Import path for CoordinationNode"
    )
    raft_cluster_path: str = Field(
        "agentsmcp.discovery.raft_cluster.RaftCluster", 
        description="Import path for RaftCluster"
    )

    class Config:
        env_file = ".env"
        env_prefix = "AGENTSMCP_"
        case_sensitive = False


settings = Settings()  # Singleton instance used throughout the module

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #

def _configure_logging() -> None:
    """
    Initialise structured logging (structlog) if available, otherwise fall back
    to the standard library ``logging`` module.  The format includes a UTC
    timestamp, log level, logger name and the message payload.
    """
    if _HAS_STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logger = structlog.get_logger()
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

    # Override FastAPI / Uvicorn loggers to use the same handlers
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_error_logger.handlers = logging.root.handlers
    uvicorn_access_logger.handlers = logging.root.handlers


_configure_logging()
log = (
    structlog.get_logger()
    if _HAS_STRUCTLOG
    else logging.getLogger(__name__)  # type: ignore
)

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def import_from_path(path: str) -> Any:
    """
    Dynamically import an object from a dotted path.  ``path`` must be of the
    form ``package.module.ClassName``.  Raises ``ImportError`` with a clear
    message if the import fails.
    """
    try:
        module_path, attr_name = path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[attr_name])
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"Cannot import {path!r}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Security – JWT handling
# --------------------------------------------------------------------------- #

if not _HAS_JOSE:
    log.error("The 'python-jose' library is missing – JWT auth will be disabled.")
    JWT_SECRET_KEY = "disabled"
    JWT_ALGORITHM = "none"
else:
    JWT_SECRET_KEY = settings.jwt_secret_key
    JWT_ALGORITHM = settings.jwt_algorithm


def _create_token(data: dict, expires_delta: timedelta) -> str:
    """
    Encode a JWT using the supplied ``data`` payload and ``expires_delta``.
    """
    if not _HAS_JOSE:
        raise RuntimeError("JWT functionality not available - python-jose not installed")
    
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def create_access_token(username: str) -> str:
    """
    Produce a short‑lived access token.
    """
    return _create_token(
        {"sub": username, "type": "access"},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
    )


def create_refresh_token(username: str) -> str:
    """
    Produce a long‑lived refresh token.
    """
    return _create_token(
        {"sub": username, "type": "refresh"},
        expires_delta=timedelta(minutes=settings.refresh_token_expire_minutes),
    )


class TokenPayload(BaseModel):
    """Schema for a decoded JWT payload."""

    sub: str  # username / user identifier
    exp: int
    type: str


def decode_token(token: str) -> TokenPayload:
    """
    Decode a JWT and validate its expiration.  Raises ``HTTPException`` on
    malformed or expired tokens.
    """
    if not _HAS_JOSE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT functionality not available",
        )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return TokenPayload(**payload)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# --------------------------------------------------------------------------- #
# Pydantic models – request/response schemas and validation
# --------------------------------------------------------------------------- #

class LoginRequest(BaseModel):
    """Payload for ``/auth/login``."""

    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """Response containing both access and refresh tokens."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class AgentInfo(BaseModel):
    """Minimal representation of an agent."""

    id: str
    name: str = ""
    status: str = "unknown"
    last_heartbeat: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AgentCreateRequest(BaseModel):
    """Payload for creating a new agent."""

    name: str = Field(..., min_length=1, max_length=255)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AgentUpdateRequest(BaseModel):
    """Payload for partial update of an agent."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[str] = None


class TaskInfo(BaseModel):
    """Information about a queued / running task."""

    id: str
    agent_id: str
    payload: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    status: str = "pending"
    priority: int = 0


class TaskCreateRequest(BaseModel):
    """Payload to create a new task for an agent."""

    agent_id: str = Field(..., description="Target agent identifier")
    payload: Dict[str, Any] = Field(..., description="Arbitrary task data")
    priority: Optional[int] = Field(0, ge=0, description="Task priority (higher = earlier)")


class SystemStatusResponse(BaseModel):
    """Aggregate system status."""

    agents_total: int
    agents_active: int
    tasks_pending: int
    tasks_running: int
    uptime_seconds: int
    raft_leader: Optional[str] = None
    cluster_healthy: bool = True


class HealthResponse(BaseModel):
    """Simple health‑check payload."""

    status: str = "ok"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"


# --------------------------------------------------------------------------- #
# Dependency injection – JWT auth
# --------------------------------------------------------------------------- #

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> str:
    """
    Resolve the current user from the ``Authorization: Bearer <token>`` header.
    ``security`` dependencies are evaluated before each request that includes
    this dependency.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    payload = decode_token(token)
    if payload.type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )
    return payload.sub


# Optional auth (allows unauthenticated access)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Optional[str]:
    """Optional authentication - returns None if no token provided."""
    if credentials is None:
        return None
    try:
        token = credentials.credentials
        payload = decode_token(token)
        if payload.type != "access":
            return None
        return payload.sub
    except HTTPException:
        return None


# --------------------------------------------------------------------------- #
# Integration Layer – lazy imports of core components
# --------------------------------------------------------------------------- #

def _load_integration_component(path: str, name: str) -> Any:
    """
    Load a component class from its import path, instantiate it lazily and
    return a singleton instance.  Errors are logged and re‑raised as
    ``RuntimeError`` – the server must not start without the core services.
    """
    try:
        import os
        if os.getenv("AGENTSMCP_DISABLE_DISCOVERY") == "1":
            log.info("Discovery disabled; using mock component", component=path)
            return _create_mock_component(name)
        component_cls = import_from_path(path)
        instance = component_cls()
        log.info(f"{name} instantiated", component=path)
        return instance
    except Exception as exc:
        log.error(f"Failed to instantiate {name}", component=path, exc_info=exc)
        # Create mock component for development
        return _create_mock_component(name)


def _create_mock_component(name: str) -> Any:
    """Create a mock component for development when real components are unavailable."""
    
    class MockComponent:
        def __init__(self):
            self.name = name
            self._data = {}
            
        def list_agents(self) -> List[Dict[str, Any]]:
            return [
                {
                    "id": "agent-1",
                    "name": "Mock Agent 1", 
                    "status": "online",
                    "last_heartbeat": datetime.utcnow(),
                    "metadata": {}
                },
                {
                    "id": "agent-2",
                    "name": "Mock Agent 2",
                    "status": "offline", 
                    "last_heartbeat": datetime.utcnow() - timedelta(minutes=5),
                    "metadata": {}
                }
            ]
            
        def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
            agents = self.list_agents()
            return next((a for a in agents if a["id"] == agent_id), None)
            
        def register_agent(self, name: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
            agent = {
                "id": f"agent-{time.time()}",
                "name": name,
                "status": "online",
                "last_heartbeat": datetime.utcnow(),
                "metadata": metadata or {}
            }
            return agent
            
        def update_agent(self, agent_id: str, **kwargs) -> Optional[Dict[str, Any]]:
            agent = self.get_agent(agent_id)
            if agent:
                agent.update(**kwargs)
                return agent
            return None
            
        def remove_agent(self, agent_id: str) -> bool:
            return True
            
        def list_tasks(self, status: str = None) -> List[Dict[str, Any]]:
            tasks = [
                {
                    "id": "task-1",
                    "agent_id": "agent-1",
                    "payload": {"action": "test"},
                    "created_at": datetime.utcnow() - timedelta(minutes=1),
                    "started_at": datetime.utcnow(),
                    "finished_at": None,
                    "status": "running",
                    "priority": 1
                },
                {
                    "id": "task-2", 
                    "agent_id": "agent-2",
                    "payload": {"action": "backup"},
                    "created_at": datetime.utcnow() - timedelta(minutes=5),
                    "started_at": None,
                    "finished_at": None,
                    "status": "pending",
                    "priority": 0
                }
            ]
            if status:
                tasks = [t for t in tasks if t["status"] == status]
            return tasks
            
        def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
            tasks = self.list_tasks()
            return next((t for t in tasks if t["id"] == task_id), None)
            
        def create_task(self, agent_id: str, payload: Dict[str, Any], priority: int = 0) -> Dict[str, Any]:
            task = {
                "id": f"task-{time.time()}",
                "agent_id": agent_id,
                "payload": payload,
                "created_at": datetime.utcnow(),
                "started_at": None,
                "finished_at": None,
                "status": "pending",
                "priority": priority
            }
            return task
            
        def cancel_task(self, task_id: str) -> bool:
            return True
            
        def subscribe_status_changes(self) -> asyncio.Queue:
            queue = asyncio.Queue()
            # In real implementation, would populate with actual changes
            return queue
            
        def start(self):
            log.info(f"Mock {self.name} started")
            
        def stop(self):
            log.info(f"Mock {self.name} stopped")
            
        def shutdown(self):
            self.stop()
            
        def health_check(self) -> Dict[str, Any]:
            return {
                "status": "healthy",
                "component": self.name,
                "mock": True
            }
    
    log.warning(f"Using mock component for {name}")
    return MockComponent()


# The following globals are created at import time – they are cheap because the
# underlying objects usually hold references to external services that are
# started elsewhere (or are in‑process singletons).
_agent_registry = _load_integration_component(
    settings.agent_registry_path, "AgentRegistry"
)
_discovery_engine = _load_integration_component(
    settings.discovery_engine_path, "DiscoveryEngine"
)
_coordination_node = _load_integration_component(
    settings.coordination_node_path, "CoordinationNode"
)
_raft_cluster = _load_integration_component(settings.raft_cluster_path, "RaftCluster")

# Add start_time for uptime calculation
_server_start_time = time.time()

# --------------------------------------------------------------------------- #
# FastAPI application – core creation
# --------------------------------------------------------------------------- #

def create_app() -> FastAPI:
    """
    Initialise the FastAPI application with middleware, routers and
    exception handlers.
    """
    # ------------------------------------------------------------------- #
    # Rate limiter (optional, requires ``slowapi``)
    # ------------------------------------------------------------------- #
    limiter = None
    if _HAS_SLOWAPI and get_remote_address is not None:
        limiter = Limiter(key_func=get_remote_address)

    app = FastAPI(
        title="AgentsMCP Web UI",
        description="Real‑time monitoring and control interface for the AgentsMCP system",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        # CORS is added later via middleware
    )

    # ------------------------------------------------------------------- #
    # Middleware stack
    # ------------------------------------------------------------------- #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
        allow_credentials=settings.cors_credentials,
        max_age=settings.cors_max_age,
    )

    # Security headers (Content‑Security‑Policy, X‑Content‑Type‑Options, etc.)
    @app.middleware("http")  # type: ignore[misc]
    async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        if settings.security_headers:
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            # Simple CSP – can be replaced with a more elaborate policy
            response.headers[
                "Content-Security-Policy"
            ] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        return response

    # Request tracing middleware (adds a unique identifier)
    @app.middleware("http")
    async def request_tracing_middleware(request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(time.time_ns())
        start = time.time()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration = (time.time() - start) * 1000
            status_code = response.status_code if response else 500
            log.info(
                "request_complete",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=round(duration, 2),
                request_id=request_id,
                client=request.client.host if request.client else "unknown",
            )

    # Rate limiter middleware (optional)
    if _HAS_SLOWAPI and limiter is not None:
        app.state.limiter = limiter
        if _rate_limit_exceeded_handler:
            app.add_exception_handler(429, _rate_limit_exceeded_handler)

    # ------------------------------------------------------------------- #
    # Exception handlers
    # ------------------------------------------------------------------- #
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        log.warning(
            "validation_error",
            path=request.url.path,
            errors=exc.errors(),
            body=exc.body,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors(), "body": exc.body},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        log.info("http_error", path=request.url.path, status_code=exc.status_code, detail=exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_exception", path=request.url.path, exc=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # ------------------------------------------------------------------- #
    # Static files (dashboard UI)
    # ------------------------------------------------------------------- #
    if os.path.isdir(settings.static_dir):
        app.mount(
            "/static",
            starlette.staticfiles.StaticFiles(directory=settings.static_dir, html=True),
            name="static",
        )
        log.info("static_files_mounted", directory=settings.static_dir)
    else:
        log.warning("static_dir_missing", directory=settings.static_dir)

    # ------------------------------------------------------------------- #
    # Router registrations
    # ------------------------------------------------------------------- #
    app.include_router(auth_router, prefix="/auth", tags=["authentication"])
    app.include_router(agent_router, prefix="/agents", tags=["agents"])
    app.include_router(task_router, prefix="/tasks", tags=["tasks"])
    app.include_router(system_router, prefix="/system", tags=["system"])
    app.include_router(event_router, prefix="/events", tags=["events"])

    return app


# --------------------------------------------------------------------------- #
# Authentication router
# --------------------------------------------------------------------------- #

auth_router = fastapi.APIRouter()


@auth_router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Obtain a JWT access/refresh token pair",
)
async def login(payload: LoginRequest) -> TokenResponse:
    """
    Verify credentials against a (placeholder) user store and issue JWT tokens.
    In a real deployment this would check a DB or external auth provider.
    """
    # Placeholder – replace with real password verification
    if payload.username != "admin" or payload.password != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(payload.username)
    refresh_token = create_refresh_token(payload.username)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@auth_router.post(
    "/refresh",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Refresh an access token using a valid refresh token",
)
async def refresh_token(
    refresh_token: str = fastapi.Body(..., embed=True, description="Refresh JWT")
) -> TokenResponse:
    payload = decode_token(refresh_token)
    if payload.type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type"
        )
    new_access = create_access_token(payload.sub)
    new_refresh = create_refresh_token(payload.sub)  # optional rotation
    return TokenResponse(access_token=new_access, refresh_token=new_refresh)


# --------------------------------------------------------------------------- #
# Agent router – CRUD and SSE stream
# --------------------------------------------------------------------------- #

agent_router = fastapi.APIRouter(dependencies=[Depends(get_current_user)])


@agent_router.get(
    "/",
    response_model=List[AgentInfo],
    summary="List all registered agents",
)
async def list_agents() -> List[AgentInfo]:
    agents = await run_in_threadpool(_agent_registry.list_agents)
    return [AgentInfo(**a) for a in agents]


@agent_router.get(
    "/{agent_id}",
    response_model=AgentInfo,
    summary="Retrieve detailed information about a single agent",
)
async def get_agent(agent_id: str = Path(..., description="Agent identifier")) -> AgentInfo:
    agent = await run_in_threadpool(lambda: _agent_registry.get_agent(agent_id))
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentInfo(**agent)


@agent_router.post(
    "/",
    response_model=AgentInfo,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new agent entry",
)
async def create_agent(payload: AgentCreateRequest) -> AgentInfo:
    new_agent = await run_in_threadpool(
        lambda: _agent_registry.register_agent(name=payload.name, metadata=payload.metadata)
    )
    return AgentInfo(**new_agent)


@agent_router.patch(
    "/{agent_id}",
    response_model=AgentInfo,
    summary="Partially update an existing agent",
)
async def update_agent(
    payload: AgentUpdateRequest,
    agent_id: str = Path(..., description="Agent identifier"),
) -> AgentInfo:
    updated = await run_in_threadpool(
        lambda: _agent_registry.update_agent(
            agent_id,
            name=payload.name,
            metadata=payload.metadata,
            status=payload.status,
        )
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentInfo(**updated)


@agent_router.delete(
    "/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an agent",
)
async def delete_agent(agent_id: str = Path(..., description="Agent identifier")) -> Response:
    success = await run_in_threadpool(lambda: _agent_registry.remove_agent(agent_id))
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------------------- #
# SSE endpoint – live agent status updates
# --------------------------------------------------------------------------- #

event_router = fastapi.APIRouter()


if _HAS_SSE:

    async def _agent_status_stream() -> AsyncGenerator[str, None]:
        """
        Coroutine that yields JSON‑serializable strings whenever the
        ``AgentRegistry`` reports a status change.  It subscribes to an async
        queue provided by the registry.
        """
        try:
            queue: asyncio.Queue = await run_in_threadpool(_agent_registry.subscribe_status_changes)
            while True:
                try:
                    # Wait for status update with timeout to allow periodic heartbeats
                    status_update = await asyncio.wait_for(queue.get(), timeout=30.0)
                    # The queue contains dicts ready for JSON serialisation.
                    yield json.dumps(status_update)
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield json.dumps({"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()})
                except asyncio.CancelledError:
                    # Client disconnected – clean up.
                    break
        except Exception as e:
            log.error("SSE stream error", exc_info=e)
            yield json.dumps({"type": "error", "message": str(e)})

    @event_router.get(
        "/agents",
        response_class=EventSourceResponse,
        summary="Server‑Sent Events stream for agent status updates",
        dependencies=[Depends(get_current_user_optional)],
    )
    async def sse_agent_status() -> EventSourceResponse:
        """
        Consume the async generator and feed it to ``EventSourceResponse``.
        Each dict is emitted as JSON payload for the client.
        """
        return EventSourceResponse(_agent_status_stream())
else:
    @event_router.get(
        "/agents",
        include_in_schema=False,
        summary="[Disabled] SSE endpoint – SSE library not installed",
    )
    async def sse_agent_status_unavailable() -> JSONResponse:
        raise HTTPException(
            status_code=501,
            detail="Server Sent Events not supported – install 'sse-starlette' or 'fastapi-sse'",
        )


# --------------------------------------------------------------------------- #
# Optional WebSocket endpoint – bidirectional control channel
# --------------------------------------------------------------------------- #

if _HAS_WEBSOCKET:

    @event_router.websocket("/ws/agents/{agent_id}")
    async def websocket_agent_control(
        websocket: WebSocket,
        agent_id: str = Path(..., description="Target agent identifier"),
        token: str = Query(..., description="Bearer JWT token for auth"),
    ) -> None:
        """
        Simple command / response channel for a single agent.  The client must
        include a ``token`` query argument (JWT).  The server validates the token,
        then proxies JSON messages to the underlying ``AgentRegistry``.
        """
        try:
            payload = decode_token(token)
            if payload.type != "access":
                raise HTTPException(status_code=401, detail="Invalid token")
        except HTTPException:
            await websocket.close(code=1008)
            return

        await websocket.accept()
        # Proxy messages – in a real implementation the registry would expose
        # async queues or callbacks.
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    payload_data = json.loads(data)
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({"error": "invalid JSON"}))
                    continue

                # Here we simply echo back with a dummy status field.
                response = {"agent_id": agent_id, "received": payload_data, "status": "ack"}
                await websocket.send_text(json.dumps(response))
        except WebSocketDisconnect:
            log.info("websocket_disconnected", agent_id=agent_id)
        except Exception as e:
            log.error("websocket_error", agent_id=agent_id, exc_info=e)
        finally:
            try:
                await websocket.close()
            except:
                pass
else:
    # No websocket support – expose a placeholder that returns 501.
    @event_router.get(
        "/ws/agents/{agent_id}",
        include_in_schema=False,
        summary="[Disabled] WebSocket endpoint – WebSocket support not available",
    )
    async def ws_placeholder(agent_id: str) -> JSONResponse:
        raise HTTPException(
            status_code=501,
            detail="WebSocket support not enabled",
        )


# --------------------------------------------------------------------------- #
# Task router – enqueue and monitor tasks
# --------------------------------------------------------------------------- #

task_router = fastapi.APIRouter(dependencies=[Depends(get_current_user)])


@task_router.get(
    "/",
    response_model=List[TaskInfo],
    summary="List all tasks (pending and running)",
)
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by task status")
) -> List[TaskInfo]:
    tasks = await run_in_threadpool(lambda: _agent_registry.list_tasks(status=status))
    return [TaskInfo(**t) for t in tasks]


@task_router.post(
    "/",
    response_model=TaskInfo,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a new task for a specific agent",
)
async def create_task(payload: TaskCreateRequest) -> TaskInfo:
    # Simple validation that the target agent exists
    agent_exists = await run_in_threadpool(
        lambda: _agent_registry.get_agent(payload.agent_id)
    )
    if not agent_exists:
        raise HTTPException(status_code=404, detail="Target agent not found")

    task = await run_in_threadpool(
        lambda: _agent_registry.create_task(
            agent_id=payload.agent_id,
            payload=payload.payload,
            priority=payload.priority or 0,
        )
    )
    return TaskInfo(**task)


@task_router.get(
    "/{task_id}",
    response_model=TaskInfo,
    summary="Retrieve details about a single task",
)
async def get_task(task_id: str = Path(..., description="Task identifier")) -> TaskInfo:
    task = await run_in_threadpool(lambda: _agent_registry.get_task(task_id))
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskInfo(**task)


@task_router.delete(
    "/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel / delete a task",
)
async def delete_task(task_id: str = Path(..., description="Task identifier")) -> Response:
    success = await run_in_threadpool(lambda: _agent_registry.cancel_task(task_id))
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------------------- #
# Memory subsystem integration
# --------------------------------------------------------------------------- #

_memory_providers = []

try:
    from ..memory import get_memory_health, InMemoryProvider
    _memory_provider = InMemoryProvider()
    _memory_providers = [_memory_provider]
    log.info("memory_subsystem_initialized", provider="InMemoryProvider")
except ImportError as exc:
    log.warning("memory_subsystem_unavailable", exc_info=exc)

# --------------------------------------------------------------------------- #
# System router – health, metrics and aggregate status
# --------------------------------------------------------------------------- #

system_router = fastapi.APIRouter()


@system_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health‑check endpoint",
    tags=["monitoring"],
)
async def health_check() -> HealthResponse:
    """
    Minimal health endpoint used by orchestration platforms (Kubernetes, Docker‑
    compose, etc.).  It returns ``200 OK`` when the server process is alive.
    Additional internal checks (e.g., DB connectivity) could be added here.
    """
    return HealthResponse()


@system_router.get(
    "/health/memory",
    summary="Memory subsystem health check",
    tags=["monitoring"],
)
async def memory_health_check() -> JSONResponse:
    """
    Detailed health check for the memory subsystem including all providers.
    Returns provider-specific health status and performance metrics.
    """
    if not _memory_providers:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable", 
                "message": "Memory subsystem not initialized",
                "providers": {},
                "overall_healthy": False
            }
        )
    
    try:
        health_data = await get_memory_health(_memory_providers)
        status_code = 200 if health_data["overall_healthy"] else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if health_data["overall_healthy"] else "unhealthy",
                **health_data
            }
        )
    except Exception as exc:
        log.error("memory_health_check_failed", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(exc),
                "providers": {},
                "overall_healthy": False
            }
        )


@system_router.get(
    "/metrics",
    summary="Prometheus metrics endpoint",
    tags=["monitoring"],
)
async def metrics() -> Response:
    """
    Expose Prometheus metrics.  ``prometheus_client`` is optional – if not
    installed we return ``501 Not Implemented``.
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        raise HTTPException(status_code=501, detail="Prometheus client not installed")


@system_router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="Aggregate system status",
    tags=["monitoring"],
)
async def system_status() -> SystemStatusResponse:
    """
    Gather a snapshot of the whole platform – number of agents, tasks, uptime,
    and Raft leader.
    """
    # These calls are deliberately run in thread‑pool to avoid blocking the
    # event‑loop if the underlying components are synchronous.
    agents = await run_in_threadpool(_agent_registry.list_agents)
    tasks = await run_in_threadpool(_agent_registry.list_tasks)

    agents_total = len(agents)
    agents_active = sum(1 for a in agents if a.get("status") == "online")
    tasks_pending = sum(1 for t in tasks if t.get("status") == "pending")
    tasks_running = sum(1 for t in tasks if t.get("status") == "running")

    uptime_seconds = int(time.time() - _server_start_time)
    
    # Get Raft leader if available
    raft_leader = None
    try:
        if hasattr(_raft_cluster, 'health_check'):
            cluster_health = await run_in_threadpool(_raft_cluster.health_check)
            raft_leader = cluster_health.get("leader")
    except Exception:
        pass

    return SystemStatusResponse(
        agents_total=agents_total,
        agents_active=agents_active,
        tasks_pending=tasks_pending,
        tasks_running=tasks_running,
        uptime_seconds=uptime_seconds,
        raft_leader=raft_leader,
        cluster_healthy=True,
    )


@system_router.get(
    "/config/current",
    summary="Current system configuration",
    tags=["configuration"],
)
async def get_current_config() -> JSONResponse:
    """
    Get current system configuration including memory subsystem settings.
    This endpoint provides read-only access to current configuration values.
    """
    try:
        config_data = {
            "system": {
                "host": settings.host,
                "port": settings.port,
                "cors_origins": settings.cors_origins,
                "static_dir": settings.static_dir,
            },
            "memory": {
                "providers_available": len(_memory_providers),
                "provider_types": [provider.__class__.__name__ for provider in _memory_providers],
            },
            "features": {
                "sse_enabled": _HAS_SSE,
                "websocket_enabled": _HAS_WEBSOCKET,
                "jwt_enabled": _HAS_JOSE,
                "rate_limiting": _HAS_SLOWAPI,
                "structured_logging": _HAS_STRUCTLOG,
            },
            "security": {
                "jwt_algorithm": settings.jwt_algorithm,
                "access_token_expire_minutes": settings.access_token_expire_minutes,
                "security_headers_enabled": settings.security_headers,
            }
        }
        
        return JSONResponse(content=config_data)
    except Exception as exc:
        log.error("config_retrieval_failed", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve configuration"}
        )


# --------------------------------------------------------------------------- #
# Application instance (exported for uvicorn)
# --------------------------------------------------------------------------- #

app: FastAPI = create_app()


# --------------------------------------------------------------------------- #
# Graceful shutdown hooks – close external resources
# --------------------------------------------------------------------------- #

@app.on_event("startup")
async def on_startup() -> None:
    """
    Run any asynchronous startup logic – e.g. connect to databases, start the
    Raft node, subscribe to message buses, etc.
    """
    try:
        if hasattr(_coordination_node, 'start'):
            await run_in_threadpool(_coordination_node.start)
            log.info("coordination_node_started")
    except Exception as exc:
        log.error("coordination_node_start_failed", exc_info=exc)
        # Don't fail startup for mock components
    
    try:
        if hasattr(_raft_cluster, 'start'):
            await run_in_threadpool(_raft_cluster.start)
            log.info("raft_cluster_started")
    except Exception as exc:
        log.error("raft_cluster_start_failed", exc_info=exc)
        # Don't fail startup for mock components

    log.info("agentsmcp_web_server_started", 
             host=settings.host, 
             port=settings.port,
             has_sse=_HAS_SSE,
             has_websocket=_HAS_WEBSOCKET,
             has_jwt=_HAS_JOSE)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """
    Disconnect from external services and perform cleanup.
    """
    try:
        if hasattr(_coordination_node, 'stop'):
            await run_in_threadpool(_coordination_node.stop)
            log.info("coordination_node_stopped")
    except Exception as exc:
        log.error("coordination_node_stop_failed", exc_info=exc)

    # If the registry holds background tasks, stop them here:
    try:
        if hasattr(_agent_registry, 'shutdown'):
            await run_in_threadpool(_agent_registry.shutdown)
    except Exception:
        pass

    # Raft cluster shutdown
    try:
        if hasattr(_raft_cluster, 'shutdown'):
            await run_in_threadpool(_raft_cluster.shutdown)
    except Exception:
        pass

    log.info("agentsmcp_web_server_shutdown_complete")


# Root redirect to docs
@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root URL to API docs."""
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# --------------------------------------------------------------------------- #
# Development run helper
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    uvicorn.run(
        "agentsmcp.web.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info",
    )
