import logging
import os
from datetime import datetime
from typing import Optional

import uvicorn
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .agent_manager import AgentManager
from .config import Config
from .logging_config import configure_logging
from .settings import AppSettings
from .events import EventBus

try:
    from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore
except Exception:  # pragma: no cover
    Instrumentator = None  # type: ignore


class SpawnRequest(BaseModel):
    agent_type: str
    task: str
    timeout: Optional[int] = 300


class SpawnResponse(BaseModel):
    job_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    job_id: str
    state: str
    output: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AgentServer:
    """FastAPI server for the AgentsMCP system."""

    def __init__(self, config: Config, settings: Optional[AppSettings] = None):
        settings = settings or AppSettings()
        self.settings = settings
        configure_logging(level=settings.log_level, fmt=settings.log_format)
        self.log = logging.getLogger(__name__)
        self.config = config
        self.app = FastAPI(
            title="AgentsMCP",
            description="CLI-driven MCP agent system with extensible RAG pipeline",
            version="1.0.0",
        )

        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.events = EventBus()
        self.agent_manager = AgentManager(config, events=self.events)
        self._setup_routes()
        self._setup_exception_handlers()
        self._setup_metrics()
        # Discovery announce (AD2)
        try:
            if getattr(self.config, "discovery_enabled", False):
                from .discovery.announcer import Announcer

                Announcer(self.config).announce()
        except Exception:
            self.log.warning("Discovery announcer failed to run")

    def _setup_routes(self):
        """Set up API routes."""

        @self.app.get("/")
        async def root():
            return {
                "service": "AgentsMCP",
                "version": "1.0.0",
                "description": (
                    "CLI-driven MCP agent system with extensible RAG pipeline"
                ),
                "endpoints": {
                    "spawn": "POST /spawn - Spawn a new agent",
                    "status": "GET /status/{job_id} - Get job status",
                    "health": "GET /health - Health check",
                },
            }

        @self.app.get("/health")
        async def health():
            """Basic health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.utcnow()}

        @self.app.get("/health/ready")
        async def readiness():
            """Readiness check - confirms all dependencies are available."""
            try:
                # Check storage connectivity
                await self.agent_manager.storage.get_job_status("health-check")

                return {
                    "status": "ready",
                    "timestamp": datetime.utcnow(),
                    "storage": "connected",
                    "agents": len(self.config.agents),
                }
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "status": "not ready",
                        "error": str(e),
                        "timestamp": datetime.utcnow(),
                    },
                )

        @self.app.get("/health/live")
        async def liveness():
            """Liveness check - confirms the service is alive."""
            return {
                "status": "alive",
                "timestamp": datetime.utcnow(),
                "uptime": datetime.utcnow().timestamp(),
            }

        @self.app.get("/stats")
        async def metrics():
            """Basic stats endpoint (non-Prometheus)."""
            active_jobs = len(
                [
                    job
                    for job in self.agent_manager.jobs.values()
                    if job.status.state.value in ["pending", "running"]
                ]
            )

            return {
                "timestamp": datetime.utcnow(),
                "total_jobs": len(self.agent_manager.jobs),
                "active_jobs": active_jobs,
                "agent_types": list(self.config.agents.keys()),
                "storage_type": self.config.storage.type,
            }

        @self.app.get("/metrics")
        async def metrics_series():
            """Very simple metrics for UI charts (WUI4)."""
            counts = {
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "timeout": 0,
                "running": 0,
                "pending": 0,
            }
            for job in self.agent_manager.jobs.values():
                st = job.status.state.value
                if st in counts:
                    counts[st] += 1
            return {"timestamp": datetime.utcnow(), "counts": counts}

        # Capabilities for handshake (AD4)
        @self.app.get("/capabilities")
        async def capabilities():
            return {
                "agent_id": "agentsmcp-local",
                "name": "agentsmcp",
                "capabilities": list(self.config.agents.keys()),
                "transport": "http",
                "endpoint": f"http://{self.config.server.host}:{self.config.server.port}",
            }

        # Coordination endpoints (AD4)
        @self.app.get("/coord/ping")
        async def coord_ping():
            return {"pong": True}

        class HandshakeRequest(BaseModel):
            agent_id: str
            name: str
            capabilities: list[str] = []
            endpoint: str
            token: str | None = None

        @self.app.post("/coord/handshake")
        async def coord_handshake(req: HandshakeRequest):
            # Enforce allowlist/token when configured (AD5)
            allow = set(getattr(self.config, "discovery_allowlist", []) or [])
            shared = getattr(self.config, "discovery_token", None)
            if allow and (req.agent_id not in allow and req.name not in allow):
                raise HTTPException(status_code=403, detail="not allowed")
            if shared and req.token != shared:
                raise HTTPException(status_code=403, detail="bad token")
            return {"ok": True, "received": req.model_dump()}

        # Jobs listing for UI (WUI5)
        @self.app.get("/jobs")
        async def list_jobs():
            def _j(job):
                return {
                    "job_id": job.job_id,
                    "agent": job.agent_type,
                    "state": job.status.state.value,
                    "updated_at": job.status.updated_at,
                }
            return {"jobs": [_j(j) for j in self.agent_manager.jobs.values()]}

        # SSE events (WUI1)
        @self.app.get("/events")
        async def events():
            async def gen():
                async for chunk in self.events.subscribe():
                    yield chunk
            return StreamingResponse(gen(), media_type="text/event-stream")

        # Minimal UI scaffold (WUI3, WUI6)
        if getattr(self.config, "ui_enabled", True):
            try:
                static_dir = os.path.join(os.path.dirname(__file__), "web", "static")
                os.makedirs(static_dir, exist_ok=True)
                self.app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")
            except Exception:
                # Non-fatal if static can't be mounted
                pass

        # Settings API (minimal) for Web UI parity
        @self.app.get("/settings")
        async def get_settings():
            try:
                providers = {
                    name: {
                        "api_base": getattr(cfg, "api_base", None),
                        "has_key": bool(getattr(cfg, "api_key", None)),
                    }
                    for name, cfg in (self.config.providers or {}).items()
                }
                agents = {
                    name: {
                        "model": ag.model,
                        "provider": getattr(ag.provider, "value", str(ag.provider)),
                    }
                    for name, ag in self.config.agents.items()
                }
                return {"providers": providers, "agents": agents}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        class SettingsUpdate(BaseModel):
            providers: Optional[dict] = None
            agents: Optional[dict] = None

        @self.app.put("/settings")
        async def update_settings(body: SettingsUpdate):
            try:
                if body.providers:
                    for name, upd in body.providers.items():
                        if name not in self.config.providers:
                            # create new entry if needed
                            from .config import ProviderConfig, ProviderType
                            try:
                                ptype = ProviderType(name)
                            except Exception:
                                continue
                            self.config.providers[name] = ProviderConfig(name=ptype)
                        p = self.config.providers[name]
                        if "api_base" in upd:
                            p.api_base = upd.get("api_base")
                        if "api_key" in upd:
                            # accept empty to clear
                            p.api_key = upd.get("api_key")
                if body.agents:
                    for name, upd in body.agents.items():
                        if name in self.config.agents:
                            ag = self.config.agents[name]
                            if "model" in upd:
                                ag.model = upd.get("model")
                # persist to default path
                try:
                    path = self.config.default_config_path()
                    self.config.save_to_file(path)
                except Exception:
                    pass
                return {"ok": True}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Provider models listing for Web UI
        @self.app.get("/providers/{provider}/models")
        async def list_provider_models(provider: str):
            try:
                from .providers import list_models as pv_list_models
                from .config import ProviderType, ProviderConfig
                base_cfg = self.config.providers.get(provider)
                cfg = ProviderConfig(
                    name=ProviderType(provider),
                    api_key=getattr(base_cfg, "api_key", None) if base_cfg else None,
                    api_base=getattr(base_cfg, "api_base", None) if base_cfg else None,
                )
                models = pv_list_models(cfg.name, cfg)
                return {"models": [m.to_dict() for m in models]}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        # MCP config (minimal) — gated by config.mcp_api_enabled
        if getattr(self.config, "mcp_api_enabled", False):
            @self.app.get("/mcp")
            async def get_mcp():
                try:
                    servers = [
                        {
                            "name": s.name,
                            "enabled": s.enabled,
                            "transport": s.transport,
                            "command": s.command,
                            "url": s.url,
                        }
                        for s in (self.config.mcp or [])
                    ]
                    flags = {
                        "stdio": bool(getattr(self.config, "mcp_stdio_enabled", True)),
                        "ws": bool(getattr(self.config, "mcp_ws_enabled", False)),
                        "sse": bool(getattr(self.config, "mcp_sse_enabled", False)),
                    }
                    return {"servers": servers, "flags": flags}
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

            class MCPUpdate(BaseModel):
                action: str
                name: Optional[str] = None
                transport: Optional[str] = None
                command: Optional[list[str]] = None
                url: Optional[str] = None
                enabled: Optional[bool] = None
                # flags
                stdio: Optional[bool] = None
                ws: Optional[bool] = None
                sse: Optional[bool] = None

            @self.app.put("/mcp")
            async def update_mcp(body: MCPUpdate):
                try:
                    action = body.action
                    lst = list(self.config.mcp or [])
                    if action == "add" and body.name:
                        from .config import MCPServerConfig
                        if any(s.name == body.name for s in lst):
                            raise HTTPException(status_code=400, detail="exists")
                        lst.append(MCPServerConfig(name=body.name, transport=body.transport or "stdio", command=body.command, url=body.url, enabled=bool(body.enabled if body.enabled is not None else True)))
                    elif action == "set_flags":
                        # Merge provided flags into config
                        if body.stdio is not None:
                            setattr(self.config, "mcp_stdio_enabled", bool(body.stdio))
                        if body.ws is not None:
                            setattr(self.config, "mcp_ws_enabled", bool(body.ws))
                        if body.sse is not None:
                            setattr(self.config, "mcp_sse_enabled", bool(body.sse))
                        try:
                            path = self.config.default_config_path()
                            self.config.save_to_file(path)
                        except Exception:
                            pass
                        return {
                            "ok": True,
                            "flags": {
                                "stdio": bool(getattr(self.config, "mcp_stdio_enabled", True)),
                                "ws": bool(getattr(self.config, "mcp_ws_enabled", False)),
                                "sse": bool(getattr(self.config, "mcp_sse_enabled", False)),
                            },
                        }
                    elif action in ("enable", "disable") and body.name:
                        found = False
                        for s in lst:
                            if s.name == body.name:
                                s.enabled = (action == "enable")
                                found = True
                                break
                        if not found:
                            raise HTTPException(status_code=404, detail="not found")
                    elif action == "remove" and body.name:
                        lst = [s for s in lst if s.name != body.name]
                    else:
                        raise HTTPException(status_code=400, detail="bad action")
                    self.config.mcp = lst  # type: ignore[assignment]
                    try:
                        path = self.config.default_config_path()
                        self.config.save_to_file(path)
                    except Exception:
                        pass
                    return {"ok": True}
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

            @self.app.get("/mcp/status")
            async def mcp_status():
                try:
                    from .mcp.manager import MCPServer as _M, get_global_manager as _get
                    servers = []
                    for s in (self.config.mcp or []):
                        servers.append(_M(name=s.name, command=s.command, transport=s.transport, url=s.url, env=s.env or {}, cwd=s.cwd, enabled=s.enabled))
                    mgr = _get(
                        servers,
                        allow_stdio=bool(getattr(self.config, "mcp_stdio_enabled", True)),
                        allow_ws=bool(getattr(self.config, "mcp_ws_enabled", False)),
                        allow_sse=bool(getattr(self.config, "mcp_sse_enabled", False)),
                    )
                    st = await mgr.get_status()
                    return {"manager": mgr.get_config(), "servers": st}
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

        # Discovery status (read-only)
        @self.app.get("/discovery")
        async def discovery_status():
            try:
                return {
                    "enabled": bool(getattr(self.config, "discovery_enabled", False)),
                    "allowlist": getattr(self.config, "discovery_allowlist", []),
                    "has_token": bool(getattr(self.config, "discovery_token", None)),
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Costs (optional)
        @self.app.get("/costs")
        async def get_costs():
            try:
                try:
                    from .cost.tracker import CostTracker  # type: ignore
                except Exception:
                    raise HTTPException(status_code=501, detail="costs unavailable")
                t = CostTracker()
                return {
                    "total": t.total_cost,
                    "daily": getattr(t, "get_daily_cost", lambda: None)(),
                    "breakdown": getattr(t, "get_breakdown", lambda: None)(),
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/spawn", response_model=SpawnResponse)
        async def spawn_agent(request: SpawnRequest):
            """Spawn a new agent to handle a task."""
            try:
                # Instantiate manager at call-time to support testing/mocking
                mgr = AgentManager(self.config)

                job_id = await mgr.spawn_agent(
                    request.agent_type, request.task, request.timeout or 300
                )

                return SpawnResponse(
                    job_id=job_id,
                    status="spawned",
                    message=f"Agent {request.agent_type} spawned successfully",
                )

            except HTTPException:
                raise
            except Exception as e:
                self.log.exception("Spawn failed: %s", e)
                msg = str(e)
                if "Unknown agent type" in msg or "No configuration found for agent type" in msg:
                    raise HTTPException(status_code=400, detail=msg)
                raise HTTPException(status_code=500, detail=msg)

        @self.app.get("/status/{job_id}", response_model=StatusResponse)
        async def get_job_status(job_id: str):
            """Get the status of a running job."""
            try:
                mgr = AgentManager(self.config)
                status = await mgr.get_job_status(job_id)

                if not status:
                    raise HTTPException(status_code=404, detail="Job not found")

                return StatusResponse(
                    job_id=job_id,
                    state=status.state.name,
                    output=status.output,
                    error=status.error,
                    created_at=status.created_at,
                    updated_at=status.updated_at,
                )

            except HTTPException:
                raise
            except Exception as e:
                self.log.exception("Status fetch failed: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/jobs/{job_id}")
        async def cancel_job(job_id: str):
            """Cancel a running job."""
            try:
                mgr = AgentManager(self.config)
                success = await mgr.cancel_job(job_id)

                if not success:
                    raise HTTPException(status_code=404, detail="Job not found")

                return {"job_id": job_id, "status": "cancelled"}

            except HTTPException:
                raise
            except Exception as e:
                self.log.exception("Cancel failed: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_exception_handlers(self) -> None:
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            self.log.debug("Request: %s %s", request.method, request.url.path)
            response = await call_next(request)
            self.log.debug(
                "Response: %s %s -> %s",
                request.method,
                request.url.path,
                response.status_code,
            )
            return response

    def _setup_metrics(self) -> None:
        # Gate metrics by settings to avoid overhead by default
        if getattr(self, "settings", None) and not self.settings.prometheus_enabled:
            return
        if Instrumentator is not None:
            try:
                Instrumentator().instrument(self.app).expose(self.app)
            except Exception:
                self.log.warning("Prometheus instrumentation failed to initialize")

        @self.app.get("/agents")
        async def list_agents():
            """List available agent types."""
            return {
                "agents": list(self.config.agents.keys()),
                "configs": {
                    name: {
                        "type": config.type,
                        "model": config.model,
                        "tools": config.tools,
                    }
                    for name, config in self.config.agents.items()
                },
            }

    async def start(self):
        """Start the server."""
        config = uvicorn.Config(
            self.app,
            host=self.config.server.host,
            port=self.config.server.port,
            log_level=logging.getLevelName(logging.getLogger().level).lower(),
        )
        server = uvicorn.Server(config)
        await server.serve()

        # On shutdown, try to close MCP clients if any
        try:
            from .mcp.manager import get_global_manager as _get
            servers = []
            for s in (self.config.mcp or []):
                from .mcp.manager import MCPServer as _M
                servers.append(_M(name=s.name, command=s.command, transport=s.transport, url=s.url, env=s.env or {}, cwd=s.cwd, enabled=s.enabled))
            mgr = _get(
                servers,
                allow_stdio=bool(getattr(self.config, "mcp_stdio_enabled", True)),
                allow_ws=bool(getattr(self.config, "mcp_ws_enabled", False)),
                allow_sse=bool(getattr(self.config, "mcp_sse_enabled", False)),
            )
            await mgr.close_all()
        except Exception:
            pass


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create a FastAPI app configured for AgentsMCP.

    If ``config_path`` is provided, attempts to load YAML config from it; otherwise
    tries the path from ``AGENTSMCP_CONFIG`` env var, then falls back to
    ``agentsmcp.yaml`` in the working directory. If none is found, uses defaults.
    Environment variables ``AGENTSMCP_HOST`` and ``AGENTSMCP_PORT`` override
    server host/port if set.
    """
    cfg: Config
    env = AppSettings()
    path = config_path or os.getenv("AGENTSMCP_CONFIG") or "agentsmcp.yaml"
    try:
        from pathlib import Path

        if path and Path(path).exists():
            cfg = Config.from_file(Path(path))
        else:
            cfg = Config()
    except Exception:
        # On any config load error, fall back to defaults
        cfg = Config()

    # Merge environment overrides
    cfg = env.to_runtime_config(cfg)

    server = AgentServer(cfg, settings=env)
    return server.app


# Export only a factory by default to avoid import-time side effects.
# Use with uvicorn's factory mode:
#   uvicorn agentsmcp.server:create_app --factory
# If you need a concrete app variable for legacy compatibility, create it in
# your launcher script instead of at import time to keep startup fast.
app = None  # intentionally not initialized at import time
