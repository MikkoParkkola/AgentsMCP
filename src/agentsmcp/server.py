import logging
import os
from datetime import datetime
from typing import Optional

import uvicorn
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .agent_manager import AgentManager
from .config import Config
from .orchestrator_factory import OrchestratorFactory, OrchestratorMode
from .logging_config import configure_logging
# Import from settings.py file directly, not the settings/ directory
import agentsmcp.settings as settings_module
from .events import EventBus
from .models import EnvelopeParser, EnvelopeStatus, EnvelopeError

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
        settings = settings or settings_module.AppSettings()
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
        
        # Initialize orchestrator using factory - simple mode by default
        self.orchestrator = OrchestratorFactory.create(config)
        
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

        def _wants_envelope(request: Request) -> bool:
            """Determine if client requested envelope; defaults to legacy (False).

            Honors:
            - Header X-Envelope: 1|true|yes
            - Query param envelope=1|true|yes
            - Accept: application/vnd.agentsmcp.envelope+json
            - Header X-Legacy-Format: 1|true|yes forces legacy
            - Query param legacy=1|true|yes forces legacy
            """
            try:
                qp = request.query_params
                # Force legacy if explicitly requested
                legacy = str(qp.get("legacy", "")).lower() in {"1", "true", "yes"} or (
                    request.headers.get("x-legacy-format", "").lower() in {"1", "true", "yes"}
                )
                if legacy:
                    return False
                return EnvelopeParser.wants_envelope(
                    headers=request.headers, query_params=request.query_params
                )
            except Exception:
                return False

        def _wrap_response(request: Request, payload, status: EnvelopeStatus = EnvelopeStatus.SUCCESS, errors: list[EnvelopeError] | None = None):
            if not _wants_envelope(request):
                return payload
            req_id = request.headers.get("x-request-id")
            env = EnvelopeParser.build_envelope(
                payload,
                status=status,
                errors=errors,
                request_id=req_id,
                source="agentsmcp-server",
            )
            return env.model_dump(mode="json")

        @self.app.get("/")
        async def root(request: Request):
            payload = {
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
            return _wrap_response(request, payload)

        @self.app.get("/health")
        async def health(request: Request):
            """Basic health check endpoint."""
            return _wrap_response(request, {"status": "healthy", "timestamp": datetime.utcnow()})

        @self.app.get("/health/ready")
        async def readiness(request: Request):
            """Readiness check - confirms all dependencies are available."""
            try:
                # Check storage connectivity
                await self.agent_manager.storage.get_job_status("health-check")

                return _wrap_response(request, {
                    "status": "ready",
                    "timestamp": datetime.utcnow(),
                    "storage": "connected",
                    "agents": len(self.config.agents),
                })
            except Exception as e:
                raise HTTPException(status_code=503, detail=str(e))

        @self.app.post("/refresh")
        async def refresh(request: Request):
            """Invalidate cached detector/model data so the UI can refresh quickly."""
            try:
                from .config.env_detector import refresh_env_detector_cache
                refresh_env_detector_cache()
                return _wrap_response(request, {"ok": True, "refreshed": True, "timestamp": datetime.utcnow()})
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health/live")
        async def liveness(request: Request):
            """Liveness check - confirms the service is alive."""
            return _wrap_response(request, {
                "status": "alive",
                "timestamp": datetime.utcnow(),
                "uptime": datetime.utcnow().timestamp(),
            })

        @self.app.get("/stats")
        async def metrics(request: Request):
            """Basic stats endpoint (non-Prometheus)."""
            active_jobs = len(
                [
                    job
                    for job in self.agent_manager.jobs.values()
                    if job.status.state.value in ["pending", "running"]
                ]
            )

            return _wrap_response(request, {
                "timestamp": datetime.utcnow(),
                "total_jobs": len(self.agent_manager.jobs),
                "active_jobs": active_jobs,
                "agent_types": list(self.config.agents.keys()),
                "storage_type": self.config.storage.type,
            })

        @self.app.get("/metrics")
        async def metrics_series(request: Request):
            """Very simple metrics for UI charts (WUI4)."""
            counts = {
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "timeout": 0,
                "running": 0,
                "pending": 0,
            }
            durations = list(getattr(self.agent_manager.metrics, "get", lambda k, d=None: [])("durations", [])) if isinstance(getattr(self.agent_manager, "metrics", None), dict) else []
            # Fallback gather from job statuses
            for job in self.agent_manager.jobs.values():
                st = job.status.state.value
                if st in counts:
                    counts[st] += 1
                if st == "running":
                    counts["running"] += 0  # already counted
            # Include queue insights
            queue_len = getattr(self.agent_manager, "queue_size", lambda: 0)()
            # Aggregate simple latency stats if present
            avg_ms = 0.0
            if durations:
                avg_ms = (sum(durations) / max(1, len(durations))) * 1000.0
            payload = {
                "timestamp": datetime.utcnow(),
                "counts": counts,
                "queue": {"size": queue_len},
                "throughput": {
                    "running": self.agent_manager.metrics.get("running", 0) if isinstance(self.agent_manager.metrics, dict) else counts.get("running", 0),
                    "completed": self.agent_manager.metrics.get("completed", 0) if isinstance(self.agent_manager.metrics, dict) else counts.get("completed", 0),
                    "failed": self.agent_manager.metrics.get("failed", 0) if isinstance(self.agent_manager.metrics, dict) else counts.get("failed", 0),
                    "avg_ms": avg_ms,
                },
            }
            return _wrap_response(request, payload)

        # Capabilities for handshake (AD4)
        @self.app.get("/capabilities")
        async def capabilities(request: Request):
            return _wrap_response(request, {
                "agent_id": "agentsmcp-local",
                "name": "agentsmcp",
                "capabilities": list(self.config.agents.keys()),
                "transport": "http",
                "endpoint": f"http://{self.config.server.host}:{self.config.server.port}",
            })

        # Coordination endpoints (AD4)
        @self.app.get("/coord/ping")
        async def coord_ping(request: Request):
            return _wrap_response(request, {"pong": True})

        class HandshakeRequest(BaseModel):
            agent_id: str
            name: str
            capabilities: list[str] = []
            endpoint: str
            token: str | None = None

        @self.app.post("/coord/handshake")
        async def coord_handshake(request: Request):
            body = await request.json()
            payload, _meta = EnvelopeParser.parse_body(body)
            req = HandshakeRequest.model_validate(payload)
            # Enforce allowlist/token when configured (AD5)
            allow = set(getattr(self.config, "discovery_allowlist", []) or [])
            shared = getattr(self.config, "discovery_token", None)
            if allow and (req.agent_id not in allow and req.name not in allow):
                raise HTTPException(status_code=403, detail="not allowed")
            if shared and req.token != shared:
                raise HTTPException(status_code=403, detail="bad token")
            return _wrap_response(request, {"ok": True, "received": req.model_dump()})

        # Jobs listing for UI (WUI5)
        @self.app.get("/jobs")
        async def list_jobs(request: Request):
            def _j(job):
                return {
                    "job_id": job.job_id,
                    "agent": job.agent_type,
                    "state": job.status.state.value,
                    "updated_at": job.status.updated_at,
                }
            return _wrap_response(request, {"jobs": [_j(j) for j in self.agent_manager.jobs.values()]})

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
        async def get_settings(request: Request):
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
                return _wrap_response(request, {"providers": providers, "agents": agents})
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        class SettingsUpdate(BaseModel):
            providers: Optional[dict] = None
            agents: Optional[dict] = None

        @self.app.put("/settings")
        async def update_settings(body: SettingsUpdate, request: Request):
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
                return _wrap_response(request, {"ok": True})
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Provider models listing for Web UI
        @self.app.get("/providers/{provider}/models")
        async def list_provider_models(provider: str, request: Request):
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
                return _wrap_response(request, {"models": [m.to_dict() for m in models]})
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        # MCP config (minimal) — gated by config.mcp_api_enabled
        if getattr(self.config, "mcp_api_enabled", False):
            @self.app.get("/mcp")
            async def get_mcp(request: Request):
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
                    return _wrap_response(request, {"servers": servers, "flags": flags})
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
            async def update_mcp(body: MCPUpdate, request: Request):
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
                        return _wrap_response(request, {
                            "ok": True,
                            "flags": {
                                "stdio": bool(getattr(self.config, "mcp_stdio_enabled", True)),
                                "ws": bool(getattr(self.config, "mcp_ws_enabled", False)),
                                "sse": bool(getattr(self.config, "mcp_sse_enabled", False)),
                            },
                        })
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
                    return _wrap_response(request, {"ok": True})
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

            @self.app.get("/mcp/status")
            async def mcp_status(request: Request):
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
                    return _wrap_response(request, {"manager": mgr.get_config(), "servers": st})
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

        # Discovery status (read-only)
        @self.app.get("/discovery")
        async def discovery_status(request: Request):
            try:
                return _wrap_response(request, {
                    "enabled": bool(getattr(self.config, "discovery_enabled", False)),
                    "allowlist": getattr(self.config, "discovery_allowlist", []),
                    "has_token": bool(getattr(self.config, "discovery_token", None)),
                })
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Costs (optional)
        @self.app.get("/costs")
        async def get_costs(request: Request):
            try:
                try:
                    from .cost.tracker import CostTracker  # type: ignore
                except Exception:
                    raise HTTPException(status_code=501, detail="costs unavailable")
                t = CostTracker()
                return _wrap_response(request, {
                    "total": t.total_cost,
                    "daily": getattr(t, "get_daily_cost", lambda: None)(),
                    "breakdown": getattr(t, "get_breakdown", lambda: None)(),
                })
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/spawn")
        async def spawn_agent(raw_request: Request):
            """Spawn a new agent to handle a task."""
            try:
                # Instantiate manager at call-time to support testing/mocking
                mgr = AgentManager(self.config)

                body = await raw_request.json()
                payload, _meta = EnvelopeParser.parse_body(body)
                req = SpawnRequest.model_validate(payload)

                job_id = await mgr.spawn_agent(
                    req.agent_type, req.task, req.timeout or 300
                )

                payload_out = {
                    "job_id": job_id,
                    "status": "spawned",
                    "message": f"Agent {req.agent_type} spawned successfully",
                }
                return _wrap_response(raw_request, payload_out)

            except HTTPException:
                raise
            except Exception as e:
                self.log.exception("Spawn failed: %s", e)
                msg = str(e)
                if "Unknown agent type" in msg or "No configuration found for agent type" in msg:
                    raise HTTPException(status_code=400, detail=msg)
                raise HTTPException(status_code=500, detail=msg)

        @self.app.get("/status/{job_id}")
        async def get_job_status(job_id: str, request: Request):
            """Get the status of a running job."""
            try:
                mgr = AgentManager(self.config)
                status = await mgr.get_job_status(job_id)

                if not status:
                    raise HTTPException(status_code=404, detail="Job not found")

                payload = {
                    "job_id": job_id,
                    "state": status.state.name,
                    "output": status.output,
                    "error": status.error,
                    "created_at": status.created_at,
                    "updated_at": status.updated_at,
                }
                return _wrap_response(request, payload)

            except HTTPException:
                raise
            except Exception as e:
                self.log.exception("Status fetch failed: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/jobs/{job_id}")
        async def cancel_job(job_id: str, request: Request):
            """Cancel a running job."""
            try:
                mgr = AgentManager(self.config)
                success = await mgr.cancel_job(job_id)

                if not success:
                    raise HTTPException(status_code=404, detail="Job not found")

                return _wrap_response(request, {"job_id": job_id, "status": "cancelled"})

            except HTTPException:
                raise
            except Exception as e:
                self.log.exception("Cancel failed: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        # Simple orchestration endpoints
        class SimpleTaskRequest(BaseModel):
            task: str
            complexity: Optional[str] = "moderate"  # simple, moderate, complex
            timeout: Optional[int] = 300
            cost_sensitive: Optional[bool] = False
            context_size_estimate: Optional[int] = 1000

        @self.app.post("/simple/execute")
        async def simple_execute(raw_request: Request):
            """Execute task using simple orchestration (default mode)."""
            try:
                body = await raw_request.json()
                payload, _meta = EnvelopeParser.parse_body(body)
                req = SimpleTaskRequest.model_validate(payload)

                # Convert to TaskRequest
                from .simple_orchestrator import TaskRequest, TaskComplexity
                task_request = TaskRequest(
                    task=req.task,
                    complexity=TaskComplexity(req.complexity or "moderate"),
                    context_size_estimate=req.context_size_estimate or 1000,
                    cost_sensitive=req.cost_sensitive or False
                )

                result = await self.orchestrator.execute_task(task_request, timeout=req.timeout or 300)
                return _wrap_response(raw_request, result)

            except HTTPException:
                raise
            except Exception as e:
                self.log.exception("Simple execute failed: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        class ChatRequest(BaseModel):
            messages: list[dict[str, str]]
            preferred_model: Optional[str] = None

        @self.app.post("/simple/chat")
        async def simple_chat(raw_request: Request):
            """Chat using simple orchestration with model selection."""
            try:
                body = await raw_request.json()
                payload, _meta = EnvelopeParser.parse_body(body)
                req = ChatRequest.model_validate(payload)

                result = await self.orchestrator.chat(req.messages, req.preferred_model)
                return _wrap_response(raw_request, result)

            except HTTPException:
                raise
            except Exception as e:
                self.log.exception("Simple chat failed: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/simple/config")
        async def get_orchestrator_config(request: Request):
            """Get current orchestrator configuration and recommendations."""
            try:
                recommendations = OrchestratorFactory.get_mode_recommendations(self.config)
                
                # Get current preferences if available
                preferences = {}
                if hasattr(self.orchestrator, 'model_preferences'):
                    for role, pref in self.orchestrator.model_preferences.items():
                        preferences[role.value] = {
                            "primary_model": pref.primary_model,
                            "fallback_models": pref.fallback_models,
                            "cost_threshold": pref.cost_threshold
                        }

                return _wrap_response(request, {
                    "mode": "simple",  # Current server uses simple mode
                    "recommendations": recommendations,
                    "preferences": preferences
                })

            except Exception as e:
                self.log.exception("Config fetch failed: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        class PreferencesRequest(BaseModel):
            preferences: dict[str, dict[str, any]]

        @self.app.put("/simple/config")
        async def update_orchestrator_preferences(raw_request: Request):
            """Update model preferences for simple orchestrator."""
            try:
                body = await raw_request.json()
                payload, _meta = EnvelopeParser.parse_body(body)
                req = PreferencesRequest.model_validate(payload)

                if hasattr(self.orchestrator, 'configure_preferences'):
                    self.orchestrator.configure_preferences(req.preferences)
                    return _wrap_response(raw_request, {"status": "updated"})
                else:
                    raise HTTPException(status_code=501, detail="Orchestrator does not support runtime preferences")

            except HTTPException:
                raise
            except Exception as e:
                self.log.exception("Preferences update failed: %s", e)
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

        @self.app.exception_handler(HTTPException)
        async def http_exc_handler(request: Request, exc: HTTPException):
            # Respect legacy clients: return default shape when requested
            try:
                qp = request.query_params
                legacy = str(qp.get("legacy", "")).lower() in {"1", "true", "yes"} or (
                    str(qp.get("envelope", "")).lower() in {"0", "false", "no"}
                ) or (request.headers.get("x-legacy-format", "").lower() in {"1", "true", "yes"})
            except Exception:
                legacy = False
            if legacy:
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

            # Normalize errors into envelope; keep status code
            detail = exc.detail
            # Extract message and code if provided
            if isinstance(detail, dict) and "message" in detail:
                message = str(detail.get("message"))
                code = str(detail.get("code", exc.status_code))
                details = {k: v for k, v in detail.items() if k not in {"message", "code"}}
            else:
                message = str(detail)
                code = str(exc.status_code)
                details = None
            errors = [EnvelopeError(code=code, message=message, details=details)]
            env = EnvelopeParser.build_envelope(
                None,
                status=EnvelopeStatus.ERROR,
                errors=errors,
                request_id=request.headers.get("x-request-id"),
                source="agentsmcp-server",
            )
            payload = env.model_dump(mode="json")
            return JSONResponse(status_code=exc.status_code, content=payload)

        @self.app.exception_handler(Exception)
        async def generic_exc_handler(request: Request, exc: Exception):
            try:
                qp = request.query_params
                legacy = str(qp.get("legacy", "")).lower() in {"1", "true", "yes"} or (
                    str(qp.get("envelope", "")).lower() in {"0", "false", "no"}
                ) or (request.headers.get("x-legacy-format", "").lower() in {"1", "true", "yes"})
            except Exception:
                legacy = False
            if legacy:
                return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

            errors = [EnvelopeError(code="internal_error", message=str(exc))]
            env = EnvelopeParser.build_envelope(
                None,
                status=EnvelopeStatus.ERROR,
                errors=errors,
                request_id=request.headers.get("x-request-id"),
                source="agentsmcp-server",
            )
            return JSONResponse(status_code=500, content=env.model_dump(mode="json"))

        @self.app.exception_handler(ValidationError)
        async def validation_exc_handler(request: Request, exc: ValidationError):
            try:
                qp = request.query_params
                legacy = str(qp.get("legacy", "")).lower() in {"1", "true", "yes"} or (
                    str(qp.get("envelope", "")).lower() in {"0", "false", "no"}
                ) or (request.headers.get("x-legacy-format", "").lower() in {"1", "true", "yes"})
            except Exception:
                legacy = False
            if legacy:
                return JSONResponse(status_code=422, content={"detail": str(exc)})
            errors = [EnvelopeError(code="validation_error", message=str(exc))]
            env = EnvelopeParser.build_envelope(
                None, status=EnvelopeStatus.ERROR, errors=errors, request_id=request.headers.get("x-request-id"), source="agentsmcp-server"
            )
            return JSONResponse(status_code=422, content=env.model_dump(mode="json"))

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
        async def list_agents(request: Request):
            """List available agent types."""
            return _wrap_response(request, {
                "agents": list(self.config.agents.keys()),
                "configs": {
                    name: {
                        "type": config.type,
                        "model": config.model,
                        "tools": config.tools,
                    }
                    for name, config in self.config.agents.items()
                },
            })

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
    env = settings_module.AppSettings()
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
