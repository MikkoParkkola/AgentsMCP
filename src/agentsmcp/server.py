import logging
import os
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent_manager import AgentManager
from .config import Config
from .logging_config import configure_logging
from .settings import AppSettings

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

        self.agent_manager = AgentManager(config)
        self._setup_routes()
        self._setup_exception_handlers()
        self._setup_metrics()

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

        @self.app.post("/spawn", response_model=SpawnResponse)
        async def spawn_agent(request: SpawnRequest):
            """Spawn a new agent to handle a task."""
            try:
                # Validate agent type exists in config
                if request.agent_type not in self.config.agents:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown agent type: {request.agent_type}",
                    )

                job_id = await self.agent_manager.spawn_agent(
                    request.agent_type, request.task, request.timeout or 300
                )

                return SpawnResponse(
                    job_id=job_id,
                    status="spawned",
                    message=f"Agent {request.agent_type} spawned successfully",
                )

            except Exception as e:
                self.log.exception("Spawn failed: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/status/{job_id}", response_model=StatusResponse)
        async def get_job_status(job_id: str):
            """Get the status of a running job."""
            try:
                status = await self.agent_manager.get_job_status(job_id)

                if not status:
                    raise HTTPException(status_code=404, detail="Job not found")

                return StatusResponse(
                    job_id=job_id,
                    state=status.state,
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
                success = await self.agent_manager.cancel_job(job_id)

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


# Uvicorn import target: "agentsmcp.server:app"
app = create_app()
