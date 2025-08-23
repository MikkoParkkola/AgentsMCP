import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .config import Config
from .models import JobStatus
from .agent_manager import AgentManager


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
    
    def __init__(self, config: Config):
        self.config = config
        self.app = FastAPI(
            title="AgentsMCP",
            description="CLI-driven MCP agent system with extensible RAG pipeline",
            version="1.0.0"
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
    
    def _setup_routes(self):
        """Set up API routes."""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "AgentsMCP",
                "version": "1.0.0",
                "description": "CLI-driven MCP agent system with extensible RAG pipeline",
                "endpoints": {
                    "spawn": "POST /spawn - Spawn a new agent",
                    "status": "GET /status/{job_id} - Get job status",
                    "health": "GET /health - Health check"
                }
            }
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "timestamp": datetime.utcnow()}
        
        @self.app.post("/spawn", response_model=SpawnResponse)
        async def spawn_agent(request: SpawnRequest):
            """Spawn a new agent to handle a task."""
            try:
                # Validate agent type exists in config
                if request.agent_type not in self.config.agents:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unknown agent type: {request.agent_type}"
                    )
                
                job_id = await self.agent_manager.spawn_agent(
                    request.agent_type, 
                    request.task, 
                    request.timeout or 300
                )
                
                return SpawnResponse(
                    job_id=job_id,
                    status="spawned",
                    message=f"Agent {request.agent_type} spawned successfully"
                )
                
            except Exception as e:
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
                    updated_at=status.updated_at
                )
                
            except HTTPException:
                raise
            except Exception as e:
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
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/agents")
        async def list_agents():
            """List available agent types."""
            return {
                "agents": list(self.config.agents.keys()),
                "configs": {
                    name: {
                        "type": config.type,
                        "model": config.model,
                        "tools": config.tools
                    }
                    for name, config in self.config.agents.items()
                }
            }
    
    async def start(self):
        """Start the server."""
        config = uvicorn.Config(
            self.app,
            host=self.config.server.host,
            port=self.config.server.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()