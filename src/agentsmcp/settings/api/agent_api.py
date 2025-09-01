"""
REST API endpoints for agent management.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from pydantic import BaseModel, Field

from ..services.agent_service import AgentService
from ..domain.value_objects import AgentStatus, AgentCapability
from ..domain.entities import InstructionTemplate


# Dependency functions (defined outside class to avoid 'self' reference issues)
def get_current_user_id() -> str:
    """Dependency to get current user ID from request context."""
    # This would typically extract user ID from JWT token or session
    # For now, returning a placeholder
    return "user_123"


# Request/Response Models
class CreateAgentRequest(BaseModel):
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    base_model: str = Field(..., description="Base model name")
    category: str = Field("general", description="Agent category")
    tags: List[str] = Field(default_factory=list, description="Tags")


class AgentResponse(BaseModel):
    id: str
    name: str
    description: str
    base_model: str
    status: AgentStatus
    owner_id: str
    category: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime


class CreateInstanceRequest(BaseModel):
    name: str = Field("", description="Instance name")
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom settings")
    session_id: Optional[str] = Field(None, description="Session ID")


class InstanceResponse(BaseModel):
    id: str
    agent_id: str
    name: str
    status: AgentStatus
    created_at: datetime
    success_rate: float
    total_requests: int


class AgentAPI:
    """Agent API router and endpoint definitions."""
    
    def __init__(self, agent_service: AgentService):
        self.agent_service = agent_service
        self.router = self._create_router()
    
    def _create_router(self) -> APIRouter:
        """Create FastAPI router with all endpoints."""
        router = APIRouter(prefix="/api/v1/agents", tags=["agents"])
        
        router.add_api_route(
            "/",
            self.create_agent,
            methods=["POST"],
            response_model=AgentResponse
        )
        
        router.add_api_route(
            "/{agent_id}",
            self.get_agent,
            methods=["GET"],
            response_model=AgentResponse
        )
        
        router.add_api_route(
            "/{agent_id}/instances",
            self.create_instance,
            methods=["POST"],
            response_model=InstanceResponse
        )
        
        router.add_api_route(
            "/instances/{instance_id}/start",
            self.start_instance,
            methods=["POST"],
            response_model=InstanceResponse
        )
        
        router.add_api_route(
            "/marketplace",
            self.get_marketplace,
            methods=["GET"],
            response_model=List[AgentResponse]
        )
        
        return router
    
    async def create_agent(self,
                         request: CreateAgentRequest,
                         user_id: str = Depends(get_current_user_id)) -> AgentResponse:
        """Create a new agent definition."""
        try:
            agent = await self.agent_service.create_agent_definition(
                user_id=user_id,
                name=request.name,
                description=request.description,
                base_model=request.base_model,
                category=request.category,
                tags=request.tags
            )
            
            return AgentResponse(
                id=agent.id,
                name=agent.name,
                description=agent.description,
                base_model=agent.base_model,
                status=agent.status,
                owner_id=agent.owner_id,
                category=agent.category,
                tags=agent.tags,
                created_at=agent.created_at,
                updated_at=agent.updated_at
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_agent(self,
                       agent_id: str = Path(...),
                       user_id: str = Depends(get_current_user_id)) -> AgentResponse:
        """Get agent definition by ID."""
        # Placeholder implementation
        raise HTTPException(status_code=501, detail="Not implemented")
    
    async def create_instance(self,
                            agent_id: str = Path(...),
                            request: CreateInstanceRequest = Body(...),
                            user_id: str = Depends(get_current_user_id)) -> InstanceResponse:
        """Create an agent instance."""
        try:
            instance = await self.agent_service.create_agent_instance(
                user_id=user_id,
                agent_id=agent_id,
                instance_name=request.name,
                custom_settings=request.custom_settings,
                session_id=request.session_id
            )
            
            return InstanceResponse(
                id=instance.id,
                agent_id=instance.agent_definition_id,
                name=instance.name,
                status=instance.status,
                created_at=instance.created_at,
                success_rate=instance.success_rate,
                total_requests=instance.total_requests
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def start_instance(self,
                           instance_id: str = Path(...),
                           user_id: str = Depends(get_current_user_id)) -> InstanceResponse:
        """Start an agent instance."""
        try:
            instance = await self.agent_service.start_agent_instance(
                user_id=user_id,
                instance_id=instance_id
            )
            
            return InstanceResponse(
                id=instance.id,
                agent_id=instance.agent_definition_id,
                name=instance.name,
                status=instance.status,
                created_at=instance.created_at,
                success_rate=instance.success_rate,
                total_requests=instance.total_requests
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_marketplace(self,
                            category: Optional[str] = Query(None),
                            search: str = Query(""),
                            limit: int = Query(50, le=100),
                            user_id: str = Depends(get_current_user_id)) -> List[AgentResponse]:
        """Get agents from marketplace."""
        try:
            agents = await self.agent_service.get_agent_marketplace(
                user_id=user_id,
                category=category,
                search_query=search,
                limit=limit
            )
            
            return [
                AgentResponse(
                    id=agent.id,
                    name=agent.name,
                    description=agent.description,
                    base_model=agent.base_model,
                    status=agent.status,
                    owner_id=agent.owner_id,
                    category=agent.category,
                    tags=agent.tags,
                    created_at=agent.created_at,
                    updated_at=agent.updated_at
                )
                for agent in agents
            ]
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    

class AgentAPIHandlers:
    """Container for agent API handlers."""
    
    def __init__(self, agent_service: AgentService):
        self.api = AgentAPI(agent_service)
        self.router = self.api.router