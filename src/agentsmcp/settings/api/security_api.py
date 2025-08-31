"""
REST API endpoints for security services.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Path, Body
from pydantic import BaseModel, Field

from ..services.security_service import SecurityService
from ..domain.value_objects import PermissionLevel


class GrantPermissionRequest(BaseModel):
    user_id: str = Field(..., description="Target user ID")
    resource_type: str = Field(..., description="Resource type")
    resource_id: str = Field(..., description="Resource ID")
    permission_level: PermissionLevel = Field(..., description="Permission level")
    expires_at: Optional[datetime] = Field(None, description="Expiration time")


class SecurityAPI:
    """Security API router."""
    
    def __init__(self, security_service: SecurityService):
        self.security_service = security_service
        self.router = self._create_router()
    
    def _create_router(self) -> APIRouter:
        router = APIRouter(prefix="/api/v1/security", tags=["security"])
        
        router.add_api_route(
            "/permissions/grant",
            self.grant_permission,
            methods=["POST"]
        )
        
        router.add_api_route(
            "/audit/{user_id}",
            self.get_audit_events,
            methods=["GET"]
        )
        
        return router
    
    async def grant_permission(self,
                             request: GrantPermissionRequest = Body(...),
                             granter_id: str = Depends(self._get_current_user_id)) -> Dict[str, Any]:
        """Grant permission to a user."""
        try:
            await self.security_service.grant_permission(
                granter_id=granter_id,
                user_id=request.user_id,
                resource_type=request.resource_type,
                resource_id=request.resource_id,
                permission_level=request.permission_level,
                expires_at=request.expires_at
            )
            return {"status": "success", "message": "Permission granted"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_audit_events(self,
                              user_id: str = Path(...),
                              hours: int = 24,
                              current_user_id: str = Depends(self._get_current_user_id)) -> Dict[str, Any]:
        """Get security audit events for a user."""
        try:
            audit_data = await self.security_service.audit_security_events(user_id, hours)
            return audit_data
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def _get_current_user_id(self) -> str:
        return "user_123"


class SecurityAPIHandlers:
    def __init__(self, security_service: SecurityService):
        self.api = SecurityAPI(security_service)
        self.router = self.api.router