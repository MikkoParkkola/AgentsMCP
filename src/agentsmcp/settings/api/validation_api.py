"""
REST API endpoints for validation services.
"""

from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Path, Body
from pydantic import BaseModel, Field

from ..services.validation_service import ValidationService


# Dependency functions (defined outside class to avoid 'self' reference issues)
def get_current_user_id() -> str:
    """Dependency to get current user ID from request context."""
    # This would typically extract user ID from JWT token or session
    # For now, returning a placeholder
    return "user_123"


class ValidateSettingRequest(BaseModel):
    key: str = Field(..., description="Setting key")
    value: Any = Field(..., description="Setting value")


class SmartSuggestionsRequest(BaseModel):
    context: Dict[str, Any] = Field(..., description="Context for suggestions")


class ValidationAPI:
    """Validation API router."""
    
    def __init__(self, validation_service: ValidationService):
        self.validation_service = validation_service
        self.router = self._create_router()
    
    def _create_router(self) -> APIRouter:
        router = APIRouter(prefix="/api/v1/validation", tags=["validation"])
        
        router.add_api_route(
            "/nodes/{node_id}/validate-setting",
            self.validate_setting,
            methods=["POST"]
        )
        
        router.add_api_route(
            "/suggestions",
            self.get_smart_suggestions,
            methods=["POST"]
        )
        
        return router
    
    async def validate_setting(self,
                              node_id: str = Path(...),
                              request: ValidateSettingRequest = Body(...),
                              user_id: str = Depends(get_current_user_id)) -> Dict[str, Any]:
        """Validate a setting in real-time."""
        try:
            result = await self.validation_service.validate_setting_real_time(
                user_id=user_id,
                node_id=node_id,
                key=request.key,
                value=request.value
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_smart_suggestions(self,
                                  request: SmartSuggestionsRequest = Body(...),
                                  user_id: str = Depends(get_current_user_id)) -> List[Dict[str, Any]]:
        """Get smart configuration suggestions."""
        try:
            suggestions = await self.validation_service.get_smart_suggestions(
                user_id=user_id,
                context=request.context
            )
            return suggestions
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def _get_current_user_id(self) -> str:
        return "user_123"


class ValidationAPIHandlers:
    def __init__(self, validation_service: ValidationService):
        self.api = ValidationAPI(validation_service)
        self.router = self.api.router