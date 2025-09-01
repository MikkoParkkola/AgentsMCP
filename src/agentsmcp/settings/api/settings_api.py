"""
REST API endpoints for settings management.

Provides HTTP endpoints for managing hierarchical settings
with real-time validation and conflict resolution.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.settings_service import SettingsService
from ..services.validation_service import ValidationService
from ..services.security_service import SecurityService
from ..domain.value_objects import SettingsLevel, SettingType, ConflictResolution
from ..domain.entities import UserProfile


# Dependency functions (defined outside class to avoid 'self' reference issues)
def get_current_user_id() -> str:
    """Dependency to get current user ID from request context."""
    # This would typically extract user ID from JWT token or session
    # For now, returning a placeholder
    return "user_123"


# Request/Response Models
class CreateHierarchyRequest(BaseModel):
    name: str = Field(..., description="Name of the hierarchy")
    description: str = Field("", description="Optional description")


class CreateNodeRequest(BaseModel):
    level: SettingsLevel = Field(..., description="Settings level")
    name: str = Field(..., description="Node name")
    parent_id: Optional[str] = Field(None, description="Parent node ID")


class SetSettingRequest(BaseModel):
    key: str = Field(..., description="Setting key")
    value: Any = Field(..., description="Setting value")
    type: Optional[SettingType] = Field(None, description="Setting type (auto-inferred if not provided)")
    validate_setting: bool = Field(True, description="Whether to validate the setting")


class BulkUpdateRequest(BaseModel):
    settings: Dict[str, Any] = Field(..., description="Settings to update")
    validate_setting: bool = Field(True, description="Whether to validate settings")


class ConflictResolutionRequest(BaseModel):
    resolutions: Dict[str, str] = Field(..., description="Setting key to chosen node ID mapping")


class ExportRequest(BaseModel):
    include_secrets: bool = Field(False, description="Whether to include decrypted secrets")


class ImportRequest(BaseModel):
    import_data: Dict[str, Any] = Field(..., description="Data to import")
    merge_strategy: str = Field("override", description="Merge strategy: override, skip")


class HierarchyResponse(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    root_node_id: str
    node_count: int


class NodeResponse(BaseModel):
    id: str
    name: str
    level: SettingsLevel
    parent_id: Optional[str]
    settings_count: int
    created_at: datetime
    updated_at: datetime


class SettingResponse(BaseModel):
    key: str
    value: Any
    type: SettingType
    encrypted: bool
    last_modified: datetime
    source_level: Optional[SettingsLevel] = None
    inheritance_chain: Optional[List[Dict[str, Any]]] = None


class ValidationResultResponse(BaseModel):
    valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]


class ConflictResponse(BaseModel):
    key: str
    values: List[Dict[str, Any]]
    resolution_suggestion: ConflictResolution
    recommended_action: str
    impact_assessment: str


# API Implementation
class SettingsAPI:
    """Settings API router and endpoint definitions."""
    
    def __init__(self, 
                 settings_service: SettingsService,
                 validation_service: ValidationService,
                 security_service: SecurityService):
        self.settings_service = settings_service
        self.validation_service = validation_service
        self.security_service = security_service
        self.router = self._create_router()
    
    def _create_router(self) -> APIRouter:
        """Create FastAPI router with all endpoints."""
        router = APIRouter(prefix="/api/v1/settings", tags=["settings"])
        
        # Hierarchy management
        router.add_api_route(
            "/hierarchies",
            self.create_hierarchy,
            methods=["POST"],
            response_model=HierarchyResponse
        )
        
        router.add_api_route(
            "/hierarchies/{hierarchy_id}",
            self.get_hierarchy,
            methods=["GET"],
            response_model=HierarchyResponse
        )
        
        router.add_api_route(
            "/hierarchies/{hierarchy_id}/validate",
            self.validate_hierarchy,
            methods=["POST"],
            response_model=Dict[str, Any]
        )
        
        # Node management
        router.add_api_route(
            "/hierarchies/{hierarchy_id}/nodes",
            self.create_node,
            methods=["POST"],
            response_model=NodeResponse
        )
        
        router.add_api_route(
            "/nodes/{node_id}",
            self.get_node,
            methods=["GET"],
            response_model=NodeResponse
        )
        
        # Settings management
        router.add_api_route(
            "/nodes/{node_id}/settings/{key}",
            self.set_setting,
            methods=["PUT"],
            response_model=Dict[str, Any]
        )
        
        router.add_api_route(
            "/nodes/{node_id}/settings/{key}",
            self.get_setting,
            methods=["GET"],
            response_model=SettingResponse
        )
        
        router.add_api_route(
            "/nodes/{node_id}/settings",
            self.get_effective_settings,
            methods=["GET"],
            response_model=Dict[str, SettingResponse]
        )
        
        router.add_api_route(
            "/nodes/{node_id}/settings",
            self.bulk_update_settings,
            methods=["PUT"],
            response_model=Dict[str, Any]
        )
        
        # Validation and conflicts
        router.add_api_route(
            "/nodes/{node_id}/validate",
            self.validate_settings,
            methods=["POST"],
            response_model=ValidationResultResponse
        )
        
        router.add_api_route(
            "/nodes/{node_id}/conflicts",
            self.detect_conflicts,
            methods=["GET"],
            response_model=List[ConflictResponse]
        )
        
        router.add_api_route(
            "/nodes/{node_id}/conflicts/resolve",
            self.resolve_conflicts,
            methods=["POST"],
            response_model=Dict[str, Any]
        )
        
        # Import/Export
        router.add_api_route(
            "/hierarchies/{hierarchy_id}/export",
            self.export_settings,
            methods=["POST"],
            response_model=Dict[str, Any]
        )
        
        router.add_api_route(
            "/hierarchies/import",
            self.import_settings,
            methods=["POST"],
            response_model=Dict[str, Any]
        )
        
        return router
    
    async def create_hierarchy(self, 
                             request: CreateHierarchyRequest,
                             user_id: str = Depends(get_current_user_id)) -> HierarchyResponse:
        """Create a new settings hierarchy."""
        try:
            hierarchy = await self.settings_service.create_hierarchy(
                user_id=user_id,
                name=request.name,
                description=request.description
            )
            
            return HierarchyResponse(
                id=hierarchy.id,
                name=hierarchy.name,
                created_at=hierarchy.created_at,
                updated_at=hierarchy.updated_at,
                root_node_id=hierarchy.root_node_id,
                node_count=len(hierarchy.nodes)
            )
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    async def get_hierarchy(self,
                          hierarchy_id: str = Path(..., description="Hierarchy ID"),
                          user_id: str = Depends(get_current_user_id)) -> HierarchyResponse:
        """Get a settings hierarchy by ID."""
        # This would typically get the hierarchy from the repository
        # For now, returning a placeholder response
        raise HTTPException(status_code=501, detail="Not implemented")
    
    async def create_node(self,
                        hierarchy_id: str = Path(..., description="Hierarchy ID"),
                        request: CreateNodeRequest = Body(...),
                        user_id: str = Depends(get_current_user_id)) -> NodeResponse:
        """Create a new settings node."""
        try:
            node = await self.settings_service.create_settings_node(
                user_id=user_id,
                hierarchy_id=hierarchy_id,
                level=request.level,
                name=request.name,
                parent_id=request.parent_id
            )
            
            return NodeResponse(
                id=node.id,
                name=node.name,
                level=node.level,
                parent_id=node.parent_id,
                settings_count=len(node.settings),
                created_at=node.created_at,
                updated_at=node.updated_at
            )
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    async def set_setting(self,
                        node_id: str = Path(..., description="Node ID"),
                        key: str = Path(..., description="Setting key"),
                        request: SetSettingRequest = Body(...),
                        user_id: str = Depends(get_current_user_id)) -> Dict[str, Any]:
        """Set a setting value."""
        try:
            await self.settings_service.set_setting(
                user_id=user_id,
                node_id=node_id,
                key=key,
                value=request.value,
                setting_type=request.type,
                validate=request.validate_setting
            )
            
            return {"status": "success", "message": f"Setting '{key}' updated"}
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    async def get_setting(self,
                        node_id: str = Path(..., description="Node ID"),
                        key: str = Path(..., description="Setting key"),
                        include_inheritance: bool = Query(True, description="Include inheritance info"),
                        user_id: str = Depends(get_current_user_id)) -> SettingResponse:
        """Get a setting value with optional inheritance information."""
        try:
            setting_info = await self.settings_service.get_setting(
                user_id=user_id,
                node_id=node_id,
                key=key,
                include_inheritance=include_inheritance
            )
            
            if not setting_info:
                raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
            
            return SettingResponse(
                key=key,
                value=setting_info["value"],
                type=setting_info["type"],
                encrypted=setting_info.get("encrypted", False),
                last_modified=setting_info.get("last_modified", datetime.utcnow()),
                source_level=setting_info.get("source_level"),
                inheritance_chain=setting_info.get("inheritance_chain")
            )
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    async def get_effective_settings(self,
                                   node_id: str = Path(..., description="Node ID"),
                                   use_cache: bool = Query(True, description="Use cached values"),
                                   user_id: str = Depends(get_current_user_id)) -> Dict[str, SettingResponse]:
        """Get all effective settings for a node."""
        try:
            settings = await self.settings_service.get_effective_settings(
                user_id=user_id,
                node_id=node_id,
                use_cache=use_cache
            )
            
            response = {}
            for key, setting_info in settings.items():
                response[key] = SettingResponse(
                    key=key,
                    value=setting_info["value"],
                    type=setting_info["type"],
                    encrypted=setting_info.get("encrypted", False),
                    last_modified=setting_info["last_modified"],
                    source_level=setting_info.get("source_level")
                )
            
            return response
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    async def bulk_update_settings(self,
                                 node_id: str = Path(..., description="Node ID"),
                                 request: BulkUpdateRequest = Body(...),
                                 user_id: str = Depends(get_current_user_id)) -> Dict[str, Any]:
        """Update multiple settings in a single operation."""
        try:
            result = await self.settings_service.bulk_update_settings(
                user_id=user_id,
                node_id=node_id,
                settings=request.settings,
                validate=request.validate_setting
            )
            
            return result
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    async def validate_settings(self,
                              node_id: str = Path(..., description="Node ID"),
                              user_id: str = Depends(get_current_user_id)) -> ValidationResultResponse:
        """Validate all settings for a node."""
        try:
            errors = await self.settings_service.validate_settings(
                user_id=user_id,
                node_id=node_id
            )
            
            return ValidationResultResponse(
                valid=len(errors) == 0,
                errors=errors,
                warnings=[],
                suggestions=[]
            )
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    async def detect_conflicts(self,
                             node_id: str = Path(..., description="Node ID"),
                             user_id: str = Depends(get_current_user_id)) -> List[ConflictResponse]:
        """Detect setting conflicts for a node."""
        try:
            conflicts = await self.settings_service.detect_conflicts(
                user_id=user_id,
                node_id=node_id
            )
            
            response = []
            for conflict in conflicts:
                response.append(ConflictResponse(
                    key=conflict["key"],
                    values=conflict["values"],
                    resolution_suggestion=conflict["resolution_suggestion"],
                    recommended_action=conflict["recommended_action"],
                    impact_assessment=conflict["impact_assessment"]
                ))
            
            return response
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    async def resolve_conflicts(self,
                              node_id: str = Path(..., description="Node ID"),
                              request: ConflictResolutionRequest = Body(...),
                              user_id: str = Depends(get_current_user_id)) -> Dict[str, Any]:
        """Resolve setting conflicts with user decisions."""
        try:
            result = await self.settings_service.resolve_conflicts(
                user_id=user_id,
                node_id=node_id,
                resolutions=request.resolutions
            )
            
            return result
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    async def export_settings(self,
                            hierarchy_id: str = Path(..., description="Hierarchy ID"),
                            request: ExportRequest = Body(...),
                            user_id: str = Depends(get_current_user_id)) -> Dict[str, Any]:
        """Export settings hierarchy."""
        try:
            export_data = await self.settings_service.export_settings(
                user_id=user_id,
                hierarchy_id=hierarchy_id,
                include_secrets=request.include_secrets
            )
            
            return export_data
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    async def import_settings(self,
                            request: ImportRequest = Body(...),
                            user_id: str = Depends(get_current_user_id)) -> Dict[str, Any]:
        """Import settings from external data."""
        try:
            result = await self.settings_service.import_settings(
                user_id=user_id,
                import_data=request.import_data,
                merge_strategy=request.merge_strategy
            )
            
            return result
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    async def validate_hierarchy(self,
                               hierarchy_id: str = Path(..., description="Hierarchy ID"),
                               user_id: str = Depends(get_current_user_id)) -> Dict[str, Any]:
        """Validate hierarchy consistency."""
        try:
            result = await self.validation_service.validate_hierarchy_consistency(hierarchy_id)
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# Handler class for dependency injection
class SettingsAPIHandlers:
    """Container for settings API handlers and dependencies."""
    
    def __init__(self,
                 settings_service: SettingsService,
                 validation_service: ValidationService,
                 security_service: SecurityService):
        self.api = SettingsAPI(settings_service, validation_service, security_service)
        self.router = self.api.router