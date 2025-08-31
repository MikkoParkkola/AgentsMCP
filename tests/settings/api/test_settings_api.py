"""
API tests for settings REST endpoints.
"""

import pytest
import json
import uuid
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import FastAPI

from agentsmcp.settings.api.settings_api import SettingsAPI
from agentsmcp.settings.services.settings_service import SettingsService
from agentsmcp.settings.adapters.memory_repositories import (
    InMemorySettingsRepository,
    InMemoryUserRepository,
    InMemoryCacheRepository,
    InMemoryAuditRepository,
)
from agentsmcp.settings.domain.entities import UserProfile
from agentsmcp.settings.domain.value_objects import (
    SettingsLevel,
    SettingType,
    PermissionLevel,
)
from agentsmcp.settings.events.event_publisher import EventPublisher


@pytest.fixture
def repositories():
    """Create repository fixtures."""
    return {
        "settings": InMemorySettingsRepository(),
        "user": InMemoryUserRepository(),
        "cache": InMemoryCacheRepository(),
        "audit": InMemoryAuditRepository(),
    }


@pytest.fixture
def event_publisher():
    """Create event publisher fixture."""
    return EventPublisher()


@pytest.fixture
def settings_service(repositories, event_publisher):
    """Create settings service fixture."""
    return SettingsService(
        settings_repository=repositories["settings"],
        user_repository=repositories["user"],
        cache_repository=repositories["cache"],
        audit_repository=repositories["audit"],
        event_publisher=event_publisher
    )


@pytest.fixture
def sample_user():
    """Create sample user fixture."""
    return UserProfile(
        user_id="test_user_123",
        username="testuser",
        email="test@example.com",
        permissions=[PermissionLevel.READ, PermissionLevel.WRITE]
    )


@pytest.fixture
def app(settings_service):
    """Create FastAPI test app."""
    app = FastAPI()
    settings_api = SettingsAPI(settings_service)
    app.include_router(settings_api.router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestSettingsAPI:
    """Test Settings API endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_hierarchy_endpoint(self, client, repositories, sample_user):
        """Test POST /api/v1/settings/hierarchies endpoint."""
        await repositories["user"].save_user(sample_user)
        
        request_data = {
            "name": "Test Hierarchy",
            "description": "A test hierarchy for API testing"
        }
        
        response = client.post(
            "/api/v1/settings/hierarchies",
            json=request_data,
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "hierarchy_id" in response_data
        assert isinstance(response_data["hierarchy_id"], str)
        assert "message" in response_data
        
        # Verify hierarchy was created
        hierarchy = await repositories["settings"].get_hierarchy(response_data["hierarchy_id"])
        assert hierarchy is not None
        assert hierarchy.name == "Test Hierarchy"
    
    @pytest.mark.asyncio
    async def test_get_hierarchy_endpoint(self, client, repositories, settings_service, sample_user):
        """Test GET /api/v1/settings/hierarchies/{hierarchy_id} endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy first
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Get Test Hierarchy"
        )
        
        response = client.get(
            f"/api/v1/settings/hierarchies/{hierarchy_id}",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["id"] == hierarchy_id
        assert response_data["name"] == "Get Test Hierarchy"
        assert "nodes" in response_data
        assert "created_at" in response_data
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_hierarchy(self, client, sample_user):
        """Test GET endpoint with non-existent hierarchy ID."""
        fake_id = str(uuid.uuid4())
        
        response = client.get(
            f"/api/v1/settings/hierarchies/{fake_id}",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_add_node_endpoint(self, client, repositories, settings_service, sample_user):
        """Test POST /api/v1/settings/hierarchies/{hierarchy_id}/nodes endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy first
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Node Test Hierarchy"
        )
        
        request_data = {
            "level": "global",
            "name": "Global Settings",
            "parent_id": None
        }
        
        response = client.post(
            f"/api/v1/settings/hierarchies/{hierarchy_id}/nodes",
            json=request_data,
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "node_id" in response_data
        assert isinstance(response_data["node_id"], str)
        
        # Verify node was added
        hierarchy = await repositories["settings"].get_hierarchy(hierarchy_id)
        assert response_data["node_id"] in hierarchy.nodes
    
    @pytest.mark.asyncio
    async def test_set_setting_endpoint(self, client, repositories, settings_service, sample_user):
        """Test PUT /api/v1/settings/hierarchies/{hierarchy_id}/nodes/{node_id}/settings/{key} endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy and node
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Setting Test Hierarchy"
        )
        
        node_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.USER,
            name="User Settings"
        )
        
        request_data = {
            "value": "test_value",
            "type": "string"
        }
        
        response = client.put(
            f"/api/v1/settings/hierarchies/{hierarchy_id}/nodes/{node_id}/settings/test_key",
            json=request_data,
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        # Verify setting was set
        setting = await settings_service.get_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            key="test_key"
        )
        assert setting is not None
        assert setting.value == "test_value"
    
    @pytest.mark.asyncio
    async def test_get_setting_endpoint(self, client, repositories, settings_service, sample_user):
        """Test GET /api/v1/settings/hierarchies/{hierarchy_id}/nodes/{node_id}/settings/{key} endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy, node, and setting
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Get Setting Test"
        )
        
        node_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.USER,
            name="User Settings"
        )
        
        await settings_service.set_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            key="get_test_key",
            value=42,
            setting_type=SettingType.INTEGER
        )
        
        response = client.get(
            f"/api/v1/settings/hierarchies/{hierarchy_id}/nodes/{node_id}/settings/get_test_key",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["value"] == 42
        assert response_data["type"] == "integer"
        assert response_data["source_level"] == "user"
        assert "created_at" in response_data
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_setting(self, client, repositories, settings_service, sample_user):
        """Test GET endpoint with non-existent setting key."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy and node
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Nonexistent Test"
        )
        
        node_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.USER,
            name="User Settings"
        )
        
        response = client.get(
            f"/api/v1/settings/hierarchies/{hierarchy_id}/nodes/{node_id}/settings/nonexistent_key",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_delete_setting_endpoint(self, client, repositories, settings_service, sample_user):
        """Test DELETE /api/v1/settings/hierarchies/{hierarchy_id}/nodes/{node_id}/settings/{key} endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy, node, and setting
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Delete Setting Test"
        )
        
        node_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.USER,
            name="User Settings"
        )
        
        await settings_service.set_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            key="delete_me",
            value="to_be_deleted",
            setting_type=SettingType.STRING
        )
        
        # Verify setting exists
        setting = await settings_service.get_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            key="delete_me"
        )
        assert setting is not None
        
        # Delete setting
        response = client.delete(
            f"/api/v1/settings/hierarchies/{hierarchy_id}/nodes/{node_id}/settings/delete_me",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        # Verify setting was deleted
        setting = await settings_service.get_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            key="delete_me"
        )
        assert setting is None
    
    @pytest.mark.asyncio
    async def test_get_effective_settings_endpoint(self, client, repositories, settings_service, sample_user):
        """Test GET /api/v1/settings/hierarchies/{hierarchy_id}/nodes/{node_id}/effective endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Effective Settings Test"
        )
        
        # Create parent node with settings
        parent_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.GLOBAL,
            name="Global Settings"
        )
        
        await settings_service.set_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=parent_id,
            key="parent_setting",
            value="parent_value",
            setting_type=SettingType.STRING
        )
        
        # Create child node with settings
        child_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.USER,
            name="User Settings",
            parent_id=parent_id
        )
        
        await settings_service.set_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=child_id,
            key="child_setting",
            value="child_value",
            setting_type=SettingType.STRING
        )
        
        # Get effective settings
        response = client.get(
            f"/api/v1/settings/hierarchies/{hierarchy_id}/nodes/{child_id}/effective",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "settings" in response_data
        settings = response_data["settings"]
        
        # Should have both parent and child settings
        assert "parent_setting" in settings
        assert "child_setting" in settings
        assert settings["parent_setting"]["value"] == "parent_value"
        assert settings["child_setting"]["value"] == "child_value"
    
    @pytest.mark.asyncio
    async def test_bulk_update_settings_endpoint(self, client, repositories, settings_service, sample_user):
        """Test PUT /api/v1/settings/hierarchies/{hierarchy_id}/nodes/{node_id}/bulk endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy and node
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Bulk Update Test"
        )
        
        node_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.USER,
            name="User Settings"
        )
        
        request_data = {
            "settings": {
                "string_setting": {"value": "bulk_string", "type": "string"},
                "int_setting": {"value": 100, "type": "integer"},
                "bool_setting": {"value": True, "type": "boolean"}
            }
        }
        
        response = client.put(
            f"/api/v1/settings/hierarchies/{hierarchy_id}/nodes/{node_id}/bulk",
            json=request_data,
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["updated_count"] == 3
        
        # Verify all settings were set
        for key in ["string_setting", "int_setting", "bool_setting"]:
            setting = await settings_service.get_setting(
                user_id=sample_user.user_id,
                hierarchy_id=hierarchy_id,
                node_id=node_id,
                key=key
            )
            assert setting is not None
    
    @pytest.mark.asyncio
    async def test_unauthorized_access(self, client):
        """Test API endpoints without proper authentication."""
        # Try to create hierarchy without user ID header
        response = client.post(
            "/api/v1/settings/hierarchies",
            json={"name": "Unauthorized Test"}
        )
        
        assert response.status_code == 422  # Missing required header
    
    @pytest.mark.asyncio
    async def test_invalid_request_data(self, client, sample_user):
        """Test API endpoints with invalid request data."""
        # Try to create hierarchy with missing required fields
        response = client.post(
            "/api/v1/settings/hierarchies",
            json={},  # Missing required 'name' field
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 422
        error_details = response.json()["detail"]
        assert any("name" in str(error) for error in error_details)
    
    @pytest.mark.asyncio
    async def test_invalid_setting_type(self, client, repositories, settings_service, sample_user):
        """Test setting endpoint with invalid setting type."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy and node
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Invalid Type Test"
        )
        
        node_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.USER,
            name="User Settings"
        )
        
        request_data = {
            "value": "test_value",
            "type": "invalid_type"  # Invalid type
        }
        
        response = client.put(
            f"/api/v1/settings/hierarchies/{hierarchy_id}/nodes/{node_id}/settings/test_key",
            json=request_data,
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 422
        error_details = response.json()["detail"]
        assert any("type" in str(error) for error in error_details)
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, client, sample_user):
        """Test API error handling for server errors."""
        # This test would be more meaningful with actual error conditions
        # For now, test with invalid UUID format
        response = client.get(
            "/api/v1/settings/hierarchies/invalid-uuid",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        # Should handle the invalid UUID gracefully
        assert response.status_code in [400, 404, 422]
    
    @pytest.mark.asyncio
    async def test_settings_inheritance_via_api(self, client, repositories, settings_service, sample_user):
        """Test settings inheritance through API endpoints."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Inheritance API Test"
        )
        
        # Create parent node and set a value
        response = client.post(
            f"/api/v1/settings/hierarchies/{hierarchy_id}/nodes",
            json={"level": "global", "name": "Global Settings"},
            headers={"X-User-ID": sample_user.user_id}
        )
        parent_id = response.json()["node_id"]
        
        client.put(
            f"/api/v1/settings/hierarchies/{hierarchy_id}/nodes/{parent_id}/settings/inherited_key",
            json={"value": "inherited_value", "type": "string"},
            headers={"X-User-ID": sample_user.user_id}
        )
        
        # Create child node
        response = client.post(
            f"/api/v1/settings/hierarchies/{hierarchy_id}/nodes",
            json={"level": "user", "name": "User Settings", "parent_id": parent_id},
            headers={"X-User-ID": sample_user.user_id}
        )
        child_id = response.json()["node_id"]
        
        # Get setting from child (should inherit from parent)
        response = client.get(
            f"/api/v1/settings/hierarchies/{hierarchy_id}/nodes/{child_id}/settings/inherited_key",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        setting_data = response.json()
        assert setting_data["value"] == "inherited_value"
        assert setting_data["source_level"] == "global"  # Inherited from parent


if __name__ == "__main__":
    pytest.main([__file__])