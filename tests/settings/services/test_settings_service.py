"""
Integration tests for SettingsService.
"""

import pytest
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

from agentsmcp.settings.services.settings_service import SettingsService
from agentsmcp.settings.adapters.memory_repositories import (
    InMemorySettingsRepository,
    InMemoryUserRepository,
    InMemoryCacheRepository,
    InMemoryAuditRepository,
)
from agentsmcp.settings.domain.entities import SettingsNode, SettingsHierarchy, UserProfile
from agentsmcp.settings.domain.value_objects import (
    SettingsLevel,
    SettingType,
    SettingValue,
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


class TestSettingsService:
    """Test SettingsService integration."""
    
    @pytest.mark.asyncio
    async def test_create_hierarchy(self, settings_service, repositories, sample_user):
        """Test creating a new settings hierarchy."""
        # First save the user
        await repositories["user"].save_user(sample_user)
        
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Test Hierarchy",
            description="A test hierarchy"
        )
        
        assert isinstance(hierarchy_id, str)
        
        # Verify hierarchy was saved
        hierarchy = await repositories["settings"].get_hierarchy(hierarchy_id)
        assert hierarchy is not None
        assert hierarchy.name == "Test Hierarchy"
        
        # Verify audit entry was created
        audit_entries = await repositories["audit"].get_user_entries(sample_user.user_id)
        assert len(audit_entries) >= 1
        
        create_entry = next((e for e in audit_entries if e.action == "hierarchy.create"), None)
        assert create_entry is not None
        assert create_entry.resource_id == hierarchy_id
    
    @pytest.mark.asyncio
    async def test_add_settings_node(self, settings_service, repositories, sample_user):
        """Test adding a settings node to hierarchy."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy first
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Test Hierarchy"
        )
        
        # Add a settings node
        node_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.GLOBAL,
            name="Global Settings",
            parent_id=None
        )
        
        assert isinstance(node_id, str)
        
        # Verify node was added to hierarchy
        hierarchy = await repositories["settings"].get_hierarchy(hierarchy_id)
        assert node_id in hierarchy.nodes
        
        node = hierarchy.nodes[node_id]
        assert node.name == "Global Settings"
        assert node.level == SettingsLevel.GLOBAL
    
    @pytest.mark.asyncio
    async def test_set_setting_in_node(self, settings_service, repositories, sample_user):
        """Test setting a value in a node."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy and node
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Test Hierarchy"
        )
        
        node_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.USER,
            name="User Settings"
        )
        
        # Set a setting
        await settings_service.set_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            key="test_setting",
            value="test_value",
            setting_type=SettingType.STRING
        )
        
        # Verify setting was set
        hierarchy = await repositories["settings"].get_hierarchy(hierarchy_id)
        node = hierarchy.nodes[node_id]
        setting = node.get_setting("test_setting")
        
        assert setting is not None
        assert setting.value == "test_value"
        assert setting.type == SettingType.STRING
        assert setting.source_level == SettingsLevel.USER
    
    @pytest.mark.asyncio
    async def test_get_setting_with_inheritance(self, settings_service, repositories, sample_user):
        """Test getting setting with inheritance from parent node."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Test Hierarchy"
        )
        
        # Create parent node with setting
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
            key="inherited_setting",
            value="parent_value",
            setting_type=SettingType.STRING
        )
        
        # Create child node without the setting
        child_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.USER,
            name="User Settings",
            parent_id=parent_id
        )
        
        # Get setting from child (should inherit from parent)
        setting = await settings_service.get_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=child_id,
            key="inherited_setting"
        )
        
        assert setting is not None
        assert setting.value == "parent_value"
        assert setting.source_level == SettingsLevel.GLOBAL  # From parent
    
    @pytest.mark.asyncio
    async def test_get_effective_settings(self, settings_service, repositories, sample_user):
        """Test getting all effective settings for a node."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Test Hierarchy"
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
            key="parent_only",
            value="parent_value",
            setting_type=SettingType.STRING
        )
        
        await settings_service.set_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=parent_id,
            key="shared_setting",
            value="parent_shared",
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
            key="child_only",
            value="child_value",
            setting_type=SettingType.STRING
        )
        
        await settings_service.set_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=child_id,
            key="shared_setting",
            value="child_shared",
            setting_type=SettingType.STRING
        )
        
        # Get effective settings
        effective = await settings_service.get_effective_settings(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=child_id
        )
        
        assert len(effective) == 3
        assert effective["parent_only"].value == "parent_value"
        assert effective["child_only"].value == "child_value"
        assert effective["shared_setting"].value == "child_shared"  # Child overrides parent
    
    @pytest.mark.asyncio
    async def test_remove_setting(self, settings_service, repositories, sample_user):
        """Test removing a setting from a node."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy and node
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Test Hierarchy"
        )
        
        node_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.USER,
            name="User Settings"
        )
        
        # Set a setting
        await settings_service.set_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            key="to_remove",
            value="remove_me",
            setting_type=SettingType.STRING
        )
        
        # Verify setting exists
        setting = await settings_service.get_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            key="to_remove"
        )
        assert setting is not None
        
        # Remove setting
        removed = await settings_service.remove_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            key="to_remove"
        )
        
        assert removed is True
        
        # Verify setting was removed
        setting = await settings_service.get_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            key="to_remove"
        )
        assert setting is None
    
    @pytest.mark.asyncio
    async def test_bulk_update_settings(self, settings_service, repositories, sample_user):
        """Test bulk updating multiple settings."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy and node
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Test Hierarchy"
        )
        
        node_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.USER,
            name="User Settings"
        )
        
        # Bulk update settings
        settings_data = {
            "string_setting": {"value": "test_string", "type": SettingType.STRING},
            "int_setting": {"value": 42, "type": SettingType.INTEGER},
            "bool_setting": {"value": True, "type": SettingType.BOOLEAN}
        }
        
        result = await settings_service.bulk_update_settings(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            settings_data=settings_data
        )
        
        assert result is True
        
        # Verify all settings were set
        hierarchy = await repositories["settings"].get_hierarchy(hierarchy_id)
        node = hierarchy.nodes[node_id]
        
        assert node.get_setting("string_setting").value == "test_string"
        assert node.get_setting("int_setting").value == 42
        assert node.get_setting("bool_setting").value is True
    
    @pytest.mark.asyncio
    async def test_unauthorized_access_denied(self, settings_service, repositories):
        """Test that unauthorized users cannot access settings."""
        # Create a user without permissions
        unauthorized_user = UserProfile(
            user_id="unauthorized",
            username="noperm",
            permissions=[]  # No permissions
        )
        await repositories["user"].save_user(unauthorized_user)
        
        # Try to create hierarchy - should fail
        with pytest.raises(PermissionError):
            await settings_service.create_hierarchy(
                user_id=unauthorized_user.user_id,
                name="Unauthorized Hierarchy"
            )
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, settings_service, repositories, sample_user):
        """Test that caching works correctly."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy and node
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Test Hierarchy"
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
            key="cached_setting",
            value="cached_value",
            setting_type=SettingType.STRING
        )
        
        # First get should cache the result
        setting1 = await settings_service.get_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            key="cached_setting"
        )
        
        # Second get should use cache (verify by checking cache repository)
        cache_key = f"setting:{hierarchy_id}:{node_id}:cached_setting"
        cached_value = await repositories["cache"].get(cache_key)
        
        assert cached_value is not None
        assert cached_value.value == "cached_value"
    
    @pytest.mark.asyncio
    async def test_audit_trail_creation(self, settings_service, repositories, sample_user):
        """Test that audit trail is properly created."""
        await repositories["user"].save_user(sample_user)
        
        # Create hierarchy
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Audit Test"
        )
        
        # Add node
        node_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.USER,
            name="User Settings"
        )
        
        # Set setting
        await settings_service.set_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            key="audit_setting",
            value="audit_value",
            setting_type=SettingType.STRING
        )
        
        # Check audit entries
        audit_entries = await repositories["audit"].get_user_entries(sample_user.user_id)
        
        # Should have entries for hierarchy creation, node addition, and setting change
        assert len(audit_entries) >= 3
        
        action_types = [entry.action for entry in audit_entries]
        assert "hierarchy.create" in action_types
        assert "node.add" in action_types
        assert "setting.set" in action_types
    
    @pytest.mark.asyncio 
    async def test_nonexistent_hierarchy(self, settings_service, sample_user):
        """Test handling of non-existent hierarchy."""
        fake_hierarchy_id = str(uuid.uuid4())
        
        setting = await settings_service.get_setting(
            user_id=sample_user.user_id,
            hierarchy_id=fake_hierarchy_id,
            node_id="fake_node",
            key="fake_key"
        )
        
        assert setting is None
    
    @pytest.mark.asyncio
    async def test_settings_validation(self, settings_service, repositories, sample_user):
        """Test that settings values are validated before saving."""
        await repositories["user"].save_user(sample_user)
        
        hierarchy_id = await settings_service.create_hierarchy(
            user_id=sample_user.user_id,
            name="Validation Test"
        )
        
        node_id = await settings_service.add_settings_node(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            level=SettingsLevel.USER,
            name="User Settings"
        )
        
        # Test type validation - this should work
        await settings_service.set_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            key="valid_int",
            value=123,
            setting_type=SettingType.INTEGER
        )
        
        # Verify setting was saved
        setting = await settings_service.get_setting(
            user_id=sample_user.user_id,
            hierarchy_id=hierarchy_id,
            node_id=node_id,
            key="valid_int"
        )
        assert setting.value == 123
        assert setting.type == SettingType.INTEGER


if __name__ == "__main__":
    pytest.main([__file__])