"""
Unit tests for domain entities.
"""

import pytest
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List

from agentsmcp.settings.domain.entities import (
    SettingsNode,
    SettingsHierarchy,
    UserProfile,
    AgentDefinition,
    AgentInstance,
    AuditEntry,
)
from agentsmcp.settings.domain.value_objects import (
    SettingsLevel,
    AgentStatus,
    PermissionLevel,
    SettingType,
    SettingValue,
    ValidationRule,
)


class TestSettingsNode:
    """Test SettingsNode entity."""
    
    def test_node_creation_minimal(self):
        """Test creating node with minimal parameters."""
        node = SettingsNode(
            level=SettingsLevel.USER,
            name="Test Node"
        )
        
        assert node.level == SettingsLevel.USER
        assert node.name == "Test Node"
        assert isinstance(node.id, str)
        assert len(node.id) > 0
        assert node.parent_id is None
        assert node.settings == {}
        assert node.children == []
        assert isinstance(node.created_at, datetime)
        assert isinstance(node.updated_at, datetime)
    
    def test_node_creation_with_parent(self):
        """Test creating node with parent."""
        parent_id = str(uuid.uuid4())
        node = SettingsNode(
            level=SettingsLevel.SESSION,
            name="Child Node",
            parent_id=parent_id
        )
        
        assert node.parent_id == parent_id
        assert node.level == SettingsLevel.SESSION
    
    def test_set_setting(self):
        """Test setting a value in the node."""
        node = SettingsNode(level=SettingsLevel.USER, name="Test")
        
        setting_value = SettingValue(
            value="test_value",
            type=SettingType.STRING,
            source_level=SettingsLevel.USER
        )
        
        node.set_setting("test_key", setting_value)
        
        assert "test_key" in node.settings
        assert node.settings["test_key"] == setting_value
        # updated_at should be modified
        assert node.updated_at >= node.created_at
    
    def test_get_setting_exists(self):
        """Test getting an existing setting."""
        node = SettingsNode(level=SettingsLevel.USER, name="Test")
        
        setting_value = SettingValue(
            value=42,
            type=SettingType.INTEGER,
            source_level=SettingsLevel.USER
        )
        node.set_setting("number", setting_value)
        
        retrieved = node.get_setting("number")
        assert retrieved == setting_value
        assert retrieved.value == 42
    
    def test_get_setting_not_exists(self):
        """Test getting a non-existent setting."""
        node = SettingsNode(level=SettingsLevel.USER, name="Test")
        
        result = node.get_setting("nonexistent")
        assert result is None
    
    def test_remove_setting_exists(self):
        """Test removing an existing setting."""
        node = SettingsNode(level=SettingsLevel.USER, name="Test")
        
        setting_value = SettingValue(
            value="to_remove",
            type=SettingType.STRING,
            source_level=SettingsLevel.USER
        )
        node.set_setting("remove_me", setting_value)
        
        assert "remove_me" in node.settings
        
        removed = node.remove_setting("remove_me")
        assert removed == setting_value
        assert "remove_me" not in node.settings
    
    def test_remove_setting_not_exists(self):
        """Test removing a non-existent setting."""
        node = SettingsNode(level=SettingsLevel.USER, name="Test")
        
        result = node.remove_setting("nonexistent")
        assert result is None
    
    def test_add_child(self):
        """Test adding a child node."""
        parent = SettingsNode(level=SettingsLevel.GLOBAL, name="Parent")
        child_id = str(uuid.uuid4())
        
        parent.add_child(child_id)
        
        assert child_id in parent.children
    
    def test_remove_child(self):
        """Test removing a child node."""
        parent = SettingsNode(level=SettingsLevel.GLOBAL, name="Parent")
        child_id = str(uuid.uuid4())
        
        parent.add_child(child_id)
        assert child_id in parent.children
        
        parent.remove_child(child_id)
        assert child_id not in parent.children
    
    def test_get_all_settings(self):
        """Test getting all settings from node."""
        node = SettingsNode(level=SettingsLevel.USER, name="Test")
        
        setting1 = SettingValue(value="value1", type=SettingType.STRING, source_level=SettingsLevel.USER)
        setting2 = SettingValue(value=123, type=SettingType.INTEGER, source_level=SettingsLevel.USER)
        
        node.set_setting("key1", setting1)
        node.set_setting("key2", setting2)
        
        all_settings = node.get_all_settings()
        
        assert len(all_settings) == 2
        assert all_settings["key1"] == setting1
        assert all_settings["key2"] == setting2
    
    def test_clear_settings(self):
        """Test clearing all settings from node."""
        node = SettingsNode(level=SettingsLevel.USER, name="Test")
        
        node.set_setting("key1", SettingValue(value="val1", type=SettingType.STRING, source_level=SettingsLevel.USER))
        node.set_setting("key2", SettingValue(value="val2", type=SettingType.STRING, source_level=SettingsLevel.USER))
        
        assert len(node.settings) == 2
        
        node.clear_settings()
        
        assert len(node.settings) == 0


class TestSettingsHierarchy:
    """Test SettingsHierarchy entity."""
    
    def test_hierarchy_creation(self):
        """Test creating settings hierarchy."""
        hierarchy = SettingsHierarchy(name="Test Hierarchy")
        
        assert hierarchy.name == "Test Hierarchy"
        assert isinstance(hierarchy.id, str)
        assert len(hierarchy.id) > 0
        assert hierarchy.nodes == {}
        assert hierarchy.root_node_id is None
        assert isinstance(hierarchy.created_at, datetime)
        assert isinstance(hierarchy.updated_at, datetime)
    
    def test_add_node(self):
        """Test adding a node to hierarchy."""
        hierarchy = SettingsHierarchy(name="Test")
        
        node = SettingsNode(level=SettingsLevel.GLOBAL, name="Global")
        
        hierarchy.add_node(node)
        
        assert node.id in hierarchy.nodes
        assert hierarchy.nodes[node.id] == node
    
    def test_remove_node(self):
        """Test removing a node from hierarchy."""
        hierarchy = SettingsHierarchy(name="Test")
        node = SettingsNode(level=SettingsLevel.GLOBAL, name="Global")
        
        hierarchy.add_node(node)
        assert node.id in hierarchy.nodes
        
        removed = hierarchy.remove_node(node.id)
        assert removed == node
        assert node.id not in hierarchy.nodes
    
    def test_remove_nonexistent_node(self):
        """Test removing a non-existent node."""
        hierarchy = SettingsHierarchy(name="Test")
        
        result = hierarchy.remove_node("nonexistent")
        assert result is None
    
    def test_get_node_exists(self):
        """Test getting an existing node."""
        hierarchy = SettingsHierarchy(name="Test")
        node = SettingsNode(level=SettingsLevel.USER, name="User")
        
        hierarchy.add_node(node)
        
        retrieved = hierarchy.get_node(node.id)
        assert retrieved == node
    
    def test_get_node_not_exists(self):
        """Test getting a non-existent node."""
        hierarchy = SettingsHierarchy(name="Test")
        
        result = hierarchy.get_node("nonexistent")
        assert result is None
    
    def test_set_root_node(self):
        """Test setting root node."""
        hierarchy = SettingsHierarchy(name="Test")
        root = SettingsNode(level=SettingsLevel.GLOBAL, name="Root")
        
        hierarchy.add_node(root)
        hierarchy.root_node_id = root.id
        
        assert hierarchy.root_node_id == root.id
    
    def test_resolve_setting_single_node(self):
        """Test resolving setting from single node."""
        hierarchy = SettingsHierarchy(name="Test")
        node = SettingsNode(level=SettingsLevel.USER, name="User")
        
        setting_value = SettingValue(
            value="test_value",
            type=SettingType.STRING,
            source_level=SettingsLevel.USER
        )
        node.set_setting("test_key", setting_value)
        
        hierarchy.add_node(node)
        
        resolved = hierarchy.resolve_setting(node.id, "test_key")
        assert resolved == setting_value
    
    def test_resolve_setting_with_inheritance(self):
        """Test resolving setting with inheritance from parent."""
        hierarchy = SettingsHierarchy(name="Test")
        
        # Create parent node with setting
        parent = SettingsNode(level=SettingsLevel.GLOBAL, name="Global")
        parent_setting = SettingValue(
            value="parent_value",
            type=SettingType.STRING,
            source_level=SettingsLevel.GLOBAL
        )
        parent.set_setting("inherited_key", parent_setting)
        
        # Create child node without the setting
        child = SettingsNode(level=SettingsLevel.USER, name="User", parent_id=parent.id)
        
        # Set up hierarchy
        hierarchy.add_node(parent)
        hierarchy.add_node(child)
        parent.add_child(child.id)
        
        # Should resolve from parent
        resolved = hierarchy.resolve_setting(child.id, "inherited_key")
        assert resolved == parent_setting
    
    def test_resolve_setting_child_overrides_parent(self):
        """Test child setting overrides parent setting."""
        hierarchy = SettingsHierarchy(name="Test")
        
        # Create parent node with setting
        parent = SettingsNode(level=SettingsLevel.GLOBAL, name="Global")
        parent_setting = SettingValue(
            value="parent_value",
            type=SettingType.STRING,
            source_level=SettingsLevel.GLOBAL
        )
        parent.set_setting("override_key", parent_setting)
        
        # Create child node with override
        child = SettingsNode(level=SettingsLevel.USER, name="User", parent_id=parent.id)
        child_setting = SettingValue(
            value="child_value",
            type=SettingType.STRING,
            source_level=SettingsLevel.USER
        )
        child.set_setting("override_key", child_setting)
        
        # Set up hierarchy
        hierarchy.add_node(parent)
        hierarchy.add_node(child)
        parent.add_child(child.id)
        
        # Should resolve child value (override)
        resolved = hierarchy.resolve_setting(child.id, "override_key")
        assert resolved == child_setting
        assert resolved.value == "child_value"
    
    def test_resolve_nonexistent_setting(self):
        """Test resolving non-existent setting."""
        hierarchy = SettingsHierarchy(name="Test")
        node = SettingsNode(level=SettingsLevel.USER, name="User")
        
        hierarchy.add_node(node)
        
        result = hierarchy.resolve_setting(node.id, "nonexistent")
        assert result is None
    
    def test_get_effective_settings(self):
        """Test getting all effective settings for a node."""
        hierarchy = SettingsHierarchy(name="Test")
        
        # Create parent with settings
        parent = SettingsNode(level=SettingsLevel.GLOBAL, name="Global")
        parent.set_setting("parent_only", SettingValue("parent_val", SettingType.STRING, SettingsLevel.GLOBAL))
        parent.set_setting("shared_key", SettingValue("parent_shared", SettingType.STRING, SettingsLevel.GLOBAL))
        
        # Create child with settings
        child = SettingsNode(level=SettingsLevel.USER, name="User", parent_id=parent.id)
        child.set_setting("child_only", SettingValue("child_val", SettingType.STRING, SettingsLevel.USER))
        child.set_setting("shared_key", SettingValue("child_shared", SettingType.STRING, SettingsLevel.USER))
        
        # Set up hierarchy
        hierarchy.add_node(parent)
        hierarchy.add_node(child)
        parent.add_child(child.id)
        
        effective = hierarchy.get_effective_settings(child.id)
        
        assert len(effective) == 3
        assert effective["parent_only"].value == "parent_val"
        assert effective["child_only"].value == "child_val"
        assert effective["shared_key"].value == "child_shared"  # Child overrides parent


class TestUserProfile:
    """Test UserProfile entity."""
    
    def test_user_creation_minimal(self):
        """Test creating user with minimal parameters."""
        user = UserProfile(
            user_id="user123",
            username="testuser"
        )
        
        assert user.user_id == "user123"
        assert user.username == "testuser"
        assert user.email is None
        assert user.preferences == {}
        assert user.permissions == []
        assert isinstance(user.created_at, datetime)
        assert isinstance(user.last_login, datetime)
        assert not user.is_admin
    
    def test_user_creation_full(self):
        """Test creating user with all parameters."""
        preferences = {"theme": "dark", "language": "en"}
        permissions = [PermissionLevel.READ, PermissionLevel.WRITE]
        
        user = UserProfile(
            user_id="user456",
            username="fulluser",
            email="user@example.com",
            preferences=preferences,
            permissions=permissions,
            is_admin=True
        )
        
        assert user.email == "user@example.com"
        assert user.preferences == preferences
        assert user.permissions == permissions
        assert user.is_admin is True
    
    def test_set_preference(self):
        """Test setting user preference."""
        user = UserProfile(user_id="user123", username="test")
        
        user.set_preference("theme", "light")
        
        assert user.preferences["theme"] == "light"
    
    def test_get_preference_exists(self):
        """Test getting existing preference."""
        user = UserProfile(user_id="user123", username="test")
        user.set_preference("language", "es")
        
        result = user.get_preference("language")
        assert result == "es"
    
    def test_get_preference_not_exists(self):
        """Test getting non-existent preference."""
        user = UserProfile(user_id="user123", username="test")
        
        result = user.get_preference("nonexistent")
        assert result is None
    
    def test_add_permission(self):
        """Test adding permission to user."""
        user = UserProfile(user_id="user123", username="test")
        
        user.add_permission(PermissionLevel.WRITE)
        
        assert PermissionLevel.WRITE in user.permissions
    
    def test_remove_permission(self):
        """Test removing permission from user."""
        user = UserProfile(user_id="user123", username="test", permissions=[PermissionLevel.READ, PermissionLevel.WRITE])
        
        user.remove_permission(PermissionLevel.WRITE)
        
        assert PermissionLevel.WRITE not in user.permissions
        assert PermissionLevel.READ in user.permissions
    
    def test_has_permission(self):
        """Test checking if user has permission."""
        user = UserProfile(user_id="user123", username="test", permissions=[PermissionLevel.READ])
        
        assert user.has_permission(PermissionLevel.READ)
        assert not user.has_permission(PermissionLevel.WRITE)
    
    def test_update_last_login(self):
        """Test updating last login time."""
        user = UserProfile(user_id="user123", username="test")
        original_login = user.last_login
        
        # Small delay to ensure time difference
        import time
        time.sleep(0.001)
        
        user.update_last_login()
        
        assert user.last_login > original_login


class TestAgentDefinition:
    """Test AgentDefinition entity."""
    
    def test_agent_definition_creation_minimal(self):
        """Test creating agent definition with minimal parameters."""
        agent_def = AgentDefinition(
            name="TestAgent",
            description="A test agent",
            user_id="user123"
        )
        
        assert agent_def.name == "TestAgent"
        assert agent_def.description == "A test agent"
        assert agent_def.user_id == "user123"
        assert isinstance(agent_def.id, str)
        assert agent_def.configuration == {}
        assert agent_def.validation_rules == []
        assert agent_def.status == AgentStatus.DRAFT
        assert isinstance(agent_def.created_at, datetime)
    
    def test_agent_definition_creation_full(self):
        """Test creating agent definition with all parameters."""
        config = {"model": "gpt-4", "temperature": 0.7}
        rules = [ValidationRule("required", True, "Name is required")]
        
        agent_def = AgentDefinition(
            name="FullAgent",
            description="Full agent definition",
            user_id="user456",
            configuration=config,
            validation_rules=rules,
            status=AgentStatus.ACTIVE
        )
        
        assert agent_def.configuration == config
        assert agent_def.validation_rules == rules
        assert agent_def.status == AgentStatus.ACTIVE
    
    def test_update_configuration(self):
        """Test updating agent configuration."""
        agent_def = AgentDefinition(
            name="TestAgent",
            description="Test",
            user_id="user123"
        )
        
        new_config = {"model": "claude-3", "max_tokens": 1000}
        agent_def.update_configuration(new_config)
        
        assert agent_def.configuration == new_config
        assert agent_def.updated_at > agent_def.created_at
    
    def test_add_validation_rule(self):
        """Test adding validation rule to agent."""
        agent_def = AgentDefinition(
            name="TestAgent",
            description="Test",
            user_id="user123"
        )
        
        rule = ValidationRule("pattern", r"^\w+$", "Must be alphanumeric")
        agent_def.add_validation_rule(rule)
        
        assert rule in agent_def.validation_rules
    
    def test_update_status(self):
        """Test updating agent status."""
        agent_def = AgentDefinition(
            name="TestAgent",
            description="Test",
            user_id="user123",
            status=AgentStatus.DRAFT
        )
        
        agent_def.update_status(AgentStatus.TESTING)
        
        assert agent_def.status == AgentStatus.TESTING
        assert agent_def.updated_at > agent_def.created_at


class TestAgentInstance:
    """Test AgentInstance entity."""
    
    def test_agent_instance_creation(self):
        """Test creating agent instance."""
        instance = AgentInstance(
            agent_definition_id="def123",
            user_id="user456",
            session_id="session789"
        )
        
        assert instance.agent_definition_id == "def123"
        assert instance.user_id == "user456" 
        assert instance.session_id == "session789"
        assert isinstance(instance.id, str)
        assert instance.runtime_config == {}
        assert instance.status == AgentStatus.ACTIVE
        assert instance.performance_metrics == {}
        assert isinstance(instance.created_at, datetime)
    
    def test_update_runtime_config(self):
        """Test updating runtime configuration."""
        instance = AgentInstance("def123", "user456", "session789")
        
        new_config = {"current_task": "analysis", "context": "document"}
        instance.update_runtime_config(new_config)
        
        assert instance.runtime_config == new_config
    
    def test_update_performance_metrics(self):
        """Test updating performance metrics."""
        instance = AgentInstance("def123", "user456", "session789")
        
        metrics = {"response_time": 1.5, "accuracy": 0.95}
        instance.update_performance_metrics(metrics)
        
        assert instance.performance_metrics == metrics
    
    def test_update_status(self):
        """Test updating instance status."""
        instance = AgentInstance("def123", "user456", "session789")
        
        instance.update_status(AgentStatus.INACTIVE)
        
        assert instance.status == AgentStatus.INACTIVE


class TestAuditEntry:
    """Test AuditEntry entity."""
    
    def test_audit_entry_creation(self):
        """Test creating audit entry."""
        entry = AuditEntry(
            user_id="user123",
            action="settings.update",
            resource_type="settings_node",
            resource_id="node456",
            details={"key": "test_key", "old_value": "old", "new_value": "new"}
        )
        
        assert entry.user_id == "user123"
        assert entry.action == "settings.update"
        assert entry.resource_type == "settings_node"
        assert entry.resource_id == "node456"
        assert entry.details["key"] == "test_key"
        assert isinstance(entry.id, str)
        assert isinstance(entry.timestamp, datetime)
    
    def test_audit_entry_with_metadata(self):
        """Test creating audit entry with metadata."""
        metadata = {"ip_address": "192.168.1.1", "user_agent": "test-client"}
        
        entry = AuditEntry(
            user_id="user123",
            action="agent.create",
            resource_type="agent_definition",
            resource_id="agent789",
            metadata=metadata
        )
        
        assert entry.metadata == metadata
    
    def test_audit_entry_minimal(self):
        """Test creating audit entry with minimal parameters."""
        entry = AuditEntry(
            user_id="user123",
            action="user.login",
            resource_type="user_profile",
            resource_id="user123"
        )
        
        assert entry.details == {}
        assert entry.metadata == {}


if __name__ == "__main__":
    pytest.main([__file__])