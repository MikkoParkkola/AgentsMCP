"""Tests for ExecutionContext functionality."""

import asyncio
import pytest
from datetime import datetime, timezone

from agentsmcp.cli.v3.core import ExecutionContext
from agentsmcp.cli.v3.models import (
    ExecutionMode,
    UserProfile,
    UserPreferences,
    SkillLevel,
    ResourceLimit,
    ResourceType,
    PermissionDeniedError,
    ResourceExhaustedError,
)


@pytest.fixture
def user_profile():
    """Create a test user profile."""
    return UserProfile(
        user_id="test_user",
        skill_level=SkillLevel.INTERMEDIATE,
        preferences=UserPreferences(
            verbose_output=True,
            suggestion_level=3
        )
    )


@pytest.fixture
def resource_limits():
    """Create test resource limits."""
    return {
        ResourceType.CPU_TIME: ResourceLimit(
            resource_type=ResourceType.CPU_TIME,
            max_value=1.0,  # 1 second
            warning_threshold=0.8
        ),
        ResourceType.MEMORY: ResourceLimit(
            resource_type=ResourceType.MEMORY,
            max_value=100,  # 100MB
            warning_threshold=0.8
        )
    }


class TestExecutionContext:
    """Test ExecutionContext functionality."""
    
    def test_context_initialization(self, user_profile):
        """Test basic context initialization."""
        context = ExecutionContext(user_profile, ExecutionMode.CLI)
        
        assert context.user_profile == user_profile
        assert context.execution_mode == ExecutionMode.CLI
        assert context.context_id is not None
        assert context.created_at is not None
        assert not context.is_active
        assert context.current_command is None
    
    def test_capabilities_by_execution_mode(self, user_profile):
        """Test capability reporting based on execution mode."""
        # CLI mode
        cli_context = ExecutionContext(user_profile, ExecutionMode.CLI)
        cli_caps = cli_context.capabilities
        assert "text_output" in cli_caps
        assert "error_reporting" in cli_caps
        
        # TUI mode
        tui_context = ExecutionContext(user_profile, ExecutionMode.TUI)
        tui_caps = tui_context.capabilities
        assert "interactive_input" in tui_caps
        assert "real_time_updates" in tui_caps
        assert "visual_feedback" in tui_caps
        
        # Web UI mode
        web_context = ExecutionContext(user_profile, ExecutionMode.WEB_UI)
        web_caps = web_context.capabilities
        assert "rich_formatting" in web_caps
        assert "file_upload" in web_caps
        
        # API mode
        api_context = ExecutionContext(user_profile, ExecutionMode.API)
        api_caps = api_context.capabilities
        assert "json_response" in api_caps
        assert "structured_data" in api_caps


class TestPermissionManager:
    """Test permission management."""
    
    def test_default_permissions_by_skill_level(self, user_profile):
        """Test default permissions are set based on skill level."""
        # Beginner user
        beginner_profile = UserProfile(skill_level=SkillLevel.BEGINNER)
        beginner_context = ExecutionContext(beginner_profile)
        
        assert beginner_context.check_permission("command.help")
        assert beginner_context.check_permission("file.read")
        assert not beginner_context.check_permission("file.write")
        assert not beginner_context.check_permission("system.admin")
        
        # Expert user
        expert_profile = UserProfile(skill_level=SkillLevel.EXPERT)
        expert_context = ExecutionContext(expert_profile)
        
        assert expert_context.check_permission("command.help")
        assert expert_context.check_permission("file.write")
        assert expert_context.check_permission("system.admin")
        assert expert_context.check_permission("network.access")
    
    def test_permission_grant_deny(self, user_profile):
        """Test granting and denying permissions."""
        context = ExecutionContext(user_profile)
        
        # Initially should not have dangerous command permission
        assert not context.check_permission("command.dangerous")
        
        # Grant permission
        context.permissions.grant_permission("command.dangerous")
        assert context.check_permission("command.dangerous")
        
        # Deny permission
        context.permissions.deny_permission("command.dangerous")
        assert not context.check_permission("command.dangerous")
    
    def test_require_permission_success(self, user_profile):
        """Test successful permission requirement."""
        context = ExecutionContext(user_profile)
        
        # Should not raise for existing permission
        context.require_permission("command.help")
    
    def test_require_permission_failure(self, user_profile):
        """Test permission requirement failure."""
        context = ExecutionContext(user_profile)
        
        # Should raise for missing permission
        with pytest.raises(PermissionDeniedError) as exc_info:
            context.require_permission("system.admin")
        
        assert "system.admin" in str(exc_info.value)
        assert user_profile.skill_level.value in str(exc_info.value)


class TestSessionState:
    """Test session state management."""
    
    @pytest.mark.asyncio
    async def test_session_variables(self, user_profile):
        """Test session variable management."""
        context = ExecutionContext(user_profile)
        
        # Set and get variable
        await context.session.set_variable("test_var", "test_value")
        value = await context.session.get_variable("test_var")
        assert value == "test_value"
        
        # Get non-existent variable with default
        default_value = await context.session.get_variable("missing_var", "default")
        assert default_value == "default"
    
    @pytest.mark.asyncio
    async def test_context_stack(self, user_profile):
        """Test context stack management."""
        context = ExecutionContext(user_profile)
        
        # Push context frames
        await context.session.push_context({"key1": "value1"})
        await context.session.push_context({"key2": "value2"})
        
        # Get merged context
        merged = await context.session.get_current_context()
        assert merged["key1"] == "value1"
        assert merged["key2"] == "value2"
        
        # Pop context
        popped = await context.session.pop_context()
        assert popped == {"key2": "value2"}
        
        # Remaining context
        remaining = await context.session.get_current_context()
        assert remaining == {"key1": "value1"}
        assert "key2" not in remaining


class TestResourceMonitoring:
    """Test resource monitoring and limits."""
    
    def test_resource_monitor_initialization(self, user_profile, resource_limits):
        """Test resource monitor initialization."""
        context = ExecutionContext(
            user_profile,
            resource_limits=resource_limits
        )
        
        assert ResourceType.CPU_TIME in context.resource_monitor.limits
        assert ResourceType.MEMORY in context.resource_monitor.limits
        
        cpu_limit = context.resource_monitor.limits[ResourceType.CPU_TIME]
        assert cpu_limit.max_value == 1.0
        assert cpu_limit.warning_threshold == 0.8
    
    def test_resource_monitoring_lifecycle(self, user_profile, resource_limits):
        """Test resource monitoring start/stop."""
        context = ExecutionContext(
            user_profile,
            resource_limits=resource_limits
        )
        
        # Start monitoring
        context.resource_monitor.start_monitoring()
        assert context.resource_monitor._monitoring is True
        assert context.resource_monitor.start_time is not None
        
        # Get usage (should not raise)
        usage = context.resource_monitor.get_usage()
        assert isinstance(usage, dict)
        
        # Stop monitoring
        final_usage = context.resource_monitor.stop_monitoring()
        assert context.resource_monitor._monitoring is False
        assert isinstance(final_usage, dict)


class TestSkillLevelAdaptation:
    """Test skill level-based content adaptation."""
    
    def test_beginner_content_adaptation(self):
        """Test content adaptation for beginners."""
        beginner_profile = UserProfile(skill_level=SkillLevel.BEGINNER)
        context = ExecutionContext(beginner_profile)
        
        # Test suggestion limiting and explanation addition
        content = {
            "suggestions": [
                {"description": "First suggestion"},
                {"description": "Second suggestion"}, 
                {"description": "Third suggestion"},
                {"description": "Fourth suggestion"},
            ]
        }
        
        adapted = context.adapt_for_skill_level(content)
        
        # Should limit suggestions for beginners
        assert len(adapted["suggestions"]) <= 3
        
        # Should add explanations
        for suggestion in adapted["suggestions"]:
            assert "explanation" in suggestion
    
    def test_expert_content_adaptation(self):
        """Test content adaptation for experts."""
        expert_profile = UserProfile(skill_level=SkillLevel.EXPERT)
        context = ExecutionContext(expert_profile)
        
        content = {"data": "test"}
        adapted = context.adapt_for_skill_level(content)
        
        # Should add technical details for experts
        assert adapted["advanced_options"] is True
        assert adapted["technical_details"] is True


class TestCommandExecutionContext:
    """Test command execution context management."""
    
    @pytest.mark.asyncio
    async def test_command_execution_context_manager(self, user_profile):
        """Test command execution context manager."""
        context = ExecutionContext(user_profile)
        
        assert context.current_command is None
        assert not context.is_active
        
        async with context.command_execution("test_command"):
            assert context.current_command == "test_command"
            assert context.is_active
            
            # Simulate some work
            await asyncio.sleep(0.01)
        
        # Context should be cleaned up
        assert context.current_command is None
        assert not context.is_active
    
    @pytest.mark.asyncio
    async def test_command_execution_with_timeout(self, user_profile):
        """Test command execution context manager (timeout handling moved to command engine)."""
        context = ExecutionContext(user_profile)
        
        # Test that context manager works normally - timeout handling is now in CommandEngine
        async with context.command_execution("slow_command"):
            # Sleep briefly
            await asyncio.sleep(0.01)
        
        # Context should be cleaned up
        assert context.current_command is None
        assert not context.is_active
    
    @pytest.mark.asyncio
    async def test_audit_trail_generation(self, user_profile):
        """Test audit trail generation during command execution."""
        context = ExecutionContext(user_profile)
        
        # Should start with empty audit trail
        audit_entries = context.get_audit_trail()
        assert len(audit_entries) == 0
        
        # Execute a command
        async with context.command_execution("test_command"):
            await asyncio.sleep(0.01)
        
        # Should have audit entries
        audit_entries = context.get_audit_trail()
        assert len(audit_entries) >= 1
        
        # Check audit entry structure
        entry = audit_entries[-1]
        assert entry["event"] == "command_completed"
        assert entry["command"] == "test_command" 
        assert entry["context_id"] == context.context_id
        assert entry["user_id"] == user_profile.user_id
        assert "duration_ms" in entry
        assert "timestamp" in entry


class TestContextInformation:
    """Test context information and debugging support."""
    
    def test_context_info_completeness(self, user_profile):
        """Test comprehensive context information."""
        context = ExecutionContext(user_profile, ExecutionMode.TUI)
        
        info = context.get_context_info()
        
        # Verify all expected fields are present
        expected_fields = [
            "context_id", "user_id", "skill_level", "execution_mode",
            "capabilities", "is_active", "current_command", "session_id",
            "created_at", "permissions_count", "audit_entries_count"
        ]
        
        for field in expected_fields:
            assert field in info
        
        # Verify field values
        assert info["user_id"] == user_profile.user_id
        assert info["skill_level"] == user_profile.skill_level.value
        assert info["execution_mode"] == ExecutionMode.TUI.value
        assert isinstance(info["capabilities"], list)
        assert len(info["capabilities"]) > 0
        assert not info["is_active"]
        assert info["permissions_count"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])