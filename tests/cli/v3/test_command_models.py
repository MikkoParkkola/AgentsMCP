"""Tests for CLI v3 command models validation and serialization."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from agentsmcp.cli.v3.models import (
    ExecutionMode,
    SkillLevel,
    CommandStatus,
    ResourceType,
    UserProfile,
    UserPreferences,
    ResourceLimit,
    CommandRequest,
    CommandResult,
    ExecutionMetrics,
    NextAction,
    Suggestion,
    CommandError,
    AuditLogEntry,
)


class TestEnums:
    """Test enum definitions."""
    
    def test_execution_mode_values(self):
        """Test ExecutionMode enum values."""
        assert ExecutionMode.CLI == "cli"
        assert ExecutionMode.TUI == "tui"
        assert ExecutionMode.WEB_UI == "web_ui"
        assert ExecutionMode.API == "api"
    
    def test_skill_level_values(self):
        """Test SkillLevel enum values."""
        assert SkillLevel.BEGINNER == "beginner"
        assert SkillLevel.INTERMEDIATE == "intermediate"
        assert SkillLevel.EXPERT == "expert"
    
    def test_command_status_values(self):
        """Test CommandStatus enum values."""
        assert CommandStatus.PENDING == "pending"
        assert CommandStatus.RUNNING == "running"
        assert CommandStatus.COMPLETED == "completed"
        assert CommandStatus.FAILED == "failed"
        assert CommandStatus.TIMEOUT == "timeout"
        assert CommandStatus.CANCELLED == "cancelled"
    
    def test_resource_type_values(self):
        """Test ResourceType enum values."""
        assert ResourceType.CPU_TIME == "cpu_time"
        assert ResourceType.MEMORY == "memory"
        assert ResourceType.NETWORK == "network"
        assert ResourceType.STORAGE == "storage"
        assert ResourceType.API_CALLS == "api_calls"
        assert ResourceType.TOKEN_COUNT == "token_count"


class TestUserModels:
    """Test user profile and preferences models."""
    
    def test_user_preferences_defaults(self):
        """Test UserPreferences default values."""
        prefs = UserPreferences()
        
        assert prefs.theme == "default"
        assert prefs.verbose_output is False
        assert prefs.auto_confirm is False
        assert prefs.suggestion_level == 3
        assert prefs.default_timeout_ms == 30000
    
    def test_user_preferences_validation(self):
        """Test UserPreferences field validation."""
        # Valid preferences
        prefs = UserPreferences(
            suggestion_level=5,
            default_timeout_ms=60000
        )
        assert prefs.suggestion_level == 5
        
        # Invalid suggestion level
        with pytest.raises(ValidationError):
            UserPreferences(suggestion_level=6)  # Max is 5
        
        with pytest.raises(ValidationError):
            UserPreferences(suggestion_level=-1)  # Min is 0
        
        # Invalid timeout
        with pytest.raises(ValidationError):
            UserPreferences(default_timeout_ms=500)  # Min is 1000
        
        with pytest.raises(ValidationError):
            UserPreferences(default_timeout_ms=400000)  # Max is 300000
    
    def test_user_profile_defaults(self):
        """Test UserProfile default values and validation."""
        profile = UserProfile()
        
        assert profile.user_id is not None  # Auto-generated UUID
        assert profile.skill_level == SkillLevel.INTERMEDIATE
        assert isinstance(profile.preferences, UserPreferences)
        assert profile.command_history == []
        assert profile.favorite_commands == []
        assert isinstance(profile.last_active, datetime)
    
    def test_user_profile_with_data(self):
        """Test UserProfile with provided data."""
        profile = UserProfile(
            user_id="test_user",
            skill_level=SkillLevel.EXPERT,
            command_history=["help", "status", "test"],
            favorite_commands=["help", "status"]
        )
        
        assert profile.user_id == "test_user"
        assert profile.skill_level == SkillLevel.EXPERT
        assert len(profile.command_history) == 3
        assert len(profile.favorite_commands) == 2
    
    def test_user_profile_list_limits(self):
        """Test UserProfile list field limits."""
        # Command history limit (100 items) - should raise validation error
        long_history = [f"command_{i}" for i in range(150)]
        with pytest.raises(ValidationError):
            UserProfile(command_history=long_history)
        
        # Favorite commands limit (20 items) - should raise validation error
        long_favorites = [f"fav_{i}" for i in range(30)]
        with pytest.raises(ValidationError):
            UserProfile(favorite_commands=long_favorites)


class TestResourceModels:
    """Test resource limit models."""
    
    def test_resource_limit_valid(self):
        """Test valid ResourceLimit creation."""
        limit = ResourceLimit(
            resource_type=ResourceType.MEMORY,
            max_value=100,
            warning_threshold=0.8
        )
        
        assert limit.resource_type == ResourceType.MEMORY
        assert limit.max_value == 100
        assert limit.current_usage == 0  # Default
        assert limit.warning_threshold == 0.8
    
    def test_resource_limit_validation(self):
        """Test ResourceLimit validation rules."""
        # Max value must be positive
        with pytest.raises(ValidationError):
            ResourceLimit(
                resource_type=ResourceType.CPU_TIME,
                max_value=0
            )
        
        with pytest.raises(ValidationError):
            ResourceLimit(
                resource_type=ResourceType.CPU_TIME,
                max_value=-10
            )
        
        # Current usage can't be negative
        with pytest.raises(ValidationError):
            ResourceLimit(
                resource_type=ResourceType.MEMORY,
                max_value=100,
                current_usage=-5
            )
        
        # Warning threshold bounds
        with pytest.raises(ValidationError):
            ResourceLimit(
                resource_type=ResourceType.MEMORY,
                max_value=100,
                warning_threshold=0.05  # Below 0.1 minimum
            )
        
        with pytest.raises(ValidationError):
            ResourceLimit(
                resource_type=ResourceType.MEMORY,
                max_value=100,
                warning_threshold=1.5  # Above 1.0 maximum
            )


class TestCommandModels:
    """Test command request and response models."""
    
    def test_command_request_defaults(self):
        """Test CommandRequest default values."""
        request = CommandRequest(command_type="test")
        
        assert request.command_type == "test"
        assert request.request_id is not None
        assert request.args == {}
        assert request.raw_input is None
        assert isinstance(request.timestamp, datetime)
        assert request.priority == 5
        assert request.timeout_ms is None
    
    def test_command_request_validation(self):
        """Test CommandRequest validation."""
        # Command type cannot be empty
        with pytest.raises(ValidationError):
            CommandRequest(command_type="")
        
        # Priority bounds
        with pytest.raises(ValidationError):
            CommandRequest(command_type="test", priority=0)  # Min is 1
        
        with pytest.raises(ValidationError):
            CommandRequest(command_type="test", priority=11)  # Max is 10
        
        # Timeout bounds
        with pytest.raises(ValidationError):
            CommandRequest(command_type="test", timeout_ms=500)  # Min is 1000
        
        with pytest.raises(ValidationError):
            CommandRequest(command_type="test", timeout_ms=700000)  # Max is 600000
    
    def test_suggestion_model(self):
        """Test Suggestion model."""
        suggestion = Suggestion(
            text="Try this command",
            command="help",
            confidence=0.8,
            category="assistance"
        )
        
        assert suggestion.text == "Try this command"
        assert suggestion.command == "help"
        assert suggestion.confidence == 0.8
        assert suggestion.category == "assistance"
        
        # Confidence bounds
        with pytest.raises(ValidationError):
            Suggestion(text="test", confidence=-0.1)
        
        with pytest.raises(ValidationError):
            Suggestion(text="test", confidence=1.1)
    
    def test_next_action_model(self):
        """Test NextAction model."""
        action = NextAction(
            command="status",
            description="Check system status",
            confidence=0.9
        )
        
        assert action.command == "status"
        assert action.description == "Check system status"
        assert action.confidence == 0.9
        assert action.category == "workflow"  # Default
        assert action.estimated_time_ms is None
    
    def test_command_error_model(self):
        """Test CommandError model."""
        error = CommandError(
            error_code="ValidationFailed",
            message="Invalid arguments provided",
            recovery_suggestions=["Check help", "Fix arguments"]
        )
        
        assert error.error_code == "ValidationFailed"
        assert error.message == "Invalid arguments provided"
        assert len(error.recovery_suggestions) == 2
        assert isinstance(error.timestamp, datetime)
    
    def test_execution_metrics_model(self):
        """Test ExecutionMetrics model."""
        metrics = ExecutionMetrics(
            duration_ms=1500,
            tokens_used=100,
            cost_usd=0.05,
            memory_peak_mb=50
        )
        
        assert metrics.duration_ms == 1500
        assert metrics.tokens_used == 100
        assert metrics.cost_usd == 0.05
        assert metrics.cpu_time_ms == 0  # Default
        assert metrics.memory_peak_mb == 50
        
        # Non-negative validation
        with pytest.raises(ValidationError):
            ExecutionMetrics(duration_ms=-100)
        
        with pytest.raises(ValidationError):
            ExecutionMetrics(duration_ms=100, tokens_used=-10)


class TestCommandResult:
    """Test CommandResult model and validation."""
    
    def test_command_result_success(self):
        """Test successful CommandResult creation."""
        result = CommandResult(
            request_id="test_req",
            success=True,
            status=CommandStatus.COMPLETED,
            data={"output": "success"}
        )
        
        assert result.request_id == "test_req"
        assert result.success is True
        assert result.status == CommandStatus.COMPLETED
        assert result.data["output"] == "success"
        assert result.suggestions == []
        assert result.errors == []
        assert result.warnings == []
        assert isinstance(result.timestamp, datetime)
    
    def test_command_result_failure(self):
        """Test failed CommandResult creation."""
        error = CommandError(
            error_code="TestError",
            message="Test error occurred"
        )
        
        result = CommandResult(
            request_id="test_req",
            success=False,
            status=CommandStatus.FAILED,
            errors=[error]
        )
        
        assert result.success is False
        assert result.status == CommandStatus.FAILED
        assert len(result.errors) == 1
        assert result.errors[0].error_code == "TestError"
    
    def test_command_result_success_validation(self):
        """Test success field consistency validation."""
        # Success=True with COMPLETED status should work
        result = CommandResult(
            request_id="test",
            success=True,
            status=CommandStatus.COMPLETED
        )
        assert result.success is True
        
        # Success=False with FAILED status should work
        result = CommandResult(
            request_id="test",
            success=False,
            status=CommandStatus.FAILED
        )
        assert result.success is False
        
        # Success=True with FAILED status should fail
        with pytest.raises(ValidationError):
            CommandResult(
                request_id="test",
                success=True,
                status=CommandStatus.FAILED
            )
        
        # Success=False with COMPLETED status should fail
        with pytest.raises(ValidationError):
            CommandResult(
                request_id="test",
                success=False,
                status=CommandStatus.COMPLETED
            )


class TestAuditLogEntry:
    """Test audit log entry model."""
    
    def test_audit_log_entry_creation(self):
        """Test AuditLogEntry creation."""
        entry = AuditLogEntry(
            request_id="req_123",
            user_id="user_456",
            command_type="test_command",
            execution_mode=ExecutionMode.CLI,
            success=True,
            duration_ms=500
        )
        
        assert entry.entry_id is not None
        assert entry.request_id == "req_123"
        assert entry.user_id == "user_456"
        assert entry.command_type == "test_command"
        assert entry.execution_mode == ExecutionMode.CLI
        assert entry.success is True
        assert entry.duration_ms == 500
        assert isinstance(entry.timestamp, datetime)
        assert entry.resource_usage == {}
        assert entry.metadata == {}
    
    def test_audit_log_entry_with_data(self):
        """Test AuditLogEntry with resource usage and metadata."""
        entry = AuditLogEntry(
            request_id="req_123",
            user_id="user_456", 
            command_type="complex_command",
            execution_mode=ExecutionMode.TUI,
            success=True,
            duration_ms=2000,
            resource_usage={"memory_mb": 50, "cpu_time_s": 0.5},
            metadata={"context": "test", "priority": "high"}
        )
        
        assert entry.resource_usage["memory_mb"] == 50
        assert entry.resource_usage["cpu_time_s"] == 0.5
        assert entry.metadata["context"] == "test"
        assert entry.metadata["priority"] == "high"


class TestModelSerialization:
    """Test model serialization and deserialization."""
    
    def test_user_profile_serialization(self):
        """Test UserProfile JSON serialization."""
        profile = UserProfile(
            user_id="test_user",
            skill_level=SkillLevel.EXPERT,
            command_history=["help", "status"]
        )
        
        # Serialize to dict
        data = profile.model_dump()
        assert data["user_id"] == "test_user"
        assert data["skill_level"] == "expert"
        assert data["command_history"] == ["help", "status"]
        
        # Deserialize from dict
        restored = UserProfile.model_validate(data)
        assert restored.user_id == profile.user_id
        assert restored.skill_level == profile.skill_level
        assert restored.command_history == profile.command_history
    
    def test_command_request_serialization(self):
        """Test CommandRequest JSON serialization."""
        request = CommandRequest(
            command_type="test",
            args={"param1": "value1", "param2": 42},
            priority=3
        )
        
        # Serialize to dict
        data = request.model_dump()
        assert data["command_type"] == "test"
        assert data["args"]["param1"] == "value1"
        assert data["args"]["param2"] == 42
        assert data["priority"] == 3
        
        # Deserialize from dict
        restored = CommandRequest.model_validate(data)
        assert restored.command_type == request.command_type
        assert restored.args == request.args
        assert restored.priority == request.priority
    
    def test_command_result_serialization(self):
        """Test CommandResult JSON serialization."""
        result = CommandResult(
            request_id="test_req",
            success=True,
            status=CommandStatus.COMPLETED,
            data={"output": "success", "count": 5},
            suggestions=[
                Suggestion(text="Next step", confidence=0.8)
            ]
        )
        
        # Serialize to dict
        data = result.model_dump()
        assert data["success"] is True
        assert data["status"] == "completed"
        assert data["data"]["output"] == "success"
        assert len(data["suggestions"]) == 1
        assert data["suggestions"][0]["text"] == "Next step"
        
        # Deserialize from dict
        restored = CommandResult.model_validate(data)
        assert restored.success == result.success
        assert restored.status == result.status
        assert restored.data == result.data
        assert len(restored.suggestions) == 1
        assert restored.suggestions[0].text == result.suggestions[0].text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])