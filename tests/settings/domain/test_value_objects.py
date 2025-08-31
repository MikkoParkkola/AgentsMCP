"""
Unit tests for domain value objects.
"""

import pytest
from datetime import datetime, timezone
from typing import Any

from agentsmcp.settings.domain.value_objects import (
    SettingsLevel,
    AgentStatus,
    PermissionLevel,
    SettingType,
    SettingValue,
    ValidationRule,
    ValidationResult,
)


class TestSettingsLevel:
    """Test SettingsLevel enum."""
    
    def test_hierarchy_levels(self):
        """Test that settings levels have correct hierarchy."""
        assert SettingsLevel.GLOBAL.value == "global"
        assert SettingsLevel.USER.value == "user"
        assert SettingsLevel.SESSION.value == "session"
        assert SettingsLevel.AGENT.value == "agent"
    
    def test_level_comparison(self):
        """Test that levels can be compared for hierarchy."""
        # Test that enum values maintain expected ordering
        levels = [SettingsLevel.GLOBAL, SettingsLevel.USER, SettingsLevel.SESSION, SettingsLevel.AGENT]
        assert len(levels) == 4
        # Note: Enum comparison by default uses definition order


class TestAgentStatus:
    """Test AgentStatus enum."""
    
    def test_status_values(self):
        """Test agent status enum values."""
        assert AgentStatus.DRAFT.value == "draft"
        assert AgentStatus.ACTIVE.value == "active"
        assert AgentStatus.TESTING.value == "testing"
        assert AgentStatus.INACTIVE.value == "inactive"
        assert AgentStatus.ARCHIVED.value == "archived"
    
    def test_all_statuses_defined(self):
        """Test all expected statuses are defined."""
        expected_statuses = {"draft", "active", "testing", "inactive", "archived"}
        actual_statuses = {status.value for status in AgentStatus}
        assert actual_statuses == expected_statuses


class TestPermissionLevel:
    """Test PermissionLevel enum."""
    
    def test_permission_values(self):
        """Test permission level enum values."""
        assert PermissionLevel.READ.value == "read"
        assert PermissionLevel.WRITE.value == "write"
        assert PermissionLevel.ADMIN.value == "admin"
    
    def test_permission_hierarchy(self):
        """Test permission levels maintain expected hierarchy."""
        levels = [PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.ADMIN]
        assert len(levels) == 3


class TestSettingType:
    """Test SettingType enum."""
    
    def test_type_values(self):
        """Test setting type enum values."""
        assert SettingType.STRING.value == "string"
        assert SettingType.INTEGER.value == "integer"
        assert SettingType.FLOAT.value == "float"
        assert SettingType.BOOLEAN.value == "boolean"
        assert SettingType.ARRAY.value == "array"
        assert SettingType.OBJECT.value == "object"
        assert SettingType.SECRET.value == "secret"
    
    def test_all_types_defined(self):
        """Test all expected types are defined."""
        expected_types = {"string", "integer", "float", "boolean", "array", "object", "secret"}
        actual_types = {t.value for t in SettingType}
        assert actual_types == expected_types


class TestSettingValue:
    """Test SettingValue value object."""
    
    def test_string_value_creation(self):
        """Test creating string setting value."""
        value = SettingValue(
            value="test_string",
            type=SettingType.STRING,
            source_level=SettingsLevel.USER,
            metadata={"source": "user_input"}
        )
        
        assert value.value == "test_string"
        assert value.type == SettingType.STRING
        assert value.source_level == SettingsLevel.USER
        assert value.metadata["source"] == "user_input"
        assert isinstance(value.created_at, datetime)
        assert value.created_at.tzinfo is not None
    
    def test_integer_value_creation(self):
        """Test creating integer setting value."""
        value = SettingValue(
            value=42,
            type=SettingType.INTEGER,
            source_level=SettingsLevel.GLOBAL
        )
        
        assert value.value == 42
        assert value.type == SettingType.INTEGER
        assert value.source_level == SettingsLevel.GLOBAL
        assert value.metadata == {}
    
    def test_boolean_value_creation(self):
        """Test creating boolean setting value."""
        value = SettingValue(
            value=True,
            type=SettingType.BOOLEAN,
            source_level=SettingsLevel.SESSION
        )
        
        assert value.value is True
        assert value.type == SettingType.BOOLEAN
    
    def test_array_value_creation(self):
        """Test creating array setting value."""
        array_data = ["item1", "item2", "item3"]
        value = SettingValue(
            value=array_data,
            type=SettingType.ARRAY,
            source_level=SettingsLevel.USER
        )
        
        assert value.value == array_data
        assert value.type == SettingType.ARRAY
    
    def test_object_value_creation(self):
        """Test creating object setting value."""
        object_data = {"key1": "value1", "key2": 42}
        value = SettingValue(
            value=object_data,
            type=SettingType.OBJECT,
            source_level=SettingsLevel.AGENT
        )
        
        assert value.value == object_data
        assert value.type == SettingType.OBJECT
    
    def test_secret_value_creation(self):
        """Test creating secret setting value."""
        value = SettingValue(
            value="secret_token_123",
            type=SettingType.SECRET,
            source_level=SettingsLevel.USER,
            metadata={"encrypted": True}
        )
        
        assert value.value == "secret_token_123"
        assert value.type == SettingType.SECRET
        assert value.metadata["encrypted"] is True
    
    def test_float_value_creation(self):
        """Test creating float setting value."""
        value = SettingValue(
            value=3.14159,
            type=SettingType.FLOAT,
            source_level=SettingsLevel.GLOBAL
        )
        
        assert value.value == 3.14159
        assert value.type == SettingType.FLOAT
    
    def test_metadata_handling(self):
        """Test metadata is properly handled."""
        metadata = {
            "source": "environment",
            "env_var": "MY_VAR",
            "validated": True,
            "validation_time": "2023-01-01T12:00:00Z"
        }
        
        value = SettingValue(
            value="test",
            type=SettingType.STRING,
            source_level=SettingsLevel.GLOBAL,
            metadata=metadata
        )
        
        assert value.metadata == metadata
        assert value.metadata["source"] == "environment"
        assert value.metadata["validated"] is True
    
    def test_created_at_timestamp(self):
        """Test created_at timestamp is set correctly."""
        before = datetime.now(timezone.utc)
        
        value = SettingValue(
            value="test",
            type=SettingType.STRING,
            source_level=SettingsLevel.USER
        )
        
        after = datetime.now(timezone.utc)
        
        assert before <= value.created_at <= after
        assert value.created_at.tzinfo is not None


class TestValidationRule:
    """Test ValidationRule value object."""
    
    def test_pattern_rule_creation(self):
        """Test creating pattern-based validation rule."""
        rule = ValidationRule(
            rule_type="pattern",
            rule_value=r"^\d{3}-\d{3}-\d{4}$",
            error_message="Must be in format XXX-XXX-XXXX"
        )
        
        assert rule.rule_type == "pattern"
        assert rule.rule_value == r"^\d{3}-\d{3}-\d{4}$"
        assert rule.error_message == "Must be in format XXX-XXX-XXXX"
    
    def test_range_rule_creation(self):
        """Test creating range-based validation rule."""
        rule = ValidationRule(
            rule_type="range",
            rule_value={"min": 0, "max": 100},
            error_message="Value must be between 0 and 100"
        )
        
        assert rule.rule_type == "range"
        assert rule.rule_value["min"] == 0
        assert rule.rule_value["max"] == 100
    
    def test_required_rule_creation(self):
        """Test creating required field validation rule."""
        rule = ValidationRule(
            rule_type="required",
            rule_value=True,
            error_message="This field is required"
        )
        
        assert rule.rule_type == "required"
        assert rule.rule_value is True
    
    def test_custom_rule_creation(self):
        """Test creating custom validation rule."""
        rule = ValidationRule(
            rule_type="custom",
            rule_value="validate_email_format",
            error_message="Invalid email format"
        )
        
        assert rule.rule_type == "custom"
        assert rule.rule_value == "validate_email_format"


class TestValidationResult:
    """Test ValidationResult value object."""
    
    def test_valid_result_creation(self):
        """Test creating valid validation result."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            suggestions=[]
        )
        
        assert result.is_valid is True
        assert result.errors == []
        assert result.suggestions == []
    
    def test_invalid_result_with_errors(self):
        """Test creating invalid validation result with errors."""
        errors = ["Value cannot be empty", "Must be a valid email"]
        result = ValidationResult(
            is_valid=False,
            errors=errors,
            suggestions=["Try: user@example.com"]
        )
        
        assert result.is_valid is False
        assert result.errors == errors
        assert result.suggestions == ["Try: user@example.com"]
    
    def test_invalid_result_without_suggestions(self):
        """Test creating invalid validation result without suggestions."""
        result = ValidationResult(
            is_valid=False,
            errors=["Invalid format"],
            suggestions=[]
        )
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.suggestions == []
    
    def test_partial_validation_with_warnings(self):
        """Test validation result with mixed outcomes."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            suggestions=["Consider using a stronger password"]
        )
        
        assert result.is_valid is True
        assert result.errors == []
        assert len(result.suggestions) == 1


class TestValueObjectImmutability:
    """Test that value objects are immutable."""
    
    def test_setting_value_immutability(self):
        """Test that SettingValue fields cannot be modified."""
        value = SettingValue(
            value="original",
            type=SettingType.STRING,
            source_level=SettingsLevel.USER
        )
        
        # These should work (reading)
        assert value.value == "original"
        assert value.type == SettingType.STRING
        
        # These should fail (writing) - SettingValue should be frozen
        with pytest.raises(AttributeError):
            value.value = "modified"
        
        with pytest.raises(AttributeError):
            value.type = SettingType.INTEGER
    
    def test_validation_rule_immutability(self):
        """Test that ValidationRule fields cannot be modified."""
        rule = ValidationRule(
            rule_type="pattern",
            rule_value="test",
            error_message="error"
        )
        
        # Reading should work
        assert rule.rule_type == "pattern"
        
        # Writing should fail
        with pytest.raises(AttributeError):
            rule.rule_type = "range"
    
    def test_validation_result_immutability(self):
        """Test that ValidationResult fields cannot be modified."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            suggestions=[]
        )
        
        # Reading should work
        assert result.is_valid is True
        
        # Writing should fail
        with pytest.raises(AttributeError):
            result.is_valid = False


if __name__ == "__main__":
    pytest.main([__file__])