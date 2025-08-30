"""Tests for registry models and data structures."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from src.agentsmcp.cli.v3.models.registry_models import (
    CommandDefinition,
    CommandMetadata,
    ParameterDefinition,
    ParameterType,
    CommandCategory,
    SecurityLevel,
    SkillLevel,
    DiscoveryRequest,
    ValidationRequest,
    ValidationResult,
    ValidationIssue,
    ValidationConstraint,
    CommandExample,
    CommandDependency,
)
from src.agentsmcp.cli.v3.models.command_models import ExecutionMode


class TestParameterDefinition:
    """Test parameter definition validation."""
    
    def test_valid_parameter_creation(self):
        """Test creating a valid parameter definition."""
        param = ParameterDefinition(
            name="test_param",
            type=ParameterType.STRING,
            description="A test parameter",
            required=True,
            examples=["example1", "example2"],
            aliases=["tp", "test"]
        )
        
        assert param.name == "test_param"
        assert param.type == ParameterType.STRING
        assert param.description == "A test parameter"
        assert param.required is True
        assert len(param.examples) == 2
        assert len(param.aliases) == 2
    
    def test_invalid_parameter_name(self):
        """Test that invalid parameter names are rejected."""
        with pytest.raises(ValueError, match="Parameter name must contain only"):
            ParameterDefinition(
                name="test param!",  # Invalid characters
                type=ParameterType.STRING,
                description="A test parameter"
            )
    
    def test_parameter_with_constraints(self):
        """Test parameter with validation constraints."""
        constraint = ValidationConstraint(
            type="min_length",
            value=5,
            message="Must be at least 5 characters"
        )
        
        param = ParameterDefinition(
            name="string_param",
            type=ParameterType.STRING,
            description="A string parameter with constraints",
            constraints=[constraint]
        )
        
        assert len(param.constraints) == 1
        assert param.constraints[0].type == "min_length"
        assert param.constraints[0].value == 5


class TestCommandDefinition:
    """Test command definition validation."""
    
    def test_minimal_command_definition(self):
        """Test creating a minimal valid command definition."""
        cmd = CommandDefinition(
            name="test_command",
            description="A test command"
        )
        
        assert cmd.name == "test_command"
        assert cmd.description == "A test command"
        assert cmd.category == CommandCategory.CORE  # Default
        assert cmd.security_level == SecurityLevel.SAFE  # Default
        assert ExecutionMode.CLI in cmd.supported_modes  # Default
        assert ExecutionMode.TUI in cmd.supported_modes  # Default
    
    def test_complete_command_definition(self):
        """Test creating a complete command definition."""
        param = ParameterDefinition(
            name="input_file",
            type=ParameterType.FILE_PATH,
            description="Input file path",
            required=True
        )
        
        example = CommandExample(
            command="test_command --input_file example.txt",
            description="Process an example file",
            skill_level=SkillLevel.BEGINNER
        )
        
        dependency = CommandDependency(
            command="required_command",
            version_min="1.0.0",
            optional=False
        )
        
        cmd = CommandDefinition(
            name="test_command",
            aliases=["tc", "test"],
            description="A comprehensive test command",
            long_description="This is a detailed description of the test command",
            category=CommandCategory.ADVANCED,
            security_level=SecurityLevel.ELEVATED,
            supported_modes=[ExecutionMode.CLI, ExecutionMode.TUI, ExecutionMode.API],
            required_permissions=["file.read", "file.write"],
            min_skill_level=SkillLevel.INTERMEDIATE,
            parameters=[param],
            examples=[example],
            dependencies=[dependency],
            version="2.1.0",
            author="Test Author",
            tags=["testing", "example"]
        )
        
        assert cmd.name == "test_command"
        assert len(cmd.aliases) == 2
        assert cmd.category == CommandCategory.ADVANCED
        assert cmd.security_level == SecurityLevel.ELEVATED
        assert len(cmd.supported_modes) == 3
        assert len(cmd.required_permissions) == 2
        assert cmd.min_skill_level == SkillLevel.INTERMEDIATE
        assert len(cmd.parameters) == 1
        assert len(cmd.examples) == 1
        assert len(cmd.dependencies) == 1
        assert cmd.version == "2.1.0"
        assert len(cmd.tags) == 2
    
    def test_deprecated_command_validation(self):
        """Test that deprecated commands must have replacement."""
        with pytest.raises(ValueError, match="Deprecated commands must specify a replacement"):
            CommandDefinition(
                name="old_command",
                description="An old command",
                deprecated=True
                # Missing replacement
            )
        
        # Should work with replacement
        cmd = CommandDefinition(
            name="old_command",
            description="An old command",
            deprecated=True,
            replacement="new_command"
        )
        assert cmd.deprecated is True
        assert cmd.replacement == "new_command"
    
    def test_invalid_command_name(self):
        """Test that invalid command names are rejected."""
        with pytest.raises(ValueError, match="Command name must contain only"):
            CommandDefinition(
                name="invalid command!",  # Invalid characters
                description="Invalid command"
            )


class TestCommandMetadata:
    """Test command metadata functionality."""
    
    def test_command_metadata_creation(self):
        """Test creating command metadata."""
        cmd_def = CommandDefinition(
            name="test_cmd",
            description="Test command",
            tags=["test", "example"]
        )
        
        metadata = CommandMetadata(
            definition=cmd_def,
            handler_class="TestHandler",
            plugin_source="test_plugin"
        )
        
        assert metadata.definition.name == "test_cmd"
        assert metadata.handler_class == "TestHandler"
        assert metadata.plugin_source == "test_plugin"
        assert metadata.usage_count == 0
        assert metadata.success_rate == 1.0
        assert isinstance(metadata.registration_id, str)
    
    def test_search_keywords_generation(self):
        """Test automatic search keyword generation."""
        cmd_def = CommandDefinition(
            name="file_copy",
            aliases=["fc", "copy"],
            description="Copy files from source to destination",
            tags=["file", "copy", "utility"],
            category=CommandCategory.CORE
        )
        
        metadata = CommandMetadata(
            definition=cmd_def,
            handler_class="FileCopyHandler"
        )
        
        keywords = metadata.search_keywords
        
        # Should include command name, aliases, description words, tags, category
        assert "file_copy" in keywords
        assert "fc" in keywords
        assert "copy" in keywords
        assert "files" in keywords
        assert "source" in keywords
        assert "destination" in keywords
        assert "file" in keywords
        assert "utility" in keywords
        assert "core" in keywords


class TestDiscoveryRequest:
    """Test discovery request validation."""
    
    def test_basic_discovery_request(self):
        """Test creating a basic discovery request."""
        request = DiscoveryRequest(
            pattern="file copy",
            skill_level=SkillLevel.INTERMEDIATE,
            mode=ExecutionMode.CLI
        )
        
        assert request.pattern == "file copy"
        assert request.skill_level == SkillLevel.INTERMEDIATE
        assert request.mode == ExecutionMode.CLI
        assert request.fuzzy_matching is True  # Default
        assert request.max_results == 20  # Default
        assert request.include_deprecated is False  # Default
    
    def test_discovery_request_with_filters(self):
        """Test discovery request with various filters."""
        request = DiscoveryRequest(
            pattern="search",
            categories=[CommandCategory.CORE, CommandCategory.ADVANCED],
            include_deprecated=True,
            include_experimental=True,
            fuzzy_matching=False,
            max_results=10,
            current_project_type="python",
            recent_commands=["file.list", "git.status"],
            user_preferences={"preferred_tools": ["grep", "find"]}
        )
        
        assert len(request.categories) == 2
        assert request.include_deprecated is True
        assert request.include_experimental is True
        assert request.fuzzy_matching is False
        assert request.max_results == 10
        assert request.current_project_type == "python"
        assert len(request.recent_commands) == 2
        assert "preferred_tools" in request.user_preferences


class TestValidationRequest:
    """Test validation request creation."""
    
    def test_basic_validation_request(self):
        """Test creating a basic validation request."""
        request = ValidationRequest(
            command="test_command",
            args={"input_file": "test.txt", "verbose": True},
            context={"user_id": "test_user"}
        )
        
        assert request.command == "test_command"
        assert len(request.args) == 2
        assert request.args["input_file"] == "test.txt"
        assert request.args["verbose"] is True
        assert request.context["user_id"] == "test_user"
        assert request.security_check is True  # Default
        assert request.parameter_validation is True  # Default
    
    def test_validation_request_with_options(self):
        """Test validation request with all options."""
        request = ValidationRequest(
            command="risky_command",
            args={"target": "/etc/passwd"},
            context={"environment": "production"},
            security_check=True,
            parameter_validation=True,
            dependency_check=True,
            permission_check=True,
            user_skill_level=SkillLevel.EXPERT,
            execution_mode=ExecutionMode.API,
            user_permissions=["admin", "file.read"]
        )
        
        assert request.command == "risky_command"
        assert request.security_check is True
        assert request.dependency_check is True
        assert request.permission_check is True
        assert request.user_skill_level == SkillLevel.EXPERT
        assert request.execution_mode == ExecutionMode.API
        assert len(request.user_permissions) == 2


class TestValidationResult:
    """Test validation result creation and validation."""
    
    def test_successful_validation_result(self):
        """Test creating a successful validation result."""
        result = ValidationResult(
            valid=True,
            command_found=True,
            issues=[],
            suggestions=["Command executed successfully"],
            parameter_validation={"input_file": True, "verbose": True},
            security_assessment={"security_level": "safe", "safe": True},
            dependency_status={"required_dep": True},
            estimated_execution_time_ms=1500,
            estimated_resource_usage={"memory_mb": 50, "cpu_percent": 10}
        )
        
        assert result.valid is True
        assert result.command_found is True
        assert len(result.issues) == 0
        assert len(result.suggestions) == 1
        assert len(result.parameter_validation) == 2
        assert result.security_assessment["safe"] is True
        assert result.dependency_status["required_dep"] is True
        assert result.estimated_execution_time_ms == 1500
    
    def test_failed_validation_result(self):
        """Test creating a failed validation result."""
        error_issue = ValidationIssue(
            severity="error",
            code="MISSING_PARAMETER",
            message="Required parameter missing",
            parameter="input_file",
            suggestion="Provide the input_file parameter"
        )
        
        warning_issue = ValidationIssue(
            severity="warning",
            code="DEPRECATED_COMMAND",
            message="This command is deprecated",
            suggestion="Use new_command instead"
        )
        
        result = ValidationResult(
            valid=False,
            command_found=True,
            issues=[error_issue, warning_issue],
            suggestions=["Fix the error and try again"],
            parameter_validation={"input_file": False},
            security_assessment={"security_level": "safe", "safe": True},
            dependency_status={}
        )
        
        assert result.valid is False
        assert result.command_found is True
        assert len(result.issues) == 2
        assert result.issues[0].severity == "error"
        assert result.issues[1].severity == "warning"
        assert result.parameter_validation["input_file"] is False
    
    def test_validation_result_consistency(self):
        """Test that validation result consistency is enforced."""
        error_issue = ValidationIssue(
            severity="error",
            code="VALIDATION_ERROR",
            message="Validation failed"
        )
        
        # Should automatically set valid=False when there are errors
        result = ValidationResult(
            valid=True,  # This should be corrected
            command_found=True,
            issues=[error_issue]
        )
        
        assert result.valid is False  # Should be corrected by validator


class TestValidationIssue:
    """Test validation issue creation."""
    
    def test_validation_issue_creation(self):
        """Test creating validation issues."""
        issue = ValidationIssue(
            severity="error",
            code="INVALID_PARAMETER",
            message="Parameter validation failed",
            parameter="test_param",
            suggestion="Check parameter format"
        )
        
        assert issue.severity == "error"
        assert issue.code == "INVALID_PARAMETER"
        assert issue.message == "Parameter validation failed"
        assert issue.parameter == "test_param"
        assert issue.suggestion == "Check parameter format"
    
    def test_invalid_severity(self):
        """Test that invalid severity levels are rejected."""
        with pytest.raises(ValueError, match="Severity must be error, warning, or info"):
            ValidationIssue(
                severity="critical",  # Invalid severity
                code="TEST_CODE",
                message="Test message"
            )
    
    def test_valid_severity_levels(self):
        """Test all valid severity levels."""
        for severity in ["error", "warning", "info"]:
            issue = ValidationIssue(
                severity=severity,
                code="TEST_CODE",
                message="Test message"
            )
            assert issue.severity == severity


class TestCommandDependency:
    """Test command dependency validation."""
    
    def test_basic_dependency(self):
        """Test creating a basic dependency."""
        dep = CommandDependency(
            command="required_command"
        )
        
        assert dep.command == "required_command"
        assert dep.version_min is None
        assert dep.version_max is None
        assert dep.optional is False
        assert dep.fallback is None
    
    def test_versioned_dependency(self):
        """Test creating a dependency with version constraints."""
        dep = CommandDependency(
            command="versioned_command",
            version_min="1.2.0",
            version_max="2.0.0",
            optional=True,
            fallback="alternative_command"
        )
        
        assert dep.command == "versioned_command"
        assert dep.version_min == "1.2.0"
        assert dep.version_max == "2.0.0"
        assert dep.optional is True
        assert dep.fallback == "alternative_command"


class TestCommandExample:
    """Test command example validation."""
    
    def test_command_example(self):
        """Test creating a command example."""
        example = CommandExample(
            command="test_command --input file.txt --verbose",
            description="Process a file with verbose output",
            expected_output="File processed successfully",
            skill_level=SkillLevel.INTERMEDIATE
        )
        
        assert example.command == "test_command --input file.txt --verbose"
        assert example.description == "Process a file with verbose output"
        assert example.expected_output == "File processed successfully"
        assert example.skill_level == SkillLevel.INTERMEDIATE


# Integration test for the full workflow
class TestRegistryModelsIntegration:
    """Integration tests for registry models."""
    
    def test_complete_command_workflow(self):
        """Test creating a complete command with all components."""
        # Create parameter with constraint
        constraint = ValidationConstraint(
            type="min_length",
            value=1,
            message="Filename cannot be empty"
        )
        
        param = ParameterDefinition(
            name="filename",
            type=ParameterType.FILE_PATH,
            description="Path to the input file",
            required=True,
            constraints=[constraint],
            examples=["input.txt", "/path/to/file.dat"],
            aliases=["f", "file"]
        )
        
        # Create example
        example = CommandExample(
            command="process_file --filename input.txt",
            description="Process a simple text file",
            expected_output="File processed: 100 lines",
            skill_level=SkillLevel.BEGINNER
        )
        
        # Create dependency
        dependency = CommandDependency(
            command="file_reader",
            version_min="1.0.0",
            optional=False
        )
        
        # Create command definition
        cmd_def = CommandDefinition(
            name="process_file",
            aliases=["pf", "process"],
            description="Process files with various operations",
            long_description="A comprehensive file processing command that supports multiple formats",
            category=CommandCategory.CORE,
            security_level=SecurityLevel.SAFE,
            supported_modes=[ExecutionMode.CLI, ExecutionMode.TUI],
            required_permissions=["file.read"],
            min_skill_level=SkillLevel.BEGINNER,
            parameters=[param],
            examples=[example],
            dependencies=[dependency],
            version="1.0.0",
            author="Test Suite",
            tags=["file", "processing", "utility"]
        )
        
        # Create metadata
        metadata = CommandMetadata(
            definition=cmd_def,
            handler_class="ProcessFileHandler",
            plugin_source=None
        )
        
        # Verify everything is properly connected
        assert metadata.definition.name == "process_file"
        assert len(metadata.definition.parameters) == 1
        assert metadata.definition.parameters[0].name == "filename"
        assert len(metadata.definition.parameters[0].constraints) == 1
        assert len(metadata.definition.examples) == 1
        assert len(metadata.definition.dependencies) == 1
        assert metadata.handler_class == "ProcessFileHandler"
        
        # Verify search keywords were generated
        assert "process_file" in metadata.search_keywords
        assert "process" in metadata.search_keywords
        assert "file" in metadata.search_keywords
        assert "processing" in metadata.search_keywords
    
    def test_validation_workflow(self):
        """Test the complete validation workflow."""
        # Create validation request
        request = ValidationRequest(
            command="process_file",
            args={
                "filename": "test.txt",
                "verbose": True
            },
            context={
                "user_id": "test_user",
                "environment": "development"
            },
            security_check=True,
            parameter_validation=True,
            dependency_check=True,
            permission_check=True,
            user_skill_level=SkillLevel.INTERMEDIATE,
            execution_mode=ExecutionMode.CLI,
            user_permissions=["file.read", "file.write"]
        )
        
        # Create validation issues
        warning_issue = ValidationIssue(
            severity="warning",
            code="FILE_NOT_FOUND",
            message="File 'test.txt' not found",
            parameter="filename",
            suggestion="Verify the file path exists"
        )
        
        # Create validation result
        result = ValidationResult(
            valid=True,  # Warning doesn't make it invalid
            command_found=True,
            issues=[warning_issue],
            suggestions=["Check file paths before execution"],
            parameter_validation={"filename": True, "verbose": True},
            security_assessment={
                "security_level": "safe",
                "safe": True,
                "risk_factors": []
            },
            dependency_status={"file_reader": True},
            estimated_execution_time_ms=2000,
            estimated_resource_usage={
                "memory_mb": 25,
                "cpu_percent": 5,
                "disk_io_mb": 10,
                "network_kb": 0
            }
        )
        
        # Verify the complete validation result
        assert result.valid is True
        assert result.command_found is True
        assert len(result.issues) == 1
        assert result.issues[0].severity == "warning"
        assert len(result.parameter_validation) == 2
        assert result.security_assessment["safe"] is True
        assert result.dependency_status["file_reader"] is True
        assert result.estimated_execution_time_ms == 2000
        assert "memory_mb" in result.estimated_resource_usage