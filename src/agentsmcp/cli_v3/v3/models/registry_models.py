"""Registry models for CLI v3 command registry system.

This module defines the Pydantic data structures used by the command registry,
discovery engine, and validation system for centralized command management.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from .command_models import ExecutionMode, SkillLevel


class CommandCategory(str, Enum):
    """Command categories for organization and filtering."""
    CORE = "core"
    ADVANCED = "advanced"
    PLUGIN = "plugin"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    SYSTEM = "system"


class CommandPriority(str, Enum):
    """Command priority for execution and resource management."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class SecurityLevel(str, Enum):
    """Security levels for command validation."""
    SAFE = "safe"
    ELEVATED = "elevated"
    DANGEROUS = "dangerous"
    SYSTEM = "system"


class ParameterType(str, Enum):
    """Parameter types for validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    FILE_PATH = "file_path"
    DIRECTORY_PATH = "directory_path"
    URL = "url"
    EMAIL = "email"
    JSON = "json"
    REGEX = "regex"


class ValidationConstraint(BaseModel):
    """Parameter validation constraint."""
    
    type: str = Field(..., description="Constraint type (min_length, max_length, pattern, etc.)")
    value: Union[int, float, str, bool, List[Any]] = Field(..., description="Constraint value")
    message: Optional[str] = Field(None, description="Custom validation message")


class ParameterDefinition(BaseModel):
    """Command parameter definition with validation rules."""
    
    name: str = Field(..., min_length=1, description="Parameter name")
    type: ParameterType = Field(..., description="Parameter data type")
    description: str = Field(..., min_length=1, description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default_value: Any = Field(None, description="Default value if not provided")
    constraints: List[ValidationConstraint] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list, description="Usage examples")
    aliases: List[str] = Field(default_factory=list, description="Alternative parameter names")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Parameter name must contain only alphanumeric characters, hyphens, and underscores")
        return v


class CommandExample(BaseModel):
    """Command usage example with explanation."""
    
    command: str = Field(..., min_length=1, description="Example command")
    description: str = Field(..., min_length=1, description="What the example demonstrates")
    expected_output: Optional[str] = Field(None, description="Expected command output")
    skill_level: SkillLevel = Field(default=SkillLevel.BEGINNER, description="Target skill level")


class CommandDependency(BaseModel):
    """Command dependency specification."""
    
    command: str = Field(..., min_length=1, description="Required command name")
    version_min: Optional[str] = Field(None, description="Minimum required version")
    version_max: Optional[str] = Field(None, description="Maximum compatible version")
    optional: bool = Field(default=False, description="Whether dependency is optional")
    fallback: Optional[str] = Field(None, description="Fallback command if dependency unavailable")


class CommandDefinition(BaseModel):
    """Complete command definition for registry."""
    
    name: str = Field(..., min_length=1, description="Unique command name")
    aliases: List[str] = Field(default_factory=list, description="Command aliases")
    description: str = Field(..., min_length=1, description="Brief command description")
    long_description: Optional[str] = Field(None, description="Detailed command explanation")
    category: CommandCategory = Field(default=CommandCategory.CORE)
    priority: CommandPriority = Field(default=CommandPriority.NORMAL)
    security_level: SecurityLevel = Field(default=SecurityLevel.SAFE)
    
    # Execution requirements
    supported_modes: List[ExecutionMode] = Field(
        default_factory=lambda: [ExecutionMode.CLI, ExecutionMode.TUI],
        description="Supported execution modes"
    )
    required_permissions: List[str] = Field(default_factory=list, description="Required user permissions")
    min_skill_level: SkillLevel = Field(default=SkillLevel.BEGINNER, description="Minimum user skill level")
    
    # Parameters and validation
    parameters: List[ParameterDefinition] = Field(default_factory=list)
    examples: List[CommandExample] = Field(default_factory=list)
    
    # Dependencies and versioning
    dependencies: List[CommandDependency] = Field(default_factory=list)
    version: str = Field(default="1.0.0", description="Command version")
    deprecated: bool = Field(default=False, description="Whether command is deprecated")
    replacement: Optional[str] = Field(None, description="Replacement command if deprecated")
    
    # Metadata
    author: Optional[str] = Field(None, description="Command author")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v.replace('_', '').replace('-', '').replace('.', '').isalnum():
            raise ValueError("Command name must contain only alphanumeric characters, hyphens, underscores, and dots")
        return v
    
    @model_validator(mode='after')
    def validate_deprecated_replacement(self):
        if self.deprecated and not self.replacement:
            raise ValueError("Deprecated commands must specify a replacement")
        return self


class CommandMetadata(BaseModel):
    """Command metadata for registry storage."""
    
    definition: CommandDefinition = Field(..., description="Complete command definition")
    registration_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique registration ID")
    handler_class: str = Field(..., description="Handler class name")
    plugin_source: Optional[str] = Field(None, description="Plugin source if external command")
    
    # Runtime statistics
    usage_count: int = Field(default=0, ge=0, description="Total usage count")
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Success rate")
    avg_execution_time_ms: int = Field(default=0, ge=0, description="Average execution time")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    
    # Discovery optimization
    search_keywords: List[str] = Field(default_factory=list, description="Optimized search keywords")
    fuzzy_variants: List[str] = Field(default_factory=list, description="Common misspellings and variants")
    
    @model_validator(mode='after')
    def generate_search_keywords(self):
        """Auto-generate search keywords from definition."""
        if not self.search_keywords:
            keywords = set()
            
            # Add command name and aliases
            keywords.add(self.definition.name.lower())
            keywords.update(alias.lower() for alias in self.definition.aliases)
            
            # Add words from description
            desc_words = self.definition.description.lower().split()
            keywords.update(word.strip('.,!?') for word in desc_words if len(word) > 2)
            
            # Add tags
            keywords.update(tag.lower() for tag in self.definition.tags)
            
            # Add category
            keywords.add(self.definition.category.value)
            
            self.search_keywords = sorted(list(keywords))
        
        return self


class DiscoveryRequest(BaseModel):
    """Request for command discovery with filtering criteria."""
    
    pattern: str = Field(..., min_length=1, description="Search pattern")
    skill_level: SkillLevel = Field(default=SkillLevel.INTERMEDIATE, description="User skill level")
    mode: ExecutionMode = Field(default=ExecutionMode.CLI, description="Target execution mode")
    
    # Filtering options
    categories: List[CommandCategory] = Field(default_factory=list, description="Filter by categories")
    include_deprecated: bool = Field(default=False, description="Include deprecated commands")
    include_experimental: bool = Field(default=False, description="Include experimental commands")
    fuzzy_matching: bool = Field(default=True, description="Enable fuzzy matching")
    max_results: int = Field(default=20, ge=1, le=100, description="Maximum results to return")
    
    # Context for smart suggestions
    current_project_type: Optional[str] = Field(None, description="Current project context")
    recent_commands: List[str] = Field(default_factory=list, max_items=10, description="Recent command history")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")


class DiscoveryResult(BaseModel):
    """Individual command discovery result with scoring."""
    
    metadata: CommandMetadata = Field(..., description="Command metadata")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Search relevance score")
    match_reasons: List[str] = Field(default_factory=list, description="Why this command matched")
    usage_rank: int = Field(ge=0, description="Usage-based ranking")
    fuzzy_match: bool = Field(default=False, description="Whether match was fuzzy")


class ValidationRequest(BaseModel):
    """Request for command validation."""
    
    command: str = Field(..., min_length=1, description="Command to validate")
    args: Dict[str, Any] = Field(default_factory=dict, description="Command arguments")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    
    # Validation options
    security_check: bool = Field(default=True, description="Perform security validation")
    parameter_validation: bool = Field(default=True, description="Validate parameters")
    dependency_check: bool = Field(default=True, description="Check dependencies")
    permission_check: bool = Field(default=True, description="Check permissions")
    
    # User context
    user_skill_level: SkillLevel = Field(default=SkillLevel.INTERMEDIATE)
    execution_mode: ExecutionMode = Field(default=ExecutionMode.CLI)
    user_permissions: List[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    """Individual validation issue."""
    
    severity: str = Field(..., description="Issue severity: error, warning, info")
    code: str = Field(..., description="Machine-readable issue code")
    message: str = Field(..., description="Human-readable issue message")
    parameter: Optional[str] = Field(None, description="Related parameter if applicable")
    suggestion: Optional[str] = Field(None, description="Suggested fix")
    
    @field_validator('severity')
    @classmethod
    def validate_severity(cls, v):
        if v not in ['error', 'warning', 'info']:
            raise ValueError("Severity must be error, warning, or info")
        return v


class ValidationResult(BaseModel):
    """Complete validation result."""
    
    valid: bool = Field(..., description="Overall validation result")
    command_found: bool = Field(..., description="Whether command exists")
    issues: List[ValidationIssue] = Field(default_factory=list, description="Validation issues")
    suggestions: List[str] = Field(default_factory=list, description="General suggestions")
    
    # Detailed results
    parameter_validation: Dict[str, bool] = Field(default_factory=dict, description="Per-parameter validation")
    security_assessment: Dict[str, Any] = Field(default_factory=dict, description="Security analysis")
    dependency_status: Dict[str, bool] = Field(default_factory=dict, description="Dependency availability")
    
    # Performance estimates
    estimated_execution_time_ms: Optional[int] = Field(None, ge=0)
    estimated_resource_usage: Dict[str, Union[int, float]] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Ensure validation result consistency."""
        has_errors = any(issue.severity == 'error' for issue in self.issues)
        if has_errors and self.valid:
            self.valid = False
        return self


class RegistryStats(BaseModel):
    """Registry statistics and health information."""
    
    total_commands: int = Field(ge=0)
    active_commands: int = Field(ge=0)
    deprecated_commands: int = Field(ge=0)
    plugin_commands: int = Field(ge=0)
    
    # Category breakdown
    commands_by_category: Dict[CommandCategory, int] = Field(default_factory=dict)
    commands_by_skill_level: Dict[SkillLevel, int] = Field(default_factory=dict)
    
    # Performance metrics
    avg_lookup_time_ms: float = Field(ge=0.0)
    avg_discovery_time_ms: float = Field(ge=0.0)
    avg_validation_time_ms: float = Field(ge=0.0)
    
    # Usage statistics
    most_used_commands: List[str] = Field(default_factory=list, max_items=10)
    recent_registrations: int = Field(default=0, ge=0)
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Custom exceptions for registry system
class RegistryError(Exception):
    """Base exception for registry errors."""
    pass


class CommandAlreadyExistsError(RegistryError):
    """Raised when attempting to register a command that already exists."""
    pass


class InvalidDefinitionError(RegistryError):
    """Raised when command definition is invalid."""
    pass


class RegistryCorruptedError(RegistryError):
    """Raised when registry data is corrupted."""
    pass


class CircularDependencyError(RegistryError):
    """Raised when circular dependencies are detected."""
    pass