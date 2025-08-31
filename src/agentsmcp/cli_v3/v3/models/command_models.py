"""Command models for CLI v3 core command engine.

This module defines the Pydantic data structures used throughout the command
execution pipeline for type safety, validation, and serialization.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class ExecutionMode(str, Enum):
    """Execution interface modes for command routing."""
    CLI = "cli"
    TUI = "tui" 
    WEB_UI = "web_ui"
    API = "api"


class SkillLevel(str, Enum):
    """User skill level for progressive disclosure."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class CommandStatus(str, Enum):
    """Command execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ResourceType(str, Enum):
    """Resource types for limit enforcement."""
    CPU_TIME = "cpu_time"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    API_CALLS = "api_calls"
    TOKEN_COUNT = "token_count"


class UserPreferences(BaseModel):
    """User interface and behavior preferences."""
    
    theme: str = Field(default="default", description="UI theme preference")
    verbose_output: bool = Field(default=False, description="Show detailed output")
    auto_confirm: bool = Field(default=False, description="Auto-confirm safe operations")
    suggestion_level: int = Field(
        default=3, 
        ge=0, 
        le=5, 
        description="Level of suggestions to show (0=none, 5=maximum)"
    )
    default_timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=300000,
        description="Default command timeout in milliseconds"
    )


class UserProfile(BaseModel):
    """User context and preferences for personalization."""
    
    user_id: str = Field(default_factory=lambda: str(uuid4()))
    skill_level: SkillLevel = Field(default=SkillLevel.INTERMEDIATE)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    command_history: List[str] = Field(default_factory=list, max_items=100)
    favorite_commands: List[str] = Field(default_factory=list, max_items=20)
    last_active: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResourceLimit(BaseModel):
    """Resource limit configuration."""
    
    resource_type: ResourceType
    max_value: Union[int, float] = Field(gt=0, description="Maximum allowed value")
    current_usage: Union[int, float] = Field(default=0, ge=0)
    warning_threshold: float = Field(
        default=0.8, 
        ge=0.1, 
        le=1.0,
        description="Warning threshold as fraction of max"
    )


class CommandRequest(BaseModel):
    """Structured command request with context."""
    
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    command_type: str = Field(..., min_length=1, description="Command type identifier")
    args: Dict[str, Any] = Field(default_factory=dict, description="Command arguments")
    raw_input: Optional[str] = Field(None, description="Original user input if natural language")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = Field(default=5, ge=1, le=10, description="Execution priority (1=highest)")
    timeout_ms: Optional[int] = Field(None, ge=1000, le=600000)


class Suggestion(BaseModel):
    """Smart suggestion for user assistance."""
    
    text: str = Field(..., min_length=1, description="Suggestion text")
    command: Optional[str] = Field(None, description="Executable command if applicable")
    category: str = Field(default="general", description="Suggestion category")
    confidence: float = Field(ge=0.0, le=1.0, description="Suggestion confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NextAction(BaseModel):
    """Suggested next action after command completion."""
    
    command: str = Field(..., min_length=1, description="Suggested command")
    description: str = Field(..., min_length=1, description="Human-readable description")
    confidence: float = Field(ge=0.0, le=1.0, description="Action relevance confidence")
    category: str = Field(default="workflow", description="Action category")
    estimated_time_ms: Optional[int] = Field(None, ge=0, description="Estimated execution time")


class ExecutionMetrics(BaseModel):
    """Command execution performance and usage metrics."""
    
    duration_ms: int = Field(ge=0, description="Total execution time")
    tokens_used: int = Field(default=0, ge=0, description="LLM tokens consumed")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated cost in USD")
    cpu_time_ms: int = Field(default=0, ge=0, description="CPU time consumed")
    memory_peak_mb: int = Field(default=0, ge=0, description="Peak memory usage")
    network_calls: int = Field(default=0, ge=0, description="Network requests made")
    cache_hits: int = Field(default=0, ge=0, description="Cache hits")
    cache_misses: int = Field(default=0, ge=0, description="Cache misses")


class CommandError(BaseModel):
    """Structured command execution error."""
    
    error_code: str = Field(..., min_length=1, description="Machine-readable error code")
    message: str = Field(..., min_length=1, description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")
    recovery_suggestions: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CommandResult(BaseModel):
    """Complete command execution result."""
    
    request_id: str = Field(..., description="Matching request ID")
    success: bool = Field(..., description="Overall execution success")
    status: CommandStatus = Field(..., description="Detailed execution status")
    data: Any = Field(None, description="Command output data")
    suggestions: List[Suggestion] = Field(default_factory=list)
    errors: List[CommandError] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @model_validator(mode='after')
    def validate_success_consistency(self):
        """Ensure success field matches status."""
        if self.status == CommandStatus.COMPLETED and not self.success:
            raise ValueError("Success must be True when status is COMPLETED")
        elif self.status in [CommandStatus.FAILED, CommandStatus.TIMEOUT] and self.success:
            raise ValueError(f"Success must be False when status is {self.status}")
        return self


class AuditLogEntry(BaseModel):
    """Audit trail entry for command execution."""
    
    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str = Field(..., description="Related command request ID")
    user_id: str = Field(..., description="User who executed command")
    command_type: str = Field(..., description="Command that was executed")
    execution_mode: ExecutionMode = Field(..., description="Interface mode used")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = Field(..., description="Command execution success")
    duration_ms: int = Field(ge=0, description="Execution duration")
    resource_usage: Dict[str, Union[int, float]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Custom exceptions for command engine
class CommandEngineError(Exception):
    """Base exception for command engine errors."""
    pass


class CommandNotFoundError(CommandEngineError):
    """Raised when a requested command is not found."""
    pass


class ValidationFailedError(CommandEngineError):
    """Raised when command validation fails."""
    pass


class ExecutionTimeoutError(CommandEngineError):
    """Raised when command execution times out."""
    pass


class PermissionDeniedError(CommandEngineError):
    """Raised when user lacks permission for command."""
    pass


class ResourceExhaustedError(CommandEngineError):
    """Raised when resource limits are exceeded."""
    pass