"""Natural language processing data models for CLI v3 architecture.

This module defines the Pydantic data structures used for natural language
parsing, command interpretation, and LLM integration.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class ParsingMethod(str, Enum):
    """Method used for parsing natural language input."""
    LLM = "llm"
    RULE_BASED = "rule_based" 
    HYBRID = "hybrid"


class ConfidenceLevel(str, Enum):
    """Confidence level classifications."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class LLMConfig(BaseModel):
    """Configuration for local LLM integration."""
    
    model_name: str = Field(default="gpt-oss:20b", description="Local LLM model name")
    max_tokens: int = Field(default=1024, ge=1, le=8192, description="Maximum output tokens")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0, description="Request timeout")
    context_window: int = Field(default=32000, ge=1000, description="Model context window size")
    enable_tools: bool = Field(default=False, description="Enable tool use in LLM calls")
    
    @field_validator('model_name')
    def validate_model_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class ConversationContext(BaseModel):
    """Context for natural language processing with conversation history."""
    
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    command_history: List[str] = Field(default_factory=list, max_items=50)
    recent_files: List[str] = Field(default_factory=list, max_items=20)
    current_directory: str = Field(default=".", description="Current working directory")
    project_state: Dict[str, Any] = Field(default_factory=dict, description="Project metadata")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_command(self, command: str) -> None:
        """Add a command to history."""
        self.command_history.append(command)
        if len(self.command_history) > 50:
            self.command_history.pop(0)
        self.last_activity = datetime.now(timezone.utc)
    
    def add_file(self, filepath: str) -> None:
        """Add a file to recent files."""
        if filepath not in self.recent_files:
            self.recent_files.append(filepath)
        if len(self.recent_files) > 20:
            self.recent_files.pop(0)


class ParsedCommand(BaseModel):
    """Structured command parsed from natural language."""
    
    action: str = Field(..., min_length=1, description="Primary action/command type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Command parameters")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Parsing confidence score")
    method: ParsingMethod = Field(..., description="Method used for parsing")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator('action')
    def validate_action(cls, v):
        if not v or not v.strip():
            raise ValueError("Action cannot be empty")
        return v.strip()


class CommandInterpretation(BaseModel):
    """Alternative interpretation of a natural language command."""
    
    command: ParsedCommand = Field(..., description="Parsed command structure")
    rationale: str = Field(..., min_length=1, description="Explanation of interpretation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Interpretation confidence")
    examples: List[str] = Field(default_factory=list, description="Usage examples")
    
    @field_validator('rationale')
    def validate_rationale(cls, v):
        if not v or not v.strip():
            raise ValueError("Rationale cannot be empty")
        return v.strip()


class PatternMatch(BaseModel):
    """Rule-based pattern match result."""
    
    pattern: str = Field(..., description="Matched pattern identifier")
    action: str = Field(..., description="Corresponding action")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(..., ge=0.0, le=1.0)
    priority: int = Field(default=5, ge=1, le=10, description="Pattern priority (1=highest)")


class NLPError(BaseModel):
    """Natural language processing error details."""
    
    error_type: str = Field(..., description="Error classification")
    message: str = Field(..., min_length=1, description="Human-readable error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")
    recovery_suggestions: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ParsingResult(BaseModel):
    """Complete result from natural language parsing operation."""
    
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    success: bool = Field(..., description="Overall parsing success")
    structured_command: Optional[ParsedCommand] = Field(None, description="Primary parsed command")
    alternative_interpretations: List[CommandInterpretation] = Field(
        default_factory=list,
        description="Alternative command interpretations"
    )
    explanation: str = Field(..., min_length=1, description="What the system understood")
    method_used: ParsingMethod = Field(..., description="Parsing method that succeeded")
    processing_time_ms: int = Field(default=0, ge=0, description="Processing duration")
    errors: List[NLPError] = Field(default_factory=list, description="Any errors encountered")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TokenUsage(BaseModel):
    """LLM token usage tracking."""
    
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost: float = Field(default=0.0, ge=0.0, description="Estimated cost in USD")
    
    def __post_init__(self):
        """Calculate total tokens after initialization."""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """Response from LLM processing."""
    
    content: str = Field(..., description="LLM response content")
    finish_reason: str = Field(default="stop", description="Why generation stopped")
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    model_name: str = Field(..., description="Model used for generation")
    processing_time_ms: int = Field(default=0, ge=0)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CommandExample(BaseModel):
    """Example of natural language to command mapping."""
    
    natural_input: str = Field(..., min_length=1, description="Natural language input")
    expected_action: str = Field(..., min_length=1, description="Expected command action")
    expected_parameters: Dict[str, Any] = Field(default_factory=dict)
    description: str = Field(..., min_length=1, description="Example description")
    category: str = Field(default="general", description="Example category")


class NLPMetrics(BaseModel):
    """Metrics for NLP system performance."""
    
    total_requests: int = Field(default=0, ge=0)
    successful_parses: int = Field(default=0, ge=0)
    failed_parses: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    rule_based_matches: int = Field(default=0, ge=0)
    average_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    average_processing_time_ms: float = Field(default=0.0, ge=0.0)
    total_tokens_used: int = Field(default=0, ge=0)
    estimated_total_cost: float = Field(default=0.0, ge=0.0)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_parses / self.total_requests) * 100.0


# Custom exceptions for NLP processing
class NLPError(Exception):
    """Base exception for natural language processing errors."""
    pass


class ParsingFailedError(NLPError):
    """Raised when parsing completely fails."""
    pass


class AmbiguousInputError(NLPError):
    """Raised when input has multiple valid interpretations."""
    pass


class LLMUnavailableError(NLPError):
    """Raised when LLM service is unavailable."""
    pass


class ContextTooLargeError(NLPError):
    """Raised when context exceeds model limits."""
    pass


class UnsupportedLanguageError(NLPError):
    """Raised when input language is not supported."""
    pass