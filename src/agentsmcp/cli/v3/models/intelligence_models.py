"""Intelligence system models for user profiling, learning, and personalization.

This module defines the Pydantic data structures used by the user intelligence
system for learning user patterns, skill detection, and personalized experiences.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator, root_validator


class SkillLevel(str, Enum):
    """User skill level for progressive disclosure and personalization."""
    BEGINNER = "beginner"      # 0-10 commands, frequent help requests
    INTERMEDIATE = "intermediate"  # 11-50 commands, some advanced features
    EXPERT = "expert"          # 51-200 commands, regular advanced usage
    POWER = "power"            # 200+ commands, custom workflows


class CommandCategory(str, Enum):
    """Categories for command classification and pattern analysis."""
    SETUP = "setup"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    DEBUGGING = "debugging"
    CONFIGURATION = "configuration"
    HELP = "help"
    ADVANCED = "advanced"


class SuggestionType(str, Enum):
    """Types of suggestions the system can provide."""
    NEXT_ACTION = "next_action"
    FEATURE_INTRODUCTION = "feature_introduction"
    OPTIMIZATION = "optimization"
    ERROR_PREVENTION = "error_prevention"
    WORKFLOW_COMPLETION = "workflow_completion"
    SKILL_ADVANCEMENT = "skill_advancement"


class LearningEventType(str, Enum):
    """Types of learning events for pattern recognition."""
    COMMAND_EXECUTED = "command_executed"
    ERROR_OCCURRED = "error_occurred"
    HELP_REQUESTED = "help_requested"
    FEATURE_DISCOVERED = "feature_discovered"
    WORKFLOW_COMPLETED = "workflow_completed"
    FEEDBACK_PROVIDED = "feedback_provided"


class ProgressiveDisclosureLevel(int, Enum):
    """Progressive disclosure levels for UI complexity management."""
    MINIMAL = 1        # Basic commands only (beginners)
    BASIC = 2          # Common advanced features (intermediate)
    STANDARD = 3       # Full feature set (experts)
    ADVANCED = 4       # Power user shortcuts and customization
    DEVELOPER = 5      # Developer/debugging features


class UserAction(BaseModel):
    """Individual user action for learning and pattern recognition."""
    
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    command: str = Field(..., min_length=1, description="Command executed")
    category: CommandCategory = Field(default=CommandCategory.EXECUTION)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = Field(..., description="Whether command succeeded")
    duration_ms: int = Field(ge=0, description="Command execution time")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    errors: List[str] = Field(default_factory=list, description="Error messages if failed")
    
    
class SessionContext(BaseModel):
    """Context information for a user session."""
    
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    commands_used: List[str] = Field(default_factory=list)
    errors_encountered: List[str] = Field(default_factory=list)
    help_requests: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    unique_commands: int = Field(default=0, ge=0)
    advanced_features_used: List[str] = Field(default_factory=list)


class UserFeedback(BaseModel):
    """User feedback for suggestion and system improvement."""
    
    feedback_id: str = Field(default_factory=lambda: str(uuid4()))
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comment: Optional[str] = Field(None, description="Optional text feedback")
    suggestion_id: Optional[str] = Field(None, description="Related suggestion ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    category: str = Field(default="general", description="Feedback category")
    helpful: Optional[bool] = Field(None, description="Whether suggestion was helpful")


class CommandPattern(BaseModel):
    """Detected command usage pattern for workflow recognition."""
    
    pattern_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., min_length=1, description="Human-readable pattern name")
    commands: List[str] = Field(..., min_items=2, description="Command sequence")
    frequency: int = Field(default=1, ge=1, description="How often pattern occurs")
    success_rate: float = Field(ge=0.0, le=1.0, description="Pattern success rate")
    avg_duration_ms: int = Field(ge=0, description="Average completion time")
    context_triggers: List[str] = Field(default_factory=list, description="Context that triggers pattern")
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = Field(ge=0.0, le=1.0, description="Pattern confidence score")


class Suggestion(BaseModel):
    """Personalized suggestion for user assistance."""
    
    suggestion_id: str = Field(default_factory=lambda: str(uuid4()))
    type: SuggestionType = Field(..., description="Type of suggestion")
    text: str = Field(..., min_length=1, description="Suggestion text")
    command: Optional[str] = Field(None, description="Executable command if applicable")
    category: str = Field(default="general", description="Suggestion category")
    confidence: float = Field(ge=0.0, le=1.0, description="Suggestion confidence score")
    priority: int = Field(default=5, ge=1, le=10, description="Suggestion priority (1=highest)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    skill_level_target: SkillLevel = Field(default=SkillLevel.INTERMEDIATE, description="Target skill level")
    expires_at: Optional[datetime] = Field(None, description="When suggestion expires")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('expires_at')
    def validate_expiry(cls, v):
        if v and v <= datetime.now(timezone.utc):
            raise ValueError("Expiry time must be in the future")
        return v


class LearningMetrics(BaseModel):
    """Metrics for learning algorithm performance tracking."""
    
    total_actions: int = Field(default=0, ge=0, description="Total user actions recorded")
    patterns_detected: int = Field(default=0, ge=0, description="Command patterns identified")
    suggestions_made: int = Field(default=0, ge=0, description="Suggestions provided")
    suggestions_accepted: int = Field(default=0, ge=0, description="Suggestions user acted on")
    accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall prediction accuracy")
    learning_rate: float = Field(default=0.1, ge=0.001, le=1.0, description="Algorithm learning rate")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def acceptance_rate(self) -> float:
        """Calculate suggestion acceptance rate."""
        if self.suggestions_made == 0:
            return 0.0
        return self.suggestions_accepted / self.suggestions_made


class UserProfile(BaseModel):
    """Complete user profile with preferences, skills, and patterns."""
    
    user_id: str = Field(default_factory=lambda: str(uuid4()))
    skill_level: SkillLevel = Field(default=SkillLevel.BEGINNER)
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    command_patterns: List[CommandPattern] = Field(default_factory=list)
    total_commands: int = Field(default=0, ge=0, description="Total commands executed")
    session_count: int = Field(default=0, ge=0, description="Total sessions")
    avg_session_duration_ms: int = Field(default=0, ge=0)
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    help_request_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    advanced_features_used: Set[str] = Field(default_factory=set)
    favorite_commands: List[str] = Field(default_factory=list, max_items=20)
    common_errors: Dict[str, int] = Field(default_factory=dict)
    learning_metrics: LearningMetrics = Field(default_factory=LearningMetrics)
    progressive_disclosure_level: ProgressiveDisclosureLevel = Field(default=ProgressiveDisclosureLevel.MINIMAL)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field(default="1.0.0", description="Profile schema version")
    
    @root_validator
    def validate_consistency(cls, values):
        """Ensure profile data consistency."""
        total_commands = values.get('total_commands', 0)
        skill_level = values.get('skill_level', SkillLevel.BEGINNER)
        
        # Validate skill level consistency with command count
        if skill_level == SkillLevel.BEGINNER and total_commands > 10:
            values['skill_level'] = SkillLevel.INTERMEDIATE
        elif skill_level == SkillLevel.INTERMEDIATE and total_commands > 50:
            values['skill_level'] = SkillLevel.EXPERT
        elif skill_level == SkillLevel.EXPERT and total_commands > 200:
            values['skill_level'] = SkillLevel.POWER
            
        return values
    
    @property
    def expertise_score(self) -> float:
        """Calculate overall user expertise score (0.0-1.0)."""
        base_score = min(self.total_commands / 200.0, 1.0)  # Commands contribution
        success_bonus = self.success_rate * 0.2  # Success rate bonus
        pattern_bonus = min(len(self.command_patterns) / 10.0, 0.2)  # Pattern recognition bonus
        advanced_bonus = min(len(self.advanced_features_used) / 20.0, 0.1)  # Advanced features bonus
        help_penalty = self.help_request_rate * 0.1  # Penalty for frequent help requests
        
        return min(base_score + success_bonus + pattern_bonus + advanced_bonus - help_penalty, 1.0)


class IntelligenceError(Exception):
    """Base exception for intelligence system errors."""
    pass


class ProfileCorruptedError(IntelligenceError):
    """Raised when user profile data is corrupted or invalid."""
    pass


class InsufficientDataError(IntelligenceError):
    """Raised when insufficient data is available for analysis."""
    pass


class AnalysisFailedError(IntelligenceError):
    """Raised when pattern analysis or learning fails."""
    pass


class StorageUnavailableError(IntelligenceError):
    """Raised when user profile storage is unavailable."""
    pass