"""Core data models for dynamic agent loading system.

This module defines the fundamental data structures used by the task classifier
and team orchestrator for dynamic agent loading and team composition.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timezone

from pydantic import BaseModel, Field, validator


class TaskType(str, Enum):
    """High-level task classification."""
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    ANALYSIS = "analysis"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    BUG_FIX = "bug_fix"
    MAINTENANCE = "maintenance"
    RESEARCH = "research"


class ComplexityLevel(str, Enum):
    """Task complexity assessment."""
    TRIVIAL = "trivial"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(str, Enum):
    """Risk assessment for task execution."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TechnologyStack(str, Enum):
    """Supported technology stacks."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    REACT = "react"
    NODEJS = "nodejs"
    API = "api"
    DATABASE = "database"
    DEVOPS = "devops"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    TUI = "tui"
    CLI = "cli"
    MACHINE_LEARNING = "machine_learning"
    DATA_ANALYSIS = "data_analysis"


class CoordinationStrategy(str, Enum):
    """Team coordination approaches."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"
    PIPELINE = "pipeline"


class TaskClassification(BaseModel):
    """Classification result for a given task."""
    
    task_type: TaskType = Field(..., description="Primary task type")
    complexity: ComplexityLevel = Field(..., description="Estimated complexity level")
    required_roles: List[str] = Field(default_factory=list, description="Essential roles for task completion")
    optional_roles: List[str] = Field(default_factory=list, description="Beneficial but not essential roles")
    technologies: List[TechnologyStack] = Field(default_factory=list, description="Technology stacks involved")
    estimated_effort: int = Field(..., ge=1, le=100, description="Effort estimation (1-100 scale)")
    risk_level: RiskLevel = Field(..., description="Risk assessment")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    
    @validator('required_roles', 'optional_roles', pre=True)
    def validate_roles(cls, v):
        if not isinstance(v, list):
            return []
        return [role.strip() for role in v if isinstance(role, str) and role.strip()]
    
    @validator('keywords', pre=True)
    def validate_keywords(cls, v):
        if not isinstance(v, list):
            return []
        return [kw.lower().strip() for kw in v if isinstance(kw, str) and kw.strip()]


class AgentSpec(BaseModel):
    """Specification for an agent in a team composition."""
    
    role: str = Field(..., description="Agent role identifier")
    model_assignment: str = Field(..., description="Preferred model/agent type")
    priority: int = Field(default=1, ge=1, le=10, description="Priority level (1=highest, 10=lowest)")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Resource needs")
    specializations: List[str] = Field(default_factory=list, description="Specialized capabilities")
    
    @validator('role')
    def validate_role(cls, v):
        if not v or not v.strip():
            raise ValueError("Role cannot be empty")
        return v.strip()


class ResourceConstraints(BaseModel):
    """Resource limitations for team composition."""
    
    max_agents: int = Field(default=5, ge=1, le=20, description="Maximum number of agents")
    memory_limit: Optional[int] = Field(default=None, description="Memory limit in MB")
    time_budget: Optional[int] = Field(default=None, description="Time budget in seconds")
    cost_budget: Optional[float] = Field(default=None, description="Cost budget in EUR")
    parallel_limit: int = Field(default=3, ge=1, description="Maximum parallel agents")
    
    @validator('cost_budget')
    def validate_cost_budget(cls, v):
        if v is not None and v < 0:
            raise ValueError("Cost budget cannot be negative")
        return v


class TeamComposition(BaseModel):
    """Composed team for task execution."""
    
    primary_team: List[AgentSpec] = Field(..., description="Primary agents for the task")
    fallback_agents: List[AgentSpec] = Field(default_factory=list, description="Backup agents")
    load_order: List[str] = Field(..., description="Order in which to load agents")
    coordination_strategy: CoordinationStrategy = Field(..., description="How agents coordinate")
    estimated_cost: Optional[float] = Field(default=None, description="Estimated execution cost in EUR")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Composition confidence")
    rationale: str = Field(default="", description="Reasoning behind team selection")
    
    @validator('primary_team')
    def validate_primary_team(cls, v):
        if not v:
            raise ValueError("Primary team cannot be empty")
        return v
    
    @validator('load_order')
    def validate_load_order(cls, v, values):
        primary_roles = {agent.role for agent in values.get('primary_team', [])}
        if not all(role in primary_roles for role in v):
            raise ValueError("Load order must include only primary team roles")
        return v


class TaskResult(BaseModel):
    """Result of task execution."""
    
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Execution status")
    artifacts: List[str] = Field(default_factory=list, description="Generated artifacts")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Execution metrics")
    errors: List[str] = Field(default_factory=list, description="Encountered errors")
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = {'pending', 'running', 'completed', 'failed', 'cancelled'}
        if v.lower() not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return v.lower()


class TeamPerformanceMetrics(BaseModel):
    """Performance tracking for team compositions."""
    
    team_id: str = Field(..., description="Team composition identifier")
    task_type: TaskType = Field(..., description="Type of task executed")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Historical success rate")
    average_duration: Optional[float] = Field(default=None, description="Average completion time in seconds")
    average_cost: Optional[float] = Field(default=None, description="Average execution cost in EUR")
    total_executions: int = Field(default=0, ge=0, description="Total number of executions")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @validator('total_executions')
    def validate_executions(cls, v):
        if v < 0:
            raise ValueError("Total executions cannot be negative")
        return v


class ClassificationCache(BaseModel):
    """Cache entry for task classifications."""
    
    objective_hash: str = Field(..., description="Hash of the objective text")
    context_hash: str = Field(..., description="Hash of the context")
    classification: TaskClassification = Field(..., description="Cached classification")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hit_count: int = Field(default=1, ge=1, description="Number of cache hits")
    
    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if cache entry has expired."""
        age = datetime.now(timezone.utc) - self.created_at
        return age.total_seconds() > ttl_seconds


# Exception classes for error handling
class TaskClassificationError(Exception):
    """Base exception for task classification errors."""
    pass


class InvalidObjective(TaskClassificationError):
    """Raised when the objective is invalid or empty."""
    pass


class InsufficientContext(TaskClassificationError):
    """Raised when insufficient context is provided for classification."""
    pass


class UnsupportedTaskType(TaskClassificationError):
    """Raised when the task type is not supported by the system."""
    pass


class TeamCompositionError(Exception):
    """Base exception for team composition errors."""
    pass


class InsufficientResources(TeamCompositionError):
    """Raised when resource constraints cannot be satisfied."""
    pass


class NoSuitableAgents(TeamCompositionError):
    """Raised when no suitable agents are available for the task."""
    pass