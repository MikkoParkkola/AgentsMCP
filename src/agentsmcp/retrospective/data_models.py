"""Enhanced retrospective data models for individual agent and comprehensive analysis.

This module defines comprehensive data structures for the enhanced retrospective system
including individual agent retrospectives, agile coach analysis, and enforcement tracking.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..roles.base import RoleName


class RetrospectiveType(str, Enum):
    """Types of retrospectives that can be conducted."""
    INDIVIDUAL = "individual"
    COMPREHENSIVE = "comprehensive"
    POST_TASK = "post_task"
    SPRINT = "sprint"
    INCIDENT = "incident"


class ImprovementCategory(str, Enum):
    """Categories for improvement actions."""
    PERFORMANCE = "performance"
    COORDINATION = "coordination"
    TOOL_USAGE = "tool_usage"
    DECISION_MAKING = "decision_making"
    COMMUNICATION = "communication"
    PROCESS = "process"
    LEARNING = "learning"


class PriorityLevel(str, Enum):
    """Priority levels for action points."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ImplementationStatus(str, Enum):
    """Status of action point implementation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class DecisionPoint:
    """Represents a decision made during task execution."""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    decision_description: str = ""
    options_considered: List[str] = field(default_factory=list)
    chosen_option: str = ""
    rationale: str = ""
    outcome: Optional[str] = None
    confidence_level: float = 0.0  # 0.0 to 1.0


@dataclass
class ToolUsage:
    """Represents usage of a tool during task execution."""
    tool_name: str
    usage_count: int = 0
    success_rate: float = 0.0
    total_time_seconds: float = 0.0
    outcomes: List[str] = field(default_factory=list)
    efficiency_assessment: Optional[str] = None


@dataclass
class Challenge:
    """Represents a challenge encountered during execution."""
    challenge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    category: str = ""  # e.g., "technical", "coordination", "resource"
    impact_level: str = "medium"  # low, medium, high, critical
    resolution_attempted: Optional[str] = None
    resolution_successful: bool = False
    lessons_learned: List[str] = field(default_factory=list)


@dataclass
class PerformanceAssessment:
    """Agent's self-assessment of performance."""
    overall_score: float = 0.0  # 0.0 to 1.0
    efficiency_score: float = 0.0
    quality_score: float = 0.0
    collaboration_score: float = 0.0
    learning_score: float = 0.0
    self_assessment_notes: str = ""
    areas_of_strength: List[str] = field(default_factory=list)
    areas_for_improvement: List[str] = field(default_factory=list)


@dataclass
class SelfImprovementAction:
    """Individual improvement action identified by agent."""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    category: ImprovementCategory = ImprovementCategory.PERFORMANCE
    priority: PriorityLevel = PriorityLevel.MEDIUM
    estimated_effort: str = ""  # e.g., "1 hour", "1 day"
    expected_benefit: str = ""
    implementation_notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class IndividualRetrospective:
    """Complete individual agent retrospective data."""
    agent_role: RoleName
    task_id: str
    retrospective_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    retrospective_type: RetrospectiveType = RetrospectiveType.INDIVIDUAL
    
    # Core retrospective data
    execution_summary: str = ""
    what_went_well: List[str] = field(default_factory=list)
    what_could_improve: List[str] = field(default_factory=list)
    challenges_encountered: List[Challenge] = field(default_factory=list)
    
    # Performance self-assessment
    performance_assessment: PerformanceAssessment = field(default_factory=PerformanceAssessment)
    
    # Decision and tool analysis
    decisions_made: List[DecisionPoint] = field(default_factory=list)
    tools_used: List[ToolUsage] = field(default_factory=list)
    
    # Learning and improvement
    key_learnings: List[str] = field(default_factory=list)
    knowledge_gaps_identified: List[str] = field(default_factory=list)
    self_improvement_actions: List[SelfImprovementAction] = field(default_factory=list)
    
    # Collaboration insights
    collaboration_feedback: str = ""
    team_dynamics_observations: List[str] = field(default_factory=list)
    communication_effectiveness: float = 0.0  # 0.0 to 1.0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = 0.0
    agent_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemicIssue:
    """Systemic issue identified across multiple agents."""
    issue_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    category: str = ""
    severity: str = "medium"  # low, medium, high, critical
    affected_agents: List[str] = field(default_factory=list)
    occurrence_count: int = 0
    root_cause_analysis: str = ""
    potential_solutions: List[str] = field(default_factory=list)


@dataclass
class PatternAnalysis:
    """Analysis of patterns identified across agent retrospectives."""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str = ""  # success, failure, collaboration, etc.
    pattern_description: str = ""
    confidence_score: float = 0.0  # 0.0 to 1.0
    supporting_evidence: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class CrossAgentInsights:
    """Insights from analyzing interactions between agents."""
    collaboration_effectiveness: float = 0.0  # 0.0 to 1.0
    communication_patterns: List[str] = field(default_factory=list)
    coordination_challenges: List[str] = field(default_factory=list)
    synergy_opportunities: List[str] = field(default_factory=list)
    role_boundary_issues: List[str] = field(default_factory=list)


@dataclass
class SystemicImprovement:
    """System-wide improvement recommendation."""
    improvement_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    category: ImprovementCategory = ImprovementCategory.PROCESS
    priority: PriorityLevel = PriorityLevel.MEDIUM
    impact_assessment: str = ""
    effort_assessment: str = ""
    implementation_approach: str = ""
    success_metrics: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ActionPoint:
    """Specific action point for orchestrator enforcement."""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    category: str = ""
    priority: PriorityLevel = PriorityLevel.MEDIUM
    
    # Implementation details
    implementation_type: str = ""  # automatic, manual, configuration
    implementation_steps: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    
    # Tracking
    status: ImplementationStatus = ImplementationStatus.PENDING
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Impact and effort
    estimated_effort_hours: float = 0.0
    expected_impact: str = ""
    success_metrics: List[str] = field(default_factory=list)
    
    # Implementation tracking
    implementation_notes: str = ""
    validation_results: List[str] = field(default_factory=list)


@dataclass
class PriorityMatrix:
    """Impact/effort analysis of action points."""
    high_impact_low_effort: List[ActionPoint] = field(default_factory=list)
    high_impact_high_effort: List[ActionPoint] = field(default_factory=list)
    low_impact_low_effort: List[ActionPoint] = field(default_factory=list)
    low_impact_high_effort: List[ActionPoint] = field(default_factory=list)


@dataclass
class ImplementationRoadmap:
    """Sequenced implementation plan for action points."""
    roadmap_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phases: List[Dict[str, Any]] = field(default_factory=list)  # Each phase contains action_ids and timeline
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # action_id -> [dependent_action_ids]
    estimated_total_duration: str = ""
    critical_path: List[str] = field(default_factory=list)  # action_ids on critical path


@dataclass
class ComprehensiveRetrospectiveReport:
    """Complete comprehensive retrospective report from agile coach analysis."""
    task_id: str
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    retrospective_type: RetrospectiveType = RetrospectiveType.COMPREHENSIVE
    
    # Input data summary
    individual_retrospectives_count: int = 0
    participating_agents: List[str] = field(default_factory=list)
    
    # Analysis results
    pattern_analysis: List[PatternAnalysis] = field(default_factory=list)
    systemic_issues: List[SystemicIssue] = field(default_factory=list)
    cross_agent_insights: CrossAgentInsights = field(default_factory=CrossAgentInsights)
    
    # Synthesized insights
    overall_team_performance: float = 0.0  # 0.0 to 1.0
    collaboration_effectiveness: float = 0.0
    learning_outcomes: List[str] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    improvement_opportunities: List[str] = field(default_factory=list)
    
    # Recommendations and actions
    systemic_improvements: List[SystemicImprovement] = field(default_factory=list)
    action_points: List[ActionPoint] = field(default_factory=list)
    priority_matrix: PriorityMatrix = field(default_factory=PriorityMatrix)
    implementation_roadmap: ImplementationRoadmap = field(default_factory=ImplementationRoadmap)
    
    # Future recommendations
    next_task_recommendations: List[str] = field(default_factory=list)
    team_optimization_suggestions: List[str] = field(default_factory=list)
    process_improvements: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    analysis_duration_seconds: float = 0.0
    agile_coach_notes: str = ""


@dataclass
class IndividualRetrospectiveConfig:
    """Configuration for individual agent retrospectives."""
    timeout_seconds: int = 30
    max_challenges: int = 10
    max_learnings: int = 10
    max_improvement_actions: int = 5
    include_decision_analysis: bool = True
    include_tool_usage_analysis: bool = True
    include_collaboration_feedback: bool = True
    anonymize_sensitive_data: bool = True
    performance_assessment_required: bool = True


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator enforcement system."""
    enforcement_enabled: bool = True
    auto_implement_safe_actions: bool = False
    max_concurrent_implementations: int = 3
    implementation_timeout_seconds: int = 300
    require_manual_approval_for_critical: bool = True
    rollback_enabled: bool = True
    validation_required: bool = True


@dataclass
class EnforcementPlan:
    """Plan for enforcing action point implementation."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_points: List[ActionPoint] = field(default_factory=list)
    implementation_sequence: List[str] = field(default_factory=list)  # action_ids in order
    validation_steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_completion_time: str = ""
    rollback_procedures: List[str] = field(default_factory=list)


@dataclass
class ValidationCriterion:
    """Criteria for validating action point completion."""
    action_id: str
    criterion_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    validation_type: str = ""  # automated, manual, metric-based
    validation_script: Optional[str] = None
    expected_outcome: str = ""
    tolerance: Optional[str] = None  # for metric-based validations


@dataclass
class ReadinessAssessment:
    """Assessment of system readiness for next task."""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    overall_readiness_score: float = 0.0  # 0.0 to 1.0
    blocking_issues: List[str] = field(default_factory=list)
    pending_action_points: List[str] = field(default_factory=list)  # action_ids
    system_health_indicators: Dict[str, float] = field(default_factory=dict)
    readiness_notes: str = ""
    next_task_clearance: bool = False
    estimated_time_to_ready: Optional[str] = None


# Pydantic models for API compatibility

class IndividualRetrospectiveModel(BaseModel):
    """Pydantic model for individual retrospective API."""
    retrospective_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_role: str
    task_id: str
    execution_summary: str = ""
    what_went_well: List[str] = Field(default_factory=list)
    what_could_improve: List[str] = Field(default_factory=list)
    key_learnings: List[str] = Field(default_factory=list)
    performance_score: float = Field(ge=0.0, le=1.0, default=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ComprehensiveRetrospectiveReportModel(BaseModel):
    """Pydantic model for comprehensive retrospective report API."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    individual_retrospectives_count: int = 0
    participating_agents: List[str] = Field(default_factory=list)
    overall_team_performance: float = Field(ge=0.0, le=1.0, default=0.0)
    action_points_count: int = 0
    systemic_issues_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ActionPointModel(BaseModel):
    """Pydantic model for action point API."""
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    priority: str = "medium"
    status: str = "pending"
    estimated_effort_hours: float = Field(ge=0.0, default=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))