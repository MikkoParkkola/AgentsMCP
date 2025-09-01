"""
Core data models for the thinking and planning system.

These models define the structure for thinking processes, task decomposition,
evaluation criteria, and execution planning.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum


class ThinkingPhase(Enum):
    """Phases of the thinking process."""
    ANALYZE_REQUEST = "analyze"      # Understand the request
    EXPLORE_OPTIONS = "explore"      # Generate multiple approaches  
    EVALUATE_APPROACHES = "evaluate" # Score and rank options
    SELECT_STRATEGY = "select"       # Choose best approach
    DECOMPOSE_TASKS = "decompose"    # Break into sub-tasks
    PLAN_EXECUTION = "plan"          # Schedule execution order
    REFLECT_ADJUST = "reflect"       # Meta-cognitive review


class TaskType(Enum):
    """Types of tasks for execution planning."""
    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    COORDINATION = "coordination"
    VALIDATION = "validation"


class DecompositionStrategy(Enum):
    """Strategies for breaking down complex tasks."""
    SEQUENTIAL = "sequential"      # Break into ordered steps
    PARALLEL = "parallel"         # Identify parallel execution opportunities
    HIERARCHICAL = "hierarchical" # Create task hierarchies
    DEPENDENCY_DRIVEN = "dependency_driven" # Focus on dependency analysis


class EvaluationCriterion(Enum):
    """Criteria for evaluating approaches."""
    FEASIBILITY = "feasibility"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    COST = "cost"
    TIME_TO_COMPLETION = "time_to_completion"
    RISK_LEVEL = "risk_level"


@dataclass
class ThinkingStep:
    """A single step in the thinking process."""
    phase: ThinkingPhase
    timestamp: datetime
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    duration_ms: Optional[int] = None


@dataclass
class Approach:
    """A potential approach to solving a problem."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    steps: List[str] = field(default_factory=list)
    estimated_effort: Optional[float] = None
    risks: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationScore:
    """Evaluation score for a specific criterion."""
    criterion: EvaluationCriterion
    score: float  # 0.0 to 1.0
    weight: float = 1.0
    rationale: str = ""
    
    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class EvaluationCriteria:
    """Criteria and weights for approach evaluation."""
    criteria: Dict[EvaluationCriterion, float] = field(default_factory=dict)
    custom_criteria: Dict[str, float] = field(default_factory=dict)
    
    def normalize_weights(self):
        """Normalize all weights to sum to 1.0."""
        total_weight = sum(self.criteria.values()) + sum(self.custom_criteria.values())
        if total_weight > 0:
            for criterion in self.criteria:
                self.criteria[criterion] /= total_weight
            for criterion in self.custom_criteria:
                self.custom_criteria[criterion] /= total_weight


@dataclass
class RankedApproach:
    """An approach with its evaluation score and ranking."""
    approach: Approach
    total_score: float
    individual_scores: Dict[EvaluationCriterion, EvaluationScore]
    rank: int
    selection_rationale: str = ""


@dataclass 
class SubTask:
    """A decomposed sub-task."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    task_type: TaskType = TaskType.IMPLEMENTATION
    priority: int = 0  # Higher numbers = higher priority
    estimated_duration: Optional[timedelta] = None
    required_resources: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # IDs of dependent tasks
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyGraph:
    """Graph representing task dependencies."""
    tasks: Dict[str, SubTask] = field(default_factory=dict)
    edges: Dict[str, List[str]] = field(default_factory=dict)  # task_id -> [dependent_task_ids]
    
    def add_task(self, task: SubTask):
        """Add a task to the dependency graph."""
        self.tasks[task.id] = task
        if task.id not in self.edges:
            self.edges[task.id] = []
    
    def add_dependency(self, task_id: str, depends_on_id: str):
        """Add a dependency relationship."""
        if depends_on_id not in self.edges:
            self.edges[depends_on_id] = []
        if task_id not in self.edges[depends_on_id]:
            self.edges[depends_on_id].append(task_id)
    
    def get_parallel_groups(self) -> List[List[str]]:
        """Identify groups of tasks that can run in parallel."""
        visited = set()
        parallel_groups = []
        
        def find_parallel_tasks(task_id: str) -> List[str]:
            if task_id in visited:
                return []
            
            visited.add(task_id)
            group = [task_id]
            
            # Find tasks with no remaining dependencies
            for other_id, other_task in self.tasks.items():
                if (other_id not in visited and 
                    all(dep_id in visited for dep_id in other_task.dependencies)):
                    group.extend(find_parallel_tasks(other_id))
            
            return group
        
        for task_id in self.tasks:
            group = find_parallel_tasks(task_id)
            if group:
                parallel_groups.append(group)
        
        return parallel_groups
    
    def detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies."""
        def dfs(node: str, path: List[str], visited: set) -> Optional[List[str]]:
            if node in path:
                # Found cycle - return the cycle portion
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]
            
            if node in visited:
                return None
                
            visited.add(node)
            path.append(node)
            
            for dependent in self.edges.get(node, []):
                cycle = dfs(dependent, path, visited)
                if cycle:
                    return cycle
            
            path.pop()
            return None
        
        cycles = []
        visited = set()
        
        for task_id in self.tasks:
            if task_id not in visited:
                cycle = dfs(task_id, [], visited)
                if cycle:
                    cycles.append(cycle)
        
        return cycles


@dataclass
class ResourceConstraints:
    """Resource constraints for execution planning."""
    max_concurrent_tasks: int = 10
    available_agents: List[str] = field(default_factory=list)
    memory_limit_mb: Optional[int] = None
    time_limit: Optional[timedelta] = None
    priority_weights: Dict[TaskType, float] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """Resource allocation for a specific task."""
    task_id: str
    assigned_agents: List[str] = field(default_factory=list)
    allocated_memory_mb: int = 0
    estimated_start_time: Optional[datetime] = None
    estimated_end_time: Optional[datetime] = None


@dataclass
class Checkpoint:
    """A checkpoint in the execution plan."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    completed_tasks: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    rollback_instructions: str = ""


@dataclass
class ExecutionSchedule:
    """Optimized task execution schedule."""
    tasks: List[SubTask] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)  # Task IDs in execution order
    parallel_groups: List[List[str]] = field(default_factory=list)
    resource_allocations: Dict[str, ResourceAllocation] = field(default_factory=dict)
    checkpoints: List[Checkpoint] = field(default_factory=list)
    estimated_total_duration: Optional[timedelta] = None
    critical_path: List[str] = field(default_factory=list)


@dataclass
class StrategyAdjustment:
    """Adjustment to thinking/execution strategy."""
    adjustment_type: str
    description: str
    impact_level: str  # "low", "medium", "high"
    implementation_steps: List[str] = field(default_factory=list)
    expected_improvement: str = ""


@dataclass
class QualityAssessment:
    """Assessment of thinking process quality."""
    overall_quality: float  # 0.0 to 1.0
    phase_scores: Dict[ThinkingPhase, float] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_areas: List[str] = field(default_factory=list)
    confidence_accuracy: Optional[float] = None  # How well-calibrated was confidence


@dataclass
class ExecutionResult:
    """Result of executing a planned task."""
    task_id: str
    success: bool
    duration: timedelta
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)


@dataclass
class ThinkingResult:
    """Complete result of a thinking process."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_request: str = ""
    selected_approach: Optional[RankedApproach] = None
    execution_plan: Optional[ExecutionSchedule] = None
    thinking_trace: List[ThinkingStep] = field(default_factory=list)
    confidence: float = 0.0
    total_thinking_time_ms: int = 0
    quality_assessment: Optional[QualityAssessment] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_thinking_step(self, step: ThinkingStep):
        """Add a thinking step to the trace."""
        self.thinking_trace.append(step)
    
    def get_phase_duration(self, phase: ThinkingPhase) -> int:
        """Get total duration spent in a specific phase."""
        return sum(step.duration_ms or 0 for step in self.thinking_trace if step.phase == phase)


@dataclass
class PlanningState:
    """State of an ongoing planning process."""
    request_id: str
    current_phase: ThinkingPhase
    thinking_result: ThinkingResult
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    is_complete: bool = False
    error: Optional[str] = None
    
    def update_phase(self, phase: ThinkingPhase):
        """Update the current phase and timestamp."""
        self.current_phase = phase
        self.last_updated = datetime.now()