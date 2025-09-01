"""
Configuration models and defaults for the thinking and planning system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import timedelta

from .models import ThinkingPhase, EvaluationCriterion, DecompositionStrategy, TaskType


@dataclass
class ThinkingConfig:
    """Configuration for the thinking framework."""
    
    # General settings
    enabled: bool = True
    timeout_ms: int = 500  # Maximum thinking time for simple tasks
    complex_timeout_ms: int = 2000  # Maximum thinking time for complex tasks
    
    # Phase configuration
    enabled_phases: List[ThinkingPhase] = field(default_factory=lambda: [
        ThinkingPhase.ANALYZE_REQUEST,
        ThinkingPhase.EXPLORE_OPTIONS,
        ThinkingPhase.EVALUATE_APPROACHES,
        ThinkingPhase.SELECT_STRATEGY,
        ThinkingPhase.DECOMPOSE_TASKS,
        ThinkingPhase.PLAN_EXECUTION
    ])
    
    # When to enable reflection phase (expensive)
    enable_reflection_for_complex_tasks: bool = True
    enable_reflection_for_failed_executions: bool = True
    
    # Approach generation and evaluation
    min_approaches_to_generate: int = 2
    max_approaches_to_generate: int = 5
    evaluation_criteria_weights: Dict[EvaluationCriterion, float] = field(default_factory=lambda: {
        EvaluationCriterion.FEASIBILITY: 0.25,
        EvaluationCriterion.PERFORMANCE: 0.20,
        EvaluationCriterion.RELIABILITY: 0.20,
        EvaluationCriterion.TIME_TO_COMPLETION: 0.15,
        EvaluationCriterion.RISK_LEVEL: 0.10,
        EvaluationCriterion.MAINTAINABILITY: 0.10
    })
    
    # Task decomposition settings
    max_decomposition_depth: int = 3
    max_subtasks_per_level: int = 10
    preferred_decomposition_strategy: DecompositionStrategy = DecompositionStrategy.DEPENDENCY_DRIVEN
    
    # Execution planning settings
    max_concurrent_tasks: int = 10
    enable_parallel_execution: bool = True
    checkpoint_frequency: int = 5  # Insert checkpoint every N tasks
    
    # Resource constraints
    default_memory_limit_mb: int = 1024
    default_time_limit_minutes: int = 30
    
    # Metacognitive monitoring
    enable_quality_monitoring: bool = True
    confidence_calibration_enabled: bool = True
    adaptation_threshold: float = 0.7  # Quality score below which to adapt strategies
    
    # State persistence
    enable_state_persistence: bool = True
    state_cleanup_after_hours: int = 24
    max_stored_states: int = 100
    
    # Performance tuning
    cache_thinking_results: bool = True
    cache_ttl_minutes: int = 60
    enable_thinking_shortcuts: bool = True  # Skip phases for very simple requests
    
    def is_complex_task(self, request: str) -> bool:
        """Determine if a request represents a complex task."""
        complexity_indicators = [
            'implement', 'create', 'build', 'design', 'architect',
            'multiple', 'system', 'integrate', 'optimize', 'refactor',
            'comprehensive', 'end-to-end', 'full-stack'
        ]
        
        return (
            len(request) > 100 or
            any(indicator in request.lower() for indicator in complexity_indicators) or
            request.count('.') > 3 or  # Multiple sentences
            request.count(',') > 2     # Multiple clauses
        )
    
    def get_timeout_for_request(self, request: str) -> int:
        """Get appropriate timeout for a request."""
        return self.complex_timeout_ms if self.is_complex_task(request) else self.timeout_ms
    
    def should_enable_reflection(self, request: str, had_execution_errors: bool = False) -> bool:
        """Determine if reflection phase should be enabled."""
        return (
            self.enable_reflection_for_failed_executions and had_execution_errors
        ) or (
            self.enable_reflection_for_complex_tasks and self.is_complex_task(request)
        )


@dataclass
class ApproachEvaluatorConfig:
    """Configuration for approach evaluation."""
    
    # Scoring settings
    use_weighted_scoring: bool = True
    normalize_scores: bool = True
    confidence_threshold: float = 0.6  # Minimum confidence to proceed with approach
    
    # Evaluation criteria
    required_criteria: List[EvaluationCriterion] = field(default_factory=lambda: [
        EvaluationCriterion.FEASIBILITY,
        EvaluationCriterion.RISK_LEVEL
    ])
    
    # Performance settings
    max_evaluation_time_ms: int = 200
    parallel_evaluation: bool = True


@dataclass  
class TaskDecomposerConfig:
    """Configuration for task decomposition."""
    
    # Decomposition limits
    max_depth: int = 3
    max_subtasks_total: int = 50
    min_task_complexity_for_decomposition: int = 2  # Skip trivial tasks
    
    # Dependency analysis
    enable_dependency_detection: bool = True
    enable_parallel_task_identification: bool = True
    detect_circular_dependencies: bool = True
    
    # Task categorization
    task_type_keywords: Dict[TaskType, List[str]] = field(default_factory=lambda: {
        TaskType.ANALYSIS: ['analyze', 'investigate', 'research', 'examine'],
        TaskType.IMPLEMENTATION: ['implement', 'create', 'build', 'develop', 'write'],
        TaskType.TESTING: ['test', 'verify', 'validate', 'check', 'ensure'],
        TaskType.DOCUMENTATION: ['document', 'write docs', 'explain', 'describe'],
        TaskType.COORDINATION: ['coordinate', 'sync', 'integrate', 'merge'],
        TaskType.VALIDATION: ['review', 'approve', 'confirm', 'validate']
    })


@dataclass
class ExecutionPlannerConfig:
    """Configuration for execution planning."""
    
    # Scheduling settings
    optimization_strategy: str = "critical_path"  # or "load_balanced", "fastest"
    enable_resource_optimization: bool = True
    enable_checkpoint_insertion: bool = True
    
    # Resource management
    default_agent_concurrency: int = 3
    memory_safety_margin: float = 0.1  # Reserve 10% of memory limit
    time_safety_margin: float = 0.2    # Reserve 20% of time limit
    
    # Risk management
    insert_validation_checkpoints: bool = True
    enable_rollback_planning: bool = True
    risk_tolerance: str = "medium"  # low, medium, high


@dataclass
class MetacognitiveConfig:
    """Configuration for metacognitive monitoring."""
    
    # Quality assessment
    enable_phase_scoring: bool = True
    enable_confidence_calibration: bool = True
    enable_strategy_adaptation: bool = True
    
    # Learning settings
    learning_rate: float = 0.1
    adaptation_sensitivity: float = 0.05
    min_samples_for_adaptation: int = 5
    
    # Feedback collection
    collect_execution_feedback: bool = True
    feedback_weight: float = 0.3  # Weight of execution results vs. thinking quality


@dataclass
class StateManagementConfig:
    """Configuration for state persistence."""
    
    # Storage settings
    storage_backend: str = "memory"  # memory, file, redis
    encryption_enabled: bool = True
    compression_enabled: bool = True
    
    # Cleanup policies
    auto_cleanup_enabled: bool = True
    max_age_hours: int = 24
    max_states_per_user: int = 100
    
    # Recovery settings
    enable_state_recovery: bool = True
    recovery_timeout_ms: int = 1000
    validate_recovered_state: bool = True


# Default configuration instance
DEFAULT_THINKING_CONFIG = ThinkingConfig()
DEFAULT_EVALUATOR_CONFIG = ApproachEvaluatorConfig()
DEFAULT_DECOMPOSER_CONFIG = TaskDecomposerConfig()
DEFAULT_PLANNER_CONFIG = ExecutionPlannerConfig()
DEFAULT_METACOGNITIVE_CONFIG = MetacognitiveConfig()
DEFAULT_STATE_CONFIG = StateManagementConfig()


def create_lightweight_config() -> ThinkingConfig:
    """Create a lightweight configuration for simple tasks."""
    config = ThinkingConfig()
    config.timeout_ms = 200
    config.complex_timeout_ms = 500
    config.enabled_phases = [
        ThinkingPhase.ANALYZE_REQUEST,
        ThinkingPhase.EVALUATE_APPROACHES,
        ThinkingPhase.SELECT_STRATEGY
    ]
    config.min_approaches_to_generate = 1
    config.max_approaches_to_generate = 2
    config.enable_reflection_for_complex_tasks = False
    config.enable_quality_monitoring = False
    return config


def create_comprehensive_config() -> ThinkingConfig:
    """Create a comprehensive configuration for complex tasks."""
    config = ThinkingConfig()
    config.timeout_ms = 1000
    config.complex_timeout_ms = 5000
    config.enabled_phases = list(ThinkingPhase)  # All phases
    config.min_approaches_to_generate = 3
    config.max_approaches_to_generate = 7
    config.enable_reflection_for_complex_tasks = True
    config.enable_quality_monitoring = True
    config.confidence_calibration_enabled = True
    return config