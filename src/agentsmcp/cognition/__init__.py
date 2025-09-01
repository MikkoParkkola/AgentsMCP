"""
Advanced LLM Thinking and Planning System for AgentsMCP.

This module implements deliberative planning loops that enhance decision-making
quality through structured thinking phases before action execution.

Core Components:
- ThinkingFramework: Main coordination of thinking phases
- ApproachEvaluator: Multi-option evaluation and ranking
- TaskDecomposer: Intelligent task breakdown with dependency analysis
- ExecutionPlanner: Sub-task scheduling and resource allocation
- MetacognitiveMonitor: Self-reflection and strategy adjustment
- PlanningStateManager: State persistence and recovery
- ThinkingOrchestrator: Orchestrator integration with thinking capabilities
"""

from .thinking_framework import ThinkingFramework, ThinkingResult
from .approach_evaluator import ApproachEvaluator, RankedApproach
from .task_decomposer import TaskDecomposer, SubTask, DependencyGraph
from .execution_planner import ExecutionPlanner, ExecutionSchedule
from .metacognitive_monitor import MetacognitiveMonitor, QualityAssessment
from .planning_state_manager import (
    PlanningStateManager, PlanningState, StateMetadata, StateRecoveryInfo,
    create_state_manager, save_thinking_result, load_thinking_result
)
from .orchestrator_wrapper import (
    ThinkingOrchestrator, ThinkingOrchestratorConfig,
    create_thinking_orchestrator, create_fast_thinking_orchestrator,
    create_comprehensive_thinking_orchestrator
)
from .models import (
    ThinkingPhase, ThinkingStep, Approach, EvaluationCriteria,
    DecompositionStrategy, ResourceConstraints, StrategyAdjustment,
    PersistenceFormat, CheckpointStrategy, CleanupPolicy,
    OrchestratorIntegrationMode, ThinkingScope, PerformanceProfile
)
from .config import ThinkingConfig, DEFAULT_THINKING_CONFIG

__version__ = "1.0.0"

__all__ = [
    # Core framework
    "ThinkingFramework",
    "ThinkingResult", 
    "ThinkingConfig",
    "DEFAULT_THINKING_CONFIG",
    
    # Evaluation system
    "ApproachEvaluator",
    "RankedApproach",
    
    # Task decomposition
    "TaskDecomposer", 
    "SubTask",
    "DependencyGraph",
    
    # Execution planning
    "ExecutionPlanner",
    "ExecutionSchedule",
    
    # Metacognitive monitoring
    "MetacognitiveMonitor",
    "QualityAssessment",
    
    # State management
    "PlanningStateManager",
    "PlanningState",
    "StateMetadata",
    "StateRecoveryInfo",
    "create_state_manager",
    "save_thinking_result",
    "load_thinking_result",
    
    # Orchestrator integration
    "ThinkingOrchestrator",
    "ThinkingOrchestratorConfig", 
    "create_thinking_orchestrator",
    "create_fast_thinking_orchestrator",
    "create_comprehensive_thinking_orchestrator",
    
    # Models and configuration
    "ThinkingPhase",
    "ThinkingStep", 
    "Approach",
    "EvaluationCriteria",
    "DecompositionStrategy",
    "ResourceConstraints",
    "StrategyAdjustment",
    "PersistenceFormat",
    "CheckpointStrategy",
    "CleanupPolicy",
    "OrchestratorIntegrationMode",
    "ThinkingScope",
    "PerformanceProfile"
]