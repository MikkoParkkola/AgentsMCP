"""
AgentsMCP Orchestration Module

Unified integration API for dynamic agent loading with:
- Dynamic task classification and team composition
- Agile coach integration for complex tasks  
- Intelligent orchestration with fallback mechanisms
- Comprehensive retrospectives and continuous improvement
- Backward compatibility with existing team runner API

AGENTS.md v2 Two-Tier Orchestration System with:
- Single main loop coordination (Tier 1)
- Stateless agent functions (Tier 2)
- Quality gates and validation
- Structured envelope communication

Legacy systems (seamless, emotional, symphony) remain for backward compatibility.
"""

# Core orchestration components - Dynamic team runner v2 is now the default
from .team_runner_v2 import run_team  # Enhanced team runner with dynamic orchestration
from .team_runner import run_team as run_team_legacy  # Legacy team runner for fallback

# NEW: Strict Orchestrator-Only Communication Architecture
from .orchestrator import Orchestrator, OrchestratorConfig, OrchestratorMode, OrchestratorResponse
from .task_classifier import TaskClassifier
from .team_composer import TeamComposer
from .dynamic_orchestrator import DynamicOrchestrator
from .agile_coach import AgileCoachIntegration
from .retrospective_engine import RetrospectiveEngine

# Data models and types
from .models import (
    TaskClassification,
    TaskType,
    ComplexityLevel,
    RiskLevel,
    TechnologyStack,
    CoordinationStrategy,
    AgentSpec,
    ResourceConstraints,
    TeamComposition,
    TeamPerformanceMetrics,
    TaskResult,
)

# Resource management
from .resource_manager import ResourceManager, ResourceType

# Execution engine
from .execution_engine import ExecutionEngine, TeamExecution, ExecutionStatus, ExecutionProgress

# AGENTS.md v2 Two-Tier Architecture
try:
    from .coordinator import MainCoordinator
    from .delegation import DelegationEngine
    from .state_machine import TaskState, TaskStateMachine
    from .quality_gates import QualityGate, QualityGateManager
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

# Legacy orchestration systems (backward compatibility)
try:
    from .seamless_coordinator import SeamlessCoordinator
    from .emotional_orchestrator import EmotionalOrchestrator
    from .symphony_mode import SymphonyMode
    from .predictive_spawner import PredictiveSpawner
    from .orchestration_manager import OrchestrationManager
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False

# Export list - prioritize dynamic orchestration components
__all__ = [
    # Core team running functions (backward compatible)
    "run_team",  # Now uses dynamic orchestration v2
    "run_team_legacy",  # Original team runner
    
    # NEW: Strict Orchestrator-Only Communication Architecture
    "Orchestrator",
    "OrchestratorConfig", 
    "OrchestratorMode",
    "OrchestratorResponse",
    
    # Dynamic orchestration components
    "TaskClassifier",
    "TeamComposer", 
    "DynamicOrchestrator",
    "AgileCoachIntegration",
    "RetrospectiveEngine",
    
    # Data models and types
    "TaskClassification",
    "TaskType",
    "ComplexityLevel", 
    "RiskLevel",
    "TechnologyStack",
    "CoordinationStrategy",
    "AgentSpec",
    "ResourceConstraints",
    "TeamComposition",
    "TeamPerformanceMetrics",
    "TaskResult",
    
    # Resource and execution management
    "ResourceManager",
    "ResourceType",
    "ExecutionEngine",
    "TeamExecution", 
    "ExecutionStatus",
    "ExecutionProgress",
]

# Add AGENTS.md v2 components if available
if V2_AVAILABLE:
    __all__.extend([
        "MainCoordinator",
        "DelegationEngine", 
        "TaskState",
        "TaskStateMachine",
        "QualityGate",
        "QualityGateManager",
    ])

# Add legacy components if available
if LEGACY_AVAILABLE:
    __all__.extend([
        'SeamlessCoordinator',
        'EmotionalOrchestrator', 
        'SymphonyMode',
        'PredictiveSpawner',
        'OrchestrationManager'
    ])

# Version and compatibility information
__version__ = "3.0.0"
__api_version__ = "v3"

# Compatibility notes
COMPATIBILITY_NOTES = {
    "run_team": "Enhanced team runner v2 with dynamic orchestration and intelligent task classification (default)",
    "run_team_legacy": "Original team runner API maintained for backward compatibility",
    "dynamic_orchestration": "New v3 API with intelligent task classification and team composition",
    "agile_integration": "Built-in agile coach for planning and retrospectives", 
    "continuous_improvement": "Automated retrospectives and performance optimization"
}