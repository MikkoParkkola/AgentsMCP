"""
AgentsMCP Orchestration Module

AGENTS.md v2 Two-Tier Orchestration System with:
- Single main loop coordination (Tier 1)
- Stateless agent functions (Tier 2)
- Quality gates and validation
- Structured envelope communication

Legacy systems (seamless, emotional, symphony) remain for backward compatibility.
"""

# AGENTS.md v2 Two-Tier Architecture
from .coordinator import MainCoordinator
from .delegation import DelegationEngine
from .state_machine import TaskState, TaskStateMachine
from .quality_gates import QualityGate, QualityGateManager

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

# Event system from parent module (imports handled locally in coordinator to avoid circular imports)

# Export list - prioritize v2 components
__all__ = [
    # AGENTS.md v2 Two-Tier Architecture
    "MainCoordinator",
    "DelegationEngine", 
    "TaskState",
    "TaskStateMachine",
    "QualityGate",
    "QualityGateManager",
    
    # Event system
    "EventBus",
    "Event",
    "JobStarted",
    "JobCompleted", 
    "JobFailed",
    "AgentSpawned",
    "AgentTerminated",
    "ResourceLimitExceeded"
]

# Add legacy components if available
if LEGACY_AVAILABLE:
    __all__.extend([
        'SeamlessCoordinator',
        'EmotionalOrchestrator', 
        'SymphonyMode',
        'PredictiveSpawner',
        'OrchestrationManager'
    ])