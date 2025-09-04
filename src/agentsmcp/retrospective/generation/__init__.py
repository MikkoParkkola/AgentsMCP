"""
Improvement Generation Engine

This module generates specific, actionable improvement opportunities
based on analysis results with impact estimation and prioritization.
"""

from .improvement_generator import (
    ImprovementGenerator,
    ImprovementOpportunity,
    ImprovementType,
    ImplementationEffort,
    RiskLevel
)
from .suggestion_templates import SuggestionTemplates
from .impact_estimation import ImpactEstimator
from .improvement_engine import (
    ImprovementEngine,
    ImprovementFilter,
    ImprovementGenerationConfig
)
from .improvement_prioritizer import (
    ImprovementPrioritizer,
    PrioritizationStrategy,
    ResourceConstraints,
    StrategicGoals,
    PrioritizationResult
)
from .improvement_implementer import (
    ImprovementImplementer,
    ImplementationStatus,
    ImplementationResult,
    SafetyCheck
)
from .integration_examples import (
    ImprovementWorkflowExamples,
    demo_basic_improvement_generation,
    demo_quick_wins_filtering,
    demo_advanced_prioritization,
    demo_complete_workflow
)

__all__ = [
    # Core generation
    "ImprovementGenerator",
    "ImprovementOpportunity", 
    "ImprovementType",
    "ImplementationEffort",
    "RiskLevel",
    "SuggestionTemplates",
    "ImpactEstimator",
    
    # Engine and coordination
    "ImprovementEngine",
    "ImprovementFilter",
    "ImprovementGenerationConfig",
    
    # Prioritization
    "ImprovementPrioritizer",
    "PrioritizationStrategy",
    "ResourceConstraints",
    "StrategicGoals",
    "PrioritizationResult",
    
    # Implementation
    "ImprovementImplementer",
    "ImplementationStatus",
    "ImplementationResult",
    "SafetyCheck",
    
    # Integration examples
    "ImprovementWorkflowExamples",
    "demo_basic_improvement_generation",
    "demo_quick_wins_filtering", 
    "demo_advanced_prioritization",
    "demo_complete_workflow"
]