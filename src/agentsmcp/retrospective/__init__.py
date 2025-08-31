"""Enhanced retrospective system for AgentsMCP.

This package provides comprehensive retrospective capabilities including:
- Individual agent retrospectives with self-assessment
- Agile coach comprehensive analysis and pattern recognition
- Orchestrator enforcement system for action point implementation
- Seamless integration with existing agent lifecycle

Main Components:
- IndividualRetrospectiveFramework: Individual agent retrospectives
- AgileCoachAnalyzer: Comprehensive multi-agent analysis
- OrchestratorEnforcementSystem: Action point enforcement and validation
- EnhancedRetrospectiveIntegration: Integration layer for existing systems
"""

from .individual_framework import IndividualRetrospectiveFramework
from .coach_analyzer import AgileCoachAnalyzer
from .enforcement import OrchestratorEnforcementSystem
from .integration_layer import EnhancedRetrospectiveIntegration
from .self_assessment import AgentSelfAssessmentSystem

from .data_models import (
    # Core retrospective models
    IndividualRetrospective,
    ComprehensiveRetrospectiveReport,
    
    # Configuration models
    IndividualRetrospectiveConfig,
    OrchestratorConfig,
    
    # Analysis models
    PatternAnalysis,
    SystemicIssue,
    CrossAgentInsights,
    
    # Action and enforcement models
    ActionPoint,
    EnforcementPlan,
    ReadinessAssessment,
    
    # Assessment models
    PerformanceAssessment,
    DecisionPoint,
    Challenge,
    SelfImprovementAction,
    
    # Enums
    RetrospectiveType,
    ImprovementCategory,
    PriorityLevel,
    ImplementationStatus,
)

__all__ = [
    # Main framework classes
    "IndividualRetrospectiveFramework",
    "AgileCoachAnalyzer", 
    "OrchestratorEnforcementSystem",
    "EnhancedRetrospectiveIntegration",
    "AgentSelfAssessmentSystem",
    
    # Data models
    "IndividualRetrospective",
    "ComprehensiveRetrospectiveReport",
    "IndividualRetrospectiveConfig",
    "OrchestratorConfig",
    "PatternAnalysis",
    "SystemicIssue",
    "CrossAgentInsights",
    "ActionPoint",
    "EnforcementPlan", 
    "ReadinessAssessment",
    "PerformanceAssessment",
    "DecisionPoint",
    "Challenge",
    "SelfImprovementAction",
    
    # Enums
    "RetrospectiveType",
    "ImprovementCategory", 
    "PriorityLevel",
    "ImplementationStatus",
]

# Package metadata
__version__ = "1.0.0"
__author__ = "AgentsMCP Development Team"
__description__ = "Enhanced retrospective system with individual agent analysis, comprehensive coaching, and enforcement"