"""User Intelligence System for AgentsMCP CLI v3.

This module provides adaptive user profiling, learning, and personalized
suggestions to enhance the user experience through intelligent behavior
patterns recognition and progressive disclosure.
"""

from .intelligence_models import (
    UserProfile,
    SkillLevel,
    UserAction,
    SessionContext,
    UserFeedback,
    Suggestion,
    CommandPattern,
    ProgressiveDisclosureLevel,
    LearningMetrics,
)

from .user_profiler import UserProfiler
from .learning_engine import LearningEngine  
from .suggestion_engine import SuggestionEngine

__all__ = [
    # Models
    "UserProfile",
    "SkillLevel", 
    "UserAction",
    "SessionContext",
    "UserFeedback",
    "Suggestion",
    "CommandPattern",
    "ProgressiveDisclosureLevel",
    "LearningMetrics",
    
    # Core components
    "UserProfiler",
    "LearningEngine",
    "SuggestionEngine",
]