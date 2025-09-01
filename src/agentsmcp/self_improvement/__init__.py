"""AgentsMCP Self-Improvement Module

This module provides comprehensive self-improvement capabilities that continuously
optimize system performance after each user task completion.

Core Components:
- PerformanceAnalyzer: Multi-dimensional performance measurement
- ImprovementDetector: Identify optimization opportunities  
- ImprovementImplementer: Safe application of improvements
- MetricsCollector: Comprehensive telemetry and data collection
- ImprovementHistory: Change tracking and rollback system
- UserFeedbackIntegrator: User satisfaction and feedback analysis
- ContinuousOptimizer: Main orchestration and scheduling

The system implements safety mechanisms including:
- Quality gates for all improvements
- Automatic rollback for performance regressions
- A/B testing for improvement validation
- Staged rollout with monitoring
- User override capabilities
"""

from .performance_analyzer import PerformanceAnalyzer
from .improvement_detector import ImprovementDetector
from .improvement_implementer import ImprovementImplementer
from .metrics_collector import MetricsCollector
from .improvement_history import ImprovementHistory
from .user_feedback_integrator import UserFeedbackIntegrator
from .continuous_optimizer import ContinuousOptimizer

__all__ = [
    'PerformanceAnalyzer',
    'ImprovementDetector', 
    'ImprovementImplementer',
    'MetricsCollector',
    'ImprovementHistory',
    'UserFeedbackIntegrator',
    'ContinuousOptimizer'
]