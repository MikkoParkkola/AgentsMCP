"""
AgentsMCP Selection Framework

This package implements a sophisticated A/B testing and continuous evaluation system
for provider/model/agent/tool selection in AgentsMCP. It provides:

- Continuous A/B testing framework for all selection decisions
- Multi-armed bandit algorithms (Thompson Sampling, UCB1, LinUCB)
- Performance tracking and statistical analysis
- Intelligent exploration vs exploitation balancing
- Learning system that improves selections over time

Core Components:
- AdaptiveSelector: Main intelligent selection interface
- ABTestingFramework: A/B test orchestration
- BenchmarkTracker: Performance metrics collection  
- SelectionOptimizer: Multi-armed bandit algorithms
- PerformanceAnalyzer: Statistical analysis and insights
- ExperimentManager: Experiment lifecycle management
- SelectionHistory: Historical data storage and retrieval
"""

from .adaptive_selector import AdaptiveSelector
from .ab_testing_framework import ABTestingFramework, ExperimentConfig, ExperimentResult
from .benchmark_tracker import BenchmarkTracker, SelectionMetrics
from .selection_optimizer import SelectionOptimizer, OptimizationStrategy
from .performance_analyzer import PerformanceAnalyzer, StatisticalTest
from .experiment_manager import ExperimentManager, ExperimentStatus
from .selection_history import SelectionHistory, SelectionRecord

__all__ = [
    "AdaptiveSelector",
    "ABTestingFramework", 
    "ExperimentConfig",
    "ExperimentResult",
    "BenchmarkTracker",
    "SelectionMetrics",
    "SelectionOptimizer",
    "OptimizationStrategy", 
    "PerformanceAnalyzer",
    "StatisticalTest",
    "ExperimentManager",
    "ExperimentStatus",
    "SelectionHistory",
    "SelectionRecord",
]