"""
Retrospective Analysis Engine

This module provides the intelligence core of the self-improvement system,
analyzing execution logs to identify improvement opportunities with high accuracy.
"""

from .retrospective_analyzer import RetrospectiveAnalyzer
from .pattern_detection import PatternDetector
from .bottleneck_identification import BottleneckIdentifier

__all__ = [
    "RetrospectiveAnalyzer",
    "PatternDetector", 
    "BottleneckIdentifier"
]