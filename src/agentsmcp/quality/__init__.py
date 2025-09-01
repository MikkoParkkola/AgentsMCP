"""
Quality Assurance System

This module provides comprehensive quality gates, safety checks, and code protection
mechanisms to prevent system breakage during automated code modifications.
"""

from .quality_gate_system import (
    QualityGateSystem,
    QualityGateResult,
    QualityLevel,
    QualityCheck,
    QualityGateReport,
    get_quality_gate_system
)

__all__ = [
    "QualityGateSystem",
    "QualityGateResult", 
    "QualityLevel",
    "QualityCheck",
    "QualityGateReport",
    "get_quality_gate_system"
]