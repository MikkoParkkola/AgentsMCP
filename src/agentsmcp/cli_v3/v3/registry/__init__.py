"""Command registry system for CLI v3.

This package provides centralized command management with intelligent discovery,
validation, and metadata management capabilities.
"""

from .command_registry import CommandRegistry
from .discovery_engine import DiscoveryEngine
from .validator import CommandValidator

__all__ = [
    "CommandRegistry",
    "DiscoveryEngine", 
    "CommandValidator",
]