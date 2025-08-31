"""Cross-modal coordination system for CLI v3.

This package provides functionality for synchronizing state and capabilities
across different interface modes (CLI, TUI, WebUI, API).
"""

from .modal_coordinator import ModalCoordinator
from .state_synchronizer import StateSynchronizer
from .capability_manager import CapabilityManager

__all__ = [
    'ModalCoordinator',
    'StateSynchronizer', 
    'CapabilityManager'
]