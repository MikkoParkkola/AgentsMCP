"""AgentsMCP Installation and Setup System.

This module provides a comprehensive one-command installation system that:
- Detects environment automatically (OS, Python, shell, existing tools)
- Auto-configures based on detected environment with intelligent defaults
- Performs comprehensive health checks and validation
- Provides clear recovery guidance for installation failures
- Supports multiple installation scenarios (development, production, containerized)
- Completes setup in under 2 minutes on standard hardware

The main entry point is the `install` function which orchestrates the entire
installation process through the InstallationOrchestrator.
"""

from .installer import InstallationOrchestrator
from .environment_detector import EnvironmentDetector
from .configurator import AutoConfigurator
from .health_checker import HealthChecker
from .recovery import RecoveryGuide

__all__ = [
    'InstallationOrchestrator',
    'EnvironmentDetector', 
    'AutoConfigurator',
    'HealthChecker',
    'RecoveryGuide',
    'install'
]

def install(mode: str = "auto", **kwargs) -> bool:
    """
    One-command installation entry point.
    
    Args:
        mode: Installation mode ("auto", "development", "production", "containerized")
        **kwargs: Additional installation options
        
    Returns:
        True if installation completed successfully, False otherwise
        
    Example:
        >>> from agentsmcp.setup import install
        >>> success = install(mode="auto")
        >>> if success:
        >>>     print("AgentsMCP is ready to use!")
    """
    orchestrator = InstallationOrchestrator(mode=mode, **kwargs)
    return orchestrator.run_installation()