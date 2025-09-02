"""
Reliability Package - Critical TUI reliability and timeout protection modules.

This package provides the essential components to prevent TUI hangs and ensure
reliable startup and operation of the Revolutionary TUI Interface.

Key Modules:
- startup_orchestrator: Guarantees TUI startup within 10 seconds with fallback modes
- timeout_guardian: Wraps all async operations with guaranteed cancellation
- component_initializer: Initialize TUI components with timeout protection and parallel execution
- health_monitor: Monitor TUI health and detect hang conditions with recovery actions

The primary goal is to eliminate the current issue where the TUI hangs completely
after "Initializing Revolutionary TUI Interface..." and becomes unresponsive.
"""

from .startup_orchestrator import (
    StartupOrchestrator,
    StartupResult, 
    StartupPhase,
    StartupConfig,
    coordinate_tui_startup,
    create_startup_config
)

from .timeout_guardian import (
    TimeoutGuardian,
    TimeoutState,
    OperationContext,
    get_global_guardian,
    timeout_protection,
    timeout_protected,
    protect_coro
)

from .component_initializer import (
    ComponentInitializer,
    ComponentType,
    ComponentSpec,
    ComponentResult,
    ComponentStatus,
    InitializationMode,
    InitializationMetrics,
    get_global_initializer,
    initialize_tui_components
)

from .health_monitor import (
    HealthMonitor,
    HealthStatus,
    MetricType,
    AlertLevel,
    HealthMetric,
    PerformanceReport,
    HangDetectionConfig,
    get_global_health_monitor,
    start_tui_health_monitoring
)

from .recovery_manager import (
    RecoveryManager,
    RecoveryStrategy,
    RecoveryStatus,
    ComponentFailureType,
    ComponentFailure,
    RecoveryResult,
    RecoveryConfig,
    ComponentRecoveryInterface,
    get_global_recovery_manager,
    setup_tui_recovery_system,
    trigger_component_recovery
)

__all__ = [
    # Startup orchestration
    'StartupOrchestrator',
    'StartupResult',
    'StartupPhase', 
    'StartupConfig',
    'coordinate_tui_startup',
    'create_startup_config',
    
    # Timeout protection
    'TimeoutGuardian',
    'TimeoutState',
    'OperationContext', 
    'get_global_guardian',
    'timeout_protection',
    'timeout_protected',
    'protect_coro',
    
    # Component initialization
    'ComponentInitializer',
    'ComponentType',
    'ComponentSpec',
    'ComponentResult', 
    'ComponentStatus',
    'InitializationMode',
    'InitializationMetrics',
    'get_global_initializer',
    'initialize_tui_components',
    
    # Health monitoring
    'HealthMonitor',
    'HealthStatus',
    'MetricType',
    'AlertLevel',
    'HealthMetric',
    'PerformanceReport',
    'HangDetectionConfig',
    'get_global_health_monitor',
    'start_tui_health_monitoring',
    
    # Recovery management
    'RecoveryManager',
    'RecoveryStrategy',
    'RecoveryStatus',
    'ComponentFailureType',
    'ComponentFailure',
    'RecoveryResult',
    'RecoveryConfig',
    'ComponentRecoveryInterface',
    'get_global_recovery_manager',
    'setup_tui_recovery_system',
    'trigger_component_recovery'
]

# Version info
__version__ = "1.0.0"
__author__ = "reliability_architect_c1"
__description__ = "Critical TUI reliability and timeout protection"