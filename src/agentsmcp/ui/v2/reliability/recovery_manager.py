"""
Recovery Manager - Automatic TUI hang recovery without full restart.

This module provides automatic recovery from TUI component hangs and failures.
When the health monitor detects hangs or failures, this system triggers targeted
recovery actions to restore functionality without restarting the entire TUI.

Key Features:
- Automatic hang recovery within 3 seconds
- Component-level restart without full TUI restart
- Multiple recovery strategies based on failure type
- Integration with health_monitor and timeout_guardian
- Recovery performance monitoring and fallback modes

Recovery Strategies:
- RESTART_COMPONENT: Restart the specific failed component
- FALLBACK_MODE: Switch to minimal functionality mode
- EMERGENCY_SHUTDOWN: Clean shutdown if recovery fails

ICD Compliance:
- Inputs: component_hang_event, recovery_strategy, timeout_seconds
- Outputs: RecoveryResult, recovery_success, fallback_activated
- Performance: Recovery completes within 3s or triggers fallback
- Error Handling: Failed recovery triggers escalation to fallback mode
"""

import asyncio
import logging
import time
import traceback
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Union, Type
import sys

from .timeout_guardian import TimeoutGuardian, get_global_guardian, TimeoutState
from .health_monitor import HealthMonitor, HealthStatus, PerformanceReport, get_global_health_monitor

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Available recovery strategies for component failures."""
    RESTART_COMPONENT = "restart_component"      # Restart the specific component
    FALLBACK_MODE = "fallback_mode"             # Switch to minimal functionality
    EMERGENCY_SHUTDOWN = "emergency_shutdown"    # Clean shutdown as last resort


class RecoveryStatus(Enum):
    """Status of a recovery operation."""
    PENDING = "pending"           # Recovery operation in progress
    SUCCESS = "success"           # Recovery completed successfully
    FAILED = "failed"            # Recovery failed
    TIMEOUT = "timeout"          # Recovery timed out
    FALLBACK = "fallback"        # Switched to fallback mode
    ESCALATED = "escalated"      # Escalated to higher recovery level


class ComponentFailureType(Enum):
    """Types of component failures that can trigger recovery."""
    HANG = "hang"                    # Component stopped responding
    CRASH = "crash"                  # Component crashed with exception
    MEMORY_LEAK = "memory_leak"      # Excessive memory usage
    PERFORMANCE_DEGRADED = "perf"    # Severe performance degradation
    INITIALIZATION_FAILED = "init"   # Component failed to initialize
    RENDERING_FAILED = "render"      # Component rendering failures


@dataclass
class ComponentFailure:
    """Information about a component failure."""
    component_name: str
    failure_type: ComponentFailureType
    timestamp: datetime
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    severity: str = "medium"
    recovery_attempts: int = 0
    last_recovery_attempt: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""
    strategy: RecoveryStrategy
    status: RecoveryStatus
    component_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    fallback_activated: bool = False
    recovery_actions_taken: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Whether the recovery was successful."""
        return self.status == RecoveryStatus.SUCCESS
        
    @property
    def failed(self) -> bool:
        """Whether the recovery failed."""
        return self.status in [RecoveryStatus.FAILED, RecoveryStatus.TIMEOUT]


@dataclass
class RecoveryConfig:
    """Configuration for recovery operations."""
    # Timeouts
    component_restart_timeout_s: float = 3.0       # Time limit for component restart
    fallback_activation_timeout_s: float = 1.0     # Time limit for fallback activation
    emergency_shutdown_timeout_s: float = 5.0      # Time limit for emergency shutdown
    
    # Retry limits
    max_recovery_attempts: int = 3                  # Max recovery attempts per component
    recovery_cooldown_s: float = 10.0              # Min time between recovery attempts
    
    # Thresholds
    hang_detection_threshold_s: float = 5.0        # Hang detection threshold
    memory_threshold_mb: float = 500.0             # Memory usage threshold for recovery
    cpu_threshold_percent: float = 80.0            # CPU usage threshold
    
    # Recovery strategies
    default_strategy: RecoveryStrategy = RecoveryStrategy.RESTART_COMPONENT
    fallback_on_multiple_failures: bool = True     # Switch to fallback after multiple failures
    emergency_shutdown_on_critical: bool = True    # Emergency shutdown on critical failures


class ComponentRecoveryInterface:
    """Interface that components can implement for custom recovery."""
    
    async def prepare_for_recovery(self) -> None:
        """Prepare component for recovery (save state, cleanup, etc.)."""
        pass
        
    async def perform_recovery_restart(self) -> bool:
        """Perform component-specific restart. Return True if successful."""
        return False
        
    async def activate_fallback_mode(self) -> bool:
        """Activate fallback mode for this component. Return True if successful."""
        return False
        
    async def cleanup_after_failure(self) -> None:
        """Cleanup after component failure."""
        pass


class RecoveryManager:
    """
    Manages automatic recovery from TUI component hangs and failures.
    
    Integrates with health monitor to detect failures and applies appropriate
    recovery strategies to restore functionality without full TUI restart.
    """
    
    def __init__(
        self,
        config: Optional[RecoveryConfig] = None,
        timeout_guardian: Optional[TimeoutGuardian] = None,
        health_monitor: Optional[HealthMonitor] = None
    ):
        """Initialize the recovery manager."""
        self._config = config or RecoveryConfig()
        self._guardian = timeout_guardian or get_global_guardian()
        self._health_monitor = health_monitor or get_global_health_monitor()
        
        # Recovery state
        self._active_recoveries: Dict[str, RecoveryResult] = {}
        self._component_failures: Dict[str, ComponentFailure] = {}
        self._recovery_history: List[RecoveryResult] = []
        self._fallback_mode_active = False
        self._emergency_mode_active = False
        
        # Component registry and recovery interfaces
        self._component_registry: Dict[str, Any] = {}  # Will be populated by TUI components
        self._recovery_interfaces: Dict[str, ComponentRecoveryInterface] = {}
        
        # Callbacks
        self._recovery_callbacks: Set[Callable[[RecoveryResult], None]] = set()
        self._fallback_callbacks: Set[Callable[[str], None]] = set()
        
        # Statistics
        self._total_recoveries = 0
        self._successful_recoveries = 0
        self._failed_recoveries = 0
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Setup health monitor integration
        self._setup_health_monitor_integration()
        
        logger.info("Recovery manager initialized with 3s recovery timeout")
        
    def _setup_health_monitor_integration(self):
        """Setup integration with health monitor for automatic recovery."""
        if self._health_monitor:
            # Add callbacks for health status and hang detection
            self._health_monitor.add_health_callback(self._handle_health_report)
            self._health_monitor.add_hang_callback(self._handle_hang_detection)
            logger.debug("Recovery manager integrated with health monitor")
            
    async def _handle_health_report(self, report: PerformanceReport) -> None:
        """Handle health report from health monitor."""
        # Check for conditions that require recovery
        if report.recovery_suggested or report.hang_detected:
            # Determine which component(s) need recovery
            failed_components = await self._identify_failed_components(report)
            
            # Trigger recovery for each failed component
            for component_name, failure_type in failed_components.items():
                await self._trigger_component_recovery(component_name, failure_type, report)
                
    async def _handle_hang_detection(self, reason: str) -> None:
        """Handle hang detection event from health monitor."""
        logger.warning(f"Hang detected: {reason}")
        
        # For general hangs, try to recover the most likely problematic components
        # This is a heuristic-based approach when specific component isn't identified
        await self._trigger_general_hang_recovery(reason)
        
    async def _identify_failed_components(self, report: PerformanceReport) -> Dict[str, ComponentFailureType]:
        """Identify which components have failed based on health report."""
        failed_components = {}
        
        # Memory-related failures
        memory_metric = report.metrics.get("memory_usage")
        if memory_metric and memory_metric.value > self._config.memory_threshold_mb:
            # Identify memory-heavy components (simplified heuristic)
            failed_components["display_renderer"] = ComponentFailureType.MEMORY_LEAK
            
        # Performance-related failures
        response_metric = report.metrics.get("response_time")  
        if response_metric and response_metric.value > 1000:  # 1s response time
            failed_components["input_handler"] = ComponentFailureType.PERFORMANCE_DEGRADED
            
        # General hang detection
        if report.hang_detected:
            # Try to recover core components that commonly cause hangs
            failed_components["event_system"] = ComponentFailureType.HANG
            failed_components["display_manager"] = ComponentFailureType.HANG
            
        return failed_components
        
    async def _trigger_component_recovery(
        self,
        component_name: str,
        failure_type: ComponentFailureType,
        context: Optional[PerformanceReport] = None
    ) -> RecoveryResult:
        """Trigger recovery for a specific component."""
        async with self._lock:
            # Check if recovery is already in progress
            if component_name in self._active_recoveries:
                logger.warning(f"Recovery already in progress for {component_name}")
                return self._active_recoveries[component_name]
                
            # Check recovery cooldown
            if not await self._can_attempt_recovery(component_name):
                logger.warning(f"Recovery cooldown active for {component_name}")
                return await self._create_failed_result(
                    component_name, 
                    RecoveryStrategy.RESTART_COMPONENT,
                    "Recovery cooldown active"
                )
                
            # Create failure record
            failure = ComponentFailure(
                component_name=component_name,
                failure_type=failure_type,
                timestamp=datetime.now(),
                error_message=f"Failure detected: {failure_type.value}",
                metadata={'context': str(context) if context else None}
            )
            
            # Update failure tracking
            if component_name in self._component_failures:
                self._component_failures[component_name].recovery_attempts += 1
                self._component_failures[component_name].last_recovery_attempt = datetime.now()
            else:
                self._component_failures[component_name] = failure
                
            # Determine recovery strategy
            strategy = await self._determine_recovery_strategy(component_name, failure)
            
            # Start recovery operation
            recovery_result = RecoveryResult(
                strategy=strategy,
                status=RecoveryStatus.PENDING,
                component_name=component_name,
                start_time=datetime.now()
            )
            
            self._active_recoveries[component_name] = recovery_result
            
        # Execute recovery (outside of lock to prevent blocking)
        try:
            logger.info(f"Starting recovery for {component_name} using strategy {strategy.value}")
            await self._execute_recovery(recovery_result, failure)
            
        except Exception as e:
            logger.error(f"Recovery execution failed for {component_name}: {e}")
            recovery_result.status = RecoveryStatus.FAILED
            recovery_result.error_message = str(e)
            
        finally:
            async with self._lock:
                # Complete recovery
                recovery_result.end_time = datetime.now()
                if recovery_result.start_time and recovery_result.end_time:
                    duration = recovery_result.end_time - recovery_result.start_time
                    recovery_result.duration_ms = duration.total_seconds() * 1000
                    
                # Update statistics
                self._total_recoveries += 1
                if recovery_result.success:
                    self._successful_recoveries += 1
                else:
                    self._failed_recoveries += 1
                    
                # Move to history
                self._recovery_history.append(recovery_result)
                self._active_recoveries.pop(component_name, None)
                
                # Notify callbacks
                await self._notify_recovery_callbacks(recovery_result)
                
        return recovery_result
        
    async def _trigger_general_hang_recovery(self, reason: str) -> List[RecoveryResult]:
        """Trigger recovery for general hang conditions."""
        logger.warning(f"Triggering general hang recovery: {reason}")
        
        # List of core components that commonly cause hangs
        core_components = [
            "display_renderer",
            "input_handler", 
            "event_system",
            "layout_engine"
        ]
        
        # Trigger recovery for core components
        recovery_results = []
        for component_name in core_components:
            try:
                result = await self._trigger_component_recovery(
                    component_name,
                    ComponentFailureType.HANG,
                    None
                )
                recovery_results.append(result)
                
                # If recovery fails quickly, don't try remaining components
                if result.failed and result.duration_ms < 1000:
                    logger.warning("Fast failure detected, switching to fallback mode")
                    break
                    
            except Exception as e:
                logger.error(f"Failed to trigger recovery for {component_name}: {e}")
                
        return recovery_results
        
    async def _can_attempt_recovery(self, component_name: str) -> bool:
        """Check if recovery can be attempted for a component."""
        failure = self._component_failures.get(component_name)
        if not failure:
            return True
            
        # Check max attempts
        if failure.recovery_attempts >= self._config.max_recovery_attempts:
            logger.warning(f"Max recovery attempts exceeded for {component_name}")
            return False
            
        # Check cooldown period
        if failure.last_recovery_attempt:
            cooldown_elapsed = (datetime.now() - failure.last_recovery_attempt).total_seconds()
            if cooldown_elapsed < self._config.recovery_cooldown_s:
                return False
                
        return True
        
    async def _determine_recovery_strategy(
        self,
        component_name: str,
        failure: ComponentFailure
    ) -> RecoveryStrategy:
        """Determine the appropriate recovery strategy for a component failure."""
        
        # If already in emergency mode, only allow shutdown
        if self._emergency_mode_active:
            return RecoveryStrategy.EMERGENCY_SHUTDOWN
            
        # If multiple components are failing, switch to fallback mode
        active_failures = len([f for f in self._component_failures.values() 
                              if (datetime.now() - f.timestamp).total_seconds() < 30])
        
        if active_failures >= 3 and self._config.fallback_on_multiple_failures:
            return RecoveryStrategy.FALLBACK_MODE
            
        # For critical components or repeated failures, use fallback
        critical_components = {"display_renderer", "input_handler", "event_system"}
        if component_name in critical_components and failure.recovery_attempts > 1:
            return RecoveryStrategy.FALLBACK_MODE
            
        # Default to component restart for most cases
        return self._config.default_strategy
        
    async def _execute_recovery(self, result: RecoveryResult, failure: ComponentFailure) -> None:
        """Execute the recovery operation with timeout protection."""
        component_name = result.component_name
        strategy = result.strategy
        
        try:
            # Use timeout guardian to ensure recovery completes within time limit
            timeout = self._get_strategy_timeout(strategy)
            
            async with self._guardian.protect_operation(f"recovery_{component_name}", timeout):
                if strategy == RecoveryStrategy.RESTART_COMPONENT:
                    success = await self._restart_component(component_name, result)
                    
                elif strategy == RecoveryStrategy.FALLBACK_MODE:
                    success = await self._activate_fallback_mode(component_name, result)
                    
                elif strategy == RecoveryStrategy.EMERGENCY_SHUTDOWN:
                    success = await self._emergency_shutdown(component_name, result)
                    
                else:
                    raise ValueError(f"Unknown recovery strategy: {strategy}")
                    
                # Update result status
                if success:
                    result.status = RecoveryStatus.SUCCESS
                    logger.info(f"Recovery successful for {component_name} using {strategy.value}")
                else:
                    result.status = RecoveryStatus.FAILED
                    result.error_message = f"Recovery strategy {strategy.value} returned False"
                    
        except asyncio.TimeoutError:
            result.status = RecoveryStatus.TIMEOUT
            result.error_message = f"Recovery timed out after {timeout}s"
            logger.error(f"Recovery timed out for {component_name}")
            
            # Escalate to fallback mode if restart timed out
            if strategy == RecoveryStrategy.RESTART_COMPONENT:
                logger.warning(f"Escalating {component_name} recovery to fallback mode")
                await self._escalate_recovery(result)
                
        except Exception as e:
            result.status = RecoveryStatus.FAILED
            result.error_message = str(e)
            result.metadata['stack_trace'] = traceback.format_exc()
            logger.error(f"Recovery failed for {component_name}: {e}")
            
    def _get_strategy_timeout(self, strategy: RecoveryStrategy) -> float:
        """Get timeout for a recovery strategy."""
        timeouts = {
            RecoveryStrategy.RESTART_COMPONENT: self._config.component_restart_timeout_s,
            RecoveryStrategy.FALLBACK_MODE: self._config.fallback_activation_timeout_s,
            RecoveryStrategy.EMERGENCY_SHUTDOWN: self._config.emergency_shutdown_timeout_s
        }
        return timeouts.get(strategy, 3.0)
        
    async def _restart_component(self, component_name: str, result: RecoveryResult) -> bool:
        """Restart a specific component."""
        result.recovery_actions_taken.append("Preparing component for restart")
        
        try:
            # Try custom recovery interface first
            recovery_interface = self._recovery_interfaces.get(component_name)
            if recovery_interface:
                result.recovery_actions_taken.append("Using component recovery interface")
                
                # Prepare for recovery
                await recovery_interface.prepare_for_recovery()
                result.recovery_actions_taken.append("Component prepared for recovery")
                
                # Attempt restart
                success = await recovery_interface.perform_recovery_restart()
                if success:
                    result.recovery_actions_taken.append("Component restart successful")
                    return True
                else:
                    result.recovery_actions_taken.append("Component restart failed")
                    
            # Fallback to generic component restart
            result.recovery_actions_taken.append("Using generic component restart")
            component = self._component_registry.get(component_name)
            
            if component and hasattr(component, 'restart'):
                result.recovery_actions_taken.append("Calling component restart method")
                success = await component.restart()
                if success:
                    result.recovery_actions_taken.append("Generic restart successful")
                    return True
                    
            # If no restart method, try stop/start cycle
            if component:
                result.recovery_actions_taken.append("Attempting stop/start cycle")
                
                if hasattr(component, 'stop'):
                    await component.stop()
                    result.recovery_actions_taken.append("Component stopped")
                    
                if hasattr(component, 'start'):
                    await component.start()
                    result.recovery_actions_taken.append("Component started")
                    return True
                    
            # Last resort: log that component needs manual intervention
            result.recovery_actions_taken.append("Component restart not supported - manual intervention needed")
            return False
            
        except Exception as e:
            result.recovery_actions_taken.append(f"Restart failed with error: {str(e)}")
            logger.error(f"Component restart failed for {component_name}: {e}")
            return False
            
    async def _activate_fallback_mode(self, component_name: str, result: RecoveryResult) -> bool:
        """Activate fallback mode for the TUI."""
        result.recovery_actions_taken.append("Activating fallback mode")
        
        try:
            # Try component-specific fallback first
            recovery_interface = self._recovery_interfaces.get(component_name)
            if recovery_interface:
                result.recovery_actions_taken.append("Using component fallback mode")
                success = await recovery_interface.activate_fallback_mode()
                if success:
                    result.recovery_actions_taken.append("Component fallback mode activated")
                    result.fallback_activated = True
                    return True
                    
            # Activate global fallback mode
            result.recovery_actions_taken.append("Activating global fallback mode")
            self._fallback_mode_active = True
            result.fallback_activated = True
            
            # Notify fallback callbacks
            for callback in self._fallback_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(f"Fallback activated for {component_name}")
                    else:
                        callback(f"Fallback activated for {component_name}")
                except Exception as e:
                    logger.error(f"Fallback callback failed: {e}")
                    
            result.recovery_actions_taken.append("Global fallback mode activated")
            logger.warning("TUI switched to fallback mode")
            return True
            
        except Exception as e:
            result.recovery_actions_taken.append(f"Fallback activation failed: {str(e)}")
            logger.error(f"Failed to activate fallback mode for {component_name}: {e}")
            return False
            
    async def _emergency_shutdown(self, component_name: str, result: RecoveryResult) -> bool:
        """Perform emergency shutdown."""
        result.recovery_actions_taken.append("Initiating emergency shutdown")
        logger.critical(f"Emergency shutdown initiated for {component_name}")
        
        try:
            self._emergency_mode_active = True
            
            # Stop all active recoveries
            result.recovery_actions_taken.append("Stopping all active recoveries")
            for active_component in list(self._active_recoveries.keys()):
                if active_component != component_name:
                    self._active_recoveries[active_component].status = RecoveryStatus.ESCALATED
                    
            # Attempt graceful shutdown of components
            result.recovery_actions_taken.append("Attempting graceful component shutdown")
            
            for comp_name, component in self._component_registry.items():
                try:
                    if hasattr(component, 'emergency_shutdown'):
                        await component.emergency_shutdown()
                    elif hasattr(component, 'stop'):
                        await asyncio.wait_for(component.stop(), timeout=1.0)
                except Exception as e:
                    logger.warning(f"Failed to shutdown component {comp_name}: {e}")
                    
            result.recovery_actions_taken.append("Emergency shutdown completed")
            return True
            
        except Exception as e:
            result.recovery_actions_taken.append(f"Emergency shutdown failed: {str(e)}")
            logger.error(f"Emergency shutdown failed: {e}")
            return False
            
    async def _escalate_recovery(self, result: RecoveryResult) -> None:
        """Escalate recovery to a higher level strategy."""
        component_name = result.component_name
        
        if result.strategy == RecoveryStrategy.RESTART_COMPONENT:
            # Escalate to fallback mode
            logger.warning(f"Escalating {component_name} recovery to fallback mode")
            result.status = RecoveryStatus.ESCALATED
            result.recovery_actions_taken.append("Escalated to fallback mode")
            
            # Trigger fallback mode recovery
            fallback_result = RecoveryResult(
                strategy=RecoveryStrategy.FALLBACK_MODE,
                status=RecoveryStatus.PENDING,
                component_name=component_name,
                start_time=datetime.now()
            )
            
            try:
                await self._activate_fallback_mode(component_name, fallback_result)
                result.fallback_activated = fallback_result.fallback_activated
                result.recovery_actions_taken.extend(fallback_result.recovery_actions_taken)
            except Exception as e:
                logger.error(f"Escalated recovery failed: {e}")
                
    async def _create_failed_result(
        self,
        component_name: str,
        strategy: RecoveryStrategy,
        error_message: str
    ) -> RecoveryResult:
        """Create a failed recovery result."""
        return RecoveryResult(
            strategy=strategy,
            status=RecoveryStatus.FAILED,
            component_name=component_name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_ms=0.0,
            error_message=error_message
        )
        
    async def _notify_recovery_callbacks(self, result: RecoveryResult) -> None:
        """Notify recovery completion callbacks."""
        for callback in self._recovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Recovery callback failed: {e}")
                
    # Public API methods
    
    def register_component(self, name: str, component: Any) -> None:
        """Register a component for recovery management."""
        self._component_registry[name] = component
        logger.debug(f"Registered component for recovery: {name}")
        
    def register_recovery_interface(self, name: str, interface: ComponentRecoveryInterface) -> None:
        """Register a custom recovery interface for a component."""
        self._recovery_interfaces[name] = interface
        logger.debug(f"Registered recovery interface: {name}")
        
    def add_recovery_callback(self, callback: Callable[[RecoveryResult], None]) -> None:
        """Add callback for recovery completion events."""
        self._recovery_callbacks.add(callback)
        
    def remove_recovery_callback(self, callback: Callable[[RecoveryResult], None]) -> None:
        """Remove recovery completion callback."""
        self._recovery_callbacks.discard(callback)
        
    def add_fallback_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for fallback mode activation."""
        self._fallback_callbacks.add(callback)
        
    def remove_fallback_callback(self, callback: Callable[[str], None]) -> None:
        """Remove fallback mode callback."""
        self._fallback_callbacks.discard(callback)
        
    async def manual_recovery(
        self,
        component_name: str,
        strategy: Optional[RecoveryStrategy] = None,
        reason: str = "Manual recovery request"
    ) -> RecoveryResult:
        """Manually trigger recovery for a component."""
        logger.info(f"Manual recovery requested for {component_name}: {reason}")
        
        failure_type = ComponentFailureType.HANG  # Default for manual recovery
        return await self._trigger_component_recovery(component_name, failure_type)
        
    async def get_recovery_status(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current recovery status."""
        async with self._lock:
            if component_name:
                # Status for specific component
                active_recovery = self._active_recoveries.get(component_name)
                failure_info = self._component_failures.get(component_name)
                
                return {
                    'component_name': component_name,
                    'active_recovery': {
                        'status': active_recovery.status.value if active_recovery else None,
                        'strategy': active_recovery.strategy.value if active_recovery else None,
                        'start_time': active_recovery.start_time.isoformat() if active_recovery else None
                    } if active_recovery else None,
                    'failure_history': {
                        'total_attempts': failure_info.recovery_attempts if failure_info else 0,
                        'last_attempt': failure_info.last_recovery_attempt.isoformat() if failure_info and failure_info.last_recovery_attempt else None,
                        'failure_type': failure_info.failure_type.value if failure_info else None
                    } if failure_info else None
                }
            else:
                # Overall recovery status
                return {
                    'active_recoveries': len(self._active_recoveries),
                    'total_recoveries': self._total_recoveries,
                    'successful_recoveries': self._successful_recoveries,
                    'failed_recoveries': self._failed_recoveries,
                    'success_rate': (self._successful_recoveries / max(1, self._total_recoveries)) * 100,
                    'fallback_mode_active': self._fallback_mode_active,
                    'emergency_mode_active': self._emergency_mode_active,
                    'components_with_failures': len(self._component_failures),
                    'active_recovery_components': list(self._active_recoveries.keys())
                }
                
    async def get_recovery_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recovery operation history."""
        async with self._lock:
            # Get most recent recovery results
            recent_history = self._recovery_history[-limit:] if limit > 0 else self._recovery_history
            
            return [
                {
                    'component_name': result.component_name,
                    'strategy': result.strategy.value,
                    'status': result.status.value,
                    'start_time': result.start_time.isoformat(),
                    'duration_ms': result.duration_ms,
                    'success': result.success,
                    'fallback_activated': result.fallback_activated,
                    'error_message': result.error_message,
                    'actions_taken': result.recovery_actions_taken
                }
                for result in recent_history
            ]
            
    def is_fallback_mode_active(self) -> bool:
        """Check if fallback mode is currently active."""
        return self._fallback_mode_active
        
    def is_emergency_mode_active(self) -> bool:
        """Check if emergency mode is currently active."""
        return self._emergency_mode_active
        
    async def reset_component_failure_history(self, component_name: str) -> None:
        """Reset failure history for a component."""
        async with self._lock:
            self._component_failures.pop(component_name, None)
            logger.info(f"Reset failure history for {component_name}")
            
    async def reset_fallback_mode(self) -> bool:
        """Reset fallback mode (attempt to return to normal operation)."""
        if not self._fallback_mode_active:
            return True
            
        logger.info("Attempting to reset fallback mode")
        
        try:
            # Only reset if no active recoveries
            if self._active_recoveries:
                logger.warning("Cannot reset fallback mode - active recoveries in progress")
                return False
                
            # Reset fallback mode
            self._fallback_mode_active = False
            logger.info("Fallback mode reset - returning to normal operation")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset fallback mode: {e}")
            return False


# Global recovery manager instance
_global_recovery_manager: Optional[RecoveryManager] = None


def get_global_recovery_manager() -> RecoveryManager:
    """Get the global recovery manager instance."""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = RecoveryManager()
    return _global_recovery_manager


async def setup_tui_recovery_system(
    config: Optional[RecoveryConfig] = None,
    component_registry: Optional[Dict[str, Any]] = None
) -> RecoveryManager:
    """
    Setup the TUI recovery system with automatic hang recovery.
    
    Args:
        config: Recovery configuration
        component_registry: Dictionary of components to manage
        
    Returns:
        RecoveryManager instance
    """
    recovery_manager = get_global_recovery_manager()
    
    # Update configuration if provided
    if config:
        recovery_manager._config = config
        
    # Register components if provided
    if component_registry:
        for name, component in component_registry.items():
            recovery_manager.register_component(name, component)
            
    logger.info("TUI recovery system setup complete")
    return recovery_manager


async def trigger_component_recovery(
    component_name: str,
    failure_type: ComponentFailureType = ComponentFailureType.HANG,
    reason: str = "Manual trigger"
) -> RecoveryResult:
    """
    Convenience function to trigger component recovery.
    
    Args:
        component_name: Name of component to recover
        failure_type: Type of failure detected
        reason: Reason for recovery
        
    Returns:
        RecoveryResult with outcome
    """
    recovery_manager = get_global_recovery_manager()
    return await recovery_manager.manual_recovery(component_name, None, reason)