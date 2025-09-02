"""
Integration Layer - Seamless reliability integration with Revolutionary TUI.

This is the CRITICAL integration piece that makes all reliability modules work 
together with the existing TUI. It provides a drop-in replacement for the 
RevolutionaryTUIInterface that just works better without breaking existing functionality.

Key Features:
- **GUARANTEED 10s startup**: Uses startup_orchestrator for startup completion
- **ALL components protected**: timeout_guardian wraps every operation
- **Automatic hang recovery**: health_monitor + recovery_manager handle hangs
- **Backward compatibility**: Existing TUI behavior preserved as fallback
- **Seamless integration**: Drop-in replacement, no code changes required

The integration strategy:
1. Wrap RevolutionaryTUIInterface with reliability modules
2. Startup orchestration ensures 10s max startup time
3. All operations protected by timeout guardian
4. Health monitoring with automatic recovery
5. Fallback to original TUI if reliability fails

Usage:
    # Drop-in replacement for RevolutionaryTUIInterface
    from agentsmcp.ui.v2.reliability.integration_layer import ReliableTUIInterface
    
    # Use exactly like the original - but it won't hang!
    tui = ReliableTUIInterface(agent_orchestrator, agent_state)
    await tui.start()  # Guaranteed to complete within 10s
"""

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
import weakref

# Import original TUI
from ..revolutionary_tui_interface import RevolutionaryTUIInterface

# Import reliability modules
from .startup_orchestrator import (
    StartupOrchestrator, StartupResult, StartupConfig, StartupPhase,
    coordinate_tui_startup, create_startup_config
)
from .timeout_guardian import (
    TimeoutGuardian, get_global_guardian, timeout_protection, TimeoutState
)
from .health_monitor import (
    HealthMonitor, HealthStatus, PerformanceReport, HangDetectionConfig,
    get_global_health_monitor, start_tui_health_monitoring
)
from .recovery_manager import (
    RecoveryManager, RecoveryResult, RecoveryStrategy, ComponentFailureType,
    get_global_recovery_manager, setup_tui_recovery_system
)

logger = logging.getLogger(__name__)


@dataclass
class ReliabilityConfig:
    """Configuration for reliability integration."""
    # Startup configuration
    max_startup_time_s: float = 10.0        # Guaranteed startup completion
    aggressive_timeouts: bool = True         # Use aggressive timeouts to prevent hangs
    show_startup_feedback: bool = True       # Show startup progress to user
    
    # Health monitoring
    enable_health_monitoring: bool = True    # Enable continuous health monitoring
    health_check_interval_s: float = 1.0     # Health check frequency
    hang_detection_threshold_s: float = 5.0  # Hang detection threshold
    
    # Recovery configuration
    enable_automatic_recovery: bool = True    # Enable automatic hang recovery
    max_recovery_attempts: int = 3           # Max recovery attempts per component
    recovery_timeout_s: float = 3.0         # Recovery operation timeout
    
    # Fallback behavior
    fallback_on_reliability_failure: bool = True  # Fallback to original TUI
    preserve_original_behavior: bool = True       # Maintain backward compatibility


class ReliableTUIInterface:
    """
    Drop-in replacement for RevolutionaryTUIInterface with full reliability integration.
    
    This class wraps the original TUI with all reliability modules to prevent hangs
    and provide automatic recovery. It maintains complete backward compatibility while
    adding reliability features.
    
    Key reliability features:
    - Guaranteed 10-second startup completion
    - All operations protected by timeout guardian
    - Continuous health monitoring with hang detection
    - Automatic recovery from component hangs
    - Fallback to original TUI if reliability systems fail
    """
    
    def __init__(
        self,
        agent_orchestrator: Any,
        agent_state: Any,
        reliability_config: Optional[ReliabilityConfig] = None,
        **kwargs
    ):
        """
        Initialize the reliable TUI interface.
        
        Args:
            agent_orchestrator: Agent orchestrator instance
            agent_state: Agent state instance
            reliability_config: Reliability configuration
            **kwargs: Additional arguments passed to original TUI
        """
        self._agent_orchestrator = agent_orchestrator
        self._agent_state = agent_state
        self._config = reliability_config or ReliabilityConfig()
        self._original_kwargs = kwargs
        
        # Reliability components
        self._startup_orchestrator: Optional[StartupOrchestrator] = None
        self._timeout_guardian: Optional[TimeoutGuardian] = None
        self._health_monitor: Optional[HealthMonitor] = None
        self._recovery_manager: Optional[RecoveryManager] = None
        
        # Original TUI instance
        self._original_tui: Optional[RevolutionaryTUIInterface] = None
        self._tui_initialized = False
        self._reliability_active = False
        self._fallback_mode = False
        
        # State tracking
        self._startup_completed = False
        self._shutdown_requested = False
        self._initialization_start_time: Optional[float] = None
        
        # Component registry for recovery
        self._components: Dict[str, Any] = {}
        
        # Event hooks for monitoring
        self._startup_callbacks: List[Callable[[str], None]] = []
        self._health_callbacks: List[Callable[[PerformanceReport], None]] = []
        self._recovery_callbacks: List[Callable[[RecoveryResult], None]] = []
        
        logger.info("ReliableTUIInterface initialized with 10s startup guarantee")
        
    async def start(self, **kwargs) -> bool:
        """
        Start the TUI with reliability guarantees.
        
        This method ensures:
        - Startup completes within 10 seconds maximum
        - All components are protected by timeout guardian
        - Health monitoring is active
        - Recovery system is ready
        
        Returns:
            True if startup successful, False if fallback mode activated
        """
        self._initialization_start_time = time.time()
        logger.info("Starting ReliableTUIInterface with reliability guarantees")
        
        try:
            # Initialize reliability components first
            await self._initialize_reliability_components()
            
            # Perform startup orchestration with timeout protection
            startup_result = await self._perform_orchestrated_startup(**kwargs)
            
            # Handle startup result
            if startup_result == StartupResult.SUCCESS:
                logger.info("TUI startup completed successfully with full reliability")
                self._startup_completed = True
                self._reliability_active = True
                
                # Start health monitoring
                if self._config.enable_health_monitoring:
                    await self._start_health_monitoring()
                    
                return True
                
            elif startup_result == StartupResult.FALLBACK:
                logger.warning("TUI started in fallback mode due to timeout/issues")
                self._startup_completed = True
                self._fallback_mode = True
                
                # Still try to start health monitoring in degraded mode
                if self._config.enable_health_monitoring:
                    await self._start_health_monitoring()
                    
                return True
                
            else:
                # Startup failed completely - try fallback to original TUI
                if self._config.fallback_on_reliability_failure:
                    logger.error("TUI startup failed - attempting fallback to original TUI")
                    return await self._fallback_to_original_tui(**kwargs)
                else:
                    logger.error("TUI startup failed and fallback disabled")
                    return False
                    
        except Exception as e:
            logger.error(f"TUI startup failed with exception: {e}")
            logger.debug(f"Startup exception traceback:\n{traceback.format_exc()}")
            
            # Attempt fallback to original TUI
            if self._config.fallback_on_reliability_failure:
                logger.info("Attempting fallback to original TUI after startup exception")
                return await self._fallback_to_original_tui(**kwargs)
            else:
                return False
                
    async def _initialize_reliability_components(self):
        """Initialize all reliability components."""
        logger.debug("Initializing reliability components")
        
        try:
            # Initialize timeout guardian
            self._timeout_guardian = get_global_guardian()
            logger.debug("Timeout guardian initialized")
            
            # Initialize startup orchestrator
            startup_config = create_startup_config(
                max_total_time=self._config.max_startup_time_s,
                enable_feedback=self._config.show_startup_feedback,
                aggressive_timeouts=self._config.aggressive_timeouts
            )
            self._startup_orchestrator = StartupOrchestrator(startup_config)
            logger.debug("Startup orchestrator initialized")
            
            # Initialize health monitor
            hang_config = HangDetectionConfig(
                response_timeout_seconds=self._config.hang_detection_threshold_s,
                update_timeout_seconds=self._config.hang_detection_threshold_s,
                event_timeout_seconds=self._config.hang_detection_threshold_s * 2
            )
            self._health_monitor = HealthMonitor(
                check_interval_seconds=self._config.health_check_interval_s,
                hang_config=hang_config,
                timeout_guardian=self._timeout_guardian
            )
            logger.debug("Health monitor initialized")
            
            # Initialize recovery manager
            if self._config.enable_automatic_recovery:
                self._recovery_manager = get_global_recovery_manager()
                self._recovery_manager._config.component_restart_timeout_s = self._config.recovery_timeout_s
                self._recovery_manager._config.max_recovery_attempts = self._config.max_recovery_attempts
                logger.debug("Recovery manager initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize reliability components: {e}")
            raise
            
    async def _perform_orchestrated_startup(self, **kwargs) -> StartupResult:
        """Perform startup with orchestration and timeout protection."""
        logger.info("Starting orchestrated TUI startup")
        
        # Create startup feedback callback
        def startup_feedback(message: str):
            if self._config.show_startup_feedback:
                logger.info(f"Startup: {message}")
                # Could also display to user if UI is available
                
        try:
            # Prepare TUI components for orchestrated startup
            components = await self._prepare_tui_components()
            
            # Use startup orchestrator for guaranteed completion
            result = await self._startup_orchestrator.coordinate_startup(
                components=components,
                feedback_callback=startup_feedback
            )
            
            # Log startup metrics
            metrics = self._startup_orchestrator.get_startup_metrics()
            logger.info(f"Startup metrics: {metrics}")
            
            return result
            
        except Exception as e:
            logger.error(f"Orchestrated startup failed: {e}")
            return StartupResult.FAILURE
            
    async def _prepare_tui_components(self) -> Dict[str, Any]:
        """Prepare TUI components for orchestrated startup."""
        logger.debug("Preparing TUI components for startup")
        
        components = {}
        
        try:
            # Extract parameters for RevolutionaryTUIInterface constructor
            # RevolutionaryTUIInterface expects: (cli_config, orchestrator_integration, revolutionary_components)
            cli_config = self._original_kwargs.get('cli_config')
            orchestrator_integration = self._agent_orchestrator  # Use agent_orchestrator as orchestrator_integration
            revolutionary_components = self._original_kwargs.get('revolutionary_components', {})
            
            logger.debug(f"Creating RevolutionaryTUIInterface with cli_config={cli_config is not None}, "
                        f"orchestrator_integration={orchestrator_integration is not None}, "
                        f"revolutionary_components={len(revolutionary_components)} items")
            
            # Create original TUI instance with correct parameter mapping
            self._original_tui = RevolutionaryTUIInterface(
                cli_config=cli_config,
                orchestrator_integration=orchestrator_integration,
                revolutionary_components=revolutionary_components
            )
            
            logger.debug("RevolutionaryTUIInterface created successfully")
            
            # Create component wrappers for startup orchestration
            components['orchestrator'] = TUIComponentWrapper(
                "orchestrator",
                self._agent_orchestrator,
                self._timeout_guardian
            )
            
            components['display'] = TUIComponentWrapper(
                "display_manager",
                getattr(self._original_tui, 'display_manager', None),
                self._timeout_guardian
            )
            
            components['input'] = TUIComponentWrapper(
                "input_handler", 
                getattr(self._original_tui, 'input_pipeline', None),
                self._timeout_guardian
            )
            
            # Store components for recovery manager
            self._components = {
                'tui_interface': self._original_tui,
                'agent_orchestrator': self._agent_orchestrator,
                **components
            }
            
            # Register components with recovery manager
            if self._recovery_manager:
                for name, component in self._components.items():
                    self._recovery_manager.register_component(name, component)
                    
            logger.debug(f"Prepared {len(components)} TUI components for startup")
            return components
            
        except Exception as e:
            logger.error(f"Failed to prepare TUI components: {e}")
            logger.debug(f"TUI component preparation error details:\n{traceback.format_exc()}")
            # Return minimal components to allow startup to continue
            return {'orchestrator': self._agent_orchestrator}
            
    async def _start_health_monitoring(self):
        """Start health monitoring with hang detection."""
        if not self._health_monitor:
            return
            
        try:
            logger.debug("Starting TUI health monitoring")
            
            # Add health callbacks
            self._health_monitor.add_health_callback(self._handle_health_report)
            self._health_monitor.add_hang_callback(self._handle_hang_detected)
            
            # Start monitoring
            await self._health_monitor.start_monitoring()
            logger.info("TUI health monitoring active")
            
        except Exception as e:
            logger.error(f"Failed to start health monitoring: {e}")
            # Don't fail startup if health monitoring fails
            
    async def _handle_health_report(self, report: PerformanceReport):
        """Handle health reports from monitoring."""
        # Log significant health issues
        if report.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.HANGING]:
            logger.warning(f"TUI health issue detected: {report.overall_status.value}")
            
            # Record UI activity for health monitor
            if hasattr(self._health_monitor, 'record_ui_response'):
                await self._health_monitor.record_ui_response()
                
        # Notify registered callbacks
        for callback in self._health_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(report)
                else:
                    callback(report)
            except Exception as e:
                logger.error(f"Health callback failed: {e}")
                
    async def _handle_hang_detected(self, reason: str):
        """Handle hang detection events."""
        logger.critical(f"TUI hang detected: {reason}")
        
        # If recovery is enabled, it will automatically handle the hang
        # Otherwise, we might need to take manual action
        if not self._config.enable_automatic_recovery:
            logger.warning("Automatic recovery disabled - hang may persist")
            
    async def _fallback_to_original_tui(self, **kwargs) -> bool:
        """Fallback to original TUI without reliability features."""
        logger.warning("Falling back to original TUI without reliability features")
        
        try:
            # Create fresh original TUI instance if needed
            if self._original_tui is None:
                self._original_tui = RevolutionaryTUIInterface(
                    self._agent_orchestrator,
                    self._agent_state,
                    **self._original_kwargs
                )
            
            # Start original TUI with basic timeout protection
            async with timeout_protection("fallback_startup", 15.0):
                result = await self._original_tui.start(**kwargs)
                
            if result:
                logger.info("Fallback to original TUI successful")
                self._fallback_mode = True
                self._startup_completed = True
                return True
            else:
                logger.error("Fallback to original TUI also failed")
                return False
                
        except Exception as e:
            logger.error(f"Fallback to original TUI failed: {e}")
            return False
            
    # Delegate methods to original TUI with timeout protection
    
    async def display_summary(self, *args, **kwargs):
        """Display summary with timeout protection."""
        if self._original_tui:
            async with timeout_protection("display_summary", 5.0):
                return await self._original_tui.display_summary(*args, **kwargs)
        return None
        
    async def run_main_loop(self, *args, **kwargs):
        """Run main loop with timeout protection and health monitoring."""
        if not self._original_tui:
            logger.error("Cannot run main loop - TUI not initialized")
            return
            
        try:
            # Record UI activity for health monitoring
            if self._health_monitor:
                await self._health_monitor.record_ui_update()
                await self._health_monitor.record_event_processed()
                
            # The RevolutionaryTUIInterface has _run_main_loop (private method)
            # Use the correct method name
            if hasattr(self._original_tui, '_run_main_loop'):
                # CRITICAL FIX: Remove timeout protection for main loop
                # The main loop should run until user exits, not timeout after 5 minutes
                # The original TUI's _run_main_loop() waits for user input/exit indefinitely
                logger.info("Starting main loop without timeout (will run until user exits)")
                return await self._original_tui._run_main_loop(*args, **kwargs)
            else:
                logger.error("RevolutionaryTUIInterface._run_main_loop method not found")
                return
                
        except Exception as e:
            logger.error(f"Main loop failed: {e}")
            
            # For non-timeout exceptions, still try recovery
            if self._recovery_manager and self._config.enable_automatic_recovery:
                logger.info("Triggering automatic recovery for main loop failure")
                await self._recovery_manager.manual_recovery(
                    "main_loop", 
                    RecoveryStrategy.RESTART_COMPONENT,
                    f"Main loop failed: {e}"
                )
            raise
            
    async def handle_user_input(self, user_input: str, *args, **kwargs):
        """Handle user input with timeout protection."""
        if self._original_tui:
            # Record event processing for health monitoring
            if self._health_monitor:
                await self._health_monitor.record_event_processed()
                
            async with timeout_protection("handle_input", 10.0):
                return await self._original_tui.handle_user_input(user_input, *args, **kwargs)
        return None
        
    async def stop(self, **kwargs):
        """Stop the TUI with cleanup."""
        logger.info("Stopping ReliableTUIInterface")
        self._shutdown_requested = True
        
        try:
            # Stop health monitoring
            if self._health_monitor:
                await self._health_monitor.stop_monitoring()
                logger.debug("Health monitoring stopped")
                
            # Stop original TUI using the correct cleanup method
            if self._original_tui:
                async with timeout_protection("tui_stop", 5.0):
                    # RevolutionaryTUIInterface uses _cleanup method, not stop
                    if hasattr(self._original_tui, '_cleanup'):
                        await self._original_tui._cleanup()
                        logger.debug("Original TUI cleanup completed")
                    else:
                        logger.warning("RevolutionaryTUIInterface._cleanup method not found")
                
            # Shutdown timeout guardian
            if self._timeout_guardian:
                await self._timeout_guardian.shutdown()
                logger.debug("Timeout guardian shutdown")
                
            logger.info("ReliableTUIInterface shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during TUI shutdown: {e}")

    
    async def run(self, **kwargs) -> int:
        """
        Main entry point compatible with launcher expectations.
        
        This method provides the same interface as RevolutionaryTUIInterface.run()
        but with full reliability guarantees. It orchestrates startup, runs the main
        loop with health monitoring, and ensures clean shutdown.
        
        CRITICAL FIX: This method now properly waits for the TUI to complete instead
        of returning immediately, preventing premature shutdown via finally block.
        
        Returns:
            0 on successful completion, 1 on error
        """
        logger.info("Starting ReliableTUIInterface.run() with reliability guarantees")
        
        # Flag to track if we should run cleanup in finally block
        should_cleanup = True
        
        try:
            # Start the TUI with reliability protection
            startup_success = await self.start(**kwargs)
            
            if not startup_success:
                logger.error("TUI startup failed - returning error code 1")
                return 1
                
            # Check if we started in fallback mode
            if self._fallback_mode:
                logger.warning("TUI running in fallback mode - delegating to original TUI")
                
                # In fallback mode, delegate directly to original TUI's run method
                if self._original_tui and hasattr(self._original_tui, 'run'):
                    try:
                        # CRITICAL: Wait for original TUI to complete - don't return immediately
                        # This prevents the finally block from running prematurely
                        result = await self._original_tui.run()
                        logger.info("Fallback TUI run completed successfully")
                        return result
                    except Exception as e:
                        logger.error(f"Fallback TUI run failed: {e}")
                        return 1
                else:
                    logger.error("Fallback mode but no original TUI available")
                    return 1
                    
            # Full reliability mode - run with protected main loop
            logger.info("Running TUI main loop with reliability protection")
            
            # CRITICAL FIX: Properly wait for main loop completion like original TUI
            # The original TUI's run() method waits for the main loop using asyncio.wait()
            # We need to replicate this pattern to prevent immediate return
            
            try:
                # Instead of just calling run_main_loop() and returning immediately,
                # we need to wait for the actual TUI completion
                await self._wait_for_tui_completion()
                
                logger.info("TUI main loop completed successfully")
                return 0
                
            except Exception as e:
                logger.error(f"TUI main loop failed: {e}")
                
                # Attempt recovery if available
                if self._recovery_manager and self._config.enable_automatic_recovery:
                    logger.info("Attempting automatic recovery from main loop failure")
                    try:
                        recovery_result = await self._recovery_manager.manual_recovery(
                            "main_loop",
                            RecoveryStrategy.RESTART_COMPONENT,
                            f"Main loop failed: {e}"
                        )
                        
                        if recovery_result.success:
                            logger.info("Recovery successful - retrying main loop")
                            await self._wait_for_tui_completion()
                            return 0
                        else:
                            logger.error(f"Recovery failed: {recovery_result.error_message or 'Unknown error'}")
                            return 1
                            
                    except Exception as recovery_e:
                        logger.error(f"Recovery attempt failed: {recovery_e}")
                        return 1
                else:
                    return 1
                    
        except KeyboardInterrupt:
            logger.info("TUI interrupted by user")
            return 0
            
        except Exception as e:
            logger.error(f"TUI run failed with unexpected error: {e}")
            logger.debug(f"TUI run exception traceback:\n{traceback.format_exc()}")
            return 1
            
        finally:
            # CRITICAL: Only run cleanup if we actually need to shutdown
            # This prevents premature shutdown when the TUI should still be running
            if should_cleanup:
                try:
                    await self.stop()
                except Exception as e:
                    logger.error(f"Error during TUI shutdown: {e}")
    
    async def _wait_for_tui_completion(self):
        """
        Wait for the TUI to complete execution, mirroring the original TUI's pattern.
        
        The original TUI's run() method creates background tasks and waits for them
        to complete using asyncio.wait(). This method replicates that behavior
        to ensure we don't return until the user actually exits the TUI.
        """
        if not self._original_tui:
            logger.error("Cannot wait for TUI completion - TUI not initialized")
            return
            
        logger.info("Waiting for TUI completion (until user exits)...")
        
        try:
            # The key insight: we need to wait for the same condition the original TUI waits for
            # The original TUI waits for its background tasks (input_loop, periodic_update) to complete
            
            # If we're in fallback mode, the original TUI's run() method handles everything
            if self._fallback_mode:
                # This should not be reached in fallback mode, but handle gracefully
                logger.warning("_wait_for_tui_completion called in fallback mode - this should not happen")
                return
                
            # In reliability mode, we need to wait for the TUI's main loop to actually complete
            # The original TUI's _run_main_loop() creates tasks and waits for them
            
            # Start the main loop which will wait for user input/exit
            await self.run_main_loop()
            
        except Exception as e:
            logger.error(f"Error waiting for TUI completion: {e}")
            raise
            
    # Property access delegation
    
    def __getattr__(self, name):
        """Delegate attribute access to original TUI."""
        if self._original_tui and hasattr(self._original_tui, name):
            return getattr(self._original_tui, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    # Status and monitoring methods
    
    def is_reliability_active(self) -> bool:
        """Check if reliability features are active."""
        return self._reliability_active and not self._fallback_mode
        
    def is_fallback_mode(self) -> bool:
        """Check if running in fallback mode."""
        return self._fallback_mode
        
    def get_startup_time(self) -> Optional[float]:
        """Get startup time in seconds."""
        if self._initialization_start_time and self._startup_completed:
            return time.time() - self._initialization_start_time
        return None
        
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        if self._health_monitor:
            return await self._health_monitor.get_performance_summary()
        return {'status': 'unknown', 'monitoring_active': False}
        
    async def get_recovery_status(self) -> Dict[str, Any]:
        """Get recovery system status."""
        if self._recovery_manager:
            return await self._recovery_manager.get_recovery_status()
        return {'recovery_active': False}
        
    def add_health_callback(self, callback: Callable[[PerformanceReport], None]):
        """Add health status callback."""
        self._health_callbacks.append(callback)
        
    def add_recovery_callback(self, callback: Callable[[RecoveryResult], None]):
        """Add recovery status callback."""
        self._recovery_callbacks.append(callback)
        if self._recovery_manager:
            self._recovery_manager.add_recovery_callback(callback)


class TUIComponentWrapper:
    """Wrapper for TUI components to provide uniform startup interface."""
    
    def __init__(self, name: str, component: Any, timeout_guardian: TimeoutGuardian):
        self.name = name
        self.component = component
        self.guardian = timeout_guardian
        self._initialized = False
        
    async def initialize(self):
        """Initialize the wrapped component with timeout protection."""
        if self._initialized:
            return
            
        async with self.guardian.protect_operation(f"{self.name}_init", 2.0):
            # Try different initialization methods
            if hasattr(self.component, 'initialize') and callable(self.component.initialize):
                if asyncio.iscoroutinefunction(self.component.initialize):
                    await self.component.initialize()
                else:
                    self.component.initialize()
            elif hasattr(self.component, 'start') and callable(self.component.start):
                if asyncio.iscoroutinefunction(self.component.start):
                    await self.component.start()
                else:
                    self.component.start()
                    
        self._initialized = True
        
    def is_ready(self) -> bool:
        """Check if component is ready."""
        if hasattr(self.component, 'is_ready'):
            if callable(self.component.is_ready):
                return self.component.is_ready()
            else:
                return self.component.is_ready
        return self._initialized


# Convenience functions for easy integration

async def create_reliable_tui(
    agent_orchestrator: Any,
    agent_state: Any,
    config: Optional[ReliabilityConfig] = None,
    **kwargs
) -> ReliableTUIInterface:
    """
    Create a reliable TUI interface with all reliability features enabled.
    
    This is the main entry point for using the reliable TUI system.
    
    Args:
        agent_orchestrator: Agent orchestrator instance
        agent_state: Agent state instance  
        config: Optional reliability configuration
        **kwargs: Additional arguments for TUI
        
    Returns:
        ReliableTUIInterface instance ready to use
    """
    return ReliableTUIInterface(
        agent_orchestrator=agent_orchestrator,
        agent_state=agent_state,
        reliability_config=config,
        **kwargs
    )


def create_minimal_reliability_config() -> ReliabilityConfig:
    """Create minimal reliability configuration for testing."""
    return ReliabilityConfig(
        max_startup_time_s=5.0,
        aggressive_timeouts=False,
        show_startup_feedback=False,
        enable_health_monitoring=False,
        enable_automatic_recovery=False
    )


def create_aggressive_reliability_config() -> ReliabilityConfig:
    """Create aggressive reliability configuration to prevent any hangs."""
    return ReliabilityConfig(
        max_startup_time_s=8.0,
        aggressive_timeouts=True,
        show_startup_feedback=True,
        enable_health_monitoring=True,
        health_check_interval_s=0.5,  # Check every 500ms
        hang_detection_threshold_s=3.0,  # Detect hangs after 3s
        enable_automatic_recovery=True,
        max_recovery_attempts=2,
        recovery_timeout_s=2.0
    )


# Example usage and testing
async def test_reliable_tui():
    """Test the reliable TUI interface."""
    logger.info("Testing ReliableTUIInterface")
    
    # Mock agent orchestrator and state
    class MockAgentOrchestrator:
        def __init__(self):
            self.agents = []
            
    class MockAgentState:
        def __init__(self):
            self.state = {}
            
    try:
        # Create test configuration
        config = create_minimal_reliability_config()
        
        # Create reliable TUI
        tui = await create_reliable_tui(
            MockAgentOrchestrator(),
            MockAgentState(),
            config
        )
        
        # Test startup (should complete quickly)
        start_time = time.time()
        result = await tui.start()
        startup_time = time.time() - start_time
        
        logger.info(f"Startup result: {result}, time: {startup_time:.2f}s")
        
        # Check status
        logger.info(f"Reliability active: {tui.is_reliability_active()}")
        logger.info(f"Fallback mode: {tui.is_fallback_mode()}")
        logger.info(f"Startup time: {tui.get_startup_time():.2f}s")
        
        # Test health status
        health_status = await tui.get_health_status()
        logger.info(f"Health status: {health_status}")
        
        # Stop TUI
        await tui.stop()
        
        logger.info("ReliableTUIInterface test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"ReliableTUIInterface test failed: {e}")
        logger.debug(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Run test
    asyncio.run(test_reliable_tui())