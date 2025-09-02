"""
Unified TUI Coordinator - Single integration point that eliminates dual TUI system conflicts.

Provides the single point of control for ALL TUI functionality, eliminating conflicts
between Rich-based TUI, custom v2 TUI, and orchestrator integration. Manages component
lifecycle, mode switching, and provides clean interface for orchestrator integration.

ICD Compliance:
- Inputs: tui_mode, component_config, orchestrator_integration  
- Outputs: tui_instance, mode_active, integration_status
- Performance: TUI startup within 2 seconds; mode switching within 500ms
- Key Functions: Mode management, component integration, orchestrator liaison
"""

import asyncio
import logging
import sys
import time
import signal
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Type, Tuple
from datetime import datetime
import weakref

try:
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import core infrastructure modules
from .terminal_controller import get_terminal_controller, TerminalController, AlternateScreenMode, CursorVisibility
from .logging_isolation_manager import get_logging_isolation_manager, LoggingIsolationManager
from .text_layout_engine import TextLayoutEngine, WrapMode, OverflowHandling
from .input_rendering_pipeline import InputRenderingPipeline, InputMode
from .display_manager import get_display_manager, DisplayManager, RefreshMode, ContentUpdate

# Import existing TUI components
from .orchestrator_integration import OrchestratorTUIIntegration, OrchestratorIntegrationConfig
from .revolutionary_tui_interface import RevolutionaryTUIInterface
from .fixed_working_tui import FixedWorkingTUI

logger = logging.getLogger(__name__)


class TUIMode(Enum):
    """TUI operation modes."""
    REVOLUTIONARY = "revolutionary"    # Full revolutionary TUI with animations
    BASIC = "basic"                   # Simple, stable TUI interface
    FALLBACK = "fallback"             # Minimal fallback mode
    DISABLED = "disabled"             # TUI completely disabled


class TUIStatus(Enum):
    """TUI system status."""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SWITCHING_MODE = "switching_mode"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class ComponentConfig:
    """Configuration for TUI components."""
    enable_animations: bool = True
    enable_rich_rendering: bool = True
    max_fps: int = 60
    enable_logging_isolation: bool = True
    enable_alternate_screen: bool = True
    cursor_visibility: CursorVisibility = CursorVisibility.AUTO
    terminal_size_detection: bool = True
    performance_monitoring: bool = True
    error_recovery: bool = True


@dataclass
class IntegrationStatus:
    """Status of orchestrator integration."""
    orchestrator_connected: bool = False
    conversation_manager_active: bool = False
    agent_isolation_active: bool = False
    strict_mode_enabled: bool = True
    last_orchestrator_response: Optional[datetime] = None
    integration_errors: List[str] = field(default_factory=list)


@dataclass
class TUIInstance:
    """Represents an active TUI instance."""
    mode: TUIMode
    interface: Any  # The actual TUI interface instance
    startup_time: datetime
    last_activity: datetime
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    is_healthy: bool = True
    error_count: int = 0


class UnifiedTUICoordinator:
    """
    Unified TUI Coordinator - The single integration point for all TUI functionality.
    
    This is THE controller that eliminates all conflicts between multiple TUI implementations
    by providing a single, coordinated entry point. No other component should directly
    instantiate or manage TUI interfaces.
    """
    
    def __init__(self):
        """Initialize the unified TUI coordinator."""
        self._lock = asyncio.Lock()
        self._initialized = False
        
        # Core infrastructure
        self._terminal_controller: Optional[TerminalController] = None
        self._logging_manager: Optional[LoggingIsolationManager] = None
        self._text_layout_engine: Optional[TextLayoutEngine] = None
        self._input_pipeline: Optional[InputRenderingPipeline] = None
        self._display_manager: Optional[DisplayManager] = None
        
        # TUI state management
        self._current_mode = TUIMode.DISABLED
        self._target_mode = TUIMode.DISABLED
        self._status = TUIStatus.INACTIVE
        self._current_instance: Optional[TUIInstance] = None
        self._mode_switch_in_progress = False
        
        # Component configuration
        self._component_config = ComponentConfig()
        self._orchestrator_integration: Optional[OrchestratorTUIIntegration] = None
        self._integration_status = IntegrationStatus()
        
        # Available TUI implementations
        self._tui_implementations: Dict[TUIMode, Type] = {
            TUIMode.REVOLUTIONARY: RevolutionaryTUIInterface,
            TUIMode.BASIC: FixedWorkingTUI,
            TUIMode.FALLBACK: self._create_fallback_tui
        }
        
        # Performance tracking
        self._startup_times: Dict[TUIMode, float] = {}
        self._switch_times: List[float] = []
        self._health_check_interval = 5.0  # seconds
        
        # Event callbacks
        self._mode_change_callbacks: Set[Callable[[TUIMode, TUIMode], None]] = set()
        self._status_change_callbacks: Set[Callable[[TUIStatus], None]] = set()
        self._error_callbacks: Set[Callable[[str, Exception], None]] = set()
        
        # Background tasks
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._cleanup_handlers_registered = False
        
        # Error recovery
        self._max_error_count = 3
        self._error_recovery_delay = 1.0
        self._fallback_on_errors = True
        
    async def initialize(self, 
                        component_config: Optional[ComponentConfig] = None,
                        orchestrator_integration: Optional[OrchestratorTUIIntegration] = None) -> bool:
        """
        Initialize the unified TUI coordinator.
        
        Args:
            component_config: Configuration for TUI components
            orchestrator_integration: Orchestrator integration instance
            
        Returns:
            True if initialization successful, False otherwise
        """
        async with self._lock:
            if self._initialized:
                return True
            
            start_time = time.time()
            
            try:
                logger.info("Initializing Unified TUI Coordinator...")
                
                # Update configuration
                if component_config:
                    self._component_config = component_config
                
                if orchestrator_integration:
                    self._orchestrator_integration = orchestrator_integration
                
                # Initialize core infrastructure in dependency order
                await self._initialize_infrastructure()
                
                # Register cleanup handlers
                self._register_cleanup_handlers()
                
                # Start health monitoring
                self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
                
                self._status = TUIStatus.INACTIVE
                self._initialized = True
                
                initialization_time = time.time() - start_time
                logger.info(f"Unified TUI Coordinator initialized in {initialization_time:.2f}s")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize unified TUI coordinator: {e}")
                self._status = TUIStatus.ERROR
                return False
    
    async def start_tui(self,
                       tui_mode: TUIMode,
                       orchestrator_integration: Any = None) -> Tuple[Optional[TUIInstance], bool, Dict[str, Any]]:
        """
        Start TUI in specified mode.
        
        Args:
            tui_mode: Desired TUI mode
            orchestrator_integration: Optional orchestrator integration
            
        Returns:
            Tuple of (tui_instance, mode_active, integration_status)
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        async with self._lock:
            try:
                logger.info(f"Starting TUI in {tui_mode.value} mode...")
                
                # Update status
                old_status = self._status
                self._status = TUIStatus.INITIALIZING
                self._notify_status_change(self._status)
                
                # Shutdown existing instance if active
                if self._current_instance:
                    await self._shutdown_current_instance()
                
                # Initialize orchestrator integration if provided
                if orchestrator_integration:
                    self._orchestrator_integration = orchestrator_integration
                    await self._setup_orchestrator_integration()
                
                # Create and start TUI instance
                tui_instance = await self._create_tui_instance(tui_mode)
                if not tui_instance:
                    raise Exception(f"Failed to create TUI instance for mode {tui_mode.value}")
                
                # Start the TUI interface
                await self._start_tui_interface(tui_instance)
                
                # Update state
                self._current_mode = tui_mode
                self._target_mode = tui_mode
                self._current_instance = tui_instance
                self._status = TUIStatus.ACTIVE
                
                # Track performance (ICD requirement: startup within 2 seconds)
                startup_time = time.time() - start_time
                self._startup_times[tui_mode] = startup_time
                
                if startup_time > 2.0:
                    logger.warning(f"TUI startup took {startup_time:.2f}s (>2s target)")
                
                # Notify callbacks
                self._notify_mode_change(TUIMode.DISABLED, tui_mode)
                self._notify_status_change(TUIStatus.ACTIVE)
                
                logger.info(f"TUI {tui_mode.value} mode started successfully in {startup_time:.2f}s")
                
                return (
                    tui_instance,
                    True,
                    {
                        'startup_time_seconds': startup_time,
                        'orchestrator_connected': self._integration_status.orchestrator_connected,
                        'mode': tui_mode.value,
                        'status': self._status.value
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to start TUI: {e}")
                self._status = TUIStatus.ERROR
                self._notify_error("tui_startup", e)
                
                # Attempt fallback if enabled and not already in fallback
                if (self._fallback_on_errors and 
                    tui_mode != TUIMode.FALLBACK and 
                    tui_mode != TUIMode.DISABLED):
                    logger.info("Attempting fallback TUI mode due to startup error...")
                    return await self.start_tui(TUIMode.FALLBACK)
                
                return None, False, {'error': str(e)}
    
    async def switch_mode(self, new_mode: TUIMode) -> Tuple[bool, Dict[str, Any]]:
        """
        Switch TUI to a different mode.
        
        Args:
            new_mode: Target TUI mode
            
        Returns:
            Tuple of (success, performance_metrics)
        """
        if not self._initialized or self._mode_switch_in_progress:
            return False, {'error': 'Coordinator not ready or switch in progress'}
        
        if new_mode == self._current_mode:
            return True, {'message': 'Already in target mode'}
        
        start_time = time.time()
        
        async with self._lock:
            try:
                self._mode_switch_in_progress = True
                self._target_mode = new_mode
                self._status = TUIStatus.SWITCHING_MODE
                
                logger.info(f"Switching TUI mode from {self._current_mode.value} to {new_mode.value}")
                
                old_mode = self._current_mode
                
                # Gracefully shutdown current instance
                if self._current_instance:
                    await self._shutdown_current_instance()
                
                # Start new mode
                if new_mode != TUIMode.DISABLED:
                    tui_instance, mode_active, _ = await self.start_tui(new_mode)
                    if not mode_active:
                        raise Exception(f"Failed to start {new_mode.value} mode")
                else:
                    # Complete shutdown
                    self._current_mode = TUIMode.DISABLED
                    self._current_instance = None
                    self._status = TUIStatus.INACTIVE
                
                # Track performance (ICD requirement: mode switching within 500ms)
                switch_time = time.time() - start_time
                self._switch_times.append(switch_time)
                
                if switch_time > 0.5:
                    logger.warning(f"Mode switch took {switch_time:.2f}s (>500ms target)")
                
                # Notify callbacks
                self._notify_mode_change(old_mode, new_mode)
                
                logger.info(f"Mode switch completed in {switch_time:.3f}s")
                
                return True, {
                    'switch_time_seconds': switch_time,
                    'old_mode': old_mode.value,
                    'new_mode': new_mode.value
                }
                
            except Exception as e:
                logger.error(f"Failed to switch TUI mode: {e}")
                self._status = TUIStatus.ERROR
                self._notify_error("mode_switch", e)
                
                return False, {'error': str(e)}
            
            finally:
                self._mode_switch_in_progress = False
    
    async def _initialize_infrastructure(self) -> None:
        """Initialize core infrastructure components in correct order."""
        logger.debug("Initializing infrastructure components...")
        
        # Terminal controller (base dependency)
        self._terminal_controller = await get_terminal_controller()
        if not await self._terminal_controller.initialize():
            raise Exception("Failed to initialize terminal controller")
        
        # Logging isolation manager
        self._logging_manager = await get_logging_isolation_manager()
        if not await self._logging_manager.initialize():
            raise Exception("Failed to initialize logging isolation manager")
        
        # Text layout engine
        self._text_layout_engine = TextLayoutEngine()
        
        # Input rendering pipeline
        self._input_pipeline = InputRenderingPipeline()
        self._input_pipeline.configure(
            max_width=80,  # Will be updated with actual terminal size
            cursor_style=self._component_config.cursor_visibility
        )
        
        # Display manager
        self._display_manager = await get_display_manager()
        if not await self._display_manager.initialize():
            raise Exception("Failed to initialize display manager")
        
        logger.debug("Infrastructure components initialized successfully")
    
    async def _create_tui_instance(self, mode: TUIMode) -> Optional[TUIInstance]:
        """Create a TUI instance for the specified mode."""
        try:
            if mode not in self._tui_implementations:
                raise Exception(f"Unsupported TUI mode: {mode.value}")
            
            tui_class = self._tui_implementations[mode]
            
            # Prepare constructor arguments
            constructor_args = {
                'cli_config': self._component_config,
                'orchestrator_integration': self._orchestrator_integration,
                'revolutionary_components': {
                    'terminal_controller': self._terminal_controller,
                    'logging_manager': self._logging_manager,
                    'text_layout_engine': self._text_layout_engine,
                    'input_pipeline': self._input_pipeline,
                    'display_manager': self._display_manager
                }
            }
            
            # Handle special case for fallback TUI
            if mode == TUIMode.FALLBACK:
                interface = await tui_class()
            else:
                # Filter arguments based on what the class accepts
                if mode == TUIMode.REVOLUTIONARY:
                    interface = tui_class(**constructor_args)
                else:
                    # Basic TUI may not accept all arguments
                    interface = tui_class()
            
            return TUIInstance(
                mode=mode,
                interface=interface,
                startup_time=datetime.now(),
                last_activity=datetime.now(),
                is_healthy=True,
                error_count=0
            )
            
        except Exception as e:
            logger.error(f"Failed to create TUI instance for mode {mode.value}: {e}")
            return None
    
    async def _start_tui_interface(self, tui_instance: TUIInstance) -> None:
        """Start a TUI interface instance."""
        interface = tui_instance.interface
        
        # Activate logging isolation
        if self._component_config.enable_logging_isolation and self._logging_manager:
            await self._logging_manager.activate_isolation(tui_active=True)
        
        # Enter alternate screen if enabled
        if (self._component_config.enable_alternate_screen and 
            self._terminal_controller):
            await self._terminal_controller.enter_alternate_screen(
                AlternateScreenMode.AUTO if self._component_config.enable_alternate_screen else AlternateScreenMode.DISABLED
            )
        
        # Start the interface
        if hasattr(interface, 'start') and callable(interface.start):
            if asyncio.iscoroutinefunction(interface.start):
                await interface.start()
            else:
                interface.start()
        elif hasattr(interface, 'run') and callable(interface.run):
            # Don't await run() as it may be blocking - start in background
            if not asyncio.iscoroutinefunction(interface.run):
                # Start synchronous run in thread pool
                asyncio.create_task(asyncio.get_event_loop().run_in_executor(None, interface.run))
            else:
                asyncio.create_task(interface.run())
    
    async def _shutdown_current_instance(self) -> None:
        """Gracefully shutdown the current TUI instance."""
        if not self._current_instance:
            return
        
        try:
            interface = self._current_instance.interface
            
            # Call shutdown/stop method if available
            if hasattr(interface, 'stop') and callable(interface.stop):
                if asyncio.iscoroutinefunction(interface.stop):
                    await interface.stop()
                else:
                    interface.stop()
            elif hasattr(interface, 'shutdown') and callable(interface.shutdown):
                if asyncio.iscoroutinefunction(interface.shutdown):
                    await interface.shutdown()
                else:
                    interface.shutdown()
            
            # Exit alternate screen
            if self._terminal_controller:
                await self._terminal_controller.exit_alternate_screen()
            
            # Deactivate logging isolation
            if self._logging_manager:
                await self._logging_manager.deactivate_isolation()
            
        except Exception as e:
            logger.error(f"Error shutting down TUI instance: {e}")
        finally:
            self._current_instance = None
    
    async def _setup_orchestrator_integration(self) -> None:
        """Setup orchestrator integration."""
        if not self._orchestrator_integration:
            return
        
        try:
            # Initialize orchestrator integration
            if hasattr(self._orchestrator_integration, 'initialize'):
                if asyncio.iscoroutinefunction(self._orchestrator_integration.initialize):
                    await self._orchestrator_integration.initialize()
                else:
                    self._orchestrator_integration.initialize()
            
            # Update integration status
            self._integration_status.orchestrator_connected = True
            self._integration_status.strict_mode_enabled = True
            self._integration_status.last_orchestrator_response = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to setup orchestrator integration: {e}")
            self._integration_status.integration_errors.append(str(e))
    
    async def _create_fallback_tui(self) -> Any:
        """Create minimal fallback TUI."""
        class MinimalFallbackTUI:
            """Minimal fallback TUI implementation."""
            
            def __init__(self):
                self.active = False
                self.console = Console() if RICH_AVAILABLE else None
            
            async def start(self):
                self.active = True
                if self.console:
                    self.console.print("[yellow]Fallback TUI mode active[/yellow]")
                else:
                    print("Fallback TUI mode active")
            
            async def stop(self):
                self.active = False
                if self.console:
                    self.console.print("[yellow]Fallback TUI stopping[/yellow]")
                else:
                    print("Fallback TUI stopping")
        
        return MinimalFallbackTUI()
    
    async def _health_monitor_loop(self) -> None:
        """Monitor TUI health and perform recovery if needed."""
        while self._initialized:
            try:
                await self._check_tui_health()
                await asyncio.sleep(self._health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(self._health_check_interval)
    
    async def _check_tui_health(self) -> None:
        """Check TUI health and perform recovery if needed."""
        if not self._current_instance or self._status != TUIStatus.ACTIVE:
            return
        
        try:
            # Check if interface is responsive
            interface = self._current_instance.interface
            is_healthy = True
            
            # Check if interface has health check method
            if hasattr(interface, 'is_healthy') and callable(interface.is_healthy):
                if asyncio.iscoroutinefunction(interface.is_healthy):
                    is_healthy = await interface.is_healthy()
                else:
                    is_healthy = interface.is_healthy()
            
            # Update health status
            self._current_instance.is_healthy = is_healthy
            
            if not is_healthy:
                self._current_instance.error_count += 1
                logger.warning(f"TUI health check failed (error count: {self._current_instance.error_count})")
                
                # Trigger recovery if too many errors
                if self._current_instance.error_count >= self._max_error_count:
                    logger.error("TUI health check failed too many times, triggering recovery")
                    await self._trigger_recovery()
            else:
                # Reset error count on successful health check
                self._current_instance.error_count = 0
                self._current_instance.last_activity = datetime.now()
        
        except Exception as e:
            logger.error(f"TUI health check error: {e}")
    
    async def _trigger_recovery(self) -> None:
        """Trigger TUI recovery."""
        try:
            logger.info("Triggering TUI recovery...")
            
            current_mode = self._current_mode
            
            # Try to restart in same mode first
            await asyncio.sleep(self._error_recovery_delay)
            success, _ = await self.switch_mode(current_mode)
            
            if not success and self._fallback_on_errors and current_mode != TUIMode.FALLBACK:
                # Fall back to fallback mode
                logger.info("Recovery failed, falling back to fallback mode")
                await asyncio.sleep(self._error_recovery_delay)
                await self.switch_mode(TUIMode.FALLBACK)
            
        except Exception as e:
            logger.error(f"TUI recovery failed: {e}")
            self._status = TUIStatus.ERROR
    
    def _register_cleanup_handlers(self) -> None:
        """Register cleanup handlers for graceful shutdown."""
        if self._cleanup_handlers_registered:
            return
        
        def cleanup_handler(signum=None, frame=None):
            """Emergency cleanup handler."""
            try:
                if self._current_instance:
                    # Synchronous cleanup
                    if self._terminal_controller:
                        asyncio.run_coroutine_threadsafe(
                            self._terminal_controller.exit_alternate_screen(),
                            asyncio.get_event_loop()
                        )
                    
                    if self._logging_manager:
                        asyncio.run_coroutine_threadsafe(
                            self._logging_manager.deactivate_isolation(),
                            asyncio.get_event_loop()
                        )
            except:
                pass  # Best effort cleanup
        
        # Register signal handlers
        try:
            signal.signal(signal.SIGINT, cleanup_handler)
            signal.signal(signal.SIGTERM, cleanup_handler)
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, cleanup_handler)
        except:
            pass  # May not work in all environments
        
        # Register atexit handler
        import atexit
        atexit.register(cleanup_handler)
        
        self._cleanup_handlers_registered = True
    
    def _notify_mode_change(self, old_mode: TUIMode, new_mode: TUIMode) -> None:
        """Notify mode change callbacks."""
        for callback in self._mode_change_callbacks:
            try:
                callback(old_mode, new_mode)
            except Exception:
                pass  # Don't let callback errors break coordinator
    
    def _notify_status_change(self, status: TUIStatus) -> None:
        """Notify status change callbacks."""
        for callback in self._status_change_callbacks:
            try:
                callback(status)
            except Exception:
                pass  # Don't let callback errors break coordinator
    
    def _notify_error(self, error_type: str, error: Exception) -> None:
        """Notify error callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(error_type, error)
            except Exception:
                pass  # Don't let callback errors break coordinator
    
    # Public API methods
    
    @asynccontextmanager
    async def tui_session(self, mode: TUIMode):
        """Context manager for TUI sessions."""
        tui_instance = None
        try:
            tui_instance, mode_active, status = await self.start_tui(mode)
            if not mode_active:
                raise Exception(f"Failed to start TUI in {mode.value} mode")
            
            yield tui_instance
            
        finally:
            if tui_instance:
                await self.switch_mode(TUIMode.DISABLED)
    
    def register_mode_change_callback(self, callback: Callable[[TUIMode, TUIMode], None]) -> None:
        """Register a callback for mode changes."""
        self._mode_change_callbacks.add(callback)
    
    def register_status_change_callback(self, callback: Callable[[TUIStatus], None]) -> None:
        """Register a callback for status changes."""
        self._status_change_callbacks.add(callback)
    
    def register_error_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """Register a callback for errors."""
        self._error_callbacks.add(callback)
    
    def get_current_mode(self) -> TUIMode:
        """Get current TUI mode."""
        return self._current_mode
    
    def get_status(self) -> TUIStatus:
        """Get current TUI status."""
        return self._status
    
    def get_integration_status(self) -> IntegrationStatus:
        """Get orchestrator integration status."""
        return self._integration_status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_switch_time = sum(self._switch_times) / len(self._switch_times) if self._switch_times else 0
        
        return {
            'current_mode': self._current_mode.value,
            'status': self._status.value,
            'startup_times': dict(self._startup_times),
            'average_switch_time_ms': avg_switch_time * 1000,
            'switch_count': len(self._switch_times),
            'health_status': self._current_instance.is_healthy if self._current_instance else None,
            'error_count': self._current_instance.error_count if self._current_instance else 0,
            'initialized': self._initialized,
            'infrastructure': {
                'terminal_controller': self._terminal_controller is not None,
                'logging_manager': self._logging_manager is not None,
                'display_manager': self._display_manager is not None,
                'text_layout_engine': self._text_layout_engine is not None,
                'input_pipeline': self._input_pipeline is not None
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown the coordinator and all components."""
        if not self._initialized:
            return
        
        logger.info("Shutting down Unified TUI Coordinator...")
        
        self._status = TUIStatus.SHUTTING_DOWN
        self._initialized = False
        
        # Cancel health monitor
        if self._health_monitor_task and not self._health_monitor_task.done():
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown current TUI instance
        if self._current_instance:
            await self._shutdown_current_instance()
        
        # Cleanup infrastructure
        if self._display_manager:
            await self._display_manager.cleanup()
        
        if self._logging_manager:
            await self._logging_manager.cleanup()
        
        if self._terminal_controller:
            await self._terminal_controller.cleanup()
        
        # Clear state
        self._mode_change_callbacks.clear()
        self._status_change_callbacks.clear()
        self._error_callbacks.clear()
        
        logger.info("Unified TUI Coordinator shutdown complete")


# Singleton instance for global access
_unified_tui_coordinator: Optional[UnifiedTUICoordinator] = None


async def get_unified_tui_coordinator() -> UnifiedTUICoordinator:
    """
    Get or create the global unified TUI coordinator instance.
    
    Returns:
        UnifiedTUICoordinator instance
    """
    global _unified_tui_coordinator
    
    if _unified_tui_coordinator is None:
        _unified_tui_coordinator = UnifiedTUICoordinator()
        await _unified_tui_coordinator.initialize()
    
    return _unified_tui_coordinator


async def cleanup_unified_tui_coordinator() -> None:
    """
    Cleanup the global unified TUI coordinator.
    """
    global _unified_tui_coordinator
    
    if _unified_tui_coordinator:
        await _unified_tui_coordinator.shutdown()
        _unified_tui_coordinator = None


# Convenience functions for common TUI operations

async def start_revolutionary_tui(orchestrator_integration: Any = None) -> Tuple[bool, Dict[str, Any]]:
    """Start the revolutionary TUI mode."""
    coordinator = await get_unified_tui_coordinator()
    tui_instance, mode_active, status = await coordinator.start_tui(
        TUIMode.REVOLUTIONARY, 
        orchestrator_integration=orchestrator_integration
    )
    return mode_active, status


async def start_basic_tui(orchestrator_integration: Any = None) -> Tuple[bool, Dict[str, Any]]:
    """Start the basic TUI mode."""
    coordinator = await get_unified_tui_coordinator()
    tui_instance, mode_active, status = await coordinator.start_tui(
        TUIMode.BASIC,
        orchestrator_integration=orchestrator_integration
    )
    return mode_active, status


async def switch_tui_mode(new_mode: TUIMode) -> Tuple[bool, Dict[str, Any]]:
    """Switch TUI to a different mode."""
    coordinator = await get_unified_tui_coordinator()
    return await coordinator.switch_mode(new_mode)


async def stop_tui() -> Tuple[bool, Dict[str, Any]]:
    """Stop the current TUI."""
    coordinator = await get_unified_tui_coordinator()
    return await coordinator.switch_mode(TUIMode.DISABLED)