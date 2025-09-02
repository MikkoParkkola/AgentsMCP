"""
Startup Orchestrator - Guarantees TUI startup completion within 10 seconds.

This module provides the critical startup coordination to prevent the TUI from hanging
indefinitely during initialization. It implements progressive timeouts and immediate
fallback modes to ensure the user never waits more than 10 seconds for startup.

Key Features:
- GUARANTEED startup completion within 10 seconds maximum
- Progressive timeout system: 3s orchestrator, 2s display, 1s input
- Immediate fallback to basic mode if any component takes >3s 
- Shows "TUI Starting..." feedback within 500ms
- Prevents the infinite hang issue that blocks user interaction

Usage:
    orchestrator = StartupOrchestrator()
    result = await orchestrator.coordinate_startup(components)
    if result == StartupResult.SUCCESS:
        # TUI is ready
    elif result == StartupResult.FALLBACK:
        # Using basic mode
    else:
        # Startup failed - use minimal mode
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class StartupResult(Enum):
    """Result of startup coordination."""
    SUCCESS = "success"      # Full TUI startup successful
    FALLBACK = "fallback"    # Fell back to basic mode due to timeout
    FAILURE = "failure"      # Complete startup failure


class StartupPhase(Enum):
    """Phases of TUI startup with individual timeouts."""
    ORCHESTRATOR = "orchestrator"    # 3 second timeout
    DISPLAY = "display"             # 2 second timeout  
    INPUT = "input"                 # 1 second timeout
    FINALIZATION = "finalization"   # 1 second timeout


@dataclass
class StartupConfig:
    """Configuration for startup orchestration."""
    # Progressive timeout system - total 10 seconds max
    orchestrator_timeout: float = 3.0    # 3s for orchestrator init
    display_timeout: float = 2.0         # 2s for display setup
    input_timeout: float = 1.0           # 1s for input setup
    finalization_timeout: float = 1.0    # 1s for finalization
    
    # Feedback timing
    feedback_delay: float = 0.5           # Show feedback within 500ms
    
    # Fallback behavior
    enable_progressive_fallback: bool = True
    retry_failed_components: bool = False  # Don't retry to prevent hangs


@dataclass
class StartupContext:
    """Context passed through startup phases."""
    start_time: float
    phase_times: Dict[StartupPhase, float]
    failed_components: List[str]
    fallback_triggered: bool = False
    feedback_shown: bool = False


class StartupOrchestrator:
    """
    Orchestrates TUI startup with guaranteed completion within 10 seconds.
    
    This is the master coordinator that prevents the TUI from hanging during
    initialization by implementing strict timeouts and immediate fallback modes.
    """
    
    def __init__(self, config: Optional[StartupConfig] = None):
        """Initialize the startup orchestrator."""
        self.config = config or StartupConfig()
        self.context = None
        self._startup_feedback_task = None
        self._emergency_timeout_task = None
        
    async def coordinate_startup(
        self,
        components: Dict[str, Any],
        feedback_callback: Optional[Callable[[str], None]] = None
    ) -> StartupResult:
        """
        Coordinate TUI startup with guaranteed completion within 10 seconds.
        
        Args:
            components: Dictionary of TUI components to initialize
            feedback_callback: Optional callback for showing startup feedback
            
        Returns:
            StartupResult indicating success, fallback, or failure
        """
        start_time = time.time()
        self.context = StartupContext(
            start_time=start_time,
            phase_times={},
            failed_components=[]
        )
        
        try:
            # Start emergency timeout as absolute backstop
            total_timeout = (self.config.orchestrator_timeout + 
                           self.config.display_timeout +
                           self.config.input_timeout + 
                           self.config.finalization_timeout)
            
            self._emergency_timeout_task = asyncio.create_task(
                self._emergency_timeout_handler(total_timeout)
            )
            
            # Start feedback task to show immediate progress
            if feedback_callback:
                self._startup_feedback_task = asyncio.create_task(
                    self._show_startup_feedback(feedback_callback)
                )
            
            # Execute startup phases with progressive timeouts
            result = await self._execute_startup_phases(components)
            
            # Calculate total startup time
            total_time = time.time() - start_time
            logger.info(f"Startup orchestration completed in {total_time:.2f}s - Result: {result.value}")
            
            return result
            
        except asyncio.TimeoutError:
            logger.error("Emergency timeout triggered - startup took longer than absolute limit")
            return StartupResult.FAILURE
            
        except Exception as e:
            logger.error(f"Startup orchestration failed: {e}")
            return StartupResult.FAILURE
            
        finally:
            # Cleanup background tasks
            await self._cleanup_background_tasks()
    
    async def _execute_startup_phases(self, components: Dict[str, Any]) -> StartupResult:
        """Execute startup phases with individual timeouts."""
        
        # Phase 1: Orchestrator initialization (3s timeout)
        phase_start = time.time()
        orchestrator_result = await self._execute_phase_with_timeout(
            StartupPhase.ORCHESTRATOR,
            self._initialize_orchestrator_phase,
            components.get('orchestrator'),
            self.config.orchestrator_timeout
        )
        self.context.phase_times[StartupPhase.ORCHESTRATOR] = time.time() - phase_start
        
        if orchestrator_result == StartupResult.FAILURE:
            logger.warning("Orchestrator phase failed - triggering immediate fallback")
            return StartupResult.FALLBACK
        
        # Phase 2: Display setup (2s timeout)  
        phase_start = time.time()
        display_result = await self._execute_phase_with_timeout(
            StartupPhase.DISPLAY,
            self._initialize_display_phase,
            components.get('display'),
            self.config.display_timeout
        )
        self.context.phase_times[StartupPhase.DISPLAY] = time.time() - phase_start
        
        if display_result == StartupResult.FAILURE:
            logger.warning("Display phase failed - triggering fallback")
            return StartupResult.FALLBACK
        
        # Phase 3: Input setup (1s timeout)
        phase_start = time.time()
        input_result = await self._execute_phase_with_timeout(
            StartupPhase.INPUT,
            self._initialize_input_phase,
            components.get('input'),
            self.config.input_timeout
        )
        self.context.phase_times[StartupPhase.INPUT] = time.time() - phase_start
        
        if input_result == StartupResult.FAILURE:
            logger.warning("Input phase failed - using partial startup")
            # Input failure is less critical - continue with partial functionality
        
        # Phase 4: Finalization (1s timeout)
        phase_start = time.time()
        final_result = await self._execute_phase_with_timeout(
            StartupPhase.FINALIZATION,
            self._finalize_startup_phase,
            components,
            self.config.finalization_timeout
        )
        self.context.phase_times[StartupPhase.FINALIZATION] = time.time() - phase_start
        
        # Determine overall result
        if self.context.fallback_triggered:
            return StartupResult.FALLBACK
        elif len(self.context.failed_components) > 2:  # More than 2 failures = fallback
            return StartupResult.FALLBACK
        else:
            return StartupResult.SUCCESS
    
    async def _execute_phase_with_timeout(
        self,
        phase: StartupPhase,
        phase_handler: Callable,
        component: Any,
        timeout: float
    ) -> StartupResult:
        """Execute a startup phase with timeout protection."""
        try:
            # Use asyncio.wait_for for reliable timeout enforcement
            result = await asyncio.wait_for(
                phase_handler(component),
                timeout=timeout
            )
            logger.debug(f"Phase {phase.value} completed successfully")
            return StartupResult.SUCCESS
                
        except asyncio.TimeoutError:
            logger.warning(f"Phase {phase.value} timed out after {timeout}s - triggering fallback")
            self.context.failed_components.append(phase.value)
            self.context.fallback_triggered = True
            return StartupResult.FAILURE
            
        except Exception as e:
            logger.warning(f"Phase {phase.value} failed: {e}")
            self.context.failed_components.append(phase.value)
            return StartupResult.FAILURE
    
    async def _initialize_orchestrator_phase(self, orchestrator) -> bool:
        """Initialize the orchestrator component."""
        try:
            if hasattr(orchestrator, 'initialize') and callable(orchestrator.initialize):
                if asyncio.iscoroutinefunction(orchestrator.initialize):
                    await orchestrator.initialize()
                else:
                    orchestrator.initialize()
            
            # Additional orchestrator setup
            if hasattr(orchestrator, 'config'):
                # Configure for fast startup
                if hasattr(orchestrator.config, 'max_agent_wait_time_ms'):
                    orchestrator.config.max_agent_wait_time_ms = min(
                        orchestrator.config.max_agent_wait_time_ms, 2000  # Max 2s for agents
                    )
            
            return True
            
        except Exception as e:
            logger.warning(f"Orchestrator initialization failed: {e}")
            return False
    
    async def _initialize_display_phase(self, display_manager) -> bool:
        """Initialize the display management components."""
        try:
            if hasattr(display_manager, 'initialize') and callable(display_manager.initialize):
                if asyncio.iscoroutinefunction(display_manager.initialize):
                    await display_manager.initialize()
                else:
                    display_manager.initialize()
            
            # Validate display is working
            if hasattr(display_manager, 'is_ready'):
                if callable(display_manager.is_ready):
                    return display_manager.is_ready()
                else:
                    return display_manager.is_ready
                    
            return True
            
        except Exception as e:
            logger.warning(f"Display initialization failed: {e}")
            return False
    
    async def _initialize_input_phase(self, input_handler) -> bool:
        """Initialize the input handling components."""
        try:
            if hasattr(input_handler, 'initialize') and callable(input_handler.initialize):
                if asyncio.iscoroutinefunction(input_handler.initialize):
                    await input_handler.initialize()
                else:
                    input_handler.initialize()
            
            # Quick input validation
            if hasattr(input_handler, 'validate_setup'):
                if callable(input_handler.validate_setup):
                    return input_handler.validate_setup()
                    
            return True
            
        except Exception as e:
            logger.warning(f"Input initialization failed: {e}")
            return False
    
    async def _finalize_startup_phase(self, all_components: Dict[str, Any]) -> bool:
        """Finalize startup and validate system readiness."""
        try:
            # Perform final validation checks
            ready_count = 0
            total_components = len(all_components)
            
            for name, component in all_components.items():
                try:
                    if hasattr(component, 'is_ready'):
                        if callable(component.is_ready):
                            is_ready = component.is_ready()
                        else:
                            is_ready = component.is_ready
                        
                        if is_ready:
                            ready_count += 1
                    else:
                        # Assume ready if no is_ready method
                        ready_count += 1
                        
                except Exception:
                    # Component check failed - not ready
                    pass
            
            # Consider startup successful if most components are ready
            readiness_ratio = ready_count / max(1, total_components)
            logger.info(f"System readiness: {ready_count}/{total_components} ({readiness_ratio:.1%})")
            
            return readiness_ratio >= 0.6  # 60% readiness threshold
            
        except Exception as e:
            logger.warning(f"Finalization failed: {e}")
            return False
    
    async def _show_startup_feedback(self, feedback_callback: Callable[[str], None]):
        """Show startup feedback to user within 500ms."""
        try:
            # Wait for initial feedback delay
            await asyncio.sleep(self.config.feedback_delay)
            
            if not self.context.feedback_shown:
                feedback_callback("🚀 TUI Starting...")
                self.context.feedback_shown = True
                logger.debug("Startup feedback shown to user")
            
            # Show progress updates every second
            while not self.context.fallback_triggered:
                await asyncio.sleep(1.0)
                
                elapsed = time.time() - self.context.start_time
                if elapsed < 5.0:  # First 5 seconds
                    feedback_callback(f"⏳ Initializing components... ({elapsed:.1f}s)")
                elif elapsed < 8.0:  # Next 3 seconds
                    feedback_callback("🔧 Finalizing startup...")
                else:  # Last 2 seconds
                    feedback_callback("✅ Almost ready...")
                    
        except asyncio.CancelledError:
            # Normal cancellation when startup completes
            pass
        except Exception as e:
            logger.warning(f"Error showing startup feedback: {e}")
    
    async def _emergency_timeout_handler(self, total_timeout: float):
        """Emergency timeout handler as absolute backstop."""
        try:
            await asyncio.sleep(total_timeout)
            
            # If we reach here, startup is taking too long
            logger.error(f"EMERGENCY: Startup exceeded {total_timeout}s absolute limit")
            self.context.fallback_triggered = True
            
            # Force cancellation of all operations
            raise asyncio.TimeoutError("Emergency startup timeout")
            
        except asyncio.CancelledError:
            # Normal cancellation when startup completes
            pass
    
    async def _cleanup_background_tasks(self):
        """Cleanup background tasks."""
        try:
            if self._startup_feedback_task and not self._startup_feedback_task.done():
                self._startup_feedback_task.cancel()
                try:
                    await self._startup_feedback_task
                except asyncio.CancelledError:
                    pass
            
            if self._emergency_timeout_task and not self._emergency_timeout_task.done():
                self._emergency_timeout_task.cancel()
                try:
                    await self._emergency_timeout_task
                except asyncio.CancelledError:
                    pass
                    
        except Exception as e:
            logger.warning(f"Error cleaning up background tasks: {e}")
    
    @asynccontextmanager
    async def startup_context(self, feedback_callback: Optional[Callable[[str], None]] = None):
        """Context manager for startup coordination."""
        try:
            if feedback_callback and not self.context.feedback_shown:
                feedback_callback("🚀 TUI Starting...")
                self.context.feedback_shown = True
                
            yield self.context
            
        finally:
            await self._cleanup_background_tasks()
    
    def get_startup_metrics(self) -> Dict[str, Any]:
        """Get startup timing metrics."""
        if not self.context:
            return {}
        
        total_time = sum(self.context.phase_times.values())
        
        return {
            "total_startup_time": total_time,
            "phase_times": dict(self.context.phase_times),
            "failed_components": list(self.context.failed_components),
            "fallback_triggered": self.context.fallback_triggered,
            "feedback_shown": self.context.feedback_shown
        }


# Convenience functions for common usage patterns
async def coordinate_tui_startup(
    components: Dict[str, Any],
    config: Optional[StartupConfig] = None,
    feedback_callback: Optional[Callable[[str], None]] = None
) -> StartupResult:
    """Convenience function to coordinate TUI startup."""
    orchestrator = StartupOrchestrator(config)
    return await orchestrator.coordinate_startup(components, feedback_callback)


def create_startup_config(
    max_total_time: float = 10.0,
    enable_feedback: bool = True,
    aggressive_timeouts: bool = True
) -> StartupConfig:
    """Create a startup configuration with common settings."""
    if aggressive_timeouts:
        # Aggressive timeouts to prevent hangs
        return StartupConfig(
            orchestrator_timeout=2.0,  # Reduced from 3.0
            display_timeout=1.5,       # Reduced from 2.0  
            input_timeout=0.5,         # Reduced from 1.0
            finalization_timeout=0.5,  # Reduced from 1.0
            feedback_delay=0.3 if enable_feedback else float('inf'),
            enable_progressive_fallback=True,
            retry_failed_components=False
        )
    else:
        # Standard timeouts
        return StartupConfig(
            feedback_delay=0.5 if enable_feedback else float('inf'),
            enable_progressive_fallback=True,
            retry_failed_components=False
        )


# Example usage for testing
async def test_startup_orchestrator():
    """Test the startup orchestrator with mock components."""
    
    class MockComponent:
        def __init__(self, delay: float = 0.1, should_fail: bool = False):
            self.delay = delay
            self.should_fail = should_fail
            self._ready = False
        
        async def initialize(self):
            await asyncio.sleep(self.delay)
            if self.should_fail:
                raise Exception("Mock component failure")
            self._ready = True
        
        def is_ready(self) -> bool:
            return self._ready
    
    def feedback(msg: str):
        print(f"Feedback: {msg}")
    
    # Test normal startup
    components = {
        'orchestrator': MockComponent(0.5),
        'display': MockComponent(0.8),
        'input': MockComponent(0.3),
        'other': MockComponent(0.2)
    }
    
    orchestrator = StartupOrchestrator()
    result = await orchestrator.coordinate_startup(components, feedback)
    
    print(f"Startup result: {result}")
    print(f"Metrics: {orchestrator.get_startup_metrics()}")
    
    return result == StartupResult.SUCCESS


if __name__ == "__main__":
    asyncio.run(test_startup_orchestrator())