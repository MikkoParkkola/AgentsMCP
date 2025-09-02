"""
Component Initializer - Initialize TUI components with timeout protection.

This module provides the critical component initialization layer to prevent TUI hangs
during startup. It implements parallel initialization with individual timeouts and
graceful fallback when components fail to initialize properly.

Key Features:
- Initialize TUI components (terminal_controller, logging_isolation_manager, display_manager, event_system) 
- Individual 3-second timeouts per component using timeout_guardian
- Parallel initialization where possible to speed startup
- Graceful handling of failed components - continue with partial functionality
- Returns Dict[str, Component] of successfully initialized components

ICD Compliance:
- Inputs: component_specs, timeout_per_component, parallel_mode
- Outputs: initialized_components, failed_components, initialization_metrics
- Performance: Component initialization must complete within 5s total
- Error Handling: Failed components should not crash entire system
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Type, Union
import weakref
import traceback

from .timeout_guardian import TimeoutGuardian, TimeoutState, get_global_guardian
from ..terminal_controller import TerminalController, AlternateScreenMode, CursorVisibility
from ..logging_isolation_manager import LoggingIsolationManager, LogLevel
from ..display_manager import DisplayManager, RefreshMode
from ..event_system import AsyncEventSystem

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """TUI component types that need initialization."""
    TERMINAL_CONTROLLER = "terminal_controller"
    LOGGING_ISOLATION_MANAGER = "logging_isolation_manager" 
    DISPLAY_MANAGER = "display_manager"
    EVENT_SYSTEM = "event_system"


class InitializationMode(Enum):
    """Component initialization modes."""
    PARALLEL = "parallel"       # Initialize components in parallel
    SEQUENTIAL = "sequential"   # Initialize components one by one
    ADAPTIVE = "adaptive"       # Choose based on dependencies


class ComponentStatus(Enum):
    """Status of component initialization."""
    PENDING = "pending"         # Not started
    INITIALIZING = "initializing"  # In progress
    INITIALIZED = "initialized"    # Successfully initialized
    FAILED = "failed"             # Failed to initialize
    TIMEOUT = "timeout"           # Timed out during initialization


@dataclass
class ComponentSpec:
    """Specification for component initialization."""
    component_type: ComponentType
    component_class: Type
    init_args: Dict[str, Any] = field(default_factory=dict)
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 3.0
    required: bool = True  # If False, failure won't stop startup
    dependencies: List[ComponentType] = field(default_factory=list)
    parallel_safe: bool = True  # Can be initialized in parallel
    
    
@dataclass
class ComponentResult:
    """Result of component initialization."""
    component_type: ComponentType
    status: ComponentStatus
    component: Optional[Any] = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    timeout_occurred: bool = False
    

@dataclass
class InitializationMetrics:
    """Metrics collected during initialization."""
    total_duration_seconds: float
    components_attempted: int
    components_successful: int
    components_failed: int
    components_timeout: int
    parallel_operations: int
    sequential_operations: int
    critical_path_duration: float
    

class ComponentInitializer:
    """
    Initialize TUI components with timeout protection and parallel execution.
    
    Prevents TUI hangs by initializing components with guaranteed timeouts and
    graceful fallback when individual components fail to initialize.
    """
    
    def __init__(self, timeout_guardian: Optional[TimeoutGuardian] = None):
        """Initialize the component initializer."""
        self._guardian = timeout_guardian or get_global_guardian()
        self._results: Dict[ComponentType, ComponentResult] = {}
        self._lock = asyncio.Lock()
        self._initialization_start: Optional[datetime] = None
        self._initialization_end: Optional[datetime] = None
        
        # Default component specifications
        self._default_specs = self._create_default_specs()
        
    def _create_default_specs(self) -> Dict[ComponentType, ComponentSpec]:
        """Create default component specifications."""
        return {
            ComponentType.TERMINAL_CONTROLLER: ComponentSpec(
                component_type=ComponentType.TERMINAL_CONTROLLER,
                component_class=TerminalController,
                timeout_seconds=3.0,
                required=True,
                dependencies=[],
                parallel_safe=True
            ),
            ComponentType.LOGGING_ISOLATION_MANAGER: ComponentSpec(
                component_type=ComponentType.LOGGING_ISOLATION_MANAGER, 
                component_class=LoggingIsolationManager,
                timeout_seconds=3.0,
                required=True,
                dependencies=[],
                parallel_safe=True
            ),
            ComponentType.DISPLAY_MANAGER: ComponentSpec(
                component_type=ComponentType.DISPLAY_MANAGER,
                component_class=DisplayManager,
                timeout_seconds=3.0,
                required=True,
                dependencies=[ComponentType.TERMINAL_CONTROLLER, ComponentType.LOGGING_ISOLATION_MANAGER],
                parallel_safe=False  # Depends on terminal controller
            ),
            ComponentType.EVENT_SYSTEM: ComponentSpec(
                component_type=ComponentType.EVENT_SYSTEM,
                component_class=AsyncEventSystem,
                timeout_seconds=3.0,
                required=False,  # Can continue without event system
                dependencies=[],
                parallel_safe=True
            )
        }
        
    async def initialize_components(
        self,
        component_specs: Optional[Dict[ComponentType, ComponentSpec]] = None,
        mode: InitializationMode = InitializationMode.ADAPTIVE,
        total_timeout_seconds: float = 5.0
    ) -> Dict[str, Any]:
        """
        Initialize TUI components with timeout protection.
        
        Args:
            component_specs: Component specifications to use (defaults to built-in specs)
            mode: Initialization mode (parallel, sequential, adaptive)
            total_timeout_seconds: Total timeout for all initialization (default 5s)
            
        Returns:
            Dict containing:
            - 'components': Dict[str, Component] of successfully initialized components
            - 'failed': Dict[str, Exception] of failed components
            - 'metrics': InitializationMetrics
        """
        specs = component_specs or self._default_specs
        
        async with self._lock:
            self._initialization_start = datetime.now()
            self._results.clear()
            
        logger.info(f"Starting component initialization with {len(specs)} components")
        
        try:
            # Use timeout guardian to protect entire initialization process
            async with self._guardian.protect_operation("component_initialization", total_timeout_seconds):
                if mode == InitializationMode.PARALLEL:
                    results = await self._initialize_parallel(specs)
                elif mode == InitializationMode.SEQUENTIAL:
                    results = await self._initialize_sequential(specs)
                else:  # ADAPTIVE
                    results = await self._initialize_adaptive(specs)
                    
        except asyncio.TimeoutError:
            logger.error(f"Component initialization timed out after {total_timeout_seconds}s")
            results = await self._handle_timeout_scenario()
            
        except Exception as e:
            logger.error(f"Component initialization failed with exception: {e}")
            results = await self._handle_error_scenario(e)
            
        finally:
            self._initialization_end = datetime.now()
            
        return results
        
    async def _initialize_parallel(self, specs: Dict[ComponentType, ComponentSpec]) -> Dict[str, Any]:
        """Initialize components in parallel where possible."""
        logger.info("Using parallel initialization mode")
        
        # Separate components by dependencies
        independent_specs = {ct: spec for ct, spec in specs.items() if not spec.dependencies and spec.parallel_safe}
        dependent_specs = {ct: spec for ct, spec in specs.items() if spec.dependencies or not spec.parallel_safe}
        
        # Initialize independent components first in parallel
        if independent_specs:
            tasks = []
            for component_type, spec in independent_specs.items():
                task = asyncio.create_task(
                    self._initialize_single_component(component_type, spec),
                    name=f"init_{component_type.value}"
                )
                tasks.append(task)
                
            # Wait for independent components to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        # Then initialize dependent components sequentially
        for component_type, spec in dependent_specs.items():
            await self._initialize_single_component(component_type, spec)
            
        return await self._compile_results()
        
    async def _initialize_sequential(self, specs: Dict[ComponentType, ComponentSpec]) -> Dict[str, Any]:
        """Initialize components sequentially."""
        logger.info("Using sequential initialization mode")
        
        # Sort by dependencies (topological sort)
        ordered_specs = self._resolve_initialization_order(specs)
        
        for component_type, spec in ordered_specs.items():
            await self._initialize_single_component(component_type, spec)
            
        return await self._compile_results()
        
    async def _initialize_adaptive(self, specs: Dict[ComponentType, ComponentSpec]) -> Dict[str, Any]:
        """Use adaptive initialization based on component characteristics."""
        logger.info("Using adaptive initialization mode")
        
        # Start with parallel for independent components, then handle dependencies
        return await self._initialize_parallel(specs)
        
    async def _initialize_single_component(
        self,
        component_type: ComponentType,
        spec: ComponentSpec
    ) -> ComponentResult:
        """Initialize a single component with timeout protection."""
        result = ComponentResult(
            component_type=component_type,
            status=ComponentStatus.PENDING,
            start_time=datetime.now()
        )
        
        async with self._lock:
            self._results[component_type] = result
            
        logger.info(f"Initializing {component_type.value} with {spec.timeout_seconds}s timeout")
        
        try:
            result.status = ComponentStatus.INITIALIZING
            
            # Use timeout guardian for individual component
            async with self._guardian.protect_operation(f"init_{component_type.value}", spec.timeout_seconds):
                # Create component instance
                if spec.init_args or spec.init_kwargs:
                    component = spec.component_class(*spec.init_args, **spec.init_kwargs)
                else:
                    component = spec.component_class()
                    
                # Call async initialization if available
                if hasattr(component, 'initialize') and asyncio.iscoroutinefunction(component.initialize):
                    await component.initialize()
                elif hasattr(component, 'initialize'):
                    component.initialize()
                    
                result.component = component
                result.status = ComponentStatus.INITIALIZED
                logger.info(f"Successfully initialized {component_type.value}")
                
        except asyncio.TimeoutError:
            logger.error(f"Component {component_type.value} initialization timed out")
            result.status = ComponentStatus.TIMEOUT
            result.timeout_occurred = True
            
        except Exception as e:
            logger.error(f"Component {component_type.value} initialization failed: {e}")
            result.error = e
            result.status = ComponentStatus.FAILED
            
        finally:
            result.end_time = datetime.now()
            if result.start_time:
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                
        async with self._lock:
            self._results[component_type] = result
            
        return result
        
    def _resolve_initialization_order(
        self,
        specs: Dict[ComponentType, ComponentSpec]
    ) -> Dict[ComponentType, ComponentSpec]:
        """Resolve component initialization order based on dependencies."""
        ordered = {}
        remaining = specs.copy()
        
        while remaining:
            # Find components with no unresolved dependencies
            ready = []
            for component_type, spec in remaining.items():
                if not spec.dependencies or all(dep in ordered for dep in spec.dependencies):
                    ready.append(component_type)
                    
            if not ready:
                # Circular dependency or missing dependency - just add remaining components
                logger.warning("Circular dependency detected, adding remaining components")
                for component_type in remaining:
                    ordered[component_type] = remaining[component_type]
                break
                
            # Add ready components to ordered list
            for component_type in ready:
                ordered[component_type] = remaining.pop(component_type)
                
        return ordered
        
    async def _compile_results(self) -> Dict[str, Any]:
        """Compile final results from component initialization."""
        async with self._lock:
            results = self._results.copy()
            
        components = {}
        failed = {}
        
        for component_type, result in results.items():
            if result.status == ComponentStatus.INITIALIZED and result.component:
                components[component_type.value] = result.component
            else:
                error = result.error or Exception(f"Component failed with status: {result.status}")
                failed[component_type.value] = error
                
        metrics = self._calculate_metrics(results)
        
        logger.info(f"Component initialization complete: {len(components)} successful, {len(failed)} failed")
        
        return {
            'components': components,
            'failed': failed,
            'metrics': metrics
        }
        
    async def _handle_timeout_scenario(self) -> Dict[str, Any]:
        """Handle scenario where entire initialization timed out."""
        logger.warning("Handling timeout scenario - returning partial results")
        return await self._compile_results()
        
    async def _handle_error_scenario(self, error: Exception) -> Dict[str, Any]:
        """Handle scenario where initialization failed with exception."""
        logger.error(f"Handling error scenario: {error}")
        return await self._compile_results()
        
    def _calculate_metrics(self, results: Dict[ComponentType, ComponentResult]) -> InitializationMetrics:
        """Calculate initialization metrics."""
        if not self._initialization_start or not self._initialization_end:
            total_duration = 0.0
        else:
            total_duration = (self._initialization_end - self._initialization_start).total_seconds()
            
        successful = sum(1 for r in results.values() if r.status == ComponentStatus.INITIALIZED)
        failed = sum(1 for r in results.values() if r.status == ComponentStatus.FAILED)
        timeout = sum(1 for r in results.values() if r.status == ComponentStatus.TIMEOUT)
        
        # Calculate critical path duration (longest component initialization)
        critical_path = 0.0
        for result in results.values():
            if result.duration_seconds:
                critical_path = max(critical_path, result.duration_seconds)
                
        return InitializationMetrics(
            total_duration_seconds=total_duration,
            components_attempted=len(results),
            components_successful=successful,
            components_failed=failed,
            components_timeout=timeout,
            parallel_operations=successful + failed + timeout,  # Approximation
            sequential_operations=0,  # Approximation
            critical_path_duration=critical_path
        )
        
    async def get_component(self, component_type: ComponentType) -> Optional[Any]:
        """Get a specific initialized component."""
        async with self._lock:
            result = self._results.get(component_type)
            if result and result.status == ComponentStatus.INITIALIZED:
                return result.component
        return None
        
    async def get_initialization_status(self) -> Dict[str, Any]:
        """Get current initialization status."""
        async with self._lock:
            status = {}
            for component_type, result in self._results.items():
                status[component_type.value] = {
                    'status': result.status.value,
                    'duration': result.duration_seconds,
                    'timeout_occurred': result.timeout_occurred,
                    'error': str(result.error) if result.error else None
                }
        return status


# Global component initializer instance
_global_initializer: Optional[ComponentInitializer] = None


def get_global_initializer() -> ComponentInitializer:
    """Get the global component initializer instance."""
    global _global_initializer
    if _global_initializer is None:
        _global_initializer = ComponentInitializer()
    return _global_initializer


async def initialize_tui_components(
    timeout_seconds: float = 5.0,
    mode: InitializationMode = InitializationMode.ADAPTIVE
) -> Dict[str, Any]:
    """
    Convenience function to initialize TUI components with default settings.
    
    Args:
        timeout_seconds: Total timeout for initialization (default 5s)
        mode: Initialization mode
        
    Returns:
        Dict with 'components', 'failed', and 'metrics' keys
    """
    initializer = get_global_initializer()
    return await initializer.initialize_components(
        mode=mode,
        total_timeout_seconds=timeout_seconds
    )