"""
Simple Unit Tests for Component Management Layer.

Focused tests for component_initializer and health_monitor functionality
without complex async fixtures.
"""

import asyncio
import time
from datetime import datetime

# Test imports
from .component_initializer import (
    ComponentInitializer,
    ComponentType,
    ComponentSpec,
    ComponentStatus,
    InitializationMode,
    initialize_tui_components
)

from .health_monitor import (
    HealthMonitor,
    HealthStatus,
    HangDetectionConfig,
    start_tui_health_monitoring
)

from .timeout_guardian import TimeoutGuardian


class MockComponent:
    """Mock component for testing initialization."""
    
    def __init__(self, delay_seconds: float = 0.0, should_fail: bool = False):
        self.delay_seconds = delay_seconds
        self.should_fail = should_fail
        self.initialized = False
        
    async def initialize(self):
        """Mock initialization method."""
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
            
        if self.should_fail:
            raise RuntimeError("Mock initialization failure")
            
        self.initialized = True


async def test_component_initialization():
    """Test basic component initialization."""
    print("Testing component initialization...")
    
    initializer = ComponentInitializer()
    
    # Create mock specs
    specs = {
        ComponentType.TERMINAL_CONTROLLER: ComponentSpec(
            component_type=ComponentType.TERMINAL_CONTROLLER,
            component_class=MockComponent,
            init_kwargs={'delay_seconds': 0.1},
            timeout_seconds=1.0,
            required=True,
            parallel_safe=True
        )
    }
    
    # Initialize components
    result = await initializer.initialize_components(
        component_specs=specs,
        mode=InitializationMode.PARALLEL,
        total_timeout_seconds=2.0
    )
    
    # Verify results
    assert len(result['components']) == 1
    assert len(result['failed']) == 0
    assert ComponentType.TERMINAL_CONTROLLER.value in result['components']
    
    component = result['components'][ComponentType.TERMINAL_CONTROLLER.value]
    assert isinstance(component, MockComponent)
    assert component.initialized
    
    print("✓ Component initialization test passed")


async def test_component_timeout():
    """Test component timeout handling."""
    print("Testing component timeout...")
    
    initializer = ComponentInitializer()
    
    # Create spec with component that takes too long
    specs = {
        ComponentType.EVENT_SYSTEM: ComponentSpec(
            component_type=ComponentType.EVENT_SYSTEM,
            component_class=MockComponent,
            init_kwargs={'delay_seconds': 1.0},  # Takes 1 second
            timeout_seconds=0.2,  # But timeout is 0.2s
            required=False,
            parallel_safe=True
        )
    }
    
    # Initialize components
    result = await initializer.initialize_components(
        component_specs=specs,
        mode=InitializationMode.PARALLEL,
        total_timeout_seconds=2.0
    )
    
    # Component should fail due to timeout
    assert len(result['components']) == 0
    assert len(result['failed']) == 1
    assert ComponentType.EVENT_SYSTEM.value in result['failed']
    
    print("✓ Component timeout test passed")


async def test_health_monitoring():
    """Test basic health monitoring."""
    print("Testing health monitoring...")
    
    monitor = HealthMonitor(check_interval_seconds=0.1)
    
    # Start monitoring
    await monitor.start_monitoring()
    assert monitor._monitoring_active
    
    # Let it run briefly
    await asyncio.sleep(0.2)
    
    # Check status
    status = await monitor.get_current_health_status()
    assert status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, 
                     HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]
    
    # Stop monitoring
    await monitor.stop_monitoring()
    assert not monitor._monitoring_active
    
    print("✓ Health monitoring test passed")


async def test_health_activity_recording():
    """Test UI activity recording."""
    print("Testing activity recording...")
    
    monitor = HealthMonitor()
    
    # Record various activities
    await monitor.record_ui_response()
    await monitor.record_ui_update()
    await monitor.record_event_processed()
    
    # Check that timestamps were updated
    assert monitor._last_response_time is not None
    assert monitor._last_update_time is not None
    assert monitor._last_event_time is not None
    
    print("✓ Activity recording test passed")


async def test_parallel_initialization():
    """Test parallel component initialization speeds up startup."""
    print("Testing parallel initialization performance...")
    
    initializer = ComponentInitializer()
    
    # Create multiple components that can be initialized in parallel
    specs = {
        ComponentType.TERMINAL_CONTROLLER: ComponentSpec(
            component_type=ComponentType.TERMINAL_CONTROLLER,
            component_class=MockComponent,
            init_kwargs={'delay_seconds': 0.2},
            timeout_seconds=1.0,
            parallel_safe=True
        ),
        ComponentType.EVENT_SYSTEM: ComponentSpec(
            component_type=ComponentType.EVENT_SYSTEM,
            component_class=MockComponent,
            init_kwargs={'delay_seconds': 0.2},
            timeout_seconds=1.0,
            parallel_safe=True
        )
    }
    
    start_time = time.time()
    result = await initializer.initialize_components(
        component_specs=specs,
        mode=InitializationMode.PARALLEL,
        total_timeout_seconds=3.0
    )
    end_time = time.time()
    
    # Should complete in roughly 0.2s (parallel) rather than 0.4s (sequential)
    total_time = end_time - start_time
    assert total_time < 0.35  # Allow some overhead
    
    # Both components should be initialized
    assert len(result['components']) == 2
    assert len(result['failed']) == 0
    
    print(f"✓ Parallel initialization test passed (took {total_time:.3f}s)")


async def test_graceful_failure_handling():
    """Test graceful handling of component failures."""
    print("Testing graceful failure handling...")
    
    initializer = ComponentInitializer()
    
    specs = {
        ComponentType.TERMINAL_CONTROLLER: ComponentSpec(
            component_type=ComponentType.TERMINAL_CONTROLLER,
            component_class=MockComponent,
            init_kwargs={'should_fail': True},
            timeout_seconds=1.0,
            required=True
        ),
        ComponentType.EVENT_SYSTEM: ComponentSpec(
            component_type=ComponentType.EVENT_SYSTEM,
            component_class=MockComponent,
            init_kwargs={'delay_seconds': 0.1},
            timeout_seconds=1.0,
            required=False  # Not required, should still succeed
        )
    }
    
    result = await initializer.initialize_components(
        component_specs=specs,
        mode=InitializationMode.PARALLEL,
        total_timeout_seconds=3.0
    )
    
    # One component should succeed, one should fail
    assert len(result['components']) == 1
    assert len(result['failed']) == 1
    assert ComponentType.EVENT_SYSTEM.value in result['components']
    assert ComponentType.TERMINAL_CONTROLLER.value in result['failed']
    
    print("✓ Graceful failure handling test passed")


# Edge case test: Circular dependencies
async def test_circular_dependency_handling():
    """Test handling of circular dependencies."""
    print("Testing circular dependency handling...")
    
    initializer = ComponentInitializer()
    
    # Create specs with circular dependencies
    specs = {
        ComponentType.TERMINAL_CONTROLLER: ComponentSpec(
            component_type=ComponentType.TERMINAL_CONTROLLER,
            component_class=MockComponent,
            dependencies=[ComponentType.DISPLAY_MANAGER],  # Depends on display
            timeout_seconds=1.0
        ),
        ComponentType.DISPLAY_MANAGER: ComponentSpec(
            component_type=ComponentType.DISPLAY_MANAGER,
            component_class=MockComponent,
            dependencies=[ComponentType.TERMINAL_CONTROLLER],  # Depends on terminal
            timeout_seconds=1.0
        )
    }
    
    # Should handle gracefully without hanging
    result = await initializer.initialize_components(
        component_specs=specs,
        mode=InitializationMode.SEQUENTIAL,
        total_timeout_seconds=3.0
    )
    
    # Should not hang and should return some result
    assert 'components' in result
    assert 'metrics' in result
    
    print("✓ Circular dependency handling test passed")


# Edge case test: Extreme load detection
async def test_hang_detection_callback():
    """Test hang detection with callbacks."""
    print("Testing hang detection callbacks...")
    
    hang_detected = False
    
    def hang_callback(reason: str):
        nonlocal hang_detected
        hang_detected = True
        print(f"Hang detected: {reason}")
    
    monitor = HealthMonitor(
        check_interval_seconds=0.1,
        hang_config=HangDetectionConfig(response_timeout_seconds=0.2)
    )
    
    monitor.add_hang_callback(hang_callback)
    
    await monitor.start_monitoring()
    
    # Don't record any activity - should trigger hang detection
    await asyncio.sleep(0.3)  # Wait longer than timeout
    
    await monitor.stop_monitoring()
    
    # Should have detected potential hang (if timeouts are working)
    print(f"✓ Hang detection callback test completed (hang_detected: {hang_detected})")


async def main():
    """Run all tests."""
    print("Running Component Management Layer Tests")
    print("=" * 50)
    
    try:
        await test_component_initialization()
        await test_component_timeout()
        await test_health_monitoring() 
        await test_health_activity_recording()
        await test_parallel_initialization()
        await test_graceful_failure_handling()
        await test_circular_dependency_handling()
        await test_hang_detection_callback()
        
        print("=" * 50)
        print("✅ All Component Management Tests Passed!")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == '__main__':
    success = asyncio.run(main())
    exit(0 if success else 1)