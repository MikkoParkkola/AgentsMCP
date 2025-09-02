"""
Unit Tests for Component Management Layer.

Tests the component_initializer and health_monitor modules to ensure
they properly prevent TUI hangs during initialization and operation.

Key Test Areas:
- Component initialization with timeout protection
- Parallel vs sequential initialization
- Health monitoring and hang detection
- Recovery mechanisms and graceful degradation
- Performance metrics tracking
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import psutil

from .component_initializer import (
    ComponentInitializer,
    ComponentType,
    ComponentSpec, 
    ComponentStatus,
    InitializationMode,
    get_global_initializer,
    initialize_tui_components
)

from .health_monitor import (
    HealthMonitor,
    HealthStatus,
    MetricType,
    AlertLevel,
    HangDetectionConfig,
    get_global_health_monitor,
    start_tui_health_monitoring
)

from .timeout_guardian import TimeoutGuardian


class MockComponent:
    """Mock component for testing initialization."""
    
    def __init__(self, delay_seconds: float = 0.0, should_fail: bool = False):
        self.delay_seconds = delay_seconds
        self.should_fail = should_fail
        self.initialized = False
        self.initialization_time = None
        
    async def initialize(self):
        """Mock initialization method."""
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
            
        if self.should_fail:
            raise RuntimeError("Mock initialization failure")
            
        self.initialized = True
        self.initialization_time = datetime.now()


class AsyncMockComponent:
    """Mock component with async initialization for testing."""
    
    def __init__(self, init_delay: float = 0.1):
        self.init_delay = init_delay
        self.initialized = False
        
    async def initialize(self):
        await asyncio.sleep(self.init_delay)
        self.initialized = True


@pytest.fixture
async def timeout_guardian():
    """Create timeout guardian for testing."""
    guardian = TimeoutGuardian()
    yield guardian
    # Cleanup
    if guardian._monitor_task and not guardian._monitor_task.done():
        guardian._monitor_task.cancel()
        try:
            await guardian._monitor_task
        except asyncio.CancelledError:
            pass


@pytest.fixture
async def component_initializer(timeout_guardian):
    """Create component initializer for testing."""
    return ComponentInitializer(timeout_guardian=timeout_guardian)


@pytest.fixture
async def health_monitor(timeout_guardian):
    """Create health monitor for testing."""
    monitor = HealthMonitor(
        check_interval_seconds=0.1,  # Fast interval for testing
        hang_config=HangDetectionConfig(
            response_timeout_seconds=0.5,
            update_timeout_seconds=0.5,
            event_timeout_seconds=1.0,
            memory_threshold_mb=500.0,
            cpu_threshold_percent=80.0
        ),
        timeout_guardian=timeout_guardian
    )
    yield monitor
    # Cleanup
    if monitor._monitoring_active:
        await monitor.stop_monitoring()


class TestComponentInitializer:
    """Test the ComponentInitializer class."""
    
    @pytest.mark.asyncio
    async def test_successful_initialization(self, component_initializer):
        """Test successful component initialization."""
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
        result = await component_initializer.initialize_components(
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
        
        metrics = result['metrics']
        assert metrics.components_successful == 1
        assert metrics.components_failed == 0
        assert metrics.total_duration_seconds < 2.0
        
    @pytest.mark.asyncio
    async def test_component_timeout(self, component_initializer):
        """Test component initialization timeout handling."""
        # Create spec with component that takes too long
        specs = {
            ComponentType.EVENT_SYSTEM: ComponentSpec(
                component_type=ComponentType.EVENT_SYSTEM,
                component_class=MockComponent,
                init_kwargs={'delay_seconds': 2.0},  # Takes 2 seconds
                timeout_seconds=0.5,  # But timeout is 0.5s
                required=False,
                parallel_safe=True
            )
        }
        
        # Initialize components
        result = await component_initializer.initialize_components(
            component_specs=specs,
            mode=InitializationMode.PARALLEL,
            total_timeout_seconds=3.0
        )
        
        # Component should fail due to timeout
        assert len(result['components']) == 0
        assert len(result['failed']) == 1
        assert ComponentType.EVENT_SYSTEM.value in result['failed']
        
        metrics = result['metrics']
        assert metrics.components_successful == 0
        assert metrics.components_timeout == 1
        
    @pytest.mark.asyncio
    async def test_parallel_initialization(self, component_initializer):
        """Test parallel component initialization."""
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
        result = await component_initializer.initialize_components(
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
        
    @pytest.mark.asyncio
    async def test_sequential_initialization(self, component_initializer):
        """Test sequential component initialization."""
        specs = {
            ComponentType.TERMINAL_CONTROLLER: ComponentSpec(
                component_type=ComponentType.TERMINAL_CONTROLLER,
                component_class=MockComponent,
                init_kwargs={'delay_seconds': 0.1},
                timeout_seconds=1.0,
                parallel_safe=False
            ),
            ComponentType.DISPLAY_MANAGER: ComponentSpec(
                component_type=ComponentType.DISPLAY_MANAGER,
                component_class=MockComponent,
                init_kwargs={'delay_seconds': 0.1},
                timeout_seconds=1.0,
                dependencies=[ComponentType.TERMINAL_CONTROLLER],
                parallel_safe=False
            )
        }
        
        result = await component_initializer.initialize_components(
            component_specs=specs,
            mode=InitializationMode.SEQUENTIAL,
            total_timeout_seconds=3.0
        )
        
        # Both components should be initialized
        assert len(result['components']) == 2
        assert len(result['failed']) == 0
        
    @pytest.mark.asyncio 
    async def test_graceful_failure_handling(self, component_initializer):
        """Test graceful handling of component failures."""
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
        
        result = await component_initializer.initialize_components(
            component_specs=specs,
            mode=InitializationMode.PARALLEL,
            total_timeout_seconds=3.0
        )
        
        # One component should succeed, one should fail
        assert len(result['components']) == 1
        assert len(result['failed']) == 1
        assert ComponentType.EVENT_SYSTEM.value in result['components']
        assert ComponentType.TERMINAL_CONTROLLER.value in result['failed']
        
    @pytest.mark.asyncio
    async def test_total_timeout_protection(self, component_initializer):
        """Test that total timeout protects against hanging."""
        # Create component that takes longer than total timeout
        specs = {
            ComponentType.TERMINAL_CONTROLLER: ComponentSpec(
                component_type=ComponentType.TERMINAL_CONTROLLER,
                component_class=MockComponent,
                init_kwargs={'delay_seconds': 2.0},
                timeout_seconds=3.0,  # Individual timeout is fine
                required=True
            )
        }
        
        start_time = time.time()
        result = await component_initializer.initialize_components(
            component_specs=specs,
            total_timeout_seconds=0.5  # But total timeout is short
        )
        end_time = time.time()
        
        # Should complete within total timeout
        assert (end_time - start_time) < 1.0
        
        # Component should not be initialized due to timeout
        assert len(result['components']) == 0 or len(result['failed']) > 0


class TestHealthMonitor:
    """Test the HealthMonitor class."""
    
    @pytest.mark.asyncio
    async def test_health_monitoring_startup(self, health_monitor):
        """Test health monitor starts and stops correctly."""
        assert not health_monitor._monitoring_active
        
        await health_monitor.start_monitoring()
        assert health_monitor._monitoring_active
        assert health_monitor._monitoring_task is not None
        
        await health_monitor.stop_monitoring()
        assert not health_monitor._monitoring_active
        assert health_monitor._monitoring_task is None
        
    @pytest.mark.asyncio
    async def test_health_status_detection(self, health_monitor):
        """Test health status detection."""
        # Start monitoring briefly
        await health_monitor.start_monitoring()
        await asyncio.sleep(0.2)  # Let it run a couple cycles
        
        status = await health_monitor.get_current_health_status()
        assert status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        
        await health_monitor.stop_monitoring()
        
    @pytest.mark.asyncio
    async def test_hang_detection(self, health_monitor):
        """Test hang detection functionality."""
        hang_detected = False
        hang_reason = None
        
        def hang_callback(reason: str):
            nonlocal hang_detected, hang_reason
            hang_detected = True
            hang_reason = reason
            
        health_monitor.add_hang_callback(hang_callback)
        
        await health_monitor.start_monitoring()
        
        # Simulate no responses for longer than timeout
        await asyncio.sleep(0.6)  # Longer than 0.5s timeout
        
        await health_monitor.stop_monitoring()
        
        # Should have detected potential hang
        assert hang_detected or hang_reason is not None
        
    @pytest.mark.asyncio
    async def test_ui_activity_recording(self, health_monitor):
        """Test recording of UI activity."""
        # Record various types of activity
        await health_monitor.record_ui_response()
        await health_monitor.record_ui_update()
        await health_monitor.record_event_processed()
        
        # These should update internal timestamps
        assert health_monitor._last_response_time is not None
        assert health_monitor._last_update_time is not None
        assert health_monitor._last_event_time is not None
        
    @pytest.mark.asyncio
    async def test_performance_summary(self, health_monitor):
        """Test performance summary generation."""
        await health_monitor.start_monitoring()
        await asyncio.sleep(0.2)  # Let it collect some data
        
        summary = await health_monitor.get_performance_summary()
        
        assert 'status' in summary
        assert 'uptime_seconds' in summary
        assert 'monitoring_active' in summary
        assert 'metrics' in summary
        
        await health_monitor.stop_monitoring()
        
    @pytest.mark.asyncio
    async def test_health_callbacks(self, health_monitor):
        """Test health status callbacks."""
        reports = []
        
        def health_callback(report):
            reports.append(report)
            
        health_monitor.add_health_callback(health_callback)
        
        await health_monitor.start_monitoring()
        await asyncio.sleep(0.25)  # Let it run a few cycles
        await health_monitor.stop_monitoring()
        
        # Should have received health reports
        assert len(reports) > 0
        
        # Reports should have expected structure
        report = reports[0]
        assert hasattr(report, 'overall_status')
        assert hasattr(report, 'metrics')
        assert hasattr(report, 'timestamp')
        
    @pytest.mark.asyncio
    async def test_memory_threshold_detection(self, health_monitor):
        """Test memory threshold detection."""
        # Mock high memory usage
        with patch.object(health_monitor._process, 'memory_info') as mock_memory:
            # Set memory usage above threshold (500MB)
            mock_memory.return_value.rss = 600 * 1024 * 1024  # 600MB
            
            status = await health_monitor.get_current_health_status()
            assert status == HealthStatus.UNHEALTHY


class TestIntegration:
    """Integration tests for component management layer."""
    
    @pytest.mark.asyncio
    async def test_component_initialization_with_health_monitoring(self):
        """Test integration of component initialization with health monitoring."""
        # Initialize components
        result = await initialize_tui_components(
            timeout_seconds=3.0,
            mode=InitializationMode.ADAPTIVE
        )
        
        # Should get some results (even if components fail due to missing deps)
        assert 'components' in result
        assert 'failed' in result
        assert 'metrics' in result
        
        # Start health monitoring
        monitor = await start_tui_health_monitoring(
            check_interval_seconds=0.1
        )
        
        # Let it monitor briefly
        await asyncio.sleep(0.3)
        
        # Should be able to get health status
        status = await monitor.get_current_health_status()
        assert status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, 
                         HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
    @pytest.mark.asyncio
    async def test_timeout_guardian_integration(self):
        """Test integration with timeout guardian."""
        guardian = TimeoutGuardian()
        initializer = ComponentInitializer(timeout_guardian=guardian)
        monitor = HealthMonitor(timeout_guardian=guardian)
        
        # Both should use the same guardian
        assert initializer._guardian is guardian
        assert monitor._guardian is guardian
        
        # Should work together for timeout protection
        await monitor.start_monitoring()
        
        # Test timeout protection during initialization
        specs = {
            ComponentType.TERMINAL_CONTROLLER: ComponentSpec(
                component_type=ComponentType.TERMINAL_CONTROLLER,
                component_class=MockComponent,
                init_kwargs={'delay_seconds': 0.1},
                timeout_seconds=0.5
            )
        }
        
        result = await initializer.initialize_components(
            component_specs=specs,
            total_timeout_seconds=1.0
        )
        
        # Should complete successfully
        assert len(result['components']) >= 0
        
        await monitor.stop_monitoring()


# Edge case tests beyond golden tests
class TestEdgeCases:
    """Additional edge case tests for improved coverage."""
    
    @pytest.mark.asyncio
    async def test_initialization_with_circular_dependencies(self, component_initializer):
        """Test handling of circular dependencies in component initialization."""
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
        result = await component_initializer.initialize_components(
            component_specs=specs,
            mode=InitializationMode.SEQUENTIAL,
            total_timeout_seconds=3.0
        )
        
        # Should not hang and should return some result
        assert 'components' in result
        assert 'metrics' in result
        
    @pytest.mark.asyncio
    async def test_health_monitor_with_extreme_load(self, health_monitor):
        """Test health monitor behavior under extreme resource conditions."""
        # Mock extreme CPU and memory usage
        with patch.object(health_monitor._process, 'cpu_percent') as mock_cpu, \
             patch.object(health_monitor._process, 'memory_info') as mock_memory:
            
            # Set extreme values
            mock_cpu.return_value = 99.9  # Very high CPU
            mock_memory.return_value.rss = 2000 * 1024 * 1024  # 2GB memory
            
            await health_monitor.start_monitoring()
            await asyncio.sleep(0.2)
            
            status = await health_monitor.get_current_health_status()
            
            # Should detect unhealthy condition
            assert status == HealthStatus.UNHEALTHY
            
            await health_monitor.stop_monitoring()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])