"""
Test suite for Unified TUI Coordinator integration module.

Tests the unified coordination functionality, mode switching,
component integration, and performance requirements.
"""

import asyncio
import pytest
import time
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Mock dependencies before importing
class MockTerminalController:
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        self.initialized = True
        return True
    
    async def enter_alternate_screen(self, mode):
        pass
    
    async def exit_alternate_screen(self):
        pass
    
    async def cleanup(self):
        pass

class MockLoggingIsolationManager:
    def __init__(self):
        self.initialized = False
        self.isolation_active = False
    
    async def initialize(self):
        self.initialized = True
        return True
    
    async def activate_isolation(self, tui_active=True):
        self.isolation_active = True
    
    async def deactivate_isolation(self):
        self.isolation_active = False
    
    async def cleanup(self):
        pass

class MockDisplayManager:
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        self.initialized = True
        return True
    
    async def cleanup(self):
        pass

class MockTextLayoutEngine:
    pass

class MockInputRenderingPipeline:
    def configure(self, **kwargs):
        pass

class MockRevolutionaryTUI:
    def __init__(self, **kwargs):
        self.started = False
        self.healthy = True
    
    async def start(self):
        self.started = True
    
    async def stop(self):
        self.started = False
    
    def is_healthy(self):
        return self.healthy

class MockBasicTUI:
    def __init__(self):
        self.started = False
        self.healthy = True
    
    def start(self):  # Sync start
        self.started = True
    
    def stop(self):
        self.started = False
    
    def is_healthy(self):
        return self.healthy

class MockOrchestratorIntegration:
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        self.initialized = True

# Mock imports
mock_modules = {
    'src.agentsmcp.ui.v2.terminal_controller': Mock(
        get_terminal_controller=AsyncMock(return_value=MockTerminalController()),
        TerminalController=MockTerminalController,
        AlternateScreenMode=Mock(AUTO='auto', DISABLED='disabled'),
        CursorVisibility=Mock(AUTO='auto')
    ),
    'src.agentsmcp.ui.v2.logging_isolation_manager': Mock(
        get_logging_isolation_manager=AsyncMock(return_value=MockLoggingIsolationManager()),
        LoggingIsolationManager=MockLoggingIsolationManager
    ),
    'src.agentsmcp.ui.v2.display_manager': Mock(
        get_display_manager=AsyncMock(return_value=MockDisplayManager()),
        DisplayManager=MockDisplayManager,
        RefreshMode=Mock(ADAPTIVE='adaptive'),
        ContentUpdate=Mock
    ),
    'src.agentsmcp.ui.v2.text_layout_engine': Mock(
        TextLayoutEngine=MockTextLayoutEngine,
        WrapMode=Mock(SMART='smart'),
        OverflowHandling=Mock(ADAPTIVE='adaptive')
    ),
    'src.agentsmcp.ui.v2.input_rendering_pipeline': Mock(
        InputRenderingPipeline=MockInputRenderingPipeline,
        InputMode=Mock(SINGLE_LINE='single_line')
    ),
    'src.agentsmcp.ui.v2.orchestrator_integration': Mock(
        OrchestratorTUIIntegration=MockOrchestratorIntegration,
        OrchestratorIntegrationConfig=Mock
    ),
    'src.agentsmcp.ui.v2.revolutionary_tui_interface': Mock(
        RevolutionaryTUIInterface=MockRevolutionaryTUI
    ),
    'src.agentsmcp.ui.v2.fixed_working_tui': Mock(
        FixedWorkingTUI=MockBasicTUI
    )
}

with patch.dict('sys.modules', mock_modules):
    from src.agentsmcp.ui.v2.unified_tui_coordinator import (
        UnifiedTUICoordinator, get_unified_tui_coordinator, cleanup_unified_tui_coordinator,
        TUIMode, TUIStatus, ComponentConfig, IntegrationStatus, TUIInstance,
        start_revolutionary_tui, start_basic_tui, switch_tui_mode, stop_tui
    )


class TestTUIMode:
    """Test TUI mode enumeration."""
    
    def test_tui_modes(self):
        """Test TUI mode values."""
        assert TUIMode.REVOLUTIONARY.value == "revolutionary"
        assert TUIMode.BASIC.value == "basic"
        assert TUIMode.FALLBACK.value == "fallback"
        assert TUIMode.DISABLED.value == "disabled"


class TestTUIStatus:
    """Test TUI status enumeration."""
    
    def test_tui_status(self):
        """Test TUI status values."""
        assert TUIStatus.INACTIVE.value == "inactive"
        assert TUIStatus.INITIALIZING.value == "initializing"
        assert TUIStatus.ACTIVE.value == "active"
        assert TUIStatus.SWITCHING_MODE.value == "switching_mode"
        assert TUIStatus.ERROR.value == "error"
        assert TUIStatus.SHUTTING_DOWN.value == "shutting_down"


class TestComponentConfig:
    """Test component configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ComponentConfig()
        
        assert config.enable_animations == True
        assert config.enable_rich_rendering == True
        assert config.max_fps == 60
        assert config.enable_logging_isolation == True
        assert config.enable_alternate_screen == True
        assert config.terminal_size_detection == True
        assert config.performance_monitoring == True
        assert config.error_recovery == True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ComponentConfig(
            enable_animations=False,
            max_fps=30,
            enable_logging_isolation=False
        )
        
        assert config.enable_animations == False
        assert config.max_fps == 30
        assert config.enable_logging_isolation == False


class TestUnifiedTUICoordinator:
    """Test unified TUI coordinator functionality."""
    
    @pytest.fixture
    async def coordinator(self):
        """Create a coordinator instance for testing."""
        coordinator = UnifiedTUICoordinator()
        success = await coordinator.initialize()
        assert success
        
        yield coordinator
        
        await coordinator.shutdown()
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test coordinator initialization."""
        coordinator = UnifiedTUICoordinator()
        
        # Should start uninitialized
        assert not coordinator._initialized
        assert coordinator._status == TUIStatus.INACTIVE
        assert coordinator._current_mode == TUIMode.DISABLED
        
        # Initialize
        success = await coordinator.initialize()
        assert success
        assert coordinator._initialized
        assert coordinator._status == TUIStatus.INACTIVE
        
        # Verify infrastructure is initialized
        assert coordinator._terminal_controller is not None
        assert coordinator._logging_manager is not None
        assert coordinator._display_manager is not None
        assert coordinator._text_layout_engine is not None
        assert coordinator._input_pipeline is not None
        
        await coordinator.shutdown()
    
    @pytest.mark.asyncio
    async def test_start_revolutionary_tui(self, coordinator):
        """Test starting revolutionary TUI mode."""
        tui_instance, mode_active, status = await coordinator.start_tui(TUIMode.REVOLUTIONARY)
        
        assert tui_instance is not None
        assert mode_active == True
        assert coordinator.get_current_mode() == TUIMode.REVOLUTIONARY
        assert coordinator.get_status() == TUIStatus.ACTIVE
        assert 'startup_time_seconds' in status
        
        # Check performance requirement (startup within 2 seconds)
        assert status['startup_time_seconds'] < 2.0
    
    @pytest.mark.asyncio
    async def test_start_basic_tui(self, coordinator):
        """Test starting basic TUI mode."""
        tui_instance, mode_active, status = await coordinator.start_tui(TUIMode.BASIC)
        
        assert tui_instance is not None
        assert mode_active == True
        assert coordinator.get_current_mode() == TUIMode.BASIC
        assert coordinator.get_status() == TUIStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_start_fallback_tui(self, coordinator):
        """Test starting fallback TUI mode."""
        tui_instance, mode_active, status = await coordinator.start_tui(TUIMode.FALLBACK)
        
        assert tui_instance is not None
        assert mode_active == True
        assert coordinator.get_current_mode() == TUIMode.FALLBACK
        assert coordinator.get_status() == TUIStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_mode_switching(self, coordinator):
        """Test switching between TUI modes."""
        # Start in revolutionary mode
        await coordinator.start_tui(TUIMode.REVOLUTIONARY)
        assert coordinator.get_current_mode() == TUIMode.REVOLUTIONARY
        
        # Switch to basic mode
        success, metrics = await coordinator.switch_mode(TUIMode.BASIC)
        assert success
        assert coordinator.get_current_mode() == TUIMode.BASIC
        assert 'switch_time_seconds' in metrics
        
        # Check performance requirement (mode switching within 500ms)
        assert metrics['switch_time_seconds'] < 0.5
        
        # Switch to disabled
        success, metrics = await coordinator.switch_mode(TUIMode.DISABLED)
        assert success
        assert coordinator.get_current_mode() == TUIMode.DISABLED
        assert coordinator.get_status() == TUIStatus.INACTIVE
    
    @pytest.mark.asyncio
    async def test_mode_switching_same_mode(self, coordinator):
        """Test switching to same mode (should be no-op)."""
        await coordinator.start_tui(TUIMode.BASIC)
        
        success, metrics = await coordinator.switch_mode(TUIMode.BASIC)
        assert success
        assert 'message' in metrics
        assert 'Already in target mode' in metrics['message']
    
    @pytest.mark.asyncio
    async def test_orchestrator_integration(self, coordinator):
        """Test orchestrator integration setup."""
        orchestrator_integration = MockOrchestratorIntegration()
        
        tui_instance, mode_active, status = await coordinator.start_tui(
            TUIMode.REVOLUTIONARY,
            orchestrator_integration=orchestrator_integration
        )
        
        assert mode_active
        assert coordinator._orchestrator_integration is orchestrator_integration
        assert coordinator._integration_status.orchestrator_connected
    
    @pytest.mark.asyncio
    async def test_tui_session_context_manager(self, coordinator):
        """Test TUI session context manager."""
        session_entered = False
        session_exited = False
        
        async with coordinator.tui_session(TUIMode.BASIC) as tui_instance:
            session_entered = True
            assert tui_instance is not None
            assert coordinator.get_current_mode() == TUIMode.BASIC
            assert coordinator.get_status() == TUIStatus.ACTIVE
        
        session_exited = True
        assert session_entered and session_exited
        assert coordinator.get_current_mode() == TUIMode.DISABLED
        assert coordinator.get_status() == TUIStatus.INACTIVE
    
    @pytest.mark.asyncio
    async def test_callback_system(self, coordinator):
        """Test callback registration and notification."""
        mode_change_called = False
        status_change_called = False
        error_called = False
        
        def mode_change_callback(old_mode, new_mode):
            nonlocal mode_change_called
            mode_change_called = True
            assert old_mode == TUIMode.DISABLED
            assert new_mode == TUIMode.BASIC
        
        def status_change_callback(status):
            nonlocal status_change_called
            status_change_called = True
        
        def error_callback(error_type, error):
            nonlocal error_called
            error_called = True
        
        # Register callbacks
        coordinator.register_mode_change_callback(mode_change_callback)
        coordinator.register_status_change_callback(status_change_callback)
        coordinator.register_error_callback(error_callback)
        
        # Trigger mode change
        await coordinator.start_tui(TUIMode.BASIC)
        
        assert mode_change_called
        assert status_change_called
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, coordinator):
        """Test TUI health monitoring."""
        # Start TUI
        await coordinator.start_tui(TUIMode.REVOLUTIONARY)
        
        # Health monitoring should be running
        assert coordinator._health_monitor_task is not None
        assert not coordinator._health_monitor_task.done()
        
        # Current instance should be healthy
        assert coordinator._current_instance is not None
        assert coordinator._current_instance.is_healthy
        assert coordinator._current_instance.error_count == 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_simulation(self, coordinator):
        """Test error recovery mechanism."""
        # Start TUI with a mock that can simulate health issues
        await coordinator.start_tui(TUIMode.REVOLUTIONARY)
        
        # Simulate health check failure
        if coordinator._current_instance:
            coordinator._current_instance.is_healthy = False
            coordinator._current_instance.error_count = coordinator._max_error_count
        
        # Manually trigger recovery (normally done by health monitor)
        await coordinator._trigger_recovery()
        
        # Should recover or fall back
        assert coordinator.get_status() in [TUIStatus.ACTIVE, TUIStatus.ERROR]
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, coordinator):
        """Test performance metrics collection."""
        # Start TUI to generate some metrics
        await coordinator.start_tui(TUIMode.BASIC)
        await coordinator.switch_mode(TUIMode.REVOLUTIONARY)
        await coordinator.switch_mode(TUIMode.DISABLED)
        
        metrics = coordinator.get_performance_metrics()
        
        required_keys = [
            'current_mode', 'status', 'startup_times', 'average_switch_time_ms',
            'switch_count', 'initialized', 'infrastructure'
        ]
        
        for key in required_keys:
            assert key in metrics
        
        assert metrics['initialized'] == True
        assert metrics['switch_count'] > 0
        assert 'infrastructure' in metrics
        
        # Check infrastructure status
        infra = metrics['infrastructure']
        assert infra['terminal_controller'] == True
        assert infra['logging_manager'] == True
        assert infra['display_manager'] == True
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, coordinator):
        """Test graceful shutdown process."""
        # Start TUI
        await coordinator.start_tui(TUIMode.REVOLUTIONARY)
        assert coordinator.get_status() == TUIStatus.ACTIVE
        
        # Shutdown
        await coordinator.shutdown()
        assert coordinator.get_status() == TUIStatus.SHUTTING_DOWN
        assert not coordinator._initialized
        
        # Health monitor should be stopped
        if coordinator._health_monitor_task:
            assert coordinator._health_monitor_task.done()


class TestConvenienceFunctions:
    """Test convenience functions for TUI operations."""
    
    @pytest.mark.asyncio
    async def test_start_revolutionary_tui_function(self):
        """Test start_revolutionary_tui convenience function."""
        mode_active, status = await start_revolutionary_tui()
        
        assert mode_active == True
        assert 'startup_time_seconds' in status
        
        # Cleanup
        await stop_tui()
        await cleanup_unified_tui_coordinator()
    
    @pytest.mark.asyncio
    async def test_start_basic_tui_function(self):
        """Test start_basic_tui convenience function."""
        mode_active, status = await start_basic_tui()
        
        assert mode_active == True
        
        # Cleanup
        await stop_tui()
        await cleanup_unified_tui_coordinator()
    
    @pytest.mark.asyncio
    async def test_switch_tui_mode_function(self):
        """Test switch_tui_mode convenience function."""
        # Start first
        await start_basic_tui()
        
        # Switch mode
        success, metrics = await switch_tui_mode(TUIMode.REVOLUTIONARY)
        assert success
        assert 'switch_time_seconds' in metrics
        
        # Cleanup
        await stop_tui()
        await cleanup_unified_tui_coordinator()
    
    @pytest.mark.asyncio
    async def test_stop_tui_function(self):
        """Test stop_tui convenience function."""
        # Start first
        await start_basic_tui()
        
        # Stop
        success, metrics = await stop_tui()
        assert success
        
        # Cleanup
        await cleanup_unified_tui_coordinator()


class TestIntegrationScenarios:
    """Test integration scenarios with multiple components."""
    
    @pytest.mark.asyncio
    async def test_singleton_coordinator_access(self):
        """Test singleton coordinator access pattern."""
        coordinator1 = await get_unified_tui_coordinator()
        coordinator2 = await get_unified_tui_coordinator()
        
        # Should be same instance
        assert coordinator1 is coordinator2
        
        # Cleanup
        await cleanup_unified_tui_coordinator()
    
    @pytest.mark.asyncio
    async def test_multiple_mode_switches(self):
        """Test multiple rapid mode switches."""
        coordinator = await get_unified_tui_coordinator()
        
        try:
            modes = [TUIMode.BASIC, TUIMode.REVOLUTIONARY, TUIMode.FALLBACK, TUIMode.BASIC]
            
            for mode in modes:
                success, metrics = await coordinator.switch_mode(mode)
                assert success
                assert coordinator.get_current_mode() == mode
                
                # Brief pause to allow for cleanup
                await asyncio.sleep(0.01)
            
            # All switches should complete successfully
            final_metrics = coordinator.get_performance_metrics()
            assert final_metrics['switch_count'] == len(modes)
            
        finally:
            await cleanup_unified_tui_coordinator()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test handling of concurrent operations."""
        coordinator = await get_unified_tui_coordinator()
        
        try:
            # Start TUI
            await coordinator.start_tui(TUIMode.BASIC)
            
            # Try concurrent mode switches (should handle gracefully)
            tasks = []
            for i in range(3):
                mode = TUIMode.REVOLUTIONARY if i % 2 == 0 else TUIMode.BASIC
                task = asyncio.create_task(coordinator.switch_mode(mode))
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # At least one should succeed, others may be blocked or fail gracefully
            successful_switches = sum(1 for result in results if isinstance(result, tuple) and result[0])
            assert successful_switches >= 1
            
            # Final state should be stable
            assert coordinator.get_status() in [TUIStatus.ACTIVE, TUIStatus.INACTIVE]
            
        finally:
            await cleanup_unified_tui_coordinator()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        coordinator = await get_unified_tui_coordinator()
        
        try:
            # Simulate initialization failure by patching
            with patch.object(coordinator, '_create_tui_instance', return_value=None):
                tui_instance, mode_active, status = await coordinator.start_tui(TUIMode.REVOLUTIONARY)
                
                assert tui_instance is None
                assert mode_active == False
                assert 'error' in status
            
            # Should still be able to start normally after error
            tui_instance, mode_active, status = await coordinator.start_tui(TUIMode.FALLBACK)
            assert mode_active == True
            assert coordinator.get_current_mode() == TUIMode.FALLBACK
            
        finally:
            await cleanup_unified_tui_coordinator()


class TestPerformanceRequirements:
    """Test ICD performance requirements."""
    
    @pytest.mark.asyncio
    async def test_startup_time_requirement(self):
        """Test TUI startup within 2 seconds requirement."""
        coordinator = UnifiedTUICoordinator()
        await coordinator.initialize()
        
        try:
            start_time = time.time()
            tui_instance, mode_active, status = await coordinator.start_tui(TUIMode.REVOLUTIONARY)
            startup_time = time.time() - start_time
            
            assert mode_active
            assert startup_time < 2.0  # ICD requirement
            assert status['startup_time_seconds'] < 2.0
            
        finally:
            await coordinator.shutdown()
    
    @pytest.mark.asyncio
    async def test_mode_switching_time_requirement(self):
        """Test mode switching within 500ms requirement."""
        coordinator = UnifiedTUICoordinator()
        await coordinator.initialize()
        
        try:
            # Start first
            await coordinator.start_tui(TUIMode.BASIC)
            
            # Test switch time
            start_time = time.time()
            success, metrics = await coordinator.switch_mode(TUIMode.REVOLUTIONARY)
            switch_time = time.time() - start_time
            
            assert success
            assert switch_time < 0.5  # ICD requirement
            assert metrics['switch_time_seconds'] < 0.5
            
        finally:
            await coordinator.shutdown()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])