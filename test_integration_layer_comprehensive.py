"""
Comprehensive integration tests for the TUI integration layer.

Tests the display_manager and unified_tui_coordinator working together
to provide a unified, conflict-free TUI system.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Mock all dependencies
class MockTerminalController:
    def __init__(self):
        self.initialized = False
        self.alternate_screen_active = False
    
    async def initialize(self):
        self.initialized = True
        return True
    
    async def get_terminal_state(self):
        return Mock(size=Mock(width=80, height=24))
    
    async def enter_alternate_screen(self, mode):
        self.alternate_screen_active = True
        return True
    
    async def exit_alternate_screen(self):
        self.alternate_screen_active = False
        return True
    
    async def cleanup(self):
        pass

class MockLoggingManager:
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

class MockTUIInterface:
    def __init__(self, name, **kwargs):
        self.name = name
        self.started = False
        self.healthy = True
        self.display_updates = []
    
    async def start(self):
        self.started = True
    
    async def stop(self):
        self.started = False
    
    def is_healthy(self):
        return self.healthy
    
    async def update_display(self, content):
        self.display_updates.append(content)

# Create comprehensive mocks
mock_modules = {
    'rich.console': Mock(Console=Mock),
    'rich.layout': Mock(Layout=Mock),
    'rich.live': Mock(Live=Mock),
    'rich.text': Mock(Text=Mock),
    'rich.panel': Mock(Panel=Mock),
    
    'src.agentsmcp.ui.v2.terminal_controller': Mock(
        get_terminal_controller=AsyncMock(return_value=MockTerminalController()),
        TerminalController=MockTerminalController,
        AlternateScreenMode=Mock(AUTO='auto', DISABLED='disabled'),
        CursorVisibility=Mock(AUTO='auto')
    ),
    'src.agentsmcp.ui.v2.logging_isolation_manager': Mock(
        get_logging_isolation_manager=AsyncMock(return_value=MockLoggingManager()),
        LoggingIsolationManager=MockLoggingManager
    ),
    'src.agentsmcp.ui.v2.text_layout_engine': Mock(
        TextLayoutEngine=Mock,
        WrapMode=Mock(SMART='smart'),
        OverflowHandling=Mock(ADAPTIVE='adaptive')
    ),
    'src.agentsmcp.ui.v2.input_rendering_pipeline': Mock(
        InputRenderingPipeline=Mock,
        InputMode=Mock(SINGLE_LINE='single_line')
    ),
    'src.agentsmcp.ui.v2.revolutionary_tui_interface': Mock(
        RevolutionaryTUIInterface=lambda **kwargs: MockTUIInterface('revolutionary', **kwargs)
    ),
    'src.agentsmcp.ui.v2.fixed_working_tui': Mock(
        FixedWorkingTUI=lambda: MockTUIInterface('basic')
    ),
    'src.agentsmcp.ui.v2.orchestrator_integration': Mock(
        OrchestratorTUIIntegration=Mock,
        OrchestratorIntegrationConfig=Mock
    )
}

with patch.dict('sys.modules', mock_modules):
    from src.agentsmcp.ui.v2.display_manager import (
        DisplayManager, DisplayRegion, ContentUpdate, RefreshMode, RegionType
    )
    from src.agentsmcp.ui.v2.unified_tui_coordinator import (
        UnifiedTUICoordinator, TUIMode, TUIStatus, ComponentConfig
    )


class TestIntegrationLayerIntegration:
    """Test integration between display manager and unified TUI coordinator."""
    
    @pytest.fixture
    async def integration_setup(self):
        """Setup integrated display manager and TUI coordinator."""
        # Create instances
        display_manager = DisplayManager()
        coordinator = UnifiedTUICoordinator()
        
        # Initialize both
        display_success = await display_manager.initialize()
        coordinator_success = await coordinator.initialize()
        
        assert display_success
        assert coordinator_success
        
        yield display_manager, coordinator
        
        # Cleanup
        await coordinator.shutdown()
        await display_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_coordinated_initialization(self, integration_setup):
        """Test that both components initialize successfully together."""
        display_manager, coordinator = integration_setup
        
        # Both should be initialized
        assert display_manager._initialized
        assert coordinator._initialized
        
        # Both should have their dependencies
        assert display_manager._terminal_controller is not None
        assert display_manager._logging_manager is not None
        assert coordinator._terminal_controller is not None
        assert coordinator._logging_manager is not None
    
    @pytest.mark.asyncio
    async def test_tui_startup_with_display_coordination(self, integration_setup):
        """Test TUI startup with display manager coordination."""
        display_manager, coordinator = integration_setup
        
        # Register display regions that TUI will use
        regions = [
            DisplayRegion(
                region_id='tui_header',
                region_type=RegionType.HEADER,
                x=0, y=0, width=80, height=3
            ),
            DisplayRegion(
                region_id='tui_main',
                region_type=RegionType.MAIN,
                x=0, y=3, width=80, height=18
            ),
            DisplayRegion(
                region_id='tui_footer',
                region_type=RegionType.FOOTER,
                x=0, y=21, width=80, height=3
            )
        ]
        
        for region in regions:
            success = await display_manager.register_region(region)
            assert success
        
        # Start TUI - should coordinate with display manager
        tui_instance, mode_active, status = await coordinator.start_tui(TUIMode.REVOLUTIONARY)
        
        assert mode_active
        assert coordinator.get_status() == TUIStatus.ACTIVE
        
        # Display manager should be aware of the TUI
        metrics = display_manager.get_performance_metrics()
        assert metrics['region_count'] >= len(regions)
    
    @pytest.mark.asyncio
    async def test_conflict_free_display_updates(self, integration_setup):
        """Test that display updates don't conflict when TUI is active."""
        display_manager, coordinator = integration_setup
        
        # Start TUI
        await coordinator.start_tui(TUIMode.BASIC)
        
        # Register TUI regions
        tui_region = DisplayRegion(
            region_id='tui_content',
            region_type=RegionType.MAIN,
            x=0, y=0, width=80, height=24
        )
        await display_manager.register_region(tui_region)
        
        # Try to update content while TUI is active
        update = ContentUpdate(
            region_id='tui_content',
            content='TUI is running smoothly',
            requester='integration_test'
        )
        
        display_updated, conflict_detected, metrics = await display_manager.update_content(
            layout_regions={'tui_content': tui_region},
            content_updates=[update]
        )
        
        # Should work without conflicts
        assert display_updated
        assert not conflict_detected
        assert metrics['updates_applied'] == 1
    
    @pytest.mark.asyncio
    async def test_mode_switching_with_display_coordination(self, integration_setup):
        """Test TUI mode switching coordinates properly with display manager."""
        display_manager, coordinator = integration_setup
        
        # Track display manager state during mode switches
        initial_metrics = display_manager.get_performance_metrics()
        
        # Start in basic mode
        await coordinator.start_tui(TUIMode.BASIC)
        basic_metrics = display_manager.get_performance_metrics()
        
        # Switch to revolutionary mode
        success, switch_metrics = await coordinator.switch_mode(TUIMode.REVOLUTIONARY)
        assert success
        
        revolutionary_metrics = display_manager.get_performance_metrics()
        
        # Switch to fallback mode
        success, _ = await coordinator.switch_mode(TUIMode.FALLBACK)
        assert success
        
        fallback_metrics = display_manager.get_performance_metrics()
        
        # Each mode switch should maintain display manager consistency
        assert basic_metrics['initialized']
        assert revolutionary_metrics['initialized']
        assert fallback_metrics['initialized']
        
        # No conflicts should occur during mode switches
        assert fallback_metrics['metrics']['conflict_count'] == 0
    
    @pytest.mark.asyncio
    async def test_logging_isolation_coordination(self, integration_setup):
        """Test that logging isolation is coordinated between components."""
        display_manager, coordinator = integration_setup
        
        # Start TUI with logging isolation
        config = ComponentConfig(enable_logging_isolation=True)
        await coordinator.initialize(component_config=config)
        
        tui_instance, mode_active, status = await coordinator.start_tui(TUIMode.REVOLUTIONARY)
        assert mode_active
        
        # Both components should have access to logging manager
        assert coordinator._logging_manager is not None
        assert display_manager._logging_manager is not None
        
        # Logging isolation should be active
        assert coordinator._logging_manager.isolation_active
    
    @pytest.mark.asyncio
    async def test_terminal_state_coordination(self, integration_setup):
        """Test terminal state is properly coordinated between components."""
        display_manager, coordinator = integration_setup
        
        # Start TUI with alternate screen
        config = ComponentConfig(enable_alternate_screen=True)
        await coordinator.initialize(component_config=config)
        
        # Both should share the same terminal controller
        assert coordinator._terminal_controller is display_manager._terminal_controller
        
        # Start TUI - should activate alternate screen
        await coordinator.start_tui(TUIMode.REVOLUTIONARY)
        
        # Terminal controller should be in alternate screen mode
        assert coordinator._terminal_controller.alternate_screen_active
        
        # Stop TUI - should restore normal screen
        await coordinator.switch_mode(TUIMode.DISABLED)
        assert not coordinator._terminal_controller.alternate_screen_active
    
    @pytest.mark.asyncio
    async def test_performance_coordination(self, integration_setup):
        """Test performance requirements are met with both components active."""
        display_manager, coordinator = integration_setup
        
        # Test startup performance with both components
        start_time = time.time()
        tui_instance, mode_active, status = await coordinator.start_tui(TUIMode.REVOLUTIONARY)
        startup_time = time.time() - start_time
        
        assert mode_active
        assert startup_time < 2.0  # ICD requirement
        
        # Test display update performance while TUI is running
        region = DisplayRegion(
            region_id='perf_test',
            region_type=RegionType.MAIN,
            x=0, y=0, width=80, height=24
        )
        await display_manager.register_region(region)
        
        update = ContentUpdate(
            region_id='perf_test',
            content='Performance test content',
            refresh_mode=RefreshMode.PARTIAL,
            requester='perf_test'
        )
        
        update_start = time.time()
        display_updated, conflict_detected, metrics = await display_manager.update_content(
            layout_regions={'perf_test': region},
            content_updates=[update],
            refresh_mode=RefreshMode.PARTIAL
        )
        update_time = time.time() - update_start
        
        assert display_updated
        assert not conflict_detected
        assert update_time < 0.01  # ICD requirement for partial updates (10ms)
        
        # Test mode switching performance
        switch_start = time.time()
        success, switch_metrics = await coordinator.switch_mode(TUIMode.BASIC)
        switch_time = time.time() - switch_start
        
        assert success
        assert switch_time < 0.5  # ICD requirement (500ms)
    
    @pytest.mark.asyncio
    async def test_error_handling_coordination(self, integration_setup):
        """Test coordinated error handling between components."""
        display_manager, coordinator = integration_setup
        
        # Simulate display manager error
        with patch.object(display_manager, 'update_content', side_effect=Exception("Display error")):
            # TUI should still function
            tui_instance, mode_active, status = await coordinator.start_tui(TUIMode.FALLBACK)
            assert mode_active  # Should fall back gracefully
        
        # Simulate TUI startup error
        with patch.object(coordinator, '_create_tui_instance', return_value=None):
            tui_instance, mode_active, status = await coordinator.start_tui(TUIMode.REVOLUTIONARY)
            assert not mode_active
            assert 'error' in status
            
            # Display manager should still be functional
            region = DisplayRegion(
                region_id='error_test',
                region_type=RegionType.MAIN,
                x=0, y=0, width=80, height=24
            )
            success = await display_manager.register_region(region)
            assert success
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_coordination(self, integration_setup):
        """Test coordinated graceful shutdown of both components."""
        display_manager, coordinator = integration_setup
        
        # Start TUI with active display updates
        await coordinator.start_tui(TUIMode.REVOLUTIONARY)
        
        region = DisplayRegion(
            region_id='shutdown_test',
            region_type=RegionType.MAIN,
            x=0, y=0, width=80, height=24
        )
        await display_manager.register_region(region)
        
        update = ContentUpdate(
            region_id='shutdown_test',
            content='Running content',
            requester='shutdown_test'
        )
        await display_manager.update_content(
            layout_regions={'shutdown_test': region},
            content_updates=[update]
        )
        
        # Both should be active
        assert coordinator.get_status() == TUIStatus.ACTIVE
        assert display_manager._initialized
        
        # Shutdown coordinator first
        await coordinator.shutdown()
        assert coordinator.get_status() == TUIStatus.SHUTTING_DOWN
        assert not coordinator._initialized
        
        # Display manager should still be functional for final cleanup
        assert display_manager._initialized
        
        # Final display manager cleanup
        await display_manager.cleanup()
        assert not display_manager._initialized


class TestRealWorldScenarios:
    """Test real-world usage scenarios with both components."""
    
    @pytest.mark.asyncio
    async def test_chat_interface_simulation(self):
        """Simulate a chat interface using both components."""
        with patch.dict('sys.modules', mock_modules):
            display_manager = DisplayManager()
            coordinator = UnifiedTUICoordinator()
            
            try:
                # Initialize
                await display_manager.initialize()
                await coordinator.initialize()
                
                # Setup chat interface regions
                chat_regions = [
                    DisplayRegion(
                        region_id='chat_header',
                        region_type=RegionType.HEADER,
                        x=0, y=0, width=100, height=2
                    ),
                    DisplayRegion(
                        region_id='chat_messages',
                        region_type=RegionType.MAIN,
                        x=0, y=2, width=100, height=20
                    ),
                    DisplayRegion(
                        region_id='chat_input',
                        region_type=RegionType.FOOTER,
                        x=0, y=22, width=100, height=2
                    )
                ]
                
                for region in chat_regions:
                    await display_manager.register_region(region)
                
                # Start revolutionary TUI for chat
                await coordinator.start_tui(TUIMode.REVOLUTIONARY)
                
                # Simulate chat messages
                messages = [
                    "Welcome to AgentsMCP Chat!",
                    "How can I help you today?",
                    "User: Hello, I need help with...",
                    "Assistant: I'd be happy to help!"
                ]
                
                for i, message in enumerate(messages):
                    update = ContentUpdate(
                        region_id='chat_messages',
                        content=f"Message {i+1}: {message}",
                        requester='chat_simulation'
                    )
                    
                    display_updated, conflict_detected, metrics = await display_manager.update_content(
                        layout_regions={region.region_id: region for region in chat_regions},
                        content_updates=[update]
                    )
                    
                    assert display_updated
                    assert not conflict_detected
                    
                    # Brief pause between messages
                    await asyncio.sleep(0.01)
                
                # Switch to basic mode (user preference)
                success, _ = await coordinator.switch_mode(TUIMode.BASIC)
                assert success
                
                # Continue chat in basic mode
                final_update = ContentUpdate(
                    region_id='chat_messages',
                    content="Chat continuing in basic mode...",
                    requester='chat_simulation'
                )
                
                display_updated, conflict_detected, metrics = await display_manager.update_content(
                    layout_regions={region.region_id: region for region in chat_regions},
                    content_updates=[final_update]
                )
                
                assert display_updated
                assert not conflict_detected
                
            finally:
                await coordinator.shutdown()
                await display_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_multi_panel_dashboard_simulation(self):
        """Simulate a multi-panel dashboard with real-time updates."""
        with patch.dict('sys.modules', mock_modules):
            display_manager = DisplayManager()
            coordinator = UnifiedTUICoordinator()
            
            try:
                await display_manager.initialize()
                await coordinator.initialize()
                
                # Setup dashboard regions
                dashboard_regions = [
                    DisplayRegion(
                        region_id='status_panel',
                        region_type=RegionType.HEADER,
                        x=0, y=0, width=120, height=4
                    ),
                    DisplayRegion(
                        region_id='metrics_panel',
                        region_type=RegionType.SIDEBAR,
                        x=0, y=4, width=40, height=16
                    ),
                    DisplayRegion(
                        region_id='logs_panel',
                        region_type=RegionType.MAIN,
                        x=40, y=4, width=80, height=16
                    ),
                    DisplayRegion(
                        region_id='input_panel',
                        region_type=RegionType.FOOTER,
                        x=0, y=20, width=120, height=4
                    )
                ]
                
                for region in dashboard_regions:
                    await display_manager.register_region(region)
                
                # Start TUI in revolutionary mode for rich dashboard
                await coordinator.start_tui(TUIMode.REVOLUTIONARY)
                
                # Simulate concurrent updates to different panels
                update_tasks = []
                
                # Status updates
                for i in range(5):
                    update = ContentUpdate(
                        region_id='status_panel',
                        content=f"Status Update {i+1}: System operational",
                        priority=1,
                        requester='status_updater'
                    )
                    task = display_manager.update_content(
                        layout_regions={r.region_id: r for r in dashboard_regions},
                        content_updates=[update]
                    )
                    update_tasks.append(task)
                
                # Metrics updates
                for i in range(3):
                    update = ContentUpdate(
                        region_id='metrics_panel',
                        content=f"CPU: {50 + i*10}%, Memory: {60 + i*5}%",
                        priority=2,
                        requester='metrics_updater'
                    )
                    task = display_manager.update_content(
                        layout_regions={r.region_id: r for r in dashboard_regions},
                        content_updates=[update]
                    )
                    update_tasks.append(task)
                
                # Log updates
                for i in range(7):
                    update = ContentUpdate(
                        region_id='logs_panel',
                        content=f"[{datetime.now()}] Log entry {i+1}",
                        priority=0,
                        requester='log_updater'
                    )
                    task = display_manager.update_content(
                        layout_regions={r.region_id: r for r in dashboard_regions},
                        content_updates=[update]
                    )
                    update_tasks.append(task)
                
                # Wait for all updates to complete
                results = await asyncio.gather(*update_tasks)
                
                # All updates should succeed without conflicts
                for display_updated, conflict_detected, metrics in results:
                    assert display_updated or not conflict_detected  # Either succeeded or no conflict
                
                # Check final performance metrics
                coordinator_metrics = coordinator.get_performance_metrics()
                display_metrics = display_manager.get_performance_metrics()
                
                assert coordinator_metrics['current_mode'] == 'revolutionary'
                assert display_metrics['region_count'] == len(dashboard_regions)
                
            finally:
                await coordinator.shutdown()
                await display_manager.cleanup()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])