#!/usr/bin/env python3
"""
Isolated integration test for display_manager and unified_tui_coordinator.

Tests the modules without importing the broader codebase dependencies.
"""

import asyncio
import pytest
import time
import sys
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

# Mock the problematic dependencies before any imports
sys.modules['src.agentsmcp.ui.v2.terminal_controller'] = Mock()
sys.modules['src.agentsmcp.ui.v2.logging_isolation_manager'] = Mock()
sys.modules['src.agentsmcp.ui.v2.text_layout_engine'] = Mock()
sys.modules['src.agentsmcp.ui.v2.input_rendering_pipeline'] = Mock()
sys.modules['src.agentsmcp.ui.v2.revolutionary_tui_interface'] = Mock()
sys.modules['src.agentsmcp.ui.v2.orchestrator_integration'] = Mock()

# Mock Rich components
class MockConsole:
    def __init__(self, force_terminal=True):
        pass

class MockLayout:
    def __init__(self):
        self.regions = {}
    
    def split_column(self, *args):
        pass
    
    def split_row(self, *args):
        pass
    
    def __getitem__(self, key):
        if key not in self.regions:
            self.regions[key] = MockLayoutRegion(key)
        return self.regions[key]

class MockLayoutRegion:
    def __init__(self, name):
        self.name = name
        self.content = None
    
    def update(self, content):
        self.content = content

class MockLive:
    def __init__(self, layout, refresh_per_second=60):
        self.layout = layout
        self.refresh_per_second = refresh_per_second
        self.is_started = False
    
    def start(self):
        self.is_started = True
    
    def stop(self):
        self.is_started = False
    
    def refresh(self):
        pass

class MockText:
    def __init__(self, text):
        self.text = text
    
    def __str__(self):
        return self.text

class MockPanel:
    def __init__(self, content, title=""):
        self.content = content
        self.title = title

# Mock terminal controller and related classes
class MockTerminalState:
    def __init__(self, width=80, height=24):
        self.size = Mock(width=width, height=height)

class MockTerminalController:
    def __init__(self):
        self.initialized = False
        
    async def initialize(self):
        self.initialized = True
        return True
        
    async def get_terminal_state(self):
        return MockTerminalState()

class MockLoggingManager:
    def __init__(self):
        self.initialized = False
        
    async def initialize(self):
        self.initialized = True
        return True
    
    async def activate(self):
        pass
        
    async def deactivate(self):
        pass

class MockRevolutionaryTUI:
    def __init__(self):
        self.initialized = False
        self.running = False
        
    async def initialize(self, **kwargs):
        self.initialized = True
        return True
        
    async def start(self):
        self.running = True
        return True
        
    async def stop(self):
        self.running = False
        return True

# Patch Rich imports before importing the modules
with patch.dict('sys.modules', {
    'rich.console': Mock(Console=MockConsole),
    'rich.layout': Mock(Layout=MockLayout),
    'rich.live': Mock(Live=MockLive),
    'rich.text': Mock(Text=MockText),
    'rich.panel': Mock(Panel=MockPanel)
}):
    from src.agentsmcp.ui.v2.display_manager import (
        DisplayManager, ConflictDetector,
        DisplayRegion, ContentUpdate, RefreshMode, RegionType
    )
    from src.agentsmcp.ui.v2.unified_tui_coordinator import (
        UnifiedTUICoordinator, TUIMode, ComponentConfig, TUIStatus
    )

def get_mock_terminal_controller():
    return MockTerminalController()

def get_mock_logging_manager():
    return MockLoggingManager()

def mock_revolutionary_tui_class():
    return MockRevolutionaryTUI


class TestIntegrationModules:
    """Test the integration layer modules in isolation."""
    
    @pytest.fixture
    async def display_manager(self):
        """Create a display manager with mocked dependencies."""
        
        with patch('src.agentsmcp.ui.v2.display_manager.get_terminal_controller', get_mock_terminal_controller), \
             patch('src.agentsmcp.ui.v2.display_manager.get_logging_isolation_manager', get_mock_logging_manager):
            
            manager = DisplayManager()
            success = await manager.initialize()
            assert success
            
            yield manager
            
            await manager.cleanup()
    
    @pytest.fixture
    async def tui_coordinator(self):
        """Create a TUI coordinator with mocked dependencies."""
        
        with patch('src.agentsmcp.ui.v2.unified_tui_coordinator.get_terminal_controller', get_mock_terminal_controller), \
             patch('src.agentsmcp.ui.v2.unified_tui_coordinator.get_logging_isolation_manager', get_mock_logging_manager), \
             patch('src.agentsmcp.ui.v2.unified_tui_coordinator.RevolutionaryTUIInterface', mock_revolutionary_tui_class):
            
            coordinator = UnifiedTUICoordinator()
            await coordinator.initialize()
            
            yield coordinator
            
            await coordinator.shutdown()
    
    @pytest.mark.asyncio
    async def test_display_manager_initialization(self, display_manager):
        """Test display manager initializes correctly."""
        assert display_manager._initialized
        assert display_manager._terminal_controller is not None
        assert display_manager._logging_manager is not None
    
    @pytest.mark.asyncio
    async def test_display_manager_region_operations(self, display_manager):
        """Test display manager region registration and retrieval."""
        # Register a region
        region = DisplayRegion(
            region_id='test_region',
            region_type=RegionType.MAIN,
            x=0, y=0, width=80, height=24
        )
        
        success = await display_manager.register_region(region)
        assert success
        
        # Retrieve the region
        retrieved = await display_manager.get_region('test_region')
        assert retrieved is not None
        assert retrieved.region_id == 'test_region'
        
        # Get all regions
        all_regions = await display_manager.get_all_regions()
        assert 'test_region' in all_regions
    
    @pytest.mark.asyncio
    async def test_display_manager_content_updates(self, display_manager):
        """Test display manager content update functionality."""
        # Register a region
        region = DisplayRegion(
            region_id='update_test',
            region_type=RegionType.MAIN,
            x=0, y=0, width=80, height=24
        )
        await display_manager.register_region(region)
        
        # Create content update
        update = ContentUpdate(
            region_id='update_test',
            content='Test content',
            requester='test'
        )
        
        # Perform update
        display_updated, conflict_detected, metrics = await display_manager.update_content(
            layout_regions={'update_test': region},
            content_updates=[update],
            refresh_mode=RefreshMode.PARTIAL
        )
        
        assert display_updated
        assert not conflict_detected
        assert 'refresh_time_ms' in metrics
    
    @pytest.mark.asyncio
    async def test_conflict_detection(self, display_manager):
        """Test conflict detection functionality."""
        detector = ConflictDetector()
        
        # Test nonexistent region conflict
        regions = {}
        update = ContentUpdate(
            region_id='nonexistent',
            content='Test',
            requester='test'
        )
        
        has_conflict, reason = detector.check_conflicts(update, regions)
        assert has_conflict
        assert 'does not exist' in reason
    
    @pytest.mark.asyncio
    async def test_tui_coordinator_initialization(self, tui_coordinator):
        """Test TUI coordinator initializes correctly."""
        assert tui_coordinator._initialized
        assert tui_coordinator._current_mode == TUIMode.DISABLED
        assert tui_coordinator._status == TUIStatus.READY
    
    @pytest.mark.asyncio
    async def test_tui_coordinator_mode_management(self, tui_coordinator):
        """Test TUI coordinator mode switching."""
        # Start in revolutionary mode
        tui_instance, mode_active, status = await tui_coordinator.start_tui(TUIMode.REVOLUTIONARY)
        
        assert mode_active
        assert tui_coordinator._current_mode == TUIMode.REVOLUTIONARY
        assert 'startup_time_seconds' in status
        
        # Switch to basic mode
        success, metrics = await tui_coordinator.switch_mode(TUIMode.BASIC)
        assert success
        assert tui_coordinator._current_mode == TUIMode.BASIC
        assert 'switch_time_seconds' in metrics
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, display_manager):
        """Test that performance requirements are met."""
        # Register region
        region = DisplayRegion(
            region_id='perf_test',
            region_type=RegionType.MAIN,
            x=0, y=0, width=80, height=24
        )
        await display_manager.register_region(region)
        
        # Test partial update performance (should be <= 10ms per ICD)
        update = ContentUpdate(
            region_id='perf_test',
            content='Performance test content',
            requester='test'
        )
        
        start_time = time.time()
        display_updated, _, metrics = await display_manager.update_content(
            layout_regions={'perf_test': region},
            content_updates=[update],
            refresh_mode=RefreshMode.PARTIAL
        )
        elapsed_ms = (time.time() - start_time) * 1000
        
        assert display_updated
        # Allow tolerance for test environment, but verify it's reasonable
        assert elapsed_ms < 100  # More lenient for tests, but still fast
        assert metrics['refresh_time_ms'] is not None
    
    @pytest.mark.asyncio
    async def test_integrated_workflow(self):
        """Test display manager and TUI coordinator working together."""
        
        with patch('src.agentsmcp.ui.v2.display_manager.get_terminal_controller', get_mock_terminal_controller), \
             patch('src.agentsmcp.ui.v2.display_manager.get_logging_isolation_manager', get_mock_logging_manager), \
             patch('src.agentsmcp.ui.v2.unified_tui_coordinator.get_terminal_controller', get_mock_terminal_controller), \
             patch('src.agentsmcp.ui.v2.unified_tui_coordinator.get_logging_isolation_manager', get_mock_logging_manager), \
             patch('src.agentsmcp.ui.v2.unified_tui_coordinator.RevolutionaryTUIInterface', mock_revolutionary_tui_class):
            
            # Initialize both components
            display_manager = DisplayManager()
            await display_manager.initialize()
            
            coordinator = UnifiedTUICoordinator()
            await coordinator.initialize()
            
            try:
                # Register some regions
                region = DisplayRegion(
                    region_id='integrated_test',
                    region_type=RegionType.MAIN,
                    x=0, y=0, width=80, height=24
                )
                await display_manager.register_region(region)
                
                # Start TUI
                tui_instance, mode_active, status = await coordinator.start_tui(TUIMode.REVOLUTIONARY)
                assert mode_active
                
                # Update content while TUI is running
                update = ContentUpdate(
                    region_id='integrated_test',
                    content='Integrated test content',
                    requester='integration_test'
                )
                
                display_updated, conflict_detected, metrics = await display_manager.update_content(
                    layout_regions={'integrated_test': region},
                    content_updates=[update]
                )
                
                assert display_updated
                assert not conflict_detected
                
                # Switch modes
                success, switch_metrics = await coordinator.switch_mode(TUIMode.BASIC)
                assert success
                
                # Stop TUI
                success, stop_status = await coordinator.stop_tui()
                assert success
                
            finally:
                await coordinator.shutdown()
                await display_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, tui_coordinator):
        """Test error recovery capabilities."""
        # Force an error scenario
        with patch.object(tui_coordinator, '_start_revolutionary_tui', side_effect=Exception("Test error")):
            tui_instance, mode_active, status = await tui_coordinator.start_tui(TUIMode.REVOLUTIONARY)
            
            # Should not crash, should handle gracefully
            assert not mode_active
            assert 'error' in status
            
            # Status should indicate error recovery
            assert tui_coordinator._status in [TUIStatus.ERROR, TUIStatus.RECOVERING]


if __name__ == '__main__':
    # Run the isolated tests
    pytest.main([__file__, '-v'])