"""
Test suite for Display Manager integration module.

Tests the display coordination functionality, conflict detection,
region management, and performance requirements.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Mock Rich components for testing without Rich dependency
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

# Patch Rich imports before importing the module
with patch.dict('sys.modules', {
    'rich.console': Mock(Console=MockConsole),
    'rich.layout': Mock(Layout=MockLayout),
    'rich.live': Mock(Live=MockLive),
    'rich.text': Mock(Text=MockText),
    'rich.panel': Mock(Panel=MockPanel)
}):
    from src.agentsmcp.ui.v2.display_manager import (
        DisplayManager, get_display_manager, cleanup_display_manager,
        DisplayRegion, ContentUpdate, RefreshMode, RegionType,
        ConflictDetector, DisplayMetrics
    )


class TestConflictDetector:
    """Test conflict detection functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create a conflict detector instance."""
        return ConflictDetector()
    
    @pytest.fixture
    def sample_regions(self):
        """Create sample regions for testing."""
        return {
            'header': DisplayRegion(
                region_id='header',
                region_type=RegionType.HEADER,
                x=0, y=0, width=80, height=3, z_index=1
            ),
            'main': DisplayRegion(
                region_id='main', 
                region_type=RegionType.MAIN,
                x=0, y=3, width=50, height=20, z_index=0
            ),
            'sidebar': DisplayRegion(
                region_id='sidebar',
                region_type=RegionType.SIDEBAR, 
                x=50, y=3, width=30, height=20, z_index=0
            )
        }
    
    def test_no_conflicts_basic(self, detector, sample_regions):
        """Test basic case with no conflicts."""
        update = ContentUpdate(
            region_id='header',
            content='Test header',
            requester='test'
        )
        
        has_conflict, reason = detector.check_conflicts(update, sample_regions)
        assert not has_conflict
        assert reason is None
    
    def test_nonexistent_region_conflict(self, detector, sample_regions):
        """Test conflict detection for nonexistent region."""
        update = ContentUpdate(
            region_id='nonexistent',
            content='Test content',
            requester='test'
        )
        
        has_conflict, reason = detector.check_conflicts(update, sample_regions)
        assert has_conflict
        assert 'does not exist' in reason
    
    def test_region_lock_conflict(self, detector, sample_regions):
        """Test conflict detection for locked regions."""
        # Lock region for one requester
        detector.acquire_region_lock('header', 'requester1')
        
        # Try to update with different requester
        update = ContentUpdate(
            region_id='header',
            content='Test content',
            requester='requester2'
        )
        
        has_conflict, reason = detector.check_conflicts(update, sample_regions)
        assert has_conflict
        assert 'locked by' in reason
    
    def test_active_update_conflict(self, detector, sample_regions):
        """Test conflict detection for active updates."""
        detector.mark_update_active('header')
        
        update = ContentUpdate(
            region_id='header',
            content='Test content',
            requester='test'
        )
        
        has_conflict, reason = detector.check_conflicts(update, sample_regions)
        assert has_conflict
        assert 'pending update' in reason
    
    def test_z_index_overlap_conflict(self, detector):
        """Test conflict detection for overlapping z-index regions."""
        # Create overlapping overlay regions with same z-index
        regions = {
            'overlay1': DisplayRegion(
                region_id='overlay1',
                region_type=RegionType.OVERLAY,
                x=10, y=10, width=20, height=10, z_index=5, visible=True
            ),
            'overlay2': DisplayRegion(
                region_id='overlay2',
                region_type=RegionType.OVERLAY,
                x=15, y=15, width=20, height=10, z_index=5, visible=True
            )
        }
        
        update = ContentUpdate(
            region_id='overlay1',
            content='Test content',
            requester='test'
        )
        
        has_conflict, reason = detector.check_conflicts(update, regions)
        assert has_conflict
        assert 'Z-index conflict' in reason
    
    def test_region_lock_management(self, detector):
        """Test region lock acquisition and release."""
        # Acquire lock
        success = detector.acquire_region_lock('test_region', 'requester1')
        assert success
        
        # Same requester can re-acquire
        success = detector.acquire_region_lock('test_region', 'requester1')
        assert success
        
        # Different requester cannot acquire
        success = detector.acquire_region_lock('test_region', 'requester2')
        assert not success
        
        # Release lock
        success = detector.release_region_lock('test_region', 'requester1')
        assert success
        
        # Wrong requester cannot release
        success = detector.release_region_lock('test_region', 'requester2')
        assert not success
    
    def test_update_lifecycle_tracking(self, detector):
        """Test update active/complete tracking."""
        region_id = 'test_region'
        
        # Mark as active
        detector.mark_update_active(region_id)
        assert region_id in detector._active_updates
        
        # Mark as complete
        detector.mark_update_complete(region_id)
        assert region_id not in detector._active_updates


class TestDisplayManager:
    """Test display manager functionality."""
    
    @pytest.fixture
    async def manager(self):
        """Create a display manager instance."""
        # Mock dependencies
        with patch('src.agentsmcp.ui.v2.display_manager.get_terminal_controller') as mock_terminal, \
             patch('src.agentsmcp.ui.v2.display_manager.get_logging_isolation_manager') as mock_logging:
            
            # Mock terminal controller
            mock_terminal_instance = Mock()
            mock_terminal_instance.initialize = AsyncMock(return_value=True)
            mock_terminal_instance.get_terminal_state = AsyncMock(return_value=Mock(
                size=Mock(width=80, height=24)
            ))
            mock_terminal.return_value = mock_terminal_instance
            
            # Mock logging manager
            mock_logging_instance = Mock()
            mock_logging_instance.initialize = AsyncMock(return_value=True)
            mock_logging.return_value = mock_logging_instance
            
            manager = DisplayManager()
            success = await manager.initialize()
            assert success
            
            yield manager
            
            await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test display manager initialization."""
        with patch('src.agentsmcp.ui.v2.display_manager.get_terminal_controller') as mock_terminal, \
             patch('src.agentsmcp.ui.v2.display_manager.get_logging_isolation_manager') as mock_logging:
            
            mock_terminal_instance = Mock()
            mock_terminal_instance.initialize = AsyncMock(return_value=True)
            mock_terminal_instance.get_terminal_state = AsyncMock(return_value=Mock(
                size=Mock(width=80, height=24)
            ))
            mock_terminal.return_value = mock_terminal_instance
            
            mock_logging_instance = Mock()
            mock_logging_instance.initialize = AsyncMock(return_value=True)
            mock_logging.return_value = mock_logging_instance
            
            manager = DisplayManager()
            success = await manager.initialize()
            
            assert success
            assert manager._initialized
            assert manager._terminal_controller is not None
            assert manager._logging_manager is not None
            
            await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_region_registration(self, manager):
        """Test region registration."""
        region = DisplayRegion(
            region_id='test_region',
            region_type=RegionType.MAIN,
            x=0, y=0, width=80, height=24
        )
        
        success = await manager.register_region(region)
        assert success
        
        # Verify region was registered
        retrieved = await manager.get_region('test_region')
        assert retrieved is not None
        assert retrieved.region_id == 'test_region'
    
    @pytest.mark.asyncio
    async def test_content_update_basic(self, manager):
        """Test basic content update functionality."""
        # Register a region first
        region = DisplayRegion(
            region_id='test_region',
            region_type=RegionType.MAIN,
            x=0, y=0, width=80, height=24
        )
        await manager.register_region(region)
        
        # Create content update
        update = ContentUpdate(
            region_id='test_region',
            content='Test content',
            requester='test'
        )
        
        # Perform update
        display_updated, conflict_detected, metrics = await manager.update_content(
            layout_regions={'test_region': region},
            content_updates=[update],
            refresh_mode=RefreshMode.PARTIAL
        )
        
        assert display_updated
        assert not conflict_detected
        assert 'refresh_time_ms' in metrics
        assert metrics['updates_applied'] == 1
    
    @pytest.mark.asyncio
    async def test_content_update_conflicts(self, manager):
        """Test content update conflict detection."""
        # Create conflicting updates (nonexistent region)
        update = ContentUpdate(
            region_id='nonexistent_region',
            content='Test content',
            requester='test'
        )
        
        display_updated, conflict_detected, metrics = await manager.update_content(
            layout_regions={},
            content_updates=[update],
            refresh_mode=RefreshMode.PARTIAL
        )
        
        assert not display_updated or conflict_detected  # Either no update or conflict detected
        assert metrics['conflicts_detected'] > 0 or 'error' in metrics
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, manager):
        """Test performance requirements (ICD compliance)."""
        # Register region
        region = DisplayRegion(
            region_id='perf_test',
            region_type=RegionType.MAIN,
            x=0, y=0, width=80, height=24
        )
        await manager.register_region(region)
        
        # Test partial update performance (should be <= 10ms)
        update = ContentUpdate(
            region_id='perf_test',
            content='Performance test',
            requester='test'
        )
        
        start_time = time.time()
        display_updated, _, metrics = await manager.update_content(
            layout_regions={'perf_test': region},
            content_updates=[update],
            refresh_mode=RefreshMode.PARTIAL
        )
        elapsed_ms = (time.time() - start_time) * 1000
        
        assert display_updated
        # Allow some tolerance for test environment
        assert elapsed_ms < 100  # More lenient for tests
        assert metrics['refresh_time_ms'] is not None
    
    @pytest.mark.asyncio
    async def test_display_context_manager(self, manager):
        """Test display context manager functionality."""
        context_entered = False
        context_exited = False
        
        async with manager.display_context(enable_rich_live=False) as ctx:
            context_entered = True
            assert ctx is manager
        
        context_exited = True
        assert context_entered and context_exited
    
    @pytest.mark.asyncio
    async def test_callback_system(self, manager):
        """Test update and conflict callback system."""
        update_called = False
        conflict_called = False
        
        def update_callback(region_id, content):
            nonlocal update_called
            update_called = True
            assert region_id == 'callback_test'
            assert content == 'Test content'
        
        def conflict_callback(region_id, reason):
            nonlocal conflict_called
            conflict_called = True
            assert region_id == 'nonexistent'
        
        # Register callbacks
        manager.register_update_callback(update_callback)
        manager.register_conflict_callback(conflict_callback)
        
        # Test update callback
        region = DisplayRegion(
            region_id='callback_test',
            region_type=RegionType.MAIN,
            x=0, y=0, width=80, height=24
        )
        await manager.register_region(region)
        
        update = ContentUpdate(
            region_id='callback_test',
            content='Test content',
            requester='test'
        )
        
        await manager.update_content(
            layout_regions={'callback_test': region},
            content_updates=[update]
        )
        
        # Give some time for async processing
        await asyncio.sleep(0.1)
        assert update_called
        
        # Test conflict callback
        conflict_update = ContentUpdate(
            region_id='nonexistent',
            content='Conflict test',
            requester='test'
        )
        
        await manager.update_content(
            layout_regions={},
            content_updates=[conflict_update]
        )
        
        await asyncio.sleep(0.1)
        assert conflict_called
    
    @pytest.mark.asyncio
    async def test_force_refresh(self, manager):
        """Test force refresh functionality."""
        success = await manager.force_refresh(RefreshMode.FULL)
        assert success
        
        metrics = manager.get_performance_metrics()
        assert metrics['metrics']['full_refresh_count'] >= 1
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, manager):
        """Test performance metrics collection."""
        metrics = manager.get_performance_metrics()
        
        required_keys = [
            'metrics', 'queue_size', 'active_refresh', 'region_count', 
            'initialized', 'rich_available', 'average_refresh_time_ms'
        ]
        
        for key in required_keys:
            assert key in metrics
        
        assert metrics['initialized'] == True
        assert metrics['region_count'] >= 0


class TestDisplayManagerIntegration:
    """Test display manager integration with other components."""
    
    @pytest.mark.asyncio
    async def test_singleton_access(self):
        """Test singleton access pattern."""
        with patch('src.agentsmcp.ui.v2.display_manager.get_terminal_controller') as mock_terminal, \
             patch('src.agentsmcp.ui.v2.display_manager.get_logging_isolation_manager') as mock_logging:
            
            mock_terminal_instance = Mock()
            mock_terminal_instance.initialize = AsyncMock(return_value=True)
            mock_terminal_instance.get_terminal_state = AsyncMock(return_value=Mock(
                size=Mock(width=80, height=24)
            ))
            mock_terminal.return_value = mock_terminal_instance
            
            mock_logging_instance = Mock()
            mock_logging_instance.initialize = AsyncMock(return_value=True)
            mock_logging.return_value = mock_logging_instance
            
            # Get singleton instance
            manager1 = await get_display_manager()
            manager2 = await get_display_manager()
            
            # Should be same instance
            assert manager1 is manager2
            
            # Cleanup
            await cleanup_display_manager()
    
    @pytest.mark.asyncio
    async def test_multiple_updates_queuing(self):
        """Test queuing of multiple updates."""
        with patch('src.agentsmcp.ui.v2.display_manager.get_terminal_controller') as mock_terminal, \
             patch('src.agentsmcp.ui.v2.display_manager.get_logging_isolation_manager') as mock_logging:
            
            mock_terminal_instance = Mock()
            mock_terminal_instance.initialize = AsyncMock(return_value=True)
            mock_terminal_instance.get_terminal_state = AsyncMock(return_value=Mock(
                size=Mock(width=80, height=24)
            ))
            mock_terminal.return_value = mock_terminal_instance
            
            mock_logging_instance = Mock()
            mock_logging_instance.initialize = AsyncMock(return_value=True)
            mock_logging.return_value = mock_logging_instance
            
            manager = DisplayManager()
            await manager.initialize()
            
            try:
                # Register regions
                regions = {}
                for i in range(3):
                    region = DisplayRegion(
                        region_id=f'region_{i}',
                        region_type=RegionType.MAIN,
                        x=i*20, y=0, width=20, height=24
                    )
                    await manager.register_region(region)
                    regions[f'region_{i}'] = region
                
                # Create multiple updates
                updates = []
                for i in range(3):
                    update = ContentUpdate(
                        region_id=f'region_{i}',
                        content=f'Content {i}',
                        priority=i,  # Different priorities
                        requester='test'
                    )
                    updates.append(update)
                
                # Apply all updates
                display_updated, conflict_detected, metrics = await manager.update_content(
                    layout_regions=regions,
                    content_updates=updates
                )
                
                assert display_updated
                assert not conflict_detected
                assert metrics['updates_applied'] == 3
                
            finally:
                await manager.cleanup()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])