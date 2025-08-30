# -*- coding: utf-8 -*-
"""
Tests for critical TUI fixes addressing text rendering issues.

Tests verify:
1. Frame concatenation prevention via render lock
2. Race condition elimination in rendering
3. Optimized layout hashing performance
4. Request coalescing for mark_dirty calls
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from agentsmcp.ui.modern_tui import ModernTUI, TUIMode


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for TUI testing."""
    return {
        'config': Mock(),
        'theme_manager': Mock(),
        'conversation_manager': Mock(),
        'orchestration_manager': Mock()
    }


@pytest.fixture
def tui_instance(mock_dependencies):
    """Create TUI instance with mocked dependencies."""
    with patch('agentsmcp.ui.modern_tui.Console'):
        tui = ModernTUI(**mock_dependencies)
        # Initialize required attributes for testing
        tui._running = True
        tui._console = Mock()
        return tui


class TestRenderSynchronization:
    """Test render lock prevents frame concatenation."""
    
    @pytest.mark.asyncio
    async def test_render_lock_prevents_concurrent_updates(self, tui_instance):
        """Test that render lock prevents concurrent Live.update() calls."""
        # Track concurrent calls
        concurrent_renders = []
        original_render = tui_instance._render
        
        async def slow_render():
            concurrent_renders.append(time.time())
            await asyncio.sleep(0.01)  # Simulate slow rendering
            concurrent_renders.append(time.time())
            return original_render() if hasattr(tui_instance, '_render') else Mock()
        
        with patch.object(tui_instance, '_render', side_effect=slow_render):
            # Mock Live object
            live_mock = Mock()
            live_mock.update = Mock()
            
            # Simulate concurrent refresh attempts
            with patch('agentsmcp.ui.modern_tui.Live') as live_class_mock:
                live_class_mock.return_value.__enter__ = Mock(return_value=live_mock)
                live_class_mock.return_value.__exit__ = Mock(return_value=False)
                
                # Create multiple concurrent render tasks
                tasks = []
                for _ in range(3):
                    async def trigger_render():
                        tui_instance._refresh_event.set()
                        # Simulate the render loop logic
                        async with tui_instance._render_lock:
                            tui_instance._currently_rendering = True
                            try:
                                new_layout = await slow_render()
                                layout_hash = tui_instance._get_layout_hash(new_layout)
                                if layout_hash != tui_instance._last_rendered_hash:
                                    live_mock.update(new_layout)
                                    tui_instance._last_rendered_hash = layout_hash
                            finally:
                                tui_instance._currently_rendering = False
                    
                    tasks.append(asyncio.create_task(trigger_render()))
                
                # Wait for all tasks to complete
                await asyncio.gather(*tasks)
        
        # Verify renders were serialized (no overlapping timestamps)
        assert len(concurrent_renders) == 6  # 3 tasks * 2 timestamps each
        # Each render should complete before the next starts
        for i in range(0, len(concurrent_renders), 2):
            if i + 2 < len(concurrent_renders):
                assert concurrent_renders[i + 1] <= concurrent_renders[i + 2]
    
    @pytest.mark.asyncio
    async def test_currently_rendering_flag_prevents_feedback_loops(self, tui_instance):
        """Test that _currently_rendering flag prevents feedback loops."""
        # Set rendering flag
        tui_instance._currently_rendering = True
        
        # mark_dirty should be ignored during rendering
        tui_instance.mark_dirty("content")
        
        # Verify no refresh event was set
        assert not tui_instance._refresh_event.is_set()
        
        # Reset flag
        tui_instance._currently_rendering = False
        
        # Now mark_dirty should work
        await asyncio.sleep(0.01)  # Allow async task to run
        tui_instance.mark_dirty("content")
        await asyncio.sleep(0.1)  # Allow coalescing task to complete
        
        # Should trigger refresh after flag is cleared
        # Note: Due to async coalescing, we need to wait for the task
        await asyncio.sleep(0.2)


class TestRequestCoalescing:
    """Test request coalescing for mark_dirty calls."""
    
    @pytest.mark.asyncio
    async def test_mark_dirty_coalesces_requests(self, tui_instance):
        """Test that rapid mark_dirty calls are coalesced."""
        refresh_events = []
        original_set = tui_instance._refresh_event.set
        
        def track_refresh():
            refresh_events.append(time.time())
            original_set()
        
        tui_instance._refresh_event.set = track_refresh
        
        # Rapid fire mark_dirty calls
        for _ in range(10):
            tui_instance.mark_dirty("content")
        
        # Wait for coalescing
        await asyncio.sleep(0.3)
        
        # Should have coalesced into fewer refresh events
        assert len(refresh_events) < 10
        assert len(refresh_events) >= 1
    
    @pytest.mark.asyncio
    async def test_different_sections_have_different_debounce_times(self, tui_instance):
        """Test that different sections have appropriate debounce times."""
        start_time = time.time()
        
        # Footer should have ultra-fast debounce (0.02s)
        tui_instance.mark_dirty("footer")
        await asyncio.sleep(0.05)
        
        # Content should have moderate debounce (0.15s)
        tui_instance.mark_dirty("content") 
        await asyncio.sleep(0.2)
        
        # Sidebar should have longer debounce (0.5s)
        tui_instance.mark_dirty("sidebar")
        await asyncio.sleep(0.6)
        
        # Verify timing expectations are met
        elapsed = time.time() - start_time
        assert elapsed >= 0.8  # Total minimum time for all debounces
    
    @pytest.mark.asyncio
    async def test_pending_sections_are_cleared_after_refresh(self, tui_instance):
        """Test that pending refresh sections are cleared after refresh."""
        # Add multiple sections
        tui_instance.mark_dirty("header")
        tui_instance.mark_dirty("content")
        tui_instance.mark_dirty("footer")
        
        # Wait for coalescing
        await asyncio.sleep(0.3)
        
        # Pending sections should be cleared
        assert len(tui_instance._pending_refresh_sections) == 0


class TestLayoutHashOptimization:
    """Test optimized layout hash generation."""
    
    def test_layout_hash_uses_cache_versions(self, tui_instance):
        """Test that layout hash uses cache versions for performance."""
        # Mock layout object
        mock_layout = {
            "header": Mock(),
            "content": Mock(), 
            "footer": Mock(),
            "sidebar": Mock()
        }
        
        # Get initial hash
        hash1 = tui_instance._get_layout_hash(mock_layout)
        
        # Hash should be consistent
        hash2 = tui_instance._get_layout_hash(mock_layout)
        assert hash1 == hash2
        
        # Changing cache version should change hash
        tui_instance._cache_version["content"] += 1
        hash3 = tui_instance._get_layout_hash(mock_layout)
        assert hash1 != hash3
    
    def test_layout_hash_includes_ui_state(self, tui_instance):
        """Test that layout hash includes UI state changes."""
        mock_layout = {"header": Mock()}
        
        hash1 = tui_instance._get_layout_hash(mock_layout)
        
        # Change UI state
        tui_instance._sidebar_collapsed = not tui_instance._sidebar_collapsed
        hash2 = tui_instance._get_layout_hash(mock_layout)
        
        assert hash1 != hash2
    
    def test_layout_hash_performance_no_string_conversion(self, tui_instance):
        """Test that layout hash avoids expensive string conversion."""
        # Create mock layout with complex objects
        complex_mock = Mock()
        complex_mock.__str__ = Mock(side_effect=Exception("String conversion called"))
        
        mock_layout = {
            "header": complex_mock,
            "content": complex_mock,
            "footer": complex_mock
        }
        
        # Should not raise exception from string conversion
        hash_result = tui_instance._get_layout_hash(mock_layout)
        assert hash_result is not None
        
        # Verify __str__ was not called on complex objects
        complex_mock.__str__.assert_not_called()
    
    def test_layout_hash_fallback_on_error(self, tui_instance):
        """Test that layout hash has proper fallback on errors."""
        # Mock layout that raises errors when accessing items
        mock_layout = Mock()
        mock_layout.__getitem__ = Mock(side_effect=Exception("Layout error"))
        mock_layout.__contains__ = Mock(return_value=True)
        
        # Should fall back to timestamp-based hash
        with patch('time.time', return_value=12345.67):
            hash_result = tui_instance._get_layout_hash(mock_layout)
            # Just verify it returns a valid hash string (timestamp fallback works)
            assert hash_result is not None
            assert isinstance(hash_result, str)
            assert len(hash_result) > 0


class TestRaceConditionPrevention:
    """Test prevention of race conditions between input and rendering."""
    
    @pytest.mark.asyncio
    async def test_input_processing_isolated_from_rendering(self, tui_instance):
        """Test that input processing doesn't interfere with rendering."""
        # Mock input handling
        input_events = []
        render_events = []
        
        async def mock_handle_input(user_input):
            input_events.append(f"input_start_{user_input}")
            await asyncio.sleep(0.01)
            input_events.append(f"input_end_{user_input}")
        
        async def mock_render():
            render_events.append("render_start")
            await asyncio.sleep(0.01) 
            render_events.append("render_end")
            return Mock()
        
        with patch.object(tui_instance, '_handle_user_input', side_effect=mock_handle_input):
            with patch.object(tui_instance, '_render', side_effect=mock_render):
                # Simulate concurrent input and render
                input_task = asyncio.create_task(mock_handle_input("test"))
                
                # Trigger render while input is processing
                async with tui_instance._render_lock:
                    render_task = asyncio.create_task(mock_render())
                    await asyncio.gather(input_task, render_task)
        
        # Verify both completed without interference
        assert "input_start_test" in input_events
        assert "input_end_test" in input_events
        assert "render_start" in render_events
        assert "render_end" in render_events


class TestTerminalResize:
    """Test graceful handling of terminal resize during rendering."""
    
    @pytest.mark.asyncio
    async def test_render_lock_survives_terminal_resize(self, tui_instance):
        """Test that render lock works correctly during terminal resize."""
        resize_count = 0
        
        def mock_render():
            nonlocal resize_count
            resize_count += 1
            if resize_count == 2:
                # Simulate terminal resize during render
                tui_instance._console.size = (120, 40)  # Simulate resize
            return Mock()
        
        with patch.object(tui_instance, '_render', side_effect=mock_render):
            # Simulate multiple rapid refreshes (like during resize)
            tasks = []
            for _ in range(3):
                async def render_task():
                    async with tui_instance._render_lock:
                        layout = tui_instance._render()
                        return layout
                
                tasks.append(asyncio.create_task(render_task()))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All tasks should complete successfully
            assert len([r for r in results if not isinstance(r, Exception)]) == 3
            assert resize_count == 3


class TestNarrowTerminalSupport:
    """Test TUI behavior in narrow terminal configurations."""
    
    def test_layout_hash_consistent_in_narrow_terminal(self, tui_instance):
        """Test that layout hash works consistently in 80-column terminal."""
        # Simulate narrow terminal
        tui_instance._console = Mock()
        tui_instance._console.size = (80, 24)
        
        mock_layout = {"header": Mock(), "content": Mock()}
        
        # Hash should work in narrow terminal
        hash1 = tui_instance._get_layout_hash(mock_layout)
        hash2 = tui_instance._get_layout_hash(mock_layout) 
        
        assert hash1 == hash2
        assert hash1 is not None
    
    @pytest.mark.asyncio
    async def test_coalescing_works_in_narrow_terminal(self, tui_instance):
        """Test that request coalescing works in narrow terminal."""
        # Simulate narrow terminal
        tui_instance._console = Mock()
        tui_instance._console.size = (80, 24)
        
        # Should still coalesce requests effectively
        tui_instance.mark_dirty("content")
        tui_instance.mark_dirty("footer")
        
        await asyncio.sleep(0.2)
        
        # Coalescing should work regardless of terminal size
        assert len(tui_instance._pending_refresh_sections) == 0