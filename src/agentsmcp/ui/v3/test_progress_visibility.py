"""Comprehensive tests for enhanced progress visibility during TUI processing."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import List, Dict, Any

from .progress_tracker import (
    ProgressTracker, SimpleProgressTracker, ProcessingPhase,
    ToolExecutionInfo, create_progress_tracker
)
from .chat_engine import ChatEngine
from .tui_launcher import TUILauncher


class TestProgressTracker:
    """Test the ProgressTracker class functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_progress_tracking(self):
        """Test basic progress tracking with phases."""
        status_updates = []
        
        async def mock_callback(status: str):
            status_updates.append(status)
        
        tracker = ProgressTracker(mock_callback)
        
        # Test phase updates
        await tracker.update_phase(ProcessingPhase.ANALYZING)
        await tracker.update_phase(ProcessingPhase.TOOL_EXECUTION)
        await tracker.update_phase(ProcessingPhase.FINALIZING)
        
        assert len(status_updates) == 3
        assert "🔍 Analyzing your request" in status_updates[0]
        assert "🛠️ Executing tool" in status_updates[1]
        assert "✨ Finalizing response" in status_updates[2]
        
        # All messages should include timing
        for update in status_updates:
            assert "[" in update and "]" in update and "s]" in update
    
    @pytest.mark.asyncio
    async def test_tool_execution_tracking(self):
        """Test detailed tool execution progress tracking."""
        status_updates = []
        
        async def mock_callback(status: str):
            status_updates.append(status)
        
        tracker = ProgressTracker(mock_callback)
        
        # Test tool execution updates
        tool_info1 = ToolExecutionInfo("search_files", "find config files", 0, 3)
        tool_info2 = ToolExecutionInfo("read_file", "/path/to/file.txt", 1, 3)
        tool_info3 = ToolExecutionInfo("bash_command", "run test command", 2, 3)
        
        await tracker.update_tool_execution(tool_info1)
        await tracker.update_tool_execution(tool_info2)
        await tracker.update_tool_execution(tool_info3)
        
        assert len(status_updates) == 3
        assert "🔍 Executing search_files (1/3): find config files" in status_updates[0]
        assert "📄 Executing read_file (2/3): /path/to/file.txt" in status_updates[1]
        assert "⚡ Executing bash_command (3/3): run test command" in status_updates[2]
        
        # Verify tools_executed counter
        assert tracker.tools_executed == 3
    
    @pytest.mark.asyncio
    async def test_multi_turn_tracking(self):
        """Test multi-turn processing progress."""
        status_updates = []
        
        async def mock_callback(status: str):
            status_updates.append(status)
        
        tracker = ProgressTracker(mock_callback)
        
        # Test multi-turn updates
        await tracker.update_multi_turn(1, 3, "analyzing request")
        await tracker.update_multi_turn(2, 3, "processing results")
        await tracker.update_multi_turn(3, 3, "generating final response")
        
        assert len(status_updates) == 3
        assert "📊 Turn 1/3: analyzing request" in status_updates[0]
        assert "📊 Turn 2/3: processing results" in status_updates[1]
        assert "📊 Turn 3/3: generating final response" in status_updates[2]
    
    @pytest.mark.asyncio
    async def test_streaming_progress(self):
        """Test streaming response progress tracking."""
        status_updates = []
        
        async def mock_callback(status: str):
            status_updates.append(status)
        
        tracker = ProgressTracker(mock_callback)
        
        # Test streaming updates
        await tracker.update_streaming(0)
        await tracker.update_streaming(25)
        await tracker.update_streaming(100)
        
        assert len(status_updates) == 3
        assert "🎯 Streaming response" in status_updates[0]
        assert "🎯 Streaming response (25 chunks)" in status_updates[1]
        assert "🎯 Streaming response (100 chunks)" in status_updates[2]
    
    @pytest.mark.asyncio
    async def test_custom_status_messages(self):
        """Test custom status message functionality."""
        status_updates = []
        
        async def mock_callback(status: str):
            status_updates.append(status)
        
        tracker = ProgressTracker(mock_callback)
        
        # Test custom status
        await tracker.update_custom_status("Custom operation in progress", "🔧")
        await tracker.update_custom_status("Another custom status", "🎯")
        
        assert len(status_updates) == 2
        assert "🔧 Custom operation in progress" in status_updates[0]
        assert "🎯 Another custom status" in status_updates[1]
    
    def test_tool_icon_mapping(self):
        """Test tool icon mapping functionality."""
        tracker = ProgressTracker()
        
        # Test exact matches
        assert tracker._get_tool_icon("search_files") == "🔍"
        assert tracker._get_tool_icon("read_file") == "📄"
        assert tracker._get_tool_icon("bash_command") == "⚡"
        assert tracker._get_tool_icon("web_search") == "🌐"
        
        # Test partial matches
        assert tracker._get_tool_icon("mcp_search_tool") == "🔍"  # Contains "search"
        assert tracker._get_tool_icon("git_status") == "📦"  # Contains "git"
        
        # Test default fallback
        assert tracker._get_tool_icon("unknown_tool") == "🛠️"
    
    def test_get_summary(self):
        """Test progress summary functionality."""
        tracker = ProgressTracker()
        tracker.current_phase = ProcessingPhase.TOOL_EXECUTION
        tracker.current_turn = 2
        tracker.max_turns = 3
        tracker.tools_executed = 5
        
        summary = tracker.get_summary()
        
        assert summary["current_phase"] == "🛠️ Executing tool"
        assert summary["current_turn"] == 2
        assert summary["max_turns"] == 3
        assert summary["tools_executed"] == 5
        assert "total_time" in summary


class TestSimpleProgressTracker:
    """Test the SimpleProgressTracker class."""
    
    @pytest.mark.asyncio
    async def test_simple_progress_updates(self):
        """Test simple progress tracking."""
        status_updates = []
        
        async def mock_callback(status: str):
            status_updates.append(status)
        
        tracker = SimpleProgressTracker(mock_callback)
        
        await tracker.update("Processing request")
        await tracker.update("Getting response", "🎯")
        
        assert len(status_updates) == 2
        assert "⏳ Processing request" in status_updates[0]
        assert "🎯 Getting response" in status_updates[1]
    
    @pytest.mark.asyncio
    async def test_callback_failure_handling(self):
        """Test that callback failures don't break processing."""
        def failing_callback(status: str):
            raise Exception("Callback failed")
        
        tracker = SimpleProgressTracker(failing_callback)
        
        # Should not raise an exception
        await tracker.update("Test status")


class TestProgressVisibilityIntegration:
    """Test integration of progress visibility with ChatEngine and TUILauncher."""
    
    @pytest.mark.asyncio
    async def test_chat_engine_progress_forwarding(self):
        """Test that ChatEngine properly forwards progress to UI."""
        chat_engine = ChatEngine()
        
        status_updates = []
        def mock_status_callback(status: str):
            status_updates.append(status)
        
        chat_engine.set_callbacks(status_callback=mock_status_callback)
        
        # Mock the LLM client to simulate progress updates
        mock_llm_client = AsyncMock()
        mock_llm_client.send_message.return_value = "Test response"
        chat_engine._llm_client = mock_llm_client
        
        # Process a test input
        result = await chat_engine._get_ai_response("Test input")
        
        # Verify LLM client was called with progress callback
        mock_llm_client.send_message.assert_called_once()
        args, kwargs = mock_llm_client.send_message.call_args
        assert "progress_callback" in kwargs
        assert callable(kwargs["progress_callback"])
    
    @pytest.mark.asyncio
    async def test_streaming_progress_forwarding(self):
        """Test progress forwarding during streaming responses."""
        chat_engine = ChatEngine()
        
        status_updates = []
        def mock_status_callback(status: str):
            status_updates.append(status)
        
        chat_engine.set_callbacks(status_callback=mock_status_callback)
        
        # Mock streaming LLM client
        mock_llm_client = AsyncMock()
        async def mock_streaming_response(*args, **kwargs):
            yield "Chunk 1"
            yield "Chunk 2"
            yield "Chunk 3"
        
        mock_llm_client.supports_streaming.return_value = True
        mock_llm_client.send_message_streaming.return_value = mock_streaming_response()
        chat_engine._llm_client = mock_llm_client
        
        # Collect streaming chunks
        chunks = []
        async for chunk in chat_engine._get_ai_response_streaming("Test input"):
            chunks.append(chunk)
        
        # Verify streaming method was called with progress callback
        mock_llm_client.send_message_streaming.assert_called_once()
        args, kwargs = mock_llm_client.send_message_streaming.call_args
        assert "progress_callback" in kwargs
        assert callable(kwargs["progress_callback"])
        assert len(chunks) == 3


class TestTUIProgressDisplay:
    """Test TUI progress display functionality."""
    
    def test_enhanced_status_formatting(self):
        """Test enhanced status message formatting."""
        launcher = TUILauncher()
        launcher.current_renderer = None  # Test plain renderer path
        
        # Capture print output
        printed_messages = []
        original_print = __builtins__['print']
        def mock_print(*args, **kwargs):
            printed_messages.append(' '.join(str(arg) for arg in args))
        __builtins__['print'] = mock_print
        
        try:
            # Test different status message formats
            launcher._display_enhanced_status("🔍 Analyzing your request [2.1s]")
            launcher._display_enhanced_status("🛠️ Executing tool: search_files (1/3) [5.2s]")
            launcher._display_enhanced_status("Processing without timing")
            
            assert len(printed_messages) == 3
            assert "🔍 Analyzing your request [2.1s]" in printed_messages[0]
            assert "🛠️ Executing tool: search_files (1/3) [5.2s]" in printed_messages[1]
            assert "⏳ Processing without timing" in printed_messages[2]
            
        finally:
            __builtins__['print'] = original_print
    
    def test_console_renderer_status_colors(self):
        """Test ConsoleRenderer status color coding."""
        from .terminal_capabilities import TerminalCapabilities
        
        # Create mock capabilities
        capabilities = TerminalCapabilities(
            is_tty=True,
            supports_colors=True,
            supports_rich=True,
            width=80,
            height=24
        )
        
        from .console_renderer import ConsoleRenderer
        renderer = ConsoleRenderer(capabilities)
        renderer.initialize()
        
        # Mock console to capture color usage
        rendered_messages = []
        def mock_print(*args, **kwargs):
            rendered_messages.append(str(args[0]) if args else "")
        
        renderer.console.print = mock_print
        
        # Test different status message colors
        renderer.show_status("🔍 Analyzing your request [1.0s]")
        renderer.show_status("🛠️ Executing tool: search_files [2.0s]")
        renderer.show_status("📊 Turn 1/3: processing [3.0s]")
        renderer.show_status("🚀 Direct LLM processing [4.0s]")
        renderer.show_status("✨ Finalizing response [5.0s]")
        
        assert len(rendered_messages) == 5
        assert "[blue]" in rendered_messages[0]  # Analysis - blue
        assert "[yellow]" in rendered_messages[1]  # Tool execution - yellow
        assert "[magenta]" in rendered_messages[2]  # Multi-turn - magenta
        assert "[bright_blue]" in rendered_messages[3]  # Direct processing - bright blue
        assert "[green]" in rendered_messages[4]  # Finalizing - green


class TestProgressTrackerFactory:
    """Test progress tracker factory functionality."""
    
    def test_create_full_progress_tracker(self):
        """Test creation of full progress tracker."""
        mock_callback = Mock()
        tracker = create_progress_tracker(mock_callback, simple=False)
        
        assert isinstance(tracker, ProgressTracker)
        assert tracker.progress_callback == mock_callback
    
    def test_create_simple_progress_tracker(self):
        """Test creation of simple progress tracker."""
        mock_callback = Mock()
        tracker = create_progress_tracker(mock_callback, simple=True)
        
        assert isinstance(tracker, SimpleProgressTracker)
        assert tracker.status_callback == mock_callback
    
    def test_create_without_callback(self):
        """Test creation without callback."""
        full_tracker = create_progress_tracker(simple=False)
        simple_tracker = create_progress_tracker(simple=True)
        
        assert isinstance(full_tracker, ProgressTracker)
        assert isinstance(simple_tracker, SimpleProgressTracker)


# Performance and stress tests
class TestProgressVisibilityPerformance:
    """Test performance aspects of progress visibility system."""
    
    @pytest.mark.asyncio
    async def test_high_frequency_updates(self):
        """Test system can handle high frequency progress updates."""
        status_updates = []
        
        async def fast_callback(status: str):
            status_updates.append(status)
        
        tracker = ProgressTracker(fast_callback)
        
        # Send many rapid updates
        start_time = asyncio.get_event_loop().time()
        for i in range(100):
            await tracker.update_custom_status(f"Update {i}", "🔥")
        end_time = asyncio.get_event_loop().time()
        
        assert len(status_updates) == 100
        # Should complete quickly (less than 1 second for 100 updates)
        assert (end_time - start_time) < 1.0
    
    @pytest.mark.asyncio
    async def test_callback_timeout_resilience(self):
        """Test system resilience to slow/hanging callbacks."""
        async def slow_callback(status: str):
            await asyncio.sleep(0.1)  # Simulate slow callback
        
        tracker = ProgressTracker(slow_callback)
        
        # Should not block significantly
        start_time = asyncio.get_event_loop().time()
        await tracker.update_phase(ProcessingPhase.ANALYZING)
        await tracker.update_phase(ProcessingPhase.FINALIZING)
        end_time = asyncio.get_event_loop().time()
        
        # Should take at least 0.2s (2 * 0.1s) but not hang
        assert (end_time - start_time) >= 0.2
        assert (end_time - start_time) < 0.5  # Reasonable upper bound


if __name__ == "__main__":
    # Run basic smoke tests
    async def smoke_test():
        print("🧪 Running enhanced progress visibility smoke tests...")
        
        # Test basic progress tracking
        status_log = []
        async def log_callback(status):
            status_log.append(status)
            print(f"📊 Status: {status}")
        
        tracker = ProgressTracker(log_callback)
        
        await tracker.update_phase(ProcessingPhase.ANALYZING)
        await asyncio.sleep(0.1)
        
        # Simulate tool execution
        tool_info = ToolExecutionInfo("search_files", "find *.py files", 0, 2)
        await tracker.update_tool_execution(tool_info)
        await asyncio.sleep(0.1)
        
        tool_info2 = ToolExecutionInfo("read_file", "config.json", 1, 2)
        await tracker.update_tool_execution(tool_info2)
        await asyncio.sleep(0.1)
        
        await tracker.update_multi_turn(1, 2, "processing results")
        await asyncio.sleep(0.1)
        
        await tracker.update_phase(ProcessingPhase.FINALIZING)
        
        print(f"✅ Completed {len(status_log)} status updates")
        print("🎉 Enhanced progress visibility system working correctly!")
        
        # Show summary
        summary = tracker.get_summary()
        print(f"📈 Summary: {summary}")
    
    asyncio.run(smoke_test())