#!/usr/bin/env python3
"""Test streaming integration in TUI components."""

import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Add src to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine, MessageRole
from agentsmcp.ui.v3.tui_launcher import TUILauncher
from agentsmcp.ui.v3.console_renderer import ConsoleRenderer
from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
from agentsmcp.ui.v3.terminal_capabilities import TerminalCapabilities


class MockLLMClient:
    """Mock LLM client for testing streaming functionality."""
    
    def __init__(self, supports_streaming=True, streaming_chunks=None):
        self.supports_streaming_flag = supports_streaming
        self.streaming_chunks = streaming_chunks or ["Hello", " there!", " How can I help?"]
        self.conversation_history = []
    
    def supports_streaming(self) -> bool:
        return self.supports_streaming_flag
    
    async def send_message_streaming(self, message: str):
        """Mock streaming response."""
        self.conversation_history.append({"role": "user", "content": message})
        for chunk in self.streaming_chunks:
            yield chunk
            await asyncio.sleep(0.01)  # Simulate network delay
    
    async def send_message(self, message: str) -> str:
        """Mock non-streaming response."""
        self.conversation_history.append({"role": "user", "content": message})
        return "".join(self.streaming_chunks)


class TestStreamingChatEngine:
    """Test streaming functionality in ChatEngine."""
    
    @pytest.mark.asyncio
    async def test_streaming_detection(self):
        """Test that streaming capability is properly detected."""
        engine = ChatEngine()
        
        # Mock LLM client with streaming support
        mock_client = MockLLMClient(supports_streaming=True)
        engine._llm_client = mock_client
        
        should_stream = await engine._should_use_streaming()
        assert should_stream is True
        
        # Test without streaming support
        mock_client.supports_streaming_flag = False
        should_stream = await engine._should_use_streaming()
        assert should_stream is False
    
    @pytest.mark.asyncio
    async def test_streaming_response_generation(self):
        """Test that streaming responses are properly generated."""
        engine = ChatEngine()
        mock_client = MockLLMClient(
            supports_streaming=True,
            streaming_chunks=["Hello", " world", "!"]
        )
        engine._llm_client = mock_client
        
        # Collect streaming chunks
        chunks = []
        async for chunk in engine._get_ai_response_streaming("test message"):
            chunks.append(chunk)
        
        assert chunks == ["Hello", " world", "!"]
        assert len(mock_client.conversation_history) == 1
        assert mock_client.conversation_history[0]["content"] == "test message"
    
    @pytest.mark.asyncio
    async def test_streaming_fallback(self):
        """Test fallback to non-streaming when streaming is not supported."""
        engine = ChatEngine()
        mock_client = MockLLMClient(supports_streaming=False)
        mock_client.send_message = AsyncMock(return_value="Non-streaming response")
        engine._llm_client = mock_client
        
        chunks = []
        async for chunk in engine._get_ai_response_streaming("test message"):
            chunks.append(chunk)
        
        assert chunks == ["Non-streaming response"]
        mock_client.send_message.assert_called_once_with("test message")
    
    @pytest.mark.asyncio
    async def test_streaming_message_handling(self):
        """Test complete streaming message handling workflow."""
        engine = ChatEngine()
        mock_client = MockLLMClient(
            supports_streaming=True,
            streaming_chunks=["Stream", "ing", " test"]
        )
        engine._llm_client = mock_client
        
        # Track callbacks
        status_updates = []
        messages = []
        
        def status_callback(status):
            status_updates.append(status)
        
        def message_callback(message):
            messages.append(message)
        
        engine.set_callbacks(
            status_callback=status_callback,
            message_callback=message_callback
        )
        
        # Process streaming message
        result = await engine.process_input("Hello")
        
        # Verify results
        assert result is True
        assert len(messages) >= 2  # User message + AI message
        assert messages[0].role == MessageRole.USER
        assert messages[0].content == "Hello"
        
        # Check that streaming updates were sent
        streaming_updates = [s for s in status_updates if s.startswith("streaming_update:")]
        assert len(streaming_updates) > 0
        
        # Verify final message content
        ai_message = messages[-1]
        assert ai_message.role == MessageRole.ASSISTANT
        assert ai_message.content == "Streaming test"


class TestStreamingRenderers:
    """Test streaming support in renderers."""
    
    def test_console_renderer_streaming_init(self):
        """Test ConsoleRenderer streaming initialization."""
        capabilities = TerminalCapabilities(
            is_tty=True,
            width=80,
            height=24,
            supports_colors=True,
            supports_unicode=True,
            supports_rich=True,
            is_fast_terminal=True,
            max_refresh_rate=60,
            force_plain=False,
            force_simple=False
        )
        
        renderer = ConsoleRenderer(capabilities)
        renderer.initialize()
        
        assert hasattr(renderer, '_streaming_active')
        assert hasattr(renderer, '_current_streaming_content')
        assert renderer._streaming_active is False
        assert renderer._current_streaming_content == ""
    
    def test_plain_cli_renderer_streaming_init(self):
        """Test PlainCLIRenderer streaming initialization."""
        capabilities = TerminalCapabilities(
            is_tty=True,
            width=80,
            height=24,
            supports_colors=False,
            supports_unicode=True,
            supports_rich=False,
            is_fast_terminal=True,
            max_refresh_rate=60,
            force_plain=False,
            force_simple=False
        )
        
        renderer = PlainCLIRenderer(capabilities)
        renderer.initialize()
        
        assert hasattr(renderer, '_streaming_active')
        assert hasattr(renderer, '_current_streaming_content')
        assert renderer._streaming_active is False
        assert renderer._current_streaming_content == ""
    
    @patch('builtins.print')
    def test_console_renderer_streaming_updates(self, mock_print):
        """Test ConsoleRenderer streaming update handling."""
        capabilities = TerminalCapabilities(
            is_tty=True,
            width=80,
            height=24,
            supports_colors=True,
            supports_unicode=True,
            supports_rich=True,
            is_fast_terminal=True,
            max_refresh_rate=60,
            force_plain=False,
            force_simple=False
        )
        
        renderer = ConsoleRenderer(capabilities)
        renderer.initialize()
        
        # Test streaming updates
        renderer.handle_streaming_update("Hello")
        assert renderer._streaming_active is True
        assert renderer._current_streaming_content == "Hello"
        
        renderer.handle_streaming_update("Hello world")
        assert renderer._current_streaming_content == "Hello world"
        
        # Test finalization
        renderer.display_chat_message("assistant", "Hello world", "12:00:00")
        assert renderer._streaming_active is False
        assert renderer._current_streaming_content == ""
    
    @patch('builtins.print')
    def test_plain_cli_renderer_streaming_updates(self, mock_print):
        """Test PlainCLIRenderer streaming update handling."""
        capabilities = TerminalCapabilities(
            is_tty=True,
            width=80,
            height=24,
            supports_colors=False,
            supports_unicode=True,
            supports_rich=False,
            is_fast_terminal=True,
            max_refresh_rate=60,
            force_plain=False,
            force_simple=False
        )
        
        renderer = PlainCLIRenderer(capabilities)
        renderer.initialize()
        
        # Test streaming updates
        renderer.handle_streaming_update("Hello")
        assert renderer._streaming_active is True
        assert renderer._current_streaming_content == "Hello"
        
        # Verify print was called with streaming format
        mock_print.assert_called()
        args = mock_print.call_args
        assert "🤖 AI (streaming):" in str(args)
        
        renderer.handle_streaming_update("Hello world")
        assert renderer._current_streaming_content == "Hello world"
        
        # Test finalization
        renderer.display_chat_message("assistant", "Hello world", "12:00:00")
        assert renderer._streaming_active is False


class TestStreamingTUILauncher:
    """Test streaming support in TUILauncher."""
    
    def test_streaming_update_handling(self):
        """Test that TUILauncher properly handles streaming updates."""
        launcher = TUILauncher()
        launcher.current_renderer = MagicMock()
        launcher.current_renderer.handle_streaming_update = MagicMock()
        
        # Test streaming update callback
        launcher._on_status_change("streaming_update:Hello world")
        
        launcher.current_renderer.handle_streaming_update.assert_called_once_with("Hello world")
    
    def test_streaming_fallback(self):
        """Test fallback when renderer doesn't support streaming."""
        launcher = TUILauncher()
        launcher.current_renderer = MagicMock()
        # Remove streaming method to simulate unsupported renderer
        del launcher.current_renderer.handle_streaming_update
        
        with patch('builtins.print') as mock_print:
            launcher._handle_streaming_update("Test content")
            mock_print.assert_called()
            args = mock_print.call_args
            assert "🤖 AI:" in str(args)
    
    def test_non_streaming_status_handling(self):
        """Test that non-streaming status updates are handled normally."""
        launcher = TUILauncher()
        launcher.current_renderer = MagicMock()
        launcher.current_renderer.show_status = MagicMock()
        
        # Test regular status update
        launcher._on_status_change("Processing message...")
        
        launcher.current_renderer.show_status.assert_called_once_with("Processing message...")


async def test_end_to_end_streaming():
    """Test complete end-to-end streaming workflow."""
    # Create components
    engine = ChatEngine()
    
    # Mock LLM client with realistic streaming
    mock_client = MockLLMClient(
        supports_streaming=True,
        streaming_chunks=["I", " can", " help", " you", " with", " that", "!"]
    )
    engine._llm_client = mock_client
    
    # Track all updates
    all_updates = []
    
    def capture_callback(status):
        all_updates.append(status)
    
    engine.set_callbacks(status_callback=capture_callback)
    
    # Process message
    await engine.process_input("Hello AI")
    
    # Verify streaming updates were generated
    streaming_updates = [u for u in all_updates if u.startswith("streaming_update:")]
    assert len(streaming_updates) > 0
    
    # Verify complete message was built
    final_content = streaming_updates[-1].replace("streaming_update:", "")
    assert final_content == "I can help you with that!"
    
    # Verify conversation state
    assert len(engine.state.messages) == 2
    assert engine.state.messages[0].role == MessageRole.USER
    assert engine.state.messages[1].role == MessageRole.ASSISTANT
    assert engine.state.messages[1].content == "I can help you with that!"


def run_tests():
    """Run all streaming integration tests."""
    print("🧪 Testing Streaming Response Integration...")
    
    # Test ChatEngine streaming
    print("\n1. Testing ChatEngine streaming detection...")
    asyncio.run(TestStreamingChatEngine().test_streaming_detection())
    print("✅ ChatEngine streaming detection works")
    
    print("\n2. Testing streaming response generation...")
    asyncio.run(TestStreamingChatEngine().test_streaming_response_generation())
    print("✅ Streaming response generation works")
    
    print("\n3. Testing streaming fallback...")
    asyncio.run(TestStreamingChatEngine().test_streaming_fallback())
    print("✅ Streaming fallback works")
    
    print("\n4. Testing complete message handling...")
    asyncio.run(TestStreamingChatEngine().test_streaming_message_handling())
    print("✅ Complete message handling works")
    
    # Test renderers
    print("\n5. Testing renderer streaming initialization...")
    TestStreamingRenderers().test_console_renderer_streaming_init()
    TestStreamingRenderers().test_plain_cli_renderer_streaming_init()
    print("✅ Renderer streaming initialization works")
    
    print("\n6. Testing renderer streaming updates...")
    TestStreamingRenderers().test_console_renderer_streaming_updates()
    TestStreamingRenderers().test_plain_cli_renderer_streaming_updates()
    print("✅ Renderer streaming updates work")
    
    # Test TUILauncher
    print("\n7. Testing TUILauncher streaming support...")
    TestStreamingTUILauncher().test_streaming_update_handling()
    TestStreamingTUILauncher().test_streaming_fallback()
    TestStreamingTUILauncher().test_non_streaming_status_handling()
    print("✅ TUILauncher streaming support works")
    
    # End-to-end test
    print("\n8. Testing end-to-end streaming workflow...")
    asyncio.run(test_end_to_end_streaming())
    print("✅ End-to-end streaming workflow works")
    
    print("\n🎉 All streaming integration tests passed!")
    print("\n📊 Expected User Experience After Implementation:")
    print("Before: > Hello, what can you help with?")
    print("        ⏳ Processing your message...")
    print("        [Long wait with no updates]")
    print("        🤖 AI: I can help you with many tasks...")
    print("\nAfter:  > Hello, what can you help with?")
    print("        🤖 AI (streaming): I can help")
    print("        🤖 AI (streaming): I can help you with")
    print("        🤖 AI (streaming): I can help you with many tasks including...")
    print("        [Real-time response building]")


if __name__ == "__main__":
    run_tests()