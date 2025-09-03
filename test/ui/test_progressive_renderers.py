"""Test suite for progressive enhancement UI renderers."""

import pytest
from unittest.mock import Mock, patch, call
from src.agentsmcp.ui.v3 import (
    PlainCLIRenderer,
    SimpleTUIRenderer,
    RichTUIRenderer,
    TerminalCapabilities,
    UIState
)


@pytest.fixture
def minimal_capabilities():
    """Minimal terminal capabilities - no TTY, no colors."""
    return TerminalCapabilities(
        is_tty=False,
        width=80,
        height=24,
        supports_colors=False,
        supports_unicode=False,
        supports_rich=False,
        is_fast_terminal=False,
        max_refresh_rate=10,
        force_plain=True,
        force_simple=False
    )


@pytest.fixture
def basic_capabilities():
    """Basic TTY with colors but no Rich support."""
    return TerminalCapabilities(
        is_tty=True,
        width=80,
        height=24,
        supports_colors=True,
        supports_unicode=True,
        supports_rich=False,
        is_fast_terminal=True,
        max_refresh_rate=30,
        force_plain=False,
        force_simple=True
    )


@pytest.fixture
def full_capabilities():
    """Full terminal capabilities including Rich support."""
    return TerminalCapabilities(
        is_tty=True,
        width=120,
        height=40,
        supports_colors=True,
        supports_unicode=True,
        supports_rich=True,
        is_fast_terminal=True,
        max_refresh_rate=60,
        force_plain=False,
        force_simple=False
    )


@pytest.fixture
def ui_state():
    """Basic UI state for testing."""
    return UIState(
        current_input="",
        is_processing=False,
        status_message="",
        messages=[]
    )


class TestPlainCLIRenderer:
    """Test PlainCLIRenderer functionality."""
    
    def test_initialization_success(self, minimal_capabilities, ui_state):
        """Test successful initialization of plain CLI renderer."""
        with patch('builtins.print') as mock_print:
            renderer = PlainCLIRenderer(minimal_capabilities)
            renderer.state = ui_state
            
            assert renderer.initialize() is True
            
            # Check initialization output
            expected_calls = [
                call("🤖 AI Command Composer - Plain Text Mode"),
                call("=" * 50),
                call("Commands: /quit, /help, /clear"),
                call()
            ]
            mock_print.assert_has_calls(expected_calls)
    
    def test_initialization_failure(self, minimal_capabilities, ui_state):
        """Test initialization failure handling."""
        with patch('builtins.print', side_effect=Exception("Print failed")):
            renderer = PlainCLIRenderer(minimal_capabilities)
            renderer.state = ui_state
            
            assert renderer.initialize() is False
    
    def test_cleanup(self, minimal_capabilities, ui_state):
        """Test renderer cleanup."""
        with patch('builtins.print') as mock_print:
            renderer = PlainCLIRenderer(minimal_capabilities)
            renderer.state = ui_state
            
            renderer.cleanup()
            
            mock_print.assert_any_call()
            mock_print.assert_any_call("Goodbye! 👋")
    
    def test_render_frame_with_input(self, minimal_capabilities, ui_state):
        """Test rendering with current input."""
        with patch('builtins.print') as mock_print:
            renderer = PlainCLIRenderer(minimal_capabilities)
            renderer.state = ui_state
            renderer.state.current_input = "hello world"
            
            renderer.render_frame()
            
            mock_print.assert_called_with("💬 > hello world", end="", flush=True)
    
    def test_handle_input_normal(self, minimal_capabilities, ui_state):
        """Test normal input handling."""
        with patch('builtins.input', return_value="test message"):
            with patch('builtins.print'):
                renderer = PlainCLIRenderer(minimal_capabilities)
                renderer.state = ui_state
                
                result = renderer.handle_input()
                
                assert result == "test message"
                assert renderer.state.current_input == ""
    
    def test_handle_input_empty(self, minimal_capabilities, ui_state):
        """Test empty input handling."""
        with patch('builtins.input', return_value=""):
            with patch('builtins.print'):
                renderer = PlainCLIRenderer(minimal_capabilities)
                renderer.state = ui_state
                
                result = renderer.handle_input()
                
                assert result is None
    
    def test_handle_input_keyboard_interrupt(self, minimal_capabilities, ui_state):
        """Test keyboard interrupt handling."""
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            with patch('builtins.print'):
                renderer = PlainCLIRenderer(minimal_capabilities)
                renderer.state = ui_state
                
                result = renderer.handle_input()
                
                assert result == "/quit"
    
    def test_show_message_info(self, minimal_capabilities, ui_state):
        """Test showing info message."""
        with patch('builtins.print') as mock_print:
            renderer = PlainCLIRenderer(minimal_capabilities)
            renderer.state = ui_state
            
            renderer.show_message("Test info", "info")
            
            mock_print.assert_called_with("ℹ️ Test info")
    
    def test_show_message_error(self, minimal_capabilities, ui_state):
        """Test showing error message."""
        with patch('builtins.print') as mock_print:
            renderer = PlainCLIRenderer(minimal_capabilities)
            renderer.state = ui_state
            
            renderer.show_error("Test error")
            
            mock_print.assert_called_with("❌ Test error")
    
    def test_processing_state_blocks_input(self, minimal_capabilities, ui_state):
        """Test that processing state blocks input handling."""
        renderer = PlainCLIRenderer(minimal_capabilities)
        renderer.state = ui_state
        renderer.state.is_processing = True
        
        result = renderer.handle_input()
        
        assert result is None


class TestSimpleTUIRenderer:
    """Test SimpleTUIRenderer functionality."""
    
    def test_initialization_with_tty(self, basic_capabilities, ui_state):
        """Test initialization with TTY support."""
        with patch('builtins.print') as mock_print:
            renderer = SimpleTUIRenderer(basic_capabilities)
            renderer.state = ui_state
            
            assert renderer.initialize() is True
            
            # Should clear screen and draw header
            mock_print.assert_any_call("\033[2J\033[H", end="")
    
    def test_initialization_failure(self, basic_capabilities, ui_state):
        """Test initialization failure handling."""
        with patch('builtins.print', side_effect=Exception("Init failed")):
            renderer = SimpleTUIRenderer(basic_capabilities)
            renderer.state = ui_state
            
            assert renderer.initialize() is False
    
    def test_cleanup_with_tty(self, basic_capabilities, ui_state):
        """Test cleanup with TTY support."""
        with patch('builtins.print') as mock_print:
            renderer = SimpleTUIRenderer(basic_capabilities)
            renderer.state = ui_state
            
            renderer.cleanup()
            
            # Should clear screen and show goodbye
            mock_print.assert_any_call("\033[2J\033[H", end="")
            mock_print.assert_any_call("Goodbye! 👋")
    
    def test_render_frame_with_tty(self, basic_capabilities, ui_state):
        """Test frame rendering with TTY."""
        with patch('builtins.print') as mock_print:
            renderer = SimpleTUIRenderer(basic_capabilities)
            renderer.state = ui_state
            renderer.state.current_input = "test"
            
            renderer.render_frame()
            
            # Should use ANSI escape sequences
            calls = [call.args[0] for call in mock_print.call_args_list]
            assert any("\033[s" in str(call) for call in calls)  # Save cursor
    
    @patch('select.select')
    def test_handle_input_enter_key(self, mock_select, basic_capabilities, ui_state):
        """Test handling Enter key press."""
        mock_select.return_value = ([True], [], [])
        
        with patch('sys.stdin.read', return_value=chr(13)):  # Enter key
            renderer = SimpleTUIRenderer(basic_capabilities)
            renderer.state = ui_state
            renderer._input_buffer = "test message"
            
            result = renderer.handle_input()
            
            assert result == "test message"
            assert renderer._input_buffer == ""
            assert renderer.state.current_input == ""
    
    @patch('select.select')
    def test_handle_input_backspace(self, mock_select, basic_capabilities, ui_state):
        """Test handling backspace key."""
        mock_select.return_value = ([True], [], [])
        
        with patch('sys.stdin.read', return_value=chr(127)):  # Backspace
            renderer = SimpleTUIRenderer(basic_capabilities)
            renderer.state = ui_state
            renderer._input_buffer = "test"
            renderer._cursor_pos = 4
            
            result = renderer.handle_input()
            
            assert result is None
            assert renderer._input_buffer == "tes"
            assert renderer._cursor_pos == 3
    
    @patch('select.select')
    def test_handle_input_regular_char(self, mock_select, basic_capabilities, ui_state):
        """Test handling regular character input."""
        mock_select.return_value = ([True], [], [])
        
        with patch('sys.stdin.read', return_value='a'):
            renderer = SimpleTUIRenderer(basic_capabilities)
            renderer.state = ui_state
            renderer._input_buffer = "test"
            renderer._cursor_pos = 2
            
            result = renderer.handle_input()
            
            assert result is None
            assert renderer._input_buffer == "teast"
            assert renderer._cursor_pos == 3
            assert renderer.state.current_input == "teast"
    
    def test_show_message_with_tty(self, basic_capabilities, ui_state):
        """Test showing message with TTY."""
        with patch('builtins.print') as mock_print:
            renderer = SimpleTUIRenderer(basic_capabilities)
            renderer.state = ui_state
            
            renderer.show_message("Test message", "success")
            
            # Should position cursor and show message
            calls = [str(call) for call in mock_print.call_args_list]
            assert any("✅ Test message" in call for call in calls)
    
    def test_fallback_to_plain_input(self, basic_capabilities, ui_state):
        """Test fallback to plain input when TTY unavailable."""
        # Override TTY capability
        basic_capabilities.is_tty = False
        
        with patch('builtins.input', return_value="fallback test"):
            renderer = SimpleTUIRenderer(basic_capabilities)
            renderer.state = ui_state
            
            result = renderer.handle_input()
            
            assert result == "fallback test"


class TestRichTUIRenderer:
    """Test RichTUIRenderer functionality."""
    
    def test_initialization_without_rich_support(self, basic_capabilities, ui_state):
        """Test that renderer fails without Rich support."""
        # Ensure Rich support is disabled
        basic_capabilities.supports_rich = False
        
        renderer = RichTUIRenderer(basic_capabilities)
        renderer.state = ui_state
        
        assert renderer.initialize() is False
    
    @patch('rich.console.Console')
    @patch('rich.live.Live')
    def test_initialization_with_rich_support(self, mock_live, mock_console, full_capabilities, ui_state):
        """Test successful initialization with Rich support."""
        renderer = RichTUIRenderer(full_capabilities)
        renderer.state = ui_state
        
        result = renderer.initialize()
        
        assert result is True
        assert mock_console.called
        assert mock_live.called
    
    def test_initialization_failure(self, full_capabilities, ui_state):
        """Test initialization failure handling."""
        with patch('rich.console.Console', side_effect=Exception("Rich failed")):
            renderer = RichTUIRenderer(full_capabilities)
            renderer.state = ui_state
            
            assert renderer.initialize() is False
    
    @patch('rich.live.Live')
    def test_cleanup(self, mock_live, full_capabilities, ui_state):
        """Test renderer cleanup."""
        mock_live_instance = Mock()
        mock_live.return_value = mock_live_instance
        
        with patch('rich.console.Console'):
            renderer = RichTUIRenderer(full_capabilities)
            renderer.state = ui_state
            renderer.initialize()
            
            renderer.cleanup()
            
            mock_live_instance.stop.assert_called_once()
    
    @patch('rich.live.Live')
    @patch('rich.console.Console')
    def test_render_frame(self, mock_console, mock_live, full_capabilities, ui_state):
        """Test frame rendering."""
        mock_live_instance = Mock()
        mock_live.return_value = mock_live_instance
        
        renderer = RichTUIRenderer(full_capabilities)
        renderer.state = ui_state
        renderer.initialize()
        
        renderer.render_frame()
        
        mock_live_instance.refresh.assert_called()
    
    @patch('select.select')
    @patch('rich.live.Live')
    @patch('rich.console.Console')
    def test_handle_input_enter(self, mock_console, mock_live, mock_select, full_capabilities, ui_state):
        """Test handling Enter key in Rich TUI."""
        mock_select.return_value = ([True], [], [])
        
        with patch('sys.stdin.read', return_value=chr(13)):
            renderer = RichTUIRenderer(full_capabilities)
            renderer.state = ui_state
            renderer._input_buffer = "rich test"
            
            result = renderer.handle_input()
            
            assert result == "rich test"
            assert renderer._input_buffer == ""
            assert renderer.state.current_input == ""
    
    @patch('rich.live.Live')
    @patch('rich.console.Console')  
    def test_show_message(self, mock_console, mock_live, full_capabilities, ui_state):
        """Test showing message in Rich TUI."""
        renderer = RichTUIRenderer(full_capabilities)
        renderer.state = ui_state
        
        renderer.show_message("Rich message", "warning")
        
        assert "[yellow]Rich message[/yellow]" in renderer.state.status_message
    
    @patch('rich.live.Live')
    @patch('rich.console.Console')
    def test_message_rendering_with_messages(self, mock_console, mock_live, full_capabilities, ui_state):
        """Test rendering messages in Rich TUI."""
        renderer = RichTUIRenderer(full_capabilities)
        renderer.state = ui_state
        renderer.state.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = renderer._render_messages()
        
        # Check that the Text object contains our messages
        assert "Hello" in str(result)
        assert "Hi there!" in str(result)
    
    @patch('rich.live.Live')
    @patch('rich.console.Console')
    def test_message_rendering_empty(self, mock_console, mock_live, full_capabilities, ui_state):
        """Test rendering with no messages."""
        renderer = RichTUIRenderer(full_capabilities)
        renderer.state = ui_state
        
        result = renderer._render_messages()
        
        assert "No messages yet" in str(result)


class TestRendererIntegration:
    """Integration tests for renderer selection and fallbacks."""
    
    def test_renderer_imports(self):
        """Test that all renderers can be imported successfully."""
        from src.agentsmcp.ui.v3 import (
            PlainCLIRenderer,
            SimpleTUIRenderer, 
            RichTUIRenderer
        )
        
        assert PlainCLIRenderer is not None
        assert SimpleTUIRenderer is not None
        assert RichTUIRenderer is not None
    
    def test_all_renderers_inherit_from_base(self):
        """Test that all renderers inherit from UIRenderer."""
        from src.agentsmcp.ui.v3 import UIRenderer
        
        minimal_caps = TerminalCapabilities(
            is_tty=False,
            width=80,
            height=24,
            supports_colors=False,
            supports_unicode=False,
            supports_rich=False,
            is_fast_terminal=False,
            max_refresh_rate=10,
            force_plain=True,
            force_simple=False
        )
        
        plain_renderer = PlainCLIRenderer(minimal_caps)
        simple_renderer = SimpleTUIRenderer(minimal_caps)
        rich_renderer = RichTUIRenderer(minimal_caps)
        
        assert isinstance(plain_renderer, UIRenderer)
        assert isinstance(simple_renderer, UIRenderer)
        assert isinstance(rich_renderer, UIRenderer)
    
    def test_progressive_enhancement_selection(self):
        """Test that appropriate renderers are selected based on capabilities."""
        # Minimal environment should work with Plain
        minimal_caps = TerminalCapabilities(
            is_tty=False,
            width=80,
            height=24,
            supports_colors=False,
            supports_unicode=False,
            supports_rich=False,
            is_fast_terminal=False,
            max_refresh_rate=10,
            force_plain=True,
            force_simple=False
        )
        plain_renderer = PlainCLIRenderer(minimal_caps)
        assert plain_renderer.capabilities.is_tty is False
        
        # TTY environment should work with Simple
        basic_caps = TerminalCapabilities(
            is_tty=True,
            width=80,
            height=24,
            supports_colors=True,
            supports_unicode=True,
            supports_rich=False,
            is_fast_terminal=True,
            max_refresh_rate=30,
            force_plain=False,
            force_simple=True
        )
        simple_renderer = SimpleTUIRenderer(basic_caps)
        assert simple_renderer.capabilities.is_tty is True
        assert simple_renderer.capabilities.supports_rich is False
        
        # Full environment should work with Rich
        full_caps = TerminalCapabilities(
            is_tty=True,
            width=120,
            height=40,
            supports_colors=True,
            supports_unicode=True,
            supports_rich=True,
            is_fast_terminal=True,
            max_refresh_rate=60,
            force_plain=False,
            force_simple=False
        )
        rich_renderer = RichTUIRenderer(full_caps)
        assert rich_renderer.capabilities.supports_rich is True


if __name__ == "__main__":
    pytest.main([__file__])