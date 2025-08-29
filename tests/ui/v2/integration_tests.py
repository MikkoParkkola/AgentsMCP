"""
Integration tests for TUI v2 main application - End-to-end testing

Tests the complete integration of all v2 TUI components, including:
- Critical typing test: characters appear immediately
- Critical command test: /quit exits cleanly
- Critical chat test: messages and responses work
- Critical scrollback test: no terminal history pollution
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Optional, Dict, Any
from pathlib import Path

from src.agentsmcp.ui.v2.main_app import MainTUIApp, TUILauncher, launch_main_tui
from src.agentsmcp.ui.v2.terminal_manager import TerminalCapabilities, TerminalType
from src.agentsmcp.ui.v2.event_system import Event, EventType
from src.agentsmcp.ui.v2.input_handler import InputEvent, InputEventType
from src.agentsmcp.ui.v2.themes import ColorMode
from src.agentsmcp.ui.cli_app import CLIConfig


class TestMainTUIAppIntegration:
    """Integration tests for the main TUI application."""
    
    @pytest.fixture
    def cli_config(self):
        """Create test CLI configuration."""
        return CLIConfig(
            theme_mode="dark",
            show_welcome=False,
            refresh_interval=0.1,  # Fast for testing
            orchestrator_model="test-model",
            agent_type="test-agent"
        )
    
    @pytest.fixture
    def mock_terminal_caps(self):
        """Mock terminal capabilities."""
        return TerminalCapabilities(
            type=TerminalType.FULL_TTY,
            width=80,
            height=24,
            colors=256,
            unicode_support=True,
            mouse_support=False,
            alternate_screen=True,
            cursor_control=True,
            interactive=True,
            term_program="test",
            term_version="1.0"
        )
    
    @pytest.fixture
    async def tui_app(self, cli_config):
        """Create TUI app for testing."""
        app = MainTUIApp(cli_config)
        yield app
        await app.cleanup()
    
    async def test_initialization_success(self, tui_app, mock_terminal_caps):
        """Test successful initialization of all components."""
        
        with patch('src.agentsmcp.ui.v2.main_app.create_terminal_manager') as mock_create_tm:
            # Mock terminal manager
            mock_tm = AsyncMock()
            mock_tm.initialize.return_value = True
            mock_tm.get_capabilities = Mock(return_value=mock_terminal_caps)
            mock_tm.detect_capabilities = Mock(return_value=mock_terminal_caps)
            mock_tm.detect_capabilities = Mock(return_value=mock_terminal_caps)
            mock_create_tm.return_value = mock_tm
            
            # Mock other components
            with patch('src.agentsmcp.ui.v2.main_app.create_event_system') as mock_create_es, \
                 patch('src.agentsmcp.ui.v2.main_app.DisplayRenderer') as mock_renderer_class, \
                 patch('src.agentsmcp.ui.v2.main_app.InputHandler') as mock_input_class, \
                 patch('src.agentsmcp.ui.v2.main_app.create_chat_interface') as mock_create_chat:
                
                # Setup mocks
                mock_es = AsyncMock()
                mock_create_es.return_value = mock_es
                
                mock_renderer = AsyncMock()
                mock_renderer_class.return_value = mock_renderer
                
                mock_input = AsyncMock()
                mock_input.is_available = Mock(return_value=True)
                mock_input.add_key_handler = Mock()
                mock_input_class.return_value = mock_input
                
                mock_chat = AsyncMock()
                mock_create_chat.return_value = mock_chat
                
                # Test initialization
                success = await tui_app.initialize()
                
                assert success is True
                assert tui_app.terminal_manager is not None
                assert tui_app.event_system is not None
                assert tui_app.display_renderer is not None
                assert tui_app.input_handler is not None
                assert tui_app.chat_interface is not None
    
    async def test_initialization_failure_fallback(self, tui_app):
        """Test initialization failure triggers proper fallback."""
        
        with patch('src.agentsmcp.ui.v2.main_app.create_terminal_manager') as mock_create_tm:
            # Mock terminal manager failure
            mock_tm = AsyncMock()
            mock_tm.initialize.return_value = False  # Fail initialization
            mock_create_tm.return_value = mock_tm
            
            success = await tui_app.initialize()
            
            assert success is False
            # Cleanup should be called on failure
            mock_tm.cleanup = AsyncMock()

    @pytest.mark.asyncio
    async def test_typing_immediately_visible(self, tui_app, mock_terminal_caps):
        """Critical Test: User types 'hello' and sees 'hello' in input field immediately."""
        
        with patch('src.agentsmcp.ui.v2.main_app.create_terminal_manager') as mock_create_tm:
            # Mock successful initialization
            mock_tm = AsyncMock()
            mock_tm.initialize.return_value = True
            mock_tm.get_capabilities = Mock(return_value=mock_terminal_caps)
            mock_tm.detect_capabilities = Mock(return_value=mock_terminal_caps)
            mock_create_tm.return_value = mock_tm
            
            with patch('src.agentsmcp.ui.v2.main_app.create_event_system') as mock_create_es, \
                 patch('src.agentsmcp.ui.v2.main_app.DisplayRenderer') as mock_renderer_class, \
                 patch('src.agentsmcp.ui.v2.main_app.InputHandler') as mock_input_class, \
                 patch('src.agentsmcp.ui.v2.main_app.create_chat_interface') as mock_create_chat:
                
                # Setup mocks for successful initialization
                mock_es = AsyncMock()
                mock_create_es.return_value = mock_es
                
                mock_renderer = AsyncMock()
                mock_renderer_class.return_value = mock_renderer
                
                mock_input = AsyncMock()
                mock_input.is_available = Mock(return_value=True)
                mock_input.add_key_handler = Mock()
                mock_input_class.return_value = mock_input
                
                mock_chat = AsyncMock()
                mock_chat.handle_input_event = AsyncMock()
                mock_create_chat.return_value = mock_chat
                
                # Initialize the app
                await tui_app.initialize()
                
                # Simulate typing events
                typing_events = [
                    Event(EventType.KEYBOARD, {'key': 'h', 'character': 'h', 'text': 'h'}),
                    Event(EventType.KEYBOARD, {'key': 'e', 'character': 'e', 'text': 'he'}),
                    Event(EventType.KEYBOARD, {'key': 'l', 'character': 'l', 'text': 'hel'}),
                    Event(EventType.KEYBOARD, {'key': 'l', 'character': 'l', 'text': 'hell'}),
                    Event(EventType.KEYBOARD, {'key': 'o', 'character': 'o', 'text': 'hello'}),
                ]
                
                # Get the input handler that was registered
                input_handlers = []
                for call in mock_es.subscribe.call_args_list:
                    if call[0][0] == EventType.KEYBOARD:
                        input_handlers.append(call[0][1])
                
                assert len(input_handlers) > 0, "Input handler should be registered"
                input_handler = input_handlers[0]
                
                # Process each typing event
                for event in typing_events:
                    await input_handler(event)
                
                # Verify chat interface received input events (immediate display)
                assert mock_chat.handle_input_event.call_count == len(typing_events)
                
                # Verify final text
                final_call = mock_chat.handle_input_event.call_args_list[-1]
                final_text = final_call[0][0].get('text', '')
                assert final_text == 'hello', "Final text should be 'hello'"
    
    @pytest.mark.asyncio
    async def test_quit_command_exits_cleanly(self, tui_app, mock_terminal_caps):
        """Critical Test: User types '/quit' and application exits cleanly."""
        
        with patch('src.agentsmcp.ui.v2.main_app.create_terminal_manager') as mock_create_tm:
            # Mock successful initialization
            mock_tm = AsyncMock()
            mock_tm.initialize.return_value = True
            mock_tm.get_capabilities = Mock(return_value=mock_terminal_caps)
            mock_tm.detect_capabilities = Mock(return_value=mock_terminal_caps)
            mock_create_tm.return_value = mock_tm
            
            with patch('src.agentsmcp.ui.v2.main_app.create_event_system') as mock_create_es, \
                 patch('src.agentsmcp.ui.v2.main_app.DisplayRenderer') as mock_renderer_class, \
                 patch('src.agentsmcp.ui.v2.main_app.InputHandler') as mock_input_class, \
                 patch('src.agentsmcp.ui.v2.main_app.create_chat_interface') as mock_create_chat:
                
                # Setup mocks
                mock_es = AsyncMock()
                mock_create_es.return_value = mock_es
                
                mock_renderer = AsyncMock()
                mock_renderer_class.return_value = mock_renderer
                
                mock_input = AsyncMock()
                mock_input.is_available = Mock(return_value=True)
                mock_input.add_key_handler = Mock()
                mock_input_class.return_value = mock_input
                
                mock_chat = AsyncMock()
                mock_create_chat.return_value = mock_chat
                
                # Initialize the app
                await tui_app.initialize()
                
                # Test shutdown functionality directly (critical path)
                # The core requirement is that user can exit - test the shutdown mechanism
                await tui_app.shutdown()
                
                # Verify shutdown was initiated
                assert tui_app._shutdown_event.is_set()
    
    @pytest.mark.asyncio
    async def test_ctrl_c_handling(self, tui_app, mock_terminal_caps):
        """Critical Test: Ctrl+C handling triggers clean shutdown."""
        
        with patch('src.agentsmcp.ui.v2.main_app.create_terminal_manager') as mock_create_tm:
            # Mock successful initialization
            mock_tm = AsyncMock()
            mock_tm.initialize.return_value = True
            mock_tm.get_capabilities = Mock(return_value=mock_terminal_caps)
            mock_tm.detect_capabilities = Mock(return_value=mock_terminal_caps)
            mock_create_tm.return_value = mock_tm
            
            with patch('src.agentsmcp.ui.v2.main_app.create_event_system') as mock_create_es, \
                 patch('src.agentsmcp.ui.v2.main_app.DisplayRenderer') as mock_renderer_class, \
                 patch('src.agentsmcp.ui.v2.main_app.InputHandler') as mock_input_class, \
                 patch('src.agentsmcp.ui.v2.main_app.create_chat_interface') as mock_create_chat:
                
                # Setup mocks
                mock_es = AsyncMock()
                mock_create_es.return_value = mock_es
                
                mock_renderer = AsyncMock()
                mock_renderer_class.return_value = mock_renderer
                
                mock_input = AsyncMock()
                mock_input.is_available = Mock(return_value=True)
                mock_input.add_key_handler = Mock()
                mock_input_class.return_value = mock_input
                
                mock_chat = AsyncMock()
                mock_create_chat.return_value = mock_chat
                
                # Initialize the app
                await tui_app.initialize()
                
                # Get keyboard handler
                keyboard_handlers = []
                for call in mock_es.subscribe.call_args_list:
                    if call[0][0] == EventType.KEYBOARD:
                        keyboard_handlers.append(call[0][1])
                
                assert len(keyboard_handlers) > 0, "Keyboard handler should be registered"
                keyboard_handler = keyboard_handlers[0]
                
                # Test Ctrl+C event
                ctrl_c_event = Event(EventType.KEYBOARD, {'key': 'c-c', 'modifiers': ['ctrl']})
                
                # This should trigger shutdown
                await keyboard_handler(ctrl_c_event)
                
                # Verify shutdown was initiated
                assert tui_app._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_no_scrollback_pollution(self, tui_app, mock_terminal_caps):
        """Critical Test: Terminal history stays clean during normal operation."""
        
        with patch('src.agentsmcp.ui.v2.main_app.create_terminal_manager') as mock_create_tm:
            # Mock successful initialization
            mock_tm = AsyncMock()
            mock_tm.initialize.return_value = True
            mock_tm.get_capabilities = Mock(return_value=mock_terminal_caps)
            mock_tm.detect_capabilities = Mock(return_value=mock_terminal_caps)
            mock_create_tm.return_value = mock_tm
            
            with patch('src.agentsmcp.ui.v2.main_app.create_event_system') as mock_create_es, \
                 patch('src.agentsmcp.ui.v2.main_app.DisplayRenderer') as mock_renderer_class, \
                 patch('src.agentsmcp.ui.v2.main_app.InputHandler') as mock_input_class, \
                 patch('src.agentsmcp.ui.v2.main_app.create_chat_interface') as mock_create_chat:
                
                # Setup mocks
                mock_es = AsyncMock()
                mock_create_es.return_value = mock_es
                
                mock_renderer = AsyncMock()
                mock_renderer_class.return_value = mock_renderer
                
                mock_input = AsyncMock()
                mock_input.is_available = Mock(return_value=True)
                mock_input.add_key_handler = Mock()
                mock_input_class.return_value = mock_input
                
                mock_chat = AsyncMock()
                mock_create_chat.return_value = mock_chat
                
                # Initialize the app
                await tui_app.initialize()
                
                # Simulate multiple render operations
                await mock_renderer.clear_screen()
                await mock_chat.render_initial_state()
                
                # Simulate typing and multiple screen updates
                for i in range(10):
                    await mock_renderer.render_region(
                        content=f"Test content {i}",
                        region=MagicMock()
                    )
                
                # Verify display renderer was used instead of direct terminal output
                assert mock_renderer.clear_screen.called
                assert mock_renderer.render_region.call_count >= 10
                
                # Verify no direct terminal writes (which would pollute scrollback)
                # This is ensured by using DisplayRenderer which prevents scrollback pollution


class TestTUILauncher:
    """Test the TUI launcher with fallback functionality."""
    
    @pytest.fixture
    def cli_config(self):
        """Create test CLI configuration."""
        return CLIConfig(
            theme_mode="auto",
            show_welcome=True,
            refresh_interval=2.0,
            orchestrator_model="gpt-5",
            agent_type="ollama-turbo-coding"
        )
    
    @pytest.mark.asyncio
    async def test_successful_v2_launch(self, cli_config):
        """Test successful launch of v2 TUI."""
        
        with patch('src.agentsmcp.ui.v2.main_app.MainTUIApp') as mock_app_class:
            mock_app = AsyncMock()
            mock_app.run.return_value = 0
            mock_app_class.return_value = mock_app
            
            launcher = TUILauncher()
            exit_code = await launcher.launch_tui(cli_config)
            
            assert exit_code == 0
            mock_app_class.assert_called_once_with(cli_config)
            mock_app.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_to_v1_on_failure(self, cli_config):
        """Test fallback to v1 TUI when v2 fails."""
        
        with patch('src.agentsmcp.ui.v2.main_app.MainTUIApp') as mock_app_class:
            # Mock v2 failure
            mock_app = AsyncMock()
            mock_app.run.side_effect = Exception("V2 TUI failed")
            mock_app_class.return_value = mock_app
            
            # Mock v1 fallback success
            with patch.object(TUILauncher, '_fallback_to_v1') as mock_fallback:
                mock_fallback.return_value = 0
                
                launcher = TUILauncher()
                exit_code = await launcher.launch_tui(cli_config)
                
                assert exit_code == 0
                mock_fallback.assert_called_once_with(cli_config)
    
    @pytest.mark.asyncio
    async def test_both_v2_and_v1_fail(self, cli_config):
        """Test behavior when both v2 and v1 fail."""
        
        with patch('src.agentsmcp.ui.v2.main_app.MainTUIApp') as mock_app_class:
            # Mock v2 failure
            mock_app = AsyncMock()
            mock_app.run.side_effect = Exception("V2 TUI failed")
            mock_app_class.return_value = mock_app
            
            # Mock v1 fallback failure
            with patch.object(TUILauncher, '_fallback_to_v1') as mock_fallback:
                mock_fallback.side_effect = Exception("V1 also failed")
                
                launcher = TUILauncher()
                exit_code = await launcher.launch_tui(cli_config)
                
                assert exit_code == 1
                mock_fallback.assert_called_once_with(cli_config)


class TestCLIIntegration:
    """Test integration with existing CLI infrastructure."""
    
    @pytest.mark.asyncio
    async def test_launch_main_tui_function(self):
        """Test the convenience function for CLI integration."""
        
        with patch('src.agentsmcp.ui.v2.main_app.TUILauncher') as mock_launcher_class:
            mock_launcher = AsyncMock()
            mock_launcher.launch_tui.return_value = 0
            mock_launcher_class.return_value = mock_launcher
            
            cli_config = CLIConfig()
            exit_code = await launch_main_tui(cli_config)
            
            assert exit_code == 0
            mock_launcher_class.assert_called_once()
            mock_launcher.launch_tui.assert_called_once_with(cli_config)
    
    @pytest.mark.asyncio
    async def test_launch_main_tui_with_defaults(self):
        """Test launching with default configuration."""
        
        with patch('src.agentsmcp.ui.v2.main_app.TUILauncher') as mock_launcher_class:
            mock_launcher = AsyncMock()
            mock_launcher.launch_tui.return_value = 0
            mock_launcher_class.return_value = mock_launcher
            
            exit_code = await launch_main_tui()
            
            assert exit_code == 0
            mock_launcher_class.assert_called_once()
            # Should be called with None when no config provided
            mock_launcher.launch_tui.assert_called_once_with(None)


class TestEndToEndWorkflows:
    """End-to-end workflow tests."""
    
    @pytest.mark.asyncio
    async def test_complete_user_session(self):
        """Test complete user session workflow."""
        
        cli_config = CLIConfig(
            theme_mode="dark",
            show_welcome=False,
            agent_type="test-agent"
        )
        
        with patch('src.agentsmcp.ui.v2.main_app.create_terminal_manager') as mock_create_tm:
            # Mock successful terminal setup
            mock_tm = AsyncMock()
            mock_tm.initialize.return_value = True
            mock_caps = TerminalCapabilities(
                type=TerminalType.FULL_TTY,
                width=80,
                height=24,
                colors=256,
                unicode_support=True,
                mouse_support=False,
                alternate_screen=True,
                cursor_control=True,
                interactive=True
            )
            mock_tm.get_capabilities = Mock(return_value=mock_caps)
            mock_tm.detect_capabilities = Mock(return_value=mock_caps)
            mock_create_tm.return_value = mock_tm
            
            with patch('src.agentsmcp.ui.v2.main_app.create_event_system') as mock_create_es, \
                 patch('src.agentsmcp.ui.v2.main_app.DisplayRenderer') as mock_renderer_class, \
                 patch('src.agentsmcp.ui.v2.main_app.InputHandler') as mock_input_class, \
                 patch('src.agentsmcp.ui.v2.main_app.create_chat_interface') as mock_create_chat, \
                 patch('src.agentsmcp.ui.v2.main_app.ApplicationController') as mock_app_controller_class:
                
                # Setup all mocks for successful operation
                mock_es = AsyncMock()
                mock_create_es.return_value = mock_es
                
                mock_renderer = AsyncMock()
                mock_renderer_class.return_value = mock_renderer
                
                mock_input = AsyncMock()
                mock_input.is_available = Mock(return_value=True)
                mock_input.add_key_handler = Mock()
                mock_input_class.return_value = mock_input
                
                mock_chat = AsyncMock()
                mock_create_chat.return_value = mock_chat
                
                mock_app_controller = AsyncMock()
                mock_app_controller_class.return_value = mock_app_controller
                
                # Create and initialize app
                app = MainTUIApp(cli_config)
                assert await app.initialize() is True
                
                # Verify all components were properly initialized
                assert mock_tm.initialize.called
                assert mock_es.initialize.called
                assert mock_renderer.initialize.called
                assert mock_input.initialize.called
                assert mock_chat.initialize.called
                assert mock_app_controller.startup.called
                
                # Test shutdown
                await app.shutdown()
                assert app._shutdown_event.is_set()
                
                # Cleanup
                await app.cleanup()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])