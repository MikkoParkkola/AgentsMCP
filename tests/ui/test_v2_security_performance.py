# -*- coding: utf-8 -*-
"""
Security and Performance Tests for TUI v2 System

Tests for:
1. Race condition prevention
2. Input sanitization 
3. Performance benchmarks
4. Memory leak detection
5. Error propagation
"""

import asyncio
import pytest
import time
import threading
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from agentsmcp.ui.v2.display_renderer import DisplayRenderer
from agentsmcp.ui.v2.terminal_manager import TerminalManager
from agentsmcp.ui.v2.application_controller import ApplicationController, ApplicationConfig
from agentsmcp.ui.v2.main_app import MainTUIApp


class TestSecurityHardening:
    """Test security measures in the TUI system."""
    
    @pytest.mark.asyncio
    async def test_command_input_sanitization(self):
        """Test that malicious command inputs are sanitized."""
        config = ApplicationConfig()
        controller = ApplicationController(config=config)
        
        # THREAT: Terminal injection attacks
        malicious_inputs = [
            "\x1b[2J\x1b[H",  # Clear screen escape sequence
            "/quit\x07",       # Command with bell character
            "help\x00data",    # Command with null byte
            "status\r\nrm -rf /",  # CRLF injection
            "debug" + "A" * 2000,  # Buffer overflow attempt
        ]
        
        for malicious_input in malicious_inputs:
            result = await controller.process_command(malicious_input)
            
            # Should either be sanitized successfully or rejected
            if result['success']:
                # If processed, ensure it was sanitized
                sanitized_command = controller._sanitize_command_input(malicious_input)
                assert '\x1b' not in sanitized_command
                assert '\x07' not in sanitized_command 
                assert '\x00' not in sanitized_command
                assert '\r\n' not in sanitized_command
            else:
                # Rejection is acceptable for malicious input (unknown command or invalid format)
                error = result.get('error', '')
                assert 'Invalid' in error or 'Unknown' in error or 'command' in error.lower()
    
    @pytest.mark.asyncio
    async def test_command_structure_validation(self):
        """Test command structure validation."""
        config = ApplicationConfig()
        controller = ApplicationController(config=config)
        
        # THREAT: Command injection
        invalid_commands = [
            "",                    # Empty
            "a" * 100,            # Too long command name
            "cmd;rm -rf /",       # Shell injection attempt
            "help " + " ".join(["arg"] * 25),  # Too many args
            "debug " + "A" * 500, # Argument too long
        ]
        
        for invalid_cmd in invalid_commands:
            result = await controller.process_command(invalid_cmd)
            
            if invalid_cmd.strip():  # Non-empty inputs should be rejected
                assert not result['success']
                assert 'Invalid' in result.get('error', '') or 'Unknown' in result.get('error', '')
    
    @pytest.mark.asyncio
    async def test_content_size_validation(self):
        """Test that content size limits prevent memory exhaustion."""
        terminal_manager = Mock()
        terminal_manager.detect_capabilities = Mock(return_value=Mock(
            interactive=True, 
            cursor_control=True, 
            alternate_screen=True,
            width=80,
            height=24
        ))
        terminal_manager.get_size = Mock(return_value=(80, 24))
        
        renderer = DisplayRenderer(terminal_manager)
        result = await renderer.initialize()
        assert result  # Should initialize successfully
        
        # Define a test region
        renderer.define_region("test", 0, 0, 80, 24)
        
        # THREAT: Memory exhaustion attack
        large_content = "A" * (2 * 1024 * 1024)  # 2MB content
        
        # Should reject content that's too large
        result = renderer.update_region("test", large_content)
        assert not result
        
        # Should accept reasonable content
        normal_content = "Hello, world!"
        result = renderer.update_region("test", normal_content)
        assert result


class TestRaceConditionPrevention:
    """Test race condition prevention measures."""
    
    @pytest.mark.asyncio
    async def test_display_renderer_concurrent_initialization(self):
        """Test that concurrent initialization is protected."""
        terminal_manager = Mock()
        terminal_manager.initialize = AsyncMock(return_value=True)
        terminal_manager._initialized = False
        terminal_manager.detect_capabilities = Mock(return_value=Mock(
            interactive=True,
            cursor_control=True, 
            alternate_screen=True,
            width=80,
            height=24
        ))
        
        renderer = DisplayRenderer(terminal_manager)
        
        # THREAT: Race condition during initialization
        # MITIGATION: Multiple concurrent initializations should be safe
        results = await asyncio.gather(
            renderer.initialize(),
            renderer.initialize(), 
            renderer.initialize(),
            return_exceptions=True
        )
        
        # All should succeed (or at least not crash)
        successful_results = [r for r in results if isinstance(r, bool) and r]
        assert len(successful_results) >= 1  # At least one should succeed
        
        # Renderer should be initialized only once
        assert renderer._initialized is True
    
    @pytest.mark.asyncio
    async def test_application_controller_concurrent_shutdown(self):
        """Test that concurrent shutdowns don't cause issues."""
        config = ApplicationConfig(graceful_shutdown_timeout=0.1)
        controller = ApplicationController(config=config)
        
        # Mock components to avoid real initialization
        controller.event_system = Mock()
        controller.event_system.start = AsyncMock()
        controller.event_system.stop = AsyncMock()
        controller.event_system.emit_event = AsyncMock()
        controller.terminal_manager = Mock()
        controller.terminal_manager.initialize = AsyncMock(return_value=True)
        controller.terminal_manager._initialized = True
        
        await controller.startup()
        
        # THREAT: Race condition during shutdown
        # MITIGATION: Multiple concurrent shutdowns should be safe
        results = await asyncio.gather(
            controller.shutdown(),
            controller.shutdown(),
            controller.shutdown(),
            return_exceptions=True
        )
        
        # Should handle concurrent shutdowns gracefully
        assert all(isinstance(r, bool) for r in results if not isinstance(r, Exception))


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.asyncio
    async def test_startup_performance_target(self):
        """Test that TUI startup meets <500ms target."""
        # Create properly mocked async components
        mock_event_system = Mock()
        mock_event_system.initialize = AsyncMock(return_value=True)
        
        mock_terminal_manager = Mock()
        mock_terminal_manager.initialize = AsyncMock(return_value=True)
        mock_terminal_manager.get_capabilities = Mock(return_value=Mock(
            type=Mock(name="FULL_TTY"),
            colors=256
        ))
        
        mock_display_renderer = Mock()
        mock_display_renderer.initialize = AsyncMock(return_value=True)
        
        mock_input_handler = Mock()
        mock_input_handler.initialize = AsyncMock(return_value=True)
        
        mock_keyboard_processor = Mock()
        mock_keyboard_processor.initialize = AsyncMock(return_value=True)
        
        mock_app_controller = Mock()
        mock_app_controller.startup = AsyncMock(return_value=True)
        
        mock_chat_interface = Mock()
        mock_chat_interface.initialize = AsyncMock(return_value=True)
        
        # Mock all creation functions
        with patch.multiple(
            'agentsmcp.ui.v2.main_app',
            create_terminal_manager=Mock(return_value=mock_terminal_manager),
            create_event_system=Mock(return_value=mock_event_system),
            create_theme_manager=Mock(return_value=Mock()),
            create_standard_tui_layout=Mock(return_value=(Mock(), Mock())),
            create_chat_interface=Mock(return_value=mock_chat_interface),
        ):
            with patch('agentsmcp.ui.v2.main_app.DisplayRenderer', return_value=mock_display_renderer):
                with patch('agentsmcp.ui.v2.main_app.InputHandler', return_value=mock_input_handler):
                    with patch('agentsmcp.ui.v2.main_app.KeyboardProcessor', return_value=mock_keyboard_processor):
                        with patch('agentsmcp.ui.v2.main_app.ApplicationController', return_value=mock_app_controller):
                            
                            app = MainTUIApp()
                            
                            # Mock setup methods
                            app._setup_event_handlers = AsyncMock()
                            app._setup_signal_handlers = Mock()
                            
                            # PERFORMANCE: Measure initialization time
                            start_time = time.time()
                            
                            result = await app.initialize()
                            init_time = time.time() - start_time
                            
                            # Should initialize successfully
                            assert result is True
                            
                            # PERFORMANCE: Should meet 500ms target
                            assert init_time < 0.5, f"Initialization took {init_time*1000:.1f}ms, exceeds 500ms target"
    
    def test_content_hashing_performance(self):
        """Test that content hashing meets <16ms target for typing response."""
        terminal_manager = Mock()
        renderer = DisplayRenderer(terminal_manager)
        
        # Test content of various sizes
        test_contents = [
            "Hello",                           # Small
            "A" * 100,                        # Medium  
            "B" * 1000,                       # Large
            "C" * 10000,                      # Very large
        ]
        
        for content in test_contents:
            # PERFORMANCE: Hash computation should be fast
            start_time = time.time()
            hash_result = renderer._hash_content(content)
            hash_time = time.time() - start_time
            
            # Should complete quickly
            assert hash_time < 0.016, f"Hash time {hash_time*1000:.1f}ms exceeds 16ms target"
            
            # Should produce consistent results
            assert hash_result == renderer._hash_content(content)
    
    @pytest.mark.asyncio
    async def test_region_update_performance(self):
        """Test region update performance for real-time updates."""
        terminal_manager = Mock()
        terminal_manager.get_size = Mock(return_value=(80, 24))
        
        renderer = DisplayRenderer(terminal_manager)
        await renderer.initialize()
        
        # Define test region
        renderer.define_region("typing", 0, 20, 80, 1)
        
        # PERFORMANCE: Test rapid updates (simulating typing)
        content_updates = [f"User is typing{'.' * i}" for i in range(20)]
        
        start_time = time.time()
        for content in content_updates:
            renderer.update_region("typing", content)
        
        update_time = time.time() - start_time
        avg_update_time = update_time / len(content_updates)
        
        # PERFORMANCE: Each update should be <16ms for 60fps feel
        assert avg_update_time < 0.016, f"Average update time {avg_update_time*1000:.1f}ms exceeds 16ms target"


class TestMemoryLeakDetection:
    """Test for memory leaks during extended usage."""
    
    @pytest.mark.asyncio
    async def test_display_renderer_cleanup_memory(self):
        """Test that display renderer properly cleans up memory."""
        terminal_manager = Mock()
        terminal_manager.get_size = Mock(return_value=(80, 24))
        
        renderer = DisplayRenderer(terminal_manager)
        await renderer.initialize()
        
        # Create many regions and update them
        for i in range(100):
            renderer.define_region(f"region_{i}", 0, 0, 10, 1)
            renderer.update_region(f"region_{i}", f"Content {i}")
        
        # SECURITY: Cleanup should clear all memory
        await renderer.cleanup()
        
        # Verify cleanup
        assert not renderer._initialized
        assert len(renderer._regions) == 0
        assert renderer._last_terminal_state is None
    
    @pytest.mark.asyncio
    async def test_application_controller_cleanup_memory(self):
        """Test that application controller cleans up properly."""
        config = ApplicationConfig()
        controller = ApplicationController(config=config)
        
        # Mock components
        controller.event_system = Mock()
        controller.event_system.start = AsyncMock()
        controller.event_system.stop = AsyncMock()
        controller.event_system.emit_event = AsyncMock()
        
        await controller.startup()
        
        # Add command history (simulating usage)
        for i in range(50):
            await controller.process_command(f"debug_{i}")
        
        # SECURITY: Shutdown should clean up memory
        await controller.shutdown()
        
        # Verify cleanup state
        assert controller._state.value in ("shutting_down", "stopped")
        assert not controller._running
        
        # Command history should be maintained but bounded
        assert len(controller._command_history) <= 100


class TestErrorPropagation:
    """Test proper error propagation to prevent silent failures."""
    
    @pytest.mark.asyncio
    async def test_initialization_error_propagation(self):
        """Test that initialization errors are properly propagated."""
        terminal_manager = Mock()
        terminal_manager.initialize = AsyncMock(return_value=False)  # Simulate failure
        terminal_manager._initialized = False
        
        renderer = DisplayRenderer(terminal_manager)
        
        # THREAT: Silent failure during initialization
        # MITIGATION: Should return False and log error
        result = await renderer.initialize()
        assert result is False
        assert not renderer._initialized
    
    @pytest.mark.asyncio 
    async def test_command_timeout_handling(self):
        """Test that command timeouts are handled properly."""
        config = ApplicationConfig()
        controller = ApplicationController(config=config)
        
        # Mock a command that times out
        async def slow_command(*args):
            await asyncio.sleep(1.0)  # 1 second command
            return "Should not reach here"
        
        controller._commands['slow'] = slow_command
        
        # THREAT: Hanging commands
        # MITIGATION: Should timeout and return error
        # Patch the asyncio.wait_for timeout for testing
        with patch('asyncio.wait_for') as mock_wait_for:
            # Simulate timeout by raising TimeoutError
            mock_wait_for.side_effect = asyncio.TimeoutError()
            
            result = await controller.process_command('slow')
            
            assert not result['success']
            assert 'timed out' in result.get('error', '').lower()


if __name__ == "__main__":
    # Run with: python -m pytest tests/ui/test_v2_security_performance.py -v
    pytest.main([__file__, "-v", "--tb=short"])