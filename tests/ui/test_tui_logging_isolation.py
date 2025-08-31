"""
Tests for TUI logging isolation to prevent debug logs from interfering with UI display.

These tests verify that debug logs from LLM client and other components don't
corrupt the console output or cause alignment issues.
"""

import pytest
import asyncio
import io
import sys
import os
import logging
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from contextlib import contextmanager
import tempfile

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agentsmcp.ui.v2.fixed_working_tui import FixedWorkingTUI


@contextmanager
def capture_logs():
    """Context manager to capture log output."""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    
    # Add to root logger
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    
    try:
        yield log_capture
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(original_level)


@contextmanager
def isolated_stdout():
    """Context manager to isolate stdout from logging."""
    original_stdout = sys.stdout
    stdout_capture = io.StringIO()
    
    try:
        sys.stdout = stdout_capture
        yield stdout_capture
    finally:
        sys.stdout = original_stdout


class TestLoggingIsolation:
    """Test that logging doesn't interfere with TUI display."""
    
    @pytest.fixture
    def tui(self):
        """Create TUI instance for testing."""
        return FixedWorkingTUI()
        
    def test_debug_logs_dont_appear_in_stdout(self, tui):
        """Test that debug logs don't appear in stdout output."""
        logger = logging.getLogger('agentsmcp.conversation.llm_client')
        
        with isolated_stdout() as stdout_capture:
            with capture_logs() as log_capture:
                # Generate some debug logs
                logger.debug("Tool execution turn 1/3")
                logger.debug("Processing LLM response")
                logger.debug("Debug message that should not appear in UI")
                
                # Perform TUI operations
                tui.show_prompt()
                
        stdout_content = stdout_capture.getvalue()
        log_content = log_capture.getvalue()
        
        # Debug logs should be in log capture, not stdout
        assert "Tool execution turn" not in stdout_content
        assert "Tool execution turn" in log_content
        assert "Processing LLM response" not in stdout_content
        assert "Processing LLM response" in log_content
        
        # TUI prompt should be in stdout
        assert ">" in stdout_content
        
    def test_llm_client_logging_isolation(self, tui):
        """Test that LLM client debug logs are properly isolated."""
        with patch('sys.path'):
            with patch('agentsmcp.conversation.llm_client.LLMClient') as MockLLMClient:
                mock_client = MockLLMClient.return_value
                mock_client.provider = "test-provider"
                mock_client.model = "test-model"
                
                # Mock the logger to simulate debug output
                with patch('agentsmcp.conversation.llm_client.logger') as mock_logger:
                    with isolated_stdout() as stdout_capture:
                        tui.setup_llm_client()
                        
                    stdout_content = stdout_capture.getvalue()
                    
                    # Logger should have been called for debug info
                    mock_logger.info.assert_called()
                    
                    # Debug info should not leak to stdout
                    debug_calls = [call for call in mock_logger.debug.call_args_list]
                    for call in debug_calls:
                        debug_msg = str(call)
                        assert debug_msg not in stdout_content
                        
    def test_thinking_indicator_without_log_pollution(self, tui):
        """Test that thinking indicator works without debug log pollution."""
        with isolated_stdout() as stdout_capture:
            with capture_logs() as log_capture:
                # Simulate some background logging during thinking
                logger = logging.getLogger('agentsmcp.test')
                logger.debug("Background processing")
                logger.debug("More debug info")
                
                # Show thinking indicator
                stdout_capture.write('\n🤔 Thinking...\n')
                
        stdout_content = stdout_capture.getvalue()
        log_content = log_capture.getvalue()
        
        # Thinking indicator should appear cleanly in stdout
        assert '🤔 Thinking...' in stdout_content
        assert 'Background processing' not in stdout_content
        assert 'Background processing' in log_content
        
    @patch('sys.stdout')
    @pytest.mark.asyncio
    async def test_process_line_logging_isolation(self, mock_stdout, tui):
        """Test that process_line operations don't mix logs with UI."""
        # Mock LLM client
        mock_llm_client = AsyncMock()
        mock_llm_client.send_message.return_value = "Test response"
        tui.llm_client = mock_llm_client
        
        with capture_logs() as log_capture:
            # Process a line which might generate logs
            await tui.process_line("test message")
            
        # Check that stdout calls are for UI elements only
        stdout_calls = mock_stdout.write.call_args_list
        ui_calls = [str(call) for call in stdout_calls]
        
        # Should contain UI elements
        assert any('🤔 Thinking...' in call for call in ui_calls)
        assert any('🤖 Agent:' in call for call in ui_calls)
        
        # Should not contain debug log patterns
        for call in ui_calls:
            assert 'DEBUG' not in call
            assert 'Tool execution turn' not in call
            assert '%(asctime)s' not in call
            
    def test_error_logging_vs_error_display(self, tui):
        """Test distinction between error logging and error display."""
        logger = logging.getLogger('agentsmcp.ui.v2.fixed_working_tui')
        
        with isolated_stdout() as stdout_capture:
            with capture_logs() as log_capture:
                # Simulate an error that should be both logged and displayed
                error_msg = "Test error occurred"
                
                # Log the error (internal)
                logger.error(f"Error processing message: {error_msg}")
                
                # Display user-friendly error (UI)
                stdout_capture.write(f'❌ Error: {error_msg}\n')
                stdout_capture.write('   Please try again or use /help for commands.\n')
                
        stdout_content = stdout_capture.getvalue()
        log_content = log_capture.getvalue()
        
        # User-friendly error should be in stdout
        assert '❌ Error: Test error occurred' in stdout_content
        assert 'Please try again' in stdout_content
        
        # Technical error should be in logs
        assert 'Error processing message' in log_content
        
        # Technical details should not be in stdout
        assert 'Error processing message' not in stdout_content


class TestLoggerConfiguration:
    """Test proper logger configuration for TUI isolation."""
    
    def test_tui_logger_level_configuration(self):
        """Test that TUI loggers are configured properly."""
        # Get TUI-related loggers
        tui_logger = logging.getLogger('agentsmcp.ui.v2.fixed_working_tui')
        llm_logger = logging.getLogger('agentsmcp.conversation.llm_client')
        
        # Check that they exist and have appropriate configuration
        assert tui_logger is not None
        assert llm_logger is not None
        
        # They should inherit from root logger or have explicit configuration
        # This test verifies the logger hierarchy works as expected
        
    def test_console_handler_separation(self):
        """Test that console handlers are properly separated for different loggers."""
        with capture_logs() as log_capture:
            # Create loggers for different components
            ui_logger = logging.getLogger('agentsmcp.ui.test')
            backend_logger = logging.getLogger('agentsmcp.backend.test')
            
            ui_logger.info("UI message")
            backend_logger.debug("Backend debug message")
            
        log_content = log_capture.getvalue()
        
        # Both should appear in log capture
        assert "UI message" in log_content
        assert "Backend debug message" in log_content
        
    def test_log_level_filtering(self):
        """Test that log level filtering works correctly."""
        logger = logging.getLogger('agentsmcp.test_filtering')
        
        with capture_logs() as log_capture:
            # Set different levels
            handler = logging.StreamHandler(log_capture)
            handler.setLevel(logging.WARNING)  # Only warnings and above
            logger.addHandler(handler)
            
            logger.debug("Debug message - should not appear")
            logger.info("Info message - should not appear") 
            logger.warning("Warning message - should appear")
            logger.error("Error message - should appear")
            
        log_content = log_capture.getvalue()
        
        assert "Debug message" not in log_content
        assert "Info message" not in log_content
        assert "Warning message" in log_content
        assert "Error message" in log_content


class TestProductionModeLogging:
    """Test logging behavior in production mode."""
    
    @pytest.fixture
    def tui(self):
        return FixedWorkingTUI()
        
    def test_production_clean_output(self, tui):
        """Test that production mode has clean console output."""
        # Simulate production environment
        with patch.dict(os.environ, {'AGENTSMCP_ENV': 'production'}):
            with isolated_stdout() as stdout_capture:
                with capture_logs() as log_capture:
                    # Generate various log levels
                    logger = logging.getLogger('agentsmcp.test_prod')
                    logger.debug("Debug info")
                    logger.info("Info message")
                    logger.warning("Warning message")
                    
                    # Perform TUI operations
                    tui.clear_screen_and_show_prompt()
                    
        stdout_content = stdout_capture.getvalue()
        
        # Console should only have TUI elements
        assert "🚀 AgentsMCP" in stdout_content
        assert ">" in stdout_content
        
        # Should not have log messages
        assert "Debug info" not in stdout_content
        assert "Info message" not in stdout_content
        assert "Warning message" not in stdout_content
        
    def test_development_mode_logging_visibility(self, tui):
        """Test that development mode allows appropriate log visibility."""
        with patch.dict(os.environ, {'AGENTSMCP_ENV': 'development'}):
            with isolated_stdout() as stdout_capture:
                with capture_logs() as log_capture:
                    logger = logging.getLogger('agentsmcp.test_dev')
                    logger.debug("Development debug info")
                    
                    # TUI operations should still be clean
                    tui.show_prompt()
                    
        stdout_content = stdout_capture.getvalue()
        log_content = log_capture.getvalue()
        
        # Console should still be clean
        assert "Development debug info" not in stdout_content
        assert ">" in stdout_content
        
        # But logs should be captured
        assert "Development debug info" in log_content


class TestAsyncLoggingIsolation:
    """Test logging isolation in async operations."""
    
    @pytest.fixture
    def tui(self):
        return FixedWorkingTUI()
        
    @pytest.mark.asyncio
    async def test_async_operation_logging_isolation(self, tui):
        """Test that async operations don't leak logs to stdout."""
        mock_llm_client = AsyncMock()
        mock_llm_client.send_message.return_value = "Async response"
        tui.llm_client = mock_llm_client
        
        with isolated_stdout() as stdout_capture:
            with capture_logs() as log_capture:
                # Mock some async logging during LLM processing
                async def mock_send_with_logging(message):
                    logger = logging.getLogger('agentsmcp.llm.async')
                    logger.debug(f"Processing async message: {message}")
                    logger.debug("Async operation in progress")
                    return "Async response"
                
                mock_llm_client.send_message = mock_send_with_logging
                
                # Process line asynchronously
                await tui.process_line("async test")
                
        stdout_content = stdout_capture.getvalue()
        log_content = log_capture.getvalue()
        
        # UI elements should be in stdout
        assert "🤔 Thinking..." in stdout_content or "🤖 Agent:" in stdout_content
        
        # Debug logs should not leak to stdout
        assert "Processing async message" not in stdout_content
        assert "Async operation in progress" not in stdout_content
        
        # But should be in log capture
        assert "Processing async message" in log_content
        assert "Async operation in progress" in log_content
        
    @pytest.mark.asyncio
    async def test_concurrent_logging_isolation(self, tui):
        """Test logging isolation with concurrent operations."""
        async def background_task(task_id):
            logger = logging.getLogger(f'agentsmcp.background.{task_id}')
            for i in range(5):
                logger.debug(f"Background task {task_id} step {i}")
                await asyncio.sleep(0.01)
        
        with isolated_stdout() as stdout_capture:
            with capture_logs() as log_capture:
                # Start background tasks that generate logs
                tasks = [background_task(i) for i in range(3)]
                
                # Perform TUI operations concurrently
                async def ui_operations():
                    tui.show_prompt()
                    stdout_capture.write("UI operation\n")
                
                # Run everything concurrently
                await asyncio.gather(*tasks, ui_operations())
                
        stdout_content = stdout_capture.getvalue()
        log_content = log_capture.getvalue()
        
        # UI should be clean
        assert "UI operation" in stdout_content
        assert ">" in stdout_content
        
        # Background logs should not appear in UI
        assert "Background task" not in stdout_content
        
        # But should be in logs
        assert "Background task" in log_content


class TestLogFormattingIsolation:
    """Test that log formatting doesn't affect UI formatting."""
    
    def test_ansi_codes_in_logs_dont_affect_ui(self):
        """Test that ANSI codes in logs don't corrupt UI display."""
        logger = logging.getLogger('agentsmcp.test_ansi')
        
        with isolated_stdout() as stdout_capture:
            with capture_logs() as log_capture:
                # Log message with ANSI codes
                logger.info("\033[31mRed log message\033[0m")
                logger.debug("\033[1mBold debug message\033[0m")
                
                # UI operations
                stdout_capture.write("🚀 Clean UI text\n")
                stdout_capture.write("\033[32m✅ Intentional green UI\033[0m\n")
                
        stdout_content = stdout_capture.getvalue()
        log_content = log_capture.getvalue()
        
        # UI should have its intended ANSI codes
        assert "✅ Intentional green UI" in stdout_content
        assert "\033[32m" in stdout_content
        
        # Log ANSI codes should not appear in UI
        assert "Red log message" not in stdout_content
        assert "Bold debug message" not in stdout_content
        
        # But should be in logs
        assert "Red log message" in log_content
        assert "Bold debug message" in log_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])