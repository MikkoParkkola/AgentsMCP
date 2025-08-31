"""
Pytest configuration and shared fixtures for TUI testing.

This file provides shared fixtures, test configuration, and utilities
for all TUI-related tests.
"""

import pytest
import asyncio
import sys
import os
import logging
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agentsmcp.ui.v2.fixed_working_tui import FixedWorkingTUI


# Configure pytest for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def tui_instance():
    """Create a fresh TUI instance for each test."""
    tui = FixedWorkingTUI()
    yield tui
    
    # Cleanup
    if hasattr(tui, 'restore_terminal'):
        try:
            tui.restore_terminal()
        except:
            pass


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    mock_client = AsyncMock()
    mock_client.provider = "test-provider"
    mock_client.model = "test-model"
    mock_client.send_message = AsyncMock(return_value="Test response")
    mock_client.clear_history = Mock()
    return mock_client


@pytest.fixture
def tui_with_mock_llm(tui_instance, mock_llm_client):
    """Create TUI instance with mock LLM client attached."""
    tui_instance.llm_client = mock_llm_client
    return tui_instance, mock_llm_client


@pytest.fixture
def capture_logs():
    """Fixture to capture logging output."""
    import io
    
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Add to root logger
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    
    yield log_capture
    
    # Cleanup
    root_logger.removeHandler(handler)
    root_logger.setLevel(original_level)


@pytest.fixture
def isolated_stdout():
    """Fixture to isolate stdout for testing."""
    import io
    
    original_stdout = sys.stdout
    stdout_capture = io.StringIO()
    
    sys.stdout = stdout_capture
    yield stdout_capture
    sys.stdout = original_stdout


@pytest.fixture
def mock_terminal():
    """Fixture providing a mock terminal for testing."""
    from .test_tui_utilities import TerminalSimulator
    return TerminalSimulator()


@pytest.fixture
def performance_benchmarker():
    """Fixture providing performance benchmarking."""
    from .test_tui_utilities import PerformanceBenchmarker
    return PerformanceBenchmarker()


# Test data fixtures
@pytest.fixture
def test_responses():
    """Provide test response data."""
    return [
        "Simple response",
        "Multi-line response\nWith second line\nAnd third line",
        "Response with code:\n```python\nprint('hello')\n```",
        "Very long response that might wrap across terminal lines. " * 5,
        "Response with special characters: !@#$%^&*() 🌟 你好",
        "",  # Empty response
        "   ",  # Whitespace only
        "Single word",
        "Response\nwith\nmany\nlines\nof\ncontent\nto\ntest\nformatting"
    ]


@pytest.fixture
def test_inputs():
    """Provide test input data."""
    return [
        "hello",
        "write a python function", 
        "/help",
        "/clear",
        "/quit",
        "",
        "   ",
        "input with special chars !@#$%^&*()",
        "very long input that might exceed normal terminal width boundaries and test wrapping behavior",
        "unicode input 🚀 你好 🌟",
        "multi\nline\ninput",
        "\t\ttabbed input",
        "input with\rcarriage return",
        "mixed\n\r\tspecial chars"
    ]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for TUI testing."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "alignment: marks tests for alignment issues"
    )
    config.addinivalue_line(
        "markers", "logging: marks tests for logging isolation"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests for integration scenarios" 
    )
    config.addinivalue_line(
        "markers", "performance: marks tests for performance validation"
    )
    config.addinivalue_line(
        "markers", "regression: marks tests for known regression scenarios"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for TUI-specific requirements."""
    for item in items:
        # Auto-mark tests based on file names
        if "alignment" in item.fspath.basename:
            item.add_marker(pytest.mark.alignment)
        if "logging" in item.fspath.basename:
            item.add_marker(pytest.mark.logging)
        if "integration" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)


# Utility functions for test setup
def setup_test_environment():
    """Setup test environment for TUI testing."""
    # Disable real terminal interactions during testing
    os.environ['AGENTSMCP_TEST_MODE'] = '1'
    
    # Configure logging for testing
    logging.getLogger('agentsmcp').setLevel(logging.DEBUG)


def cleanup_test_environment():
    """Cleanup test environment after TUI testing."""
    if 'AGENTSMCP_TEST_MODE' in os.environ:
        del os.environ['AGENTSMCP_TEST_MODE']


# Session-level setup and teardown
@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Setup test session for TUI testing."""
    setup_test_environment()
    yield
    cleanup_test_environment()


# Custom assertions for TUI testing
class TUIAssertions:
    """Custom assertions for TUI testing."""
    
    @staticmethod
    def assert_no_progressive_indentation(lines):
        """Assert that no progressive indentation exists in output lines."""
        from .test_tui_utilities import AlignmentAnalyzer
        
        analyzer = AlignmentAnalyzer()
        issues = analyzer.analyze_progressive_indentation(lines)
        
        assert len(issues) == 0, f"Progressive indentation detected: {issues}"
    
    @staticmethod
    def assert_consistent_response_formatting(lines):
        """Assert that response formatting is consistent."""
        from .test_tui_utilities import AlignmentAnalyzer
        
        analyzer = AlignmentAnalyzer()
        issues = analyzer.analyze_response_formatting(lines)
        
        assert len(issues) == 0, f"Inconsistent response formatting: {issues}"
    
    @staticmethod
    def assert_no_debug_logs_in_output(output_text):
        """Assert that debug logs don't appear in UI output."""
        debug_indicators = [
            'DEBUG',
            'Tool execution turn',
            '%(asctime)s',
            'logging.StreamHandler',
            'agentsmcp.conversation.llm_client:'
        ]
        
        for indicator in debug_indicators:
            assert indicator not in output_text, f"Debug log indicator '{indicator}' found in UI output"
    
    @staticmethod
    def assert_cursor_position_valid(cursor_row, cursor_col, terminal_width, terminal_height):
        """Assert that cursor position is valid."""
        assert 0 <= cursor_row < terminal_height, f"Cursor row {cursor_row} out of bounds (height: {terminal_height})"
        assert 0 <= cursor_col < terminal_width, f"Cursor column {cursor_col} out of bounds (width: {terminal_width})"
    
    @staticmethod
    def assert_conversation_flow_valid(lines):
        """Assert that conversation flow follows expected patterns."""
        from .test_tui_utilities import OutputPatternMatcher
        
        matcher = OutputPatternMatcher()
        analysis = matcher.verify_conversation_flow(lines)
        
        assert analysis['valid'], f"Invalid conversation flow: {analysis['issues']}"


@pytest.fixture
def tui_assertions():
    """Provide TUI assertion utilities."""
    return TUIAssertions()


# Test execution helpers
def run_tui_test(test_func, *args, **kwargs):
    """Run a TUI test with proper setup and teardown."""
    setup_test_environment()
    try:
        return test_func(*args, **kwargs)
    finally:
        cleanup_test_environment()


def run_async_tui_test(async_test_func, *args, **kwargs):
    """Run an async TUI test with proper setup and teardown."""
    async def wrapped_test():
        setup_test_environment()
        try:
            return await async_test_func(*args, **kwargs)
        finally:
            cleanup_test_environment()
    
    return asyncio.run(wrapped_test())


# Export utilities for use in tests
__all__ = [
    'TUIAssertions',
    'run_tui_test', 
    'run_async_tui_test',
    'setup_test_environment',
    'cleanup_test_environment'
]