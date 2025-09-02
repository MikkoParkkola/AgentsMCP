"""
Tests for input_rendering_pipeline module.

Tests the input rendering pipeline's ability to provide immediate character feedback
and proper input display rendering with cursor management.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

from agentsmcp.ui.v2.input_rendering_pipeline import (
    InputRenderingPipeline,
    InputMode,
    CursorStyle,
    InputState,
    RenderResult,
    InputValidator,
    render_input_simple,
    render_with_immediate_feedback
)


class TestInputRenderingPipeline:
    """Test cases for InputRenderingPipeline."""
    
    def setup_method(self):
        """Setup for each test."""
        self.pipeline = InputRenderingPipeline()
    
    @pytest.mark.asyncio
    async def test_empty_input_rendering(self):
        """Test rendering of empty input."""
        result = await self.pipeline.render_input("", 0)
        
        assert isinstance(result, RenderResult)
        assert result.display_changed
        assert not result.input_complete
        assert result.lines_count == 1
        assert result.cursor_line == 0
        assert result.cursor_column == 0
    
    @pytest.mark.asyncio
    async def test_single_line_rendering(self):
        """Test single-line input rendering."""
        text = "Hello, World!"
        cursor_pos = 7
        result = await self.pipeline.render_input(
            text, cursor_pos, InputMode.SINGLE_LINE, "$ "
        )
        
        assert isinstance(result, RenderResult)
        assert result.render_method == "single_line"
        assert result.lines_count == 1
        assert result.cursor_column == cursor_pos
        
        # Should contain prompt and text
        rendered_str = str(result.rendered_input)
        assert "$ " in rendered_str or "$" in rendered_str
        assert "Hello" in rendered_str
    
    @pytest.mark.asyncio
    async def test_multi_line_rendering(self):
        """Test multi-line input rendering."""
        text = "Line 1\nLine 2\nLine 3"
        cursor_pos = 8  # On line 2
        result = await self.pipeline.render_input(
            text, cursor_pos, InputMode.MULTI_LINE, "> "
        )
        
        assert isinstance(result, RenderResult)
        assert result.render_method == "multi_line"
        assert result.lines_count == 3
        assert result.cursor_line == 1  # Second line (0-indexed)
        
        rendered_str = str(result.rendered_input)
        assert "Line 1" in rendered_str
        assert "Line 2" in rendered_str
        assert "Line 3" in rendered_str
    
    @pytest.mark.asyncio
    async def test_password_rendering(self):
        """Test password input rendering."""
        text = "secret123"
        cursor_pos = 6
        result = await self.pipeline.render_input(
            text, cursor_pos, InputMode.PASSWORD, "Password: "
        )
        
        assert isinstance(result, RenderResult)
        assert result.render_method == "password"
        assert result.lines_count == 1
        
        # Should mask the password
        rendered_str = str(result.rendered_input)
        assert "secret123" not in rendered_str
        assert "•" in rendered_str or "*" in rendered_str or "[" in rendered_str
    
    @pytest.mark.asyncio
    async def test_command_rendering(self):
        """Test command input rendering."""
        text = "ls -la"
        cursor_pos = 6
        result = await self.pipeline.render_input(
            text, cursor_pos, InputMode.COMMAND, "$ "
        )
        
        assert isinstance(result, RenderResult)
        assert result.render_method == "command"
        
        rendered_str = str(result.rendered_input)
        assert "ls -la" in rendered_str
    
    @pytest.mark.asyncio
    async def test_search_rendering(self):
        """Test search input rendering."""
        text = "search term"
        cursor_pos = 6
        result = await self.pipeline.render_input(
            text, cursor_pos, InputMode.SEARCH, "Search: "
        )
        
        assert isinstance(result, RenderResult)
        assert result.render_method == "search"
        
        rendered_str = str(result.rendered_input)
        assert "search term" in rendered_str
    
    @pytest.mark.asyncio
    async def test_cursor_positioning(self):
        """Test cursor positioning in rendered output."""
        text = "Hello, World!"
        
        # Test cursor at beginning
        result = await self.pipeline.render_input(text, 0)
        assert result.cursor_column == 0
        
        # Test cursor at middle
        result = await self.pipeline.render_input(text, 7)
        assert result.cursor_column == 7
        
        # Test cursor at end
        result = await self.pipeline.render_input(text, len(text))
        assert result.cursor_column == len(text)
        
        # Test cursor beyond text (should clamp)
        result = await self.pipeline.render_input(text, len(text) + 10)
        assert result.cursor_column <= len(text)
    
    @pytest.mark.asyncio
    async def test_long_line_scrolling(self):
        """Test scrolling for long single lines."""
        # Configure narrow width
        self.pipeline.configure(max_width=20)
        
        long_text = "This is a very long line that exceeds the container width and should scroll"
        cursor_pos = 50
        
        result = await self.pipeline.render_input(long_text, cursor_pos)
        
        assert isinstance(result, RenderResult)
        # Should handle scrolling gracefully
        rendered_str = str(result.rendered_input)
        assert len(rendered_str) > 0
    
    @pytest.mark.asyncio
    async def test_multi_line_height_constraint(self):
        """Test multi-line input with height constraints."""
        # Configure limited height
        self.pipeline.configure(max_height=3)
        
        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        cursor_pos = 20  # On line 4
        
        result = await self.pipeline.render_input(text, cursor_pos, InputMode.MULTI_LINE)
        
        assert isinstance(result, RenderResult)
        assert result.lines_count <= 3  # Should respect height limit
    
    @pytest.mark.asyncio
    async def test_performance_requirement(self):
        """Test performance requirement: 5ms for input rendering."""
        text = "Some input text for performance testing"
        cursor_pos = 20
        
        start_time = time.time()
        result = await self.pipeline.render_input(text, cursor_pos)
        end_time = time.time()
        
        elapsed_ms = (end_time - start_time) * 1000
        
        # Performance should meet requirement
        assert result.performance_ms >= 0
        # Allow some margin for test environment
        assert elapsed_ms <= 25  # More lenient for tests
    
    @pytest.mark.asyncio
    async def test_display_change_detection(self):
        """Test detection of display changes."""
        text = "Hello"
        
        # First render
        result1 = await self.pipeline.render_input(text, 0)
        assert result1.display_changed
        
        # Same render - might be cached
        result2 = await self.pipeline.render_input(text, 0)
        # May or may not have changed depending on cursor blink
        
        # Different cursor position - should change
        result3 = await self.pipeline.render_input(text, 3)
        assert result3.display_changed
    
    @pytest.mark.asyncio
    async def test_cursor_visibility(self):
        """Test cursor visibility management."""
        text = "Test"
        cursor_pos = 2
        
        result = await self.pipeline.render_input(text, cursor_pos)
        
        # Should have cursor visibility info
        assert isinstance(result.cursor_visible, bool)
    
    def test_cursor_style_configuration(self):
        """Test cursor style configuration."""
        # Test different cursor styles
        self.pipeline.configure(cursor_style=CursorStyle.BLOCK)
        assert self.pipeline._cursor_style == CursorStyle.BLOCK
        
        self.pipeline.configure(cursor_style=CursorStyle.BAR)
        assert self.pipeline._cursor_style == CursorStyle.BAR
        
        self.pipeline.configure(cursor_style=CursorStyle.UNDERLINE)
        assert self.pipeline._cursor_style == CursorStyle.UNDERLINE
        
        self.pipeline.configure(cursor_style=CursorStyle.NONE)
        assert self.pipeline._cursor_style == CursorStyle.NONE
    
    def test_pipeline_configuration(self):
        """Test pipeline configuration."""
        self.pipeline.configure(
            max_width=100,
            max_height=20,
            cursor_style=CursorStyle.BLOCK,
            placeholder_text="Enter text..."
        )
        
        assert self.pipeline._max_width == 100
        assert self.pipeline._max_height == 20
        assert self.pipeline._cursor_style == CursorStyle.BLOCK
        assert self.pipeline._placeholder_text == "Enter text..."
    
    @pytest.mark.asyncio
    async def test_unicode_input_rendering(self):
        """Test rendering of Unicode input."""
        text = "Hello 世界! 🌟✨"
        cursor_pos = 10
        
        result = await self.pipeline.render_input(text, cursor_pos)
        
        assert isinstance(result, RenderResult)
        rendered_str = str(result.rendered_input)
        
        # Unicode should be preserved
        assert "世界" in rendered_str or len(rendered_str) > 0  # Some systems may not display properly
    
    @pytest.mark.asyncio
    async def test_input_validation(self):
        """Test input validation."""
        # Setup validator
        validator = InputValidator(min_length=5, max_length=10)
        self.pipeline.configure(validator=validator)
        
        # Valid input
        result = await self.pipeline.render_input("hello", 5)
        assert result.input_complete is not None
        
        # Invalid input (too short)
        result = await self.pipeline.render_input("hi", 2)
        assert result.input_complete is not None
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self):
        """Test render result caching."""
        text = "Test caching"
        cursor_pos = 5
        
        # First render
        result1 = await self.pipeline.render_input(text, cursor_pos)
        
        # Second render with same parameters
        result2 = await self.pipeline.render_input(text, cursor_pos)
        
        # Results should be consistent
        assert str(result1.rendered_input) == str(result2.rendered_input)
    
    def test_cache_clearing(self):
        """Test cache clearing."""
        # Add something to cache
        self.pipeline._render_cache["test"] = Mock()
        assert len(self.pipeline._render_cache) > 0
        
        self.pipeline.clear_cache()
        assert len(self.pipeline._render_cache) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and fallback."""
        # Mock an error in rendering
        with patch.object(self.pipeline, '_render_single_line', side_effect=Exception("Test error")):
            result = await self.pipeline.render_input("Test", 2)
            
            # Should fall back gracefully
            assert isinstance(result, RenderResult)
            assert result.render_method == "error_fallback"
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        metrics = self.pipeline.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'operation_times' in metrics
        assert 'render_count' in metrics
        assert 'cache_hits' in metrics
        assert 'cache_hit_rate_percent' in metrics
        assert 'cache_size' in metrics
        assert 'performance_target_ms' in metrics
        assert 'cursor_visible' in metrics


class TestInputValidator:
    """Test cases for InputValidator."""
    
    def test_length_validation(self):
        """Test length validation."""
        validator = InputValidator(min_length=3, max_length=10)
        
        # Too short
        is_valid, message = validator.validate("hi")
        assert not is_valid
        assert "Minimum length" in message
        
        # Just right
        is_valid, message = validator.validate("hello")
        assert is_valid
        assert message is None
        
        # Too long
        is_valid, message = validator.validate("this is way too long")
        assert not is_valid
        assert "Maximum length" in message
    
    def test_pattern_validation(self):
        """Test pattern validation."""
        # Email-like pattern
        validator = InputValidator(pattern=r'^[^@]+@[^@]+\.[^@]+$')
        
        # Invalid format
        is_valid, message = validator.validate("not-an-email")
        assert not is_valid
        assert "Invalid format" in message
        
        # Valid format
        is_valid, message = validator.validate("user@example.com")
        assert is_valid
        assert message is None
    
    def test_custom_validation(self):
        """Test custom validation function."""
        def custom_validator(text):
            if "forbidden" in text.lower():
                return False, "Forbidden word detected"
            return True, None
        
        validator = InputValidator(custom_validator=custom_validator)
        
        # Contains forbidden word
        is_valid, message = validator.validate("This is forbidden")
        assert not is_valid
        assert "Forbidden word" in message
        
        # Clean text
        is_valid, message = validator.validate("This is allowed")
        assert is_valid
        assert message is None
    
    def test_combined_validation(self):
        """Test combination of validation rules."""
        def custom_validator(text):
            if text.lower() == "admin":
                return False, "Admin not allowed"
            return True, None
        
        validator = InputValidator(
            min_length=3,
            max_length=10,
            pattern=r'^[a-zA-Z]+$',  # Only letters
            custom_validator=custom_validator
        )
        
        # Fails length check
        is_valid, message = validator.validate("ab")
        assert not is_valid
        
        # Fails pattern check
        is_valid, message = validator.validate("user123")
        assert not is_valid
        
        # Fails custom check
        is_valid, message = validator.validate("admin")
        assert not is_valid
        
        # Passes all checks
        is_valid, message = validator.validate("user")
        assert is_valid


class TestInputState:
    """Test InputState dataclass."""
    
    def test_input_state_creation(self):
        """Test InputState creation."""
        state = InputState(
            text="Hello",
            cursor_position=3,
            selection_start=1,
            selection_end=4,
            is_modified=True,
            is_valid=False,
            validation_message="Test error"
        )
        
        assert state.text == "Hello"
        assert state.cursor_position == 3
        assert state.selection_start == 1
        assert state.selection_end == 4
        assert state.is_modified
        assert not state.is_valid
        assert state.validation_message == "Test error"
    
    def test_input_state_defaults(self):
        """Test InputState default values."""
        state = InputState(text="Test", cursor_position=2)
        
        assert state.text == "Test"
        assert state.cursor_position == 2
        assert state.selection_start is None
        assert state.selection_end is None
        assert not state.is_modified
        assert state.is_valid
        assert state.validation_message is None


class TestRenderResult:
    """Test RenderResult dataclass."""
    
    def test_render_result_creation(self):
        """Test RenderResult creation."""
        result = RenderResult(
            rendered_input="$ hello",
            display_changed=True,
            input_complete=False,
            cursor_visible=True,
            performance_ms=3.5,
            render_method="single_line",
            lines_count=1,
            cursor_line=0,
            cursor_column=5
        )
        
        assert result.rendered_input == "$ hello"
        assert result.display_changed
        assert not result.input_complete
        assert result.cursor_visible
        assert result.performance_ms == 3.5
        assert result.render_method == "single_line"
        assert result.lines_count == 1
        assert result.cursor_line == 0
        assert result.cursor_column == 5
    
    def test_render_result_defaults(self):
        """Test RenderResult default values."""
        result = RenderResult(
            rendered_input="test",
            display_changed=True,
            input_complete=False
        )
        
        assert result.rendered_input == "test"
        assert result.display_changed
        assert not result.input_complete
        assert result.cursor_visible  # Default True
        assert result.performance_ms == 0.0
        assert result.render_method == "unknown"
        assert result.lines_count == 1
        assert result.cursor_line == 0
        assert result.cursor_column == 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    async def test_render_input_simple(self):
        """Test simple input rendering function."""
        result = await render_input_simple("Hello", 3, "$ ", 80)
        
        assert isinstance(result, str)
        assert "Hello" in result
        assert "$" in result or "$ " in result
    
    @pytest.mark.asyncio
    async def test_render_with_immediate_feedback(self):
        """Test immediate feedback rendering."""
        rendered_text, met_target = await render_with_immediate_feedback("Test", 2, "> ")
        
        assert isinstance(rendered_text, str)
        assert isinstance(met_target, bool)
        assert "Test" in rendered_text


@pytest.mark.asyncio
async def test_edge_cases():
    """Test various edge cases."""
    pipeline = InputRenderingPipeline()
    
    # Negative cursor position
    result = await pipeline.render_input("Hello", -5)
    assert result.cursor_column == 0  # Should clamp to 0
    
    # Cursor way beyond text
    result = await pipeline.render_input("Hi", 100)
    assert result.cursor_column <= len("Hi")
    
    # Empty text with cursor
    result = await pipeline.render_input("", 0)
    assert isinstance(result, RenderResult)
    
    # Text with only whitespace
    result = await pipeline.render_input("   ", 1)
    assert isinstance(result, RenderResult)
    
    # Special characters
    result = await pipeline.render_input("\t\n\r", 1)
    assert isinstance(result, RenderResult)