"""
Comprehensive tests for ANSIMarkdownProcessor - Advanced text rendering with ANSI colors.

This test suite verifies that the ANSIMarkdownProcessor:
1. Processes markdown formatting correctly
2. Applies ANSI color codes properly
3. Handles text wrapping while preserving ANSI codes
4. Provides performance-optimized rendering
5. Handles edge cases and malformed input gracefully
6. Strips ANSI codes when needed
"""

import pytest
import re
from unittest.mock import Mock, patch

from agentsmcp.ui.v2.ansi_markdown_processor import (
    ANSIMarkdownProcessor, ANSIColor, ANSIStyle, TextStyle, RenderConfig,
    process_markdown, render_markdown_lines, strip_markdown_and_ansi
)


class TestTextStyle:
    """Test TextStyle data class functionality."""
    
    def test_basic_style_creation(self):
        """Test basic text style creation."""
        style = TextStyle(color=ANSIColor.RED, bold=True)
        
        assert style.color == ANSIColor.RED
        assert style.bold is True
        assert style.italic is False
        assert style.underline is False
        assert style.dim is False
    
    def test_style_to_ansi_conversion(self):
        """Test conversion of style to ANSI escape sequence."""
        # Test bold only
        style = TextStyle(bold=True)
        assert style.to_ansi() == "\x1b[1m"
        
        # Test color only
        style = TextStyle(color=ANSIColor.RED)
        assert style.to_ansi() == "\x1b[31m"
        
        # Test multiple attributes
        style = TextStyle(color=ANSIColor.BLUE, bold=True, underline=True)
        ansi = style.to_ansi()
        assert "1" in ansi  # Bold
        assert "4" in ansi  # Underline
        assert "34" in ansi  # Blue
        assert ansi.startswith("\x1b[")
        assert ansi.endswith("m")
    
    def test_empty_style_to_ansi(self):
        """Test empty style conversion."""
        style = TextStyle()
        assert style.to_ansi() == ""


class TestRenderConfig:
    """Test RenderConfig configuration."""
    
    def test_default_configuration(self):
        """Test default render configuration values."""
        config = RenderConfig()
        
        assert config.width == 80
        assert config.enable_colors is True
        assert config.enable_markdown is True
        assert config.enable_wrapping is True
        assert config.indent_code_blocks == 2
        assert config.list_indent == 2
        assert config.header_style == "bold_yellow"
        assert config.code_style == "cyan"
    
    def test_custom_configuration(self):
        """Test custom render configuration."""
        config = RenderConfig(
            width=120,
            enable_colors=False,
            header_style="underline"
        )
        
        assert config.width == 120
        assert config.enable_colors is False
        assert config.header_style == "underline"


class TestANSIMarkdownProcessor:
    """Test ANSI markdown processor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create processor with default config."""
        return ANSIMarkdownProcessor()
    
    @pytest.fixture
    def no_color_processor(self):
        """Create processor with colors disabled."""
        config = RenderConfig(enable_colors=False)
        return ANSIMarkdownProcessor(config)
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.config.width == 80
        assert processor.config.enable_colors is True
        assert len(processor._style_map) > 0
        assert len(processor._patterns) > 0
        assert processor._in_code_block is False
    
    def test_plain_text_processing(self, processor):
        """Test processing of plain text without markdown."""
        text = "This is plain text with no formatting."
        result = processor.process_text(text)
        
        # Should return text unchanged
        assert result == text
    
    def test_empty_text_processing(self, processor):
        """Test processing of empty or None text."""
        assert processor.process_text("") == ""
        assert processor.process_text(None) == None
    
    def test_markdown_disabled_processing(self):
        """Test processing with markdown disabled."""
        config = RenderConfig(enable_markdown=False)
        processor = ANSIMarkdownProcessor(config)
        
        text = "**Bold text** and *italic text*"
        result = processor.process_text(text)
        
        # Should return original text without processing
        assert result == text
    
    def test_header_processing_with_colors(self, processor):
        """Test markdown header processing with colors."""
        text = "# Header 1\n## Header 2\n### Header 3"
        result = processor.process_text(text)
        
        # Should contain ANSI color codes
        assert "\x1b[" in result
        assert ANSIStyle.RESET.value in result
        
        # Should preserve header structure
        assert "# " in result
        assert "## " in result
        assert "### " in result
    
    def test_header_processing_without_colors(self, no_color_processor):
        """Test markdown header processing without colors."""
        text = "# Header 1\n## Header 2"
        result = no_color_processor.process_text(text)
        
        # Should not contain ANSI codes
        assert "\x1b[" not in result
        
        # Should preserve original formatting
        assert result == text
    
    def test_bold_text_processing(self, processor):
        """Test bold text processing."""
        text = "This is **bold text** in a sentence."
        result = processor.process_text(text)
        
        # Should contain ANSI bold codes
        assert "\x1b[1m" in result or "\x1b[" in result
        assert ANSIStyle.RESET.value in result
        
        # Should preserve the text content
        assert "bold text" in result
    
    def test_italic_text_processing(self, processor):
        """Test italic text processing."""
        text = "This is *italic text* and _also italic_."
        result = processor.process_text(text)
        
        # Should contain ANSI codes
        assert "\x1b[" in result
        assert ANSIStyle.RESET.value in result
        
        # Should process both * and _ variants
        assert "italic text" in result
        assert "also italic" in result
    
    def test_inline_code_processing(self, processor):
        """Test inline code processing."""
        text = "Use `print('hello')` to output text."
        result = processor.process_text(text)
        
        # Should contain ANSI codes
        assert "\x1b[" in result
        assert ANSIStyle.RESET.value in result
        
        # Should preserve code content
        assert "print('hello')" in result
    
    def test_code_block_processing(self, processor):
        """Test code block processing."""
        text = """
```python
def hello():
    print("Hello, world!")
```
"""
        result = processor.process_text(text)
        
        # Should contain ANSI codes
        assert "\x1b[" in result
        
        # Should indent code properly
        lines = result.split('\n')
        code_lines = [line for line in lines if 'def hello' in line or 'print' in line]
        for line in code_lines:
            assert line.startswith('  ')  # Should be indented
    
    def test_list_processing(self, processor):
        """Test markdown list processing."""
        text = """
- Item 1
- Item 2
  - Nested item
* Another item
+ Plus item

1. Numbered item
2. Second item
"""
        result = processor.process_text(text)
        
        # Should contain ANSI codes for list markers
        assert "\x1b[" in result
        
        # Should replace list markers with bullets
        assert "•" in result  # Bullet character
    
    def test_blockquote_processing(self, processor):
        """Test blockquote processing."""
        text = "> This is a quoted text.\n> Second line of quote."
        result = processor.process_text(text)
        
        # Should contain ANSI codes
        assert "\x1b[" in result
        
        # Should preserve quote markers
        assert ">" in result
    
    def test_horizontal_rule_processing(self, processor):
        """Test horizontal rule processing."""
        text = "Text above\n\n---\n\nText below"
        result = processor.process_text(text)
        
        # Should contain ANSI codes
        assert "\x1b[" in result
        
        # Should replace with unicode rule
        assert "─" in result
    
    def test_link_processing(self, processor):
        """Test markdown link processing."""
        text = "Visit [GitHub](https://github.com) for code."
        result = processor.process_text(text)
        
        # Should contain ANSI codes
        assert "\x1b[" in result
        
        # Should preserve link text but remove URL in color mode
        assert "GitHub" in result
        assert "https://github.com" not in result
    
    def test_strikethrough_processing(self, processor):
        """Test strikethrough text processing."""
        text = "This is ~~deleted text~~ in a sentence."
        result = processor.process_text(text)
        
        # Should contain strikethrough ANSI code
        assert ANSIStyle.STRIKETHROUGH.value in result
        assert "deleted text" in result
    
    def test_complex_markdown_combination(self, processor):
        """Test complex markdown with multiple formatting types."""
        text = """
# Main Header

This paragraph has **bold**, *italic*, and `inline code`.

## Code Example

```python
def example():
    return "**not bold in code**"
```

- List item with **bold**
- List item with [link](http://example.com)

> Quote with *emphasis*
"""
        result = processor.process_text(text)
        
        # Should contain multiple ANSI sequences
        ansi_count = result.count('\x1b[')
        assert ansi_count > 5  # Multiple formatting elements
        
        # Should preserve structure
        assert "Main Header" in result
        assert "Code Example" in result
        assert "def example" in result
        assert "List item" in result
        assert "Quote with" in result
    
    def test_nested_formatting_handling(self, processor):
        """Test handling of nested formatting."""
        text = "**Bold with *italic inside* text**"
        result = processor.process_text(text)
        
        # Should handle nested formatting
        assert "\x1b[" in result
        assert "Bold with" in result
        assert "italic inside" in result
    
    def test_text_wrapping_plain_text(self, processor):
        """Test text wrapping for plain text."""
        processor.config.width = 20
        processor.config.enable_wrapping = True
        
        long_text = "This is a very long line of text that should be wrapped at the specified width"
        result = processor.process_text(long_text)
        
        lines = result.split('\n')
        for line in lines:
            # Each line should be within width limits
            assert len(line) <= processor.config.width
    
    def test_text_wrapping_with_ansi_codes(self, processor):
        """Test text wrapping preservation of ANSI codes."""
        processor.config.width = 30
        processor.config.enable_wrapping = True
        
        text = "This is **very bold text** that should wrap correctly with ANSI codes preserved"
        result = processor.process_text(text)
        
        lines = result.split('\n')
        assert len(lines) > 1  # Should have wrapped
        
        # ANSI codes should still be present
        assert "\x1b[" in result
    
    def test_wrapping_disabled(self, processor):
        """Test behavior with wrapping disabled."""
        processor.config.enable_wrapping = False
        processor.config.width = 10  # Very small width
        
        long_text = "This is a very long line that should not be wrapped"
        result = processor.process_text(long_text)
        
        # Should not wrap despite small width
        assert '\n' not in result.strip()
    
    def test_visual_length_calculation(self, processor):
        """Test visual length calculation excluding ANSI codes."""
        text_with_ansi = "\x1b[1mBold\x1b[0m text"
        visual_length = processor._calculate_visual_length(text_with_ansi)
        
        # Should count only visible characters
        assert visual_length == len("Bold text")
    
    def test_ansi_detection(self, processor):
        """Test ANSI code detection in lines."""
        plain_line = "This is plain text"
        ansi_line = "This has \x1b[1mANSI\x1b[0m codes"
        
        assert processor._line_contains_ansi(plain_line) is False
        assert processor._line_contains_ansi(ansi_line) is True
    
    def test_render_lines_functionality(self, processor):
        """Test rendering text as formatted lines."""
        text = "# Header\n\nParagraph with **bold** text."
        lines = processor.render_lines(text, width=40, indent_prefix="  ")
        
        assert isinstance(lines, list)
        assert len(lines) > 0
        
        # Each line should start with indent prefix
        for line in lines:
            if line.strip():  # Non-empty lines
                assert line.startswith("  ")
    
    def test_render_lines_width_enforcement(self, processor):
        """Test that render_lines enforces width limits."""
        text = "Very long line of text that exceeds width"
        lines = processor.render_lines(text, width=20, indent_prefix=">> ")
        
        for line in lines:
            assert len(line) <= 20
    
    def test_strip_ansi_functionality(self, processor):
        """Test ANSI code stripping."""
        ansi_text = "\x1b[1mBold\x1b[0m and \x1b[31mRed\x1b[0m text"
        clean_text = processor.strip_ansi(ansi_text)
        
        assert clean_text == "Bold and Red text"
        assert "\x1b[" not in clean_text
    
    def test_style_mapping_retrieval(self, processor):
        """Test style mapping retrieval."""
        # Test existing style
        bold_style = processor._get_style('bold')
        assert bold_style.bold is True
        
        # Test non-existing style returns default
        unknown_style = processor._get_style('nonexistent')
        assert isinstance(unknown_style, TextStyle)
        assert unknown_style.bold is False
    
    def test_config_get_set(self, processor):
        """Test configuration get/set methods."""
        original_config = processor.get_config()
        assert original_config.width == 80
        
        new_config = RenderConfig(width=100, enable_colors=False)
        processor.set_config(new_config)
        
        updated_config = processor.get_config()
        assert updated_config.width == 100
        assert updated_config.enable_colors is False
    
    def test_error_handling_in_processing(self, processor):
        """Test error handling during text processing."""
        # Mock a pattern to raise an exception
        with patch.object(processor._patterns['bold'], 'sub', side_effect=Exception("Test error")):
            text = "**Bold text**"
            result = processor.process_text(text)
            
            # Should return original text on error
            assert result == text


class TestMarkdownPatternMatching:
    """Test specific markdown pattern matching."""
    
    @pytest.fixture
    def processor(self):
        return ANSIMarkdownProcessor()
    
    def test_bold_pattern_matching(self, processor):
        """Test bold pattern matching edge cases."""
        test_cases = [
            ("**bold**", True),
            ("**bold text**", True),
            ("** not bold **", True),  # Spaces are allowed
            ("*not bold*", False),     # Single asterisk
            ("***not handled***", False),  # Triple asterisk
        ]
        
        for text, should_match in test_cases:
            matches = processor._patterns['bold'].findall(text)
            if should_match:
                assert len(matches) > 0, f"'{text}' should match bold pattern"
            else:
                assert len(matches) == 0, f"'{text}' should not match bold pattern"
    
    def test_italic_pattern_matching(self, processor):
        """Test italic pattern matching edge cases."""
        test_cases = [
            ("*italic*", True),
            ("*italic text*", True),
            ("_italic_", True),
            ("_italic text_", True),
            ("* not italic *", False),  # Leading space
            ("*not italic *", False),   # Trailing space
            ("**not italic**", False),  # Double asterisk
        ]
        
        for text, should_match in test_cases:
            matches = processor._patterns['italic'].findall(text)
            underscore_matches = processor._patterns['italic_underscore'].findall(text)
            
            total_matches = len(matches) + len(underscore_matches)
            
            if should_match:
                assert total_matches > 0, f"'{text}' should match italic pattern"
            else:
                assert total_matches == 0, f"'{text}' should not match italic pattern"
    
    def test_code_block_pattern_matching(self, processor):
        """Test code block pattern matching."""
        test_cases = [
            ("```\ncode\n```", True),
            ("```python\ncode\n```", True),
            ("```\ncode", False),  # Missing closing
            ("code\n```", False),   # Missing opening
        ]
        
        for text, should_match in test_cases:
            matches = processor._patterns['code_block'].findall(text)
            if should_match:
                assert len(matches) > 0, f"'{text}' should match code block pattern"
            else:
                assert len(matches) == 0, f"'{text}' should not match code block pattern"
    
    def test_header_pattern_matching(self, processor):
        """Test header pattern matching."""
        test_cases = [
            ("# Header", True),
            ("## Header", True),
            ("### Header", True),
            ("#### Header", True),
            ("##### Header", True),
            ("###### Header", True),
            ("####### Not Header", False),  # Too many #
            (" # Not Header", False),       # Leading space
            ("# ", True),                   # Just marker
        ]
        
        for text, should_match in test_cases:
            matches = processor._patterns['header'].findall(text)
            if should_match:
                assert len(matches) > 0, f"'{text}' should match header pattern"
            else:
                assert len(matches) == 0, f"'{text}' should not match header pattern"
    
    def test_link_pattern_matching(self, processor):
        """Test link pattern matching."""
        test_cases = [
            ("[text](url)", True),
            ("[GitHub](https://github.com)", True),
            ("[](empty)", True),
            ("[no url]", False),
            ("(no text)[url]", False),
        ]
        
        for text, should_match in test_cases:
            matches = processor._patterns['link'].findall(text)
            if should_match:
                assert len(matches) > 0, f"'{text}' should match link pattern"
            else:
                assert len(matches) == 0, f"'{text}' should not match link pattern"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_process_markdown_function(self):
        """Test process_markdown convenience function."""
        text = "**Bold** text"
        result = process_markdown(text, width=80, enable_colors=True)
        
        assert isinstance(result, str)
        assert "\x1b[" in result  # Should have ANSI codes
        assert "Bold" in result
    
    def test_process_markdown_no_colors(self):
        """Test process_markdown with colors disabled."""
        text = "**Bold** text"
        result = process_markdown(text, enable_colors=False)
        
        assert "\x1b[" not in result  # Should not have ANSI codes
        assert result == text  # Should return original
    
    def test_render_markdown_lines_function(self):
        """Test render_markdown_lines convenience function."""
        text = "# Header\nParagraph"
        lines = render_markdown_lines(text, width=40, indent="  ")
        
        assert isinstance(lines, list)
        assert len(lines) > 0
        for line in lines:
            if line.strip():
                assert line.startswith("  ")
    
    def test_strip_markdown_and_ansi_function(self):
        """Test strip_markdown_and_ansi convenience function."""
        text = "\x1b[1m**Bold**\x1b[0m and `code`"
        result = strip_markdown_and_ansi(text)
        
        # Should remove both ANSI and markdown
        assert "\x1b[" not in result
        assert "**" not in result
        assert "`" not in result
        assert "Bold" in result
        assert "code" in result


class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""
    
    @pytest.fixture
    def processor(self):
        return ANSIMarkdownProcessor()
    
    def test_large_text_processing(self, processor):
        """Test processing of large text blocks."""
        # Create large text with various formatting
        large_text = "\n".join([
            f"# Header {i}",
            f"This is paragraph {i} with **bold** and *italic* text.",
            f"- List item {i}",
            "```python",
            f"def function_{i}():",
            f"    return {i}",
            "```",
            ""
        ] for i in range(100))
        
        # Should complete without timeout or memory issues
        result = processor.process_text(large_text)
        
        assert isinstance(result, str)
        assert len(result) > len(large_text)  # Should have added ANSI codes
        assert result.count('\x1b[') > 100  # Many ANSI sequences
    
    def test_deeply_nested_formatting(self, processor):
        """Test handling of deeply nested formatting."""
        nested_text = "**bold *italic `code` italic* bold**"
        result = processor.process_text(nested_text)
        
        # Should handle nested formatting gracefully
        assert isinstance(result, str)
        assert "\x1b[" in result
    
    def test_malformed_markdown_handling(self, processor):
        """Test handling of malformed markdown."""
        malformed_cases = [
            "**unclosed bold",
            "*unclosed italic",
            "```unclosed code block",
            "[unclosed link(",
            "# Header with **unclosed bold",
        ]
        
        for malformed_text in malformed_cases:
            result = processor.process_text(malformed_text)
            
            # Should not crash and should return a string
            assert isinstance(result, str)
            # Should handle gracefully (may or may not apply formatting)
    
    def test_unicode_text_processing(self, processor):
        """Test processing of Unicode text."""
        unicode_text = "**粗体** and *斜体* and `代码` with 🔥 emoji"
        result = processor.process_text(unicode_text)
        
        assert isinstance(result, str)
        assert "粗体" in result
        assert "斜体" in result
        assert "代码" in result
        assert "🔥" in result
    
    def test_empty_and_whitespace_handling(self, processor):
        """Test handling of empty and whitespace-only content."""
        test_cases = [
            "",
            " ",
            "\n",
            "\t",
            "   \n  \n   ",
        ]
        
        for text in test_cases:
            result = processor.process_text(text)
            # Should handle without crashing
            assert isinstance(result, str)
    
    def test_very_long_lines(self, processor):
        """Test handling of very long lines."""
        processor.config.width = 80
        processor.config.enable_wrapping = True
        
        # Create a very long line
        very_long_line = "word " * 500  # 2500+ characters
        result = processor.process_text(very_long_line)
        
        lines = result.split('\n')
        
        # Should wrap appropriately
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= processor.config.width
    
    def test_many_small_elements(self, processor):
        """Test processing many small formatting elements."""
        # Text with many small bold elements
        many_elements = " ".join([f"**{i}**" for i in range(100)])
        result = processor.process_text(many_elements)
        
        # Should handle many elements efficiently
        assert isinstance(result, str)
        assert result.count('\x1b[') >= 100  # Many ANSI sequences
    
    def test_regex_catastrophic_backtracking_prevention(self, processor):
        """Test prevention of regex catastrophic backtracking."""
        # Patterns that could cause catastrophic backtracking
        problematic_patterns = [
            "*" * 100 + "text",
            "**" * 50 + "text",
            "`" * 100 + "code" + "`" * 100,
        ]
        
        for pattern in problematic_patterns:
            # Should complete in reasonable time (not hang)
            result = processor.process_text(pattern)
            assert isinstance(result, str)
    
    def test_wrapping_ansi_line_edge_cases(self, processor):
        """Test edge cases in ANSI line wrapping."""
        processor.config.width = 20
        
        # Line with ANSI codes longer than width
        long_ansi_line = f"\x1b[1m{'x' * 50}\x1b[0m"
        wrapped_lines = processor._wrap_ansi_line(long_ansi_line)
        
        assert isinstance(wrapped_lines, list)
        assert len(wrapped_lines) > 1
        
        # Each line should be within visual width limits
        for line in wrapped_lines:
            visual_length = processor._calculate_visual_length(line)
            assert visual_length <= processor.config.width


class TestSpecificBugFixes:
    """Test specific bug fixes and regression prevention."""
    
    @pytest.fixture
    def processor(self):
        return ANSIMarkdownProcessor()
    
    def test_bold_italic_overlap_handling(self, processor):
        """Test handling of overlapping bold and italic patterns."""
        text = "***bold and italic***"
        result = processor.process_text(text)
        
        # Should handle gracefully without infinite loops
        assert isinstance(result, str)
    
    def test_code_block_language_detection(self, processor):
        """Test code block language detection."""
        text = "```python\nprint('hello')\n```"
        result = processor.process_text(text)
        
        # Should process code block with language
        assert "print('hello')" in result
        assert "\x1b[" in result  # Should have styling
    
    def test_list_marker_replacement_consistency(self, processor):
        """Test consistent list marker replacement."""
        text = "- Item 1\n* Item 2\n+ Item 3"
        result = processor.process_text(text)
        
        # All markers should be replaced with bullets
        bullet_count = result.count("•")
        assert bullet_count == 3
    
    def test_quote_preservation_with_nesting(self, processor):
        """Test quote marker preservation with nested quotes."""
        text = "> Quote level 1\n>> Quote level 2"
        result = processor.process_text(text)
        
        # Should preserve both quote levels
        assert ">" in result
        assert ">>" in result or ">" in result  # Nested quotes handled
    
    def test_horizontal_rule_width_calculation(self, processor):
        """Test horizontal rule width calculation."""
        processor.config.width = 40
        text = "---"
        result = processor.process_text(text)
        
        # Should create rule appropriate to width
        rule_line = [line for line in result.split('\n') if '─' in line][0]
        visual_length = processor._calculate_visual_length(rule_line)
        assert visual_length <= processor.config.width