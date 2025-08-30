"""
ANSI Markdown Processor - Advanced text rendering with ANSI color codes.

This module provides sophisticated markdown-style text processing with ANSI colors,
handling edge cases and providing consistent formatting for the TUI.
"""

import re
import textwrap
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ANSIColor(Enum):
    """ANSI color codes."""
    BLACK = "\x1b[30m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    WHITE = "\x1b[37m"
    
    # Bright colors
    BRIGHT_BLACK = "\x1b[90m"
    BRIGHT_RED = "\x1b[91m"
    BRIGHT_GREEN = "\x1b[92m"
    BRIGHT_YELLOW = "\x1b[93m"
    BRIGHT_BLUE = "\x1b[94m"
    BRIGHT_MAGENTA = "\x1b[95m"
    BRIGHT_CYAN = "\x1b[96m"
    BRIGHT_WHITE = "\x1b[97m"


class ANSIStyle(Enum):
    """ANSI style codes."""
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    ITALIC = "\x1b[3m"
    UNDERLINE = "\x1b[4m"
    BLINK = "\x1b[5m"
    REVERSE = "\x1b[7m"
    STRIKETHROUGH = "\x1b[9m"


@dataclass
class TextStyle:
    """Text styling configuration."""
    color: Optional[ANSIColor] = None
    background: Optional[ANSIColor] = None
    bold: bool = False
    italic: bool = False
    underline: bool = False
    dim: bool = False
    
    def to_ansi(self) -> str:
        """Convert to ANSI escape sequence."""
        codes = []
        
        if self.bold:
            codes.append("1")
        if self.dim:
            codes.append("2")
        if self.italic:
            codes.append("3")
        if self.underline:
            codes.append("4")
        if self.color:
            codes.append(self.color.value[2:-1])  # Extract number from \x1b[XXm
        
        if codes:
            return f"\x1b[{';'.join(codes)}m"
        return ""


@dataclass
class RenderConfig:
    """Configuration for text rendering."""
    width: int = 80
    enable_colors: bool = True
    enable_markdown: bool = True
    enable_wrapping: bool = True
    indent_code_blocks: int = 2
    list_indent: int = 2
    header_style: str = "bold_yellow"  # bold_yellow, underline, etc.
    code_style: str = "cyan"
    emphasis_style: str = "bold"
    strong_style: str = "bold"
    quote_style: str = "dim"
    link_style: str = "blue_underline"


class ANSIMarkdownProcessor:
    """Advanced markdown processor with ANSI color support."""
    
    def __init__(self, config: Optional[RenderConfig] = None):
        """
        Initialize the ANSI markdown processor.
        
        Args:
            config: Rendering configuration
        """
        self.config = config or RenderConfig()
        
        # Style mappings
        self._style_map = {
            'bold': TextStyle(bold=True),
            'italic': TextStyle(italic=True),
            'underline': TextStyle(underline=True),
            'dim': TextStyle(dim=True),
            'bold_yellow': TextStyle(color=ANSIColor.YELLOW, bold=True),
            'cyan': TextStyle(color=ANSIColor.CYAN),
            'blue': TextStyle(color=ANSIColor.BLUE),
            'blue_underline': TextStyle(color=ANSIColor.BLUE, underline=True),
            'green': TextStyle(color=ANSIColor.GREEN),
            'red': TextStyle(color=ANSIColor.RED),
            'magenta': TextStyle(color=ANSIColor.MAGENTA),
        }
        
        # Regex patterns for markdown elements
        self._patterns = {
            'code_block': re.compile(r'^```(\w+)?\s*\n(.*?)\n```$', re.MULTILINE | re.DOTALL),
            'inline_code': re.compile(r'`([^`]+)`'),
            'bold': re.compile(r'\*\*(.+?)\*\*'),
            'italic': re.compile(r'(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)'),
            'italic_underscore': re.compile(r'_(.+?)_'),
            'strikethrough': re.compile(r'~~(.+?)~~'),
            'header': re.compile(r'^(\s*#+)\s*(.+)$', re.MULTILINE),
            'list_item': re.compile(r'^(\s*)([-*+])\s+(.+)$', re.MULTILINE),
            'ordered_list': re.compile(r'^(\s*)(\d+\.)\s+(.+)$', re.MULTILINE),
            'quote': re.compile(r'^(\s*>)\s*(.+)$', re.MULTILINE),
            'link': re.compile(r'\[([^\]]+)\]\(([^\)]+)\)'),
            'hr': re.compile(r'^(\s*)([-=]{3,})\s*$', re.MULTILINE),
        }
        
        # Track code block state for proper rendering
        self._in_code_block = False
        
    def process_text(self, text: str) -> str:
        """
        Process text with markdown formatting and ANSI colors.
        
        Args:
            text: Raw text to process
            
        Returns:
            Processed text with ANSI formatting
        """
        if not text or not self.config.enable_markdown:
            return text
        
        try:
            # Process in order of precedence
            result = self._process_code_blocks(text)
            result = self._process_headers(result)
            result = self._process_lists(result)
            result = self._process_quotes(result)
            result = self._process_horizontal_rules(result)
            result = self._process_inline_formatting(result)
            result = self._process_links(result)
            
            if self.config.enable_wrapping:
                result = self._apply_text_wrapping(result)
            
            return result
            
        except Exception as e:
            logger.warning(f"Error processing markdown text: {e}")
            return text  # Return original text on error
    
    def _process_code_blocks(self, text: str) -> str:
        """Process code blocks with syntax highlighting."""
        def replace_code_block(match):
            language = match.group(1) or ''
            code = match.group(2)
            
            if not self.config.enable_colors:
                return f"\n{code}\n"
            
            style = self._get_style('cyan')
            lines = []
            
            for line in code.split('\n'):
                indent = ' ' * self.config.indent_code_blocks
                styled_line = f"{indent}{style.to_ansi()}{line}{ANSIStyle.RESET.value}"
                lines.append(styled_line)
            
            return '\n' + '\n'.join(lines) + '\n'
        
        return self._patterns['code_block'].sub(replace_code_block, text)
    
    def _process_headers(self, text: str) -> str:
        """Process markdown headers."""
        def replace_header(match):
            level_markers = match.group(1)
            content = match.group(2)
            level = len(level_markers.strip())
            
            if not self.config.enable_colors:
                return f"{level_markers} {content}"
            
            style = self._get_style(self.config.header_style)
            return f"{level_markers} {style.to_ansi()}{content}{ANSIStyle.RESET.value}"
        
        return self._patterns['header'].sub(replace_header, text)
    
    def _process_lists(self, text: str) -> str:
        """Process markdown lists."""
        def replace_list_item(match):
            indent = match.group(1)
            marker = match.group(2)
            content = match.group(3)
            
            if not self.config.enable_colors:
                return f"{indent}{marker} {content}"
            
            bullet_style = self._get_style('magenta')
            bullet = f"{bullet_style.to_ansi()}•{ANSIStyle.RESET.value}"
            
            return f"{indent}{bullet} {content}"
        
        def replace_ordered_item(match):
            indent = match.group(1)
            number = match.group(2)
            content = match.group(3)
            
            if not self.config.enable_colors:
                return f"{indent}{number} {content}"
            
            number_style = self._get_style('cyan')
            styled_number = f"{number_style.to_ansi()}{number}{ANSIStyle.RESET.value}"
            
            return f"{indent}{styled_number} {content}"
        
        # Process unordered lists first
        result = self._patterns['list_item'].sub(replace_list_item, text)
        # Then ordered lists
        result = self._patterns['ordered_list'].sub(replace_ordered_item, result)
        
        return result
    
    def _process_quotes(self, text: str) -> str:
        """Process blockquotes."""
        def replace_quote(match):
            quote_marker = match.group(1)
            content = match.group(2)
            
            if not self.config.enable_colors:
                return f"{quote_marker} {content}"
            
            style = self._get_style(self.config.quote_style)
            return f"{quote_marker} {style.to_ansi()}{content}{ANSIStyle.RESET.value}"
        
        return self._patterns['quote'].sub(replace_quote, text)
    
    def _process_horizontal_rules(self, text: str) -> str:
        """Process horizontal rules."""
        def replace_hr(match):
            indent = match.group(1)
            rule_chars = match.group(2)
            
            if not self.config.enable_colors:
                return f"{indent}{rule_chars}"
            
            style = self._get_style('dim')
            rule = '─' * min(len(rule_chars), self.config.width - len(indent))
            return f"{indent}{style.to_ansi()}{rule}{ANSIStyle.RESET.value}"
        
        return self._patterns['hr'].sub(replace_hr, text)
    
    def _process_inline_formatting(self, text: str) -> str:
        """Process inline formatting (bold, italic, code, etc.)."""
        # Process inline code first (highest precedence)
        def replace_inline_code(match):
            code = match.group(1)
            if not self.config.enable_colors:
                return f"`{code}`"
            
            style = self._get_style(self.config.code_style)
            return f"{style.to_ansi()}{code}{ANSIStyle.RESET.value}"
        
        result = self._patterns['inline_code'].sub(replace_inline_code, text)
        
        # Process bold
        def replace_bold(match):
            content = match.group(1)
            if not self.config.enable_colors:
                return content
            
            style = self._get_style(self.config.strong_style)
            return f"{style.to_ansi()}{content}{ANSIStyle.RESET.value}"
        
        result = self._patterns['bold'].sub(replace_bold, result)
        
        # Process italic (both * and _ variants)
        def replace_italic(match):
            content = match.group(1)
            if not self.config.enable_colors:
                return content
            
            style = self._get_style(self.config.emphasis_style)
            return f"{style.to_ansi()}{content}{ANSIStyle.RESET.value}"
        
        result = self._patterns['italic'].sub(replace_italic, result)
        result = self._patterns['italic_underscore'].sub(replace_italic, result)
        
        # Process strikethrough
        def replace_strikethrough(match):
            content = match.group(1)
            if not self.config.enable_colors:
                return content
            
            return f"{ANSIStyle.STRIKETHROUGH.value}{content}{ANSIStyle.RESET.value}"
        
        result = self._patterns['strikethrough'].sub(replace_strikethrough, result)
        
        return result
    
    def _process_links(self, text: str) -> str:
        """Process markdown links."""
        def replace_link(match):
            link_text = match.group(1)
            url = match.group(2)
            
            if not self.config.enable_colors:
                return f"{link_text} ({url})"
            
            style = self._get_style(self.config.link_style)
            return f"{style.to_ansi()}{link_text}{ANSIStyle.RESET.value}"
        
        return self._patterns['link'].sub(replace_link, text)
    
    def _apply_text_wrapping(self, text: str) -> str:
        """Apply text wrapping while preserving ANSI codes."""
        if not self.config.enable_wrapping or self.config.width <= 0:
            return text
        
        lines = text.split('\n')
        wrapped_lines = []
        
        for line in lines:
            if self._line_contains_ansi(line):
                # Handle ANSI-styled lines specially
                wrapped_lines.extend(self._wrap_ansi_line(line))
            else:
                # Regular text wrapping
                if len(line) <= self.config.width:
                    wrapped_lines.append(line)
                else:
                    wrapped_lines.extend(textwrap.wrap(
                        line, 
                        width=self.config.width,
                        break_long_words=False,
                        break_on_hyphens=True
                    ))
        
        return '\n'.join(wrapped_lines)
    
    def _line_contains_ansi(self, line: str) -> bool:
        """Check if line contains ANSI escape codes."""
        return '\x1b[' in line
    
    def _wrap_ansi_line(self, line: str) -> List[str]:
        """Wrap a line containing ANSI codes."""
        # This is a simplified approach - for production use,
        # consider a more sophisticated ANSI-aware wrapper
        
        # For now, preserve ANSI lines as-is if they're not too long
        visual_length = self._calculate_visual_length(line)
        
        if visual_length <= self.config.width:
            return [line]
        
        # If too long, attempt simple breaking
        # This is a fallback - more complex logic could be added
        parts = line.split(' ')
        current_line = ""
        result_lines = []
        
        for part in parts:
            test_line = f"{current_line} {part}" if current_line else part
            if self._calculate_visual_length(test_line) <= self.config.width:
                current_line = test_line
            else:
                if current_line:
                    result_lines.append(current_line)
                current_line = part
        
        if current_line:
            result_lines.append(current_line)
        
        return result_lines or [line]
    
    def _calculate_visual_length(self, text: str) -> int:
        """Calculate visual length of text (excluding ANSI codes)."""
        # Remove ANSI escape sequences for length calculation
        ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
        clean_text = ansi_pattern.sub('', text)
        return len(clean_text)
    
    def _get_style(self, style_name: str) -> TextStyle:
        """Get style configuration by name."""
        return self._style_map.get(style_name, TextStyle())
    
    def render_lines(self, text: str, width: int, indent_prefix: str = '') -> List[str]:
        """
        Render text as a list of formatted lines.
        
        Args:
            text: Text to render
            width: Target width for lines
            indent_prefix: Prefix to add to each line
            
        Returns:
            List of formatted lines
        """
        # Update config width
        original_width = self.config.width
        self.config.width = max(1, width - len(indent_prefix))
        
        try:
            processed_text = self.process_text(text)
            lines = processed_text.split('\n')
            
            result_lines = []
            for line in lines:
                prefixed_line = indent_prefix + line
                # Ensure line doesn't exceed width
                if len(prefixed_line) > width:
                    prefixed_line = prefixed_line[:width]
                result_lines.append(prefixed_line)
            
            return result_lines
            
        finally:
            # Restore original width
            self.config.width = original_width
    
    def strip_ansi(self, text: str) -> str:
        """Remove all ANSI escape sequences from text."""
        ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
        return ansi_pattern.sub('', text)
    
    def get_config(self) -> RenderConfig:
        """Get the current rendering configuration."""
        return self.config
    
    def set_config(self, config: RenderConfig):
        """Set the rendering configuration."""
        self.config = config


# Convenience functions for common use cases

def process_markdown(text: str, width: int = 80, enable_colors: bool = True) -> str:
    """
    Quick markdown processing with default settings.
    
    Args:
        text: Text to process
        width: Target width for wrapping
        enable_colors: Whether to enable ANSI colors
        
    Returns:
        Processed text
    """
    config = RenderConfig(width=width, enable_colors=enable_colors)
    processor = ANSIMarkdownProcessor(config)
    return processor.process_text(text)


def render_markdown_lines(text: str, width: int = 80, indent: str = '') -> List[str]:
    """
    Render markdown text as formatted lines.
    
    Args:
        text: Text to render
        width: Target width
        indent: Indentation prefix
        
    Returns:
        List of formatted lines
    """
    config = RenderConfig(width=width)
    processor = ANSIMarkdownProcessor(config)
    return processor.render_lines(text, width, indent)


def strip_markdown_and_ansi(text: str) -> str:
    """
    Strip both markdown formatting and ANSI codes.
    
    Args:
        text: Text to clean
        
    Returns:
        Plain text
    """
    processor = ANSIMarkdownProcessor()
    # First strip ANSI
    clean_text = processor.strip_ansi(text)
    
    # Then strip markdown (simple patterns)
    patterns = [
        (r'\*\*(.+?)\*\*', r'\1'),  # Bold
        (r'(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)', r'\1'),  # Italic
        (r'`([^`]+)`', r'\1'),  # Inline code
        (r'^(\s*#+)\s*(.+)$', r'\2'),  # Headers
        (r'\[([^\]]+)\]\([^\)]+\)', r'\1'),  # Links
    ]
    
    result = clean_text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.MULTILINE)
    
    return result