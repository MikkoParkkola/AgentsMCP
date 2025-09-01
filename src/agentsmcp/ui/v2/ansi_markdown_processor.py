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
            'bold_yellow': TextStyle(color=ANSIColor.BRIGHT_YELLOW, bold=True),
            'cyan': TextStyle(color=ANSIColor.BRIGHT_CYAN),
            'blue': TextStyle(color=ANSIColor.BRIGHT_BLUE),
            'blue_underline': TextStyle(color=ANSIColor.BRIGHT_BLUE, underline=True),
            'green': TextStyle(color=ANSIColor.BRIGHT_GREEN),
            'red': TextStyle(color=ANSIColor.BRIGHT_RED),
            'magenta': TextStyle(color=ANSIColor.BRIGHT_MAGENTA),
            'yellow': TextStyle(color=ANSIColor.BRIGHT_YELLOW),
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
            'table_row': re.compile(r'^\s*\|(.+)\|\s*$', re.MULTILINE),
            'table_separator': re.compile(r'^\s*\|[\s:|-]+\|\s*$', re.MULTILINE),
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
            result = self._process_tables(result)
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
    
    def _process_tables(self, text: str) -> str:
        """Process markdown tables with proper alignment."""
        lines = text.split('\n')
        result_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this line looks like a table row
            if self._patterns['table_row'].match(line):
                # Find the complete table
                table_start = i
                table_lines = []
                
                # Collect all consecutive table lines
                while i < len(lines) and (self._patterns['table_row'].match(lines[i]) or self._patterns['table_separator'].match(lines[i])):
                    table_lines.append(lines[i])
                    i += 1
                
                # Process the table
                formatted_table = self._format_table(table_lines)
                result_lines.extend(formatted_table)
            else:
                result_lines.append(line)
                i += 1
        
        return '\n'.join(result_lines)
    
    def _format_table(self, table_lines: List[str]) -> List[str]:
        """Format a markdown table with proper alignment and colors."""
        if not table_lines:
            return []
        
        # Parse table structure
        rows = []
        separator_idx = None
        
        for idx, line in enumerate(table_lines):
            if self._patterns['table_separator'].match(line):
                separator_idx = idx
            else:
                # Extract cells from table row
                cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first/last
                rows.append(cells)
        
        if not rows:
            return table_lines
        
        # Calculate column widths
        num_cols = len(rows[0]) if rows else 0
        col_widths = [0] * num_cols
        
        for row in rows:
            for i, cell in enumerate(row[:num_cols]):
                # Calculate visual width (without ANSI codes)
                visual_width = self._calculate_visual_length(cell)
                col_widths[i] = max(col_widths[i], visual_width, 3)  # Minimum width 3
        
        # Apply maximum width constraint
        max_col_width = max(10, (self.config.width - num_cols * 3) // num_cols)
        col_widths = [min(w, max_col_width) for w in col_widths]
        
        formatted_rows = []
        
        for row_idx, row in enumerate(rows):
            # Add separator after header row if we found one
            if row_idx == 1 and separator_idx is not None and len(formatted_rows) > 0:
                if self.config.enable_colors:
                    style = self._get_style('dim')
                    separator = f"{style.to_ansi()}├{'─┼─'.join(['─' * w for w in col_widths])}┤{ANSIStyle.RESET.value}"
                else:
                    separator = f"├{'─┼─'.join(['─' * w for w in col_widths])}┤"
                formatted_rows.append(separator)
            
            # Format the data row
            formatted_cells = []
            for i, cell in enumerate(row[:num_cols]):
                width = col_widths[i]
                
                # Truncate if too long
                visual_len = self._calculate_visual_length(cell)
                if visual_len > width:
                    # Simple truncation - could be improved with ellipsis
                    cell = cell[:width-1] + '…'
                
                # Pad to width
                padding = width - self._calculate_visual_length(cell)
                padded_cell = cell + (' ' * padding)
                
                # Apply header styling for first row
                if row_idx == 0 and self.config.enable_colors:
                    style = self._get_style('bold')
                    padded_cell = f"{style.to_ansi()}{padded_cell}{ANSIStyle.RESET.value}"
                
                formatted_cells.append(padded_cell)
            
            # Assemble the row with borders
            if self.config.enable_colors:
                border_style = self._get_style('dim')
                border_char = f"{border_style.to_ansi()}│{ANSIStyle.RESET.value}"
            else:
                border_char = '│'
                
            row_content = f" {border_char} ".join(formatted_cells)
            formatted_row = f"{border_char} {row_content} {border_char}"
            formatted_rows.append(formatted_row)
        
        # Add top and bottom borders
        if self.config.enable_colors:
            style = self._get_style('dim')
            top_border = f"{style.to_ansi()}┌{'─┬─'.join(['─' * w for w in col_widths])}┐{ANSIStyle.RESET.value}"
            bottom_border = f"{style.to_ansi()}└{'─┴─'.join(['─' * w for w in col_widths])}┘{ANSIStyle.RESET.value}"
        else:
            top_border = f"┌{'─┬─'.join(['─' * w for w in col_widths])}┐"
            bottom_border = f"└{'─┴─'.join(['─' * w for w in col_widths])}┘"
        
        return [top_border] + formatted_rows + [bottom_border]
    
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
        """Apply text wrapping while preserving ANSI codes and structure."""
        if not self.config.enable_wrapping or self.config.width <= 0:
            return text
        
        lines = text.split('\n')
        wrapped_lines = []
        
        for line in lines:
            if not line.strip():
                # Preserve empty lines as-is
                wrapped_lines.append(line)
                continue
                
            # Check if this is a list item (starts with bullet or number)
            list_match = re.match(r'^(\s*)(•|\d+\.|[-*+])\s+(.+)$', line)
            if list_match:
                indent_prefix = list_match.group(1)  # Leading whitespace
                marker = list_match.group(2)         # Bullet or number
                content = list_match.group(3)        # Content after marker
                
                # Calculate the indentation for continuation lines
                marker_width = len(marker) + 1  # marker + space
                continuation_indent = indent_prefix + ' ' * marker_width
                
                # Wrap the content
                available_width = self.config.width - len(continuation_indent)
                if available_width > 10:  # Ensure reasonable width
                    if self._line_contains_ansi(content):
                        wrapped_content = self._wrap_ansi_line(content, available_width)
                    else:
                        wrapped_content = textwrap.wrap(
                            content,
                            width=available_width,
                            break_long_words=False,
                            break_on_hyphens=True
                        ) or [content]
                    
                    # Add the first line with the marker
                    first_line = f"{indent_prefix}{marker} {wrapped_content[0]}"
                    wrapped_lines.append(first_line)
                    
                    # Add continuation lines with proper indentation
                    for continuation_line in wrapped_content[1:]:
                        wrapped_lines.append(f"{continuation_indent}{continuation_line}")
                else:
                    # If not enough space for proper wrapping, keep as-is
                    wrapped_lines.append(line)
            elif self._line_contains_ansi(line):
                # Handle ANSI-styled lines specially
                wrapped_lines.extend(self._wrap_ansi_line(line, self.config.width))
            else:
                # Regular text wrapping
                if len(line) <= self.config.width:
                    wrapped_lines.append(line)
                else:
                    wrapped_content = textwrap.wrap(
                        line, 
                        width=self.config.width,
                        break_long_words=False,
                        break_on_hyphens=True
                    ) or [line]
                    wrapped_lines.extend(wrapped_content)
        
        return '\n'.join(wrapped_lines)
    
    def _line_contains_ansi(self, line: str) -> bool:
        """Check if line contains ANSI escape codes."""
        return '\x1b[' in line
    
    def _wrap_ansi_line(self, line: str, max_width: int = None) -> List[str]:
        """Wrap a line containing ANSI codes."""
        # Use provided width or fall back to config width
        target_width = max_width if max_width is not None else self.config.width
        
        # For now, preserve ANSI lines as-is if they're not too long
        visual_length = self._calculate_visual_length(line)
        
        if visual_length <= target_width:
            return [line]
        
        # If too long, attempt simple breaking at word boundaries
        parts = line.split(' ')
        current_line = ""
        result_lines = []
        
        for part in parts:
            test_line = f"{current_line} {part}" if current_line else part
            if self._calculate_visual_length(test_line) <= target_width:
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
        # Update config width for processing
        original_width = self.config.width
        self.config.width = max(20, width - len(indent_prefix))
        
        try:
            processed_text = self.process_text(text)
            lines = processed_text.split('\n')
            
            result_lines = []
            for line in lines:
                # Apply prefix
                prefixed_line = indent_prefix + line
                
                # Check visual length (excluding ANSI codes)
                visual_length = self._calculate_visual_length(prefixed_line)
                
                if visual_length <= width:
                    # Line fits - add as is
                    result_lines.append(prefixed_line)
                else:
                    # Line too long - wrap it properly
                    if self._line_contains_ansi(line):
                        # For ANSI lines, wrap carefully to preserve formatting
                        wrapped = self._wrap_ansi_line(line, width - len(indent_prefix))
                        for wrapped_line in wrapped:
                            result_lines.append(indent_prefix + wrapped_line)
                    else:
                        # For plain text, use standard wrapping
                        wrapped = textwrap.wrap(
                            line, 
                            width=width - len(indent_prefix),
                            break_long_words=False,
                            expand_tabs=False
                        )
                        if wrapped:
                            for wrapped_line in wrapped:
                                result_lines.append(indent_prefix + wrapped_line)
                        else:
                            # Empty or whitespace-only line
                            result_lines.append(indent_prefix)
            
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
    config = RenderConfig(
        width=width,
        enable_colors=True,
        enable_markdown=True,
        enable_wrapping=True,
        header_style='bold_yellow',
        code_style='cyan',
        emphasis_style='italic',
        strong_style='bold',
        quote_style='dim',
        link_style='blue_underline'
    )
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