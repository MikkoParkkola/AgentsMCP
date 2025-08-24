"""
Revolutionary UI Components - Building Blocks for Beautiful CLIs

Comprehensive collection of reusable UI components inspired by:
- Claude Code's clean typography and spacing
- Codex CLI's interactive elements
- Gemini CLI's status indicators and progress bars

Features:
- Adaptive theming integration
- Smooth animations and transitions
- Accessibility-compliant design
- Responsive layout system
"""

import os
import sys
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from .theme_manager import ThemeManager

logger = logging.getLogger(__name__)

@dataclass
class BoxStyle:
    """Configuration for box drawing"""
    top_left: str = "┌"
    top_right: str = "┐"
    bottom_left: str = "└"
    bottom_right: str = "┘"
    horizontal: str = "─"
    vertical: str = "│"
    cross: str = "┼"
    t_down: str = "┬"
    t_up: str = "┴"
    t_right: str = "├"
    t_left: str = "┤"

@dataclass
class ProgressConfig:
    """Configuration for progress bars"""
    width: int = 40
    fill_char: str = "█"
    empty_char: str = "░"
    show_percentage: bool = True
    show_eta: bool = True
    show_rate: bool = False

class UIComponents:
    """
    Revolutionary UI Components Collection
    
    Provides beautiful, themeable UI components for creating
    sophisticated command-line interfaces.
    """
    
    def __init__(self, theme_manager: Optional[ThemeManager] = None):
        self.theme_manager = theme_manager or ThemeManager()
        self.terminal_width = self._get_terminal_width()
        self.terminal_height = self._get_terminal_height()
        
        # Box drawing styles
        self.box_styles = {
            'light': BoxStyle(),
            'heavy': BoxStyle("┏", "┓", "┗", "┛", "━", "┃", "╋", "┳", "┻", "┣", "┫"),
            'double': BoxStyle("╔", "╗", "╚", "╝", "═", "║", "╬", "╦", "╩", "╠", "╣"),
            'rounded': BoxStyle("╭", "╮", "╰", "╯", "─", "│", "┼", "┬", "┴", "├", "┤")
        }
        
        # Animation state
        self._animations = {}
        self._animation_lock = threading.Lock()
    
    def _get_terminal_width(self) -> int:
        """Get terminal width with fallback"""
        try:
            return os.get_terminal_size().columns
        except (OSError, ValueError):
            return 80  # Fallback width
    
    def _get_terminal_height(self) -> int:
        """Get terminal height with fallback"""
        try:
            return os.get_terminal_size().lines
        except (OSError, ValueError):
            return 24  # Fallback height
    
    def heading(self, text: str, level: int = 1, centered: bool = False, 
               underline: bool = True) -> str:
        """
        Create beautiful headings with proper hierarchy
        
        Args:
            text: Heading text
            level: Heading level (1-6)
            centered: Whether to center the heading
            underline: Whether to add underline decoration
            
        Returns:
            Formatted heading string
        """
        theme = self.theme_manager.get_current_theme()
        
        # Style based on level
        if level == 1:
            styled_text = theme.heading_style + text
            underline_char = "═"
        elif level == 2:
            styled_text = theme.subheading_style + text
            underline_char = "─"
        elif level == 3:
            styled_text = theme.palette.text_primary + text
            underline_char = "·"
        else:
            styled_text = theme.palette.text_secondary + text
            underline_char = "·"
        
        # Apply theming
        result = self.theme_manager.style_text(styled_text, 'heading_style' if level <= 2 else 'body_style')
        
        # Center if requested
        if centered:
            padding = (self.terminal_width - len(text)) // 2
            result = " " * padding + result
        
        # Add underline
        if underline and level <= 3:
            underline_text = underline_char * len(text)
            if centered:
                underline_text = " " * padding + underline_text
            result += "\n" + self.theme_manager.colorize(underline_text, 'separator')
        
        return result
    
    def box(self, content: str, title: str = "", style: str = 'light', 
           padding: int = 1, width: Optional[int] = None) -> str:
        """
        Create beautiful boxes around content
        
        Args:
            content: Content to box
            title: Optional title for the box
            style: Box style ('light', 'heavy', 'double', 'rounded')
            padding: Internal padding
            width: Fixed width (auto-calculated if None)
            
        Returns:
            Boxed content string
        """
        box_style = self.box_styles.get(style, self.box_styles['light'])
        theme = self.theme_manager.get_current_theme()
        
        # Split content into lines
        content_lines = content.split('\n')
        
        # Calculate dimensions
        content_width = max(len(line) for line in content_lines) if content_lines else 0
        title_width = len(title) if title else 0
        
        if width is None:
            width = max(content_width + 2 * padding, title_width + 4)
        
        # Build box
        lines = []
        
        # Top border with title
        if title:
            title_padding = (width - len(title) - 2) // 2
            top_line = (box_style.top_left + 
                       box_style.horizontal * title_padding +
                       f" {title} " +
                       box_style.horizontal * (width - title_padding - len(title) - 2) +
                       box_style.top_right)
        else:
            top_line = (box_style.top_left + 
                       box_style.horizontal * width +
                       box_style.top_right)
        
        lines.append(self.theme_manager.colorize(top_line, 'border'))
        
        # Padding lines (top)
        for _ in range(padding):
            padding_line = (box_style.vertical + 
                           " " * width + 
                           box_style.vertical)
            lines.append(self.theme_manager.colorize(padding_line, 'border'))
        
        # Content lines
        for line in content_lines:
            content_line = (box_style.vertical + 
                           " " * padding +
                           line.ljust(width - 2 * padding) +
                           " " * padding +
                           box_style.vertical)
            # Mix content and border colors
            border_color = self.theme_manager.get_color('border')
            formatted_line = (border_color + box_style.vertical +
                            theme.body_style + " " * padding + line.ljust(width - 2 * padding) + " " * padding +
                            border_color + box_style.vertical)
            lines.append(formatted_line)
        
        # Padding lines (bottom)
        for _ in range(padding):
            padding_line = (box_style.vertical + 
                           " " * width + 
                           box_style.vertical)
            lines.append(self.theme_manager.colorize(padding_line, 'border'))
        
        # Bottom border
        bottom_line = (box_style.bottom_left + 
                      box_style.horizontal * width +
                      box_style.bottom_right)
        lines.append(self.theme_manager.colorize(bottom_line, 'border'))
        
        return '\n'.join(lines)
    
    def progress_bar(self, progress: float, config: Optional[ProgressConfig] = None,
                    label: str = "", eta: Optional[timedelta] = None) -> str:
        """
        Create beautiful progress bars with animations
        
        Args:
            progress: Progress value (0.0 to 1.0)
            config: Progress bar configuration
            label: Progress label
            eta: Estimated time to completion
            
        Returns:
            Formatted progress bar string
        """
        if config is None:
            config = ProgressConfig()
        
        theme = self.theme_manager.get_current_theme()
        
        # Calculate filled/empty segments
        filled_width = int(progress * config.width)
        empty_width = config.width - filled_width
        
        # Create bar components
        filled_part = self.theme_manager.colorize(
            config.fill_char * filled_width, 'success', reset=False
        )
        empty_part = self.theme_manager.colorize(
            config.empty_char * empty_width, 'text_muted', reset=True
        )
        
        progress_bar = f"[{filled_part}{empty_part}]"
        
        # Add percentage
        percentage = ""
        if config.show_percentage:
            percentage = f" {progress * 100:5.1f}%"
        
        # Add ETA
        eta_text = ""
        if config.show_eta and eta is not None:
            eta_text = f" ETA: {self._format_timedelta(eta)}"
        
        # Combine components
        result = progress_bar + self.theme_manager.colorize(percentage, 'text_secondary', reset=False)
        
        if eta_text:
            result += self.theme_manager.colorize(eta_text, 'text_muted')
        
        # Add label if provided
        if label:
            label_styled = self.theme_manager.colorize(label, 'text_primary')
            result = f"{label_styled}\n{result}"
        
        return result
    
    def status_indicator(self, status: str, message: str = "", 
                        icon: Optional[str] = None) -> str:
        """
        Create status indicators with appropriate colors and icons
        
        Args:
            status: Status type ('success', 'error', 'warning', 'info', 'loading')
            message: Status message
            icon: Custom icon (uses default if None)
            
        Returns:
            Formatted status indicator string
        """
        theme = self.theme_manager.get_current_theme()
        
        # Define status styles and icons
        status_config = {
            'success': {'color': 'success', 'icon': '✓'},
            'error': {'color': 'error', 'icon': '✗'},
            'warning': {'color': 'warning', 'icon': '⚠'},
            'info': {'color': 'info', 'icon': 'ⓘ'},
            'loading': {'color': 'primary', 'icon': '⟳'},
            'active': {'color': 'primary', 'icon': '●'},
            'inactive': {'color': 'text_muted', 'icon': '○'}
        }
        
        config = status_config.get(status, status_config['info'])
        display_icon = icon or config['icon']
        
        # Format components
        icon_colored = self.theme_manager.colorize(display_icon, config['color'])
        
        if message:
            message_colored = self.theme_manager.colorize(message, 'text_primary')
            return f"{icon_colored} {message_colored}"
        else:
            return icon_colored
    
    def table(self, headers: List[str], rows: List[List[str]], 
             title: str = "", max_width: Optional[int] = None) -> str:
        """
        Create beautiful tables with proper alignment and theming
        
        Args:
            headers: Column headers
            rows: Table rows
            title: Optional table title
            max_width: Maximum table width
            
        Returns:
            Formatted table string
        """
        if not headers or not rows:
            return ""
        
        theme = self.theme_manager.get_current_theme()
        
        # Calculate column widths
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Apply max width constraint
        if max_width:
            total_width = sum(col_widths) + len(headers) * 3 + 1
            if total_width > max_width:
                # Proportionally reduce column widths
                reduction_factor = (max_width - len(headers) * 3 - 1) / sum(col_widths)
                col_widths = [max(8, int(w * reduction_factor)) for w in col_widths]
        
        # Build table
        lines = []
        
        # Title
        if title:
            lines.append(self.theme_manager.style_text(title, 'heading_style'))
            lines.append("")
        
        # Top border
        top_border = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
        lines.append(self.theme_manager.colorize(top_border, 'border'))
        
        # Header row
        header_cells = []
        for i, header in enumerate(headers):
            cell = f" {header.ljust(col_widths[i])} "
            header_cells.append(self.theme_manager.colorize(cell, 'text_primary'))
        
        header_row = self.theme_manager.colorize("│", 'border') + self.theme_manager.colorize("│", 'border').join(header_cells) + self.theme_manager.colorize("│", 'border')
        lines.append(header_row)
        
        # Header separator
        header_sep = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
        lines.append(self.theme_manager.colorize(header_sep, 'border'))
        
        # Data rows
        for row in rows:
            data_cells = []
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    cell_text = str(cell).ljust(col_widths[i])
                    if i < len(col_widths):
                        cell_formatted = f" {cell_text} "
                        data_cells.append(self.theme_manager.colorize(cell_formatted, 'text_secondary'))
            
            if data_cells:
                data_row = self.theme_manager.colorize("│", 'border') + self.theme_manager.colorize("│", 'border').join(data_cells) + self.theme_manager.colorize("│", 'border')
                lines.append(data_row)
        
        # Bottom border
        bottom_border = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"
        lines.append(self.theme_manager.colorize(bottom_border, 'border'))
        
        return '\n'.join(lines)
    
    def card(self, title: str, content: str, status: Optional[str] = None,
            width: Optional[int] = None) -> str:
        """
        Create card-style content blocks
        
        Args:
            title: Card title
            content: Card content
            status: Optional status indicator
            width: Card width
            
        Returns:
            Formatted card string
        """
        if width is None:
            width = min(60, self.terminal_width - 4)
        
        theme = self.theme_manager.get_current_theme()
        
        # Build card header
        header_parts = [self.theme_manager.style_text(title, 'heading_style')]
        
        if status:
            status_indicator = self.status_indicator(status)
            header_parts.append(status_indicator)
        
        header = " ".join(header_parts)
        
        # Build card content
        card_content = f"{header}\n{theme.palette.separator}{'─' * (width - 2)}\n{content}"
        
        return self.box(card_content, style='rounded', padding=2, width=width)
    
    def list_item(self, text: str, level: int = 0, bullet: str = "•",
                 status: Optional[str] = None) -> str:
        """
        Create formatted list items with proper indentation
        
        Args:
            text: Item text
            level: Indentation level
            bullet: Bullet character
            status: Optional status indicator
            
        Returns:
            Formatted list item string
        """
        indent = "  " * level
        
        if status:
            bullet = self.status_indicator(status, "")
        else:
            bullet = self.theme_manager.colorize(bullet, 'primary')
        
        text_colored = self.theme_manager.colorize(text, 'text_primary')
        
        return f"{indent}{bullet} {text_colored}"
    
    def separator(self, char: str = "─", width: Optional[int] = None,
                 label: str = "") -> str:
        """
        Create decorative separators
        
        Args:
            char: Separator character
            width: Separator width
            label: Optional label in the middle
            
        Returns:
            Formatted separator string
        """
        if width is None:
            width = self.terminal_width - 2
        
        if label:
            label_len = len(label)
            left_width = (width - label_len - 2) // 2
            right_width = width - label_len - 2 - left_width
            
            left_part = self.theme_manager.colorize(char * left_width, 'separator')
            label_part = self.theme_manager.colorize(f" {label} ", 'text_primary')
            right_part = self.theme_manager.colorize(char * right_width, 'separator')
            
            return f"{left_part}{label_part}{right_part}"
        else:
            return self.theme_manager.colorize(char * width, 'separator')
    
    def panel(self, content: str, title: str = "", status: str = "info") -> str:
        """
        Create information panels with status-based theming
        
        Args:
            content: Panel content
            title: Panel title
            status: Panel status for color theming
            
        Returns:
            Formatted panel string
        """
        # Choose border style based on status
        style_map = {
            'success': 'light',
            'error': 'heavy',
            'warning': 'double',
            'info': 'rounded'
        }
        
        box_style = style_map.get(status, 'light')
        
        # Add status indicator to title if provided
        if title:
            status_icon = self.status_indicator(status, "")
            full_title = f"{status_icon} {title}"
        else:
            full_title = ""
        
        return self.box(content, title=full_title, style=box_style, padding=1)
    
    def metric_display(self, label: str, value: str, unit: str = "",
                      change: Optional[float] = None, format_large_numbers: bool = True) -> str:
        """
        Display metrics with optional change indicators
        
        Args:
            label: Metric label
            value: Metric value
            unit: Optional unit
            change: Optional change percentage
            format_large_numbers: Whether to format large numbers
            
        Returns:
            Formatted metric display string
        """
        theme = self.theme_manager.get_current_theme()
        
        # Format value
        if format_large_numbers and value.replace('.', '').isdigit():
            num_value = float(value)
            if num_value >= 1_000_000:
                formatted_value = f"{num_value / 1_000_000:.1f}M"
            elif num_value >= 1_000:
                formatted_value = f"{num_value / 1_000:.1f}K"
            else:
                formatted_value = value
        else:
            formatted_value = value
        
        # Add unit
        if unit:
            formatted_value = f"{formatted_value} {unit}"
        
        # Style components
        label_styled = self.theme_manager.colorize(label, 'text_muted')
        value_styled = self.theme_manager.colorize(formatted_value, 'primary')
        
        result = f"{label_styled}: {value_styled}"
        
        # Add change indicator
        if change is not None:
            if change > 0:
                change_text = f"↗ +{change:.1f}%"
                change_color = 'success'
            elif change < 0:
                change_text = f"↘ {change:.1f}%"
                change_color = 'error'
            else:
                change_text = "→ 0.0%"
                change_color = 'text_muted'
            
            change_styled = self.theme_manager.colorize(change_text, change_color)
            result = f"{result} {change_styled}"
        
        return result
    
    def loading_spinner(self, message: str = "Loading", animation: str = "dots") -> str:
        """
        Create loading spinners (static representation)
        
        Args:
            message: Loading message
            animation: Animation type ('dots', 'spinner', 'bar')
            
        Returns:
            Formatted loading indicator string
        """
        animations = {
            'dots': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
            'spinner': ['|', '/', '─', '\\'],
            'bar': ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        }
        
        # For static display, just show first frame
        spinner_chars = animations.get(animation, animations['dots'])
        spinner = self.theme_manager.colorize(spinner_chars[0], 'primary')
        message_styled = self.theme_manager.colorize(message, 'text_primary')
        
        return f"{spinner} {message_styled}"
    
    def _format_timedelta(self, td: timedelta) -> str:
        """Format timedelta for display"""
        total_seconds = int(td.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    def multi_column_layout(self, columns: List[Tuple[str, int]], 
                           gap: int = 2) -> str:
        """
        Create multi-column layouts
        
        Args:
            columns: List of (content, width) tuples
            gap: Gap between columns
            
        Returns:
            Formatted multi-column layout string
        """
        if not columns:
            return ""
        
        # Split content into lines for each column
        column_lines = []
        max_lines = 0
        
        for content, width in columns:
            lines = content.split('\n')
            # Wrap lines that are too long
            wrapped_lines = []
            for line in lines:
                if len(line) <= width:
                    wrapped_lines.append(line.ljust(width))
                else:
                    # Simple word wrap
                    words = line.split()
                    current_line = ""
                    for word in words:
                        if len(current_line + word) <= width:
                            current_line += word + " "
                        else:
                            wrapped_lines.append(current_line.ljust(width))
                            current_line = word + " "
                    if current_line:
                        wrapped_lines.append(current_line.ljust(width))
            
            column_lines.append(wrapped_lines)
            max_lines = max(max_lines, len(wrapped_lines))
        
        # Pad columns to same height
        for col_lines in column_lines:
            width = columns[column_lines.index(col_lines)][1]
            while len(col_lines) < max_lines:
                col_lines.append(" " * width)
        
        # Combine columns
        result_lines = []
        gap_str = " " * gap
        
        for i in range(max_lines):
            row_parts = []
            for j, col_lines in enumerate(column_lines):
                row_parts.append(col_lines[i])
            result_lines.append(gap_str.join(row_parts))
        
        return '\n'.join(result_lines)
    
    def refresh_terminal_size(self):
        """Refresh terminal dimensions"""
        self.terminal_width = self._get_terminal_width()
        self.terminal_height = self._get_terminal_height()
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def move_cursor(self, x: int, y: int) -> str:
        """Generate ANSI code to move cursor"""
        return f"\033[{y};{x}H"
    
    def hide_cursor(self) -> str:
        """Generate ANSI code to hide cursor"""
        return "\033[?25l"
    
    def show_cursor(self) -> str:
        """Generate ANSI code to show cursor"""
        return "\033[?25h"