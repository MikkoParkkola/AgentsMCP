"""
Simple color and styling system for TUI.

Basic color support with fallbacks, clean ASCII art borders and separators,
and high contrast modes for accessibility.
"""

import os
import logging
from typing import Dict, Optional, Any, Tuple, NamedTuple, List
from dataclasses import dataclass
from enum import Enum

from .terminal_manager import TerminalManager


logger = logging.getLogger(__name__)


class ColorMode(Enum):
    """Color mode options."""
    NONE = "none"           # No colors (monochrome)
    BASIC = "basic"         # 8 basic colors
    EXTENDED = "extended"   # 16 colors
    COLOR_256 = "256"       # 256 colors
    TRUE_COLOR = "true"     # 24-bit RGB


class ContrastMode(Enum):
    """Contrast mode for accessibility."""
    NORMAL = "normal"
    HIGH = "high"
    MAXIMUM = "maximum"


class Color(NamedTuple):
    """Represents a color with fallback options."""
    name: str
    true_color: Tuple[int, int, int]  # RGB values
    color_256: int                    # 256-color palette index
    color_16: int                     # 16-color palette index
    color_8: int                      # 8-color palette index
    mono: str                         # Monochrome representation


# Predefined colors with fallbacks
COLORS = {
    'black': Color('black', (0, 0, 0), 0, 0, 0, ' '),
    'red': Color('red', (204, 102, 102), 203, 9, 1, '#'),
    'green': Color('green', (181, 206, 168), 150, 10, 2, '+'),
    'yellow': Color('yellow', (240, 198, 116), 221, 11, 3, '='),
    'blue': Color('blue', (129, 162, 190), 67, 12, 4, '-'),
    'magenta': Color('magenta', (178, 148, 187), 139, 13, 5, '*'),
    'cyan': Color('cyan', (138, 190, 183), 73, 14, 6, '~'),
    'white': Color('white', (235, 235, 235), 188, 15, 7, '.'),
    
    # Extended colors
    'gray': Color('gray', (128, 128, 128), 244, 8, 0, ':'),
    'bright_red': Color('bright_red', (255, 102, 102), 203, 9, 1, '#'),
    'bright_green': Color('bright_green', (102, 255, 102), 46, 10, 2, '+'),
    'bright_yellow': Color('bright_yellow', (255, 255, 102), 11, 11, 3, '='),
    'bright_blue': Color('bright_blue', (102, 178, 255), 75, 12, 4, '-'),
    'bright_magenta': Color('bright_magenta', (255, 102, 255), 207, 13, 5, '*'),
    'bright_cyan': Color('bright_cyan', (102, 255, 255), 51, 14, 6, '~'),
    'bright_white': Color('bright_white', (255, 255, 255), 15, 15, 7, '.'),
    
    # Semantic colors
    'primary': Color('primary', (129, 162, 190), 67, 12, 4, '-'),
    'secondary': Color('secondary', (181, 206, 168), 150, 10, 2, '+'),
    'success': Color('success', (181, 206, 168), 150, 10, 2, '+'),
    'warning': Color('warning', (240, 198, 116), 221, 11, 3, '='),
    'error': Color('error', (204, 102, 102), 203, 9, 1, '#'),
    'info': Color('info', (138, 190, 183), 73, 14, 6, '~'),
    'muted': Color('muted', (128, 128, 128), 244, 8, 0, ':'),
}


@dataclass
class ColorScheme:
    """Color scheme definition."""
    name: str
    foreground: str = 'white'
    background: str = 'black'
    accent: str = 'blue'
    success: str = 'green'
    warning: str = 'yellow' 
    error: str = 'red'
    muted: str = 'gray'
    border: str = 'gray'
    highlight: str = 'cyan'


# Predefined color schemes
COLOR_SCHEMES = {
    'default': ColorScheme(
        name='default',
        foreground='white',
        background='black',
        accent='blue',
        success='green',
        warning='yellow',
        error='red',
        muted='gray',
        border='gray',
        highlight='cyan'
    ),
    
    'dark': ColorScheme(
        name='dark',
        foreground='bright_white',
        background='black',
        accent='bright_blue',
        success='bright_green',
        warning='bright_yellow',
        error='bright_red',
        muted='gray',
        border='white',
        highlight='bright_cyan'
    ),
    
    'light': ColorScheme(
        name='light',
        foreground='black',
        background='white',
        accent='blue',
        success='green',
        warning='yellow',
        error='red',
        muted='gray',
        border='black',
        highlight='cyan'
    ),
    
    'high_contrast': ColorScheme(
        name='high_contrast',
        foreground='bright_white',
        background='black',
        accent='bright_yellow',
        success='bright_green',
        warning='bright_yellow',
        error='bright_red',
        muted='white',
        border='bright_white',
        highlight='bright_white'
    ),
    
    'monochrome': ColorScheme(
        name='monochrome',
        foreground='white',
        background='black',
        accent='white',
        success='white',
        warning='white',
        error='white',
        muted='gray',
        border='white',
        highlight='white'
    )
}


class BorderStyle(Enum):
    """Border style options."""
    NONE = "none"
    SIMPLE = "simple"        # ASCII characters
    ROUNDED = "rounded"      # Unicode rounded corners
    DOUBLE = "double"        # Unicode double lines
    THICK = "thick"          # Unicode thick lines


# Border character sets
BORDER_CHARS = {
    BorderStyle.NONE: {
        'top_left': ' ', 'top': ' ', 'top_right': ' ',
        'left': ' ', 'right': ' ',
        'bottom_left': ' ', 'bottom': ' ', 'bottom_right': ' ',
        'horizontal': ' ', 'vertical': ' ',
        'cross': ' '
    },
    BorderStyle.SIMPLE: {
        'top_left': '+', 'top': '-', 'top_right': '+',
        'left': '|', 'right': '|',
        'bottom_left': '+', 'bottom': '-', 'bottom_right': '+',
        'horizontal': '-', 'vertical': '|',
        'cross': '+'
    },
    BorderStyle.ROUNDED: {
        'top_left': '╭', 'top': '─', 'top_right': '╮',
        'left': '│', 'right': '│',
        'bottom_left': '╰', 'bottom': '─', 'bottom_right': '╯',
        'horizontal': '─', 'vertical': '│',
        'cross': '┼'
    },
    BorderStyle.DOUBLE: {
        'top_left': '╔', 'top': '═', 'top_right': '╗',
        'left': '║', 'right': '║',
        'bottom_left': '╚', 'bottom': '═', 'bottom_right': '╝',
        'horizontal': '═', 'vertical': '║',
        'cross': '╬'
    },
    BorderStyle.THICK: {
        'top_left': '┏', 'top': '━', 'top_right': '┓',
        'left': '┃', 'right': '┃',
        'bottom_left': '┗', 'bottom': '━', 'bottom_right': '┛',
        'horizontal': '━', 'vertical': '┃',
        'cross': '╋'
    }
}


class ThemeManager:
    """
    Simple color and styling system for TUI.
    
    Provides color support with fallbacks, border styles, and
    high contrast modes for accessibility.
    """
    
    def __init__(self, terminal_manager: Optional[TerminalManager] = None):
        """Initialize theme manager."""
        self.terminal_manager = terminal_manager or TerminalManager()
        
        # Current settings
        self._color_mode = ColorMode.NONE
        self._contrast_mode = ContrastMode.NORMAL
        self._border_style = BorderStyle.SIMPLE
        self._color_scheme = COLOR_SCHEMES['default']
        
        # Cache for escape sequences
        self._escape_cache: Dict[str, str] = {}
        
        # Initialize based on terminal capabilities
        self._detect_capabilities()
    
    def _detect_capabilities(self):
        """Detect terminal color and styling capabilities."""
        caps = self.terminal_manager.detect_capabilities()
        
        # Determine color mode
        if caps.colors >= 16777216:  # 24-bit color
            self._color_mode = ColorMode.TRUE_COLOR
        elif caps.colors >= 256:
            self._color_mode = ColorMode.COLOR_256
        elif caps.colors >= 16:
            self._color_mode = ColorMode.EXTENDED
        elif caps.colors >= 8:
            self._color_mode = ColorMode.BASIC
        else:
            self._color_mode = ColorMode.NONE
        
        # Set border style based on unicode support
        if caps.unicode_support:
            self._border_style = BorderStyle.ROUNDED
        else:
            self._border_style = BorderStyle.SIMPLE
        
        # Check for high contrast preference
        if os.getenv('FORCE_COLOR') == '0':
            self._color_mode = ColorMode.NONE
            self._color_scheme = COLOR_SCHEMES['monochrome']
        
        logger.debug(f"Theme initialized: colors={self._color_mode.value}, borders={self._border_style.value}")
    
    def set_color_scheme(self, scheme_name: str) -> bool:
        """Set color scheme by name."""
        if scheme_name in COLOR_SCHEMES:
            self._color_scheme = COLOR_SCHEMES[scheme_name]
            self._escape_cache.clear()
            return True
        return False
    
    def set_contrast_mode(self, mode: ContrastMode):
        """Set contrast mode."""
        self._contrast_mode = mode
        
        # Auto-adjust color scheme for high contrast
        if mode == ContrastMode.HIGH:
            self.set_color_scheme('high_contrast')
        elif mode == ContrastMode.MAXIMUM:
            self.set_color_scheme('monochrome')
        
        self._escape_cache.clear()
    
    def set_border_style(self, style: BorderStyle):
        """Set border style."""
        self._border_style = style
    
    def get_color_escape(self, color_name: str, background: bool = False) -> str:
        """
        Get ANSI escape sequence for a color.
        
        Args:
            color_name: Name of the color
            background: If True, return background color escape
            
        Returns:
            ANSI escape sequence or empty string
        """
        if self._color_mode == ColorMode.NONE:
            return ""
        
        cache_key = f"{color_name}:{'bg' if background else 'fg'}"
        if cache_key in self._escape_cache:
            return self._escape_cache[cache_key]
        
        color = COLORS.get(color_name)
        if not color:
            return ""
        
        escape = ""
        
        if self._color_mode == ColorMode.TRUE_COLOR:
            # 24-bit RGB
            r, g, b = color.true_color
            escape = f"\033[{'48' if background else '38'};2;{r};{g};{b}m"
        
        elif self._color_mode == ColorMode.COLOR_256:
            # 256-color palette
            escape = f"\033[{'48' if background else '38'};5;{color.color_256}m"
        
        elif self._color_mode == ColorMode.EXTENDED:
            # 16-color palette
            base = 40 if background else 30
            escape = f"\033[{base + color.color_16}m"
        
        elif self._color_mode == ColorMode.BASIC:
            # 8-color palette
            base = 40 if background else 30
            escape = f"\033[{base + color.color_8}m"
        
        self._escape_cache[cache_key] = escape
        return escape
    
    def get_reset_escape(self) -> str:
        """Get reset escape sequence."""
        return "\033[0m" if self._color_mode != ColorMode.NONE else ""
    
    def colorize(self, text: str, color: Optional[str] = None, background: Optional[str] = None) -> str:
        """
        Colorize text with foreground and/or background colors.
        
        Args:
            text: Text to colorize
            color: Foreground color name
            background: Background color name
            
        Returns:
            Colorized text with escape sequences
        """
        if self._color_mode == ColorMode.NONE:
            return text
        
        parts = []
        
        if color:
            parts.append(self.get_color_escape(color, False))
        if background:
            parts.append(self.get_color_escape(background, True))
        
        parts.append(text)
        
        if parts[:-1]:  # If we added any escape sequences
            parts.append(self.get_reset_escape())
        
        return "".join(parts)
    
    def style_text(self, text: str, 
                   color: Optional[str] = None,
                   background: Optional[str] = None,
                   bold: bool = False,
                   dim: bool = False,
                   underline: bool = False) -> str:
        """
        Apply multiple styles to text.
        
        Args:
            text: Text to style
            color: Foreground color
            background: Background color
            bold: Make text bold
            dim: Make text dim
            underline: Underline text
            
        Returns:
            Styled text
        """
        if self._color_mode == ColorMode.NONE:
            return text
        
        parts = []
        
        # Color escapes
        if color:
            parts.append(self.get_color_escape(color, False))
        if background:
            parts.append(self.get_color_escape(background, True))
        
        # Style escapes
        if bold:
            parts.append("\033[1m")
        if dim:
            parts.append("\033[2m")
        if underline:
            parts.append("\033[4m")
        
        parts.append(text)
        
        if parts[:-1]:  # If we added any escape sequences
            parts.append(self.get_reset_escape())
        
        return "".join(parts)
    
    def get_themed_color(self, semantic_name: str) -> str:
        """Get themed color by semantic name."""
        color_name = getattr(self._color_scheme, semantic_name, 'white')
        return color_name
    
    def get_border_chars(self, style: Optional[BorderStyle] = None) -> Dict[str, str]:
        """Get border characters for the current or specified style."""
        border_style = style or self._border_style
        return BORDER_CHARS[border_style].copy()
    
    def draw_box(self, width: int, height: int, 
                 title: Optional[str] = None,
                 border_color: Optional[str] = None,
                 title_color: Optional[str] = None,
                 style: Optional[BorderStyle] = None) -> List[str]:
        """
        Draw a text box with borders.
        
        Args:
            width: Box width including borders
            height: Box height including borders
            title: Optional title text
            border_color: Border color name
            title_color: Title color name
            style: Border style override
            
        Returns:
            List of strings representing the box lines
        """
        if width < 3 or height < 3:
            return [""] * height
        
        border_style = style or self._border_style
        chars = BORDER_CHARS[border_style]
        
        # Colors
        border_col = border_color or self.get_themed_color('border')
        title_col = title_color or self.get_themed_color('accent')
        
        lines = []
        
        # Top border
        top_line = chars['top_left'] + chars['top'] * (width - 2) + chars['top_right']
        if title and len(title) < width - 4:
            # Insert title in top border
            title_text = f" {title} "
            if title_col:
                title_text = self.colorize(title_text, title_col)
            
            start_pos = (width - len(title) - 2) // 2
            top_line = (chars['top_left'] + chars['top'] * (start_pos - 1) + 
                       title_text + 
                       chars['top'] * (width - start_pos - len(title) - 3) + 
                       chars['top_right'])
        
        if border_col:
            top_line = self.colorize(top_line, border_col)
        lines.append(top_line)
        
        # Middle lines
        side_char = self.colorize(chars['left'], border_col) if border_col else chars['left']
        for _ in range(height - 2):
            line = side_char + " " * (width - 2) + side_char
            lines.append(line)
        
        # Bottom border
        bottom_line = chars['bottom_left'] + chars['bottom'] * (width - 2) + chars['bottom_right']
        if border_col:
            bottom_line = self.colorize(bottom_line, border_col)
        lines.append(bottom_line)
        
        return lines
    
    def draw_separator(self, width: int, char: Optional[str] = None, color: Optional[str] = None) -> str:
        """
        Draw a horizontal separator line.
        
        Args:
            width: Separator width
            char: Character to use (default from border style)
            color: Color name
            
        Returns:
            Separator line string
        """
        sep_char = char or BORDER_CHARS[self._border_style]['horizontal']
        line = sep_char * width
        
        if color:
            line = self.colorize(line, color)
        
        return line
    
    def get_monochrome_indicator(self, color_name: str) -> str:
        """Get monochrome indicator for a color."""
        color = COLORS.get(color_name)
        return color.mono if color else ' '
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get theme capabilities information."""
        return {
            'color_mode': self._color_mode.value,
            'colors_available': self.terminal_manager.detect_capabilities().colors,
            'unicode_support': self.terminal_manager.supports_unicode(),
            'contrast_mode': self._contrast_mode.value,
            'border_style': self._border_style.value,
            'color_scheme': self._color_scheme.name,
            'escape_cache_size': len(self._escape_cache)
        }


# Convenience functions
def create_theme_manager(terminal_manager: Optional[TerminalManager] = None) -> ThemeManager:
    """Create and return a new ThemeManager instance."""
    return ThemeManager(terminal_manager)


def detect_preferred_scheme() -> str:
    """Detect preferred color scheme from environment."""
    # Check environment variables
    if os.getenv('FORCE_COLOR') == '0':
        return 'monochrome'
    
    # Check for dark mode preferences
    term_program = os.getenv('TERM_PROGRAM', '').lower()
    if 'dark' in term_program:
        return 'dark'
    
    # Check color term
    colorterm = os.getenv('COLORTERM', '').lower()
    if colorterm in ('truecolor', '24bit'):
        return 'dark'
    
    # Default
    return 'default'