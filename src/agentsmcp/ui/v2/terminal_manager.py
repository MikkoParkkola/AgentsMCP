"""
Terminal capability detection without Rich dependencies.

Provides clean TTY detection that actually works, terminal dimensions 
and capability queries, with safe fallback modes for non-interactive environments.
"""

import os
import sys
import shutil
import subprocess
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum


class TerminalType(Enum):
    """Types of terminal environments."""
    FULL_TTY = "full_tty"          # Full interactive terminal
    PARTIAL_TTY = "partial_tty"    # Some TTY capabilities
    PIPE = "pipe"                  # Piped input/output
    REDIRECTED = "redirected"      # Redirected streams
    UNKNOWN = "unknown"            # Cannot determine


@dataclass
class TerminalCapabilities:
    """Terminal capability information."""
    type: TerminalType
    width: int
    height: int
    colors: int
    unicode_support: bool
    mouse_support: bool
    alternate_screen: bool
    cursor_control: bool
    interactive: bool
    term_program: Optional[str] = None
    term_version: Optional[str] = None


class TerminalManager:
    """
    Terminal capability detection and management without Rich dependencies.
    
    Provides reliable TTY detection that works in actual terminal environments.
    """
    
    def __init__(self):
        """Initialize the terminal manager."""
        self._capabilities: Optional[TerminalCapabilities] = None
        self._cache_valid = False
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the terminal manager asynchronously.
        
        Performs initial terminal capability detection and validation.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Detect initial capabilities
            self._capabilities = self.detect_capabilities()
            
            # Validate basic terminal functionality
            if self._capabilities.type == TerminalType.UNKNOWN:
                return False
            
            # Mark as initialized
            self._initialized = True
            return True
            
        except Exception as e:
            # Log error but don't raise - return False to indicate failure
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to initialize terminal manager: {e}")
            return False
    
    def get_capabilities(self) -> Optional[TerminalCapabilities]:
        """Get current terminal capabilities.
        
        Returns:
            TerminalCapabilities object if initialized, None otherwise
        """
        if not self._initialized:
            return None
        return self._capabilities
        
    def detect_capabilities(self, force_refresh: bool = False) -> TerminalCapabilities:
        """
        Detect terminal capabilities.
        
        Args:
            force_refresh: Force re-detection even if cached
            
        Returns:
            TerminalCapabilities object
        """
        if self._capabilities and not force_refresh and self._cache_valid:
            return self._capabilities
            
        # Detect terminal type
        terminal_type = self._detect_terminal_type()
        
        # Get dimensions
        width, height = self._get_terminal_size()
        
        # Detect color support
        colors = self._detect_color_support()
        
        # Check various capabilities
        unicode_support = self._check_unicode_support()
        mouse_support = self._check_mouse_support()
        alternate_screen = self._check_alternate_screen_support()
        cursor_control = self._check_cursor_control_support()
        
        # Determine if interactive
        interactive = terminal_type in (TerminalType.FULL_TTY, TerminalType.PARTIAL_TTY)
        
        # Get terminal program info
        term_program = os.getenv("TERM_PROGRAM")
        term_version = os.getenv("TERM_PROGRAM_VERSION")
        
        self._capabilities = TerminalCapabilities(
            type=terminal_type,
            width=width,
            height=height,
            colors=colors,
            unicode_support=unicode_support,
            mouse_support=mouse_support,
            alternate_screen=alternate_screen,
            cursor_control=cursor_control,
            interactive=interactive,
            term_program=term_program,
            term_version=term_version
        )
        self._cache_valid = True
        
        return self._capabilities
    
    def _detect_terminal_type(self) -> TerminalType:
        """Detect the type of terminal environment."""
        # Check if stdin, stdout, stderr are TTYs
        stdin_tty = sys.stdin.isatty()
        stdout_tty = sys.stdout.isatty()
        stderr_tty = sys.stderr.isatty()
        
        # Check for /dev/tty availability (Unix only)
        dev_tty_available = False
        if hasattr(os, 'name') and os.name != 'nt':  # Not Windows
            try:
                with open('/dev/tty', 'r') as tty_file:
                    dev_tty_available = True
            except (OSError, IOError):
                dev_tty_available = False
        
        # Check environment variables
        term = os.getenv('TERM', '').lower()
        term_program = os.getenv('TERM_PROGRAM', '').lower()
        
        # Full TTY: all streams are TTY or /dev/tty is available with good TERM
        if (stdin_tty and stdout_tty) or (dev_tty_available and term and term != 'dumb'):
            return TerminalType.FULL_TTY
        
        # Partial TTY: some streams are TTY
        if stdin_tty or stdout_tty or stderr_tty:
            return TerminalType.PARTIAL_TTY
        
        # Check if we're in a pipe
        if not stdin_tty and not stdout_tty:
            # Check if it looks like a pipe (not redirected to file)
            try:
                # Try to get file stats - pipes and regular files behave differently
                stdin_stat = os.fstat(sys.stdin.fileno())
                stdout_stat = os.fstat(sys.stdout.fileno())
                
                # Check if it's a pipe or FIFO
                import stat
                if (stat.S_ISFIFO(stdin_stat.st_mode) or 
                    stat.S_ISFIFO(stdout_stat.st_mode)):
                    return TerminalType.PIPE
                else:
                    return TerminalType.REDIRECTED
            except:
                # If we can't determine, assume redirected
                return TerminalType.REDIRECTED
        
        return TerminalType.UNKNOWN
    
    def _get_terminal_size(self) -> Tuple[int, int]:
        """Get terminal dimensions."""
        # Try shutil.get_terminal_size first (most reliable)
        try:
            size = shutil.get_terminal_size()
            return size.columns, size.lines
        except:
            pass
        
        # Try environment variables
        try:
            width = int(os.getenv('COLUMNS', '0'))
            height = int(os.getenv('LINES', '0'))
            if width > 0 and height > 0:
                return width, height
        except:
            pass
        
        # Try tput (Unix only)
        if hasattr(os, 'name') and os.name != 'nt':
            try:
                width = int(subprocess.check_output(['tput', 'cols']).decode().strip())
                height = int(subprocess.check_output(['tput', 'lines']).decode().strip())
                return width, height
            except:
                pass
        
        # Default fallback
        return 80, 24
    
    def _detect_color_support(self) -> int:
        """Detect number of colors supported."""
        term = os.getenv('TERM', '').lower()
        colorterm = os.getenv('COLORTERM', '').lower()
        
        # True color support
        if colorterm in ('truecolor', '24bit'):
            return 16777216  # 24-bit color
        
        # Check TERM variable for color indicators
        if any(indicator in term for indicator in ['256color', '256']):
            return 256
        elif any(indicator in term for indicator in ['color', 'ansi']):
            return 16
        elif term and term != 'dumb':
            return 8
        else:
            return 0  # No color support
    
    def _check_unicode_support(self) -> bool:
        """Check if terminal supports Unicode."""
        # Check encoding
        encoding = getattr(sys.stdout, 'encoding', '') or ''
        if 'utf' in encoding.lower():
            return True
        
        # Check environment
        lang = os.getenv('LANG', '').lower()
        lc_all = os.getenv('LC_ALL', '').lower()
        lc_ctype = os.getenv('LC_CTYPE', '').lower()
        
        return any('utf' in var for var in [lang, lc_all, lc_ctype])
    
    def _check_mouse_support(self) -> bool:
        """Check if terminal supports mouse events."""
        term = os.getenv('TERM', '').lower()
        term_program = os.getenv('TERM_PROGRAM', '').lower()
        
        # Known terminals with mouse support
        mouse_terminals = [
            'xterm', 'screen', 'tmux', 'iterm', 'konsole', 
            'gnome-terminal', 'alacritty', 'kitty'
        ]
        
        return any(terminal in term for terminal in mouse_terminals) or \
               any(terminal in term_program for terminal in mouse_terminals)
    
    def _check_alternate_screen_support(self) -> bool:
        """Check if terminal supports alternate screen."""
        term = os.getenv('TERM', '').lower()
        
        # Most modern terminals support alternate screen
        unsupported = ['dumb', 'cons25']
        return term and not any(unsupp in term for unsupp in unsupported)
    
    def _check_cursor_control_support(self) -> bool:
        """Check if terminal supports cursor control."""
        term = os.getenv('TERM', '').lower()
        
        # Most terminals with any capability support cursor control
        return term and term != 'dumb'
    
    def is_interactive(self) -> bool:
        """Check if we're in an interactive terminal."""
        caps = self.detect_capabilities()
        return caps.interactive
    
    def supports_colors(self) -> bool:
        """Check if terminal supports colors."""
        caps = self.detect_capabilities()
        return caps.colors > 0
    
    def supports_unicode(self) -> bool:
        """Check if terminal supports Unicode."""
        caps = self.detect_capabilities()
        return caps.unicode_support
    
    def get_size(self) -> Tuple[int, int]:
        """Get terminal size as (width, height)."""
        caps = self.detect_capabilities()
        return caps.width, caps.height
    
    def get_safe_width(self, margin: int = 2) -> int:
        """Get safe width for content (with margin)."""
        width, _ = self.get_size()
        return max(10, width - margin)
    
    def refresh_capabilities(self):
        """Force refresh of terminal capabilities."""
        self._cache_valid = False
        self.detect_capabilities(force_refresh=True)
    
    def get_terminal_info(self) -> Dict[str, Any]:
        """Get comprehensive terminal information."""
        caps = self.detect_capabilities()
        
        return {
            "type": caps.type.value,
            "interactive": caps.interactive,
            "dimensions": {
                "width": caps.width,
                "height": caps.height
            },
            "capabilities": {
                "colors": caps.colors,
                "unicode": caps.unicode_support,
                "mouse": caps.mouse_support,
                "alternate_screen": caps.alternate_screen,
                "cursor_control": caps.cursor_control
            },
            "environment": {
                "term": os.getenv('TERM'),
                "term_program": caps.term_program,
                "term_version": caps.term_version,
                "colorterm": os.getenv('COLORTERM'),
                "lang": os.getenv('LANG'),
                "platform": sys.platform
            },
            "streams": {
                "stdin_tty": sys.stdin.isatty(),
                "stdout_tty": sys.stdout.isatty(),
                "stderr_tty": sys.stderr.isatty()
            }
        }
    
    def print_terminal_info(self):
        """Print comprehensive terminal information."""
        info = self.get_terminal_info()
        
        print("Terminal Information:")
        print(f"  Type: {info['type']}")
        print(f"  Interactive: {info['interactive']}")
        print(f"  Size: {info['dimensions']['width']}x{info['dimensions']['height']}")
        print(f"  Colors: {info['capabilities']['colors']}")
        print(f"  Unicode: {info['capabilities']['unicode']}")
        print(f"  Mouse: {info['capabilities']['mouse']}")
        
        print("\nEnvironment:")
        for key, value in info['environment'].items():
            print(f"  {key.upper()}: {value}")
        
        print("\nStreams:")
        for key, value in info['streams'].items():
            print(f"  {key}: {value}")
    
    async def cleanup(self):
        """Cleanup terminal manager resources.
        
        Resets terminal state and clears cached capabilities.
        """
        try:
            self._capabilities = None
            self._cache_valid = False
            self._initialized = False
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error during terminal manager cleanup: {e}")


# Convenience function
def create_terminal_manager() -> TerminalManager:
    """Create and return a new TerminalManager instance."""
    return TerminalManager()


# Global instance for convenience
_global_terminal_manager: Optional[TerminalManager] = None


def get_terminal_manager() -> TerminalManager:
    """Get or create the global terminal manager instance."""
    global _global_terminal_manager
    if _global_terminal_manager is None:
        _global_terminal_manager = TerminalManager()
    return _global_terminal_manager