"""Terminal capability detection and progressive enhancement logic."""

import os
import sys
import shutil
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class TerminalCapabilities:
    """Detected terminal capabilities for progressive enhancement."""
    
    # Basic terminal properties
    is_tty: bool
    width: int
    height: int
    
    # Feature support
    supports_colors: bool
    supports_unicode: bool
    supports_rich: bool
    
    # Performance hints
    is_fast_terminal: bool
    max_refresh_rate: int
    
    # Fallback indicators
    force_plain: bool
    force_simple: bool
    

def detect_terminal_capabilities() -> TerminalCapabilities:
    """Detect current terminal capabilities for progressive enhancement."""
    
    # Basic TTY detection with override
    is_tty = sys.stdout.isatty() and sys.stdin.isatty()
    if os.environ.get('AGENTSMCP_FORCE_RICH'):
        is_tty = True  # Force TTY for Rich mode
    
    # Terminal size detection with fallbacks
    try:
        size = shutil.get_terminal_size(fallback=(80, 24))
        width, height = size.columns, size.lines
    except Exception:
        width, height = 80, 24
    
    # Color support detection
    supports_colors = _detect_color_support()
    
    # Unicode support detection
    supports_unicode = _detect_unicode_support()
    
    # Rich library compatibility
    supports_rich = _detect_rich_support(is_tty, supports_colors, supports_unicode)
    
    # Performance characteristics
    is_fast_terminal = _detect_terminal_performance()
    max_refresh_rate = 60 if is_fast_terminal else 30
    
    # Force fallbacks based on environment
    force_plain = _should_force_plain()
    force_simple = _should_force_simple()
    
    return TerminalCapabilities(
        is_tty=is_tty,
        width=width,
        height=height,
        supports_colors=supports_colors,
        supports_unicode=supports_unicode,
        supports_rich=supports_rich,
        is_fast_terminal=is_fast_terminal,
        max_refresh_rate=max_refresh_rate,
        force_plain=force_plain,
        force_simple=force_simple
    )


def _detect_color_support() -> bool:
    """Detect if terminal supports colors."""
    # Check TERM environment variable
    term = os.environ.get('TERM', '').lower()
    if 'color' in term or term in ('xterm', 'xterm-256color', 'screen', 'tmux'):
        return True
    
    # Check for explicit color support indicators
    if os.environ.get('COLORTERM') or os.environ.get('FORCE_COLOR'):
        return True
    
    # Conservative fallback
    return sys.stdout.isatty()


def _detect_unicode_support() -> bool:
    """Detect if terminal supports Unicode properly."""
    try:
        # Test with a simple unicode character
        sys.stdout.write('\u2713')  # checkmark
        sys.stdout.flush()
        return True
    except (UnicodeEncodeError, UnicodeError):
        return False


def _detect_rich_support(is_tty: bool, colors: bool, unicode: bool) -> bool:
    """Detect if Rich library should work well in this terminal."""
    # Force Rich mode override - user explicitly wants Rich UI
    if os.environ.get('AGENTSMCP_FORCE_RICH'):
        return True
    
    if not is_tty:
        return False
    
    # Basic requirements for Rich
    if not (colors and unicode):
        return False
    
    # Check for problematic terminals
    term = os.environ.get('TERM', '').lower()
    if term in ('dumb', 'emacs'):
        return False
    
    # Check CI environments that might have issues
    ci_vars = ['CI', 'CONTINUOUS_INTEGRATION', 'BUILD_NUMBER']
    if any(os.environ.get(var) for var in ci_vars):
        return False
    
    return True


def _detect_terminal_performance() -> bool:
    """Detect if terminal can handle high refresh rates."""
    # Modern terminals are generally fast
    term = os.environ.get('TERM', '').lower()
    fast_terminals = ['xterm', 'alacritty', 'kitty', 'iterm', 'terminal']
    
    return any(fast in term for fast in fast_terminals)


def _should_force_plain() -> bool:
    """Check if we should force plain text mode."""
    # Force Rich mode override - user explicitly wants Rich UI
    if os.environ.get('AGENTSMCP_FORCE_RICH'):
        return False
    
    # Explicit environment variable
    if os.environ.get('AGENTSMCP_FORCE_PLAIN'):
        return True
    
    # Non-interactive environments
    if not sys.stdout.isatty():
        return True
    
    # Minimal terminals
    term = os.environ.get('TERM', '').lower()
    if term in ('dumb', 'unknown', ''):
        return True
    
    return False


def _should_force_simple() -> bool:
    """Check if we should force simple TUI mode (no Rich)."""
    # Explicit environment variable
    if os.environ.get('AGENTSMCP_FORCE_SIMPLE'):
        return True
    
    # Terminals that work but have Rich issues
    term = os.environ.get('TERM', '').lower()
    simple_only = ['screen', 'tmux']
    
    return any(simple in term for simple in simple_only)