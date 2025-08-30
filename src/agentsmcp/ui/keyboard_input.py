"""
Cross-platform keyboard input handler for terminal applications.

Provides unified keyboard input handling across Windows, macOS, and Linux
with support for arrow keys, escape sequences, and special keys.
"""

import sys
import os
from typing import Optional, Tuple
from enum import Enum

class InputMode(Enum):
    """Input mode indicators"""
    PER_KEY = "per_key"  # Normal per-key input
    LINE_BASED = "line_based"  # Only line-based input possible
    TIMEOUT = "timeout"  # No input within timeout

class KeyCode(Enum):
    """Enumeration of special key codes"""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    ENTER = "enter"
    ESCAPE = "escape"
    BACKSPACE = "backspace"
    DELETE = "delete"
    TAB = "tab"
    SPACE = "space"
    HOME = "home"
    END = "end"
    PAGE_UP = "page_up"
    PAGE_DOWN = "page_down"


class KeyboardInput:
    """Cross-platform keyboard input handler"""
    
    def __init__(self):
        self.is_windows = sys.platform.startswith('win')
        self.is_unix = not self.is_windows
        self.is_interactive = self._detect_interactive_capability()
        self.hybrid_mode = False  # Enable enhanced line-based mode
        
        # Platform-specific modules
        if self.is_windows:
            try:
                import msvcrt
                self.msvcrt = msvcrt
            except ImportError:
                raise RuntimeError("msvcrt not available on this Windows system")
        else:
            try:
                import termios
                import tty
                import select
                self.termios = termios
                self.tty = tty
                self.select = select
            except ImportError:
                raise RuntimeError("termios/tty not available on this Unix system")
    
    def _detect_interactive_capability(self) -> bool:
        """
        Detect if we can handle interactive keyboard input.
        Checks both stdin TTY and /dev/tty availability for robust detection.
        Enhanced to be more permissive and try multiple detection methods.
        """
        # For Unix systems, check both stdin and /dev/tty availability
        if self.is_unix:
            try:
                import termios
                import tty
                
                # ENHANCED: Try multiple detection methods
                stdin_is_tty = sys.stdin.isatty()
                stdout_is_tty = sys.stdout.isatty()
                stderr_is_tty = sys.stderr.isatty()
                dev_tty_available = False
                
                # Check if /dev/tty is available as fallback (works in many environments)
                try:
                    if os.path.exists('/dev/tty'):
                        with open('/dev/tty', 'r') as tty_file:
                            # Try to get terminal attributes to confirm it works
                            termios.tcgetattr(tty_file.fileno())
                            dev_tty_available = True
                except (OSError, termios.error):
                    dev_tty_available = False
                
                # ENHANCED: Check for terminal environment variables as additional hints
                terminal_env_hints = bool(
                    os.getenv('TERM') and 
                    os.getenv('TERM') != 'dumb' and
                    (os.getenv('TERM_PROGRAM') or os.getenv('TERMINAL_EMULATOR'))
                )
                
                # Be more permissive: consider interactive if ANY of these are true:
                # 1. stdin is TTY, 2. stdout is TTY, 3. /dev/tty available + good env, 4. strong terminal env hints
                # FALLBACK: If /dev/tty is available and we're in a terminal environment, assume interactive
                return (stdin_is_tty or stdout_is_tty or 
                       (dev_tty_available and terminal_env_hints) or 
                       (terminal_env_hints and stderr_is_tty) or
                       # Fallback for environments like Claude Code where streams aren't TTY but /dev/tty works
                       (dev_tty_available and os.getenv('TERM') and os.getenv('TERM') != 'dumb'))
                
            except ImportError:
                # termios/tty not available at all
                return False
        
        # For Windows, check for console availability
        if self.is_windows:
            try:
                import msvcrt
                # Enhanced: accept if any of stdin/stdout is TTY
                return sys.stdin.isatty() or sys.stdout.isatty()
            except ImportError:
                return False
        
        return False
    
    def get_key(self, timeout: Optional[float] = None) -> Tuple[Optional[KeyCode], Optional[str], InputMode]:
        """
        Get a single keypress from the terminal.
        
        Args:
            timeout: Optional timeout in seconds. None for blocking wait.
            
        Returns:
            Tuple of (KeyCode, character, mode). One of KeyCode/character will be None.
            - For special keys: (KeyCode.*, None, PER_KEY)
            - For regular characters: (None, character, PER_KEY)
            - For timeout/no input: (None, None, TIMEOUT)
            - For fallback to line-based: (None, None, LINE_BASED)
        """
        # Fallback for non-interactive environments
        if not self.is_interactive:
            return None, None, InputMode.LINE_BASED
            
        if self.is_windows:
            return self._get_key_windows(timeout)
        else:
            return self._get_key_unix(timeout)
    
    def _get_key_windows(self, timeout: Optional[float] = None) -> Tuple[Optional[KeyCode], Optional[str], InputMode]:
        """Windows implementation of get_key"""
        if timeout is not None:
            # Windows doesn't have easy timeout support for getch
            # For now, just do non-blocking check
            if not self.msvcrt.kbhit():
                return None, None, InputMode.TIMEOUT
        
        # Get first character
        ch = self.msvcrt.getch()
        
        # Handle special keys (they come as two characters)
        if ch in (b'\x00', b'\xe0'):  # Extended key prefix
            ch2 = self.msvcrt.getch()
            key_map = {
                b'H': KeyCode.UP,
                b'P': KeyCode.DOWN,
                b'K': KeyCode.LEFT,
                b'M': KeyCode.RIGHT,
                b'G': KeyCode.HOME,
                b'O': KeyCode.END,
                b'I': KeyCode.PAGE_UP,
                b'Q': KeyCode.PAGE_DOWN,
                b'S': KeyCode.DELETE,
            }
            return key_map.get(ch2), None, InputMode.PER_KEY
        
        # Handle regular special characters
        if ch == b'\r' or ch == b'\n':
            return KeyCode.ENTER, None, InputMode.PER_KEY
        elif ch == b'\x1b':  # ESC
            return KeyCode.ESCAPE, None, InputMode.PER_KEY
        elif ch == b'\x08':  # Backspace
            return KeyCode.BACKSPACE, None, InputMode.PER_KEY
        elif ch == b'\t':    # Tab
            return KeyCode.TAB, None, InputMode.PER_KEY
        elif ch == b' ':     # Space
            return KeyCode.SPACE, None, InputMode.PER_KEY
        else:
            # Regular character
            try:
                return None, ch.decode('utf-8', errors='ignore'), InputMode.PER_KEY
            except:
                return None, None, InputMode.TIMEOUT
    
    # REMOVED: _get_key_fallback method that used blocking input()
    # This was causing the "looks interactive but acts line-based" issue
    
    def _get_key_unix(self, timeout: Optional[float] = None) -> Tuple[Optional[KeyCode], Optional[str], InputMode]:
        """Unix implementation of get_key"""
        # CRITICAL FIX: Try /dev/tty first if stdin fails, then fall back to stdin
        input_file = None
        fd = None
        old_settings = None
        
        # Try /dev/tty first (works in subprocess environments)
        try:
            if os.path.exists('/dev/tty'):
                # Open TTY in binary for robust, exact key decoding (macOS/iTerm2 friendly)
                input_file = open('/dev/tty', 'rb', buffering=0)
                fd = input_file.fileno()
                old_settings = self.termios.tcgetattr(fd)
        except (self.termios.error, OSError, FileNotFoundError):
            pass
        
        # Fall back to stdin if /dev/tty failed
        if fd is None:
            try:
                fd = sys.stdin.fileno()
                old_settings = self.termios.tcgetattr(fd)
                # Use buffered binary reader on stdin as well
                input_file = sys.stdin.buffer
            except (self.termios.error, OSError) as e:
                # CRITICAL FIX: If we can't access terminal attributes, signal line-based mode
                return None, None, InputMode.LINE_BASED
        
        try:
            # Set terminal to raw mode
            self.tty.setraw(fd)
            
            # FIXED: Reduce timeout precision to prevent first character drops
            # Check for input availability with timeout
            if timeout is not None:
                ready, _, _ = self.select.select([input_file], [], [], timeout)
                if not ready:
                    return None, None, InputMode.TIMEOUT
            
            # Read first character - this should be immediate
            ch_bytes = input_file.read(1)
            if not ch_bytes:
                return None, None, InputMode.TIMEOUT
            # Decode single byte to text safely for processing
            ch = ch_bytes.decode('utf-8', errors='ignore')
            
            # FIXED: Handle slash character immediately without delay
            if ch == '/':
                return None, ch, InputMode.PER_KEY
            
            # Handle escape sequences
            if ch == '\x1b':  # ESC
                # FIXED: Shorter timeout to prevent input lag
                ready, _, _ = self.select.select([input_file], [], [], 0.05)  # Reduced from 0.1
                if ready:
                    ch2_b = input_file.read(1)
                    if not ch2_b:
                        return KeyCode.ESCAPE, None, InputMode.PER_KEY
                    ch2 = ch2_b.decode('utf-8', errors='ignore')
                    if ch2 == '[':
                        # ANSI escape sequence
                        ch3 = input_file.read(1).decode('utf-8', errors='ignore')
                        key_map = {
                            'A': KeyCode.UP,
                            'B': KeyCode.DOWN,
                            'C': KeyCode.RIGHT,
                            'D': KeyCode.LEFT,
                            'H': KeyCode.HOME,
                            'F': KeyCode.END,
                        }
                        special_key = key_map.get(ch3)
                        if special_key:
                            return special_key, None, InputMode.PER_KEY
                        
                        # Multi-character sequences (Page Up/Down, Delete, etc.)
                        if ch3 in '0123456789':
                            # Read until we find the final character
                            sequence = ch3
                            while True:
                                # FIXED: Shorter timeout for sequence reading
                                ready, _, _ = self.select.select([input_file], [], [], 0.05)
                                if not ready:
                                    break
                                next_ch_b = input_file.read(1)
                                if not next_ch_b:
                                    break
                                next_ch = next_ch_b.decode('utf-8', errors='ignore')
                                sequence += next_ch
                                if next_ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ~':
                                    break
                            
                            # Map common sequences
                            if sequence == '5~':
                                return KeyCode.PAGE_UP, None, InputMode.PER_KEY
                            elif sequence == '6~':
                                return KeyCode.PAGE_DOWN, None, InputMode.PER_KEY
                            elif sequence == '3~':
                                return KeyCode.DELETE, None, InputMode.PER_KEY
                
                # Just ESC key
                return KeyCode.ESCAPE, None, InputMode.PER_KEY
            
            # Handle other special characters
            elif ch == '\n' or ch == '\r':
                return KeyCode.ENTER, None, InputMode.PER_KEY
            elif ch == '\x7f' or ch == '\x08':  # DEL or Backspace
                return KeyCode.BACKSPACE, None, InputMode.PER_KEY
            elif ch == '\t':
                return KeyCode.TAB, None, InputMode.PER_KEY
            elif ch == ' ':
                return KeyCode.SPACE, None, InputMode.PER_KEY
            else:
                # FIXED: Regular character - return immediately
                return None, ch, InputMode.PER_KEY
                
        finally:
            # Restore original terminal settings
            self.termios.tcsetattr(fd, self.termios.TCSADRAIN, old_settings)
            # Close /dev/tty file if we opened it
            if input_file and input_file != sys.stdin:
                try:
                    input_file.close()
                except:
                    pass
    
    def flush_input(self):
        """Clear any pending input from the buffer"""
        if not self.is_interactive:
            return  # No need to flush in non-interactive mode
            
        if self.is_windows:
            while self.msvcrt.kbhit():
                self.msvcrt.getch()
        else:
            # Unix: set non-blocking and read all available
            fd = sys.stdin.fileno()
            try:
                old_settings = self.termios.tcgetattr(fd)
                self.tty.setraw(fd)
                while True:
                    ready, _, _ = self.select.select([sys.stdin], [], [], 0)
                    if not ready:
                        break
                    sys.stdin.read(1)
            except self.termios.error:
                pass  # Terminal not available
            finally:
                try:
                    self.termios.tcsetattr(fd, self.termios.TCSADRAIN, old_settings)
                except self.termios.error:
                    pass


class MenuSelector:
    """
    Generic menu selection component with arrow key navigation.
    
    Provides a reusable interface for selecting items from a list
    with visual highlighting and keyboard navigation.
    """
    
    def __init__(self, keyboard_input: Optional[KeyboardInput] = None):
        self.keyboard = keyboard_input or KeyboardInput()
        
    def select_from_options(self, options: list, current_index: int = 0, 
                          allow_escape: bool = True) -> Tuple[Optional[bool], int]:
        """
        Display a menu and let user select with arrow keys.
        
        Args:
            options: List of options to display
            current_index: Initial selected index
            allow_escape: Whether to allow ESC key to cancel
            
        Returns:
            Tuple of (success, selected_index)
            - (True, index) if user selected an option
            - (False, -1) if user cancelled/escaped
            - (None, index) if navigation occurred (caller should redraw)
        """
        if not options:
            return False, -1
        
        selected_index = max(0, min(current_index, len(options) - 1))
        
        # Clear any pending input
        self.keyboard.flush_input()
        
        # Get next keypress
        key_code, char, input_mode = self.keyboard.get_key()
        
        if key_code == KeyCode.UP:
            selected_index = (selected_index - 1) % len(options)
            return None, selected_index  # Navigation occurred
            
        elif key_code == KeyCode.DOWN:
            selected_index = (selected_index + 1) % len(options)
            return None, selected_index  # Navigation occurred
            
        elif key_code == KeyCode.ENTER:
            return True, selected_index
            
        elif key_code == KeyCode.ESCAPE and allow_escape:
            return False, -1
            
        elif char and char.lower() == 'q' and allow_escape:
            return False, -1
            
        # No action taken
        return None, selected_index
    
    def get_single_key(self, allowed_keys: list = None, allow_escape: bool = True) -> Tuple[bool, str]:
        """
        Wait for a single key press from allowed keys.
        
        Args:
            allowed_keys: List of allowed character keys. None for any key.
            allow_escape: Whether to allow ESC key to cancel
            
        Returns:
            Tuple of (success, key)
            - (True, key) if valid key pressed
            - (False, "") if cancelled/escaped
        """
        while True:
            key_code, char, input_mode = self.keyboard.get_key()
            
            if key_code == KeyCode.ESCAPE and allow_escape:
                return False, ""
            
            if key_code == KeyCode.ENTER:
                return True, "enter"
            
            if char:
                if char.lower() == 'q' and allow_escape:
                    return False, ""
                    
                if allowed_keys is None or char.lower() in [k.lower() for k in allowed_keys]:
                    return True, char
