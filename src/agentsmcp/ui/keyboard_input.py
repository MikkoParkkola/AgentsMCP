"""
Cross-platform keyboard input handler for terminal applications.

Provides unified keyboard input handling across Windows, macOS, and Linux
with support for arrow keys, escape sequences, and special keys.
"""

import sys
import os
from typing import Optional, Tuple
from enum import Enum

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
        Only returns True if we can actually access terminal attributes for raw mode.
        """
        # For Unix systems, only return True if we can actually access terminal attributes
        if self.is_unix:
            try:
                import termios
                import tty
                
                # Must be a TTY
                if not sys.stdin.isatty():
                    return False
                
                fd = sys.stdin.fileno()
                # Try to get terminal attributes - if this works, we can do raw input
                old_settings = termios.tcgetattr(fd)
                return True
                
            except (ImportError, OSError, termios.error):
                # termios not available or terminal attributes inaccessible
                return False
        
        # For Windows, check for console availability
        if self.is_windows:
            try:
                import msvcrt
                # Must be a TTY and have proper console access
                return sys.stdin.isatty() and sys.stdout.isatty()
            except ImportError:
                return False
        
        return False
    
    def get_key(self, timeout: Optional[float] = None) -> Tuple[Optional[KeyCode], Optional[str]]:
        """
        Get a single keypress from the terminal.
        
        Args:
            timeout: Optional timeout in seconds. None for blocking wait.
            
        Returns:
            Tuple of (KeyCode, character). One will be None.
            - For special keys: (KeyCode.*, None)
            - For regular characters: (None, character)
            - For timeout/no input: (None, None)
        """
        # Fallback for non-interactive environments
        if not self.is_interactive:
            return self._get_key_fallback(timeout)
            
        if self.is_windows:
            return self._get_key_windows(timeout)
        else:
            return self._get_key_unix(timeout)
    
    def _get_key_windows(self, timeout: Optional[float] = None) -> Tuple[Optional[KeyCode], Optional[str]]:
        """Windows implementation of get_key"""
        if timeout is not None:
            # Windows doesn't have easy timeout support for getch
            # For now, just do non-blocking check
            if not self.msvcrt.kbhit():
                return None, None
        
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
            return key_map.get(ch2), None
        
        # Handle regular special characters
        if ch == b'\r' or ch == b'\n':
            return KeyCode.ENTER, None
        elif ch == b'\x1b':  # ESC
            return KeyCode.ESCAPE, None
        elif ch == b'\x08':  # Backspace
            return KeyCode.BACKSPACE, None
        elif ch == b'\t':    # Tab
            return KeyCode.TAB, None
        elif ch == b' ':     # Space
            return KeyCode.SPACE, None
        else:
            # Regular character
            try:
                return None, ch.decode('utf-8', errors='ignore')
            except:
                return None, None
    
    def _get_key_fallback(self, timeout: Optional[float] = None) -> Tuple[Optional[KeyCode], Optional[str]]:
        """Fallback implementation for non-interactive environments"""
        # In non-interactive mode, simulate arrow key behavior with regular input
        try:
            line = input().strip()
            if line.lower() in ['q', 'quit', 'exit']:
                return KeyCode.ESCAPE, None
            elif line.lower() in ['', 'enter']:
                return KeyCode.ENTER, None
            elif line.lower() in ['up', 'u']:
                return KeyCode.UP, None
            elif line.lower() in ['down', 'd']:
                return KeyCode.DOWN, None
            else:
                return None, line
        except (EOFError, KeyboardInterrupt):
            return KeyCode.ESCAPE, None
    
    def _get_key_lenient_fallback(self, timeout: Optional[float] = None) -> Tuple[Optional[KeyCode], Optional[str]]:
        """
        Lenient fallback for environments where we believe we're in a terminal 
        but can't access raw terminal attributes (e.g., containers, IDEs, certain shells).
        
        This provides better UX by giving clear instructions and handling common cases.
        """
        # Try to read a single character using standard input
        # This won't work for arrow keys, but provides a better UX
        try:
            print("Navigation: [u]p, [d]own, [enter] to select, [q] to quit: ", end='', flush=True)
            line = input().strip().lower()
            
            if line in ['q', 'quit', 'exit']:
                return KeyCode.ESCAPE, None
            elif line in ['', 'enter']:
                return KeyCode.ENTER, None
            elif line in ['up', 'u']:
                return KeyCode.UP, None
            elif line in ['down', 'd']:
                return KeyCode.DOWN, None
            elif line in ['left', 'l']:
                return KeyCode.LEFT, None
            elif line in ['right', 'r']:
                return KeyCode.RIGHT, None
            else:
                # Treat other input as regular character input
                return None, line
                
        except (EOFError, KeyboardInterrupt):
            return KeyCode.ESCAPE, None
    
    def _get_key_unix(self, timeout: Optional[float] = None) -> Tuple[Optional[KeyCode], Optional[str]]:
        """Unix implementation of get_key"""
        # Check if we can access terminal attributes
        fd = sys.stdin.fileno()
        old_settings = None
        
        try:
            old_settings = self.termios.tcgetattr(fd)
        except (self.termios.error, OSError) as e:
            # If we can't access terminal attributes, we're not actually interactive
            # This should not happen if _detect_interactive_capability worked correctly
            raise RuntimeError(
                f"Cannot access terminal attributes despite detecting interactive capability: {e}\n"
                "Please run this program in a proper terminal emulator."
            )
        
        try:
            # Set terminal to raw mode
            self.tty.setraw(sys.stdin.fileno())
            
            # FIXED: Reduce timeout precision to prevent first character drops
            # Check for input availability with timeout
            if timeout is not None:
                ready, _, _ = self.select.select([sys.stdin], [], [], timeout)
                if not ready:
                    return None, None
            
            # Read first character - this should be immediate
            ch = sys.stdin.read(1)
            
            # FIXED: Handle slash character immediately without delay
            if ch == '/':
                return None, ch
            
            # Handle escape sequences
            if ch == '\x1b':  # ESC
                # FIXED: Shorter timeout to prevent input lag
                ready, _, _ = self.select.select([sys.stdin], [], [], 0.05)  # Reduced from 0.1
                if ready:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        # ANSI escape sequence
                        ch3 = sys.stdin.read(1)
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
                            return special_key, None
                        
                        # Multi-character sequences (Page Up/Down, Delete, etc.)
                        if ch3 in '0123456789':
                            # Read until we find the final character
                            sequence = ch3
                            while True:
                                # FIXED: Shorter timeout for sequence reading
                                ready, _, _ = self.select.select([sys.stdin], [], [], 0.05)
                                if not ready:
                                    break
                                next_ch = sys.stdin.read(1)
                                sequence += next_ch
                                if next_ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ~':
                                    break
                            
                            # Map common sequences
                            if sequence == '5~':
                                return KeyCode.PAGE_UP, None
                            elif sequence == '6~':
                                return KeyCode.PAGE_DOWN, None
                            elif sequence == '3~':
                                return KeyCode.DELETE, None
                
                # Just ESC key
                return KeyCode.ESCAPE, None
            
            # Handle other special characters
            elif ch == '\n' or ch == '\r':
                return KeyCode.ENTER, None
            elif ch == '\x7f' or ch == '\x08':  # DEL or Backspace
                return KeyCode.BACKSPACE, None
            elif ch == '\t':
                return KeyCode.TAB, None
            elif ch == ' ':
                return KeyCode.SPACE, None
            else:
                # FIXED: Regular character - return immediately
                return None, ch
                
        finally:
            # Restore original terminal settings
            self.termios.tcsetattr(fd, self.termios.TCSADRAIN, old_settings)
    
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
        key_code, char = self.keyboard.get_key()
        
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
            key_code, char = self.keyboard.get_key()
            
            if key_code == KeyCode.ESCAPE and allow_escape:
                return False, ""
            
            if key_code == KeyCode.ENTER:
                return True, "enter"
            
            if char:
                if char.lower() == 'q' and allow_escape:
                    return False, ""
                    
                if allowed_keys is None or char.lower() in [k.lower() for k in allowed_keys]:
                    return True, char