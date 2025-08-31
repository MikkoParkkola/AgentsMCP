'''\
Cross-platform keyboard input handler for terminal applications.

Provides unified keyboard input handling across Windows, macOS, and Linux
with support for arrow keys, escape sequences, and special keys.
'''\

import sys
import os
from typing import Optional, Tuple, List
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

# ---------------------------------------------------------------------------
# New enum for mouse events – added to support scrolling and clicks
# ---------------------------------------------------------------------------
class MouseEvent(Enum):
    """Simple mouse event abstraction used by the TUI"""
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    CLICK_LEFT = "click_left"
    CLICK_RIGHT = "click_right"
    MOVE = "move"

class KeyboardInput:
    """Cross-platform keyboard input handler"""
    
    def __init__(self):
        self.is_windows = sys.platform.startswith('win')
        self.is_unix = not self.is_windows
        self.is_interactive = self._detect_interactive_capability()
        self.hybrid_mode = False  # Enable enhanced line-based mode
        self._fd = None                # Cached file descriptor for /dev/tty or stdin
        self._orig_settings = None     # Original termios settings (Unix only)
        self._buffer: List[str] = []   # Simple line buffer for backspace handling
        
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
            # ---------------------------------------------------------
            # Cache terminal descriptor once during construction – this avoids
            # opening /dev/tty on every key press and reduces system‑call
            # overhead.
            # ---------------------------------------------------------
            self._initialize_unix_fd()
            # Enable mouse reporting (SGR mode) for scrolling support
            if self._fd is not None:
                try:
                    os.write(self._fd, b'\x1b[?1000h\x1b[?1006h')  # Enable mouse tracking and SGR encoding
                except Exception:
                    pass
    
    # -------------------------------------------------------------------
    # Unix helper: open /dev/tty (or fall back to stdin) and store the fd
    # -------------------------------------------------------------------
    def _initialize_unix_fd(self):
        """Open the appropriate input stream and cache its file descriptor.
        The descriptor is kept open for the lifetime of the KeyboardInput
        instance and will be closed when ``close()`` is called or the object is
        garbage‑collected.
        """
        try:
            if os.path.exists('/dev/tty'):
                # Open in binary mode to get raw bytes without any decoding
                self._fd_file = open('/dev/tty', 'rb', buffering=0)
                self._fd = self._fd_file.fileno()
            else:
                raise FileNotFoundError
        except Exception:
            # Fallback to stdin – this works in many CI environments
            try:
                self._fd = sys.stdin.fileno()
                self._fd_file = sys.stdin.buffer
            except Exception:
                self._fd = None
                self._fd_file = None
                return
        # Store original terminal attributes so we can restore them later
        try:
            self._orig_settings = self.termios.tcgetattr(self._fd)
        except Exception:
            self._orig_settings = None
    
    def close(self):
        """Restore terminal settings and close any opened file descriptors.
        Users should call this when the TUI is shutting down to avoid leaving the
        terminal in raw mode.
        """
        if self.is_unix and self._fd is not None and self._orig_settings is not None:
            try:
                self.termios.tcsetattr(self._fd, self.termios.TCSADRAIN, self._orig_settings)
            except Exception:
                pass
        # Disable mouse reporting if it was enabled
        if self.is_unix and self._fd is not None:
            try:
                os.write(self._fd, b'\x1b[?1000l\x1b[?1006l')
            except Exception:
                pass
        if hasattr(self, "_fd_file") and self._fd_file not in (sys.stdin, sys.stdout, sys.stderr):
            try:
                self._fd_file.close()
            except Exception:
                pass
        self._fd = None
        self._orig_settings = None
    
    def __del__(self):
        # Ensure resources are cleaned up even if the caller forgets ``close``
        self.close()
    
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
    
    def get_key(self, timeout: Optional[float] = None) -> Tuple[Optional[KeyCode], Optional[str], InputMode, Optional[MouseEvent]]:
        """
        Get a single keypress from the terminal.
        
        Args:
            timeout: Optional timeout in seconds. None for blocking wait.
            
        Returns:
            Tuple of (KeyCode, character, mode, mouse_event). One of KeyCode/character will be None.
            - For special keys: (KeyCode.*, None, PER_KEY, None)
            - For regular characters: (None, character, PER_KEY, None)
            - For mouse events: (None, None, PER_KEY, MouseEvent.*)
            - For timeout/no input: (None, None, TIMEOUT, None)
            - For fallback to line-based: (None, None, LINE_BASED, None)
        """
        # Fallback for non-interactive environments
        if not self.is_interactive:
            return None, None, InputMode.LINE_BASED, None
        
        if self.is_windows:
            return self._get_key_windows(timeout)
        else:
            return self._get_key_unix(timeout)
    
    def _get_key_windows(self, timeout: Optional[float] = None) -> Tuple[Optional[KeyCode], Optional[str], InputMode, Optional[MouseEvent]]:
        """Windows implementation of get_key"""
        if timeout is not None:
            # Windows doesn't have easy timeout support for getch
            # For now, just do non-blocking check
            if not self.msvcrt.kbhit():
                return None, None, InputMode.TIMEOUT, None
        
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
            return key_map.get(ch2), None, InputMode.PER_KEY, None
        
        # Handle regular special characters
        if ch == b'\r' or ch == b'\n':
            return KeyCode.ENTER, None, InputMode.PER_KEY, None
        elif ch == b'\x1b':  # ESC
            return KeyCode.ESCAPE, None, InputMode.PER_KEY, None
        elif ch == b'\x08':  # Backspace
            # Update internal buffer
            if self._buffer:
                self._buffer.pop()
            return KeyCode.BACKSPACE, None, InputMode.PER_KEY, None
        elif ch == b'\t':    # Tab
            return KeyCode.TAB, None, InputMode.PER_KEY, None
        elif ch == b' ':     # Space
            self._buffer.append(' ')
            return KeyCode.SPACE, None, InputMode.PER_KEY, None
        else:
            # Regular character
            try:
                char = ch.decode('utf-8', errors='ignore')
                self._buffer.append(char)
                return None, char, InputMode.PER_KEY, None
            except:
                return None, None, InputMode.TIMEOUT, None
    
    def _get_key_unix(self, timeout: Optional[float] = None) -> Tuple[Optional[KeyCode], Optional[str], InputMode, Optional[MouseEvent]]:
        """Unix implementation of get_key"""
        if self._fd is None:
            # Critical fallback – we cannot read raw keys, so use line based mode
            return None, None, InputMode.LINE_BASED, None
        
        try:
            # Set terminal to raw mode (only once per instance – cached settings are restored on close)
            self.tty.setraw(self._fd)
            
            # Check for input availability with timeout
            if timeout is not None:
                ready, _, _ = self.select.select([self._fd_file], [], [], timeout)
                if not ready:
                    return None, None, InputMode.TIMEOUT, None
            
            # Read first byte
            ch_bytes = self._fd_file.read(1)
            if not ch_bytes:
                return None, None, InputMode.TIMEOUT, None
            ch = ch_bytes.decode('utf-8', errors='ignore')
            
            # Handle escape sequences (reduced timeout for faster response)
            if ch == '\x1b':  # ESC
                # Peek to see if this is a mouse event (SGR format starts with "[<")
                ready, _, _ = self.select.select([self._fd_file], [], [], 0.05)
                if ready:
                    nxt_b = self._fd_file.read(1)
                    if not nxt_b:
                        return KeyCode.ESCAPE, None, InputMode.PER_KEY, None
                    nxt = nxt_b.decode('utf-8', errors='ignore')
                    if nxt == '[':
                        # Could be mouse event or normal CSI sequence
                        rest = self._fd_file.read(1).decode('utf-8', errors='ignore')
                        if rest == '<':
                            # SGR mouse event – read until final M or m
                            seq = ''
                            while True:
                                part = self._fd_file.read(1).decode('utf-8', errors='ignore')
                                if not part:
                                    break
                                seq += part
                                if part in ('M', 'm'):
                                    break
                            # seq now looks like "64;10;20M" etc.
                            try:
                                button_str, _, _ = seq.partition('M')
                                button_code = int(button_str.split(';')[0])
                                if button_code == 64:
                                    return None, None, InputMode.PER_KEY, MouseEvent.SCROLL_UP
                                elif button_code == 65:
                                    return None, None, InputMode.PER_KEY, MouseEvent.SCROLL_DOWN
                            except Exception:
                                pass
                        # Not a mouse event – fall back to normal CSI handling
                        ch2 = rest
                    else:
                        ch2 = nxt
                else:
                    # No further bytes – plain ESC key
                    return KeyCode.ESCAPE, None, InputMode.PER_KEY, None
                # Normal CSI handling for arrow keys etc.
                if ch2 == '[':
                    ch3 = self._fd_file.read(1).decode('utf-8', errors='ignore')
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
                        return special_key, None, InputMode.PER_KEY, None
                    # Multi‑character sequences (Page Up/Down, Delete, etc.)
                    if ch3 in '0123456789':
                        sequence = ch3
                        while True:
                            ready, _, _ = self.select.select([self._fd_file], [], [], 0.05)
                            if not ready:
                                break
                            nxt_b = self._fd_file.read(1)
                            if not nxt_b:
                                break
                            nxt = nxt_b.decode('utf-8', errors='ignore')
                            sequence += nxt
                            if nxt in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ~':
                                break
                        if sequence == '5~':
                            return KeyCode.PAGE_UP, None, InputMode.PER_KEY, None
                        elif sequence == '6~':
                            return KeyCode.PAGE_DOWN, None, InputMode.PER_KEY, None
                        elif sequence == '3~':
                            return KeyCode.DELETE, None, InputMode.PER_KEY, None
                # If we got here the ESC was solitary
                return KeyCode.ESCAPE, None, InputMode.PER_KEY, None
            # Other special characters
            if ch in ('\n', '\r'):
                return KeyCode.ENTER, None, InputMode.PER_KEY, None
            if ch in ('\x7f', '\x08'):
                # Backspace – modify buffer
                if self._buffer:
                    self._buffer.pop()
                return KeyCode.BACKSPACE, None, InputMode.PER_KEY, None
            if ch == '\t':
                return KeyCode.TAB, None, InputMode.PER_KEY, None
            if ch == ' ':
                self._buffer.append(' ')
                return KeyCode.SPACE, None, InputMode.PER_KEY, None
            # Regular printable character
            self._buffer.append(ch)
            return None, ch, InputMode.PER_KEY, None
        finally:
            # Restore original terminal settings (if we cached them)
            if self._orig_settings is not None:
                try:
                    self.termios.tcsetattr(self._fd, self.termios.TCSADRAIN, self._orig_settings)
                except Exception:
                    pass
    
    def get_current_line(self) -> str:
        """Return the current line buffer as a string. Useful for UI rendering."""
        return ''.join(self._buffer)
    
    def flush_input(self):
        """Clear any pending input from the buffer"""
        if not self.is_interactive:
            return  # No need to flush in non-interactive mode
        
        if self.is_windows:
            while self.msvcrt.kbhit():
                self.msvcrt.getch()
        else:
            # Unix: read all pending bytes in non‑blocking mode
            if self._fd is None:
                return
            try:
                old = self.termios.tcgetattr(self._fd)
                self.tty.setraw(self._fd)
                while True:
                    ready, _, _ = self.select.select([self._fd_file], [], [], 0)
                    if not ready:
                        break
                    self._fd_file.read(1)
            except Exception:
                pass
            finally:
                if old is not None:
                    try:
                        self.termios.tcsetattr(self._fd, self.termios.TCSADRAIN, old)
                    except Exception:
                        pass
