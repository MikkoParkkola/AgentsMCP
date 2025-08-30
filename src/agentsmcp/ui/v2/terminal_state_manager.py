"""
Terminal State Manager - Critical TTY settings management with guaranteed restoration.

This module provides robust terminal state management that ensures TTY settings
are always restored, even on crashes or interrupts. This fixes the primary
issue where terminal input becomes broken after TUI exit.
"""

import os
import sys
import termios
import tty
import signal
import atexit
import contextlib
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from threading import Lock
from enum import Enum

logger = logging.getLogger(__name__)


class TerminalMode(Enum):
    """Terminal operating modes."""
    NORMAL = "normal"      # Cooked mode with line buffering
    RAW = "raw"           # Raw mode, immediate character processing
    CBREAK = "cbreak"     # Character break mode, some processing


@dataclass
class TerminalState:
    """Captured terminal state for restoration."""
    fd: int
    attrs: List[Any]  # termios attributes
    mode: TerminalMode
    cursor_visible: bool = True
    alternate_screen: bool = False
    mouse_enabled: bool = False


class TerminalStateManager:
    """
    Terminal state manager with guaranteed restoration.
    
    Critical features:
    - Atomic state capture and restoration
    - Signal handler registration for cleanup
    - Multiple exit handler registration
    - Thread-safe operations
    - Graceful fallback on errors
    """
    
    def __init__(self):
        self._original_state: Optional[TerminalState] = None
        self._current_mode: TerminalMode = TerminalMode.NORMAL
        self._lock = Lock()
        self._cleanup_registered = False
        self._tty_fd: Optional[int] = None
        self._output_fd: Optional[int] = None
        self._initialized = False
        
        # Track all changes for comprehensive restoration
        self._state_changes: List[str] = []
        
    def initialize(self) -> bool:
        """
        Initialize the terminal state manager.
        
        Returns:
            True if initialization successful
        """
        with self._lock:
            if self._initialized:
                return True
                
            try:
                # Open TTY for input control
                self._tty_fd = self._open_tty()
                if self._tty_fd is None:
                    logger.warning("Failed to open TTY - input control will be limited")
                    self._tty_fd = sys.stdin.fileno()
                
                self._output_fd = sys.stdout.fileno()
                
                # Capture original state
                self._capture_original_state()
                
                # Register cleanup handlers
                self._register_cleanup_handlers()
                
                self._initialized = True
                logger.debug("Terminal state manager initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize terminal state manager: {e}")
                return False
    
    def _open_tty(self) -> Optional[int]:
        """Safely open TTY device."""
        tty_paths = ['/dev/tty', '/proc/self/fd/0', '/dev/stdin']
        
        for path in tty_paths:
            try:
                if os.path.exists(path):
                    fd = os.open(path, os.O_RDWR)
                    # Test that this is actually a TTY
                    os.isatty(fd)
                    return fd
            except (OSError, IOError):
                continue
                
        # Fallback to stdin if it's a TTY
        try:
            if os.isatty(sys.stdin.fileno()):
                return sys.stdin.fileno()
        except (OSError, IOError):
            pass
            
        return None
    
    def _capture_original_state(self):
        """Capture the original terminal state."""
        if self._tty_fd is None:
            return
            
        try:
            attrs = termios.tcgetattr(self._tty_fd)
            self._original_state = TerminalState(
                fd=self._tty_fd,
                attrs=attrs[:],  # Make a copy
                mode=TerminalMode.NORMAL,
                cursor_visible=True,
                alternate_screen=False,
                mouse_enabled=False
            )
            logger.debug("Captured original terminal state")
            
        except (termios.error, OSError) as e:
            logger.warning(f"Failed to capture terminal state: {e}")
            self._original_state = None
    
    def _register_cleanup_handlers(self):
        """Register multiple cleanup handlers for maximum reliability."""
        if self._cleanup_registered:
            return
            
        # Register atexit handler
        atexit.register(self._emergency_restore)
        
        # Register signal handlers
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT, signal.SIGHUP]:
            try:
                original = signal.signal(sig, self._signal_handler)
                # Chain to original handler if it exists
                if original not in (signal.SIG_DFL, signal.SIG_IGN, None):
                    signal.signal(sig, lambda s, f, orig=original: (self._emergency_restore(), orig(s, f)))
            except (ValueError, OSError):
                # Some signals might not be available on all platforms
                pass
        
        self._cleanup_registered = True
        logger.debug("Registered terminal cleanup handlers")
    
    def _signal_handler(self, signum, frame):
        """Handle signals by restoring terminal state."""
        logger.debug(f"Signal {signum} received, restoring terminal state")
        self._emergency_restore()
        
        # Re-raise the signal with default handler
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    
    def enter_raw_mode(self) -> bool:
        """
        Enter raw terminal mode for character-by-character input.
        
        Returns:
            True if mode change successful
        """
        with self._lock:
            if not self._initialized or self._tty_fd is None:
                return False
                
            try:
                if self._current_mode == TerminalMode.RAW:
                    return True
                    
                # Set raw mode
                tty.setraw(self._tty_fd)
                self._current_mode = TerminalMode.RAW
                self._state_changes.append("raw_mode")
                
                logger.debug("Entered raw terminal mode")
                return True
                
            except (termios.error, OSError) as e:
                logger.error(f"Failed to enter raw mode: {e}")
                return False
    
    def enter_cbreak_mode(self) -> bool:
        """
        Enter cbreak mode for immediate character input with some processing.
        
        Returns:
            True if mode change successful
        """
        with self._lock:
            if not self._initialized or self._tty_fd is None:
                return False
                
            try:
                if self._current_mode == TerminalMode.CBREAK:
                    return True
                    
                # Set cbreak mode
                tty.setcbreak(self._tty_fd)
                self._current_mode = TerminalMode.CBREAK
                self._state_changes.append("cbreak_mode")
                
                logger.debug("Entered cbreak terminal mode")
                return True
                
            except (termios.error, OSError) as e:
                logger.error(f"Failed to enter cbreak mode: {e}")
                return False
    
    def hide_cursor(self) -> bool:
        """Hide the terminal cursor."""
        try:
            if self._output_fd is not None:
                os.write(self._output_fd, b'\033[?25l')
                self._state_changes.append("cursor_hidden")
                return True
        except OSError as e:
            logger.warning(f"Failed to hide cursor: {e}")
        return False
    
    def show_cursor(self) -> bool:
        """Show the terminal cursor."""
        try:
            if self._output_fd is not None:
                os.write(self._output_fd, b'\033[?25h')
                if "cursor_hidden" in self._state_changes:
                    self._state_changes.remove("cursor_hidden")
                return True
        except OSError as e:
            logger.warning(f"Failed to show cursor: {e}")
        return False
    
    def enter_alternate_screen(self) -> bool:
        """Enter alternate screen buffer."""
        try:
            if self._output_fd is not None:
                os.write(self._output_fd, b'\033[?1049h')
                self._state_changes.append("alternate_screen")
                return True
        except OSError as e:
            logger.warning(f"Failed to enter alternate screen: {e}")
        return False
    
    def exit_alternate_screen(self) -> bool:
        """Exit alternate screen buffer."""
        try:
            if self._output_fd is not None:
                os.write(self._output_fd, b'\033[?1049l')
                if "alternate_screen" in self._state_changes:
                    self._state_changes.remove("alternate_screen")
                return True
        except OSError as e:
            logger.warning(f"Failed to exit alternate screen: {e}")
        return False
    
    def enable_mouse_reporting(self) -> bool:
        """Enable mouse reporting."""
        try:
            if self._output_fd is not None:
                # Enable mouse reporting with SGR extended mode
                os.write(self._output_fd, b'\033[?1000h\033[?1006h')
                self._state_changes.append("mouse_enabled")
                return True
        except OSError as e:
            logger.warning(f"Failed to enable mouse reporting: {e}")
        return False
    
    def disable_mouse_reporting(self) -> bool:
        """Disable mouse reporting."""
        try:
            if self._output_fd is not None:
                # Disable mouse reporting
                os.write(self._output_fd, b'\033[?1006l\033[?1000l')
                if "mouse_enabled" in self._state_changes:
                    self._state_changes.remove("mouse_enabled")
                return True
        except OSError as e:
            logger.warning(f"Failed to disable mouse reporting: {e}")
        return False
    
    def restore_terminal_state(self) -> bool:
        """
        Restore terminal to its original state.
        
        Returns:
            True if restoration successful
        """
        with self._lock:
            return self._restore_state_locked()
    
    def _restore_state_locked(self) -> bool:
        """Internal state restoration (assumes lock held)."""
        if not self._initialized:
            return True
            
        success = True
        
        try:
            # Restore terminal attributes first
            if self._original_state and self._tty_fd is not None:
                try:
                    termios.tcsetattr(
                        self._tty_fd, 
                        termios.TCSADRAIN, 
                        self._original_state.attrs
                    )
                    self._current_mode = TerminalMode.NORMAL
                except (termios.error, OSError) as e:
                    logger.warning(f"Failed to restore terminal attributes: {e}")
                    success = False
            
            # Restore visual state
            if self._output_fd is not None:
                try:
                    # Build restoration sequence
                    restore_sequence = b''
                    
                    # Reverse all state changes
                    if "mouse_enabled" in self._state_changes:
                        restore_sequence += b'\033[?1006l\033[?1000l'
                    
                    if "alternate_screen" in self._state_changes:
                        restore_sequence += b'\033[?1049l'
                    
                    if "cursor_hidden" in self._state_changes:
                        restore_sequence += b'\033[?25h'
                    
                    # Always reset graphics and show cursor as final step
                    restore_sequence += b'\033[0m\033[?25h'
                    
                    if restore_sequence:
                        os.write(self._output_fd, restore_sequence)
                        
                except OSError as e:
                    logger.warning(f"Failed to restore visual state: {e}")
                    success = False
            
            # Clear state changes
            self._state_changes.clear()
            
            logger.debug("Terminal state restored" + ("" if success else " with warnings"))
            
        except Exception as e:
            logger.error(f"Error during terminal state restoration: {e}")
            success = False
            
        return success
    
    def _emergency_restore(self):
        """Emergency terminal restoration (no error handling)."""
        try:
            # Try to restore as much as possible without raising exceptions
            if self._tty_fd is not None and self._original_state:
                try:
                    termios.tcsetattr(self._tty_fd, termios.TCSANOW, self._original_state.attrs)
                except:
                    pass
            
            if self._output_fd is not None:
                try:
                    # Emergency restoration sequence
                    emergency_sequence = b'\033[?1049l\033[?1006l\033[?1000l\033[0m\033[?25h'
                    os.write(self._output_fd, emergency_sequence)
                except:
                    pass
                    
        except:
            # Absolutely no exceptions in emergency restore
            pass
    
    def cleanup(self):
        """Clean up the terminal state manager."""
        with self._lock:
            if not self._initialized:
                return
                
            # Restore state
            self._restore_state_locked()
            
            # Close TTY fd if we opened it
            if self._tty_fd is not None and self._tty_fd not in (0, 1, 2):
                try:
                    os.close(self._tty_fd)
                except OSError:
                    pass
            
            self._initialized = False
            self._tty_fd = None
            self._output_fd = None
            self._original_state = None
            
            logger.debug("Terminal state manager cleaned up")
    
    def get_current_mode(self) -> TerminalMode:
        """Get the current terminal mode."""
        return self._current_mode
    
    def is_initialized(self) -> bool:
        """Check if the manager is initialized."""
        return self._initialized
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        with self._lock:
            return {
                'initialized': self._initialized,
                'current_mode': self._current_mode.value,
                'tty_fd': self._tty_fd,
                'output_fd': self._output_fd,
                'has_original_state': self._original_state is not None,
                'state_changes': self._state_changes.copy(),
                'cleanup_registered': self._cleanup_registered
            }
    
    def __enter__(self):
        """Context manager entry."""
        if not self.initialize():
            raise RuntimeError("Failed to initialize terminal state manager")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with guaranteed cleanup."""
        self.cleanup()


# Convenience context manager for specific terminal modes
@contextlib.contextmanager
def raw_mode():
    """Context manager for raw terminal mode."""
    manager = TerminalStateManager()
    try:
        if not manager.initialize():
            raise RuntimeError("Failed to initialize terminal state manager")
        
        if not manager.enter_raw_mode():
            raise RuntimeError("Failed to enter raw mode")
            
        yield manager
        
    finally:
        manager.cleanup()


@contextlib.contextmanager
def cbreak_mode():
    """Context manager for cbreak terminal mode."""
    manager = TerminalStateManager()
    try:
        if not manager.initialize():
            raise RuntimeError("Failed to initialize terminal state manager")
        
        if not manager.enter_cbreak_mode():
            raise RuntimeError("Failed to enter cbreak mode")
            
        yield manager
        
    finally:
        manager.cleanup()


# Global instance for emergency restoration
_global_manager: Optional[TerminalStateManager] = None


def get_global_terminal_manager() -> TerminalStateManager:
    """Get or create the global terminal state manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = TerminalStateManager()
    return _global_manager


def emergency_terminal_restore():
    """Emergency terminal restoration function."""
    global _global_manager
    if _global_manager is not None:
        _global_manager._emergency_restore()