"""
Terminal Controller - Centralized terminal state management and control.

Provides centralized terminal state management to prevent pollution and conflicts
between different UI components. Manages alternate screen buffer, cursor visibility,
and terminal size detection with robust cleanup capabilities.

ICD Compliance:
- Inputs: terminal_size, alternate_screen_mode, cursor_visibility
- Outputs: terminal_state, size_changed_event, cleanup_result  
- Performance: Terminal operations within 100ms
- Key Functions: Size detection, alternate screen buffer management, cleanup
"""

import asyncio
import os
import sys
import shutil
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any, Callable, Set
from datetime import datetime
import weakref


class AlternateScreenMode(Enum):
    """Alternate screen buffer modes."""
    DISABLED = "disabled"
    ENABLED = "enabled"
    AUTO = "auto"  # Auto-detect based on terminal capabilities


class CursorVisibility(Enum):
    """Cursor visibility states."""
    VISIBLE = "visible"
    HIDDEN = "hidden"
    AUTO = "auto"  # Restore original state


@dataclass
class TerminalSize:
    """Terminal dimensions."""
    width: int
    height: int
    timestamp: datetime
    
    def __post_init__(self):
        """Validate terminal size values."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid terminal size: {self.width}x{self.height}")


@dataclass 
class TerminalState:
    """Complete terminal state information."""
    size: TerminalSize
    alternate_screen_active: bool
    cursor_visible: bool
    original_cursor_state: bool
    tty_available: bool
    capabilities: Dict[str, Any]
    last_updated: datetime
    

class SizeChangedEvent:
    """Event fired when terminal size changes."""
    
    def __init__(self, old_size: TerminalSize, new_size: TerminalSize):
        self.old_size = old_size
        self.new_size = new_size
        self.timestamp = datetime.now()
        self.delta_width = new_size.width - old_size.width
        self.delta_height = new_size.height - old_size.height


@dataclass
class CleanupResult:
    """Result of terminal cleanup operations."""
    success: bool
    alternate_screen_restored: bool
    cursor_restored: bool
    error_message: Optional[str] = None
    operations_completed: int = 0
    total_operations: int = 0


class TerminalController:
    """
    Centralized terminal state management and control.
    
    Prevents pollution and conflicts between different TUI components by providing
    a single point of control for terminal operations.
    """
    
    def __init__(self):
        """Initialize the terminal controller."""
        self._lock = asyncio.Lock()
        self._initialized = False
        self._active_contexts: Set[weakref.ref] = set()
        
        # Terminal state
        self._current_state: Optional[TerminalState] = None
        self._original_cursor_visible = True  # Assume visible initially
        self._alternate_screen_active = False
        self._cleanup_registered = False
        
        # Size monitoring
        self._size_callbacks: Set[Callable[[SizeChangedEvent], None]] = set()
        self._size_monitor_task: Optional[asyncio.Task] = None
        self._size_monitor_active = False
        
        # Terminal capabilities cache
        self._capabilities_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[datetime] = None
        
        # Performance tracking
        self._operation_times: Dict[str, float] = {}
    
    async def initialize(self) -> bool:
        """
        Initialize the terminal controller.
        
        Returns:
            True if initialization successful, False otherwise
        """
        async with self._lock:
            if self._initialized:
                return True
                
            try:
                # Detect initial terminal state
                initial_size = await self._detect_terminal_size()
                capabilities = await self._detect_capabilities()
                
                self._current_state = TerminalState(
                    size=initial_size,
                    alternate_screen_active=False,
                    cursor_visible=True,
                    original_cursor_state=True,
                    tty_available=sys.stdout.isatty(),
                    capabilities=capabilities,
                    last_updated=datetime.now()
                )
                
                # Register cleanup handlers
                self._register_cleanup_handlers()
                
                # Start size monitoring if in TTY
                if self._current_state.tty_available:
                    await self._start_size_monitoring()
                
                self._initialized = True
                return True
                
            except Exception as e:
                # Log error but don't raise - return False to indicate failure
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to initialize terminal controller: {e}")
                return False
    
    async def get_terminal_state(self) -> Optional[TerminalState]:
        """
        Get current terminal state.
        
        Returns:
            TerminalState object if initialized, None otherwise
        """
        if not self._initialized:
            return None
            
        # Update size if needed
        current_size = await self._detect_terminal_size()
        if (self._current_state and 
            (current_size.width != self._current_state.size.width or
             current_size.height != self._current_state.size.height)):
            await self._handle_size_change(current_size)
            
        return self._current_state
    
    async def enter_alternate_screen(self, mode: AlternateScreenMode = AlternateScreenMode.AUTO) -> bool:
        """
        Enter alternate screen buffer.
        
        Args:
            mode: How to handle alternate screen activation
            
        Returns:
            True if alternate screen was activated, False otherwise
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # THREAT: UI freeze from blocking terminal operations
            # MITIGATION: Apply timeout to async operations
            async with asyncio.timeout(2.0):  # 2 second timeout
                async with self._lock:
                    if not self._initialized or not self._current_state:
                        return False
                    
                    # Check if already active
                    if self._alternate_screen_active:
                        return True
                    
                    # Determine if we should activate based on mode
                    should_activate = False
                    if mode == AlternateScreenMode.ENABLED:
                        should_activate = True
                    elif mode == AlternateScreenMode.AUTO:
                        should_activate = (
                            self._current_state.tty_available and 
                            self._current_state.capabilities.get('alternate_screen', False)
                        )
                    
                    if not should_activate:
                        return False
                    
                    # Activate alternate screen
                    if sys.stdout.isatty():
                        try:
                            # ANSI escape sequence for alternate screen
                            sys.stdout.write('\x1b[?1049h')
                            sys.stdout.flush()
                            self._alternate_screen_active = True
                            
                            # Update state
                            if self._current_state:
                                self._current_state.alternate_screen_active = True
                                self._current_state.last_updated = datetime.now()
                            
                            return True
                        except Exception:
                            return False
                    
                    return False
                
        except asyncio.TimeoutError:
            # Terminal operation timed out
            return False
        finally:
            operation_time = asyncio.get_event_loop().time() - start_time
            self._operation_times['enter_alternate_screen'] = operation_time
    
    async def exit_alternate_screen(self) -> bool:
        """
        Exit alternate screen buffer.
        
        Returns:
            True if alternate screen was deactivated, False otherwise
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self._lock:
                if not self._alternate_screen_active:
                    return True
                
                # Deactivate alternate screen
                if sys.stdout.isatty():
                    try:
                        # ANSI escape sequence to exit alternate screen
                        sys.stdout.write('\x1b[?1049l')
                        sys.stdout.flush()
                        self._alternate_screen_active = False
                        
                        # Update state
                        if self._current_state:
                            self._current_state.alternate_screen_active = False
                            self._current_state.last_updated = datetime.now()
                        
                        return True
                    except Exception:
                        return False
                
                return True
                
        finally:
            operation_time = asyncio.get_event_loop().time() - start_time
            self._operation_times['exit_alternate_screen'] = operation_time
    
    async def set_cursor_visibility(self, visibility: CursorVisibility) -> bool:
        """
        Set cursor visibility.
        
        Args:
            visibility: Desired cursor visibility state
            
        Returns:
            True if cursor visibility was changed, False otherwise
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self._lock:
                if not self._initialized or not self._current_state:
                    return False
                
                if not sys.stdout.isatty():
                    return False
                
                try:
                    if visibility == CursorVisibility.HIDDEN:
                        # Hide cursor
                        sys.stdout.write('\x1b[?25l')
                        sys.stdout.flush()
                        self._current_state.cursor_visible = False
                    elif visibility == CursorVisibility.VISIBLE:
                        # Show cursor
                        sys.stdout.write('\x1b[?25h')
                        sys.stdout.flush()
                        self._current_state.cursor_visible = True
                    elif visibility == CursorVisibility.AUTO:
                        # Restore to original state
                        if self._original_cursor_visible:
                            sys.stdout.write('\x1b[?25h')
                            self._current_state.cursor_visible = True
                        else:
                            sys.stdout.write('\x1b[?25l')
                            self._current_state.cursor_visible = False
                        sys.stdout.flush()
                    
                    self._current_state.last_updated = datetime.now()
                    return True
                    
                except Exception:
                    return False
                    
        finally:
            operation_time = asyncio.get_event_loop().time() - start_time
            self._operation_times['set_cursor_visibility'] = operation_time
    
    def register_size_change_callback(self, callback: Callable[[SizeChangedEvent], None]) -> None:
        """
        Register a callback for terminal size changes.
        
        Args:
            callback: Function to call when size changes
        """
        # Note: Using asyncio.run since this is a sync function but we have an async lock
        # This is not ideal but needed for backward compatibility
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, we can't use async with from a sync function
            # Instead, just add the callback directly - the lock is mainly for initialization
            self._size_callbacks.add(callback)
        except RuntimeError:
            # No event loop running, safe to create one temporarily
            asyncio.run(self._add_callback_async(callback))
    
    async def _add_callback_async(self, callback: Callable[[SizeChangedEvent], None]) -> None:
        """Helper method to add callback with async lock."""
        async with self._lock:
            self._size_callbacks.add(callback)
    
    def unregister_size_change_callback(self, callback: Callable[[SizeChangedEvent], None]) -> None:
        """
        Unregister a size change callback.
        
        Args:
            callback: Function to remove from callbacks
        """
        # Same pattern as register - avoid async with in sync function
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, just remove directly
            self._size_callbacks.discard(callback)
        except RuntimeError:
            # No event loop running, safe to create one temporarily
            asyncio.run(self._remove_callback_async(callback))
    
    async def _remove_callback_async(self, callback: Callable[[SizeChangedEvent], None]) -> None:
        """Helper method to remove callback with async lock."""
        async with self._lock:
            self._size_callbacks.discard(callback)
    
    @contextmanager
    def terminal_context(self, 
                        alternate_screen: AlternateScreenMode = AlternateScreenMode.AUTO,
                        cursor_visibility: CursorVisibility = CursorVisibility.AUTO):
        """
        Context manager for terminal operations.
        
        Args:
            alternate_screen: How to handle alternate screen
            cursor_visibility: How to handle cursor visibility
        """
        class TerminalContext:
            def __init__(self, controller: 'TerminalController'):
                self.controller = controller
                self.ref = weakref.ref(self)
                controller._active_contexts.add(self.ref)
        
        context = TerminalContext(self)
        
        try:
            # Setup terminal state
            if asyncio.iscoroutinefunction(self.enter_alternate_screen):
                # We're in async context, need to handle differently
                pass
            else:
                asyncio.create_task(self.enter_alternate_screen(alternate_screen))
                asyncio.create_task(self.set_cursor_visibility(cursor_visibility))
            
            yield context
            
        finally:
            # Cleanup terminal state
            try:
                if asyncio.iscoroutinefunction(self.exit_alternate_screen):
                    asyncio.create_task(self.exit_alternate_screen())
                    asyncio.create_task(self.set_cursor_visibility(CursorVisibility.AUTO))
                
                self._active_contexts.discard(context.ref)
            except Exception:
                pass  # Best effort cleanup
    
    async def cleanup(self) -> CleanupResult:
        """
        Cleanup terminal controller and restore terminal state.
        
        Returns:
            CleanupResult with cleanup status and details
        """
        operations_completed = 0
        total_operations = 3  # alternate screen, cursor, size monitor
        
        try:
            # Stop size monitoring
            if self._size_monitor_task and not self._size_monitor_task.done():
                self._size_monitor_task.cancel()
                try:
                    await self._size_monitor_task
                except asyncio.CancelledError:
                    pass
                operations_completed += 1
            
            # Restore alternate screen
            alternate_screen_restored = await self.exit_alternate_screen()
            if alternate_screen_restored:
                operations_completed += 1
            
            # Restore cursor
            cursor_restored = await self.set_cursor_visibility(CursorVisibility.AUTO)
            if cursor_restored:
                operations_completed += 1
            
            # Clear state
            async with self._lock:
                self._initialized = False
                self._current_state = None
                self._size_callbacks.clear()
                self._active_contexts.clear()
            
            return CleanupResult(
                success=True,
                alternate_screen_restored=alternate_screen_restored,
                cursor_restored=cursor_restored,
                operations_completed=operations_completed,
                total_operations=total_operations
            )
            
        except Exception as e:
            return CleanupResult(
                success=False,
                alternate_screen_restored=False,
                cursor_restored=False,
                error_message=str(e),
                operations_completed=operations_completed,
                total_operations=total_operations
            )
    
    async def _detect_terminal_size(self) -> TerminalSize:
        """Detect current terminal size."""
        try:
            # Try multiple methods for reliability
            width, height = 0, 0
            
            # Method 1: shutil.get_terminal_size (most reliable)
            try:
                size = shutil.get_terminal_size()
                width, height = size.columns, size.lines
            except:
                pass
            
            # Method 2: Environment variables with validation
            if width <= 0 or height <= 0:
                try:
                    # THREAT: Environment variable injection
                    # MITIGATION: Validate and bound terminal dimensions
                    env_width = os.getenv('COLUMNS', '0')
                    env_height = os.getenv('LINES', '0')
                    
                    # Sanitize and validate environment variables
                    if env_width.isdigit():
                        width = int(env_width)
                    if env_height.isdigit():
                        height = int(env_height)
                    
                    # Enforce reasonable bounds to prevent memory exhaustion
                    width = max(1, min(width, 1000))
                    height = max(1, min(height, 1000))
                except:
                    pass
            
            # Method 3: Default fallback
            if width <= 0 or height <= 0:
                width, height = 80, 24
            
            # Final bounds checking
            width = max(1, min(width, 1000))
            height = max(1, min(height, 1000))
            
            return TerminalSize(
                width=width,
                height=height,
                timestamp=datetime.now()
            )
            
        except Exception:
            # Final fallback with safe dimensions
            return TerminalSize(width=80, height=24, timestamp=datetime.now())
    
    async def _detect_capabilities(self) -> Dict[str, Any]:
        """Detect terminal capabilities."""
        if self._capabilities_cache and self._cache_timestamp:
            # Use cache if recent (within 30 seconds)
            age = (datetime.now() - self._cache_timestamp).total_seconds()
            if age < 30:
                return self._capabilities_cache
        
        capabilities = {}
        
        try:
            # Basic TTY detection
            capabilities['tty'] = sys.stdout.isatty()
            
            # Color support
            term = os.getenv('TERM', '').lower()
            colorterm = os.getenv('COLORTERM', '').lower()
            
            if colorterm in ('truecolor', '24bit'):
                capabilities['colors'] = 16777216
            elif '256color' in term or '256' in term:
                capabilities['colors'] = 256
            elif 'color' in term or 'ansi' in term:
                capabilities['colors'] = 16
            else:
                capabilities['colors'] = 0
            
            # Alternate screen support
            capabilities['alternate_screen'] = (
                capabilities['tty'] and 
                term and 
                term != 'dumb' and
                'screen' not in term  # Screen/tmux handle differently
            )
            
            # Unicode support
            encoding = getattr(sys.stdout, 'encoding', '') or ''
            capabilities['unicode'] = 'utf' in encoding.lower()
            
            # Mouse support (basic detection)
            capabilities['mouse'] = (
                capabilities['tty'] and
                any(t in term for t in ['xterm', 'screen', 'tmux'])
            )
            
            self._capabilities_cache = capabilities
            self._cache_timestamp = datetime.now()
            
        except Exception:
            # Minimal fallback capabilities
            capabilities = {
                'tty': False,
                'colors': 0,
                'alternate_screen': False,
                'unicode': False,
                'mouse': False
            }
        
        return capabilities
    
    async def _start_size_monitoring(self) -> None:
        """Start monitoring terminal size changes."""
        if self._size_monitor_active:
            return
        
        self._size_monitor_active = True
        self._size_monitor_task = asyncio.create_task(self._size_monitor_loop())
    
    async def _size_monitor_loop(self) -> None:
        """Monitor terminal size changes in background."""
        last_size = None
        
        try:
            while self._size_monitor_active:
                try:
                    current_size = await self._detect_terminal_size()
                    
                    if (last_size and 
                        (current_size.width != last_size.width or 
                         current_size.height != last_size.height)):
                        await self._handle_size_change(current_size)
                    
                    last_size = current_size
                    
                    # Check every 500ms
                    await asyncio.sleep(0.5)
                    
                except asyncio.CancelledError:
                    break
                except Exception:
                    # Continue monitoring despite errors
                    await asyncio.sleep(1.0)
                    
        finally:
            self._size_monitor_active = False
    
    async def _handle_size_change(self, new_size: TerminalSize) -> None:
        """Handle terminal size change event."""
        if not self._current_state:
            return
        
        old_size = self._current_state.size
        
        # Update current state
        self._current_state.size = new_size
        self._current_state.last_updated = datetime.now()
        
        # Notify callbacks
        if self._size_callbacks:
            event = SizeChangedEvent(old_size, new_size)
            for callback in list(self._size_callbacks):  # Copy to avoid modification during iteration
                try:
                    callback(event)
                except Exception:
                    # Don't let callback errors break size monitoring
                    pass
    
    def _register_cleanup_handlers(self) -> None:
        """Register cleanup handlers for graceful shutdown."""
        if self._cleanup_registered:
            return
        
        def cleanup_handler(signum=None, frame=None):
            """Emergency cleanup handler."""
            try:
                if self._alternate_screen_active and sys.stdout.isatty():
                    sys.stdout.write('\x1b[?1049l\x1b[?25h')
                    sys.stdout.flush()
            except:
                pass
        
        # Register signal handlers
        try:
            signal.signal(signal.SIGINT, cleanup_handler)
            signal.signal(signal.SIGTERM, cleanup_handler)
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, cleanup_handler)
        except:
            pass  # May not work in all environments
        
        # Register atexit handler
        import atexit
        atexit.register(cleanup_handler)
        
        self._cleanup_registered = True
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for terminal operations."""
        return {
            'operation_times': dict(self._operation_times),
            'active_contexts': len(self._active_contexts),
            'size_callbacks': len(self._size_callbacks),
            'monitoring_active': self._size_monitor_active,
            'initialized': self._initialized
        }


# Singleton instance for global access
_terminal_controller: Optional[TerminalController] = None


async def get_terminal_controller() -> TerminalController:
    """
    Get or create the global terminal controller instance.
    
    Returns:
        TerminalController instance
    """
    global _terminal_controller
    
    if _terminal_controller is None:
        _terminal_controller = TerminalController()
        await _terminal_controller.initialize()
    
    return _terminal_controller


async def cleanup_terminal_controller() -> CleanupResult:
    """
    Cleanup the global terminal controller.
    
    Returns:
        CleanupResult with cleanup status
    """
    global _terminal_controller
    
    if _terminal_controller:
        result = await _terminal_controller.cleanup()
        _terminal_controller = None
        return result
    
    return CleanupResult(success=True, alternate_screen_restored=True, cursor_restored=True)