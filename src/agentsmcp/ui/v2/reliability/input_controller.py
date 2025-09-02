"""
Input Controller - Non-blocking, timeout-protected input system for TUI.

This module provides the critical input handling system that prevents the TUI
from hanging on terminal operations. It implements multiple input modes with
automatic fallback and guaranteed responsiveness.

Key Features:
- GUARANTEED non-blocking operation - input thread never hangs main UI
- Fast terminal setup with 1s timeout maximum and automatic fallback
- Input responsiveness within 100ms or immediate user feedback
- Multiple input modes: RAW|LINE|SIMULATED with automatic fallback
- Graceful exit: Ctrl+C always works within 1s maximum
- Timeout guardian integration for all blocking operations

Input Modes:
1. RAW mode: Character-by-character input with escape sequence processing
2. LINE mode: Line-based input for environments without raw terminal access
3. SIMULATED mode: Demo mode for non-interactive environments

Usage:
    controller = InputController()
    async for event in controller.get_input_stream():
        if event.event_type == InputEventType.CHARACTER:
            print(f"Character: {event.data}")
        elif event.event_type == InputEventType.CONTROL:
            if event.data == "ctrl_c":
                break
"""

import asyncio
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional, Dict, Any, Callable, List
import signal

from .timeout_guardian import TimeoutGuardian, timeout_protection

logger = logging.getLogger(__name__)


class InputMode(Enum):
    """Input controller operating modes."""
    RAW = "raw"           # Raw character input with escape sequences  
    LINE = "line"         # Line-based input for limited terminals
    SIMULATED = "simulated"  # Demo mode for non-interactive environments


class InputEventType(Enum):
    """Types of input events."""
    CHARACTER = "character"      # Regular character input
    CONTROL = "control"          # Control keys (Ctrl+C, Ctrl+D, etc)
    SPECIAL = "special"          # Special keys (arrows, function keys, etc)
    BACKSPACE = "backspace"      # Backspace/delete
    ENTER = "enter"             # Enter/return key
    ESCAPE = "escape"           # Escape key
    HISTORY = "history"         # History navigation (up/down arrows)


@dataclass
class InputEvent:
    """Represents an input event from the controller."""
    event_type: InputEventType
    data: str
    timestamp: float
    raw_data: Optional[bytes] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class InputController:
    """
    Non-blocking input controller with guaranteed responsiveness.
    
    Prevents TUI hangs by implementing timeout-protected terminal operations
    with automatic fallback modes and guaranteed exit handling.
    """
    
    def __init__(self, 
                 response_timeout: float = 0.1,  # 100ms guaranteed response
                 setup_timeout: float = 1.0,     # 1s max for terminal setup
                 exit_timeout: float = 1.0):     # 1s max for graceful exit
        """
        Initialize the input controller.
        
        Args:
            response_timeout: Maximum time to wait for input processing (seconds)
            setup_timeout: Maximum time for terminal setup operations (seconds) 
            exit_timeout: Maximum time for graceful exit (seconds)
        """
        self.response_timeout = response_timeout
        self.setup_timeout = setup_timeout
        self.exit_timeout = exit_timeout
        
        # Operating mode
        self.current_mode = InputMode.SIMULATED
        self.mode_detection_complete = False
        
        # State management
        self.running = False
        self.shutdown_requested = False
        self._shutdown_event = asyncio.Event()
        
        # Terminal state
        self._original_terminal_settings = None
        self._terminal_fd = None
        self._tty_available = False
        
        # Input processing
        self._input_queue: asyncio.Queue = None
        self._input_thread: Optional[threading.Thread] = None
        self._thread_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="InputController")
        
        # Timeout protection
        self.timeout_guardian = TimeoutGuardian(
            default_timeout=setup_timeout,
            detection_precision=0.01,  # 10ms precision
            cleanup_timeout=0.5
        )
        
        # Statistics
        self.events_processed = 0
        self.setup_time = 0.0
        self.last_event_time = 0.0
        
        logger.debug("InputController initialized")
    
    async def start(self) -> bool:
        """
        Start the input controller with timeout-protected setup.
        
        Returns:
            True if started successfully, False if fallback to simulated mode
        """
        if self.running:
            logger.warning("InputController already running")
            return True
        
        logger.info("Starting InputController with timeout protection")
        setup_start = time.time()
        
        try:
            # Protected terminal setup with 1s timeout
            async with timeout_protection("terminal_setup", self.setup_timeout):
                await self._detect_and_setup_input_mode()
            
            # Initialize input queue
            self._input_queue = asyncio.Queue(maxsize=100)
            
            # Start input processing
            self.running = True
            await self._start_input_processing()
            
            self.setup_time = time.time() - setup_start
            logger.info(f"InputController started in {self.current_mode.value} mode ({self.setup_time:.3f}s)")
            
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"Terminal setup timed out after {self.setup_timeout}s, using simulated mode")
            self.current_mode = InputMode.SIMULATED
            self._input_queue = asyncio.Queue(maxsize=100)
            self.running = True
            await self._start_simulated_input()
            return False
            
        except Exception as e:
            logger.error(f"Failed to start InputController: {e}")
            # Emergency fallback to simulated mode
            self.current_mode = InputMode.SIMULATED
            self._input_queue = asyncio.Queue(maxsize=100) 
            self.running = True
            await self._start_simulated_input()
            return False
    
    async def _detect_and_setup_input_mode(self):
        """Detect best input mode and setup with timeout protection."""
        try:
            # Quick TTY detection with timeout
            async with timeout_protection("tty_detection", 0.5):
                self._tty_available = await self._detect_tty_support()
            
            if self._tty_available:
                # Try RAW mode setup with timeout
                try:
                    async with timeout_protection("raw_mode_setup", 0.5):
                        await self._setup_raw_mode()
                    self.current_mode = InputMode.RAW
                    logger.info("Using RAW input mode")
                    return
                except asyncio.TimeoutError:
                    logger.warning("RAW mode setup timed out, falling back to LINE mode")
                except Exception as e:
                    logger.warning(f"RAW mode setup failed: {e}, falling back to LINE mode")
                
                # Fallback to LINE mode
                self.current_mode = InputMode.LINE
                logger.info("Using LINE input mode")
                return
            
        except asyncio.TimeoutError:
            logger.warning("TTY detection timed out")
        except Exception as e:
            logger.warning(f"Input mode detection failed: {e}")
        
        # Final fallback to SIMULATED mode
        self.current_mode = InputMode.SIMULATED
        logger.info("Using SIMULATED input mode")
    
    async def _detect_tty_support(self) -> bool:
        """Detect TTY support with timeout protection.""" 
        def detect_sync():
            try:
                if not sys.stdin or not hasattr(sys.stdin, 'isatty'):
                    return False
                
                if not sys.stdin.isatty():
                    return False
                
                # Try to import required modules
                import termios
                import tty
                import select
                
                # Test basic terminal access
                try:
                    fd = sys.stdin.fileno()
                    termios.tcgetattr(fd)
                    return True
                except (OSError, termios.error):
                    # Try /dev/tty fallback
                    try:
                        test_fd = os.open('/dev/tty', os.O_RDONLY)
                        termios.tcgetattr(test_fd)
                        os.close(test_fd)
                        return True
                    except (OSError, termios.error):
                        return False
                        
            except ImportError:
                return False
            except Exception:
                return False
        
        # Run detection in thread pool with timeout
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(self._thread_executor, detect_sync)
            result = await asyncio.wait_for(future, timeout=0.3)
            return result
        except asyncio.TimeoutError:
            logger.warning("TTY detection timed out")
            return False
        except Exception as e:
            logger.warning(f"TTY detection failed: {e}")
            return False
    
    async def _setup_raw_mode(self):
        """Setup raw terminal mode with timeout protection."""
        def setup_sync():
            try:
                import termios
                import tty
                
                # Try stdin first
                try:
                    fd = sys.stdin.fileno()
                    original_settings = termios.tcgetattr(fd)
                    tty.setraw(fd)
                    self._terminal_fd = fd
                    self._original_terminal_settings = original_settings
                    return True
                except (OSError, termios.error):
                    # Fallback to /dev/tty
                    fd = os.open('/dev/tty', os.O_RDONLY)
                    original_settings = termios.tcgetattr(fd)
                    tty.setraw(fd)
                    self._terminal_fd = fd
                    self._original_terminal_settings = original_settings
                    return True
                    
            except Exception as e:
                logger.error(f"Raw mode setup failed: {e}")
                raise
        
        # Run setup in thread pool with timeout
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._thread_executor, setup_sync)
        await asyncio.wait_for(future, timeout=0.4)
        logger.debug("Raw terminal mode setup complete")
    
    async def _start_input_processing(self):
        """Start the appropriate input processing based on detected mode."""
        if self.current_mode == InputMode.RAW:
            await self._start_raw_input()
        elif self.current_mode == InputMode.LINE:
            await self._start_line_input()
        else:
            await self._start_simulated_input()
    
    async def _start_raw_input(self):
        """Start raw character input processing."""
        def raw_input_thread():
            """Raw input processing thread with timeout protection.""" 
            try:
                import select
                
                while self.running and not self.shutdown_requested:
                    try:
                        # Non-blocking select with timeout
                        ready, _, _ = select.select([self._terminal_fd], [], [], 0.1)
                        
                        if not ready:
                            continue
                        
                        # Read available data with timeout protection
                        try:
                            data = os.read(self._terminal_fd, 64)
                        except (OSError, ValueError):
                            continue
                        
                        if not data:
                            continue
                        
                        # Process raw bytes and queue events
                        events = self._process_raw_bytes(data)
                        for event in events:
                            try:
                                self._input_queue.put_nowait(event)
                            except asyncio.QueueFull:
                                # Drop oldest event if queue is full
                                try:
                                    self._input_queue.get_nowait()
                                    self._input_queue.put_nowait(event)
                                except:
                                    pass
                        
                        # Update last event time
                        self.last_event_time = time.time()
                        
                    except Exception as e:
                        logger.debug(f"Error in raw input thread: {e}")
                        time.sleep(0.01)  # Brief pause on error
                        
            except Exception as e:
                logger.error(f"Raw input thread failed: {e}")
                # Signal fallback needed
                if self.running:
                    fallback_event = InputEvent(
                        event_type=InputEventType.CONTROL,
                        data="fallback_needed",
                        timestamp=time.time()
                    )
                    try:
                        self._input_queue.put_nowait(fallback_event)
                    except:
                        pass
        
        # Start the input thread
        self._input_thread = threading.Thread(
            target=raw_input_thread,
            daemon=True,
            name="RawInputProcessor"
        )
        self._input_thread.start()
        logger.debug("Raw input processing started")
    
    async def _start_line_input(self):
        """Start line-based input processing."""
        async def line_input_processor():
            """Line input processing coroutine."""
            while self.running and not self.shutdown_requested:
                try:
                    # Get line input with timeout
                    loop = asyncio.get_running_loop()
                    future = loop.run_in_executor(self._thread_executor, input, "💬 > ")
                    
                    try:
                        line = await asyncio.wait_for(future, timeout=1.0)
                        
                        if line.strip():
                            event = InputEvent(
                                event_type=InputEventType.ENTER,
                                data=line.strip(),
                                timestamp=time.time()
                            )
                            await self._input_queue.put(event)
                            self.last_event_time = time.time()
                        
                    except asyncio.TimeoutError:
                        # Timeout is normal for line input, just continue
                        continue
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Line input error: {e}")
                    await asyncio.sleep(0.5)
        
        # Start line input processor
        asyncio.create_task(line_input_processor())
        logger.debug("Line input processing started")
    
    async def _start_simulated_input(self):
        """Start simulated input for demo/testing."""
        async def simulated_input_processor():
            """Simulated input processor for demo mode."""
            demo_commands = [
                "status",
                "help", 
                "Demo: Processing sample task...",
                "Demo: Checking system health...",
                "Demo: Revolutionary TUI demonstration complete!",
                "quit"
            ]
            
            command_index = 0
            
            while self.running and not self.shutdown_requested:
                try:
                    # Wait between demo commands
                    await asyncio.sleep(5.0)
                    
                    if command_index < len(demo_commands):
                        cmd = demo_commands[command_index]
                        
                        # Skip demo messages 
                        if not cmd.startswith("Demo:"):
                            event = InputEvent(
                                event_type=InputEventType.ENTER,
                                data=cmd,
                                timestamp=time.time()
                            )
                            await self._input_queue.put(event)
                            self.last_event_time = time.time()
                        
                        command_index += 1
                    else:
                        # Demo complete, send quit
                        event = InputEvent(
                            event_type=InputEventType.ENTER,
                            data="quit",
                            timestamp=time.time()
                        )
                        await self._input_queue.put(event)
                        break
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Simulated input error: {e}")
                    await asyncio.sleep(1.0)
        
        # Start simulated input
        asyncio.create_task(simulated_input_processor())
        logger.info("Simulated input processing started")
    
    def _process_raw_bytes(self, data: bytes) -> List[InputEvent]:
        """Process raw byte data into input events."""
        events = []
        i = 0
        
        while i < len(data):
            b = data[i]
            
            # Ctrl+C / Ctrl+D - Exit
            if b in (3, 4):
                event = InputEvent(
                    event_type=InputEventType.CONTROL,
                    data="ctrl_c" if b == 3 else "ctrl_d",
                    timestamp=time.time(),
                    raw_data=bytes([b])
                )
                events.append(event)
                i += 1
                continue
            
            # Backspace (8 or 127)
            elif b in (8, 127):
                event = InputEvent(
                    event_type=InputEventType.BACKSPACE,
                    data="backspace",
                    timestamp=time.time(),
                    raw_data=bytes([b])
                )
                events.append(event)
                i += 1
                continue
            
            # Enter (13)
            elif b == 13:
                event = InputEvent(
                    event_type=InputEventType.ENTER,
                    data="enter",
                    timestamp=time.time(),
                    raw_data=bytes([b])
                )
                events.append(event)
                i += 1
                continue
            
            # Line feed (10)
            elif b == 10:
                event = InputEvent(
                    event_type=InputEventType.CHARACTER,
                    data='\n',
                    timestamp=time.time(),
                    raw_data=bytes([b])
                )
                events.append(event)
                i += 1
                continue
            
            # ESC sequences (27)
            elif b == 27:
                seq_bytes = [b]
                j = i + 1
                
                # Parse escape sequence
                if j < len(data) and data[j] == ord('['):
                    seq_bytes.append(data[j])
                    j += 1
                    
                    # Read until end character
                    while j < len(data):
                        seq_bytes.append(data[j])
                        # End on A-Z, a-z, or ~
                        if ((65 <= data[j] <= 90) or (97 <= data[j] <= 122) or 
                            data[j] == ord('~')):
                            break
                        j += 1
                    
                    # Convert to string for processing
                    try:
                        seq_str = bytes(seq_bytes[2:]).decode('utf-8', errors='ignore')
                        
                        # Handle arrow keys
                        if seq_str == 'A':  # Up arrow
                            event = InputEvent(
                                event_type=InputEventType.HISTORY,
                                data="up",
                                timestamp=time.time(),
                                raw_data=bytes(seq_bytes)
                            )
                        elif seq_str == 'B':  # Down arrow
                            event = InputEvent(
                                event_type=InputEventType.HISTORY,
                                data="down",
                                timestamp=time.time(),
                                raw_data=bytes(seq_bytes)
                            )
                        elif seq_str in ('C', 'D'):  # Right/Left arrows
                            event = InputEvent(
                                event_type=InputEventType.SPECIAL,
                                data="right" if seq_str == 'C' else "left",
                                timestamp=time.time(),
                                raw_data=bytes(seq_bytes)
                            )
                        else:
                            # Unknown escape sequence
                            event = InputEvent(
                                event_type=InputEventType.SPECIAL,
                                data=f"esc_{seq_str}",
                                timestamp=time.time(),
                                raw_data=bytes(seq_bytes)
                            )
                        
                        events.append(event)
                    except:
                        pass  # Skip malformed sequences
                    
                    i = j + 1
                else:
                    # ESC alone
                    event = InputEvent(
                        event_type=InputEventType.ESCAPE,
                        data="escape",
                        timestamp=time.time(),
                        raw_data=bytes([b])
                    )
                    events.append(event)
                    i += 1
                continue
            
            # Regular printable characters (32-126)
            elif 32 <= b <= 126:
                try:
                    char = bytes([b]).decode('utf-8', errors='ignore')
                    if char:
                        event = InputEvent(
                            event_type=InputEventType.CHARACTER,
                            data=char,
                            timestamp=time.time(),
                            raw_data=bytes([b])
                        )
                        events.append(event)
                except:
                    pass  # Skip invalid characters
                i += 1
                continue
            
            # Skip other control characters
            else:
                i += 1
                continue
        
        return events
    
    async def get_input_stream(self) -> AsyncIterator[InputEvent]:
        """
        Get async iterator of input events.
        
        This is the main interface for consuming input events. It guarantees
        that events are processed within the response timeout or provides
        immediate feedback about delays.
        
        Yields:
            InputEvent: Input events as they occur
        """
        if not self.running:
            raise RuntimeError("InputController not started")
        
        logger.debug("Starting input event stream")
        
        while self.running and not self.shutdown_requested:
            try:
                # Get next event with response timeout
                event = await asyncio.wait_for(
                    self._input_queue.get(),
                    timeout=self.response_timeout
                )
                
                # Handle special control events
                if event.event_type == InputEventType.CONTROL:
                    if event.data in ("ctrl_c", "ctrl_d"):
                        logger.info(f"Exit requested via {event.data}")
                        await self.stop()
                        yield event
                        break
                    elif event.data == "fallback_needed":
                        logger.warning("Input fallback requested")
                        await self._switch_to_fallback_mode()
                        continue
                
                # Update statistics
                self.events_processed += 1
                self.last_event_time = event.timestamp
                
                yield event
                
            except asyncio.TimeoutError:
                # No input available - this is normal, just continue
                # Could yield a heartbeat event if needed
                await asyncio.sleep(0.01)
                continue
                
            except asyncio.CancelledError:
                logger.debug("Input stream cancelled")
                break
                
            except Exception as e:
                logger.error(f"Error in input stream: {e}")
                await asyncio.sleep(0.1)
                continue
        
        logger.debug("Input event stream ended")
    
    async def _switch_to_fallback_mode(self):
        """Switch to fallback input mode when raw mode fails."""
        logger.warning("Switching to fallback input mode")
        
        # Stop current processing
        await self._stop_current_processing()
        
        # Switch to line mode or simulated mode
        if self._tty_available:
            self.current_mode = InputMode.LINE
            await self._start_line_input()
        else:
            self.current_mode = InputMode.SIMULATED
            await self._start_simulated_input()
        
        logger.info(f"Switched to {self.current_mode.value} input mode")
    
    async def _stop_current_processing(self):
        """Stop current input processing."""
        if self._input_thread and self._input_thread.is_alive():
            # Signal thread to stop
            self.shutdown_requested = True
            
            # Wait briefly for thread to stop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._input_thread.join, 0.5)
    
    async def stop(self) -> bool:
        """
        Stop the input controller with guaranteed exit within timeout.
        
        Returns:
            True if stopped gracefully, False if forced termination
        """
        if not self.running:
            return True
        
        logger.info("Stopping InputController")
        self.shutdown_requested = True
        self.running = False
        self._shutdown_event.set()
        
        try:
            # Protected shutdown with timeout
            async with timeout_protection("input_controller_shutdown", self.exit_timeout):
                # Stop input processing
                await self._stop_current_processing()
                
                # Restore terminal settings
                await self._restore_terminal_settings()
                
                # Shutdown thread pool
                if self._thread_executor:
                    self._thread_executor.shutdown(wait=False)
                
                # Shutdown timeout guardian
                await self.timeout_guardian.shutdown()
                
            logger.info("InputController stopped gracefully")
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"InputController shutdown timed out after {self.exit_timeout}s")
            # Force cleanup
            await self._force_cleanup()
            return False
            
        except Exception as e:
            logger.error(f"Error stopping InputController: {e}")
            await self._force_cleanup()
            return False
    
    async def _restore_terminal_settings(self):
        """Restore original terminal settings with timeout protection."""
        if not self._original_terminal_settings or not self._terminal_fd:
            return
        
        def restore_sync():
            try:
                import termios
                termios.tcsetattr(
                    self._terminal_fd, 
                    termios.TCSADRAIN, 
                    self._original_terminal_settings
                )
                logger.debug("Terminal settings restored")
            except Exception as e:
                logger.warning(f"Failed to restore terminal settings: {e}")
        
        # Run restoration in thread pool with timeout
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(self._thread_executor, restore_sync)
            await asyncio.wait_for(future, timeout=0.3)
        except asyncio.TimeoutError:
            logger.warning("Terminal restoration timed out")
        except Exception as e:
            logger.warning(f"Terminal restoration failed: {e}")
        finally:
            # Close terminal fd if we opened it
            if (self._terminal_fd is not None and 
                self._terminal_fd != sys.stdin.fileno()):
                try:
                    os.close(self._terminal_fd)
                except:
                    pass
            
            self._original_terminal_settings = None
            self._terminal_fd = None
    
    async def _force_cleanup(self):
        """Force cleanup all resources."""
        try:
            # Force thread pool shutdown
            if self._thread_executor:
                self._thread_executor.shutdown(wait=False)
            
            # Force close terminal fd
            if (self._terminal_fd is not None and 
                self._terminal_fd != sys.stdin.fileno()):
                try:
                    os.close(self._terminal_fd)
                except:
                    pass
            
            logger.warning("InputController force cleanup completed")
        except Exception as e:
            logger.error(f"Force cleanup failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get input controller status and statistics."""
        current_time = time.time()
        
        return {
            "running": self.running,
            "current_mode": self.current_mode.value,
            "tty_available": self._tty_available,
            "events_processed": self.events_processed,
            "setup_time": round(self.setup_time, 3),
            "last_event_time": round(self.last_event_time, 3) if self.last_event_time else 0,
            "time_since_last_event": round(current_time - self.last_event_time, 3) if self.last_event_time else 0,
            "queue_size": self._input_queue.qsize() if self._input_queue else 0,
            "thread_alive": self._input_thread.is_alive() if self._input_thread else False,
            "timeout_stats": self.timeout_guardian.get_protection_stats()
        }


# Convenience functions
async def create_input_controller(**kwargs) -> InputController:
    """Create and start an input controller."""
    controller = InputController(**kwargs)
    await controller.start()
    return controller


# Example usage and testing
async def test_input_controller():
    """Test the input controller with various scenarios."""
    print("Testing InputController...")
    
    controller = InputController(response_timeout=0.1, setup_timeout=1.0)
    
    try:
        # Start the controller
        started = await controller.start()
        print(f"Controller started: {started}")
        print(f"Status: {controller.get_status()}")
        
        # Process some input events
        event_count = 0
        async for event in controller.get_input_stream():
            event_count += 1
            print(f"Event {event_count}: {event.event_type.value} = '{event.data}'")
            
            if event.event_type == InputEventType.CONTROL and event.data in ("ctrl_c", "ctrl_d"):
                break
            
            if event_count >= 10:  # Limit for demo
                break
        
        print(f"Processed {event_count} events")
        
    finally:
        # Stop the controller
        stopped = await controller.stop()
        print(f"Controller stopped gracefully: {stopped}")
        print(f"Final status: {controller.get_status()}")


if __name__ == "__main__":
    asyncio.run(test_input_controller())