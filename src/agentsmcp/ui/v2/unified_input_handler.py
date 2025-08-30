"""
Unified Input Handler - Immediate character echo and conflict resolution.

This module provides a unified input handling system that:
1. Shows typed characters immediately (fixes the main typing issue)
2. Replaces competing input handlers to prevent conflicts
3. Provides raw terminal input with proper processing
4. Handles special keys and sequences correctly
"""

import os
import sys
import asyncio
import select
import termios
import threading
import time
import logging
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from .terminal_state_manager import TerminalStateManager, TerminalMode

logger = logging.getLogger(__name__)


class InputEventType(Enum):
    """Types of input events."""
    CHARACTER = "character"
    SPECIAL_KEY = "special_key"
    MOUSE = "mouse"
    RESIZE = "resize"
    PASTE = "paste"
    ERROR = "error"


@dataclass
class InputEvent:
    """Input event data."""
    event_type: InputEventType
    data: Any
    timestamp: float
    raw_bytes: Optional[bytes] = None
    
    
class InputProcessor(ABC):
    """Abstract base class for input processors."""
    
    @abstractmethod
    async def process_event(self, event: InputEvent) -> bool:
        """
        Process an input event.
        
        Args:
            event: Input event to process
            
        Returns:
            True if event was handled, False to continue processing
        """
        pass


class CharacterEchoProcessor(InputProcessor):
    """Processor that provides immediate character echo."""
    
    def __init__(self, output_handler: Callable[[str], None]):
        """
        Initialize character echo processor.
        
        Args:
            output_handler: Function to call for immediate output
        """
        self.output_handler = output_handler
        self.buffer = ""
        self.cursor_pos = 0
        
    async def process_event(self, event: InputEvent) -> bool:
        """Process character events with immediate echo."""
        if event.event_type == InputEventType.CHARACTER:
            char = event.data.get('character', '')
            
            if char and len(char) == 1 and char.isprintable():
                # Insert character at cursor position
                self.buffer = self.buffer[:self.cursor_pos] + char + self.buffer[self.cursor_pos:]
                self.cursor_pos += 1
                
                # Immediate echo
                self.output_handler(self._render_buffer())
                return True
                
        elif event.event_type == InputEventType.SPECIAL_KEY:
            key = event.data.get('key', '')
            
            if key == 'backspace' and self.cursor_pos > 0:
                # Remove character before cursor
                self.buffer = self.buffer[:self.cursor_pos-1] + self.buffer[self.cursor_pos:]
                self.cursor_pos -= 1
                
                # Immediate echo
                self.output_handler(self._render_buffer())
                return True
                
            elif key == 'delete' and self.cursor_pos < len(self.buffer):
                # Remove character at cursor
                self.buffer = self.buffer[:self.cursor_pos] + self.buffer[self.cursor_pos+1:]
                
                # Immediate echo
                self.output_handler(self._render_buffer())
                return True
                
            elif key == 'left' and self.cursor_pos > 0:
                self.cursor_pos -= 1
                self.output_handler(self._render_buffer())
                return True
                
            elif key == 'right' and self.cursor_pos < len(self.buffer):
                self.cursor_pos += 1
                self.output_handler(self._render_buffer())
                return True
                
        return False
    
    def _render_buffer(self) -> str:
        """Render the current buffer with cursor indicator."""
        # Insert cursor indicator (block character)
        cursor_char = "█"
        display_buffer = self.buffer[:self.cursor_pos] + cursor_char + self.buffer[self.cursor_pos:]
        return display_buffer
    
    def get_buffer(self) -> str:
        """Get the current buffer content."""
        return self.buffer
    
    def clear_buffer(self):
        """Clear the input buffer."""
        self.buffer = ""
        self.cursor_pos = 0
        self.output_handler("")
    
    def set_buffer(self, text: str):
        """Set the buffer content."""
        self.buffer = text
        self.cursor_pos = len(text)
        self.output_handler(self._render_buffer())


class UnifiedInputHandler:
    """
    Unified input handler that provides immediate character echo
    and replaces competing input handlers.
    """
    
    def __init__(self):
        self.terminal_manager = TerminalStateManager()
        self.processors: List[InputProcessor] = []
        self.event_handlers: Dict[InputEventType, List[Callable]] = {}
        self.running = False
        self._input_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Input state
        self.echo_processor: Optional[CharacterEchoProcessor] = None
        self._last_input_time = 0.0
        
        # Sequence parsing state
        self._escape_buffer = b''
        self._parsing_sequence = False
        
    async def initialize(self, output_handler: Optional[Callable[[str], None]] = None) -> bool:
        """
        Initialize the unified input handler.
        
        Args:
            output_handler: Function to call for immediate character output
            
        Returns:
            True if initialization successful
        """
        try:
            # Initialize terminal state manager
            if not self.terminal_manager.initialize():
                logger.error("Failed to initialize terminal state manager")
                return False
            
            # Set up echo processor if output handler provided
            if output_handler:
                self.echo_processor = CharacterEchoProcessor(output_handler)
                self.add_processor(self.echo_processor)
            
            # Enter raw mode for immediate character input
            if not self.terminal_manager.enter_raw_mode():
                logger.warning("Failed to enter raw mode - input may be delayed")
            
            # Get current event loop
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.warning("No running event loop - some features may not work")
            
            logger.debug("Unified input handler initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize unified input handler: {e}")
            return False
    
    def add_processor(self, processor: InputProcessor):
        """Add an input processor."""
        if processor not in self.processors:
            self.processors.append(processor)
    
    def remove_processor(self, processor: InputProcessor):
        """Remove an input processor."""
        if processor in self.processors:
            self.processors.remove(processor)
    
    def add_event_handler(self, event_type: InputEventType, handler: Callable):
        """Add an event handler for specific event types."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: InputEventType, handler: Callable):
        """Remove an event handler."""
        if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
    
    async def start(self) -> bool:
        """Start the input handler."""
        if self.running:
            return True
            
        try:
            self.running = True
            self._stop_event.clear()
            
            # Start input reading thread
            self._input_thread = threading.Thread(
                target=self._input_thread_main,
                name="unified-input-handler",
                daemon=True
            )
            self._input_thread.start()
            
            logger.debug("Unified input handler started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start input handler: {e}")
            self.running = False
            return False
    
    async def stop(self):
        """Stop the input handler."""
        if not self.running:
            return
            
        self.running = False
        self._stop_event.set()
        
        # Wait for input thread to finish
        if self._input_thread and self._input_thread.is_alive():
            self._input_thread.join(timeout=1.0)
            if self._input_thread.is_alive():
                logger.warning("Input thread did not terminate cleanly")
        
        logger.debug("Unified input handler stopped")
    
    def _input_thread_main(self):
        """Main input reading thread."""
        try:
            tty_fd = self.terminal_manager._tty_fd
            if tty_fd is None:
                logger.error("No TTY file descriptor available")
                return
            
            while not self._stop_event.is_set():
                try:
                    # Use select for non-blocking read with timeout
                    ready, _, _ = select.select([tty_fd], [], [], 0.1)
                    
                    if ready:
                        # Read available data
                        data = os.read(tty_fd, 64)
                        if data:
                            self._process_raw_input(data)
                            
                except (OSError, IOError) as e:
                    if not self._stop_event.is_set():
                        logger.warning(f"Input read error: {e}")
                        time.sleep(0.1)
                    break
                    
        except Exception as e:
            logger.error(f"Input thread error: {e}")
            
        logger.debug("Input thread terminated")
    
    def _process_raw_input(self, data: bytes):
        """Process raw input data."""
        timestamp = time.time()
        
        for byte_val in data:
            try:
                self._process_byte(byte_val, timestamp)
            except Exception as e:
                logger.debug(f"Error processing byte {byte_val}: {e}")
    
    def _process_byte(self, byte_val: int, timestamp: float):
        """Process a single input byte."""
        # Handle escape sequences
        if self._parsing_sequence:
            self._escape_buffer += bytes([byte_val])
            
            # Check if sequence is complete
            if self._is_sequence_complete(self._escape_buffer):
                event = self._parse_escape_sequence(self._escape_buffer, timestamp)
                self._escape_buffer = b''
                self._parsing_sequence = False
                
                if event:
                    self._dispatch_event(event)
            
            # Prevent buffer overflow
            elif len(self._escape_buffer) > 16:
                self._escape_buffer = b''
                self._parsing_sequence = False
                
            return
        
        # Start of escape sequence
        if byte_val == 27:  # ESC
            self._parsing_sequence = True
            self._escape_buffer = bytes([byte_val])
            return
        
        # Special control characters
        if byte_val in (3, 4):  # Ctrl+C, Ctrl+D
            event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                data={'key': 'ctrl-c' if byte_val == 3 else 'ctrl-d'},
                timestamp=timestamp,
                raw_bytes=bytes([byte_val])
            )
            self._dispatch_event(event)
            return
        
        # Backspace/Delete
        if byte_val in (8, 127):
            event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                data={'key': 'backspace'},
                timestamp=timestamp,
                raw_bytes=bytes([byte_val])
            )
            self._dispatch_event(event)
            return
        
        # Enter/Return
        if byte_val == 13:
            event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                data={'key': 'enter'},
                timestamp=timestamp,
                raw_bytes=bytes([byte_val])
            )
            self._dispatch_event(event)
            return
        
        # Line feed (Ctrl+J)
        if byte_val == 10:
            event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                data={'key': 'ctrl-j', 'insert_newline': True},
                timestamp=timestamp,
                raw_bytes=bytes([byte_val])
            )
            self._dispatch_event(event)
            return
        
        # Tab
        if byte_val == 9:
            event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                data={'key': 'tab'},
                timestamp=timestamp,
                raw_bytes=bytes([byte_val])
            )
            self._dispatch_event(event)
            return
        
        # Regular printable character
        if 32 <= byte_val <= 126:
            try:
                char = chr(byte_val)
                event = InputEvent(
                    event_type=InputEventType.CHARACTER,
                    data={'character': char},
                    timestamp=timestamp,
                    raw_bytes=bytes([byte_val])
                )
                self._dispatch_event(event)
            except ValueError:
                # Invalid character
                pass
    
    def _is_sequence_complete(self, buffer: bytes) -> bool:
        """Check if escape sequence is complete."""
        if len(buffer) < 2:
            return False
        
        if buffer[1] == ord('['):  # CSI sequence
            # CSI sequences end with a letter or ~
            last_byte = buffer[-1]
            return (65 <= last_byte <= 90) or (97 <= last_byte <= 122) or last_byte == ord('~')
        
        # Other sequences (usually single character after ESC)
        return len(buffer) >= 2
    
    def _parse_escape_sequence(self, buffer: bytes, timestamp: float) -> Optional[InputEvent]:
        """Parse an escape sequence into an input event."""
        if len(buffer) < 2:
            return None
        
        if buffer[1] == ord('['):  # CSI sequence
            seq = buffer[2:].decode('utf-8', errors='ignore')
            
            if seq == 'A':
                return InputEvent(InputEventType.SPECIAL_KEY, {'key': 'up'}, timestamp, buffer)
            elif seq == 'B':
                return InputEvent(InputEventType.SPECIAL_KEY, {'key': 'down'}, timestamp, buffer)
            elif seq == 'C':
                return InputEvent(InputEventType.SPECIAL_KEY, {'key': 'right'}, timestamp, buffer)
            elif seq == 'D':
                return InputEvent(InputEventType.SPECIAL_KEY, {'key': 'left'}, timestamp, buffer)
            elif seq == '3~':
                return InputEvent(InputEventType.SPECIAL_KEY, {'key': 'delete'}, timestamp, buffer)
            elif seq == '5~':
                return InputEvent(InputEventType.SPECIAL_KEY, {'key': 'page_up'}, timestamp, buffer)
            elif seq == '6~':
                return InputEvent(InputEventType.SPECIAL_KEY, {'key': 'page_down'}, timestamp, buffer)
            elif seq.startswith('<') and (seq.endswith('M') or seq.endswith('m')):
                # Mouse event
                return self._parse_mouse_event(seq, timestamp, buffer)
        
        # Unrecognized sequence
        return None
    
    def _parse_mouse_event(self, seq: str, timestamp: float, buffer: bytes) -> Optional[InputEvent]:
        """Parse a mouse event sequence."""
        import re
        
        # SGR mouse format: <button;x;y[M|m]
        match = re.match(r'<(\d+);(\d+);(\d+)([Mm])', seq)
        if not match:
            return None
        
        button = int(match.group(1))
        x = int(match.group(2))
        y = int(match.group(3))
        pressed = match.group(4) == 'M'
        
        # Decode button
        if button == 64:
            button_name = 'wheel_up'
        elif button == 65:
            button_name = 'wheel_down'
        elif button & 3 == 0:
            button_name = 'left'
        elif button & 3 == 1:
            button_name = 'middle'
        elif button & 3 == 2:
            button_name = 'right'
        else:
            button_name = 'unknown'
        
        return InputEvent(
            event_type=InputEventType.MOUSE,
            data={
                'button': button_name,
                'x': x,
                'y': y,
                'pressed': pressed,
                'raw_button': button
            },
            timestamp=timestamp,
            raw_bytes=buffer
        )
    
    def _dispatch_event(self, event: InputEvent):
        """Dispatch an event to processors and handlers."""
        try:
            # Process with processors first (they can handle/consume events)
            for processor in self.processors:
                try:
                    # Schedule processor call in event loop if available
                    if self._loop and not self._loop.is_closed():
                        future = asyncio.run_coroutine_threadsafe(
                            processor.process_event(event),
                            self._loop
                        )
                        # Don't wait for result to avoid blocking
                    
                except Exception as e:
                    logger.debug(f"Error in processor: {e}")
            
            # Call event handlers
            handlers = self.event_handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    if self._loop and not self._loop.is_closed():
                        if asyncio.iscoroutinefunction(handler):
                            asyncio.run_coroutine_threadsafe(handler(event), self._loop)
                        else:
                            self._loop.call_soon_threadsafe(handler, event)
                    else:
                        handler(event)
                        
                except Exception as e:
                    logger.debug(f"Error in event handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error dispatching event: {e}")
    
    def get_input_buffer(self) -> str:
        """Get the current input buffer if echo processor is available."""
        if self.echo_processor:
            return self.echo_processor.get_buffer()
        return ""
    
    def clear_input_buffer(self):
        """Clear the input buffer if echo processor is available."""
        if self.echo_processor:
            self.echo_processor.clear_buffer()
    
    def set_input_buffer(self, text: str):
        """Set the input buffer content if echo processor is available."""
        if self.echo_processor:
            self.echo_processor.set_buffer(text)
    
    async def cleanup(self):
        """Clean up the input handler."""
        await self.stop()
        self.terminal_manager.cleanup()
        self.processors.clear()
        self.event_handlers.clear()
        
        logger.debug("Unified input handler cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Use synchronous cleanup
        try:
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self.cleanup())
        except RuntimeError:
            # No running loop, create one
            asyncio.run(self.cleanup())