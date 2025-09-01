"""
Chat Input Component - Real-time character display and input handling.

This is the CRITICAL component that fixes the typing visibility issue.
Users must see characters as they type, and input must be submitted properly.
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from ..event_system import AsyncEventSystem, Event, EventType
from ..input_handler import InputHandler, InputEvent, InputEventType
from ..keyboard_processor import KeyboardProcessor, KeySequence, ShortcutContext

logger = logging.getLogger(__name__)


class InputMode(Enum):
    """Input mode for the chat input."""
    SINGLE_LINE = "single_line"
    MULTI_LINE = "multi_line"
    COMMAND = "command"


@dataclass
class ChatInputState:
    """State of the chat input component."""
    text: str = ""
    cursor_position: int = 0
    mode: InputMode = InputMode.SINGLE_LINE
    multiline_buffer: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    history_position: int = -1
    show_cursor: bool = True
    command_prefix: str = "/"
    multiline_mode: bool = False
    # Paste detection state
    paste_buffer: str = ""
    paste_start_time: float = 0
    is_pasting: bool = False


@dataclass
class ChatInputEvent:
    """Event emitted by chat input component."""
    event_type: str  # 'submit', 'mode_change', 'text_change'
    text: str = ""
    mode: Optional[InputMode] = None
    data: Optional[Dict[str, Any]] = None


class ChatInput:
    """
    Chat input component with real-time character display.
    
    This component solves the critical typing visibility issue by:
    1. Showing characters immediately as they are typed
    2. Managing cursor position and display
    3. Supporting multi-line input with Shift+Enter
    4. Detecting and handling commands (starting with /)
    5. Integrating with keyboard shortcuts
    """
    
    def __init__(self, 
                 event_system: AsyncEventSystem,
                 input_handler: Optional[InputHandler] = None,
                 keyboard_processor: Optional[KeyboardProcessor] = None):
        """Initialize the chat input component."""
        self.event_system = event_system
        self.input_handler = input_handler or InputHandler()
        self.keyboard_processor = keyboard_processor
        
        # Component state
        self.state = ChatInputState()
        self._initialized = False
        self._active = False
        self._callbacks: Dict[str, Callable] = {}
        
        # Display settings
        self.prompt_text = "> "
        self.multiline_prompt = "... "
        self.max_history = 100
        import os as _os
        self.cursor_char = _os.getenv("AGENTS_TUI_V2_CARET_CHAR", "█") or "█"
        self.cursor_blink_interval = 0.5
        self._cursor_visible = True
        self._cursor_task: Optional[asyncio.Task] = None
        
        # Real-time echo settings
        # We render via ChatInterface/DisplayRenderer, not direct stdout
        self._echo_enabled = False
        self._immediate_display = False
        
        # Paste detection settings
        self._paste_threshold_ms = 50  # Time threshold to detect paste operations
        self._paste_timeout_ms = 200   # Time to wait after last character for paste completion
        self._paste_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> bool:
        """Initialize the chat input component."""
        if self._initialized:
            return True
        
        try:
            # Enable immediate character echo for real-time display
            if self.input_handler:
                try:
                    # Render via DisplayRenderer, do not echo to stdout
                    self.input_handler.set_echo(False)
                except Exception:
                    pass
                # Always set up handlers (works for both PTK and fallback raw modes)
                self._setup_input_handlers()
                logger.info("Chat input initialized with real-time character display")
            else:
                logger.warning("Input handler missing, using fallback mode")
                self._setup_fallback_input()
            
            # Setup keyboard shortcuts
            if self.keyboard_processor:
                await self._setup_keyboard_shortcuts()
            
            # Start cursor blinking
            await self._start_cursor_blinking()
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize chat input: {e}")
            return False
    
    def _setup_input_handlers(self):
        """Setup input event handlers for immediate character display."""
        if not self.input_handler:
            return
        
        # Handle regular character input
        self.input_handler.add_key_handler('*', self._handle_character_input)
        
        # Handle special keys
        self.input_handler.add_key_handler('enter', self._handle_enter_key)
        self.input_handler.add_key_handler('c-j', self._handle_ctrl_j)
        self.input_handler.add_key_handler('backspace', self._handle_backspace)
        self.input_handler.add_key_handler('left', self._handle_cursor_left)
        self.input_handler.add_key_handler('right', self._handle_cursor_right)
        self.input_handler.add_key_handler('home', self._handle_home)
        self.input_handler.add_key_handler('end', self._handle_end)
        self.input_handler.add_key_handler('up', self._handle_history_up)
        self.input_handler.add_key_handler('down', self._handle_history_down)
        self.input_handler.add_key_handler('c-c', self._handle_ctrl_c)
        self.input_handler.add_key_handler('c-d', self._handle_ctrl_d)
        self.input_handler.add_key_handler('c-u', self._handle_ctrl_u)
        self.input_handler.add_key_handler('c-k', self._handle_ctrl_k)
        
        # Handle Shift+Enter for multiline
        self.input_handler.add_key_handler('s-enter', self._handle_shift_enter)
    
    def _setup_fallback_input(self):
        """Setup fallback input handling when prompt_toolkit is not available."""
        logger.info("Setting up fallback input handling")
        # For fallback, we'll need to handle input differently
        # This is less ideal but ensures basic functionality
        self._echo_enabled = False  # Terminal will handle echo
        self._immediate_display = False
    
    async def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for chat input."""
        if not self.keyboard_processor:
            return
        
        # Ctrl+C for cancel/clear
        self.keyboard_processor.add_shortcut(
            KeySequence(['c'], {'ctrl'}),
            self._handle_cancel,
            ShortcutContext.INPUT,
            "Cancel input"
        )
        
        # Ctrl+U for clear line
        self.keyboard_processor.add_shortcut(
            KeySequence(['u'], {'ctrl'}),
            self._handle_clear_line,
            ShortcutContext.INPUT,
            "Clear line"
        )
        
        # Tab for completion (future feature)
        self.keyboard_processor.add_shortcut(
            KeySequence(['tab']),
            self._handle_tab_completion,
            ShortcutContext.INPUT,
            "Tab completion"
        )

        # F2 to toggle multiline mode
        self.keyboard_processor.add_shortcut(
            KeySequence(['f2']),
            self._toggle_multiline_mode,
            ShortcutContext.INPUT,
            "Toggle multiline mode"
        )
    
    async def _start_cursor_blinking(self):
        """Start the cursor blinking task."""
        if self._cursor_task:
            self._cursor_task.cancel()
        
        self._cursor_task = asyncio.create_task(self._cursor_blink_loop())
    
    async def _cursor_blink_loop(self):
        """Cursor blinking animation loop."""
        try:
            while True:
                if self._active:
                    self._cursor_visible = not self._cursor_visible
                    await self._emit_text_change_event()
                
                await asyncio.sleep(self.cursor_blink_interval)
        except asyncio.CancelledError:
            pass
    
    async def activate(self):
        """Activate the chat input for user interaction."""
        self._active = True
        self.state.show_cursor = True
        
        # Show initial prompt
        await self._display_current_line()
        
        logger.debug("Chat input activated")
    
    async def deactivate(self):
        """Deactivate the chat input."""
        self._active = False
        self.state.show_cursor = False
        
        if self._cursor_task:
            self._cursor_task.cancel()
            self._cursor_task = None
        
        if self._paste_task:
            self._paste_task.cancel()
            self._paste_task = None
        
        # Reset paste state
        self.state.is_pasting = False
        self.state.paste_buffer = ""
        self.state.paste_start_time = 0
        
        logger.debug("Chat input deactivated")
    
    def _handle_character_input(self, event: InputEvent):
        """Handle regular character input with immediate display and paste detection."""
        if not self._active or event.event_type != InputEventType.CHARACTER:
            return
        
        char = event.character
        if not char or len(char) != 1:
            return
        
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Check if this might be part of a paste operation
        if self._is_paste_event(current_time):
            # Handle as part of paste
            self._handle_paste_character(char, current_time)
        else:
            # Handle as normal typing
            self._handle_normal_character(char)
        
        logger.debug(f"Character input: '{char}', is_pasting: {self.state.is_pasting}")
    
    def _is_paste_event(self, current_time: float) -> bool:
        """Determine if current character is part of a paste operation."""
        if not self.state.is_pasting:
            # Not currently pasting, check if rapid input suggests paste
            if self.state.paste_start_time > 0:
                time_since_last = current_time - self.state.paste_start_time
                return time_since_last < self._paste_threshold_ms
            return False
        else:
            # Already pasting, continue if within timeout
            time_since_last = current_time - self.state.paste_start_time
            return time_since_last < self._paste_timeout_ms
    
    def _handle_paste_character(self, char: str, current_time: float):
        """Handle a character as part of a paste operation."""
        if not self.state.is_pasting:
            # Start new paste operation
            self.state.is_pasting = True
            self.state.paste_buffer = char
            logger.debug("Started paste operation")
        else:
            # Add to existing paste buffer
            self.state.paste_buffer += char
        
        self.state.paste_start_time = current_time
        
        # Cancel existing paste completion task
        if self._paste_task:
            self._paste_task.cancel()
        
        # Schedule paste completion check
        self._paste_task = asyncio.create_task(self._complete_paste_operation())
    
    def _handle_normal_character(self, char: str):
        """Handle a single character as normal typing."""
        # Insert character at cursor position
        text = self.state.text
        pos = self.state.cursor_position
        
        self.state.text = text[:pos] + char + text[pos:]
        self.state.cursor_position += 1
        
        # Update timestamp for paste detection
        self.state.paste_start_time = time.time() * 1000
        
        # Emit text change event (ChatInterface will render input line)
        asyncio.create_task(self._emit_text_change_event())
    
    async def _complete_paste_operation(self):
        """Complete the paste operation after timeout."""
        try:
            # Wait for paste timeout
            await asyncio.sleep(self._paste_timeout_ms / 1000.0)
            
            if self.state.is_pasting and self.state.paste_buffer:
                # Insert the entire paste buffer at cursor position
                text = self.state.text
                pos = self.state.cursor_position
                
                self.state.text = text[:pos] + self.state.paste_buffer + text[pos:]
                self.state.cursor_position += len(self.state.paste_buffer)
                
                # Reset paste state
                self.state.paste_buffer = ""
                self.state.is_pasting = False
                self.state.paste_start_time = 0
                
                # Emit text change event for the complete paste
                await self._emit_text_change_event()
                
                logger.info(f"Completed paste operation: {len(self.state.paste_buffer)} characters")
                
        except asyncio.CancelledError:
            # Task was cancelled, likely due to more input
            pass
    
    def _handle_enter_key(self, event: InputEvent):
        """Handle Enter key - either submit or insert newline in multiline mode."""
        if not self._active:
            return
        
        # If we're currently pasting, treat enter as part of paste content
        if self.state.is_pasting:
            self._handle_paste_character('\n', time.time() * 1000)
            return
        
        if self.state.multiline_mode:
            # Insert newline
            t = self.state.text
            pos = self.state.cursor_position
            self.state.text = t[:pos] + "\n" + t[pos:]
            self.state.cursor_position = pos + 1
            asyncio.create_task(self._emit_text_change_event())
        else:
            text = self.state.text.strip()
            if text:
                asyncio.create_task(self._submit_input(text))
    
    def _handle_shift_enter(self, event: InputEvent):
        """Handle Shift+Enter by inserting a newline at cursor."""
        if not self._active:
            return
        t = self.state.text
        pos = self.state.cursor_position
        self.state.text = t[:pos] + "\n" + t[pos:]
        self.state.cursor_position = pos + 1
        asyncio.create_task(self._emit_text_change_event())

    def _handle_ctrl_j(self, event: InputEvent):
        """Ctrl+J: in multiline mode submit, otherwise insert newline as fallback."""
        if not self._active:
            return
        if self.state.multiline_mode:
            text = self.state.text.strip()
            if text:
                asyncio.create_task(self._submit_input(text))
        else:
            # Insert newline when not in multiline mode as a fallback
            t = self.state.text
            pos = self.state.cursor_position
            self.state.text = t[:pos] + "\n" + t[pos:]
            self.state.cursor_position = pos + 1
            asyncio.create_task(self._emit_text_change_event())
    
    def _handle_backspace(self, event: InputEvent):
        """Handle backspace key."""
        if not self._active or self.state.cursor_position == 0:
            return
        
        # Remove character before cursor
        text = self.state.text
        pos = self.state.cursor_position
        
        self.state.text = text[:pos-1] + text[pos:]
        self.state.cursor_position -= 1
        
        # Update display
        asyncio.create_task(self._display_current_line())
        asyncio.create_task(self._emit_text_change_event())
    
    def _handle_cursor_left(self, event: InputEvent):
        """Handle left arrow key."""
        if not self._active or self.state.cursor_position == 0:
            return
        
        self.state.cursor_position -= 1
        asyncio.create_task(self._display_current_line())
    
    def _handle_cursor_right(self, event: InputEvent):
        """Handle right arrow key."""
        if not self._active or self.state.cursor_position >= len(self.state.text):
            return
        
        self.state.cursor_position += 1
        asyncio.create_task(self._display_current_line())

    def _handle_home(self, event: InputEvent):
        """Move cursor to start of line."""
        if not self._active:
            return
        # Move to beginning
        # If multiline, move to beginning of entire buffer for simplicity
        self.state.cursor_position = 0
        asyncio.create_task(self._display_current_line())

    def _handle_end(self, event: InputEvent):
        """Move cursor to end of line."""
        if not self._active:
            return
        self.state.cursor_position = len(self.state.text)
        asyncio.create_task(self._display_current_line())
    
    def _handle_history_up(self, event: InputEvent):
        """Handle up arrow - navigate command history."""
        if not self._active or not self.state.history:
            return
        
        if self.state.history_position < len(self.state.history) - 1:
            self.state.history_position += 1
            self.state.text = self.state.history[-(self.state.history_position + 1)]
            self.state.cursor_position = len(self.state.text)
            asyncio.create_task(self._display_current_line())
    
    def _handle_history_down(self, event: InputEvent):
        """Handle down arrow - navigate command history."""
        if not self._active:
            return
        
        if self.state.history_position > 0:
            self.state.history_position -= 1
            self.state.text = self.state.history[-(self.state.history_position + 1)]
            self.state.cursor_position = len(self.state.text)
            asyncio.create_task(self._display_current_line())
        elif self.state.history_position == 0:
            self.state.history_position = -1
            self.state.text = ""
            self.state.cursor_position = 0
            asyncio.create_task(self._display_current_line())

    def _handle_ctrl_u(self, event: InputEvent):
        """Clear from start to cursor."""
        if not self._active:
            return
        t = self.state.text
        pos = self.state.cursor_position
        self.state.text = t[pos:]
        self.state.cursor_position = 0
        asyncio.create_task(self._display_current_line())
        asyncio.create_task(self._emit_text_change_event())

    def _handle_ctrl_k(self, event: InputEvent):
        """Clear from cursor to end."""
        if not self._active:
            return
        t = self.state.text
        pos = self.state.cursor_position
        self.state.text = t[:pos]
        asyncio.create_task(self._display_current_line())
        asyncio.create_task(self._emit_text_change_event())
    
    def _handle_ctrl_c(self, event: InputEvent):
        """Handle Ctrl+C - cancel current input."""
        asyncio.create_task(self._handle_cancel(None))
    
    def _handle_ctrl_d(self, event: InputEvent):
        """Handle Ctrl+D - EOF or quit."""
        if not self.state.text:
            # Empty line - trigger quit
            asyncio.create_task(self._submit_input("/quit"))
        else:
            # Clear current line
            asyncio.create_task(self._handle_clear_line(None))
    
    async def _handle_cancel(self, event: Optional[Event]) -> bool:
        """Handle cancel operation."""
        self.state.text = ""
        self.state.cursor_position = 0
        self.state.multiline_buffer.clear()
        self.state.mode = InputMode.SINGLE_LINE
        self.state.history_position = -1
        
        await self._display_current_line()
        await self._emit_text_change_event()
        
        return True
    
    async def _handle_clear_line(self, event: Optional[Event]) -> bool:
        """Handle clear line operation."""
        self.state.text = ""
        self.state.cursor_position = 0
        
        await self._display_current_line()
        await self._emit_text_change_event()
        
        return True
    
    async def _handle_tab_completion(self, event: Optional[Event]) -> bool:
        """Handle tab completion (future feature)."""
        # TODO: Implement command/text completion
        return True

    async def _toggle_multiline_mode(self, event: Optional[Event]) -> bool:
        """Toggle multiline mode."""
        self.state.multiline_mode = not self.state.multiline_mode
        await self._emit_mode_change_event()
        return True
    
    async def _submit_input(self, text: str):
        """Submit user input."""
        # Add to history if not empty and not duplicate
        if text and (not self.state.history or self.state.history[-1] != text):
            self.state.history.append(text)
            if len(self.state.history) > self.max_history:
                self.state.history = self.state.history[-self.max_history:]
        
        # Reset state
        full_text = text
        if self.state.multiline_buffer:
            full_text = "\n".join(self.state.multiline_buffer + [text])
        
        self.state.text = ""
        self.state.cursor_position = 0
        self.state.multiline_buffer.clear()
        self.state.mode = InputMode.SINGLE_LINE
        self.state.history_position = -1
        
        # Detect command mode
        is_command = full_text.startswith(self.state.command_prefix)
        
        # Emit submit event
        await self._emit_submit_event(full_text, is_command)
        
        # Update display via renderer through text_change callback
        await self._emit_text_change_event()
        
        logger.info(f"Input submitted: '{full_text}' (command: {is_command})")
    
    async def _display_current_line(self):
        """Display the current input line with cursor."""
        if not self._active:
            return
        
        # Build display text
        prompt = self.multiline_prompt if self.state.mode == InputMode.MULTI_LINE else self.prompt_text
        text = self.state.text
        cursor_pos = self.state.cursor_position
        
        # Insert cursor if visible and active
        display_text = text
        if self.state.show_cursor and self._cursor_visible:
            display_text = text[:cursor_pos] + self.cursor_char + text[cursor_pos:]
        
        # Full line to display
        line = prompt + display_text
        
        # Rendering is handled by ChatInterface updating the DisplayRenderer region.
        # This method intentionally performs no direct terminal output.
    
    async def _emit_submit_event(self, text: str, is_command: bool):
        """Emit input submit event."""
        event_data = ChatInputEvent(
            event_type="submit",
            text=text,
            mode=self.state.mode,
            data={"is_command": is_command}
        )
        
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "chat_input",
                "action": "submit",
                "text": text,
                "is_command": is_command,
                "event_data": event_data
            }
        )
        
        await self.event_system.emit_event(event)
        
        # Call registered callback if available
        if "submit" in self._callbacks:
            await self._callbacks["submit"](event_data)
    
    async def _emit_text_change_event(self):
        """Emit text change event."""
        event_data = ChatInputEvent(
            event_type="text_change",
            text=self.state.text,
            mode=self.state.mode
        )
        
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "chat_input",
                "action": "text_change",
                "text": self.state.text,
                "cursor_position": self.state.cursor_position,
                "event_data": event_data
            }
        )
        
        await self.event_system.emit_event(event)
        
        # Call registered callback if available
        if "text_change" in self._callbacks:
            await self._callbacks["text_change"](event_data)
    
    async def _emit_mode_change_event(self):
        """Emit mode change event."""
        event_data = ChatInputEvent(
            event_type="mode_change",
            mode=self.state.mode
        )
        
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "chat_input",
                "action": "mode_change",
                "mode": self.state.mode.value,
                "event_data": event_data
            }
        )
        
        await self.event_system.emit_event(event)
        
        # Call registered callback if available
        if "mode_change" in self._callbacks:
            await self._callbacks["mode_change"](event_data)
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for specific event types."""
        self._callbacks[event_type] = callback
    
    def remove_callback(self, event_type: str):
        """Remove callback for specific event types."""
        if event_type in self._callbacks:
            del self._callbacks[event_type]
    
    def get_current_text(self) -> str:
        """Get the current input text."""
        if self.state.multiline_buffer:
            return "\n".join(self.state.multiline_buffer + [self.state.text])
        return self.state.text
    
    def set_text(self, text: str):
        """Set the current input text."""
        self.state.text = text
        self.state.cursor_position = len(text)
        if self._active:
            asyncio.create_task(self._display_current_line())
            asyncio.create_task(self._emit_text_change_event())
    
    def clear(self):
        """Clear the current input."""
        if self._active:
            asyncio.create_task(self._handle_clear_line(None))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chat input statistics."""
        return {
            "active": self._active,
            "initialized": self._initialized,
            "text_length": len(self.state.text),
            "cursor_position": self.state.cursor_position,
            "mode": self.state.mode.value,
            "history_length": len(self.state.history),
            "multiline_lines": len(self.state.multiline_buffer),
            "echo_enabled": self._echo_enabled,
            "immediate_display": self._immediate_display
        }
    
    async def cleanup(self):
        """Cleanup the chat input component."""
        await self.deactivate()
        
        if self._paste_task:
            self._paste_task.cancel()
            self._paste_task = None
        
        if self.input_handler:
            # Remove all handlers
            for key in ['*', 'enter', 'backspace', 'left', 'right', 'up', 'down', 'c-c', 'c-d', 's-enter']:
                self.input_handler.remove_key_handler(key)
        
        self._callbacks.clear()
        self._initialized = False
        
        logger.debug("Chat input component cleaned up")


# Utility function for easy instantiation
def create_chat_input(event_system: AsyncEventSystem, 
                     input_handler: Optional[InputHandler] = None,
                     keyboard_processor: Optional[KeyboardProcessor] = None) -> ChatInput:
    """Create and return a new ChatInput instance."""
    return ChatInput(event_system, input_handler, keyboard_processor)
