"""
Robust keyboard input handler using prompt_toolkit for reliable key detection.

This module provides immediate character echo and reliable input handling 
that works in real TTY environments where the current system fails.
"""

import asyncio
import sys
from typing import Optional, Callable, Any, Union
from enum import Enum
from dataclasses import dataclass

try:
    from prompt_toolkit import Application
    from prompt_toolkit.application import get_app
    from prompt_toolkit.input import create_input
    from prompt_toolkit.output import create_output
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.filters import Condition
    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    _HAS_PROMPT_TOOLKIT = False


class InputEventType(Enum):
    """Types of input events that can be handled."""
    KEY_PRESS = "key_press"
    SPECIAL_KEY = "special_key"
    CHARACTER = "character"
    CTRL_KEY = "ctrl_key"


@dataclass
class InputEvent:
    """Represents an input event from the keyboard."""
    event_type: InputEventType
    key: Optional[str] = None
    character: Optional[str] = None
    ctrl: bool = False
    alt: bool = False
    shift: bool = False
    data: Optional[dict] = None


class InputHandler:
    """
    Robust keyboard input handler using prompt_toolkit.
    
    Provides immediate character echo and reliable key detection
    that works in real TTY environments.
    """
    
    def __init__(self):
        """Initialize the input handler."""
        self.available = _HAS_PROMPT_TOOLKIT
        self._app: Optional[Application] = None
        self._input = None
        self._output = None
        self._callbacks = {}
        self._running = False
        self._echo_enabled = True
        
        if self.available:
            self._initialize_prompt_toolkit()
    
    def _initialize_prompt_toolkit(self):
        """Initialize prompt_toolkit components."""
        try:
            self._input = create_input()
            self._output = create_output()
        except Exception as e:
            # Fallback if prompt_toolkit initialization fails
            self.available = False
            
    def set_echo(self, enabled: bool):
        """Enable or disable character echo."""
        self._echo_enabled = enabled
    
    def add_key_handler(self, key: str, callback: Callable[[InputEvent], Any]):
        """
        Add a key handler for a specific key or key combination.
        
        Args:
            key: Key identifier (e.g., 'c-c', 'enter', 'a', 'escape')
            callback: Function to call when key is pressed
        """
        if not self.available:
            return
            
        self._callbacks[key] = callback
    
    def remove_key_handler(self, key: str):
        """Remove a key handler."""
        if key in self._callbacks:
            del self._callbacks[key]
    
    def create_key_bindings(self) -> Optional[KeyBindings]:
        """Create key bindings for the application."""
        if not self.available:
            return None
            
        kb = KeyBindings()
        
        # Handle Ctrl+C
        @kb.add(Keys.ControlC)
        def _(event):
            """Handle Ctrl+C gracefully."""
            input_event = InputEvent(
                event_type=InputEventType.CTRL_KEY,
                key="c-c",
                ctrl=True
            )
            if "c-c" in self._callbacks:
                self._callbacks["c-c"](input_event)
            else:
                # Default: exit application
                event.app.exit()
        
        # Handle Enter
        @kb.add(Keys.ControlM)  # Enter key
        def _(event):
            input_event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                key="enter"
            )
            if "enter" in self._callbacks:
                self._callbacks["enter"](input_event)
        
        # Handle Escape
        @kb.add(Keys.Escape)
        def _(event):
            input_event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                key="escape"
            )
            if "escape" in self._callbacks:
                self._callbacks["escape"](input_event)
        
        # Handle arrow keys
        @kb.add(Keys.Up)
        def _(event):
            input_event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                key="up"
            )
            if "up" in self._callbacks:
                self._callbacks["up"](input_event)
        
        @kb.add(Keys.Down)
        def _(event):
            input_event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                key="down"
            )
            if "down" in self._callbacks:
                self._callbacks["down"](input_event)
        
        @kb.add(Keys.Left)
        def _(event):
            input_event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                key="left"
            )
            if "left" in self._callbacks:
                self._callbacks["left"](input_event)
        
        @kb.add(Keys.Right)
        def _(event):
            input_event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                key="right"
            )
            if "right" in self._callbacks:
                self._callbacks["right"](input_event)
        
        # Handle backspace
        @kb.add(Keys.Backspace)
        def _(event):
            input_event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                key="backspace"
            )
            if "backspace" in self._callbacks:
                self._callbacks["backspace"](input_event)
        
        # Handle regular characters
        @kb.add(Keys.Any)
        def _(event):
            """Handle any other key press."""
            char = event.data
            
            # Create appropriate input event
            if len(char) == 1 and ord(char) < 32:  # Control character
                input_event = InputEvent(
                    event_type=InputEventType.CTRL_KEY,
                    key=f"c-{chr(ord(char) + ord('a') - 1)}" if ord(char) > 0 else "c-@",
                    character=char,
                    ctrl=True
                )
                key_id = input_event.key
            else:
                input_event = InputEvent(
                    event_type=InputEventType.CHARACTER,
                    character=char
                )
                key_id = char
            
            # Echo character if enabled
            if self._echo_enabled and input_event.event_type == InputEventType.CHARACTER:
                sys.stdout.write(char)
                sys.stdout.flush()
            
            # Call handler if registered
            if key_id in self._callbacks:
                self._callbacks[key_id](input_event)
        
        return kb
    
    def create_simple_app(self, layout_content: Optional[str] = None) -> Optional[Application]:
        """
        Create a simple prompt_toolkit application for input handling.
        
        Args:
            layout_content: Optional content to display
            
        Returns:
            Application instance or None if not available
        """
        if not self.available:
            return None
        
        kb = self.create_key_bindings()
        if not kb:
            return None
        
        # Simple layout - just a window that shows content
        if layout_content:
            control = FormattedTextControl(
                text=FormattedText([("", layout_content)])
            )
        else:
            control = FormattedTextControl(text="")
        
        layout = Layout(Window(control))
        
        app = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=False,
            input=self._input,
            output=self._output,
        )
        
        return app
    
    async def run_async(self, app: Optional[Application] = None):
        """
        Run the input handler asynchronously.
        
        Args:
            app: Optional Application instance to run
        """
        if not self.available:
            return
            
        if app is None:
            app = self.create_simple_app()
            
        if app is None:
            return
        
        self._app = app
        self._running = True
        
        try:
            await app.run_async()
        finally:
            self._running = False
            self._app = None
    
    def stop(self):
        """Stop the input handler."""
        if self._app and self._running:
            self._app.exit()
    
    def get_single_key(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Get a single keypress (synchronous fallback).
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Key character or None if timeout/error
        """
        if not self.available:
            # Fallback to standard input
            try:
                if timeout:
                    import select
                    ready, _, _ = select.select([sys.stdin], [], [], timeout)
                    if not ready:
                        return None
                
                return sys.stdin.read(1)
            except:
                return None
        
        # For prompt_toolkit, we'd need to create a temporary app
        # This is a simplified fallback
        try:
            char = self._input.read()
            return char if char else None
        except:
            return None
    
    def is_available(self) -> bool:
        """Check if the input handler is available and functional."""
        return self.available
    
    def get_capabilities(self) -> dict:
        """Get input handler capabilities."""
        return {
            "immediate_echo": self.available,
            "key_detection": self.available,
            "async_support": self.available,
            "prompt_toolkit": self.available,
            "fallback_available": True
        }


# Convenience function for simple use cases
def create_input_handler() -> InputHandler:
    """Create and return a new InputHandler instance."""
    return InputHandler()