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
    
    def __init__(self, terminal_manager=None, event_system=None):
        """Initialize the input handler.
        
        Args:
            terminal_manager: Terminal manager instance (optional)
            event_system: Event system instance (optional)
        """
        self.available = _HAS_PROMPT_TOOLKIT
        self._app: Optional[Application] = None
        self._input = None
        self._output = None
        self._callbacks = {}
        self._running = False
        self._echo_enabled = True
        self._initialized = False
        
        # Store references for integration
        self.terminal_manager = terminal_manager
        self.event_system = event_system
        
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
    
    async def initialize(self) -> bool:
        """
        Initialize the input handler asynchronously.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if self._initialized:
                return True
            
            # Perform any async initialization here
            if self.available:
                self._initialized = True
                return True
            else:
                # Fallback mode - still consider it successful
                self._initialized = True
                return True
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to initialize input handler: {e}")
            return False
    
    async def cleanup(self):
        """
        Cleanup the input handler.
        
        Stops any running input processing and cleans up resources.
        """
        try:
            self._running = False
            
            if self._app:
                self._app.exit()
                self._app = None
            
            self._callbacks.clear()
            self._initialized = False
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error during input handler cleanup: {e}")
            
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

        # Ctrl+J: dedicated mapping (used for submit in multiline mode)
        @kb.add(Keys.ControlJ)
        def _(event):
            input_event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                key="c-j"
            )
            if "c-j" in self._callbacks:
                self._callbacks["c-j"](input_event)
        
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

        # Page Up / Page Down
        @kb.add(Keys.PageUp)
        def _(event):
            input_event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                key="page_up"
            )
            if "page_up" in self._callbacks:
                self._callbacks["page_up"](input_event)

        @kb.add(Keys.PageDown)
        def _(event):
            input_event = InputEvent(
                event_type=InputEventType.SPECIAL_KEY,
                key="page_down"
            )
            if "page_down" in self._callbacks:
                self._callbacks["page_down"](input_event)
        
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
            
            # Do not echo directly; rendering is managed by the UI
            
            import inspect, asyncio as _asyncio
            # Helper to invoke callback (sync or async)
            def _invoke(cb):
                try:
                    result = cb(input_event)
                    if inspect.iscoroutine(result):
                        try:
                            loop = _asyncio.get_running_loop()
                            loop.create_task(result)
                        except RuntimeError:
                            # No running loop; best-effort run
                            _asyncio.run(result)
                except Exception:
                    # Swallow to avoid breaking input loop
                    pass

            # Call handler if registered (specific)
            if key_id in self._callbacks:
                _invoke(self._callbacks[key_id])
            # Wildcard handler for any character
            if '*' in self._callbacks:
                _invoke(self._callbacks['*'])
        
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
            
        if app is not None:
            # Run provided PTK application (may manage its own output)
            self._app = app
            self._running = True
            try:
                await app.run_async()
            finally:
                self._running = False
                self._app = None
            return

        # Lightweight key reader that doesn't render UI
        try:
            from prompt_toolkit.keys import Keys  # type: ignore
        except Exception:
            return

        loop = asyncio.get_running_loop()
        self._running = True

        def reader():
            try:
                with create_input() as inp:
                    for kp in inp.read_keys():
                        if not self._running:
                            break
                        k = getattr(kp, 'key', None)
                        data = getattr(kp, 'data', None)
                        # Map to InputEvent
                        evt = None
                        if k == Keys.ControlC:
                            evt = InputEvent(InputEventType.CTRL_KEY, key='c-c', ctrl=True)
                        elif k == Keys.ControlD:
                            evt = InputEvent(InputEventType.CTRL_KEY, key='c-d', ctrl=True)
                        elif k == Keys.Backspace:
                            evt = InputEvent(InputEventType.SPECIAL_KEY, key='backspace')
                        elif k == Keys.ControlJ:
                            # Ctrl+J is handled explicitly by UI (e.g., submit in multiline mode)
                            evt = InputEvent(InputEventType.SPECIAL_KEY, key='c-j')
                        elif k == Keys.Enter or k == Keys.ControlM:
                            evt = InputEvent(InputEventType.SPECIAL_KEY, key='enter')
                        elif k == Keys.Home:
                            evt = InputEvent(InputEventType.SPECIAL_KEY, key='home')
                        elif k == Keys.End:
                            evt = InputEvent(InputEventType.SPECIAL_KEY, key='end')
                        elif k == Keys.ControlU:
                            evt = InputEvent(InputEventType.CTRL_KEY, key='c-u', ctrl=True)
                        elif k == Keys.ControlK:
                            evt = InputEvent(InputEventType.CTRL_KEY, key='c-k', ctrl=True)
                        elif k == Keys.Escape:
                            evt = InputEvent(InputEventType.SPECIAL_KEY, key='escape')
                        elif k == Keys.Up:
                            evt = InputEvent(InputEventType.SPECIAL_KEY, key='up')
                        elif k == Keys.Down:
                            evt = InputEvent(InputEventType.SPECIAL_KEY, key='down')
                        elif k == Keys.Left:
                            evt = InputEvent(InputEventType.SPECIAL_KEY, key='left')
                        elif k == Keys.Right:
                            evt = InputEvent(InputEventType.SPECIAL_KEY, key='right')
                        elif k == Keys.PageUp:
                            evt = InputEvent(InputEventType.SPECIAL_KEY, key='page_up')
                        elif k == Keys.PageDown:
                            evt = InputEvent(InputEventType.SPECIAL_KEY, key='page_down')
                        else:
                            ch = data or (k if isinstance(k, str) else None)
                            if ch and len(ch) == 1 and ch.isprintable():
                                evt = InputEvent(InputEventType.CHARACTER, character=ch)
                            # Map raw newlines conservatively to Enter for reliability
                            elif ch == '\n' or ch == '\r':
                                evt = InputEvent(InputEventType.SPECIAL_KEY, key='enter')
                        if evt is None:
                            continue
                        # Dispatch to callbacks on the asyncio loop
                        def dispatch():
                            # Specific
                            cb = self._callbacks.get(evt.key or evt.character)
                            if cb:
                                try:
                                    res = cb(evt)
                                    if asyncio.iscoroutine(res):
                                        loop.create_task(res)
                                except Exception:
                                    pass
                            # Wildcard
                            cb2 = self._callbacks.get('*')
                            if cb2:
                                try:
                                    res = cb2(evt)
                                    if asyncio.iscoroutine(res):
                                        loop.create_task(res)
                                except Exception:
                                    pass
                        loop.call_soon_threadsafe(dispatch)
            except Exception:
                pass

        import threading
        t = threading.Thread(target=reader, name="InputReader", daemon=True)
        t.start()
        try:
            while self._running:
                await asyncio.sleep(0.05)
        finally:
            self._running = False
    
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
