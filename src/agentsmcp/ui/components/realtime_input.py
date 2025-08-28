"""
Real-Time Input Field Component

A Rich-based input component that shows typing as it happens, with cursor positioning
and real-time visual feedback for AgentsMCP's TUI interface.
"""

from __future__ import annotations

import asyncio
import re
import signal
import sys
from typing import Any, Awaitable, Callable, List, Optional, Union

# Rich imports with graceful fallback
try:
    from rich.console import Console, RenderableType
    from rich.panel import Panel
    from rich.text import Text
    from rich.style import Style
except ImportError:  # pragma: no cover
    Console = None
    Panel = None
    Text = None
    Style = None
    RenderableType = Any


class RealTimeInputField:
    """Real-time input field that shows typing as it happens.
    
    Features:
    - Character-by-character input handling
    - Visual cursor with blinking effect
    - Multi-line support with proper wrapping
    - Input sanitization (ANSI escape sequences)
    - Configurable dimensions and styling
    
    Args:
        console: Rich Console instance for rendering
        max_width: Maximum width of input panel
        max_height: Maximum height of input panel (lines)
        initial_text: Starting content
        prompt: Text shown before input area
        show_cursor: Whether to show blinking cursor
        cursor_style: Rich style for cursor display
    """
    
    def __init__(
        self,
        console: Optional[Console] = None,
        *,
        max_width: int = 80,
        max_height: int = 5,
        initial_text: str = "",
        prompt: str = "",
        show_cursor: bool = True,
        cursor_style: str = "reverse bold"
    ) -> None:
        self.console = console or (Console() if Console else None)
        self.max_width = max_width
        self.max_height = max_height
        self.prompt = prompt
        self.show_cursor = show_cursor
        self.cursor_style = cursor_style
        
        # Input buffer state - FIXED: Initialize properly
        self._lines: List[str] = initial_text.split('\n') if initial_text else [""]
        self._cursor_row = len(self._lines) - 1
        self._cursor_col = len(self._lines[-1])
        self._cursor_visible = True
        self._initialized = False  # Track initialization state
        
        # Event callbacks
        self._submit_callbacks: List[Callable[[str], Union[None, Awaitable[None]]]] = []
        self._change_callbacks: List[Callable[[str], Union[None, Awaitable[None]]]] = []
        
        # Runtime state
        self._running = False
        self._cursor_task: Optional[asyncio.Task] = None
        
        # Input sanitization
        self._ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        
        # FIXED: Initialize properly from start
        self._ensure_initialized()
        
    def on_submit(self, callback: Callable[[str], Union[None, Awaitable[None]]]) -> None:
        """Register callback for when user submits input (Enter key)."""
        self._submit_callbacks.append(callback)
        
    def on_change(self, callback: Callable[[str], Union[None, Awaitable[None]]]) -> None:
        """Register callback for when input content changes."""
        self._change_callbacks.append(callback)
        
    def get_current_input(self) -> str:
        """Get current input content as string."""
        return '\n'.join(self._lines)
        
    def clear_input(self) -> None:
        """Clear input buffer and reset cursor."""
        self._lines = [""]
        self._cursor_row = 0
        self._cursor_col = 0
        # Try to trigger change, but don't fail if no event loop
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._trigger_change())
        except RuntimeError:
            # No event loop running, trigger synchronously
            self._trigger_change_sync()
        
    def set_input(self, text: str) -> None:
        """Set input content programmatically."""
        self._lines = text.split('\n') if text else [""]
        self._cursor_row = len(self._lines) - 1
        self._cursor_col = len(self._lines[-1])
        # Try to trigger change, but don't fail if no event loop
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._trigger_change())
        except RuntimeError:
            # No event loop running, trigger synchronously
            self._trigger_change_sync()
        
    def render(self) -> RenderableType:
        """Render the input field as a Rich renderable."""
        if not Text or not Panel:
            return f"{self.prompt}{self.get_current_input()}"
        
        # FIXED: Ensure we're properly initialized before rendering
        self._ensure_initialized()
            
        # Build display text with cursor
        display_lines = []
        
        for i, line in enumerate(self._lines):
            text_line = Text()
            
            if i == self._cursor_row:
                # Insert cursor in current line
                before_cursor = line[:self._cursor_col]
                after_cursor = line[self._cursor_col:]
                
                text_line.append(before_cursor)
                
                if self.show_cursor and self._cursor_visible:
                    # Show cursor as highlighted space or character
                    cursor_char = after_cursor[0] if after_cursor else " "
                    text_line.append(cursor_char, style=self.cursor_style)
                    text_line.append(after_cursor[1:] if len(after_cursor) > 1 else "")
                else:
                    text_line.append(after_cursor)
            else:
                text_line.append(line)
                
            display_lines.append(text_line)
        
        # Combine lines
        if len(display_lines) == 1:
            content = display_lines[0]
        else:
            content = Text()
            for i, line in enumerate(display_lines):
                if i > 0:
                    content.append('\n')
                content.append_text(line)
        
        # Add prompt if specified - FIXED: Ensure prompt is properly shown
        if self.prompt:
            prompt_text = Text(self.prompt, style="bold cyan")
            full_content = Text()
            full_content.append_text(prompt_text)
            # FIXED: Add current input content after prompt
            current_input = self.get_current_input()
            if current_input or self.show_cursor:
                full_content.append_text(content)
            content = full_content
            
        return Panel(
            content,
            width=min(self.max_width, 80) if self.max_width else None,
            height=min(self.max_height, len(self._lines) + 2),
            title="Input" if not self.prompt else None,
            border_style="blue"
        )
        
    async def handle_key(self, key: str) -> bool:
        """Handle a key press. Returns True if handled, False otherwise."""
        if not key:
            return False
            
        # FIXED: Ensure field is initialized before handling input
        self._ensure_initialized()
        
        # FIXED: Handle slash commands properly - ensure "/" is always processed
        if key == "/":
            # Always insert "/" character for command mode
            self._insert_char(key)
            await self._trigger_change()
            return True
            
        # Handle special keys
        if key == "enter":
            await self._handle_submit()
            return True
        elif key == "backspace":
            self._handle_backspace()
            await self._trigger_change()
            return True
        elif key == "delete":
            self._handle_delete()
            await self._trigger_change()
            return True
        elif key == "left":
            self._move_cursor_left()
            return True
        elif key == "right":
            self._move_cursor_right()
            return True
        elif key == "up":
            self._move_cursor_up()
            return True
        elif key == "down":
            self._move_cursor_down()
            return True
        elif key == "home":
            self._cursor_col = 0
            return True
        elif key == "end":
            self._cursor_col = len(self._lines[self._cursor_row])
            return True
        elif len(key) == 1 and key.isprintable():
            # FIXED: Regular character input with immediate echo
            self._insert_char(key)
            await self._trigger_change()
            return True
            
        return False
        
    def sanitize_input(self, text: str) -> str:
        """Remove ANSI escape sequences and clean input."""
        if not text:
            return ""
            
        # Remove ANSI escape sequences
        cleaned = self._ansi_escape.sub('', text)
        
        # Handle different line endings
        cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
        
        return cleaned
        
    async def start_cursor_blink(self) -> None:
        """Start cursor blinking animation."""
        if self._cursor_task:
            self._cursor_task.cancel()
            
        async def blink():
            while self._running:
                await asyncio.sleep(0.5)  # Faster blink interval for responsiveness
                self._cursor_visible = not self._cursor_visible
                
        self._cursor_task = asyncio.create_task(blink())
        
    def stop_cursor_blink(self) -> None:
        """Stop cursor blinking animation."""
        if self._cursor_task:
            self._cursor_task.cancel()
            self._cursor_task = None
        self._cursor_visible = True
        
    # Private methods
    def _insert_char(self, char: str) -> None:
        """Insert character at cursor position."""
        # FIXED: Ensure we're initialized before inserting
        self._ensure_initialized()
        
        current_line = self._lines[self._cursor_row]
        new_line = current_line[:self._cursor_col] + char + current_line[self._cursor_col:]
        self._lines[self._cursor_row] = new_line
        self._cursor_col += 1
        
        # FIXED: Ensure cursor stays visible during typing and force immediate display
        self._cursor_visible = True
        
    def _handle_backspace(self) -> None:
        """Handle backspace key."""
        if self._cursor_col > 0:
            # Delete character before cursor in current line
            current_line = self._lines[self._cursor_row]
            new_line = current_line[:self._cursor_col-1] + current_line[self._cursor_col:]
            self._lines[self._cursor_row] = new_line
            self._cursor_col -= 1
        elif self._cursor_row > 0:
            # Merge with previous line
            current_line = self._lines.pop(self._cursor_row)
            self._cursor_row -= 1
            self._cursor_col = len(self._lines[self._cursor_row])
            self._lines[self._cursor_row] += current_line
            
        # Ensure cursor stays visible during editing
        self._cursor_visible = True
            
    def _handle_delete(self) -> None:
        """Handle delete key."""
        current_line = self._lines[self._cursor_row]
        if self._cursor_col < len(current_line):
            # Delete character at cursor
            new_line = current_line[:self._cursor_col] + current_line[self._cursor_col+1:]
            self._lines[self._cursor_row] = new_line
        elif self._cursor_row < len(self._lines) - 1:
            # Merge with next line
            next_line = self._lines.pop(self._cursor_row + 1)
            self._lines[self._cursor_row] += next_line
            
    def _move_cursor_left(self) -> None:
        """Move cursor left."""
        if self._cursor_col > 0:
            self._cursor_col -= 1
        elif self._cursor_row > 0:
            self._cursor_row -= 1
            self._cursor_col = len(self._lines[self._cursor_row])
            
    def _move_cursor_right(self) -> None:
        """Move cursor right."""
        current_line = self._lines[self._cursor_row]
        if self._cursor_col < len(current_line):
            self._cursor_col += 1
        elif self._cursor_row < len(self._lines) - 1:
            self._cursor_row += 1
            self._cursor_col = 0
            
    def _move_cursor_up(self) -> None:
        """Move cursor up."""
        if self._cursor_row > 0:
            self._cursor_row -= 1
            self._cursor_col = min(self._cursor_col, len(self._lines[self._cursor_row]))
            
    def _move_cursor_down(self) -> None:
        """Move cursor down."""
        if self._cursor_row < len(self._lines) - 1:
            self._cursor_row += 1
            self._cursor_col = min(self._cursor_col, len(self._lines[self._cursor_row]))
            
    async def _handle_submit(self) -> None:
        """Handle input submission."""
        content = self.get_current_input()
        for callback in self._submit_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(content)
                else:
                    callback(content)
            except Exception:
                pass  # Graceful callback failure
                
    async def _trigger_change(self) -> None:
        """Trigger change callbacks."""
        content = self.get_current_input()
        for callback in self._change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(content)
                else:
                    callback(content)
            except Exception:
                pass  # Graceful callback failure
                
    def _trigger_change_sync(self) -> None:
        """Trigger change callbacks synchronously (for non-async contexts)."""
        content = self.get_current_input()
        for callback in self._change_callbacks:
            try:
                if not asyncio.iscoroutinefunction(callback):
                    callback(content)
                # Skip async callbacks when not in async context
            except Exception:
                pass  # Graceful callback failure
    
    async def handle_submit(self, text: str) -> None:
        """Handle line submission from terminal input."""
        # Set the text in the field
        self.set_input(text)
        
        # Trigger submit callbacks 
        await self._handle_submit()
        
        # Clear after processing
        self.clear_input()
    
    def _ensure_initialized(self) -> None:
        """FIXED: Ensure input field is properly initialized and ready."""
        if self._initialized:
            return
            
        # Reset to clean state if needed
        if not self._lines or (len(self._lines) == 1 and not self._lines[0]):
            self._lines = [""]
            self._cursor_row = 0
            self._cursor_col = 0
        
        # Ensure cursor is in valid position
        if self._cursor_row >= len(self._lines):
            self._cursor_row = len(self._lines) - 1
        if self._cursor_col > len(self._lines[self._cursor_row]):
            self._cursor_col = len(self._lines[self._cursor_row])
            
        # Mark as initialized
        self._cursor_visible = True
        self._initialized = True