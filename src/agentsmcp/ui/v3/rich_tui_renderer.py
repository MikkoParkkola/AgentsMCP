"""Rich TUI renderer - full-featured beautiful interface."""

import asyncio
import sys
import select
import termios
import tty
from typing import Optional
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.align import Align
from .ui_renderer_base import UIRenderer


class RichTUIRenderer(UIRenderer):
    """Full-featured Rich-based TUI renderer."""
    
    def __init__(self, capabilities):
        super().__init__(capabilities)
        self.console = None
        self.live = None
        self.layout = None
        self._input_buffer = ""
        self._cursor_pos = 0
        self._original_terminal_attrs = None
        
    def initialize(self) -> bool:
        """Initialize Rich TUI renderer with proper terminal size detection."""
        try:
            # Check if Rich should work in this environment
            if not self.capabilities.supports_rich:
                return False
            
            # Initialize Rich console with dynamic terminal detection
            # Don't force terminal dimensions - let Rich auto-detect for better responsiveness
            self.console = Console(
                force_terminal=self.capabilities.is_tty,
                color_system="auto" if self.capabilities.supports_colors else None,
                # Remove fixed width/height to allow dynamic resizing
            )
            
            # Get current terminal size from console for layout calculations
            terminal_width = self.console.size.width
            terminal_height = self.console.size.height
            
            # Create responsive layout with proper size constraints
            self.layout = Layout()
            
            # Calculate responsive layout sizes based on terminal dimensions
            # Ensure minimum viable sizes while being responsive
            header_size = max(2, min(3, terminal_height // 8))  # 2-3 lines depending on height
            status_size = 1  # Always 1 line for status
            input_size = max(2, min(4, terminal_height // 6))   # 2-4 lines depending on height
            
            # Main area gets remaining space (with minimum of 5 lines)
            remaining_height = terminal_height - header_size - input_size - status_size
            if remaining_height < 5:
                # Terminal too small - adjust other areas
                header_size = 2
                input_size = 2
                remaining_height = max(3, terminal_height - 5)
            
            self.layout.split_column(
                Layout(name="header", size=header_size),
                Layout(name="main", ratio=1),  # Takes remaining space
                Layout(name="input", size=input_size),
                Layout(name="status", size=status_size)
            )
            
            # Store terminal dimensions for content formatting
            self._terminal_width = terminal_width
            self._terminal_height = terminal_height
            
            # Initialize layout content
            self._update_layout()
            
            # Start Live display with adaptive refresh rate
            # Lower refresh rate for better performance and less flashing
            refresh_rate = min(10, self.capabilities.max_refresh_rate)  # Max 10 FPS to reduce flashing
            
            # Store original refresh rate for input handling
            self._original_refresh_rate = refresh_rate
            
            self.live = Live(
                self.layout,
                console=self.console,
                screen=True,
                redirect_stderr=False,
                refresh_per_second=refresh_rate
            )
            self.live.start()
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize Rich TUI: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up Rich TUI renderer."""
        # FIX: Prevent multiple cleanup calls more robustly
        if getattr(self, '_cleanup_called', False):
            return
        self._cleanup_called = True
        
        try:
            # Restore terminal attributes if they were modified
            if self._original_terminal_attrs is not None and sys.stdin.isatty():
                try:
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._original_terminal_attrs)
                except Exception:
                    pass  # Ignore terminal restoration errors
            
            if self.live and hasattr(self.live, 'stop'):
                self.live.stop()
                self.live = None
            
            if self.console and hasattr(self.console, 'print'):
                self.console.print("Goodbye! 👋")
                
        except Exception:
            pass  # Ignore cleanup errors
    
    def render_frame(self) -> None:
        """Render current frame in Rich TUI."""
        try:
            if not self.live:
                return
                
            self._update_layout()
            self.live.refresh()
            
        except Exception as e:
            if self.console:
                self.console.print(f"[red]Render error: {e}[/red]")
    
    def handle_input(self) -> Optional[str]:
        """Handle user input without interfering with Rich Live display."""
        try:
            if self.state.is_processing:
                return None
            
            # Check if we have input available without blocking
            if not sys.stdin.isatty():
                # Non-TTY environment - fallback to basic input
                return self._handle_non_tty_input()
            
            # Non-blocking input detection
            if not select.select([sys.stdin], [], [], 0)[0]:
                # No input available
                return None
            
            # Setup terminal for raw character input
            old_attrs = None
            try:
                old_attrs = termios.tcgetattr(sys.stdin.fileno())
                tty.setraw(sys.stdin.fileno())
                
                # Read single character
                char = sys.stdin.read(1)
                if not char:
                    return None
                    
                char_code = ord(char)
                
                # Handle special characters
                if char_code == 13 or char_code == 10:  # Enter
                    # Complete input ready
                    complete_input = self._input_buffer
                    self._input_buffer = ""
                    self._cursor_pos = 0
                    self.state.current_input = ""
                    return complete_input if complete_input.strip() else None
                    
                elif char_code == 127 or char_code == 8:  # Backspace/Delete
                    if self._cursor_pos > 0:
                        self._input_buffer = (
                            self._input_buffer[:self._cursor_pos-1] + 
                            self._input_buffer[self._cursor_pos:]
                        )
                        self._cursor_pos -= 1
                        self.state.current_input = self._input_buffer
                        
                elif char_code == 3:  # Ctrl+C
                    self._input_buffer = ""
                    self._cursor_pos = 0
                    self.state.current_input = ""
                    return "/quit"
                    
                elif char_code >= 32 and char_code <= 126:  # Printable characters
                    # Insert character at cursor position
                    self._input_buffer = (
                        self._input_buffer[:self._cursor_pos] + 
                        char + 
                        self._input_buffer[self._cursor_pos:]
                    )
                    self._cursor_pos += 1
                    self.state.current_input = self._input_buffer
                    
                # Handle arrow keys (multi-byte sequences)
                elif char_code == 27:  # ESC sequence
                    # Try to read arrow key sequence
                    if select.select([sys.stdin], [], [], 0.01)[0]:
                        second_char = sys.stdin.read(1)
                        if second_char == '[' and select.select([sys.stdin], [], [], 0.01)[0]:
                            third_char = sys.stdin.read(1)
                            if third_char == 'C':  # Right arrow
                                if self._cursor_pos < len(self._input_buffer):
                                    self._cursor_pos += 1
                            elif third_char == 'D':  # Left arrow
                                if self._cursor_pos > 0:
                                    self._cursor_pos -= 1
                
                return None  # Input not complete yet
                
            finally:
                # Restore terminal attributes
                if old_attrs is not None:
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_attrs)
                    
        except Exception as e:
            # Reset input state on error
            self._input_buffer = ""
            self._cursor_pos = 0
            self.state.current_input = ""
            if self.console:
                self.console.print(f"[red]Input error: {e}[/red]")
            return None
    
    def _handle_non_tty_input(self) -> Optional[str]:
        """Fallback input handler for non-TTY environments."""
        try:
            # Use select with a very short timeout for non-blocking check
            if select.select([sys.stdin], [], [], 0.01)[0]:
                line = sys.stdin.readline().strip()
                return line if line else None
            return None
        except Exception:
            return None
    
    def show_message(self, message: str, level: str = "info") -> None:
        """Show a message in Rich TUI."""
        try:
            colors = {
                "info": "blue",
                "success": "green",
                "warning": "yellow",
                "error": "red"
            }
            color = colors.get(level, "blue")
            
            self.state.status_message = f"[{color}]{message}[/{color}]"
            
        except Exception as e:
            if self.console:
                self.console.print(f"[red]Message display error: {e}[/red]")
    
    def show_error(self, error: str) -> None:
        """Show an error in Rich TUI."""
        self.show_message(error, "error")
    
    def _update_layout(self) -> None:
        """Update layout content with proper terminal size handling."""
        try:
            # Get current terminal dimensions (may have changed)
            current_width = self.console.size.width
            current_height = self.console.size.height
            
            # Update stored dimensions if terminal was resized
            if hasattr(self, '_terminal_width'):
                self._terminal_width = current_width
                self._terminal_height = current_height
            
            # Header - keep it concise to fit terminal width
            header_text = "🤖 AI Command Composer - Rich TUI"
            if current_width < len(header_text) + 4:  # Account for panel borders
                header_text = "🤖 AI Command Composer"
            if current_width < len(header_text) + 4:
                header_text = "🤖 AI Chat"
                
            self.layout["header"].update(
                Panel(
                    Align.center(header_text),
                    style="bold blue"
                )
            )
            
            # Main area (messages) - content is wrapped in _render_messages
            main_content = self._render_messages()
            self.layout["main"].update(
                Panel(main_content, title="Chat", border_style="green")
            )
            
            # Input area with cursor - constrain to terminal width
            prompt = "💬 > "
            input_text = self.state.current_input or ""
            
            # Calculate available width for input (account for panel borders and prompt)
            input_available_width = max(20, current_width - len(prompt) - 8)  # Leave room for borders
            
            # Truncate input text if it's too long for display
            display_input = input_text
            display_cursor_pos = self._cursor_pos
            
            if len(input_text) > input_available_width:
                # Show end of input with ellipsis at start
                display_input = "..." + input_text[-(input_available_width-3):]
                # Adjust cursor position for display
                cursor_offset = len(input_text) - len(display_input) + 3  # +3 for "..."
                display_cursor_pos = max(0, min(len(display_input), self._cursor_pos - cursor_offset + 3))
            else:
                display_cursor_pos = min(self._cursor_pos, len(display_input))
            
            # Build input content with cursor
            input_content = Text()
            input_content.append(prompt, style="bold yellow")
            
            # Add text with cursor visualization
            if display_input:
                if display_cursor_pos >= 0 and display_cursor_pos <= len(display_input):
                    # Text before cursor
                    if display_cursor_pos > 0:
                        input_content.append(display_input[:display_cursor_pos], style="white")
                    
                    # Cursor (show as block or underscore)
                    if display_cursor_pos < len(display_input):
                        input_content.append(display_input[display_cursor_pos], style="reverse white")
                        if display_cursor_pos + 1 < len(display_input):
                            input_content.append(display_input[display_cursor_pos + 1:], style="white")
                    else:
                        input_content.append("▋", style="reverse white")  # Block cursor at end
                else:
                    input_content.append(display_input, style="white")
                    input_content.append("▋", style="reverse white")
            else:
                # Empty input with cursor
                input_content.append("▋", style="reverse white")
            
            # Create input title that fits terminal width
            input_title = "Type your message (Enter to send)"
            if current_width < len(input_title) + 8:
                input_title = "Type message (Enter)"
            if current_width < len(input_title) + 8:
                input_title = "Input"
            
            self.layout["input"].update(
                Panel(input_content, title=input_title, border_style="yellow")
            )
            
            # Status area - truncate status if needed
            status_text = "Ready"
            if self.state.is_processing:
                status_text = "⏳ Processing..."
            elif self.state.status_message:
                status_text = self.state.status_message
            
            # Truncate status to fit terminal width
            max_status_width = current_width - 4  # Account for padding
            if len(status_text) > max_status_width:
                status_text = status_text[:max_status_width-3] + "..."
                
            self.layout["status"].update(status_text)
            
        except Exception as e:
            if self.console:
                self.console.print(f"[red]Layout update error: {e}[/red]")
    
    def _render_messages(self) -> Text:
        """Render chat messages with proper text wrapping and scrolling."""
        try:
            if not self.state.messages:
                return Text("No messages yet. Type something to get started!", style="dim")
            
            content = Text()
            
            # Calculate available width for message content (account for panel borders and padding)
            available_width = max(40, getattr(self, '_terminal_width', 80) - 8)  # Leave room for borders/padding
            
            # Show ALL messages - don't artificially limit message count
            # Let the terminal's natural scrolling handle display limits
            messages_to_show = self.state.messages
            
            for msg in messages_to_show:
                role = msg.get("role", "unknown")
                text = msg.get("content", "")
                
                # Determine role prefix and style
                if role == "user":
                    prefix = "You: "
                    style = "bold blue"
                elif role == "assistant":
                    prefix = "AI: "
                    style = "green"
                else:
                    prefix = f"{role}: "
                    style = "dim"
                
                # Handle long messages with proper text wrapping
                if len(text) > available_width - len(prefix):
                    # Split long messages into wrapped lines
                    import textwrap
                    wrapped_lines = textwrap.wrap(
                        text, 
                        width=available_width - len(prefix),
                        break_long_words=True,
                        break_on_hyphens=True
                    )
                    
                    if wrapped_lines:
                        # First line with prefix
                        content.append(f"{prefix}{wrapped_lines[0]}\n", style=style)
                        
                        # Subsequent lines with proper indentation
                        indent = " " * len(prefix)
                        for line in wrapped_lines[1:]:
                            content.append(f"{indent}{line}\n", style=style)
                    else:
                        content.append(f"{prefix}[empty message]\n", style=style)
                else:
                    # Short message - no wrapping needed
                    content.append(f"{prefix}{text}\n", style=style)
                
                # Add a blank line between messages for readability
                content.append("\n")
            
            return content
            
        except Exception as e:
            return Text(f"Error rendering messages: {e}", style="red")