"""Rich TUI renderer - full-featured beautiful interface."""

import asyncio
import sys
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
        
    def initialize(self) -> bool:
        """Initialize Rich TUI renderer."""
        try:
            # Check if Rich should work in this environment
            if not self.capabilities.supports_rich:
                return False
            
            # Initialize Rich console with terminal detection
            self.console = Console(
                force_terminal=self.capabilities.is_tty,
                width=self.capabilities.width,
                height=self.capabilities.height,
                color_system="auto" if self.capabilities.supports_colors else None
            )
            
            # Create layout
            self.layout = Layout()
            self.layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main", ratio=1),
                Layout(name="input", size=3),
                Layout(name="status", size=1)
            )
            
            # Initialize layout content
            self._update_layout()
            
            # Start Live display
            self.live = Live(
                self.layout,
                console=self.console,
                screen=True,
                redirect_stderr=False,
                refresh_per_second=self.capabilities.max_refresh_rate
            )
            self.live.start()
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize Rich TUI: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up Rich TUI renderer."""
        try:
            if self.live:
                self.live.stop()
                self.live = None
            
            if self.console:
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
        """Handle user input in Rich TUI mode."""
        try:
            import select
            
            # Check for available input (non-blocking)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                char = sys.stdin.read(1)
                
                if ord(char) == 13:  # Enter key
                    result = self._input_buffer.strip()
                    self._input_buffer = ""
                    self._cursor_pos = 0
                    self.state.current_input = ""
                    return result if result else None
                    
                elif ord(char) == 127:  # Backspace
                    if self._cursor_pos > 0:
                        self._input_buffer = (
                            self._input_buffer[:self._cursor_pos-1] + 
                            self._input_buffer[self._cursor_pos:]
                        )
                        self._cursor_pos -= 1
                        self.state.current_input = self._input_buffer
                        
                elif ord(char) >= 32:  # Printable character
                    self._input_buffer = (
                        self._input_buffer[:self._cursor_pos] + 
                        char + 
                        self._input_buffer[self._cursor_pos:]
                    )
                    self._cursor_pos += 1
                    self.state.current_input = self._input_buffer
                    
            return None
            
        except Exception as e:
            if self.console:
                self.console.print(f"[red]Input error: {e}[/red]")
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
        """Update layout content."""
        try:
            # Header
            self.layout["header"].update(
                Panel(
                    Align.center("🤖 AI Command Composer - Rich TUI"),
                    style="bold blue"
                )
            )
            
            # Main area (messages)
            main_content = self._render_messages()
            self.layout["main"].update(
                Panel(main_content, title="Chat", border_style="green")
            )
            
            # Input area
            input_content = Text(f"> {self.state.current_input}")
            if self._cursor_pos < len(self.state.current_input):
                input_content.stylize("reverse", self._cursor_pos + 2, self._cursor_pos + 3)
            
            self.layout["input"].update(
                Panel(input_content, title="Input", border_style="yellow")
            )
            
            # Status area
            status_text = "Ready"
            if self.state.is_processing:
                status_text = "⏳ Processing..."
            elif self.state.status_message:
                status_text = self.state.status_message
                
            self.layout["status"].update(status_text)
            
        except Exception as e:
            if self.console:
                self.console.print(f"[red]Layout update error: {e}[/red]")
    
    def _render_messages(self) -> Text:
        """Render chat messages."""
        try:
            if not self.state.messages:
                return Text("No messages yet. Type something to get started!", style="dim")
            
            content = Text()
            for msg in self.state.messages[-10:]:  # Show last 10 messages
                role = msg.get("role", "unknown")
                text = msg.get("content", "")
                
                if role == "user":
                    content.append(f"You: {text}\n", style="bold blue")
                elif role == "assistant":
                    content.append(f"AI: {text}\n", style="green")
                else:
                    content.append(f"{role}: {text}\n", style="dim")
                    
            return content
            
        except Exception as e:
            return Text(f"Error rendering messages: {e}", style="red")