"""Console-Style Flow Renderer - eliminates all panel layout issues."""

import sys
from typing import Optional, List
from rich.console import Console
from rich.text import Text
from rich.rule import Rule
from rich.padding import Padding
from .ui_renderer_base import UIRenderer
from .console_message_formatter import ConsoleMessageFormatter


class ConsoleRenderer(UIRenderer):
    """Console-style renderer using Rich formatting without complex layouts."""
    
    def __init__(self, capabilities):
        super().__init__(capabilities)
        self.console = None
        self.formatter = None
        self._cleanup_called = False
        self._header_shown = False  # Prevent duplicate headers
        
    def initialize(self) -> bool:
        """Initialize console renderer with Rich formatting."""
        try:
            # Initialize Rich console
            self.console = Console(
                force_terminal=self.capabilities.is_tty,
                color_system="auto" if self.capabilities.supports_colors else None,
                width=self.capabilities.width if self.capabilities.width else None
            )
            
            # Initialize message formatter
            self.formatter = ConsoleMessageFormatter(self.console)
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize Console renderer: {e}")
            return False
    
    def show_welcome(self) -> None:
        """Show welcome header - only once."""
        if self._header_shown or not self.console:
            return
            
        try:
            # Clean, simple header
            welcome_text = Text.assemble(
                ("🤖 ", "bold blue"),
                ("AgentsMCP Console", "bold white"),
                (" - Rich Styled", "dim yellow")
            )
            
            self.console.print()  # Blank line for spacing
            self.console.print(welcome_text, justify="center")
            self.console.print(Rule(style="dim blue"), style="dim")
            self.console.print()
            
            self._header_shown = True
            
        except Exception as e:
            print(f"Welcome display error: {e}")
    
    def show_ready(self) -> None:
        """Show ready message."""
        try:
            if self.console:
                ready_text = Text("Ready! Type your message or /help for commands.", style="dim green")
                self.console.print(ready_text)
                self.console.print()
                
        except Exception as e:
            print(f"Ready display error: {e}")
    
    def cleanup(self) -> None:
        """Clean console renderer cleanup."""
        if self._cleanup_called:
            return
        self._cleanup_called = True
        
        # No special cleanup needed - console renderer is stateless
        pass
    
    def render_frame(self) -> None:
        """No frame rendering needed for console style."""
        pass
    
    def handle_input(self) -> Optional[str]:
        """Handle user input with Rich prompt styling."""
        try:
            if self.state.is_processing:
                return None
            
            # Simple Rich-styled prompt
            if self.console:
                self.console.print("💬 [yellow]>[/yellow] ", end="")
            
            try:
                user_input = input().strip()
                return user_input if user_input else None
            except (EOFError, KeyboardInterrupt):
                return "/quit"
            except Exception:
                return "/quit"
                
        except Exception:
            return "/quit"
    
    def show_message(self, message: str, level: str = "info") -> None:
        """Show a simple message with Rich styling."""
        try:
            if not self.console:
                print(message)
                return
                
            colors = {
                "info": "blue",
                "success": "green", 
                "warning": "yellow",
                "error": "red"
            }
            color = colors.get(level, "blue")
            
            self.console.print(f"[{color}]{message}[/{color}]")
                
        except Exception as e:
            print(f"Message display error: {e}")
    
    def display_chat_message(self, role: str, content: str, timestamp: str = None) -> None:
        """Display a chat message using console formatter."""
        try:
            if not self.console or not self.formatter:
                return
            
            # Use the dedicated formatter for consistent styling
            self.formatter.format_and_display_message(role, content, timestamp)
                
        except Exception as e:
            print(f"Chat message display error: {e}")
    
    def show_status(self, status: str) -> None:
        """Show status message."""
        try:
            if self.console and status and status != "Ready":
                # Only show non-"Ready" status to avoid spam
                self.console.print(f"[dim cyan]⏳ {status}[/dim cyan]")
                
        except Exception as e:
            print(f"Status display error: {e}")
    
    def show_thinking(self) -> None:
        """Show thinking indicator."""
        try:
            if self.console:
                self.console.print("[dim cyan]🤔 Thinking...[/dim cyan]")
        except Exception as e:
            print(f"Thinking display error: {e}")
    
    def show_error(self, error: str) -> None:
        """Show an error message."""
        self.show_message(error, "error")
    
    def show_goodbye(self) -> None:
        """Show goodbye message."""
        try:
            if self.console:
                self.console.print()  # Blank line
                goodbye_text = Text("👋 Goodbye!", style="bold yellow")
                self.console.print(goodbye_text)
                
        except Exception as e:
            print("👋 Goodbye!")