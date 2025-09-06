"""Console-Style Flow Renderer - eliminates all panel layout issues."""

import sys
import uuid
from typing import Optional, List
from rich.console import Console
from rich.text import Text
from rich.rule import Rule
from rich.padding import Padding
from .ui_renderer_base import UIRenderer
from .console_message_formatter import ConsoleMessageFormatter
from .streaming_state_manager import StreamingStateManager


class ConsoleRenderer(UIRenderer):
    """Console-style renderer using Rich formatting without complex layouts."""
    
    def __init__(self, capabilities):
        super().__init__(capabilities)
        self.console = None
        self.formatter = None
        self._cleanup_called = False
        self._header_shown = False  # Prevent duplicate headers
        
        # Initialize streaming state manager
        self.streaming_manager = StreamingStateManager(supports_tty=capabilities.is_tty)
        
        # Input history management
        self._input_history = []
        self._max_history = 1000
        self._history_pos = -1
        
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
        
        # Clean up streaming state
        if self.streaming_manager:
            self.streaming_manager.force_cleanup()
    
    def render_frame(self) -> None:
        """No frame rendering needed for console style."""
        pass
    
    
    def handle_input(self) -> Optional[str]:
        """Handle user input with Rich prompt styling and basic history support."""
        try:
            if self.state.is_processing:
                return None
            
            try:
                # Get input with readline support for basic history/editing
                import readline
                
                # Configure readline for better terminal experience
                readline.set_startup_hook(None)
                readline.clear_history()
                
                # Add current history to readline
                for item in self._input_history[-50:]:  # Last 50 items for performance
                    readline.add_history(item)
                
                # Add timestamp to prompt  
                import datetime
                timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
                user_input = input(f"{timestamp} 💬 > ").strip()
                
                # Add to our history if not empty and different from last
                if user_input and (not self._input_history or self._input_history[-1] != user_input):
                    self._input_history.append(user_input)
                    # Keep history within limits
                    if len(self._input_history) > self._max_history:
                        self._input_history = self._input_history[-self._max_history:]
                
                return user_input if user_input else None
                
            except ImportError:
                # Fallback without readline
                user_input = input().strip()
                
                # Still maintain our basic history
                if user_input and (not self._input_history or self._input_history[-1] != user_input):
                    self._input_history.append(user_input)
                    if len(self._input_history) > self._max_history:
                        self._input_history = self._input_history[-self._max_history:]
                
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
    
    def show_status(self, status: str) -> None:
        """Show status message with enhanced formatting for detailed progress."""
        try:
            if not self.console or not status or status == "Ready":
                return
                
            # Enhanced status formatting with different colors for different types
            if status.startswith("🔍"):
                # Analysis phase - blue
                self.console.print(f"[blue]{status}[/blue]")
            elif status.startswith("🛠️"):
                # Tool execution - yellow
                self.console.print(f"[yellow]{status}[/yellow]")
            elif status.startswith("📊"):
                # Multi-turn processing - magenta
                self.console.print(f"[magenta]{status}[/magenta]")
            elif status.startswith("🔄"):
                # Processing results - cyan
                self.console.print(f"[cyan]{status}[/cyan]")
            elif status.startswith("✨") or status.startswith("🎯"):
                # Finalizing/streaming - green
                self.console.print(f"[green]{status}[/green]")
            elif status.startswith("🚀"):
                # Direct processing - bright blue
                self.console.print(f"[bright_blue]{status}[/bright_blue]")
            elif "Tool execution" in status or "tool:" in status.lower():
                # Tool-related status - bright yellow
                self.console.print(f"[bright_yellow]{status}[/bright_yellow]")
            else:
                # Default status - dim cyan
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
    
    def handle_streaming_update(self, content: str) -> None:
        """Handle real-time streaming updates with robust state management."""
        try:
            if not self.console:
                return
                
            # Start streaming session if not already active
            if not self.streaming_manager.is_streaming_active():
                session_id = str(uuid.uuid4())[:8]  # Short session ID
                self.streaming_manager.start_streaming_session(session_id)
            
            # Use streaming state manager to handle the update
            self.streaming_manager.display_streaming_update(content)
            
        except Exception as e:
            print(f"\nStreaming update error: {e}")
            # Force cleanup on error
            if self.streaming_manager:
                self.streaming_manager.force_cleanup()
    
    def display_chat_message(self, role: str, content: str, timestamp: str = None) -> None:
        """Display a chat message using console formatter."""
        try:
            if not self.console or not self.formatter:
                return
            
            # If we were streaming and this is the final assistant message
            if self.streaming_manager.is_streaming_active() and role == "assistant":
                # Complete the streaming session first
                self.streaming_manager.complete_streaming_session()
                
                # Display the complete final message using formatter
                self.formatter.format_and_display_message(role, content, timestamp)
                return
            
            # Use the dedicated formatter for consistent styling
            self.formatter.format_and_display_message(role, content, timestamp)
                
        except Exception as e:
            print(f"Chat message display error: {e}")
    
    def show_goodbye(self) -> None:
        """Show goodbye message."""
        try:
            if self.console:
                # Clean up any active streaming
                if self.streaming_manager.is_streaming_active():
                    self.streaming_manager.complete_streaming_session()
                
                self.console.print()  # Blank line
                goodbye_text = Text("👋 Goodbye!", style="bold yellow")
                self.console.print(goodbye_text)
                
        except Exception as e:
            print("👋 Goodbye!")