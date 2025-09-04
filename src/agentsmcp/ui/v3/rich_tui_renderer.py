"""Rich TUI renderer - PHASE 3: Rich Live display panels with stable input handling."""

import sys
from typing import Optional
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from .ui_renderer_base import UIRenderer


class RichTUIRenderer(UIRenderer):
    """PHASE 3: Rich TUI with Live display panels and stable input handling."""
    
    def __init__(self, capabilities):
        super().__init__(capabilities)
        self.console = None
        self.live = None
        self.layout = None
        self._cleanup_called = False  # Guard against multiple cleanup calls
        self._conversation_history = []  # Track messages for the conversation panel
        self._current_status = "Ready"  # Track current status
        # Input history management
        self._input_history = []
        self._max_history = 1000  # Track current status
        
    def initialize(self) -> bool:
        """PHASE 3: Initialize Rich TUI with Live display panels."""
        try:
            # Check if Rich should work in this environment
            if not self.capabilities.supports_rich:
                return False
            
            # Initialize Rich console
            self.console = Console(
                force_terminal=self.capabilities.is_tty,
                color_system="auto" if self.capabilities.supports_colors else None
            )
            
            # PHASE 3: Create Rich Layout with panels
            self.layout = Layout()
            
            # Split layout into header, body (conversation + status), and footer
            # Use smaller fixed sizes to fit better in terminal
            self.layout.split_column(
                Layout(name="header", size=1),  # Reduced from 3 to 1
                Layout(name="body", ratio=1),
                Layout(name="footer", size=1)   # Reduced from 3 to 1
            )
            
            # Split body into conversation and status panels  
            self.layout["body"].split_row(
                Layout(name="conversation", ratio=3),
                Layout(name="status", ratio=1)
            )
            
            # Initialize panels but DON'T update them yet - wait for Live display to start
            # This prevents immediate rendering to console during init
            self._initialize_panels()
            
            # Create Live display but DON'T start it yet to prevent duplicate headers
            if self.capabilities.is_tty:
                self.live = Live(
                    self.layout,
                    console=self.console,
                    refresh_per_second=2,  # Moderate refresh rate
                    screen=False,  # Don't take over full screen - allows input mixing
                    vertical_overflow="ellipsis"  # Handle overflow gracefully
                )
                # Live display will be started by start_live_display() method after init is complete
            else:
                # Fallback for non-TTY environments
                self.console.print("🤖 [bold blue]Rich Console TUI (Live Display)[/bold blue]")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize Rich TUI with Live display: {e}")
            return False

    
    def _initialize_panels(self) -> None:
        """Initialize panel content without rendering to prevent header duplication."""
        # Just prepare the layout structure - don't call layout.update() yet
        # This prevents immediate console rendering during initialization
        pass
    
    def start_live_display(self) -> None:
        """Start the Live display after initialization is complete to prevent duplicate headers."""
        try:
            if self.live and self.capabilities.is_tty:
                if not (hasattr(self.live, '_started') and self.live._started):
                    # Now that we're ready to start Live display, initialize all panels
                    self._update_header()
                    self._update_conversation_panel()
                    self._update_status_panel()
                    self._update_footer()
                    # Start the Live display
                    self.live.start()
        except Exception as e:
            print(f"Failed to start Live display: {e}")
    
    def _update_header(self) -> None:
        """Update the header panel."""
        header_text = Text.assemble(
            ("🤖 ", "bold blue"),
            ("AgentsMCP TUI", "bold white"),
            (" - PHASE 3", "dim yellow")
        )
        self.layout["header"].update(
            Panel(header_text, style="blue", padding=(0, 1), height=1)
        )
    
    def _update_conversation_panel(self) -> None:
        """Update the conversation panel with message history and markdown rendering."""
        if not self._conversation_history:
            content = Text("No messages yet. Start a conversation!", style="dim italic")
        else:
            # Show last 10 messages to keep it manageable
            recent_messages = self._conversation_history[-10:]
            content = Text()
            
            # More accurate width calculation for conversation panel
            # The layout is conversation (ratio=3) : status (ratio=1), so conversation gets 75% of body width
            console_width = self.console.size.width
            # Account for: panel borders (2 chars each side = 4), padding (2 chars each side = 4 total)
            # The Rich layout automatically handles the separation between conversation and status panels
            conversation_width = int(console_width * 0.75) - 6  # Panel borders + padding
            
            for msg_data in recent_messages:
                # Handle both old string format and new structured format
                if isinstance(msg_data, dict):
                    role = msg_data.get("role", "unknown")
                    msg_content = msg_data.get("content", "")
                    time_prefix = msg_data.get("timestamp", "")
                    is_markdown = msg_data.get("is_markdown", False)
                    
                    # Format role header with timestamp
                    if role == "user":
                        role_header = f"{time_prefix}[bold blue]👤 You:[/bold blue] "
                    elif role == "assistant":
                        role_header = f"{time_prefix}[bold green]🤖 AI:[/bold green] "
                    elif role == "system":
                        role_header = f"{time_prefix}[dim yellow]ℹ️ System:[/dim yellow] "
                    else:
                        role_header = f"{time_prefix}[dim]❓ {role}:[/dim] "
                    
                    # Add role header
                    content.append(role_header)
                    
                    # For AI responses with markdown, render simplified text version
                    if role == "assistant" and is_markdown:
                        # Convert markdown to plain text for the panel (simplified)
                        # Remove basic markdown formatting for panel display
                        import re
                        plain_content = re.sub(r'\*\*(.*?)\*\*', r'\1', msg_content)  # Bold
                        plain_content = re.sub(r'\*(.*?)\*', r'\1', plain_content)     # Italic
                        plain_content = re.sub(r'`(.*?)`', r'\1', plain_content)      # Code
                        plain_content = re.sub(r'#{1,6}\s+(.*)', r'\1', plain_content)  # Headers
                        plain_content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', plain_content)  # Links
                        msg_content = plain_content
                    
                    # Handle line wrapping for the message content
                    msg_lines = msg_content.split('\n')
                    for msg_line in msg_lines:
                        if len(msg_line) > conversation_width:
                            import textwrap
                            wrapped_lines = textwrap.wrap(msg_line, width=conversation_width)
                            for line in wrapped_lines:
                                content.append(f"{line}\n")
                        else:
                            content.append(f"{msg_line}\n")
                    
                else:
                    # Handle legacy string format for backward compatibility
                    msg_lines = msg_data.split('\n')
                    for msg_line in msg_lines:
                        if len(msg_line) > conversation_width:
                            import textwrap
                            wrapped_lines = textwrap.wrap(msg_line, width=conversation_width)
                            for line in wrapped_lines:
                                content.append(f"{line}\n")
                        else:
                            content.append(f"{msg_line}\n")
                
                # Add separator between messages
                content.append("\n")
        
        self.layout["conversation"].update(
            Panel(content, title="[bold white]Conversation", border_style="green", padding=(0, 1))
        )
    
    def _update_status_panel(self) -> None:
        """Update the status panel with current information."""
        # Create a simple status table
        table = Table.grid(padding=(0, 1))
        table.add_column("Label", style="bold cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Status:", self._current_status)
        table.add_row("Messages:", str(len(self._conversation_history)))
        table.add_row("Time:", Text.from_markup("[dim]Live[/dim]"))
        
        self.layout["status"].update(
            Panel(table, title="[bold white]Status", border_style="yellow", padding=(0, 1))
        )
    
    def _update_footer(self) -> None:
        """Update the footer panel with help text."""
        footer_text = Text.assemble(
            ("Commands: ", "bold white"),
            ("/help", "cyan"),
            (", ", "white"),
            ("/quit", "cyan"),
            (", ", "white"),
            ("/clear", "cyan"),
            ("  •  ", "dim"),
            ("Type your message and press Enter", "dim italic")
        )
        self.layout["footer"].update(
            Panel(footer_text, style="dim", padding=(0, 1), height=1)
        )

    def cleanup(self) -> None:
        """PHASE 3: Cleanup with Live display management."""
        if self._cleanup_called:
            return  # Prevent multiple cleanup calls
        self._cleanup_called = True
        
        try:
            # Stop Live display if it's running
            if self.live and hasattr(self.live, 'stop'):
                self.live.stop()
            # Rich renderer cleanup - NO goodbye message here
            # Let the TUI launcher handle the single goodbye message
        except Exception:
            pass  # Ignore cleanup errors
    
    def render_frame(self) -> None:
        """PHASE 3: Update Live display panels."""
        try:
            if self.live and self.layout:
                # Update all panels to reflect current state
                self._update_conversation_panel()
                self._update_status_panel()
                # Header and footer are static, so no need to update
        except Exception as e:
            # Fallback to console print if Live display fails
            pass
    
    def handle_input(self) -> Optional[str]:
        """PHASE 3: Input handling compatible with Live display and history support."""
        try:
            if self.state.is_processing:
                return None
            
            # For screen=False Live display, we can do minimal interrupt input
            if self.live and hasattr(self.live, '_started') and self.live._started:
                # Temporarily pause Live display for clean input
                self.live.stop()
                
                # Enhanced input prompt with readline support
                try:
                    self.console.print("\n💬 [yellow]>[/yellow] ", end="")
                    
                    # Try to use readline for better input experience
                    try:
                        import readline
                        
                        # Configure readline
                        readline.set_startup_hook(None)
                        readline.clear_history()
                        
                        # Add recent history to readline
                        for item in self._input_history[-50:]:  # Last 50 for performance
                            readline.add_history(item)
                        
                        user_input = input().strip()
                        
                    except ImportError:
                        # Fallback without readline
                        user_input = input().strip()
                    
                    # Add to history
                    if user_input and (not self._input_history or self._input_history[-1] != user_input):
                        self._input_history.append(user_input)
                        if len(self._input_history) > self._max_history:
                            self._input_history = self._input_history[-self._max_history:]
                    
                except (EOFError, KeyboardInterrupt):
                    user_input = "/quit"
                except Exception:
                    user_input = "/quit"
                
                # Restart Live display immediately
                self.live.start()
                
                return user_input if user_input else None
            else:
                # Fallback input for non-Live display
                try:
                    # Try with readline support
                    try:
                        import readline
                        readline.set_startup_hook(None)
                        readline.clear_history()
                        for item in self._input_history[-50:]:
                            readline.add_history(item)
                    except ImportError:
                        pass
                    
                    user_input = input("💬 > ").strip()
                    
                    # Add to history
                    if user_input and (not self._input_history or self._input_history[-1] != user_input):
                        self._input_history.append(user_input)
                        if len(self._input_history) > self._max_history:
                            self._input_history = self._input_history[-self._max_history:]
                    
                    return user_input if user_input else None
                except (EOFError, KeyboardInterrupt):
                    return "/quit"
                    
        except Exception as e:
            # Return /quit on persistent errors to avoid infinite loops
            return "/quit"
    
    def show_message(self, message: str, level: str = "info") -> None:
        """Show a simple Rich formatted message."""
        try:
            colors = {
                "info": "blue",
                "success": "green", 
                "warning": "yellow",
                "error": "red"
            }
            color = colors.get(level, "blue")
            
            # Direct console print with Rich formatting
            if self.console:
                self.console.print(f"[{color}]{message}[/{color}]")
            
        except Exception as e:
            print(f"Message display error: {e}")
    
    def display_chat_message(self, role: str, content: str, timestamp: str = None) -> None:
        """Display a chat message with appropriate formatting and markdown rendering."""
        try:
            if not self.console:
                return
            
            # Add timestamp prefix if provided
            time_prefix = f"{timestamp} " if timestamp else ""
            
            # Create structured message for conversation history
            message_data = {
                "role": role,
                "content": content,
                "timestamp": time_prefix,
                "is_markdown": role == "assistant"  # Only render markdown for AI responses
            }
            
            # Add to conversation history for Live display panels
            self._conversation_history.append(message_data)
            
            # Format message for display with markdown support for AI responses
            if role == "user":
                display_msg = f"{time_prefix}[bold blue]👤 You:[/bold blue] {content}"
            elif role == "assistant":
                # For AI responses, use Rich Markdown rendering
                from rich.markdown import Markdown
                try:
                    # Create markdown object for rich rendering
                    markdown_content = Markdown(content)
                    # Create display message with markdown
                    display_msg = f"{time_prefix}[bold green]🤖 AI:[/bold green]"
                except Exception:
                    # Fallback to plain text if markdown parsing fails
                    display_msg = f"{time_prefix}[bold green]🤖 AI:[/bold green] {content}"
                    markdown_content = None
            elif role == "system":
                display_msg = f"{time_prefix}[dim yellow]ℹ️ System:[/dim yellow] {content}"
            else:
                display_msg = f"{time_prefix}[dim]❓ {role}:[/dim] {content}"
            
            # If Live display is active, ONLY update panels - no direct console.print
            if self.live and hasattr(self.live, '_started') and self.live._started:
                # Just update the conversation panel - Live display will handle rendering
                # No need to explicitly refresh - Live display auto-refreshes
                self._update_conversation_panel()
            else:
                # Direct console print ONLY if Live display not active
                if role == "assistant" and 'markdown_content' in locals() and markdown_content:
                    # Print role header first
                    self.console.print(display_msg)
                    # Then print markdown content with proper indentation
                    self.console.print(markdown_content, style="", crop=False)
                else:
                    # Standard console print for non-markdown content
                    self.console.print(display_msg)
                
        except Exception as e:
            print(f"Chat message display error: {e}")
    
    def show_status(self, status: str) -> None:
        """Show status message and update Live display."""
        try:
            # Update internal status
            self._current_status = status
            
            # Update Live display panels if active
            if self.live and self.layout:
                self._update_status_panel()
                
                # Only show status in console if Live display is not active
                # and avoid spamming "Ready" messages
            elif self.console and status and status != "Ready":
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
        """Show an error in Rich TUI."""
        self.show_message(error, "error")
    
    # PHASE 2: Remove all complex layout and message rendering
    # We're using simple console.print() instead