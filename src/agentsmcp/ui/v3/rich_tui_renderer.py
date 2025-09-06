"""Rich TUI renderer - PHASE 3: Rich Live display panels with stable input handling."""

import os
import sys
from typing import Optional
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from .ui_renderer_base import UIRenderer
from .progress_display import ProgressDisplay, AgentStatus, AgentProgress


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
        # Progress display integration
        self._progress_display = None
        self._agent_progress = {}  # Store agent progress information
        
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
                    
                    # For AI responses with markdown, preserve markdown for rendering
                    if role == "assistant" and is_markdown:
                        # Keep markdown content for Rich rendering
                        # We'll render it properly in the display below
                        pass
                    
                    # Handle markdown content vs plain text content
                    if role == "assistant" and is_markdown:
                        try:
                            # Create markdown object for Rich rendering with width constraints
                            markdown_obj = Markdown(msg_content)
                            # Add the markdown object directly to content
                            content.append(markdown_obj)
                            content.append("\n")
                        except Exception:
                            # Fallback to plain text if markdown parsing fails
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
                        # Handle regular text content with line wrapping
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
        """Update the status panel with comprehensive progress information."""
        try:
            # Create a comprehensive status display with progress information
            from rich.console import Group
            from datetime import datetime
            
            elements = []
            
            # Basic status information
            status_table = Table.grid(padding=(0, 0))
            status_table.add_column("Label", style="bold cyan")
            status_table.add_column("Value", style="white")
            
            status_table.add_row("Status:", self._current_status)
            status_table.add_row("Messages:", str(len(self._conversation_history)))
            status_table.add_row("Time:", datetime.now().strftime("%H:%M:%S"))
            
            elements.append(status_table)
            
            # Agent progress section if we have progress data
            if self._agent_progress:
                elements.append(Text("\n"))  # Separator
                elements.append(Text("🤖 Agents:", style="bold yellow"))
                
                for agent_id, progress_info in self._agent_progress.items():
                    agent_name = progress_info.get("name", agent_id)[:12]  # Truncate long names
                    status = progress_info.get("status", "unknown")
                    percentage = progress_info.get("progress", 0.0)
                    current_step = progress_info.get("step", "")
                    elapsed_ms = progress_info.get("elapsed_ms", 0)
                    
                    # Status icon
                    status_icons = {
                        "idle": "⏸️", "planning": "🎯", "in_progress": "🟢",
                        "waiting": "🟡", "blocked": "🔴", "completed": "✅", "error": "❌"
                    }
                    icon = status_icons.get(status, "❓")
                    
                    # Progress bar (simplified for small space)
                    bar_width = 8
                    filled = int((percentage / 100) * bar_width)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    
                    # Time formatting
                    time_str = ""
                    if elapsed_ms > 0:
                        if elapsed_ms < 1000:
                            time_str = f"({elapsed_ms}ms)"
                        elif elapsed_ms < 60000:
                            time_str = f"({elapsed_ms/1000:.1f}s)"
                        else:
                            time_str = f"({elapsed_ms//60000}m{(elapsed_ms%60000)//1000}s)"
                    
                    # Agent line
                    agent_line = f"{icon} {agent_name:<8} [{bar}] {percentage:3.0f}% {time_str}"
                    elements.append(Text(agent_line, style="dim" if status == "completed" else "white"))
                    
                    # Current step (if any and space allows)
                    if current_step and len(current_step) > 0:
                        step_text = current_step[:20] + "..." if len(current_step) > 20 else current_step
                        elements.append(Text(f"    └─ {step_text}", style="dim cyan"))
            
            # Task timing information from progress display
            if self._progress_display:
                try:
                    status_line = self._progress_display.format_status_line()
                    if status_line and status_line.strip():
                        elements.append(Text("\n"))  # Separator
                        elements.append(Text("⏱️ Task Status:", style="bold green"))
                        elements.append(Text(status_line, style="dim white"))
                except Exception:
                    pass  # Ignore progress display errors
            
            # Combine all elements into a group
            status_content = Group(*elements)
            
            self.layout["status"].update(
                Panel(status_content, title="[bold white]System Status", border_style="yellow", padding=(0, 1))
            )
            
        except Exception as e:
            # Fallback to simple status panel
            table = Table.grid(padding=(0, 1))
            table.add_column("Label", style="bold cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Status:", self._current_status)
            table.add_row("Messages:", str(len(self._conversation_history)))
            table.add_row("Error:", f"Status update failed: {str(e)[:20]}")
            
            self.layout["status"].update(
                Panel(table, title="[bold white]Status", border_style="red", padding=(0, 1))
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
        """PHASE 3: Enhanced input handling with FORCE_RICH mode adaptation and graceful EOF handling."""
        try:
            if self.state.is_processing:
                return None
            
            # Detect FORCE_RICH mode and non-TTY environment
            force_rich_mode = os.environ.get('AGENTSMCP_FORCE_RICH') == '1'
            is_non_tty = not self.capabilities.is_tty
            
            # For screen=False Live display, we can do minimal interrupt input
            if self.live and hasattr(self.live, '_started') and self.live._started:
                # Temporarily pause Live display for clean input
                self.live.stop()
                
                # Enhanced input prompt with readline support and FORCE_RICH adaptation
                try:
                    # Special handling for FORCE_RICH in non-TTY environments
                    if force_rich_mode and is_non_tty:
                        # Provide informative messaging for non-TTY FORCE_RICH mode
                        self.console.print("\n[dim yellow]⚠️  FORCE_RICH mode in non-TTY environment detected[/dim yellow]")
                        self.console.print("[dim]Input handling adapted for compatibility[/dim]")
                    
                    # Timestamp-enabled prompt will be handled by input() calls below
                    
                    # Try to use readline for better input experience
                    try:
                        import readline
                        
                        # Configure readline
                        readline.set_startup_hook(None)
                        readline.clear_history()
                        
                        # Add recent history to readline
                        for item in self._input_history[-50:]:  # Last 50 for performance
                            readline.add_history(item)
                        
                        # Non-blocking input polling for FORCE_RICH mode
                        if force_rich_mode and is_non_tty:
                            # Use non-blocking input when possible in FORCE_RICH non-TTY mode
                            try:
                                import select
                                import sys
                                import datetime
                                
                                # Check if input is available with timeout
                                if select.select([sys.stdin], [], [], 0.1)[0]:
                                    timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
                                    user_input = input(f"{timestamp} 💬 > ").strip()
                                else:
                                    # No immediate input available, return None to continue display
                                    return None
                            except (ImportError, OSError):
                                # Fallback to blocking input if select not available
                                import datetime
                                timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
                                user_input = input(f"{timestamp} 💬 > ").strip()
                        else:
                            # Standard blocking input with multi-line support
                            import datetime
                            timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
                            
                            # Check for multi-line input (like Claude Code CLI)
                            user_input = input(f"{timestamp} 💬 > ").strip()
                            
                            # If input seems incomplete (very long or ends with special chars), allow continuation
                            lines = [user_input]
                            while (len(user_input) > 200 or user_input.endswith(',') or user_input.endswith('.') or user_input.endswith('and')) and len(lines) < 10:
                                try:
                                    continuation = input("... ").strip()
                                    if not continuation:  # Empty line signals end
                                        break
                                    lines.append(continuation)
                                    user_input = " ".join(lines)
                                except (EOFError, KeyboardInterrupt):
                                    break
                            
                            user_input = " ".join(lines)
                        
                    except ImportError:
                        # Fallback without readline
                        if force_rich_mode and is_non_tty:
                            # Attempt non-blocking input without readline
                            try:
                                import select
                                import sys
                                import datetime
                                
                                if select.select([sys.stdin], [], [], 0.1)[0]:
                                    timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
                                    user_input = input(f"{timestamp} 💬 > ").strip()
                                else:
                                    return None
                            except (ImportError, OSError):
                                import datetime
                                timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
                                user_input = input(f"{timestamp} 💬 > ").strip()
                        else:
                            import datetime
                            timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
                            user_input = input(f"{timestamp} 💬 > ").strip()
                    
                    # Add to history
                    if user_input and (not self._input_history or self._input_history[-1] != user_input):
                        self._input_history.append(user_input)
                        if len(self._input_history) > self._max_history:
                            self._input_history = self._input_history[-self._max_history:]
                    
                except (EOFError, KeyboardInterrupt):
                    # Enhanced EOF handling for FORCE_RICH mode
                    if force_rich_mode and is_non_tty:
                        self.console.print("\n[dim yellow]ℹ️  EOF detected in FORCE_RICH non-TTY mode[/dim yellow]")
                        self.console.print("[dim]Rich panels remain displayed. Use /quit to exit or provide input.[/dim]")
                        self.console.print("[dim]Tip: Run in a real terminal for better input experience.[/dim]")
                        # Return None instead of immediate /quit to keep panels displayed
                        return None
                    else:
                        user_input = "/quit"
                except Exception as e:
                    # More graceful error handling for FORCE_RICH mode
                    if force_rich_mode and is_non_tty:
                        self.console.print(f"\n[dim red]Input error in FORCE_RICH mode: {str(e)[:50]}[/dim red]")
                        self.console.print("[dim]Continuing with panel display. Type /quit to exit.[/dim]")
                        return None
                    else:
                        user_input = "/quit"
                
                # Restart Live display immediately
                self.live.start()
                
                return user_input if user_input else None
            else:
                # Fallback input for non-Live display with FORCE_RICH adaptation
                try:
                    # Special messaging for FORCE_RICH fallback mode
                    if force_rich_mode and is_non_tty:
                        print("⚠️  FORCE_RICH fallback: Live display unavailable, using static mode")
                    
                    # Try with readline support
                    try:
                        import readline
                        readline.set_startup_hook(None)
                        readline.clear_history()
                        for item in self._input_history[-50:]:
                            readline.add_history(item)
                    except ImportError:
                        pass
                    
                    # Add timestamp to user input prompt
                    import datetime
                    timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
                    user_input = input(f"{timestamp} 💬 > ").strip()
                    
                    # Add to history
                    if user_input and (not self._input_history or self._input_history[-1] != user_input):
                        self._input_history.append(user_input)
                        if len(self._input_history) > self._max_history:
                            self._input_history = self._input_history[-self._max_history:]
                    
                    return user_input if user_input else None
                except (EOFError, KeyboardInterrupt):
                    # Enhanced EOF handling for fallback mode
                    if force_rich_mode and is_non_tty:
                        print("ℹ️  EOF in FORCE_RICH mode - use /quit command to exit properly")
                        return None
                    else:
                        return "/quit"
                    
        except Exception as e:
            # Enhanced error recovery for FORCE_RICH mode
            if force_rich_mode and is_non_tty:
                if self.console:
                    self.console.print(f"\n[dim red]Critical input error: {str(e)[:50]}[/dim red]")
                    self.console.print("[dim]FORCE_RICH mode continuing. Use /quit to exit safely.[/dim]")
                else:
                    print(f"Critical input error in FORCE_RICH mode: {str(e)[:50]}")
                    print("Use /quit to exit safely.")
                return None
            else:
                # Return /quit on persistent errors in normal mode to avoid infinite loops
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
        """Show status message and update Live display with enhanced agent tracking."""
        try:
            # Update internal status
            self._current_status = status
            
            # Parse status for agent progress information
            self._parse_agent_status_update(status)
            
            # Update Live display panels if active
            if self.live and self.layout:
                self._update_status_panel()
                
                # Only show status in console if Live display is not active
                # and avoid spamming "Ready" messages
            elif self.console and status and status != "Ready":
                self.console.print(f"[dim cyan]⏳ {status}[/dim cyan]")
                
        except Exception as e:
            print(f"Status display error: {e}")
    
    def _parse_agent_status_update(self, status: str) -> None:
        """Parse status messages for agent progress information."""
        try:
            # Extract agent information from enhanced status messages
            if "Agent-" in status or "🛠️" in status or "🔍" in status or "✨" in status:
                # Try to extract agent name and progress info
                import re
                
                # Look for agent names in status
                agent_match = re.search(r'Agent-([A-Z]+)', status)
                if agent_match:
                    agent_name = agent_match.group(1).lower()
                    
                    # Update or create agent progress entry
                    if agent_name not in self._agent_progress:
                        self._agent_progress[agent_name] = {
                            "name": agent_name,
                            "status": "in_progress",
                            "progress": 10.0,
                            "step": "",
                            "elapsed_ms": 0
                        }
                    
                    # Update progress based on status content
                    progress_info = self._agent_progress[agent_name]
                    
                    if "executing" in status.lower():
                        progress_info["progress"] = min(progress_info["progress"] + 20, 90)
                        progress_info["step"] = "Executing"
                    elif "analyzing" in status.lower():
                        progress_info["progress"] = min(progress_info["progress"] + 15, 70)
                        progress_info["step"] = "Analyzing"
                    elif "completed" in status.lower() or "done" in status.lower():
                        progress_info["progress"] = 100.0
                        progress_info["status"] = "completed"
                        progress_info["step"] = "Completed"
                    elif "error" in status.lower() or "failed" in status.lower():
                        progress_info["status"] = "error"
                        progress_info["step"] = "Error"
            
            # Clean up old completed agents after some time
            self._cleanup_old_agent_progress()
            
        except Exception:
            pass  # Ignore parsing errors
    
    def _cleanup_old_agent_progress(self) -> None:
        """Clean up old completed agent progress entries."""
        try:
            import time
            current_time = time.time()
            
            # Remove completed agents after 30 seconds
            agents_to_remove = []
            for agent_id, progress_info in self._agent_progress.items():
                if progress_info.get("status") == "completed":
                    # Add cleanup timestamp if not present
                    if "cleanup_time" not in progress_info:
                        progress_info["cleanup_time"] = current_time
                    elif current_time - progress_info["cleanup_time"] > 30:
                        agents_to_remove.append(agent_id)
            
            for agent_id in agents_to_remove:
                del self._agent_progress[agent_id]
                
        except Exception:
            pass  # Ignore cleanup errors
    
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
    
    def set_progress_display(self, progress_display: ProgressDisplay) -> None:
        """Set the progress display system for enhanced agent status tracking."""
        self._progress_display = progress_display
        
        # Set up callback for progress updates
        if progress_display:
            progress_display.update_callback = self._on_progress_update
    
    def _on_progress_update(self, progress_text: str) -> None:
        """Handle progress updates from the progress display system."""
        try:
            # Update status panel when progress changes
            if self.live and self.layout:
                self._update_status_panel()
        except Exception:
            pass  # Ignore progress update errors
    
    def update_agent_progress(self, agent_id: str, progress: float, step: str = None, status: str = "in_progress") -> None:
        """Update progress for a specific agent."""
        try:
            if agent_id not in self._agent_progress:
                self._agent_progress[agent_id] = {
                    "name": agent_id,
                    "status": status,
                    "progress": 0.0,
                    "step": "",
                    "elapsed_ms": 0
                }
            
            progress_info = self._agent_progress[agent_id]
            progress_info["progress"] = min(100.0, max(0.0, progress))
            progress_info["status"] = status
            if step:
                progress_info["step"] = step
            
            # Update the status panel
            if self.live and self.layout:
                self._update_status_panel()
                
        except Exception as e:
            print(f"Agent progress update error: {e}")
    
    def complete_task_display(self) -> None:
        """
        Complete the current task display and stop progress updates.
        This addresses the endless status loop in Rich TUI mode.
        """
        try:
            # Complete the connected ProgressDisplay if available
            if self._progress_display:
                self._progress_display.complete_task()
            
            # Mark all agents as completed
            for agent_info in self._agent_progress.values():
                agent_info["status"] = "completed"
                agent_info["progress"] = 100.0
            
            # Update the status to show task completion
            self._current_status = "✅ Task completed successfully"
            
            # Perform final status panel update
            if self.live and self.layout:
                self._update_status_panel()
                
        except Exception as e:
            print(f"Task completion error: {e}")