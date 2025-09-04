"""Console message formatter - handles Rich text formatting for chat messages."""

import textwrap
from typing import Optional
from rich.console import Console
from rich.text import Text
from rich.padding import Padding


class ConsoleMessageFormatter:
    """Formats and displays chat messages with Rich styling in console flow."""
    
    def __init__(self, console: Console):
        self.console = console
        # Calculate safe text width (leave margin for icons and styling)
        self.text_width = max(50, console.size.width - 8) if console.size.width else 72
    
    def format_and_display_message(self, role: str, content: str, timestamp: str = None) -> None:
        """Format and display a chat message with proper styling and wrapping."""
        try:
            # Add timestamp prefix if provided
            time_prefix = f"{timestamp} " if timestamp else ""
            
            # Format role-specific styling
            if role == "user":
                icon_text = Text.assemble((time_prefix, "dim"), ("👤 You: ", "bold blue"))
                content_style = "white"
            elif role == "assistant":
                icon_text = Text.assemble((time_prefix, "dim"), ("🤖 AI: ", "bold green"))
                content_style = "white"
            elif role == "system":
                icon_text = Text.assemble((time_prefix, "dim"), ("ℹ️ System: ", "bold yellow"))
                content_style = "yellow"
                # Special handling for help messages
                if "Commands:" in content and "• /" in content:
                    self.console.print(icon_text)
                    self.format_help_message(content)
                    self.console.print()
                    return
            else:
                icon_text = Text.assemble((time_prefix, "dim"), (f"❓ {role}: ", "dim"))
                content_style = "dim"
            
            # Display the icon/role line
            self.console.print(icon_text)
            
            # Handle content formatting with proper wrapping
            self._display_wrapped_content(content, content_style)
            
            # Add spacing between messages
            self.console.print()
            
        except Exception as e:
            print(f"Message formatting error: {e}")
    
    def _display_wrapped_content(self, content: str, style: str) -> None:
        """Display content with proper text wrapping and indentation."""
        try:
            # Handle existing line breaks first
            content_lines = content.split('\n')
            
            for line in content_lines:
                if not line.strip():
                    # Preserve empty lines
                    self.console.print()
                    continue
                
                # Wrap long lines at word boundaries
                if len(line) > self.text_width:
                    wrapped_lines = textwrap.wrap(
                        line, 
                        width=self.text_width,
                        break_long_words=False,
                        break_on_hyphens=False
                    )
                    for wrapped_line in wrapped_lines:
                        # Indent content slightly for better readability
                        indented_text = Text(f"  {wrapped_line}", style=style)
                        self.console.print(indented_text)
                else:
                    # Indent content slightly for better readability
                    indented_text = Text(f"  {line}", style=style)
                    self.console.print(indented_text)
                    
        except Exception as e:
            # Fallback to simple display
            self.console.print(f"  {content}", style=style)
    
    def format_help_message(self, help_text: str) -> None:
        """Format help message with proper bullet point handling."""
        try:
            lines = help_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    self.console.print()
                    continue
                
                # Handle bullet points
                if line.startswith('• '):
                    # Format bullet points with proper styling
                    command_part = line[2:].split(' - ', 1)
                    if len(command_part) == 2:
                        command, description = command_part
                        bullet_text = Text.assemble(
                            ("  • ", "cyan"),
                            (command, "bold cyan"),
                            (" - ", "white"),
                            (description, "dim white")
                        )
                        self.console.print(bullet_text)
                    else:
                        # Simple bullet point
                        bullet_text = Text.assemble(
                            ("  • ", "cyan"),
                            (line[2:], "white")
                        )
                        self.console.print(bullet_text)
                else:
                    # Regular text line
                    if len(line) > self.text_width:
                        wrapped_lines = textwrap.wrap(line, width=self.text_width)
                        for wrapped_line in wrapped_lines:
                            self.console.print(f"  {wrapped_line}", style="white")
                    else:
                        self.console.print(f"  {line}", style="white")
            
        except Exception as e:
            # Fallback to simple display
            self.console.print(f"  {help_text}", style="white")
    
    def format_system_info(self, info_text: str) -> None:
        """Format system information messages."""
        try:
            # System info gets special formatting
            info_lines = info_text.split('\n')
            
            for line in info_lines:
                line = line.strip()
                if not line:
                    self.console.print()
                    continue
                
                # Handle special formatting for status info
                if line.startswith('•'):
                    info_text = Text.assemble(
                        ("  ", ""),
                        (line, "dim cyan")
                    )
                    self.console.print(info_text)
                else:
                    self.console.print(f"  {line}", style="dim cyan")
                    
        except Exception as e:
            self.console.print(f"  {info_text}", style="dim cyan")