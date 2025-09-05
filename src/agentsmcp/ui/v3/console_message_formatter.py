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
                # Special handling for feature showcase messages
                if content.startswith("FEATURE_SHOWCASE_FORMAT:"):
                    showcase_content = content[24:]  # Remove "FEATURE_SHOWCASE_FORMAT:" prefix
                    self.format_feature_showcase(showcase_content)
                    return
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
        """Format help message with proper markdown rendering and rich formatting."""
        try:
            # Use Rich's Markdown renderer for proper formatting
            from rich.markdown import Markdown
            
            # Create markdown object with proper rendering
            markdown = Markdown(help_text)
            
            # Display the rendered markdown
            self.console.print(markdown)
            
        except ImportError:
            # Fallback to manual formatting if Rich markdown not available
            self._format_help_manual(help_text)
        except Exception as e:
            # Fallback to manual formatting on any error
            self._format_help_manual(help_text)
    
    def _format_help_manual(self, help_text: str) -> None:
        """Manual formatting for help message if Rich markdown fails."""
        try:
            lines = help_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    self.console.print()
                    continue
                
                # Handle headers with **bold** markdown
                if line.startswith('💬 **') and line.endswith(':**'):
                    # Extract header text and make it bold
                    header_text = line[4:-3]  # Remove emoji and markdown
                    header_display = Text.assemble(
                        ("💬 ", "blue"),
                        (header_text + ":", "bold blue")
                    )
                    self.console.print(header_display)
                    continue
                    
                # Handle other sections with emoji headers
                if '**' in line and line.endswith(':**'):
                    # Find emoji and header
                    parts = line.split(' **', 1)
                    if len(parts) == 2:
                        emoji_part = parts[0]
                        header_part = parts[1][:-3]  # Remove ":**"
                        header_display = Text.assemble(
                            (emoji_part + " ", "yellow"),
                            (header_part + ":", "bold yellow")
                        )
                        self.console.print(header_display)
                        continue
                
                # Handle bullet points
                if line.startswith('• '):
                    # Parse command and description
                    bullet_content = line[2:]  # Remove bullet
                    if ' - ' in bullet_content:
                        command, description = bullet_content.split(' - ', 1)
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
                            (bullet_content, "white")
                        )
                        self.console.print(bullet_text)
                elif line.startswith('  • '):
                    # Indented bullet (sub-bullet)
                    bullet_content = line[4:]  # Remove indented bullet
                    if ' - ' in bullet_content:
                        command, description = bullet_content.split(' - ', 1)
                        bullet_text = Text.assemble(
                            ("    • ", "dim cyan"),
                            (command, "cyan"),
                            (" - ", "dim white"),
                            (description, "dim white")
                        )
                        self.console.print(bullet_text)
                    else:
                        bullet_text = Text.assemble(
                            ("    • ", "dim cyan"),
                            (bullet_content, "dim white")
                        )
                        self.console.print(bullet_text)
                else:
                    # Regular text line - handle **bold** markdown
                    if '**' in line:
                        # Simple bold text handling
                        formatted_line = Text()
                        parts = line.split('**')
                        for i, part in enumerate(parts):
                            if i % 2 == 0:
                                # Regular text
                                formatted_line.append(f"  {part}", style="white")
                            else:
                                # Bold text
                                formatted_line.append(part, style="bold white")
                        self.console.print(formatted_line)
                    else:
                        # Plain text
                        if len(line) > self.text_width:
                            wrapped_lines = textwrap.wrap(line, width=self.text_width)
                            for wrapped_line in wrapped_lines:
                                self.console.print(f"  {wrapped_line}", style="white")
                        else:
                            self.console.print(f"  {line}", style="white")
            
        except Exception as e:
            # Ultimate fallback to simple display
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
    
    def format_feature_showcase(self, showcase_message: str) -> None:
        """Format feature showcase message with Rich markdown rendering for already-existing features."""
        try:
            # Use Rich's Markdown renderer for proper formatting
            from rich.markdown import Markdown
            from rich.panel import Panel
            
            # Create markdown object with proper rendering
            markdown = Markdown(showcase_message)
            
            # Wrap in a panel for visual distinction
            panel = Panel(
                markdown,
                title="🎯 Feature Already Available", 
                title_align="left",
                border_style="green",
                padding=(0, 1)
            )
            
            # Display the formatted showcase
            self.console.print(panel)
            
        except ImportError:
            # Fallback to manual formatting if Rich markdown not available
            self._format_showcase_manual(showcase_message)
        except Exception as e:
            # Fallback to manual formatting on any error
            self._format_showcase_manual(showcase_message)
    
    def _format_showcase_manual(self, showcase_message: str) -> None:
        """Manual formatting for feature showcase if Rich markdown fails."""
        try:
            lines = showcase_message.split('\n')
            
            # Header
            header_text = Text.assemble(
                ("🎯 ", "green"),
                ("Feature Already Available", "bold green")
            )
            self.console.print(header_text)
            self.console.print()
            
            for line in lines:
                line = line.strip()
                if not line:
                    self.console.print()
                    continue
                
                # Skip markdown header that we already displayed
                if line == "🎯 **Feature Already Available**":
                    continue
                
                # Handle checkmark lines
                if line.startswith('✅'):
                    check_text = Text.assemble(
                        ("  ", ""),
                        (line, "bold green")
                    )
                    self.console.print(check_text)
                    continue
                
                # Handle **Try it now:** sections
                if line.startswith('**Try it now:**'):
                    try_text = Text.assemble(
                        ("  ", ""),
                        ("Try it now:", "bold yellow")
                    )
                    self.console.print(try_text)
                    continue
                
                # Handle **Related features:** sections  
                if line.startswith('**Related features'):
                    related_text = Text.assemble(
                        ("  ", ""),
                        ("Related features you might like:", "bold cyan")
                    )
                    self.console.print(related_text)
                    continue
                
                # Handle **Detection evidence:** sections
                if line.startswith('**Detection evidence:**'):
                    evidence_text = Text.assemble(
                        ("  ", ""),
                        ("Detection evidence:", "bold blue")
                    )
                    self.console.print(evidence_text)
                    continue
                
                # Handle code blocks
                if line.startswith('```bash'):
                    continue  # Skip opening code fence
                elif line.startswith('```'):
                    continue  # Skip closing code fence
                elif line.startswith('$ '):
                    # Command examples
                    command_text = Text.assemble(
                        ("    ", ""),
                        (line, "bold magenta")
                    )
                    self.console.print(command_text)
                    continue
                
                # Handle bullet points
                if line.startswith('• '):
                    bullet_content = line[2:]  # Remove bullet
                    if bullet_content.startswith('`') and bullet_content.endswith('`'):
                        # Code bullet
                        code_content = bullet_content[1:-1]  # Remove backticks
                        bullet_text = Text.assemble(
                            ("    • ", "cyan"),
                            (code_content, "bold cyan")
                        )
                    else:
                        bullet_text = Text.assemble(
                            ("    • ", "cyan"),
                            (bullet_content, "white")
                        )
                    self.console.print(bullet_text)
                    continue
                
                # Regular text - handle **bold** markdown
                if '**' in line:
                    formatted_line = Text()
                    parts = line.split('**')
                    for i, part in enumerate(parts):
                        if i % 2 == 0:
                            # Regular text
                            formatted_line.append(f"  {part}" if i == 0 else part, style="white")
                        else:
                            # Bold text
                            formatted_line.append(part, style="bold white")
                    self.console.print(formatted_line)
                else:
                    # Plain text
                    if len(line) > self.text_width:
                        wrapped_lines = textwrap.wrap(line, width=self.text_width)
                        for wrapped_line in wrapped_lines:
                            self.console.print(f"  {wrapped_line}", style="white")
                    else:
                        self.console.print(f"  {line}", style="white")
            
            # Add bottom spacing
            self.console.print()
            
        except Exception as e:
            # Ultimate fallback to simple display
            self.console.print(f"  {showcase_message}", style="white")