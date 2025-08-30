"""
Chat History Display Components

Provides enhanced chat history display functionality for AgentsMCP's terminal interface.
Handles message formatting, code highlighting, and conversation management.
"""

from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field

try:
    from rich.console import Console, RenderableType
    from rich.text import Text
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.align import Align
    from rich.console import Group
    from rich import box
except ImportError:  # pragma: no cover
    Console = None
    RenderableType = None
    Text = None
    Panel = None
    Syntax = None
    Align = None
    Group = None
    box = None


@dataclass
class ChatMessage:
    """Represents a single chat message with metadata."""
    content: str
    message_type: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChatHistoryDisplay:
    """Enhanced chat history display with rich formatting and code highlighting.
    
    Provides conversation history management with support for different message types,
    syntax highlighting, and graceful fallback when Rich is not available.
    """
    
    # Message type styling
    MESSAGE_STYLES = {
        'user': 'cyan bold',
        'assistant': 'white',
        'system': 'dim yellow',
        'error': 'red bold',
        'tool': 'green',
    }
    
    def __init__(self, console: Optional[Console] = None, max_history: int = 100):
        """Initialize chat history display.
        
        Args:
            console: Rich console instance for rendering
            max_history: Maximum number of messages to keep in history
        """
        self.console = console
        self.max_history = max_history
        self._messages: List[ChatMessage] = []
        self._code_block_pattern = re.compile(r'```(\w+)?\n(.*?)```', re.DOTALL)
        
    def add_message(
        self, 
        content: str, 
        message_type: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message to the chat history.
        
        Args:
            content: Message content
            message_type: Type of message (user, assistant, system, error, tool)
            metadata: Optional metadata associated with the message
        """
        if message_type not in self.MESSAGE_STYLES:
            raise ValueError(f"Unknown message type: {message_type}")
            
        message = ChatMessage(
            content=content.strip(),
            message_type=message_type,
            metadata=metadata or {}
        )
        
        self._messages.append(message)
        
        # Maintain history limit
        if len(self._messages) > self.max_history:
            self._messages = self._messages[-self.max_history:]
    
    def clear_history(self) -> None:
        """Clear all messages from history."""
        self._messages.clear()
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent messages.
        
        Args:
            count: Number of recent messages to return
            
        Returns:
            List of message dictionaries
        """
        recent = self._messages[-count:] if count > 0 else self._messages
        return [
            {
                'content': msg.content,
                'message_type': msg.message_type,
                'timestamp': msg.timestamp,
                'metadata': msg.metadata
            }
            for msg in recent
        ]
    
    def filter_messages(
        self, 
        message_type: Optional[str] = None,
        since: Optional[datetime.datetime] = None
    ) -> List[Dict[str, Any]]:
        """Filter messages by type and/or timestamp.
        
        Args:
            message_type: Filter by message type
            since: Filter messages since this timestamp
            
        Returns:
            List of filtered message dictionaries
        """
        filtered = self._messages
        
        if message_type:
            if message_type not in self.MESSAGE_STYLES:
                raise ValueError(f"Unknown message type: {message_type}")
            filtered = [msg for msg in filtered if msg.message_type == message_type]
        
        if since:
            filtered = [msg for msg in filtered if msg.timestamp >= since]
        
        return [
            {
                'content': msg.content,
                'message_type': msg.message_type,
                'timestamp': msg.timestamp,
                'metadata': msg.metadata
            }
            for msg in filtered
        ]
    
    def render_history(self, height: Optional[int] = None) -> Union[RenderableType, str]:
        """Render the chat history for display.
        
        Args:
            height: Optional height limit for the display
            
        Returns:
            Rich renderable or plain string if Rich not available
        """
        if not Console:  # Rich not available
            return self._render_plain_text()
        
        if not self._messages:
            empty_text = Text("No conversation history yet", style="dim")
            return Panel(empty_text, title="Chat History", border_style="dim")
        
        # Build renderable elements for each message
        elements = []
        
        for msg in self._messages:
            elements.append(self._render_message(msg))
        
        # Create the history group
        if Group:
            history_content = Group(*elements)
        else:
            # Fallback if Group not available
            history_content = elements[0] if elements else Text("", style="dim")
        
        # Wrap in panel
        panel_kwargs = {
            'title': 'Chat History',
            'border_style': 'dim',
            'padding': (0, 1)
        }
        
        if height:
            panel_kwargs['height'] = height
        
        if box and hasattr(box, 'ROUNDED'):
            panel_kwargs['box'] = box.ROUNDED
        
        return Panel(history_content, **panel_kwargs)
    
    def _render_message(self, message: ChatMessage) -> Union[RenderableType, str]:
        """Render a single message with appropriate formatting."""
        if not Text:
            return f"[{message.message_type.upper()}] {message.content}"
        
        style = self.MESSAGE_STYLES.get(message.message_type, 'white')
        
        # Check for code blocks
        if self._has_code_blocks(message.content):
            return self._render_message_with_code(message.content, style)
        
        # Regular message
        text = Text()
        text.append(f"[{message.message_type.upper()}] ", style="dim")
        text.append(message.content, style=style)
        text.overflow = "fold"
        
        return text
    
    def _render_message_with_code(self, content: str, base_style: str) -> Union[RenderableType, str]:
        """Render message containing code blocks with syntax highlighting."""
        if not Text or not Syntax:
            return content
        
        parts = []
        last_end = 0
        
        # Find and process code blocks
        for match in self._code_block_pattern.finditer(content):
            # Add text before code block
            before = content[last_end:match.start()]
            if before:
                parts.append(Text(before, style=base_style))
            
            # Add syntax highlighted code
            language = match.group(1) or 'text'
            code = match.group(2)
            
            try:
                syntax = Syntax(
                    code,
                    language,
                    theme="monokai",
                    line_numbers=False,
                    word_wrap=True
                )
                parts.append(syntax)
            except Exception:
                # Fallback to plain code display
                parts.append(Text(code, style="bold white on black"))
            
            last_end = match.end()
        
        # Add remaining text after last code block
        remaining = content[last_end:]
        if remaining:
            parts.append(Text(remaining, style=base_style))
        
        # Return group or first element
        if Group and len(parts) > 1:
            return Group(*parts)
        elif parts:
            return parts[0]
        else:
            return Text(content, style=base_style)
    
    def _has_code_blocks(self, content: str) -> bool:
        """Check if content contains code blocks."""
        return bool(self._code_block_pattern.search(content))
    
    def _render_plain_text(self) -> str:
        """Render history as plain text when Rich is not available."""
        if not self._messages:
            return "No conversation history yet"
        
        lines = []
        for msg in self._messages:
            timestamp = msg.timestamp.strftime("%H:%M:%S")
            lines.append(f"[{timestamp}] [{msg.message_type.upper()}] {msg.content}")
        
        return "\n".join(lines)
    
    def show_typing_indicator(self, message: str = "AI is thinking...") -> None:
        """Show visual indicator that AI is processing."""
        if self.console:
            self.console.print(f"[dim yellow]{message}[/dim yellow]")
    
    def clear_typing_indicator(self) -> None:
        """Clear the typing indicator."""
        if self.console:
            # Use Rich's proper method to control the display instead of raw escape sequences
            # In a TUI context, the typing indicator should be handled by the main rendering loop
            pass