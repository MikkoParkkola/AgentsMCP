"""Message formatting utilities for different UI contexts."""

from typing import List, Dict, Any
from .chat_engine import ChatMessage, MessageRole
import textwrap


class MessageFormatter:
    """Formats chat messages for different UI contexts."""
    
    @staticmethod
    def format_for_plain_cli(message: ChatMessage, width: int = 80) -> str:
        """Format message for plain CLI display."""
        role_prefixes = {
            MessageRole.USER: "👤 You:",
            MessageRole.ASSISTANT: "🤖 AI:",
            MessageRole.SYSTEM: "ℹ️ System:"
        }
        
        prefix = role_prefixes.get(message.role, "❓ Unknown:")
        
        # Wrap content to fit terminal width
        wrapped_lines = textwrap.wrap(
            message.content, 
            width=width-10,  # Leave room for prefix
            initial_indent="",
            subsequent_indent="    "
        )
        
        if not wrapped_lines:
            return f"{prefix} [empty message]"
        
        result = f"{prefix} {wrapped_lines[0]}"
        for line in wrapped_lines[1:]:
            result += f"\n    {line}"
        
        return result
    
    @staticmethod  
    def format_for_simple_tui(message: ChatMessage, width: int = 80) -> str:
        """Format message for simple TUI display."""
        role_symbols = {
            MessageRole.USER: "👤",
            MessageRole.ASSISTANT: "🤖", 
            MessageRole.SYSTEM: "ℹ️"
        }
        
        symbol = role_symbols.get(message.role, "❓")
        
        # Simple format with symbol
        wrapped_lines = textwrap.wrap(message.content, width=width-4)
        if not wrapped_lines:
            return f"{symbol} [empty message]"
        
        result = f"{symbol} {wrapped_lines[0]}"
        for line in wrapped_lines[1:]:
            result += f"\n  {line}"
            
        return result
    
    @staticmethod
    def format_conversation_summary(messages: List[ChatMessage], max_messages: int = 10) -> str:
        """Format a summary of recent conversation."""
        if not messages:
            return "No conversation history."
        
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
        
        summary_lines = [f"Recent Conversation ({len(recent_messages)} messages):"]
        
        for i, msg in enumerate(recent_messages, 1):
            role_name = msg.role.value.capitalize()
            preview = msg.content  # Show full content, no truncation
            summary_lines.append(f"  {i}. {role_name}: {preview}")
        
        if len(messages) > max_messages:
            summary_lines.append(f"... and {len(messages) - max_messages} earlier messages")
        
        return "\n".join(summary_lines)


class StatusFormatter:
    """Formats status messages for different UI contexts."""
    
    @staticmethod
    def format_processing_status(is_processing: bool, custom_message: str = "") -> str:
        """Format processing status message."""
        if is_processing:
            return custom_message or "⏳ Processing your message..."
        else:
            return custom_message or "✅ Ready for your input"
    
    @staticmethod
    def format_error_status(error: str) -> str:
        """Format error status message."""
        return f"❌ Error: {error}"
    
    @staticmethod
    def format_success_status(message: str) -> str:
        """Format success status message."""
        return f"✅ {message}"