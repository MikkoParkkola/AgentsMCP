"""Core chat engine - business logic separated from UI concerns."""

import asyncio
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import time


class MessageRole(Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """A single message in the chat conversation."""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            'role': self.role.value,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class ChatState:
    """Current state of the chat engine."""
    messages: List[ChatMessage] = field(default_factory=list)
    is_processing: bool = False
    last_error: Optional[str] = None
    session_id: str = field(default_factory=lambda: f"session_{int(time.time())}")
    
    def add_message(self, role: MessageRole, content: str, **metadata) -> ChatMessage:
        """Add a message to the conversation."""
        message = ChatMessage(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        return message
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history as list of dicts."""
        return [msg.to_dict() for msg in self.messages]
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages.clear()


class ChatEngine:
    """Core chat engine handling AI conversation logic."""
    
    def __init__(self):
        self.state = ChatState()
        self._status_callback: Optional[Callable[[str], None]] = None
        self._message_callback: Optional[Callable[[ChatMessage], None]] = None
        self._error_callback: Optional[Callable[[str], None]] = None
        
        # Built-in commands
        self.commands = {
            '/help': self._handle_help_command,
            '/quit': self._handle_quit_command,
            '/clear': self._handle_clear_command,
            '/history': self._handle_history_command,
            '/status': self._handle_status_command
        }
    
    def set_callbacks(self, 
                     status_callback: Optional[Callable[[str], None]] = None,
                     message_callback: Optional[Callable[[ChatMessage], None]] = None,
                     error_callback: Optional[Callable[[str], None]] = None):
        """Set callbacks for UI updates."""
        self._status_callback = status_callback
        self._message_callback = message_callback
        self._error_callback = error_callback
    
    def _notify_status(self, status: str) -> None:
        """Notify UI of status change."""
        if self._status_callback:
            self._status_callback(status)
    
    def _notify_message(self, message: ChatMessage) -> None:
        """Notify UI of new message."""
        if self._message_callback:
            self._message_callback(message)
    
    def _notify_error(self, error: str) -> None:
        """Notify UI of error."""
        self.state.last_error = error
        if self._error_callback:
            self._error_callback(error)
    
    async def process_input(self, user_input: str) -> bool:
        """
        Process user input and return True if should continue, False if should quit.
        """
        try:
            user_input = user_input.strip()
            if not user_input:
                return True
            
            # Handle built-in commands
            if user_input.startswith('/'):
                return await self._handle_command(user_input)
            
            # Handle regular chat message
            return await self._handle_chat_message(user_input)
            
        except Exception as e:
            self._notify_error(f"Error processing input: {str(e)}")
            return True  # Continue despite error
    
    async def _handle_command(self, command_input: str) -> bool:
        """Handle built-in commands."""
        parts = command_input.split(' ', 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command in self.commands:
            return await self.commands[command](args)
        else:
            self._notify_error(f"Unknown command: {command}. Type /help for available commands.")
            return True
    
    async def _handle_chat_message(self, user_input: str) -> bool:
        """Handle regular chat message."""
        try:
            # Add user message to history
            user_message = self.state.add_message(MessageRole.USER, user_input)
            self._notify_message(user_message)
            
            # Set processing state
            self.state.is_processing = True
            self._notify_status("Processing your message...")
            
            # Simulate AI processing (replace with actual AI call)
            response = await self._get_ai_response(user_input)
            
            # Add AI response to history
            ai_message = self.state.add_message(MessageRole.ASSISTANT, response)
            self._notify_message(ai_message)
            
            # Clear processing state
            self.state.is_processing = False
            self._notify_status("Ready")
            
            return True
            
        except Exception as e:
            self.state.is_processing = False
            self._notify_error(f"Error getting AI response: {str(e)}")
            return True
    
    async def _get_ai_response(self, user_input: str) -> str:
        """
        Get AI response to user input using the real LLMClient.
        """
        try:
            # Import and initialize the real LLM client
            from ...conversation.llm_client import LLMClient
            
            # Initialize LLM client if not already done
            if not hasattr(self, '_llm_client'):
                # Set TUI mode to prevent console contamination
                import os
                os.environ['AGENTSMCP_TUI_MODE'] = '1'
                self._llm_client = LLMClient()
            
            # Get response from real LLM
            response = await self._llm_client.send_message(user_input)
            return response
            
        except Exception as e:
            # Fallback to simple response if LLM fails
            import logging
            logging.error(f"LLM client failed: {e}")
            
            # Provide helpful fallback responses
            if "hello" in user_input.lower():
                return "Hello! How can I help you today?"
            elif any(word in user_input.lower() for word in ["help", "commands"]):
                return "I'm an AI assistant. You can ask me questions or have a conversation. Type /help for available commands."
            elif any(word in user_input.lower() for word in ["agent", "role"]):
                return "I have access to various specialized agents including coding agents, QA engineers, and other roles. What would you like me to help you with?"
            else:
                return f"I understand you're asking about: \"{user_input}\". I'm having trouble connecting to the full AI system right now, but I can still help with basic commands. Try typing /help for available options."
    
    async def _handle_help_command(self, args: str) -> bool:
        """Handle /help command."""
        help_message = """Available Commands:
• /help - Show this help message
• /quit - Exit the application  
• /clear - Clear conversation history
• /history - Show conversation history
• /status - Show current status

Just type your message and press Enter to chat with the AI!"""
        
        help_msg = self.state.add_message(MessageRole.SYSTEM, help_message)
        self._notify_message(help_msg)
        return True
    
    async def _handle_quit_command(self, args: str) -> bool:
        """Handle /quit command."""
        self._notify_status("Goodbye!")
        return False  # Signal to quit
    
    async def _handle_clear_command(self, args: str) -> bool:
        """Handle /clear command."""
        message_count = len(self.state.messages)
        self.state.clear_history()
        
        clear_msg = self.state.add_message(
            MessageRole.SYSTEM, 
            f"Cleared {message_count} messages from conversation history."
        )
        self._notify_message(clear_msg)
        return True
    
    async def _handle_history_command(self, args: str) -> bool:
        """Handle /history command."""
        if not self.state.messages:
            history_msg = self.state.add_message(
                MessageRole.SYSTEM,
                "No conversation history available."
            )
        else:
            history_text = f"Conversation History ({len(self.state.messages)} messages):\n"
            for i, msg in enumerate(self.state.messages[-10:], 1):  # Show last 10
                role_symbol = {"user": "👤", "assistant": "🤖", "system": "ℹ️"}
                symbol = role_symbol.get(msg.role.value, "❓")
                history_text += f"{i}. {symbol} {msg.content}\n"  # Show full content
            
            history_msg = self.state.add_message(MessageRole.SYSTEM, history_text.strip())
        
        self._notify_message(history_msg)
        return True
    
    async def _handle_status_command(self, args: str) -> bool:
        """Handle /status command."""
        status_info = f"""Current Status:
• Session ID: {self.state.session_id}
• Messages: {len(self.state.messages)}
• Processing: {self.state.is_processing}
• Last Error: {self.state.last_error or 'None'}"""
        
        status_msg = self.state.add_message(MessageRole.SYSTEM, status_info)
        self._notify_message(status_msg)
        return True
    
    def get_state(self) -> ChatState:
        """Get current chat state."""
        return self.state
    
    def is_processing(self) -> bool:
        """Check if currently processing a message."""
        return self.state.is_processing


class MockAIProvider:
    """Mock AI provider for testing and development."""
    
    def __init__(self):
        self.responses = [
            "That's an interesting question! Let me think about that.",
            "I can help you with that. Here's what I think...",
            "That's a great point. Have you considered...",
            "I understand what you're asking. In my experience...",
            "That reminds me of something similar. Let me explain..."
        ]
        self.response_index = 0
    
    async def get_response(self, user_input: str, conversation_history: List[Dict[str, Any]]) -> str:
        """Get mock AI response."""
        await asyncio.sleep(0.3)  # Simulate processing time
        
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        
        return response