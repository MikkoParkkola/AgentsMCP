"""Core chat engine - business logic separated from UI concerns."""

import asyncio
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime


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
        
        # Initialize LLMClient once to preserve conversation history
        self._llm_client = None
        self._initialize_llm_client()
        
        # Built-in commands with new diagnostic and control commands
        self.commands = {
            '/help': self._handle_help_command,
            '/quit': self._handle_quit_command,
            '/clear': self._handle_clear_command,
            '/history': self._handle_history_command,
            '/status': self._handle_status_command,
            '/config': self._handle_config_command,
            '/providers': self._handle_providers_command,
            '/preprocessing': self._handle_preprocessing_command
        }
    
    @staticmethod
    def _format_timestamp() -> str:
        """Format current time as [hh:mm:ss] timestamp."""
        return datetime.now().strftime("[%H:%M:%S]")
    
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
    
    def _initialize_llm_client(self) -> None:
        """Initialize LLM client once and preserve it throughout the session."""
        try:
            # Set TUI mode to prevent console contamination
            import os
            os.environ['AGENTSMCP_TUI_MODE'] = '1'
            
            # Import and create LLMClient only once
            from ...conversation.llm_client import LLMClient
            self._llm_client = LLMClient()
        except Exception as e:
            import logging
            logging.error(f"Failed to initialize LLM client: {e}")
            self._llm_client = None
    
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
        Get AI response to user input using the real LLMClient with detailed error reporting.
        """
        try:
            # Use the existing LLM client (initialized once in __init__)
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                return "❌ Failed to initialize LLM client. Please check your configuration with /config command."
            
            # Get response from real LLM - it now handles its own error reporting
            response = await self._llm_client.send_message(user_input)
            return response
            
        except Exception as e:
            # This should rarely happen now since LLMClient handles most errors internally
            import logging
            logging.error(f"Unexpected error in chat engine: {e}")
            
            return f"❌ Unexpected system error: {str(e)}\n\n💡 This may indicate a system-level issue. Try:\n  • Restarting the TUI\n  • Checking your terminal environment\n  • Running with debug mode enabled"
    
    async def _handle_help_command(self, args: str) -> bool:
        """Handle /help command."""
        help_message = """🤖 AI Command Composer - Help
==================================

💬 **Chat Commands:**
• /help - Show this help message
• /quit - Exit the application  
• /clear - Clear conversation history
• /history - Show conversation history

🔧 **Diagnostic Commands:**
• /status - Show basic system status
• /config - Show detailed LLM configuration
• /providers - Show LLM provider status
• /preprocessing [on/off/toggle/status] - Control preprocessing mode

🚀 **Quick Setup Guide:**
If you're getting connection errors:

1. **Check Configuration**: `/config`
2. **See Available Providers**: `/providers`
3. **Set up a Provider**:
   • OpenAI: `export OPENAI_API_KEY=your_key`
   • Anthropic: `export ANTHROPIC_API_KEY=your_key` 
   • Ollama: `ollama serve` (free, runs locally)
   • OpenRouter: `export OPENROUTER_API_KEY=your_key`

📊 **Preprocessing Modes:**
• **On** (default): Multi-turn tool execution, slower but more capable
• **Off**: Direct LLM responses only, faster but simpler

💡 **Tips:**
• Type `/config` if you see connection errors
• Use `/preprocessing off` for faster responses
• All environment variables should be set before starting TUI"""
        
        self._notify_message(ChatMessage(
            role=MessageRole.ASSISTANT,
            content=help_message,
            timestamp=self._format_timestamp()
        ))
        return True
    
    async def _handle_quit_command(self, args: str) -> bool:
        """Handle /quit command."""
        # Don't show goodbye here - let TUI launcher handle it
        return False  # Signal to quit
    
    async def _handle_clear_command(self, args: str) -> bool:
        """Handle /clear command."""
        message_count = len(self.state.messages)
        self.state.clear_history()
        
        # Also clear the LLMClient's conversation history to keep them in sync
        if self._llm_client is not None:
            self._llm_client.conversation_history.clear()
        
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

    async def _handle_config_command(self, args: str) -> bool:
        """Handle /config command to show detailed configuration status."""
        try:
            # Use the existing LLM client (initialized once in __init__)
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                self._notify_error("Failed to initialize LLM client")
                return True
            
            # Get configuration status
            config_status = self._llm_client.get_configuration_status()
            
            # Build detailed status message
            status_msg = "🔧 LLM Configuration Status\n"
            status_msg += "=" * 40 + "\n\n"
            
            # Current settings
            status_msg += f"📊 Current Settings:\n"
            status_msg += f"  • Provider: {config_status['current_provider']}\n"
            status_msg += f"  • Model: {config_status['current_model']}\n"
            status_msg += f"  • Preprocessing: {'✅ Enabled' if config_status['preprocessing_enabled'] else '❌ Disabled'}\n"
            status_msg += f"  • MCP Tools: {'✅ Available' if config_status['mcp_tools_available'] else '❌ Not Available'}\n\n"
            
            # Provider status
            status_msg += "🔌 Provider Status:\n"
            for provider, pstatus in config_status['providers'].items():
                icon = "✅" if pstatus['configured'] else "❌"
                status_msg += f"  {icon} {provider.upper()}:\n"
                
                if provider == "ollama":
                    service_icon = "✅" if pstatus['service_available'] else "❌"
                    status_msg += f"      Service: {service_icon} {'Running' if pstatus['service_available'] else 'Not Running'}\n"
                    if not pstatus['service_available']:
                        status_msg += f"      💡 Start with: ollama serve\n"
                else:
                    key_icon = "✅" if pstatus['api_key_present'] else "❌"
                    status_msg += f"      API Key: {key_icon} {'Configured' if pstatus['api_key_present'] else 'Missing'}\n"
                    if not pstatus['api_key_present']:
                        status_msg += f"      💡 Set: {provider.upper()}_API_KEY environment variable\n"
                
                if pstatus['last_error']:
                    status_msg += f"      ⚠️ Last Error: {pstatus['last_error']}\n"
                status_msg += "\n"
            
            # Configuration issues
            if config_status['configuration_issues']:
                status_msg += "⚠️ Configuration Issues:\n"
                for issue in config_status['configuration_issues']:
                    status_msg += f"  • {issue}\n"
                status_msg += "\n"
            
            # Help section
            status_msg += "💡 Commands:\n"
            status_msg += "  • /providers - Show only provider status\n" 
            status_msg += "  • /preprocessing - Control preprocessing mode\n"
            status_msg += "  • /help - Show all available commands\n"
            
            self._notify_message(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=status_msg,
                timestamp=self._format_timestamp()
            ))
            return True
            
        except Exception as e:
            self._notify_error(f"Error getting configuration status: {str(e)}")
            return True

    async def _handle_providers_command(self, args: str) -> bool:
        """Handle /providers command to show provider status."""
        try:
            # Use the existing LLM client (initialized once in __init__)
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                self._notify_error("Failed to initialize LLM client")
                return True
            
            # Get configuration status
            config_status = self._llm_client.get_configuration_status()
            
            # Build providers status message
            status_msg = "🔌 LLM Provider Status\n"
            status_msg += "=" * 30 + "\n\n"
            
            # Count configured providers
            configured_count = sum(1 for p in config_status['providers'].values() if p['configured'])
            status_msg += f"📊 Summary: {configured_count}/{len(config_status['providers'])} providers configured\n\n"
            
            # Provider details
            for provider, pstatus in config_status['providers'].items():
                icon = "🟢" if pstatus['configured'] else "🔴"
                status_msg += f"{icon} **{provider.upper()}**\n"
                
                if provider == "ollama":
                    if pstatus['service_available']:
                        status_msg += "   ✅ Service running locally\n"
                    else:
                        status_msg += "   ❌ Service not running\n"
                        status_msg += "   💡 Start with: `ollama serve`\n"
                else:
                    if pstatus['api_key_present']:
                        status_msg += "   ✅ API key configured\n"
                    else:
                        status_msg += "   ❌ API key missing\n"
                        status_msg += f"   💡 Set: `{provider.upper()}_API_KEY` environment variable\n"
                
                if pstatus['last_error']:
                    status_msg += f"   ⚠️ Error: {pstatus['last_error']}\n"
                status_msg += "\n"
            
            # Current selection
            status_msg += f"🎯 Current: **{config_status['current_provider']}** ({config_status['current_model']})\n\n"
            
            # Quick setup guide
            status_msg += "🚀 Quick Setup:\n"
            status_msg += "  • **OpenAI**: `export OPENAI_API_KEY=your_key`\n"
            status_msg += "  • **Anthropic**: `export ANTHROPIC_API_KEY=your_key`\n"
            status_msg += "  • **Ollama**: `ollama serve` (free, local)\n"
            status_msg += "  • **OpenRouter**: `export OPENROUTER_API_KEY=your_key`\n"
            
            self._notify_message(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=status_msg,
                timestamp=self._format_timestamp()
            ))
            return True
            
        except Exception as e:
            self._notify_error(f"Error getting provider status: {str(e)}")
            return True

    async def _handle_preprocessing_command(self, args: str) -> bool:
        """Handle /preprocessing command to control preprocessing mode."""
        try:
            # Use the existing LLM client (initialized once in __init__)
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                self._notify_error("Failed to initialize LLM client")
                return True
            
            args = args.strip().lower()
            
            if args == "on":
                result = self._llm_client.toggle_preprocessing(True)
            elif args == "off":
                result = self._llm_client.toggle_preprocessing(False)
            elif args == "toggle":
                result = self._llm_client.toggle_preprocessing()
            elif args == "status" or args == "":
                result = self._llm_client.get_preprocessing_status()
            else:
                result = "❌ Invalid preprocessing command.\n\n💡 Usage:\n  • /preprocessing on - Enable preprocessing\n  • /preprocessing off - Disable preprocessing\n  • /preprocessing toggle - Switch mode\n  • /preprocessing status - Show current mode"
            
            self._notify_message(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=result,
                timestamp=self._format_timestamp()
            ))
            return True
            
        except Exception as e:
            self._notify_error(f"Error handling preprocessing command: {str(e)}")
            return True
    
    def get_state(self) -> ChatState:
        """Get current chat state."""
        return self.state
    
    def is_processing(self) -> bool:
        """Check if currently processing a message."""
        return self.state.is_processing

    
    async def cleanup(self) -> None:
        """Clean up ChatEngine resources."""
        try:
            # Clean up LLM client if it exists
            if hasattr(self, '_llm_client') and self._llm_client:
                # Check if LLM client has cleanup method
                if hasattr(self._llm_client, 'cleanup'):
                    await self._llm_client.cleanup()
                elif hasattr(self._llm_client, 'close'):
                    await self._llm_client.close()
                self._llm_client = None
            
            # Clear callbacks to prevent hanging references
            self._status_callback = None
            self._message_callback = None  
            self._error_callback = None
            
            # Clear state
            self.state.messages.clear()
            self.state.is_processing = False
            
        except Exception as e:
            # Log cleanup errors but don't raise them
            import logging
            logging.warning(f"ChatEngine cleanup warning: {e}")


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