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
            '/preprocessing': self._handle_preprocessing_command,
            '/timeouts': self._handle_timeouts_command
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
    
    def _notify_streaming_update(self, content: str) -> None:
        """Notify UI of streaming response update."""
        if self._status_callback:
            self._status_callback(f"streaming_update:{content}")
    
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
        """Handle regular chat message with streaming support."""
        try:
            # Check if preprocessing is enabled and show optimized prompt
            preprocessing_enabled = getattr(self._llm_client, 'preprocessing_enabled', True) if self._llm_client else True
            
            if preprocessing_enabled and self._llm_client:
                # Add original user message to history
                user_message = self.state.add_message(MessageRole.USER, user_input)
                self._notify_message(user_message)
                
                # Show status while optimizing
                self._notify_status("📝 Optimizing prompt...")
                
                # Get optimized prompt
                optimized_prompt = await self._llm_client.optimize_prompt(user_input)
                
                # Show optimized prompt if it's different from original
                if optimized_prompt != user_input and len(optimized_prompt.strip()) > len(user_input.strip()) * 0.8:
                    # Create and show optimized prompt message
                    optimized_message = ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=f"📝 Optimized prompt: {optimized_prompt}",
                        timestamp=self._format_timestamp()
                    )
                    self._notify_message(optimized_message)
                    # Use optimized prompt for processing
                    actual_prompt = optimized_prompt
                else:
                    # Use original prompt if optimization didn't improve it
                    actual_prompt = user_input
            else:
                # Preprocessing disabled - just show user message with timestamp
                user_message = self.state.add_message(MessageRole.USER, user_input)
                self._notify_message(user_message)
                actual_prompt = user_input
            
            # Set processing state
            self.state.is_processing = True
            self._notify_status("Processing your message...")
            
            # Check if streaming is available and enabled
            if await self._should_use_streaming():
                await self._handle_streaming_response(actual_prompt)
            else:
                # Fallback to batch processing
                response = await self._get_ai_response(actual_prompt)
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
    
    async def _should_use_streaming(self) -> bool:
        """Check if streaming should be used for responses."""
        try:
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                return False
            
            return self._llm_client.supports_streaming()
        except Exception:
            return False
    
    async def _handle_streaming_response(self, user_input: str) -> None:
        """Handle streaming AI response with real-time updates."""
        try:
            # Stream response chunks directly without creating placeholder message
            full_response = ""
            async for chunk in self._get_ai_response_streaming(user_input):
                if chunk:  # Only process non-empty chunks
                    full_response += chunk
                    # Notify UI of streaming update
                    self._notify_streaming_update(full_response)
            
            # After streaming is complete, create final message and display it properly
            ai_message = self.state.add_message(MessageRole.ASSISTANT, full_response)
            self._notify_message(ai_message)
            
        except Exception as e:
            # Handle streaming errors
            error_msg = f"❌ Streaming error: {str(e)}"
            ai_message = self.state.add_message(MessageRole.ASSISTANT, error_msg)
            self._notify_message(ai_message)
    
    async def _get_ai_response_streaming(self, user_input: str):
        """Stream AI response in real-time chunks with progress tracking."""
        try:
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                yield "❌ Failed to initialize LLM client. Please check your configuration with /config command."
                return
            
            # Create progress callback to forward to UI
            async def progress_callback(status: str):
                """Forward progress updates to TUI."""
                if self._status_callback:
                    self._status_callback(status)
            
            # Use streaming if supported, otherwise fallback to batch
            if self._llm_client.supports_streaming():
                async for chunk in self._llm_client.send_message_streaming(user_input, progress_callback=progress_callback):
                    yield chunk
            else:
                # Fallback to batch processing with progress tracking
                response = await self._llm_client.send_message(user_input, progress_callback=progress_callback)
                yield response
                
        except Exception as e:
            import logging
            logging.error(f"Error in streaming response: {e}")
            yield f"❌ Streaming error: {str(e)}"
    
    async def _get_ai_response(self, user_input: str) -> str:
        """
        Get AI response to user input using the real LLMClient with detailed error reporting and progress tracking.
        """
        try:
            # Use the existing LLM client (initialized once in __init__)
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                return "❌ Failed to initialize LLM client. Please check your configuration with /config command."
            
            # Create progress callback to forward to UI
            async def progress_callback(status: str):
                """Forward progress updates to TUI."""
                if self._status_callback:
                    self._status_callback(status)
            
            # Get response from real LLM with progress tracking
            response = await self._llm_client.send_message(user_input, progress_callback=progress_callback)
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
• /timeouts [set <type> <seconds>|reset|status] - Manage request timeouts

🛠️ **Preprocessing Commands:**
• /preprocessing on/off/toggle - Control preprocessing mode
• /preprocessing status - Show current preprocessing status
• /preprocessing provider <provider> - Set preprocessing provider
• /preprocessing model <model> - Set preprocessing model  
• /preprocessing config - Show detailed preprocessing configuration

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
• **On** (default): Multi-turn tool execution + prompt optimization
• **Off**: Direct LLM responses only, faster but simpler

🎯 **Advanced Preprocessing:**
• **Custom Provider/Model**: Use different models for preprocessing vs responses
• **Example**: Fast local Ollama for preprocessing, powerful Anthropic for responses
• **Commands**: 
  - `/preprocessing provider ollama`
  - `/preprocessing model gpt-oss:20b`
  - `/preprocessing config`

⏱️ **Timeout Issues:**
• Use `/timeouts` to check current timeout settings
• Use `/timeouts set complex_task 600` for large operations
• Default timeouts work for most simple questions
• Streaming responses have separate timeout settings

💡 **Tips:**
• Type `/config` if you see connection errors
• Use `/preprocessing off` for faster responses
• Use `/preprocessing config` to see optimization setup
• Mix providers: fast local preprocessing + powerful cloud responses
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
            
            # Timeout configuration
            status_msg += "⏱️ Timeout Settings:\n"
            status_msg += f"  • Default Request: {self._llm_client._get_timeout('default', 300)}s\n"
            status_msg += f"  • Anthropic: {self._llm_client._get_timeout('anthropic', 300)}s\n"
            status_msg += f"  • OpenRouter: {self._llm_client._get_timeout('openrouter', 300)}s\n"
            status_msg += f"  • Local Ollama: {self._llm_client._get_timeout('local_ollama', 300)}s\n"
            status_msg += f"  • Ollama Turbo: {self._llm_client._get_timeout('ollama_turbo', 300)}s\n"
            status_msg += f"  • Proxy: {self._llm_client._get_timeout('proxy', 300)}s\n\n"
            
            # Help section
            status_msg += "💡 Commands:\n"
            status_msg += "  • /providers - Show only provider status\n" 
            status_msg += "  • /preprocessing - Control preprocessing mode\n"
            status_msg += "  • /timeouts - Manage timeout settings\n"
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
            
            args = args.strip()
            parts = args.split(' ', 1) if args else ['']
            command = parts[0].lower()
            arg_value = parts[1] if len(parts) > 1 else ""
            
            if command == "on":
                result = self._llm_client.toggle_preprocessing(True)
            elif command == "off":
                result = self._llm_client.toggle_preprocessing(False)
            elif command == "toggle":
                result = self._llm_client.toggle_preprocessing()
            elif command == "status" or command == "":
                result = self._llm_client.get_preprocessing_status()
            elif command == "provider":
                if not arg_value:
                    result = "❌ Provider name required\n💡 Usage: /preprocessing provider <provider>\n📋 Valid providers: ollama, ollama-turbo, openai, anthropic, openrouter"
                else:
                    result = self._llm_client.set_preprocessing_provider(arg_value)
            elif command == "model":
                if not arg_value:
                    result = "❌ Model name required\n💡 Usage: /preprocessing model <model>\n📋 Example: /preprocessing model gpt-oss:20b"
                else:
                    result = self._llm_client.set_preprocessing_model(arg_value)
            elif command == "config":
                result = self._llm_client.get_preprocessing_config()
            else:
                result = """❌ Invalid preprocessing command.

💡 Usage:
  • /preprocessing on - Enable preprocessing
  • /preprocessing off - Disable preprocessing
  • /preprocessing toggle - Switch mode
  • /preprocessing status - Show current mode
  • /preprocessing provider <provider> - Set preprocessing provider
  • /preprocessing model <model> - Set preprocessing model
  • /preprocessing config - Show detailed configuration

🚀 Examples:
  • /preprocessing provider ollama
  • /preprocessing model gpt-oss:20b
  • /preprocessing config"""
            
            self._notify_message(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=result,
                timestamp=self._format_timestamp()
            ))
            return True
            
        except Exception as e:
            self._notify_error(f"Error handling preprocessing command: {str(e)}")
            return True

    async def _handle_timeouts_command(self, args: str) -> bool:
        """Handle /timeouts command to show and configure timeouts."""
        try:
            # Use the existing LLM client (initialized once in __init__)
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                self._notify_error("Failed to initialize LLM client")
                return True
            
            args = args.strip().lower()
            
            if not args or args == "status":
                # Show current timeouts
                result = self._get_timeout_status()
            elif args.startswith("set "):
                # Set timeout: /timeouts set anthropic 60
                result = self._set_timeout(args[4:])
            elif args == "reset":
                # Reset to defaults - show current since we can't actually reset
                result = self._reset_timeouts()
            else:
                result = "❌ Invalid timeouts command.\n\n💡 Usage:\n  • /timeouts - Show current timeouts\n  • /timeouts set <type> <seconds> - Set timeout\n  • /timeouts reset - Show defaults"
            
            self._notify_message(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=result,
                timestamp=self._format_timestamp()
            ))
            return True
            
        except Exception as e:
            self._notify_error(f"Error handling timeouts command: {str(e)}")
            return True

    def _get_timeout_status(self) -> str:
        """Get detailed timeout configuration status."""
        # Get all timeout types from the codebase
        timeout_types = {
            "default": ("Default Request", 300),
            "anthropic": ("Anthropic API", 300),
            "openrouter": ("OpenRouter API", 300),
            "local_ollama": ("Local Ollama", 300),
            "ollama_turbo": ("Ollama Turbo", 300),
            "proxy": ("Proxy Requests", 300)
        }
        
        status_msg = "⏱️ Timeout Configuration\n"
        status_msg += "=" * 40 + "\n\n"
        
        for timeout_key, (timeout_name, default_val) in timeout_types.items():
            current_val = self._llm_client._get_timeout(timeout_key, default_val)
            status_msg += f"  • {timeout_name}: {current_val}s\n"
        
        status_msg += "\n💡 Commands:\n"
        status_msg += "  • /timeouts set <type> <seconds> - Adjust timeout\n"
        status_msg += "  • /timeouts reset - Show default values\n"
        status_msg += "  • /config - Show full configuration\n\n"
        
        status_msg += "📊 Recommended Values:\n"
        status_msg += "  • Simple questions: 30-60s\n"
        status_msg += "  • Complex analysis: 120-300s\n"
        status_msg += "  • Large file operations: 300-600s\n\n"
        
        status_msg += "🔧 Available timeout types:\n"
        status_msg += "  • default, anthropic, openrouter\n"
        status_msg += "  • local_ollama, ollama_turbo, proxy\n"
        
        return status_msg

    def _set_timeout(self, args: str) -> str:
        """Set timeout value for a specific type."""
        try:
            parts = args.split()
            if len(parts) != 2:
                return "❌ Invalid format. Use: /timeouts set <type> <seconds>"
            
            timeout_type, timeout_str = parts
            timeout_value = float(timeout_str)
            
            if timeout_value <= 0:
                return "❌ Timeout value must be greater than 0"
            
            if timeout_value > 3600:  # 1 hour max
                return "❌ Timeout value cannot exceed 3600 seconds (1 hour)"
            
            # Note: Since we can't actually modify the config at runtime,
            # we inform the user about how timeouts are configured
            return f"""ℹ️ Timeout Configuration Information

Current timeout for '{timeout_type}': {self._llm_client._get_timeout(timeout_type, 300)}s
Requested value: {timeout_value}s

⚠️ **Timeout Configuration Method:**
Timeouts are currently read from configuration at startup.
To modify timeouts, you would need to:

1. Set environment variable or config file
2. Restart the TUI application

📝 **Configuration Options:**
• Environment: Set timeout values in your startup config
• Config file: Add timeouts section to configuration
• Runtime: Not currently supported

💡 Use '/timeouts' to see current values and defaults."""

        except ValueError:
            return "❌ Invalid timeout value. Must be a number."
        except Exception as e:
            return f"❌ Error setting timeout: {str(e)}"

    def _reset_timeouts(self) -> str:
        """Show default timeout values."""
        return """⏱️ Default Timeout Values
===============================

  • Default Request: 300s (5 minutes)
  • Anthropic API: 300s (5 minutes)
  • OpenRouter API: 300s (5 minutes)
  • Local Ollama: 300s (5 minutes)
  • Ollama Turbo: 300s (5 minutes)
  • Proxy Requests: 300s (5 minutes)

ℹ️ **Current vs Default:**
Use '/timeouts' to see your current configuration.

🔧 **To Reset:**
Timeout configuration is set at startup. To use defaults:
1. Remove any custom timeout configuration
2. Restart the TUI application"""
    
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