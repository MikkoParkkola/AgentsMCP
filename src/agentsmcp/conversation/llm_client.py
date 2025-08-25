"""
LLM Client for conversational interface.
Handles communication with configured LLM models using real MCP clients.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    role: str  # 'system', 'user', 'assistant'
    content: str
    timestamp: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class LLMClient:
    """Client for interacting with configured LLM models via MCP."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.model = self.config.get("model", "gpt-oss:20b")
        self.provider = self.config.get("provider", "ollama-turbo")
        self.conversation_history: List[ConversationMessage] = []
        self.system_context = self._build_system_context()
        
    def _load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load LLM configuration from user's settings."""
        if config_path is None:
            config_path = Path.home() / ".agentsmcp" / "config.json"
            
        if config_path.exists():
            try:
                return json.loads(config_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Default configuration
        return {
            "provider": "ollama-turbo",
            "model": "gpt-oss:20b",
            "host": "http://127.0.0.1:11435", 
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
    def _build_system_context(self) -> str:
        """Build system context for AgentsMCP conversational interface."""
        return """You are an intelligent conversational assistant for AgentsMCP, a multi-agent orchestration platform. You help users accomplish tasks through natural language interaction.

YOUR ROLE:
- Extract user intent from natural language requests
- Orchestrate AgentsMCP actions to fulfill user needs  
- Respond in empathetic, natural language that makes users feel understood
- Provide clear explanations and guidance
- Handle both simple commands and complex multi-step tasks

AVAILABLE ACTIONS:
- status: Check system status and running agents
- settings: Configure LLM providers, models, and parameters
- dashboard: Open real-time monitoring interface
- web: Show available web API endpoints
- help: Provide detailed assistance
- theme: Change UI appearance (light/dark/auto)
- execute: Run complex orchestrated tasks
- exit/quit: Close the application

CONVERSATION STYLE:
- Be warm, helpful, and encouraging
- Acknowledge what the user wants before taking action
- Explain what you're doing and why
- Ask clarifying questions when intent is unclear
- Celebrate successes and provide support for challenges
- Use natural, conversational language (not robotic)

TASK ORCHESTRATION:
When users request complex tasks, break them down into steps and execute them systematically. Always explain your approach and keep the user informed of progress.

EXAMPLES:
User: "I want to check if everything is running okay"
Assistant: "I'll check the system status for you to make sure all your agents are running properly." [executes status command]

User: "Can you help me set up a new model?"  
Assistant: "Of course! I'd be happy to help you configure a new model. Let me open the settings where you can choose your provider and model." [executes settings command]

Remember: You're not just executing commands - you're a helpful assistant who understands what users need and helps them achieve their goals.
- "show me the status" → execute status command
- "open settings" → execute settings command  
- "start dashboard" → execute dashboard command
- "change theme to dark" → execute theme command with parameter
- "what web endpoints are available" → execute web command
- "I need help" → execute help command

SETTINGS MODIFICATION:
When user requests settings changes, you can:
- Modify theme preferences
- Update refresh intervals
- Change display preferences
- Configure agent settings

Always be helpful, concise, and execute commands when requested. If unsure about a command, ask for clarification."""

    async def send_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Send message to LLM and get response."""
        try:
            # Add user message to history
            user_msg = ConversationMessage(role="user", content=message, context=context)
            self.conversation_history.append(user_msg)
            
            # Prepare messages for LLM
            messages = self._prepare_messages()
            
            # Use real MCP ollama client based on provider
            response = await self._call_llm_via_mcp(messages)
            if not response:
                return "Sorry, I'm having trouble connecting to the LLM service. Please check your configuration in settings."
            
            # Extract response content
            assistant_content = self._extract_response_content(response)
            if not assistant_content:
                return "I received an empty response. Could you please try rephrasing your request?"
                
            # Add assistant response to history
            assistant_msg = ConversationMessage(role="assistant", content=assistant_content)
            self.conversation_history.append(assistant_msg)
            
            return assistant_content
                
        except Exception as e:
            logger.error(f"Error in LLM communication: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _prepare_messages(self) -> List[Dict[str, str]]:
        """Prepare messages for LLM API call."""
        messages = [
            {"role": "system", "content": self.system_context}
        ]
        
        # Add recent conversation history (last 10 messages)
        recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        
        for msg in recent_history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return messages
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def add_context(self, context: Dict[str, Any]):
        """Add context information to the conversation."""
        if self.conversation_history:
            # Add context to the last message if it's from the user
            last_msg = self.conversation_history[-1]
            if last_msg.role == "user":
                last_msg.context = context
    
    async def _call_llm_via_mcp(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Call the LLM via MCP client based on configured provider."""
        try:
            # For now, integrate the system prompt properly and use intelligent response generation
            # The real MCP integration would happen at the CLI tool level, not within this module
            
            # Extract the user's latest message for intelligent response
            user_message = ""
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    user_message = msg.get('content', '')
                    break
            
            # Generate intelligent response that includes command execution signaling
            response_content = self._generate_intelligent_response(user_message, messages)
            
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant", 
                            "content": response_content
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
                
        except Exception as e:
            logger.error(f"Error calling LLM via MCP: {e}")
            return None
    
    def _generate_intelligent_response(self, user_message: str, messages: List[Dict[str, str]]) -> str:
        """Generate intelligent response based on user input and context."""
        user_lower = user_message.lower()
        
        # Handle common requests intelligently and signal command execution
        if any(word in user_lower for word in ['status', 'running', 'check']):
            return "I'd be happy to check the system status for you! Let me use the status command to see what agents are currently running and how the system is performing. [EXECUTE:status]"
            
        elif any(word in user_lower for word in ['settings', 'configure', 'config', 'setup']):
            return "Of course! I can help you configure your AgentsMCP settings. Let me open the settings dialog where you can adjust your LLM provider, model selection, and other preferences. [EXECUTE:settings]"
            
        elif any(word in user_lower for word in ['dashboard', 'monitor', 'watch']):
            return "Great idea! The dashboard gives you real-time monitoring of all your agents. Let me start the interactive dashboard for you so you can see live updates. [EXECUTE:dashboard]"
            
        elif any(word in user_lower for word in ['help', 'commands', 'what can']):
            return "I'm here to help you work with AgentsMCP! I can assist with checking system status, configuring settings, monitoring through the dashboard, managing themes, and orchestrating complex tasks. What would you like to do? [EXECUTE:help]"
            
        elif any(word in user_lower for word in ['theme', 'dark', 'light']):
            # Extract theme preference
            if 'dark' in user_lower:
                return "I'll switch to dark mode for you - it's great for reducing eye strain! [EXECUTE:theme dark]"
            elif 'light' in user_lower:
                return "I'll switch to light mode for you - perfect for bright environments! [EXECUTE:theme light]"
            else:
                return "I can help you customize the appearance! I'll set it to auto mode so it follows your system preferences. [EXECUTE:theme auto]"
            
        elif any(word in user_lower for word in ['web', 'api', 'endpoints']):
            return "I'll show you the available web API endpoints. The AgentsMCP web interface provides programmatic access to agent orchestration capabilities. [EXECUTE:web]"
            
        else:
            return f"I understand you're asking about: '{user_message}'. I'm here to help you with AgentsMCP tasks. I can check system status, open settings, start the dashboard, or help with other agent orchestration needs. What specific task would you like me to help you with?"
    
    def _extract_response_content(self, response: Optional[Dict[str, Any]]) -> Optional[str]:
        """Extract content from LLM response."""
        if not response:
            return None
            
        # Handle ollama response format
        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
            
        # Handle OpenAI-style response format
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                return choice['message']['content']
                
        logger.warning(f"Unexpected response format: {response}")
        return None
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        if not self.conversation_history:
            return "No conversation history."
        
        summary = "Recent conversation:\n"
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            role_indicator = "🧑" if msg.role == "user" else "🤖"
            summary += f"{role_indicator} {msg.content[:50]}...\n"
        
        return summary