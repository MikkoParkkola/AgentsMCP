"""
MCP utilities for AgentsMCP conversational interface.
Provides access to MCP tools like ollama-turbo for LLM integration.
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Global MCP client cache
_ollama_client = None


def get_ollama_client():
    """Get or create ollama-turbo MCP client for conversational interface."""
    global _ollama_client
    
    if _ollama_client is not None:
        return _ollama_client
    
    try:
        # Try to import and use the MCP ollama-turbo client
        # This would be provided by the MCP framework
        
        # For now, create a mock client that returns structured responses
        # In production, this would use the actual MCP ollama-turbo integration
        _ollama_client = MockOllamaClient()
        logger.info("Mock Ollama client initialized for conversational interface")
        return _ollama_client
        
    except Exception as e:
        logger.error(f"Failed to initialize ollama client: {e}")
        return None


class MockOllamaClient:
    """Mock ollama client for development/testing purposes."""
    
    async def chat_completion(self, model: str, messages: list, temperature: float = 0.7) -> dict:
        """Mock chat completion that returns intelligent responses based on context."""
        try:
            # Get the last user message
            user_message = None
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    user_message = msg.get('content', '').lower()
                    break
            
            if not user_message:
                return self._create_response("I'm here to help! What would you like to do with AgentsMCP?")
            
            # Pattern matching for common requests
            if any(word in user_message for word in ['status', 'running', 'agents', 'state']):
                return self._create_response("I'll check the system status for you. → execute status")
            
            elif any(word in user_message for word in ['settings', 'configure', 'config', 'preferences']):
                return self._create_response("I'll open the settings dialog where you can configure AgentsMCP. → execute settings")
            
            elif any(word in user_message for word in ['dashboard', 'monitor', 'monitoring']):
                return self._create_response("Starting the monitoring dashboard to show real-time system status. → execute dashboard")
            
            elif any(word in user_message for word in ['web', 'api', 'endpoints']):
                return self._create_response("Here are the available web API endpoints: → execute web")
            
            elif any(word in user_message for word in ['help', 'commands', 'what can']):
                return self._create_response("Here are all the available commands: → execute help")
            
            elif 'theme' in user_message:
                if 'dark' in user_message:
                    return self._create_response("Switching to dark theme. → execute theme dark")
                elif 'light' in user_message:
                    return self._create_response("Switching to light theme. → execute theme light")
                elif 'auto' in user_message:
                    return self._create_response("Setting theme to auto mode. → execute theme auto")
                else:
                    return self._create_response("I can help you change the theme. Available options are: dark, light, or auto. → execute theme")
            
            elif any(word in user_message for word in ['exit', 'quit', 'bye', 'goodbye']):
                return self._create_response("Goodbye! Exiting AgentsMCP. → execute exit")
            
            else:
                # General conversation
                return self._create_response(
                    "I understand you want to interact with AgentsMCP. "
                    "I can help you with status checks, settings, dashboard monitoring, "
                    "web API information, theme changes, and more. "
                    "What would you like to do?"
                )
                
        except Exception as e:
            logger.error(f"Mock ollama client error: {e}")
            return self._create_response("I encountered an error processing your request. Please try again.")
    
    def _create_response(self, content: str) -> dict:
        """Create a structured response like the real ollama API."""
        return {
            'choices': [
                {
                    'message': {
                        'content': content,
                        'role': 'assistant'
                    }
                }
            ]
        }