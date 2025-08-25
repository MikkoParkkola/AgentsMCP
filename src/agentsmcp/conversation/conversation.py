"""
Core conversation management for AgentsMCP conversational interface.
Handles natural language input parsing and command execution.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .llm_client import LLMClient, ConversationMessage

logger = logging.getLogger(__name__)


@dataclass
class CommandIntent:
    """Represents a parsed command intent from natural language."""
    command: str
    parameters: Dict[str, Any]
    confidence: float
    raw_text: str


class ConversationManager:
    """Manages conversational interface for AgentsMCP."""
    
    def __init__(self, command_interface=None, theme_manager=None):
        self.command_interface = command_interface
        self.theme_manager = theme_manager
        self.llm_client = LLMClient()
        self.command_patterns = self._build_command_patterns()
        
    def _build_command_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for recognizing command intents."""
        return {
            "status": [
                r"show.*status", r"check.*status", r"system.*status",
                r"what.*running", r"agents.*running", r"current.*state"
            ],
            "settings": [
                r"open.*settings", r"configure", r"change.*settings",
                r"modify.*config", r"preferences", r"settings"
            ],
            "dashboard": [
                r"start.*dashboard", r"open.*dashboard", r"monitor",
                r"dashboard", r"show.*dashboard", r"monitoring"
            ],
            "web": [
                r"web.*api", r"api.*endpoints", r"web.*interface",
                r"what.*endpoints", r"api.*info", r"web.*info"
            ],
            "help": [
                r"help", r"how.*use", r"commands.*available",
                r"what.*can.*do", r"usage", r"guide"
            ],
            "theme": [
                r"change.*theme", r"theme.*to", r"switch.*theme",
                r"set.*theme", r"theme.*dark", r"theme.*light"
            ],
            "exit": [
                r"exit", r"quit", r"bye", r"goodbye",
                r"stop", r"close", r"end"
            ]
        }
    
    async def process_input(self, user_input: str) -> str:
        """Process user input and return response."""
        try:
            # First try to extract direct commands
            command_intent = self._extract_command_intent(user_input)
            
            if command_intent and command_intent.confidence > 0.7:
                # Execute direct command
                result = await self._execute_command(command_intent)
                if result:
                    return result
            
            # Fall back to LLM conversation
            response = await self.llm_client.send_message(user_input, {
                "available_commands": list(self.command_patterns.keys()),
                "current_theme": getattr(self.theme_manager, 'current_theme', 'auto') if self.theme_manager else 'auto'
            })
            
            # Check if LLM response contains command execution request
            command_request = self._extract_command_from_llm_response(response)
            if command_request:
                command_result = await self._execute_command(command_request)
                if command_result:
                    # Combine LLM response with command result
                    return f"{response}\n\n{command_result}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return f"Sorry, I encountered an error processing your request: {str(e)}"
    
    def _extract_command_intent(self, text: str) -> Optional[CommandIntent]:
        """Extract command intent from natural language text."""
        text_lower = text.lower().strip()
        
        best_match = None
        best_confidence = 0.0
        
        for command, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    confidence = len(re.findall(pattern, text_lower)) * 0.3
                    confidence += 1.0 if text_lower.startswith(pattern.split('.*')[0]) else 0.0
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        parameters = self._extract_parameters(command, text_lower)
                        best_match = CommandIntent(
                            command=command,
                            parameters=parameters,
                            confidence=confidence,
                            raw_text=text
                        )
        
        return best_match
    
    def _extract_parameters(self, command: str, text: str) -> Dict[str, Any]:
        """Extract parameters for specific commands."""
        parameters = {}
        
        if command == "theme":
            if "dark" in text:
                parameters["theme"] = "dark"
            elif "light" in text:
                parameters["theme"] = "light"
            elif "auto" in text:
                parameters["theme"] = "auto"
        
        return parameters
    
    def _extract_command_from_llm_response(self, response: str) -> Optional[CommandIntent]:
        """Extract command execution requests from LLM responses."""
        # Look for command execution markers in LLM response
        command_markers = [
            r"execute\s+(\w+)(?:\s+(\w+))?",
            r"run\s+(\w+)(?:\s+(\w+))?",
            r"→\s*execute\s+(\w+)(?:\s+(\w+))?",
            r"\[execute:\s*(\w+)(?:\s+(\w+))?\]"
        ]
        
        for pattern in command_markers:
            match = re.search(pattern, response.lower())
            if match:
                command = match.group(1)
                param = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                
                if command in self.command_patterns:
                    parameters = {}
                    if param and command == "theme":
                        parameters["theme"] = param
                    
                    return CommandIntent(
                        command=command,
                        parameters=parameters,
                        confidence=1.0,
                        raw_text=response
                    )
        
        return None
    
    async def _execute_command(self, command_intent: CommandIntent) -> Optional[str]:
        """Execute a command based on intent."""
        if not self.command_interface:
            return f"Command '{command_intent.command}' recognized but command interface not available."
        
        try:
            command = command_intent.command
            params = command_intent.parameters
            
            if command == "status":
                return await self.command_interface.handle_command("status")
            elif command == "settings":
                return await self.command_interface.handle_command("settings")
            elif command == "dashboard":
                return await self.command_interface.handle_command("dashboard")
            elif command == "web":
                return await self.command_interface.handle_command("web")
            elif command == "help":
                return await self.command_interface.handle_command("help")
            elif command == "theme":
                if "theme" in params:
                    return await self.command_interface.handle_command(f"theme {params['theme']}")
                else:
                    return await self.command_interface.handle_command("theme")
            elif command == "exit":
                return await self.command_interface.handle_command("exit")
            else:
                return f"Command '{command}' not yet implemented in conversational interface."
                
        except Exception as e:
            logger.error(f"Error executing command {command_intent.command}: {e}")
            return f"Sorry, I couldn't execute the '{command_intent.command}' command: {str(e)}"
    
    async def handle_settings_modification(self, modification_request: str) -> str:
        """Handle settings modification requests from conversation."""
        try:
            # Parse the modification request
            if "theme" in modification_request.lower():
                if "dark" in modification_request.lower():
                    if self.theme_manager:
                        self.theme_manager.current_theme = "dark"
                        return "✅ Theme changed to dark mode."
                elif "light" in modification_request.lower():
                    if self.theme_manager:
                        self.theme_manager.current_theme = "light"
                        return "✅ Theme changed to light mode."
                elif "auto" in modification_request.lower():
                    if self.theme_manager:
                        self.theme_manager.current_theme = "auto"
                        return "✅ Theme changed to auto mode."
            
            return "I understand you want to modify settings, but I need more specific instructions. You can say things like 'change theme to dark' or 'set theme to light'."
            
        except Exception as e:
            logger.error(f"Error modifying settings: {e}")
            return f"Sorry, I couldn't modify the settings: {str(e)}"
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get current conversation context."""
        return {
            "conversation_length": len(self.llm_client.conversation_history),
            "last_command": getattr(self, '_last_command', None),
            "theme": getattr(self.theme_manager, 'current_theme', 'auto') if self.theme_manager else 'auto',
            "commands_available": list(self.command_patterns.keys())
        }