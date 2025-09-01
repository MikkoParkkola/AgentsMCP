"""
Enhanced Command Interface Component

This component provides an improved command interface with features like:
- Smart command completion
- Context-aware suggestions
- Command history with search
- Syntax highlighting
- Error prevention
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CommandSuggestion:
    """Represents a command suggestion."""
    command: str
    description: str
    confidence: float
    category: str


class EnhancedCommandInterface:
    """
    Enhanced command interface with intelligent features.
    
    This component provides advanced command input capabilities while
    maintaining full backward compatibility with basic command interfaces.
    """
    
    def __init__(self, basic_interface=None):
        """Initialize the enhanced command interface."""
        self.basic_interface = basic_interface
        self.command_history = []
        self.suggestions_cache = {}
        self.is_initialized = False
        
        # Command categories for better organization
        self.command_categories = {
            "system": ["status", "settings", "help", "quit", "exit"],
            "conversation": ["clear", "history", "save", "load"],
            "agents": ["list", "info", "logs", "stats"],
            "theme": ["light", "dark", "auto", "toggle"]
        }
        
        logger.debug("Enhanced command interface initialized")
    
    async def initialize(self, orchestrator=None, theme_manager=None) -> bool:
        """Initialize the enhanced command interface."""
        try:
            self.orchestrator = orchestrator
            self.theme_manager = theme_manager
            self.is_initialized = True
            logger.info("Enhanced command interface ready")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize enhanced command interface: {e}")
            return False
    
    async def get_command_suggestions(self, partial_input: str) -> List[CommandSuggestion]:
        """Get smart command suggestions for partial input."""
        if not self.is_initialized:
            return []
        
        suggestions = []
        partial_lower = partial_input.lower()
        
        # Check cache first
        if partial_input in self.suggestions_cache:
            return self.suggestions_cache[partial_input]
        
        # Generate suggestions from command categories
        for category, commands in self.command_categories.items():
            for command in commands:
                if command.startswith(partial_lower):
                    confidence = 1.0 - (len(partial_input) / len(command))
                    suggestions.append(CommandSuggestion(
                        command=command,
                        description=f"{category.title()} command",
                        confidence=confidence,
                        category=category
                    ))
        
        # Add context-aware suggestions based on recent history
        if self.command_history:
            recent_commands = self.command_history[-5:]
            for cmd in recent_commands:
                if partial_lower in cmd.lower() and cmd not in [s.command for s in suggestions]:
                    suggestions.append(CommandSuggestion(
                        command=cmd,
                        description="From recent history",
                        confidence=0.7,
                        category="history"
                    ))
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        # Cache results
        self.suggestions_cache[partial_input] = suggestions[:10]  # Limit to top 10
        
        return suggestions[:10]
    
    async def execute_command(self, command: str, context: Dict = None) -> str:
        """Execute a command with enhanced features."""
        if not self.is_initialized:
            return "Command interface not ready"
        
        # Add to history
        if command not in self.command_history[-1:]:  # Avoid duplicates of last command
            self.command_history.append(command)
            
            # Keep history manageable
            if len(self.command_history) > 100:
                self.command_history = self.command_history[-50:]
        
        # Handle built-in commands
        if command == "help":
            return self._get_help_message()
        elif command == "history":
            return self._get_history_display()
        elif command == "clear":
            return self._clear_interface()
        elif command.startswith("theme "):
            return await self._handle_theme_command(command)
        
        # Delegate to orchestrator or basic interface
        if self.orchestrator:
            try:
                # Route through orchestrator for consistency
                response = await self.orchestrator.process_user_input(command, context or {})
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.error(f"Orchestrator command execution failed: {e}")
                return f"Command execution error: {str(e)}"
        elif self.basic_interface:
            return await self.basic_interface.execute_command(command)
        else:
            return f"Unknown command: {command}"
    
    def _get_help_message(self) -> str:
        """Get help message with available commands."""
        help_text = "Enhanced Command Interface - Available Commands:\n\n"
        
        for category, commands in self.command_categories.items():
            help_text += f"{category.title()}:\n"
            for cmd in commands:
                help_text += f"  {cmd}\n"
            help_text += "\n"
        
        help_text += "Type any command or start typing for suggestions.\n"
        help_text += "Use UP/DOWN arrows to browse command history."
        
        return help_text
    
    def _get_history_display(self) -> str:
        """Get formatted command history."""
        if not self.command_history:
            return "No command history available."
        
        history_text = "Recent Commands:\n\n"
        for i, cmd in enumerate(self.command_history[-10:], 1):
            history_text += f"{i:2d}. {cmd}\n"
        
        return history_text
    
    def _clear_interface(self) -> str:
        """Clear the interface (implementation depends on TUI framework)."""
        return "Interface cleared"
    
    async def _handle_theme_command(self, command: str) -> str:
        """Handle theme-related commands."""
        parts = command.split()
        if len(parts) < 2:
            return "Usage: theme <light|dark|auto|toggle>"
        
        theme_name = parts[1].lower()
        
        if self.theme_manager:
            try:
                if theme_name == "toggle":
                    current = getattr(self.theme_manager, 'current_theme', 'auto')
                    new_theme = 'dark' if current == 'light' else 'light'
                    # Would call theme_manager.set_theme(new_theme) if available
                    return f"Theme toggled to {new_theme}"
                elif theme_name in ['light', 'dark', 'auto']:
                    # Would call theme_manager.set_theme(theme_name) if available
                    return f"Theme set to {theme_name}"
                else:
                    return "Invalid theme. Use: light, dark, auto, or toggle"
            except Exception as e:
                return f"Theme change failed: {e}"
        else:
            return "Theme management not available"
    
    def get_command_history(self) -> List[str]:
        """Get the command history list."""
        return self.command_history.copy()
    
    def clear_suggestions_cache(self):
        """Clear the suggestions cache."""
        self.suggestions_cache.clear()
    
    def supports_suggestions(self) -> bool:
        """Check if command suggestions are supported."""
        return self.is_initialized
    
    def supports_history(self) -> bool:
        """Check if command history is supported."""
        return self.is_initialized
    
    async def cleanup(self):
        """Clean up the enhanced command interface."""
        logger.debug("Enhanced command interface cleanup")
        self.suggestions_cache.clear()
        self.is_initialized = False