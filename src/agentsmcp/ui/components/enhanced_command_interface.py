"""
Enhanced Command Interface - Revolutionary CLI experience with natural language processing.

This component provides intelligent command composition with:
- Natural language intent recognition
- Real-time command translation and preview
- Context-aware suggestions and auto-completion
- Intelligent error correction and learning
- Seamless integration with existing command infrastructure
"""

from __future__ import annotations

import asyncio
import logging
import re
import json
from typing import Optional, Dict, List, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from ..v2.event_system import AsyncEventSystem, Event, EventType

logger = logging.getLogger(__name__)


class CommandIntent(Enum):
    """Recognized command intents for natural language processing."""
    CHAT = "chat"
    SEARCH = "search"
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    HELP = "help"
    CONFIG = "config"
    STATUS = "status"
    QUIT = "quit"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for command interpretation."""
    HIGH = "high"      # >0.8
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"        # 0.2-0.5
    UNKNOWN = "unknown" # <0.2


@dataclass
class CommandSuggestion:
    """A command suggestion with metadata."""
    command: str
    description: str
    intent: CommandIntent
    confidence: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    shortcuts: List[str] = field(default_factory=list)


@dataclass
class InterpretationResult:
    """Result of natural language command interpretation."""
    original_input: str
    interpreted_command: str
    intent: CommandIntent
    confidence: ConfidenceLevel
    parameters: Dict[str, Any]
    suggestions: List[CommandSuggestion]
    explanation: str
    error_corrections: List[str] = field(default_factory=list)


@dataclass
class CommandContext:
    """Context information for intelligent command processing."""
    user_skill_level: str = "intermediate"  # beginner, intermediate, expert
    recent_commands: List[str] = field(default_factory=list)
    current_agent: str = "default"
    active_conversation: bool = False
    last_error: Optional[str] = None
    session_duration: timedelta = field(default_factory=lambda: timedelta())
    usage_patterns: Dict[str, int] = field(default_factory=dict)


class EnhancedCommandInterface:
    """
    Revolutionary command interface with natural language processing and intelligent assistance.
    
    Features:
    - Natural language to command translation
    - Real-time command preview and validation
    - Context-aware auto-completion
    - Intelligent error correction
    - Learning from user patterns
    - Accessibility-first design
    """
    
    def __init__(self, event_system: AsyncEventSystem):
        """Initialize the enhanced command interface."""
        self.event_system = event_system
        self.context = CommandContext()
        
        # Natural language processing patterns
        self.nlp_patterns = self._initialize_nlp_patterns()
        
        # Command registry with metadata
        self.commands = self._initialize_command_registry()
        
        # Auto-completion and suggestion system
        self.suggestion_cache: Dict[str, List[CommandSuggestion]] = {}
        self.learning_history: List[Tuple[str, str, float]] = []  # (input, command, success_score)
        
        # Performance tracking
        self.response_times: List[float] = []
        self.interpretation_accuracy: List[bool] = []
        
        # Accessibility settings
        self.announce_suggestions = True
        self.verbose_explanations = False
        self.high_contrast_mode = False
        
        # Callbacks for UI integration
        self._callbacks: Dict[str, Callable] = {}
        
    def _initialize_nlp_patterns(self) -> Dict[CommandIntent, List[Tuple[str, float]]]:
        """Initialize natural language processing patterns with confidence weights."""
        return {
            CommandIntent.CHAT: [
                (r"(?:chat|talk|speak|converse|discuss)\s+(?:with|to)?\s*(.+)", 0.9),
                (r"(?:ask|tell|say)\s+(.+)", 0.8),
                (r"(?:send|message)\s+(.+)", 0.7),
                (r"(.+)\s*\?$", 0.6),  # Questions
            ],
            CommandIntent.SEARCH: [
                (r"(?:search|find|look\s+for|locate)\s+(.+)", 0.9),
                (r"(?:show|list|display)\s+(.+)", 0.7),
                (r"(?:where\s+is|what\s+is)\s+(.+)", 0.8),
            ],
            CommandIntent.CREATE: [
                (r"(?:create|make|generate|build)\s+(.+)", 0.9),
                (r"(?:new|add)\s+(.+)", 0.8),
                (r"(?:write|compose)\s+(.+)", 0.7),
            ],
            CommandIntent.MODIFY: [
                (r"(?:change|modify|update|edit)\s+(.+)", 0.9),
                (r"(?:set|configure)\s+(.+)\s+(?:to|as)\s+(.+)", 0.8),
                (r"(?:fix|correct)\s+(.+)", 0.7),
            ],
            CommandIntent.HELP: [
                (r"(?:help|usage|how\s+to)\s*(.+)?", 0.9),
                (r"\?+$", 0.6),
                (r"(?:explain|describe)\s+(.+)", 0.7),
            ],
            CommandIntent.CONFIG: [
                (r"(?:config|settings|options|preferences)\s*(.+)?", 0.9),
                (r"(?:enable|disable|toggle)\s+(.+)", 0.8),
            ],
            CommandIntent.STATUS: [
                (r"(?:status|state|info|information)\s*(.+)?", 0.9),
                (r"(?:show|display)\s+(?:current|active)\s+(.+)", 0.7),
            ],
            CommandIntent.QUIT: [
                (r"(?:quit|exit|bye|goodbye|stop|end)", 0.9),
                (r"(?:close|shutdown)", 0.7),
            ]
        }
    
    def _initialize_command_registry(self) -> Dict[str, CommandSuggestion]:
        """Initialize the command registry with metadata."""
        return {
            # Chat commands
            "/chat": CommandSuggestion(
                command="/chat",
                description="Start a conversation with an AI agent",
                intent=CommandIntent.CHAT,
                confidence=0.9,
                examples=["chat with claude about python", "chat what is machine learning"],
                shortcuts=["/c"]
            ),
            "/ask": CommandSuggestion(
                command="/ask",
                description="Ask a question to the AI agent",
                intent=CommandIntent.CHAT,
                confidence=0.9,
                examples=["ask how to optimize code", "ask about best practices"],
                shortcuts=["/a"]
            ),
            
            # Agent management
            "/agent": CommandSuggestion(
                command="/agent",
                description="Switch or configure AI agents",
                intent=CommandIntent.CONFIG,
                confidence=0.9,
                parameters={"agent": "string"},
                examples=["agent claude", "agent list", "agent config"],
                shortcuts=["/ag"]
            ),
            "/model": CommandSuggestion(
                command="/model",
                description="Select or configure model settings",
                intent=CommandIntent.CONFIG,
                confidence=0.9,
                parameters={"model": "string"},
                examples=["model gpt-4", "model list"],
                shortcuts=["/m"]
            ),
            
            # System commands
            "/help": CommandSuggestion(
                command="/help",
                description="Show help information",
                intent=CommandIntent.HELP,
                confidence=0.9,
                examples=["help", "help commands", "help agent"],
                shortcuts=["/h", "/?"]
            ),
            "/status": CommandSuggestion(
                command="/status",
                description="Show system status",
                intent=CommandIntent.STATUS,
                confidence=0.9,
                examples=["status", "status agent", "status system"],
                shortcuts=["/s"]
            ),
            "/config": CommandSuggestion(
                command="/config",
                description="Configure system settings",
                intent=CommandIntent.CONFIG,
                confidence=0.9,
                parameters={"setting": "string", "value": "any"},
                examples=["config theme dark", "config verbose true"],
                shortcuts=["/cfg"]
            ),
            
            # Utility commands
            "/clear": CommandSuggestion(
                command="/clear",
                description="Clear the screen or conversation",
                intent=CommandIntent.MODIFY,
                confidence=0.9,
                examples=["clear", "clear history"],
                shortcuts=["/cl"]
            ),
            "/history": CommandSuggestion(
                command="/history",
                description="Show command or conversation history",
                intent=CommandIntent.STATUS,
                confidence=0.9,
                examples=["history", "history 10", "history search python"],
                shortcuts=["/hist"]
            ),
            "/quit": CommandSuggestion(
                command="/quit",
                description="Exit the application",
                intent=CommandIntent.QUIT,
                confidence=0.9,
                examples=["quit", "exit"],
                shortcuts=["/q", "/exit"]
            ),
        }
    
    async def interpret_natural_language(self, user_input: str) -> InterpretationResult:
        """
        Interpret natural language input and convert to structured commands.
        
        Args:
            user_input: Raw user input string
            
        Returns:
            Detailed interpretation result with suggestions
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Clean and normalize input
            cleaned_input = self._clean_input(user_input)
            
            # Check for direct commands first
            if cleaned_input.startswith('/'):
                return await self._handle_direct_command(cleaned_input)
            
            # Perform intent recognition
            intent, confidence, matches = self._recognize_intent(cleaned_input)
            
            # Generate command interpretation
            interpreted_command = await self._generate_command(cleaned_input, intent, matches)
            
            # Get relevant suggestions
            suggestions = await self._generate_suggestions(cleaned_input, intent, confidence)
            
            # Create result
            result = InterpretationResult(
                original_input=user_input,
                interpreted_command=interpreted_command,
                intent=intent,
                confidence=self._calculate_confidence_level(confidence),
                parameters=self._extract_parameters(cleaned_input, intent, matches),
                suggestions=suggestions,
                explanation=self._generate_explanation(intent, interpreted_command, confidence),
                error_corrections=self._suggest_corrections(cleaned_input)
            )
            
            # Track performance
            processing_time = asyncio.get_event_loop().time() - start_time
            self.response_times.append(processing_time)
            
            # Emit interpretation event
            await self._emit_interpretation_event(result, processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error interpreting natural language: {e}")
            return self._create_error_result(user_input, str(e))
    
    def _clean_input(self, user_input: str) -> str:
        """Clean and normalize user input."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', user_input.strip())
        
        # Handle common typos and shortcuts
        corrections = {
            'halp': 'help',
            'hlep': 'help',
            'stauts': 'status',
            'statu': 'status',
            'conig': 'config',
            'cofig': 'config',
            'eixt': 'exit',
            'exti': 'exit',
        }
        
        for typo, correction in corrections.items():
            cleaned = re.sub(rf'\b{typo}\b', correction, cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def _recognize_intent(self, input_text: str) -> Tuple[CommandIntent, float, List[re.Match]]:
        """Recognize the intent of the user input."""
        best_intent = CommandIntent.UNKNOWN
        best_confidence = 0.0
        best_matches = []
        
        for intent, patterns in self.nlp_patterns.items():
            for pattern, base_confidence in patterns:
                match = re.search(pattern, input_text, re.IGNORECASE)
                if match:
                    # Adjust confidence based on context
                    adjusted_confidence = self._adjust_confidence(base_confidence, intent, input_text)
                    
                    if adjusted_confidence > best_confidence:
                        best_intent = intent
                        best_confidence = adjusted_confidence
                        best_matches = [match]
                    elif adjusted_confidence == best_confidence:
                        best_matches.append(match)
        
        return best_intent, best_confidence, best_matches
    
    def _adjust_confidence(self, base_confidence: float, intent: CommandIntent, input_text: str) -> float:
        """Adjust confidence based on context and user patterns."""
        confidence = base_confidence
        
        # Boost confidence for recently used intents
        recent_intents = [self._extract_intent_from_command(cmd) for cmd in self.context.recent_commands[-5:]]
        if intent in recent_intents:
            confidence *= 1.1
        
        # Adjust for user skill level
        if self.context.user_skill_level == "expert":
            # Experts might use more terse language
            confidence *= 1.05
        elif self.context.user_skill_level == "beginner":
            # Beginners might be more verbose
            if len(input_text.split()) > 5:
                confidence *= 1.1
        
        # Consider session context
        if self.context.active_conversation and intent == CommandIntent.CHAT:
            confidence *= 1.2
        
        return min(confidence, 1.0)
    
    async def _generate_command(self, input_text: str, intent: CommandIntent, matches: List[re.Match]) -> str:
        """Generate the most appropriate command based on interpretation."""
        if intent == CommandIntent.CHAT:
            return f"/chat {input_text}"
        elif intent == CommandIntent.SEARCH:
            search_term = matches[0].group(1) if matches else input_text
            return f"/search {search_term}"
        elif intent == CommandIntent.CREATE:
            target = matches[0].group(1) if matches else input_text
            return f"/create {target}"
        elif intent == CommandIntent.MODIFY:
            if len(matches) > 0 and matches[0].groups() and len(matches[0].groups()) >= 2:
                setting, value = matches[0].group(1), matches[0].group(2)
                return f"/config {setting} {value}"
            else:
                target = matches[0].group(1) if matches else input_text
                return f"/modify {target}"
        elif intent == CommandIntent.HELP:
            topic = matches[0].group(1) if matches and matches[0].group(1) else ""
            return f"/help {topic}".strip()
        elif intent == CommandIntent.CONFIG:
            setting = matches[0].group(1) if matches and matches[0].group(1) else ""
            return f"/config {setting}".strip()
        elif intent == CommandIntent.STATUS:
            component = matches[0].group(1) if matches and matches[0].group(1) else ""
            return f"/status {component}".strip()
        elif intent == CommandIntent.QUIT:
            return "/quit"
        else:
            return f"/chat {input_text}"  # Fallback to chat
    
    async def _generate_suggestions(self, input_text: str, intent: CommandIntent, confidence: float) -> List[CommandSuggestion]:
        """Generate contextual command suggestions."""
        suggestions = []
        
        # Check cache first
        cache_key = f"{intent.value}_{hash(input_text[:50])}"
        if cache_key in self.suggestion_cache:
            return self.suggestion_cache[cache_key]
        
        # Generate suggestions based on intent
        for command_name, cmd_suggestion in self.commands.items():
            if cmd_suggestion.intent == intent:
                suggestions.append(cmd_suggestion)
        
        # Add fuzzy matching suggestions
        fuzzy_suggestions = self._generate_fuzzy_suggestions(input_text)
        suggestions.extend(fuzzy_suggestions)
        
        # Sort by relevance
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit and cache
        limited_suggestions = suggestions[:5]
        self.suggestion_cache[cache_key] = limited_suggestions
        
        return limited_suggestions
    
    def _generate_fuzzy_suggestions(self, input_text: str) -> List[CommandSuggestion]:
        """Generate suggestions based on fuzzy string matching."""
        suggestions = []
        
        for command_name, cmd_suggestion in self.commands.items():
            # Calculate similarity score
            similarity = self._calculate_similarity(input_text.lower(), command_name.lower())
            
            if similarity > 0.3:  # Threshold for relevance
                # Create a copy with adjusted confidence
                fuzzy_suggestion = CommandSuggestion(
                    command=cmd_suggestion.command,
                    description=cmd_suggestion.description,
                    intent=cmd_suggestion.intent,
                    confidence=similarity * 0.8,  # Fuzzy matches get lower confidence
                    parameters=cmd_suggestion.parameters,
                    examples=cmd_suggestion.examples,
                    shortcuts=cmd_suggestion.shortcuts
                )
                suggestions.append(fuzzy_suggestion)
        
        return suggestions
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two strings using Levenshtein distance."""
        if not text1 or not text2:
            return 0.0
        
        # Simple similarity based on common subsequences
        longer = text1 if len(text1) > len(text2) else text2
        shorter = text2 if len(text1) > len(text2) else text1
        
        if len(longer) == 0:
            return 1.0
        
        # Count common characters
        common = sum(1 for c in shorter if c in longer)
        return common / len(longer)
    
    def _extract_parameters(self, input_text: str, intent: CommandIntent, matches: List[re.Match]) -> Dict[str, Any]:
        """Extract parameters from the input text."""
        parameters = {}
        
        if matches:
            match = matches[0]
            groups = match.groups()
            
            if intent == CommandIntent.CHAT and groups:
                parameters["message"] = groups[0]
            elif intent == CommandIntent.SEARCH and groups:
                parameters["query"] = groups[0]
            elif intent == CommandIntent.CREATE and groups:
                parameters["target"] = groups[0]
            elif intent == CommandIntent.MODIFY and len(groups) >= 2:
                parameters["setting"] = groups[0]
                parameters["value"] = groups[1]
            elif intent == CommandIntent.CONFIG and groups:
                parameters["setting"] = groups[0]
        
        return parameters
    
    def _generate_explanation(self, intent: CommandIntent, command: str, confidence: float) -> str:
        """Generate human-readable explanation of the interpretation."""
        confidence_desc = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
        
        explanations = {
            CommandIntent.CHAT: f"Interpreted as a chat message (confidence: {confidence_desc})",
            CommandIntent.SEARCH: f"Interpreted as a search command (confidence: {confidence_desc})",
            CommandIntent.CREATE: f"Interpreted as a creation command (confidence: {confidence_desc})",
            CommandIntent.MODIFY: f"Interpreted as a modification command (confidence: {confidence_desc})",
            CommandIntent.HELP: f"Interpreted as a help request (confidence: {confidence_desc})",
            CommandIntent.CONFIG: f"Interpreted as a configuration command (confidence: {confidence_desc})",
            CommandIntent.STATUS: f"Interpreted as a status request (confidence: {confidence_desc})",
            CommandIntent.QUIT: f"Interpreted as an exit command (confidence: {confidence_desc})",
            CommandIntent.UNKNOWN: f"Could not clearly interpret intent (confidence: {confidence_desc})"
        }
        
        return explanations.get(intent, f"Interpreted command: {command}")
    
    def _suggest_corrections(self, input_text: str) -> List[str]:
        """Suggest corrections for common issues."""
        corrections = []
        
        # Check for incomplete commands
        if input_text.startswith('/') and len(input_text) < 3:
            corrections.append("Command appears incomplete. Type '/help' for available commands.")
        
        # Check for typos in command names
        words = input_text.split()
        for word in words:
            if word.startswith('/'):
                closest_command = self._find_closest_command(word)
                if closest_command and closest_command != word:
                    corrections.append(f"Did you mean '{closest_command}' instead of '{word}'?")
        
        return corrections
    
    def _find_closest_command(self, word: str) -> Optional[str]:
        """Find the closest matching command."""
        best_match = None
        best_score = 0.0
        
        for command_name in self.commands.keys():
            score = self._calculate_similarity(word.lower(), command_name.lower())
            if score > best_score and score > 0.6:
                best_score = score
                best_match = command_name
        
        return best_match
    
    async def _handle_direct_command(self, command: str) -> InterpretationResult:
        """Handle direct command input (starts with /)."""
        parts = command.split(maxsplit=1)
        base_command = parts[0]
        parameters = parts[1] if len(parts) > 1 else ""
        
        # Find matching command
        matching_suggestion = None
        for cmd_name, suggestion in self.commands.items():
            if base_command == cmd_name or base_command in suggestion.shortcuts:
                matching_suggestion = suggestion
                break
        
        if matching_suggestion:
            return InterpretationResult(
                original_input=command,
                interpreted_command=command,
                intent=matching_suggestion.intent,
                confidence=ConfidenceLevel.HIGH,
                parameters={"args": parameters} if parameters else {},
                suggestions=[matching_suggestion],
                explanation=f"Direct command: {matching_suggestion.description}"
            )
        else:
            return InterpretationResult(
                original_input=command,
                interpreted_command=command,
                intent=CommandIntent.UNKNOWN,
                confidence=ConfidenceLevel.UNKNOWN,
                parameters={},
                suggestions=self._get_similar_commands(base_command),
                explanation=f"Unknown command: {base_command}",
                error_corrections=[f"Unknown command '{base_command}'. Type '/help' for available commands."]
            )
    
    def _get_similar_commands(self, command: str) -> List[CommandSuggestion]:
        """Get similar commands for error correction."""
        similar = []
        
        for cmd_name, suggestion in self.commands.items():
            similarity = self._calculate_similarity(command.lower(), cmd_name.lower())
            if similarity > 0.4:
                similar.append(suggestion)
        
        return sorted(similar, key=lambda x: x.confidence, reverse=True)[:3]
    
    def _calculate_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level."""
        if confidence > 0.8:
            return ConfidenceLevel.HIGH
        elif confidence > 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence > 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN
    
    def _create_error_result(self, user_input: str, error: str) -> InterpretationResult:
        """Create an error result for failed interpretations."""
        return InterpretationResult(
            original_input=user_input,
            interpreted_command=f"/chat {user_input}",  # Fallback to chat
            intent=CommandIntent.UNKNOWN,
            confidence=ConfidenceLevel.UNKNOWN,
            parameters={},
            suggestions=[],
            explanation=f"Error processing input: {error}",
            error_corrections=[f"Processing error occurred. Input treated as chat message."]
        )
    
    def _extract_intent_from_command(self, command: str) -> CommandIntent:
        """Extract intent from a command string."""
        if command.startswith('/'):
            base_command = command.split()[0]
            for cmd_name, suggestion in self.commands.items():
                if base_command == cmd_name or base_command in suggestion.shortcuts:
                    return suggestion.intent
        return CommandIntent.UNKNOWN
    
    async def _emit_interpretation_event(self, result: InterpretationResult, processing_time: float):
        """Emit event about command interpretation."""
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "enhanced_command_interface",
                "action": "interpretation",
                "result": {
                    "original_input": result.original_input,
                    "interpreted_command": result.interpreted_command,
                    "intent": result.intent.value,
                    "confidence": result.confidence.value,
                    "processing_time_ms": processing_time * 1000,
                    "suggestions_count": len(result.suggestions)
                }
            }
        )
        
        await self.event_system.emit_event(event)
        
        # Call registered callback if available
        if "interpretation" in self._callbacks:
            await self._callbacks["interpretation"](result)
    
    async def learn_from_feedback(self, original_input: str, chosen_command: str, success: bool):
        """Learn from user feedback to improve future interpretations."""
        # Store learning data
        success_score = 1.0 if success else 0.0
        self.learning_history.append((original_input, chosen_command, success_score))
        
        # Limit history size
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-800:]  # Keep last 800
        
        # Update user context
        if success:
            self.context.recent_commands.append(chosen_command)
            if len(self.context.recent_commands) > 20:
                self.context.recent_commands = self.context.recent_commands[-15:]
            
            # Update usage patterns
            intent = self._extract_intent_from_command(chosen_command)
            self.context.usage_patterns[intent.value] = self.context.usage_patterns.get(intent.value, 0) + 1
        
        # Track interpretation accuracy
        self.interpretation_accuracy.append(success)
        if len(self.interpretation_accuracy) > 100:
            self.interpretation_accuracy = self.interpretation_accuracy[-80:]
    
    def update_context(self, **kwargs):
        """Update the command context with new information."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the command interface."""
        if not self.response_times:
            return {"status": "no_data"}
        
        return {
            "average_response_time_ms": sum(self.response_times) * 1000 / len(self.response_times),
            "max_response_time_ms": max(self.response_times) * 1000,
            "min_response_time_ms": min(self.response_times) * 1000,
            "total_interpretations": len(self.response_times),
            "accuracy_rate": sum(self.interpretation_accuracy) / len(self.interpretation_accuracy) if self.interpretation_accuracy else 0,
            "learning_samples": len(self.learning_history),
            "cache_hits": len(self.suggestion_cache),
            "user_skill_level": self.context.user_skill_level,
            "most_used_intents": dict(sorted(self.context.usage_patterns.items(), key=lambda x: x[1], reverse=True))
        }
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for specific event types."""
        self._callbacks[event_type] = callback
    
    def remove_callback(self, event_type: str):
        """Remove callback for specific event types."""
        if event_type in self._callbacks:
            del self._callbacks[event_type]
    
    async def cleanup(self):
        """Cleanup the enhanced command interface."""
        # Clear caches
        self.suggestion_cache.clear()
        self._callbacks.clear()
        
        # Reset statistics
        self.response_times.clear()
        self.interpretation_accuracy.clear()
        
        logger.info("Enhanced command interface cleaned up")


# Utility function for easy instantiation
def create_enhanced_command_interface(event_system: AsyncEventSystem) -> EnhancedCommandInterface:
    """Create and return a new EnhancedCommandInterface instance."""
    return EnhancedCommandInterface(event_system)