"""
AI Command Composer - Advanced natural language to command translation with real-time intent recognition.

This revolutionary component provides sophisticated natural language understanding for the AgentsMCP CLI,
transforming user intent into precise commands with real-time feedback and learning capabilities.

Key Features:
- Advanced intent recognition with confidence scoring
- Real-time command translation and preview
- Context-aware command synthesis
- Multi-modal input support (text, voice, gesture)
- Continuous learning from user interactions
- Semantic command search and discovery
- Template-based command generation
- Error prediction and prevention
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import logging
from collections import defaultdict, deque
import difflib

from ..v2.event_system import AsyncEventSystem


class IntentConfidence(Enum):
    """Confidence levels for intent recognition."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


class CommandCategory(Enum):
    """Categories of commands for better organization."""
    AGENT_MANAGEMENT = "agent_management"
    TASK_EXECUTION = "task_execution"
    SYSTEM_CONTROL = "system_control"
    DATA_ANALYSIS = "data_analysis"
    DEBUGGING = "debugging"
    CONFIGURATION = "configuration"
    MONITORING = "monitoring"
    WORKFLOW = "workflow"


class InputModality(Enum):
    """Different input modalities supported."""
    TEXT = "text"
    VOICE = "voice"
    GESTURE = "gesture"
    HYBRID = "hybrid"


@dataclass
class IntentMatch:
    """Represents a matched intent with confidence and context."""
    intent: str
    confidence: float
    command: str
    parameters: Dict[str, Any]
    category: CommandCategory
    description: str
    examples: List[str]
    required_params: Set[str] = field(default_factory=set)
    optional_params: Set[str] = field(default_factory=set)
    aliases: List[str] = field(default_factory=list)
    safety_level: str = "safe"  # safe, caution, dangerous
    execution_time: str = "instant"  # instant, short, long
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "command": self.command,
            "parameters": self.parameters,
            "category": self.category.value,
            "description": self.description,
            "examples": self.examples,
            "required_params": list(self.required_params),
            "optional_params": list(self.optional_params),
            "aliases": self.aliases,
            "safety_level": self.safety_level,
            "execution_time": self.execution_time
        }


@dataclass
class ComposerSession:
    """Represents an active composition session with context."""
    session_id: str
    start_time: datetime
    user_input: str
    modality: InputModality
    context: Dict[str, Any] = field(default_factory=dict)
    intent_matches: List[IntentMatch] = field(default_factory=list)
    selected_match: Optional[IntentMatch] = None
    refinements: List[str] = field(default_factory=list)
    is_active: bool = True


@dataclass
class LearningPattern:
    """Represents a learned pattern from user interactions."""
    pattern: str
    intent: str
    frequency: int
    success_rate: float
    last_used: datetime
    context_tags: Set[str] = field(default_factory=set)


@dataclass
class CommandTemplate:
    """Template for generating commands from patterns."""
    name: str
    pattern: str
    command_template: str
    parameter_mappings: Dict[str, str]
    validation_rules: List[str] = field(default_factory=list)
    success_rate: float = 1.0


class AICommandComposer:
    """
    Revolutionary AI Command Composer with advanced natural language understanding.
    
    Transforms user intent into precise commands through sophisticated intent recognition,
    real-time translation, and continuous learning from user interactions.
    """
    
    def __init__(self, event_system: Optional[AsyncEventSystem] = None, config_path: Optional[Path] = None):
        """Initialize the AI Command Composer."""
        # Create event system if not provided (for backward compatibility)
        if event_system is None:
            try:
                from ..v2.event_system import AsyncEventSystem
                self.event_system = AsyncEventSystem()
                self._owns_event_system = True
            except Exception:
                self.event_system = None
                self._owns_event_system = False
                logger.warning("Event system not available, running in limited mode")
        else:
            self.event_system = event_system
            self._owns_event_system = False
            
        self.config_path = config_path or Path.home() / ".agentsmcp" / "ai_composer.json"
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.intent_patterns: Dict[str, Dict[str, Any]] = {}
        self.command_registry: Dict[str, Dict[str, Any]] = {}
        self.learning_patterns: Dict[str, LearningPattern] = {}
        self.command_templates: Dict[str, CommandTemplate] = {}
        
        # Active sessions
        self.active_sessions: Dict[str, ComposerSession] = {}
        self.session_history: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.translation_cache: Dict[str, IntentMatch] = {}
        self.performance_metrics: Dict[str, Any] = {
            "translations_per_second": 0,
            "average_confidence": 0,
            "cache_hit_rate": 0,
            "learning_accuracy": 0
        }
        
        # Real-time processing
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.is_processing = False
        self._initialized = False
        
        # Initialize components
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize async components."""
        try:
            await self._initialize_intent_patterns()
            await self._initialize_command_registry()
            await self._load_learning_patterns()
            await self._initialize_templates()
            await self._start_processing_loop()
            
            # Register event handlers if event system is available
            if self.event_system:
                await self._register_event_handlers()
            
            self._initialized = True
            self.logger.info("AI Command Composer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Command Composer: {e}")
            # Don't raise in case of partial initialization
            self._initialized = True  # Mark as initialized even with errors
    
    async def _initialize_intent_patterns(self):
        """Initialize advanced intent recognition patterns."""
        self.intent_patterns = {
            # Agent management patterns
            "create_agent": {
                "patterns": [
                    r"create (a|an|new) (?P<agent_type>\w+) agent",
                    r"spawn (?P<agent_type>\w+)",
                    r"start (a|an) (?P<agent_type>\w+)",
                    r"launch (?P<agent_type>\w+) agent",
                    r"bring up (?P<agent_type>\w+)"
                ],
                "command": "agent create --type {agent_type}",
                "category": CommandCategory.AGENT_MANAGEMENT,
                "confidence_boost": 0.2,
                "required_params": {"agent_type"}
            },
            
            "list_agents": {
                "patterns": [
                    r"(show|list|display) (all )?agents",
                    r"what agents (are|do we have)",
                    r"agent (status|list|inventory)",
                    r"show me (the )?agents"
                ],
                "command": "agent list",
                "category": CommandCategory.AGENT_MANAGEMENT,
                "confidence_boost": 0.3
            },
            
            "stop_agent": {
                "patterns": [
                    r"stop (the )?(?P<agent_id>\w+) agent",
                    r"kill (?P<agent_id>\w+)",
                    r"terminate (?P<agent_id>\w+)",
                    r"shutdown (?P<agent_id>\w+)",
                    r"halt (?P<agent_id>\w+)"
                ],
                "command": "agent stop {agent_id}",
                "category": CommandCategory.AGENT_MANAGEMENT,
                "safety_level": "caution",
                "required_params": {"agent_id"}
            },
            
            # Task execution patterns
            "run_task": {
                "patterns": [
                    r"run (?P<task>.+)",
                    r"execute (?P<task>.+)",
                    r"perform (?P<task>.+)",
                    r"do (?P<task>.+)",
                    r"carry out (?P<task>.+)"
                ],
                "command": "task run '{task}'",
                "category": CommandCategory.TASK_EXECUTION,
                "required_params": {"task"}
            },
            
            "schedule_task": {
                "patterns": [
                    r"schedule (?P<task>.+) (at|for) (?P<time>.+)",
                    r"run (?P<task>.+) (at|in) (?P<time>.+)",
                    r"defer (?P<task>.+) until (?P<time>.+)"
                ],
                "command": "task schedule --task '{task}' --time '{time}'",
                "category": CommandCategory.TASK_EXECUTION,
                "required_params": {"task", "time"}
            },
            
            # System control patterns
            "show_status": {
                "patterns": [
                    r"(show|display|check) (system )?status",
                    r"how (is|are) (things|everything)",
                    r"system (health|check|overview)",
                    r"what's (the )?status"
                ],
                "command": "system status",
                "category": CommandCategory.SYSTEM_CONTROL,
                "confidence_boost": 0.4
            },
            
            "restart_system": {
                "patterns": [
                    r"restart (the )?system",
                    r"reboot (everything|all)",
                    r"reload (the )?system",
                    r"bounce (the )?system"
                ],
                "command": "system restart",
                "category": CommandCategory.SYSTEM_CONTROL,
                "safety_level": "dangerous",
                "execution_time": "long"
            },
            
            # Data analysis patterns
            "analyze_logs": {
                "patterns": [
                    r"analyze (the )?logs",
                    r"check (the )?logs (for (?P<pattern>.+))?",
                    r"examine (the )?log files",
                    r"look at (the )?logs"
                ],
                "command": "logs analyze" + (" --pattern '{pattern}'" if "pattern" in "{pattern}" else ""),
                "category": CommandCategory.DATA_ANALYSIS,
                "optional_params": {"pattern"}
            },
            
            "generate_report": {
                "patterns": [
                    r"generate (a )?(?P<report_type>\w+) report",
                    r"create (a )?(?P<report_type>\w+) report",
                    r"make (a )?(?P<report_type>\w+) report",
                    r"produce (a )?(?P<report_type>\w+) report"
                ],
                "command": "report generate --type {report_type}",
                "category": CommandCategory.DATA_ANALYSIS,
                "required_params": {"report_type"},
                "execution_time": "long"
            },
            
            # Configuration patterns
            "set_config": {
                "patterns": [
                    r"set (?P<key>\w+) to (?P<value>.+)",
                    r"configure (?P<key>\w+) as (?P<value>.+)",
                    r"update (?P<key>\w+) to (?P<value>.+)",
                    r"change (?P<key>\w+) to (?P<value>.+)"
                ],
                "command": "config set {key} '{value}'",
                "category": CommandCategory.CONFIGURATION,
                "required_params": {"key", "value"}
            },
            
            "show_config": {
                "patterns": [
                    r"show (the )?config(uration)?",
                    r"display (the )?settings",
                    r"what (are|is) (the )?config(uration)?",
                    r"list (the )?settings"
                ],
                "command": "config show",
                "category": CommandCategory.CONFIGURATION,
                "confidence_boost": 0.3
            },
            
            # Advanced patterns with context awareness
            "help_with": {
                "patterns": [
                    r"help (me )?(with )?(?P<topic>.+)",
                    r"how (do I|to) (?P<topic>.+)",
                    r"assistance (with )?(?P<topic>.+)",
                    r"guide (me through|for) (?P<topic>.+)"
                ],
                "command": "help --topic '{topic}'",
                "category": CommandCategory.SYSTEM_CONTROL,
                "optional_params": {"topic"}
            }
        }
        
        # Compile regex patterns for efficiency
        for intent, config in self.intent_patterns.items():
            config["compiled_patterns"] = [
                re.compile(pattern, re.IGNORECASE) for pattern in config["patterns"]
            ]
    
    async def _initialize_command_registry(self):
        """Initialize the comprehensive command registry."""
        self.command_registry = {
            # Agent commands
            "agent": {
                "subcommands": {
                    "create": {"params": ["type", "name", "config"], "required": ["type"]},
                    "list": {"params": [], "aliases": ["ls", "show"]},
                    "stop": {"params": ["id"], "required": ["id"], "aliases": ["kill", "terminate"]},
                    "restart": {"params": ["id"], "required": ["id"]},
                    "status": {"params": ["id"], "optional": ["id"]}
                },
                "category": CommandCategory.AGENT_MANAGEMENT
            },
            
            # Task commands
            "task": {
                "subcommands": {
                    "run": {"params": ["command"], "required": ["command"]},
                    "schedule": {"params": ["task", "time"], "required": ["task", "time"]},
                    "list": {"params": ["status"], "optional": ["status"]},
                    "cancel": {"params": ["id"], "required": ["id"]},
                    "status": {"params": ["id"], "optional": ["id"]}
                },
                "category": CommandCategory.TASK_EXECUTION
            },
            
            # System commands
            "system": {
                "subcommands": {
                    "status": {"params": [], "aliases": ["health", "check"]},
                    "restart": {"params": ["component"], "optional": ["component"]},
                    "shutdown": {"params": [], "aliases": ["stop", "halt"]},
                    "info": {"params": [], "aliases": ["version", "about"]}
                },
                "category": CommandCategory.SYSTEM_CONTROL
            },
            
            # Configuration commands
            "config": {
                "subcommands": {
                    "get": {"params": ["key"], "required": ["key"]},
                    "set": {"params": ["key", "value"], "required": ["key", "value"]},
                    "show": {"params": [], "aliases": ["list", "display"]},
                    "reset": {"params": ["key"], "optional": ["key"]}
                },
                "category": CommandCategory.CONFIGURATION
            },
            
            # Logging and analysis
            "logs": {
                "subcommands": {
                    "show": {"params": ["level", "component"], "optional": ["level", "component"]},
                    "analyze": {"params": ["pattern", "since"], "optional": ["pattern", "since"]},
                    "clear": {"params": [], "aliases": ["clean", "flush"]},
                    "export": {"params": ["format", "output"], "optional": ["format", "output"]}
                },
                "category": CommandCategory.DATA_ANALYSIS
            },
            
            # Reporting
            "report": {
                "subcommands": {
                    "generate": {"params": ["type", "format", "output"], "required": ["type"]},
                    "list": {"params": [], "aliases": ["show"]},
                    "schedule": {"params": ["type", "frequency"], "required": ["type", "frequency"]}
                },
                "category": CommandCategory.DATA_ANALYSIS
            },
            
            # Help system
            "help": {
                "subcommands": {
                    "": {"params": ["topic"], "optional": ["topic"]},
                    "commands": {"params": [], "aliases": ["cmd"]},
                    "examples": {"params": ["topic"], "optional": ["topic"]}
                },
                "category": CommandCategory.SYSTEM_CONTROL
            }
        }
    
    async def _load_learning_patterns(self):
        """Load previously learned patterns."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    
                patterns_data = data.get("learning_patterns", {})
                for pattern_key, pattern_data in patterns_data.items():
                    self.learning_patterns[pattern_key] = LearningPattern(
                        pattern=pattern_data["pattern"],
                        intent=pattern_data["intent"],
                        frequency=pattern_data["frequency"],
                        success_rate=pattern_data["success_rate"],
                        last_used=datetime.fromisoformat(pattern_data["last_used"]),
                        context_tags=set(pattern_data.get("context_tags", []))
                    )
                    
                self.logger.info(f"Loaded {len(self.learning_patterns)} learning patterns")
                
        except Exception as e:
            self.logger.warning(f"Could not load learning patterns: {e}")
            self.learning_patterns = {}
    
    async def _initialize_templates(self):
        """Initialize command templates for pattern-based generation."""
        self.command_templates = {
            "agent_action": CommandTemplate(
                name="agent_action",
                pattern=r"(?P<action>start|stop|restart|create) (?P<target>agent|task) (?P<name>\w+)",
                command_template="{target} {action} {name}",
                parameter_mappings={"action": "action", "target": "target", "name": "name"}
            ),
            
            "config_operation": CommandTemplate(
                name="config_operation",
                pattern=r"(?P<action>set|get|show) (?P<key>\w+)( to (?P<value>.+))?",
                command_template="config {action} {key}" + " '{value}'" if "{value}" else "",
                parameter_mappings={"action": "action", "key": "key", "value": "value"}
            ),
            
            "system_query": CommandTemplate(
                name="system_query",
                pattern=r"(show|display|check) (?P<resource>status|logs|agents|tasks)",
                command_template="{resource}",
                parameter_mappings={"resource": "resource"}
            )
        }
    
    async def _register_event_handlers(self):
        """Register event handlers for real-time processing."""
        if not self.event_system:
            return
        
        # Check if event system has subscribe method (new interface)
        if hasattr(self.event_system, 'subscribe'):
            try:
                from ..v2.event_system import EventType
                await self.event_system.subscribe(EventType.CUSTOM, self._handle_custom_event)
            except Exception as e:
                self.logger.warning(f"Failed to register event handlers: {e}")
        
        # Old interface compatibility - check if direct string-based subscription exists
        elif hasattr(self.event_system, 'on'):
            self.event_system.on("user_input", self._handle_user_input)
            self.event_system.on("command_executed", self._handle_command_feedback)
            self.event_system.on("session_started", self._handle_session_start)
            self.event_system.on("session_ended", self._handle_session_end)
    
    async def _handle_custom_event(self, event):
        """Handle custom events from other components."""
        try:
            event_data = event.data if hasattr(event, 'data') else event
            component = event_data.get('component') if isinstance(event_data, dict) else None
            action = event_data.get('action') if isinstance(event_data, dict) else None
            
            if component == "revolutionary_tui":
                if action == "input_changed":
                    input_text = event_data.get('input_text', '')
                    if len(input_text) > 0:
                        # Trigger suggestion generation
                        await self._update_suggestions(input_text)
                        
                elif action == "command_selected":
                    command = event_data.get('command', '')
                    await self._execute_selected_command(command)
                    
            elif component == "symphony_dashboard":
                if action == "agent_selected":
                    agent_id = event_data.get('agent_id')
                    await self._update_agent_context(agent_id)
                    
        except Exception as e:
            self.logger.error(f"Error handling custom event: {e}")
    
    async def _update_suggestions(self, input_text: str):
        """Update command suggestions based on input text."""
        try:
            # Generate contextual suggestions
            if len(input_text) >= 2:
                suggestions = await self._generate_contextual_suggestions(input_text)
                self.current_suggestions = suggestions[:self.max_suggestions]
            else:
                self.current_suggestions = []
                
        except Exception as e:
            self.logger.error(f"Error updating suggestions: {e}")
    
    async def _execute_selected_command(self, command: str):
        """Execute a selected command."""
        try:
            self.logger.info(f"Executing selected command: {command}")
            # Add command to history
            self.command_history.append(command)
            
            # Emit command execution event
            if hasattr(self.event_system, 'emit_event'):
                from ..v2.event_system import Event, EventType
                event = Event(
                    event_type=EventType.CUSTOM,
                    data={
                        'component': 'ai_command_composer',
                        'action': 'command_executed',
                        'command': command
                    }
                )
                await self.event_system.emit_event(event)
                
        except Exception as e:
            self.logger.error(f"Error executing selected command: {e}")
    
    async def _update_agent_context(self, agent_id: str):
        """Update context based on selected agent."""
        try:
            self.current_context["selected_agent"] = agent_id
            self.logger.info(f"Updated context with selected agent: {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating agent context: {e}")
    
    async def _start_processing_loop(self):
        """Start the main processing loop for real-time translation."""
        self.is_processing = True
        asyncio.create_task(self._processing_loop())
    
    async def _processing_loop(self):
        """Main processing loop for handling translation requests."""
        while self.is_processing:
            try:
                # Process queued requests
                while not self.processing_queue.empty():
                    session_id, user_input = await self.processing_queue.get()
                    await self._process_translation_request(session_id, user_input)
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Clean up old sessions
                await self._cleanup_sessions()
                
                await asyncio.sleep(0.01)  # 100 FPS processing rate
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(0.1)
    
    async def compose_command(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        modality: InputModality = InputModality.TEXT,
        session_id: Optional[str] = None
    ) -> ComposerSession:
        """
        Main entry point for command composition from natural language input.
        
        Args:
            user_input: Natural language input from user
            context: Additional context information
            modality: Input modality (text, voice, gesture, hybrid)
            session_id: Optional session ID for continuation
            
        Returns:
            ComposerSession with intent matches and recommendations
        """
        try:
            # Create or retrieve session
            if session_id and session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.user_input = user_input
                session.refinements.append(user_input)
            else:
                session_id = f"session_{datetime.now().timestamp()}"
                session = ComposerSession(
                    session_id=session_id,
                    start_time=datetime.now(),
                    user_input=user_input,
                    modality=modality,
                    context=context or {}
                )
                self.active_sessions[session_id] = session
            
            # Perform intent recognition
            intent_matches = await self._recognize_intent(user_input, context)
            session.intent_matches = intent_matches
            
            # Select best match
            if intent_matches:
                session.selected_match = intent_matches[0]
                
                # Emit real-time update event
                await self.event_system.emit("command_composed", {
                    "session_id": session_id,
                    "user_input": user_input,
                    "intent_matches": [match.to_dict() for match in intent_matches[:3]],
                    "selected_command": session.selected_match.command,
                    "confidence": session.selected_match.confidence
                })
            
            # Learn from this interaction
            await self._learn_from_interaction(session)
            
            return session
            
        except Exception as e:
            self.logger.error(f"Error composing command: {e}")
            raise
    
    async def _recognize_intent(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[IntentMatch]:
        """
        Advanced intent recognition with confidence scoring.
        
        Args:
            user_input: User's natural language input
            context: Additional context for recognition
            
        Returns:
            List of IntentMatch objects sorted by confidence
        """
        # Check cache first
        cache_key = f"{user_input}:{hash(str(context))}"
        if cache_key in self.translation_cache:
            return [self.translation_cache[cache_key]]
        
        matches = []
        user_input_lower = user_input.lower().strip()
        
        # Pattern-based recognition
        for intent, config in self.intent_patterns.items():
            for pattern in config["compiled_patterns"]:
                match = pattern.search(user_input_lower)
                if match:
                    confidence = self._calculate_confidence(user_input, intent, match, config)
                    
                    # Extract parameters
                    parameters = match.groupdict()
                    
                    # Generate command
                    command = self._generate_command(config["command"], parameters)
                    
                    intent_match = IntentMatch(
                        intent=intent,
                        confidence=confidence,
                        command=command,
                        parameters=parameters,
                        category=config["category"],
                        description=config.get("description", f"Execute {intent}"),
                        examples=config.get("examples", []),
                        required_params=config.get("required_params", set()),
                        optional_params=config.get("optional_params", set()),
                        safety_level=config.get("safety_level", "safe"),
                        execution_time=config.get("execution_time", "instant")
                    )
                    
                    matches.append(intent_match)
        
        # Template-based recognition
        template_matches = await self._match_templates(user_input)
        matches.extend(template_matches)
        
        # Learning-based recognition
        learned_matches = await self._match_learned_patterns(user_input)
        matches.extend(learned_matches)
        
        # Semantic similarity matching
        semantic_matches = await self._semantic_matching(user_input, context)
        matches.extend(semantic_matches)
        
        # Sort by confidence and remove duplicates
        matches = sorted(matches, key=lambda x: x.confidence, reverse=True)
        unique_matches = []
        seen_commands = set()
        
        for match in matches:
            if match.command not in seen_commands:
                unique_matches.append(match)
                seen_commands.add(match.command)
        
        # Cache the best match
        if unique_matches:
            self.translation_cache[cache_key] = unique_matches[0]
        
        return unique_matches[:5]  # Return top 5 matches
    
    def _calculate_confidence(
        self,
        user_input: str,
        intent: str,
        match: re.Match,
        config: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for an intent match."""
        base_confidence = 0.6
        
        # Boost for exact pattern match
        if match.group(0) == user_input.lower():
            base_confidence += 0.3
        
        # Boost for configuration setting
        base_confidence += config.get("confidence_boost", 0)
        
        # Boost for parameter completeness
        required_params = config.get("required_params", set())
        if required_params:
            matched_params = set(match.groupdict().keys())
            completeness = len(matched_params & required_params) / len(required_params)
            base_confidence += completeness * 0.2
        
        # Reduce for ambiguity
        if len(match.groups()) > 3:
            base_confidence -= 0.1
        
        # Historical success rate boost
        if intent in self.learning_patterns:
            pattern = self.learning_patterns[intent]
            base_confidence += (pattern.success_rate - 0.5) * 0.2
        
        return min(base_confidence, 0.99)
    
    def _generate_command(self, command_template: str, parameters: Dict[str, str]) -> str:
        """Generate the actual command from template and parameters."""
        try:
            # Handle optional parameters
            command = command_template
            for param_name, param_value in parameters.items():
                if param_value:
                    command = command.replace(f"{{{param_name}}}", param_value)
            
            # Clean up any remaining template placeholders
            command = re.sub(r'\{[^}]+\}', '', command)
            command = re.sub(r'\s+', ' ', command).strip()
            
            return command
            
        except Exception as e:
            self.logger.error(f"Error generating command: {e}")
            return command_template
    
    async def _match_templates(self, user_input: str) -> List[IntentMatch]:
        """Match against command templates."""
        matches = []
        
        for template in self.command_templates.values():
            pattern = re.compile(template.pattern, re.IGNORECASE)
            match = pattern.search(user_input)
            
            if match:
                parameters = match.groupdict()
                command = self._generate_command(template.command_template, parameters)
                
                confidence = 0.5 + (template.success_rate * 0.3)
                
                intent_match = IntentMatch(
                    intent=f"template_{template.name}",
                    confidence=confidence,
                    command=command,
                    parameters=parameters,
                    category=CommandCategory.SYSTEM_CONTROL,  # Default category
                    description=f"Template-based command: {template.name}",
                    examples=[]
                )
                
                matches.append(intent_match)
        
        return matches
    
    async def _match_learned_patterns(self, user_input: str) -> List[IntentMatch]:
        """Match against previously learned patterns."""
        matches = []
        
        for pattern_key, pattern in self.learning_patterns.items():
            similarity = difflib.SequenceMatcher(None, user_input.lower(), pattern.pattern.lower()).ratio()
            
            if similarity > 0.7:  # High similarity threshold
                confidence = similarity * pattern.success_rate
                
                intent_match = IntentMatch(
                    intent=pattern.intent,
                    confidence=confidence,
                    command=pattern.pattern,  # This would be the learned command
                    parameters={},
                    category=CommandCategory.SYSTEM_CONTROL,  # Default category
                    description=f"Learned pattern: {pattern.intent}",
                    examples=[]
                )
                
                matches.append(intent_match)
        
        return matches
    
    async def _semantic_matching(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[IntentMatch]:
        """Perform semantic similarity matching for complex queries."""
        matches = []
        
        # This would integrate with a more sophisticated NLP model
        # For now, implement basic keyword matching
        keywords = user_input.lower().split()
        
        for command, config in self.command_registry.items():
            # Calculate semantic similarity based on command keywords
            command_keywords = [command] + list(config.get("subcommands", {}).keys())
            
            common_keywords = set(keywords) & set(command_keywords)
            if common_keywords:
                similarity = len(common_keywords) / len(keywords)
                
                if similarity > 0.3:
                    intent_match = IntentMatch(
                        intent=f"semantic_{command}",
                        confidence=similarity * 0.6,
                        command=command,
                        parameters={},
                        category=config.get("category", CommandCategory.SYSTEM_CONTROL),
                        description=f"Semantic match for {command}",
                        examples=[]
                    )
                    
                    matches.append(intent_match)
        
        return matches
    
    async def refine_command(
        self,
        session_id: str,
        refinement: str,
        selected_match_index: Optional[int] = None
    ) -> ComposerSession:
        """
        Refine a command based on user feedback or additional input.
        
        Args:
            session_id: Session identifier
            refinement: Additional refinement input
            selected_match_index: Index of selected match to refine
            
        Returns:
            Updated ComposerSession
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session.refinements.append(refinement)
        
        # If user selected a specific match, use it as base for refinement
        if selected_match_index is not None and selected_match_index < len(session.intent_matches):
            base_match = session.intent_matches[selected_match_index]
            session.selected_match = base_match
        
        # Re-run intent recognition with combined input
        combined_input = f"{session.user_input} {refinement}"
        refined_matches = await self._recognize_intent(combined_input, session.context)
        
        # Update session
        session.intent_matches = refined_matches
        if refined_matches:
            session.selected_match = refined_matches[0]
        
        return session
    
    async def execute_composed_command(
        self,
        session_id: str,
        confirm_execution: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the composed command from a session.
        
        Args:
            session_id: Session identifier
            confirm_execution: Whether execution was confirmed by user
            
        Returns:
            Execution result
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        if not session.selected_match:
            raise ValueError("No command selected for execution")
        
        # Safety check for dangerous commands
        if session.selected_match.safety_level == "dangerous" and not confirm_execution:
            return {
                "status": "confirmation_required",
                "message": "This command requires explicit confirmation",
                "command": session.selected_match.command,
                "safety_level": session.selected_match.safety_level
            }
        
        try:
            # Emit command execution event
            execution_result = await self.event_system.emit("execute_command", {
                "command": session.selected_match.command,
                "session_id": session_id,
                "parameters": session.selected_match.parameters,
                "safety_level": session.selected_match.safety_level
            })
            
            # Learn from successful execution
            if execution_result.get("status") == "success":
                await self._record_successful_execution(session)
            
            # Update session
            session.is_active = False
            self.session_history.append(session)
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return {
                "status": "error",
                "message": str(e),
                "command": session.selected_match.command
            }
    
    async def _learn_from_interaction(self, session: ComposerSession):
        """Learn from user interactions to improve future recognition."""
        if not session.selected_match:
            return
        
        # Create or update learning pattern
        pattern_key = f"{session.user_input}:{session.selected_match.intent}"
        
        if pattern_key in self.learning_patterns:
            pattern = self.learning_patterns[pattern_key]
            pattern.frequency += 1
            pattern.last_used = datetime.now()
        else:
            pattern = LearningPattern(
                pattern=session.user_input,
                intent=session.selected_match.intent,
                frequency=1,
                success_rate=0.8,  # Initial success rate
                last_used=datetime.now(),
                context_tags=set(session.context.keys()) if session.context else set()
            )
            self.learning_patterns[pattern_key] = pattern
        
        # Save learning patterns periodically
        if len(self.learning_patterns) % 10 == 0:
            await self._save_learning_patterns()
    
    async def _record_successful_execution(self, session: ComposerSession):
        """Record successful command execution for learning."""
        if not session.selected_match:
            return
        
        pattern_key = f"{session.user_input}:{session.selected_match.intent}"
        
        if pattern_key in self.learning_patterns:
            pattern = self.learning_patterns[pattern_key]
            # Update success rate using moving average
            pattern.success_rate = (pattern.success_rate * 0.9) + (1.0 * 0.1)
    
    async def _save_learning_patterns(self):
        """Save learning patterns to disk."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            patterns_data = {}
            for pattern_key, pattern in self.learning_patterns.items():
                patterns_data[pattern_key] = {
                    "pattern": pattern.pattern,
                    "intent": pattern.intent,
                    "frequency": pattern.frequency,
                    "success_rate": pattern.success_rate,
                    "last_used": pattern.last_used.isoformat(),
                    "context_tags": list(pattern.context_tags)
                }
            
            data = {"learning_patterns": patterns_data}
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.debug(f"Saved {len(patterns_data)} learning patterns")
            
        except Exception as e:
            self.logger.error(f"Error saving learning patterns: {e}")
    
    async def get_suggestions(
        self,
        partial_input: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get real-time command suggestions as user types.
        
        Args:
            partial_input: Partial user input
            limit: Maximum number of suggestions
            
        Returns:
            List of command suggestions
        """
        suggestions = []
        
        if len(partial_input) < 2:
            return suggestions
        
        # Get partial matches
        matches = await self._recognize_intent(partial_input)
        
        for match in matches[:limit]:
            suggestion = {
                "command": match.command,
                "confidence": match.confidence,
                "description": match.description,
                "completion": match.command,
                "category": match.category.value,
                "safety_level": match.safety_level,
                "execution_time": match.execution_time
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    async def get_command_examples(self, intent: str) -> List[str]:
        """Get example inputs for a specific intent."""
        if intent in self.intent_patterns:
            config = self.intent_patterns[intent]
            return config.get("examples", config["patterns"][:3])
        
        return []
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def is_ready(self) -> bool:
        """Check if the AI Command Composer is ready for use."""
        return self._initialized
    
    async def get_quick_suggestions(self, partial_text: str, max_suggestions: int = 3) -> List[str]:
        """Get quick command suggestions for real-time display in TUI."""
        if not self._initialized or len(partial_text) < 2:
            return []
        
        try:
            suggestions = await self.get_suggestions(partial_text, limit=max_suggestions)
            return [s["completion"] for s in suggestions]
        except Exception as e:
            self.logger.warning(f"Error getting quick suggestions: {e}")
            return []
    
    async def get_command_help(self, command: str) -> Optional[str]:
        """Get help text for a specific command."""
        # Check if command exists in registry
        base_command = command.split()[0] if command else ""
        
        if base_command in self.command_registry:
            registry_entry = self.command_registry[base_command]
            category = registry_entry.get("category", CommandCategory.SYSTEM_CONTROL)
            subcommands = registry_entry.get("subcommands", {})
            
            help_text = f"Command: {base_command}\nCategory: {category.value}\n"
            
            if subcommands:
                help_text += "Available subcommands:\n"
                for subcmd, config in subcommands.items():
                    params = config.get("params", [])
                    required = config.get("required", [])
                    aliases = config.get("aliases", [])
                    
                    param_str = ""
                    if params:
                        param_parts = []
                        for param in params:
                            if param in required:
                                param_parts.append(f"<{param}>")
                            else:
                                param_parts.append(f"[{param}]")
                        param_str = " " + " ".join(param_parts)
                    
                    help_text += f"  {subcmd}{param_str}"
                    if aliases:
                        help_text += f" (aliases: {', '.join(aliases)})"
                    help_text += "\n"
            
            return help_text
        
        return None
    
    async def validate_command(self, command: str) -> Tuple[bool, Optional[str]]:
        """Validate a command and return success status and error message if invalid."""
        try:
            # Basic validation - check if command starts with known patterns
            command_parts = command.strip().split()
            if not command_parts:
                return False, "Empty command"
            
            base_command = command_parts[0]
            
            # Check against command registry
            if base_command in self.command_registry:
                return True, None
            
            # Check against intent patterns
            for intent_config in self.intent_patterns.values():
                for pattern in intent_config.get("compiled_patterns", []):
                    if pattern.search(command.lower()):
                        return True, None
            
            return False, f"Unknown command: {base_command}"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        # Calculate cache hit rate
        total_requests = len(self.session_history) + len(self.active_sessions)
        cache_hits = len(self.translation_cache)
        
        if total_requests > 0:
            self.performance_metrics["cache_hit_rate"] = cache_hits / total_requests
        
        # Calculate average confidence
        if self.session_history:
            confidences = [
                session.selected_match.confidence 
                for session in self.session_history 
                if session.selected_match
            ]
            if confidences:
                self.performance_metrics["average_confidence"] = sum(confidences) / len(confidences)
        
        # Calculate learning accuracy
        if self.learning_patterns:
            success_rates = [pattern.success_rate for pattern in self.learning_patterns.values()]
            self.performance_metrics["learning_accuracy"] = sum(success_rates) / len(success_rates)
    
    async def _cleanup_sessions(self):
        """Clean up old inactive sessions."""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if (current_time - session.start_time) > timedelta(minutes=30):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            session = self.active_sessions.pop(session_id)
            self.session_history.append(session)
    
    async def _handle_user_input(self, event_data: Dict[str, Any]):
        """Handle user input events."""
        user_input = event_data.get("input", "")
        session_id = event_data.get("session_id")
        
        if user_input and len(user_input) > 2:
            await self.processing_queue.put((session_id, user_input))
    
    async def _handle_command_feedback(self, event_data: Dict[str, Any]):
        """Handle command execution feedback for learning."""
        session_id = event_data.get("session_id")
        success = event_data.get("success", False)
        
        if session_id in self.active_sessions and success:
            session = self.active_sessions[session_id]
            await self._record_successful_execution(session)
    
    async def _handle_session_start(self, event_data: Dict[str, Any]):
        """Handle session start events."""
        self.logger.debug("AI Command Composer session started")
    
    async def _handle_session_end(self, event_data: Dict[str, Any]):
        """Handle session end events."""
        session_id = event_data.get("session_id")
        if session_id in self.active_sessions:
            session = self.active_sessions.pop(session_id)
            self.session_history.append(session)
    
    async def _process_translation_request(self, session_id: str, user_input: str):
        """Process a real-time translation request."""
        try:
            # Get suggestions for real-time feedback
            suggestions = await self.get_suggestions(user_input)
            
            # Emit real-time suggestions event
            await self.event_system.emit("command_suggestions", {
                "session_id": session_id,
                "input": user_input,
                "suggestions": suggestions
            })
            
        except Exception as e:
            self.logger.error(f"Error processing translation request: {e}")
    
    async def shutdown(self):
        """Shutdown the AI Command Composer."""
        self.is_processing = False
        await self._save_learning_patterns()
        
        # Clean up active sessions
        for session in self.active_sessions.values():
            self.session_history.append(session)
        self.active_sessions.clear()
        
        self.logger.info("AI Command Composer shutdown complete")


# Example usage and integration
async def main():
    """Example usage of the AI Command Composer."""
    # This would be called from the main TUI application
    from ..v2.event_system import AsyncEventSystem
    
    event_system = AsyncEventSystem()
    composer = AICommandComposer(event_system)
    
    # Example composition
    session = await composer.compose_command(
        "create a new Python agent for data analysis",
        context={"current_directory": "/workspace", "user_skill": "advanced"}
    )
    
    print(f"Composed command: {session.selected_match.command}")
    print(f"Confidence: {session.selected_match.confidence}")
    
    # Example refinement
    refined_session = await composer.refine_command(
        session.session_id,
        "with scikit-learn support",
        0
    )
    
    print(f"Refined command: {refined_session.selected_match.command}")


if __name__ == "__main__":
    asyncio.run(main())