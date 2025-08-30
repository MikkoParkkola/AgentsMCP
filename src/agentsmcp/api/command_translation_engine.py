"""
Command Translation Engine with ML-Powered Suggestions

Advanced command translation system that converts natural language input
into precise CLI commands with contextual suggestions and validation.
"""

import re
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from .base import APIBase, APIResponse, APIError
from .nlp_processor import CommandIntent, IntentPrediction
from .intent_recognition_service import IntentRecognitionService


class CommandComplexity(str, Enum):
    """Command complexity levels for progressive disclosure."""
    BEGINNER = "beginner"      # Simple, basic commands
    INTERMEDIATE = "intermediate"  # Standard commands with options
    ADVANCED = "advanced"      # Complex multi-step commands
    EXPERT = "expert"         # Advanced scripting and automation


@dataclass
class CommandTemplate:
    """Template for command generation."""
    pattern: str
    template: str
    description: str
    complexity: CommandComplexity
    examples: List[str]
    parameters: List[Dict[str, Any]]
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


@dataclass 
class TranslationResult:
    """Result of command translation."""
    original_text: str
    suggested_command: str
    confidence: float
    intent: CommandIntent
    parameters: Dict[str, Any]
    alternatives: List[str]
    explanation: str
    complexity: CommandComplexity
    validation_status: str
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class CommandSuggestion:
    """Contextual command suggestion."""
    command: str
    description: str
    confidence: float
    usage_example: str
    complexity: CommandComplexity
    relevant_context: List[str]


class CommandTranslationEngine(APIBase):
    """Advanced command translation engine with ML-powered suggestions."""
    
    def __init__(self, intent_service: Optional[IntentRecognitionService] = None):
        super().__init__("command_translation_engine")
        self.intent_service = intent_service or IntentRecognitionService()
        self.command_templates: Dict[CommandIntent, List[CommandTemplate]] = {}
        self.parameter_extractors = {}
        self.validation_rules = {}
        self.usage_analytics = {}
        
        # Initialize templates and extractors
        asyncio.create_task(self._initialize_templates())
        asyncio.create_task(self._initialize_extractors())
    
    async def _initialize_templates(self):
        """Initialize command templates for different intents."""
        self.command_templates = {
            CommandIntent.CHAT: [
                CommandTemplate(
                    pattern=r"(?i)(?:chat|talk|discuss)\s+(?:with|about|to)?\s*(.+)?",
                    template="agentsmcp chat {agent} {message}",
                    description="Start a chat conversation with an agent",
                    complexity=CommandComplexity.BEGINNER,
                    examples=[
                        "agentsmcp chat --agent claude",
                        "agentsmcp chat --agent gpt4 --message 'Hello!'",
                        "agentsmcp chat --interactive"
                    ],
                    parameters=[
                        {"name": "agent", "type": "string", "required": False, "description": "Agent to chat with"},
                        {"name": "message", "type": "string", "required": False, "description": "Initial message"},
                        {"name": "interactive", "type": "flag", "description": "Start interactive chat mode"}
                    ],
                    aliases=["talk", "discuss", "converse"]
                ),
            ],
            CommandIntent.PIPELINE: [
                CommandTemplate(
                    pattern=r"(?i)(?:run|execute|start)\s+(?:a\s+)?(?:pipeline|workflow|process)(?:\s+(.+))?",
                    template="agentsmcp pipeline run {name} {config}",
                    description="Execute a data processing pipeline",
                    complexity=CommandComplexity.INTERMEDIATE,
                    examples=[
                        "agentsmcp pipeline run data-analysis",
                        "agentsmcp pipeline run --config myconfig.yaml",
                        "agentsmcp pipeline run etl --dry-run"
                    ],
                    parameters=[
                        {"name": "name", "type": "string", "required": True, "description": "Pipeline name"},
                        {"name": "config", "type": "file", "required": False, "description": "Configuration file"},
                        {"name": "dry-run", "type": "flag", "description": "Test run without execution"}
                    ],
                    aliases=["workflow", "process", "job"]
                ),
                CommandTemplate(
                    pattern=r"(?i)(?:create|build|setup)\s+(?:a\s+)?(?:pipeline|workflow)(?:\s+(.+))?",
                    template="agentsmcp pipeline create {name} {template}",
                    description="Create a new pipeline from template",
                    complexity=CommandComplexity.ADVANCED,
                    examples=[
                        "agentsmcp pipeline create my-pipeline --template data-science",
                        "agentsmcp pipeline create etl-job --template basic"
                    ],
                    parameters=[
                        {"name": "name", "type": "string", "required": True, "description": "New pipeline name"},
                        {"name": "template", "type": "choice", "choices": ["basic", "data-science", "web-scraping"], "description": "Pipeline template"}
                    ]
                ),
            ],
            CommandIntent.DISCOVERY: [
                CommandTemplate(
                    pattern=r"(?i)(?:find|discover|search|list|show)\s+(?:available\s+)?(?:agents?|services?|tools?)(?:\s+(.+))?",
                    template="agentsmcp discovery {action} {filter}",
                    description="Discover available agents and services",
                    complexity=CommandComplexity.BEGINNER,
                    examples=[
                        "agentsmcp discovery list",
                        "agentsmcp discovery search --type agent",
                        "agentsmcp discovery list --capability nlp"
                    ],
                    parameters=[
                        {"name": "action", "type": "choice", "choices": ["list", "search"], "required": True},
                        {"name": "type", "type": "choice", "choices": ["agent", "service", "tool"], "description": "Resource type"},
                        {"name": "capability", "type": "string", "description": "Required capability"}
                    ]
                ),
            ],
            CommandIntent.AGENT_MANAGEMENT: [
                CommandTemplate(
                    pattern=r"(?i)(?:spawn|create|start|launch)\s+(?:an?\s+)?agent(?:\s+(.+))?",
                    template="agentsmcp agents create {name} {type}",
                    description="Create and start a new agent",
                    complexity=CommandComplexity.INTERMEDIATE,
                    examples=[
                        "agentsmcp agents create my-agent --type claude",
                        "agentsmcp agents create data-agent --type gpt4 --capability analysis"
                    ],
                    parameters=[
                        {"name": "name", "type": "string", "required": True, "description": "Agent name"},
                        {"name": "type", "type": "choice", "choices": ["claude", "gpt4", "gemini", "local"], "required": True},
                        {"name": "capability", "type": "string", "description": "Primary capability"}
                    ]
                ),
                CommandTemplate(
                    pattern=r"(?i)(?:stop|terminate|kill|shutdown)\s+(?:the\s+)?agent(?:\s+(.+))?",
                    template="agentsmcp agents stop {name}",
                    description="Stop a running agent",
                    complexity=CommandComplexity.BEGINNER,
                    examples=[
                        "agentsmcp agents stop my-agent",
                        "agentsmcp agents stop --all"
                    ],
                    parameters=[
                        {"name": "name", "type": "string", "description": "Agent name"},
                        {"name": "all", "type": "flag", "description": "Stop all agents"}
                    ]
                ),
            ],
            CommandIntent.SYMPHONY_MODE: [
                CommandTemplate(
                    pattern=r"(?i)(?:enable|activate|start)\s+(?:symphony|orchestration|coordination)\s*mode",
                    template="agentsmcp symphony enable {agents}",
                    description="Enable symphony mode for multi-agent coordination",
                    complexity=CommandComplexity.ADVANCED,
                    examples=[
                        "agentsmcp symphony enable",
                        "agentsmcp symphony enable --agents agent1,agent2,agent3",
                        "agentsmcp symphony enable --auto-scale"
                    ],
                    parameters=[
                        {"name": "agents", "type": "string", "description": "Comma-separated agent names"},
                        {"name": "auto-scale", "type": "flag", "description": "Enable automatic scaling"}
                    ]
                ),
            ],
            CommandIntent.CONFIG: [
                CommandTemplate(
                    pattern=r"(?i)(?:show|display|view)\s+(?:current\s+)?(?:config|settings?)",
                    template="agentsmcp config show {section}",
                    description="Display current configuration",
                    complexity=CommandComplexity.BEGINNER,
                    examples=[
                        "agentsmcp config show",
                        "agentsmcp config show agents",
                        "agentsmcp config show --format json"
                    ],
                    parameters=[
                        {"name": "section", "type": "choice", "choices": ["agents", "discovery", "pipeline"], "description": "Config section"},
                        {"name": "format", "type": "choice", "choices": ["yaml", "json"], "default": "yaml"}
                    ]
                ),
                CommandTemplate(
                    pattern=r"(?i)(?:set|update|change)\s+(?:config|setting)(?:\s+(.+))?",
                    template="agentsmcp config set {key} {value}",
                    description="Update configuration setting",
                    complexity=CommandComplexity.INTERMEDIATE,
                    examples=[
                        "agentsmcp config set agents.default_model claude",
                        "agentsmcp config set discovery.auto_refresh true"
                    ],
                    parameters=[
                        {"name": "key", "type": "string", "required": True, "description": "Configuration key"},
                        {"name": "value", "type": "string", "required": True, "description": "New value"}
                    ]
                ),
            ],
            CommandIntent.HELP: [
                CommandTemplate(
                    pattern=r"(?i)(?:help|assistance|guide|how\s+to)",
                    template="agentsmcp {command} --help",
                    description="Get help for commands",
                    complexity=CommandComplexity.BEGINNER,
                    examples=[
                        "agentsmcp --help",
                        "agentsmcp chat --help",
                        "agentsmcp pipeline --help"
                    ],
                    parameters=[
                        {"name": "command", "type": "string", "description": "Command to get help for"}
                    ]
                ),
            ],
        }
    
    async def _initialize_extractors(self):
        """Initialize parameter extraction functions."""
        self.parameter_extractors = {
            "agent_name": self._extract_agent_name,
            "file_path": self._extract_file_path,
            "pipeline_name": self._extract_pipeline_name,
            "model_name": self._extract_model_name,
            "config_key": self._extract_config_key,
            "capability": self._extract_capability,
            "flags": self._extract_flags,
        }
        
        # Initialize validation rules
        self.validation_rules = {
            "agent_name": {"pattern": r"^[a-zA-Z0-9_-]+$", "max_length": 50},
            "file_path": {"extensions": [".yaml", ".yml", ".json", ".py"], "max_length": 200},
            "pipeline_name": {"pattern": r"^[a-zA-Z0-9_-]+$", "max_length": 50},
            "model_name": {"choices": ["claude", "gpt4", "gemini", "llama", "local"]},
        }
    
    def _extract_agent_name(self, text: str) -> Optional[str]:
        """Extract agent name from text."""
        patterns = [
            r"(?i)(?:agent|model)\s+([a-zA-Z0-9_-]+)",
            r"(?i)(?:with|using)\s+([a-zA-Z0-9_-]+)",
            r"(?i)--agent\s+([a-zA-Z0-9_-]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _extract_file_path(self, text: str) -> Optional[str]:
        """Extract file path from text."""
        patterns = [
            r"([a-zA-Z0-9_/.-]+\.(?:yaml|yml|json|py|md))",
            r"(?i)--config\s+([a-zA-Z0-9_/.-]+)",
            r"(?i)--file\s+([a-zA-Z0-9_/.-]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _extract_pipeline_name(self, text: str) -> Optional[str]:
        """Extract pipeline name from text."""
        patterns = [
            r"(?i)(?:pipeline|workflow)\s+([a-zA-Z0-9_-]+)",
            r"(?i)(?:run|execute)\s+([a-zA-Z0-9_-]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _extract_model_name(self, text: str) -> Optional[str]:
        """Extract model name from text."""
        models = ["claude", "gpt4", "gpt-4", "gemini", "llama", "mixtral"]
        text_lower = text.lower()
        
        for model in models:
            if model in text_lower:
                return model.replace("-", "")  # Normalize
        return None
    
    def _extract_config_key(self, text: str) -> Optional[str]:
        """Extract configuration key from text."""
        patterns = [
            r"(?i)(?:set|update)\s+([a-zA-Z0-9_.]+)",
            r"([a-zA-Z0-9_.]+)\s*=",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _extract_capability(self, text: str) -> Optional[str]:
        """Extract capability from text."""
        capabilities = ["nlp", "analysis", "coding", "data", "web", "vision", "audio"]
        text_lower = text.lower()
        
        for capability in capabilities:
            if capability in text_lower:
                return capability
        return None
    
    def _extract_flags(self, text: str) -> List[str]:
        """Extract command flags from text."""
        flag_patterns = [
            r"--([a-zA-Z0-9-]+)(?:\s|$)",
            r"-([a-zA-Z])(?:\s|$)",
        ]
        
        flags = []
        for pattern in flag_patterns:
            flags.extend(re.findall(pattern, text))
        
        # Convert common natural language to flags
        flag_mappings = {
            "interactive": "interactive",
            "dry run": "dry-run", 
            "all": "all",
            "json": "format json",
            "yaml": "format yaml",
            "help": "help",
            "verbose": "verbose",
            "quiet": "quiet",
        }
        
        text_lower = text.lower()
        for phrase, flag in flag_mappings.items():
            if phrase in text_lower:
                flags.append(flag)
        
        return list(set(flags))  # Remove duplicates
    
    async def translate_command(
        self, 
        text: str,
        user_skill_level: str = "intermediate",
        context: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """
        Translate natural language to CLI command with validation.
        
        Provides ML-powered suggestions with contextual understanding
        and progressive disclosure based on user skill level.
        """
        return await self._execute_with_metrics(
            "translate_command",
            self._translate_command_internal,
            text,
            user_skill_level,
            context or {}
        )
    
    async def _translate_command_internal(
        self,
        text: str,
        user_skill_level: str,
        context: Dict[str, Any]
    ) -> TranslationResult:
        """Internal command translation logic."""
        if not text or not text.strip():
            raise APIError("Empty command text", "INVALID_INPUT", 400)
        
        # Step 1: Get intent classification
        intent_response = await self.intent_service.classify_intent(text)
        if intent_response.status != "success":
            raise APIError("Intent classification failed", "INTENT_ERROR", 500)
        
        intent_pred: IntentPrediction = intent_response.data
        
        # Step 2: Find matching templates
        templates = self.command_templates.get(intent_pred.intent, [])
        if not templates:
            return TranslationResult(
                original_text=text,
                suggested_command="agentsmcp --help",
                confidence=0.1,
                intent=intent_pred.intent,
                parameters={},
                alternatives=["agentsmcp --help"],
                explanation="No specific command template found. Use --help for guidance.",
                complexity=CommandComplexity.BEGINNER,
                validation_status="unknown_intent",
                warnings=["Intent not recognized - falling back to help"]
            )
        
        # Step 3: Match against templates and extract parameters
        best_match = None
        best_score = 0.0
        
        for template in templates:
            score = self._score_template_match(text, template, intent_pred.confidence)
            if score > best_score:
                best_score = score
                best_match = template
        
        if not best_match or best_score < 0.3:
            # Low confidence match - provide generic command
            return self._create_generic_result(text, intent_pred, templates)
        
        # Step 4: Extract parameters
        parameters = self._extract_all_parameters(text, best_match)
        
        # Step 5: Generate command
        suggested_command = self._generate_command_from_template(best_match, parameters)
        
        # Step 6: Validate command
        validation_status, warnings = self._validate_generated_command(suggested_command, parameters)
        
        # Step 7: Generate alternatives
        alternatives = self._generate_alternatives(templates, parameters, user_skill_level)
        
        # Step 8: Apply progressive disclosure
        complexity = self._determine_complexity(best_match, user_skill_level)
        explanation = self._generate_explanation(best_match, parameters, complexity)
        
        return TranslationResult(
            original_text=text,
            suggested_command=suggested_command,
            confidence=min(intent_pred.confidence * best_score, 0.95),
            intent=intent_pred.intent,
            parameters=parameters,
            alternatives=alternatives,
            explanation=explanation,
            complexity=complexity,
            validation_status=validation_status,
            warnings=warnings
        )
    
    def _score_template_match(
        self, 
        text: str, 
        template: CommandTemplate,
        intent_confidence: float
    ) -> float:
        """Score how well a template matches the input text."""
        # Base score from regex pattern
        pattern_match = re.search(template.pattern, text)
        if not pattern_match:
            return 0.0
        
        base_score = 0.6  # Base score for pattern match
        
        # Boost score based on match coverage
        match_length = len(pattern_match.group(0))
        text_length = len(text.strip())
        coverage = match_length / text_length
        base_score += coverage * 0.2
        
        # Boost score based on intent confidence
        base_score += intent_confidence * 0.2
        
        # Check for alias matches
        text_lower = text.lower()
        for alias in template.aliases:
            if alias in text_lower:
                base_score += 0.1
                break
        
        return min(base_score, 1.0)
    
    def _extract_all_parameters(
        self, 
        text: str, 
        template: CommandTemplate
    ) -> Dict[str, Any]:
        """Extract all parameters for a command template."""
        parameters = {}
        
        # Extract using registered extractors
        for extractor_name, extractor_func in self.parameter_extractors.items():
            try:
                value = extractor_func(text)
                if value:
                    parameters[extractor_name] = value
            except Exception as e:
                self.logger.warning(f"Parameter extraction failed for {extractor_name}: {e}")
        
        # Extract template-specific parameters
        for param_def in template.parameters:
            param_name = param_def["name"]
            if param_name not in parameters:
                # Try to extract based on parameter definition
                extracted_value = self._extract_parameter_by_definition(text, param_def)
                if extracted_value:
                    parameters[param_name] = extracted_value
        
        return parameters
    
    def _extract_parameter_by_definition(
        self, 
        text: str, 
        param_def: Dict[str, Any]
    ) -> Optional[Any]:
        """Extract parameter value based on parameter definition."""
        param_name = param_def["name"]
        param_type = param_def.get("type", "string")
        
        if param_type == "flag":
            # Check if flag is mentioned in text
            flag_keywords = [param_name, param_name.replace("-", " ")]
            text_lower = text.lower()
            return any(keyword in text_lower for keyword in flag_keywords)
        
        elif param_type == "choice":
            choices = param_def.get("choices", [])
            text_lower = text.lower()
            for choice in choices:
                if choice in text_lower:
                    return choice
        
        elif param_type == "file":
            return self._extract_file_path(text)
        
        return None
    
    def _generate_command_from_template(
        self, 
        template: CommandTemplate,
        parameters: Dict[str, Any]
    ) -> str:
        """Generate CLI command from template and parameters."""
        command = template.template
        
        # Replace template placeholders
        for param_def in template.parameters:
            param_name = param_def["name"]
            placeholder = f"{{{param_name}}}"
            
            if placeholder in command:
                value = parameters.get(param_name)
                if value:
                    if param_def.get("type") == "flag":
                        # For flags, replace with --flag-name or remove if False
                        if value:
                            command = command.replace(placeholder, f"--{param_name}")
                        else:
                            command = command.replace(placeholder, "")
                    else:
                        command = command.replace(placeholder, str(value))
                else:
                    # Use default value or remove placeholder
                    default = param_def.get("default")
                    if default:
                        command = command.replace(placeholder, str(default))
                    elif param_def.get("required"):
                        command = command.replace(placeholder, f"<{param_name}>")
                    else:
                        command = command.replace(placeholder, "")
        
        # Clean up extra spaces
        command = re.sub(r'\s+', ' ', command).strip()
        
        return command
    
    def _validate_generated_command(
        self, 
        command: str,
        parameters: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """Validate the generated command."""
        warnings = []
        
        # Check for placeholder values still present
        if "<" in command and ">" in command:
            warnings.append("Command contains placeholder values that need to be filled")
            return "incomplete", warnings
        
        # Validate individual parameters
        for param_name, param_value in parameters.items():
            rules = self.validation_rules.get(param_name, {})
            
            # Pattern validation
            if "pattern" in rules:
                if not re.match(rules["pattern"], str(param_value)):
                    warnings.append(f"Invalid format for {param_name}")
            
            # Length validation
            if "max_length" in rules:
                if len(str(param_value)) > rules["max_length"]:
                    warnings.append(f"{param_name} exceeds maximum length")
            
            # Choice validation
            if "choices" in rules:
                if param_value not in rules["choices"]:
                    warnings.append(f"Invalid choice for {param_name}: {param_value}")
        
        # Determine overall status
        if warnings:
            return "warning", warnings
        else:
            return "valid", warnings
    
    def _generate_alternatives(
        self,
        templates: List[CommandTemplate],
        parameters: Dict[str, Any],
        user_skill_level: str
    ) -> List[str]:
        """Generate alternative command suggestions."""
        alternatives = []
        
        # Generate variations of the same command
        for template in templates[:3]:  # Top 3 templates
            try:
                alt_command = self._generate_command_from_template(template, parameters)
                if alt_command not in alternatives:
                    alternatives.append(alt_command)
            except Exception:
                continue
        
        # Add skill-level appropriate alternatives
        if user_skill_level == "beginner":
            alternatives.append("agentsmcp --help")
        elif user_skill_level == "expert":
            # Add advanced options
            for alt in alternatives[:]:
                if "--verbose" not in alt:
                    alternatives.append(f"{alt} --verbose")
        
        return alternatives[:5]  # Limit to 5 alternatives
    
    def _determine_complexity(
        self, 
        template: CommandTemplate,
        user_skill_level: str
    ) -> CommandComplexity:
        """Determine appropriate complexity level for progressive disclosure."""
        base_complexity = template.complexity
        
        # Adjust based on user skill level
        complexity_order = [
            CommandComplexity.BEGINNER,
            CommandComplexity.INTERMEDIATE,
            CommandComplexity.ADVANCED,
            CommandComplexity.EXPERT
        ]
        
        current_index = complexity_order.index(base_complexity)
        
        if user_skill_level == "beginner" and current_index > 1:
            return CommandComplexity.INTERMEDIATE
        elif user_skill_level == "expert":
            return CommandComplexity.EXPERT
        
        return base_complexity
    
    def _generate_explanation(
        self,
        template: CommandTemplate,
        parameters: Dict[str, Any],
        complexity: CommandComplexity
    ) -> str:
        """Generate contextual explanation for the command."""
        base_explanation = template.description
        
        if complexity == CommandComplexity.BEGINNER:
            return f"{base_explanation}. This is a basic command that's safe to run."
        elif complexity == CommandComplexity.EXPERT:
            param_details = ", ".join([f"{k}: {v}" for k, v in parameters.items() if v])
            return f"{base_explanation}. Parameters: {param_details}. Advanced usage with full control."
        else:
            return f"{base_explanation}. Use --help for more options."
    
    def _create_generic_result(
        self,
        text: str,
        intent_pred: IntentPrediction,
        templates: List[CommandTemplate]
    ) -> TranslationResult:
        """Create a generic result when no good template match is found."""
        generic_commands = {
            CommandIntent.CHAT: "agentsmcp chat",
            CommandIntent.PIPELINE: "agentsmcp pipeline list",
            CommandIntent.DISCOVERY: "agentsmcp discovery list",
            CommandIntent.AGENT_MANAGEMENT: "agentsmcp agents list",
            CommandIntent.SYMPHONY_MODE: "agentsmcp symphony status",
            CommandIntent.CONFIG: "agentsmcp config show",
            CommandIntent.HELP: "agentsmcp --help",
        }
        
        generic_command = generic_commands.get(intent_pred.intent, "agentsmcp --help")
        
        return TranslationResult(
            original_text=text,
            suggested_command=generic_command,
            confidence=intent_pred.confidence * 0.5,  # Lower confidence for generic
            intent=intent_pred.intent,
            parameters={},
            alternatives=[f"{generic_command} --help"],
            explanation=f"Generic {intent_pred.intent.value} command. Use --help for specific options.",
            complexity=CommandComplexity.BEGINNER,
            validation_status="generic",
            warnings=["Using generic command - consider being more specific"]
        )
    
    async def get_contextual_suggestions(
        self,
        current_input: str,
        user_context: Dict[str, Any] = None
    ) -> APIResponse:
        """Get contextual command suggestions based on current input."""
        return await self._execute_with_metrics(
            "get_contextual_suggestions",
            self._get_contextual_suggestions_internal,
            current_input,
            user_context or {}
        )
    
    async def _get_contextual_suggestions_internal(
        self,
        current_input: str,
        user_context: Dict[str, Any]
    ) -> List[CommandSuggestion]:
        """Generate contextual suggestions for partial input."""
        suggestions = []
        
        if len(current_input) < 2:
            # Provide common starter suggestions
            common_starters = [
                CommandSuggestion(
                    command="agentsmcp chat",
                    description="Start a conversation with an agent",
                    confidence=0.9,
                    usage_example="agentsmcp chat --agent claude",
                    complexity=CommandComplexity.BEGINNER,
                    relevant_context=["new_user", "getting_started"]
                ),
                CommandSuggestion(
                    command="agentsmcp discovery list",
                    description="See what's available",
                    confidence=0.8,
                    usage_example="agentsmcp discovery list --type agent",
                    complexity=CommandComplexity.BEGINNER,
                    relevant_context=["exploration", "discovery"]
                ),
            ]
            return common_starters
        
        # For longer inputs, try to classify intent and suggest completions
        try:
            intent_response = await self.intent_service.classify_intent(current_input)
            if intent_response.status == "success":
                intent_pred: IntentPrediction = intent_response.data
                templates = self.command_templates.get(intent_pred.intent, [])
                
                for template in templates:
                    if self._matches_partial_input(current_input, template):
                        suggestion = CommandSuggestion(
                            command=template.template.split()[0:2],  # First 2 parts
                            description=template.description,
                            confidence=intent_pred.confidence,
                            usage_example=template.examples[0] if template.examples else "",
                            complexity=template.complexity,
                            relevant_context=[intent_pred.intent.value]
                        )
                        suggestions.append(suggestion)
        except Exception as e:
            self.logger.warning(f"Failed to generate contextual suggestions: {e}")
        
        return suggestions[:5]  # Top 5 suggestions
    
    def _matches_partial_input(self, partial_input: str, template: CommandTemplate) -> bool:
        """Check if template could match the partial input."""
        partial_lower = partial_input.lower()
        
        # Check against template pattern keywords
        for alias in template.aliases + [template.template.split()[1]]:
            if alias.startswith(partial_lower) or partial_lower in alias:
                return True
        
        return False
    
    async def learn_from_feedback(
        self,
        original_text: str,
        suggested_command: str,
        user_feedback: str,
        actual_command: str = None
    ) -> APIResponse:
        """Learn from user feedback to improve suggestions."""
        return await self._execute_with_metrics(
            "learn_from_feedback",
            self._learn_from_feedback_internal,
            original_text,
            suggested_command,
            user_feedback,
            actual_command
        )
    
    async def _learn_from_feedback_internal(
        self,
        original_text: str,
        suggested_command: str,
        user_feedback: str,
        actual_command: Optional[str]
    ) -> Dict[str, Any]:
        """Internal logic for learning from user feedback."""
        feedback_entry = {
            "timestamp": datetime.utcnow(),
            "original_text": original_text,
            "suggested_command": suggested_command,
            "user_feedback": user_feedback,
            "actual_command": actual_command,
            "feedback_type": self._classify_feedback(user_feedback)
        }
        
        # Store feedback for model improvement
        feedback_key = f"{original_text}:{suggested_command}"
        if feedback_key not in self.usage_analytics:
            self.usage_analytics[feedback_key] = []
        
        self.usage_analytics[feedback_key].append(feedback_entry)
        
        # If we have enough negative feedback, trigger template adjustment
        negative_feedback_count = sum(
            1 for entry in self.usage_analytics[feedback_key]
            if entry["feedback_type"] in ["negative", "correction"]
        )
        
        if negative_feedback_count >= 3:
            await self._adjust_templates_based_on_feedback(feedback_key)
        
        return {
            "feedback_recorded": True,
            "total_feedback_entries": len(self.usage_analytics.get(feedback_key, [])),
            "improvement_triggered": negative_feedback_count >= 3
        }
    
    def _classify_feedback(self, feedback: str) -> str:
        """Classify user feedback type."""
        feedback_lower = feedback.lower()
        
        positive_indicators = ["good", "correct", "right", "perfect", "yes", "thanks"]
        negative_indicators = ["wrong", "bad", "no", "incorrect", "not what", "error"]
        
        if any(indicator in feedback_lower for indicator in positive_indicators):
            return "positive"
        elif any(indicator in feedback_lower for indicator in negative_indicators):
            return "negative"
        elif "should be" in feedback_lower or "meant" in feedback_lower:
            return "correction"
        else:
            return "neutral"
    
    async def _adjust_templates_based_on_feedback(self, feedback_key: str):
        """Adjust command templates based on accumulated feedback."""
        # This would implement machine learning-based template adjustment
        # For now, we'll log the need for manual template review
        self.logger.info(f"Template adjustment needed for: {feedback_key}")
        
        # In a full implementation, this would:
        # 1. Analyze patterns in negative feedback
        # 2. Adjust template patterns and weights
        # 3. Update parameter extraction rules
        # 4. Re-train classification models
    
    async def get_translation_analytics(self) -> APIResponse:
        """Get analytics about command translation performance."""
        return await self._execute_with_metrics(
            "get_translation_analytics",
            self._get_translation_analytics_internal
        )
    
    async def _get_translation_analytics_internal(self) -> Dict[str, Any]:
        """Internal logic for getting translation analytics."""
        total_feedback = len(self.usage_analytics)
        
        if total_feedback == 0:
            return {
                "total_translations": 0,
                "feedback_entries": 0,
                "accuracy_estimate": 0.0,
                "most_common_intents": [],
                "improvement_areas": []
            }
        
        # Calculate accuracy based on positive feedback
        positive_feedback = 0
        total_feedback_entries = 0
        
        for entries in self.usage_analytics.values():
            total_feedback_entries += len(entries)
            positive_feedback += sum(
                1 for entry in entries
                if entry["feedback_type"] == "positive"
            )
        
        accuracy_estimate = (positive_feedback / total_feedback_entries) if total_feedback_entries > 0 else 0.0
        
        return {
            "total_translations": total_feedback,
            "feedback_entries": total_feedback_entries,
            "accuracy_estimate": accuracy_estimate,
            "templates_available": sum(len(templates) for templates in self.command_templates.values()),
            "intents_supported": len(self.command_templates)
        }