"""Rule-based pattern matching for natural language commands.

This module provides fast fallback parsing when LLM is unavailable,
using regex patterns and keyword matching for common commands.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Pattern, Any
from dataclasses import dataclass

from ..models.nlp_models import PatternMatch, ParsedCommand, ParsingMethod


logger = logging.getLogger(__name__)


@dataclass
class CommandPattern:
    """Definition of a command pattern with matching rules."""
    
    pattern_id: str
    action: str
    regex_patterns: List[str]
    keywords: List[str]
    parameter_extractors: Dict[str, str]  # parameter_name -> regex pattern
    examples: List[str]
    priority: int = 5
    confidence_base: float = 0.8


class PatternMatcher:
    """Rule-based pattern matcher for common CLI commands."""
    
    def __init__(self):
        self.patterns: List[CommandPattern] = []
        self.compiled_patterns: Dict[str, List[Pattern]] = {}
        self._initialize_patterns()
        self._compile_patterns()
        
        logger.info(f"Initialized PatternMatcher with {len(self.patterns)} patterns")
    
    def _initialize_patterns(self) -> None:
        """Initialize predefined command patterns."""
        
        # Analysis and exploration patterns
        self.patterns.extend([
            CommandPattern(
                pattern_id="analyze_code",
                action="analyze",
                regex_patterns=[
                    r"analyze\s+(?:my\s+)?code",
                    r"check\s+(?:my\s+)?code(?:\s+for\s+issues)?",
                    r"review\s+(?:my\s+)?code",
                    r"audit\s+(?:my\s+)?code",
                    r"scan\s+(?:my\s+)?code"
                ],
                keywords=["analyze", "check", "review", "audit", "scan", "code"],
                parameter_extractors={
                    "target": r"(?:analyze|check|review|audit|scan)\s+(?:my\s+)?(?:code\s+in\s+)?(.+?)(?:\s|$)",
                    "type": r"(?:for\s+)?(issues|bugs|security|performance|style)"
                },
                examples=[
                    "analyze my code",
                    "check code for issues", 
                    "review the code in src/",
                    "audit code for security issues"
                ],
                priority=2,
                confidence_base=0.85
            ),
            
            CommandPattern(
                pattern_id="analyze_project",
                action="analyze",
                regex_patterns=[
                    r"analyze\s+(?:the\s+|this\s+)?project",
                    r"check\s+(?:the\s+|this\s+)?project",
                    r"examine\s+(?:the\s+|this\s+)?project",
                    r"investigate\s+(?:the\s+|this\s+)?(?:project|repository|repo)",
                    r"scan\s+(?:the\s+|this\s+)?(?:project|repository|repo)"
                ],
                keywords=["analyze", "check", "examine", "investigate", "scan", "project", "repository", "repo"],
                parameter_extractors={
                    "target": r"(?:analyze|check|examine|investigate|scan)\s+(?:the\s+|this\s+)?(?:project|repository|repo)\s*(?:in\s+)?(.+?)(?:\s|$)",
                    "deep": r"(deep|thorough|comprehensive|detailed)"
                },
                examples=[
                    "analyze the project",
                    "check this project for issues",
                    "examine the repository",
                    "investigate project structure"
                ],
                priority=2,
                confidence_base=0.90
            ),
            
            CommandPattern(
                pattern_id="help_command",
                action="help",
                regex_patterns=[
                    r"help",
                    r"show\s+help",
                    r"what\s+can\s+(?:you|I)\s+do",
                    r"available\s+commands",
                    r"list\s+commands",
                    r"how\s+(?:do\s+I|to)"
                ],
                keywords=["help", "commands", "usage", "how"],
                parameter_extractors={
                    "topic": r"help\s+(?:with\s+)?(.+?)(?:\s|$)",
                    "command": r"help\s+(?:with\s+)?(?:the\s+)?(\w+)(?:\s+command)?"
                },
                examples=[
                    "help",
                    "show help",
                    "what can you do",
                    "help with analyze command"
                ],
                priority=1,
                confidence_base=0.95
            ),
            
            CommandPattern(
                pattern_id="status_check",
                action="status",
                regex_patterns=[
                    r"status",
                    r"check\s+status",
                    r"show\s+status",
                    r"what.?s\s+(?:the\s+)?status",
                    r"how\s+(?:are\s+)?(?:things|we)\s+(?:doing|going)",
                    r"system\s+(?:status|health)"
                ],
                keywords=["status", "health", "check", "system"],
                parameter_extractors={
                    "component": r"(?:status|check)\s+(?:of\s+)?(.+?)(?:\s|$)"
                },
                examples=[
                    "status",
                    "check status", 
                    "show system status",
                    "what's the status"
                ],
                priority=1,
                confidence_base=0.90
            ),
            
            CommandPattern(
                pattern_id="start_tui",
                action="tui",
                regex_patterns=[
                    r"start\s+(?:the\s+)?tui",
                    r"open\s+(?:the\s+)?tui",
                    r"launch\s+(?:the\s+)?tui",
                    r"run\s+(?:the\s+)?tui",
                    r"tui\s+mode",
                    r"interactive\s+mode",
                    r"start\s+interactive"
                ],
                keywords=["tui", "interactive", "start", "launch", "open", "run"],
                parameter_extractors={},
                examples=[
                    "start the tui",
                    "open tui",
                    "launch interactive mode",
                    "run tui"
                ],
                priority=1,
                confidence_base=0.95
            ),
            
            CommandPattern(
                pattern_id="setup_project", 
                action="init",
                regex_patterns=[
                    r"(?:help\s+(?:me\s+)?)?set\s?up\s+(?:the\s+)?project",
                    r"initialize\s+(?:the\s+)?project",
                    r"init(?:ialize)?\s+(?:the\s+)?project",
                    r"create\s+(?:new\s+)?project",
                    r"bootstrap\s+(?:the\s+)?project",
                    r"get\s+started"
                ],
                keywords=["setup", "initialize", "init", "create", "bootstrap", "started"],
                parameter_extractors={
                    "template": r"(?:setup|init|create)\s+(?:a\s+|an\s+)?(\w+)\s+project",
                    "interactive": r"(interactive|guided|wizard)"
                },
                examples=[
                    "help me set up the project",
                    "initialize project",
                    "create new project", 
                    "bootstrap the project"
                ],
                priority=2,
                confidence_base=0.85
            ),
            
            CommandPattern(
                pattern_id="optimize_costs",
                action="optimize",
                regex_patterns=[
                    r"optimize\s+(?:my\s+)?costs?",
                    r"reduce\s+(?:my\s+)?costs?",
                    r"lower\s+(?:my\s+)?costs?",
                    r"save\s+(?:on\s+)?costs?",
                    r"cost\s+optimization",
                    r"minimize\s+(?:my\s+)?costs?"
                ],
                keywords=["optimize", "reduce", "lower", "save", "minimize", "costs", "cost"],
                parameter_extractors={
                    "target": r"optimize\s+(?:my\s+)?(.+?)(?:\s|$)"
                },
                examples=[
                    "optimize my costs",
                    "reduce costs", 
                    "cost optimization",
                    "save on costs"
                ],
                priority=3,
                confidence_base=0.85
            ),
            
            CommandPattern(
                pattern_id="run_command",
                action="run", 
                regex_patterns=[
                    r"run\s+(.+)",
                    r"execute\s+(.+)",
                    r"start\s+(.+)",
                    r"launch\s+(.+)"
                ],
                keywords=["run", "execute", "start", "launch"],
                parameter_extractors={
                    "command": r"(?:run|execute|start|launch)\s+(.+?)(?:\s|$)",
                    "args": r"(?:run|execute|start|launch)\s+\w+\s+(.+?)(?:\s|$)"
                },
                examples=[
                    "run analyze --target .",
                    "execute tests",
                    "start monitoring",
                    "launch dashboard"
                ],
                priority=4,
                confidence_base=0.75
            ),
            
            CommandPattern(
                pattern_id="file_operations",
                action="file",
                regex_patterns=[
                    r"(?:read|show|display|cat)\s+(?:file\s+)?(.+)",
                    r"(?:list|ls)\s+(?:files?\s+(?:in\s+)?)?(.+)",
                    r"(?:edit|modify|change)\s+(?:file\s+)?(.+)",
                    r"(?:create|make|touch)\s+(?:file\s+)?(.+)"
                ],
                keywords=["read", "show", "list", "edit", "create", "file", "files"],
                parameter_extractors={
                    "operation": r"(read|show|display|cat|list|ls|edit|modify|change|create|make|touch)",
                    "path": r"(?:read|show|display|cat|list|ls|edit|modify|change|create|make|touch)\s+(?:file\s+)?(.+?)(?:\s|$)"
                },
                examples=[
                    "read file config.json",
                    "list files in src/",
                    "edit main.py",
                    "create new file"
                ],
                priority=3,
                confidence_base=0.80
            ),
            
            # Configuration and settings patterns
            CommandPattern(
                pattern_id="settings_config",
                action="settings",
                regex_patterns=[
                    r"settings?",
                    r"config(?:uration)?",
                    r"preferences?",
                    r"options?",
                    r"configure",
                    r"set\s+up\s+settings",
                    r"change\s+settings"
                ],
                keywords=["settings", "config", "configuration", "preferences", "options"],
                parameter_extractors={
                    "section": r"(?:settings|config|preferences)\s+(?:for\s+)?(.+?)(?:\s|$)"
                },
                examples=[
                    "settings",
                    "configuration", 
                    "change settings",
                    "config for models"
                ],
                priority=1,
                confidence_base=0.90
            ),
            
            # Monitoring and dashboard patterns
            CommandPattern(
                pattern_id="dashboard_monitor",
                action="dashboard",
                regex_patterns=[
                    r"dashboard",
                    r"monitor(?:ing)?",
                    r"watch",
                    r"observe",
                    r"track",
                    r"show\s+(?:me\s+)?(?:the\s+)?dashboard",
                    r"open\s+dashboard"
                ],
                keywords=["dashboard", "monitor", "monitoring", "watch", "observe", "track"],
                parameter_extractors={},
                examples=[
                    "dashboard",
                    "monitoring",
                    "show dashboard",
                    "open dashboard"
                ],
                priority=2,
                confidence_base=0.85
            )
        ])
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        for pattern in self.patterns:
            compiled = []
            for regex_pattern in pattern.regex_patterns:
                try:
                    compiled.append(re.compile(regex_pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Failed to compile pattern '{regex_pattern}': {e}")
                    continue
            
            self.compiled_patterns[pattern.pattern_id] = compiled
    
    def match_patterns(self, natural_input: str) -> List[PatternMatch]:
        """Match input against all patterns and return scored matches."""
        input_normalized = natural_input.strip().lower()
        matches: List[PatternMatch] = []
        
        for pattern in self.patterns:
            match_score = self._calculate_pattern_match_score(pattern, input_normalized)
            if match_score > 0.0:
                parameters = self._extract_parameters(pattern, natural_input)
                
                match = PatternMatch(
                    pattern=pattern.pattern_id,
                    action=pattern.action,
                    parameters=parameters,
                    confidence=min(match_score * pattern.confidence_base, 1.0),
                    priority=pattern.priority
                )
                matches.append(match)
        
        # Sort by priority (lower number = higher priority) then by confidence
        matches.sort(key=lambda m: (m.priority, -m.confidence))
        
        logger.debug(f"Found {len(matches)} pattern matches for: '{natural_input}'")
        return matches
    
    def _calculate_pattern_match_score(self, pattern: CommandPattern, input_normalized: str) -> float:
        """Calculate match score for a pattern against input."""
        scores = []
        
        # Check regex patterns
        regex_score = self._check_regex_patterns(pattern.pattern_id, input_normalized)
        if regex_score > 0.0:
            scores.append(regex_score)
        
        # Check keyword matches
        keyword_score = self._check_keyword_matches(pattern.keywords, input_normalized)
        if keyword_score > 0.0:
            scores.append(keyword_score)
        
        if not scores:
            return 0.0
        
        # Return highest score (best match method)
        return max(scores)
    
    def _check_regex_patterns(self, pattern_id: str, input_normalized: str) -> float:
        """Check if any regex patterns match the input."""
        compiled_patterns = self.compiled_patterns.get(pattern_id, [])
        
        for compiled_pattern in compiled_patterns:
            match = compiled_pattern.search(input_normalized)
            if match:
                # Score based on match coverage
                match_coverage = len(match.group(0)) / len(input_normalized)
                return min(match_coverage * 1.2, 1.0)  # Boost regex matches slightly
        
        return 0.0
    
    def _check_keyword_matches(self, keywords: List[str], input_normalized: str) -> float:
        """Check keyword matches and calculate score."""
        input_words = set(input_normalized.split())
        keyword_set = set(keyword.lower() for keyword in keywords)
        
        matches = input_words.intersection(keyword_set)
        if not matches:
            return 0.0
        
        # Score based on keyword coverage
        keyword_coverage = len(matches) / len(keyword_set)
        word_coverage = len(matches) / len(input_words) if input_words else 0.0
        
        # Balanced score considering both keyword and word coverage
        return (keyword_coverage + word_coverage) / 2.0
    
    def _extract_parameters(self, pattern: CommandPattern, natural_input: str) -> Dict[str, Any]:
        """Extract parameters from input using pattern extractors."""
        parameters: Dict[str, Any] = {}
        
        for param_name, extractor_pattern in pattern.parameter_extractors.items():
            try:
                regex = re.compile(extractor_pattern, re.IGNORECASE)
                match = regex.search(natural_input)
                if match:
                    # Use first capture group if available, otherwise full match
                    value = match.group(1) if match.groups() else match.group(0)
                    parameters[param_name] = value.strip()
            except re.error as e:
                logger.warning(f"Failed to extract parameter '{param_name}' with pattern '{extractor_pattern}': {e}")
                continue
        
        return parameters
    
    def get_best_match(self, natural_input: str) -> Optional[PatternMatch]:
        """Get the best pattern match for input."""
        matches = self.match_patterns(natural_input)
        if not matches:
            return None
        
        return matches[0]  # Already sorted by priority and confidence
    
    def get_pattern_examples(self, action: str) -> List[str]:
        """Get example inputs for a given action."""
        examples = []
        for pattern in self.patterns:
            if pattern.action == action:
                examples.extend(pattern.examples)
        return examples
    
    def parse_command_fallback(self, natural_input: str, confidence_threshold: float = 0.3) -> Optional[ParsedCommand]:
        """Parse command using rule-based patterns as fallback."""
        best_match = self.get_best_match(natural_input)
        
        if not best_match or best_match.confidence < confidence_threshold:
            return None
        
        return ParsedCommand(
            action=best_match.action,
            parameters=best_match.parameters,
            confidence=best_match.confidence,
            method=ParsingMethod.RULE_BASED
        )
    
    def add_custom_pattern(self, pattern: CommandPattern) -> None:
        """Add a custom pattern for domain-specific commands."""
        self.patterns.append(pattern)
        
        # Compile new pattern
        compiled = []
        for regex_pattern in pattern.regex_patterns:
            try:
                compiled.append(re.compile(regex_pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Failed to compile custom pattern '{regex_pattern}': {e}")
                continue
        
        self.compiled_patterns[pattern.pattern_id] = compiled
        logger.info(f"Added custom pattern: {pattern.pattern_id}")
    
    def get_supported_actions(self) -> List[str]:
        """Get list of all supported actions."""
        return list(set(pattern.action for pattern in self.patterns))
    
    def get_pattern_info(self, pattern_id: str) -> Optional[CommandPattern]:
        """Get detailed information about a pattern."""
        for pattern in self.patterns:
            if pattern.pattern_id == pattern_id:
                return pattern
        return None