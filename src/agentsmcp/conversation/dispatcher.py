"""
Natural Language to Command Dispatcher for AgentsMCP.

Converts natural language requests into structured commands that can be
executed by the agent orchestration system.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CommandIntent:
    """Represents a parsed command intent from natural language."""
    command: str
    parameters: Dict[str, Any]
    confidence: float
    raw_text: str
    agent_type: Optional[str] = None


class NaturalLanguageDispatcher:
    """Converts natural language to structured commands using pattern matching and LLM analysis."""
    
    def __init__(self):
        self.command_patterns = self._build_command_patterns()
        self.task_patterns = self._build_task_patterns()
        
    def _build_command_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for direct command recognition."""
        return {
            "status": [
                r"show.*status", r"check.*status", r"system.*status",
                r"what.*running", r"agents.*running", r"current.*state",
                r"is.*working", r"health.*check"
            ],
            "settings": [
                r"open.*settings", r"configure", r"change.*settings",
                r"modify.*config", r"preferences", r"settings",
                r"setup.*provider", r"change.*model"
            ],
            "dashboard": [
                r"start.*dashboard", r"open.*dashboard", r"monitor",
                r"dashboard", r"show.*dashboard", r"monitoring",
                r"real.*time", r"watch.*agents"
            ],
            "web": [
                r"web.*api", r"api.*endpoints", r"web.*interface",
                r"what.*endpoints", r"api.*info", r"web.*info",
                r"rest.*api", r"http.*api"
            ],
            "help": [
                r"help", r"how.*use", r"commands.*available",
                r"what.*can.*do", r"usage", r"guide", r"instructions"
            ],
            "theme": [
                r"change.*theme", r"theme.*to", r"switch.*theme",
                r"set.*theme", r"theme.*dark", r"theme.*light",
                r"dark.*mode", r"light.*mode"
            ]
        }
    
    def _build_task_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for complex task recognition requiring agent orchestration."""
        return {
            "test_and_fix": [
                r"test.*fix.*issues?", r"run.*tests?.*fix", r"check.*fix.*problems?",
                r"test.*project.*fix", r"fix.*all.*issues?", r"debug.*fix",
                r"run.*tests?.*repair", r"validate.*fix"
            ],
            "code_review": [
                r"review.*code", r"code.*review", r"check.*code.*quality",
                r"analyze.*code", r"audit.*code", r"inspect.*code"
            ],
            "build_deploy": [
                r"build.*deploy", r"deploy.*application", r"create.*build",
                r"compile.*deploy", r"package.*deploy", r"release.*build"
            ],
            "refactor": [
                r"refactor.*code", r"clean.*up.*code", r"improve.*code.*structure",
                r"restructure.*code", r"optimize.*code"
            ],
            "documentation": [
                r"generate.*docs", r"create.*documentation", r"document.*code",
                r"write.*readme", r"api.*documentation"
            ],
            "security_scan": [
                r"security.*scan", r"check.*vulnerabilities", r"security.*audit",
                r"scan.*security.*issues", r"vulnerability.*assessment"
            ]
        }
    
    async def parse_request(self, user_input: str) -> CommandIntent:
        """Parse natural language input into a command intent."""
        text_lower = user_input.lower().strip()
        
        # First, try direct command patterns
        command_intent = self._match_direct_commands(text_lower, user_input)
        if command_intent:
            return command_intent
            
        # Try complex task patterns
        task_intent = self._match_task_patterns(text_lower, user_input)
        if task_intent:
            return task_intent
            
        # Fallback: create a generic task intent
        return CommandIntent(
            command="generic_task",
            parameters={"description": user_input, "requires_agent": True},
            confidence=0.5,
            raw_text=user_input,
            agent_type="codex"  # Default to codex for generic tasks
        )
    
    def _match_direct_commands(self, text_lower: str, original_text: str) -> Optional[CommandIntent]:
        """Match direct CLI commands using patterns."""
        best_match = None
        best_confidence = 0.0
        
        for command, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    confidence = len(re.findall(pattern, text_lower)) * 0.4
                    confidence += 1.0 if text_lower.startswith(pattern.split('.*')[0]) else 0.2
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        parameters = self._extract_command_parameters(command, text_lower)
                        best_match = CommandIntent(
                            command=command,
                            parameters=parameters,
                            confidence=confidence,
                            raw_text=original_text
                        )
        
        return best_match if best_confidence > 0.6 else None
    
    def _match_task_patterns(self, text_lower: str, original_text: str) -> Optional[CommandIntent]:
        """Match complex task patterns requiring agent orchestration."""
        for task_type, patterns in self.task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    confidence = 0.8  # High confidence for task patterns
                    
                    # Determine which agent type is best for this task
                    agent_type = self._determine_agent_for_task(task_type, text_lower)
                    
                    parameters = {
                        "task_type": task_type,
                        "description": original_text,
                        "requires_orchestration": True
                    }
                    
                    # Add task-specific parameters
                    if task_type == "test_and_fix":
                        parameters.update({
                            "run_tests": True,
                            "fix_issues": True,
                            "comprehensive": True
                        })
                    elif task_type == "build_deploy":
                        parameters.update({
                            "build": True,
                            "deploy": "deploy" in text_lower,
                            "target": self._extract_build_target(text_lower)
                        })
                    
                    return CommandIntent(
                        command="orchestrate_task",
                        parameters=parameters,
                        confidence=confidence,
                        raw_text=original_text,
                        agent_type=agent_type
                    )
        
        return None
    
    def _extract_command_parameters(self, command: str, text: str) -> Dict[str, Any]:
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
    
    def _determine_agent_for_task(self, task_type: str, text: str) -> str:
        """Determine the best agent type for a given task."""
        # Task type to agent mapping
        task_agent_map = {
            "test_and_fix": "codex",  # Codex is great for testing and fixing
            "code_review": "claude",  # Claude excels at code analysis
            "build_deploy": "codex",  # Codex handles builds well
            "refactor": "claude",     # Claude good for refactoring
            "documentation": "claude", # Claude excellent at documentation
            "security_scan": "codex"  # Codex good for security analysis
        }
        
        # Context-specific overrides
        if "large" in text or "complex" in text or "comprehensive" in text:
            return "claude"  # Use Claude for large/complex tasks (big context window)
        elif "local" in text or "private" in text or "offline" in text:
            return "ollama"  # Use Ollama for local/private work
            
        return task_agent_map.get(task_type, "codex")
    
    def _extract_build_target(self, text: str) -> str:
        """Extract build target from text."""
        targets = ["linux", "windows", "macos", "docker", "arm", "x86"]
        for target in targets:
            if target in text:
                return target
        return "default"
    
    async def should_use_agent_orchestration(self, command_intent: CommandIntent) -> bool:
        """Determine if a command requires agent orchestration."""
        orchestration_commands = [
            "orchestrate_task", "test_and_fix", "code_review", 
            "build_deploy", "refactor", "documentation", "security_scan"
        ]
        
        return (
            command_intent.command in orchestration_commands or
            command_intent.parameters.get("requires_orchestration", False) or
            command_intent.parameters.get("requires_agent", False)
        )