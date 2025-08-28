"""Intelligent command suggestion system for AgentsMCP CLI."""

from __future__ import annotations
import re
import json
import click
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
import os


@dataclass
class CommandUsage:
    """Track command usage for learning purposes."""
    command: str
    timestamp: datetime
    success: bool
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "context": self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CommandUsage:
        return cls(
            command=data["command"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            success=data["success"],
            context=data["context"]
        )


@dataclass
class Suggestion:
    """A command suggestion with context and confidence."""
    command: str
    description: str
    confidence: float
    category: str  # "typo", "next-step", "workflow", "learning"
    context: Optional[str] = None
    
    def format_display(self) -> str:
        """Format suggestion for display."""
        emoji_map = {
            "typo": "🔤",
            "next-step": "➡️", 
            "workflow": "🔄",
            "learning": "💡"
        }
        emoji = emoji_map.get(self.category, "💡")
        
        confidence_indicator = ""
        if self.confidence >= 0.9:
            confidence_indicator = " ⭐"
        elif self.confidence >= 0.7:
            confidence_indicator = " ✨"
        
        result = f"{emoji} {click.style(self.command, fg='cyan', bold=True)}{confidence_indicator}"
        if self.context:
            result += f"\n   {click.style(self.context, fg='white', dim=True)}"
        result += f"\n   {self.description}"
        return result


class ContextEngine:
    """Analyze current context to inform suggestions."""
    
    def __init__(self):
        self.current_dir = Path.cwd()
        self.config_exists = self._check_config_exists()
        self.git_repo = self._check_git_repo()
    
    def _check_config_exists(self) -> bool:
        """Check if AgentsMCP config exists."""
        try:
            from agentsmcp.paths import default_user_config_path
            return default_user_config_path().exists()
        except:
            return False
    
    def _check_git_repo(self) -> bool:
        """Check if we're in a git repository."""
        return (self.current_dir / '.git').exists()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context information."""
        return {
            "config_exists": self.config_exists,
            "git_repo": self.git_repo,
            "current_dir": str(self.current_dir),
            "has_package_json": (self.current_dir / "package.json").exists(),
            "has_requirements": (self.current_dir / "requirements.txt").exists(),
            "has_pyproject": (self.current_dir / "pyproject.toml").exists(),
        }


class TypoDetector:
    """Detect typos and suggest corrections for commands."""
    
    def __init__(self):
        # Common command mappings from errors.py
        self.command_mappings = {
            'start': 'run simple',
            'exec': 'run simple', 
            'execute': 'run simple',
            'launch': 'run interactive',
            'chat': 'run interactive',
            'talk': 'run interactive',
            'price': 'monitor costs',
            'cost': 'monitor costs',
            'spend': 'monitor costs',
            'money': 'monitor budget',
            'ai': 'knowledge models',
            'model': 'knowledge models',
            'llm': 'knowledge models',
            'learn': 'knowledge rag',
            'search': 'knowledge rag',
            'configure': 'init config',
            'settings': 'config show',
            'install': 'init setup',
            'create': 'init setup',
            'new': 'init setup',
        }
        
        # Valid commands for fuzzy matching
        self.valid_commands = [
            'init setup', 'init config', 'init demo',
            'run simple', 'run interactive', 'run advanced',
            'monitor costs', 'monitor budget', 'monitor usage',
            'knowledge models', 'knowledge rag', 'knowledge update',
            'server start', 'server status', 'server stop',
            'config show', 'config edit', 'config validate'
        ]
    
    def suggest_correction(self, invalid_command: str) -> Optional[Suggestion]:
        """Suggest correction for potentially misspelled command."""
        invalid_lower = invalid_command.lower().strip()
        
        # Check exact mappings first
        if invalid_lower in self.command_mappings:
            correct_cmd = self.command_mappings[invalid_lower]
            return Suggestion(
                command=correct_cmd,
                description=f"Did you mean '{correct_cmd}' instead of '{invalid_command}'?",
                confidence=0.95,
                category="typo",
                context="Common command alias"
            )
        
        # Fuzzy matching against valid commands
        best_match = None
        best_ratio = 0.0
        
        for valid_cmd in self.valid_commands:
            ratio = SequenceMatcher(None, invalid_lower, valid_cmd.lower()).ratio()
            if ratio > best_ratio and ratio > 0.6:  # Minimum similarity threshold
                best_ratio = ratio
                best_match = valid_cmd
        
        if best_match:
            return Suggestion(
                command=best_match,
                description=f"Did you mean '{best_match}' instead of '{invalid_command}'?",
                confidence=best_ratio,
                category="typo",
                context=f"Fuzzy match (similarity: {best_ratio:.1%})"
            )
        
        return None


class WorkflowEngine:
    """Suggest next logical steps based on current state and history."""
    
    def __init__(self, context_engine: ContextEngine):
        self.context_engine = context_engine
        
        # Workflow sequences
        self.workflows = {
            "first_time_setup": [
                "init setup",
                "run simple 'hello world'", 
                "monitor costs",
                "knowledge models"
            ],
            "development": [
                "run interactive",
                "knowledge rag",
                "monitor usage",
                "server start"
            ],
            "cost_management": [
                "monitor costs",
                "monitor budget --check",
                "config show",
                "run simple --cost-sensitive"
            ]
        }
    
    def suggest_next_steps(self, last_command: Optional[str] = None) -> List[Suggestion]:
        """Suggest next logical steps."""
        context = self.context_engine.get_context()
        suggestions = []
        
        # First-time user workflow
        if not context["config_exists"]:
            suggestions.append(Suggestion(
                command="init setup",
                description="Set up AgentsMCP for first-time use with guided configuration",
                confidence=0.95,
                category="workflow",
                context="First-time setup needed"
            ))
            return suggestions
        
        # Post-command suggestions
        if last_command:
            post_command_suggestions = self._get_post_command_suggestions(last_command)
            suggestions.extend(post_command_suggestions)
        
        # Context-based suggestions
        if context["git_repo"] and (context["has_package_json"] or context["has_pyproject"]):
            suggestions.append(Suggestion(
                command="run interactive",
                description="Start interactive coding session for this project",
                confidence=0.8,
                category="workflow", 
                context="Development project detected"
            ))
        
        return suggestions
    
    def _get_post_command_suggestions(self, last_command: str) -> List[Suggestion]:
        """Get suggestions based on the last executed command."""
        suggestions = []
        
        post_command_map = {
            "init setup": [
                ("run simple 'hello world'", "Test your setup with a simple task", 0.9),
                ("knowledge models", "Explore available AI models", 0.7)
            ],
            "run simple": [
                ("monitor costs", "Check the cost of your last task", 0.8),
                ("run interactive", "Try interactive mode for more control", 0.7)
            ],
            "monitor costs": [
                ("monitor budget --check", "Check your spending against budget", 0.8),
                ("config show", "Review cost settings", 0.6)
            ],
            "knowledge models": [
                ("run interactive", "Start a session with your preferred model", 0.8),
                ("knowledge rag", "Set up knowledge retrieval", 0.7)
            ]
        }
        
        if last_command in post_command_map:
            for cmd, desc, confidence in post_command_map[last_command]:
                suggestions.append(Suggestion(
                    command=cmd,
                    description=desc,
                    confidence=confidence,
                    category="next-step",
                    context=f"After '{last_command}'"
                ))
        
        return suggestions


class UsageTracker:
    """Track command usage for learning and personalization."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path.home() / ".agentsmcp" / "suggestions"
        self.data_dir = data_dir
        self.usage_file = self.data_dir / "usage_history.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.usage_history: List[CommandUsage] = self._load_history()
    
    def _load_history(self) -> List[CommandUsage]:
        """Load usage history from disk."""
        if not self.usage_file.exists():
            return []
        
        try:
            with open(self.usage_file, 'r') as f:
                data = json.load(f)
            return [CommandUsage.from_dict(item) for item in data]
        except (json.JSONDecodeError, KeyError):
            return []
    
    def _save_history(self):
        """Save usage history to disk."""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump([usage.to_dict() for usage in self.usage_history], f)
        except OSError:
            pass  # Fail silently if we can't save
    
    def record_usage(self, command: str, success: bool = True, context: Optional[Dict[str, Any]] = None):
        """Record command usage."""
        usage = CommandUsage(
            command=command,
            timestamp=datetime.now(),
            success=success,
            context=context or {}
        )
        self.usage_history.append(usage)
        
        # Keep only recent history (last 100 commands)
        self.usage_history = self.usage_history[-100:]
        self._save_history()
    
    def get_frequently_used(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get frequently used commands."""
        # Count commands from last 30 days
        cutoff = datetime.now() - timedelta(days=30)
        recent_commands = [
            usage.command for usage in self.usage_history 
            if usage.timestamp > cutoff and usage.success
        ]
        
        # Count occurrences
        command_counts = {}
        for cmd in recent_commands:
            command_counts[cmd] = command_counts.get(cmd, 0) + 1
        
        # Sort by count and return top N
        return sorted(command_counts.items(), key=lambda x: x[1], reverse=True)[:limit]


class IntelligentSuggestionSystem:
    """Main suggestion system orchestrator."""
    
    def __init__(self):
        self.context_engine = ContextEngine()
        self.typo_detector = TypoDetector()
        self.workflow_engine = WorkflowEngine(self.context_engine)
        self.usage_tracker = UsageTracker()
    
    def suggest_for_invalid_command(self, invalid_command: str) -> List[Suggestion]:
        """Get suggestions for an invalid command (typo detection)."""
        suggestions = []
        
        # Try typo correction first
        typo_suggestion = self.typo_detector.suggest_correction(invalid_command)
        if typo_suggestion:
            suggestions.append(typo_suggestion)
        
        # Add workflow suggestions as alternatives
        workflow_suggestions = self.workflow_engine.suggest_next_steps()
        suggestions.extend(workflow_suggestions[:2])  # Limit to top 2
        
        return suggestions
    
    def suggest_next_actions(self, last_command: Optional[str] = None) -> List[Suggestion]:
        """Get suggestions for next actions."""
        suggestions = []
        
        # Workflow-based suggestions
        workflow_suggestions = self.workflow_engine.suggest_next_steps(last_command)
        suggestions.extend(workflow_suggestions)
        
        # Learning-based suggestions from usage history
        frequently_used = self.usage_tracker.get_frequently_used(3)
        for cmd, count in frequently_used:
            if cmd != last_command:  # Don't suggest the same command again
                suggestions.append(Suggestion(
                    command=cmd,
                    description=f"You've used this command {count} times recently",
                    confidence=min(0.7 + (count * 0.05), 0.9),
                    category="learning",
                    context="Frequently used"
                ))
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def record_command_usage(self, command: str, success: bool = True):
        """Record that a command was used."""
        context = self.context_engine.get_context()
        self.usage_tracker.record_usage(command, success, context)


def display_suggestions(suggestions: List[Suggestion], title: str = "💡 Suggestions"):
    """Display suggestions to the user."""
    if not suggestions:
        return
    
    click.echo(f"\n{click.style(title, fg='blue', bold=True)}")
    click.echo("─" * len(title.replace('💡 ', '')))
    
    for i, suggestion in enumerate(suggestions, 1):
        click.echo(f"\n{i}. {suggestion.format_display()}")
    
    click.echo()  # Extra newline for spacing


# Global instance
_suggestion_system = IntelligentSuggestionSystem()


def get_suggestion_system() -> IntelligentSuggestionSystem:
    """Get the global suggestion system instance."""
    return _suggestion_system