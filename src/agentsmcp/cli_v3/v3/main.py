"""Main CLI v3 entry point - Revolutionary UX with natural language support.

This module provides the unified entry point that integrates all CLI v3 components
into a revolutionary command-line experience with natural language processing,
progressive disclosure, and intelligent command routing.
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .core.command_engine import CommandEngine, CommandHandler, IntelligenceProvider, TelemetryCollector, CrossModalCoordinator
from .nlp.processor import NaturalLanguageProcessor
from .intelligence.user_profiler import UserProfiler
from .coordination.modal_coordinator import ModalCoordinator
from .registry.command_registry import CommandRegistry
from .models.command_models import (
    CommandRequest, ExecutionMode, UserProfile, SkillLevel, 
    CommandEngineError, CommandNotFoundError, ValidationFailedError
)
from .models.nlp_models import ConversationContext, LLMConfig
from .migration.legacy_adapter import LegacyAdapter

logger = logging.getLogger(__name__)
console = Console()

class CliV3IntelligenceProvider(IntelligenceProvider):
    """Intelligence provider integration for CLI v3."""
    
    def __init__(self, nlp_processor: NaturalLanguageProcessor):
        self.nlp_processor = nlp_processor
    
    async def parse_natural_language(self, input_text: str, context) -> CommandRequest:
        """Parse natural language input into structured command request."""
        # Convert execution context to conversation context
        conv_context = ConversationContext(
            command_history=context.user_profile.command_history,
            current_directory=".",
            recent_files=[],
            user_preferences={}
        )
        
        # Parse the natural language
        parsing_result = await self.nlp_processor.parse_command(input_text, conv_context)
        
        if not parsing_result.success or not parsing_result.structured_command:
            raise ValidationFailedError("Failed to parse natural language input")
        
        parsed = parsing_result.structured_command
        
        # Convert to CommandRequest
        return CommandRequest(
            command_type=parsed.action,
            arguments=parsed.parameters,
            options={},
            metadata={"confidence": parsed.confidence, "method": parsed.method.value}
        )
    
    async def analyze_user_intent(self, request: CommandRequest, context) -> Dict[str, Any]:
        """Analyze user intent and provide additional context."""
        return {
            "intent_confidence": request.metadata.get("confidence", 0.8),
            "parsing_method": request.metadata.get("method", "unknown"),
            "suggested_improvements": []
        }
    
    async def generate_smart_suggestions(self, context, recent_commands: List[str], current_result: Any = None) -> List:
        """Generate contextual smart suggestions."""
        conv_context = ConversationContext(
            command_history=recent_commands,
            current_directory=".",
            recent_files=[],
            user_preferences={}
        )
        
        suggestions = await self.nlp_processor.get_command_suggestions(conv_context)
        
        # Convert to Suggestion objects (simplified for now)
        from ..models.command_models import Suggestion
        return [
            Suggestion(
                text=suggestion,
                command=suggestion.split()[0] if suggestion else "help",
                confidence=0.8,
                category="contextual"
            )
            for suggestion in suggestions[:5]
        ]

class CliV3TelemetryCollector(TelemetryCollector):
    """Telemetry collector for CLI v3 metrics."""
    
    def __init__(self):
        self.metrics = {
            "total_commands": 0,
            "natural_language_commands": 0,
            "direct_commands": 0,
            "errors": 0,
            "avg_response_time": 0.0
        }
    
    async def record_command_execution(self, request, result, metrics, context) -> None:
        """Record command execution metrics."""
        self.metrics["total_commands"] += 1
        
        # Track natural language vs direct commands
        if request.metadata.get("is_natural_language"):
            self.metrics["natural_language_commands"] += 1
        else:
            self.metrics["direct_commands"] += 1
        
        # Update average response time
        if self.metrics["avg_response_time"] == 0:
            self.metrics["avg_response_time"] = metrics.duration_ms
        else:
            self.metrics["avg_response_time"] = (
                self.metrics["avg_response_time"] * 0.9 + metrics.duration_ms * 0.1
            )
    
    async def record_error(self, error: Exception, request, context) -> None:
        """Record error occurrence."""
        self.metrics["errors"] += 1
        logger.warning(f"Command error: {type(error).__name__}: {error}")
    
    async def record_user_behavior(self, event: str, data: Dict[str, Any], context) -> None:
        """Record user behavior patterns."""
        logger.debug(f"User behavior: {event} - {data}")

class CliV3CrossModalCoordinator(CrossModalCoordinator):
    """Cross-modal coordinator for CLI v3."""
    
    def __init__(self):
        self.active_modes = set()
        self.session_state = {}
    
    async def notify_mode_switch(self, from_mode: ExecutionMode, to_mode: ExecutionMode, context) -> None:
        """Notify of interface mode switch."""
        self.active_modes.discard(from_mode)
        self.active_modes.add(to_mode)
        logger.info(f"Mode switch: {from_mode} -> {to_mode}")
    
    async def sync_session_state(self, context, target_modes: List[ExecutionMode]) -> None:
        """Synchronize session state across modes."""
        # In CLI v3, we primarily coordinate with TUI and WebUI
        logger.debug(f"Syncing session state across modes: {target_modes}")

# Core command handlers
class HelpCommandHandler(CommandHandler):
    """Handler for help commands."""
    
    def __init__(self):
        super().__init__("help")
        self.required_permissions = []
    
    async def execute(self, request: CommandRequest, context) -> Any:
        """Execute help command."""
        topic = request.arguments.get("topic", "")
        
        if topic:
            return self._get_topic_help(topic, context)
        else:
            return self._get_general_help(context)
    
    def _get_general_help(self, context) -> Dict[str, Any]:
        """Generate general help based on user skill level."""
        skill_level = context.user_profile.skill_level
        
        if skill_level == SkillLevel.BEGINNER:
            commands = [
                "help - Show this help message",
                "status - Check system status", 
                "run <task> - Execute a simple task",
                "tui - Launch interactive interface"
            ]
        elif skill_level == SkillLevel.INTERMEDIATE:
            commands = [
                "help [topic] - Show help for specific topic",
                "status - Check system status",
                "run <task> - Execute task with options",
                "analyze <file> - Analyze code or data",
                "tui - Launch interactive interface",
                "config - Manage configuration"
            ]
        else:  # EXPERT
            commands = [
                "help [topic] - Contextual help system",
                "status [--detailed] - System health metrics",
                "run <task> [--complexity=<level>] - Advanced task execution",
                "analyze <target> [--deep] - Deep analysis tools",
                "tui [--advanced] - Power user interface",
                "config edit - Direct configuration editing",
                "debug - Debug and diagnostic tools"
            ]
        
        return {
            "title": "AgentsMCP CLI v3 - Revolutionary Multi-Agent Orchestration",
            "description": "Natural language command interface with intelligent assistance",
            "commands": commands,
            "skill_level": skill_level.value,
            "natural_language": True
        }
    
    def _get_topic_help(self, topic: str, context) -> Dict[str, Any]:
        """Get help for specific topic."""
        help_topics = {
            "natural": {
                "title": "Natural Language Commands",
                "content": [
                    "You can use natural language with AgentsMCP:",
                    "  agentsmcp 'analyze my Python code'",
                    "  agentsmcp 'help me set up the project'",
                    "  agentsmcp 'start the interactive interface'",
                    "  agentsmcp 'optimize my costs'",
                    "",
                    "Natural language commands are processed intelligently",
                    "and converted to appropriate structured commands."
                ]
            },
            "modes": {
                "title": "Interface Modes",
                "content": [
                    "AgentsMCP supports multiple interface modes:",
                    "  CLI - Command line interface (this mode)",
                    "  TUI - Terminal user interface (interactive)",
                    "  WebUI - Web-based interface",
                    "  API - Programmatic access",
                    "",
                    "Use 'agentsmcp tui' to launch interactive mode.",
                    "Session state is synchronized across all modes."
                ]
            },
            "skill": {
                "title": "Skill Levels & Progressive Disclosure",
                "content": [
                    "AgentsMCP adapts to your skill level:",
                    "  Beginner - Simple commands, guided help",
                    "  Intermediate - More options, explanations",
                    "  Expert - Full command set, minimal hand-holding",
                    "",
                    "Set your skill level with:",
                    "  agentsmcp config set skill_level <level>"
                ]
            }
        }
        
        if topic in help_topics:
            return help_topics[topic]
        else:
            return {
                "title": f"Help for '{topic}'",
                "content": [f"No help available for topic '{topic}'", "Use 'help' for general help."]
            }

class StatusCommandHandler(CommandHandler):
    """Handler for status commands."""
    
    def __init__(self, engine: CommandEngine, registry: CommandRegistry):
        super().__init__("status")
        self.engine = engine
        self.registry = registry
    
    async def execute(self, request: CommandRequest, context) -> Any:
        """Execute status command."""
        detailed = request.arguments.get("detailed", False)
        
        # Get basic status
        engine_status = self.engine.get_engine_status()
        registry_stats = self.registry.get_registry_stats()
        
        status = {
            "system": "healthy",
            "uptime": engine_status["uptime_seconds"],
            "commands": {
                "total": registry_stats.total_commands,
                "active": registry_stats.active_commands,
                "deprecated": registry_stats.deprecated_commands
            },
            "engine": {
                "active_commands": engine_status["active_commands"],
                "handlers": engine_status["registered_handlers"],
                "intelligence": engine_status["intelligence_provider"],
                "telemetry": engine_status["telemetry_collector"]
            }
        }
        
        if detailed:
            status.update({
                "performance": {
                    "avg_lookup_time_ms": registry_stats.avg_lookup_time_ms,
                    "avg_discovery_time_ms": registry_stats.avg_discovery_time_ms,
                    "avg_validation_time_ms": registry_stats.avg_validation_time_ms
                },
                "usage": {
                    "most_used": registry_stats.most_used_commands[:5]
                },
                "categories": registry_stats.commands_by_category,
                "skill_levels": registry_stats.commands_by_skill_level
            })
        
        return status

class TuiLaunchHandler(CommandHandler):
    """Handler for launching TUI interface."""
    
    def __init__(self):
        super().__init__("tui")
    
    async def execute(self, request: CommandRequest, context) -> Any:
        """Launch TUI interface."""
        try:
            # Import TUI launcher
            from ....ui.v3.tui_launcher import launch_tui
            
            # Launch TUI asynchronously
            exit_code = await launch_tui()
            
            return {
                "launched": True,
                "exit_code": exit_code,
                "message": "TUI interface completed"
            }
            
        except ImportError:
            return {
                "launched": False,
                "error": "TUI interface not available",
                "message": "Install TUI dependencies or check installation"
            }
        except Exception as e:
            return {
                "launched": False, 
                "error": str(e),
                "message": "Failed to launch TUI interface"
            }

class NaturalLanguageHandler(CommandHandler):
    """Handler for natural language input processing."""
    
    def __init__(self):
        super().__init__("natural")
    
    async def execute(self, request: CommandRequest, context) -> Any:
        """Process natural language command."""
        # This handler gets the converted command after NLP processing
        # It mainly serves as a pass-through with logging
        
        return {
            "processed": True,
            "original_input": request.metadata.get("original_input"),
            "confidence": request.metadata.get("confidence", 0.0),
            "method": request.metadata.get("method", "unknown")
        }


class CliV3Main:
    """Main CLI v3 application class."""
    
    def __init__(self):
        self.engine = CommandEngine()
        self.registry = CommandRegistry()
        self.nlp_processor = NaturalLanguageProcessor()
        self.user_profiler = UserProfiler()
        self.modal_coordinator = ModalCoordinator()
        self.legacy_adapter = LegacyAdapter()
        
        # Set up integrations
        self.intelligence_provider = CliV3IntelligenceProvider(self.nlp_processor)
        self.telemetry_collector = CliV3TelemetryCollector()
        self.cross_modal_coordinator = CliV3CrossModalCoordinator()
        
        # Configure engine
        self.engine.set_intelligence_provider(self.intelligence_provider)
        self.engine.set_telemetry_collector(self.telemetry_collector) 
        self.engine.set_cross_modal_coordinator(self.cross_modal_coordinator)
        
        # Register core handlers
        self._register_core_handlers()
        
        # Initialize startup time
        self.startup_time = time.time()
        
        logger.info("CLI v3 main application initialized")
    
    def _register_core_handlers(self):
        """Register core command handlers."""
        # Register built-in handlers
        help_handler = HelpCommandHandler()
        self.engine.register_handler(help_handler)
        
        status_handler = StatusCommandHandler(self.engine, self.registry)
        self.engine.register_handler(status_handler)
        
        tui_handler = TuiLaunchHandler()
        self.engine.register_handler(tui_handler)
        
        natural_handler = NaturalLanguageHandler()
        self.engine.register_handler(natural_handler)
        
        logger.info("Registered core command handlers")
    
    async def execute_command(
        self, 
        command_input: str,
        execution_mode: ExecutionMode = ExecutionMode.CLI,
        user_profile: Optional[UserProfile] = None
    ) -> Tuple[bool, Any, str]:
        """Execute a command through the v3 engine.
        
        Args:
            command_input: Raw command input (natural language or structured)
            execution_mode: Interface execution mode
            user_profile: User context profile
            
        Returns:
            Tuple of (success, result_data, message)
        """
        start_time = time.time()
        
        try:
            # Get or create user profile
            if user_profile is None:
                user_profile = await self.user_profiler.get_or_create_profile("default")
            
            # Determine if input is natural language
            is_natural_language = self._is_natural_language_input(command_input)
            
            if is_natural_language:
                # Parse natural language
                command_request = await self._parse_natural_language_input(
                    command_input, user_profile
                )
                command_request.metadata["is_natural_language"] = True
                command_request.metadata["original_input"] = command_input
            else:
                # Parse structured command
                command_request = self._parse_structured_command(command_input)
                command_request.metadata["is_natural_language"] = False
            
            # Execute through engine
            result, metrics, next_actions = await self.engine.execute_command(
                command_request, execution_mode, user_profile
            )
            
            # Update user profile with command
            user_profile.command_history.append(command_input)
            if len(user_profile.command_history) > 50:
                user_profile.command_history = user_profile.command_history[-50:]
            
            await self.user_profiler.update_profile(user_profile)
            
            # Calculate total time
            total_time = time.time() - start_time
            
            if result.success:
                message = f"Command completed in {total_time*1000:.0f}ms"
                if next_actions:
                    message += f" • {len(next_actions)} suggestions available"
                return True, result.data, message
            else:
                error_msg = "Command failed"
                if result.errors:
                    error_msg = result.errors[0].message
                return False, result.data, error_msg
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return False, None, f"Execution error: {str(e)}"
    
    def _is_natural_language_input(self, command_input: str) -> bool:
        """Determine if input is natural language vs structured command."""
        command_input = command_input.strip()
        
        # Check for quoted natural language
        if (command_input.startswith('"') and command_input.endswith('"')) or \
           (command_input.startswith("'") and command_input.endswith("'")):
            return True
        
        # Check for natural language indicators
        natural_indicators = [
            "help me", "i want", "i need", "can you", "please", "how do i",
            "show me", "tell me", "what is", "analyze my", "check my"
        ]
        
        command_lower = command_input.lower()
        if any(indicator in command_lower for indicator in natural_indicators):
            return True
        
        # Check for sentence structure (contains common English words)
        sentence_words = ["the", "a", "an", "and", "or", "but", "my", "this", "that"]
        words = command_lower.split()
        if len(words) > 3 and any(word in sentence_words for word in words):
            return True
        
        # Check for structured command patterns
        structured_patterns = [
            command_input.split()[0] in ["help", "status", "run", "config", "tui", "analyze"],
            "--" in command_input,
            command_input.startswith("-"),
            "=" in command_input
        ]
        
        if any(structured_patterns):
            return False
        
        # Default to natural language for longer inputs
        return len(words) > 2
    
    async def _parse_natural_language_input(
        self, 
        command_input: str, 
        user_profile: UserProfile
    ) -> CommandRequest:
        """Parse natural language input through NLP processor."""
        # Clean quoted input
        if (command_input.startswith('"') and command_input.endswith('"')) or \
           (command_input.startswith("'") and command_input.endswith("'")):
            command_input = command_input[1:-1]
        
        # Use intelligence provider to parse
        context_mock = type('obj', (object,), {'user_profile': user_profile})()
        return await self.intelligence_provider.parse_natural_language(command_input, context_mock)
    
    def _parse_structured_command(self, command_input: str) -> CommandRequest:
        """Parse structured command input."""
        parts = command_input.split()
        if not parts:
            raise ValidationFailedError("Empty command input")
        
        command_type = parts[0]
        arguments = {}
        options = {}
        
        # Basic argument parsing
        current_key = None
        for part in parts[1:]:
            if part.startswith("--"):
                # Long option
                if "=" in part:
                    key, value = part[2:].split("=", 1)
                    options[key] = value
                else:
                    current_key = part[2:]
            elif part.startswith("-"):
                # Short option
                current_key = part[1:]
            elif current_key:
                # Value for previous option
                options[current_key] = part
                current_key = None
            else:
                # Positional argument
                arg_index = len(arguments)
                arguments[f"arg_{arg_index}"] = part
        
        return CommandRequest(
            command_type=command_type,
            arguments=arguments,
            options=options
        )
    
    def get_startup_performance(self) -> Dict[str, float]:
        """Get startup performance metrics."""
        current_time = time.time()
        startup_duration = current_time - self.startup_time
        
        return {
            "startup_time_ms": startup_duration * 1000,
            "components_loaded": 6,  # engine, registry, nlp, profiler, coordinator, adapter
            "handlers_registered": len(self.engine._handlers),
            "ready": startup_duration < 0.2  # Target: <200ms startup
        }
    
    async def shutdown(self):
        """Gracefully shutdown the CLI v3 system."""
        logger.info("Shutting down CLI v3...")
        
        try:
            # Save any pending data
            await self.registry.save_registry()
            await self.user_profiler.save_profiles()
            
            # Shutdown components
            await self.engine.shutdown()
            
            logger.info("CLI v3 shutdown complete")
        except Exception as e:
            logger.error(f"Error during CLI v3 shutdown: {e}")

# Main entry functions

async def create_cli_v3() -> CliV3Main:
    """Create and initialize CLI v3 main application."""
    return CliV3Main()

def display_result(success: bool, result_data: Any, message: str, skill_level: SkillLevel = SkillLevel.INTERMEDIATE):
    """Display command result with appropriate formatting for skill level."""
    
    if success:
        # Success display
        if skill_level == SkillLevel.BEGINNER:
            console.print(f"✅ {message}", style="green")
            if isinstance(result_data, dict) and "title" in result_data:
                console.print(f"\n[bold]{result_data['title']}[/bold]")
                if "commands" in result_data:
                    for cmd in result_data["commands"]:
                        console.print(f"  {cmd}")
        else:
            # Intermediate/Expert display with more detail
            console.print(Panel(
                Text(message, style="green bold"),
                title="✅ Success",
                border_style="green"
            ))
            
            if isinstance(result_data, dict):
                if result_data.get("natural_language"):
                    console.print("\n💡 [dim]Tip: You can use natural language commands like 'analyze my code' or 'help me setup'[/dim]\n")
                
                # Display structured result data
                if "title" in result_data:
                    console.print(Markdown(f"## {result_data['title']}"))
                
                if "description" in result_data:
                    console.print(result_data["description"])
                    
                if "commands" in result_data:
                    console.print("\n[bold]Available Commands:[/bold]")
                    for cmd in result_data["commands"]:
                        console.print(f"  • {cmd}")
    else:
        # Error display
        console.print(Panel(
            Text(message, style="red"),
            title="❌ Error", 
            border_style="red"
        ))
        
        if skill_level == SkillLevel.BEGINNER:
            console.print("\n💡 [dim]Need help? Try: agentsmcp help[/dim]")

def display_beautiful_help():
    """Display beautiful help output with Unicode/ASCII fallbacks."""
    try:
        # Try Unicode version first
        help_content = """
# AgentsMCP CLI v3 🚀

Revolutionary multi-agent orchestration with natural language support.

## Natural Language Commands
```bash
agentsmcp "analyze my Python code"
agentsmcp "help me set up the project"
agentsmcp "start the interactive interface"
agentsmcp "optimize my costs"
```

## Structured Commands
```bash
agentsmcp help [topic]          # Show help information
agentsmcp status [--detailed]   # System status
agentsmcp tui                   # Launch interactive TUI
agentsmcp run <task>            # Execute task
```

## Interface Modes
- **CLI** - Command line (current)
- **TUI** - Interactive terminal UI
- **WebUI** - Browser interface  
- **API** - Programmatic access

## Progressive Disclosure
The CLI adapts to your skill level:
- 🌱 **Beginner** - Guided experience
- 🌿 **Intermediate** - Balanced power/simplicity
- 🌳 **Expert** - Full control

---
*Type `agentsmcp help <topic>` for detailed information*
        """
        
        console.print(Markdown(help_content))
        
    except Exception:
        # Fallback to ASCII version
        print("""
AgentsMCP CLI v3 - Revolutionary Multi-Agent Orchestration

NATURAL LANGUAGE COMMANDS:
  agentsmcp "analyze my Python code"
  agentsmcp "help me set up the project" 
  agentsmcp "start the interactive interface"
  agentsmcp "optimize my costs"

STRUCTURED COMMANDS:
  help [topic]          Show help information
  status [--detailed]   System status
  tui                   Launch interactive TUI
  run <task>            Execute task

INTERFACE MODES:
  CLI    - Command line (current)
  TUI    - Interactive terminal UI
  WebUI  - Browser interface
  API    - Programmatic access

The CLI adapts to your skill level for the perfect balance
of power and simplicity.

Type 'agentsmcp help <topic>' for detailed information.
        """)

# Click integration for the new CLI
@click.group(invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--skill-level", type=click.Choice(["beginner", "intermediate", "expert"]), 
              default="intermediate", help="Set skill level for progressive disclosure")
@click.option("--mode", type=click.Choice(["cli", "tui", "web", "api"]), 
              default="cli", help="Interface execution mode")
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.pass_context
def main(ctx, skill_level: str, mode: str, debug: bool):
    """AgentsMCP CLI v3 - Revolutionary multi-agent orchestration with natural language support.
    
    Examples:
      agentsmcp "analyze my code"           # Natural language
      agentsmcp help                        # Structured command  
      agentsmcp status --detailed           # Advanced options
      agentsmcp tui                         # Launch TUI
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    # Store context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['skill_level'] = SkillLevel(skill_level.upper())
    ctx.obj['mode'] = ExecutionMode(mode.upper())
    ctx.obj['debug'] = debug
    
    # If no subcommand provided, show help
    if ctx.invoked_subcommand is None:
        display_beautiful_help()

@main.command("execute")
@click.argument("command", required=True)
@click.pass_context
def execute_command(ctx, command: str):
    """Execute a command (natural language or structured)."""
    
    async def run_command():
        cli_app = await create_cli_v3()
        
        try:
            success, result, message = await cli_app.execute_command(
                command,
                execution_mode=ctx.obj['mode'],
                user_profile=None  # Will be auto-created
            )
            
            display_result(success, result, message, ctx.obj['skill_level'])
            
        except Exception as e:
            console.print(f"❌ Execution failed: {e}", style="red")
            return 1
        finally:
            await cli_app.shutdown()
        
        return 0 if success else 1
    
    # Run the async command
    exit_code = asyncio.run(run_command())
    ctx.exit(exit_code)

if __name__ == "__main__":
    main()