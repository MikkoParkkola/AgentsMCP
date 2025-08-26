"""
Revolutionary Command Interface - Beautiful CLI Interactions

Intelligent command-line interface inspired by:
- Claude Code's conversational interaction patterns
- Codex CLI's smart command suggestions
- Gemini CLI's context-aware assistance

Features:
- Smart command completion and suggestions
- Interactive parameter input with validation
- Beautiful command history and favorites
- Context-aware help and documentation
- Conversational command processing
"""

import asyncio
import sys
import os
import readline
import rlcompleter
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
import json
import shlex
import logging

from .theme_manager import ThemeManager
from .ui_components import UIComponents
from ..agent_manager import AgentManager
from ..config import Config, ProviderType, ProviderConfig
from ..providers import list_models as providers_list_models, ProviderError
from ..providers_validate import validate_provider_config, ValidationResult
from ..config_write import persist_provider_api_key
from ..stream import generate_stream_from_text, openai_stream_text
from rich.prompt import Prompt, Confirm
from rich.table import Table as RichTable
from rich.panel import Panel
from ..orchestration.orchestration_manager import OrchestrationManager
from ..conversation.conversation import ConversationManager
# Import settings UI after initial setup to avoid circular imports
try:
    from .modern_settings_ui import run_modern_settings_dialog
except ImportError:
    run_modern_settings_dialog = None

logger = logging.getLogger(__name__)

@dataclass
class CommandDefinition:
    """Definition of a CLI command"""
    name: str
    description: str
    handler: Callable
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    category: str = "general"
    examples: List[str] = field(default_factory=list)
    hidden: bool = False

@dataclass
class CommandHistory:
    """Command history entry"""
    command: str
    timestamp: datetime
    success: bool
    execution_time: float
    result_preview: str = ""

@dataclass
class InterfaceConfig:
    """Configuration for command interface"""
    prompt_style: str = "agentsmcp"
    show_suggestions: bool = True
    auto_complete: bool = True
    command_history_size: int = 1000
    enable_colors: bool = True
    interactive_mode: bool = True

class CommandInterface:
    """
    Revolutionary Command Interface
    
    Provides intelligent, beautiful command-line interactions with
    smart completion, context awareness, and conversational assistance.
    """
    
    def __init__(self, orchestration_manager: OrchestrationManager,
                 theme_manager: Optional[ThemeManager] = None,
                 config: Optional[InterfaceConfig] = None,
                 agent_manager: Optional[AgentManager] = None,
                 app_config: Optional[Config] = None):
        self.orchestration_manager = orchestration_manager
        self.theme_manager = theme_manager or ThemeManager()
        self.ui = UIComponents(self.theme_manager)
        self.config = config or InterfaceConfig()
        
        # Agent orchestration and configuration
        self.agent_manager = agent_manager
        self.app_config = app_config
        # Default to cloud ollama-turbo coding agent (gpt-oss:120b)
        self.current_agent = "ollama-turbo-coding"
        self.session_history: List[Tuple[str, str]] = []  # (role, text) pairs
        self.context_percent = 0  # Context trimming percentage
        self.stream_enabled = True  # Enable streaming by default
        
        # Command registry
        self.commands: Dict[str, CommandDefinition] = {}
        self.aliases: Dict[str, str] = {}
        self.categories: Dict[str, List[str]] = {}
        
        # Interface state
        self.is_running = False
        self.current_context: Dict[str, Any] = {}
        self.command_history: List[CommandHistory] = []
        self.favorites: List[str] = []
        
        # Completion and suggestions
        self.completer = None
        self.last_suggestions: List[str] = []
        
        # Conversational interface
        self.conversation_manager = ConversationManager(
            command_interface=self, 
            theme_manager=self.theme_manager,
            agent_manager=getattr(self, 'agent_manager', None)
        )
        self.conversational_mode = True  # Enable conversational mode by default
        
        # Initialize commands and readline
        self._register_built_in_commands()
        self._setup_readline()
    
    def _register_built_in_commands(self):
        """Register built-in CLI commands"""
        commands = [
            CommandDefinition(
                name="status",
                description="Show comprehensive system status",
                handler=self._cmd_status,
                category="monitoring",
                examples=["status", "status --detailed"],
                parameters=[
                    {"name": "detailed", "type": "bool", "default": False, "help": "Show detailed status information"}
                ]
            ),
            CommandDefinition(
                name="dashboard",
                description="Launch interactive status dashboard",
                handler=self._cmd_dashboard,
                category="monitoring",
                examples=["dashboard", "dashboard --refresh 0.5"],
                parameters=[
                    {"name": "refresh", "type": "float", "default": 1.0, "help": "Refresh interval in seconds"},
                    {"name": "compact", "type": "bool", "default": False, "help": "Use compact display mode"}
                ]
            ),
            CommandDefinition(
                name="execute",
                description="Execute a task using orchestration system",
                handler=self._cmd_execute,
                aliases=["run", "exec"],
                category="orchestration",
                examples=[
                    "execute 'Create a web API for user management'",
                    "execute --mode symphony 'Build a complete e-commerce system'"
                ],
                parameters=[
                    {"name": "task", "type": "str", "required": True, "help": "Task description to execute"},
                    {"name": "mode", "type": "str", "choices": ["seamless", "symphony", "predictive", "hybrid"], 
                     "default": "hybrid", "help": "Execution mode"},
                    {"name": "context", "type": "json", "help": "Additional context as JSON"}
                ]
            ),
            CommandDefinition(
                name="agents",
                description="Manage and monitor agents",
                handler=self._cmd_agents,
                category="orchestration",
                examples=["agents list", "agents spawn data-scientist", "agents status"],
                parameters=[
                    {"name": "action", "type": "str", "choices": ["list", "spawn", "status", "terminate"], 
                     "required": True, "help": "Action to perform"},
                    {"name": "specialization", "type": "str", "help": "Agent specialization for spawn action"},
                    {"name": "agent_id", "type": "str", "help": "Agent ID for specific actions"}
                ]
            ),
            CommandDefinition(
                name="symphony",
                description="Control Symphony mode orchestration",
                handler=self._cmd_symphony,
                category="orchestration",
                examples=["symphony start", "symphony status", "symphony stop"],
                parameters=[
                    {"name": "action", "type": "str", "choices": ["start", "stop", "status", "pause"], 
                     "required": True, "help": "Symphony action"},
                    {"name": "tasks", "type": "list", "help": "List of tasks for symphony execution"}
                ]
            ),
            CommandDefinition(
                name="theme",
                description="Manage UI theme settings",
                handler=self._cmd_theme,
                category="interface",
                examples=["theme auto", "theme dark", "theme light", "theme info"],
                parameters=[
                    {"name": "mode", "type": "str", "choices": ["auto", "dark", "light", "info"], 
                     "default": "info", "help": "Theme mode or info"}
                ]
            ),
            CommandDefinition(
                name="history",
                description="Show command history with filtering",
                handler=self._cmd_history,
                category="interface",
                examples=["history", "history --filter execute", "history --count 20"],
                parameters=[
                    {"name": "filter", "type": "str", "help": "Filter history by command or pattern"},
                    {"name": "count", "type": "int", "default": 20, "help": "Number of history items to show"}
                ]
            ),
            CommandDefinition(
                name="help",
                description="Show help information for commands",
                handler=self._cmd_help,
                aliases=["h", "?"],
                category="interface",
                examples=["help", "help execute", "help --category orchestration"],
                parameters=[
                    {"name": "command", "type": "str", "help": "Specific command to get help for"},
                    {"name": "category", "type": "str", "help": "Show commands in specific category"}
                ]
            ),
            CommandDefinition(
                name="config",
                description="Manage interface configuration",
                handler=self._cmd_config,
                category="interface",
                examples=["config show", "config set auto_complete true"],
                parameters=[
                    {"name": "action", "type": "str", "choices": ["show", "set", "reset"], 
                     "default": "show", "help": "Configuration action"},
                    {"name": "key", "type": "str", "help": "Configuration key"},
                    {"name": "value", "type": "str", "help": "Configuration value"}
                ]
            ),
            CommandDefinition(
                name="clear",
                description="Clear the terminal screen",
                handler=self._cmd_clear,
                aliases=["cls"],
                category="interface",
                examples=["clear"]
            ),
            CommandDefinition(
                name="exit",
                description="Exit the interactive interface",
                handler=self._cmd_exit,
                aliases=["quit", "q"],
                category="interface",
                examples=["exit"]
            )
        ]
        
        # Register commands
        for cmd in commands:
            self.register_command(cmd)
        
        # Register settings command if available
        if run_modern_settings_dialog:
            settings_cmd = CommandDefinition(
                name="settings",
                description="Configure LLM provider, model and generation settings",
                handler=self._cmd_settings,
                category="interface",
                examples=["settings"],
                parameters=[]
            )
            self.register_command(settings_cmd)

        # Always register keys command so users can check state even without the full settings UI
        keys_cmd = CommandDefinition(
            name="keys",
            description="Show API key status for supported providers (config first, then env)",
            handler=self._cmd_keys,
            category="interface",
            examples=["keys"],
            parameters=[]
        )
        self.register_command(keys_cmd)

        # Export-keys helper (prints ready-to-copy export lines)
        export_keys_cmd = CommandDefinition(
            name="export-keys",
            description="Print export commands for API keys (placeholders by default; use --include-values to print actual keys)",
            handler=self._cmd_export_keys,
            category="interface",
            examples=["export-keys", "export-keys --include-values"],
            parameters=[{"name": "include_values", "type": "bool", "default": False}]
        )
        self.register_command(export_keys_cmd)

        # Provider order insight
        provider_order_cmd = CommandDefinition(
            name="provider-order",
            description="Show the provider fallback order and which will be skipped due to missing keys",
            handler=self._cmd_provider_order,
            category="interface",
            examples=["provider-order"],
            parameters=[]
        )
        self.register_command(provider_order_cmd)
        # Register generate-config command
        generate_config_cmd = CommandDefinition(
            name="generate-config",
            description="Generate MCP client configuration with auto-discovered paths",
            handler=self._cmd_generate_config,
            category="interface", 
            examples=["generate-config"],
            parameters=[]
        )
        self.register_command(generate_config_cmd)

        # Session management commands
        session_commands = [
                CommandDefinition(
                    name="analyze",
                    description="Analyze the current repository for structure and issues",
                    handler=self._cmd_analyze,
                    category="session",
                    examples=["analyze"],
                    parameters=[]
                ),
                CommandDefinition(
                    name="provider-use",
                    description="Set the primary provider for this session and persist to user settings",
                    handler=self._cmd_provider_use,
                    category="session",
                    examples=["provider-use openai", "provider-use ollama-turbo"],
                    parameters=[{"name": "provider", "type": "str", "help": "Provider name"}]
                ),
                CommandDefinition(
                    name="agent",
                    description="Switch AI agent (codex/claude/ollama)",
                    handler=self._cmd_agent_switch,
                    category="session",
                    examples=["agent codex", "agent claude", "agent ollama"],
                    parameters=[
                        {"name": "agent_type", "type": "str", "choices": ["codex", "claude", "ollama"], 
                         "help": "Agent type to switch to"}
                    ]
                ),
                CommandDefinition(
                    name="model",
                    description="Set model for current session",
                    handler=self._cmd_model,
                    category="session",
                    examples=["model gpt-4", "model claude-3-5-sonnet"],
                    parameters=[
                        {"name": "model_name", "type": "str", "help": "Model name"}
                    ]
                ),
                CommandDefinition(
                    name="provider",
                    description="Set provider for current session",
                    handler=self._cmd_provider,
                    category="session",
                    examples=["provider openai", "provider anthropic"],
                    parameters=[
                        {"name": "provider_name", "type": "str", "choices": ["openai", "anthropic", "ollama", "openrouter"], 
                         "help": "Provider name"}
                    ]
                ),
                CommandDefinition(
                    name="stream",
                    description="Toggle streaming mode",
                    handler=self._cmd_stream,
                    category="session",
                    examples=["stream on", "stream off"],
                    parameters=[
                        {"name": "mode", "type": "str", "choices": ["on", "off"], 
                         "default": "on", "help": "Streaming mode"}
                    ]
                ),
                CommandDefinition(
                    name="context",
                    description="Set context trimming percentage",
                    handler=self._cmd_context,
                    category="session",
                    examples=["context 50", "context off"],
                    parameters=[
                        {"name": "percentage", "type": "str", "help": "Context percentage (0-100 or 'off')"}
                    ]
                ),
                CommandDefinition(
                    name="new",
                    description="Start new conversation session",
                    handler=self._cmd_new_session,
                    category="session",
                    examples=["new"],
                    parameters=[]
                ),
                CommandDefinition(
                    name="save",
                    description="Save current session configuration",
                    handler=self._cmd_save_config,
                    category="session",
                    examples=["save"],
                    parameters=[]
                ),
                CommandDefinition(
                    name="models",
                    description="List and select available models",
                    handler=self._cmd_models,
                    category="session",
                    examples=["models", "models openai"],
                    parameters=[
                        {"name": "provider", "type": "str", "help": "Provider to list models for"}
                    ]
                ),
        ]

        # Large prompt editor
        edit_cmd = CommandDefinition(
            name="edit",
            description="Open a multiline prompt editor and optionally execute",
            handler=self._cmd_edit,
            category="session",
            examples=["edit", "edit --execute"],
            parameters=[
                {"name": "execute", "type": "bool", "default": False, "help": "Execute immediately after editing"}
            ]
        )
        self.register_command(edit_cmd)

        for cmd in session_commands:
            self.register_command(cmd)
    
    def register_command(self, command: CommandDefinition):
        """Register a new command"""
        self.commands[command.name] = command
        
        # Register aliases
        for alias in command.aliases:
            self.aliases[alias] = command.name
        
        # Add to category
        if command.category not in self.categories:
            self.categories[command.category] = []
        if command.name not in self.categories[command.category]:
            self.categories[command.category].append(command.name)
        
        logger.debug(f"Registered command: {command.name} (category: {command.category})")
    
    def _setup_readline(self):
        """Setup readline for command completion and history"""
        if not self.config.auto_complete:
            return
        
        try:
            # Set up completion
            readline.set_completer(self._complete_command)
            readline.parse_and_bind("tab: complete")
            
            # Set up history
            readline.set_history_length(self.config.command_history_size)
            
            # Load history if it exists
            history_file = os.path.expanduser("~/.agentsmcp_history")
            if os.path.exists(history_file):
                readline.read_history_file(history_file)
            
        except Exception as e:
            logger.warning(f"Failed to setup readline: {e}")
    
    def _complete_command(self, text: str, state: int) -> Optional[str]:
        """Command completion function for readline"""
        if state == 0:
            # First call - generate completion list
            line = readline.get_line_buffer()
            parts = shlex.split(line) if line else []
            
            if not parts or (len(parts) == 1 and not line.endswith(' ')):
                # Completing command name
                commands = list(self.commands.keys()) + list(self.aliases.keys())
                self.last_suggestions = [cmd for cmd in commands if cmd.startswith(text)]
            else:
                # Completing command parameters
                command_name = parts[0]
                if command_name in self.aliases:
                    command_name = self.aliases[command_name]
                
                if command_name in self.commands:
                    self.last_suggestions = self._get_parameter_suggestions(
                        command_name, parts[1:], text
                    )
                else:
                    self.last_suggestions = []
        
        # Return next suggestion
        if state < len(self.last_suggestions):
            return self.last_suggestions[state]
        return None
    
    def _get_parameter_suggestions(self, command_name: str, current_params: List[str], 
                                 current_text: str) -> List[str]:
        """Get parameter suggestions for a command"""
        command = self.commands[command_name]
        suggestions = []
        
        # Add parameter names that start with current text
        for param in command.parameters:
            param_name = f"--{param['name']}"
            if param_name.startswith(current_text):
                suggestions.append(param_name)
        
        # Add choice values if current parameter has choices
        if current_params and current_params[-1].startswith('--'):
            param_name = current_params[-1][2:]  # Remove --
            for param in command.parameters:
                if param['name'] == param_name and 'choices' in param:
                    for choice in param['choices']:
                        if choice.startswith(current_text):
                            suggestions.append(choice)
        
        return suggestions
    
    async def start_interactive_mode(self):
        """Start interactive command interface"""
        logger.info("🚀 Starting Revolutionary Command Interface")
        
        self.is_running = True
        
        # Show welcome message
        await self._show_welcome()
        
        try:
            while self.is_running:
                try:
                    # Get command input with conversational prompt and autocomplete
                    prompt = self._generate_conversational_prompt()
                    command_line = self._get_input_with_autocomplete(prompt).strip()
                    
                    # Show tab completion hint if user starts typing "/"
                    if command_line == "/":
                        print(self.theme_manager.colorize("💡 Press TAB to see all available commands", 'info'))
                        continue
                    
                    if not command_line:
                        continue
                    
                    # Process command (conversational or direct)
                    start_time = datetime.now()
                    success, result = await self._process_input(command_line)
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    # Add to history
                    history_entry = CommandHistory(
                        command=command_line,
                        timestamp=start_time,
                        success=success,
                        execution_time=execution_time,
                        result_preview=str(result)[:100] if result else ""
                    )
                    self.command_history.append(history_entry)
                    
                    # Trim history if too long
                    if len(self.command_history) > self.config.command_history_size:
                        self.command_history.pop(0)
                    
                except KeyboardInterrupt:
                    print("\n" + self.theme_manager.colorize("Use 'exit' to quit gracefully.", 'text_muted'))
                    continue
                except EOFError:
                    # Ctrl+D pressed
                    break
                    
        except Exception as e:
            logger.error(f"Interactive mode error: {e}")
            print(self.theme_manager.colorize(f"❌ Interface error: {e}", 'error'))
        
        finally:
            await self._cleanup_interface()
        
        logger.info("👋 Command interface session ended")
    
    def _create_web_interface_info(self) -> str:
        """Create web interface info card"""
        web_url = "http://localhost:8000"
        api_url = f"{web_url}/docs"
        health_url = f"{web_url}/health"
        jobs_url = f"{web_url}/jobs"
        
        web_content = f"""🌐 Web API: {web_url}
📋 API Docs: {api_url}
💚 Health Check: {health_url}
🔄 Jobs API: {jobs_url}

Access the REST API for programmatic agent spawning,
job monitoring, and system health checks."""
        
        return self.ui.box(
            web_content,
            title="🚀 Web API Available",
            style='rounded',
            width=min(self.ui.terminal_width - 4, 90)
        )

    async def _show_welcome(self):
        """Show a simplified welcome message and system status"""
        # Create a简洁 welcome message
        title = "🎼 AgentsMCP CLI"
        
        welcome_content = [
            "Welcome to AgentsMCP.",
            "",
            "Commands start with '/' (e.g., /help, /agents, /execute)",
            "Other input is treated as conversation with AI agents.",
            "",
            f"Theme: {self.theme_manager.get_current_theme().name}",
            f"LLM: ollama-turbo (conversational mode enabled)",
        ]
        
        welcome_text = '\n'.join(welcome_content)
        welcome_card = self.ui.card(title, welcome_text, status="success")
        
        print(welcome_card)
        print()  # Extra spacing
    
    def _generate_prompt(self) -> str:
        """Generate command prompt with styling"""
        if self.config.prompt_style == "agentsmcp":
            # Main AgentsMCP styled prompt
            prompt_symbol = "🎼"
            prompt_text = "agentsmcp"
        else:
            prompt_symbol = ">"
            prompt_text = "cli"
        
        # Style components
        symbol_styled = self.theme_manager.colorize(prompt_symbol, 'primary')
        text_styled = self.theme_manager.colorize(prompt_text, 'secondary')
        
        return f"{symbol_styled} {text_styled} ▶ "
    
    def _generate_conversational_prompt(self) -> str:
        """Generate a clean, standard CLI prompt"""
        if self.config.prompt_style == "agentsmcp":
            # Simple, clean prompt
            prompt_symbol = "🎼"
            prompt_text = "agentsmcp"
            
            # Styled components
            symbol_styled = self.theme_manager.colorize(prompt_symbol, 'primary')
            text_styled = self.theme_manager.colorize(prompt_text, 'secondary')
            
            # Simple prompt without box styling
            return f"{symbol_styled} {text_styled} ▶ "
        else:
            return self._generate_prompt()

    def _get_input_with_autocomplete(self, prompt: str) -> str:
        """Get input with command autocomplete support"""
        try:
            import readline
            
            # Set up command completion
            def complete_command(text, state):
                """Tab completion function for commands"""
                # Only complete if text starts with '/'
                if not text.startswith('/'):
                    return None
                
                # Remove '/' for matching
                command_text = text[1:].lower()
                
                # Get all command names
                command_names = sorted([f"/{cmd}" for cmd in self.commands.keys()])
                
                # Filter commands that start with the text
                if command_text == "":
                    # If just "/", show all commands
                    matches = command_names
                else:
                    matches = [cmd for cmd in command_names if cmd[1:].lower().startswith(command_text)]
                
                if state < len(matches):
                    return matches[state]
                return None
            
            # Set up readline
            readline.set_completer(complete_command)
            readline.parse_and_bind("tab: complete")
            
            # Get input
            return input(prompt)
            
        except ImportError:
            # Fallback to regular input if readline not available
            return input(prompt)
        except Exception:
            # Fallback on any error
            return input(prompt)
    
    def _show_prompt_context(self) -> str:
        """Show context above the prompt (optional enhancement for future)"""
        context_items = []
        
        # Add conversation context if available
        if hasattr(self.conversation_manager, 'get_conversation_context'):
            context = self.conversation_manager.get_conversation_context()
            if context.get('conversation_length', 0) > 0:
                context_items.append(f"💭 {context['conversation_length']} messages")
        
        # Add system status
        if hasattr(self.orchestration_manager, 'is_running'):
            status = "🟢 Ready" if self.orchestration_manager.is_running else "🔵 Standby"
            context_items.append(status)
        
        if context_items:
            context_bar = " | ".join(context_items)
            return self.theme_manager.colorize(f"[{context_bar}]", 'text_muted')
        
        return ""
    
    def _get_daily_wisdom(self) -> str:
        """Get daily wisdom message that persists until user prompt"""
        wisdoms = [
            "Great orchestration begins with understanding each agent's strengths.",
            "The symphony of AI agents creates harmony through purposeful collaboration.",
            "Every complex task can be decomposed into elegant, manageable orchestrations.",
            "Trust in your agents, but verify through comprehensive monitoring.",
            "The future belongs to those who can orchestrate intelligence at scale.",
            "Patience and precision in agent management yields extraordinary results.",
            "Innovation happens when diverse AI specialists work in perfect harmony."
        ]
        
        # Use date-based selection for consistent daily wisdom
        import hashlib
        from datetime import date
        
        today = date.today().isoformat()
        hash_input = f"{today}-agentsmcp-wisdom"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        wisdom_index = int(hash_value, 16) % len(wisdoms)
        
        return wisdoms[wisdom_index]
    
    async def _process_input(self, user_input: str) -> Tuple[bool, Any]:
        """Process user input through conversational or command interface"""
        try:
            # First check if it's a direct command
            if self._is_direct_command(user_input):
                return await self._process_command(user_input)
            
            # Otherwise, process through conversational interface
            if self.conversational_mode:
                response = await self.conversation_manager.process_input(user_input)
                print()
                # Add clear visual separation for conversational responses
                print(self.theme_manager.colorize("🤖 AgentsMCP Response:", 'primary'))
                print("-" * 50)
                print(response)
                print("-" * 50)
                print()
                return True, response
            else:
                # Fallback to direct command processing
                return await self._process_command(user_input)
                
        except Exception as e:
            error_msg = f"Input processing error: {e}"
            print(self.theme_manager.colorize(f"❌ {error_msg}", 'error'))
            return False, error_msg

    def _estimate_tokens(self, text: str) -> int:
        """Lightweight token estimate (~chars/4)."""
        try:
            return max(1, int(len(text) / 4))
        except Exception:
            return len(text)

    async def _cmd_edit(self, execute: bool = False) -> str:
        """Open a multiline editor for large prompts and optionally execute it."""
        import tempfile
        from pathlib import Path

        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        tmp = None
        content = ""
        try:
            if editor:
                tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt")
                tmp_path = Path(tmp.name)
                tmp.close()
                # Open editor
                os.system(f"{editor} {tmp_path}")
                content = tmp_path.read_text(encoding="utf-8", errors="ignore")
            else:
                # Fallback: simple here-doc style input
                print(self.theme_manager.colorize("Enter your prompt. Finish with Ctrl-D (EOF) on a new line.", 'info'))
                lines: List[str] = []
                try:
                    while True:
                        line = sys.stdin.readline()
                        if not line:
                            break
                        lines.append(line)
                except KeyboardInterrupt:
                    pass
                content = "".join(lines)
        finally:
            if tmp:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        content = content.strip()
        if not content:
            return self.ui.box("No content captured.", title="Edit Prompt", style='warning')

        # Show token estimate and length
        tokens = self._estimate_tokens(content)
        summary = f"Captured {len(content):,} chars (~{tokens:,} tokens)."

        # Execute now?
        if execute:
            resp = await self.conversation_manager.process_input(content)
            return self.ui.box(
                f"{summary}\n\nResponse:\n{resp}", title="Edit & Execute", style='success'
            )
        else:
            # Stash into context; next input can be "/send" or paste manually
            self.current_context['last_edited_prompt'] = content
            return self.ui.box(
                f"{summary}\nStored in context as last_edited_prompt. Use: /execute '<short instruction>' or paste to run.",
                title="Edit Prompt", style='info'
            )
    
    def _is_direct_command(self, user_input: str) -> bool:
        """Check if input is a direct command (starts with /)"""
        try:
            # Commands must start with / to be treated as direct commands
            if not user_input.startswith('/'):
                return False
            
            # Remove the / prefix for command processing
            command_without_prefix = user_input[1:].strip()
            if not command_without_prefix:
                return False
            
            parts = command_without_prefix.split()
            if not parts:
                return False
            
            command_name = parts[0]
            
            # Resolve aliases
            if command_name in self.aliases:
                command_name = self.aliases[command_name]
            
            # Check if this is a known command
            return command_name in self.commands
            
        except:
            return False
    
    async def handle_command(self, command: str) -> str:
        """Handle command execution from conversational interface"""
        try:
            success, result = await self._process_command(command)
            return str(result) if result else "Command executed successfully"
        except Exception as e:
            return f"Command execution failed: {e}"
    
    async def _process_command(self, command_line: str) -> Tuple[bool, Any]:
        """Process a command line input"""
        try:
            # Remove / prefix if present (already validated in _is_direct_command)
            if command_line.startswith('/'):
                command_line = command_line[1:].strip()
            
            # Parse command
            parts = shlex.split(command_line)
            if not parts:
                return True, None
            
            command_name = parts[0]
            args = parts[1:]
            
            # Resolve aliases
            if command_name in self.aliases:
                command_name = self.aliases[command_name]
            
            # Check if command exists
            if command_name not in self.commands:
                error_msg = f"Unknown command: {command_name}"
                print(self.theme_manager.colorize(f"❌ {error_msg}", 'error'))
                
                # Suggest similar commands
                suggestions = self._get_command_suggestions(command_name)
                if suggestions:
                    suggestions_text = ", ".join(suggestions)
                    print(self.theme_manager.colorize(f"💡 Did you mean: {suggestions_text}?", 'info'))
                
                return False, error_msg
            
            # Get command definition
            command = self.commands[command_name]
            
            # Parse parameters
            params = self._parse_command_parameters(command, args)
            
            # Execute command
            result = await command.handler(**params)
            
            return True, result
            
        except Exception as e:
            error_msg = f"Command execution error: {e}"
            print(self.theme_manager.colorize(f"❌ {error_msg}", 'error'))
            return False, error_msg
    
    def _parse_command_parameters(self, command: CommandDefinition, 
                                args: List[str]) -> Dict[str, Any]:
        """Parse command line arguments into parameters"""
        params = {}
        assigned_params = set()  # Track which params have been explicitly set
        i = 0
        
        # Set default values
        for param in command.parameters:
            if 'default' in param:
                params[param['name']] = param['default']
        
        # Parse arguments
        while i < len(args):
            arg = args[i]
            
            if arg.startswith('--'):
                # Named parameter
                param_name = arg[2:]
                
                # Find parameter definition
                param_def = None
                for p in command.parameters:
                    if p['name'] == param_name:
                        param_def = p
                        break
                
                if not param_def:
                    raise ValueError(f"Unknown parameter: --{param_name}")
                
                # Get parameter value
                if param_def.get('type') == 'bool':
                    params[param_name] = True
                    assigned_params.add(param_name)
                else:
                    if i + 1 >= len(args):
                        raise ValueError(f"Parameter --{param_name} requires a value")
                    
                    value = args[i + 1]
                    params[param_name] = self._convert_parameter_value(value, param_def)
                    assigned_params.add(param_name)
                    i += 1
                
            else:
                # Positional parameter
                # Find first parameter that hasn't been explicitly assigned
                assigned = False
                for param in command.parameters:
                    if param['name'] not in assigned_params:
                        params[param['name']] = self._convert_parameter_value(arg, param)
                        assigned_params.add(param['name'])
                        assigned = True
                        break
                
                if not assigned:
                    # No more parameters to assign, treat as extra positional args
                    if 'args' not in params:
                        params['args'] = []
                    params['args'].append(arg)
            
            i += 1
        
        # Check required parameters
        for param in command.parameters:
            if param.get('required') and param['name'] not in params:
                raise ValueError(f"Required parameter missing: --{param['name']}")
        
        return params
    
    def _convert_parameter_value(self, value: str, param_def: Dict[str, Any]) -> Any:
        """Convert string parameter value to appropriate type"""
        param_type = param_def.get('type', 'str')
        
        if param_type == 'str':
            return value
        elif param_type == 'int':
            return int(value)
        elif param_type == 'float':
            return float(value)
        elif param_type == 'bool':
            return value.lower() in ('true', '1', 'yes', 'on')
        elif param_type == 'json':
            return json.loads(value)
        elif param_type == 'list':
            return json.loads(value) if value.startswith('[') else value.split(',')
        else:
            return value
    
    def _get_command_suggestions(self, command_name: str) -> List[str]:
        """Get command suggestions for typos"""
        suggestions = []
        
        # Simple similarity matching
        for cmd_name in self.commands.keys():
            if self._similarity_score(command_name, cmd_name) > 0.6:
                suggestions.append(cmd_name)
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _similarity_score(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings"""
        if not s1 or not s2:
            return 0.0
        
        # Simple Levenshtein distance-based similarity
        max_len = max(len(s1), len(s2))
        distance = self._levenshtein_distance(s1, s2)
        return 1.0 - (distance / max_len)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        
        return prev_row[-1]
    
    async def _get_quick_status(self) -> str:
        """Get quick status summary"""
        try:
            status = await self.orchestration_manager.get_system_status()
            
            system_status = status.get("system_status", "unknown")
            mode = status.get("orchestration_mode", "unknown")
            uptime = status.get("uptime", "0:00:00")
            
            status_items = [
                self.ui.status_indicator("success" if system_status == "running" else "error", 
                                       f"System: {system_status.title()}"),
                self.ui.metric_display("Mode", mode.title()),
                self.ui.metric_display("Uptime", str(uptime))
            ]
            
            return " | ".join(status_items)
            
        except Exception as e:
            return self.theme_manager.colorize(f"Status unavailable: {e}", 'warning')
    
    # Command handlers
    async def _cmd_status(self, detailed: bool = False) -> str:
        """Handle status command"""
        try:
            status = await self.orchestration_manager.get_system_status()
            
            if detailed:
                # Show detailed status
                status_json = json.dumps(status, indent=2, default=str)
                return self.ui.panel(status_json, "Detailed System Status", "info")
            else:
                # Show summary status
                summary_items = [
                    ("System Status", status.get("system_status", "unknown").title()),
                    ("Session ID", status.get("session_id", "N/A")[:8] if status.get("session_id") else "N/A"),
                    ("Orchestration Mode", status.get("orchestration_mode", "unknown").title()),
                    ("Uptime", str(status.get("uptime", "0:00:00"))),
                    ("Total Tasks", str(status.get("performance_metrics", {}).get("total_tasks_completed", 0))),
                ]
                
                summary_content = '\n'.join([
                    self.ui.metric_display(label, value) for label, value in summary_items
                ])
                
                result = self.ui.card("📊 System Status", summary_content, "success")
                print(result)
                return "Status displayed"
                
        except Exception as e:
            error_msg = f"Failed to get system status: {e}"
            print(self.theme_manager.colorize(f"❌ {error_msg}", 'error'))
            return error_msg
    
    async def _cmd_dashboard(self, refresh: float = 1.0, compact: bool = False) -> str:
        """Handle dashboard command"""
        try:
            from .status_dashboard import StatusDashboard, DashboardConfig
            
            # For interactive testing, disable auto_refresh to prevent infinite loops
            config = DashboardConfig(
                refresh_interval=refresh,
                compact_mode=compact,
                auto_refresh=False  # Fix: Disable auto-refresh to prevent timeouts
            )
            
            dashboard = StatusDashboard(self.orchestration_manager, self.theme_manager, config)
            
            print(self.theme_manager.colorize("📊 Displaying dashboard snapshot...", 'info'))
            print()
            
            # Show a single dashboard snapshot instead of entering infinite loop
            await dashboard._update_dashboard()
            
            return "Dashboard snapshot displayed successfully"
            
        except Exception as e:
            error_msg = f"Failed to display dashboard: {e}"
            print(self.theme_manager.colorize(f"❌ {error_msg}", 'error'))
            return error_msg
    
    async def _cmd_execute(self, task: str, mode: str = "hybrid", 
                         context: Optional[Dict[str, Any]] = None) -> str:
        """Handle execute command"""
        try:
            print(self.ui.loading_spinner(f"Executing task in {mode} mode"))
            print()
            
            # Initialize orchestration manager if needed
            if not hasattr(self.orchestration_manager, 'is_running') or not self.orchestration_manager.is_running:
                print(self.theme_manager.colorize("🔧 Initializing orchestration system...", 'info'))
                await self.orchestration_manager.initialize(mode)
                print()
            
            # Execute task
            result = await self.orchestration_manager.execute_task(task, context)
            
            # Display result
            task_id = result.get("task_id", "unknown")
            execution_strategy = result.get("execution_strategy", "unknown")
            completion_time = result.get("completion_time", "unknown")
            
            result_summary = [
                f"Task ID: {task_id}",
                f"Strategy: {execution_strategy.title()}",
                f"Completion Time: {completion_time}",
                f"Status: {'✅ Success' if not result.get('error') else '❌ Failed'}"
            ]
            
            if result.get('error'):
                result_summary.append(f"Error: {result['error']}")
            
            result_content = '\n'.join(result_summary)
            result_card = self.ui.card("🎯 Task Execution Result", result_content, 
                                     "success" if not result.get('error') else "error")
            
            print(result_card)
            
            return f"Task executed: {task_id}"
            
        except Exception as e:
            error_msg = f"Task execution failed: {e}"
            print(self.theme_manager.colorize(f"❌ {error_msg}", 'error'))
            return error_msg
    
    async def _cmd_agents(self, action: str, specialization: Optional[str] = None, 
                        agent_id: Optional[str] = None) -> str:
        """Handle agents command"""
        try:
            if action == "list":
                status = await self.orchestration_manager.get_system_status()
                component_status = status.get("component_status", {})
                
                # Extract agent information
                agents_info = []
                
                # Symphony mode agents
                symphony_status = component_status.get("symphony_mode", {})
                if isinstance(symphony_status, dict) and "agent_status" in symphony_status:
                    for agent_id, agent_info in symphony_status["agent_status"].items():
                        agents_info.append([
                            agent_id,
                            agent_info.get("specialization", "unknown"),
                            f"{agent_info.get('performance', 0.0):.2f}",
                            f"{agent_info.get('load', 0.0):.2f}",
                            "symphony"
                        ])
                
                if agents_info:
                    headers = ["Agent ID", "Specialization", "Performance", "Load", "Source"]
                    table = self.ui.table(headers, agents_info, "Active Agents")
                    print(table)
                else:
                    print(self.theme_manager.colorize("No active agents found", 'text_muted'))
                
                return f"Listed {len(agents_info)} agents"
                
            elif action == "spawn":
                if not specialization:
                    raise ValueError("Specialization required for spawn action")
                
                # This would integrate with the predictive spawner
                print(self.theme_manager.colorize(f"🚀 Spawning {specialization} agent...", 'info'))
                
                # Simulate spawn request
                result = {
                    "request_id": f"spawn_{datetime.now().strftime('%H%M%S')}",
                    "specialization": specialization,
                    "estimated_spawn_time": "30 seconds"
                }
                
                spawn_content = f"Spawn request submitted:\nRequest ID: {result['request_id']}\nSpecialization: {specialization}\nETA: {result['estimated_spawn_time']}"
                spawn_card = self.ui.card("🤖 Agent Spawn Request", spawn_content, "info")
                print(spawn_card)
                
                return f"Spawn requested: {specialization}"
                
            else:
                return f"Agent action '{action}' not yet implemented"
                
        except Exception as e:
            error_msg = f"Agents command failed: {e}"
            print(self.theme_manager.colorize(f"❌ {error_msg}", 'error'))
            return error_msg
    
    async def _cmd_symphony(self, action: str, tasks: Optional[List[str]] = None) -> str:
        """Handle symphony command"""
        try:
            if action == "start":
                if not tasks:
                    tasks = ["Example task for symphony demonstration"]
                
                print(self.theme_manager.colorize("🎼 Starting Symphony mode orchestration...", 'info'))
                
                # This would integrate with symphony mode
                symphony_info = f"Symphony started with {len(tasks)} tasks"
                symphony_card = self.ui.card("🎼 Symphony Mode", symphony_info, "success")
                print(symphony_card)
                
                return f"Symphony started with {len(tasks)} tasks"
                
            elif action == "status":
                # Get symphony status
                status_info = "Symphony mode status would be displayed here"
                status_card = self.ui.card("🎼 Symphony Status", status_info, "info")
                print(status_card)
                
                return "Symphony status displayed"
                
            else:
                return f"Symphony action '{action}' not yet implemented"
                
        except Exception as e:
            error_msg = f"Symphony command failed: {e}"
            print(self.theme_manager.colorize(f"❌ {error_msg}", 'error'))
            return error_msg
    
    async def _cmd_theme(self, mode: str = "info") -> str:
        """Handle theme command"""
        if mode == "info":
            theme_info = self.theme_manager.get_theme_info()
            
            info_content = [
                f"Current Theme: {theme_info['name']}",
                f"Theme Type: {theme_info['type'].title()}",
                f"Detection Method: {theme_info['detection_method'].title()}",
                f"Available Themes: {', '.join(theme_info['available_themes'])}",
                f"Colorama Available: {'✅' if theme_info['colorama_available'] else '❌'}"
            ]
            
            info_text = '\n'.join(info_content)
            info_card = self.ui.card("🎨 Theme Information", info_text, "info")
            print(info_card)
            
        elif mode in ["dark", "light"]:
            old_theme = self.theme_manager.get_current_theme().name
            self.theme_manager.set_theme(mode)
            new_theme = self.theme_manager.get_current_theme().name
            
            change_msg = f"Theme changed from '{old_theme}' to '{new_theme}'"
            print(self.theme_manager.colorize(f"🎨 {change_msg}", 'success'))
            
        elif mode == "auto":
            old_theme = self.theme_manager.get_current_theme().name
            self.theme_manager.auto_detect_theme()
            new_theme = self.theme_manager.get_current_theme().name
            
            change_msg = f"Theme auto-detected: '{new_theme}' (was '{old_theme}')"
            print(self.theme_manager.colorize(f"🎨 {change_msg}", 'success'))
        
        return f"Theme operation: {mode}"
    
    async def _cmd_history(self, filter: Optional[str] = None, count: int = 20) -> str:
        """Handle history command"""
        history_items = self.command_history[-count:] if count > 0 else self.command_history
        
        if filter:
            history_items = [h for h in history_items if filter.lower() in h.command.lower()]
        
        if not history_items:
            print(self.theme_manager.colorize("No matching history items found", 'text_muted'))
            return "No history items"
        
        # Build history table
        history_data = []
        for i, item in enumerate(reversed(history_items), 1):
            status_icon = "✅" if item.success else "❌"
            history_data.append([
                str(i),
                item.command,
                item.timestamp.strftime("%H:%M:%S"),
                f"{item.execution_time:.2f}s",
                status_icon
            ])
        
        headers = ["#", "Command", "Time", "Duration", "Status"]
        history_table = self.ui.table(headers, history_data, "Command History")
        print(history_table)
        
        return f"Displayed {len(history_items)} history items"
    
    async def _cmd_help(self, command: Optional[str] = None, 
                      category: Optional[str] = None) -> str:
        """Handle help command"""
        if command:
            # Show help for specific command
            if command in self.aliases:
                command = self.aliases[command]
            
            if command not in self.commands:
                print(self.theme_manager.colorize(f"❌ Unknown command: {command}", 'error'))
                return f"Unknown command: {command}"
            
            cmd = self.commands[command]
            
            # Build detailed help
            help_content = [
                f"Description: {cmd.description}",
                f"Category: {cmd.category.title()}"
            ]
            
            if cmd.aliases:
                help_content.append(f"Aliases: {', '.join(cmd.aliases)}")
            
            if cmd.parameters:
                help_content.append("\nParameters:")
                for param in cmd.parameters:
                    param_info = f"  --{param['name']}"
                    if param.get('required'):
                        param_info += " (required)"
                    if 'default' in param:
                        param_info += f" [default: {param['default']}]"
                    param_info += f": {param.get('help', 'No description')}"
                    help_content.append(param_info)
            
            if cmd.examples:
                help_content.append("\nExamples:")
                for example in cmd.examples:
                    help_content.append(f"  {example}")
            
            help_text = '\n'.join(help_content)
            help_card = self.ui.card(f"📚 Help: {command}", help_text, "info")
            print(help_card)
            
        elif category:
            # Show commands in category
            if category not in self.categories:
                available_cats = ', '.join(self.categories.keys())
                print(self.theme_manager.colorize(f"❌ Unknown category: {category}", 'error'))
                print(self.theme_manager.colorize(f"Available categories: {available_cats}", 'info'))
                return f"Unknown category: {category}"
            
            commands_in_cat = self.categories[category]
            
            cat_content = []
            for cmd_name in commands_in_cat:
                cmd = self.commands[cmd_name]
                if not cmd.hidden:
                    cat_content.append(f"  {cmd_name}: {cmd.description}")
            
            cat_text = '\n'.join(cat_content)
            cat_card = self.ui.card(f"📚 Commands in '{category.title()}'", cat_text, "info")
            print(cat_card)
            
        else:
            # Show simplified general help
            help_sections = []
            
            for cat_name, cmd_names in self.categories.items():
                visible_commands = [
                    cmd_name for cmd_name in cmd_names 
                    if not self.commands[cmd_name].hidden
                ]
                
                if visible_commands:
                    section_content = []
                    # Show all commands in each category
                    for cmd_name in visible_commands:
                        cmd = self.commands[cmd_name]
                        section_content.append(f"  {cmd_name:<15} - {cmd.description}")
                    
                    help_sections.append(f"\n{cat_name.title()}:\n" + '\n'.join(section_content))
            
            general_help = [
                "Available command categories:",
                *help_sections,
                "\nUse 'help <command>' for detailed help on a specific command.",
                "Use 'help --category <category>' to see all commands in a category."
            ]
            
            help_text = '\n'.join(general_help)
            help_card = self.ui.card("📚 AgentsMCP Command Help", help_text, "info")
            print(help_card)
        
        return "Help displayed"
    
    async def _cmd_config(self, action: str = "show", key: Optional[str] = None, 
                        value: Optional[str] = None) -> str:
        """Handle config command"""
        if action == "show":
            config_items = [
                ("Prompt Style", self.config.prompt_style),
                ("Auto Complete", str(self.config.auto_complete)),
                ("Show Suggestions", str(self.config.show_suggestions)),
                ("Interactive Mode", str(self.config.interactive_mode)),
                ("Command History Size", str(self.config.command_history_size)),
                ("Enable Colors", str(self.config.enable_colors))
            ]
            
            config_content = '\n'.join([
                self.ui.metric_display(label, val) for label, val in config_items
            ])
            
            config_card = self.ui.card("⚙️ Interface Configuration", config_content, "info")
            print(config_card)
            
        elif action == "set":
            if not key or value is None:
                raise ValueError("Both key and value required for set action")
            
            if hasattr(self.config, key):
                # Convert value to appropriate type
                current_val = getattr(self.config, key)
                if isinstance(current_val, bool):
                    new_val = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_val, int):
                    new_val = int(value)
                elif isinstance(current_val, float):
                    new_val = float(value)
                else:
                    new_val = value
                
                setattr(self.config, key, new_val)
                print(self.theme_manager.colorize(f"✅ Set {key} = {new_val}", 'success'))
            else:
                raise ValueError(f"Unknown config key: {key}")
        
        return f"Config {action} completed"
    
    async def _cmd_clear(self) -> str:
        """Handle clear command"""
        self.ui.clear_screen()
        return "Screen cleared"

    async def _cmd_analyze(self) -> str:
        """Analyze the current repository (direct execution)."""
        try:
            if hasattr(self, 'conversation_manager') and hasattr(self.conversation_manager, '_analyze_repository_directly'):
                result = self.conversation_manager._analyze_repository_directly()
                return self.ui.box(result, title="📦 Repository Analysis", style='light')
            else:
                return "❌ Analyze not available in this mode"
        except Exception as e:
            return f"❌ Analyze failed: {e}"

    async def _cmd_provider_use(self, provider: str = None) -> str:
        """Set primary provider for session and persist to user settings."""
        valid = ["ollama-turbo", "openai", "openrouter", "anthropic", "ollama", "codex"]
        if not provider or provider.lower() not in valid:
            current_provider = "unknown"
            try:
                if hasattr(self, 'conversation_manager') and hasattr(self.conversation_manager, 'llm_client'):
                    current_provider = getattr(self.conversation_manager.llm_client, 'provider', 'unknown')
            except Exception:
                pass
            body = (
                f"Usage: /provider-use <provider>\n"
                f"Valid: {', '.join(valid)}\n"
                f"Current: {current_provider}"
            )
            return self.ui.box(body, title="Provider Use", style='info')
        prov = provider.lower()
        try:
            # Update runtime LLM client
            if hasattr(self, 'conversation_manager') and hasattr(self.conversation_manager, 'llm_client'):
                self.conversation_manager.llm_client.provider = prov
                logger.info(f"Updated LLM client provider to: {prov}")
            
            # Persist to user settings via orchestration manager  
            if hasattr(self, 'orchestration_manager'):
                if hasattr(self.orchestration_manager, 'save_user_settings'):
                    self.orchestration_manager.save_user_settings({"provider": prov})
                    logger.info(f"Persisted provider setting: {prov}")
                elif hasattr(self.orchestration_manager, 'user_settings'):
                    # Fallback: update in-memory settings
                    if not self.orchestration_manager.user_settings:
                        self.orchestration_manager.user_settings = {}
                    self.orchestration_manager.user_settings["provider"] = prov
                    logger.info(f"Updated in-memory provider setting: {prov}")
            
            return self.ui.box(f"✅ Primary provider set to: {prov}", title="Provider Updated", style='success')
        except Exception as e:
            logger.error(f"Failed to set provider {prov}: {e}")
            return self.ui.box(f"❌ Failed to set provider: {e}", title="Error", style='error')
    
    async def _cmd_settings(self) -> str:
        """Handle settings command with async-safe execution"""
        if not run_modern_settings_dialog:
            print(self.theme_manager.colorize("❌ Settings UI not available", 'error'))
            return "Settings UI not available"
        
        try:
            print(self.ui.clear_screen())
            
            # Use ThreadPoolExecutor to avoid asyncio.run() conflicts
            import concurrent.futures
            import asyncio
            
            def run_settings_sync():
                """Synchronous wrapper for settings dialog"""
                return run_modern_settings_dialog(self.theme_manager, self.ui)
            
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, use ThreadPoolExecutor
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_settings_sync)
                    success = future.result(timeout=300)  # 5 minute timeout
            except RuntimeError:
                # No running loop, safe to call directly
                success = run_modern_settings_dialog(self.theme_manager, self.ui)
            
            print(self.ui.clear_screen())
            
            if success:
                print(self.theme_manager.colorize("✅ Settings updated successfully", 'success'))
                return "Settings updated successfully"
            else:
                # Don't show "cancelled" message as it's confusing for users just exploring
                print(self.theme_manager.colorize("ℹ️ Settings menu closed", 'info'))
                return "Settings menu accessed"
        except Exception as e:
            error_msg = f"Settings command failed: {e}"
            print(self.theme_manager.colorize(f"❌ {error_msg}", 'error'))
            logger.error(f"Settings command error: {e}")
            return error_msg

    async def _cmd_keys(self) -> str:
        """Display API key status across providers.

        Logic: prefer user config (~/.agentsmcp/config.json), fallback to env var.
        Nothing is printed in clear; only masked status + source.
        """
        import os
        try:
            cfg = {}
            if hasattr(self, 'orchestration_manager') and self.orchestration_manager:
                cfg = getattr(self.orchestration_manager, 'user_settings', {}) or {}
            api_keys = (cfg or {}).get('api_keys', {})
        except Exception as e:
            logger.warning(f"Could not load user settings: {e}")
            api_keys = {}

        def mask(val: str) -> str:
            if not val:
                return 'Not set'
            tail = val[-4:] if len(val) >= 4 else val
            return f"••••••••{tail}"

        def status(provider_key: str, env_var: str) -> str:
            # Config (provider-specific) first, then env
            conf_val = api_keys.get(provider_key, '')
            env_val = os.getenv(env_var, '')
            if conf_val:
                return f"Config: {mask(conf_val)}"
            if env_val:
                return f"Env({env_var}): {mask(env_val)}"
            return "Missing"

        rows = [
            ["Ollama Turbo", status("ollama-turbo", "OLLAMA_API_KEY")],
            ["OpenAI", status("openai", "OPENAI_API_KEY")],
            ["OpenRouter", status("openrouter", "OPENROUTER_API_KEY")],
            ["Anthropic", status("anthropic", "ANTHROPIC_API_KEY")],
            ["GitHub (MCP)", ("Env(GITHUB_TOKEN): " + mask(os.getenv('GITHUB_TOKEN',''))) if os.getenv('GITHUB_TOKEN') else "Missing"],
        ]

        # Build a simple table-like content
        header = f"Provider{' ' * 18}Status\n" + ("-" * 60)
        lines = []
        for name, stat in rows:
            pad = max(0, 24 - len(name))
            lines.append(f"{name}{' ' * pad}{stat}")
        content = header + "\n" + "\n".join(lines) + "\n\nUse /settings to set config API key, or export env vars."
        return self.ui.box(content, title="🔑 API Keys", style='light')

    async def _cmd_export_keys(self, include_values: bool = False) -> str:
        """Print ready-to-copy export commands for providers.

        - By default prints placeholders (safer). Pass --include-values to output actual values.
        """
        import os
        cfg = getattr(self.orchestration_manager, 'user_settings', {}) if hasattr(self, 'orchestration_manager') else {}
        api_keys = (cfg or {}).get('api_keys', {})
        lines = ["# Copy and paste into your shell"]

        def emit(provider_key: str, env_var: str, label: str):
            conf_val = api_keys.get(provider_key, '')
            env_val = os.getenv(env_var, '')
            use_val = conf_val or env_val
            if include_values and use_val:
                val = use_val.replace("'", "'\\''")
                lines.append(f"export {env_var}='{val}'  # {label}")
            else:
                src = "config" if conf_val else ("env" if env_val else "missing")
                placeholder = "<paste-your-key>" if not include_values else "<unavailable>"
                comment = f"# {label} ({src})"
                lines.append(f"export {env_var}='{placeholder}'  {comment}")

        emit("ollama-turbo", "OLLAMA_API_KEY", "Ollama Turbo")
        emit("openai", "OPENAI_API_KEY", "OpenAI")
        emit("openrouter", "OPENROUTER_API_KEY", "OpenRouter")
        emit("anthropic", "ANTHROPIC_API_KEY", "Anthropic (Claude)")
        # GitHub MCP doesn't live in config; env only
        if include_values and os.getenv('GITHUB_TOKEN'):
            v = os.getenv('GITHUB_TOKEN').replace("'", "'\\''")
            lines.append(f"export GITHUB_TOKEN='{v}'  # GitHub MCP")
        else:
            lines.append("export GITHUB_TOKEN='<paste-your-token>'  # GitHub MCP (env)")

        script = "\n".join(lines)
        return self.ui.box(script, title="🧰 Export Keys", style='light')

    async def _cmd_provider_order(self) -> str:
        """Display the current provider fallback order and status.

        Order: primary -> openai -> openrouter -> anthropic -> ollama -> codex
        Providers without keys are skipped (except local ollama).
        """
        import os
        # Determine primary provider
        primary = None
        if hasattr(self, 'conversation_manager') and hasattr(self.conversation_manager, 'llm_client'):
            primary = getattr(self.conversation_manager.llm_client, 'provider', None)
        if not primary and hasattr(self, 'orchestration_manager'):
            primary = (getattr(self.orchestration_manager, 'user_settings', {}) or {}).get('provider')
        primary = (primary or 'ollama-turbo').lower()

        order = []
        for p in [primary, 'openai', 'openrouter', 'anthropic', 'ollama', 'codex']:
            if p not in order:
                order.append(p)

        cfg = getattr(self.orchestration_manager, 'user_settings', {}) if hasattr(self, 'orchestration_manager') else {}
        api_keys = (cfg or {}).get('api_keys', {})
        env_map = {
            'ollama-turbo': 'OLLAMA_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
        }

        lines = []
        header = f"Primary: {primary}\n" + ("-" * 60)
        for i, p in enumerate(order, 1):
            if p == 'ollama':
                status = 'will try (local, no key required, timeout ~120s)'
            else:
                conf = api_keys.get(p, '')
                envv = env_map.get(p)
                envp = os.getenv(envv, '') if envv else ''
                if conf:
                    status = f"will try (config)"
                elif envp:
                    status = f"will try (env {envv})"
                else:
                    status = 'skipped (no key)'
            lines.append(f"{i}. {p:12} {status}")

        # Show timeouts if available
        timeouts = {}
        try:
            settings = getattr(self.orchestration_manager, 'user_settings', {}) if hasattr(self, 'orchestration_manager') else {}
            timeouts = (settings or {}).get('timeouts', {})
        except Exception:
            pass
        if timeouts:
            lines.append("")
            lines.append("Timeouts (seconds):")
            for k, v in timeouts.items():
                lines.append(f"  - {k}: {v}")
        lines.append("")
        lines.append("Note: Local Ollama may be slow on first run; timeout is set high by default (~120s). Adjust via ~/.agentsmcp/config.json → timeouts.local_ollama.")
        content = header + "\n" + "\n".join(lines)
        return self.ui.box(content, title="🧭 Provider Fallback Order", style='light')
    
    async def _cmd_generate_config(self) -> str:
        """Handle generate-config command"""
        try:
            print(self.theme_manager.colorize("🔧 Generating MCP client configuration...", 'info'))
            
            # Generate configuration using orchestration manager
            config_text = self.orchestration_manager.generate_client_config()
            
            # Display the configuration in a nice box
            config_card = self.ui.card(
                "🚀 MCP Client Configuration", 
                "Configuration generated successfully! Copy the text below:",
                "success"
            )
            print(config_card)
            print()
            
            # Print the configuration with a border
            print(self.theme_manager.colorize("=" * 80, 'accent'))
            print(config_text)
            print(self.theme_manager.colorize("=" * 80, 'accent'))
            print()
            
            # Show helpful next steps
            next_steps = """Next Steps:
1. Copy the configuration above
2. Save it to your MCP client config file:
   - Claude Desktop: ~/.config/Claude/claude_desktop_config.json
   - Claude Code CLI: ~/.config/claude-code/config.json
3. Set required environment variables (see comments above)
4. Restart your MCP client"""
            
            steps_card = self.ui.card("📋 Next Steps", next_steps, "info")
            print(steps_card)
            
            return "MCP configuration generated successfully"
            
        except Exception as e:
            error_msg = f"Generate config failed: {e}"
            print(self.theme_manager.colorize(f"❌ {error_msg}", 'error'))
            return error_msg

    # Session management command handlers
    async def _cmd_agent_switch(self, agent_type: str = None) -> str:
        """Switch to a different AI agent"""
        if agent_type is None or not agent_type.strip():
            # Show current and available agents with usage hint
            available = list(self.app_config.agents.keys()) if self.app_config else []
            body = (
                f"Current agent: {self.current_agent}\n"
                f"Available: {', '.join(available) if available else 'n/a'}\n\n"
                f"Usage: /agent <agent_type>"
            )
            return self.ui.box(body, title="Agent", style='info')
        if not self.agent_manager:
            return "❌ Agent manager not available. Enable agent orchestration to switch agents."
        
        self.current_agent = agent_type
        
        return self.ui.box(
            f"🤖 Agent switched to: {agent_type.upper()}\n"
            f"   Ready for orchestration and task delegation",
            title="Agent Switch",
            style='success'
        )
    
    async def _cmd_model(self, model_name: str = None) -> str:
        """Set model for current session"""
        try:
            if model_name is None or not model_name.strip():
                # Show current model and prioritized defaults with usage
                if not self.app_config:
                    return self.ui.box("❌ No configuration available", title="Error", style='error')
                agent_cfg = self.app_config.agents.get(self.current_agent)
                current = agent_cfg.model if agent_cfg else 'n/a'
                priority = ", ".join(agent_cfg.model_priority) if agent_cfg and agent_cfg.model_priority else 'n/a'
                body = (
                    f"Current agent: {self.current_agent}\n"
                    f"Current model: {current}\n"
                    f"Priority: {priority}\n\n"
                    f"Usage: /model <model_name>"
                )
                return self.ui.box(body, title="Model Info", style='info')
            
            if not self.app_config or not self.app_config.agents:
                return self.ui.box("❌ No agent configuration available", title="Error", style='error')
            
            agent_config = self.app_config.agents.get(self.current_agent)
            if not agent_config:
                return self.ui.box(f"❌ Agent '{self.current_agent}' not found in configuration", title="Error", style='error')
            
            # Update the agent configuration
            agent_config.model = model_name
            
            # Also update the LLM client if available
            if hasattr(self, 'conversation_manager') and hasattr(self.conversation_manager, 'llm_client'):
                self.conversation_manager.llm_client.model = model_name
                logger.info(f"Updated LLM client model to: {model_name}")
            
            return self.ui.box(
                f"✅ Model set to: {model_name}\n"
                f"   Agent: {self.current_agent}",
                title="Model Updated",
                style='success'
            )
        except Exception as e:
            logger.error(f"Failed to set model {model_name}: {e}")
            return self.ui.box(f"❌ Failed to set model: {e}", title="Error", style='error')
    
    async def _cmd_provider(self, provider_name: str = None) -> str:
        """Set provider for current session"""
        if not provider_name:
            # Show interactive provider selection
            return await self._interactive_provider_selection()
        
        if not self.app_config or not self.app_config.agents:
            return "❌ No agent configuration available"
        
        try:
            provider_type = ProviderType(provider_name)
            agent_config = self.app_config.agents.get(self.current_agent)
            if agent_config:
                agent_config.provider = provider_type
            
            # Validate provider configuration
            if hasattr(self.app_config, 'providers') and provider_name in self.app_config.providers:
                base_cfg = self.app_config.providers[provider_name]
                vcfg = ProviderConfig(
                    name=provider_type,
                    api_key=getattr(base_cfg, 'api_key', None),
                    api_base=getattr(base_cfg, 'api_base', None)
                )
                vres = validate_provider_config(provider_type, vcfg)
                validation_msg = f"\n   Validation: {vres.reason}" if not vres.ok else ""
            else:
                validation_msg = "\n   ⚠️  Provider not fully configured"
            
            return self.ui.box(
                f"🔌 Provider set to: {provider_name}\n"
                f"   Agent: {self.current_agent}{validation_msg}",
                title="Provider Updated",
                style='info'
            )
        except ValueError:
            return f"❌ Invalid provider: {provider_name}. Use: openai, anthropic, ollama, openrouter"
    
    async def _cmd_stream(self, mode: str = "on") -> str:
        """Toggle streaming mode"""
        if mode in ("on", "true", "1"):
            self.stream_enabled = True
            status = "enabled"
        elif mode in ("off", "false", "0"):
            self.stream_enabled = False  
            status = "disabled"
        else:
            # Toggle current state
            self.stream_enabled = not self.stream_enabled
            status = "enabled" if self.stream_enabled else "disabled"
        
        return f"🌊 Streaming {status}"
    
    async def _cmd_context(self, percentage: str) -> str:
        """Set context trimming percentage"""
        if percentage.lower() in ("off", "0"):
            self.context_percent = 0
            return "🧠 Context trimming disabled"
        
        try:
            p = int(percentage)
            if 0 <= p <= 100:
                self.context_percent = p
                return f"🧠 Context set to last ~{p}% of budget"
            else:
                return "❌ Context percentage must be 0-100 or 'off'"
        except ValueError:
            return "❌ Invalid percentage. Use a number 0-100 or 'off'"
    
    async def _cmd_new_session(self) -> str:
        """Start a new conversation session"""
        self.session_history.clear()
        return self.ui.box(
            "🆕 New session started\n"
            "   Conversation history cleared",
            title="New Session",
            style='success'
        )
    
    async def _cmd_save_config(self) -> str:
        """Save current session configuration"""
        if not self.app_config:
            return "❌ No configuration to save"
        
        try:
            from pathlib import Path
            config_path = Path("agentsmcp.yaml")
            self.app_config.save_to_file(config_path)
            return f"💾 Configuration saved to {config_path}"
        except Exception as e:
            return f"❌ Failed to save configuration: {e}"
    
    async def _cmd_models(self, provider: str = None) -> str:
        """List and select available models"""
        if not self.app_config:
            return "❌ No configuration available"
        
        # Determine provider
        if not provider:
            agent_config = self.app_config.agents.get(self.current_agent)
            if agent_config:
                provider = agent_config.provider.value if hasattr(agent_config.provider, 'value') else str(agent_config.provider)
            else:
                provider = "openai"
        
        try:
            provider_type = ProviderType(provider)
            
            # Get provider configuration
            base_cfg = None
            if hasattr(self.app_config, 'providers') and provider in self.app_config.providers:
                base_cfg = self.app_config.providers[provider]
            
            pconfig = ProviderConfig(
                name=provider_type,
                api_key=getattr(base_cfg, 'api_key', None) if base_cfg else None,
                api_base=getattr(base_cfg, 'api_base', None) if base_cfg else None
            )
            
            # List models
            models = providers_list_models(provider_type, pconfig)
            
            if not models:
                return f"❌ No models found for provider: {provider}"
            
            # Create models table
            content = [f"Available Models for {provider}:", ""]
            for i, model in enumerate(models[:10], 1):  # Show first 10
                content.append(f"{i:2}. {model.id}")
                if model.name and model.name != model.id:
                    content[-1] += f" ({model.name})"
            
            
            return self.ui.box(
                "\n".join(content),
                title="Models",
                style='info'
            )
            
        except ProviderError as e:
            return f"❌ Failed to list models: {e}"
        except ValueError:
            return f"❌ Invalid provider: {provider}"
    
    async def _interactive_provider_selection(self) -> str:
        """Interactive provider selection"""
        providers = ["openai", "anthropic", "ollama", "openrouter"]
        
        content = ["Available Providers:", ""]
        for i, provider in enumerate(providers, 1):
            configured = ""
            if hasattr(self.app_config, 'providers') and provider in self.app_config.providers:
                configured = " ✓"
            content.append(f"{i}. {provider}{configured}")
        
        return self.ui.box(
            "\n".join(content) + "\n\nUse: provider <name> to select",
            title="Providers",
            style='info'
        )

    async def _cmd_exit(self) -> str:
        """Handle exit command"""
        print(self.theme_manager.colorize("👋 Goodbye! Thanks for using AgentsMCP!", 'success'))
        self.is_running = False
        return "Exiting"
    
    async def _cleanup_interface(self):
        """Cleanup interface resources"""
        # Save command history
        try:
            history_file = os.path.expanduser("~/.agentsmcp_history")
            with open(history_file, 'w') as f:
                for entry in self.command_history[-100:]:  # Save last 100 commands
                    f.write(f"{entry.command}\n")
        except Exception as e:
            logger.warning(f"Failed to save command history: {e}")
        
        logger.info("Interface cleanup completed")
    
    async def execute_single_command(self, command_line: str) -> Tuple[bool, Any]:
        """Execute a single command (non-interactive mode)"""
        return await self._process_command(command_line)
