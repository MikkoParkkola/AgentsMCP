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
from ..orchestration.orchestration_manager import OrchestrationManager

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
                 config: Optional[InterfaceConfig] = None):
        self.orchestration_manager = orchestration_manager
        self.theme_manager = theme_manager or ThemeManager()
        self.ui = UIComponents(self.theme_manager)
        self.config = config or InterfaceConfig()
        
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
                    # Get command input
                    prompt = self._generate_prompt()
                    command_line = input(prompt).strip()
                    
                    if not command_line:
                        continue
                    
                    # Process command
                    start_time = datetime.now()
                    success, result = await self._process_command(command_line)
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
    
    async def _show_welcome(self):
        """Show welcome message and system status"""
        # Create welcome card
        title = "🎼 AgentsMCP Revolutionary CLI"
        
        welcome_content = [
            "Welcome to the future of AI agent orchestration!",
            "",
            self.ui.status_indicator("info", "Type 'help' for available commands"),
            self.ui.status_indicator("info", "Type 'dashboard' for real-time monitoring"),  
            self.ui.status_indicator("info", "Type 'execute \"your task\"' to run orchestration"),
            "",
            f"Theme: {self.theme_manager.get_current_theme().name}",
            f"Mode: {self.config.prompt_style.title()}",
        ]
        
        welcome_text = '\n'.join(welcome_content)
        welcome_card = self.ui.card(title, welcome_text, status="success")
        
        print(welcome_card)
        print()  # Extra spacing
        
        # Show quick status if orchestration manager is initialized
        if hasattr(self.orchestration_manager, 'is_running') and self.orchestration_manager.is_running:
            status_summary = await self._get_quick_status()
            print(status_summary)
            print()
    
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
    
    async def _process_command(self, command_line: str) -> Tuple[bool, Any]:
        """Process a command line input"""
        try:
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
                else:
                    if i + 1 >= len(args):
                        raise ValueError(f"Parameter --{param_name} requires a value")
                    
                    value = args[i + 1]
                    params[param_name] = self._convert_parameter_value(value, param_def)
                    i += 1
                
            else:
                # Positional parameter
                # Find first required parameter without a value
                for param in command.parameters:
                    if param.get('required') and param['name'] not in params:
                        params[param['name']] = self._convert_parameter_value(arg, param)
                        break
                else:
                    # No more required parameters, treat as positional
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
            
            config = DashboardConfig(
                refresh_interval=refresh,
                compact_mode=compact,
                auto_refresh=True
            )
            
            dashboard = StatusDashboard(self.orchestration_manager, self.theme_manager, config)
            
            print(self.theme_manager.colorize("🚀 Launching interactive dashboard...", 'info'))
            print(self.theme_manager.colorize("Press Ctrl+C to exit dashboard", 'text_muted'))
            print()
            
            await dashboard.start_dashboard()
            
            return "Dashboard session completed"
            
        except Exception as e:
            error_msg = f"Failed to start dashboard: {e}"
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
                item.command[:40] + "..." if len(item.command) > 40 else item.command,
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
            # Show general help
            help_sections = []
            
            for cat_name, cmd_names in self.categories.items():
                visible_commands = [
                    cmd_name for cmd_name in cmd_names 
                    if not self.commands[cmd_name].hidden
                ]
                
                if visible_commands:
                    section_content = []
                    for cmd_name in visible_commands[:5]:  # Show first 5 commands
                        cmd = self.commands[cmd_name]
                        section_content.append(f"  {cmd_name}: {cmd.description}")
                    
                    if len(visible_commands) > 5:
                        section_content.append(f"  ... and {len(visible_commands) - 5} more")
                    
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