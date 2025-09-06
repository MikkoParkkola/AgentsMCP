"""
Main CLI Application for AgentsMCP

Revolutionary command-line interface that orchestrates all UI components
into a beautiful, cohesive experience with Apple-style design principles.
"""
import asyncio
import signal
import sys
import os
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import time

from .theme_manager import ThemeManager, Fore

# Legacy modern TUI removed - using only the working TUI implementation
from .ui_components import UIComponents
from .status_dashboard import StatusDashboard
from .command_interface import CommandInterface
from .statistics_display import StatisticsDisplay


@dataclass
class CLIConfig:
    """Configuration for the CLI application"""
    theme_mode: str = "auto"  # auto, light, dark
    auto_refresh: bool = True
    refresh_interval: float = 2.0
    show_welcome: bool = True
    enable_colors: bool = True
    debug_mode: bool = False
    log_level: str = "WARNING"  # Hide INFO logs in interactive mode
    interface_mode: str = "interactive"  # interactive, dashboard, stats
    orchestrator_model: str = "gpt-5"
    agent_type: str = "ollama-turbo-coding"
    model_override: Optional[str] = None
    provider_override: Optional[str] = None
    streaming: bool = True


class CLIApp:
    """Main CLI Application - Revolutionary interface for AgentsMCP"""
    
    # Collection of funny rules and quotes to make users smile
    BROA_RULES = [
        "Rule #1: If it compiles, ship it. If it doesn't, blame the compiler.",
        "Rule #2: There are only 10 types of people: those who understand binary and those who don't.",
        "Rule #3: A user interface is like a joke - if you have to explain it, it's not that good.",
        "Rule #4: 99 bugs in the code, 99 bugs in the code. Take one down, patch it around, 117 bugs in the code.",
        "Rule #5: Programming is like sex: one mistake and you have to support it for the rest of your life.",
        "Rule #6: Always code as if the person who ends up maintaining your code is a violent psychopath who knows where you live.",
        "Rule #7: The best way to accelerate a computer running Windows is to throw it out the window.",
        "Rule #8: A SQL query goes into a bar, walks up to two tables and asks: 'Can I join you?'",
        "Rule #9: Why do programmers prefer dark mode? Because light attracts bugs!",
        "Rule #10: In order to understand recursion, you must first understand recursion.",
        "Rule #11: There are two hard things in computer science: cache invalidation, naming things, and off-by-one errors.",
        "Rule #12: A byte walks into a bar. The bartender asks, 'Are you feeling a bit off?' The byte replies, 'No, just a bit.'",
        "Rule #13: How many programmers does it take to change a light bulb? None, that's a hardware problem.",
        "Rule #14: Programming is 10% science, 20% ingenuity, and 70% getting the ingenuity to work with the science.",
        "Rule #15: The first 90% of the code accounts for the first 90% of the development time. The remaining 10% accounts for the other 90%.",
        "Rule #16: Walking on water and developing software from a specification are easy if both are frozen.",
        "Rule #17: It's not a bug, it's an undocumented feature!",
        "Rule #18: Debugging is twice as hard as writing the code in the first place.",
        "Rule #19: If debugging is the process of removing bugs, then programming must be the process of putting them in.",
        "Rule #20: Any fool can write code that a computer can understand. Good programmers write code that humans can understand."
    ]
    
    FUNNY_QUOTES = [
        "Code never lies, comments sometimes do. 🤔",
        "Programming is the art of telling another human being what one wants the computer to do. 🎨",
        "The most important property of a program is whether it accomplishes the intention of its user. ✨",
        "Simplicity is the ultimate sophistication. 🌟",
        "Code is like humor. When you have to explain it, it's bad. 😄",
        "First, solve the problem. Then, write the code. 🧠",
        "Experience is the name everyone gives to their mistakes. 🎓",
        "The best error message is the one that never shows up. 🚫",
        "Deleted code is debugged code. 🗑️",
        "Programming isn't about what you know; it's about what you can figure out. 🔍"
    ]
    
    def __init__(self, config: CLIConfig = None, mode: str = "interactive"):
        """Create the CLI application driver.

        Parameters
        ----------
        config: CLIConfig
            Global configuration for the CLI application.
        mode: str
            The UI mode to launch. Known values are:
            * "interactive" – legacy line-oriented REPL.
            * "dashboard"   – dashboard UI (unchanged).
            * "tui"         – modern TUI (default for the ``run interactive``
                              command after this fix).
        """
        self.config = config or CLIConfig()
        self.current_mode = mode
        
        # Configure logging for interactive mode
        from ..logging_config import configure_logging
        configure_logging(level=self.config.log_level, fmt="text")
        
        # Initialize core components
        self.theme_manager = ThemeManager()
        self.ui = UIComponents(self.theme_manager)
        # Create lightweight orchestration manager for CLI (avoids async initialization issues)
        from ..orchestration.orchestration_manager import OrchestrationManager
        self.orchestration_manager = self._create_cli_orchestration_manager()
        
        # Initialize timezone handling
        self._setup_timezone()
        
        from .status_dashboard import DashboardConfig
        dashboard_config = DashboardConfig(
            auto_refresh=self.config.auto_refresh,
            refresh_interval=self.config.refresh_interval
        )
        
        self.status_dashboard = StatusDashboard(
            orchestration_manager=self.orchestration_manager,
            theme_manager=self.theme_manager, 
            config=dashboard_config
        )
        # Create agent manager and load configuration for enhanced interactive mode
        self.agent_manager = None
        self.app_config = None
        try:
            from ..agent_manager import AgentManager
            from ..config import Config
            from pathlib import Path
            
            # Load configuration if available
            if Path("agentsmcp.yaml").exists():
                self.app_config = Config.from_file(Path("agentsmcp.yaml"))
                self.agent_manager = AgentManager(self.app_config)
        except Exception as e:
            # Continue without agent manager if it fails
            pass
        
        self.command_interface = CommandInterface(
            self.orchestration_manager, 
            self.theme_manager,
            agent_manager=self.agent_manager,
            app_config=self.app_config
        )
        
        # Configure session defaults from CLI config
        if hasattr(self.command_interface, 'current_agent'):
            self.command_interface.current_agent = self.config.agent_type
        if hasattr(self.command_interface, 'stream_enabled'):
            self.command_interface.stream_enabled = self.config.streaming
        self.statistics_display = StatisticsDisplay(self.theme_manager, {
            'auto_refresh': self.config.auto_refresh,
            'refresh_interval': self.config.refresh_interval
        })
        
        # Application state
        self.is_running = False
        # Keep the mode passed to constructor, don't override with config default
        # self.current_mode = self.config.interface_mode  # BUG: This overrides the constructor parameter
        self.session_start_time = None
        # Track active TUI shell (if any) so signal handler can stop it
        self._current_tui_shell = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.is_running = False
        print(f"\n{self.theme_manager.current_theme.palette.warning}Shutting down gracefully...{Fore.RESET}")
        # If a TUI shell is running, request it to stop
        try:
            shell = getattr(self, "_current_tui_shell", None)
            if shell is not None:
                shell.running = False
        except Exception:
            pass
    
    async def start(self) -> Dict[str, Any]:
        """Start the CLI application"""
        if self.config.debug_mode:
            print(f"🔧 Debug: CLIApp.start() called with mode: {self.current_mode}")
            
        self.is_running = True
        self.session_start_time = asyncio.get_event_loop().time()
        
        if self.config.debug_mode:
            print(f"🔧 Debug: Session start time: {self.session_start_time}")
            print(f"🔧 Debug: Theme mode: {self.config.theme_mode}")
        
        # Apply theme configuration
        if self.config.theme_mode != "auto":
            self.theme_manager.set_theme(self.config.theme_mode)
        else:
            self.theme_manager.auto_detect_theme()
        
        if self.config.debug_mode:
            print(f"🔧 Debug: Theme applied, checking welcome screen...")
            print(f"🔧 Debug: show_welcome={self.config.show_welcome}, current_mode={self.current_mode}")
        
        # Show welcome screen (skip for TUI to avoid ASCII art)
        if self.config.show_welcome and self.current_mode != "tui":
            await self._show_welcome()

        # Robust non-TTY fallback: if interactive mode is requested but stdin is not a TTY,
        # avoid launching the TUI which requires a real terminal. Prefer dashboard.
        try:
            import sys as _sys
            if self.current_mode == "interactive" and not _sys.stdin.isatty():
                # Use dashboard in non-TTY environments
                print(self.theme_manager.colorize("(no TTY detected) Launching dashboard", 'text_muted'))
                self.current_mode = "dashboard"
        except Exception:
            # If detection fails, keep original mode
            pass
        
        # Start main application loop
        if self.config.debug_mode:
            print(f"🔧 Debug: Starting main application loop with mode: {self.current_mode}")
            
        try:
            if self.current_mode == "interactive":
                if self.config.debug_mode:
                    print("🔧 Debug: Launching interactive mode...")
                # Legacy basic REPL (unchanged).
                await self._run_interactive_mode()
            elif self.current_mode == "dashboard":
                if self.config.debug_mode:
                    print("🔧 Debug: Launching dashboard mode...")
                await self._run_dashboard_mode()
            elif self.current_mode == "stats":
                if self.config.debug_mode:
                    print("🔧 Debug: Launching statistics mode...")
                await self._run_statistics_mode()
            elif self.current_mode == "tui":
                if self.config.debug_mode:
                    print("🔧 Debug: Launching TUI mode - calling _run_modern_tui()...")
                # Launch the rich terminal interface
                try:
                    await self._run_modern_tui()
                except Exception as exc:  # pragma: no cover – defensive
                    import logging, os as _os
                    logging.getLogger(__name__).exception(
                        "Failed to start TUI"
                    )
                    # V3-ONLY MODE - No V2 fallback, enhanced error reporting
                    await self._show_error(f"V3 TUI failed: {exc}")
                    print("❌ V2 fallback DISABLED - debugging V3 issues")
                    print(f"   Error details: {str(exc)}")
                    print("   This error needs to be fixed in the V3 system")
                    print("   Falling back to simple PlainCLIRenderer instead of V2...")
                    
                    # Try the simple PlainCLI fallback instead of V2
                    try:
                        await self._run_simple_plain_cli()
                    except Exception as cli_exc:
                        print(f"❌ Even PlainCLI fallback failed: {cli_exc}")
                        print("   Falling back to basic interactive mode...")
                        await self._run_interactive_mode()
                    
                    return {
                        "status": "error_with_fallback",
                        "session_duration": 0,
                        "final_mode": "plain_cli_fallback",
                        "error": str(exc),
                        "note": "V2 Revolutionary TUI disabled - used PlainCLI fallback"
                    }
            else:
                await self._run_interactive_mode()  # Default fallback
                
        except KeyboardInterrupt:
            pass  # Handled by signal handler
        except Exception as e:
            await self._show_error(f"Application error: {str(e)}")
        finally:
            await self._cleanup()
        
        session_duration = asyncio.get_event_loop().time() - self.session_start_time
        return {
            "status": "completed",
            "session_duration": session_duration,
            "final_mode": self.current_mode
        }
    
    async def _show_welcome(self):
        """Display simplified welcome screen"""
        theme = self.theme_manager.current_theme
        
        # Clear screen and show cursor
        print(self.ui.clear_screen())
        print(self.ui.move_cursor(1, 1))
        
        # Simple welcome message
        welcome_content = f"""{theme.palette.secondary}Welcome to AgentsMCP!{Fore.RESET}

{theme.palette.text_muted}A revolutionary multi-agent orchestration platform.{Fore.RESET}

{theme.palette.primary}Type 'help' to get started.{Fore.RESET}
        """
        
        welcome_box = self.ui.box(
            welcome_content.strip(),
            title="🚀 AgentsMCP",
            style='light',
            width=min(self.ui.terminal_width - 4, 60)
        )
        print(welcome_box)
        
        # Wait briefly then clear for a cleaner interface
        await asyncio.sleep(1)
        print(self.ui.clear_screen())
    
    async def _run_interactive_mode(self):
        """Run the interactive command interface"""
        print(self.ui.clear_screen())
        
        # Show简洁 mode header
        header = self.ui.box(
            "Interactive Mode - Type 'help' for commands",
            title="🎮 AgentsMCP",
            style='light',
            width=min(self.ui.terminal_width - 4, 60)
        )
        print(header)
        print()
        
        # Start command interface
        await self.command_interface.start_interactive_mode()
    
    async def _run_dashboard_mode(self):
        """Run the status dashboard"""
        print(self.ui.clear_screen())
        
        # Show dashboard header
        header = self.ui.box(
            "📊 Real-time Orchestration Dashboard - Press 'q' to exit",
            title="Dashboard Mode", 
            style='light',
            width=80
        )
        print(header)
        
        # Start dashboard
        await self.status_dashboard.start_dashboard()
    
    # V2 TUI COMPLETELY REMOVED - _run_tui_shell method disabled
    # This method previously launched V2 Revolutionary TUI - now disabled to force V3 debugging
    async def _run_tui_shell(self):
        """V2 TUI DISABLED - This method is no longer functional.
        
        Previously launched V2 Revolutionary TUI system, but now disabled
        to force exclusive V3 usage and enable proper V3 debugging.
        """
        print("❌ V2 TUI system DISABLED - _run_tui_shell() is no longer functional")
        print("   This method previously launched V2 Revolutionary TUI")
        print("   V2 is disabled to force V3-only execution for debugging")
        print("   Use V3 TUI system via _run_modern_tui() instead")
        return 1  # Return error code to indicate this method is disabled

    async def _run_statistics_mode(self):
        """Run the statistics display"""
        print(self.ui.clear_screen())
        
        # Show statistics header
        header = self.ui.box(
            "📈 Advanced Statistics & Trends - Press 'q' to exit",
            title="Statistics Mode",
            style='light', 
            width=80
        )
        print(header)
        
        # Start metrics simulation for demo
        asyncio.create_task(self.statistics_display.simulate_metrics(3600))
        
        # Start statistics display
        await self.statistics_display.start_display()

    async def _run_modern_tui(self):
        """Launch V3 TUI system ONLY - V2 Revolutionary TUI completely disabled.

        This method forces the system to use V3 TUI exclusively:
        - Uses the new v3 architecture with progressive enhancement
        - Provides detailed error reporting for debugging
        - Falls back to simple PlainCLIRenderer if V3 completely fails
        - NO V2 fallback to allow proper V3 debugging
        """
        if self.config.debug_mode:
            print("🔧 Debug: _run_modern_tui() called - V2 DISABLED, V3-ONLY MODE")
            
        print("🚀 Starting V3 TUI system (V2 Revolutionary TUI DISABLED)...")
        
        try:
            if self.config.debug_mode:
                print("🔧 Debug: Importing V3 tui_launcher...")
                
            # Import and launch V3 TUI system ONLY
            from .v3.tui_launcher import launch_tui
            
            if self.config.debug_mode:
                print("🔧 Debug: Calling V3 launch_tui()...")
                print(f"🔧 Debug: Config debug_mode: {self.config.debug_mode}")
                
            print("✅ V3 TUI launcher imported successfully")
            exit_code = await launch_tui()
            
            if exit_code != 0:
                print(f"⚠️  V3 TUI exited with code: {exit_code}")
                print("   This indicates a V3 issue that needs debugging")
            
            return exit_code
            
        except ImportError as e:
            print(f"❌ V3 TUI components missing: {e}")
            import traceback
            print("📋 Full import error traceback:")
            traceback.print_exc()
            print("   Falling back to simple PlainCLIRenderer...")
            return await self._run_simple_plain_cli()
        except Exception as e:
            import logging
            import traceback
            logger = logging.getLogger(__name__)
            logger.error(f"V3 TUI launch failed: {e}")
            
            print(f"❌ V3 TUI failed with exception: {e}")
            print("📋 Full error traceback:")
            traceback.print_exc()
            print("   This error needs to be debugged in V3 system")
            print("   Falling back to simple PlainCLIRenderer...")
            return await self._run_simple_plain_cli()
    
    async def _run_simple_plain_cli(self):
        """Run the simplest possible CLI renderer when V3 TUI completely fails."""
        print("🔄 Starting simple PlainCLIRenderer fallback...")
        try:
            # Import V3 PlainCLIRenderer directly
            from .v3.plain_cli_renderer import PlainCLIRenderer
            from .v3.terminal_capabilities import TerminalCapabilities
            from .v3.chat_engine import ChatEngine
            
            # Create minimal capabilities for fallback
            capabilities = TerminalCapabilities(
                is_tty=False,
                width=80,
                height=24,
                supports_colors=False,
                supports_unicode=False,
                supports_rich=False
            )
            
            # Create and run plain CLI renderer
            renderer = PlainCLIRenderer(capabilities)
            chat_engine = ChatEngine()
            
            print("✅ PlainCLIRenderer initialized")
            print("📝 Type your messages and press Enter. Type '/quit' to exit.")
            print()
            
            # Simple interaction loop
            while True:
                try:
                    user_input = input("> ").strip()
                    
                    if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                        print("👋 Goodbye!")
                        return 0
                    
                    if user_input.lower() in ['/help', 'help']:
                        print("📚 Available commands:")
                        print("  /help - Show this help")
                        print("  /quit, /exit - Exit the application")
                        print("  Just type to chat!")
                        continue
                    
                    if not user_input:
                        continue
                    
                    # Process through chat engine
                    should_continue = await chat_engine.process_input(user_input)
                    if not should_continue:
                        print("👋 Goodbye!")
                        return 0
                        
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    return 0
                except EOFError:
                    print("\n👋 Goodbye!")
                    return 0
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
                    
        except ImportError as e:
            print(f"❌ Even PlainCLIRenderer not available: {e}")
            print("   Using absolute minimal interactive mode...")
            return await self._run_simple_interactive_mode()
        except Exception as e:
            print(f"❌ PlainCLIRenderer failed: {e}")
            print("   Using absolute minimal interactive mode...")
            return await self._run_simple_interactive_mode()
    
    async def _run_simple_interactive_mode(self):
        """Run a very basic interactive mode when TUI completely fails."""
        print("🤖 AgentsMCP - Simple Interactive Mode")
        print("─" * 50)
        print("Type your message and press Enter. Type 'quit' or 'exit' to exit.")
        print("Type 'help' for basic commands.")
        print()
        
        try:
            # Setup LLM client
            sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')
            from agentsmcp.conversation.llm_client import LLMClient
            llm_client = LLMClient()
            print(f"✅ Connected to {llm_client.provider} - {llm_client.model}")
            
        except Exception as e:
            print(f"⚠️  LLM client failed to initialize: {e}")
            print("   Continuing in demo mode...")
            llm_client = None
        
        print()
        
        try:
            while True:
                try:
                    user_input = input("> ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', '/quit', '/exit']:
                        print("👋 Goodbye!")
                        return 0
                    
                    if user_input.lower() in ['help', '/help']:
                        print("📚 Commands:")
                        print("  help/quit/exit - Show help or exit")
                        print("  Just type normally to chat with the LLM!")
                        continue
                    
                    if not user_input:
                        continue
                    
                    if llm_client:
                        print("\n🤖 AgentsMCP:")
                        try:
                            response = await llm_client.chat_async([{"role": "user", "content": user_input}])
                            print(response)
                        except Exception as e:
                            print(f"❌ Error: {str(e)}")
                            print("   Please try again.")
                    else:
                        print(f"⚠️  LLM client unavailable. You said: \"{user_input}\"")
                        print("   Try restarting the application to reconnect.")
                    
                    print()
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    return 0
                except EOFError:
                    print("\n👋 Goodbye!")
                    return 0
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"❌ Simple interactive mode failed: {str(e)}")
            return 1
    
    async def _show_error(self, error_message: str):
        """Display error message beautifully"""
        theme = self.theme_manager.current_theme
        
        error_box = self.ui.box(
            f"{theme.palette.error}💥 {error_message}{Fore.RESET}",
            title="Error",
            style='heavy',
            width=60
        )
        print(error_box)
        
        if self.config.debug_mode:
            import traceback
            debug_info = traceback.format_exc()
            debug_box = self.ui.box(
                debug_info,
                title="Debug Information",
                style='light',
                width=80
            )
            print(debug_box)
    
    async def _cleanup(self):
        """Cleanup resources and show goodbye message"""
        theme = self.theme_manager.current_theme
        
        # Stop all running components
        if hasattr(self.status_dashboard, 'stop_dashboard'):
            self.status_dashboard.stop_dashboard()
        
        if hasattr(self.statistics_display, 'stop_display'):
            self.statistics_display.stop_display()
        
        if hasattr(self.command_interface, 'stop'):
            self.command_interface.stop()
        
        # Minimal goodbye (avoid ASCII art); especially quiet when TUI was used
        if self.current_mode != "tui":
            print(self.ui.clear_screen())
            print(f"{theme.palette.primary}Goodbye!{Fore.RESET}")
        # Restore cursor
        print(self.ui.show_cursor(), end='')
    
    def _setup_timezone(self):
        """Setup timezone handling for the application"""
        try:
            # Get system timezone
            self.local_timezone = ZoneInfo(time.tzname[0] if time.tzname[0] else 'UTC')
        except Exception:
            self.local_timezone = None
    
    def _get_local_time(self) -> datetime:
        """Get current time in local timezone"""
        try:
            if self.local_timezone:
                return datetime.now(self.local_timezone)
            return datetime.now()
        except Exception:
            return datetime.now()
    
    def _format_timestamp(self, dt: datetime) -> str:
        """Format timestamp in local timezone"""
        try:
            if dt.tzinfo is None and self.local_timezone:
                # If naive datetime, assume it's local
                dt = dt.replace(tzinfo=self.local_timezone)
            
            # Convert to local timezone for display
            local_dt = dt.astimezone()
            return local_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            # Fallback to simple formatting
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def _create_cli_orchestration_manager(self):
        """Create a minimal orchestration manager for CLI use without async components."""
        import json
        from pathlib import Path
        from datetime import datetime
        
        class CLIOrchestrationManager:
            """Minimal orchestration manager for CLI settings and config generation."""
            
            def __init__(self):
                # Settings persistence
                self.config_dir = Path.home() / ".agentsmcp"
                self.settings_file = self.config_dir / "config.json"
                self.user_settings = {}
                self.reload_user_settings()
                
            def save_user_settings(self, settings):
                """Save user settings to configuration file."""
                try:
                    self.config_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Merge with existing settings
                    current_settings = self.user_settings.copy()
                    current_settings.update(settings)
                    
                    # Write to file
                    with open(self.settings_file, 'w') as f:
                        json.dump(current_settings, f, indent=2)
                    
                    # Update in-memory settings
                    self.user_settings = current_settings
                    
                except Exception as e:
                    raise Exception(f"Failed to save settings: {e}")
            
            def reload_user_settings(self):
                """Reload user settings from configuration file."""
                try:
                    if self.settings_file.exists():
                        with open(self.settings_file, 'r') as f:
                            self.user_settings = json.load(f)
                    else:
                        # Default settings
                        self.user_settings = {
                            "provider": "ollama-turbo",
                            "model": "gpt-oss:120b", 
                            "temperature": 0.7,
                            "max_tokens": 1024
                        }
                        
                    return self.user_settings
                    
                except Exception:
                    # Fall back to defaults
                    self.user_settings = {
                        "provider": "ollama-turbo",
                        "model": "gpt-oss:120b",
                        "temperature": 0.7, 
                        "max_tokens": 1024
                    }
                    return self.user_settings
            
            def generate_client_config(self):
                """Generate MCP client configuration with auto-discovered paths."""
                import subprocess
                import shutil
                
                # Auto-discover system paths
                node_path = shutil.which("node") or "/usr/local/bin/node"
                python_path = shutil.which("python3") or shutil.which("python") or "/usr/bin/python3"
                
                # Get current user settings
                settings = self.user_settings
                
                # Base configuration template
                config = {
                    "mcpServers": {
                        "codex": {
                            "command": "npx",
                            "args": ["-y", "@anthropic/mcp-codex"],
                            "env": {
                                "NODE_PATH": node_path,
                                "CODEX_MODEL": settings.get("model", "gpt-oss:120b"),
                                "CODEX_PROVIDER": settings.get("provider", "ollama-turbo"),
                                "CODEX_TEMPERATURE": str(settings.get("temperature", 0.7)),
                                "CODEX_MAX_TOKENS": str(settings.get("max_tokens", 1024))
                            }
                        },
                        "claude": {
                            "command": python_path,
                            "args": ["-m", "mcp_claude"],
                            "env": {
                                "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
                                "CLAUDE_MODEL": "claude-3-5-sonnet-20241022",
                                "CLAUDE_MAX_TOKENS": str(settings.get("max_tokens", 1024))
                            }
                        },
                        "ollama": {
                            "command": "npx",
                            "args": ["-y", "@anthropic/mcp-ollama"],
                            "env": {
                                "OLLAMA_HOST": "http://localhost:11434",
                                "OLLAMA_MODEL": "gpt-oss:20b"
                            }
                        },
                        "ollama-turbo": {
                            "command": "npx", 
                            "args": ["-y", "@anthropic/mcp-ollama-turbo"],
                            "env": {
                                "OLLAMA_TURBO_HOST": settings.get("ollama_host", "https://ollama.com"),
                                "OLLAMA_TURBO_MODEL": settings.get("model", "gpt-oss:120b"),
                                "OLLAMA_API_KEY": "${OLLAMA_API_KEY}"
                            }
                        },
                        "github": {
                            "command": "npx",
                            "args": ["-y", "@anthropic/mcp-github"],
                            "env": {
                                "GITHUB_TOKEN": "${GITHUB_TOKEN}"
                            }
                        },
                        "filesystem": {
                            "command": "npx",
                            "args": ["-y", "@anthropic/mcp-filesystem"],
                            "env": {
                                "FILESYSTEM_ROOT": str(Path.cwd())
                            }
                        },
                        "git": {
                            "command": "npx",
                            "args": ["-y", "@anthropic/mcp-git"],
                            "env": {
                                "GIT_WORKING_DIR": str(Path.cwd())
                            }
                        }
                    }
                }

                # Optionally include locally installed CLI MCP servers
                try:
                    claude_code_bin = shutil.which("claude-code")
                    if claude_code_bin:
                        config["mcpServers"]["claude-code-cli"] = {
                            "command": claude_code_bin,
                            "args": ["mcp-server"],
                            "env": {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"}
                        }
                except Exception:
                    pass

                try:
                    codex_bin = shutil.which("codex") or shutil.which("codex-cli")
                    if codex_bin:
                        config["mcpServers"]["codex-cli"] = {
                            "command": codex_bin,
                            "args": ["mcp-server"],
                            "env": {}
                        }
                except Exception:
                    pass
                
                # Format as pretty-printed JSON
                config_json = json.dumps(config, indent=2)
                
                # Add helpful header comments
                header = f"""# MCP Client Configuration
# Generated by AgentsMCP on {self._format_timestamp(self._get_local_time())}
# Current settings: {settings.get('provider')} / {settings.get('model')}
#
# Save this to one of:
# - ~/.config/Claude/claude_desktop_config.json  (Claude Desktop)
# - ~/.config/claude-code/config.json           (Claude Code CLI) 
# - Your MCP client's configuration file
#
# Required environment variables:
# - ANTHROPIC_API_KEY: Your Anthropic API key for Claude
# - GITHUB_TOKEN: Your GitHub personal access token
# - OLLAMA_API_KEY: Your Ollama API key (if using)
#
# Auto-discovered paths:
# - Node.js: {node_path}
# - Python: {python_path}
# - Working Directory: {Path.cwd()}

"""
                
                return header + config_json
            
            async def get_system_status(self):
                """Get basic system status information."""
                from datetime import datetime, timedelta
                import time
                
                # Basic status information for CLI mode
                current_time = self._get_local_time()
                
                return {
                    "system_status": "running",
                    "session_id": "cli-session",
                    "orchestration_mode": "cli",
                    "uptime": str(timedelta(seconds=int(time.time() - getattr(self, '_start_time', time.time())))),
                    "performance_metrics": {
                        "total_tasks_completed": 0,
                        "active_agents": 0,
                        "memory_usage": 0
                    },
                    "component_status": {
                        "cli": {"status": "active"},
                        "settings": {"status": "available"},
                        "config_generator": {"status": "available"}
                    }
                }
            
            async def initialize(self, mode="hybrid"):
                """Initialize orchestration system."""
                self.is_running = True
                self.mode = mode
                return {"status": "initialized", "mode": mode}
            
            async def execute_task(self, task, context=None):
                """Execute a task using the orchestration system."""
                from datetime import datetime
                import uuid
                
                # For CLI mode, we simulate task execution
                task_id = str(uuid.uuid4())[:8]
                
                # Simulate task analysis and execution
                print(f"🤖 Task received: {task}")
                print(f"📋 Task ID: {task_id}")
                print(f"🔍 Analyzing task requirements...")
                
                # This is a simplified simulation - in a real implementation
                # this would integrate with actual agents and orchestration
                
                if "AgentsMCP" in task or "codebase" in task or "improve" in task:
                    suggestions = [
                        "1. Add comprehensive unit tests to increase code coverage",
                        "2. Implement proper error handling with try-catch blocks and informative error messages",  
                        "3. Add type hints throughout the codebase for better IDE support and documentation",
                        "4. Optimize the CLI command parsing for better performance with large command sets",
                        "5. Add configuration validation to prevent runtime errors from invalid settings"
                    ]
                    
                    print("💡 Generated improvement suggestions:")
                    for suggestion in suggestions:
                        print(f"   {suggestion}")
                    
                    return {
                        "task_id": task_id,
                        "execution_strategy": "analysis",
                        "completion_time": self._format_timestamp(self._get_local_time()),
                        "status": "completed",
                        "results": {
                            "type": "codebase_analysis", 
                            "suggestions": suggestions,
                            "analysis": "Performed comprehensive codebase analysis and identified key areas for improvement"
                        }
                    }
                else:
                    print("⚠️ Task type not recognized for CLI simulation mode")
                    return {
                        "task_id": task_id,
                        "execution_strategy": "simulation",
                        "completion_time": self._format_timestamp(self._get_local_time()),
                        "status": "completed",
                        "results": {
                            "message": "Task simulated in CLI mode - full orchestration requires running agents"
                        }
                    }
            
            def _get_local_time(self):
                """Get current local time with timezone info"""
                from datetime import datetime
                try:
                    from zoneinfo import ZoneInfo
                    import time
                    # Get local timezone
                    local_tz = ZoneInfo(time.tzname[0])
                    return datetime.now(local_tz)
                except ImportError:
                    # Fallback for older Python versions
                    return datetime.now()
            
            def _format_timestamp(self, dt) -> str:
                """Format datetime as human-readable timestamp"""
                return dt.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
        
        return CLIOrchestrationManager()

    def switch_mode(self, new_mode: str) -> bool:
        """Switch between interface modes"""
        if new_mode in ["interactive", "dashboard", "stats", "tui"]:
            self.current_mode = new_mode
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current application status"""
        uptime = 0
        if self.session_start_time:
            uptime = asyncio.get_event_loop().time() - self.session_start_time
        
        return {
            "running": self.is_running,
            "mode": self.current_mode,
            "theme": self.theme_manager.current_theme.name,
            "uptime": uptime,
            "config": {
                "theme_mode": self.config.theme_mode,
                "auto_refresh": self.config.auto_refresh,
                "refresh_interval": self.config.refresh_interval
            }
        }


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="AgentsMCP - Revolutionary Multi-Agent Orchestration Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
        epilog="""
Examples:
  agentsmcp                           # Interactive REPL mode
  agentsmcp --mode tui                # Rich terminal interface  
  agentsmcp --mode dashboard          # Status dashboard
  agentsmcp --mode stats              # Statistics display
  agentsmcp --theme dark              # Force dark theme
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['interactive', 'dashboard', 'stats', 'tui'],
        default='interactive',
        help='Interface mode: interactive=REPL, tui=rich terminal UI, dashboard=status display, stats=metrics (default: interactive)'
    )
    
    parser.add_argument(
        '--theme', '-t', 
        choices=['auto', 'light', 'dark'],
        default='auto',
        help='Theme mode (default: auto)'
    )
    
    parser.add_argument(
        '--refresh-interval', '-r',
        type=float,
        default=2.0,
        help='Auto-refresh interval in seconds (default: 2.0)'
    )
    
    parser.add_argument(
        '--no-welcome',
        action='store_true',
        help='Skip welcome screen'
    )
    
    parser.add_argument(
        '--no-colors',
        action='store_true', 
        help='Disable color output'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    return parser


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        return {}


async def main():
    """Main entry point for the CLI application"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    file_config = {}
    if args.config:
        file_config = load_config_file(args.config)
    
    # Merge configuration sources (command line overrides file)
    config = CLIConfig(
        theme_mode=args.theme,
        auto_refresh=True,
        refresh_interval=args.refresh_interval,
        show_welcome=not args.no_welcome,
        enable_colors=not args.no_colors,
        debug_mode=args.debug,
        interface_mode=args.mode
    )
    
    # Apply file configuration
    for key, value in file_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Handle TUI mode with V3 ONLY architecture (V2 completely disabled)
    if args.mode == "tui":
        try:
            # Import and use ONLY the V3 TUI system - no V2 fallback
            from .v3.tui_launcher import launch_tui
            print("🚀 Starting V3 TUI system (V2 Revolutionary TUI DISABLED)...")
            
            if config.debug_mode:
                print("🔧 Debug: V3 direct launch - bypassing legacy CLI app routing")
                
            exit_code = await launch_tui()
            
            if exit_code != 0:
                print(f"❌ V3 TUI failed with exit code: {exit_code}")
                print("   This indicates a V3 system issue that needs debugging")
            
            return exit_code
            
        except ImportError as e:
            print(f"❌ V3 TUI not available: {e}")
            if config.debug_mode:
                import traceback
                print("📋 Full V3 import error:")
                traceback.print_exc()
            print("   V2 system DISABLED - no fallback available")
            print("   Please fix V3 import issues or run in different mode")
            return 1
        except Exception as e:
            print(f"❌ V3 TUI failed: {e}")
            if config.debug_mode:
                import traceback
                print("📋 Full V3 error:")
                traceback.print_exc()
            print("   V2 system DISABLED - debugging V3 issues")
            print("   This error needs to be fixed in V3 system")
            return 1
    
    # Create and start the CLI application (legacy system)
    app = CLIApp(config, mode=args.mode)
    
    try:
        result = await app.start()
        if config.debug_mode:
            print(f"Debug: Session result: {result}")
    except Exception as e:
        print(f"Fatal error: {e}")
        if config.debug_mode:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Ensure proper event loop handling across platforms
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
