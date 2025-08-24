"""
Main CLI Application for AgentsMCP

Revolutionary command-line interface that orchestrates all UI components
into a beautiful, cohesive experience with Apple-style design principles.
"""
import asyncio
import signal
import sys
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import argparse

from .theme_manager import ThemeManager
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
    log_level: str = "INFO"
    interface_mode: str = "interactive"  # interactive, dashboard, stats


class CLIApp:
    """Main CLI Application - Revolutionary interface for AgentsMCP"""
    
    def __init__(self, config: CLIConfig = None):
        self.config = config or CLIConfig()
        
        # Initialize core components
        self.theme_manager = ThemeManager()
        self.ui = UIComponents(self.theme_manager)
        self.status_dashboard = StatusDashboard(self.theme_manager, {
            'auto_refresh': self.config.auto_refresh,
            'refresh_interval': self.config.refresh_interval
        })
        self.command_interface = CommandInterface(self.theme_manager)
        self.statistics_display = StatisticsDisplay(self.theme_manager, {
            'auto_refresh': self.config.auto_refresh,
            'refresh_interval': self.config.refresh_interval
        })
        
        # Application state
        self.is_running = False
        self.current_mode = self.config.interface_mode
        self.session_start_time = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.is_running = False
        print(f"\n{self.theme_manager.current_theme.colors['warning']}Shutting down gracefully...{self.theme_manager.current_theme.colors['reset']}")
    
    async def start(self) -> Dict[str, Any]:
        """Start the CLI application"""
        self.is_running = True
        self.session_start_time = asyncio.get_event_loop().time()
        
        # Apply theme configuration
        if self.config.theme_mode != "auto":
            self.theme_manager.set_theme(self.config.theme_mode)
        else:
            self.theme_manager.auto_detect_theme()
        
        # Show welcome screen
        if self.config.show_welcome:
            await self._show_welcome()
        
        # Start main application loop
        try:
            if self.current_mode == "interactive":
                await self._run_interactive_mode()
            elif self.current_mode == "dashboard":
                await self._run_dashboard_mode()
            elif self.current_mode == "stats":
                await self._run_statistics_mode()
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
        """Display beautiful welcome screen"""
        theme = self.theme_manager.current_theme
        
        # Clear screen and show cursor
        print(self.ui.clear_screen())
        print(self.ui.move_cursor(1, 1))
        
        # ASCII art header
        header_art = f"""
    {theme.colors['primary']}
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  🚀 AgentsMCP - Revolutionary Multi-Agent Orchestration Platform  ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║        ▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄   ▄▄▄   ▄▄   ▄▄▄▄▄▄▄▄▄   ║
    ║       ▐░░░░░░░░▌ ▐░░░░░░░░░▌ ▐░░░░░░░░░▌ ▐░░░▌ ▐░░▌ ▐░░░░░░░░░▌  ║
    ║       ▐░█▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀  ▐░█▀▀▀▀▀▀▀  ▐░▀░░▌▐░▀░▌  ▀▀▀▀█░█▀▀▀▀  ║
    ║       ▐░▌       ▐░▌          ▐░▌          ▐░▌▐░░▐░▌▐░▌     ▐░▌      ║
    ║       ▐░█▄▄▄▄▄█░▌▐░▌ ▄▄▄▄▄▄▄▄▐░█▄▄▄▄▄▄▄  ▐░▌ ▐░▐░▌ ▐░▌     ▐░▌      ║
    ║       ▐░░░░░░░░▌ ▐░▌▐░░░░░░░░▌▐░░░░░░░░░▌ ▐░▌  ▐▐░▌  ▐░▌     ▐░▌      ║
    ║       ▐░█▀▀▀▀▀█░▌▐░▌ ▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀  ▐░▌   ▐░▌   ▐░▌     ▐░▌      ║
    ║       ▐░▌       ▐░▌       ▐░▌▐░▌          ▐░▌    ▐░▌   ▐░▌     ▐░▌      ║
    ║       ▐░▌       ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄  ▐░▌     ▐░▌   ▐░▌     ▐░▌      ║
    ║       ▐░▌       ▐░░░░░░░░░▌ ▐░░░░░░░░░▌ ▐░▌      ▐░▌  ▐░▌     ▐░▌      ║
    ║        ▀         ▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀   ▀        ▀    ▀       ▀       ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    {theme.colors['reset']}
        """
        print(header_art)
        
        # Welcome message with features
        welcome_content = f"""
{theme.colors['secondary']}Welcome to the future of multi-agent orchestration!{theme.colors['reset']}

🎼 {theme.colors['accent']}Symphony Mode{theme.colors['reset']}: Conduct agents like a maestro
🧠 {theme.colors['accent']}Predictive Spawning{theme.colors['reset']}: AI-powered agent provisioning  
📊 {theme.colors['accent']}Real-time Analytics{theme.colors['reset']}: Beautiful metrics visualization
🎨 {theme.colors['accent']}Adaptive Themes{theme.colors['reset']}: Automatic light/dark detection
⚡ {theme.colors['accent']}Lightning Fast{theme.colors['reset']}: Sub-millisecond response times

{theme.colors['muted']}Available Modes:{theme.colors['reset']}
• {theme.colors['primary']}Interactive{theme.colors['reset']}: Full-featured command interface
• {theme.colors['primary']}Dashboard{theme.colors['reset']}: Real-time orchestration monitoring
• {theme.colors['primary']}Statistics{theme.colors['reset']}: Advanced metrics and trends

{theme.colors['warning']}Press any key to continue, or Ctrl+C to exit...{theme.colors['reset']}
        """
        
        welcome_box = self.ui.box(
            welcome_content.strip(),
            title="🎯 Getting Started",
            style='heavy',
            width=75
        )
        print(welcome_box)
        
        # Wait for user input or timeout
        try:
            import select
            if select.select([sys.stdin], [], [], 3.0)[0]:  # 3 second timeout
                sys.stdin.read(1)
        except:
            await asyncio.sleep(3)  # Fallback for non-Unix systems
        
        print(self.ui.clear_screen())
    
    async def _run_interactive_mode(self):
        """Run the interactive command interface"""
        print(self.ui.clear_screen())
        
        # Show mode header
        header = self.ui.box(
            "🎮 Interactive Command Mode - Type 'help' for available commands",
            title="Interactive Mode",
            style='light',
            width=80
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
    
    async def _show_error(self, error_message: str):
        """Display error message beautifully"""
        theme = self.theme_manager.current_theme
        
        error_box = self.ui.box(
            f"{theme.colors['error']}💥 {error_message}{theme.colors['reset']}",
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
        
        # Show goodbye message
        print(self.ui.clear_screen())
        
        goodbye_message = f"""
{theme.colors['primary']}
    ╔════════════════════════════════════════════╗
    ║  🎭 Thank you for using AgentsMCP!          ║
    ║                                            ║
    ║  🚀 Ready to orchestrate the future?       ║
    ║  💫 Your agents await your next command    ║
    ╚════════════════════════════════════════════╝
{theme.colors['reset']}

{theme.colors['secondary']}Session completed successfully.{theme.colors['reset']}
{theme.colors['muted']}May your code be bug-free and your agents be swift! ✨{theme.colors['reset']}
        """
        print(goodbye_message)
        
        # Restore cursor
        print(self.ui.show_cursor(), end='')
    
    def switch_mode(self, new_mode: str) -> bool:
        """Switch between interface modes"""
        if new_mode in ["interactive", "dashboard", "stats"]:
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
        epilog="""
Examples:
  agentsmcp                           # Start in interactive mode
  agentsmcp --mode dashboard          # Start in dashboard mode
  agentsmcp --mode stats              # Start in statistics mode
  agentsmcp --theme dark              # Force dark theme
  agentsmcp --no-welcome              # Skip welcome screen
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['interactive', 'dashboard', 'stats'],
        default='interactive',
        help='Interface mode (default: interactive)'
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
    
    # Create and start the CLI application
    app = CLIApp(config)
    
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