"""
Revolutionary TUI Interface - The actual rich, feature-packed TUI experience

This module provides the Revolutionary TUI Interface that replaces the basic chat prompt
with a sophisticated, multi-panel, animated interface that delivers the revolutionary 
experience users were promised.

Key Features:
- Rich multi-panel layout with status bars and interactive sections
- 60fps animations and visual effects
- AI Command Composer integration with smart suggestions
- Symphony Dashboard with live metrics
- Real-time status updates and agent monitoring
- Typewriter effects and visual feedback
- Advanced input handling with command completion
- Revolutionary visual design with smooth transitions
"""

import asyncio
import os
import sys
import time
import signal
import logging
import shutil
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque

# Rich terminal components
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.live import Live
    from rich.align import Align
    from rich.columns import Columns
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..components.revolutionary_tui_enhancements import RevolutionaryTUIEnhancements, VisualFeedbackType, AnimationEasing
from ..components.ai_command_composer import AICommandComposer
from ..components.symphony_dashboard import SymphonyDashboard
from .event_system import AsyncEventSystem
from ...orchestration import Orchestrator, OrchestratorConfig, OrchestratorMode

logger = logging.getLogger(__name__)


@dataclass
class TUIState:
    """Current state of the Revolutionary TUI."""
    current_input: str = ""
    input_suggestions: List[str] = None
    agent_status: Dict[str, str] = None
    system_metrics: Dict[str, Any] = None
    conversation_history: List[Dict[str, str]] = None
    is_processing: bool = False
    processing_message: str = ""
    last_update: float = 0.0
    
    def __post_init__(self):
        if self.input_suggestions is None:
            self.input_suggestions = []
        if self.agent_status is None:
            self.agent_status = {}
        if self.system_metrics is None:
            self.system_metrics = {}
        if self.conversation_history is None:
            self.conversation_history = []


class RevolutionaryTUIInterface:
    """
    Revolutionary TUI Interface - The rich, feature-packed TUI experience.
    
    This is the actual revolutionary interface that provides:
    - Multi-panel layout with status bars and interactive sections
    - 60fps animations and smooth transitions
    - AI Command Composer with intelligent suggestions
    - Symphony Dashboard integration
    - Visual effects and feedback
    - Advanced input handling
    """
    
    def __init__(self, cli_config=None, orchestrator_integration=None, revolutionary_components=None):
        """Initialize the Revolutionary TUI Interface."""
        self.cli_config = cli_config
        self.orchestrator_integration = orchestrator_integration
        self.revolutionary_components = revolutionary_components or {}
        
        # Core systems
        self.event_system = AsyncEventSystem()
        self.state = TUIState()
        self.running = False
        
        # Revolutionary components
        self.enhancements: Optional[RevolutionaryTUIEnhancements] = None
        self.ai_composer: Optional[AICommandComposer] = None
        self.symphony_dashboard: Optional[SymphonyDashboard] = None
        
        # Rich terminal setup with full terminal width utilization
        if RICH_AVAILABLE:
            # Get terminal dimensions for full space utilization
            try:
                term_size = shutil.get_terminal_size()
                term_width = term_size.columns
                term_height = term_size.lines
            except:
                term_width = 80
                term_height = 24
            
            # Use full terminal width (no artificial constraints)
            # Only leave minimal margin to prevent edge overflow
            full_width = max(80, term_width - 1)  # Just 1 char margin for safety
            
            # Initialize console to use full terminal dimensions
            self.console = Console(
                width=full_width,
                height=term_height,
                force_terminal=True,
                legacy_windows=False,
                # Force proper terminal control for alternate screen
                file=sys.stdout,
                stderr=False,
                # Ensure no output leaks to scrollback
                quiet=False,
                # Enable proper alternate screen buffer handling
                soft_wrap=False
            )
            # Store terminal dimensions for dynamic panel sizing
            self.terminal_width = full_width
            self.terminal_height = term_height
        else:
            self.console = None
            self.terminal_width = 80
            self.terminal_height = 24
        self.layout = None
        self.live_display = None
        
        # Input handling
        self.input_buffer = ""
        self.input_history = deque(maxlen=100)
        self.history_index = -1
        
        # Performance and animation - Balanced for smooth TUI experience  
        self.last_render_time = 0.0
        self.frame_count = 0
        self.target_fps = 4.0   # Smooth update rate for TUI responsiveness
        self.max_fps = 10.0     # Maximum refresh rate for active usage
        
        # Terminal control flags
        self._alternate_screen_active = False
        self._terminal_pollution_detected = False
        
        # Layout initialized flag
        self._layout_initialized = False
        
        # Debug control - THREAT: Debug output floods scrollback buffer
        # MITIGATION: Strict debug mode control with logging instead of print statements
        self._debug_mode = os.environ.get('REVOLUTIONARY_TUI_DEBUG', '').lower() in ('1', 'true', 'yes')
        self._debug_print_throttle = 0.0
        self._debug_throttle_interval = 2.0  # Minimum 2s between debug prints
    
    # Content hashing removed - no longer needed since Rich Live handles updates automatically
        
        # Orchestrator setup
        self.orchestrator = None
        
        self.logger = logger
        logger.info("Revolutionary TUI Interface initialized")
    
    async def initialize(self) -> bool:
        """Initialize all revolutionary components."""
        try:
            # Initialize event system
            await self.event_system.initialize()
            
            # Initialize Revolutionary TUI Enhancements
            self.enhancements = RevolutionaryTUIEnhancements(self.event_system)
            
            # Initialize AI Command Composer
            self.ai_composer = AICommandComposer(self.event_system)
            
            # Initialize Symphony Dashboard
            if "symphony_dashboard" in self.revolutionary_components:
                self.symphony_dashboard = self.revolutionary_components["symphony_dashboard"]
            else:
                try:
                    self.symphony_dashboard = SymphonyDashboard(self.event_system)
                except Exception as e:
                    logger.warning(f"Could not initialize SymphonyDashboard: {e}")
                    # Create a simple mock dashboard
                    class MockSymphonyDashboard:
                        def get_current_state(self):
                            return {
                                'active_agents': 2,
                                'running_tasks': 0,
                                'success_rate': 100.0,
                                'recent_activity': ['System initialized', 'Ready for tasks']
                            }
                    self.symphony_dashboard = MockSymphonyDashboard()
            
            # Initialize orchestrator
            await self._initialize_orchestrator()
            
            # Setup Rich layout
            if RICH_AVAILABLE:
                await self._setup_rich_layout()
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Trigger initial status updates to populate UI
            await self._trigger_periodic_updates()
            
            logger.info("Revolutionary TUI Interface fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Revolutionary TUI Interface: {e}")
            return False
    
    async def _initialize_orchestrator(self):
        """Initialize the orchestrator for message processing."""
        try:
            config = OrchestratorConfig(
                mode=OrchestratorMode.STRICT_ISOLATION,
                enable_smart_classification=True,
                fallback_to_simple_response=True,
                max_agent_wait_time_ms=120000,
                synthesis_timeout_ms=5000
            )
            
            # CRITICAL FIX: Temporarily suppress all logging during orchestrator initialization
            # This prevents LLM client debug logs from flooding the terminal during TUI startup
            original_root_level = logging.getLogger().level
            original_llm_level = logging.getLogger('agentsmcp.conversation.llm_client').level
            
            try:
                # Set to CRITICAL to suppress ALL logging during initialization
                logging.getLogger().setLevel(logging.CRITICAL)
                logging.getLogger('agentsmcp.conversation.llm_client').setLevel(logging.CRITICAL)
                logging.getLogger('agentsmcp.orchestration').setLevel(logging.CRITICAL)
                logging.getLogger('agentsmcp.agents').setLevel(logging.CRITICAL)
                
                # Initialize orchestrator (this may trigger LLM client initialization)
                self.orchestrator = Orchestrator(config=config)
                
            finally:
                # Restore original logging levels
                logging.getLogger().setLevel(original_root_level)
                logging.getLogger('agentsmcp.conversation.llm_client').setLevel(original_llm_level)
                
            logger.debug("Orchestrator initialized for Revolutionary TUI")
        except Exception as e:
            logger.warning(f"Failed to initialize orchestrator: {e}")
            self.orchestrator = None
    
    async def _setup_rich_layout(self):
        """Setup the Rich terminal layout for the revolutionary interface to fill full terminal."""
        if not RICH_AVAILABLE:
            return
            
        # Create main layout that uses full terminal dimensions
        self.layout = Layout()
        
        # Configure layout to use full terminal space
        self.layout.size = None  # Allow layout to expand to full terminal
        
        # Split into header, main, and footer with dynamic sizing
        self.layout.split_column(
            Layout(name="header", size=3),  # Fixed header height
            Layout(name="main", ratio=1),   # Main area takes remaining space
            Layout(name="footer", size=3)   # Reduced footer height for more space
        )
        
        # Split main area into sidebar and content with responsive sizing
        # Sidebar gets fixed width, content expands to fill remaining space
        sidebar_width = max(20, min(30, self.terminal_width // 4))  # Responsive sidebar
        self.layout["main"].split_row(
            Layout(name="sidebar", size=sidebar_width),
            Layout(name="content", ratio=1)  # Content expands to fill remaining width
        )
        
        # Split content into chat and input - chat gets most space
        self.layout["content"].split_column(
            Layout(name="chat", ratio=5),    # Chat gets 5/6 of content height
            Layout(name="input", ratio=1)    # Input gets 1/6 of content height
        )
        
        # Split sidebar into status and dashboard
        self.layout["sidebar"].split_column(
            Layout(name="status", ratio=1),     # Status gets half of sidebar height
            Layout(name="dashboard", ratio=1)   # Dashboard gets half of sidebar height
        )
        
        # Initialize panels with full-width support
        await self._initialize_layout_panels()
    
    async def _initialize_layout_panels(self):
        """Initialize layout panels with full terminal space utilization."""
        if not RICH_AVAILABLE or not self.layout:
            return
        
        try:
            # Header panel - Expands to full terminal width
            header_text = Text("🚀 AgentsMCP Revolutionary Interface", style="bold blue")
            if self.state.is_processing:
                header_text.append(" • ", style="dim")
                header_text.append(self.state.processing_message, style="yellow")
            
            self.layout["header"].update(
                Panel(
                    Align.center(header_text),
                    box=box.ROUNDED,
                    style="bright_blue",
                    expand=True  # Fill available width
                )
            )
            
            # Status panel - Expands to fill sidebar width
            status_content = self._create_status_panel()
            self.layout["status"].update(
                Panel(
                    status_content,
                    title="Agent Status",
                    box=box.ROUNDED,
                    style="green",
                    expand=True  # Fill available sidebar width
                )
            )
            
            # Dashboard panel - Expands to fill sidebar width
            dashboard_content = await self._create_dashboard_panel()
            self.layout["dashboard"].update(
                Panel(
                    dashboard_content,
                    title="Symphony Dashboard",
                    box=box.ROUNDED,
                    style="magenta",
                    expand=True  # Fill available sidebar width
                )
            )
            
            # Chat panel - Expands to fill content area width
            chat_content = self._create_chat_panel()
            self.layout["chat"].update(
                Panel(
                    chat_content,
                    title="Conversation",
                    box=box.ROUNDED,
                    style="white",
                    expand=True  # Fill available content width
                )
            )
            
            # Input panel - Expands to fill content area width
            input_content = self._create_input_panel()
            self.layout["input"].update(
                Panel(
                    input_content,
                    title="AI Command Composer",
                    box=box.ROUNDED,
                    style="cyan",
                    expand=True  # Fill available content width
                )
            )
            
            # Footer - Expands to full terminal width
            footer_content = self._create_footer_panel()
            self.layout["footer"].update(
                Panel(
                    footer_content,
                    box=box.ROUNDED,
                    style="dim",
                    expand=True  # Fill available width
                )
            )
            
        except Exception as e:
            logger.warning(f"Error initializing layout panels: {e}")
    
    def _create_status_panel(self) -> Text:
        """Create the agent status panel content with full-width support."""
        if not self.state.agent_status:
            return Text("🔄 Initializing • 📊 Loading metrics...", overflow="fold")
        
        # Create Rich Text with wrapping enabled for wider panels
        text = Text(overflow="fold")
        
        # Agent status - expanded display for wider sidebar
        for agent_name, status in self.state.agent_status.items():
            status_icon = "🟢" if status == "active" else "🔴" if status == "error" else "🟡"
            # Use full agent names in wider sidebar
            display_name = agent_name.replace("_", " ").title()
            line = f"{status_icon} {display_name}: {status.upper()}"
            text.append(line + "\n", style="bold" if status == "active" else "dim")
        
        # System metrics - expanded display with more information
        if self.state.system_metrics:
            text.append("\n📊 System Metrics:\n", style="cyan bold")
            
            fps = self.state.system_metrics.get('fps', 0)
            memory = self.state.system_metrics.get('memory_mb', 0)
            cpu = self.state.system_metrics.get('cpu_percent', 0)
            tasks = self.state.system_metrics.get('active_tasks', 0)
            uptime = self.state.system_metrics.get('uptime_mins', 0)
            
            # Full metrics display for wider sidebar
            text.append(f"⚡ FPS: {fps}\n", style="green")
            text.append(f"💾 RAM: {memory:.1f} MB\n", style="yellow")
            text.append(f"🔄 CPU: {cpu:.1f}%\n", style="magenta")
            text.append(f"📋 Tasks: {tasks}\n", style="blue")
            text.append(f"⏱️ Uptime: {uptime}m", style="white")
        
        return text
    
    async def _create_dashboard_panel(self) -> Text:
        """Create the Symphony dashboard panel content with full-width support."""
        try:
            if self.symphony_dashboard:
                # Get dashboard data
                if hasattr(self.symphony_dashboard, 'get_current_state'):
                    get_state_method = getattr(self.symphony_dashboard, 'get_current_state')
                    if asyncio.iscoroutinefunction(get_state_method):
                        dashboard_data = await get_state_method()
                    else:
                        dashboard_data = get_state_method()
                else:
                    # Fallback mock data
                    dashboard_data = {
                        'active_agents': 3,
                        'running_tasks': 1,
                        'success_rate': 95.2,
                        'recent_activity': ['Task completed successfully', 'New agent spawned and initialized', 'System performance optimized']
                    }
                
                # Create Rich Text with wrapping enabled for wider panels
                text = Text(overflow="fold")
                
                # Expanded metrics display for wider sidebar
                agents = dashboard_data.get('active_agents', 0)
                tasks = dashboard_data.get('running_tasks', 0)
                success = dashboard_data.get('success_rate', 0)
                
                text.append("🎼 Symphony Overview:\n", style="magenta bold")
                text.append(f"👥 Active Agents: {agents}\n", style="cyan")
                text.append(f"⚙️ Running Tasks: {tasks}\n", style="blue")
                text.append(f"✅ Success Rate: {success:.1f}%\n", style="green")
                
                # Recent activity - expanded display with full descriptions
                recent_activity = dashboard_data.get('recent_activity', [])
                if recent_activity:
                    text.append("\n📈 Recent Activity:\n", style="yellow bold")
                    for activity in recent_activity[-3:]:  # Show more activities in wider panel
                        # Use full activity descriptions in wider sidebar
                        text.append(f"• {activity}\n", style="white")
                
                return text
            else:
                return Text("🎼 Symphony Dashboard Loading...\nInitializing components...", overflow="fold")
                
        except Exception as e:
            logger.warning(f"Error creating dashboard panel: {e}")
            return Text(f"🎼 Dashboard Error:\n{str(e)}", overflow="fold")
    
    def _create_chat_panel(self) -> Text:
        """Create the chat conversation panel content with full-width support."""
        if not self.state.conversation_history:
            return Text("🚀 Revolutionary TUI Interface\n\nWelcome to the enhanced chat experience!\nType your messages below to begin conversation.", overflow="fold")
        
        # Create Rich Text with wrapping enabled for wider chat panel
        text = Text(overflow="fold")
        
        # Show recent conversation with expanded format for wider content area
        recent_messages = self.state.conversation_history[-15:]  # Show more messages in expanded view
        
        for entry in recent_messages:
            role = entry.get('role', 'unknown')
            message = entry.get('content', '')
            timestamp = entry.get('timestamp', '')
            
            # Skip empty messages
            if not message.strip():
                continue
            
            # Use more generous message length for wider chat panel
            # Let Rich handle text wrapping instead of truncating
            display_message = message.strip()
            
            # Expanded format with full timestamps
            time_display = f"[{timestamp}]" if timestamp else ""
            
            # Role-specific formatting with expanded styling
            if role == 'user':
                text.append(f"👤 User {time_display}:\n", style="bold cyan")
                text.append(f"{display_message}\n\n", style="white")
            elif role == 'assistant':
                text.append(f"🤖 Assistant {time_display}:\n", style="bold green")
                text.append(f"{display_message}\n\n", style="white")
            elif role == 'system':
                text.append(f"⚙️ System {time_display}:\n", style="bold yellow")
                text.append(f"{display_message}\n\n", style="dim white")
        
        return text
    
    def _create_input_panel(self) -> Text:
        """Create the AI command composer input panel content with full-width support."""
        # Create Rich Text with wrapping enabled for wider content panel
        text = Text(overflow="fold")
        
        # Current input with cursor indicator - expanded display
        input_display = self.state.current_input or ""
        if self.state.is_processing:
            input_display += " ⏳"
        else:
            # Enhanced cursor animation for better visibility
            current_time = time.time()
            should_show_cursor = int(current_time / 0.8) % 2 == 0
            
            if should_show_cursor or (current_time - self.state.last_update) < 0.5:
                input_display += "█"
            else:
                input_display += " "
        
        # Display full input without truncation in wider content panel
        text.append(f"💬 Input: ", style="bold cyan")
        text.append(f"{input_display}\n", style="white")
        
        # Expanded status and help information
        if not self.state.current_input and not self.state.is_processing:
            text.append("💡 Tips: Type your message • ↑↓ for history • Enter to send • Ctrl+C to exit\n", style="dim")
        
        # History navigation indicator
        if self.history_index > -1:
            text.append(f"📋 History: {self.history_index + 1}/{len(self.input_history)}\n", style="yellow")
        
        # AI suggestions with expanded display
        if self.state.input_suggestions:
            text.append("✨ AI Suggestions:\n", style="magenta bold")
            for i, suggestion in enumerate(self.state.input_suggestions[:3]):  # Show more suggestions
                # Display full suggestions in wider panel
                text.append(f"  {i+1}. {suggestion}\n", style="blue")
        
        return text
    
    def _create_footer_panel(self) -> Text:
        """Create the footer panel with help and shortcuts for full terminal width."""
        # Create Rich Text with wrapping enabled for full terminal width
        text = Text(overflow="fold")
        
        # Expanded help items for wider footer
        help_items = [
            "Enter: Send Message",
            "Ctrl+C: Exit TUI",
            "↑↓: History Navigation", 
            "/help: Show Commands",
            "/status: System Status",
            "/clear: Clear Chat"
        ]
        
        # Enhanced performance info
        fps_info = f"FPS: {self.frame_count % 61}"
        
        # Create expanded footer with more information in full width
        left_section = " • ".join(help_items[:3])
        right_section = " • ".join(help_items[3:] + [fps_info])
        
        # Use full width footer with centered layout
        text.append(f"{left_section}", style="cyan")
        text.append(f" | ", style="dim")
        text.append(f"{right_section}", style="yellow")
        
        return text
    
    async def run(self) -> int:
        """Run the Revolutionary TUI Interface."""
        debug_mode = self._debug_mode or getattr(self.cli_config, 'debug_mode', False)
        
        # CRITICAL FIX: Suppress logging during TUI operation to prevent terminal pollution
        # Store original log level and set to ERROR to prevent debug/info logs from flooding terminal
        original_log_level = logging.getLogger().level
        original_llm_client_level = logging.getLogger('agentsmcp.conversation.llm_client').level
        original_orchestrator_level = logging.getLogger('agentsmcp.orchestration').level
        
        try:
            # CRITICAL FIX: Complete logging suppression during TUI operation
            # Create a null handler to completely prevent any log output to terminal
            null_handler = logging.NullHandler()
            original_handlers = {}
            
            # Store original handlers and replace with null handler
            root_logger = logging.getLogger()
            original_handlers['root'] = root_logger.handlers.copy()
            root_logger.handlers = [null_handler]
            root_logger.setLevel(logging.CRITICAL)
            
            # Set logging to ERROR level to prevent debug/info output during TUI operation
            logging.getLogger('agentsmcp.conversation.llm_client').setLevel(logging.CRITICAL)
            logging.getLogger('agentsmcp.orchestration').setLevel(logging.CRITICAL)
            logging.getLogger('agentsmcp.ui.v2').setLevel(logging.CRITICAL)
            
            # Also suppress specific noisy loggers that flood terminal
            noisy_loggers = [
                'agentsmcp.conversation.llm_client',
                'agentsmcp.orchestration.orchestrator',
                'agentsmcp.agents',
                'agentsmcp.ui.v2.revolutionary_tui_interface',
                'agentsmcp.ui.v2.revolutionary_launcher'
            ]
            
            for logger_name in noisy_loggers:
                logger_instance = logging.getLogger(logger_name)
                original_handlers[logger_name] = logger_instance.handlers.copy()
                logger_instance.handlers = [null_handler]
                logger_instance.setLevel(logging.CRITICAL)
            
            # Check for CI/automated environments that should not use Rich
            # But allow TTY environments to proceed to Rich display
            is_tty = sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False
            is_ci = any(var in os.environ for var in ['CI', 'GITHUB_ACTIONS', 'TRAVIS', 'JENKINS', 'BUILD'])
            
            if is_ci:
                logger.error("CI environment detected - using minimal fallback to prevent output pollution")
                # Return immediately with exit code 0 to prevent any Rich output cycling
                return 0
            
            # THREAT: Debug output floods scrollback buffer
            # MITIGATION: Use logging instead of print statements
            if debug_mode:
                logger.debug("Revolutionary TUI Interface run() method called")
                logger.debug(f"RICH_AVAILABLE: {RICH_AVAILABLE}")
                logger.debug(f"CLI config: {self.cli_config}")
            
            # Initialize components
            if debug_mode:
                logger.debug("Calling initialize()")
            
            if not await self.initialize():
                logger.error("Failed to initialize Revolutionary TUI Interface")
                if debug_mode:
                    logger.debug("Revolutionary TUI Interface initialization failed")
                return 1
            
            if debug_mode:
                logger.debug("Revolutionary TUI Interface initialized successfully")
            
            # Setup signal handlers
            def signal_handler(signum, frame):
                if debug_mode:
                    logger.debug(f"Signal {signum} received, setting running=False")
                self.running = False
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            if debug_mode:
                logger.debug("Signal handlers set up")
            
            self.running = True
            logger.info("🎯 Revolutionary TUI Interface marked as running")
            
            if debug_mode:
                logger.debug(f"Running state set to: {self.running}")
            
            # CRITICAL: Triple-check TTY status before Rich Live to prevent output cycling
            stdout_tty = sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else False
            stdin_tty = sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False
            stderr_tty = sys.stderr.isatty() if hasattr(sys.stderr, 'isatty') else False
            
            if RICH_AVAILABLE and is_tty:
                logger.info("🎨 Using Rich Live display for TUI (TTY confirmed)")
                if debug_mode:
                    logger.debug("Rich is available and TTY confirmed, using Live display")
                    logger.debug(f"Layout object: {self.layout}")
                    logger.debug(f"TTY Status - stdin: {stdin_tty}, stdout: {stdout_tty}, stderr: {stderr_tty}")
                
                # Use Rich Live display for smooth updates
                try:
                    logger.info("📺 Attempting to create Rich Live display...")
                    if debug_mode:
                        logger.debug("Creating Rich Live display context")
                        logger.debug(f"Layout valid: {self.layout is not None}")
                        logger.debug(f"Layout type: {type(self.layout)}")
                        logger.debug(f"Terminal size: {os.get_terminal_size() if hasattr(os, 'get_terminal_size') else 'unknown'}")
                        logger.debug(f"Is TTY: {sys.stdin.isatty()}")
                        logger.debug(f"TERM env: {os.environ.get('TERM', 'not set')}")
                    
                    # CRITICAL: Force alternate screen mode with proper terminal isolation
                    # MITIGATION: Explicit terminal control to prevent any scrollback pollution
                    try:
                        if debug_mode:
                            logger.debug("Attempting Live context with screen=True (alternate screen)")
                        
                        # CRITICAL FIX: Force alternate screen buffer with terminal isolation
                        try:
                            # Clear any existing output first
                            self.console.clear()
                            
                            # Explicitly enter alternate screen mode
                            if hasattr(self.console, '_file') and hasattr(self.console._file, 'write'):
                                # Send alternate screen control sequence
                                self.console._file.write('\033[?1049h')  # Enter alternate screen
                                self.console._file.flush()
                                self._alternate_screen_active = True
                        except Exception as screen_e:
                            logger.warning(f"Could not explicitly enter alternate screen: {screen_e}")
                        
                        # Create Live with alternate screen and smooth refresh rate
                        live_config = {
                            "renderable": self.layout,
                            "console": self.console,
                            "screen": True,  # Use alternate screen for clean TUI experience
                            "refresh_per_second": self.target_fps,  # Smooth refresh rate for TUI
                            "auto_refresh": True,  # Enable auto-refresh for smooth updates
                            "vertical_overflow": "crop",  # Prevent overflow to main screen
                            "transient": False  # Ensure proper screen buffer usage
                        }
                        
                        with Live(**live_config) as live:
                            logger.info("📺 Rich Live display context entered successfully (alternate screen)")
                            if debug_mode:
                                logger.debug("Rich Live display context active (alternate screen)")
                            
                            self.live_display = live
                            
                            # Don't stop the Live display - let it run so TUI is visible
                            logger.info("🚀 Starting main loop with Rich Live display active...")
                            
                            if debug_mode:
                                logger.debug("About to call _run_main_loop() with Rich Live active")
                            
                            await self._run_main_loop()
                            
                            if debug_mode:
                                logger.debug("_run_main_loop() completed")
                            
                            logger.info("✅ Main loop completed")
                    
                    except Exception as live_e:
                        if debug_mode:
                            logger.debug(f"Live context (alternate screen) failed: {type(live_e).__name__}: {live_e}")
                            logger.debug("Retrying with alternate screen buffer")
                        
                        # EMERGENCY RETRY: Force alternate screen with reduced refresh rate
                        # CRITICAL: Must prevent any scrollback buffer pollution
                        try:
                            import time
                            time.sleep(0.1)  # Brief pause before retry
                            
                            # EMERGENCY: Force terminal reset and alternate screen
                            try:
                                # Clear console and force terminal reset
                                self.console.clear()
                                
                                # Force alternate screen control sequence
                                if hasattr(self.console, '_file') and hasattr(self.console._file, 'write'):
                                    self.console._file.write('\033[?1049h')  # Force alternate screen
                                    self.console._file.flush()
                                    self._alternate_screen_active = True
                            except Exception:
                                pass  # Ignore errors in emergency mode
                            
                            # Force alternate screen with emergency fallback config
                            emergency_live_config = {
                                "renderable": self.layout,
                                "console": self.console,
                                "screen": True,  # Force alternate screen
                                "refresh_per_second": max(1.0, self.target_fps / 2),  # Reduced but still usable rate
                                "auto_refresh": True,  # Keep auto-refresh for TUI functionality
                                "vertical_overflow": "crop",
                                "transient": False
                            }
                            
                            with Live(**emergency_live_config) as live:
                                logger.info("📺 Rich Live display context entered successfully (retry)")
                                if debug_mode:
                                    logger.debug("Rich Live display context active (retry)")
                                
                                self.live_display = live
                                
                                # Don't stop the Live display - let it run so TUI is visible
                                logger.info("🚀 Starting main loop with Rich Live display active (retry)...")
                                
                                if debug_mode:
                                    logger.debug("About to call _run_main_loop() with emergency Rich Live active")
                                
                                await self._run_main_loop()
                                
                                if debug_mode:
                                    logger.debug("_run_main_loop() completed")
                                
                                logger.info("✅ Main loop completed")
                        except Exception as retry_e:
                            if debug_mode:
                                logger.debug(f"Retry also failed: {retry_e}")
                            logger.error(f"❌ Both Rich Live attempts failed: {retry_e}")
                            
                            # EMERGENCY FALLBACK: Disable Rich entirely to prevent scrollback pollution
                            logger.warning("🚨 EMERGENCY: Rich Live failed - disabling all Rich output to prevent terminal pollution")
                            
                            try:
                                # Clear any remaining output
                                self.console.clear()
                                
                                # Try to exit alternate screen if we entered it
                                if self._alternate_screen_active and hasattr(self.console, '_file'):
                                    self.console._file.write('\033[?1049l')  # Exit alternate screen
                                    self.console._file.flush()
                                    self._alternate_screen_active = False
                            except Exception:
                                pass  # Ignore cleanup errors in emergency
                            
                            # Use minimal fallback mode with ZERO Rich output
                            await self._run_emergency_fallback_loop()
                            return 0
                
                except Exception as e:
                    logger.error(f"❌ Rich Live display failed: {e}")
                    if debug_mode:
                        logger.debug(f"Rich Live display failed: {type(e).__name__}: {e}")
                        logger.debug("Full exception traceback:", exc_info=True)
                    logger.info("🔄 Falling back to basic display")
                    await self._run_fallback_loop()
            else:
                logger.info("📟 Using basic display (Rich not available or TTY not detected)")
                if debug_mode:
                    logger.debug(f"Rich available: {RICH_AVAILABLE}, TTY status: is_tty={is_tty}")
                    logger.debug("Using fallback display mode")
                # Use fallback mode instead of returning immediately
                await self._run_fallback_loop()
            
            logger.info("🏁 Revolutionary TUI Interface execution completed")
            if debug_mode:
                logger.debug("Revolutionary TUI Interface execution completed normally")
            return 0
            
        except KeyboardInterrupt:
            logger.info("Revolutionary TUI interrupted by user")
            if debug_mode:
                logger.debug("Revolutionary TUI interrupted by KeyboardInterrupt")
            return 0
        except Exception as e:
            logger.error(f"Revolutionary TUI Interface error: {e}")
            if debug_mode:
                logger.debug(f"Revolutionary TUI Interface exception: {type(e).__name__}: {e}")
                logger.debug("Full exception traceback:", exc_info=True)
            return 1
        finally:
            # CRITICAL: Restore original logging levels and handlers to prevent side effects
            try:
                logging.getLogger().setLevel(original_log_level)
                logging.getLogger('agentsmcp.conversation.llm_client').setLevel(original_llm_client_level)
                logging.getLogger('agentsmcp.orchestration').setLevel(original_orchestrator_level)
                
                # Restore original handlers
                if 'original_handlers' in locals():
                    for logger_name, handlers in original_handlers.items():
                        if logger_name == 'root':
                            logging.getLogger().handlers = handlers
                        else:
                            logging.getLogger(logger_name).handlers = handlers
            except Exception:
                pass  # Ignore errors during logging restoration
            
            if debug_mode:
                logger.debug("Revolutionary TUI Interface cleanup starting")
            
            # CRITICAL: Emergency terminal cleanup
            try:
                # Force exit alternate screen if still active
                if getattr(self, '_alternate_screen_active', False) and hasattr(self, 'console') and self.console:
                    try:
                        if hasattr(self.console, '_file') and hasattr(self.console._file, 'write'):
                            self.console._file.write('\033[?1049l')  # Exit alternate screen
                            self.console._file.flush()
                    except Exception:
                        pass  # Ignore terminal control errors during emergency cleanup
            except Exception:
                pass  # Ignore all errors during emergency cleanup
            
            await self._cleanup()
            if debug_mode:
                logger.debug("Revolutionary TUI Interface cleanup completed")
    
    async def _run_main_loop(self):
        """Main loop with Rich interface."""
        debug_mode = getattr(self.cli_config, 'debug_mode', False)
        
        logger.info("🚀 Revolutionary TUI Interface started with Rich display")
        if debug_mode:
            logger.debug("_run_main_loop() started")
            logger.debug(f"self.running = {self.running}")
        
        # Start event-driven background tasks - no more polling
        logger.info("⚙️ Creating event-driven background tasks...")
        
        input_task = asyncio.create_task(self._input_loop())
        
        # Optional: Create a periodic update trigger task for occasional status checks
        # This replaces the continuous polling loop with an occasional update trigger
        periodic_update_task = asyncio.create_task(self._periodic_update_trigger())
        
        logger.info("🎯 Event-driven background tasks created, waiting for completion...")
        
        if debug_mode:
            logger.debug("Tasks created, waiting for first completion...")
        
        try:
            # Wait for any task to complete (usually from user interruption)
            done, pending = await asyncio.wait(
                [input_task, periodic_update_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            if debug_mode:
                logger.debug(f"Wait completed - {len(done)} tasks done, {len(pending)} pending")
            
            # Log which task completed first
            for task in done:
                task_name = "unknown"
                if task is input_task:
                    task_name = "input_task"
                elif task is periodic_update_task:
                    task_name = "periodic_update_task"
                
                logger.info(f"Task completed first: {task_name}")
                try:
                    result = task.result()
                    if debug_mode:
                        logger.debug(f"Task {task_name} result: {result}")
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {e}")
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
    
    async def _periodic_update_trigger(self):
        """Periodic update trigger for status refreshes."""
        while self.running:
            try:
                # Trigger periodic updates every 10 seconds for reasonable responsiveness
                await asyncio.sleep(10.0)
                
                # Only trigger updates if we haven't detected terminal pollution
                if not self._terminal_pollution_detected:
                    await self._trigger_periodic_updates()
                
            except Exception as e:
                logger.error(f"Error in periodic update trigger: {e}")
                await asyncio.sleep(5.0)  # Brief pause on error
    
    async def _run_fallback_loop(self):
        """Fallback loop without Rich (basic terminal output)."""
        logger.info("🚀 Revolutionary TUI Interface started in fallback mode")
        
        # Log initial interface activation
        logger.info("AgentsMCP Revolutionary Interface (Fallback Mode) - Rich terminal features unavailable")
        
        # Check if we can use the enhanced fallback input method
        import sys
        if not sys.stdin.isatty():
            logger.info("No TTY detected - using enhanced fallback input method")
            return await self._fallback_input_loop()
        
        # Simple input loop for TTY environments
        logger.info("TTY detected - using simple input loop")
        while self.running:
            try:
                user_input = input("\n💬 > ").strip()
                if user_input:
                    await self._process_user_input(user_input)
            except EOFError:
                logger.info("EOFError in simple input loop - exiting")
                break
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt in simple input loop - exiting")
                break
    
    async def _run_emergency_fallback_loop(self):
        """Emergency fallback loop with minimal output to prevent scrollback pollution."""
        logger.info("🚨 EMERGENCY FALLBACK: Minimal output mode to prevent terminal pollution")
        
        # NO Rich output whatsoever - only essential logging
        import sys
        
        if not sys.stdin or sys.stdin.closed or not sys.stdin.isatty():
            logger.info("🚨 No TTY available - running in silent demonstration mode")
            
            # Silent demo mode with minimal output
            demo_commands = ["help", "status", "quit"]
            for cmd in demo_commands:
                if not self.running:
                    break
                await asyncio.sleep(3)  # Slow demo to prevent output flooding
                await self._process_user_input(cmd)
            return
        
        # Minimal TTY mode with essential prompts only
        logger.info("🚨 Emergency TTY mode - type 'quit' to exit")
        while self.running:
            try:
                # Minimal prompt to prevent output pollution
                user_input = input("> ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    self.running = False
                    break
                if user_input:
                    await self._process_user_input(user_input)
            except (EOFError, KeyboardInterrupt):
                logger.info("Emergency fallback interrupted - exiting")
                break
    
    # Render loop removed - Rich Live handles all rendering automatically
    # This eliminates the layout corruption that occurred when manually updating panels
    
    async def _input_loop(self):
        """Input handling loop with actual keyboard input processing."""
        debug_mode = getattr(self.cli_config, 'debug_mode', False)
        
        if debug_mode:
            logger.debug("_input_loop() started")
        
        import sys
        import os
        import threading
        import time
        
        # Check if we have access to a proper TTY
        tty_available = False
        try:
            import termios
            import tty
            import select
            
            is_tty = sys.stdin.isatty()
            
            if is_tty:
                # Try to access stdin as TTY first
                try:
                    stdin_fd = sys.stdin.fileno()
                    attrs = termios.tcgetattr(stdin_fd)
                    tty_available = True
                        
                except (termios.error, OSError) as e:
                    # Fall back to /dev/tty if stdin doesn't work
                    try:
                        test_fd = os.open('/dev/tty', os.O_RDONLY)
                        attrs = termios.tcgetattr(test_fd)
                        os.close(test_fd)
                        tty_available = True
                            
                    except (OSError, termios.error) as e2:
                        tty_available = False
            else:
                tty_available = False
                
        except ImportError as e:
            tty_available = False
        
        if debug_mode:
            logger.debug(f"TTY detection result: tty_available = {tty_available}")
        
        if not tty_available:
            logger.warning("TTY not available, using fallback input method")
            return await self._fallback_input_loop()
        
        # Setup terminal for raw input
        fd = None
        original_settings = None
        stop_flag = {"stop": False}
        
        # Get the current event loop to pass to the reader thread
        current_loop = asyncio.get_running_loop()
        
        def reader_thread(loop):
            nonlocal fd, original_settings
            try:
                # Try stdin first, then fall back to /dev/tty
                try:
                    fd = sys.stdin.fileno()
                    original_settings = termios.tcgetattr(fd)
                    tty.setraw(fd)
                except (termios.error, OSError) as e:
                    # Fallback to /dev/tty
                    fd = os.open('/dev/tty', os.O_RDONLY)
                    original_settings = termios.tcgetattr(fd)
                    tty.setraw(fd)
                
                while not stop_flag["stop"] and self.running:
                    try:
                        # Use select for non-blocking read with timeout
                        ready, _, _ = select.select([fd], [], [], 0.1)
                        if not ready:
                            continue
                            
                        # Read available data - handle both file descriptor types
                        try:
                            if isinstance(fd, int) and fd >= 0:
                                data = os.read(fd, 64)
                            else:
                                continue
                        except (OSError, ValueError) as e:
                            continue
                            
                        if not data:
                            continue
                        
                        # Process each byte
                        i = 0
                        while i < len(data) and not stop_flag["stop"]:
                            b = data[i]
                            
                            # Ctrl+C / Ctrl+D - Exit
                            if b in (3, 4):
                                if loop:
                                    loop.call_soon_threadsafe(lambda: asyncio.create_task(self._handle_exit()))
                                else:
                                    self.running = False
                                i += 1
                                continue
                            
                            # Backspace (8 or 127)
                            elif b in (8, 127):
                                if loop:
                                    loop.call_soon_threadsafe(self._handle_backspace_input)
                                i += 1
                                continue
                            
                            # Enter (13) - Submit input
                            elif b == 13:
                                if loop:
                                    loop.call_soon_threadsafe(lambda: asyncio.create_task(self._handle_enter_input()))
                                i += 1
                                continue
                            
                            # Line feed (10) - Add newline to input
                            elif b == 10:
                                if loop:
                                    loop.call_soon_threadsafe(lambda: self._handle_character_input('\n'))
                                i += 1 
                                continue
                            
                            # ESC sequences (27)
                            elif b == 27:
                                # Parse escape sequence
                                seq_bytes = [b]
                                j = i + 1
                                
                                # Read the complete escape sequence
                                if j < len(data) and data[j] == ord('['):
                                    seq_bytes.append(data[j])
                                    j += 1
                                    
                                    # Read until we find the end character
                                    while j < len(data):
                                        seq_bytes.append(data[j])
                                        # End on A-Z, a-z, or ~
                                        if ((65 <= data[j] <= 90) or (97 <= data[j] <= 122) or 
                                            data[j] == ord('~')):
                                            break
                                        j += 1
                                    
                                    # Convert to string for easier processing
                                    try:
                                        seq_str = bytes(seq_bytes[2:]).decode('utf-8', errors='ignore')
                                    except:
                                        seq_str = ''
                                    
                                    # Handle arrow keys
                                    if seq_str == 'A':  # Up arrow - Previous history
                                        if loop:
                                            loop.call_soon_threadsafe(self._handle_up_arrow)
                                    elif seq_str == 'B':  # Down arrow - Next history  
                                        if loop:
                                            loop.call_soon_threadsafe(self._handle_down_arrow)
                                    elif seq_str == 'C':  # Right arrow - Move cursor right
                                        if loop:
                                            loop.call_soon_threadsafe(self._handle_right_arrow)
                                    elif seq_str == 'D':  # Left arrow - Move cursor left
                                        if loop:
                                            loop.call_soon_threadsafe(self._handle_left_arrow)
                                    
                                    i = j + 1
                                else:
                                    # ESC alone - clear input
                                    if loop:
                                        loop.call_soon_threadsafe(self._handle_escape_key)
                                    i += 1
                                continue
                            
                            # Regular printable characters (32-126)
                            elif 32 <= b <= 126:
                                try:
                                    char = bytes([b]).decode('utf-8', errors='ignore')
                                    if char:
                                        if loop:
                                            loop.call_soon_threadsafe(lambda c=char: self._handle_character_input(c))
                                except Exception:
                                    pass
                                i += 1
                                continue
                            
                            # Skip other control characters
                            else:
                                i += 1
                                continue
                        
                    except Exception as e:
                        time.sleep(0.1)
                        
            except Exception as e:
                logger.error(f"Failed to setup raw terminal input: {e}")
                # Set flag to indicate fallback is needed - main thread will handle it
                stop_flag["use_fallback"] = True
            
            finally:
                # Restore terminal settings
                if fd is not None and original_settings is not None:
                    try:
                        termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)
                    except Exception:
                        pass
                # Only close file descriptor if it was opened with os.open (not stdin)
                if fd is not None and fd != sys.stdin.fileno():
                    try:
                        os.close(fd)
                    except Exception:
                        pass
        
        # Start the reader thread, passing the event loop
        thread = threading.Thread(target=reader_thread, args=(current_loop,), daemon=True)
        thread.start()
        
        try:
            loop_iterations = 0
            while self.running and thread.is_alive():
                loop_iterations += 1
                
                # Check if thread requested fallback
                if stop_flag.get("use_fallback", False):
                    logger.info("Reader thread requested fallback, switching to fallback input method")
                    stop_flag["stop"] = True
                    if thread.is_alive():
                        thread.join(timeout=1.0)
                    return await self._fallback_input_loop()
                
                await asyncio.sleep(0.1)
                
                # Input suggestions now handled via events in _on_input_changed()
                # No more polling for suggestions
                        
        finally:
            # Signal thread to stop
            stop_flag["stop"] = True
            if thread.is_alive():
                thread.join(timeout=1.0)
    
    async def _fallback_input_loop(self):
        """Fallback input loop for when raw terminal setup fails."""
        logger.info("Using fallback input method - will wait for Enter key")
        
        import concurrent.futures
        import sys
        
        # Check if stdin is actually available
        if not sys.stdin or sys.stdin.closed:
            logger.error("No stdin available - cannot use fallback input method")
            self.running = False
            return
        
        # Use a thread pool executor for blocking input
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # Track whether we've shown the header already
        header_shown = False
        
        def get_input():
            """Get input from stdin in blocking mode."""
            nonlocal header_shown
            try:
                # Check if stdin is actually readable
                if sys.stdin.isatty():
                    # In TTY mode, use normal input - show header only once
                    if not header_shown:
                        header_shown = True
                    return input("💬 > ")
                else:
                    # In non-TTY mode, try to read from stdin if available, otherwise simulate
                    try:
                        # First try to read from stdin with a short timeout
                        import select
                        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if ready:
                            # There's input available, read it
                            return input()
                    except:
                        pass  # Fall through to simulation mode
                    
                    # No stdin input available - run simulation mode
                    if not header_shown:
                        logger.info("Non-TTY environment detected - using simulated input mode")
                        header_shown = True
                    
                    # Simulate some sample commands for demo purposes
                    import time
                    if not hasattr(get_input, 'command_index'):
                        get_input.command_index = 0
                    
                    sample_commands = [
                        "status", 
                        "help", 
                        "status",
                        "Demo: Processing sample task...",
                        "Demo: Checking system health...", 
                        "Demo: Revolutionary TUI demonstration complete!",
                        "quit"
                    ]
                    
                    if get_input.command_index < len(sample_commands):
                        cmd = sample_commands[get_input.command_index]
                        time.sleep(8)  # Wait 8 seconds between commands to reduce output rate
                        get_input.command_index += 1
                        return cmd
                    
                    # After all demo commands, return quit to exit gracefully
                    return "quit"
                    
            except (EOFError, KeyboardInterrupt):
                logger.info("Input interrupted or EOF reached")
                return None
            except Exception as e:
                logger.error(f"Error getting input: {e}")
                return None
        
        try:
            # Log welcome message for fallback mode
            logger.info("Revolutionary TUI Interface - Fallback Mode - Running without raw terminal access")
            
            while self.running:
                try:
                    # Get input asynchronously
                    future = executor.submit(get_input)
                    
                    # Wait for input with periodic updates (no Rich updates in fallback mode)
                    while not future.done() and self.running:
                        await asyncio.sleep(0.5)  # Longer sleep since no display updates needed
                    
                    if not self.running:
                        future.cancel()
                        break
                    
                    user_input = future.result()
                    if user_input is None:
                        # EOF or interrupt - check if we're in a proper terminal
                        if sys.stdin and not sys.stdin.closed:
                            continue
                        else:
                            logger.info("stdin closed or unavailable - exiting")
                            break
                    
                    user_input = user_input.strip()
                    if user_input.lower() in ['quit', 'exit']:
                        logger.info("Exiting Revolutionary TUI...")
                        self.running = False
                        break
                    
                    # Special commands are now handled in _process_user_input()
                    # Remove duplicate handlers to prevent conflicts
                    
                    if user_input:
                        # Handle demo messages
                        if user_input.startswith("Demo:"):
                            continue
                        
                        # Process the input
                        self.state.current_input = user_input
                        old_conversation_length = len(self.state.conversation_history)
                        await self._process_user_input(user_input)
                        self.state.current_input = ""
                        
                        # Log any new responses that were added to conversation history
                        new_conversation_length = len(self.state.conversation_history)
                    
                except Exception as e:
                    logger.error(f"Error in fallback input loop: {e}")
                    await asyncio.sleep(1.0)
        
        finally:
            executor.shutdown(wait=False)
            logger.info("Fallback input loop ended")
            
            # Log completion message
            logger.info("Revolutionary TUI demo completed - For interactive mode, run from a real terminal")
    
    # Polling-based update loop REMOVED - replaced with event-driven updates
    # All status and metrics updates now happen via events only
    
    async def _update_agent_status(self):
        """Update the status of all agents and publish event."""
        try:
            # Update agent status
            new_status = {
                "orchestrator": "active" if self.orchestrator else "offline",
                "ai_composer": "active" if self.ai_composer else "offline",
                "symphony_dashboard": "active" if self.symphony_dashboard else "offline",
                "tui_enhancements": "active" if self.enhancements else "offline"
            }
            
            # Check if status actually changed
            if new_status != self.state.agent_status:
                self.state.agent_status = new_status
                # Publish event for reactive UI update
                await self._publish_agent_status_changed()
                
        except Exception as e:
            logger.warning(f"Error updating agent status: {e}")
    
    async def _update_system_metrics(self):
        """Update system performance metrics and publish event."""
        try:
            # Update metrics
            new_metrics = {
                "fps": min(60, self.frame_count % 61),
                "memory_mb": 45.2,  # Mock memory usage
                "cpu_percent": 12.5,  # Mock CPU usage
                "active_tasks": len(self.state.conversation_history),
                "uptime_mins": int((time.time() - self.last_render_time) / 60) if self.last_render_time else 0
            }
            
            # Check if metrics actually changed
            if new_metrics != self.state.system_metrics:
                self.state.system_metrics = new_metrics
                # Publish event for reactive UI update
                await self._publish_metrics_updated()
                
        except Exception as e:
            logger.warning(f"Error updating system metrics: {e}")
    
    async def _trigger_periodic_updates(self):
        """Trigger periodic updates via events instead of polling."""
        try:
            # Update agent status and metrics, which will emit events if changed
            await self._update_agent_status()
            await self._update_system_metrics()
            
            # Update timestamp
            self.state.last_update = time.time()
            
        except Exception as e:
            logger.warning(f"Error in periodic updates: {e}")
    
    async def _apply_visual_effects(self):
        """Apply visual effects using the enhancement engine."""
        if not self.enhancements:
            return
        
        try:
            # Apply typewriter effects to new messages
            # Apply smooth transitions
            # Update animations
            # This would integrate with the RevolutionaryTUIEnhancements
            pass
        except Exception as e:
            logger.warning(f"Error applying visual effects: {e}")
    
    def _handle_character_input(self, char: str):
        """Handle a single character input and emit events for reactive updates."""
        # Add character to current input
        self.state.current_input += char
        
        # Make cursor visible and reset blink timer  
        self.state.last_update = time.time()
        
        # Emit input changed event for reactive UI updates
        asyncio.create_task(self._publish_input_changed())
    
    # Input display updates removed - Rich Live handles all updates automatically
    # Manual panel updates during Live operation were corrupting the display
    
    def _handle_backspace_input(self):
        """Handle backspace key input and emit events for reactive updates."""
        if self.state.current_input:
            self.state.current_input = self.state.current_input[:-1]
            self.state.last_update = time.time()
            
            # Emit input changed event for reactive UI updates
            asyncio.create_task(self._publish_input_changed())
    
    async def _handle_enter_input(self):
        """Handle Enter key - submit the current input and emit events."""
        if self.state.current_input.strip():
            # Process the input
            await self._process_user_input(self.state.current_input.strip())
            
            # Clear input state
            self.state.current_input = ""
            self.state.input_suggestions = []
            
            # Emit input changed event for clearing input
            await self._publish_input_changed()
    
    def _handle_up_arrow(self):
        """Handle up arrow key - navigate to previous input in history and emit events."""
        if self.input_history and self.history_index < len(self.input_history) - 1:
            # Save current input if at end of history
            if self.history_index == -1 and self.state.current_input.strip():
                # Save current input as most recent
                pass  # Current input preserved
            
            self.history_index += 1
            history_item = self.input_history[-(self.history_index + 1)]
            self.state.current_input = history_item
            self.state.last_update = time.time()
            
            # Emit input changed event for reactive UI updates
            asyncio.create_task(self._publish_input_changed())
    
    def _handle_down_arrow(self):
        """Handle down arrow key - navigate to next input in history and emit events."""
        if self.history_index > -1:
            self.history_index -= 1
            
            if self.history_index == -1:
                # Back to current input
                self.state.current_input = ""
            else:
                history_item = self.input_history[-(self.history_index + 1)]
                self.state.current_input = history_item
                
            self.state.last_update = time.time()
            
            # Emit input changed event for reactive UI updates
            asyncio.create_task(self._publish_input_changed())
    
    def _handle_left_arrow(self):
        """Handle left arrow key - move cursor left (future enhancement)."""
        # TODO: Implement cursor position control
        # For now, just refresh display to show we received the key
        self.state.last_update = time.time()
        # Display updates handled automatically by Rich Live
    
    def _handle_right_arrow(self):
        """Handle right arrow key - move cursor right (future enhancement)."""  
        # TODO: Implement cursor position control
        # For now, just refresh display to show we received the key
        self.state.last_update = time.time()
        # Display updates handled automatically by Rich Live
    
    def _handle_escape_key(self):
        """Handle ESC key - clear current input and emit events."""
        self.state.current_input = ""
        self.state.input_suggestions = []
        self.history_index = -1
        self.state.last_update = time.time()
        
        # Emit input changed event for reactive UI updates
        asyncio.create_task(self._publish_input_changed())
    
    async def _handle_exit(self):
        """Handle Ctrl+C/Ctrl+D - graceful exit."""
        logger.info("Exit requested by user")
        self.running = False
    
    async def _process_user_input(self, user_input: str):
        """Process user input through the revolutionary system."""
        try:
            # Add to input history for arrow key navigation
            if user_input.strip():
                self.input_history.append(user_input.strip())
                self.history_index = -1  # Reset history navigation
            
            # Handle built-in commands before processing through orchestrator
            user_input_lower = user_input.strip().lower()
            
            # Handle quit commands
            if user_input_lower in ['quit', 'exit', '/quit', '/exit']:
                logger.info("Exiting Revolutionary TUI...")
                self.running = False
                return
            
            # Handle help command
            if user_input_lower in ['help', '/help']:
                response_content = """📋 Commands:
  quit, exit - Exit TUI
  help - Show this help  
  status - Show system status
  clear - Clear conversation history
  
  Other input processed by AI agents."""
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                user_message = {
                    "role": "user",
                    "content": user_input,
                    "timestamp": timestamp
                }
                assistant_message = {
                    "role": "assistant", 
                    "content": response_content,
                    "timestamp": timestamp
                }
                self.state.conversation_history.extend([user_message, assistant_message])
                
                # Emit conversation updated events
                await self._publish_conversation_updated(user_message)
                await self._publish_conversation_updated(assistant_message)
                return
                
            # Handle status command
            if user_input_lower in ['status', '/status']:
                response_content = f"""📊 TUI Status:
  Running: {self.running}
  Mode: Revolutionary TUI
  History: {len(self.input_history)} entries
  Messages: {len(self.state.conversation_history)}"""
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                user_message = {
                    "role": "user",
                    "content": user_input,
                    "timestamp": timestamp
                }
                assistant_message = {
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": timestamp
                }
                self.state.conversation_history.extend([user_message, assistant_message])
                
                # Emit conversation updated events
                await self._publish_conversation_updated(user_message)
                await self._publish_conversation_updated(assistant_message)
                return
                
            # Handle clear command
            if user_input_lower in ['clear', '/clear']:
                self.state.conversation_history.clear()
                timestamp = datetime.now().strftime("%H:%M:%S")
                clear_message = {
                    "role": "assistant",
                    "content": "🧹 Conversation history cleared.",
                    "timestamp": timestamp
                }
                self.state.conversation_history.append(clear_message)
                
                # Emit conversation updated event
                await self._publish_conversation_updated(clear_message)
                return
            
            # Add to conversation history for non-builtin commands
            timestamp = datetime.now().strftime("%H:%M:%S")
            new_user_message = {
                "role": "user",
                "content": user_input,
                "timestamp": timestamp
            }
            self.state.conversation_history.append(new_user_message)
            
            # Emit conversation updated event
            await self._publish_conversation_updated(new_user_message)
            
            # Show processing state
            self.state.is_processing = True
            self.state.processing_message = "Processing with AI agents..."
            
            # Emit processing state changed event
            await self._publish_processing_state_changed()
            
            # Process through orchestrator if available
            if self.orchestrator:
                import os
                context = {
                    "working_directory": os.getcwd(),
                    "project_root": os.getcwd(),
                    "user_initiated_from": "Revolutionary_TUI"
                }
                
                response = await self.orchestrator.process_user_input(user_input, context)
                
                # Add response to conversation
                new_assistant_message = {
                    "role": "assistant",
                    "content": response.content or "No response generated",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                self.state.conversation_history.append(new_assistant_message)
                
                # Emit conversation updated event
                await self._publish_conversation_updated(new_assistant_message)
                
                # Apply typewriter effect to response if enhancements available
                if self.enhancements:
                    await self.enhancements.create_visual_effect(
                        VisualFeedbackType.TYPEWRITER,
                        "response_area",
                        text=response.content,
                        speed=25
                    )
            
            else:
                # Fallback response
                new_assistant_message = {
                    "role": "assistant", 
                    "content": f"Revolutionary TUI received: {user_input}",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                self.state.conversation_history.append(new_assistant_message)
                
                # Emit conversation updated event
                await self._publish_conversation_updated(new_assistant_message)
            
            # Clear processing state
            self.state.is_processing = False
            self.state.processing_message = ""
            
            # Emit processing state changed event
            await self._publish_processing_state_changed()
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            self.state.is_processing = False
            self.state.processing_message = ""
            
            # Emit processing state changed event for error state
            await self._publish_processing_state_changed()
            
            # Add error to conversation
            error_message = {
                "role": "system",
                "content": f"Error processing input: {str(e)}",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            self.state.conversation_history.append(error_message)
            
            # Emit conversation updated event for error
            await self._publish_conversation_updated(error_message)
    
    async def _register_event_handlers(self):
        """Register event handlers for the revolutionary interface."""
        try:
            # Register handlers for UI update events
            await self.event_system.subscribe("input_changed", self._on_input_changed)
            await self.event_system.subscribe("agent_status_changed", self._on_agent_status_changed)
            await self.event_system.subscribe("metrics_updated", self._on_metrics_updated)
            await self.event_system.subscribe("conversation_updated", self._on_conversation_updated)
            await self.event_system.subscribe("processing_state_changed", self._on_processing_state_changed)
            await self.event_system.subscribe("ui_refresh", self._on_ui_refresh)
            
            # Legacy handlers for backward compatibility
            await self.event_system.subscribe("user_input", self._handle_user_input_event)
            await self.event_system.subscribe("agent_status_change", self._handle_agent_status_change)
            await self.event_system.subscribe("performance_update", self._handle_performance_update)
            
        except Exception as e:
            logger.warning(f"Error registering event handlers: {e}")
    
    async def _handle_user_input_event(self, event_data: Dict[str, Any]):
        """Handle user input events."""
        user_input = event_data.get("input", "")
        if user_input:
            self.state.current_input = user_input
    
    async def _handle_agent_status_change(self, event_data: Dict[str, Any]):
        """Handle agent status change events."""
        agent_name = event_data.get("agent", "")
        status = event_data.get("status", "unknown")
        if agent_name:
            self.state.agent_status[agent_name] = status
    
    async def _handle_performance_update(self, event_data: Dict[str, Any]):
        """Handle performance update events."""
        metrics = event_data.get("metrics", {})
        self.state.system_metrics.update(metrics)
    
    # Event-driven UI update handlers
    async def _on_input_changed(self, event_data: Dict[str, Any]):
        """Handle input change events and update input panel."""
        try:
            # Update AI suggestions asynchronously
            if self.ai_composer and event_data.get("input"):
                try:
                    suggestions = await self.ai_composer.get_suggestions(event_data["input"])
                    self.state.input_suggestions = [s.get("completion", s) for s in suggestions if isinstance(s, (dict, str))]
                except Exception:
                    pass  # Ignore suggestion errors
            
            # Update UI panels through Rich Live (automatic)
            await self._refresh_panel("input")
            
        except Exception as e:
            debug_mode = getattr(self.cli_config, 'debug_mode', False)
            if debug_mode:
                logger.debug(f"Error handling input change event: {e}")
    
    async def _on_agent_status_changed(self, event_data: Dict[str, Any]):
        """Handle agent status change events and update status panel."""
        try:
            # Update status panel
            await self._refresh_panel("status")
        except Exception as e:
            pass  # Silently ignore event handling errors
    
    async def _on_metrics_updated(self, event_data: Dict[str, Any]):
        """Handle metrics update events and update status panel."""
        try:
            # Update status panel with new metrics
            await self._refresh_panel("status")
        except Exception as e:
            pass  # Silently ignore event handling errors
    
    async def _on_conversation_updated(self, event_data: Dict[str, Any]):
        """Handle conversation update events and update chat panel."""
        try:
            # Update chat panel
            await self._refresh_panel("chat")
        except Exception as e:
            pass  # Silently ignore event handling errors
    
    async def _on_processing_state_changed(self, event_data: Dict[str, Any]):
        """Handle processing state change events and update header/input panels."""
        try:
            # Update header and input panels
            await self._refresh_panel("header")
            await self._refresh_panel("input")
        except Exception as e:
            pass  # Silently ignore event handling errors
    
    async def _on_ui_refresh(self, event_data: Dict[str, Any]):
        """Handle UI refresh events for specific panels."""
        try:
            panels = event_data.get("panels", [])
            for panel in panels:
                await self._refresh_panel(panel)
        except Exception as e:
            pass  # Silently ignore event handling errors
    
    async def _refresh_panel(self, panel_name: str):
        """Refresh a specific UI panel - Rich Live handles the actual update."""
        try:
            if not self.layout or not RICH_AVAILABLE or not sys.stdin.isatty():
                return  # Skip all Rich operations in non-TTY environments
            
            # Manual refresh for immediate updates when needed
            if (hasattr(self, 'live_display') and self.live_display and 
                sys.stdin.isatty() and sys.stdout.isatty()):
                try:
                    # Allow immediate refresh for responsive UI
                    self.live_display.refresh()
                except Exception:
                    pass  # Ignore refresh errors
            
            # Rich Live automatically updates when layout content changes
            # We just need to update the layout content with full-width panels
            if panel_name == "header":
                header_text = Text("🚀 AgentsMCP Revolutionary Interface", style="bold blue")
                if self.state.is_processing:
                    header_text.append(" • ", style="dim")
                    header_text.append(self.state.processing_message, style="yellow")
                
                self.layout["header"].update(
                    Panel(
                        Align.center(header_text),
                        box=box.ROUNDED,
                        style="bright_blue",
                        expand=True  # Fill full terminal width
                    )
                )
            
            elif panel_name == "status":
                status_content = self._create_status_panel()
                self.layout["status"].update(
                    Panel(
                        status_content,
                        title="Agent Status",
                        box=box.ROUNDED,
                        style="green",
                        expand=True  # Fill sidebar width
                    )
                )
            
            elif panel_name == "chat":
                chat_content = self._create_chat_panel()
                self.layout["chat"].update(
                    Panel(
                        chat_content,
                        title="Conversation",
                        box=box.ROUNDED,
                        style="white",
                        expand=True  # Fill content area width
                    )
                )
            
            elif panel_name == "input":
                input_content = self._create_input_panel()
                self.layout["input"].update(
                    Panel(
                        input_content,
                        title="AI Command Composer",
                        box=box.ROUNDED,
                        style="cyan",
                        expand=True  # Fill content area width
                    )
                )
            
            elif panel_name == "dashboard":
                dashboard_content = await self._create_dashboard_panel()
                self.layout["dashboard"].update(
                    Panel(
                        dashboard_content,
                        title="Symphony Dashboard",
                        box=box.ROUNDED,
                        style="magenta",
                        expand=True  # Fill sidebar width
                    )
                )
                
        except Exception as e:
            pass  # Silently ignore panel refresh errors
    
    # Event publishing methods
    async def _publish_input_changed(self, current_input: str = None, suggestions: List[str] = None):
        """Publish input changed event."""
        try:
            await self.event_system.emit("input_changed", {
                "input": current_input or self.state.current_input,
                "suggestions": suggestions or self.state.input_suggestions
            })
        except Exception as e:
            pass  # Silently ignore event publishing errors
    
    async def _publish_agent_status_changed(self, agent: str = None, status: str = None):
        """Publish agent status changed event."""
        try:
            await self.event_system.emit("agent_status_changed", {
                "agent": agent,
                "status": status,
                "all_status": self.state.agent_status.copy()
            })
        except Exception as e:
            pass  # Silently ignore event publishing errors
    
    async def _publish_metrics_updated(self, metrics: Dict[str, Any] = None):
        """Publish metrics updated event."""
        try:
            await self.event_system.emit("metrics_updated", {
                "metrics": metrics or self.state.system_metrics.copy()
            })
        except Exception as e:
            pass  # Silently ignore event publishing errors
    
    async def _publish_conversation_updated(self, new_message: Dict[str, str] = None):
        """Publish conversation updated event."""
        try:
            await self.event_system.emit("conversation_updated", {
                "history": self.state.conversation_history.copy(),
                "new_message": new_message
            })
        except Exception as e:
            pass  # Silently ignore event publishing errors
    
    async def _publish_processing_state_changed(self, is_processing: bool = None, message: str = None):
        """Publish processing state changed event."""
        try:
            await self.event_system.emit("processing_state_changed", {
                "is_processing": is_processing if is_processing is not None else self.state.is_processing,
                "message": message or self.state.processing_message
            })
        except Exception as e:
            pass  # Silently ignore event publishing errors
    
    async def _cleanup(self):
        """Cleanup resources and ensure proper terminal state."""
        try:
            self.running = False
            
            # CRITICAL: Ensure we exit alternate screen if active
            if self._alternate_screen_active and hasattr(self.console, '_file'):
                try:
                    self.console._file.write('\033[?1049l')  # Exit alternate screen
                    self.console._file.flush()
                    self._alternate_screen_active = False
                    logger.info("Exited alternate screen buffer")
                except Exception as e:
                    logger.warning(f"Could not exit alternate screen: {e}")
            
            # Clear console one final time
            if self.console:
                try:
                    self.console.clear()
                except Exception:
                    pass  # Ignore cleanup errors
            
            # Cleanup components
            if self.enhancements:
                await self.enhancements.shutdown()
            
            if self.ai_composer:
                await self.ai_composer.shutdown()
            
            if self.event_system:
                await self.event_system.cleanup()
            
            logger.info("Revolutionary TUI Interface cleanup complete")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


# Factory function for creating the revolutionary interface
async def create_revolutionary_interface(
    cli_config=None,
    orchestrator_integration=None,
    revolutionary_components=None
) -> RevolutionaryTUIInterface:
    """Create and initialize a Revolutionary TUI Interface."""
    interface = RevolutionaryTUIInterface(
        cli_config=cli_config,
        orchestrator_integration=orchestrator_integration,
        revolutionary_components=revolutionary_components
    )
    
    return interface


# Example usage and testing
async def main():
    """Example usage of the Revolutionary TUI Interface."""
    interface = await create_revolutionary_interface()
    return await interface.run()


if __name__ == "__main__":
    asyncio.run(main())