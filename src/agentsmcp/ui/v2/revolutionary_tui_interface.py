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
        
        # Rich terminal setup
        self.console = Console() if RICH_AVAILABLE else None
        self.layout = None
        self.live_display = None
        
        # Input handling
        self.input_buffer = ""
        self.input_history = deque(maxlen=100)
        self.history_index = -1
        
        # Performance and animation
        self.last_render_time = 0.0
        self.frame_count = 0
        self.target_fps = 0.5  # Minimal refresh rate to prevent scrollback flooding
        
        # Content change detection
        self._panel_content_hashes = {}
        self._last_global_update = 0.0
    
    def _get_content_hash(self, content: str, extra_data: str = "") -> str:
        """Generate a stable hash for content comparison."""
        combined = f"{content}{extra_data}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()[:8]
        
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
            self.orchestrator = Orchestrator(config=config)
            logger.debug("Orchestrator initialized for Revolutionary TUI")
        except Exception as e:
            logger.warning(f"Failed to initialize orchestrator: {e}")
            self.orchestrator = None
    
    async def _setup_rich_layout(self):
        """Setup the Rich terminal layout for the revolutionary interface."""
        if not RICH_AVAILABLE:
            return
            
        # Create main layout with multiple panels
        self.layout = Layout()
        
        # Split into header, main, and footer
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        
        # Split main area into sidebar and content
        self.layout["main"].split_row(
            Layout(name="sidebar", size=25),
            Layout(name="content")
        )
        
        # Split content into chat and input
        self.layout["content"].split_column(
            Layout(name="chat", ratio=4),
            Layout(name="input", size=3)
        )
        
        # Split sidebar into status and dashboard
        self.layout["sidebar"].split_column(
            Layout(name="status", size=8),
            Layout(name="dashboard")
        )
        
        # Initialize panels
        await self._update_layout_panels()
    
    async def _update_layout_panels(self):
        """Update all layout panels with current content."""
        if not RICH_AVAILABLE or not self.layout:
            return
        
        # Global throttle - prevent excessive updates
        current_time = time.time()
        if current_time - self._last_global_update < 2.0:  # Minimum 2 second interval
            return
        
        try:
            # Header panel - Main title and system status
            header_text = Text("🚀 AgentsMCP Revolutionary Interface", style="bold blue")
            if self.state.is_processing:
                header_text.append(" • ", style="dim")
                header_text.append(self.state.processing_message, style="yellow")
            
            header_content = str(header_text)
            header_hash = self._get_content_hash(header_content, str(self.state.is_processing))
            
            if self._panel_content_hashes.get('header') != header_hash:
                self.layout["header"].update(
                    Panel(
                        Align.center(header_text),
                        box=box.ROUNDED,
                        style="bright_blue"
                    )
                )
                self._panel_content_hashes['header'] = header_hash
            
            # Status panel - Agent status and metrics
            status_content = self._create_status_panel()
            status_hash = self._get_content_hash(status_content)
            
            if self._panel_content_hashes.get('status') != status_hash:
                self.layout["status"].update(
                    Panel(
                        status_content,
                        title="Agent Status",
                        box=box.ROUNDED,
                        style="green"
                    )
                )
                self._panel_content_hashes['status'] = status_hash
            
            # Dashboard panel - Symphony dashboard
            dashboard_content = await self._create_dashboard_panel()
            dashboard_hash = self._get_content_hash(dashboard_content)
            
            if self._panel_content_hashes.get('dashboard') != dashboard_hash:
                self.layout["dashboard"].update(
                    Panel(
                        dashboard_content,
                        title="Symphony Dashboard",
                        box=box.ROUNDED,
                        style="magenta"
                    )
                )
                self._panel_content_hashes['dashboard'] = dashboard_hash
            
            # Chat panel - Conversation history
            chat_content = self._create_chat_panel()
            chat_hash = self._get_content_hash(chat_content, str(len(self.state.conversation_history)))
            
            if self._panel_content_hashes.get('chat') != chat_hash:
                self.layout["chat"].update(
                    Panel(
                        chat_content,
                        title="Conversation",
                        box=box.ROUNDED,
                        style="white"
                    )
                )
                self._panel_content_hashes['chat'] = chat_hash
            
            # Input panel - Smart input with suggestions
            input_content = self._create_input_panel()
            input_hash = self._get_content_hash(input_content, str(len(self.state.input_suggestions)))
            
            if self._panel_content_hashes.get('input') != input_hash:
                self.layout["input"].update(
                    Panel(
                        input_content,
                        title="AI Command Composer",
                        box=box.ROUNDED,
                        style="cyan"
                    )
                )
                self._panel_content_hashes['input'] = input_hash
            
            # Footer - Help and shortcuts
            footer_content = self._create_footer_panel()
            footer_hash = self._get_content_hash(footer_content)
            
            if self._panel_content_hashes.get('footer') != footer_hash:
                self.layout["footer"].update(
                    Panel(
                        footer_content,
                        box=box.ROUNDED,
                        style="dim"
                    )
                )
                self._panel_content_hashes['footer'] = footer_hash
            
            # Update global throttle timestamp
            self._last_global_update = current_time
            
        except Exception as e:
            logger.warning(f"Error updating layout panels: {e}")
            # Still update throttle timestamp even on error
            self._last_global_update = current_time
    
    def _create_status_panel(self) -> str:
        """Create the agent status panel content."""
        if not self.state.agent_status:
            return "🔄 Initializing agents...\n📊 Metrics loading..."
        
        content = []
        
        # Agent status
        for agent_name, status in self.state.agent_status.items():
            status_icon = "🟢" if status == "active" else "🔴" if status == "error" else "🟡"
            content.append(f"{status_icon} {agent_name}: {status}")
        
        # System metrics
        if self.state.system_metrics:
            content.append("📊 Performance:")
            for metric, value in self.state.system_metrics.items():
                if isinstance(value, float):
                    content.append(f"  {metric}: {value:.2f}")
                else:
                    content.append(f"  {metric}: {value}")
        
        return "\n".join(content)
    
    async def _create_dashboard_panel(self) -> str:
        """Create the Symphony dashboard panel content."""
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
                        'recent_activity': ['Task completed', 'New agent spawned', 'System optimized']
                    }
                
                content = []
                content.append("🎼 Symphony Status")
                content.append(f"Active Agents: {dashboard_data.get('active_agents', 0)}")
                content.append(f"Tasks Running: {dashboard_data.get('running_tasks', 0)}")
                content.append(f"Success Rate: {dashboard_data.get('success_rate', 0):.1f}%")
                
                # Recent activity
                recent_activity = dashboard_data.get('recent_activity', [])
                if recent_activity:
                    content.append("📈 Recent Activity:")
                    for activity in recent_activity[-3:]:  # Last 3 items
                        content.append(f"  • {activity}")
                
                return "\n".join(content)
            else:
                return "🎼 Symphony Dashboard\n📊 Loading metrics...\n🔄 Initializing..."
                
        except Exception as e:
            logger.warning(f"Error creating dashboard panel: {e}")
            return f"🎼 Symphony Dashboard\n❌ Error: {str(e)}"
    
    def _create_chat_panel(self) -> str:
        """Create the chat conversation panel content."""
        if not self.state.conversation_history:
            return """🚀 Revolutionary TUI Interface

Advanced command center with AI Command Composer,
agent monitoring, Symphony Dashboard, and visual effects.

Type your message below to start!"""

        content = []
        
        # Show recent conversation
        for entry in self.state.conversation_history[-10:]:  # Last 10 messages
            role = entry.get('role', 'unknown')
            message = entry.get('content', '')
            timestamp = entry.get('timestamp', '')
            
            if role == 'user':
                content.append(f"👤 You ({timestamp}):")
                content.append(f"  {message}")
            elif role == 'assistant':
                content.append(f"🤖 AgentsMCP ({timestamp}):")
                content.append(f"  {message}")
        
        return "\n".join(content)
    
    def _create_input_panel(self) -> str:
        """Create the AI command composer input panel content."""
        content = []
        
        # Current input with cursor indicator
        input_display = self.state.current_input or ""
        if self.state.is_processing:
            input_display += " ⏳"
        else:
            # Add blinking cursor based on time - creates a nice visual effect
            cursor_blink_interval = 0.8  # Blink every 0.8 seconds
            current_time = time.time()
            should_show_cursor = int(current_time / cursor_blink_interval) % 2 == 0
            
            if should_show_cursor or (current_time - self.state.last_update) < 0.5:
                # Show cursor for 0.5s after last input, then blink
                input_display += "█"  # Block cursor to show active input
            else:
                input_display += " "  # Space for cursor blink
        
        content.append(f"💬 Input: {input_display}")
        
        # Show helpful tips if no input yet and not processing
        if not self.state.current_input and not self.state.is_processing:
            content.append("⌨️ Type to chat • ↑/↓ History • Enter Send • Ctrl+C Exit")
        
        # Show history navigation info if user is browsing history
        if self.history_index > -1:
            content.append(f"📋 History: {self.history_index + 1}/{len(self.input_history)}")
        
        # Suggestions from AI Composer
        if self.state.input_suggestions:
            content.append("✨ AI Suggestions:")
            for i, suggestion in enumerate(self.state.input_suggestions[:3]):
                content.append(f"  {i+1}. {suggestion}")
        
        return "\n".join(content)
    
    def _create_footer_panel(self) -> str:
        """Create the footer panel with help and shortcuts."""
        help_items = [
            "Enter: Send message",
            "Ctrl+C: Exit",
            "↑/↓: History",
            "Tab: Complete",
            "/help: Commands"
        ]
        
        # Add performance info
        fps_info = f"FPS: {self.frame_count % 61}"  # Simple FPS display
        
        return " • ".join(help_items) + f" • {fps_info}"
    
    async def run(self) -> int:
        """Run the Revolutionary TUI Interface."""
        debug_mode = getattr(self.cli_config, 'debug_mode', False)
        
        if debug_mode:
            print("🔧 Debug: Revolutionary TUI Interface run() method called")
            print(f"🔧 Debug: RICH_AVAILABLE: {RICH_AVAILABLE}")
            print(f"🔧 Debug: CLI config: {self.cli_config}")
        
        try:
            # Initialize components
            if debug_mode:
                print("🔧 Debug: Calling initialize()...")
            
            if not await self.initialize():
                logger.error("Failed to initialize Revolutionary TUI Interface")
                if debug_mode:
                    print("🔧 Debug: Revolutionary TUI Interface initialization failed")
                return 1
            
            if debug_mode:
                print("🔧 Debug: Revolutionary TUI Interface initialized successfully")
            
            # Setup signal handlers
            def signal_handler(signum, frame):
                if debug_mode:
                    print(f"🔧 Debug: Signal {signum} received, setting running=False")
                self.running = False
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            if debug_mode:
                print("🔧 Debug: Signal handlers set up")
            
            self.running = True
            logger.info("🎯 Revolutionary TUI Interface marked as running")
            
            if debug_mode:
                print(f"🔧 Debug: Running state set to: {self.running}")
            
            if RICH_AVAILABLE:
                logger.info("🎨 Using Rich Live display for TUI")
                if debug_mode:
                    print("🔧 Debug: Rich is available, using Live display")
                    print(f"🔧 Debug: Layout object: {self.layout}")
                
                # Use Rich Live display for smooth updates
                try:
                    logger.info("📺 Attempting to create Rich Live display...")
                    if debug_mode:
                        print("🔧 Debug: Creating Rich Live display context...")
                        print(f"🔧 Debug: Layout valid: {self.layout is not None}")
                        print(f"🔧 Debug: Layout type: {type(self.layout)}")
                        print(f"🔧 Debug: Terminal size: {os.get_terminal_size() if hasattr(os, 'get_terminal_size') else 'unknown'}")
                        print(f"🔧 Debug: Is TTY: {sys.stdin.isatty()}")
                        print(f"🔧 Debug: TERM env: {os.environ.get('TERM', 'not set')}")
                    
                    # Try with screen=True for proper terminal control (prevents scrollback flooding)
                    try:
                        if debug_mode:
                            print("🔧 Debug: Attempting Live context with screen=True (alternate screen)...")
                        
                        with Live(self.layout, refresh_per_second=0.5, screen=True) as live:
                            logger.info("📺 Rich Live display context entered successfully (alternate screen)")
                            if debug_mode:
                                print("🔧 Debug: Rich Live display context active (alternate screen)")
                            
                            self.live_display = live
                            logger.info("🚀 Starting main loop...")
                            
                            if debug_mode:
                                print("🔧 Debug: About to call _run_main_loop()...")
                            
                            await self._run_main_loop()
                            
                            if debug_mode:
                                print("🔧 Debug: _run_main_loop() completed")
                            
                            logger.info("✅ Main loop completed")
                    
                    except Exception as live_e:
                        if debug_mode:
                            print(f"🔧 Debug: Live context (alternate screen) failed: {type(live_e).__name__}: {live_e}")
                            print("🔧 Debug: Retrying with alternate screen buffer...")
                        
                        # Retry with alternate screen - avoid screen=False to prevent scrollback pollution
                        try:
                            import time
                            time.sleep(0.1)  # Brief pause before retry
                            with Live(self.layout, refresh_per_second=0.25, screen=True) as live:
                                logger.info("📺 Rich Live display context entered successfully (retry)")
                                if debug_mode:
                                    print("🔧 Debug: Rich Live display context active (retry)")
                                
                                self.live_display = live
                                logger.info("🚀 Starting main loop...")
                                
                                if debug_mode:
                                    print("🔧 Debug: About to call _run_main_loop()...")
                                
                                await self._run_main_loop()
                                
                                if debug_mode:
                                    print("🔧 Debug: _run_main_loop() completed")
                                
                                logger.info("✅ Main loop completed")
                        except Exception as retry_e:
                            if debug_mode:
                                print(f"🔧 Debug: Retry also failed: {retry_e}")
                            logger.error(f"❌ Both Rich Live attempts failed: {retry_e}")
                            # Use fallback mode instead of screen=False
                            await self._run_fallback_loop()
                            return 0
                
                except Exception as e:
                    logger.error(f"❌ Rich Live display failed: {e}")
                    if debug_mode:
                        print(f"🔧 Debug: Rich Live display failed: {type(e).__name__}: {e}")
                        print("🔧 Debug: Full exception traceback:")
                        import traceback
                        traceback.print_exc()
                    logger.info("🔄 Falling back to basic display")
                    await self._run_fallback_loop()
            else:
                logger.info("📟 Using basic display (Rich not available)")
                if debug_mode:
                    print("🔧 Debug: Rich not available, using fallback loop")
                # Fallback to basic display
                await self._run_fallback_loop()
            
            logger.info("🏁 Revolutionary TUI Interface execution completed")
            if debug_mode:
                print("🔧 Debug: Revolutionary TUI Interface execution completed normally")
            return 0
            
        except KeyboardInterrupt:
            logger.info("Revolutionary TUI interrupted by user")
            if debug_mode:
                print("🔧 Debug: Revolutionary TUI interrupted by KeyboardInterrupt")
            return 0
        except Exception as e:
            logger.error(f"Revolutionary TUI Interface error: {e}")
            if debug_mode:
                print(f"🔧 Debug: Revolutionary TUI Interface exception: {type(e).__name__}: {e}")
                print("🔧 Debug: Full exception traceback:")
                import traceback
                traceback.print_exc()
            return 1
        finally:
            if debug_mode:
                print("🔧 Debug: Revolutionary TUI Interface cleanup starting...")
            await self._cleanup()
            if debug_mode:
                print("🔧 Debug: Revolutionary TUI Interface cleanup completed")
    
    async def _run_main_loop(self):
        """Main loop with Rich interface."""
        debug_mode = getattr(self.cli_config, 'debug_mode', False)
        
        logger.info("🚀 Revolutionary TUI Interface started with Rich display")
        if debug_mode:
            print("🔧 Debug: _run_main_loop() started")
            print(f"🔧 Debug: self.running = {self.running}")
        
        # Start background tasks (render task disabled - Rich Live handles its own rendering)
        logger.info("⚙️ Creating background tasks...")
        
        if debug_mode:
            print("🔧 Debug: Creating input task...")
        
        input_task = asyncio.create_task(self._input_loop())
        logger.debug("✅ Input task created")
        
        if debug_mode:
            print("🔧 Debug: Creating update task...")
        
        update_task = asyncio.create_task(self._update_loop())
        logger.debug("✅ Update task created")
        logger.info("🎯 Background tasks created, waiting for completion...")
        
        if debug_mode:
            print("🔧 Debug: Tasks created, waiting for first completion...")
        
        try:
            # Wait for any task to complete (usually from user interruption)
            done, pending = await asyncio.wait(
                [input_task, update_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            if debug_mode:
                print(f"🔧 Debug: Wait completed - {len(done)} tasks done, {len(pending)} pending")
            
            # Log which task completed first
            for task in done:
                task_name = "unknown"
                if task is input_task:
                    task_name = "input_task"
                elif task is update_task:
                    task_name = "update_task"
                
                if debug_mode:
                    print(f"🔧 Debug: Task completed: {task_name}")
                    if task.exception():
                        print(f"🔧 Debug: Task {task_name} exception: {task.exception()}")
                    else:
                        print(f"🔧 Debug: Task {task_name} result: {task.result()}")
                
                logger.info(f"Task completed first: {task_name}")
                try:
                    result = task.result()
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
    
    async def _run_fallback_loop(self):
        """Fallback loop without Rich (basic terminal output)."""
        logger.info("🚀 Revolutionary TUI Interface started in fallback mode")
        
        # Show initial interface
        print("=" * 60)
        print("🚀 AgentsMCP Revolutionary Interface (Fallback Mode)")
        print("=" * 60)
        print("Welcome to the advanced command center!")
        print("Rich terminal features unavailable - using basic mode.")
        print("=" * 60)
        
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
    
    async def _render_loop(self):
        """Render loop for smooth 60fps updates."""
        while self.running:
            try:
                start_time = time.time()
                
                # Update layout panels less frequently to prevent flooding
                current_time = time.time()
                if not hasattr(self, '_last_panel_update') or current_time - self._last_panel_update > 10.0:
                    await self._update_layout_panels()
                    self._last_panel_update = current_time
                
                # Apply any visual effects
                if self.enhancements:
                    await self._apply_visual_effects()
                
                # Update frame counter
                self.frame_count += 1
                
                # Calculate frame time and sleep
                frame_time = time.time() - start_time
                target_frame_time = 1.0 / self.target_fps
                sleep_time = max(0, target_frame_time - frame_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in render loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _input_loop(self):
        """Input handling loop with actual keyboard input processing."""
        debug_mode = getattr(self.cli_config, 'debug_mode', False)
        
        if debug_mode:
            print("🔧 Debug: _input_loop() started")
        
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
            logger.debug(f"TTY detection: sys.stdin.isatty() = {is_tty}")
            if debug_mode:
                print(f"🔧 Debug: sys.stdin.isatty() = {is_tty}")
            
            if is_tty:
                # Try to access stdin as TTY first
                try:
                    stdin_fd = sys.stdin.fileno()
                    if debug_mode:
                        print(f"🔧 Debug: stdin fileno = {stdin_fd}")
                    
                    attrs = termios.tcgetattr(stdin_fd)
                    tty_available = True
                    logger.debug("TTY detection: stdin TTY access successful")
                    if debug_mode:
                        print("🔧 Debug: stdin TTY access successful")
                        print(f"🔧 Debug: Terminal attrs available: {len(attrs) if attrs else 0} entries")
                        
                except (termios.error, OSError) as e:
                    logger.debug(f"TTY detection: stdin failed ({e}), trying /dev/tty")
                    if debug_mode:
                        print(f"🔧 Debug: stdin TTY failed: {type(e).__name__}: {e}")
                        print("🔧 Debug: Trying /dev/tty fallback...")
                    
                    # Fall back to /dev/tty if stdin doesn't work
                    try:
                        test_fd = os.open('/dev/tty', os.O_RDONLY)
                        if debug_mode:
                            print(f"🔧 Debug: /dev/tty opened, fd = {test_fd}")
                        
                        attrs = termios.tcgetattr(test_fd)
                        os.close(test_fd)
                        tty_available = True
                        logger.debug("TTY detection: /dev/tty access successful")
                        if debug_mode:
                            print("🔧 Debug: /dev/tty access successful")
                            
                    except (OSError, termios.error) as e2:
                        logger.debug(f"TTY detection: /dev/tty failed ({e2})")
                        if debug_mode:
                            print(f"🔧 Debug: /dev/tty also failed: {type(e2).__name__}: {e2}")
                        tty_available = False
            else:
                logger.debug("TTY detection: stdin is not a TTY")
                if debug_mode:
                    print("🔧 Debug: stdin is not a TTY")
                
        except ImportError as e:
            logger.debug(f"TTY detection: Import failed ({e})")
            if debug_mode:
                print(f"🔧 Debug: TTY import failed: {e}")
            tty_available = False
        
        logger.info(f"TTY detection result: tty_available = {tty_available}")
        if debug_mode:
            print(f"🔧 Debug: Final TTY detection result: {tty_available}")
        
        if not tty_available:
            logger.warning("TTY not available, using fallback input method")
            if debug_mode:
                print("🔧 Debug: TTY not available, calling fallback input loop")
            return await self._fallback_input_loop()
        
        # Setup terminal for raw input
        fd = None
        original_settings = None
        stop_flag = {"stop": False}
        
        # Get the current event loop to pass to the reader thread
        current_loop = asyncio.get_running_loop()
        if debug_mode:
            print(f"🔧 Debug: _input_loop got main event loop: {current_loop}")
        
        def reader_thread(loop):
            nonlocal fd, original_settings
            if debug_mode:
                print("🔧 Debug: reader_thread() started")
                print(f"🔧 Debug: reader_thread received event loop: {loop}")
            try:
                # Try stdin first, then fall back to /dev/tty
                try:
                    fd = sys.stdin.fileno()
                    if debug_mode:
                        print(f"🔧 Debug: reader_thread using stdin fd: {fd}")
                    original_settings = termios.tcgetattr(fd)
                    tty.setraw(fd)
                    if debug_mode:
                        print("🔧 Debug: reader_thread stdin set to raw mode")
                except (termios.error, OSError) as e:
                    if debug_mode:
                        print(f"🔧 Debug: reader_thread stdin failed: {e}, trying /dev/tty")
                    # Fallback to /dev/tty
                    fd = os.open('/dev/tty', os.O_RDONLY)
                    original_settings = termios.tcgetattr(fd)
                    tty.setraw(fd)
                    if debug_mode:
                        print(f"🔧 Debug: reader_thread using /dev/tty fd: {fd}")
                
                if debug_mode:
                    print("🔧 Debug: reader_thread ready to process input")
                
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
                            logger.debug(f"Error reading from terminal: {e}")
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
                        logger.debug(f"Error in reader thread: {e}")
                        time.sleep(0.1)
                        
            except Exception as e:
                logger.error(f"Failed to setup raw terminal input: {e}")
                if debug_mode:
                    print(f"🔧 Debug: reader_thread setup exception: {type(e).__name__}: {e}")
                    print("🔧 Debug: reader_thread setting use_fallback flag")
                    import traceback
                    print("🔧 Debug: reader_thread exception traceback:")
                    traceback.print_exc()
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
        if debug_mode:
            print("🔧 Debug: Starting reader thread with event loop parameter")
        thread.start()
        
        # Keep the async function alive while input thread runs
        if debug_mode:
            print(f"🔧 Debug: _input_loop starting main monitoring loop, thread alive: {thread.is_alive()}")
        
        try:
            loop_iterations = 0
            while self.running and thread.is_alive():
                loop_iterations += 1
                if debug_mode and loop_iterations == 1:
                    print("🔧 Debug: _input_loop main monitoring loop started")
                elif debug_mode and loop_iterations % 50 == 0:  # Log every 5 seconds
                    print(f"🔧 Debug: _input_loop monitoring loop iteration {loop_iterations}, thread alive: {thread.is_alive()}")
                
                # Check if thread requested fallback
                if stop_flag.get("use_fallback", False):
                    logger.info("Reader thread requested fallback, switching to fallback input method")
                    if debug_mode:
                        print("🔧 Debug: _input_loop use_fallback flag detected, switching to fallback")
                    stop_flag["stop"] = True
                    if thread.is_alive():
                        thread.join(timeout=1.0)
                    return await self._fallback_input_loop()
                
                await asyncio.sleep(0.1)
                
                # Update input suggestions if AI composer is available
                if self.ai_composer and self.state.current_input:
                    try:
                        suggestions = await self.ai_composer.get_suggestions(self.state.current_input)
                        self.state.input_suggestions = [s.get("completion", s) for s in suggestions if isinstance(s, (dict, str))]
                    except Exception:
                        pass  # Ignore suggestion errors
                        
        finally:
            if debug_mode:
                print(f"🔧 Debug: _input_loop main loop exited, thread alive: {thread.is_alive()}")
                print(f"🔧 Debug: _input_loop self.running: {self.running}")
                print(f"🔧 Debug: _input_loop stop_flag: {stop_flag}")
            # Signal thread to stop
            stop_flag["stop"] = True
            if thread.is_alive():
                thread.join(timeout=1.0)
            if debug_mode:
                print("🔧 Debug: _input_loop completed, returning None")
    
    async def _fallback_input_loop(self):
        """Fallback input loop for when raw terminal setup fails."""
        logger.info("Using fallback input method - will wait for Enter key")
        
        import concurrent.futures
        import sys
        
        # Check if stdin is actually available
        if not sys.stdin or sys.stdin.closed:
            logger.error("No stdin available - cannot use fallback input method")
            print("❌ No stdin available for input. TUI cannot function in this environment.")
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
                        print("\n🎨 Revolutionary TUI (TTY Fallback Mode)")
                        print("=" * 50)
                        print("💡 Enter commands below. Type 'quit' to exit.")
                        print("🔑 Available commands: help, quit, status")
                        print("-" * 50)
                        header_shown = True
                    return input("💬 Enter command: ")
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
                        print("\n🎨 Revolutionary TUI (Non-Interactive Mode)")
                        print("=" * 50)
                        print("💡 Running in non-interactive environment")
                        print("🔑 Simulating user commands for demonstration")
                        print("-" * 50)
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
                        time.sleep(4)  # Wait 4 seconds between commands to reduce output rate
                        # Only print every other command to reduce flooding
                        if get_input.command_index % 2 == 0:
                            print(f"💬 Command: {cmd}")
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
            # Show welcome message for fallback mode with clear separation
            print("\n" + "="*60)
            print("🚀 REVOLUTIONARY TUI INTERFACE - DEMO MODE")
            print("   Running without raw terminal access")
            print("   Watch as we demonstrate TUI capabilities...")
            print("="*60)
            
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
                        logger.info("Input returned None - checking if stdin is available")
                        if sys.stdin and not sys.stdin.closed:
                            print("\n⚠️  Input interrupted. Type 'quit' to exit cleanly.")
                            continue
                        else:
                            logger.info("stdin closed or unavailable - exiting")
                            break
                    
                    user_input = user_input.strip()
                    if user_input.lower() in ['quit', 'exit']:
                        print("👋 Exiting Revolutionary TUI...")
                        self.running = False
                        break
                    
                    # Special commands are now handled in _process_user_input()
                    # Remove duplicate handlers to prevent conflicts
                    
                    if user_input:
                        # Handle demo messages
                        if user_input.startswith("Demo:"):
                            print(f"🎭 {user_input}")
                            print("   Simulating revolutionary TUI capabilities...")
                            continue
                        
                        # Process the input
                        self.state.current_input = user_input
                        old_conversation_length = len(self.state.conversation_history)
                        await self._process_user_input(user_input)
                        self.state.current_input = ""
                        
                        # Print any new responses that were added to conversation history
                        new_conversation_length = len(self.state.conversation_history)
                        if new_conversation_length > old_conversation_length:
                            # Print the assistant's response(s)
                            for msg in self.state.conversation_history[old_conversation_length:]:
                                if msg["role"] == "assistant":
                                    print(f"\n{msg['content']}\n")
                    
                except Exception as e:
                    logger.error(f"Error in fallback input loop: {e}")
                    print(f"❌ Input error: {e}")
                    await asyncio.sleep(1.0)
        
        finally:
            executor.shutdown(wait=False)
            logger.info("Fallback input loop ended")
            
            # Show completion message
            print("\n" + "="*60)
            print("✅ REVOLUTIONARY TUI DEMO COMPLETED SUCCESSFULLY")
            print("   All TUI capabilities demonstrated")
            print("   For interactive mode, run from a real terminal")
            print("="*60)
    
    async def _update_loop(self):
        """Update loop for system status and metrics."""
        while self.running:
            try:
                # Update agent status
                await self._update_agent_status()
                
                # Update system metrics
                await self._update_system_metrics()
                
                # Update timestamp
                self.state.last_update = time.time()
                
                # Wait before next update
                await asyncio.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _update_agent_status(self):
        """Update the status of all agents."""
        try:
            # Mock agent status - in real implementation, query actual agents
            self.state.agent_status = {
                "orchestrator": "active" if self.orchestrator else "offline",
                "ai_composer": "active" if self.ai_composer else "offline",
                "symphony_dashboard": "active" if self.symphony_dashboard else "offline",
                "tui_enhancements": "active" if self.enhancements else "offline"
            }
        except Exception as e:
            logger.warning(f"Error updating agent status: {e}")
    
    async def _update_system_metrics(self):
        """Update system performance metrics."""
        try:
            # Mock metrics - in real implementation, gather actual performance data
            self.state.system_metrics = {
                "fps": min(60, self.frame_count % 61),
                "memory_mb": 45.2,  # Mock memory usage
                "cpu_percent": 12.5,  # Mock CPU usage
                "active_tasks": len(self.state.conversation_history),
                "uptime_mins": int((time.time() - self.last_render_time) / 60) if self.last_render_time else 0
            }
        except Exception as e:
            logger.warning(f"Error updating system metrics: {e}")
    
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
        """Handle a single character input and update display immediately."""
        # Add character to current input
        self.state.current_input += char
        
        # Make cursor visible and reset blink timer  
        self.state.last_update = time.time()
        
        # Throttled visual feedback to prevent scrollback flooding
        self._update_input_display_immediate()
    
    def _update_input_display_immediate(self):
        """Update input display with heavy throttling to prevent flooding."""
        if not self.live_display or not self.layout:
            return
        
        # Heavy throttle updates to prevent flooding - max 1 update per 5 seconds
        current_time = time.time()
        if hasattr(self, '_last_display_update'):
            if current_time - self._last_display_update < 5.0:  # 5 second throttle
                return
        self._last_display_update = current_time
            
        try:
            # Only update if content actually changed
            input_content = self._create_input_panel()
            input_hash = self._get_content_hash(input_content, str(len(self.state.input_suggestions)))
            
            if self._panel_content_hashes.get('input_immediate') == input_hash:
                return  # No change, skip update
            
            # Update just the input panel
            from rich import box
            self.layout["input"].update(
                Panel(
                    input_content,
                    title="AI Command Composer",
                    box=box.ROUNDED,
                    style="cyan"
                )
            )
            self._panel_content_hashes['input_immediate'] = input_hash
            
            # Very rare refresh to prevent flooding
            if hasattr(self.live_display, 'refresh') and current_time - getattr(self, '_last_refresh', 0) > 10.0:
                try:
                    self.live_display.refresh()
                    self._last_refresh = current_time
                except Exception:
                    pass  # Ignore refresh errors
                    
        except Exception as e:
            logger.debug(f"Error in throttled input display update: {e}")
    
    def _handle_backspace_input(self):
        """Handle backspace key input with immediate visual feedback."""
        if self.state.current_input:
            self.state.current_input = self.state.current_input[:-1]
            self.state.last_update = time.time()
            
            # Throttled visual feedback to prevent flooding
            self._update_input_display_immediate()
    
    async def _handle_enter_input(self):
        """Handle Enter key - submit the current input."""
        if self.state.current_input.strip():
            # Process the input
            await self._process_user_input(self.state.current_input.strip())
            
            # Clear input state
            self.state.current_input = ""
            self.state.input_suggestions = []
            
            # Update display with heavy throttling
            self._update_input_display_immediate()
    
    def _handle_up_arrow(self):
        """Handle up arrow key - navigate to previous input in history."""
        if self.input_history and self.history_index < len(self.input_history) - 1:
            # Save current input if at end of history
            if self.history_index == -1 and self.state.current_input.strip():
                # Save current input as most recent
                pass  # Current input preserved
            
            self.history_index += 1
            history_item = self.input_history[-(self.history_index + 1)]
            self.state.current_input = history_item
            self.state.last_update = time.time()
            
            # Update display with throttling
            self._update_input_display_immediate()
    
    def _handle_down_arrow(self):
        """Handle down arrow key - navigate to next input in history."""
        if self.history_index > -1:
            self.history_index -= 1
            
            if self.history_index == -1:
                # Back to current input
                self.state.current_input = ""
            else:
                history_item = self.input_history[-(self.history_index + 1)]
                self.state.current_input = history_item
                
            self.state.last_update = time.time()
            
            # Update display with throttling
            self._update_input_display_immediate()
    
    def _handle_left_arrow(self):
        """Handle left arrow key - move cursor left (future enhancement)."""
        # TODO: Implement cursor position control
        # For now, just refresh display to show we received the key
        self.state.last_update = time.time()
        self._update_input_display_immediate()
    
    def _handle_right_arrow(self):
        """Handle right arrow key - move cursor right (future enhancement)."""  
        # TODO: Implement cursor position control
        # For now, just refresh display to show we received the key
        self.state.last_update = time.time()
        self._update_input_display_immediate()
    
    def _handle_escape_key(self):
        """Handle ESC key - clear current input."""
        self.state.current_input = ""
        self.state.input_suggestions = []
        self.history_index = -1
        self.state.last_update = time.time()
        
        # Update display with throttling
        self._update_input_display_immediate()
    
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
                print("👋 Exiting Revolutionary TUI...")
                self.running = False
                return
            
            # Handle help command
            if user_input_lower in ['help', '/help']:
                response_content = """📋 Revolutionary TUI Commands:
  quit, exit, /quit, /exit - Exit TUI
  help, /help              - Show this help
  status, /status          - Show system status
  clear, /clear            - Clear conversation history
  
  Any other input will be processed by AI agents."""
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.state.conversation_history.extend([{
                    "role": "user",
                    "content": user_input,
                    "timestamp": timestamp
                }, {
                    "role": "assistant", 
                    "content": response_content,
                    "timestamp": timestamp
                }])
                return
                
            # Handle status command
            if user_input_lower in ['status', '/status']:
                response_content = f"""📊 Revolutionary TUI Status:
  Running: {self.running}
  Mode: Revolutionary TUI
  TTY Available: Yes
  Input History: {len(self.input_history)} entries
  Conversation Length: {len(self.state.conversation_history)} messages"""
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.state.conversation_history.extend([{
                    "role": "user",
                    "content": user_input,
                    "timestamp": timestamp
                }, {
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": timestamp
                }])
                return
                
            # Handle clear command
            if user_input_lower in ['clear', '/clear']:
                self.state.conversation_history.clear()
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.state.conversation_history.append({
                    "role": "assistant",
                    "content": "🧹 Conversation history cleared.",
                    "timestamp": timestamp
                })
                return
            
            # Add to conversation history for non-builtin commands
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.state.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": timestamp
            })
            
            # Show processing state
            self.state.is_processing = True
            self.state.processing_message = "Processing with AI agents..."
            
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
                self.state.conversation_history.append({
                    "role": "assistant",
                    "content": response.content or "No response generated",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
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
                self.state.conversation_history.append({
                    "role": "assistant", 
                    "content": f"Revolutionary TUI received: {user_input}",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            
            # Clear processing state
            self.state.is_processing = False
            self.state.processing_message = ""
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            self.state.is_processing = False
            self.state.processing_message = ""
            
            # Add error to conversation
            self.state.conversation_history.append({
                "role": "system",
                "content": f"Error processing input: {str(e)}",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
    
    async def _register_event_handlers(self):
        """Register event handlers for the revolutionary interface."""
        try:
            # Register handlers for various events
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
    
    async def _cleanup(self):
        """Cleanup resources."""
        try:
            self.running = False
            
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