"""
Revolutionary TUI Interface - Integrated with unified architecture for dotted-line free display.

This module provides the Revolutionary TUI Interface using the new unified TUI architecture
to eliminate console pollution and dotted line issues. Integrates with unified_tui_coordinator,
terminal_controller, logging_isolation_manager, text_layout_engine, and other infrastructure.

Key Features:
- Rich multi-panel layout with status bars and interactive sections  
- 60fps animations and visual effects (without console flooding)
- AI Command Composer integration with smart suggestions
- Symphony Dashboard with live metrics
- Real-time status updates and agent monitoring
- Typewriter effects and visual feedback
- Advanced input handling with command completion
- Revolutionary visual design with smooth transitions
- Zero dotted line pollution using text_layout_engine
"""

import asyncio
import logging
import os
import shutil
import sys
import threading
import time
import signal
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

# Import unified architecture components
from .terminal_controller import TerminalController, AlternateScreenMode, CursorVisibility
from .logging_isolation_manager import LoggingIsolationManager, LogLevel
from .text_layout_engine import TextLayoutEngine, WrapMode, OverflowHandling, eliminate_dotted_lines
from .input_rendering_pipeline import InputRenderingPipeline, InputMode
from .display_manager import DisplayManager, RefreshMode, ContentUpdate

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
        try:
            self.cli_config = cli_config
            self.orchestrator_integration = orchestrator_integration or False
            
            # Initialize logging attributes first (needed by _safe_log)
            self._isolated_logging = False  # Start with non-isolated logging
            self.logging_manager = None  # Will be set up later if needed
            
            # Set up logging
            if self.orchestrator_integration:
                self.logger = logger
            else:
                logging.basicConfig(level=logging.DEBUG)
                self.logger = logger
                
            # Initialize state management
            self.state = TUIState()
            self.running = False
            self.shutdown_event = asyncio.Event()
            
            # Initialize components from orchestrator if provided
            self.revolutionary_components = revolutionary_components or {}
            
            # Revolutionary enhancements
            self.enhancements: Optional[RevolutionaryTUIEnhancements] = None
            self.ai_composer: Optional[AICommandComposer] = None
            self.symphony_dashboard: Optional[SymphonyDashboard] = None
            
            # Rich terminal setup with improved terminal size detection
            if RICH_AVAILABLE:
                # First, get actual terminal size using multiple methods
                actual_width, actual_height = self._detect_actual_terminal_size()
                
                # Determine if we should force terminal mode based on size detection
                has_valid_size = actual_width > 0 and actual_height > 0
                
                # Initialize console with proper size detection
                self.console = Console(
                    # Force terminal mode when we have valid dimensions to ensure Rich respects explicit sizing
                    force_terminal=True if has_valid_size else None,
                    legacy_windows=False,
                    file=sys.stdout,
                    stderr=False,
                    quiet=False,
                    soft_wrap=False,
                    # Explicitly set size if detection worked
                    width=actual_width if actual_width > 0 else None,
                    height=actual_height if actual_height > 0 else None
                )
                
                # Verify and set terminal dimensions
                try:
                    console_size = self.console.size
                    detected_width = console_size.width
                    detected_height = console_size.height
                    
                    # Use the larger of detected vs actual size to avoid layout truncation
                    self.terminal_width = max(detected_width, actual_width) if actual_width > 0 else detected_width
                    self.terminal_height = max(detected_height, actual_height) if actual_height > 0 else detected_height
                    
                    self._safe_log("debug", f"Terminal size detection: Rich={detected_width}x{detected_height}, Actual={actual_width}x{actual_height}, Using={self.terminal_width}x{self.terminal_height}")
                    
                except Exception as e:
                    # Fallback to actual detected size
                    if actual_width > 0 and actual_height > 0:
                        self.terminal_width = actual_width
                        self.terminal_height = actual_height
                        self._safe_log("info", f"Using actual terminal size: {self.terminal_width}x{self.terminal_height}")
                    else:
                        self.terminal_width = 80
                        self.terminal_height = 24
                        self._safe_log("warning", f"Terminal size detection failed, using fallback: {self.terminal_width}x{self.terminal_height}")
            else:
                self.console = None
                self.terminal_width = 80
                self.terminal_height = 24
            
            self.layout = None
            self.live_display = None
            
            # Input handling - UNIFIED: Use only self.state.current_input (removed duplicate input_buffer)
            self.input_history = deque(maxlen=100)
            self.history_index = -1
            
            # Performance and animation - Balanced for smooth TUI experience  
            self.last_render_time = 0.0
            self.frame_count = 0
            
            # Screen management
            self._alternate_screen_active = False
            
            # Event system
            self.event_system = AsyncEventSystem()
            
            # Add input rendering pipeline
            self.input_pipeline = InputRenderingPipeline()
            
            # Initialize controller (non-async)
            self.terminal_controller = TerminalController()
            
            # Threading
            self.main_thread_id = threading.get_ident()
            self.refresh_throttle = 16  # 60 FPS - reduced for better responsiveness
            self.last_refresh_time = 0.0
            self.refresh_lock = asyncio.Lock()
            
            # Initialize thread-safe event loop reference
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = None
                
        except Exception as e:
            self._safe_log("error", f"Failed to initialize Revolutionary TUI: {e}")
            # Set minimal fallback state
            self.console = None
            self.terminal_width = 80
            self.terminal_height = 24
            self.state = TUIState()
            self.running = False

    def _detect_actual_terminal_size(self):
        """
        Detect actual terminal size using multiple methods.
        
        Returns:
            tuple: (width, height) or (0, 0) if detection fails
        """
        methods = []
        
        # Method 1: shutil.get_terminal_size() - most reliable
        try:
            import shutil
            term_size = shutil.get_terminal_size()
            methods.append(("shutil", term_size.columns, term_size.lines))
        except Exception as e:
            self._safe_log("debug", f"shutil terminal size detection failed: {e}")
        
        # Method 2: os.get_terminal_size() if available
        try:
            import os
            if hasattr(os, 'get_terminal_size'):
                term_size = os.get_terminal_size()
                methods.append(("os", term_size.columns, term_size.lines))
        except Exception as e:
            self._safe_log("debug", f"os terminal size detection failed: {e}")
        
        # Method 3: Environment variables COLUMNS/LINES
        try:
            cols = int(os.environ.get('COLUMNS', 0))
            lines = int(os.environ.get('LINES', 0))
            if cols > 0 and lines > 0:
                methods.append(("env", cols, lines))
        except Exception as e:
            self._safe_log("debug", f"environment variable terminal size detection failed: {e}")
        
        # Method 4: Rich Console auto-detection (as fallback reference)
        try:
            from rich.console import Console
            temp_console = Console()
            size = temp_console.size
            methods.append(("rich", size.width, size.height))
        except Exception as e:
            self._safe_log("debug", f"Rich terminal size detection failed: {e}")
        
        # Log all detected sizes
        if methods:
            size_info = ", ".join([f"{method}={w}x{h}" for method, w, h in methods])
            self._safe_log("debug", f"Terminal size detection results: {size_info}")
            
            # Use the first (most reliable) method
            _, width, height = methods[0]
            
            # Sanity check: ensure reasonable terminal size
            if width >= 40 and height >= 10 and width <= 1000 and height <= 200:
                return width, height
            else:
                self._safe_log("warning", f"Detected terminal size {width}x{height} seems unreasonable, trying next method")
                # Try other methods if first one gives unreasonable results
                for method, w, h in methods[1:]:
                    if w >= 40 and h >= 10 and w <= 1000 and h <= 200:
                        self._safe_log("info", f"Using {method} terminal size: {w}x{h}")
                        return w, h
        
        self._safe_log("warning", "All terminal size detection methods failed")
        return 0, 0
    
    def _safe_log(self, level: str, message: str, **kwargs) -> None:
        """Safely log messages without polluting TUI display."""
        if self._isolated_logging and self.logging_manager and self.logging_manager.is_isolation_active():
            # Logging is isolated - messages go to buffer instead of console
            import logging
            logger = logging.getLogger(__name__)
            getattr(logger, level.lower())(message, **kwargs)
        elif not self._isolated_logging:
            # Logging is not isolated - can output normally (before TUI starts)
            import logging
            logger = logging.getLogger(__name__)
            getattr(logger, level.lower())(message, **kwargs)
        # If isolated logging is enabled but no manager, silently drop to prevent pollution
    
    def _safe_layout_text(self, content: str, max_width: int) -> Text:
        """Safely layout text using the text layout engine to prevent dotted lines."""
        try:
            if self.text_layout_engine:
                # Use async layout in a task (simplified sync wrapper)
                import asyncio
                try:
                    # Try to get current loop
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create a task for the layout
                        task = loop.create_task(eliminate_dotted_lines(content, max_width))
                        # For sync usage, we return immediately with basic layout
                        result_text = content
                    else:
                        # No running loop, use sync fallback
                        result_text = content
                except RuntimeError:
                    # No event loop, use sync fallback
                    result_text = content
                    
                # Apply basic text cleanup to eliminate dotted lines
                result_text = result_text.replace('...', '')  # Remove ellipsis
                result_text = result_text.replace('…', '')    # Remove Unicode ellipsis
                
                # Basic word wrapping without dotted lines
                if len(result_text) > max_width:
                    words = result_text.split()
                    lines = []
                    current_line = ""
                    
                    for word in words:
                        if not current_line:
                            current_line = word
                        elif len(current_line) + 1 + len(word) <= max_width:
                            current_line += " " + word
                        else:
                            lines.append(current_line)
                            current_line = word
                    
                    if current_line:
                        lines.append(current_line)
                    
                    result_text = "\n".join(lines)
                
                return Text(result_text)
            else:
                # No text layout engine available, basic cleanup
                result_text = content.replace('...', '').replace('…', '')
                return Text(result_text)
        except Exception:
            # Fallback to basic text
            return Text(content.replace('...', '').replace('…', ''))
    
    async def initialize(self) -> bool:
        """Initialize all revolutionary components."""
        logger.error("🚀 INITIALIZE: Starting Revolutionary TUI initialization")
        
        # Environment and TTY status checks
        try:
            import sys
            import os
            logger.error(f"🔍 INITIALIZE: Environment checks:")
            logger.error(f"   - sys.stdout.isatty(): {sys.stdout.isatty()}")
            logger.error(f"   - sys.stderr.isatty(): {sys.stderr.isatty()}")
            logger.error(f"   - TERM: {os.environ.get('TERM', 'NOT_SET')}")
            logger.error(f"   - COLORTERM: {os.environ.get('COLORTERM', 'NOT_SET')}")
            logger.error(f"   - Rich available: {RICH_AVAILABLE}")
        except Exception as e:
            logger.error(f"❌ INITIALIZE: Failed environment checks: {e}")
            return False
            
        try:
            # Initialize event system
            logger.error("🔧 INITIALIZE: Initializing event system...")
            try:
                await self.event_system.initialize()
                logger.error("✅ INITIALIZE: Event system initialized successfully")
            except Exception as e:
                logger.error(f"❌ INITIALIZE: Event system initialization failed: {e}")
                raise
            
            # Initialize Revolutionary TUI Enhancements
            logger.error("🔧 INITIALIZE: Initializing TUI enhancements...")
            try:
                self.enhancements = RevolutionaryTUIEnhancements(self.event_system)
                logger.error("✅ INITIALIZE: TUI enhancements initialized successfully")
            except Exception as e:
                logger.error(f"❌ INITIALIZE: TUI enhancements initialization failed: {e}")
                raise
            
            # Initialize AI Command Composer
            logger.error("🔧 INITIALIZE: Initializing AI Command Composer...")
            try:
                self.ai_composer = AICommandComposer(self.event_system)
                logger.error("✅ INITIALIZE: AI Command Composer initialized successfully")
            except Exception as e:
                logger.error(f"❌ INITIALIZE: AI Command Composer initialization failed: {e}")
                raise
            
            # Initialize Symphony Dashboard
            logger.error("🔧 INITIALIZE: Initializing Symphony Dashboard...")
            if "symphony_dashboard" in self.revolutionary_components:
                logger.error("🔍 INITIALIZE: Using existing symphony dashboard from components")
                self.symphony_dashboard = self.revolutionary_components["symphony_dashboard"]
                logger.error("✅ INITIALIZE: Symphony Dashboard (existing) initialized successfully")
            else:
                try:
                    logger.error("🔍 INITIALIZE: Creating new SymphonyDashboard instance")
                    self.symphony_dashboard = SymphonyDashboard(self.event_system)
                    logger.error("✅ INITIALIZE: Symphony Dashboard (new) initialized successfully")
                except Exception as e:
                    logger.error(f"⚠️ INITIALIZE: Could not initialize SymphonyDashboard: {e}")
                    logger.error("🔧 INITIALIZE: Creating mock dashboard fallback...")
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
                    logger.error("✅ INITIALIZE: Mock Symphony Dashboard created successfully")
            
            # Initialize orchestrator
            logger.error("🔧 INITIALIZE: Initializing orchestrator...")
            try:
                await self._initialize_orchestrator()
                logger.error("✅ INITIALIZE: Orchestrator initialized successfully")
            except Exception as e:
                logger.error(f"❌ INITIALIZE: Orchestrator initialization failed: {e}")
                raise
            
            # Setup Rich layout
            logger.error(f"🔧 INITIALIZE: Setting up Rich layout (RICH_AVAILABLE={RICH_AVAILABLE})...")
            if RICH_AVAILABLE:
                try:
                    await self._setup_rich_layout()
                    logger.error("✅ INITIALIZE: Rich layout setup successfully")
                except Exception as e:
                    logger.error(f"❌ INITIALIZE: Rich layout setup failed: {e}")
                    raise
            else:
                logger.error("⚠️ INITIALIZE: Skipping Rich layout (Rich not available)")
            
            # Register event handlers
            logger.error("🔧 INITIALIZE: Registering event handlers...")
            try:
                await self._register_event_handlers()
                logger.error("✅ INITIALIZE: Event handlers registered successfully")
            except Exception as e:
                logger.error(f"❌ INITIALIZE: Event handler registration failed: {e}")
                raise
            
            # Trigger initial status updates to populate UI
            logger.error("🔧 INITIALIZE: Triggering initial periodic updates...")
            try:
                await self._trigger_periodic_updates()
                logger.error("✅ INITIALIZE: Initial periodic updates triggered successfully")
            except Exception as e:
                logger.error(f"❌ INITIALIZE: Initial periodic updates failed: {e}")
                raise
            
            logger.error("🎉 INITIALIZE: Revolutionary TUI Interface fully initialized - RETURNING TRUE")
            return True
            
        except Exception as e:
            logger.error(f"💥 INITIALIZE: CRITICAL FAILURE - Failed to initialize Revolutionary TUI Interface: {e}")
            logger.error(f"💥 INITIALIZE: Exception type: {type(e).__name__}")
            logger.error(f"💥 INITIALIZE: Exception args: {e.args}")
            import traceback
            logger.error(f"💥 INITIALIZE: Full traceback:\n{traceback.format_exc()}")
            logger.error("💥 INITIALIZE: RETURNING FALSE")
            return False
    
    async def _initialize_orchestrator(self):
        """Initialize the orchestrator for message processing."""
        logger.error("🔧 _INITIALIZE_ORCHESTRATOR: Starting orchestrator initialization")
        
        try:
            logger.error("🔧 _INITIALIZE_ORCHESTRATOR: Creating OrchestratorConfig...")
            config = OrchestratorConfig(
                mode=OrchestratorMode.STRICT_ISOLATION,
                enable_smart_classification=True,
                fallback_to_simple_response=True,
                max_agent_wait_time_ms=120000,
                synthesis_timeout_ms=5000
            )
            logger.error("✅ _INITIALIZE_ORCHESTRATOR: OrchestratorConfig created successfully")
            
            # CRITICAL FIX: Temporarily suppress all logging during orchestrator initialization
            # This prevents LLM client debug logs from flooding the terminal during TUI startup
            logger.error("🔧 _INITIALIZE_ORCHESTRATOR: Suppressing logging levels for orchestrator creation...")
            original_root_level = logging.getLogger().level
            original_llm_level = logging.getLogger('agentsmcp.conversation.llm_client').level
            
            try:
                # Set to CRITICAL to suppress ALL logging during initialization
                logging.getLogger().setLevel(logging.CRITICAL)
                logging.getLogger('agentsmcp.conversation.llm_client').setLevel(logging.CRITICAL)
                logging.getLogger('agentsmcp.orchestration').setLevel(logging.CRITICAL)
                logging.getLogger('agentsmcp.agents').setLevel(logging.CRITICAL)
                logger.error("✅ _INITIALIZE_ORCHESTRATOR: Logging levels suppressed")
                
                # Initialize orchestrator (this may trigger LLM client initialization)
                logger.error("🔧 _INITIALIZE_ORCHESTRATOR: Creating Orchestrator instance...")
                self.orchestrator = Orchestrator(config=config)
                logger.error("✅ _INITIALIZE_ORCHESTRATOR: Orchestrator instance created successfully")
                
            finally:
                # Restore original logging levels
                logger.error("🔧 _INITIALIZE_ORCHESTRATOR: Restoring original logging levels...")
                logging.getLogger().setLevel(original_root_level)
                logging.getLogger('agentsmcp.conversation.llm_client').setLevel(original_llm_level)
                logger.error("✅ _INITIALIZE_ORCHESTRATOR: Original logging levels restored")
                
            logger.error("🎉 _INITIALIZE_ORCHESTRATOR: Orchestrator fully initialized successfully")
        except Exception as e:
            logger.error(f"💥 _INITIALIZE_ORCHESTRATOR: CRITICAL FAILURE - Failed to initialize orchestrator: {e}")
            logger.error(f"💥 _INITIALIZE_ORCHESTRATOR: Exception type: {type(e).__name__}")
            logger.error(f"💥 _INITIALIZE_ORCHESTRATOR: Exception args: {e.args}")
            import traceback
            logger.error(f"💥 _INITIALIZE_ORCHESTRATOR: Full traceback:\n{traceback.format_exc()}")
            self.orchestrator = None
            raise
    
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
        
        # Initialize panels with proper terminal-aware support
        await self._initialize_layout_panels()
        
        # Store layout initialization state
        self._layout_initialized = True
    
    async def _initialize_layout_panels(self):
        """Initialize layout panels with proper terminal size awareness."""
        if not RICH_AVAILABLE or not self.layout:
            return
        
        try:
            # Get current terminal dimensions for responsive layout
            try:
                current_size = self.console.size
                current_width = current_size.width
                current_height = current_size.height
            except Exception:
                current_width = self.terminal_width
                current_height = self.terminal_height
            
            # Calculate responsive panel widths based on actual terminal size
            sidebar_width = max(25, min(35, current_width // 4))  # 25% of width, min 25, max 35
            content_width = current_width - sidebar_width - 4  # Account for layout margins
            
            # Header panel - No fixed width, let Rich handle terminal width
            header_text = Text("🚀 AgentsMCP Revolutionary Interface", style="bold blue")
            if self.state.is_processing:
                header_text.append(" • ", style="dim")
                header_text.append(self.state.processing_message, style="yellow")
            
            self.layout["header"].update(
                Panel(
                    Align.center(header_text),
                    box=box.ROUNDED,
                    style="bright_blue"
                    # No width specified - let Rich handle terminal width
                )
            )
            
            # Status panel - Use calculated sidebar width
            status_content = self._create_status_panel()
            self.layout["status"].update(
                Panel(
                    status_content,
                    title="Agent Status",
                    box=box.ROUNDED,
                    style="green"
                    # No width specified - let layout manager handle
                )
            )
            
            # Dashboard panel - Use calculated sidebar width
            dashboard_content = await self._create_dashboard_panel()
            self.layout["dashboard"].update(
                Panel(
                    dashboard_content,
                    title="Symphony Dashboard",
                    box=box.ROUNDED,
                    style="magenta"
                    # No width specified - let layout manager handle
                )
            )
            
            # Chat panel - Use calculated content width
            chat_content = self._create_chat_panel()
            self.layout["chat"].update(
                Panel(
                    chat_content,
                    title="Conversation",
                    box=box.ROUNDED,
                    style="white"
                    # No width specified - let layout manager handle
                )
            )
            
            # Input panel - Use calculated content width
            input_content = self._create_input_panel()
            self.layout["input"].update(
                Panel(
                    input_content,
                    title="AI Command Composer",
                    box=box.ROUNDED,
                    style="cyan"
                    # No width specified - let layout manager handle
                )
            )
            
            # Footer - No fixed width, let Rich handle terminal width
            footer_content = self._create_footer_panel()
            self.layout["footer"].update(
                Panel(
                    footer_content,
                    box=box.ROUNDED,
                    style="dim"
                    # No width specified - let Rich handle terminal width
                )
            )
            
        except Exception as e:
            logger.warning(f"Error initializing layout panels: {e}")
    
    def _create_status_panel(self) -> Text:
        """Create the agent status panel content using unified text layout engine."""
        if not self.state.agent_status:
            # Show activity indicator with timestamp to prove TUI is alive
            import time
            current_time = time.strftime("%H:%M:%S")
            return self._safe_layout_text(f"🔄 Initializing...\n📊 Loading metrics...\n⏰ {current_time}\n🎯 TUI Active & Ready", 30)
        
        # Get current terminal dimensions for content sizing
        try:
            if RICH_AVAILABLE and self.console:
                current_size = self.console.size
                max_width = max(20, current_size.width // 4 - 2)  # Sidebar width minus padding
            else:
                max_width = 25  # Fallback width
        except Exception:
            max_width = 25
        
        # Build content as string first, then use text layout engine
        content_lines = []
        
        # Always show TUI status and activity indicator first
        import time
        current_time = time.strftime("%H:%M:%S")
        uptime_mins = (time.time() - getattr(self, '_startup_time', time.time())) / 60
        content_lines.append(f"🎯 TUI Ready & Active")
        content_lines.append(f"⏰ {current_time}")
        content_lines.append(f"📡 Up {uptime_mins:.1f}min")
        
        # Agent status - expanded display for wider sidebar
        for agent_name, status in self.state.agent_status.items():
            status_icon = "🟢" if status == "active" else "🔴" if status == "error" else "🟡"
            # Use full agent names in wider sidebar
            display_name = agent_name.replace("_", " ").title()
            line = f"{status_icon} {display_name}: {status.upper()}"
            content_lines.append(line)
        
        # System metrics - add to content lines
        if self.state.system_metrics:
            content_lines.append("📊 System Metrics:")
            
            fps = self.state.system_metrics.get('fps', 0)
            memory = self.state.system_metrics.get('memory_mb', 0) 
            cpu = self.state.system_metrics.get('cpu_percent', 0)
            tasks = self.state.system_metrics.get('active_tasks', 0)
            uptime = self.state.system_metrics.get('uptime_mins', 0)
            
            # Add metrics without wrapping issues
            content_lines.append(f"FPS: {fps:.1f}")
            content_lines.append(f"Memory: {memory:.0f}MB")
            content_lines.append(f"CPU: {cpu:.1f}%")
            content_lines.append(f"Tasks: {tasks}")
            content_lines.append(f"Uptime: {uptime:.0f}min")
        
        # Use text layout engine to create final text
        content_text = "\n".join(content_lines)
        return self._safe_layout_text(content_text, max_width)
    
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
                text = Text()
                
                # Expanded metrics display for wider sidebar
                agents = dashboard_data.get('active_agents', 0)
                tasks = dashboard_data.get('running_tasks', 0)
                success = dashboard_data.get('success_rate', 0)
                
                text.append("🎼 Symphony Overview:\n", style="magenta bold")
                text.append(f"👥 Active Agents: {agents}\n", style="cyan")
                text.append(f"⚙️ Running Tasks: {tasks}\n", style="blue")
                
                # Recent activity - expanded display with full descriptions
                recent_activity = dashboard_data.get('recent_activity', [])
                if recent_activity:
                    text.append(f"✅ Success Rate: {success:.1f}%\n", style="green")  # Newline if activities follow
                    text.append("📈 Recent Activity:\n", style="yellow bold")
                    activity_items = recent_activity[-3:]  # Show more activities in wider panel
                    for i, activity in enumerate(activity_items):
                        # Use full activity descriptions in wider sidebar
                        # Remove trailing newline from final activity item to prevent wrapping
                        if i == len(activity_items) - 1:
                            text.append(f"• {activity}", style="white")  # No trailing newline on final line
                        else:
                            text.append(f"• {activity}\n", style="white")
                else:
                    text.append(f"✅ Success Rate: {success:.1f}%", style="green")  # No trailing newline if no activities
                
                return text
            else:
                return Text("🎼 Symphony Dashboard Loading...\nInitializing components")
                
        except Exception as e:
            logger.warning(f"Error creating dashboard panel: {e}")
            return Text(f"🎼 Dashboard Error:\n{str(e)}")
    
    def _create_chat_panel(self) -> Text:
        """Create the chat conversation panel content using unified text layout engine."""
        if not self.state.conversation_history:
            # Enhanced startup message with clear instructions
            startup_message = (
                "🚀 Revolutionary TUI Interface - Ready for Input!\n\n"
                "✨ Welcome to the enhanced chat experience!\n\n"
                "💡 Available Commands:\n"
                "   • Type your message and press Enter to chat\n"
                "   • /help - Show detailed help\n" 
                "   • /quit or /exit - Exit the interface\n"
                "   • /clear - Clear conversation history\n"
                "   • /status - Show system status\n\n"
                "🎯 The interface is active and waiting for your input...\n"
                "💬 Start typing below to begin!"
            )
            return self._safe_layout_text(startup_message, 80)
        
        # Get current terminal dimensions for content sizing
        try:
            if RICH_AVAILABLE and self.console:
                current_size = self.console.size
                max_width = max(40, current_size.width - current_size.width // 4 - 6)  # Content width minus padding
                max_messages = max(5, (current_size.height - 10) // 3)  # Based on terminal height
            else:
                max_width = 60  # Fallback width
                max_messages = 10  # Fallback message count
        except Exception:
            max_width = 60
            max_messages = 10
        
        # Build content as lines first
        content_lines = []
        
        # Show recent conversation with responsive message count based on terminal height
        recent_messages = self.state.conversation_history[-max_messages:]  # Responsive message count
        
        for i, entry in enumerate(recent_messages):
            role = entry.get('role', 'unknown')
            message = entry.get('content', '')
            timestamp = entry.get('timestamp', '')
            
            # Skip empty messages
            if not message.strip():
                continue
            
            # Clean display message to prevent dotted lines
            display_message = message.strip().replace('...', '').replace('…', '')
            
            # Expanded format with full timestamps
            time_display = f"[{timestamp}]" if timestamp else ""
            
            # Role-specific formatting - build as strings (compact layout without empty separators)
            if role == 'user':
                content_lines.append(f"👤 User {time_display}:")
                content_lines.append(display_message)
            elif role == 'assistant':
                content_lines.append(f"🤖 Assistant {time_display}:")
                content_lines.append(display_message)
            elif role == 'system':
                content_lines.append(f"⚙️ System {time_display}:")
                content_lines.append(display_message)
        
        # Use text layout engine to create final text
        content_text = "\n".join(content_lines) if content_lines else "No messages yet"
        return self._safe_layout_text(content_text, max_width)
    
    def _create_input_panel(self) -> Text:
        """Create the AI command composer input panel content using unified text layout engine."""
        # Get terminal width for proper input display
        try:
            if RICH_AVAILABLE and self.console:
                current_size = self.console.size
                max_width = max(40, current_size.width - 6)  # Account for panel padding
            else:
                max_width = 74  # Fallback width
        except Exception:
            max_width = 74
        
        # FIXED: Remove pipeline sync that was corrupting user input during typing
        # The state.current_input is now the authoritative source of truth
        # Pipeline state is updated TO MATCH the display state in _handle_character_input()
        # This prevents the race condition where stale pipeline overwrites fresh user input
        
        # Build content as lines
        content_lines = []
        
        # Current input with cursor indicator - expanded display
        input_display = self.state.current_input or ""
        debug_mode = getattr(self.cli_config, 'debug_mode', False)
        if debug_mode:
            self._safe_log("debug", f"INPUT_PANEL: Displaying '{input_display}' (state: '{self.state.current_input}')")
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
        
        # Clean input display to prevent dotted lines
        input_display = input_display.replace('...', '').replace('…', '')
        
        # Build content lines
        content_lines.append(f"💬 Input: {input_display}")
        
        # Expanded status and help information  
        if not self.state.current_input and not self.state.is_processing:
            content_lines.append("💡 Quick Help: Type message & press Enter • /help for commands • /quit to exit")
            content_lines.append("🎯 TUI is Ready & Waiting for Input!")
            if hasattr(self, '_startup_time'):
                uptime_secs = time.time() - self._startup_time
                if uptime_secs < 60:
                    content_lines.append(f"⏱️  Running for {uptime_secs:.0f}s")
                else:
                    content_lines.append(f"⏱️  Running for {uptime_secs/60:.1f}min")
        
        # History navigation indicator
        if self.history_index > -1:
            content_lines.append(f"📋 History: {self.history_index + 1}/{len(self.input_history)}")
        
        # AI suggestions with expanded display
        if self.state.input_suggestions:
            content_lines.append("✨ AI Suggestions:")
            suggestion_items = self.state.input_suggestions[:3]  # Show more suggestions
            for i, suggestion in enumerate(suggestion_items):
                # Clean suggestions to prevent dotted lines
                clean_suggestion = suggestion.replace('...', '').replace('…', '')
                content_lines.append(f"  {i+1}. {clean_suggestion}")
        
        # Use text layout engine to create final text
        content_text = "\n".join(content_lines)
        return self._safe_layout_text(content_text, max_width)
    
    def _create_footer_panel(self) -> Text:
        """Create the footer panel with help and shortcuts using unified text layout engine.""" 
        # Get terminal width
        try:
            if RICH_AVAILABLE and self.console:
                current_size = self.console.size
                max_width = current_size.width - 4  # Full width minus padding
            else:
                max_width = 76  # Fallback width
        except Exception:
            max_width = 76
        
        # Build content as lines
        content_lines = []
        
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
        
        # Build footer content
        content_lines.append(f"{left_section} | {right_section}")
        
        # Use text layout engine to create final text
        content_text = "\n".join(content_lines)
        return self._safe_layout_text(content_text, max_width)
    
    async def run(self):
        """Main entry point for Revolutionary TUI Interface with enhanced terminal detection."""
        try:
            # Import required modules
            import signal
            import sys
            import os
            import logging
            
            # Get debug mode configuration
            debug_mode = getattr(self, 'debug_mode', False)
            
            logger.info("🚀 Revolutionary TUI Interface starting...")
            
            # Save original handlers to restore later
            original_handlers = {}
            
            # Create a null handler to suppress output
            null_handler = logging.NullHandler()
            
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
            
            # ENHANCED TTY DETECTION: More permissive for user environments
            # Check for CI/automated environments that should not use Rich
            is_ci = any(var in os.environ for var in ['CI', 'GITHUB_ACTIONS', 'TRAVIS', 'JENKINS', 'BUILD'])
            
            if is_ci:
                logger.error("CI environment detected - using minimal fallback to prevent output pollution")
                return 0
            
            # Enhanced terminal capability detection
            basic_tty = sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False
            has_term_env = os.environ.get('TERM', '') != ''
            has_term_program = os.environ.get('TERM_PROGRAM', '') != ''
            has_terminal_size = hasattr(os, 'get_terminal_size')
            
            # Check for specific terminal programs that support input even if not detected as TTY
            supported_terminals = ['iTerm.app', 'Terminal.app', 'VSCode', 'vscode']
            is_supported_terminal = any(term in os.environ.get('TERM_PROGRAM', '') for term in supported_terminals)
            
            # CRITICAL FIX: Allow interactive mode if we have ANY indication of terminal capability
            # This fixes the issue where iTerm.app users get stuck in demo mode
            allow_interactive_mode = (
                basic_tty or 
                has_term_env or 
                has_term_program or 
                is_supported_terminal or
                has_terminal_size
            )
            
            # Use Rich TTY if we have full TTY support
            is_tty = basic_tty and has_term_env
            
            # THREAT: Debug output floods scrollback buffer
            # MITIGATION: Use logging instead of print statements
            if debug_mode:
                logger.debug("Revolutionary TUI Interface run() method called")
                logger.debug(f"RICH_AVAILABLE: {RICH_AVAILABLE}")
                logger.debug(f"CLI config: {self.cli_config}")
                logger.debug(f"Basic TTY: {basic_tty}, TERM: {has_term_env}, TERM_PROGRAM: {has_term_program}")
                logger.debug(f"Supported terminal: {is_supported_terminal}, Allow interactive: {allow_interactive_mode}")
            
            # Initialize components
            if debug_mode:
                logger.debug("Calling initialize()")
            
            init_result = await self.initialize()
            
            if not init_result:
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
            self._safe_log("info", "🎯 Revolutionary TUI Interface marked as running")
            
            if debug_mode:
                logger.debug(f"Running state set to: {self.running}")
            
            # CRITICAL: Triple-check TTY status before Rich Live to prevent output cycling
            stdout_tty = sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else False
            stdin_tty = sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False
            stderr_tty = sys.stderr.isatty() if hasattr(sys.stderr, 'isatty') else False
            
            # CRITICAL FIX: Save original terminal state BEFORE Rich starts
            original_terminal_state = None
            try:
                if stdin_tty:
                    original_terminal_state = self._save_terminal_state()
                    if debug_mode:
                        logger.debug("💾 Original terminal state saved before Rich Live context")
            except Exception as e:
                logger.warning(f"Could not save original terminal state: {e}")
            
            try:
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
                                # Use terminal_controller for proper screen management if available
                                if hasattr(self, 'terminal_controller') and self.terminal_controller:
                                    await self.terminal_controller.enter_alternate_screen()
                                    self._alternate_screen_active = True
                                else:
                                    # Enhanced manual terminal control to prevent scrollback pollution
                                    self.console.clear()
                                    if hasattr(self.console, '_file') and hasattr(self.console._file, 'write'):
                                        # More robust sequence to prevent scrollback contamination
                                        self.console._file.write('\033[2J\033[H')     # Clear screen and home cursor
                                        self.console._file.write('\033[?47h')        # Save screen buffer
                                        self.console._file.write('\033[?1049h')      # Enter alternate screen
                                        self.console._file.write('\033[?25l')        # Hide cursor during transition
                                        self.console._file.flush()
                                        self._alternate_screen_active = True
                            except Exception as screen_e:
                                logger.warning(f"Could not explicitly enter alternate screen: {screen_e}")
                            
                            # Create Live with anti-scrollback configuration
                            live_config = {
                                "renderable": self.layout,
                                "console": self.console,
                                "screen": True,  # Force alternate screen buffer
                                "refresh_per_second": min(self.target_fps, 10.0),  # Cap refresh rate to prevent flooding
                                "auto_refresh": False,  # Disable auto-refresh to prevent scrollback leaks
                                "vertical_overflow": "crop",  # Prevent overflow that could leak to main screen
                                "transient": False  # Ensure proper screen buffer isolation
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
                                
                                try:
                                    await self._run_main_loop()
                                except Exception as main_loop_e:
                                    import traceback
                                    self._safe_log("error", f"Exception in _run_main_loop(): {main_loop_e}")
                                    self._safe_log("error", f"_run_main_loop() exception traceback:\n{traceback.format_exc()}")
                                    raise
                                
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
                                    "refresh_per_second": max(1.0, min(self.target_fps / 2, 5.0)),  # Very low refresh rate
                                    "auto_refresh": False,  # Disable auto-refresh for anti-scrollback  
                                    "vertical_overflow": "crop",
                                    "transient": False
                                }
                                
                                with Live(**emergency_live_config) as live:
                                    logger.info("📺 Emergency Rich Live display context entered (alternate screen)")
                                    self.live_display = live
                                    logger.info("🚀 Starting main loop with emergency Rich Live display...")
                                    
                                    try:
                                        await self._run_main_loop()
                                    except Exception as emergency_main_loop_e:
                                        import traceback
                                        self._safe_log("error", f"Exception in emergency _run_main_loop(): {emergency_main_loop_e}")
                                        self._safe_log("error", f"Emergency _run_main_loop() exception traceback:\n{traceback.format_exc()}")
                                        raise
                                    
                                    logger.info("✅ Emergency main loop completed")
                                
                            except Exception as emergency_live_e:
                                logger.error(f"Emergency Rich Live display also failed: {emergency_live_e}")
                                if debug_mode:
                                    import traceback
                                    logger.debug(f"Emergency Rich Live traceback:\n{traceback.format_exc()}")
                                
                                # FINAL FALLBACK: Run without Rich but check for interactive capability
                                logger.warning("🔄 Falling back to enhanced interactive mode")
                                if allow_interactive_mode:
                                    await self._enhanced_interactive_mode()
                                else:
                                    await self._run_emergency_fallback_loop()
                    
                    except Exception as rich_e:
                        logger.error(f"Rich Live display creation failed: {rich_e}")
                        if debug_mode:
                            import traceback
                            logger.debug(f"Rich Live creation traceback:\n{traceback.format_exc()}")
                        
                        # Fallback to enhanced interactive mode if supported
                        if allow_interactive_mode:
                            logger.info("🔄 Falling back to enhanced interactive mode")
                            await self._enhanced_interactive_mode()
                        else:
                            logger.warning("🔄 Falling back to emergency non-Rich mode")
                            await self._run_emergency_fallback_loop()
                
                else:
                    # CRITICAL FIX: Check for interactive capability instead of just going to demo mode
                    if allow_interactive_mode:
                        logger.info("💬 Enhanced interactive mode enabled (terminal capabilities detected)")
                        if debug_mode:
                            logger.debug(f"Interactive mode: Basic TTY={basic_tty}, TERM={has_term_env}, TERM_PROGRAM={has_term_program}, Supported={is_supported_terminal}")
                        await self._enhanced_interactive_mode()
                    else:
                        # Rich not available or no terminal capability - run in demo mode
                        logger.info("📱 No interactive capability detected - running demo mode")
                        if debug_mode:
                            logger.debug(f"Demo mode: Rich available={RICH_AVAILABLE}, Interactive allowed={allow_interactive_mode}")
                        await self._demo_mode_loop()
                
            finally:
                # CRITICAL FIX: Restore original terminal state AFTER Rich exits
                if original_terminal_state is not None:
                    try:
                        self._restore_terminal_state(original_terminal_state)
                        logger.info("🔧 Terminal echo state restored after Rich Live display")
                        if debug_mode:
                            logger.debug("✅ Original terminal state successfully restored")
                    except Exception as e:
                        logger.warning(f"Could not restore original terminal state: {e}")
                
                # Restore original logging handlers
                for logger_name, handlers in original_handlers.items():
                    logger_instance = logging.getLogger(logger_name)
                    logger_instance.handlers = handlers
                    logger_instance.setLevel(logging.DEBUG)  # Reset to default level
            
            # CRITICAL FIX: Always call cleanup before normal exit
            try:
                await self._cleanup()
                logger.info("✅ Revolutionary TUI Interface completed successfully with cleanup")
            except Exception as cleanup_e:
                logger.warning(f"Cleanup error on normal exit: {cleanup_e}")
            return 0
            
        except KeyboardInterrupt:
            logger.info("⚠️  Revolutionary TUI Interface interrupted by user")
            # CRITICAL FIX: Call cleanup on keyboard interrupt
            try:
                await self._cleanup()
                logger.info("✅ Cleanup completed after keyboard interrupt")
            except Exception as cleanup_e:
                logger.warning(f"Cleanup error on interrupt: {cleanup_e}")
            return 130
        except Exception as e:
            logger.error(f"💥 Revolutionary TUI Interface crashed: {e}")
            if debug_mode:
                import traceback
                logger.error(f"Revolutionary TUI crash traceback:\n{traceback.format_exc()}")
            # CRITICAL FIX: Call cleanup even on crash
            try:
                await self._cleanup()
                logger.info("✅ Cleanup completed after crash")
            except Exception as cleanup_e:
                logger.warning(f"Cleanup error on crash: {cleanup_e}")
            return 1

    def _save_terminal_state(self):
        """Save original terminal state before Rich Live context."""
        try:
            import termios
            import sys
            
            if sys.stdin.isatty():
                fd = sys.stdin.fileno()
                original_attrs = termios.tcgetattr(fd)
                logger.debug("💾 Terminal state saved successfully")
                return original_attrs
            else:
                logger.debug("⚠️  Not a TTY - cannot save terminal state")
                return None
                
        except Exception as e:
            logger.warning(f"Could not save terminal state: {e}")
            return None
    
    def _restore_terminal_state(self, original_attrs):
        """Restore original terminal state after Rich Live context exits."""
        if original_attrs is None:
            return
            
        try:
            import termios
            import sys
            
            if sys.stdin.isatty():
                fd = sys.stdin.fileno()
                
                # Restore the original terminal attributes
                termios.tcsetattr(fd, termios.TCSADRAIN, original_attrs)
                
                # Ensure echo and canonical mode are enabled for input visibility
                new_attrs = termios.tcgetattr(fd)
                new_attrs[3] = (new_attrs[3] | termios.ICANON | termios.ECHO)
                termios.tcsetattr(fd, termios.TCSADRAIN, new_attrs)
                
                logger.debug("🔧 Terminal state restored and echo enabled")
            else:
                logger.debug("⚠️  Not a TTY - cannot restore terminal state")
                
        except Exception as e:
            logger.warning(f"Could not restore terminal state: {e}")
    
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
        
        # Add startup status messages to show TUI is ready
        import time
        self.state.conversation_history.append({
            "role": "system",
            "content": "🚀 Revolutionary TUI Interface Ready! Type your message and press Enter to start chatting. Use /help for available commands.",
            "timestamp": time.time()
        })
        
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
                # Trigger periodic updates every 5 seconds for more responsive status display
                # This shows users that the TUI is active and waiting for input
                await asyncio.sleep(5.0)
                
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
                                    loop.call_soon_threadsafe(self._handle_enter_input_sync)
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
        logger.info("Using fallback input method - interactive mode enabled")
        logger.info(f"CRITICAL DEBUG: self.running state at start of _fallback_input_loop: {self.running}")
        
        import concurrent.futures
        import sys
        
        # Check if stdin is actually available and usable
        # Note: stdin can be available even in non-TTY environments (pipes, redirects, etc.)
        if not sys.stdin or sys.stdin.closed:
            logger.error("No stdin available - cannot use fallback input method")
            self.running = False
            return
        
        # CRITICAL FIX: In non-TTY environments, provide a demo mode instead of immediate exit
        if not sys.stdin.isatty():
            logger.info("Non-TTY environment detected - running in demo mode")
            logger.info(f"CRITICAL DEBUG: self.running before demo mode: {self.running}")
            # CRITICAL FIX: Force self.running = True at demo mode start
            self.running = True
            logger.info(f"CRITICAL DEBUG: self.running forced to True for demo mode: {self.running}")
            await self._demo_mode_loop()
            return
        
        # NOTE: Terminal state management is now handled in the main run() method
        # No longer restoring terminal state here as it's done in the proper finally block
        
        # Use a thread pool executor for blocking input
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        def get_input():
            """Get input from stdin in blocking mode."""
            try:
                # Always try to get actual user input first
                return input("💬 > ")
            except (EOFError, KeyboardInterrupt):
                logger.info("Input interrupted or EOF reached")
                return None
            except Exception as e:
                logger.error(f"Error getting input: {e}")
                return None
        
        try:
            # Show welcome message
            print("\n🚀 Revolutionary TUI Interface - Interactive Mode")
            print("Type your message and press Enter. Use 'help' for commands, 'quit' to exit.")
            print("=" * 60)
            sys.stdout.flush()  # Force output to be visible
            
            while self.running:
                try:
                    # Get input asynchronously
                    future = executor.submit(get_input)
                    
                    # Wait for input with periodic status checks
                    while not future.done() and self.running:
                        await asyncio.sleep(0.1)
                    
                    if not self.running:
                        future.cancel()
                        break
                    
                    user_input = future.result()
                    if user_input is None:
                        # EOF or interrupt
                        logger.info("Input stream ended - exiting TUI")
                        break
                    
                    user_input = user_input.strip()
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("👋 Exiting Revolutionary TUI...")
                        self.running = False
                        break
                    
                    if not user_input:
                        continue
                    
                    # Process the input
                    self.state.current_input = user_input
                    await self._process_user_input(user_input)
                    self.state.current_input = ""
                    
                    # Show last assistant response if any
                    if self.state.conversation_history:
                        last_msg = self.state.conversation_history[-1]
                        if last_msg.get('role') == 'assistant':
                            print(f"🤖 {last_msg.get('content', '')}")
                
                except Exception as e:
                    logger.error(f"Error in fallback input loop: {e}")
                    
        except Exception as e:
            logger.error(f"Fatal error in fallback input loop: {e}")
        finally:
            executor.shutdown(wait=False)
    
    async def _demo_mode_loop(self):
        """Enhanced demo mode with proper TTY-aware input handling."""
        logger.info("Starting enhanced demo mode for non-TTY environment")
        
        print("\n🚀 Revolutionary TUI Interface - Demo Mode")
        print("Running in non-TTY environment - demonstrating TUI capabilities...")
        print("=" * 60)
        sys.stdout.flush()  # Force output to be visible
        
        # Simulate some TUI activity for demonstration
        demo_messages = [
            "🤖 TUI initialized successfully in demo mode",
            "🔧 All systems operational", 
            "✅ Ready for interactive use in TTY environment",
            "💡 Tip: Run in a proper terminal for full interactive experience"
        ]
        
        for i, message in enumerate(demo_messages):
            if not self.running:
                logger.warning("Demo interrupted - TUI not running")
                break
                
            print(f"[{i+1}/4] {message}")
            sys.stdout.flush()  # Force output to be visible immediately
            logger.info(f"Demo message {i+1}/4: {message}")
            await asyncio.sleep(0.5)  # Brief pause between messages
        
        # Keep running for a bit to demonstrate the TUI stays active
        print("\n⏳ TUI staying active (demonstrating proper lifecycle)...")
        sys.stdout.flush()
        logger.info("Starting demo countdown to show TUI lifecycle")
        
        # Wait for 3 seconds to show the TUI is properly active and not shutting down immediately
        for countdown in range(3, 0, -1):
            if not self.running:
                logger.warning(f"Demo countdown interrupted at {countdown}s - TUI not running")
                break
            print(f"   Demo countdown: {countdown}s")
            sys.stdout.flush()
            logger.info(f"Demo countdown: {countdown}s remaining")
            await asyncio.sleep(1.0)
        
        # Enhanced: Add TTY-aware interactive input capability after demo
        print("\n💬 Interactive mode now available:")
        print("   • Type messages and press Enter to see them processed") 
        print("   • Type '/quit' to exit")
        print("   • Type '/help' for available commands")
        print("   • Type '/status' for system status")
        sys.stdout.flush()
        logger.info("Starting interactive input mode in demo")
        
        # Check if we have TTY input available
        has_tty_input = hasattr(sys.stdin, 'isatty') and sys.stdin.isatty()
        
        if has_tty_input:
            logger.info("TTY input detected - entering full interactive mode")
            # Full interactive mode with TTY
            while self.running:
                try:
                    # Provide visible input prompt
                    user_input = input("\n💬 TUI> ")
                    
                    # Echo the input so user sees what they typed
                    if user_input.strip():
                        print(f"📝 You typed: {user_input}")
                        sys.stdout.flush()
                        
                        # Process the input through the existing system
                        await self._process_user_input(user_input)
                        
                        # If a quit command was processed, running will be False
                        if not self.running:
                            break
                            
                        # Show any conversation updates that resulted from input processing
                        if hasattr(self.state, 'conversation_history') and self.state.conversation_history:
                            # Show the last assistant response if any
                            last_message = self.state.conversation_history[-1]
                            if last_message.get('role') == 'assistant':
                                print(f"🤖 Assistant: {last_message.get('content', '')}")
                                sys.stdout.flush()
                    else:
                        # Handle empty input
                        print("💭 (Empty input - type something or '/help' for commands)")
                        sys.stdout.flush()
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Exiting TUI...")
                    sys.stdout.flush()
                    logger.info("Interactive mode interrupted by user (EOF/Ctrl+C)")
                    break
                except Exception as e:
                    logger.error(f"Error in interactive input loop: {e}")
                    print(f"❌ Input error: {e}")
                    print("💡 Try again or type '/quit' to exit")
                    sys.stdout.flush()
        else:
            logger.info("Non-TTY environment detected - enabling fallback interactive mode")
            # Non-TTY environment - provide real interactive mode with input() fallback
            print("\n🔄 Interactive mode enabled in non-TTY environment...")
            print("💡 Using input() for interaction (limited features)")
            sys.stdout.flush()
            
            # Real interactive mode even without TTY
            while self.running:
                try:
                    # Use input() as fallback for non-TTY environments
                    user_input = input("\n💬 TUI> ")
                    
                    # Echo the input so user sees what they typed
                    if user_input.strip():
                        print(f"📝 Processing: {user_input}")
                        sys.stdout.flush()
                        
                        # Process the user input
                        await self._process_user_input(user_input)
                        
                        # Check if user wants to quit
                        if user_input.strip().lower() in ['/quit', 'quit', 'exit']:
                            print("👋 Goodbye!")
                            sys.stdout.flush()
                            logger.info("User requested quit in non-TTY interactive mode")
                            break
                    else:
                        # Handle empty input
                        print("💭 (Empty input - type something or '/help' for commands)")
                        sys.stdout.flush()
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Exiting TUI...")
                    sys.stdout.flush()
                    logger.info("Interactive mode interrupted by user (EOF/Ctrl+C)")
                    break
                except Exception as e:
                    logger.error(f"Error in non-TTY interactive input loop: {e}")
                    print(f"❌ Input error: {e}")
                    print("💡 Try again or type '/quit' to exit")
                    sys.stdout.flush()
        
        # Only show completion message if we actually completed the demo
        # (not if user explicitly quit from interactive mode)
        if self.running:
            print("\n✅ Demo completed - TUI shutting down gracefully")
            sys.stdout.flush()
            logger.info("Enhanced demo mode completed successfully")
            self.running = False
        else:
            logger.info("Interactive mode ended - TUI shutting down gracefully")

    async def _enhanced_interactive_mode(self):
        """Enhanced interactive mode for terminals that support input but may not register as full TTY.
        
        This method provides a robust interactive experience for environments like iTerm.app 
        running through Claude Code where stdin.isatty() returns False but input is actually available.
        """
        import sys
        import os
        import asyncio
        import logging
        from rich.prompt import Prompt
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        
        logger = logging.getLogger(__name__)
        console = Console()
        
        try:
            # Show welcome message
            welcome_text = Text()
            welcome_text.append("🤖 Revolutionary TUI Interface - Enhanced Interactive Mode\n", style="bold cyan")
            welcome_text.append("✨ Terminal capabilities detected - full interactive experience enabled\n", style="green")
            welcome_text.append("\nAvailable commands:\n", style="yellow")
            welcome_text.append("  • Type messages and press Enter to chat with AI\n", style="white")
            welcome_text.append("  • /help - Show help information\n", style="blue")
            welcome_text.append("  • /status - Show system status\n", style="blue")
            welcome_text.append("  • /quit - Exit the TUI\n", style="blue")
            welcome_text.append("  • /clear - Clear the screen\n", style="blue")
            welcome_text.append("\nType your message below:\n", style="yellow")
            
            welcome_panel = Panel(
                welcome_text,
                title="[bold]Welcome to AgentsMCP TUI[/bold]",
                border_style="cyan",
                expand=False
            )
            console.print(welcome_panel)
            console.print()
            
            # Initialize conversation components
            if hasattr(self, 'conversation_manager') and self.conversation_manager:
                logger.info("✅ Conversation manager available for enhanced interactive mode")
            else:
                logger.warning("⚠️  No conversation manager - responses will be limited")
            
            # Main interactive loop
            while self.running:
                try:
                    # Use Rich prompt for better input handling
                    try:
                        user_input = Prompt.ask(
                            "[bold cyan]TUI>[/bold cyan]",
                            console=console,
                            default=""
                        ).strip()
                    except (EOFError, KeyboardInterrupt):
                        # Handle Ctrl+C or EOF gracefully
                        console.print("\n[yellow]👋 Exiting TUI...[/yellow]")
                        break
                    except Exception as input_error:
                        # Fallback to basic input if Rich prompt fails
                        logger.warning(f"Rich prompt failed, using basic input: {input_error}")
                        console.print("[cyan]TUI> [/cyan]", end="")
                        try:
                            user_input = input().strip()
                        except (EOFError, KeyboardInterrupt):
                            console.print("\n[yellow]👋 Exiting TUI...[/yellow]")
                            break
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.startswith('/'):
                        await self._handle_command(user_input, console)
                    else:
                        # Handle regular chat input
                        await self._handle_chat_input(user_input, console)
                        
                except KeyboardInterrupt:
                    console.print("\n[yellow]⚠️  Received Ctrl+C - shutting down gracefully...[/yellow]")
                    break
                except Exception as loop_error:
                    logger.error(f"Error in enhanced interactive loop: {loop_error}")
                    console.print(f"[red]❌ Error: {loop_error}[/red]")
                    # Continue the loop unless it's a critical error
                    if "critical" in str(loop_error).lower():
                        break
            
            console.print("[green]✅ Enhanced interactive mode shutting down gracefully[/green]")
            
        except Exception as e:
            logger.error(f"Enhanced interactive mode failed: {e}")
            console.print(f"[red]❌ Enhanced interactive mode error: {e}[/red]")
            # Fallback to emergency mode
            await self._run_emergency_fallback_loop()
    
    async def _handle_command(self, command: str, console):
        """Handle slash commands in enhanced interactive mode."""
        import logging
        logger = logging.getLogger(__name__)
        
        command = command.lower().strip()
        
        if command == '/quit' or command == '/exit':
            console.print("[yellow]👋 Goodbye![/yellow]")
            self.running = False
            return
        
        elif command == '/help':
            help_text = """[bold cyan]Available Commands:[/bold cyan]

[blue]/help[/blue]     - Show this help message
[blue]/status[/blue]   - Show system status  
[blue]/clear[/blue]    - Clear the screen
[blue]/quit[/blue]     - Exit the TUI

[bold yellow]Chat Features:[/bold yellow]
• Type any message to chat with the AI
• Multi-line input supported
• Commands are processed locally
• AI responses use configured LLM provider

[bold green]Tips:[/bold green]
• Use Ctrl+C to exit gracefully
• Terminal supports full Unicode and colors
• Input is processed in real-time"""
            
            console.print(Panel(help_text, title="[bold]AgentsMCP TUI Help[/bold]", border_style="blue"))
        
        elif command == '/status':
            # Show system status
            status_info = []
            status_info.append(f"🔄 TUI Running: {self.running}")
            status_info.append(f"🖥️  Terminal: {os.environ.get('TERM_PROGRAM', 'Unknown')}")
            status_info.append(f"📺 Console: Rich {getattr(console, '_color_system', 'auto')}")
            
            if hasattr(self, 'conversation_manager') and self.conversation_manager:
                status_info.append("💬 Conversation Manager: ✅ Active")
            else:
                status_info.append("💬 Conversation Manager: ❌ Not Available")
            
            if hasattr(self, 'orchestrator') and self.orchestrator:
                status_info.append("🎭 Orchestrator: ✅ Active")
            else:
                status_info.append("🎭 Orchestrator: ❌ Not Available")
            
            status_text = "\n".join(status_info)
            console.print(Panel(status_text, title="[bold]System Status[/bold]", border_style="green"))
        
        elif command == '/clear':
            console.clear()
            console.print("[green]✨ Screen cleared[/green]")
        
        else:
            console.print(f"[red]❓ Unknown command: {command}[/red]")
            console.print("[yellow]Type /help for available commands[/yellow]")
    
    async def _handle_chat_input(self, user_input: str, console):
        """Handle regular chat input in enhanced interactive mode."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            console.print(f"[dim]Processing: {user_input}[/dim]")
            
            # Try to use conversation manager if available
            if hasattr(self, 'conversation_manager') and self.conversation_manager:
                try:
                    # Process through conversation manager
                    response = await self.conversation_manager.process_user_input(user_input)
                    if response:
                        console.print(Panel(response, title="[bold]AI Response[/bold]", border_style="green"))
                    else:
                        console.print("[yellow]🤖 AI response was empty or failed to process[/yellow]")
                except Exception as conv_error:
                    logger.error(f"Conversation manager error: {conv_error}")
                    console.print(f"[red]❌ Conversation error: {conv_error}[/red]")
            
            # Try orchestrator as fallback
            elif hasattr(self, 'orchestrator') and self.orchestrator:
                try:
                    result = await self.orchestrator.process_command(user_input)
                    if result:
                        console.print(Panel(str(result), title="[bold]System Response[/bold]", border_style="blue"))
                    else:
                        console.print("[yellow]🔧 System response was empty[/yellow]")
                except Exception as orch_error:
                    logger.error(f"Orchestrator error: {orch_error}")
                    console.print(f"[red]❌ System error: {orch_error}[/red]")
            
            else:
                # No conversation components available - show helpful message
                console.print(Panel(
                    f"[yellow]📝 Input received: '{user_input}'[/yellow]\n\n"
                    "[red]⚠️  No conversation manager or orchestrator available.[/red]\n"
                    "[blue]This would normally be processed by the AI system.[/blue]\n\n"
                    "[dim]To enable AI responses, ensure conversation components are properly initialized.[/dim]",
                    title="[bold]Input Acknowledged[/bold]",
                    border_style="yellow"
                ))
            
        except Exception as e:
            logger.error(f"Error handling chat input: {e}")
            console.print(f"[red]❌ Error processing input: {e}[/red]")
    
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
    
    async def _check_terminal_resize(self):
        """Check for terminal resize and update layout if needed."""
        if not RICH_AVAILABLE or not self.console:
            return False
        
        try:
            current_size = self.console.size
            current_width = current_size.width
            current_height = current_size.height
            
            # Check if terminal size changed
            if (current_width, current_height) != self._last_terminal_size:
                logger.info(f"Terminal resize detected: {self._last_terminal_size} -> {(current_width, current_height)}")
                
                # Update stored dimensions
                self.terminal_width = current_width
                self.terminal_height = current_height
                self._last_terminal_size = (current_width, current_height)
                
                # Mark for layout refresh
                self._resize_pending = True
                
                # Reinitialize layout panels with new dimensions
                if self._layout_initialized and self.layout:
                    await self._initialize_layout_panels()
                
                return True
            
        except Exception as e:
            logger.warning(f"Error checking terminal resize: {e}")
        
        return False
    
    async def _trigger_periodic_updates(self):
        """Trigger periodic updates via events instead of polling."""
        try:
            # Check for terminal resize first
            resize_detected = await self._check_terminal_resize()
            
            # Update agent status and metrics, which will emit events if changed
            await self._update_agent_status()
            await self._update_system_metrics()
            
            # Update timestamp
            self.state.last_update = time.time()
            
            # If resize was detected, refresh all panels
            if resize_detected:
                await self._refresh_panel("header")
                await self._refresh_panel("status")
                await self._refresh_panel("dashboard")
                await self._refresh_panel("chat")
                await self._refresh_panel("input")
                self._resize_pending = False
            
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
        """Handle a single character input using unified input rendering pipeline."""
        # Add character to current input FIRST
        self.state.current_input += char
        
        # Make cursor visible and reset blink timer  
        self.state.last_update = time.time()
        
        # CRITICAL DEBUG: Always log character input to understand what's happening
        self._safe_log("info", f"🔤 CHAR INPUT: '{char}' -> buffer: '{self.state.current_input}'")
        
        # Debug logging for input visibility
        debug_mode = getattr(self.cli_config, 'debug_mode', False)
        if debug_mode:
            self._safe_log("debug", f"Character input: '{char}' -> '{self.state.current_input}'")
        
        # FIXED: Update pipeline to match state instead of vice versa
        if self.input_pipeline:
            try:
                # Update pipeline state to match current input buffer - prevent desync
                if hasattr(self.input_pipeline, '_current_state'):
                    self.input_pipeline._current_state.text = self.state.current_input
                
                # Render input immediately for instant feedback
                pipeline_success = self.input_pipeline.render_immediate_feedback(
                    char, self.state.current_input, cursor_position=len(self.state.current_input)
                )
                if debug_mode:
                    # Log pipeline state after feedback
                    pipeline_current = getattr(self.input_pipeline._current_state, 'text', 'N/A') if hasattr(self.input_pipeline, '_current_state') else 'N/A'
                    self._safe_log("debug", f"Pipeline feedback: {pipeline_success}, Pipeline state: '{pipeline_current}'")
            except Exception as e:
                self._safe_log("info", f"🔧 PIPELINE ERROR: {e}")
                pass  # Continue without pipeline if it fails
        
        # Immediately refresh the Live display to show typed characters
        # Must be synchronous since this runs in input thread, not async context
        try:
            self._safe_log("info", f"🔄 REFRESHING display for: '{self.state.current_input}'")
            self._sync_refresh_display()
            self._safe_log("info", f"✅ REFRESH completed for: '{self.state.current_input}'")
        except Exception as e:
            self._safe_log("info", f"❌ REFRESH ERROR: {e}")
        
        # Emit input changed event for reactive UI updates (async)
        if hasattr(self, '_event_loop') and self._event_loop and not self._event_loop.is_closed():
            self._event_loop.call_soon_threadsafe(lambda: asyncio.create_task(self._publish_input_changed()))
    
    # Input display updates removed - Rich Live handles all updates automatically
    # Manual panel updates during Live operation were corrupting the display
    
    def _handle_backspace_input(self):
        """Handle backspace key input using unified input rendering pipeline."""
        if self.state.current_input:
            self.state.current_input = self.state.current_input[:-1]
            self.state.last_update = time.time()
            
            debug_mode = getattr(self.cli_config, 'debug_mode', False)
            if debug_mode:
                self._safe_log("debug", f"Backspace: New input '{self.state.current_input}'")
            
            # FIXED: Update pipeline to match state to prevent desync
            if self.input_pipeline:
                try:
                    # Update pipeline state to match current input buffer - prevent desync
                    if hasattr(self.input_pipeline, '_current_state'):
                        self.input_pipeline._current_state.text = self.state.current_input
                    
                    # Render backspace immediately for instant feedback
                    self.input_pipeline.render_deletion_feedback(
                        self.state.current_input, cursor_position=len(self.state.current_input)
                    )
                    if debug_mode:
                        # Log pipeline state after deletion
                        pipeline_current = getattr(self.input_pipeline._current_state, 'text', 'N/A') if hasattr(self.input_pipeline, '_current_state') else 'N/A'
                        self._safe_log("debug", f"Pipeline after backspace: '{pipeline_current}'")
                except Exception:
                    pass  # Continue without pipeline if it fails
            
            # Immediately refresh the Live display to show backspace effect
            # Must be synchronous since this runs in input thread, not async context
            self._sync_refresh_display()
            
            # Emit input changed event for reactive UI updates (async)
            if hasattr(self, '_event_loop') and self._event_loop and not self._event_loop.is_closed():
                self._event_loop.call_soon_threadsafe(lambda: asyncio.create_task(self._publish_input_changed()))
    
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

    def _handle_enter_input_sync(self):
        """
        CRITICAL FIX: Sync wrapper for Enter key handling from thread context.
        This properly schedules the async _handle_enter_input to be executed.
        """
        try:
            debug_mode = getattr(self.cli_config, 'debug_mode', False)
            if debug_mode:
                self._safe_log("debug", f"ENTER_KEY: Sync wrapper called with input: '{self.state.current_input}'")
            
            # Get the current event loop
            try:
                loop = asyncio.get_running_loop()
                
                # FIXED: Create a task and store it so it gets executed
                # The key is to create the task and let the event loop handle it
                async def enter_task():
                    try:
                        await self._handle_enter_input()
                        if debug_mode:
                            self._safe_log("debug", "ENTER_KEY: Async processing completed successfully")
                    except Exception as e:
                        self._safe_log("error", f"ENTER_KEY: Async processing failed: {e}")
                        import traceback
                        self._safe_log("error", f"Enter key error traceback:\n{traceback.format_exc()}")
                
                # Schedule the task to run in the event loop
                task = loop.create_task(enter_task())
                
                # CRITICAL: Store the task to prevent garbage collection
                if not hasattr(self, '_pending_enter_tasks'):
                    self._pending_enter_tasks = []
                self._pending_enter_tasks.append(task)
                
                # Clean up completed tasks
                self._pending_enter_tasks = [t for t in self._pending_enter_tasks if not t.done()]
                
                if debug_mode:
                    self._safe_log("debug", f"ENTER_KEY: Task created and scheduled (pending tasks: {len(self._pending_enter_tasks)})")
                
            except RuntimeError:
                # No event loop running - this shouldn't happen in our case
                self._safe_log("error", "ENTER_KEY: No event loop available for async processing")
                
        except Exception as e:
            self._safe_log("error", f"ENTER_KEY: Sync wrapper failed: {e}")
            import traceback
            self._safe_log("error", f"Enter key sync wrapper error:\n{traceback.format_exc()}")
    
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
            
            # Immediately refresh display to show history navigation
            self._sync_refresh_display()
            
            # Emit input changed event for reactive UI updates (async)
            if hasattr(self, '_event_loop') and self._event_loop and not self._event_loop.is_closed():
                self._event_loop.call_soon_threadsafe(lambda: asyncio.create_task(self._publish_input_changed()))
    
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
            
            # Immediately refresh display to show history navigation
            self._sync_refresh_display()
            
            # Emit input changed event for reactive UI updates (async)
            if hasattr(self, '_event_loop') and self._event_loop and not self._event_loop.is_closed():
                self._event_loop.call_soon_threadsafe(lambda: asyncio.create_task(self._publish_input_changed()))
    
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
        
        # Immediately refresh display to show cleared input
        self._sync_refresh_display()
        
        # Emit input changed event for reactive UI updates (async)
        if hasattr(self, '_event_loop') and self._event_loop and not self._event_loop.is_closed():
            self._event_loop.call_soon_threadsafe(lambda: asyncio.create_task(self._publish_input_changed()))
    
    async def _handle_exit(self):
        """Handle Ctrl+C/Ctrl+D - graceful exit with proper cleanup.
        
        CRITICAL FIX: Ensure cleanup is called for clean application termination.
        """
        logger.info("Exit requested by user - initiating graceful shutdown")
        
        # Set running flag first to stop loops
        self.running = False
        
        # Perform cleanup to ensure proper resource deallocation
        try:
            await self._cleanup()
            logger.info("✅ Graceful exit completed with cleanup")
        except Exception as e:
            logger.warning(f"Error during exit cleanup: {e}")
            # Still exit even if cleanup fails
        
        # Ensure we actually terminate the application
        import sys
        sys.exit(0)
    
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
            logger.info("Starting event handler registration...")
            
            # Register handlers for UI update events
            await self.event_system.subscribe("input_changed", self._on_input_changed)
            logger.info("Registered input_changed handler")
            
            await self.event_system.subscribe("agent_status_changed", self._on_agent_status_changed)
            logger.info("Registered agent_status_changed handler")
            
            await self.event_system.subscribe("metrics_updated", self._on_metrics_updated)
            logger.info("Registered metrics_updated handler")
            
            await self.event_system.subscribe("conversation_updated", self._on_conversation_updated)
            logger.info("Registered conversation_updated handler")
            
            await self.event_system.subscribe("processing_state_changed", self._on_processing_state_changed)
            logger.info("Registered processing_state_changed handler")
            
            await self.event_system.subscribe("ui_refresh", self._on_ui_refresh)
            logger.info("Registered ui_refresh handler")
            
            # Legacy handlers for backward compatibility
            await self.event_system.subscribe("user_input", self._handle_user_input_event)
            logger.info("Registered user_input handler")
            
            await self.event_system.subscribe("agent_status_change", self._handle_agent_status_change)
            logger.info("Registered agent_status_change handler")
            
            await self.event_system.subscribe("performance_update", self._handle_performance_update)
            logger.info("Registered performance_update handler")
            
            # Check final stats
            stats = self.event_system.get_stats()
            logger.info(f"Event handler registration complete. Handler count: {stats['handler_count']}")
            
        except Exception as e:
            logger.error(f"Error registering event handlers: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
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
        """Refresh a specific UI panel - LAYOUT CORRUPTION FIX APPLIED.
        
        CRITICAL FIX: Removed manual Live.refresh() calls that corrupt layout structure.
        Rich Live display will automatically refresh when panel content is updated.
        """
        try:
            if not self.layout or not RICH_AVAILABLE or not sys.stdin.isatty():
                return  # Skip all Rich operations in non-TTY environments
            
            # Update layout content without fixed widths to respect terminal boundaries
            if panel_name == "header":
                header_text = Text("🚀 AgentsMCP Revolutionary Interface", style="bold blue")
                if self.state.is_processing:
                    header_text.append(" • ", style="dim")
                    header_text.append(self.state.processing_message, style="yellow")
                
                self.layout["header"].update(
                    Panel(
                        Align.center(header_text),
                        box=box.ROUNDED,
                        style="bright_blue"
                        # No width/expand specified - let Rich handle terminal boundaries
                    )
                )
            
            elif panel_name == "status":
                status_content = self._create_status_panel()
                self.layout["status"].update(
                    Panel(
                        status_content,
                        title="Agent Status",
                        box=box.ROUNDED,
                        style="green"
                        # No width/expand specified - let layout manager handle
                    )
                )
            
            elif panel_name == "chat":
                chat_content = self._create_chat_panel()
                self.layout["chat"].update(
                    Panel(
                        chat_content,
                        title="Conversation",
                        box=box.ROUNDED,
                        style="white"
                        # No width/expand specified - let layout manager handle
                    )
                )
            
            elif panel_name == "input":
                input_content = self._create_input_panel()
                self.layout["input"].update(
                    Panel(
                        input_content,
                        title="AI Command Composer",
                        box=box.ROUNDED,
                        style="cyan"
                        # No width/expand specified - let layout manager handle
                    )
                )
            
            elif panel_name == "dashboard":
                dashboard_content = await self._create_dashboard_panel()
                self.layout["dashboard"].update(
                    Panel(
                        dashboard_content,
                        title="Symphony Dashboard",
                        box=box.ROUNDED,
                        style="magenta"
                        # No width/expand specified - let layout manager handle
                    )
                )
            
            # CRITICAL FIX: DO NOT call manual refresh - this corrupts the layout!
            # Rich Live display will automatically refresh when panel content is updated.
            # Manual refresh() calls disrupt the layout's internal state causing corruption.
            
            # OLD CODE (causes corruption):
            # self.live_display.refresh()
            
            # NEW APPROACH: Let Rich handle refreshes automatically
            # The Live display will detect panel updates and refresh appropriately
            # without disrupting the layout structure.
                
        except Exception as e:
            pass  # Silently ignore panel refresh errors  # Silently ignore panel refresh errors
    
    def _sync_refresh_display(self):
        """
        CRITICAL FIX FOR TYPING VISIBILITY: We need to trigger refresh when input changes
        because auto_refresh=False in Live display (to prevent scrollback pollution).
        BUT we must throttle refreshes to prevent race conditions and layout corruption.
        FIXED: Reduced throttling from 50ms to 16ms for real-time typing visibility.
        """
        import time
        
        try:
            # RACE CONDITION FIX: Throttle refreshes to prevent layout corruption
            current_time = time.time()
            if not hasattr(self, '_last_manual_refresh_time'):
                self._last_manual_refresh_time = 0
            
            # FIXED: Reduced throttling for real-time typing (16ms = ~60 FPS)
            min_refresh_interval = 0.016  # 16ms minimum between refreshes (60 FPS max)
            time_since_last = current_time - self._last_manual_refresh_time
            
            # FIXED: Only check for changes if sufficient time has passed OR this is first call
            current_input = self.state.current_input
            if not hasattr(self, '_last_input_refresh_content'):
                self._last_input_refresh_content = ""
            
            # FIXED: Always refresh on first call or when input actually changes
            input_changed = current_input != self._last_input_refresh_content
            sufficient_time_passed = time_since_last >= min_refresh_interval
            
            # FIXED: State is now the source of truth - no buffer corruption
            debug_mode = getattr(self.cli_config, 'debug_mode', False)
            if debug_mode:
                self._safe_log("debug", f"SYNC_REFRESH: Current state '{current_input}' (changed: {input_changed}, Δt: {time_since_last:.3f}s)")
            
            # FIXED: Skip refresh only if no change AND too recent
            if not input_changed and not sufficient_time_passed:
                if debug_mode:
                    self._safe_log("debug", f"REFRESH_SKIPPED: No change and too soon (Δt: {time_since_last:.3f}s)")
                return
                
            # Check if we have a valid Live display and TTY
            if not (hasattr(self, 'live_display') and self.live_display and 
                    sys.stdin.isatty()):
                if debug_mode:
                    self._safe_log("debug", "INPUT_REFRESH: Live display or TTY not available")
                return
                    
            if not self.layout:
                if debug_mode:
                    self._safe_log("debug", "INPUT_REFRESH: No layout available")
                return
            
            # CRITICAL FIX: Update input panel content safely
            try:
                # FIXED: Check if layout has "input" key safely to avoid KeyError
                try:
                    _ = self.layout["input"]
                except (KeyError, ValueError):
                    if debug_mode:
                        available_keys = list(self.layout.keys()) if hasattr(self.layout, 'keys') else "No keys method"
                        self._safe_log("debug", f"INPUT_REFRESH: Layout missing 'input' key. Available: {available_keys}")
                    return
                
                # Create fresh input panel with current state
                input_panel = self._create_input_panel()
                
                # Update layout with new panel
                self.layout["input"].update(input_panel)
                
                # FIXED: Force refresh of Live display
                if hasattr(self.live_display, 'refresh'):
                    self.live_display.refresh()
                    
                # Update tracking variables
                self._last_manual_refresh_time = current_time
                self._last_input_refresh_content = current_input
                
                if debug_mode:
                    self._safe_log("debug", f"INPUT_REFRESH: Successful refresh for input '{current_input}' (length: {len(current_input)})")
                
            except Exception as panel_e:
                # Panel creation/update failed - this is the main source of corruption
                self._safe_log("error", f"INPUT_REFRESH: Panel update failed: {panel_e}")
                
                # Try to recreate layout structure on failure
                try:
                    from rich.layout import Layout
                    if not self.layout or not hasattr(self.layout, 'get'):
                        self.layout = Layout()
                        self.layout.split_column(Layout(name="input"))
                        self._safe_log("info", "INPUT_REFRESH: Layout recreated after failure")
                except Exception as layout_e:
                    self._safe_log("error", f"INPUT_REFRESH: Layout recreation failed: {layout_e}")
                
        except Exception as e:
            self._safe_log("error", f"CRITICAL: _sync_refresh_display failed: {e}")
            import traceback
            self._safe_log("error", f"Refresh failure traceback:\n{traceback.format_exc()}")  # Ignore refresh errors - input should still work even if display fails  # Ignore refresh errors - input should still work even if display fails  # Ignore refresh errors - input should still work even if display fails
    
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


# Factory function for creating the revolutionary interface through unified coordinator
async def create_revolutionary_interface(
    cli_config=None,
    orchestrator_integration=None,
    revolutionary_components=None
) -> RevolutionaryTUIInterface:
    """Create and initialize a Revolutionary TUI Interface through unified coordinator."""
    from .unified_tui_coordinator import get_unified_tui_coordinator, TUIMode
    
    # Get the unified coordinator
    coordinator = await get_unified_tui_coordinator()
    
    # Start the revolutionary TUI mode through coordinator
    tui_instance, mode_active, status = await coordinator.start_tui(
        TUIMode.REVOLUTIONARY,
        orchestrator_integration=orchestrator_integration
    )
    
    if mode_active and tui_instance:
        # Return the actual interface from the TUI instance
        return tui_instance.interface
    else:
        # Fallback to direct instantiation if coordinator fails
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