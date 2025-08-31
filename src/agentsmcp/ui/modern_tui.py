# -*- coding: utf-8 -*-
"""
ModernTUI – the core of the progressive‑disclosure, Rich‑based user interface
for AgentsMCP.

The implementation focuses on a solid, testable foundation:

*   Mode management (Zen / Dashboard / Command‑Center)
*   Simple yet responsive layout built on :class:`rich.layout.Layout`
*   Async event loop that works both in a true TTY and in a non‑interactive
    fallback (plain‑stdout / stdin).
*   Tight integration points for the existing AgentsMCP services:
    ``ThemeManager``, ``ConversationManager`` and ``OrchestrationManager``.
*   Graceful degradation – if Rich cannot be used we fall back to a very
    lightweight CLI loop that still lets the user chat.

Only the *Zen* mode is fully functional at the moment; the other two modes
are scaffolded so that later tasks can flesh them out without needing to touch
the core framework.
"""

from __future__ import annotations

import asyncio
import os
import contextlib
import json
import logging
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional, List, Dict

# HTTP client for SSE
try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

# --------------------------------------------------------------------------- #
# Rich imports – guarded so that the module can still be imported on systems
# where Rich is not available (the CI tests for this repository run with Rich,
# but the guard makes the code defensive and easier to reason about).
# --------------------------------------------------------------------------- #
try:
    from rich.console import Console, RenderableType
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.align import Align
    from rich.columns import Columns
    from rich.rule import Rule
    from rich.table import Table
except Exception:  # pragma: no cover – only hit when Rich is not installed.
    Console = None  # type: ignore
    Layout = None  # type: ignore
    Panel = None  # type: ignore
    Text = None  # type: ignore
    Live = None  # type: ignore
    Align = None  # type: ignore
    Columns = None  # type: ignore
    Rule = None  # type: ignore
    Table = None  # type: ignore
RenderableType = Any

# --------------------------------------------------------------------------- #
# Enhanced chat components imports
# --------------------------------------------------------------------------- #
try:
    from .components.enhanced_chat import EnhancedChatInput
    from .components.chat_history import ChatHistoryDisplay
    from .components.realtime_input import RealTimeInputField
    from .keyboard_input import KeyboardInput, InputMode, KeyCode
except Exception as e:  # pragma: no cover
    # If the components cannot be imported we fall back to the legacy
    # input & history handling.  
    EnhancedChatInput = None   # type: ignore
    ChatHistoryDisplay = None  # type: ignore
    KeyboardInput = None  # type: ignore
    KeyCode = None  # type: ignore
    RealTimeInputField = None  # type: ignore


# --------------------------------------------------------------------------- #
# Log integration for TUI
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
try:
    logger.setLevel(logging.DEBUG)
except Exception:
    pass
@dataclass
class LogEntry:
    """A structured log entry for TUI display."""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    
    def format_for_display(self) -> str:
        """Format log entry for TUI display."""
        time_str = self.timestamp.strftime("%H:%M:%S.%f")[:-3]  # microseconds to milliseconds
        level_color = {
            "DEBUG": "dim",
            "INFO": "green", 
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red"
        }.get(self.level, "white")
        
        # Truncate logger name for cleaner display
        short_logger = self.logger_name.split('.')[-1] if '.' in self.logger_name else self.logger_name
        
        return f"[dim]{time_str}[/dim] [{level_color}]{self.level:5}[/{level_color}] [cyan]{short_logger}[/cyan]: {self.message}"


class TUILogHandler(logging.Handler):
    """Custom log handler that captures log messages for TUI display."""
    
    def __init__(self, max_entries: int = 200):
        super().__init__()
        self.log_entries: deque[LogEntry] = deque(maxlen=max_entries)
        self._tui_instance = None
        
        # Filter to only show relevant loggers
        self.relevant_loggers = {
            "agentsmcp",
            "agentsmcp.conversation.llm_client",
            "agentsmcp.conversation.dispatcher", 
            "agentsmcp.mcp.manager",
            "agentsmcp.orchestration",
            "agentsmcp.agents",
            "uvicorn",
            "fastapi"
        }
    
    def set_tui_instance(self, tui_instance):
        """Set reference to TUI instance for triggering refreshes."""
        self._tui_instance = tui_instance
        
    def emit(self, record: logging.LogRecord) -> None:
        """Handle a log record by storing it for TUI display."""
        try:
            # Filter to only relevant loggers to reduce noise
            if not any(record.name.startswith(logger) for logger in self.relevant_loggers):
                return
                
            # Skip overly verbose debug messages
            if record.levelno == logging.DEBUG and len(record.getMessage()) > 200:
                return
                
            entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created),
                level=record.levelname,
                logger_name=record.name,
                message=record.getMessage(),
                module=getattr(record, 'module', None),
                function=getattr(record, 'funcName', None),
                line_number=getattr(record, 'lineno', None)
            )
            
            self.log_entries.append(entry)
            
            # Trigger TUI refresh if we have a reference and it's showing logs
            if (self._tui_instance and 
                hasattr(self._tui_instance, '_current_page') and
                self._tui_instance._current_page == SidebarPage.LOGS):
                self._tui_instance.mark_dirty("content")
                
        except Exception:
            # Don't let log handling break the application
            pass

# --------------------------------------------------------------------------- #
# Public enums & constants
# --------------------------------------------------------------------------- #
class TUIMode(Enum):
    """All supported UI modes.

    * ``ZEN`` – Minimal chat view (default).
    * ``DASHBOARD`` – High‑level status / metrics overview.
    * ``COMMAND_CENTER`` – Full‑featured technical UI for power users.
    """
    ZEN = "zen"
    DASHBOARD = "dashboard"
    COMMAND_CENTER = "command_center"


class SidebarPage(Enum):
    """Available sidebar pages for hybrid TUI mode."""
    CHAT = "chat"
    JOBS = "jobs" 
    AGENTS = "agents"
    MODELS = "models"
    PROVIDERS = "providers"
    COSTS = "costs"
    MCP = "mcp"
    DISCOVERY = "discovery"
    LOGS = "logs"
    SETTINGS = "settings"


class FocusRegion(Enum):
    """Focusable regions in the hybrid TUI."""
    SIDEBAR = "sidebar"
    MAIN = "main"
    INPUT = "input"


# --------------------------------------------------------------------------- #
# Core ModernTUI implementation
# --------------------------------------------------------------------------- #
class ModernTUI:
    """
    The world‑class "Modern" TUI.

    Parameters
    ----------
    config:
        Global CLI configuration (parsed from the command line).
    theme_manager:
        Centralised theming service – provides colours / style tokens.
    conversation_manager:
        Handles chat history, sending, receiving and persisting messages.
    orchestration_manager:
        Persists UI‑related settings (e.g. last used mode).
    theme:
        The theme to use – ``"auto"`` selects automatically based on the
        terminal background.  Other strings are passed straight to
        :class:`ThemeManager`.
    no_welcome:
        If ``True`` suppresses the initial welcome banner.
    """

    # ------------------------------------------------------------------- #
    # Construction & basic initialisation
    # ------------------------------------------------------------------- #
    def __init__(
        self,
        config,
        theme_manager,
        conversation_manager,
        orchestration_manager,
        theme: str = "auto",
        no_welcome: bool = False,
    ) -> None:
        # Store injected dependencies
        self.config = config
        self.theme_manager = theme_manager
        self.conversation_manager = conversation_manager
        self.orchestration_manager = orchestration_manager

        # Runtime state
        self._theme_name = theme
        self._no_welcome = no_welcome
        self._current_mode: TUIMode = self._load_last_mode()
        self._running: bool = False

        # Rich console – created lazily so unit‑tests that monkey‑patch `Console`
        # continue to work.
        self._console: Optional[Console] = None
        self._layout: Optional[Layout] = None
        
        # ACCESSIBILITY: Configurable accessibility options
        self._accessibility_config = {
            "high_contrast": False,  # Enable high contrast colors
            "reduce_motion": False,  # Disable animations and blinking
            "increase_spacing": False,  # Add more spacing between elements
        }

        # Async helpers
        self._input_queue: asyncio.Queue[str] = asyncio.Queue()
        
        # Event-driven refresh support
        self._refresh_event: asyncio.Event = asyncio.Event()
        
        # Hybrid TUI state - load from persisted settings
        self._load_ui_state()
        self._chat_scroll_offset: int = 0  # For PgUp/PgDn scrolling (not persisted)
        self._command_palette_active: bool = False  # Always start inactive
        
        # Initialize render caching system
        self._render_cache = {}
        self._cache_version = {"header": 0, "content": 0, "footer": 0, "chat_body": 0, "sidebar": 0}
        
        # Frame deduplication system to prevent scrollback flooding
        self._last_rendered_hash = None
        self._render_counter = 0
        self._currently_rendering = False  # Guard against render feedback loops
        
        # CRITICAL FIX: Add render synchronization lock to prevent race conditions
        self._render_lock = asyncio.Lock()
        
        # CRITICAL FIX: Request coalescing for mark_dirty calls
        self._pending_refresh_sections = set()
        self._coalesce_lock = asyncio.Lock()
        
        # CRITICAL FIX: Flag for immediate footer refresh (bypass debounce)
        self._immediate_footer_refresh = False
        
        # Enhanced chat components - initialized later after console is ready
        self.enhanced_input = None
        self.chat_history = None
        self.realtime_input = None
        
        # Log handler for TUI integration
        self.log_handler = TUILogHandler(max_entries=200)
        self.log_handler.set_tui_instance(self)
        
        # Keyboard input handler for per-key events
        self._keyboard_input = None
        self._keyboard_task = None
        
        # SSE listener for real-time status updates
        self._sse_task = None
        self._last_status_update = None
        
        # Error handling and rate limiting
        self._error_counts = {}  # Track error counts per render section
        self._max_errors_per_section = 3  # Max errors before fallback
        self._error_suppression = {}  # Track suppressed errors
        self._last_error_log_time = {}  # Rate limit error logging
        
        # Track whether we're inside a Live() render context
        self._in_live_context = False
        
        # Initialize console early for RealTimeInputField
        if Console is not None:
            try:
                self._console = Console(force_terminal=True)
            except Exception:
                self._console = None
        else:
            self._console = None
        
        # Track typing activity for smart refresh rates
        self._last_keypress_time = 0.0
        self._typing_timeout = 2.0  # Consider user stopped typing after 2 seconds
        # Debug keys overlay
        self._debug_keys = bool(os.getenv("AGENTS_TUI_DEBUG_KEYS"))
        self._debug_last_key = ""
        # Track input engine for diagnostics: 'native' or 'ptk'
        self._input_engine = "native"
        
        # CRITICAL FIX: Initialize keyboard input BEFORE RealTimeInputField to avoid blocking
        # Initialize keyboard input for per-key event handling
        if KeyboardInput is not None:
            try:
                self._keyboard_input = KeyboardInput()
                if not self._keyboard_input.is_interactive:
                    # CRITICAL FIX: Don't force interactive mode if terminal access isn't available
                    # Instead, enhance line-based input to be more responsive
                    try:
                        self._print_system_message("Running in line-based input mode - enhanced for better UX")
                    except:
                        pass  # _print_system_message may not be available yet
                    
                    # Add hybrid mode flag to enable special handling
                    self._keyboard_input.hybrid_mode = True
            except Exception:
                self._keyboard_input = None
        
        # FIXED: Initialize RealTimeInputField now that console is available
        if RealTimeInputField is not None and self._console is not None:
            try:
                self.realtime_input = RealTimeInputField(
                    console=self._console,
                    prompt=">>> ",
                    placeholder="Type your message here... (or try ? for help)",
                    max_width=None,
                    max_height=3
                )
                # FIXED: Connect input events and ensure proper focus
                self._connect_input_events()
                self._ensure_input_focus()
            except Exception:
                # Fallback gracefully if initialization fails
                self.realtime_input = None
        else:
            self.realtime_input = None

    # ------------------------------------------------------------------- #
    # Debug logging helper
    # ------------------------------------------------------------------- #
    def _debug_log(self, message: str) -> None:
        """Consistent debug logging that integrates with the TUI log handler."""
        # Debug logging disabled to prevent console flooding during TUI operation
        # Temporarily enabled for input echo debugging
        if "realtime_input" in message.lower() or "footer" in message.lower():
            try:
                import sys
                print(f"DEBUG: {message}", file=sys.stderr, flush=True)
            except:
                pass

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #
    async def run(self) -> None:
        """
        Entry point for the Modern TUI.

        This method orchestrates the asynchronous life‑cycle:

        1. Detect if we are attached to a TTY – if not, fall back to a very thin
           synchronous CLI loop.
        2. Initialise the Rich ``Console`` and layout (if possible).
        3. Enter an event loop that:
           * reads user input,
           * forwards it to the appropriate manager,
           * renders the UI according to the current mode,
           * reacts to ``Ctrl‑C`` or ``/mode`` commands.
        """
        # ------------------------------------------------------------------- #
        # 1️⃣  Rich TUI with graceful degradation - removed restrictive TTY check
        # ------------------------------------------------------------------- #
        # Note: We allow Rich rendering in non-TTY environments (Docker, CI, IDEs, etc.)
        # since Rich Console with force_terminal=True can handle these cases gracefully.
        # Only fall back to basic CLI if Console import is not available.
        if not Console:
            await self._fallback_cli()
            return
            
        # ------------------------------------------------------------------- #
        # 2️⃣  Initialise Rich console & layout (if not already done)
        # ------------------------------------------------------------------- #
        if self._console is None:
            try:
                # Check if theme_manager has rich_theme method, otherwise use default
                if hasattr(self.theme_manager, 'rich_theme'):
                    console_theme = self.theme_manager.rich_theme()
                else:
                    console_theme = None
                
                self._console = Console(force_terminal=True, theme=console_theme)
            except Exception:
                self._console = Console(force_terminal=True)
        # Initialize enhanced chat components now that console is ready
        if EnhancedChatInput is not None and ChatHistoryDisplay is not None:
            try:
                self.enhanced_input = EnhancedChatInput(console=self._console)
                self.chat_history = ChatHistoryDisplay(
                    console=self._console,
                    max_history=100,
                )
            except Exception:
                # In case the ctor raises (e.g. missing runtime deps)
                self.enhanced_input = None
                self.chat_history = None
                
        # Keyboard input was already initialized earlier to avoid blocking issues
        
        self._layout = self._build_layout()
        self._running = True
        
        # Initialize input mode tracking for auto-switch and health indicator
        self._input_mode = "per_key" if (self._keyboard_input and self._keyboard_input.is_interactive) else "line"

        # If key debugging is enabled but we're not in per-key mode, surface a helpful hint
        if getattr(self, "_debug_keys", False) and self._input_mode != "per_key":
            self._enqueue_tui_log(
                "INFO",
                "AGENTS_TUI_DEBUG_KEYS is set, but input is line-based (no TTY). "
                "Per-key capture is unavailable; use a real terminal to see live key events."
            )
        
        # Start input cursor blink if available
        try:
            if self.realtime_input is not None:
                await self.realtime_input.start_cursor_blink()
        except Exception:
            pass

        # Install TUI log handler to capture log messages (prevent console flooding)
        # Attach to root and temporarily remove StreamHandlers (stdout/stderr) during TUI run.
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        self._removed_stream_handlers = []
        self._removed_child_stream_handlers = {}
        try:
            for h in list(root_logger.handlers):
                if h is self.log_handler:
                    continue
                if isinstance(h, logging.StreamHandler):
                    self._removed_stream_handlers.append(h)
                    root_logger.removeHandler(h)
            # Also remove StreamHandlers from child loggers to stop mixed console output
            for name, logger_obj in logging.Logger.manager.loggerDict.items():
                try:
                    if isinstance(logger_obj, logging.Logger):
                        removed = []
                        for h in list(logger_obj.handlers):
                            if isinstance(h, logging.StreamHandler):
                                logger_obj.removeHandler(h)
                                removed.append(h)
                        if removed:
                            self._removed_child_stream_handlers[name] = removed
                        # Ensure child loggers propagate up so our TUI handler receives logs
                        logger_obj.propagate = True
                except Exception:
                    continue
        except Exception:
            # Non-fatal
            self._removed_stream_handlers = []

        # Prepare welcome to render inside Live (avoid pre-Live scrollback flooding)
        self._show_welcome = not self._no_welcome

        # ------------------------------------------------------------------- #
        # 3️⃣  Spin up the background tasks
        # ------------------------------------------------------------------- #
        if self._keyboard_input and self._keyboard_input.is_interactive:
            # Use per-key input for interactive terminals
            input_task = asyncio.create_task(self._read_keyboard_input())
        else:
            # Fallback to line-based input for non-interactive environments
            input_task = asyncio.create_task(self._read_input())
            
        # Start SSE listener for real-time status updates
        await self._start_sse_listener()

        # Track console size and start a resize monitor to keep layout in sync
        try:
            self._last_console_size = getattr(self._console, 'size', None)
        except Exception:
            self._last_console_size = None
        self._resize_task = asyncio.create_task(self._monitor_console_resize())

        # Event-driven Live rendering with clean single-frame output
        # Use alternate screen to prevent scrollback flooding; control refresh explicitly
        self._in_live_context = True
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        # Enable bracketed paste mode so supported terminals emit single paste events
        try:
            import sys as _sys
            _sys.stdout.write('\x1b[?2004h')
            _sys.stdout.flush()
        except Exception:
            pass

        with Live(
            self._render(), 
            console=self._console, 
            refresh_per_second=20,
            auto_refresh=False,
            screen=True,
            transient=False
        ) as live:
            try:
                # Suppress stray prints from other subsystems (route to sink)
                _stdout_sink = io.StringIO()
                _stderr_sink = io.StringIO()
                with redirect_stdout(_stdout_sink), redirect_stderr(_stderr_sink):
                    while self._running:
                        # Immediate footer fast path to eliminate 1-char lag
                        if getattr(self, '_immediate_footer_refresh', False):
                            self._immediate_footer_refresh = False
                            async with self._render_lock:
                                new_layout = self._render()
                                live.update(new_layout)
                                self._last_rendered_hash = self._get_layout_hash(new_layout)
                                self._render_counter += 1
                                self._last_frame_time = time.time()
                            continue
                        # Wait for either user input OR a refresh request
                        get_task = asyncio.create_task(self._input_queue.get())
                        refresh_task = asyncio.create_task(self._refresh_event.wait())
                        done, pending = await asyncio.wait(
                            [get_task, refresh_task],
                            return_when=asyncio.FIRST_COMPLETED
                        )
                    
                    # Cancel pending tasks to prevent resource leaks
                    for task in pending:
                        task.cancel()
                        
                    # Handle user input if received
                    if get_task in done:
                        try:
                            user_input = get_task.result()
                            await self._handle_user_input(user_input)
                        except Exception:
                            pass
                    
                    # Handle refresh requests with frame deduplication
                    if self._should_refresh() or self._is_user_typing():
                        # CRITICAL FIX: Clear the refresh event BEFORE checking content to prevent feedback loops
                        self._refresh_event.clear()
                        
                        # Check for immediate footer refresh flag - bypass all debounce
                        immediate_refresh = getattr(self, '_immediate_footer_refresh', False)
                        if immediate_refresh:
                            # Reset flag immediately to prevent duplicate processing
                            self._immediate_footer_refresh = False
                        
                        # CRITICAL FIX: Single refresh path with minimal debounce for typing
                        if not immediate_refresh and not self._is_user_typing():
                            # Light debounce only for non-immediate refreshes
                            await asyncio.sleep(0.02)
                        
                        # CRITICAL FIX: Synchronize rendering to prevent frame concatenation
                        async with self._render_lock:
                            # Set rendering flag to prevent feedback loops
                            self._currently_rendering = True
                            try:
                                new_layout = self._render()
                                layout_hash = self._get_layout_hash(new_layout)
                                
                                # CRITICAL FIX: Only update if content actually changed
                                if layout_hash != self._last_rendered_hash:
                                    # Enhanced scrollback prevention: minimize frame updates
                                    # Allow immediate renders for typing/forced refresh to prevent 1-char lag
                                    current_time = time.time()
                                    min_frame_interval = 0.05
                                    if immediate_refresh or self._is_user_typing():
                                        min_frame_interval = 0.0

                                    if (not hasattr(self, '_last_frame_time')) or ((current_time - self._last_frame_time) >= min_frame_interval):
                                        live.update(new_layout)
                                        self._last_rendered_hash = layout_hash
                                        self._render_counter += 1
                                        self._last_frame_time = current_time
                                    # If too recent, skip this update (content will be rendered on next cycle)
                            finally:
                                self._currently_rendering = False
                            
            except (KeyboardInterrupt, SystemExit):
                self._running = False
            finally:
                input_task.cancel()
                # Give the background task a chance to clean up.
                with contextlib.suppress(asyncio.CancelledError):
                    await input_task
                # Stop cursor blink when leaving TUI
                try:
                    if self.realtime_input is not None:
                        self.realtime_input.stop_cursor_blink()
                except Exception:
                    pass
        # Disable bracketed paste mode on exit of Live
        try:
            import sys as _sys
            _sys.stdout.write('\x1b[?2004l')
            _sys.stdout.flush()
        except Exception:
            pass
                
                # Stop SSE listener
                await self._stop_sse_listener()
                
                # Stop resize monitor
                try:
                    if getattr(self, '_resize_task', None):
                        self._resize_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await self._resize_task
                except Exception:
                    pass

                # Restore logging handlers to previous state
                root_logger = logging.getLogger()
                if hasattr(self, "_removed_stream_handlers"):
                    for h in self._removed_stream_handlers:
                        root_logger.addHandler(h)
                    self._removed_stream_handlers.clear()
                # Restore child logger handlers
                if hasattr(self, "_removed_child_stream_handlers"):
                    for name, handlers in list(self._removed_child_stream_handlers.items()):
                        try:
                            logger_obj = logging.getLogger(name)
                            for h in handlers:
                                logger_obj.addHandler(h)
                        except Exception:
                            pass
                    self._removed_child_stream_handlers.clear()
                try:
                    root_logger.removeHandler(self.log_handler)
                except Exception:
                    pass
                self._in_live_context = False

        # Persist UI state for next launch.
        try:
            if hasattr(self.orchestration_manager, 'save_user_settings'):
                ui_state = {
                    "ui.last_mode": self._current_mode.value,
                    "ui.sidebar_collapsed": self._sidebar_collapsed,
                    "ui.current_page": self._current_page.value,
                    "ui.current_focus": self._current_focus.value,
                }
                self.orchestration_manager.save_user_settings(ui_state)
        except Exception:
            pass  # Graceful failure if persistence isn't available

    async def _monitor_console_resize(self) -> None:
        """Monitor terminal size and trigger layout refresh on change."""
        try:
            while self._running:
                try:
                    size = getattr(self._console, 'size', None)
                    if size and getattr(self, '_last_console_size', None):
                        if (size.width != self._last_console_size.width) or (size.height != self._last_console_size.height):
                            self._last_console_size = size
                            # Invalidate all sections and request refresh
                            for sec in ("header", "content", "footer", "sidebar"):
                                if sec in self._cache_version:
                                    self._cache_version[sec] += 1
                            self._refresh_event.set()
                    elif size:
                        self._last_console_size = size
                except Exception:
                    pass
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            return

    # ------------------------------------------------------------------- #
    # Event-driven refresh API
    # ------------------------------------------------------------------- #
    def mark_dirty(self, section: str = "all") -> None:
        """Mark UI section as needing refresh with request coalescing.
        
        Args:
            section: Which section to invalidate ("header", "content", "footer", "all")
        """
        # Always coalesce requests, even if rendering, so we don't miss fast events
        # CRITICAL FIX: Use async task for coalescing to avoid blocking
        asyncio.create_task(self._coalesce_mark_dirty(section))
    
    async def _coalesce_mark_dirty(self, section: str) -> None:
        """Coalesce mark_dirty requests to prevent excessive refreshes."""
        async with self._coalesce_lock:
            # Add section to pending refresh set
            if section == "all":
                self._pending_refresh_sections = {"header", "content", "footer", "sidebar"}
            else:
                self._pending_refresh_sections.add(section)
            
            # CRITICAL FIX: Enhanced debouncing with adaptive timing
            import time
            now = time.time()
            
            # Initialize timing if needed
            if not hasattr(self, '_last_dirty_time'):
                self._last_dirty_time = {}
            
            # CRITICAL FIX: Optimized debounce times for responsiveness
            section_debounce = {
                "footer": 0.02,    # Input field - CRITICAL: Ultra-fast for typing
                "header": 0.3,     # Status info
                "content": 0.15,   # Chat content
                "sidebar": 0.5,    # Sidebar
            }
            
            # Use the shortest debounce time for any pending section
            min_debounce = min(section_debounce.get(s, 0.2) for s in self._pending_refresh_sections)
            last_time = max(self._last_dirty_time.get(s, 0.0) for s in self._pending_refresh_sections)
            
            # CRITICAL FIX: Only trigger refresh if enough time has passed
            if now - last_time > min_debounce:
                # Invalidate cache for all pending sections
                for pending_section in self._pending_refresh_sections:
                    if pending_section in self._cache_version:
                        self._cache_version[pending_section] += 1
                    self._last_dirty_time[pending_section] = now
                
                # Clear pending set and trigger refresh
                self._pending_refresh_sections.clear()
                self._refresh_event.set()
    
    def _get_cached_panel(self, section: str, generator_func) -> RenderableType:
        """Get cached panel or generate new one if cache is invalid.
        
        Args:
            section: Cache section name ("header", "content", "footer")
            generator_func: Function to generate the panel if not cached
            
        Returns:
            Cached or newly generated renderable
        """
        cache_key = f"{section}_{self._cache_version[section]}"
        
        if cache_key not in self._render_cache:
            # Generate new panel and cache it
            self._render_cache[cache_key] = generator_func()
            
            # Clean old cache entries for this section
            old_keys = [k for k in self._render_cache.keys() if k.startswith(f"{section}_") and k != cache_key]
            for old_key in old_keys:
                del self._render_cache[old_key]
        
        return self._render_cache[cache_key]
        
    def _update_section_if_changed(self, section: str, new_content: RenderableType) -> bool:
        """Update layout section only if content has actually changed. Returns True if updated."""
        current_key = f"current_{section}"
        
        # CRITICAL FIX: Enhanced content comparison with better normalization
        try:
            # For Panel objects, extract meaningful content for comparison
            if hasattr(new_content, 'renderable') and hasattr(new_content, 'title'):
                content_str = str(new_content.renderable) if new_content.renderable else ""
                title_str = str(new_content.title) if new_content.title else ""
                border_str = str(new_content.border_style) if hasattr(new_content, 'border_style') else ""
                
                # Aggressive whitespace normalization to reduce false positives
                content_str = ' '.join(content_str.split())
                title_str = ' '.join(title_str.split())
                
                # Include timestamp information for time-sensitive content but normalize it
                import re
                # Normalize timestamps to reduce micro-changes (seconds only, not milliseconds)
                content_str = re.sub(r'\d{2}:\d{2}:\d{2}\.\d+', lambda m: m.group(0).split('.')[0], content_str)
                
                new_hash = hash((content_str, title_str, border_str))
            elif hasattr(new_content, '__rich__'):
                content_str = str(new_content.__rich__())
                content_str = ' '.join(content_str.split())  # Normalize whitespace
                new_hash = hash(content_str)
            else:
                content_str = str(new_content)
                content_str = ' '.join(content_str.split())  # Normalize whitespace
                new_hash = hash(content_str)
        except Exception:
            # Fallback to simple string comparison
            new_hash = hash(str(new_content))
            
        cached_hash = self._render_cache.get(current_key)
        
        # CRITICAL FIX: Only update if content has actually changed (strict comparison)
        if cached_hash != new_hash:
            # Safety guard: check if section exists in layout using try/except
            # Rich Layout's __contains__ method doesn't work as expected and triggers __getitem__
            try:
                layout_section = self._layout[section]
                layout_section.update(new_content)
                self._render_cache[current_key] = new_hash
                return True
            except KeyError:
                # Section doesn't exist, handle special cases
                if section == "content":
                    # Handle content section differently based on sidebar state
                    try:
                        if self._sidebar_collapsed:
                            # In collapsed mode, content is mapped to main_area
                            content_layout = self._layout["main_area"]
                        else:
                            # In expanded mode, content should exist as its own section
                            content_layout = self._layout["content"]
                        
                        content_layout.update(new_content)
                        self._render_cache[current_key] = new_hash
                        return True
                    except KeyError:
                        # Even the fallback failed, skip silently to avoid crashes
                        pass
                # For other sections that don't exist, skip silently
            return False
        
        # CRITICAL FIX: Content hasn't changed, no update needed - this prevents redundant Live.update() calls
        return False
        
    def _render_with_change_tracking(self) -> list:
        """Render only changed sections and return list of sections that changed."""
        if not self._layout:
            return []
            
        mode_renderer = {
            TUIMode.ZEN: self._render_zen_with_tracking,
            TUIMode.DASHBOARD: self._render_dashboard_with_tracking,
            TUIMode.COMMAND_CENTER: self._render_command_center_with_tracking,
        }.get(self._current_mode, self._render_zen_with_tracking)
        
        try:
            return mode_renderer()
        except Exception as exc:  # pragma: no cover – defensive
            # If something goes wrong we still want the UI to stay alive.
            # Use error suppression to prevent infinite loops
            self._log_suppressed_error("mode_renderer", exc)
            
            # Show a minimal error panel only if we haven't exceeded error limits
            error_count = self._error_counts.get("mode_renderer", 0)
            if error_count < self._max_errors_per_section:
                error_panel = Panel(
                    Text.from_markup("[bold red]UI Error:[/bold red] Display temporarily unavailable"),
                    title="Error",
                    border_style="red",
                )
                self._update_section_if_changed("content", error_panel)
                return ["content"]
            else:
                # Too many errors, just return empty to prevent infinite loops
                return []

    def _should_refresh(self) -> bool:
        """Return True if a refresh has been requested."""
        return self._refresh_event.is_set()
    
    def _get_layout_hash(self, layout) -> str:
        """CRITICAL FIX: Optimized hash generation using object IDs instead of string serialization."""
        try:
            # CRITICAL FIX: Use object IDs and cache versions instead of expensive string conversion
            hash_components = []
            
            # Hash based on cache versions (much faster than string conversion)
            for section, version in self._cache_version.items():
                hash_components.append(f"{section}:{version}")
            
            # Add layout structure indicators without expensive string conversion
            hash_components.extend([
                f"mode:{self._current_mode.value}",
                f"page:{self._current_page.value}",
                f"focus:{self._current_focus.value}",
                f"sidebar:{not self._sidebar_collapsed}",
                f"palette:{self._command_palette_active}"
            ])
            
            # CRITICAL FIX: Include object IDs for actual content changes
            try:
                if hasattr(layout, '__getitem__'):
                    # Use object ID instead of string conversion for performance
                    for section in ["header", "content", "footer", "sidebar"]:
                        try:
                            obj = layout[section]
                            hash_components.append(f"{section}_id:{id(obj)}")
                        except (KeyError, AttributeError):
                            pass
            except Exception:
                pass
            
            # Generate lightweight hash
            combined = "|".join(hash_components)
            return str(hash(combined))
            
        except Exception:
            # Fallback to timestamp-based hash to prevent identical frames
            import time
            return str(int(time.time() * 1000))
    
    def _is_user_typing(self) -> bool:
        """Check if user is currently typing (within typing timeout)."""
        import time
        return time.time() - self._last_keypress_time < self._typing_timeout
    
    def _status_has_meaningful_change(self, old_status: Dict, new_status: Dict) -> bool:
        """Check if status update contains meaningful changes worth refreshing for."""
        # Compare key fields that are visible in the header
        key_fields = ['active_agents', 'total_cost', 'current_model', 'connection_status']
        
        for field in key_fields:
            old_val = old_status.get(field)
            new_val = new_status.get(field)
            
            # For numeric values, only consider significant changes
            if field == 'total_cost':
                if old_val is None or new_val is None or abs(float(new_val or 0) - float(old_val or 0)) > 0.01:
                    return True
            elif old_val != new_val:
                return True
                
        return False
        
    # ------------------------------------------------------------------- #
    # Error handling and rate limiting
    # ------------------------------------------------------------------- #
    def _render_with_fallback(self, section_name: str, render_func, fallback_content: str = "Content unavailable") -> RenderableType:
        """Render with error handling and fallback content."""
        import time
        
        # Check if section has exceeded error threshold
        error_count = self._error_counts.get(section_name, 0)
        if error_count >= self._max_errors_per_section:
            # Show fallback content instead of trying to render
            return Panel(
                Text(f"{fallback_content}\n\n[dim]Render errors: {error_count}[/dim]", 
                     style="dim", justify="center"),
                title=f"{section_name.title()} (Fallback)",
                border_style="red",
            )
        
        try:
            return render_func()
        except Exception as e:
            # Increment error count
            self._error_counts[section_name] = error_count + 1
            
            # Rate limit error logging (max once per 5 seconds per section)
            now = time.time()
            last_log = self._last_error_log_time.get(section_name, 0)
            if now - last_log > 5.0:
                # Log error details to TUI logs panel instead of console
                self._enqueue_tui_log("ERROR", f"Render error in {section_name}: {e}")
                self._last_error_log_time[section_name] = now
            
            # Return fallback panel
            return Panel(
                Text(f"{fallback_content}\n\n[dim]Error: {str(e)[:50]}{'...' if len(str(e)) > 50 else ''}[/dim]",
                     style="dim", justify="center"),
                title=f"{section_name.title()} (Error)",
                border_style="yellow",
            )
    
    def _reset_error_count(self, section_name: str) -> None:
        """Reset error count for a section (called on successful render)."""
        if section_name in self._error_counts:
            del self._error_counts[section_name]
            
    def _log_suppressed_error(self, section_name: str, error: Exception) -> None:
        """Log an error once and suppress subsequent identical errors."""
        error_key = f"{section_name}:{type(error).__name__}:{str(error)}"
        if error_key not in self._error_suppression:
            self._error_suppression[error_key] = 0
            self._enqueue_tui_log("ERROR", f"First occurrence of error in {section_name}: {error}")
        self._error_suppression[error_key] += 1

    def _enqueue_tui_log(self, level: str, message: str) -> None:
        """Append a log entry to the TUI logs panel without writing to console."""
        try:
            if not hasattr(self, 'log_handler') or self.log_handler is None:
                return
            entry = LogEntry(
                timestamp=datetime.now(),
                level=level.upper(),
                logger_name="agentsmcp.tui",
                message=message,
            )
            self.log_handler.log_entries.append(entry)
            # If logs page is visible, refresh content; otherwise leave for later
            if getattr(self, '_current_page', None) == SidebarPage.LOGS:
                self.mark_dirty("content")
        except Exception:
            # Never fail due to logging issues
            pass
        
    def _connect_input_events(self) -> None:
        """Connect RealTimeInputField events to the TUI's refresh and input system."""
        if not self.realtime_input:
            return
            
        # Connect both submit and change events for real-time feedback
        async def on_input_submit(text: str) -> None:
            await self._input_queue.put(text)
            self.realtime_input.clear_input()  # Clear after submission
            self.mark_dirty("footer")  # Refresh input area on submit
            
            # Show brief "Sent!" feedback to user
            self._print_system_message("✅ Message sent!")
            
        async def on_input_change(text: str) -> None:
            # CRITICAL FIX: Force immediate visual updates for typing feedback 
            self._force_immediate_footer_refresh()
            self._last_input_text = text
            
        self.realtime_input.on_submit(on_input_submit)
        self.realtime_input.on_change(on_input_change)
        
    def _ensure_input_focus(self) -> None:
        """FIXED: Ensure input field is properly focused and ready for immediate input."""
        if not self.realtime_input:
            return
            
        try:
            # Ensure the input field is in a ready state
            self.realtime_input._ensure_initialized()
            # Force cursor to be visible and ready
            self.realtime_input._cursor_visible = True
            # Mark footer as needing refresh to show focused state
            self.mark_dirty("footer")
        except Exception:
            # Graceful failure if input setup fails
            pass

    def _render_welcome(self) -> None:
        """Display a brief welcome message."""
        if self._console:
            # Customize welcome message based on input mode
            if self._keyboard_input and self._keyboard_input.is_interactive:
                input_info = "Type directly - characters appear as you type. Press Enter to send."
            else:
                input_info = "Type your message and press Enter to send.\n[dim yellow]Note: In line-based mode, text appears when you press Enter.[/dim yellow]"
            
            welcome_panel = Panel(
                Text(f"Welcome to AgentsMCP Modern TUI!\n\n{input_info}\n\nCommands: '/help' for help, '/quit' to exit.", 
                     justify="center"),
                title="🚀 AgentsMCP",
                border_style="blue"
            )
            self._console.print(welcome_panel)
            self._console.print()  # Empty line

    # ------------------------------------------------------------------- #
    def set_accessibility_option(self, option: str, value: bool) -> None:
        """Configure accessibility options for the TUI."""
        if option in self._accessibility_config:
            self._accessibility_config[option] = value
            
            # Apply immediate changes if needed
            if option == "reduce_motion" and self.realtime_input:
                # Update cursor blinking based on motion preference
                if value:
                    # Disable cursor blinking for motion sensitivity
                    self.realtime_input.stop_cursor_blink()
                    self.realtime_input._cursor_visible = True  # Keep cursor always visible
                else:
                    # Re-enable cursor blinking
                    import asyncio
                    try:
                        asyncio.create_task(self.realtime_input.start_cursor_blink())
                    except RuntimeError:
                        pass  # No event loop running
                        
            # Refresh display to apply color changes
            self.mark_dirty("all")

    # 4️⃣  Public helpers – mode switching & introspection
    # ------------------------------------------------------------------- #
    @property
    def mode(self) -> TUIMode:
        """Current UI mode (read‑only)."""
        return self._current_mode

    async def switch_mode(self, new_mode: TUIMode) -> None:
        """
        Switch to a different UI mode.

        The method updates internal state, re‑creates the layout (if needed) and
        persists the choice via the ``OrchestrationManager``.
        """
        if not isinstance(new_mode, TUIMode):
            raise ValueError(f"Invalid mode: {new_mode!r}")

        if new_mode == self._current_mode:
            return

        self._current_mode = new_mode
        # Re‑build layout because different modes may have different panel
        # requirements.
        self._layout = self._build_layout()
        # Persist immediately – this way a crash after the switch still remembers
        # the last chosen mode.
        try:
            if hasattr(self.orchestration_manager, 'save_user_settings'):
                self.orchestration_manager.save_user_settings({"ui.last_mode": new_mode.value})
        except Exception:
            pass  # Graceful failure

    # ------------------------------------------------------------------- #
    # 5️⃣  Internal utilities
    # ------------------------------------------------------------------- #
    def _load_last_mode(self) -> TUIMode:
        """Load the last mode from ``OrchestrationManager`` or fall back to ZEN."""
        try:
            if hasattr(self.orchestration_manager, 'user_settings'):
                raw = self.orchestration_manager.user_settings.get("ui.last_mode", TUIMode.ZEN.value)
                return TUIMode(raw)
        except Exception:
            pass
        return TUIMode.ZEN
    
    def _load_ui_state(self) -> None:
        """Load hybrid TUI state from persisted settings or use defaults."""
        try:
            if hasattr(self.orchestration_manager, 'user_settings'):
                settings = self.orchestration_manager.user_settings
                
                # Load sidebar state - default to collapsed (Zen mode)
                self._sidebar_collapsed = settings.get("ui.sidebar_collapsed", True)
                
                # Load current page - default to CHAT
                page_value = settings.get("ui.current_page", SidebarPage.CHAT.value)
                try:
                    self._current_page = SidebarPage(page_value)
                except ValueError:
                    self._current_page = SidebarPage.CHAT
                
                # Load focus region - default to INPUT
                focus_value = settings.get("ui.current_focus", FocusRegion.INPUT.value)
                try:
                    self._current_focus = FocusRegion(focus_value)
                except ValueError:
                    self._current_focus = FocusRegion.INPUT
            else:
                # Use defaults if no settings available
                self._sidebar_collapsed = True
                self._current_page = SidebarPage.CHAT
                self._current_focus = FocusRegion.INPUT
        except Exception:
            # Fallback to defaults on any error
            self._sidebar_collapsed = True
            self._current_page = SidebarPage.CHAT
            self._current_focus = FocusRegion.INPUT
    
    def _save_ui_state(self) -> None:
        """Save current hybrid TUI state to persistence."""
        try:
            if hasattr(self.orchestration_manager, 'save_user_settings'):
                ui_state = {
                    "ui.sidebar_collapsed": self._sidebar_collapsed,
                    "ui.current_page": self._current_page.value,
                    "ui.current_focus": self._current_focus.value,
                }
                self.orchestration_manager.save_user_settings(ui_state)
        except Exception:
            pass  # Graceful failure if persistence isn't available

    # ------------------------------------------------------------------- #
    # Layout building – each mode gets its own layout tree.
    # ------------------------------------------------------------------- #
    def _build_layout(self) -> Layout:
        """Construct a fresh hybrid layout with optional collapsible sidebar."""
        if not Layout:
            raise RuntimeError("Rich is not available – cannot build layout.")

        base = Layout(name="root")
        
        # Always use hybrid layout structure
        base.split(
            Layout(name="header", size=3),  # Status bar with connection info
            Layout(name="main_area", ratio=1),  # Flexible main area
            Layout(name="footer", size=3),  # Hotkeys and hints
        )
        
        # Configure main area with optional sidebar
        main_area = base["main_area"]
        
        if self._sidebar_collapsed:
            # Clean zen mode - just the main content pane
            # CRITICAL FIX: Don't create a separate content layout in collapsed mode
            # The main_area itself serves as the content area when collapsed
            pass  # main_area is already the content area
        else:
            # Split main area: sidebar | content
            main_area.split_row(
                Layout(name="sidebar", size=25),  # Fixed-width sidebar
                Layout(name="content", ratio=1),  # Flexible content area
            )
            
        return base

    def _normalize_emoji_spacing(self, text: str) -> str:
        """Normalize emoji spacing for consistent alignment."""
        import re
        
        # Common wide emojis that may cause alignment issues
        emoji_fixes = {
            '⌨️': '⌨ ',   # Keyboard - add space after
            '📝': '📝 ',  # Memo - add space after
            '⚙️': '⚙ ',   # Gear - add space after
            '🤖': '🤖 ',  # Robot - add space after
            '💰': '💰 ',  # Money bag - add space after
        }
        
        # Apply fixes
        for emoji, replacement in emoji_fixes.items():
            if emoji in text and not text.endswith(' '):
                text = text.replace(emoji, replacement)
        
        return text
        
    def _get_accessible_color(self, color: str, context: str = "default") -> str:
        """Get accessibility-friendly color based on current settings."""
        if not self._accessibility_config["high_contrast"]:
            return color
            
        # High contrast color mapping
        high_contrast_colors = {
            "green": "bright_green",
            "red": "bright_red", 
            "yellow": "bright_yellow",
            "blue": "bright_blue",
            "cyan": "bright_cyan",
            "dim": "white",  # Replace dim with white for visibility
        }
        
        return high_contrast_colors.get(color, color)
        
    def _get_motion_style(self, has_animation: bool = False) -> dict:
        """Get style options considering motion sensitivity."""
        style_options = {}
        
        if self._accessibility_config["reduce_motion"] and has_animation:
            # Disable animations for motion-sensitive users
            style_options["no_animation"] = True
            
        return style_options

    # ------------------------------------------------------------------- #
    # Hybrid TUI rendering - unified approach with sidebar support
    # ------------------------------------------------------------------- #
    def _render(self) -> RenderableType:
        """Return a complete renderable representing the hybrid TUI."""
        if not self._layout:
            return Text("[red]Layout not initialised[/red]")

        # CRITICAL FIX: Set render guard to prevent feedback loops
        self._currently_rendering = True
        try:
            # Render header with status information - wrapped with error handling
            try:
                self._render_hybrid_header()
            except Exception as e:
                self._log_suppressed_error("render_header", e)
            
            # Render sidebar if not collapsed - wrapped with error handling
            if not self._sidebar_collapsed:
                try:
                    self._render_sidebar()
                except Exception as e:
                    self._log_suppressed_error("render_sidebar", e)
            
            # Render main content area based on current page - wrapped with error handling
            try:
                self._render_main_content()
            except Exception as e:
                self._log_suppressed_error("render_content", e)
            
            # Render footer with hotkeys and hints - wrapped with error handling
            try:
                self._render_hybrid_footer()
            except Exception as e:
                self._log_suppressed_error("render_footer", e)
            
        except Exception as exc:  # pragma: no cover – defensive fallback
            # If something goes wrong at the top level, show a minimal error panel
            # But use error suppression to prevent infinite loops
            self._log_suppressed_error("render_toplevel", exc)
            try:
                error_panel = Panel(
                    Text.from_markup("[bold red]UI Error:[/bold red] Interface temporarily unavailable"),
                    title="Error",
                    border_style="red",
                )
                content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
                content_layout.update(error_panel)
            except Exception:
                # Even the error panel failed - just pass silently to prevent infinite loops
                pass
        finally:
            # CRITICAL FIX: Clear render guard and cycle-specific flags after rendering is complete
            self._currently_rendering = False
            self._header_rendered_this_cycle = False
            
        return self._layout
        
    # ----------------- Hybrid TUI Component Renderers ----------------- #
    def _render_hybrid_header(self) -> None:
        """Render status header with real-time connection info and current page."""
        # CRITICAL FIX: Prevent duplicate header rendering by checking if already rendered this cycle
        if getattr(self, '_header_rendered_this_cycle', False):
            return
        self._header_rendered_this_cycle = True
        
        def generate_header():
            import time
            from datetime import datetime
            
            # Left side - app status
            left_items = []
            
            # Connection status with real-time indicator
            try:
                # Check if orchestration manager is responsive
                settings_attr = getattr(self.orchestration_manager, 'user_settings', None)
                settings = settings_attr() if callable(settings_attr) else settings_attr
                if isinstance(settings, dict):
                    color = self._get_accessible_color("green")
                    left_items.append(f"[{color}]●[/{color}] Ready")
                else:
                    color = self._get_accessible_color("yellow")
                    left_items.append(f"[{color}]●[/{color}] Limited")
            except Exception:
                color = self._get_accessible_color("red")
                left_items.append(f"[{color}]●[/{color}] Disconnected")
            
            # CRITICAL FIX: Add input mode health indicator
            input_mode = getattr(self, '_input_mode', 'unknown')
            if input_mode == "per_key":
                color = self._get_accessible_color("green")
                left_items.append(f"[{color}]⌨️  Interactive[/{color}]")
            elif input_mode == "line":
                color = self._get_accessible_color("yellow")
                left_items.append(f"[{color}]📝 Line Mode[/{color}]")
            else:
                color = self._get_accessible_color("dim")
                left_items.append(f"[{color}]❓ Unknown[/{color}]")
            
            # Current page/mode indicator
            if not self._sidebar_collapsed:
                page_name = self._current_page.value.title()
                left_items.append(f"[cyan]{page_name} Page[/cyan]")
            else:
                left_items.append("[blue]Zen Mode[/blue]")
                
            # Real-time status from SSE (if available)
            if self._last_status_update:
                status_data = self._last_status_update
                
                # Show active agents count
                agent_count = status_data.get('active_agents', 0)
                if agent_count > 0:
                    left_items.append(f"[yellow]⚙️ {agent_count} agents[/yellow]")
                
                # Show cost information
                total_cost = status_data.get('total_cost', 0)
                if total_cost > 0:
                    left_items.append(f"[green]💰 ${total_cost:.2f}[/green]")
                    
                # Show model information
                current_model = status_data.get('current_model', '')
                if current_model:
                    left_items.append(f"[cyan]🤖 {current_model}[/cyan]")
            else:
                # Fallback to static job count check
                try:
                    job_count = 0  # Replace with actual job counting
                    if job_count > 0:
                        left_items.append(f"[yellow]⚙️ {job_count} jobs[/yellow]")
                except Exception:
                    pass
                
            # Right side - system info
            right_items = []
            
            # CRITICAL FIX: Disable real-time clock to prevent scrollback flooding
            # Time display causes constant re-renders - disabled for now
            # right_items.append("[dim]Time display disabled[/dim]")
            
            # Web UI access method
            right_items.append("[dim]Web: agentsmcp dashboard --port 8000[/dim]")
            # Debug status and last key indicator (if enabled)
            if getattr(self, '_debug_keys', False):
                # Always show current debug mode so it's visible even in line-based input
                if getattr(self, '_input_mode', 'line') == 'per_key':
                    mode_label = f"keys/{getattr(self, '_input_engine', 'native')}"
                else:
                    mode_label = 'line'
                right_items.append(f"[magenta]Debug:[/] {mode_label}")
                # Show last key when available (per-key mode)
                last = getattr(self, '_debug_last_key', '')
                if last:
                    right_items.append(f"[magenta]Key:[/] {last!r}")
            
            # Memory usage (if available) - CRITICAL FIX: Cache memory info to reduce updates
            # Only update memory display every 30 seconds to prevent constant re-renders
            try:
                import psutil
                import time
                mem_cache_key = f"mem_{int(time.time()) // 30 * 30}"  # 30-second buckets
                if not hasattr(self, '_cached_memory') or self._cached_memory[0] != mem_cache_key:
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 80:
                        mem_display = f"[red]RAM: {memory_percent:.0f}%[/red]"
                    elif memory_percent > 60:
                        mem_display = f"[yellow]RAM: {memory_percent:.0f}%[/yellow]"
                    else:
                        mem_display = f"[green]RAM: {memory_percent:.0f}%[/green]"
                    self._cached_memory = (mem_cache_key, mem_display)
                right_items.append(self._cached_memory[1])
            except ImportError:
                # psutil not available, skip memory monitoring
                pass
            except Exception:
                # Error getting memory info, skip
                pass
            
            # CRITICAL FIX: Normalize emoji spacing for better alignment
            left_text = " • ".join(self._normalize_emoji_spacing(item) for item in left_items)
            right_text = " • ".join(self._normalize_emoji_spacing(item) for item in right_items)
            
            # Create two-column layout using Rich Columns for proper alignment
            left_content = Text.from_markup(left_text)
            right_content = Align.right(Text.from_markup(right_text))
            
            # Use Columns to handle alignment automatically
            if Columns:
                header_content = Columns([left_content, right_content], expand=True)
            else:
                # Fallback to simple concatenation if Columns not available
                header_content = Text()
                header_content.append(left_text)
                header_content.append("   ")  # Simple spacing
                header_content.append(right_text)
            
            return Panel(
                header_content,
                height=3,
                style="dim",
                border_style="blue",
                title="🔄 Status Lane" if not self._sidebar_collapsed else "💬 AgentsMCP",
            )
        
        # Use error handling for header rendering
        header = self._render_with_fallback("header", generate_header, "Status information unavailable")
        try:
            self._update_section_if_changed("header", header)
            self._reset_error_count("header")  # Reset error count on success
        except Exception as e:
            self._log_suppressed_error("header", e)
        
    def _render_sidebar(self) -> None:
        """Render the collapsible sidebar with pages."""
        if self._sidebar_collapsed:
            return
            
        def generate_sidebar():
            # Build sidebar menu
            menu_items = []
            for page in SidebarPage:
                page_name = page.value.title()
                if page == self._current_page:
                    # Highlight current page
                    menu_items.append(f"[reverse bold] {page_name} [/reverse bold]")
                else:
                    menu_items.append(f"  {page_name}")
            
            menu_text = "\n".join(menu_items)
            
            # Add keyboard hints at bottom
            hints = [
                "",
                "[dim]Hotkeys:[/dim]",
                "[dim]↑/↓: Navigate[/dim]",
                "[dim]Enter: Select[/dim]",
                "[dim]Ctrl+B: Toggle[/dim]"
            ]
            
            full_text = menu_text + "\n" + "\n".join(hints)
            
            return Panel(
                Text.from_markup(full_text),
                title="Pages",
                border_style="cyan",
                width=25,
            )
        
        # Use error handling for sidebar rendering
        sidebar = self._render_with_fallback("sidebar", generate_sidebar, "Sidebar unavailable")
        try:
            self._layout["sidebar"].update(sidebar)
            self._reset_error_count("sidebar")  # Reset error count on success
        except KeyError:
            # Sidebar section doesn't exist (shouldn't happen since we check _sidebar_collapsed above)
            pass
        
    def _render_main_content(self) -> None:
        """Render main content based on current page."""
        # Render welcome inside Live once to avoid pre-Live scrollback flooding
        if getattr(self, '_show_welcome', False):
            try:
                welcome_panel = Panel(
                    Text(
                        "Welcome to AgentsMCP Modern TUI!\n\n"
                        "Type directly - characters appear as you type. Press Enter to send.\n\n"
                        "Commands: '/help' for help, '/quit' to exit.",
                        justify="center",
                    ),
                    title="🚀 AgentsMCP",
                    border_style="blue",
                )
                content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
                content_layout.update(welcome_panel)
                self._show_welcome = False
                return
            except Exception:
                self._show_welcome = False
                # Fall through to normal rendering
        content_renderer = {
            SidebarPage.CHAT: self._render_chat_content,
            SidebarPage.JOBS: self._render_jobs_content,
            SidebarPage.AGENTS: self._render_agents_content,
            SidebarPage.MODELS: self._render_models_content,
            SidebarPage.PROVIDERS: self._render_providers_content,
            SidebarPage.COSTS: self._render_costs_content,
            SidebarPage.MCP: self._render_mcp_content,
            SidebarPage.DISCOVERY: self._render_discovery_content,
            SidebarPage.LOGS: self._render_logs_content,
            SidebarPage.SETTINGS: self._render_settings_content,
        }.get(self._current_page, self._render_chat_content)
        
        content_renderer()
        
    def _render_hybrid_footer(self) -> None:
        """Render footer with context-aware hotkeys and real-time input."""
        def generate_footer():
            # CRITICAL FIX: Always use realtime input when available, regardless of mode
            if self.realtime_input is not None:
                try:
                    # CRITICAL FIX: Always render realtime input to show typed characters
                    cur = self.realtime_input.get_current_input() if hasattr(self.realtime_input, 'get_current_input') else ''
                    rendered = self.realtime_input.render()
                    return rendered
                except Exception as e:
                    # Log the error for debugging but fall back to static footer
                    import traceback
                    self._log_suppressed_error("realtime_input_render", e)
                    # Fall back to static footer if realtime rendering fails
                    pass
            
            # Command palette mode
            if self._command_palette_active:
                return Panel(
                    Align.left(Text("Command Mode: Type '/' commands | Escape: Cancel", style="bold yellow")),
                    height=3,
                    border_style="yellow",
                    title="⌨️  Command Palette Active",
                )
            
            # Normal mode - build hotkey hints based on context
            hotkeys = []
            
            # Core navigation
            if self._sidebar_collapsed:
                hotkeys.append("[bold blue]Ctrl+B[/bold blue]: Show sidebar")
            else:
                hotkeys.append("[bold blue]Ctrl+B[/bold blue]: Hide sidebar")
                hotkeys.append("[bold green]Tab[/bold green]: Focus")
                
            # Current focus hints
            if self._current_focus == FocusRegion.MAIN:
                hotkeys.append("[cyan]PgUp/PgDn[/cyan]: Scroll history")
            elif self._current_focus == FocusRegion.SIDEBAR:
                hotkeys.append("[green]↑/↓[/green]: Navigate pages")
            
            # Always available
            hotkeys.append("[yellow]/[/yellow]: Commands")
            
            # Page-specific hints
            if not self._sidebar_collapsed:
                if self._current_page == SidebarPage.CHAT:
                    hotkeys.append("📝: Chat focused")
                elif self._current_page == SidebarPage.JOBS:
                    hotkeys.append("⚙️: Jobs monitoring")
                elif self._current_page == SidebarPage.AGENTS:
                    hotkeys.append("🤖: Agent management")
                elif self._current_page == SidebarPage.MODELS:
                    hotkeys.append("🧠: Model config")
                elif self._current_page == SidebarPage.LOGS:
                    hotkeys.append("📋: Activity logs")
            
            # Web UI hint
            hotkeys.append("[dim]Web: /dashboard[/dim]")
            
            footer_text = " • ".join(hotkeys)
            
            # Focus indicator in title
            focus_title = ""
            if not self._sidebar_collapsed:
                if self._current_focus == FocusRegion.INPUT:
                    focus_title = "⌨️  Input Focus"
                elif self._current_focus == FocusRegion.MAIN:
                    focus_title = "📖  Main Focus"  
                elif self._current_focus == FocusRegion.SIDEBAR:
                    focus_title = "📂  Sidebar Focus"
            else:
                focus_title = "💬  Zen Mode"
            
            return Panel(
                Align.left(Text.from_markup(footer_text)),
                height=3,
                style="dim",
                border_style="blue" if self._current_focus == FocusRegion.INPUT else "dim",
                title=focus_title,
            )
        
        # Use error handling for footer rendering; always update to ensure echo
        footer = self._render_with_fallback("footer", generate_footer, "Input hints unavailable")
        try:
            # Always update footer; typing requires immediate visual feedback
            self._layout["footer"].update(footer)
            self._reset_error_count("footer")  # Reset error count on success
        except Exception as e:
            self._log_suppressed_error("footer", e)
        
    # ----------------- Page Content Renderers ----------------- #
    def _render_chat_content(self) -> None:
        """Render chat interface with scrollable history."""
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        
        # FIXED: Only show chat history, input is handled by footer
        # Just update content with scrollable chat history
        self._render_scrollable_chat_history(content_layout)
            
    def _render_scrollable_chat_history(self, content_layout=None) -> None:
        """Render chat history with scroll support."""
        if content_layout is None:
            content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
            
        def render_enhanced_chat():
            if self.chat_history is not None:
                # Get terminal height and calculate available space
                terminal_height = self._console.size.height if self._console else 24
                available_height = max(terminal_height - 6, 5)  # Reserve space for header/footer
                
                # Render with scroll offset (note: ChatHistoryDisplay doesn't support scroll_offset yet)
                return self.chat_history.render_history(height=available_height)
            else:
                raise Exception("Enhanced chat history not available")
        
        try:
            chat_display = self._render_with_fallback("chat_history", render_enhanced_chat, "Enhanced chat unavailable")
            # FIXED: Update the main content layout directly, not a sub-layout
            content_layout.update(chat_display)
            self._reset_error_count("chat_history")  # Reset error count on success
        except Exception:
            self._render_legacy_scrollable_chat(content_layout)
            
    def _render_legacy_scrollable_chat(self, content_layout=None) -> None:
        """Legacy scrollable chat history rendering."""
        if content_layout is None:
            content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
            
        def generate_chat():
            try:
                history = self._get_chat_history()
            except Exception:
                history = []
                
            if not history:
                return Panel(
                    Text("[dim]Start a conversation... (or press Ctrl+B to explore other features)[/dim]", 
                         justify="center"),
                    border_style="blue",
                    title="Chat History",
                )
            
            # Apply scroll offset
            visible_messages = history[self._chat_scroll_offset:self._chat_scroll_offset + 20]
            
            lines = []
            for entry in visible_messages:
                role = entry.get('role', 'unknown')
                content = entry.get('content', '')
                if role == 'user':
                    lines.append(f"[cyan][USER][/cyan] {content}")
                elif role == 'assistant':
                    lines.append(f"[green][ASSISTANT][/green] {content}")
                elif role == 'system':
                    lines.append(f"[yellow][SYSTEM][/yellow] {content}")
                else:
                    lines.append(f"[{role.upper()}] {content}")
                    
            # Add scroll indicators
            scroll_info = ""
            if self._chat_scroll_offset > 0:
                scroll_info += f" ↑ {self._chat_scroll_offset} messages above"
            if len(history) > self._chat_scroll_offset + 20:
                remaining = len(history) - self._chat_scroll_offset - 20
                scroll_info += f" ↓ {remaining} messages below"
                
            if scroll_info:
                lines.append(f"[dim]{scroll_info}[/dim]")
                
            text_content = '\n'.join(lines)
            return Panel(
                Text.from_markup(text_content),
                border_style="blue",
                title="Chat History (PgUp/PgDn to scroll)",
                title_align="left",
            )
            
        # Use error handling for legacy chat rendering
        chat_panel = self._render_with_fallback("legacy_chat", generate_chat, "Chat history unavailable")
        try:
            # FIXED: Update the main content layout directly
            content_layout.update(chat_panel)
            self._reset_error_count("legacy_chat")  # Reset error count on success
        except Exception as e:
            self._log_suppressed_error("legacy_chat", e)
        
        
    def _render_jobs_content(self) -> None:
        """Render jobs monitoring page."""
        def generate_jobs():
            return Panel(
                Text("[dim]Jobs monitoring coming soon...\nHere you'll see:\n• Running tasks\n• Job queues\n• Live streaming output[/dim]"),
                title="Jobs & Tasks",
                border_style="yellow",
            )
            
        jobs_panel = self._get_cached_panel("jobs_content", generate_jobs)
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        content_layout.update(jobs_panel)
        
    def _render_agents_content(self) -> None:
        """Render agents management page."""
        def generate_agents():
            return Panel(
                Text("[dim]Agent management coming soon...\nHere you'll see:\n• Available agents\n• Agent performance\n• Model assignments[/dim]"),
                title="Agents",
                border_style="green",
            )
            
        agents_panel = self._get_cached_panel("agents_content", generate_agents)
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        content_layout.update(agents_panel)
        
    def _render_models_content(self) -> None:
        """Render models configuration page."""
        def generate_models():
            return Panel(
                Text("[dim]Model configuration coming soon...\nHere you'll see:\n• Available models\n• Performance metrics\n• Cost tracking[/dim]"),
                title="Models",
                border_style="magenta",
            )
            
        models_panel = self._get_cached_panel("models_content", generate_models)
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        content_layout.update(models_panel)
        
    def _render_providers_content(self) -> None:
        """Render providers configuration page."""
        def generate_providers():
            return Panel(
                Text("[dim]Provider settings coming soon...\nHere you'll see:\n• API configurations\n• Rate limits\n• Health status[/dim]"),
                title="Providers",
                border_style="red",
            )
            
        providers_panel = self._get_cached_panel("providers_content", generate_providers)
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        content_layout.update(providers_panel)
        
    def _render_costs_content(self) -> None:
        """Render cost monitoring page."""
        def generate_costs():
            return Panel(
                Text("[dim]Cost monitoring coming soon...\nHere you'll see:\n• Budget tracking\n• Usage analytics\n• Spend optimization[/dim]"),
                title="Costs & Budget",
                border_style="cyan",
            )
            
        costs_panel = self._get_cached_panel("costs_content", generate_costs)
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        content_layout.update(costs_panel)
        
    def _render_mcp_content(self) -> None:
        """Render MCP servers management page."""
        def generate_mcp():
            return Panel(
                Text("[dim]MCP management coming soon...\nHere you'll see:\n• Connected servers\n• Tool availability\n• Performance metrics[/dim]"),
                title="MCP Servers",
                border_style="blue",
            )
            
        mcp_panel = self._get_cached_panel("mcp_content", generate_mcp)
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        content_layout.update(mcp_panel)
        
    def _render_discovery_content(self) -> None:
        """Render service discovery page."""
        def generate_discovery():
            return Panel(
                Text("[dim]Service discovery coming soon...\nHere you'll see:\n• Available services\n• Network topology\n• Health checks[/dim]"),
                title="Discovery",
                border_style="green",
            )
            
        discovery_panel = self._get_cached_panel("discovery_content", generate_discovery)
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        content_layout.update(discovery_panel)
        
    def _render_logs_content(self) -> None:
        """Render logs monitoring page."""
        def generate_logs():
            log_entries = list(self.log_handler.log_entries)
            
            if not log_entries:
                return Panel(
                    Text("[dim]No log entries yet...\n\nBackground activity will appear here when it happens:\n• LLM processing\n• Tool execution\n• API calls\n• System events[/dim]",
                         justify="center"),
                    title="📋 Activity Logs",
                    border_style="cyan",
                )
            
            # Get recent entries - show more in logs view (last 100)
            recent_entries = log_entries[-100:] if len(log_entries) > 100 else log_entries
            
            # Reverse for newest first display
            recent_entries = list(reversed(recent_entries))
            
            # Format log entries for display
            log_lines = []
            for entry in recent_entries:
                log_lines.append(entry.format_for_display())
            
            # Add summary info at the top
            level_counts = {}
            for entry in log_entries:
                level_counts[entry.level] = level_counts.get(entry.level, 0) + 1
                
            level_summary = " | ".join([f"{level}: {count}" for level, count in sorted(level_counts.items())])
            
            summary_info = [
                f"[bold]Live Activity Monitor[/bold] [dim]({len(recent_entries)} newest shown)[/dim]",
                f"[dim]Total: {len(log_entries)} | Levels: {level_summary}[/dim]",
                "[dim]Sources: AgentsMCP components, LLM clients, orchestration[/dim]",
                ""  # Empty line for spacing
            ]
            
            full_content = "\n".join(summary_info + log_lines)
            
            # Adjust height based on terminal size for better scrolling
            terminal_height = self._console.size.height if self._console else 24
            content_height = terminal_height - 8  # Account for header/footer/panel borders
            
            return Panel(
                Text.from_markup(full_content),
                title="📋 Activity Logs (Live)",
                subtitle="[dim]Newest first • Auto-refreshing[/dim]",
                border_style="cyan",
                title_align="left",
                height=content_height
            )
            
        # Don't cache logs content as it updates frequently
        logs_panel = generate_logs()
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        content_layout.update(logs_panel)
        
    def _render_settings_content(self) -> None:
        """Render settings page."""
        def generate_settings():
            return Panel(
                Text("[dim]Settings coming soon...\nHere you'll configure:\n• UI preferences\n• Default models\n• Keyboard shortcuts[/dim]"),
                title="Settings",
                border_style="white",
            )
            
        settings_panel = self._get_cached_panel("settings_content", generate_settings)
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        content_layout.update(settings_panel)

    # ----------------- Mode specific renderers ----------------- #
    def _render_zen(self) -> None:
        """Populate the layout for Zen mode (minimal chat view)."""
        # Header – app name & mode indicator (cached)
        def generate_header():
            return Panel(
                Align.center(
                    Text(f"AgentsMCP – Zen Chat", style="bold blue"),
                    vertical="middle",
                ),
                style="dim",
                border_style="blue",
            )
        
        header = self._get_cached_panel("header", generate_header)
        self._update_section_if_changed("header", header)

        # Body – chat history using enhanced display when available
        if self.chat_history is not None:
            try:
                # Calculate available height for the chat history
                terminal_height = self._console.size.height if self._console else 24
                available_height = max(terminal_height - 6, 5)  # Reserve space for header/footer
                
                chat_display = self.chat_history.render_history(height=available_height)
                content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
                content_layout.update(chat_display)
            except Exception:
                # Fall back to legacy chat history display
                self._render_legacy_chat_history()
        else:
            # Legacy chat history display
            self._render_legacy_chat_history()

        # Footer – real-time input field or legacy static prompt
        if self.realtime_input is not None:
            # Use real-time input field with live typing display
            try:
                self._debug_log("_render_footer: calling realtime_input.render()")
                footer = self.realtime_input.render()
                self._layout["footer"].update(footer)
                self._debug_log("_render_footer: updated footer layout with realtime input")
            except Exception as e:
                self._debug_log(f"_render_footer: realtime_input.render failed: {e}")
                # Fall back to static footer if rendering fails
                self._render_static_footer()
        else:
            # Legacy static footer
            self._render_static_footer()
            
    def _render_static_footer(self) -> None:
        """Render the static legacy footer."""
        footer = Panel(
            Align.left(
                Text(
                    "[bold]Type message[/bold] (or /mode to change UI, /quit to exit) → ",
                    style="yellow",
                ),
                vertical="middle",
            ),
            style="dim",
        )
        self._layout["footer"].update(footer)
        
    def _render_legacy_chat_history(self) -> None:
        """Legacy chat history rendering for fallback with caching."""
        def generate_body():
            try:
                history = self._get_chat_history()
            except Exception:
                history = []
                
            if not history:
                body_text = Text("No conversation history yet", style="dim")
            else:
                body_text = Text()
                for entry in history[-10:]:  # Show last 10 messages
                    role = entry.get("role", "system")
                    content = entry.get("content", "")
                    if content.strip():  # Content validation - skip empty messages
                        role_style = "green" if role == "assistant" else "cyan"
                        body_text.append(f"{role.title()}: ", style=f"bold {role_style}")
                        body_text.append(f"{content}\n")
                        
            return Panel(body_text, title="Chat History", border_style="blue")
        
        content_panel = self._get_cached_panel("content", generate_body)
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        content_layout.update(content_panel)

    def _render_dashboard(self) -> None:
        """Dashboard mode – placeholder implementation."""
        header = Panel(
            Align.center(Text("AgentsMCP – Dashboard", style="bold green")),
            style="dim",
        )
        self._layout["header"].update(header)

        # Fake metrics – real implementation will pull data from orchestration_manager
        left = Panel(
            Text.from_markup(
                "[bold]Metrics[/bold]\n"
                "- Sessions: 12\n"
                "- Errors: 0\n"
                "- Uptime: 3h 27m"
            ),
            border_style="green",
        )
        right = Panel(
            Text.from_markup(
                "[bold]System[/bold]\n"
                "- CPU: 23%\n"
                "- RAM: 1.2 GiB / 8 GiB\n"
                "- Disk: 25 GiB free"
            ),
            border_style="green",
        )
        self._layout["left"].update(left)
        self._layout["right"].update(right)

        footer = Panel(
            Align.center(Text("Press /mode zen/dashboard/command to switch.")),
            style="dim",
        )
        self._layout["footer"].update(footer)

    def _render_command_center(self) -> None:
        """Command‑center mode – placeholder skeleton."""
        header = Panel(
            Align.center(Text("AgentsMCP – Command Center", style="bold magenta")),
            style="dim",
        )
        self._layout["header"].update(header)

        body = Panel(
            Text("[dim]Full technical UI coming soon…[/dim]"),
            border_style="magenta",
        )
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        content_layout.update(body)

        footer = Panel(
            Align.center(Text("Use /mode to change UI.")),
            style="dim",
        )
        self._layout["footer"].update(footer)
        
    # ------------------------------------------------------------------- #
    # Change-tracking render methods for optimized updates
    # ------------------------------------------------------------------- #
    def _render_zen_with_tracking(self) -> list:
        """Render Zen mode with change tracking. Returns list of changed sections."""
        changed = []
        
        # Header – app name & mode indicator (cached)
        def generate_header():
            return Panel(
                Align.center(
                    Text(f"AgentsMCP – Zen Chat", style="bold blue"),
                    vertical="middle",
                ),
                style="dim",
                border_style="blue",
            )
        
        header = self._get_cached_panel("header", generate_header)
        if self._update_section_if_changed("header", header):
            changed.append("header")
            
        # Body – chat history using enhanced display when available
        if self.chat_history is not None:
            try:
                # Calculate available height for the chat history
                terminal_height = self._console.size.height if self._console else 24
                available_height = max(terminal_height - 6, 5)  # Reserve space for header/footer
                
                chat_display = self.chat_history.render_history(height=available_height)
                if self._update_section_if_changed("content", chat_display):
                    changed.append("content")
            except Exception:
                # Fall back to legacy chat history display
                if self._render_legacy_chat_history_with_tracking():
                    changed.append("content")
        else:
            # Legacy chat history display
            if self._render_legacy_chat_history_with_tracking():
                changed.append("content")
                
        # Footer – real-time input field or legacy static prompt
        if self.realtime_input is not None:
            # Use real-time input field with live typing display
            try:
                self._debug_log("_render_with_change_tracking: calling realtime_input.render()")
                footer = self.realtime_input.render()
                if self._update_section_if_changed("footer", footer):
                    changed.append("footer")
                    self._debug_log("_render_with_change_tracking: footer changed=True")
                else:
                    self._debug_log("_render_with_change_tracking: footer changed=False")
            except Exception as e:
                self._debug_log(f"_render_with_change_tracking: realtime_input.render failed: {e}")
                # Fall back to static footer if rendering fails
                if self._render_static_footer_with_tracking():
                    changed.append("footer")
        else:
            # Legacy static footer
            if self._render_static_footer_with_tracking():
                changed.append("footer")
                
        return changed
        
    def _render_dashboard_with_tracking(self) -> list:
        """Render Dashboard mode with change tracking."""
        return self._render_zen_with_tracking()  # Same as Zen for now
        
    def _render_command_center_with_tracking(self) -> list:
        """Render Command Center mode with change tracking."""
        changed = []
        
        header = Panel(
            Align.center(Text("AgentsMCP – Command Center", style="bold magenta")),
            style="dim",
        )
        if self._update_section_if_changed("header", header):
            changed.append("header")
            
        body = Panel(
            Text("[dim]Full technical UI coming soon…[/dim]"),
            border_style="magenta",
        )
        if self._update_section_if_changed("content", body):
            changed.append("content")
            
        footer = Panel(
            Align.center(Text("Use /mode to change UI.")),
            style="dim",
        )
        if self._update_section_if_changed("footer", footer):
            changed.append("footer")
            
        return changed
        
    def _render_legacy_chat_history_with_tracking(self) -> bool:
        """Legacy chat history rendering with change tracking. Returns True if changed."""
        def generate_body():
            try:
                history = self._get_chat_history()
            except Exception:
                history = []
                
            if not history:
                return Panel(
                    Text("[dim]Start a conversation...[/dim]", justify="center"),
                    border_style="blue",
                    title="Chat History",
                )
                
            lines = []
            for entry in history[-20:]:  # Show last 20 messages
                role = entry.get('role', 'unknown')
                content = entry.get('content', '')
                if role == 'user':
                    lines.append(f"[cyan][USER][/cyan] {content}")
                elif role == 'assistant':
                    lines.append(f"[green][ASSISTANT][/green] {content}")
                elif role == 'system':
                    lines.append(f"[yellow][SYSTEM][/yellow] {content}")
                else:
                    lines.append(f"[{role.upper()}] {content}")
                    
            text_content = '\n'.join(lines)
            return Panel(
                Text.from_markup(text_content),
                border_style="blue",
                title="Chat History",
                title_align="left",
            )
            
        content = self._get_cached_panel("content", generate_body)
        return self._update_section_if_changed("content", content)
        
    def _render_static_footer_with_tracking(self) -> bool:
        """Render static footer with change tracking. Returns True if changed."""
        footer = Panel(
            Align.left(
                Text(
                    "[bold]Type message[/bold] (or /mode to change UI, /quit to exit) → ",
                    style="yellow",
                ),
                vertical="middle",
            ),
            style="dim",
        )
        return self._update_section_if_changed("footer", footer)

    # ------------------------------------------------------------------- #
    # Input handling – runs in a background asyncio task.
    # ------------------------------------------------------------------- #
    async def _read_input(self) -> None:
        """
        Enhanced line-based input reading that coordinates with Rich Live rendering.
        Uses stdin reading without conflicting prompts to prevent scrollback pollution.
        """
        # Native input engine
        self._input_engine = "native"
        loop = asyncio.get_running_loop()
        
        # CRITICAL FIX: Use stdin.readline() instead of input() to avoid prompt conflicts
        import sys
        
        while self._running:
            try:
                # Read from stdin without creating conflicting prompts
                try:
                    # FIXED: Check if stdin has data available before trying to read
                    import select
                    import sys
                    
                    # Use select to check if stdin has data without blocking
                    if sys.stdin.isatty():
                        # For TTY, wait with a timeout to allow graceful shutdown
                        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                        if not ready:
                            # No input available, continue the loop to check _running status
                            continue
                    
                    # FIXED: Use input() with proper blocking to wait for real user input
                    # This blocks until the user actually types something and presses Enter
                    line = await loop.run_in_executor(None, input, "")
                    if not line:  # Empty line (just Enter pressed)
                        continue  # Continue waiting for real input
                    line += '\n'  # Add newline to match readline() behavior
                    
                except EOFError:
                    # EOF reached - user pressed Ctrl+D or similar
                    # In interactive mode, this should only happen with explicit user action
                    self._running = False
                    break
                except KeyboardInterrupt:
                    # User pressed Ctrl+C
                    self._running = False
                    break
                except OSError:
                    # Handle cases where stdin is not available or select fails
                    # Fall back to a simple blocking read with timeout
                    try:
                        line = await asyncio.wait_for(
                            loop.run_in_executor(None, sys.stdin.readline),
                            timeout=1.0
                        )
                        if not line:  # True EOF - only exit if we get multiple consecutive EOFs
                            # Wait a bit and try again to distinguish between temporary EOF and real EOF
                            await asyncio.sleep(0.1)
                            continue
                    except asyncio.TimeoutError:
                        # Timeout waiting for input - continue the loop
                        continue
                    except Exception:
                        # If all input methods fail, wait a bit and continue
                        await asyncio.sleep(0.1)
                        continue
                
                # Clean the input and submit it
                clean_line = line.rstrip('\n\r')
                if clean_line.strip():  # Only process non-empty lines
                    # ENHANCED: Handle special commands in line-based mode
                    if await self._handle_line_based_special_commands(clean_line):
                        continue
                    
                    # CRITICAL FIX: In line-based mode, set the input field content first so it's visible
                    if self.realtime_input is not None:
                        # Set the input to show what was typed
                        self.realtime_input.set_input(clean_line)
                        # Force refresh to show the input
                        self._force_immediate_footer_refresh()
                        # Small delay so user can see what they typed
                        await asyncio.sleep(0.05)  # Reduced delay
                        # Now submit the input
                        await self.realtime_input.handle_submit(clean_line)
                    else:
                        # Direct to input queue for fallback
                        await self._input_queue.put(clean_line)
                        
            except Exception:
                # If stdin is not available or there's a critical error, wait and retry
                await asyncio.sleep(0.1)
                continue
    
    async def _read_keyboard_input(self) -> None:
        """
        Per-key input reading for interactive terminals.
        Enables live typing, keyboard shortcuts, and real-time interaction.
        """
        # Prefer our native raw-terminal reader by default to avoid event-loop blocking.
        # You can explicitly enable prompt_toolkit by setting AGENTS_TUI_ENABLE_PTK=1.
        use_ptk = os.getenv("AGENTS_TUI_ENABLE_PTK", "0") == "1"
        if use_ptk:
            self._input_engine = "ptk"
            try:
                from prompt_toolkit.input import create_input  # type: ignore
                from prompt_toolkit.keys import Keys  # type: ignore
                with create_input() as inp:
                    while self._running:
                        try:
                            for kp in inp.read_keys():
                                if not self._running:
                                    break
                                k = kp.key
                                key_str = None
                                # Handle bracketed paste as a single input chunk
                                if k == Keys.BracketedPaste:
                                    data = getattr(kp, 'data', '') or ''
                                    if data and self.realtime_input is not None:
                                        try:
                                            # Normalize line endings and append to current input
                                            data = data.replace('\r\n','\n').replace('\r','\n')
                                            current = self.realtime_input.get_current_input()
                                            self.realtime_input.set_input(current + data)
                                            # Force immediate input repaint; do not submit
                                            self._force_immediate_footer_refresh()
                                        except Exception:
                                            pass
                                    continue
                                if k == Keys.Up:
                                    key_str = "up"
                                elif k == Keys.Down:
                                    key_str = "down"
                                elif k == Keys.Left:
                                    key_str = "left"
                                elif k == Keys.Right:
                                    key_str = "right"
                                elif k == Keys.PageUp:
                                    key_str = "page_up"
                                elif k == Keys.PageDown:
                                    key_str = "page_down"
                                elif k == Keys.Home:
                                    key_str = "home"
                                elif k == Keys.End:
                                    key_str = "end"
                                elif k == Keys.Tab:
                                    key_str = "tab"
                                elif k == Keys.Backspace:
                                    key_str = "backspace"
                                elif k == Keys.Delete:
                                    key_str = "delete"
                                elif k == Keys.Escape:
                                    key_str = "escape"
                                elif k == Keys.Enter:
                                    key_str = "enter"
                                elif str(k).lower() in ("c-b", "control-b", "ctrl-b"):
                                    key_str = "\x02"  # Ctrl+B
                                else:
                                    data = getattr(kp, 'data', None)
                                    if data:
                                        key_str = data
                                    elif isinstance(k, str) and len(k) == 1:
                                        key_str = k
                                if not key_str:
                                    continue
                                # Update typing and debug
                                try:
                                    import time as _t
                                    self._last_keypress_time = _t.time()
                                except Exception:
                                    pass
                                if getattr(self, '_debug_keys', False):
                                    try:
                                        self._debug_last_key = key_str
                                        self.mark_dirty("header")
                                    except Exception:
                                        pass
                                # Immediate echo for '/'
                                if key_str == "/" and self.realtime_input is not None:
                                    handled = await self.realtime_input.handle_key(key_str)
                                    if handled:
                                        self._force_immediate_footer_refresh()
                                        # Yield to allow immediate footer repaint
                                        await asyncio.sleep(0)
                                        continue
                                # Hybrid/global shortcuts first
                                if hasattr(self, '_handle_hybrid_keyboard_event') and self._handle_hybrid_keyboard_event(key_str):
                                    continue
                                # Forward to realtime input
                                if self.realtime_input is not None:
                                    handled = await self.realtime_input.handle_key(key_str)
                                    if handled:
                                        self._force_immediate_footer_refresh()
                                        # Yield to allow immediate footer repaint
                                        await asyncio.sleep(0)
                                        continue
                                # Enter submits
                                if key_str == "enter" and self.realtime_input is not None:
                                    # Ensure the last typed character is painted before clearing
                                    self._force_immediate_footer_refresh()
                                    await asyncio.sleep(0)
                                    current_input = self.realtime_input.get_current_input().strip()
                                    if current_input:
                                        await self.realtime_input.handle_submit(current_input)
                                        self.realtime_input.clear_input()
                                    self.mark_dirty("footer")
                        except Exception:
                            await asyncio.sleep(0.01)
                return
            except Exception:
                # Fall back to native reader below
                pass
        loop = asyncio.get_running_loop()
        
        while self._running:
            try:
                # FIXED: Reduced timeout for more responsive input with InputMode detection
                key_code, char, input_mode = await loop.run_in_executor(
                    None, 
                    lambda: self._keyboard_input.get_key(timeout=0.05)  # Faster polling
                )
                
                # CRITICAL FIX: Handle InputMode to detect fallback to line-based input
                if input_mode == InputMode.LINE_BASED:
                    # Auto-switch from per-key to line-based input mode
                    try:
                        self._print_system_message("⚠️  Switching to line-based input mode (terminal limitations)")
                        self._print_system_message("💡 Type commands like: 'help', 'sidebar', or '/help' for assistance")
                    except:
                        pass
                    # Cancel the per-key input task and switch to line-based
                    self._input_mode = "line"  # Set flag for health indicator
                    break  # Exit per-key loop to switch to line-based
                
                # Handle timeout (no input)
                if key_code is None and char is None:
                    continue
                
                # CRITICAL FIX: Update typing activity timestamp FIRST
                import time
                self._last_keypress_time = time.time()
                
                
                # Convert KeyCode to string for compatibility
                key_str = None
                if key_code:
                    key_str = self._keycode_to_string(key_code)
                elif char:
                    key_str = char
                    
                if not key_str:
                    continue
                
                # CRITICAL FIX: Handle slash character specially for immediate echo
                if key_str == "/":
                    # Always forward slash directly to input field for immediate echo
                    if self.realtime_input is not None:
                        handled = await self.realtime_input.handle_key(key_str)
                        if handled:
                            self._force_immediate_footer_refresh()  # Force immediate refresh for slash
                            # Yield to allow immediate footer repaint
                            await asyncio.sleep(0)
                            continue
                    
                # CRITICAL FIX: Handle TUI navigation keys first (before text input)
                if await self._handle_tui_navigation(key_str, key_code):
                    continue
                
                # Try hybrid keyboard handler first (shortcuts like Ctrl+B)
                if hasattr(self, '_handle_hybrid_keyboard_event'):
                    handled = self._handle_hybrid_keyboard_event(key_str)
                    if handled:
                        continue
                
                # Forward key to RealTimeInputField if available
                if self.realtime_input is not None:
                    handled = await self.realtime_input.handle_key(key_str)
                    if handled:
                        # CRITICAL FIX: Force immediate footer refresh for typing feedback
                        self._force_immediate_footer_refresh()
                        # Yield to allow immediate footer repaint before next key is processed
                        await asyncio.sleep(0)
                        continue
                # Update debug state on any unhandled key
                if getattr(self, '_debug_keys', False):
                    try:
                        self._debug_last_key = key_str
                        self.mark_dirty("header")
                    except Exception:
                        pass
                
                # Handle Enter with immediate paint before clearing
                if key_code == KeyCode.ENTER:
                    if self.realtime_input is not None:
                        # Force a repaint to show the final character
                        self._force_immediate_footer_refresh()
                        await asyncio.sleep(0)
                        current_input = self.realtime_input.get_current_input().strip()
                        if current_input:
                            await self.realtime_input.handle_submit(current_input)
                            self.realtime_input.clear_input()
                    self.mark_dirty("footer")
                            
            except Exception:
                # On any keyboard error, fall back to line input
                try:
                    self._print_system_message("⚠️  Keyboard input error - switching to line-based mode")
                except:
                    pass
                self._input_mode = "line"
                break
        
        # CRITICAL FIX: If we exit the per-key loop, auto-switch to line-based input
        if self._running and hasattr(self, '_input_mode') and self._input_mode == "line":
            try:
                self._print_system_message("🔄 Starting line-based input mode...")
                await self._read_input()
            except Exception:
                pass
    
    async def _handle_line_based_special_commands(self, line: str) -> bool:
        """
        Handle special commands in line-based input mode.
        Returns True if the command was handled and should not be processed as regular input.
        """
        line = line.strip()
        
        # Handle slash commands (command palette)
        if line.startswith('/'):
            # Remove the leading slash
            command_part = line[1:].strip()
            
            # Show available commands if just "/" was typed
            if not command_part:
                self._show_command_help()
                return True
            
            # Handle specific slash commands
            if command_part.startswith('sidebar') or command_part.startswith('sb'):
                self._toggle_sidebar()
                return True
            elif command_part.startswith('help') or command_part.startswith('h'):
                self._show_command_help()
                return True
            elif command_part.startswith('clear') or command_part.startswith('c'):
                if hasattr(self, 'clear_chat_history'):
                    await self.clear_chat_history()
                return True
            elif command_part.lower() in ['quit', 'exit', 'q']:
                self._running = False
                # CRITICAL FIX: Shutdown command interface when TUI exits
                asyncio.create_task(self._shutdown_command_interface())
                return True
            else:
                # Let regular slash commands be processed as regular input
                return False
        
        # Handle keyboard shortcuts as text commands
        if line.lower() in ['ctrl+b', 'toggle sidebar', 'sidebar']:
            self._toggle_sidebar()
            return True
        elif line.lower() in ['help', '?']:
            self._show_command_help()
            return True
        elif line.lower() in ['clear', 'cls']:
            if hasattr(self, 'clear_chat_history'):
                await self.clear_chat_history()
            return True
        elif line.lower() in ['exit', 'quit', 'q']:
            self._running = False
            # CRITICAL FIX: Shutdown command interface when TUI exits
            asyncio.create_task(self._shutdown_command_interface())
            return True
        
        # Not a special command
        return False
    
    def _show_command_help(self):
        """Show available commands for line-based mode."""
        help_text = """
🎯 Available Commands & Shortcuts:

📝 Text Commands:
• Type your message and press Enter to send
• /help or /h - Show this help
• /sidebar or /sb - Toggle sidebar
• /clear or /c - Clear chat history
• exit, quit, q - Exit application

⌨️  Keyboard Shortcuts:
• ? - Quick help (this screen)
• Ctrl+B - Toggle sidebar
• Ctrl+Q - Quit application
• Tab - Cycle focus between areas
• Page Up/Down - Scroll chat history

♿ Accessibility Shortcuts:
• Alt+H - Toggle high contrast mode
• Alt+M - Toggle motion reduction
• Alt+S - Toggle increased spacing

💡 Tip: You can also type regular messages and they'll be sent to the AI.
        """.strip()
        
        # Show help in the system message area
        self._print_system_message(help_text)
                
    def _keycode_to_string(self, keycode: 'KeyCode') -> str:
        """Convert KeyCode enum to string representation."""
        if not KeyCode:
            return ""
            
        mapping = {
            KeyCode.UP: "up",
            KeyCode.DOWN: "down", 
            KeyCode.LEFT: "left",
            KeyCode.RIGHT: "right",
            KeyCode.ENTER: "enter",
            KeyCode.ESCAPE: "escape",
            KeyCode.BACKSPACE: "backspace",
            KeyCode.DELETE: "delete",
            KeyCode.TAB: "tab",
            KeyCode.SPACE: " ",
            KeyCode.HOME: "home",
            KeyCode.END: "end",
            KeyCode.PAGE_UP: "page_up",
            KeyCode.PAGE_DOWN: "page_down",
        }
        return mapping.get(keycode, "")
    
    async def _handle_tui_navigation(self, key_str: str, key_code) -> bool:
        """Handle TUI navigation keys (arrow keys, tab, etc.) for page/mode navigation."""
        if not key_code:
            return False
        
        # Import KeyCode if available
        if not hasattr(self, '_KeyCode'):
            try:
                from .keyboard_input import KeyCode
                self._KeyCode = KeyCode
            except ImportError:
                return False
        
        # Handle TUI navigation (not text input)
        current_input = ""
        if self.realtime_input:
            current_input = self.realtime_input.get_current_input().strip()
        
        # Only handle navigation when input field is empty (TUI navigation mode)
        if current_input:
            return False  # Let input field handle navigation within text
        
        if key_code == self._KeyCode.LEFT or key_code == self._KeyCode.RIGHT:
            # Navigate between pages/modes
            if hasattr(self, '_navigate_pages'):
                direction = 1 if key_code == self._KeyCode.RIGHT else -1
                self._navigate_pages(direction)
                self.mark_dirty("all")
                return True
        
        elif key_code == self._KeyCode.UP or key_code == self._KeyCode.DOWN:
            # Navigate within current page/mode
            if hasattr(self, '_navigate_within_page'):
                direction = 1 if key_code == self._KeyCode.DOWN else -1
                self._navigate_within_page(direction)
                self.mark_dirty("content")
                return True
        
        elif key_code == self._KeyCode.TAB:
            # Tab navigation between UI elements
            if hasattr(self, '_navigate_focus'):
                self._navigate_focus()
                self.mark_dirty("all")
                return True
        
        return False
    
    def _refresh_input_display(self) -> None:
        """Force UI refresh to show updated input state."""
        self.mark_dirty("footer")  # Input display refresh
        
    def _force_immediate_footer_refresh(self) -> None:
        """CRITICAL FIX: Force immediate footer refresh bypassing debounce for typing."""
        # Do not skip during rendering; flag immediate and let loop refresh once
        # Bump footer cache version to ensure layout hash changes
        try:
            if "footer" in self._cache_version:
                self._cache_version["footer"] += 1
        except Exception:
            pass
        
        # Set immediate refresh flag and trigger single refresh
        self._immediate_footer_refresh = True
        
        # CRITICAL FIX: Also mark footer as dirty to ensure it gets updated
        self.mark_dirty("footer")
        
        self._refresh_event.set()

    # ------------------------------------------------------------------- #
    # SSE listener for real-time status updates
    # ------------------------------------------------------------------- #
    
    async def _start_sse_listener(self) -> None:
        """Start SSE listener for real-time status updates."""
        if not _HAS_HTTPX:
            return  # Skip if httpx not available
            
        # Start SSE listener task
        if self._sse_task is None:
            self._sse_task = asyncio.create_task(self._sse_listener_loop())
    
    async def _stop_sse_listener(self) -> None:
        """Stop SSE listener."""
        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
            self._sse_task = None
    
    async def _sse_listener_loop(self) -> None:
        """Main SSE listener loop."""
        base_url = getattr(self.config, 'web_ui_base_url', 'http://localhost:8000')
        sse_url = f"{base_url}/api/events"
        
        while self._running:
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream("GET", sse_url, headers={
                        "Accept": "text/event-stream",
                        "Cache-Control": "no-cache"
                    }) as response:
                        if response.status_code != 200:
                            await asyncio.sleep(5)  # Retry after 5 seconds
                            continue
                            
                        async for line in response.aiter_lines():
                            if not self._running:
                                break
                                
                            line = line.strip()
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])  # Remove "data: " prefix
                                    await self._handle_sse_event(data)
                                except json.JSONDecodeError:
                                    continue
                                    
            except Exception:
                # Connection failed, retry after 10 seconds
                await asyncio.sleep(10)
    
    async def _handle_sse_event(self, event: Dict[str, Any]) -> None:
        """Handle incoming SSE event."""
        event_type = event.get("type", "")
        
        if event_type == "status_update":
            # Update cached status for header display, but only trigger refresh if meaningful change
            new_status = event.get("data", {})
            if not self._last_status_update or self._status_has_meaningful_change(self._last_status_update, new_status):
                self._last_status_update = new_status
                self.mark_dirty("header")  # Trigger header refresh
            
        elif event_type == "agent_spawned":
            # New agent started - could show notification in sidebar
            self.mark_dirty("content")
            
        elif event_type == "agent_completed":
            # Agent completed - update display
            self.mark_dirty("content")
            
        elif event_type == "cost_update":
            # Cost information changed - update header only if significant change
            new_cost_data = event.get("data", {})
            old_cost = self._last_status_update.get('total_cost', 0) if self._last_status_update else 0
            new_cost = new_cost_data.get('total_cost', 0)
            
            # Only update if cost change is significant (>1 cent)
            if abs(float(new_cost) - float(old_cost)) > 0.01:
                if self._last_status_update:
                    self._last_status_update.update(new_cost_data)
                else:
                    self._last_status_update = new_cost_data
                self.mark_dirty("header")

    # ------------------------------------------------------------------- #
    # User input dispatcher – interprets slash‑commands and forwards chat.
    # ------------------------------------------------------------------- #
    async def _handle_user_input(self, raw: str) -> None:
        """
        Interpret user input and manage chat history.
        Handles commands and forwards chat messages to ConversationManager.
        Supports hybrid TUI command palette.
        """
        stripped = raw.strip()
        
        # Handle command palette if active
        if self._command_palette_active:
            self._deactivate_command_palette()  # Deactivate after command
            if stripped:
                self._execute_command_palette_command(stripped)
            return
        
        # Record the user message in chat history first
        if self.chat_history is not None and stripped:
            try:
                self.chat_history.add_message(stripped, "user")
                self._auto_scroll_to_bottom()  # Auto-scroll to show new message
                self.mark_dirty("content")  # Only refresh chat history after new message
            except Exception:
                # History failure should not break the UI
                pass
        
        # FIXED: Ensure all commands properly start with "/" and handle them correctly
        if stripped.startswith("/"):
            tokens = stripped[1:].split(maxsplit=1)
            cmd = tokens[0].lower()
            arg = tokens[1] if len(tokens) > 1 else ""
            
            # FIXED: Process all slash commands with proper feedback
            if cmd == "mode":
                await self._process_mode_command(arg)
                self.mark_dirty("header")  # Mode change only affects header
            elif cmd in {"quit", "exit", "q"}:
                self._running = False
                self._print_system_message("Shutting down...")
                # CRITICAL FIX: Shutdown command interface when TUI exits
                await self._shutdown_command_interface()
            elif cmd == "help":
                # Use hybrid help if sidebar state variables exist (hybrid mode)
                if hasattr(self, '_sidebar_collapsed'):
                    self._show_hybrid_help()
                else:
                    self._show_help()
                self.mark_dirty("content")  # Help text appears in body
            elif cmd == "sidebar":
                # Handle sidebar commands in hybrid mode
                if hasattr(self, '_sidebar_collapsed'):
                    if arg.lower() in ("toggle", ""):
                        self._toggle_sidebar()
                    elif arg.lower() == "show":
                        if self._sidebar_collapsed:
                            self._toggle_sidebar()
                    elif arg.lower() == "hide":
                        if not self._sidebar_collapsed:
                            self._toggle_sidebar()
                    else:
                        self._print_system_message(f"Unknown sidebar command: {arg}")
                else:
                    self._print_system_message("Sidebar commands only available in hybrid mode")
            elif cmd == "clear":
                # FIXED: Add clear command for input history
                if self.chat_history is not None:
                    try:
                        # Clear chat history if method exists
                        if hasattr(self.chat_history, 'clear_history'):
                            self.chat_history.clear_history()
                        self.mark_dirty("content")
                        self._print_system_message("Chat history cleared")
                    except Exception:
                        self._print_system_message("Failed to clear chat history")
                else:
                    self._print_system_message("No chat history to clear")
            else:
                error_msg = f"Unknown command: /{cmd}. Try /help for available commands."
                self._print_system_message(error_msg)
                # Record the error in chat history
                if self.chat_history is not None:
                    try:
                        self.chat_history.add_message(error_msg, "error")
                        self.mark_dirty("content")  # Error appears in chat history
                    except Exception:
                        pass
        else:
            # Forward to conversation manager
            try:
                response = await self._send_chat_message(stripped)
                if response:
                    # Record the assistant response in chat history
                    if self.chat_history is not None:
                        try:
                            self.chat_history.add_message(str(response), "assistant")
                            self._auto_scroll_to_bottom()  # Auto-scroll to show new response
                            self.mark_dirty("content")  # Assistant response in chat history
                        except Exception:
                            pass
            except Exception as exc:
                error_msg = f"[red]Chat error:[/red] {exc}"
                self._print_system_message(error_msg)
                # Record the error in chat history
                if self.chat_history is not None:
                    try:
                        self.chat_history.add_message(str(exc), "error")
                        self.mark_dirty("content")  # Error appears in chat history
                    except Exception:
                        pass

    async def _process_mode_command(self, arg: str) -> None:
        """Switch UI mode based on the argument passed to ``/mode``."""
        mapping = {
            "zen": TUIMode.ZEN,
            "dashboard": TUIMode.DASHBOARD,
            "command": TUIMode.COMMAND_CENTER,
            "command_center": TUIMode.COMMAND_CENTER,
        }
        target = mapping.get(arg.lower())
        if target is None:
            self._print_system_message(
                f"Invalid mode '{arg}'. Available: zen, dashboard, command."
            )
            return
        await self.switch_mode(target)
        self._print_system_message(f"Switched to {target.value} mode")

    def _show_help(self) -> None:
        """Display help information."""
        help_text = """
Available commands:
  /mode zen          - Switch to minimal chat interface
  /mode dashboard    - Switch to metrics overview
  /mode command      - Switch to full technical interface  
  /quit or /exit     - Exit the TUI
  /clear             - Clear chat history
  /help              - Show this help message
  
Just type your message to chat with AI agents!
All commands must start with "/" character.
        """.strip()
        self._print_system_message(help_text)

    # ------------------------------------------------------------------- #
    # Chat integration helpers
    # ------------------------------------------------------------------- #
    def _get_chat_history(self):
        """Get chat history from conversation manager."""
        try:
            if hasattr(self.conversation_manager, 'get_history'):
                return self.conversation_manager.get_history()
            elif hasattr(self.conversation_manager, 'chat_history'):
                return self.conversation_manager.chat_history
            elif hasattr(self.conversation_manager, 'messages'):
                return self.conversation_manager.messages
        except Exception:
            pass
        return []

    async def _send_chat_message(self, message: str):
        """Send a chat message via conversation manager."""
        try:
            if hasattr(self.conversation_manager, 'process_input'):
                response = self.conversation_manager.process_input(message)
                if asyncio.iscoroutine(response):
                    response = await response
                return response
            elif hasattr(self.conversation_manager, 'send_message'):
                response = self.conversation_manager.send_message(message)
                if asyncio.iscoroutine(response):
                    response = await response  
                return response
            elif hasattr(self.conversation_manager, 'send_user_message'):
                response = self.conversation_manager.send_user_message(message)
                if asyncio.iscoroutine(response):
                    response = await response
                return response
        except Exception as exc:
            self._print_system_message(f"Chat integration error: {exc}")
            return None

    # ------------------------------------------------------------------- #
    # Helper: simple console output when Rich cannot be used.
    # ------------------------------------------------------------------- #
    async def _fallback_cli(self) -> None:
        """
        Very thin "fallback" UI for environments without TTY or Rich.
        It simply loops over ``input()`` and forwards messages to the
        ``ConversationManager``.  Errors are printed using ``print``.
        """
        print("=== AgentsMCP (fallback CLI) ===")
        print("Type '/quit' to exit, '/mode <name>' to change UI mode (no visual effect).")
        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            stripped = user_input.strip()
            if stripped.startswith("/"):
                # Only support /quit in fallback to keep it simple.
                cmd = stripped[1:].split(maxsplit=1)[0].lower()
                if cmd in ("quit", "exit"):
                    break
                elif cmd == "help":
                    print("Available commands: /quit, /exit")
                else:
                    print(f"Unknown command in fallback mode: {cmd}")
                continue

            try:
                resp = await self._send_chat_message(stripped)
                if resp:
                    print(f"[assistant] {resp}")
                else:
                    print("[assistant] Message processed")
            except Exception as exc:
                print(f"[error] {exc}")

    # ------------------------------------------------------------------- #
    # Hybrid TUI Keyboard Event Handling
    # ------------------------------------------------------------------- #
    def _handle_hybrid_keyboard_event(self, key: str) -> bool:
        """
        Handle keyboard events for hybrid TUI mode.
        Returns True if handled, False otherwise.
        """
        # Handle control characters (Ctrl combinations)
        if key == '\x02':  # Ctrl+B (ASCII 2)
            self._toggle_sidebar()
            return True
        elif key == '\x11':  # Ctrl+Q (ASCII 17) - alternative quit
            self._running = False
            # CRITICAL FIX: Shutdown command interface when TUI exits
            asyncio.create_task(self._shutdown_command_interface())
            return True
            
        # FIXED: Handle global shortcuts - but don't intercept "/" for command palette
        # Let the "/" character flow through to input field for proper echo
        elif key == "escape" and self._command_palette_active:
            self._deactivate_command_palette()
            return True
        elif key == "?":
            # Show help overlay when "?" is pressed (discoverability improvement)
            self._show_command_help()
            return True
        elif key.startswith("alt+"):
            # Handle accessibility keyboard shortcuts
            alt_key = key[4:]  # Remove "alt+" prefix
            if alt_key == "h":
                # Alt+H: Toggle high contrast mode
                current = self._accessibility_config.get("high_contrast", False)
                self.set_accessibility_option("high_contrast", not current)
                contrast_status = "enabled" if not current else "disabled"
                self._print_system_message(f"High contrast {contrast_status}")
                return True
            elif alt_key == "m":
                # Alt+M: Toggle motion reduction
                current = self._accessibility_config.get("reduce_motion", False)
                self.set_accessibility_option("reduce_motion", not current)
                motion_status = "reduced" if not current else "normal"
                self._print_system_message(f"Motion sensitivity: {motion_status}")
                return True
            elif alt_key == "s":
                # Alt+S: Toggle increased spacing
                current = self._accessibility_config.get("increase_spacing", False)
                self.set_accessibility_option("increase_spacing", not current)
                spacing_status = "increased" if not current else "normal"
                self._print_system_message(f"Spacing: {spacing_status}")
                return True
        elif key == "tab":
            if not self._sidebar_collapsed:
                self._cycle_focus()
            return True
        elif key in ["page_up", "pgup"]:
            if self._current_focus == FocusRegion.MAIN:
                self._scroll_chat_history(-5)  # Scroll up 5 lines
            return True
        elif key in ["page_down", "pgdn"]:
            if self._current_focus == FocusRegion.MAIN:
                self._scroll_chat_history(5)  # Scroll down 5 lines
            return True
        elif key in ["up", "down"] and self._current_focus == FocusRegion.SIDEBAR and not self._sidebar_collapsed:
            self._navigate_sidebar(key)
            return True
        elif key in ["up", "down"] and self._current_focus == FocusRegion.MAIN:
            # Fine-grained scroll when focused on main content
            delta = -1 if key == "up" else 1
            self._scroll_chat_history(delta)
            return True
        elif key in ["left", "right"] and not self._sidebar_collapsed:
            # Horizontal navigation switches focus between sidebar and main
            if key == "left":
                self._current_focus = FocusRegion.SIDEBAR
            else:
                self._current_focus = FocusRegion.MAIN
            self._save_ui_state()
            self.mark_dirty("footer")
            return True
        elif key == "enter" and self._current_focus == FocusRegion.SIDEBAR and not self._sidebar_collapsed:
            # Selecting current sidebar page moves focus to main content
            self._current_focus = FocusRegion.MAIN
            self._save_ui_state()
            self.mark_dirty("footer")
            return True
            
        return False
        
    def _toggle_sidebar(self) -> None:
        """Toggle sidebar visibility (Ctrl+B)."""
        self._sidebar_collapsed = not self._sidebar_collapsed
        
        # Rebuild layout to reflect the change
        self._layout = self._build_layout()
        
        # Mark sections dirty to trigger re-render
        self.mark_dirty("header")
        self.mark_dirty("content") 
        self.mark_dirty("footer")
        
        # Save UI state
        self._save_ui_state()
        
        # Show feedback message
        status = "hidden" if self._sidebar_collapsed else "shown"
        self._print_system_message(f"Sidebar {status} (Ctrl+B to toggle)")
        
    def _activate_command_palette(self) -> None:
        """Activate command palette with '/' prefix."""
        self._command_palette_active = True
        self._current_focus = FocusRegion.INPUT
        self.mark_dirty("footer")  # Update footer to show command palette
        
    def _deactivate_command_palette(self) -> None:
        """Deactivate command palette (Escape key)."""
        self._command_palette_active = False
        self.mark_dirty("footer")  # Update footer to normal input
        
    def _cycle_focus(self) -> None:
        """Cycle focus between regions (Tab key)."""
        if self._sidebar_collapsed:
            # Only INPUT and MAIN when sidebar is collapsed
            if self._current_focus == FocusRegion.INPUT:
                self._current_focus = FocusRegion.MAIN
            else:
                self._current_focus = FocusRegion.INPUT
        else:
            # Cycle through all regions when sidebar is visible
            if self._current_focus == FocusRegion.INPUT:
                self._current_focus = FocusRegion.MAIN
            elif self._current_focus == FocusRegion.MAIN:
                self._current_focus = FocusRegion.SIDEBAR
            else:
                self._current_focus = FocusRegion.INPUT
                
        # Save UI state
        self._save_ui_state()
        
        self.mark_dirty("footer")  # Update focus indicators
        
    def _scroll_chat_history(self, delta: int) -> None:
        """Scroll chat history by delta lines."""
        # Get total message count to limit scrolling
        try:
            history = self._get_chat_history()
            max_scroll = max(0, len(history) - 20)  # Show max 20 messages at a time
            self._chat_scroll_offset = max(0, min(max_scroll, self._chat_scroll_offset + delta))
        except Exception:
            self._chat_scroll_offset = max(0, self._chat_scroll_offset + delta)
        self.mark_dirty("content")  # Re-render chat content
    
    def _auto_scroll_to_bottom(self) -> None:
        """Auto-scroll to show the most recent messages (bottom of chat)."""
        try:
            history = self._get_chat_history()
            if len(history) > 20:
                # Scroll to show the last 20 messages
                self._chat_scroll_offset = len(history) - 20
            else:
                # If 20 or fewer messages, show from the top
                self._chat_scroll_offset = 0
            self.mark_dirty("content")
        except Exception:
            self._chat_scroll_offset = 0
        
    def _navigate_sidebar(self, direction: str) -> None:
        """Navigate sidebar pages with up/down arrows."""
        pages = list(SidebarPage)
        current_index = pages.index(self._current_page)
        
        if direction == "up" and current_index > 0:
            self._current_page = pages[current_index - 1]
        elif direction == "down" and current_index < len(pages) - 1:
            self._current_page = pages[current_index + 1]
        else:
            return  # No change needed
        
        # Save UI state
        self._save_ui_state()
            
        # Re-render sidebar and content
        self.mark_dirty("sidebar")
        self.mark_dirty("content")
        
    def _navigate_pages(self, direction: int) -> None:
        """Navigate between pages/modes using left/right arrows."""
        # If sidebar is visible, treat as hybrid page navigation; otherwise cycle modes
        if not getattr(self, '_sidebar_collapsed', True):
            pages = list(SidebarPage)
            current_index = pages.index(self._current_page)
            new_index = (current_index + direction) % len(pages)
            self._current_page = pages[new_index]
            self._save_ui_state()
        else:
            modes = list(TUIMode)
            current_index = modes.index(self._current_mode)
            new_index = (current_index + direction) % len(modes)
            self._current_mode = modes[new_index]
            self._save_ui_state()
            
    def _navigate_within_page(self, direction: int) -> None:
        """Navigate within current page using up/down arrows."""
        if hasattr(self, '_current_focus') and self._current_focus == FocusRegion.MAIN:
            # Scroll chat history when main content is focused
            self._scroll_chat_history(direction * 5)
        elif hasattr(self, '_current_focus') and self._current_focus == FocusRegion.SIDEBAR:
            # Navigate sidebar pages
            self._navigate_sidebar("down" if direction > 0 else "up")
            
    def _navigate_focus(self) -> None:
        """Navigate between UI focus regions using Tab."""
        # Use existing cycle_focus method
        self._cycle_focus()
        
    def _execute_command_palette_command(self, command: str) -> None:
        """Execute a command from the command palette."""
        command = command.strip()
        
        # Handle built-in commands with '/' prefix
        if command.startswith("/"):
            cmd_parts = command[1:].split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
            
            if cmd == "mode":
                asyncio.create_task(self._handle_mode_command(arg))
            elif cmd in ("quit", "exit"):
                asyncio.create_task(self._handle_quit_command())
            elif cmd == "help":
                self._show_hybrid_help()
            elif cmd == "sidebar":
                if arg.lower() in ("toggle", ""):
                    self._toggle_sidebar()
                elif arg.lower() == "show":
                    if self._sidebar_collapsed:
                        self._toggle_sidebar()
                elif arg.lower() == "hide":
                    if not self._sidebar_collapsed:
                        self._toggle_sidebar()
            else:
                self._print_system_message(f"Unknown command: /{cmd}")
        else:
            # Regular chat message
            asyncio.create_task(self._send_chat_message(command))
    
    async def _handle_quit_command(self):
        """Handle quit command for hybrid mode."""
        self._running = False
        self._print_system_message("Shutting down...")
        # CRITICAL FIX: Shutdown command interface when TUI exits
        await self._shutdown_command_interface()
    
    async def _shutdown_command_interface(self):
        """Shutdown the command interface to prevent prompt flooding."""
        try:
            if hasattr(self, 'conversation_manager') and self.conversation_manager:
                if hasattr(self.conversation_manager, 'command_interface') and self.conversation_manager.command_interface:
                    # Signal the command interface to stop its conversational loop
                    self.conversation_manager.command_interface.is_running = False
                    logger.debug("Command interface shutdown signal sent")
        except Exception as e:
            logger.warning(f"Error shutting down command interface: {e}")
            
    def _show_hybrid_help(self) -> None:
        """Show help for hybrid TUI mode."""
        help_text = """
Hybrid TUI Commands:
  /mode zen|dashboard|command  - Switch UI mode
  /sidebar [toggle|show|hide]  - Control sidebar
  /clear                      - Clear chat history
  /quit or /exit              - Exit the TUI
  /help                       - Show this help
  
Keyboard Shortcuts:
  Ctrl+B                      - Toggle sidebar
  Tab                         - Cycle focus (when sidebar open)
  PgUp/PgDn                   - Scroll chat history
  ↑/↓ (in sidebar)           - Navigate pages
  Escape                      - Close command palette

Sidebar Pages:
  Chat, Jobs, Agents, Models, Providers, Costs, MCP, Discovery, Logs, Settings

Just type your message to chat with AI agents!
All commands must start with "/" character.
        """.strip()
        self._print_system_message(help_text)

    # ------------------------------------------------------------------- #
    # Simple system‑message printer (works both with Rich and fallback).
    # ------------------------------------------------------------------- #
    def _print_system_message(self, message: str) -> None:
        # During TUI, prefer in-UI delivery to avoid console leakage
        if self._in_live_context and self.chat_history is not None:
            try:
                self.chat_history.add_message(str(message), "system")
                self.mark_dirty("content")
                return
            except Exception:
                pass
        # Otherwise enqueue to TUI logs if available
        if self._in_live_context:
            self._enqueue_tui_log("INFO", str(message))
            return
        # Fallback for non-TUI (e.g., fallback CLI)
        if self._console:
            try:
                self._console.print(f"[dim yellow]System: {message}[/dim yellow]")
            except Exception:
                print(f"System: {message}")
        else:
            print(f"System: {message}")
