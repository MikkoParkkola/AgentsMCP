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
import contextlib
import json
import sys
import traceback
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
    from .keyboard_input import KeyboardInput, KeyCode
except Exception as e:  # pragma: no cover
    # If the components cannot be imported we fall back to the legacy
    # input & history handling.  
    EnhancedChatInput = None   # type: ignore
    ChatHistoryDisplay = None  # type: ignore
    KeyboardInput = None  # type: ignore
    KeyCode = None  # type: ignore
    RealTimeInputField = None  # type: ignore


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
        
        # Enhanced chat components - initialized later after console is ready
        self.enhanced_input = None
        self.chat_history = None
        self.realtime_input = None
        
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
        
        # Initialize console early for RealTimeInputField
        if Console is not None:
            try:
                self._console = Console(force_terminal=True)
            except Exception:
                self._console = None
        else:
            self._console = None
        
        # Initialize RealTimeInputField now that console is available
        if RealTimeInputField is not None and self._console is not None:
            try:
                self.realtime_input = RealTimeInputField(
                    console=self._console,
                    prompt=">>> ",
                    max_width=None,
                    max_height=3
                )
                self._connect_input_events()
            except Exception:
                # Fallback gracefully if initialization fails
                self.realtime_input = None
        else:
            self.realtime_input = None

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
                
        # Initialize keyboard input for per-key event handling
        if KeyboardInput is not None:
            try:
                self._keyboard_input = KeyboardInput()
                if not self._keyboard_input.is_interactive:
                    self._print_system_message("Running in non-interactive mode - using line-based input")
            except Exception:
                self._keyboard_input = None
        
        self._layout = self._build_layout()
        self._running = True

        # Show a short welcome unless the caller asked to silence it.
        if not self._no_welcome:
            self._render_welcome()

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

        # Event-driven Live rendering – only refreshes when state actually changes
        with Live(self._render(), console=self._console) as live:
            try:
                while self._running:
                    # Wait for either user input OR a refresh request
                    done, pending = await asyncio.wait(
                        [
                            asyncio.create_task(self._input_queue.get()),
                            asyncio.create_task(self._refresh_event.wait())
                        ],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel pending tasks to prevent resource leaks
                    for task in pending:
                        task.cancel()
                        
                    # Handle user input if received
                    input_received = False
                    for task in done:
                        if hasattr(task, '_coro') and 'get' in str(task._coro):
                            try:
                                user_input = task.result()
                                await self._handle_user_input(user_input)
                                input_received = True
                            except Exception:
                                pass
                                
                    # Force refresh after input processing
                    if input_received:
                        self.mark_dirty("content")  # Input processing affects chat history
                    
                    # Handle refresh requests (with minimal debouncing for responsiveness)
                    if self._should_refresh():
                        # Minimal debounce to batch rapid changes while keeping input responsive
                        await asyncio.sleep(0.01)  # Reduced from 0.03 to 0.01 seconds
                        
                        # Update UI only if still dirty after debounce
                        if self._should_refresh():
                            # Track what actually changed to reduce Live updates
                            changed_sections = self._render_with_change_tracking()
                            if changed_sections:
                                live.update(self._layout)  # Update layout directly instead of full render
                            self._refresh_event.clear()
                            
            except (KeyboardInterrupt, SystemExit):
                self._running = False
            finally:
                input_task.cancel()
                # Give the background task a chance to clean up.
                with contextlib.suppress(asyncio.CancelledError):
                    await input_task
                
                # Stop SSE listener
                await self._stop_sse_listener()

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

    # ------------------------------------------------------------------- #
    # Event-driven refresh API
    # ------------------------------------------------------------------- #
    def mark_dirty(self, section: str = "all") -> None:
        """Mark UI section as needing refresh with smart caching.
        
        Args:
            section: Which section to invalidate ("header", "content", "footer", "all")
        """
        # Invalidate cache for specified section
        if section == "all":
            for key in self._cache_version:
                self._cache_version[key] += 1
        elif section in self._cache_version:
            self._cache_version[section] += 1
        
        # Add debouncing to prevent excessive refreshes, but use shorter debounce for input
        if not hasattr(self, '_last_dirty_time'):
            self._last_dirty_time = 0.0
        
        import time
        now = time.time()
        
        # Use shorter debounce for footer (input field) to enable real-time typing
        debounce_time = 0.02 if section == "footer" else 0.1  # 20ms for input, 100ms for others
        
        # Only trigger refresh if enough time has passed
        if now - self._last_dirty_time > debounce_time:
            self._refresh_event.set()
            self._last_dirty_time = now
    
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
        
        # Compare content by creating a hash of its string representation
        try:
            # For Panel objects, get the content and title for comparison
            if hasattr(new_content, 'renderable') and hasattr(new_content, 'title'):
                content_str = str(new_content.renderable) if new_content.renderable else ""
                title_str = str(new_content.title) if new_content.title else ""
                border_str = str(new_content.border_style) if hasattr(new_content, 'border_style') else ""
                new_hash = hash((content_str, title_str, border_str))
            elif hasattr(new_content, '__rich__'):
                new_hash = hash(str(new_content.__rich__()))
            else:
                new_hash = hash(str(new_content))
        except Exception:
            # Fallback to simple string comparison
            new_hash = hash(str(new_content))
            
        cached_hash = self._render_cache.get(current_key)
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
                # Log error details for debugging
                if self._console:
                    self._console.print(f"[red]Render error in {section_name}:[/red] {e}", file=sys.stderr)
                else:
                    print(f"Render error in {section_name}: {e}", file=sys.stderr)
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
            if self._console:
                self._console.print(f"[red]First occurrence of error in {section_name}:[/red] {error}", file=sys.stderr)
            else:
                print(f"First occurrence of error in {section_name}: {error}", file=sys.stderr)
        self._error_suppression[error_key] += 1
        
    def _connect_input_events(self) -> None:
        """Connect RealTimeInputField events to the TUI's refresh and input system."""
        if not self.realtime_input:
            return
            
        # Connect both submit and change events for real-time feedback
        async def on_input_submit(text: str) -> None:
            await self._input_queue.put(text)
            self.realtime_input.clear_input()  # Clear after submission
            self.mark_dirty("footer")  # Refresh input area on submit
            
        async def on_input_change(text: str) -> None:
            # Trigger real-time visual updates for typing feedback
            self.mark_dirty("footer")
            
        self.realtime_input.on_submit(on_input_submit)
        self.realtime_input.on_change(on_input_change)

    def _render_welcome(self) -> None:
        """Display a brief welcome message."""
        if self._console:
            welcome_panel = Panel(
                Text("Welcome to AgentsMCP Modern TUI!\nType your message or '/help' for commands.", 
                     style="dim", justify="center"),
                title="🚀 AgentsMCP",
                border_style="blue"
            )
            self._console.print(welcome_panel)
            self._console.print()  # Empty line

    # ------------------------------------------------------------------- #
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
            main_area.update(Layout(name="content"))
        else:
            # Split main area: sidebar | content
            main_area.split_row(
                Layout(name="sidebar", size=25),  # Fixed-width sidebar
                Layout(name="content", ratio=1),  # Flexible content area
            )
            
        return base

    # ------------------------------------------------------------------- #
    # Hybrid TUI rendering - unified approach with sidebar support
    # ------------------------------------------------------------------- #
    def _render(self) -> RenderableType:
        """Return a complete renderable representing the hybrid TUI."""
        if not self._layout:
            return Text("[red]Layout not initialised[/red]")

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
        return self._layout
        
    # ----------------- Hybrid TUI Component Renderers ----------------- #
    def _render_hybrid_header(self) -> None:
        """Render status header with real-time connection info and current page."""
        def generate_header():
            import time
            from datetime import datetime
            
            # Left side - app status
            left_items = []
            
            # Connection status with real-time indicator
            try:
                # Check if orchestration manager is responsive
                if hasattr(self.orchestration_manager, 'user_settings'):
                    settings = self.orchestration_manager.user_settings()
                    left_items.append("[green]●[/green] Ready")
                else:
                    left_items.append("[yellow]●[/yellow] Limited")
            except Exception:
                left_items.append("[red]●[/red] Disconnected")
            
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
            
            # Current time
            current_time = datetime.now().strftime("%H:%M:%S")
            right_items.append(f"[dim]{current_time}[/dim]")
            
            # Web UI access method
            right_items.append("[dim]Web: agentsmcp dashboard --port 8000[/dim]")
            
            # Memory usage (if available)
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 80:
                    right_items.append(f"[red]RAM: {memory_percent:.0f}%[/red]")
                elif memory_percent > 60:
                    right_items.append(f"[yellow]RAM: {memory_percent:.0f}%[/yellow]")
                else:
                    right_items.append(f"[green]RAM: {memory_percent:.0f}%[/green]")
            except ImportError:
                # psutil not available, skip memory monitoring
                pass
            except Exception:
                # Error getting memory info, skip
                pass
            
            left_text = " • ".join(left_items)
            right_text = " • ".join(right_items)
            
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
        content_renderer = {
            SidebarPage.CHAT: self._render_chat_content,
            SidebarPage.JOBS: self._render_jobs_content,
            SidebarPage.AGENTS: self._render_agents_content,
            SidebarPage.MODELS: self._render_models_content,
            SidebarPage.PROVIDERS: self._render_providers_content,
            SidebarPage.COSTS: self._render_costs_content,
            SidebarPage.MCP: self._render_mcp_content,
            SidebarPage.DISCOVERY: self._render_discovery_content,
            SidebarPage.SETTINGS: self._render_settings_content,
        }.get(self._current_page, self._render_chat_content)
        
        content_renderer()
        
    def _render_hybrid_footer(self) -> None:
        """Render footer with context-aware hotkeys and hints."""
        def generate_footer():
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
        
        # Use error handling for footer rendering
        footer = self._render_with_fallback("footer", generate_footer, "Input hints unavailable")
        try:
            self._update_section_if_changed("footer", footer)
            self._reset_error_count("footer")  # Reset error count on success
        except Exception as e:
            self._log_suppressed_error("footer", e)
        
    # ----------------- Page Content Renderers ----------------- #
    def _render_chat_content(self) -> None:
        """Render chat interface with scrollable history."""
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        
        # Split content into chat history and input
        content_layout.split(
            Layout(name="chat_history", ratio=1),
            Layout(name="chat_input", size=3),
        )
        
        # Render scrollable chat history
        self._render_scrollable_chat_history()
        
        # Render input field
        if self.realtime_input is not None:
            try:
                input_panel = self.realtime_input.render()
                content_layout["chat_input"].update(input_panel)
            except Exception:
                self._render_static_input()
        else:
            self._render_static_input()
            
    def _render_scrollable_chat_history(self) -> None:
        """Render chat history with scroll support."""
        def render_enhanced_chat():
            if self.chat_history is not None:
                # Get terminal height and calculate available space
                terminal_height = self._console.size.height if self._console else 24
                available_height = max(terminal_height - 9, 5)  # Reserve space for header/footer/input
                
                # Render with scroll offset (note: ChatHistoryDisplay doesn't support scroll_offset yet)
                return self.chat_history.render_history(height=available_height)
            else:
                raise Exception("Enhanced chat history not available")
        
        try:
            chat_display = self._render_with_fallback("chat_history", render_enhanced_chat, "Enhanced chat unavailable")
            content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
            content_layout["chat_history"].update(chat_display)
            self._reset_error_count("chat_history")  # Reset error count on success
        except Exception:
            self._render_legacy_scrollable_chat()
            
    def _render_legacy_scrollable_chat(self) -> None:
        """Legacy scrollable chat history rendering."""
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
            
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        # Use error handling for legacy chat rendering
        chat_panel = self._render_with_fallback("legacy_chat", generate_chat, "Chat history unavailable")
        try:
            content_layout["chat_history"].update(chat_panel)
            self._reset_error_count("legacy_chat")  # Reset error count on success
        except Exception as e:
            self._log_suppressed_error("legacy_chat", e)
        
    def _render_static_input(self) -> None:
        """Render static input field fallback."""
        input_panel = Panel(
            Text("Type your message (Shift+Enter for newline) >>> ", style="bold yellow"),
            height=3,
            style="dim",
        )
        content_layout = self._layout["main_area"] if self._sidebar_collapsed else self._layout["content"]
        content_layout["chat_input"].update(input_panel)
        
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
                footer = self.realtime_input.render()
                self._layout["footer"].update(footer)
            except Exception:
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
                footer = self.realtime_input.render()
                if self._update_section_if_changed("footer", footer):
                    changed.append("footer")
            except Exception:
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
        Fixed line-based input reading that works reliably.
        Reads full lines from stdin and processes them correctly.
        """
        loop = asyncio.get_running_loop()
        
        while self._running:
            try:
                # Read full line - this blocks until Enter but runs in executor
                line = await loop.run_in_executor(None, sys.stdin.readline)
                
                if not line:  # EOF
                    self._running = False
                    break
                    
                # Clean the input and submit it
                clean_line = line.rstrip('\n')
                if clean_line.strip():  # Only process non-empty lines
                    # Submit to realtime input field if available
                    if self.realtime_input is not None:
                        await self.realtime_input.handle_submit(clean_line)
                    else:
                        # Direct to input queue for fallback
                        await self._input_queue.put(clean_line)
                        
            except Exception:
                break
    
    async def _read_keyboard_input(self) -> None:
        """
        Per-key input reading for interactive terminals.
        Enables live typing, keyboard shortcuts, and real-time interaction.
        """
        loop = asyncio.get_running_loop()
        
        while self._running:
            try:
                # Read a single key with timeout
                key_code, char = await loop.run_in_executor(
                    None, 
                    lambda: self._keyboard_input.get_key(timeout=0.1)
                )
                
                # Handle timeout (no input)
                if key_code is None and char is None:
                    continue
                
                # Convert KeyCode to string for compatibility
                key_str = None
                if key_code:
                    key_str = self._keycode_to_string(key_code)
                elif char:
                    key_str = char
                    
                if not key_str:
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
                        # Input handled successfully - no need to trigger refresh here
                        # The RealTimeInputField will trigger change events automatically
                        continue
                
                # Handle special keys that create complete input
                if key_code == KeyCode.ENTER:
                    # Get current input and submit
                    if self.realtime_input is not None:
                        current_input = self.realtime_input.get_current_input().strip()
                        if current_input:
                            await self.realtime_input.handle_submit(current_input)
                            self.realtime_input.clear_input()
                    self.mark_dirty("footer")
                            
            except Exception:
                # On any keyboard error, fall back to line input
                break
                
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
    
    def _refresh_input_display(self) -> None:
        """Force UI refresh to show updated input state."""
        self.mark_dirty("footer")  # Input display refresh

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
            # Update cached status for header display
            self._last_status_update = event.get("data", {})
            self.mark_dirty("header")  # Trigger header refresh
            
        elif event_type == "agent_spawned":
            # New agent started - could show notification in sidebar
            self.mark_dirty("content")
            
        elif event_type == "agent_completed":
            # Agent completed - update display
            self.mark_dirty("content")
            
        elif event_type == "cost_update":
            # Cost information changed - update header
            self._last_status_update = event.get("data", {})
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
        
        if stripped.startswith("/"):
            tokens = stripped[1:].split(maxsplit=1)
            cmd = tokens[0].lower()
            arg = tokens[1] if len(tokens) > 1 else ""
            if cmd == "mode":
                await self._process_mode_command(arg)
                self.mark_dirty("header")  # Mode change only affects header
            elif cmd in {"quit", "exit"}:
                self._running = False
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
            else:
                error_msg = f"Unknown command: {cmd}"
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
  /help              - Show this help message
  
Just type your message to chat with AI agents!
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
            return True
            
        # Handle global shortcuts
        elif key == "/" and not self._command_palette_active:
            self._activate_command_palette()
            return True
        elif key == "escape" and self._command_palette_active:
            self._deactivate_command_palette()
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
        self.mark_dirty("body")  # Re-render chat content
    
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
        self.mark_dirty("body")
        
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
            
    def _show_hybrid_help(self) -> None:
        """Show help for hybrid TUI mode."""
        help_text = """
Hybrid TUI Commands:
  /mode zen|dashboard|command  - Switch UI mode
  /sidebar [toggle|show|hide]  - Control sidebar
  /quit or /exit              - Exit the TUI
  /help                       - Show this help
  
Keyboard Shortcuts:
  Ctrl+B                      - Toggle sidebar
  Tab                         - Cycle focus (when sidebar open)
  /                           - Open command palette  
  PgUp/PgDn                   - Scroll chat history
  ↑/↓ (in sidebar)           - Navigate pages
  Escape                      - Close command palette

Sidebar Pages:
  Chat, Jobs, Agents, Models, Providers, Costs, MCP, Discovery, Settings

Just type your message to chat with AI agents!
        """.strip()
        self._print_system_message(help_text)

    # ------------------------------------------------------------------- #
    # Simple system‑message printer (works both with Rich and fallback).
    # ------------------------------------------------------------------- #
    def _print_system_message(self, message: str) -> None:
        if self._console:
            self._console.print(f"[dim yellow]System: {message}[/dim yellow]")
        else:
            print(f"System: {message}")