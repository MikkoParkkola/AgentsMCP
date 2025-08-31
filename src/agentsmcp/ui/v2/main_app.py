"""
Main TUI Application Entry Point - Complete v2 TUI System Integration

This module provides the main application entry point that integrates all v2 TUI components
into a complete working system, replacing the broken v1 TUI with reliable functionality.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import signal
from typing import Optional, Any, Dict
from pathlib import Path

from .application_controller import ApplicationController, ApplicationConfig, ApplicationState
from .terminal_manager import create_terminal_manager, TerminalType
from .event_system import create_event_system, Event, EventType
from .input_handler import InputHandler, InputEvent, InputEventType
from .display_renderer import DisplayRenderer
from .layout_engine import create_standard_tui_layout
from .themes import create_theme_manager, detect_preferred_scheme, ColorMode
from .chat_interface import create_chat_interface, ChatInterfaceConfig
from .keyboard_processor import KeyboardProcessor, ShortcutContext
from .status_manager import StatusManager, SystemState
from .terminal_state_manager import TerminalStateManager, TerminalMode
from .unified_input_handler import UnifiedInputHandler, InputEventType as UnifiedInputEventType
from .ansi_markdown_processor import ANSIMarkdownProcessor, RenderConfig
from ..cli_app import CLIConfig
import os

logger = logging.getLogger(__name__)


class MainTUIApp:
    """Main TUI Application - Complete v2 system integration.
    
    This class coordinates all v2 components to provide a working TUI that:
    - Shows typed characters immediately (fixes original typing issue)
    - Prevents scrollback pollution (fixes original display issue) 
    - Provides clean chat interface with real-time input/output
    - Handles exit commands (/quit) and Ctrl+C gracefully
    """
    
    def __init__(self, cli_config: Optional[CLIConfig] = None):
        """Initialize the main TUI application.
        
        Args:
            cli_config: CLI configuration, if None uses defaults
        """
        self.cli_config = cli_config or CLIConfig()
        self.app_controller: Optional[ApplicationController] = None
        self.running = False
        self._shutdown_event = asyncio.Event()
        
        # Component references
        self.terminal_manager = None
        self.event_system = None
        self.input_handler = None
        self.display_renderer = None
        self.layout_engine = None
        self.theme_manager = None
        self.chat_interface = None
        self.keyboard_processor = None
        self.status_manager = None
        
        # New critical components for input fix
        self.terminal_state_manager = None
        self.unified_input_handler = None
        self.ansi_processor = None
        
        # Logging suppression during TUI to avoid stray console lines
        self._removed_stream_handlers = []
        
    async def initialize(self) -> bool:
        """Initialize all TUI components in correct order with performance optimization.
        
        Returns:
            True if initialization successful, False otherwise
        """
        import time
        start_time = time.time()
        
        try:
            logger.info("Initializing TUI v2 system...")
            
            # CRITICAL FIX: Initialize terminal state manager first for proper TTY control
            self.terminal_state_manager = TerminalStateManager()
            if not self.terminal_state_manager.initialize():
                logger.error("Failed to initialize terminal state manager")
                return False
            
            # Initialize ANSI markdown processor for text rendering
            config = RenderConfig(width=80, enable_colors=True)
            self.ansi_processor = ANSIMarkdownProcessor(config)
            
            # PERFORMANCE: Fast concurrent initialization where possible
            init_start = time.time()
            
            # 1. Terminal capabilities and management (critical path)
            self.terminal_manager = create_terminal_manager()
            term_init_start = time.time()
            if not await asyncio.wait_for(self.terminal_manager.initialize(), timeout=1.0):
                logger.error("Failed to initialize terminal manager")
                return False
            term_init_time = time.time() - term_init_start
            
            # Check terminal compatibility
            caps = self.terminal_manager.get_capabilities()
            if caps is None or caps.type == TerminalType.UNKNOWN:
                logger.error("Unknown terminal type detected or initialization failed")
                return False
                
            # Update ANSI processor width based on terminal
            if caps.width > 0:
                self.ansi_processor.config.width = caps.width
                
            logger.debug(f"Terminal init: {term_init_time*1000:.1f}ms")
            
            # 2. Parallel initialization of independent components
            async def init_event_system():
                self.event_system = create_event_system()
                await self.event_system.initialize()
            
            async def init_theme_manager():
                color_scheme = self._determine_color_scheme(caps.colors)
                self.theme_manager = create_theme_manager(self.terminal_manager)
                self.theme_manager.set_color_scheme(color_scheme)
            
            # PERFORMANCE: Initialize event system and theme concurrently
            await asyncio.gather(
                init_event_system(),
                init_theme_manager()
            )
            
            # 3. Status manager and display renderer (parallel init)
            async def init_status_manager():
                self.status_manager = StatusManager(self.event_system)
                await self.status_manager.initialize()
                await self.status_manager.set_status(
                    SystemState.LOADING,
                    "Initializing display system..."
                )
            
            async def init_display_renderer():
                self.display_renderer = DisplayRenderer(
                    terminal_manager=self.terminal_manager
                )
                await self.display_renderer.initialize()
            
            display_init_start = time.time()
            await asyncio.gather(
                init_status_manager(),
                init_display_renderer()
            )
            display_init_time = time.time() - display_init_start
            logger.debug(f"Status manager and display renderer init: {display_init_time*1000:.1f}ms")
            
            # 4. Layout engine (lightweight, can be synchronous)
            layout_start = time.time()
            self.layout_engine, self.layout_nodes = create_standard_tui_layout(
                terminal_manager=self.terminal_manager
            )
            layout_time = time.time() - layout_start
            logger.debug(f"Layout engine init: {layout_time*1000:.1f}ms")
            
            # 5. Input handling and keyboard processing (can be concurrent)
            async def init_input_handler():
                self.input_handler = InputHandler(
                    terminal_manager=self.terminal_manager,
                    event_system=self.event_system
                )
                await self.input_handler.initialize()
            
            async def init_keyboard_processor():
                # Keyboard processor depends on input handler, so wait
                await init_input_handler()
                self.keyboard_processor = KeyboardProcessor(
                    input_handler=self.input_handler,
                    event_system=self.event_system
                )
                await self.keyboard_processor.initialize()
            
            # Initialize input components
            input_start = time.time()
            await init_keyboard_processor()
            input_time = time.time() - input_start
            logger.debug(f"Input system init: {input_time*1000:.1f}ms")
            
            # 6. Application controller with status integration
            await self.status_manager.set_status(
                SystemState.LOADING,
                "Starting application controller..."
            )
            
            app_config = ApplicationConfig(
                enable_auto_save=False,  # PERFORMANCE: Disable auto-save during init
                graceful_shutdown_timeout=2.0,  # PERFORMANCE: Faster shutdown
                debug_mode=False
            )
            
            app_start = time.time()
            self.app_controller = ApplicationController(
                config=app_config,
                terminal_manager=self.terminal_manager,
                event_system=self.event_system,
                status_manager=self.status_manager
            )
            if not await asyncio.wait_for(self.app_controller.startup(), timeout=2.0):
                logger.error("Failed to startup application controller")
                await self.status_manager.set_status(
                    SystemState.ERROR,
                    "Application controller startup failed"
                )
                return False
            app_time = time.time() - app_start
            logger.debug(f"Application controller init: {app_time*1000:.1f}ms")
            
            # 7. Chat interface with status updates
            await self.status_manager.set_status(
                SystemState.LOADING,
                "Initializing chat interface..."
            )
            
            chat_start = time.time()
            await self._initialize_chat_interface_deferred()
            chat_time = time.time() - chat_start
            logger.debug(f"Chat interface init: {chat_time*1000:.1f}ms")
            
            # 8. Setup event handlers (lightweight)
            await self._setup_event_handlers()
            
            # 9. Setup signal handlers (synchronous)
            self._setup_signal_handlers()
            
            # Mark as running after successful initialization
            self.running = True
            
            # Set ready status with context information
            await self.status_manager.set_status(
                SystemState.READY,
                "AgentsMCP TUI ready for input"
            )
            
            total_time = time.time() - start_time
            logger.info(f"TUI v2 system initialized successfully in {total_time*1000:.1f}ms")
            
            # PERFORMANCE: Log if initialization is slower than target
            if total_time > 0.5:  # 500ms target
                logger.warning(f"Startup time {total_time*1000:.1f}ms exceeds 500ms target")
                await self.status_manager.set_status(
                    SystemState.WARNING,
                    f"Slow startup: {total_time*1000:.0f}ms (target: <500ms)"
                )
            
            return True
            
        except asyncio.TimeoutError as e:
            logger.error(f"TUI initialization timed out: {e}")
            await self.cleanup()
            return False
        except Exception as e:
            logger.exception(f"Failed to initialize TUI v2 system: {e}")
            await self.cleanup()
            return False
    
    async def _initialize_chat_interface_deferred(self):
        """Initialize chat interface with minimal configuration for fast startup."""
        # PERFORMANCE: Minimal config for faster startup
        chat_config = ChatInterfaceConfig(
            enable_history_search=False,  # Enable later
            enable_multiline=True,
            enable_commands=True,
            max_history_messages=100  # Lower limit initially
        )
        
        self.chat_interface = create_chat_interface(
            application_controller=self.app_controller,
            config=chat_config,
            status_manager=self.status_manager,
            display_renderer=self.display_renderer
        )
        await self.chat_interface.initialize()
    
    def _determine_color_scheme(self, colors: int) -> str:
        """Determine appropriate color scheme based on terminal capabilities."""
        if self.cli_config.theme_mode == "light":
            return "light"
        elif self.cli_config.theme_mode == "dark":
            return "dark"
        else:
            # Auto-detect based on terminal and environment
            return detect_preferred_scheme()
    
    async def _setup_event_handlers(self):
        """Setup event handlers for component integration."""
        
        # Keyboard event handling - handles input, commands, and shortcuts
        async def handle_keyboard_event(event: Event):
            if event.event_type == EventType.KEYBOARD:
                key_data = event.data
                
                # Handle quit commands
                if key_data.get('key') == 'c-c' or key_data.get('key') == 'c-d':  # Ctrl+C or Ctrl+D
                    await self.shutdown()
                elif key_data.get('type') == 'command':
                    command = key_data.get('command', '').strip().lower()
                    if command in ['/quit', '/exit', '/q']:
                        await self.shutdown()
                    else:
                        # Pass to chat interface for handling
                        if self.chat_interface and hasattr(self.chat_interface, 'handle_command'):
                            await self.chat_interface.handle_command(command)
                else:
                    # Pass to chat interface for input handling
                    if self.chat_interface and hasattr(self.chat_interface, 'handle_input_event'):
                        await self.chat_interface.handle_input_event(key_data)
        
        # Application event handling - for startup, shutdown, etc.
        async def handle_application_event(event: Event):
            if event.event_type == EventType.APPLICATION:
                action = event.data.get('action')
                if action == 'shutdown':
                    await self.shutdown()
        
        # Register event handlers
        await self.event_system.subscribe(EventType.KEYBOARD, handle_keyboard_event)
        await self.event_system.subscribe(EventType.APPLICATION, handle_application_event)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self) -> int:
        """Run the main TUI application.
        
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            # Globally suppress logging output to avoid stray console lines during TUI
            try:
                self._prev_logging_disable = logging.root.manager.disable
            except Exception:
                self._prev_logging_disable = 0
            logging.disable(logging.CRITICAL)

            # Force raw input pipeline (use minimal raw reader with backend) if requested
            if os.getenv("AGENTS_TUI_V2_FORCE_RAW_INPUT", "0") == "1":
                return await self._run_minimal_input_mode()

            # Minimal isolated input mode (default on) for stabilizing typing
            if os.getenv("AGENTS_TUI_V2_MINIMAL", "1") == "1":
                return await self._run_minimal_input_mode()

            if not await self.initialize():
                return 1
            
            self.running = True
            logger.info("Starting TUI v2 main application...")

            # Suppress console StreamHandlers to avoid out-of-place log lines
            try:
                root_logger = logging.getLogger()
                removed = []
                for h in list(root_logger.handlers):
                    if isinstance(h, logging.StreamHandler):
                        removed.append(h)
                        root_logger.removeHandler(h)
                self._removed_stream_handlers = removed
            except Exception:
                self._removed_stream_handlers = []
            
            # Display initial interface with status bar
            self.display_renderer.clear_all_regions()
            
            # Do not print welcome or status directly; chat interface will render

            # Activate chat interface so input is live and regions are painted
            try:
                if self.chat_interface and hasattr(self.chat_interface, 'activate'):
                    await self.chat_interface.activate()
            except Exception as e:
                logger.warning(f"Failed to activate chat interface: {e}")
            
            logger.info("TUI v2 interface initialized and ready")
            
            # Start input processing to capture keystrokes (echo is disabled; UI renders input)
            try:
                if self.input_handler:
                    asyncio.create_task(self.input_handler.run_async(app=None))
            except Exception as e:
                logger.warning(f"Input handler run failed: {e}")

            # Main event loop - wait for shutdown
            await self._shutdown_event.wait()
            
            logger.info("TUI v2 application shutting down...")
            return 0
            
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
            return 0
        except Exception as e:
            logger.exception(f"Unexpected error in main application: {e}")
            return 1
        finally:
            # Perform cleanup while logging is still suppressed
            await self.cleanup()
            # Restore console logging after cleanup to avoid stray lines in terminal
            try:
                if self._removed_stream_handlers:
                    root_logger = logging.getLogger()
                    for h in self._removed_stream_handlers:
                        root_logger.addHandler(h)
                    self._removed_stream_handlers.clear()
            except Exception:
                pass
            # Ensure we end at a clean state on both stdout and stderr
            # Clear current line, reset cursor to column 0, and add a newline so
            # any subsequent prints (including CLI tips) appear cleanly.
            try:
                import sys as _sys, os as _os
                if _os.getenv("AGENTS_TUI_V2_DEBUG", "0") != "1":
                    for _stream in (_sys.stdout, _sys.stderr):
                        try:
                            # Reset graphic attributes, clear line, move to col 0, newline
                            _stream.write("\x1b[0m\r\x1b[2K\n")
                            _stream.flush()
                        except Exception:
                            continue
            except Exception:
                pass
            # Re-enable logging to previous level
            try:
                logging.disable(getattr(self, '_prev_logging_disable', 0))
            except Exception:
                pass

    async def _run_minimal_input_mode(self) -> int:
        """Run a minimal isolated input loop to validate typing and exit handling."""
        try:
            # Early debug file for step tracing
            import os as _os, sys as _sys
            dbg_enabled = _os.getenv("AGENTS_TUI_V2_DEBUG", "0") == "1"
            dbg_file = None
            if dbg_enabled:
                try:
                    dbg_file = open('/tmp/tui_v2_minimal.log', 'a', buffering=1)
                    dbg_file.write("\n--- tui_v2_minimal init ---\n")
                except Exception:
                    dbg_file = None

            def _dbg(msg: str):
                if dbg_file:
                    try:
                        dbg_file.write(msg + "\n")
                    except Exception:
                        pass

            _dbg("step: creating terminal manager")
            self.terminal_manager = create_terminal_manager()
            if not await self.terminal_manager.initialize():
                try:
                    if _os.getenv("AGENTS_TUI_V2_DEBUG", "0") == "1":
                        _sys.stderr.write("[tui-v2-minimal] step: terminal init failed\n")
                        _sys.stderr.flush()
                except Exception:
                    pass
                _dbg("step: terminal init failed")
                print("❌ Terminal init failed")
                return 1
            _dbg("step: terminal init ok")
            self.display_renderer = DisplayRenderer(self.terminal_manager)
            if not await self.display_renderer.initialize():
                try:
                    if _os.getenv("AGENTS_TUI_V2_DEBUG", "0") == "1":
                        _sys.stderr.write("[tui-v2-minimal] step: display init failed\n")
                        _sys.stderr.flush()
                except Exception:
                    pass
                _dbg("step: display init failed")
                print("❌ Display init failed")
                return 1
            _dbg("step: display init ok")
            caps = self.terminal_manager.detect_capabilities()
            width, height = caps.width, caps.height
            last_size = {"w": width, "h": height}
            # Define regions: status (top), history (above input), input (bottom)
            status_y = 0
            try:
                input_lines = max(1, int(os.getenv("AGENTS_TUI_V2_INPUT_LINES", "3")))
            except Exception:
                input_lines = 3
            input_y = max(0, height - input_lines)
            usable_h = max(0, input_y - status_y)
            # Give history a useful height (10–30 lines depending on terminal)
            history_h = max(3, usable_h - 1)
            history_y = max(status_y + 1, input_y - history_h)
            _dbg(f"step: define regions status_y={status_y} history_y={history_y} input_y={input_y} width={width} history_h={history_h} input_lines={input_lines}")
            self.display_renderer.define_region("status", 0, status_y, width, 1)
            self.display_renderer.define_region("history", 0, history_y, width, history_h)
            self.display_renderer.define_region("input", 0, input_y, width, input_lines)
            
            _dbg("step: prepare prompt & caret")
            try:
                prompt = "> "
                from datetime import datetime
                state = {"text": "", "cursor": 0, "caret_visible": True}
                # Caret configuration: default to full block, overridable via env
                caret_char = os.getenv("AGENTS_TUI_V2_CARET_CHAR", "█") or "█"
                messages: list[dict] = []  # {role: 'user'|'system', text: str, time: 'HH:MM'}
                scroll_offset: int = 0  # 0 = bottom; >0 = scrolled up by that many lines
                status_text: str = "Ready | /help | /quit"
                _dbg("step: prompt & caret ok")
            except Exception as e:
                _dbg(f"step: prompt & caret failed: {e}")
                try:
                    if dbg_enabled:
                        _sys.stderr.write("[tui-v2-minimal] step: prompt & caret failed\n")
                        _sys.stderr.flush()
                except Exception:
                    pass
                return 1
            # Spinner state for outbound requests
            spinner_task: Optional[asyncio.Task] = None
            
            # Backspace debouncing to prevent multi-delete on single tap
            import time as _time
            last_bs_time: float = 0.0
            bs_series_start: float = 0.0
            BS_TAP_WINDOW = 0.12   # seconds; ignore repeats inside this window
            BS_SERIES_GRACE = 0.40 # seconds; allow repeats after this (long press)
            spinner_active: bool = False
            spinner_frames = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
            spinner_idx = 0
            input_spinner_char: str = ''
            show_input_spinner: bool = False
            # Minimum spinner visibility (ms) so it’s perceptible
            try:
                spinner_min_ms = max(0, int(os.getenv("AGENTS_TUI_V2_SPINNER_MIN_MS", "500")))
            except Exception:
                spinner_min_ms = 500
            spinner_started_at = 0.0
            # Mouse wheel scroll granularity (in lines) – default: 1 line per tick
            try:
                wheel_scroll_lines = max(1, int(os.getenv("AGENTS_TUI_V2_WHEEL_LINES", "1")))
            except Exception:
                wheel_scroll_lines = 1
            
            # Optional backend integration (ConversationManager) for real responses
            conv_mgr = None
            try:
                import os as _os
                # Default backend ON in minimal mode so chat sends to LLM
                if _os.getenv("AGENTS_TUI_V2_BACKEND", "1") == "1":
                    from ...conversation.conversation import ConversationManager
                    conv_mgr = ConversationManager()
                    # Background prewarm to reduce first-response latency
                    if _os.getenv("AGENTS_TUI_V2_BACKEND_PREWARM", "1") == "1":
                        async def _prewarm_backend():
                            try:
                                # Suppress any prints/logs during warmup
                                import contextlib, io, sys as _sys
                                devnull = io.StringIO()
                                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                                    # Warm up model capability detection (may touch network/local server)
                                    await conv_mgr.llm_client.get_model_capabilities()
                            except Exception:
                                pass
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(_prewarm_backend())
                        except Exception:
                            pass
            except Exception:
                conv_mgr = None

            import textwrap

            def _render_markdown_lines(raw_text: str, width: int, indent_prefix: str = '') -> list[str]:
                BOLD = "\x1b[1m"; ITALIC = "\x1b[3m"; CYAN = "\x1b[36m"; YELLOW = "\x1b[33m"; MAGENTA = "\x1b[35m"; RESET = "\x1b[0m"
                import re as _re, textwrap as _tw
                def style_inline(s: str) -> str:
                    s = _re.sub(r"`([^`]+)`", lambda m: f"{CYAN}{m.group(1)}{RESET}", s)
                    s = _re.sub(r"\*\*(.+?)\*\*", lambda m: f"{BOLD}{m.group(1)}{RESET}", s)
                    s = _re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", lambda m: f"{ITALIC}{m.group(1)}{RESET}", s)
                    s = _re.sub(r"_(?!\s)(.+?)(?<!\s)_", lambda m: f"{ITALIC}{m.group(1)}{RESET}", s)
                    return s
                out = []
                in_code = False
                for raw in (raw_text.split('\n') if raw_text else ['']):
                    if raw.strip().startswith('```'):
                        in_code = not in_code; continue
                    bullet = ''
                    line = raw
                    if not in_code and _re.match(r"^\s*[-*]\s+", line):
                        line = _re.sub(r"^\s*[-*]\s+", '', line)
                        bullet = f"{MAGENTA}•{RESET} "
                    if in_code:
                        seg = f"{CYAN}{line}{RESET}"
                        out.append((indent_prefix + seg)[:width])
                    else:
                        m = _re.match(r"^(\s*#+)\s*(.+)$", line)
                        if m:
                            line = f"{BOLD}{YELLOW}{m.group(2)}{RESET}"
                        else:
                            line = style_inline(line)
                        wrap = _tw.wrap((bullet + line) if bullet else line, width=max(1, width - len(indent_prefix))) or ['']
                        for i, seg in enumerate(wrap):
                            prefix = indent_prefix if i > 0 else indent_prefix
                            out.append((prefix + seg)[:width])
                return out

            # Simple modal panel state for /settings and /agents
            panel = {"active": False, "type": "", "index": 0}
            settings = {"Theme": "auto", "Provider": "ollama-turbo", "Mouse Scroll": "off"}
            agents_cfg = [
                {"name": "codex", "enabled": True},
                {"name": "claude", "enabled": True},
                {"name": "ollama-turbo", "enabled": True},
                {"name": "ollama", "enabled": True},
            ]

            def _render_panel():
                nonlocal status_text
                if not panel["active"]:
                    return
                head = f" Settings " if panel["type"] == "settings" else f" Agents "
                title = f"[ {head.strip()} ]  (Esc to close, ↑/↓ select, ←/→ change, s=save)"
                lines = [title]
                if panel["type"] == "settings":
                    items = [
                        ("Theme", settings["Theme"], ["auto", "dark", "light"]),
                        ("Provider", settings["Provider"], ["ollama-turbo", "ollama"]),
                        ("Mouse Scroll", settings["Mouse Scroll"], ["off", "on"]),
                    ]
                    for idx, (k, v, choices) in enumerate(items):
                        cursor = ">" if idx == panel["index"] else " "
                        lines.append(f"{cursor} {k}: {v}")
                else:
                    for idx, a in enumerate(agents_cfg):
                        cursor = ">" if idx == panel["index"] else " "
                        mark = "[x]" if a.get("enabled") else "[ ]"
                        lines.append(f"{cursor} {mark} {a.get('name')}")
                # Fit to history region height and width
                padded = []
                for ln in lines[:history_h]:
                    if len(ln) < width:
                        ln = ln + (" " * (width - len(ln)))
                    else:
                        ln = ln[:width]
                    padded.append(ln)
                while len(padded) < history_h:
                    padded.append(" " * width)
                self.display_renderer.update_region("history", "\n".join(padded), force=True)
                status_text = ("Editing settings | ←/→ change | s save | Esc close" if panel["type"] == "settings" else "Editing agents | Space/→ toggle | s save | Esc close")

            def _panel_cycle(delta: int):
                if panel["type"] == "settings":
                    idx = panel["index"]
                    keys = ["Theme", "Provider", "Mouse Scroll"]
                    key = keys[idx]
                    choices = {
                        "Theme": ["auto", "dark", "light"],
                        "Provider": ["ollama-turbo", "ollama"],
                        "Mouse Scroll": ["off", "on"],
                    }[key]
                    cur = settings[key]
                    try:
                        i = choices.index(cur)
                    except ValueError:
                        i = 0
                    i = (i + delta) % len(choices)
                    settings[key] = choices[i]
                else:
                    idx = panel["index"]
                    if 0 <= idx < len(agents_cfg):
                        agents_cfg[idx]["enabled"] = not agents_cfg[idx].get("enabled", True)

            def _save_settings():
                # Persist to ~/.agentsmcp/config.json
                try:
                    import json
                    cfg_dir = Path.home() / ".agentsmcp"
                    cfg_dir.mkdir(parents=True, exist_ok=True)
                    cfg_path = cfg_dir / "config.json"
                    data = {}
                    if cfg_path.exists():
                        try:
                            data = json.loads(cfg_path.read_text())
                        except Exception:
                            data = {}
                    # Map settings to config fields
                    if settings["Provider"]:
                        data["provider"] = settings["Provider"]
                    # Theme is handled by CLI, keep for future
                    data.setdefault("api_keys", data.get("api_keys", {}))
                    cfg_path.write_text(json.dumps(data, indent=2))
                except Exception:
                    pass

            def _save_agents():
                try:
                    import json
                    cfg_dir = Path.home() / ".agentsmcp"
                    cfg_dir.mkdir(parents=True, exist_ok=True)
                    agents_path = cfg_dir / "agents.json"
                    agents_path.write_text(json.dumps(agents_cfg, indent=2))
                except Exception:
                    pass

            def _open_panel(kind: str):
                panel["active"] = True
                panel["type"] = kind
                panel["index"] = 0
                _render_panel()

            def _close_panel():
                panel["active"] = False
                # Repaint history to resume chat view
                render()

            def _apply_ansi_markdown(text: str) -> str:
                """Apply ANSI color codes to markdown-style text using the advanced processor."""
                if not text:
                    return text
                
                # Use the advanced ANSI markdown processor if available
                if self.ansi_processor:
                    try:
                        return self.ansi_processor.process_text(text)
                    except Exception as e:
                        logger.warning(f"ANSI processor failed, using fallback: {e}")
                
                # Fallback to basic processing
                import re
                
                # ANSI color codes
                BOLD = "\x1b[1m"
                ITALIC = "\x1b[3m"
                CYAN = "\x1b[36m"
                YELLOW = "\x1b[33m"
                MAGENTA = "\x1b[35m"
                GREEN = "\x1b[32m"
                RED = "\x1b[31m"
                RESET = "\x1b[0m"
                
                # Apply markdown-style formatting
                # Code blocks (backticks)
                text = re.sub(r"`([^`]+)`", rf"{CYAN}\1{RESET}", text)
                
                # Bold text (**text**)
                text = re.sub(r"\*\*(.+?)\*\*", rf"{BOLD}\1{RESET}", text)
                
                # Italic text (*text* but not **text**)
                text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", rf"{ITALIC}\1{RESET}", text)
                
                # Headers (# text)
                text = re.sub(r"^(\s*#+)\s*(.+)$", rf"\1 {BOLD}{YELLOW}\2{RESET}", text, flags=re.MULTILINE)
                
                # List items (- or * at start of line)
                text = re.sub(r"^(\s*[-*])\s+", rf"\1 {MAGENTA}•{RESET} ", text, flags=re.MULTILINE)
                
                return text

            def _format_history_lines() -> list[str]:
                lines: list[str] = []
                for m in messages:
                    ts = m.get('time') or ''
                    role = m.get('role') or 'user'
                    if role == 'user':
                        who = 'You'
                    elif role == 'assistant':
                        who = 'Assistant'
                    else:
                        who = 'System'
                    raw_text = (m.get('text') or '').replace('\r\n', '\n').replace('\r', '\n')
                    text = _apply_ansi_markdown(raw_text) if role == 'assistant' else raw_text
                    prefix = f"[{ts}] {who}: " if ts else f"{who}: "
                    avail = max(1, width - len(prefix))
                    parts = text.split('\n') if text else ['']
                    first_line = True
                    for part in parts:
                        wrapped = textwrap.wrap(part, width=avail) or ['']
                        for i, seg in enumerate(wrapped):
                            if first_line and i == 0:
                                line = prefix + seg
                            else:
                                line = (' ' * len(prefix)) + seg
                            lines.append(line[:width])
                            first_line = False
                return lines

            def _total_history_lines() -> int:
                try:
                    # Build same formatted list as in render()
                    formatted = []
                    for m in messages:
                        ts = m.get('time') or ''
                        role = m.get('role') or 'user'
                        who = 'You' if role == 'user' else ('Assistant' if role == 'assistant' else 'System')
                        prefix = (f"[{ts}] {who}: " if ts else f"{who}: ")
                        content = (m.get('text') or '').replace('\r\n','\n').replace('\r','\n')
                        md_lines = _render_markdown_lines(content, width, '')
                        if not md_lines:
                            md_lines = ['']
                        for idx2, ln in enumerate(md_lines):
                            if idx2 == 0:
                                fl = (prefix + ln)[:width]
                            else:
                                fl = (" " * len(prefix) + ln)[:width]
                            formatted.append(fl)
                    return len(formatted)
                except Exception:
                    return len(messages)

            # Add render debouncing to reduce race conditions
            _render_pending = {"flag": False}
            _render_last_call = {"time": 0}
            
            def render():
                import time
                # Simple debouncing: avoid rapid consecutive render calls
                current_time = time.time()
                if current_time - _render_last_call["time"] < 0.001:  # 1ms minimum interval
                    if not _render_pending["flag"]:
                        _render_pending["flag"] = True
                        # Schedule deferred render to avoid overwhelming the display
                        try:
                            loop = asyncio.get_running_loop()
                            loop.call_later(0.002, lambda: (_render_pending.update({"flag": False}), _do_render()))
                        except Exception:
                            _do_render()  # Fallback to immediate render
                    return
                
                _render_last_call["time"] = current_time
                _do_render()
            
            def _do_render():
                # Handle terminal resize proactively
                try:
                    caps2 = self.terminal_manager.detect_capabilities()
                    if caps2.width != last_size["w"] or caps2.height != last_size["h"]:
                        last_size["w"], last_size["h"] = caps2.width, caps2.height
                        nonlocal width, height, input_lines, input_y, usable_h, history_h, history_y
                        width, height = caps2.width, caps2.height
                        input_y = max(0, height - input_lines)
                        usable_h = max(0, input_y - status_y)
                        history_h = max(3, usable_h - 1)
                        history_y = max(status_y + 1, input_y - history_h)
                        self.display_renderer.define_region("status", 0, status_y, width, 1)
                        self.display_renderer.define_region("history", 0, history_y, width, history_h)
                        self.display_renderer.define_region("input", 0, input_y, width, input_lines)
                except Exception:
                    pass
                txt = state["text"]
                pos = state["cursor"]
                caret = caret_char if state.get("caret_visible", True) else " "
                prefix_spinner = (input_spinner_char + ' ') if show_input_spinner and input_spinner_char else ''
                # Insert caret into full text at cursor
                full_with_caret = txt[:pos] + caret + txt[pos:]
                raw_lines = full_with_caret.split('\n') or [""]
                # Take last input_lines
                to_show = raw_lines[-input_lines:]
                visual = []
                for idx, l in enumerate(to_show):
                    # First displayed line gets prompt, others continuation
                    is_first = (idx == max(0, len(to_show) - input_lines))
                    line_text = f"{prefix_spinner}{prompt}{l}" if is_first else f"{prefix_spinner}... {l}"
                    if len(line_text) < width:
                        line_text = line_text + (" " * (width - len(line_text)))
                    visual.append(line_text[:width])
                while len(visual) < input_lines:
                    visual.insert(0, " " * width)
                self.display_renderer.update_region("input", "\n".join(visual), force=True)
                # Render history with scroll offset, padded (or panel if active)
                if panel["active"]:
                    _render_panel()
                elif messages is not None:
                    # Build all logical lines with Markdown++ styling
                    formatted = []
                    for m in messages:
                        ts = m.get('time') or ''
                        role = m.get('role') or 'user'
                        who = 'You' if role == 'user' else ('Assistant' if role == 'assistant' else 'System')
                        prefix = (f"[{ts}] {who}: " if ts else f"{who}: ")
                        content = (m.get('text') or '').replace('\r\n','\n').replace('\r','\n')
                        md_lines = _render_markdown_lines(content, width, '')
                        if not md_lines:
                            md_lines = ['']
                        # Attach prefix to first, indent rest
                        for idx2, ln in enumerate(md_lines):
                            if idx2 == 0:
                                fl = (prefix + ln)[:width]
                            else:
                                fl = (" " * len(prefix) + ln)[:width]
                            formatted.append(fl)
                    all_lines = formatted
                    total_lines = len(all_lines)
                    # Determine visible window from bottom with scroll_offset (in lines)
                    if total_lines <= history_h:
                        window = all_lines[:]
                    else:
                        start = max(0, total_lines - history_h - scroll_offset)
                        end = start + history_h
                        if end > total_lines:
                            end = total_lines
                            start = max(0, end - history_h)
                        window = all_lines[start:end]
                    # pad to width and ensure exactly history_h rows bottom-aligned
                    padded = [l + (" " * (width - len(l))) for l in window]
                    if len(padded) < history_h:
                        padded = ([" " * width] * (history_h - len(padded))) + padded
                    self.display_renderer.update_region("history", "\n".join(padded), force=True)
                # Render status line (low-frequency updates handled by hashing in renderer)
                st = status_text
                if len(st) < width:
                    st = st + (" " * (width - len(st)))
                self.display_renderer.update_region("status", st[:width], force=True)

            async def _spinner_loop():
                nonlocal spinner_idx, status_text
                try:
                    while spinner_active:
                        spinner_idx = (spinner_idx + 1) % len(spinner_frames)
                        ch = spinner_frames[spinner_idx]
                        status_text = f"{ch} Sending… | /help | /quit"
                        # update input-line spinner too
                        nonlocal input_spinner_char
                        input_spinner_char = ch
                        render()
                        await asyncio.sleep(0.1)
                except Exception:
                    pass

            def on_char(evt):
                ch = evt.character
                if not ch or len(ch) != 1:
                    return
                txt = state["text"]
                pos = state["cursor"]
                state["text"] = txt[:pos] + ch + txt[pos:]
                state["cursor"] = pos + 1
                state["caret_visible"] = True
                # Do not spam status; keep it as-is on typing
                render()
                # Optional quick-quit (disabled by default); enable via AGENTS_TUI_V2_QUICK_QUIT=1
                try:
                    import os as _os, asyncio as _asyncio
                    if _os.getenv("AGENTS_TUI_V2_QUICK_QUIT") == "1":
                        buff = state["text"].strip()
                        if buff in ("/quit", "/exit"):
                            _asyncio.create_task(self.shutdown())
                except Exception:
                    pass

            def on_backspace(evt):
                txt = state["text"]
                pos = state["cursor"]
                if pos > 0:
                    state["text"] = txt[:pos-1] + txt[pos:]
                    state["cursor"] = pos - 1
                state["caret_visible"] = True
                # Keep status unchanged on backspace
                render()

            async def on_enter(evt):
                text = state["text"].strip()
                if text in ("/quit", "/exit"):
                    await self.shutdown()
                    return
                # Handle simple commands in minimal mode
                nonlocal scroll_offset, status_text
                if text == "/clear":
                    messages.clear()
                    scroll_offset = 0
                    status_text = "Cleared | /help | /quit"
                elif text == "/help":
                    help_md = """# AgentsMCP TUI v2 Help

- **Submit**: press Enter
- **Newline**: press Ctrl+J (Shift+Enter depends on terminal)
- **Scroll**: Mouse wheel, PageUp/PageDown (line by line)
- **Cancel input**: Ctrl+C or `/cancel`
- **Exit**: `/quit` or Ctrl+D on empty line

## Commands
- `/help` — show this help
- `/clear` — clear chat history
- `/env` — show useful environment flags
- `/inputlines N` — set visible input height to N lines
- `/cancel` — clear current input
- `/quit` — exit the TUI

## Environment Flags
`AGENTS_TUI_V2_INPUT_LINES` (default 3), `AGENTS_TUI_V2_WHEEL_LINES` (default 1), `AGENTS_TUI_V2_SPINNER_MIN_MS` (default 500), `AGENTS_TUI_V2_POLL_MS` (default 100), `AGENTS_TUI_V2_CARET_CHAR` (default █)
"""
                    messages.append({
                        'role': 'system',
                        'text': help_md,
                        'time': datetime.now().strftime('%H:%M')
                    })
                    status_text = "Help | /help | /quit"
                elif text.startswith("/inputlines") or text.startswith("/height") or text.startswith("/il"):
                    # Adjust the visible input height at runtime
                    try:
                        parts = text.split()
                        if len(parts) >= 2:
                            new_lines = max(1, min(height-3, int(parts[1])))
                            # Recompute geometry
                            nonlocal input_lines, input_y, usable_h, history_h, history_y
                            input_lines = new_lines
                            input_y = max(0, height - input_lines)
                            usable_h = max(0, input_y - status_y)
                            history_h = max(3, usable_h - 1)
                            history_y = max(status_y + 1, input_y - history_h)
                            # Redefine regions
                            self.display_renderer.define_region("history", 0, history_y, width, history_h)
                            self.display_renderer.define_region("input", 0, input_y, width, input_lines)
                            messages.append({'role': 'system', 'text': f"Input height set to {new_lines} lines.", 'time': datetime.now().strftime('%H:%M')})
                            status_text = "Input height updated | /help | /quit"
                        else:
                            messages.append({'role': 'system', 'text': "Usage: /inputlines <N>", 'time': datetime.now().strftime('%H:%M')})
                            status_text = "Usage shown | /help | /quit"
                    except Exception:
                        messages.append({'role': 'system', 'text': "Failed to set input height.", 'time': datetime.now().strftime('%H:%M')})
                        status_text = "Error | /help | /quit"
                elif text == "/settings":
                    _open_panel("settings")
                    state["text"] = ""
                    state["cursor"] = 0
                    return
                elif text == "/agents":
                    _open_panel("agents")
                    state["text"] = ""
                    state["cursor"] = 0
                    return
                elif text.startswith("/"):
                    messages.append({
                        'role': 'system',
                        'text': f"Unknown command: {text}",
                        'time': datetime.now().strftime('%H:%M')
                    })
                    status_text = "Unknown command | /help | /quit"
                elif text == "/env":
                    env_info = [
                        "Env flags:",
                        "AGENTS_TUI_V2_MINIMAL (1=raw input mode)",
                        "AGENTS_TUI_V2_FORCE_RAW_INPUT (1=force raw input in full v2)",
                        "AGENTS_TUI_V2_INPUT_LINES (visible input lines, default 3)",
                        "AGENTS_TUI_V2_WHEEL_LINES (mouse wheel step, default 1)",
                        "AGENTS_TUI_V2_SPINNER_MIN_MS (min spinner ms, default 500)",
                        "AGENTS_TUI_V2_POLL_MS (raw reader poll ms, default 100)",
                        "AGENTS_TUI_V2_CARET_CHAR (input caret, default █)",
                    ]
                    messages.append({'role': 'system', 'text': "\n".join(env_info), 'time': datetime.now().strftime('%H:%M')})
                    status_text = "Env help | /help | /quit"
                elif text == "/cancel":
                    state["text"] = ""
                    state["cursor"] = 0
                    status_text = "Canceled | /help | /quit"
                elif text:
                    # Append user message
                    now_ts = datetime.now().strftime('%H:%M')
                    messages.append({'role': 'user', 'text': text, 'time': now_ts})
                    status_text = "⠋ Sending… | /help | /quit"
                    # Start spinner
                    nonlocal spinner_task, spinner_active, show_input_spinner, spinner_started_at
                    spinner_active = True
                    show_input_spinner = True
                    try:
                        import time as _time
                        spinner_started_at = _time.monotonic()
                    except Exception:
                        spinner_started_at = 0.0
                    try:
                        spinner_task = _asyncio.create_task(_spinner_loop())
                    except Exception:
                        spinner_task = None
                    # If backend is enabled, process via ConversationManager, else echo stub
                    async def _assistant_backend(user_text: str, user_ts: str):
                        nonlocal status_text, spinner_active, spinner_task, show_input_spinner, input_spinner_char
                        try:
                            if conv_mgr is None:
                                # Fallback: echo
                                await _asyncio.sleep(0.2)
                                reply = f"(echo) {user_text}"
                            else:
                                # Suppress stray prints/logs from backend
                                import contextlib, io, sys as _sys
                                devnull = io.StringIO()
                                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                                    reply = await conv_mgr.process_input(user_text)
                            messages.append({'role': 'assistant', 'text': reply, 'time': datetime.now().strftime('%H:%M')})
                            if len(messages) > 500:
                                del messages[:len(messages)-500]
                            status_text = "Assistant replied | /help | /quit"
                            render()
                        except Exception:
                            # Show a friendly error in history
                            messages.append({'role': 'system', 'text': 'Assistant failed to respond.', 'time': datetime.now().strftime('%H:%M')})
                            status_text = "Error | /help | /quit"
                            render()
                        finally:
                            # Stop spinner
                            try:
                                import time as _time
                                if spinner_started_at > 0:
                                    elapsed_ms = int((_time.monotonic() - spinner_started_at) * 1000)
                                    if elapsed_ms < spinner_min_ms:
                                        await _asyncio.sleep((spinner_min_ms - elapsed_ms)/1000.0)
                            except Exception:
                                pass
                            spinner_active = False
                            try:
                                if spinner_task:
                                    spinner_task.cancel()
                            except Exception:
                                pass
                            show_input_spinner = False
                            input_spinner_char = ''
                    _asyncio.create_task(_assistant_backend(text, now_ts))
                # Keep history reasonable
                if len(messages) > 500:
                    del messages[:len(messages)-500]
                # Reset scroll to bottom after submit
                scroll_offset = 0
                # Clear line after submit
                state["text"] = ""
                state["cursor"] = 0
                state["caret_visible"] = True
                render()

            # Insert a literal newline at the cursor (used for Ctrl+J in minimal mode)
            def on_newline():
                txt = state["text"]
                pos = state["cursor"]
                state["text"] = txt[:pos] + "\n" + txt[pos:]
                state["cursor"] = pos + 1
                state["caret_visible"] = True
                render()

            # Optionally enable mouse reporting (Xterm SGR mode) for wheel events
            try:
                if os.getenv("AGENTS_TUI_V2_MOUSE", "0") == "1":
                    _dbg("step: enable mouse start")
                    out = self.display_renderer._output  # sys.stdout
                    out.write('\033[?1000h')  # Enable basic mouse
                    out.write('\033[?1006h')  # Enable SGR extended mode
                    out.flush()
                    _dbg("step: enable mouse ok")
            except Exception as e:
                _dbg(f"step: enable mouse failed: {e}")

            # Paint initial line
            _dbg("step: initial render start")
            try:
                render()
                _dbg("step: initial render ok")
            except Exception as e:
                _dbg(f"step: initial render failed: {e}")
                try:
                    if dbg_enabled:
                        _sys.stderr.write("[tui-v2-minimal] step: initial render failed\n")
                        _sys.stderr.flush()
                except Exception:
                    pass
                return 1

            # Raw /dev/tty reader in a thread for reliable key capture
            import threading, termios, tty, select, asyncio as _asyncio
            debug_enabled = os.getenv("AGENTS_TUI_V2_DEBUG", "0") == "1"
            try:
                poll_ms = int(os.getenv("AGENTS_TUI_V2_POLL_MS", "100"))
                if poll_ms < 5:
                    poll_ms = 5
            except Exception:
                poll_ms = 100
            debug_file = None
            if debug_enabled:
                try:
                    debug_file = open('/tmp/tui_v2_minimal.log', 'a', buffering=1)
                    debug_file.write("\n--- tui_v2_minimal start ---\n")
                except Exception:
                    debug_file = None

            stop_flag = {"stop": False}

            # Capture the running loop to schedule thread-safe callbacks
            try:
                loop = _asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            def log_debug(msg: str):
                if debug_file:
                    try:
                        debug_file.write(msg + "\n")
                    except Exception:
                        pass

            def reader_thread():
                fd = None
                old = None
                try:
                    fd = os.open('/dev/tty', os.O_RDONLY)
                    old = termios.tcgetattr(fd)
                    tty.setraw(fd)
                    while not stop_flag["stop"]:
                        r, _, _ = select.select([fd], [], [], poll_ms/1000.0)
                        if not r:
                            continue
                        data = os.read(fd, 64)
                        if not data:
                            continue
                        i = 1  # we'll process from index 0 below; set start to 0
                        i = 0
                        n = len(data)
                        while i < n:
                            b = data[i]
                            if debug_enabled:
                                try:
                                    log_debug(f"byte[{i}]: {b} char: {bytes([b]).decode('utf-8', errors='ignore')!r} buf='{state['text']}'")
                                except Exception:
                                    pass
                            # Ctrl+C / Ctrl+D
                            if b in (3, 4):
                                if loop:
                                    loop.call_soon_threadsafe(lambda: _asyncio.create_task(self.shutdown()))
                                else:
                                    os._exit(0)
                                i += 1
                                continue
                            # Backspace with immediate processing
                            if b in (8, 127):
                                # Debounce backspace to avoid multi-delete on single tap
                                now = _time.monotonic()
                                start_new_series = (now - last_bs_time) > BS_TAP_WINDOW
                                if start_new_series:
                                    bs_series_start = now
                                within_tap_window = (now - last_bs_time) < BS_TAP_WINDOW
                                long_press = (now - bs_series_start) >= BS_SERIES_GRACE
                                # Only process if not a rapid repeat within tap window, unless long press
                                if (not within_tap_window) or long_press:
                                    last_bs_time = now
                                    # IMMEDIATE BACKSPACE PROCESSING to fix one-character-behind lag
                                    if state["text"] and state["cursor"] > 0:
                                        txt = state["text"]
                                        pos = state["cursor"]
                                        state["text"] = txt[:pos-1] + txt[pos:]
                                        state["cursor"] = pos - 1
                                        state["caret_visible"] = True
                                        
                                        # Immediate rendering for instant backspace feedback
                                        try:
                                            caret = caret_char if state.get("caret_visible", True) else " "
                                            pos = state["cursor"]
                                            full_with_caret = state["text"][:pos] + caret + state["text"][pos:]
                                            
                                            # Build visual representation
                                            visual = []
                                            for line_idx, line in enumerate(full_with_caret.split('\n')):
                                                if line_idx == 0:
                                                    visual.append(f"> {line}")
                                                else:
                                                    visual.append(f"  {line}")
                                            
                                            # Direct region update for instant backspace feedback
                                            self.display_renderer.update_region("input", "\n".join(visual), force=True)
                                        except Exception:
                                            pass  # Fail silently, async handler will eventually update
                                    
                                    # Also schedule async handler for full processing
                                    if loop:
                                        loop.call_soon_threadsafe(on_backspace, None)
                                i += 1
                                continue
                            # Enter / Ctrl+J distinction in raw mode
                            # CR (13) = Enter/Return → submit; LF (10) = Ctrl+J → insert newline
                            if b == 13:
                                # If CRLF paste, treat as newline and consume both
                                if i + 1 < n and data[i+1] == 10:
                                    if loop:
                                        loop.call_soon_threadsafe(on_newline)
                                    i += 2
                                    continue
                                if loop:
                                    loop.call_soon_threadsafe(lambda: _asyncio.create_task(on_enter(None)))
                                i += 1
                                continue
                            if b == 10:
                                if loop:
                                    loop.call_soon_threadsafe(on_newline)
                                i += 1
                                continue
                            # ESC sequences
                            if b == 27:
                                # Attempt to parse CSI sequence from current buffer
                                seq = b''
                                j = i + 1
                                if j < n and data[j] == ord('['):
                                    j += 1
                                    while j < n:
                                        seq += bytes([data[j]])
                                        # End of sequence on letter or '~'
                                        if (65 <= data[j] <= 90) or (97 <= data[j] <= 122) or data[j] == ord('~'):
                                            break
                                        j += 1
                                    seq_str = seq.decode('utf-8', errors='ignore')
                                    handled = False
                                    # Handle SGR mouse: <b;x;yM/m
                                    if '<' in seq_str and (seq_str.endswith('M') or seq_str.endswith('m')):
                                        import re as _re
                                        m = _re.match(r"<([0-9]+);([0-9]+);([0-9]+)[Mm]", seq_str)
                                        if m:
                                            try:
                                                btn = int(m.group(1))
                                            except Exception:
                                                btn = -1
                                            if btn == 64:  # Wheel up
                                                def _wheel_up():
                                                    nonlocal scroll_offset, status_text
                                                    total = _total_history_lines()
                                                    if total > 0:
                                                        scroll_offset = min(scroll_offset + wheel_scroll_lines, max(0, total - history_h))
                                                        status_text = f"Viewing older (offset {scroll_offset}) | /help | /quit"
                                                        render()
                                                if loop:
                                                    loop.call_soon_threadsafe(_wheel_up)
                                                handled = True
                                            elif btn == 65:  # Wheel down
                                                def _wheel_down():
                                                    nonlocal scroll_offset, status_text
                                                    if scroll_offset > 0:
                                                        scroll_offset = max(0, scroll_offset - wheel_scroll_lines)
                                                        status_text = (f"Viewing older (offset {scroll_offset}) | /help | /quit" if scroll_offset > 0 else "Ready | /help | /quit")
                                                        render()
                                                if loop:
                                                    loop.call_soon_threadsafe(_wheel_down)
                                                handled = True

                                    if not handled and '5~' in seq_str:  # Page Up
                                        def _pgup():
                                            nonlocal scroll_offset, status_text
                                            total = _total_history_lines()
                                            if total > 0:
                                                scroll_offset = min(scroll_offset + history_h, max(0, total - history_h))
                                                status_text = f"Viewing older (offset {scroll_offset}) | /help | /quit"
                                                render()
                                        if loop:
                                            loop.call_soon_threadsafe(_pgup)
                                        handled = True
                                    elif not handled and '6~' in seq_str:  # Page Down
                                        def _pgdn():
                                            nonlocal scroll_offset, status_text
                                            if scroll_offset > 0:
                                                scroll_offset = max(0, scroll_offset - history_h)
                                                status_text = (f"Viewing older (offset {scroll_offset}) | /help | /quit"
                                                               if scroll_offset > 0 else "Ready | /help | /quit")
                                                render()
                                        if loop:
                                            loop.call_soon_threadsafe(_pgdn)
                                        handled = True
                                    elif not handled and 'A' in seq_str:  # Up
                                        def _up():
                                            nonlocal scroll_offset, status_text
                                            if panel["active"]:
                                                if panel["index"] > 0:
                                                    panel["index"] -= 1
                                                _render_panel()
                                            else:
                                                total = _total_history_lines()
                                                if total > 0:
                                                    scroll_offset = min(scroll_offset + 1, max(0, total - history_h))
                                                    status_text = f"Viewing older (offset {scroll_offset}) | /help | /quit"
                                                    render()
                                        if loop:
                                            loop.call_soon_threadsafe(_up)
                                        handled = True
                                    elif not handled and 'B' in seq_str:  # Down
                                        def _down():
                                            nonlocal scroll_offset, status_text
                                            if panel["active"]:
                                                max_idx = 2 if panel["type"] == "settings" else max(0, len(agents_cfg) - 1)
                                                if panel["index"] < max_idx:
                                                    panel["index"] += 1
                                                _render_panel()
                                            else:
                                                if scroll_offset > 0:
                                                    scroll_offset = max(0, scroll_offset - 1)
                                                    status_text = (f"Viewing older (offset {scroll_offset}) | /help | /quit"
                                                                   if scroll_offset > 0 else "Ready | /help | /quit")
                                                    render()
                                        if loop:
                                            loop.call_soon_threadsafe(_down)
                                        handled = True
                                    elif not handled and 'C' in seq_str:  # Right
                                        def _right():
                                            if panel["active"]:
                                                _panel_cycle(+1)
                                                _render_panel()
                                        if loop:
                                            loop.call_soon_threadsafe(_right)
                                        handled = True
                                    elif not handled and 'D' in seq_str:  # Left
                                        def _left():
                                            if panel["active"]:
                                                _panel_cycle(-1)
                                                _render_panel()
                                        if loop:
                                            loop.call_soon_threadsafe(_left)
                                        handled = True
                                    # Advance index past ESC [ ... tail
                                    i = j + 1
                                    if handled:
                                        continue
                                # ESC alone: close panel or clear input
                                def _clear():
                                    if panel["active"]:
                                        _close_panel()
                                    else:
                                        state["text"] = ""
                                        state["cursor"] = 0
                                        render()
                                if loop:
                                    loop.call_soon_threadsafe(_clear)
                                i += 1
                                continue
                            # Regular printable character - CRITICAL FIX for immediate echo
                            try:
                                ch = bytes([b]).decode('utf-8', errors='ignore')
                            except Exception:
                                ch = ''
                            if ch:
                                # If a panel is active, intercept keys for save/toggle and ignore normal input
                                if panel["active"]:
                                    def _panel_key():
                                        nonlocal status_text
                                        if ch.lower() == 's':
                                            if panel["type"] == "settings":
                                                _save_settings()
                                            else:
                                                _save_agents()
                                            status_text = "Saved | Esc to close"
                                            _render_panel()
                                        elif ch == ' ' and panel["type"] == "agents":
                                            _panel_cycle(+1)
                                            _render_panel()
                                    if loop:
                                        loop.call_soon_threadsafe(_panel_key)
                                    i += 1
                                    continue
                                # IMMEDIATE CHARACTER PROCESSING to fix one-character-behind lag
                                # Update state directly in thread to avoid async scheduling delays
                                txt = state["text"]
                                pos = state["cursor"]
                                state["text"] = txt[:pos] + ch + txt[pos:]
                                state["cursor"] = pos + 1
                                state["caret_visible"] = True
                                
                                # Immediate rendering without async scheduling for instant feedback
                                try:
                                    # Fast path: render input region only for typing characters
                                    caret = caret_char if state.get("caret_visible", True) else " "
                                    full_with_caret = state["text"][:pos+1] + caret + state["text"][pos+1:]
                                    raw_lines = full_with_caret.split('\n') or [""]
                                    to_show = raw_lines[-input_lines:]
                                    visual = []
                                    for idx, l in enumerate(to_show):
                                        is_first = (idx == max(0, len(to_show) - input_lines))
                                        line_text = f"{prompt}{l}" if is_first else f"... {l}"
                                        if len(line_text) < width:
                                            line_text = line_text + (" " * (width - len(line_text)))
                                        visual.append(line_text[:width])
                                    while len(visual) < input_lines:
                                        visual.insert(0, " " * width)
                                    # Direct region update for instant feedback
                                    self.display_renderer.update_region("input", "\n".join(visual), force=True)
                                except Exception:
                                    # Fallback to async scheduling if immediate render fails
                                    class _Evt: pass
                                    ev = _Evt()
                                    ev.character = ch
                                    if loop:
                                        loop.call_soon_threadsafe(on_char, ev)
                            i += 1
                finally:
                    try:
                        if old is not None and fd is not None:
                            termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    except Exception:
                        pass
                    try:
                        if fd is not None:
                            os.close(fd)
                    except Exception:
                        pass
                    if debug_file:
                        try:
                            debug_file.write("--- tui_v2_minimal end ---\n")
                            debug_file.close()
                        except Exception:
                            pass

            _dbg("step: starting reader thread")
            t = threading.Thread(target=reader_thread, name='tui-v2-minimal-reader', daemon=True)
            t.start()
            _dbg("step: reader thread started")

            # Start a simple cursor blink task (no interference with input)
            async def _blink():
                try:
                    while self.running:
                        await _asyncio.sleep(0.6)
                        state["caret_visible"] = not state.get("caret_visible", True)
                        render()
                except Exception:
                    pass

            _dbg("step: starting blink task")
            blink_task = _asyncio.create_task(_blink())
            _dbg("step: blink task started")

            self.running = True
            _dbg("step: entering wait loop")
            await self._shutdown_event.wait()
            _dbg("step: shutdown event set")
            return 0
        except Exception as e:
            logger.exception(f"Minimal input mode failed: {e}")
            try:
                import os as _os, sys as _sys, traceback as _tb
                if _os.getenv("AGENTS_TUI_V2_DEBUG", "0") == "1":
                    _sys.stderr.write("\n[tui-v2-minimal] error: " + str(e) + "\n")
                    _sys.stderr.write(_tb.format_exc() + "\n")
                    _sys.stderr.flush()
            except Exception:
                pass
            return 1
        finally:
            try:
                stop_flag["stop"] = True
            except Exception:
                pass
            try:
                blink_task.cancel()
            except Exception:
                pass
            # Disable mouse reporting
            try:
                out = self.display_renderer._output
                out.write('\033[?1006l')
                out.write('\033[?1000l')
                out.flush()
            except Exception:
                pass
            # Cleanup will be handled in the outer run() finally to avoid double-logging
            try:
                if dbg_file:
                    dbg_file.write("--- tui_v2_minimal end ---\n")
                    dbg_file.close()
            except Exception:
                pass
    
    async def shutdown(self):
        """Initiate graceful shutdown of the application."""
        if not self.running:
            return
            
        logger.info("Initiating TUI v2 shutdown...")
        self.running = False
        
        # Stop application controller first
        if self.app_controller:
            await self.app_controller.shutdown()
        
        # Signal main loop to exit
        self._shutdown_event.set()
    
    async def cleanup(self):
        """Cleanup all components and resources."""
        import os as _os
        _silent = _os.getenv("AGENTS_TUI_V2_SILENT_CLEANUP", "1") == "1"
        if not _silent:
            logger.info("Cleaning up TUI v2 components...")
        
        # Clean up in reverse order of initialization
        components = [
            self.app_controller,
            self.chat_interface,
            self.keyboard_processor,
            self.input_handler,
            self.unified_input_handler,
            self.display_renderer,
            self.layout_engine,
            self.theme_manager,
            self.status_manager,
            self.event_system,
            self.terminal_manager,
            self.terminal_state_manager
        ]
        
        for component in components:
            if component is None:
                continue
            try:
                # Prefer sync cleanup for display renderer to exit alt screen cleanly
                if component is self.display_renderer and hasattr(component, 'cleanup_sync'):
                    component.cleanup_sync()
                elif hasattr(component, 'cleanup'):
                    # async cleanup for others
                    await component.cleanup()
            except Exception as e:
                if not _silent:
                    logger.warning(f"Error during component cleanup: {e}")

        if not _silent:
            logger.info("TUI v2 cleanup complete")
    
    async def _show_welcome_screen(self):
        """Display welcome screen with system information."""
        if not self.display_renderer or not self.status_manager:
            return
            
        welcome_content = [
            "🚀 AgentsMCP TUI v2 - AI Agent Orchestration Platform",
            "",
            "✨ Features:",
            "  • Multi-agent support (Claude, Codex, Ollama)",
            "  • Real-time chat interface",
            "  • MCP tool integration",
            "  • Enhanced error handling",
            "",
            "🎯 Quick Start:",
            "  • Type naturally to chat with AI agents",
            "  • Use /help to see available commands",
            "  • Press Ctrl+C or type /quit to exit",
            "",
            "📊 System Status:",
        ]
        
        # Add current system status
        status = self.status_manager.current_status
        context = self.status_manager.context_info
        
        # Support both dict-like and attribute-style context objects
        def _ctx_get(obj, key, default):
            try:
                if hasattr(obj, 'get'):
                    return obj.get(key, default)
            except Exception:
                pass
            try:
                return getattr(obj, key)
            except Exception:
                return default

        welcome_content.extend([
            f"  • Status: {status.icon} {status.title}",
            f"  • Agent: {_ctx_get(context, 'agent', 'Default')}",
            f"  • Model: {_ctx_get(context, 'model', 'Auto')}",
            f"  • Terminal: {self.terminal_manager.get_capabilities().type.value}",
            ""
        ])
        
        # Display welcome in a formatted box
        # Use the supported API to format a message box for printing
        try:
            welcome_box = self.display_renderer.format_message_box(
                title="Welcome to AgentsMCP",
                content=welcome_content,
                border_style="double"
            )
            sys.stdout.write(f"\r{welcome_box}\n")
            sys.stdout.write("\r\n")  # Add spacing
            sys.stdout.flush()
        except Exception:
            # Fallback: print plain text if renderer API is unavailable
            sys.stdout.write(f"\r{chr(10).join(['=== Welcome to AgentsMCP ==='] + welcome_content)}\n")
            sys.stdout.write("\r\n")
            sys.stdout.flush()
    
    async def _display_status_bar(self):
        """Display/update the main status bar using the renderer, not prints."""
        if not self.display_renderer or not self.status_manager:
            return

        # Get current status and context
        status = self.status_manager.current_status
        context = self.status_manager.context_info if hasattr(self.status_manager, 'context_info') else {}

        def _ctx_get(obj, key, default):
            try:
                if hasattr(obj, 'get'):
                    return obj.get(key, default)
            except Exception:
                pass
            try:
                return getattr(obj, key)
            except Exception:
                return default

        items = [
            f"{status.icon} {status.title}",
            f"🤖 {_ctx_get(context, 'agent', 'Default')}",
            f"⚙️ {_ctx_get(context, 'model', 'Auto')}",
            "🔧 /help",
            "❌ /quit",
        ]

        content = "  |  ".join(items)
        try:
            caps = self.display_renderer.terminal_manager.detect_capabilities()
            bar = self.display_renderer.format_status_bar(content, width=caps.width)
            self.display_renderer.update_region('status_bar', bar, force=True)
        except Exception:
            # Best-effort fallback
            self.display_renderer.update_region('status_bar', content, force=True)


class TUILauncher:
    """Launcher class for integrating with existing CLI infrastructure."""
    
    def __init__(self):
        self.app: Optional[MainTUIApp] = None
    
    async def launch_tui(self, cli_config: Optional[CLIConfig] = None) -> int:
        """Launch the TUI using the single working implementation.
        
        Args:
            cli_config: CLI configuration (currently not used by fixed implementation)
            
        Returns:
            Exit code
        """
        logger.info("Launching fixed working TUI - the only supported TUI implementation")
        from .fixed_working_tui import launch_fixed_working_tui
        return await launch_fixed_working_tui()
    


# Convenience function for CLI integration
async def launch_main_tui(cli_config: Optional[CLIConfig] = None) -> int:
    """Launch the main TUI application with CLI integration.
    
    Args:
        cli_config: CLI configuration, uses defaults if None
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    launcher = TUILauncher()
    return await launcher.launch_tui(cli_config)


# Direct execution support
if __name__ == "__main__":
    async def main():
        app = MainTUIApp()
        return await app.run()
    
    sys.exit(asyncio.run(main()))
