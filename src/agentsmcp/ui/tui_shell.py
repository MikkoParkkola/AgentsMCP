from __future__ import annotations

import asyncio
from typing import Optional, Dict, Any, List
import threading
import queue
import sys
import termios
import tty
import select
import time

from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich import box

from .theme_manager import ThemeManager
from .keyboard_input import KeyboardInput, KeyCode

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore


class TUIShell:
    """Rich-based TUI shell (sci‑fi themed).

    - Sidebar with hotkeys (1–9)
    - Header/Footer with status
    - Pages: Home, Jobs, Agents, Models, Providers, Costs, MCP, Discovery, Settings
    - Command palette (type colon-prefixed commands)
    - Selection, model application, provider/agent edits
    - Job watch pane with pause/clear/stop
    """

    PAGES = [
        "Chat",
        "Home",
        "Jobs",
        "Agents",
        "Config",   # Combined Models + Providers
        "Costs",
        "MCP",
        "Discovery",
        "Settings",
    ]
    COMMANDS = [
        # Core navigation and info
        "help", "goto", "status", "history", "clear", "exit",
        # Providers and models
        "set", "models", "apply-model", "provider", "agent", "model",
        # Jobs/watch
        "select", "cancel", "cancel-selected", "watch", "watch-selected",
        "watch-pause", "watch-resume", "watch-clear", "watch-stop",
        # Misc
        "generate-config", "keys", "export-keys", "provider-order",
        # Aliases from CLI
        "execute", "symphony", "theme", "config", "new", "save",
        # Top-level commands exposed elsewhere
        "costs", "budget", "server", "mcp", "discovery"
    ]

    # Layout constants - consistent margins and sizing throughout the UI
    _MARGIN_X = 1          # horizontal margin for content
    _PADDING_X = 1         # content padding inside boxes
    _MIN_WIDTH = 40        # minimum width for content areas
    _MIN_HEIGHT = 10       # minimum height for content areas

    def __init__(self, theme_manager: Optional[ThemeManager] = None, chat_handler=None):
        self.console = Console()
        self.theme_manager = theme_manager or ThemeManager()
        self.running = True
        self.page = "Chat"
        self.status_message: str = "Type :help for commands"
        self.input_queue: asyncio.Queue[str] = asyncio.Queue()
        self.base_url = "http://127.0.0.1:8000"
        # Preferences
        self.high_contrast: bool = False
        # Selection / cache
        self.selection: Dict[str, int] = {"Jobs": -1, "Models": -1}
        self.current_provider: str = "openai"
        self.models_cache: Dict[str, List[Dict[str, Any]]] = {}
        # Watch state
        self.watch_active: bool = False
        self.watch_job_id: Optional[str] = None
        self.watch_lines: List[str] = []
        self.watch_paused: bool = False
        self.watch_task: Optional[asyncio.Task] = None
        # Input threads (daemon) and queues
        self._key_thread: Optional[threading.Thread] = None
        self._thread_queue: "queue.Queue[object]" = queue.Queue()
        self._waiting_line_input: bool = False
        # Chat support
        self.chat_handler = chat_handler
        self.chat_buffer: str = ""
        self.chat_messages: List[Dict[str, str]] = []
        # UI state
        self.palette_visible: bool = True
        self._dirty: bool = True
        self._last_render_ts: float = 0.0
        self.focus: str = "chat"  # 'chat' or 'sidebar'
        self.sidebar_index: int = 0
        # ESC suppression to avoid stray '[' 'A' after lone ESC
        self._suppress_bracket_until: float = 0.0
        self._suppress_esc_tail: int = 0
        # Command mode suggestion state
        self.cmd_suggestions: List[str] = []
        self.cmd_sel: int = 0
        # ESC sequence state machine buffer
        self._pending_esc: str = ""

    # Layout
    def _make_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=2),
        )
        layout["body"].split_row(
            Layout(name="sidebar", size=28),
            Layout(name="main", ratio=1),
        )
        return layout

    def _calculate_content_size(self, available_width: int, available_height: int) -> tuple[int, int]:
        """Calculate content size with consistent margins and minimum constraints."""
        # Calculate effective width and height after margins
        content_width = max(self._MIN_WIDTH, available_width - 2 * (self._MARGIN_X + self._PADDING_X))
        content_height = max(self._MIN_HEIGHT, available_height - 2 * self._MARGIN_X)
        return content_width, content_height

    def _safe_text_wrap(self, text: str, width: int) -> List[str]:
        """Safely wrap text to fit within specified width."""
        import textwrap
        if width <= 0:
            return [""]
        # Use textwrap but ensure we don't exceed width
        wrapped = textwrap.wrap(text, width=width, expand_tabs=True, replace_whitespace=False)
        # Additional safety: clip any lines that are still too long
        return [line[:width] if len(line) > width else line for line in wrapped]

    # Rendering helpers
    def _render_header(self) -> Panel:
        title = f"[bold cyan]AgentsMCP[/] [magenta]•[/] [bold magenta]{self.page}[/]"
        style = "bold black on white" if self.high_contrast else "bold white on grey7"
        border = "white" if self.high_contrast else "cyan"
        return Panel(title, style=style, border_style=border, box=box.ROUNDED)

    def _render_sidebar(self) -> Panel:
        tbl = Table.grid(padding=(0, 1))
        for idx0, it in enumerate(self.PAGES):
            idx = idx0 + 1
            hotkey = f"[{idx}]"
            active = (it == self.page)
            focused_row = (self.focus == 'sidebar' and self.sidebar_index == idx0)
            color = "bold magenta" if active else ("bold white" if self.high_contrast else "white")
            pointer = "▶" if focused_row else " "
            tbl.add_row(f"{pointer} [cyan]{hotkey}[/]  [{color}]{it}[/]")
        border = "yellow" if self.focus == 'sidebar' else ("white" if self.high_contrast else "blue")
        return Panel(tbl, title="[cyan]Menu", border_style=border, box=box.ROUNDED)

    def _render_footer(self) -> Panel:
        hints = (
            "[cyan]1[/]Chat [cyan]2[/]Home [cyan]3[/]Jobs [cyan]4[/]Agents [cyan]5[/]Config "
            "[cyan]6[/]Costs [cyan]7[/]MCP [cyan]8[/]Discovery [cyan]9[/]Settings  •  "
            "[yellow]TAB[/] focus sidebar  •  [yellow]:[/] start command  •  [yellow]ESC[/] dismiss overlay  •  Ctrl+C to quit"
        )
        if self.status_message:
            hints += f"\n[white]{self.status_message}"
        return Panel(hints, border_style=("white" if self.high_contrast else "grey37"), box=box.ROUNDED)

    def _fmt_kv(self, d: Dict[str, Any]) -> str:
        rows = []
        for k, v in d.items():
            rows.append(f"[bold magenta]{k}[/]: {v}")
        return "\n".join(rows) if rows else "(empty)"

    def _render_help_panel(self) -> Panel:
        text = (
            "[cyan]Navigation[/]: 1–9 switch pages • q quit • TAB focus sidebar\n"
            "[cyan]Commands[/]: start a line with ':' (e.g., :goto Settings, :help)\n"
            "  :goto <page>  •  :select <n>  •  :cancel <job_id>  •  :cancel-selected\n"
            "  :set provider <name> api_base|api_key <val>  •  :set agent <name> model <model>\n"
            "  :models provider <name>  •  :models set <agent> <model>  •  :apply-model <agent>\n"
            "  :watch <job_id>  •  :watch-selected  •  :watch-pause|resume|clear|stop\n"
        )
        return Panel(text, title="[magenta]Help", border_style=("white" if self.high_contrast else "cyan"), box=box.ROUNDED)

    def _render_palette_panel(self) -> Panel:
        """Render a compact command palette hint when not actively filtering."""
        text = (
            "[cyan]Tip[/]: Type ':' to start a command. Examples:\n"
            "  :help    :goto Settings    :models provider ollama-turbo\n"
            "[cyan]Keys[/]: ←/→ pages • 1–9 jump • q quit • TAB focus sidebar"
        )
        return Panel(text, title="[magenta]Command Palette", border_style="cyan", box=box.ROUNDED)

    # Networking helpers
    async def _fetch_json(self, path: str, timeout: float = 2.0) -> Dict[str, Any]:
        """Fetch JSON data with proper error handling and user feedback."""
        if httpx is None:
            self.status_message = "⚠️  Network features disabled (httpx not installed)"
            return {}
        
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    return r.json()
                self.status_message = f"Server error {r.status_code}"
                return {"error": r.text, "status": r.status_code}
        except httpx.ConnectError:
            self.status_message = "📡 Cannot connect to server"
            return {}
        except httpx.TimeoutException:
            self.status_message = f"⏱️  Request timed out"
            return {}
        except Exception as e:
            self.status_message = f"Network error: {str(e)[:50]}"
            return {}

    async def _put_settings(self, body: Dict[str, Any]) -> bool:
        """Update settings with proper error handling and user feedback."""
        if httpx is None:
            self.status_message = "⚠️  Network features disabled (httpx not installed)"
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.put(f"{self.base_url}/settings", json=body)
                return r.status_code == 200
        except httpx.ConnectError:
            self.status_message = "⚠️  Cannot connect to AgentsMCP server"
            return False
        except httpx.TimeoutException:
            self.status_message = "⚠️  Request timeout - server may be busy"
            return False
        except Exception as e:
            self.status_message = f"⚠️  Settings update failed: {str(e)}"
            return False

    async def _delete_job(self, job_id: str) -> bool:
        """Delete a job with proper error handling and user feedback."""
        if httpx is None:
            self.status_message = "⚠️  Network features disabled (httpx not installed)"
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.delete(f"{self.base_url}/jobs/{job_id}")
                return r.status_code == 200
        except httpx.ConnectError:
            self.status_message = "⚠️  Cannot connect to AgentsMCP server"
            return False
        except httpx.TimeoutException:
            self.status_message = "⚠️  Request timeout - server may be busy"
            return False
        except Exception as e:
            self.status_message = f"⚠️  Job deletion failed: {str(e)}"
            return False

    async def _get_models(self, provider: str) -> List[Dict[str, Any]]:
        """Get models for a provider with proper error handling and user feedback."""
        if httpx is None:
            self.status_message = "⚠️  Network features disabled (httpx not installed)"
            return []
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/providers/{provider}/models")
                if r.status_code == 200:
                    j = r.json()
                    return list(j.get("models", []) or [])
                return []
        except httpx.ConnectError:
            self.status_message = "⚠️  Cannot connect to AgentsMCP server"
            return []
        except httpx.TimeoutException:
            self.status_message = "⚠️  Request timeout - server may be busy"
            return []
        except Exception as e:
            self.status_message = f"⚠️  Failed to get models: {str(e)}"
            return []
        except Exception:
            return []

    # Watch (polling) helpers
    async def _watch_worker(self, job_id: str) -> None:
        last = None
        try:
            while self.watch_active and self.watch_job_id == job_id:
                if not self.watch_paused:
                    st = await self._fetch_json(f"/status/{job_id}")
                    out = st.get("output") or st.get("error") or ""
                    if out and out != last:
                        lines = out.splitlines()
                        # keep short ring buffer to bound memory
                        self.watch_lines = lines[-500:]
                        last = out
                await asyncio.sleep(1.0)
        except Exception:
            pass

    async def _start_watch(self, job_id: str) -> None:
        await self._stop_watch()
        self.watch_active = True
        self.watch_job_id = job_id
        self.watch_lines.clear()
        self.watch_paused = False
        self.status_message = f"Watching job {job_id}"
        self.watch_task = asyncio.create_task(self._watch_worker(job_id))

    async def _stop_watch(self) -> None:
        self.watch_active = False
        self.watch_job_id = None
        self.watch_lines.clear()
        self.watch_paused = False
        if self.watch_task:
            self.watch_task.cancel()
            try:
                await self.watch_task
            except Exception:
                pass
            self.watch_task = None

    # Rendering
    async def _render_main_panel(self) -> Panel | Group:
        # Help overlay: show on any page when status_message displays commands
        if self.status_message.startswith("Commands:"):
            return self._render_help_panel()

        if self.page == "Chat":
            # Calculate available width for proper text wrapping
            terminal_size = self.console.size
            content_width, _ = self._calculate_content_size(terminal_size.width, terminal_size.height)
            
            lines: List[str] = []
            for msg in self.chat_messages[-50:]:
                who = "You" if msg.get("role") == "user" else "Assistant"
                text = msg.get('text', '')
                # Wrap long messages to fit content area
                if len(text) > content_width - 20:  # Account for "Who: " prefix
                    wrapped_lines = self._safe_text_wrap(text, content_width - 20)
                    for i, wrapped_line in enumerate(wrapped_lines):
                        if i == 0:
                            lines.append(f"[bold magenta]{who}[/]: {wrapped_line}")
                        else:
                            lines.append(f"     {wrapped_line}")  # Indent continuation
                else:
                    lines.append(f"[bold magenta]{who}[/]: {text}")
            lines.append("")
            caret = "▌" if self.focus == 'chat' else ""
            # Ensure chat buffer doesn't overflow
            buffer_display = self.chat_buffer
            if len(buffer_display) > content_width - 10:
                buffer_display = buffer_display[:content_width - 13] + "..."
            lines.append(f"[cyan]>[/] {buffer_display}{caret}")
            body = "\n".join(lines) if lines else f"[cyan]>[/] {buffer_display}{caret}"
            chat_panel = Panel(body, title="[magenta]Chat", border_style="green", box=box.ROUNDED)
            items = [chat_panel]
            buf = self.chat_buffer
            if buf.startswith(':'):
                self._update_cmd_suggestions()
                # Render suggestions with selection
                if self.cmd_suggestions:
                    lines = []
                    for i, s in enumerate(self.cmd_suggestions):
                        marker = '›' if i == self.cmd_sel else ' '
                        lines.append(f"{marker} {s}")
                    hint = ": to run commands • Enter to complete/execute • ↑/↓ to choose"
                    sug_text = "\n".join(lines) + f"\n[grey58]{hint}[/]"
                else:
                    sug_text = "(no matches)\n[grey58]: to run commands • Enter to execute[/]"
                items.append(Panel(sug_text, title="[magenta]Command Suggestions", border_style="cyan", box=box.ROUNDED))
            elif getattr(self, 'palette_visible', False):
                items.append(self._render_palette_panel())
            return Group(*items)

        # Other pages
        if self.page == "Home":
            stats = await self._fetch_json("/stats")
            body = self._fmt_kv({
                "active_jobs": stats.get("active_jobs", 0),
                "total_jobs": stats.get("total_jobs", 0),
                "agents": ", ".join(stats.get("agent_types", []) or []),
                "storage": stats.get("storage_type", ""),
            })
            return Panel(body or "No stats", title="[cyan]Status", border_style="green", box=box.ROUNDED)

        if self.page == "Jobs":
            jobs = await self._fetch_json("/jobs")
            job_list = jobs.get("jobs", []) or []
            tbl = Table("#", "job_id", "agent", "state", "updated", show_header=True, header_style="bold cyan")
            sel = self.selection.get("Jobs", -1)
            for i, j in enumerate(job_list[:100]):
                style = "bold yellow" if i == sel else None
                tbl.add_row(str(i + 1), j.get("job_id", ""), j.get("agent", ""), j.get("state", ""), str(j.get("updated_at", "")), style=style)
            main_panel = Panel(tbl, title="[magenta]Jobs", border_style="green", box=box.ROUNDED)
            if self.watch_active and self.watch_job_id:
                # Apply safe text wrapping to watch output
                terminal_size = self.console.size
                content_width, _ = self._calculate_content_size(terminal_size.width, terminal_size.height)
                
                if self.watch_lines:
                    # Wrap long output lines to fit content area
                    wrapped_lines = []
                    for line in self.watch_lines[-200:]:
                        if len(line) > content_width:
                            wrapped_lines.extend(self._safe_text_wrap(line, content_width))
                        else:
                            wrapped_lines.append(line)
                    content = "\n".join(wrapped_lines)
                else:
                    content = "(no output yet)"
                
                watch_title = f"[magenta]Output • Job {self.watch_job_id} ({'paused' if self.watch_paused else 'live'})"
                watch_panel = Panel(content, title=watch_title, border_style="yellow", box=box.ROUNDED)
                return Group(main_panel, watch_panel)
            return main_panel

        if self.page == "Providers":
            st = await self._fetch_json("/settings")
            providers = (st.get("providers") or {})
            tbl = Table("provider", "api_base", "has_key", header_style="bold cyan")
            for name, cfg in providers.items():
                tbl.add_row(name, str(cfg.get("api_base", "")), "yes" if cfg.get("has_key") else "no")
            return Panel(tbl, title="[magenta]Providers", border_style="green", box=box.ROUNDED)

        if self.page == "Agents":
            st = await self._fetch_json("/settings")
            agents = (st.get("agents") or {})
            tbl = Table("agent", "provider", "model", header_style="bold cyan")
            for name, cfg in agents.items():
                tbl.add_row(name, str(cfg.get("provider", "")), str(cfg.get("model", "")))
            return Panel(tbl, title="[magenta]Agents", border_style="green", box=box.ROUNDED)

        if self.page == "Config":
            # Combined Providers + Models view
            st = await self._fetch_json("/settings")
            providers = (st.get("providers") or {})
            p_tbl = Table("provider", "api_base", "has_key", header_style="bold cyan")
            for name, cfg in providers.items():
                p_tbl.add_row(name, str(cfg.get("api_base", "")), "yes" if cfg.get("has_key") else "no")
            p_panel = Panel(p_tbl, title="[magenta]Providers", border_style="green", box=box.ROUNDED)

            models = self.models_cache.get(self.current_provider) or []
            if not models:
                body = f"Provider: {self.current_provider} • Use :models provider <name> to fetch models."
                m_panel = Panel(body, title="[magenta]Models", border_style="green", box=box.ROUNDED)
            else:
                tbl = Table("#", "id", "name", header_style="bold cyan")
                sel = self.selection.get("Models", -1)
                for i, m in enumerate(models[:200]):
                    label = m.get("name") or m.get("id")
                    style = "bold yellow" if i == sel else None
                    tbl.add_row(str(i + 1), m.get("id", ""), label or "", style=style)
                hint = f"Provider: {self.current_provider} • :select <n> then :apply-model <agent>"
                m_panel = Panel(tbl, title=f"[magenta]Models • {hint}", border_style="green", box=box.ROUNDED)
            return Group(p_panel, m_panel)

        if self.page == "Costs":
            costs = await self._fetch_json("/costs")
            if costs.get("status") == 501 or costs.get("error"):
                return Panel("Costs unavailable", title="[magenta]Costs", border_style="yellow", box=box.ROUNDED)
            body = self._fmt_kv({"total": costs.get("total"), "daily": costs.get("daily")})
            return Panel(body or "No data", title="[magenta]Costs", border_style="green", box=box.ROUNDED)

        if self.page == "MCP":
            m = await self._fetch_json("/mcp")
            servers = m.get("servers", [])
            tbl = Table("name", "enabled", "transport", "endpoint", header_style="bold cyan")
            for s in servers:
                endpoint = " ".join(s.get("command") or []) if s.get("command") else (s.get("url") or "")
                tbl.add_row(s.get("name", ""), "yes" if s.get("enabled") else "no", s.get("transport", ""), endpoint)
            return Panel(tbl, title="[magenta]MCP", border_style="green", box=box.ROUNDED)

        if self.page == "Discovery":
            d = await self._fetch_json("/discovery")
            body = self._fmt_kv(d)
            return Panel(body or "No data", title="[magenta]Discovery", border_style="green", box=box.ROUNDED)

        if self.page == "Settings":
            st = await self._fetch_json("/settings")
            providers = (st.get("providers") or {})
            agents = (st.get("agents") or {})
            # Providers table
            p_tbl = Table("provider", "api_base", "has_key", header_style="bold cyan")
            for name, cfg in providers.items():
                p_tbl.add_row(name, str(cfg.get("api_base", "")), "yes" if cfg.get("has_key") else "no")
            p_panel = Panel(p_tbl, title="[magenta]Providers", border_style="green", box=box.ROUNDED)
            # Agents table
            a_tbl = Table("agent", "provider", "model", header_style="bold cyan")
            for name, cfg in agents.items():
                a_tbl.add_row(name, str(cfg.get("provider", "")), str(cfg.get("model", "")))
            a_panel = Panel(a_tbl, title="[magenta]Agents", border_style="green", box=box.ROUNDED)
            hint = Panel(
                "Use :set provider <name> api_base|api_key <val>  •  :set agent <name> model <model>  •  or :form …",
                title="[cyan]Edit Hints", border_style="yellow", box=box.ROUNDED
            )
            return Group(p_panel, a_panel, hint)

        return Panel("Work in progress", border_style="green", box=box.ROUNDED)

    def _update_cmd_suggestions(self) -> None:
        """Update command suggestions based on current chat_buffer starting with ':'"""
        try:
            content = self.chat_buffer[1:]
            parts = content.split()
            # No token typed yet -> suggest commands
            if not parts:
                pool = self.COMMANDS
                self.cmd_suggestions = pool[:12]
                self.cmd_sel = 0
                return
            first = parts[0]
            # If only command token (maybe partial)
            if len(parts) == 1 and not content.endswith(' '):
                self.cmd_suggestions = [c for c in self.COMMANDS if c.startswith(first)][:12]
                self.cmd_sel = 0 if self.cmd_sel >= len(self.cmd_suggestions) else self.cmd_sel
                return
            # If command is 'goto', suggest pages for second token
            if first == 'goto':
                page_prefix = parts[1] if len(parts) > 1 else ''
                self.cmd_suggestions = [p for p in self.PAGES if p.lower().startswith(page_prefix.lower())][:12]
                self.cmd_sel = 0 if self.cmd_sel >= len(self.cmd_suggestions) else self.cmd_sel
                return
            # Default: no special suggestions
            self.cmd_suggestions = []
            self.cmd_sel = 0
        except Exception:
            self.cmd_suggestions = []
            self.cmd_sel = 0

        

    # Input workers
    def _key_thread_fn(self) -> None:
        """Read keys using prompt_toolkit when available, else fallback to cbreak+ESC parser.

        Important: Only attempt raw/tty manipulation when stdin is a real TTY.
        In non-interactive environments (e.g., CI, redirected pipes) we avoid
        termios/tty calls which would raise and instead idle until shutdown.
        """
        # Try prompt_toolkit first for robust, portable key decoding
        try:
            from prompt_toolkit.input import create_input
            from prompt_toolkit.keys import Keys
            with create_input() as inp:
                while self.running:
                    try:
                        for kp in inp.read_keys():
                            k = kp.key
                            if k == Keys.Up:
                                self._thread_queue.put({"key": "\x1b[A"})
                            elif k == Keys.Down:
                                self._thread_queue.put({"key": "\x1b[B"})
                            elif k == Keys.Left:
                                self._thread_queue.put({"key": "\x1b[D"})
                            elif k == Keys.Right:
                                self._thread_queue.put({"key": "\x1b[C"})
                            elif k in (Keys.Enter,):
                                self._thread_queue.put({"key": "\n"})
                            elif k in (Keys.Backspace,):
                                self._thread_queue.put({"key": "\x7f"})
                            elif k in (Keys.Tab,):
                                self._thread_queue.put({"key": "\t"})
                            elif k in (Keys.Escape,):
                                self._thread_queue.put({"key": "\x1b"})
                            else:
                                # Prefer textual data for printable keys
                                data = getattr(kp, 'data', None)
                                if data:
                                    self._thread_queue.put({"key": data})
                                else:
                                    # Some prompt_toolkit builds set key to the literal character
                                    try:
                                        if isinstance(k, str) and len(k) == 1:
                                            self._thread_queue.put({"key": k})
                                    except Exception:
                                        pass
                    except Exception:
                        time.sleep(0.01)
            return
        except Exception:
            pass

        # If stdin is not a TTY, don't attempt raw mode. Just idle and let the
        # render loop run (useful for environments that can't provide input).
        try:
            if not sys.stdin.isatty():
                while self.running:
                    time.sleep(0.05)
                return
        except Exception:
            # If detection fails, treat as non-tty and idle
            while self.running:
                time.sleep(0.05)
            return

        # Fallback: per‑character input using KeyboardInput abstraction.
        try:
            kb = KeyboardInput()
        except Exception:
            kb = None  # Last resort: idle
        if kb is None:
            while self.running:
                time.sleep(0.05)
            return

        # Map KeyCode to the string tokens _handle_key expects
        def _map_keycode(code: KeyCode) -> str | None:
            if code == KeyCode.ENTER:
                return "\n"
            if code == KeyCode.ESCAPE:
                return "\x1b"
            if code == KeyCode.BACKSPACE:
                return "\x7f"
            if code == KeyCode.TAB:
                return "\t"
            if code == KeyCode.UP:
                return "\x1b[A"
            if code == KeyCode.DOWN:
                return "\x1b[B"
            if code == KeyCode.LEFT:
                return "\x1b[D"
            if code == KeyCode.RIGHT:
                return "\x1b[C"
            if code == KeyCode.SPACE:
                return " "
            # Other keys (HOME/END/PgUp/PgDn) not handled explicitly
            return None

        # Read keys in a tight loop with small timeout, enqueue immediately
        while self.running:
            try:
                code, ch = kb.get_key(timeout=0.05)
                if not self.running:
                    break
                if code is None and ch is None:
                    continue
                if code is not None:
                    mapped = _map_keycode(code)
                    if mapped is not None:
                        self._thread_queue.put({"key": mapped})
                elif ch is not None and len(ch) > 0:
                    # Send raw printable character(s). For safety, only first char.
                    self._thread_queue.put({"key": ch[0]})
            except Exception:
                time.sleep(0.01)

    async def _handle_command(self, cmd: str) -> None:
        # Normalize leading ':' for palette-style commands
        if cmd.startswith(":"):
            cmd = cmd[1:]
        parts = cmd.split()
        if not parts:
            return
        try:
            # quick navigations
            if parts[0] in [str(i) for i in range(1, 10)]:
                idx = int(parts[0]) - 1
                if 0 <= idx < len(self.PAGES):
                    self.page = self.PAGES[idx]
                    self.status_message = f"Switched to {self.page}"
                return
            if parts[0] in ("q", "quit", "exit"):
                self.running = False
                return
            if parts[0] in ("help", "h"):
                if self.status_message.startswith("Commands:"):
                    self.status_message = "Help hidden"
                else:
                    self.status_message = (
                        "Commands: :goto <page> • :select <n> • :cancel <job_id> • :cancel-selected • "
                        ":set provider <name> api_base|api_key <val> • :set agent <name> model <model> • :models provider <name> • :models set <agent> <model> • "
                        ":apply-model <agent> • :watch <job_id> • :watch-selected • :watch-pause • :watch-resume • :watch-clear • :watch-stop • :contrast on|off"
                    )
                return
            if parts[0] == "goto" and len(parts) >= 2:
                dest = parts[1].capitalize()
                if dest in self.PAGES:
                    self.page = dest
                    self.status_message = f"Switched to {dest}"
                else:
                    self.status_message = f"Unknown page: {dest}"
                return
            if parts[0] == "select" and len(parts) >= 2 and parts[1].isdigit():
                idx = int(parts[1]) - 1
                if self.page in ("Jobs", "Models"):
                    self.selection[self.page] = idx
                    self.status_message = f"Selected row {idx+1}"
                return
            if parts[0] == "cancel-selected":
                if self.page == "Jobs":
                    sel = self.selection.get("Jobs", -1)
                    jobs = await self._fetch_json("/jobs")
                    job_list = jobs.get("jobs", []) or []
                    if 0 <= sel < len(job_list):
                        ok = await self._delete_job(job_list[sel].get("job_id", ""))
                        self.status_message = "Job cancelled" if ok else "Cancel failed"
                    else:
                        self.status_message = "Nothing selected"
                else:
                    self.status_message = "Use on Jobs page"
                return
            if parts[0] == "cancel" and len(parts) >= 2:
                ok = await self._delete_job(parts[1])
                self.status_message = "Job cancelled" if ok else "Cancel failed"
                return
            if parts[0] == "apply-model" and len(parts) >= 2:
                sel = self.selection.get("Models", -1)
                models = self.models_cache.get(self.current_provider) or []
                if 0 <= sel < len(models):
                    model_id = models[sel].get("id") or models[sel].get("name")
                    agent = parts[1]
                    ok = await self._put_settings({"agents": {agent: {"model": model_id}}})
                    self.status_message = f"Applied {model_id} to {agent}" if ok else "Apply failed"
                else:
                    self.status_message = "Nothing selected"
                return
            if parts[0] == "form" and len(parts) >= 3 and parts[1] == "provider":
                name = parts[2]
                # Inline simple form
                current = await self._fetch_json("/settings")
                prov = (current.get("providers") or {}).get(name, {})
                api_base_old = prov.get("api_base", "")
                api_base = await asyncio.to_thread(self.console.input, f"api_base for {name} [{api_base_old}]: ")
                if not api_base:
                    api_base = api_base_old
                api_key = await asyncio.to_thread(self.console.input, f"api_key for {name} (blank to clear): ")
                ok = await self._put_settings({"providers": {name: {"api_base": api_base, "api_key": api_key}}})
                self.status_message = "Provider updated" if ok else "Provider update failed"
                return
            if parts[0] == "form" and len(parts) >= 3 and parts[1] == "agent":
                name = parts[2]
                current = await self._fetch_json("/settings")
                ag = (current.get("agents") or {}).get(name, {})
                model_old = ag.get("model", "")
                model = await asyncio.to_thread(self.console.input, f"model for {name} [{model_old}]: ")
                if not model:
                    model = model_old
                ok = await self._put_settings({"agents": {name: {"model": model}}})
                self.status_message = "Agent updated" if ok else "Agent update failed"
                return
            # Fallthrough: unknown command
            self.status_message = "Unknown command"
            return
        except Exception as e:
            # Ensure any unexpected errors do not break the TUI loop
            self.status_message = f"Command error: {e}"

    # Main loop
    async def run(self) -> None:
        """Run the TUI shell event/render loop using Rich Live to avoid scrollback flicker."""
        from rich.live import Live
        self.running = True
        layout = self._make_layout()
        # Start key reader as daemon thread
        self._key_thread = threading.Thread(target=self._key_thread_fn, daemon=True)
        self._key_thread.start()
        # Use alternate screen and live updates so frames don't fill scrollback.
        # transient=False keeps the TUI visible on exit; set True to clear on exit.
        try:
            with Live(layout, console=self.console, refresh_per_second=24, screen=True, transient=True) as live:
                while self.running:
                    # Drain any pending input first
                    while not self._thread_queue.empty():
                        try:
                            item = self._thread_queue.get_nowait()
                        except Exception:
                            break
                        else:
                            if isinstance(item, dict) and "key" in item:
                                await self._handle_key(item["key"])  # type: ignore[arg-type]
                            elif isinstance(item, dict) and "cmd" in item:
                                await self._handle_command(str(item["cmd"]))
                            elif isinstance(item, str):
                                await self._handle_command(item)
                    # Render only when dirty or keep-alive (1s)
                    now = asyncio.get_event_loop().time()
                    if getattr(self, '_dirty', True) or (now - getattr(self, '_last_render_ts', 0.0)) > 1.0:
                        layout["header"].update(self._render_header())
                        layout["sidebar"].update(self._render_sidebar())
                        main_panel = await self._render_main_panel()
                        layout["main"].update(main_panel)
                        layout["footer"].update(self._render_footer())
                        live.update(layout, refresh=True)
                        self._dirty = False
                        self._last_render_ts = now
                    await asyncio.sleep(0.02)
        finally:
            self.running = False
            await self._stop_watch()
            # Best-effort join of the key thread (daemon)
            try:
                if self._key_thread and self._key_thread.is_alive():
                    self._key_thread.join(timeout=0.2)
            except Exception:
                pass

    async def _handle_key(self, key: str) -> None:
        """Handle a single key or escape sequence."""
        def map_arrow(seq: str) -> Optional[str]:
            try:
                if seq in ("\x1b[A", "\x1bOA"): return 'up'
                if seq in ("\x1b[B", "\x1bOB"): return 'down'
                if seq in ("\x1b[C", "\x1bOC"): return 'right'
                if seq in ("\x1b[D", "\x1bOD"): return 'left'
                if seq.startswith("\x1b[") and len(seq) >= 3 and seq[-1] in 'ABCD':
                    return {'A':'up','B':'down','C':'right','D':'left'}[seq[-1]]
            except Exception:
                pass
            return None
        # Toggle focus with TAB
        if key == '\t':
            self.focus = 'sidebar' if self.focus == 'chat' else 'chat'
            if self.focus == 'sidebar':
                try:
                    self.sidebar_index = self.PAGES.index(self.page)
                except Exception:
                    self.sidebar_index = 0
            self._dirty = True
            return
        # Sidebar focus handling
        if self.focus == 'sidebar':
            arrow = map_arrow(key)
            if arrow == 'up' or key == 'k':
                self.sidebar_index = max(0, self.sidebar_index - 1)
                self._dirty = True
                return
            if arrow == 'down' or key == 'j':
                self.sidebar_index = min(len(self.PAGES) - 1, self.sidebar_index + 1)
                self._dirty = True
                return
            if key in ("\r", "\n") or arrow == 'right':
                self.page = self.PAGES[self.sidebar_index]
                self.status_message = f"Switched to {self.page}"
                self._dirty = True
                return
            if arrow == 'left':
                self.focus = 'chat'
                self._dirty = True
                return
            # Number quick jump
            if len(key) == 1 and key.isdigit():
                idx = int(key) - 1
                if 0 <= idx < len(self.PAGES):
                    self.sidebar_index = idx
                    self.page = self.PAGES[self.sidebar_index]
                    self.status_message = f"Switched to {self.page}"
                    self._dirty = True
                return
            # First-letter quick navigation
            if len(key) == 1 and key.isalpha():
                key_lower = key.lower()
                for i, name in enumerate(self.PAGES):
                    if name.lower().startswith(key_lower):
                        self.sidebar_index = i
                        self.page = name
                        self.status_message = f"Switched to {self.page}"
                        self._dirty = True
                        break
                return
            # Swallow other printable keys while in sidebar focus
            if len(key) == 1 and key >= ' ':
                self._dirty = True
                return
        # mark dirty on any key
        self._dirty = True
        # ESC: exit overlays/command mode
        if key == '\x1b':
            if self.status_message.startswith("Commands:"):
                self.status_message = ""
                self._dirty = True
                return
            if self.chat_buffer.startswith(':'):
                self.chat_buffer = ""
                self._dirty = True
                return
            if getattr(self, 'palette_visible', False):
                self.palette_visible = False
                self._dirty = True
                return
            if getattr(self, 'focus', 'chat') == 'sidebar':
                self.focus = 'chat'
                self._dirty = True
                return
        # Chat input handling takes precedence on Chat page
        if self.page == "Chat":
            # Quit is via Ctrl+C or :exit; do not quit on 'q' to avoid interfering with chat
            # Command suggestions navigation (when starting with ':')
            if self.chat_buffer.startswith(':'):
                arrow = None
                if key in ("\x1b[A", "\x1bOA"): arrow = 'up'
                if key in ("\x1b[B", "\x1bOB"): arrow = 'down'
                if arrow == 'up' and self.cmd_suggestions:
                    self.cmd_sel = (self.cmd_sel - 1) % len(self.cmd_suggestions)
                    self._dirty = True
                    return
                if arrow == 'down' and self.cmd_suggestions:
                    self.cmd_sel = (self.cmd_sel + 1) % len(self.cmd_suggestions)
                    self._dirty = True
                    return
            # Submit on Enter
            if key in ('\r', '\n'):
                text = self.chat_buffer.strip()
                if text:
                    # If line starts with ':', treat as command (not chat)
                    if text.startswith(':'):
                        # Autocomplete behavior before executing
                        self._update_cmd_suggestions()
                        content = text[1:]
                        parts = content.split()
                        # If selecting a command token
                        if (not parts) or (len(parts) == 1 and not content.endswith(' ')):
                            if self.cmd_suggestions:
                                chosen = self.cmd_suggestions[self.cmd_sel]
                                # Replace first token fully and add space
                                self.chat_buffer = f":{chosen} "
                                self._dirty = True
                                return
                        # If selecting goto target
                        if parts and parts[0] == 'goto' and (len(parts) == 1 or (len(parts) >= 2 and not content.endswith(' '))):
                            if self.cmd_suggestions:
                                chosen = self.cmd_suggestions[self.cmd_sel]
                                prefix = parts[0]
                                self.chat_buffer = f":{prefix} {chosen}"
                                self._dirty = True
                                return
                        # Otherwise execute the command line sans ':'
                        await self._handle_command(content)
                    else:
                        self.chat_messages.append({"role": "user", "text": text})
                        if self.chat_handler is not None:
                            try:
                                resp = await self.chat_handler(text)
                                if resp:
                                    self.chat_messages.append({"role": "assistant", "text": resp})
                            except Exception as e:
                                self.chat_messages.append({"role": "assistant", "text": f"(error: {e})"})
                    self.chat_buffer = ""
                return
            # Backspace
            if key == '\x7f':
                self.chat_buffer = self.chat_buffer[:-1]
                return
            # Ignore arrows on chat for now (allow page nav via left/right)
            if key in ("\x1b[A", "\x1b[B", "\x1bOA", "\x1bOB"):
                return
            # Show/hide command palette with '?'
            if key == '?':
                self.palette_visible = not self.palette_visible
                return
            # Printable characters
            if len(key) == 1 and key >= ' ':
                self.chat_buffer += key
                if self.chat_buffer.startswith(':'):
                    self._update_cmd_suggestions()
                return
            # Left/Right navigate pages
            arrow = map_arrow(key)
            if arrow in ('left','right'):
                try:
                    idx = self.PAGES.index(self.page)
                    idx = idx - 1 if arrow == 'left' else idx + 1
                    if 0 <= idx < len(self.PAGES):
                        self.page = self.PAGES[idx]
                        self.status_message = f"Switched to {self.page}"
                except Exception:
                    pass
                return

        # Digits 1-9 switch pages; if on Chat and buffer is empty, allow quick jump
        if len(key) == 1 and key.isdigit() and (self.page != "Chat" or self.chat_buffer == ""):
            idx = int(key) - 1
            if 0 <= idx < len(self.PAGES):
                self.page = self.PAGES[idx]
                self.status_message = f"Switched to {self.page}"
            return
        # Quit
        if key.lower() == 'q':
            self.running = False
            return
        # Arrows for selection on Jobs/Models pages
        arrow = map_arrow(key)
        if arrow in ('up','down','left','right'):
            if self.page in ("Jobs", "Models"):
                sel_key = self.page
                cur = self.selection.get(sel_key, -1)
                delta = -1 if arrow == 'up' else (1 if arrow == 'down' else 0)
                new = cur + delta
                # Clamp to bounds if we have data (we don't know length here; optimistic)
                if new < 0:
                    new = 0
                self.selection[sel_key] = new
                self.status_message = f"Selected row {new+1}"
                return
            # Left/Right: page navigation
            if arrow == 'left':
                # left
                try:
                    idx = self.PAGES.index(self.page)
                    if idx > 0:
                        self.page = self.PAGES[idx - 1]
                        self.status_message = f"Switched to {self.page}"
                        self._dirty = True
                except Exception:
                    pass
                return
            if arrow == 'right':
                try:
                    idx = self.PAGES.index(self.page)
                    if idx < len(self.PAGES) - 1:
                        self.page = self.PAGES[idx + 1]
                        self.status_message = f"Switched to {self.page}"
                        self._dirty = True
                except Exception:
                    pass
                return
        # ':' is handled to open line input in key thread; ignore here
        # Other keys: ignore silently
