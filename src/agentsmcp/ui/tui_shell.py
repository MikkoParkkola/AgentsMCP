from __future__ import annotations

import asyncio
from typing import Optional, Dict, Any, List

from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich import box

from .theme_manager import ThemeManager

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
        "Home",
        "Jobs",
        "Agents",
        "Models",
        "Providers",
        "Costs",
        "MCP",
        "Discovery",
        "Settings",
    ]

    def __init__(self, theme_manager: Optional[ThemeManager] = None):
        self.console = Console()
        self.theme_manager = theme_manager or ThemeManager()
        self.running = True
        self.page = "Home"
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

    # Rendering helpers
    def _render_header(self) -> Panel:
        title = f"[bold cyan]AgentsMCP[/] [magenta]•[/] [bold magenta]{self.page}[/]"
        style = "bold black on white" if self.high_contrast else "bold white on grey7"
        border = "white" if self.high_contrast else "cyan"
        return Panel(title, style=style, border_style=border, box=box.ROUNDED)

    def _render_sidebar(self) -> Panel:
        tbl = Table.grid(padding=(0, 1))
        for idx, it in enumerate(self.PAGES, start=1):
            hotkey = f"[{idx}]" if idx <= 9 else "   "
            color = "bold magenta" if it == self.page else ("bold white" if self.high_contrast else "white")
            tbl.add_row(f"[cyan]{hotkey}[/]  [{color}]{it}[/]")
        return Panel(tbl, title="[cyan]Menu", border_style=("white" if self.high_contrast else "blue"), box=box.ROUNDED)

    def _render_footer(self) -> Panel:
        hints = (
            "[cyan]1[/]Home [cyan]2[/]Jobs [cyan]3[/]Agents [cyan]4[/]Models [cyan]5[/]Providers "
            "[cyan]6[/]Costs [cyan]7[/]MCP [cyan]8[/]Discovery [cyan]9[/]Settings  •  "
            "[yellow]q[/] Quit  •  :help  •  :contrast on|off"
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
            "[cyan]Navigation[/]: 1–9 switch pages • q quit\n"
            "[cyan]Commands[/]:\n"
            "  :goto <page>  •  :help (toggle)  •  :contrast on|off\n"
            "  :select <n>  •  :cancel-selected  •  :cancel <job_id>\n"
            "  :set provider <name> api_base|api_key <val>\n"
            "  :set agent <name> model <model>\n"
            "  :models provider <name>  •  :models set <agent> <model>  •  :apply-model <agent>\n"
            "  :watch <job_id>  •  :watch-selected  •  :watch-pause  •  :watch-resume  •  :watch-clear  •  :watch-stop\n"
        )
        return Panel(text, title="[magenta]Help", border_style=("white" if self.high_contrast else "cyan"), box=box.ROUNDED)

    # Networking helpers
    async def _fetch_json(self, path: str, timeout: float = 2.0) -> Dict[str, Any]:
        if httpx is None:
            return {}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{self.base_url}{path}")
                if r.status_code == 200:
                    return r.json()
                return {"error": r.text, "status": r.status_code}
        except Exception:
            return {}

    async def _put_settings(self, body: Dict[str, Any]) -> bool:
        if httpx is None:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.put(f"{self.base_url}/settings", json=body)
                return r.status_code == 200
        except Exception:
            return False

    async def _delete_job(self, job_id: str) -> bool:
        if httpx is None:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.delete(f"{self.base_url}/jobs/{job_id}")
                return r.status_code == 200
        except Exception:
            return False

    async def _get_models(self, provider: str) -> List[Dict[str, Any]]:
        if httpx is None:
            return []
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/providers/{provider}/models")
                if r.status_code == 200:
                    j = r.json()
                    return list(j.get("models", []) or [])
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

        if self.page == "Home":
            stats = await self._fetch_json("/stats")
            body = self._fmt_kv({
                "active_jobs": stats.get("active_jobs"),
                "total_jobs": stats.get("total_jobs"),
                "agents": ", ".join(stats.get("agent_types", []) or []),
                "storage": stats.get("storage_type"),
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
                content = "\n".join(self.watch_lines[-200:]) if self.watch_lines else "(no output yet)"
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

        if self.page == "Models":
            models = self.models_cache.get(self.current_provider) or []
            if not models:
                body = "Use :models provider <name> to fetch models. Then :select <n> and :models set <agent> <model> or :apply-model <agent>."
                return Panel(body, title=f"[magenta]Models ({self.current_provider})", border_style="green", box=box.ROUNDED)
            tbl = Table("#", "id", "name", header_style="bold cyan")
            sel = self.selection.get("Models", -1)
            for i, m in enumerate(models[:200]):
                label = m.get("name") or m.get("id")
                style = "bold yellow" if i == sel else None
                tbl.add_row(str(i + 1), m.get("id", ""), label or "", style=style)
            hint = f"Provider: {self.current_provider} • :select <n> then :apply-model <agent>"
            return Panel(tbl, title=f"[magenta]Models • {hint}", border_style="green", box=box.ROUNDED)

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
            body = "Use :form provider <name> or :form agent <name> to edit."
            return Panel(body, title="[magenta]Settings", border_style="green", box=box.ROUNDED)

        return Panel("Work in progress", border_style="green", box=box.ROUNDED)

    # Input workers
    async def _input_worker(self) -> None:
        while self.running:
            try:
                s = await asyncio.to_thread(self.console.input, "")
                await self.input_queue.put(s.strip())
            except Exception:
                await asyncio.sleep(0.2)

    async def _handle_command(self, cmd: str) -> None:
        parts = cmd.split()
        if not parts:
            return
        try:
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