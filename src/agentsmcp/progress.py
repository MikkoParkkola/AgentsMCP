"""Progress display utilities for AgentsMCP CLI/TUI.

Provides a reusable :class:`ProgressDisplay` that wraps ``rich.progress.Progress``
and optionally shows percentage, ETA, and arbitrary KPI counters.

The component reads a simple YAML configuration file ``tui_progress.yaml``
located at the repository root.  The config keys are:

- ``show_percentage`` (bool) – show a visual progress bar and percentage.
- ``show_eta`` (bool) – display an estimated time‑of‑arrival based on elapsed time.
- ``show_kpis`` (bool) – render a table of custom key‑performance‑indicators.

If the config file is missing, all three options default to ``True``.

Typical usage::

    from agentsmcp.progress import ProgressDisplay

    prog = ProgressDisplay()
    prog.start(total=items_count, description="Processing items")
    for item in items:
        # ... do work ...
        prog.update(1, errors=error_cnt, throughput=rate)
    prog.finish()

The class is deliberately lightweight and does not depend on any global state,
so it can be instantiated in both the legacy Click‑based CLI and the modern
prompt‑toolkit UI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any

import yaml
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = {
    "show_percentage": True,
    "show_eta": True,
    "show_kpis": True,
}

def _load_config() -> Dict[str, bool]:
    """Load ``tui_progress.yaml`` from the project root.

    Returns a dictionary with boolean values for the three supported keys.
    Missing file or malformed content falls back to the defaults.
    """
    cfg_path = Path("tui_progress.yaml")
    if not cfg_path.is_file():
        return _DEFAULT_CONFIG.copy()
    try:
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        return {
            "show_percentage": bool(raw.get("show_percentage", True)),
            "show_eta": bool(raw.get("show_eta", True)),
            "show_kpis": bool(raw.get("show_kpis", True)),
        }
    except Exception:
        # If anything goes wrong, use defaults – we never want the UI to crash.
        return _DEFAULT_CONFIG.copy()

# ---------------------------------------------------------------------------
# ProgressDisplay implementation
# ---------------------------------------------------------------------------
class ProgressDisplay:
    """A thin wrapper around :class:`rich.progress.Progress`.

    The class tracks total work, elapsed time, and arbitrary KPI counters.
    It renders a progress bar (if enabled) and a small table of KPIs beneath it.
    """

    def __init__(self, description: str = "") -> None:
        self.description = description
        self._config = _load_config()
        self._total: int = 0
        self._completed: int = 0
        self._start_time: float | None = None
        self._kpis: Dict[str, Any] = {}
        self._console = Console()
        self._progress: Progress | None = None
        self._task_id: Any = None

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    def start(self, total: int, description: str | None = None) -> None:
        """Initialize the progress display.

        Parameters
        ----------
        total: int
            The total number of steps/items expected.
        description: str, optional
            Override the description set at construction time.
        """
        self._total = max(total, 0)
        self._completed = 0
        self._kpis.clear()
        self._start_time = time.time()
        if description:
            self.description = description

        if self._config["show_percentage"]:
            # Build a Progress instance with optional ETA columns.
            columns = [TextColumn(self.description)]
            columns.append(BarColumn(bar_width=None))
            columns.append(TextColumn("[progress.percentage]{task.percentage:>3.0f}%"))
            if self._config["show_eta"]:
                columns.append(TimeElapsedColumn())
                columns.append(TimeRemainingColumn())
            self._progress = Progress(*columns, console=self._console, transient=True)
            self._task_id = self._progress.add_task(self.description, total=self._total)
            self._progress.start()
        else:
            # No visual bar – just print a start line.
            self._console.print(f"[bold]{self.description}[/bold] – starting...")

    def update(self, count: int = 1, **kpis: Any) -> None:
        """Advance the progress by *count* steps and merge KPI values.

        ``**kpis`` can be any key/value pairs you wish to track (e.g.
        ``errors=2, throughput=150``).  Numeric values are summed; other
        values are overwritten.
        """
        self._completed += count
        # Merge KPI values – numeric values are summed.
        for key, value in kpis.items():
            if isinstance(value, (int, float)):
                self._kpis[key] = self._kpis.get(key, 0) + value
            else:
                self._kpis[key] = value

        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, advance=count)
        else:
            # Fallback simple textual update.
            self._console.print(
                f"[green]✔[/green] {self._completed}/{self._total} – "
                + ", ".join(f"{k}: {v}" for k, v in self._kpis.items())
            )

    def finish(self, success: bool = True) -> None:
        """Mark the operation as complete and render final KPI table.
        """
        elapsed = time.time() - (self._start_time or time.time())
        # Ensure the progress bar is stopped.
        if self._progress:
            self._progress.stop()
        # Render final summary.
        status = "[bold green]SUCCESS[/bold green]" if success else "[bold red]FAILED[/bold red]"
        self._console.print(f"{status} – elapsed: {elapsed:.2f}s")
        if self._config["show_kpis"] and self._kpis:
            table = Table(title="Key Performance Indicators")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="magenta")
            for key, value in self._kpis.items():
                table.add_row(str(key), str(value))
            self._console.print(table)

    # -------------------------------------------------------------------
    # Helper properties (optional, for external introspection)
    # -------------------------------------------------------------------
    @property
    def completed(self) -> int:
        return self._completed

    @property
    def total(self) -> int:
        return self._total

    @property
    def kpis(self) -> Dict[str, Any]:
        return dict(self._kpis)
