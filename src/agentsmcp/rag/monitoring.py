"""
agentsmcp.rag.monitoring

Comprehensive RAG freshness monitoring.

* Detects stale documents based on a TTL configured in
  :class:`agentsmcp.config.RAGFreshnessPolicyConfig`.
* Notifies the user via Rich tables and colored messages.
* Automatically removes stale documents if the policy allows it.
* Can be run once or in a background daemon that re‑checks at a set interval.

The module is intentionally written to be tolerant if any of the
expected RAG components are missing (e.g. when RAG is disabled).
"""

from __future__ import annotations

import datetime as _dt
import time as _time
import threading
import logging
import json
from typing import Iterable, List, Dict, Any, Optional

# --------------------------------------------------------------------------- #
# Imports that might be missing when RAG is disabled
# --------------------------------------------------------------------------- #
try:
    from agentsmcp.config import get_config
    from agentsmcp.rag.client import RAGClient
except Exception:
    # ----------------------------------------------------------------------- #
    # Fallback stubs used when the real modules cannot be imported.
    # ----------------------------------------------------------------------- #
    class RAGClient:  # pragma: no cover
        def __init__(self, config: Any = None):
            self.vector_store = None
            self._config = type('Config', (), {'enabled': False})()

    def get_config():  # pragma: no cover
        return type("Config", (), {"rag": None})()

# --------------------------------------------------------------------------- #
# Rich imports
# --------------------------------------------------------------------------- #
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn
from rich.text import Text
from rich import box
import click

# --------------------------------------------------------------------------- #
# Logger
# --------------------------------------------------------------------------- #
_logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _parse_timestamp(value: Any) -> Optional[_dt.datetime]:
    """
    Try to parse a timestamp from one of the common formats:
    * ISO 8601 string
    * ``YYYY-MM-DD HH:MM:SS.FFF`` string
    * POSIX epoch (int/float)
    * ``datetime.datetime`` instance
    """
    if isinstance(value, _dt.datetime):
        return value

    if isinstance(value, (int, float)):
        try:
            return _dt.datetime.fromtimestamp(value, tz=_dt.timezone.utc)
        except Exception:
            return None

    if isinstance(value, str):
        # ISO 8601
        try:
            return _dt.datetime.fromisoformat(value)
        except ValueError:
            pass

        # Common fallback
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = _dt.datetime.strptime(value, fmt)
                return dt.replace(tzinfo=_dt.timezone.utc)
            except ValueError:
                continue

    return None


# --------------------------------------------------------------------------- #
# Core monitoring class
# --------------------------------------------------------------------------- #
class FreshnessMonitor:
    """
    Helper class that checks document age, reports stale docs,
    and optionally removes them.

    Parameters
    ----------
    rag_client:
        Instance of :class:`~agentsmcp.rag.client.RAGClient` that owns the
        underlying vector store.
    policy_config:
        Configuration object with ttl_days, notification_level, auto_remove_stale, confirmation_prompt.
    """

    def __init__(
        self,
        rag_client: RAGClient,
        policy_config: Any,
    ) -> None:
        self.client = rag_client
        self.policy = policy_config
        self.console = Console()

    # --------------------------------------------------------------------- #
    # Retrieval helpers (adapt to whatever the vector store returns)
    # --------------------------------------------------------------------- #
    def _list_chunks(self) -> List[Dict[str, Any]]:
        """
        Return list of dictionary objects each representing a chunk
        stored in the vector store.

        The method is tolerant and simply returns an empty list if the
        client or vector store does not expose the expected interface.
        """
        try:
            # Check if RAG is enabled first
            if not getattr(self.client._config, 'enabled', False):
                return []

            # Try to get documents from the RAG client
            if hasattr(self.client, 'list'):
                return self.client.list()

            # Try vector store directly
            if hasattr(self.client, '_vector_store') and self.client._vector_store:
                vector_store = self.client._vector_store
                if hasattr(vector_store, "list"):
                    return vector_store.list()

            return []

        except Exception as exc:
            _logger.exception(f"Unable to retrieve chunks from vector store: {exc}")
            return []

    # --------------------------------------------------------------------- #
    # Freshness checks
    # --------------------------------------------------------------------- #
    def check_freshness(self) -> List[Dict[str, Any]]:
        """
        Scan all stored chunks and return a list of
        dictionaries for those that have exceeded the configured TTL.

        Each dictionary contains:
            * ``id``          – chunk identifier
            * ``created_at``  – parsed ``datetime`` object
            * ``age_days``    – number of days the chunk has been present
        """
        ttl_days = getattr(self.policy, 'ttl_days', 90)
        if ttl_days <= 0:
            # TTL of 0 means never stale
            return []

        now = _dt.datetime.now(tz=_dt.timezone.utc)
        ttl_delta = _dt.timedelta(days=ttl_days)
        stale_info: List[Dict[str, Any]] = []

        for chunk in self._list_chunks():
            # Expect the timestamp in ``metadata.mtime`` or ``metadata.created_at``
            meta = chunk.get("metadata") or {}
            created_at_raw = (
                meta.get("created_at") 
                or meta.get("mtime") 
                or meta.get("timestamp") 
                or chunk.get("created_at")
            )
            if not created_at_raw:
                continue

            created_at = _parse_timestamp(created_at_raw)
            if created_at is None:
                continue

            age = now - created_at
            if age > ttl_delta:
                stale_info.append(
                    {
                        "id": chunk.get("id"),
                        "created_at": created_at,
                        "age_days": int(age.days),
                        "source": meta.get("source", "Unknown"),
                    }
                )

        return stale_info

    # --------------------------------------------------------------------- #
    # User notifications
    # --------------------------------------------------------------------- #
    def notify(self, stale_items: Iterable[Dict[str, Any]]) -> None:
        """
        Pretty‑print a table of stale items.

        If no items are stale a friendly "all fresh" message is shown.
        """
        stale_list = list(stale_items)
        if not stale_list:
            self.console.print("[bold green]✅ All documents are fresh![/bold green]")
            return

        table = Table(
            title="[bold yellow]📅 Stale Documents[/bold yellow]",
            box=box.ROUNDED,
            header_style="bold magenta",
            show_lines=True,
        )
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Source", style="blue")
        table.add_column("Created At", style="magenta")
        table.add_column("Age (days)", style="yellow", justify="right")

        for item in stale_list:
            import pathlib
            source_name = pathlib.Path(item.get("source", "Unknown")).name
            table.add_row(
                str(item["id"]),
                source_name,
                item["created_at"].strftime("%Y-%m-%d %H:%M UTC"),
                str(item["age_days"]),
            )

        self.console.print(table)

        # Add a severity banner if required
        level = getattr(self.policy, 'notification_level', 'info')
        if level and level.lower() in ("warning", "critical", "error"):
            ttl_days = getattr(self.policy, 'ttl_days', 90)
            banner = Text(
                f"⚠️  {len(stale_list)} stale document(s) exceed TTL of {ttl_days} days.", 
                style=f"bold {level.lower()}"
            )
            self.console.print(banner)

    # --------------------------------------------------------------------- #
    # Cleanup handling
    # --------------------------------------------------------------------- #
    def _delete_chunk(self, chunk_id: Any) -> bool:
        """
        Delete a chunk by ID.  Returns ``True`` on success,
        ``False`` otherwise.
        """
        try:
            if hasattr(self.client, 'remove'):
                self.client.remove(chunk_id)
                return True
            elif hasattr(self.client, '_vector_store') and self.client._vector_store:
                if hasattr(self.client._vector_store, 'delete'):
                    self.client._vector_store.delete(chunk_id)
                    return True
            return False
        except Exception as exc:
            _logger.exception(f"Failed to delete chunk {chunk_id!s}: {exc}")
            return False

    def cleanup(self, stale_items: Iterable[Dict[str, Any]]) -> None:
        """
        Remove stale documents if the policy allows it.

        If ``auto_remove_stale`` is ``False`` the method just
        prints a message and returns.

        When ``confirmation_prompt`` is enabled, the user will be asked
        to confirm the deletion.
        """
        stale_list = list(stale_items)
        if not stale_list:
            self.console.print("[bold green]✅ No stale documents to remove.[/bold green]")
            return

        auto_remove = getattr(self.policy, 'auto_remove_stale', False)
        if not auto_remove:
            self.console.print("[bold yellow]⚠️  Auto‑remove is disabled. Skipping cleanup.[/bold yellow]")
            self.console.print("💡 Enable auto_remove_stale in configuration or use 'agentsmcp rag remove <id>' manually")
            return

        confirmation_prompt = getattr(self.policy, 'confirmation_prompt', True)
        msg = f"Delete {len(stale_list)} stale document(s)? This action cannot be undone."
        if confirmation_prompt:
            if not click.confirm(msg, default=False):
                self.console.print("[bold]🚫 Cleanup cancelled.[/bold]")
                return

        # Progress bar
        task_desc = Text("🗑️  Removing stale documents…", style="cyan")
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(str(task_desc), total=len(stale_list))

            removed = 0
            for item in stale_list:
                if self._delete_chunk(item["id"]):
                    removed += 1
                progress.update(task, advance=1)

        if removed == len(stale_list):
            self.console.print(
                Text(f"✅ Successfully removed all {removed} stale documents.", style="green")
            )
        else:
            self.console.print(
                Text(f"⚠️  Removed {removed} of {len(stale_list)} stale documents.", style="yellow")
            )

    # --------------------------------------------------------------------- #
    # High‑level status
    # --------------------------------------------------------------------- #
    def provide_status(self) -> None:
        """
        Human‑readable report of the freshness health.
        """
        ttl_days = getattr(self.policy, 'ttl_days', 90)
        auto_remove = getattr(self.policy, 'auto_remove_stale', False)
        confirmation = getattr(self.policy, 'confirmation_prompt', True)
        
        stale = self.check_freshness()
        
        self.console.print("\n📊 [bold]RAG Knowledge Base Freshness Status[/bold]")
        self.console.print("─" * 50)
        self.console.print(f"🕐 [bold]TTL Policy:[/bold] {ttl_days} days")
        self.console.print(f"🔄 [bold]Auto-remove:[/bold] {'✅ Enabled' if auto_remove else '❌ Disabled'}")
        self.console.print(f"❓ [bold]Confirmation prompts:[/bold] {'✅ Enabled' if confirmation else '❌ Disabled'}")
        self.console.print(f"📑 [bold]Total documents:[/bold] {len(self._list_chunks())}")
        
        if stale:
            self.console.print(f"⚠️  [bold red]Stale documents:[/bold red] {len(stale)}")
            self.console.print()
            self.notify(stale)
        else:
            self.console.print(f"✅ [bold green]Stale documents:[/bold green] 0")
            self.console.print()
            self.console.print("[bold green]🎉 Knowledge base is healthy![/bold green]")


# --------------------------------------------------------------------------- #
# Background monitoring helper
# --------------------------------------------------------------------------- #
def start_background_monitoring(
    rag_client: RAGClient,
    policy_config: Any,
    interval_minutes: int = 60,
) -> threading.Thread:
    """
    Spawn a daemon thread that periodically runs a full freshness check,
    notifies the user and deletes stale items if configured.

    Returns the thread object; it is a daemon thread, so it will not
    keep the process alive when the main thread finishes.
    """
    monitor = FreshnessMonitor(rag_client, policy_config)

    def loop() -> None:
        while True:
            try:
                stale = monitor.check_freshness()
                if stale:
                    monitor.notify(stale)
                    monitor.cleanup(stale)
                else:
                    # Periodic "all good" message
                    _logger.info("Background freshness check: all documents are fresh")
            except Exception:  # pragma: no cover - hard to inject here
                _logger.exception("Background freshness monitoring failed")
            _time.sleep(interval_minutes * 60)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t