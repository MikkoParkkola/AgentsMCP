"""
RAG freshness monitoring CLI commands.

Provides user-friendly commands to check document freshness, perform cleanup,
view status, and start background monitoring for the AgentsMCP RAG system.
"""

from __future__ import annotations

import click
import threading
import time
import logging
from typing import Optional

from rich.console import Console
from rich.text import Text

# Graceful imports for when RAG is disabled
try:
    from agentsmcp.rag.client import RAGClient
    from agentsmcp.rag.monitoring import FreshnessMonitor, start_background_monitoring
    from agentsmcp.config import get_config
except ImportError:
    # Fallback stubs when RAG is not available
    class RAGClient:
        def __init__(self, config=None):
            self._config = type('Config', (), {'enabled': False})()
    
    class FreshnessMonitor:
        def __init__(self, *args, **kwargs):
            pass
        
        def check_freshness(self):
            return []
        
        def notify(self, items):
            pass
        
        def cleanup(self, items):
            pass
        
        def provide_status(self):
            pass
    
    def start_background_monitoring(*args, **kwargs):
        return None
    
    def get_config():
        return type("Config", (), {"rag": None})()

logger = logging.getLogger(__name__)


@click.group(name="rag-freshness")
@click.pass_context
def rag_freshness(ctx):
    """🔍 RAG knowledge base freshness monitoring and management."""
    console = Console()
    
    # Check if RAG is available and enabled
    try:
        config = get_config()
        if not config.rag or not config.rag.enabled:
            console.print("[yellow]⚠️  RAG is not enabled in configuration.[/yellow]")
            console.print("💡 Enable RAG in your configuration to use freshness monitoring.")
            ctx.exit(1)
    except Exception:
        console.print("[red]❌ Failed to load RAG configuration.[/red]")
        ctx.exit(1)


@rag_freshness.command()
@click.option(
    "--format", 
    "output_format",
    type=click.Choice(["table", "json", "summary"]),
    default="table",
    help="Output format for freshness check results"
)
def check(output_format: str):
    """🔍 Check for stale documents in the knowledge base."""
    console = Console()
    
    try:
        # Initialize RAG components
        rag_client = RAGClient()
        config = get_config()
        monitor = FreshnessMonitor(rag_client, config.rag.freshness_policy)
        
        with console.status("🔄 Scanning knowledge base for stale documents...", spinner="dots"):
            stale_items = monitor.check_freshness()
        
        if output_format == "table":
            monitor.notify(stale_items)
        elif output_format == "json":
            import json
            output = {
                "total_stale": len(stale_items),
                "ttl_days": getattr(config.rag.freshness_policy, 'ttl_days', 90),
                "stale_documents": [
                    {
                        "id": item["id"],
                        "source": item.get("source", "Unknown"),
                        "age_days": item["age_days"],
                        "created_at": item["created_at"].isoformat()
                    }
                    for item in stale_items
                ]
            }
            console.print_json(json.dumps(output, indent=2))
        elif output_format == "summary":
            ttl_days = getattr(config.rag.freshness_policy, 'ttl_days', 90)
            if stale_items:
                console.print(f"[yellow]⚠️  Found {len(stale_items)} stale documents (TTL: {ttl_days} days)[/yellow]")
            else:
                console.print(f"[green]✅ All documents are fresh (TTL: {ttl_days} days)[/green]")
        
    except Exception as exc:
        console.print(f"[red]❌ Freshness check failed: {exc}[/red]")
        raise click.ClickException(f"Failed to check freshness: {exc}")


@rag_freshness.command()
@click.option(
    "--force", 
    is_flag=True,
    help="Skip confirmation prompts and force cleanup"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be cleaned up without actually removing documents"
)
def cleanup(force: bool, dry_run: bool):
    """🗑️  Remove stale documents from the knowledge base."""
    console = Console()
    
    try:
        # Initialize RAG components
        rag_client = RAGClient()
        config = get_config()
        monitor = FreshnessMonitor(rag_client, config.rag.freshness_policy)
        
        with console.status("🔄 Identifying stale documents...", spinner="dots"):
            stale_items = monitor.check_freshness()
        
        if not stale_items:
            console.print("[green]✅ No stale documents found. Knowledge base is healthy![/green]")
            return
        
        # Show what will be cleaned up
        monitor.notify(stale_items)
        
        if dry_run:
            console.print(f"\n[cyan]🔍 Dry run: Would remove {len(stale_items)} stale documents[/cyan]")
            return
        
        # Handle auto-remove policy
        auto_remove = getattr(config.rag.freshness_policy, 'auto_remove_stale', False)
        if not auto_remove:
            console.print("\n[yellow]⚠️  Auto-remove is disabled in configuration.[/yellow]")
            console.print("💡 Enable auto_remove_stale in configuration or use --force flag")
            if not force:
                return
        
        # Override confirmation if force flag is used
        if force:
            # Temporarily override the confirmation setting
            original_confirmation = getattr(config.rag.freshness_policy, 'confirmation_prompt', True)
            config.rag.freshness_policy.confirmation_prompt = False
        
        try:
            monitor.cleanup(stale_items)
        finally:
            # Restore original confirmation setting if we overrode it
            if force:
                config.rag.freshness_policy.confirmation_prompt = original_confirmation
        
    except Exception as exc:
        console.print(f"[red]❌ Cleanup failed: {exc}[/red]")
        raise click.ClickException(f"Failed to cleanup: {exc}")


@rag_freshness.command()
@click.option(
    "--detailed",
    is_flag=True,
    help="Show detailed configuration and statistics"
)
def status(detailed: bool):
    """📊 Show RAG knowledge base freshness status and configuration."""
    console = Console()
    
    try:
        # Initialize RAG components
        rag_client = RAGClient()
        config = get_config()
        monitor = FreshnessMonitor(rag_client, config.rag.freshness_policy)
        
        if detailed:
            monitor.provide_status()
            
            # Additional detailed information
            console.print("\n🔧 [bold]Detailed Configuration[/bold]")
            console.print("─" * 50)
            
            freshness_config = config.rag.freshness_policy
            console.print(f"📋 [bold]Notification level:[/bold] {getattr(freshness_config, 'notification_level', 'warning')}")
            
            # Show vector store info if available
            try:
                chunks = monitor._list_chunks()
                if chunks:
                    oldest_chunk = min(chunks, key=lambda x: x.get('metadata', {}).get('created_at', '9999'))
                    newest_chunk = max(chunks, key=lambda x: x.get('metadata', {}).get('created_at', '0000'))
                    
                    console.print(f"📅 [bold]Oldest document:[/bold] {oldest_chunk.get('metadata', {}).get('source', 'Unknown')}")
                    console.print(f"📅 [bold]Newest document:[/bold] {newest_chunk.get('metadata', {}).get('source', 'Unknown')}")
            except Exception:
                pass
        else:
            # Quick status check
            with console.status("📊 Checking status...", spinner="dots"):
                stale_items = monitor.check_freshness()
                total_docs = len(monitor._list_chunks())
            
            ttl_days = getattr(config.rag.freshness_policy, 'ttl_days', 90)
            
            console.print(f"\n📊 [bold]Knowledge Base Status[/bold]")
            console.print(f"📑 Total documents: {total_docs}")
            console.print(f"🕐 TTL policy: {ttl_days} days")
            
            if stale_items:
                console.print(f"⚠️  [red]Stale documents: {len(stale_items)}[/red]")
                console.print("💡 Run 'agentsmcp rag-freshness cleanup' to remove stale documents")
            else:
                console.print("✅ [green]All documents are fresh[/green]")
        
    except Exception as exc:
        console.print(f"[red]❌ Status check failed: {exc}[/red]")
        raise click.ClickException(f"Failed to get status: {exc}")


@rag_freshness.command()
@click.option(
    "--interval",
    type=int,
    default=60,
    help="Monitoring interval in minutes (default: 60)"
)
@click.option(
    "--daemon",
    is_flag=True,
    help="Run as background daemon (detaches from terminal)"
)
def monitor(interval: int, daemon: bool):
    """👁️  Start continuous background monitoring of document freshness."""
    console = Console()
    
    if interval < 5:
        raise click.ClickException("Monitoring interval must be at least 5 minutes")
    
    try:
        # Initialize RAG components
        rag_client = RAGClient()
        config = get_config()
        
        console.print(f"🚀 [bold]Starting RAG freshness monitoring[/bold]")
        console.print(f"⏱️  Check interval: {interval} minutes")
        console.print(f"🕐 TTL policy: {getattr(config.rag.freshness_policy, 'ttl_days', 90)} days")
        
        if daemon:
            console.print("👻 [dim]Running in daemon mode (background)[/dim]")
            console.print("💡 Use Ctrl+C to stop monitoring")
        else:
            console.print("🔍 [dim]Interactive mode - status updates will be shown[/dim]")
            console.print("💡 Use Ctrl+C to stop monitoring")
        
        # Start background monitoring
        monitor_thread = start_background_monitoring(
            rag_client, 
            config.rag.freshness_policy, 
            interval_minutes=interval
        )
        
        if daemon:
            # In daemon mode, just start and detach
            console.print("✅ [green]Background monitoring started successfully[/green]")
            return
        else:
            # In interactive mode, show periodic status updates
            try:
                monitor_instance = FreshnessMonitor(rag_client, config.rag.freshness_policy)
                
                while True:
                    time.sleep(interval * 60)  # Convert to seconds
                    
                    try:
                        stale_items = monitor_instance.check_freshness()
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        if stale_items:
                            console.print(f"\n[yellow]⚠️  [{timestamp}] Found {len(stale_items)} stale documents[/yellow]")
                        else:
                            console.print(f"\n[green]✅ [{timestamp}] All documents are fresh[/green]")
                    except Exception as e:
                        console.print(f"\n[red]❌ [{time.strftime('%Y-%m-%d %H:%M:%S')}] Monitoring error: {e}[/red]")
                        
            except KeyboardInterrupt:
                console.print("\n🛑 [yellow]Monitoring stopped by user[/yellow]")
                console.print("✅ [green]Background monitoring thread will continue if started[/green]")
        
    except Exception as exc:
        console.print(f"[red]❌ Failed to start monitoring: {exc}[/red]")
        raise click.ClickException(f"Failed to start monitoring: {exc}")


# Register the command group (this will be imported by the main CLI)
__all__ = ["rag_freshness"]