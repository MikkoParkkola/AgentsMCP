"""
RAG (Retrieval-Augmented Generation) CLI commands.

Provides CLI interface for knowledge base management operations.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from agentsmcp.rag import RAGClient, RAGError

# Import freshness commands
from .rag_freshness import rag_freshness

log = logging.getLogger(__name__)
console = Console()


@click.group(name="rag", help="📚 Manage RAG knowledge base for enhanced agent responses")
def rag_group():
    """RAG knowledge management commands."""
    pass


@rag_group.command(name="ingest", help="📥 Add documents to the knowledge base")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, help="Process directories recursively")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
def ingest_command(paths: tuple[str, ...], recursive: bool, verbose: bool) -> None:
    """Ingest documents into the RAG knowledge base.
    
    PATHS: One or more file or directory paths to ingest
    """
    if verbose:
        logging.getLogger("agentsmcp.rag").setLevel(logging.DEBUG)
    
    try:
        client = RAGClient()
        
        with console.status("[bold blue]Processing documents...") as status:
            # Convert paths to pathlib objects
            processed_paths = []
            for path_str in paths:
                path = pathlib.Path(path_str)
                if path.is_file():
                    processed_paths.append(path)
                elif path.is_dir() and recursive:
                    processed_paths.append(path)
                elif path.is_dir():
                    console.print(f"⚠️ Skipping directory {path} (use --recursive to include)")
                else:
                    console.print(f"⚠️ Skipping {path} (not found)")
            
            if not processed_paths:
                console.print("❌ No valid paths to process")
                return
            
            # Ingest documents
            client.ingest(processed_paths)
        
        console.print("✅ [bold green]Successfully ingested documents into knowledge base[/bold green]")
        
        # Show summary
        vectors = client.list()
        console.print(f"📊 Knowledge base now contains {len(vectors)} document chunks")
        
    except RAGError as e:
        console.print(f"❌ [bold red]RAG Error:[/bold red] {e}")
        console.print("💡 Hint: Enable RAG in configuration or run 'agentsmcp setup'")
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()


@rag_group.command(name="list", help="📋 List documents in the knowledge base")
@click.option("--limit", "-n", type=int, help="Limit number of results")
@click.option("--source", "-s", help="Filter by source file")
def list_command(limit: Optional[int], source: Optional[str]) -> None:
    """List all documents in the RAG knowledge base."""
    try:
        client = RAGClient()
        vectors = client.list()
        
        if not vectors:
            console.print("📭 Knowledge base is empty")
            console.print("💡 Use 'agentsmcp rag ingest <path>' to add documents")
            return
        
        # Filter by source if requested
        if source:
            vectors = [v for v in vectors if source in v.get("metadata", {}).get("source", "")]
        
        # Apply limit
        if limit:
            vectors = vectors[:limit]
        
        # Create table
        table = Table(title="📚 RAG Knowledge Base")
        table.add_column("ID", style="cyan")
        table.add_column("Source", style="green")
        table.add_column("Chunk", style="yellow") 
        table.add_column("Size", justify="right", style="blue")
        table.add_column("Modified", style="magenta")
        
        for vector in vectors:
            metadata = vector.get("metadata", {})
            vector_id = str(vector.get("id", "N/A"))
            source_path = metadata.get("source", "Unknown")
            chunk_nr = metadata.get("chunk_nr", 0)
            text = metadata.get("text", "")
            text_preview = (text[:50] + "...") if len(text) > 50 else text
            size = f"{len(text)} chars" if text else "N/A"
            
            # Format modification time
            mtime = metadata.get("mtime")
            if mtime:
                import datetime
                mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            else:
                mtime_str = "N/A"
            
            table.add_row(
                vector_id,
                pathlib.Path(source_path).name if source_path != "Unknown" else "Unknown",
                f"#{chunk_nr}: {text_preview}",
                size,
                mtime_str
            )
        
        console.print(table)
        
        if len(vectors) != len(client.list()):
            total = len(client.list())
            shown = len(vectors)
            console.print(f"\n📊 Showing {shown} of {total} total vectors")
            
    except RAGError as e:
        console.print(f"❌ [bold red]RAG Error:[/bold red] {e}")
        console.print("💡 Hint: Enable RAG in configuration or run 'agentsmcp setup'")
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@rag_group.command(name="search", help="🔍 Search the knowledge base")
@click.argument("query", required=True)
@click.option("--limit", "-k", default=5, type=int, help="Number of results to return")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed results")
def search_command(query: str, limit: int, verbose: bool) -> None:
    """Search for relevant documents in the knowledge base.
    
    QUERY: Search query text
    """
    try:
        client = RAGClient()
        
        with console.status("[bold blue]Searching knowledge base..."):
            results = client.search(query, k=limit)
        
        if not results:
            console.print(f"❌ No results found for: {query}")
            return
        
        console.print(f"🔍 [bold]Search results for:[/bold] {query}\n")
        
        for i, result in enumerate(results, 1):
            text = result.get("text", "")
            metadata = result.get("metadata", {})
            score = result.get("score", 0.0)
            
            source = metadata.get("source", "Unknown")
            chunk_nr = metadata.get("chunk_nr", 0)
            
            # Create panel for each result
            title = f"Result {i} - {pathlib.Path(source).name} (chunk #{chunk_nr})"
            score_text = f"Relevance: {score:.3f}"
            
            if verbose:
                content = f"[bold]Source:[/bold] {source}\n"
                content += f"[bold]Chunk:[/bold] #{chunk_nr}\n"
                content += f"[bold]Score:[/bold] {score:.3f}\n\n"
                content += text
            else:
                # Show preview
                preview = (text[:200] + "...") if len(text) > 200 else text
                content = f"{score_text}\n\n{preview}"
            
            panel = Panel(
                content,
                title=title,
                title_align="left",
                border_style="blue" if i == 1 else "dim"
            )
            console.print(panel)
            
    except RAGError as e:
        console.print(f"❌ [bold red]RAG Error:[/bold red] {e}")
        console.print("💡 Hint: Enable RAG in configuration or run 'agentsmcp setup'")
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@rag_group.command(name="remove", help="🗑️ Remove documents from knowledge base")
@click.argument("vector_ids", nargs=-1, required=True)
@click.option("--confirm", "-y", is_flag=True, help="Skip confirmation prompt")
def remove_command(vector_ids: tuple[str, ...], confirm: bool) -> None:
    """Remove vectors from the knowledge base.
    
    VECTOR_IDS: One or more vector IDs to remove (from 'rag list')
    """
    try:
        client = RAGClient()
        
        if not confirm:
            console.print(f"⚠️ About to remove {len(vector_ids)} vector(s): {', '.join(vector_ids)}")
            if not click.confirm("Continue?"):
                console.print("Cancelled")
                return
        
        removed_count = 0
        for vector_id in vector_ids:
            try:
                client.remove(vector_id)
                removed_count += 1
                console.print(f"✅ Removed vector {vector_id}")
            except Exception as e:
                console.print(f"❌ Failed to remove {vector_id}: {e}")
        
        console.print(f"\n📊 Successfully removed {removed_count}/{len(vector_ids)} vectors")
        
    except RAGError as e:
        console.print(f"❌ [bold red]RAG Error:[/bold red] {e}")
        console.print("💡 Hint: Enable RAG in configuration or run 'agentsmcp setup'")
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@rag_group.command(name="status", help="📊 Show RAG system status")
def status_command() -> None:
    """Show RAG system status and configuration."""
    try:
        client = RAGClient()
        vectors = client.list()
        
        # Create status panel
        status_text = Text()
        status_text.append("✅ RAG System: ", style="bold")
        status_text.append("ENABLED", style="bold green")
        status_text.append(f"\n📚 Documents: {len(vectors)} chunks")
        
        if vectors:
            # Get source statistics
            sources = {}
            for vector in vectors:
                source = vector.get("metadata", {}).get("source", "Unknown")
                sources[source] = sources.get(source, 0) + 1
            
            status_text.append(f"\n📁 Sources: {len(sources)} files")
            
            # Show top sources
            top_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)[:3]
            for source, count in top_sources:
                filename = pathlib.Path(source).name if source != "Unknown" else "Unknown"
                status_text.append(f"\n  • {filename}: {count} chunks")
            
            if len(sources) > 3:
                status_text.append(f"\n  • ... and {len(sources) - 3} more")
        
        console.print(Panel(
            status_text,
            title="📊 RAG Status",
            border_style="green"
        ))
        
    except RAGError as e:
        # RAG is disabled
        status_text = Text()
        status_text.append("❌ RAG System: ", style="bold")
        status_text.append("DISABLED", style="bold red")
        status_text.append("\n\n💡 To enable RAG:")
        status_text.append("\n  1. Run 'agentsmcp setup' for interactive configuration")
        status_text.append("\n  2. Or manually edit your configuration file")
        
        console.print(Panel(
            status_text,
            title="📊 RAG Status",
            border_style="red"
        ))
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@rag_group.command(name="clear", help="🧹 Clear all documents from knowledge base")
@click.option("--confirm", "-y", is_flag=True, help="Skip confirmation prompt")
def clear_command(confirm: bool) -> None:
    """Clear all documents from the knowledge base."""
    try:
        client = RAGClient()
        vectors = client.list()
        
        if not vectors:
            console.print("📭 Knowledge base is already empty")
            return
        
        if not confirm:
            console.print(f"⚠️ About to remove ALL {len(vectors)} vectors from knowledge base")
            console.print("⚠️ This action cannot be undone!")
            if not click.confirm("Continue?"):
                console.print("Cancelled")
                return
        
        removed_count = 0
        with console.status("[bold red]Clearing knowledge base..."):
            for vector in vectors:
                try:
                    client.remove(vector.get("id"))
                    removed_count += 1
                except Exception as e:
                    log.warning(f"Failed to remove vector {vector.get('id')}: {e}")
        
        console.print(f"✅ [bold green]Cleared {removed_count}/{len(vectors)} vectors[/bold green]")
        console.print("📭 Knowledge base is now empty")
        
    except RAGError as e:
        console.print(f"❌ [bold red]RAG Error:[/bold red] {e}")
        console.print("💡 Hint: Enable RAG in configuration or run 'agentsmcp setup'")
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


# Add freshness commands as a subcommand group
rag_group.add_command(rag_freshness)
