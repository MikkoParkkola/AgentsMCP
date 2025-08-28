"""Ultra-fast CLI entry point optimized for ≤0.20s --help performance."""

from __future__ import annotations

import sys
from typing import Optional

import click

# Minimal imports - only version info
from agentsmcp import __version__


def _delegate_to_full_cli():
    """Efficiently delegate to full CLI for actual command execution."""
    from agentsmcp.cli import main as full_main
    # Pass through all arguments to preserve behavior
    full_main()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="agentsmcp")
@click.option("--log-level", default=None, help="Log level (override env)")
@click.option("--log-format", default=None, type=click.Choice(["json", "text"]))
@click.option("--config", "config_path", default=None, help="Path to YAML config")
@click.pass_context
def main(ctx, log_level, log_format, config_path) -> None:
    """AgentsMCP - Revolutionary Multi-Agent Orchestration with Cost Intelligence."""
    if ctx.invoked_subcommand is None:
        return
    _delegate_to_full_cli()


# Minimal command stubs for help display - Click handles the rest
@main.command("simple")
def simple():
    """Execute a task using simple orchestration (recommended default)."""
    _delegate_to_full_cli()


@main.command("config") 
def config_cmd():
    """Configure AgentsMCP orchestration and model preferences."""
    _delegate_to_full_cli()


@main.command("interactive")
def interactive():
    """Launch enhanced interactive CLI with AI chat, agent orchestration, and task delegation."""
    _delegate_to_full_cli()


@main.command("dashboard")
def dashboard():
    """Launch real-time dashboard with cost monitoring and agent orchestration."""
    _delegate_to_full_cli()


@main.command("costs")
def costs():
    """Display current costs and budget status with beautiful formatting."""
    _delegate_to_full_cli()


@main.command("budget")
def budget():
    """Manage monthly budget and get cost alerts."""
    _delegate_to_full_cli()


@main.command("models")
def models():
    """Show available orchestrator models and recommendations."""
    _delegate_to_full_cli()


@main.command("optimize")
def optimize():
    """Optimize model selection for cost-effectiveness."""
    _delegate_to_full_cli()


@main.group("server")
def server():
    """Manage the API server."""
    _delegate_to_full_cli()


@main.group("mcp") 
def mcp():
    """Manage MCP servers in configuration."""
    _delegate_to_full_cli()


@main.group("roles")
def roles():
    """🛠️ Role‑based orchestration commands."""
    _delegate_to_full_cli()


@main.group("rag")
def rag():
    """📚 Manage RAG knowledge base for enhanced agent responses"""
    _delegate_to_full_cli()


@main.command("setup")
def setup():
    """🛠️ Interactive wizard that creates a valid AgentsMCP configuration."""
    _delegate_to_full_cli()


if __name__ == "__main__":
    main()