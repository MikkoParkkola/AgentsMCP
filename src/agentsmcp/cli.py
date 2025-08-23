#!/usr/bin/env python3

import asyncio
import click
from pathlib import Path
from typing import Optional

from .config import Config
from .server import AgentServer
from .agent_manager import AgentManager


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.pass_context
def cli(ctx, config: Optional[str]):
    """AgentsMCP - CLI-driven MCP agent system with extensible RAG pipeline."""
    ctx.ensure_object(dict)
    
    if config:
        config_path = Path(config)
    else:
        # Default config locations
        for default_path in [Path.cwd() / "agentsmcp.yaml", Path.home() / ".config" / "agentsmcp" / "config.yaml"]:
            if default_path.exists():
                config_path = default_path
                break
        else:
            config_path = None
    
    if config_path:
        ctx.obj['config'] = Config.from_file(config_path)
    else:
        ctx.obj['config'] = Config()


@cli.command()
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=8000, type=int, help='Server port')
@click.pass_context
def serve(ctx, host: str, port: int):
    """Start the AgentsMCP server."""
    config = ctx.obj['config']
    config.server.host = host
    config.server.port = port
    
    server = AgentServer(config)
    asyncio.run(server.start())


@cli.command()
@click.argument('agent_type')
@click.argument('task')
@click.option('--timeout', default=300, type=int, help='Task timeout in seconds')
@click.pass_context
def spawn(ctx, agent_type: str, task: str, timeout: int):
    """Spawn an agent to handle a specific task."""
    config = ctx.obj['config']
    
    async def run_agent():
        agent_manager = AgentManager(config)
        job_id = await agent_manager.spawn_agent(agent_type, task, timeout)
        click.echo(f"Agent spawned with job ID: {job_id}")
        
        # Wait for completion and show result
        result = await agent_manager.wait_for_completion(job_id)
        if result.success:
            click.echo(f"Task completed successfully: {result.output}")
        else:
            click.echo(f"Task failed: {result.error}", err=True)
    
    asyncio.run(run_agent())


@cli.command()
@click.argument('job_id')
@click.pass_context
def status(ctx, job_id: str):
    """Check the status of a running agent job."""
    config = ctx.obj['config']
    
    async def check_status():
        agent_manager = AgentManager(config)
        status = await agent_manager.get_job_status(job_id)
        
        click.echo(f"Job {job_id}: {status.state}")
        if status.output:
            click.echo(f"Output: {status.output}")
        if status.error:
            click.echo(f"Error: {status.error}", err=True)
    
    asyncio.run(check_status())


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize a new AgentsMCP configuration."""
    config_path = Path.cwd() / "agentsmcp.yaml"
    
    if config_path.exists():
        click.echo(f"Configuration already exists at {config_path}")
        return
    
    # Create default configuration
    config = Config()
    config.save_to_file(config_path)
    click.echo(f"Configuration created at {config_path}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()"""CLI entry point for AgentsMCP."""

import click

from . import __version__
from .placeholder import add as add_fn


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="agentsmcp")
def main() -> None:
    """AgentsMCP CLI."""
    # This function is the console script entry point.
    # Subcommands are defined below.
    pass


@main.command("add")
@click.argument("a", type=int)
@click.argument("b", type=int)
def add_cmd(a: int, b: int) -> None:
    """Add two integers and print the result."""
    click.echo(str(add_fn(a, b)))
