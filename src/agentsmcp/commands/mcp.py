from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from ..config import Config, MCPServerConfig
from ..settings import AppSettings


def _load_config(config_path: Optional[str]) -> Config:
    env = AppSettings()
    base: Config
    if config_path and Path(config_path).exists():
        base = Config.from_file(Path(config_path))
    elif Path("agentsmcp.yaml").exists():
        base = Config.from_file(Path("agentsmcp.yaml"))
    else:
        base = Config()
    return env.to_runtime_config(base)


def _save_config(cfg: Config, config_path: Optional[str]) -> Path:
    path = Path(config_path) if config_path else Path("agentsmcp.yaml")
    cfg.save_to_file(path)
    return path


@click.group()
def mcp() -> None:
    """Manage MCP servers in configuration."""


@mcp.command("list")
@click.option("--config", "config_path", default=None, help="Path to YAML config")
def mcp_list(config_path: Optional[str]) -> None:
    cfg = _load_config(config_path)
    rows = []
    for s in cfg.mcp or []:
        rows.append(
            f"- {s.name} enabled={s.enabled} transport={s.transport or 'stdio'} "
            f"command={s.command or ''} url={s.url or ''}"
        )
    if not rows:
        click.echo("No MCP servers configured.")
        return
    click.echo("Configured MCP servers:\n" + "\n".join(rows))


@mcp.command("enable")
@click.argument("name")
@click.option("--config", "config_path", default=None, help="Path to YAML config")
def mcp_enable(name: str, config_path: Optional[str]) -> None:
    cfg = _load_config(config_path)
    found = False
    for s in cfg.mcp or []:
        if s.name == name:
            s.enabled = True
            found = True
            break
    if not found:
        raise click.ClickException(f"MCP server not found: {name}")
    path = _save_config(cfg, config_path)
    click.echo(f"Enabled MCP server '{name}' in {path}")


@mcp.command("disable")
@click.argument("name")
@click.option("--config", "config_path", default=None, help="Path to YAML config")
def mcp_disable(name: str, config_path: Optional[str]) -> None:
    cfg = _load_config(config_path)
    found = False
    for s in cfg.mcp or []:
        if s.name == name:
            s.enabled = False
            found = True
            break
    if not found:
        raise click.ClickException(f"MCP server not found: {name}")
    path = _save_config(cfg, config_path)
    click.echo(f"Disabled MCP server '{name}' in {path}")


@mcp.command("add")
@click.argument("name")
@click.option("--transport", default="stdio", type=click.Choice(["stdio", "sse", "websocket"]))
@click.option("--command", multiple=True, help="Command (space-separated). Repeat to pass args.")
@click.option("--url", default=None, help="URL for sse/websocket transports")
@click.option("--config", "config_path", default=None, help="Path to YAML config")
def mcp_add(
    name: str,
    transport: str,
    command: tuple[str, ...],
    url: Optional[str],
    config_path: Optional[str],
) -> None:
    cfg = _load_config(config_path)
    if any(s.name == name for s in cfg.mcp or []):
        raise click.ClickException(f"MCP server already exists: {name}")
    server = MCPServerConfig(name=name, transport=transport, command=list(command) or None, url=url, enabled=True)
    cfg.mcp.append(server)  # type: ignore[arg-type]
    path = _save_config(cfg, config_path)
    click.echo(f"Added MCP server '{name}' to {path}")


@mcp.command("remove")
@click.argument("name")
@click.option("--config", "config_path", default=None, help="Path to YAML config")
def mcp_remove(name: str, config_path: Optional[str]) -> None:
    cfg = _load_config(config_path)
    before = len(cfg.mcp or [])
    cfg.mcp = [s for s in (cfg.mcp or []) if s.name != name]  # type: ignore[assignment]
    after = len(cfg.mcp or [])
    if before == after:
        raise click.ClickException(f"MCP server not found: {name}")
    path = _save_config(cfg, config_path)
    click.echo(f"Removed MCP server '{name}' from {path}")


@mcp.command("serve")
@click.option("--stdio", is_flag=True, default=True, help="Serve over stdio (default)")
@click.option("--config", "config_path", default=None, help="Path to YAML config")
def mcp_serve(stdio: bool, config_path: Optional[str]) -> None:
    """Run AgentsMCP as an MCP gateway server with version negotiation."""
    cfg = _load_config(config_path)
    if stdio:
        from ..mcp.server import run_stdio_blocking
        run_stdio_blocking(cfg)
    else:
        raise click.ClickException("Only stdio transport is currently wired in this CLI.")
