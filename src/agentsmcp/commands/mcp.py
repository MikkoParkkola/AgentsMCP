from __future__ import annotations
from pathlib import Path
from typing import Optional

import click

from ..config import Config, MCPServerConfig
# Import from settings.py file directly, not the settings/ directory
import agentsmcp.settings as settings_module
from ..mcp.manager import MCPServer as _M, get_global_manager as _get
import json


def _load_config(config_path: Optional[str]) -> Config:
    env = settings_module.AppSettings()
    base: Config
    if config_path and Path(config_path).exists():
        base = Config.from_file(Path(config_path))
    elif Path("agentsmcp.yaml").exists():
        base = Config.from_file(Path("agentsmcp.yaml"))
    else:
        base = Config()
    return env.to_runtime_config(base)


def _save_config(cfg: Config, config_path: Optional[str]) -> Path:
    # Default to per-user config under ~/.agentsmcp
    default_path = Config.default_config_path()
    path = Path(config_path) if config_path else default_path
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


@mcp.command("warmup")
@click.option("--timeout", default=20, help="Per-server timeout (seconds)")
def mcp_warmup(timeout: int) -> None:
    """Pre-fetch common MCP server packages via npx to reduce first-use latency.

    This is optional and safe to run repeatedly. Network access required.
    """
    import shutil
    import subprocess

    servers = [
        ("@anthropic/mcp-github", ["npx", "-y", "@anthropic/mcp-github", "--help"]),
        ("@anthropic/mcp-filesystem", ["npx", "-y", "@anthropic/mcp-filesystem", "--help"]),
        ("@anthropic/mcp-git", ["npx", "-y", "@anthropic/mcp-git", "--help"]),
        ("@anthropic/mcp-ollama", ["npx", "-y", "@anthropic/mcp-ollama", "--help"]),
        ("@anthropic/mcp-ollama-turbo", ["npx", "-y", "@anthropic/mcp-ollama-turbo", "--help"]),
    ]

    if not shutil.which("npx"):
        click.echo("npx not found; skipping MCP warmup.")
        return

    ok = 0
    for name, cmd in servers:
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
            ok += 1
        except Exception:
            click.echo(f"Skipped {name} (timeout or error)")
    click.echo(f"MCP warmup completed: {ok}/{len(servers)} cached")


@mcp.command("status")
@click.option("--config", "config_path", default=None, help="Path to YAML config")
def mcp_status(config_path: Optional[str]) -> None:
    """Show MCP manager status as JSON."""
    cfg = _load_config(config_path)
    servers = []
    for s in (cfg.mcp or []):
        servers.append(_M(name=s.name, command=s.command, transport=s.transport, url=s.url, env=s.env or {}, cwd=s.cwd, enabled=s.enabled))
    mgr = _get(
        servers,
        allow_stdio=bool(getattr(cfg, "mcp_stdio_enabled", True)),
        allow_ws=bool(getattr(cfg, "mcp_ws_enabled", False)),
        allow_sse=bool(getattr(cfg, "mcp_sse_enabled", False)),
    )
    import asyncio as _asyncio
    async def _run():
        st = await mgr.get_status()
        import json as _json
        click.echo(_json.dumps(st, indent=2, sort_keys=True))
    _asyncio.run(_run())


@mcp.command("set-flags")
@click.option("--stdio", type=click.Choice(["on", "off"]))
@click.option("--ws", type=click.Choice(["on", "off"]))
@click.option("--sse", type=click.Choice(["on", "off"]))
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, readable=True), required=True)
def mcp_set_flags(stdio: Optional[str], ws: Optional[str], sse: Optional[str], config_path: str) -> None:
    """Update MCP transport flags (stdio/ws/sse) in configuration.

    When no flags are provided, prints current values without persisting.
    """
    cfg = _load_config(config_path)
    changed = False
    if stdio is not None:
        cfg.mcp_stdio_enabled = (stdio == "on")  # type: ignore[attr-defined]
        changed = True
    if ws is not None:
        cfg.mcp_ws_enabled = (ws == "on")  # type: ignore[attr-defined]
        changed = True
    if sse is not None:
        cfg.mcp_sse_enabled = (sse == "on")  # type: ignore[attr-defined]
        changed = True

    if changed:
        _save_config(cfg, config_path)

    flags = {
        "stdio": bool(getattr(cfg, "mcp_stdio_enabled", True)),
        "ws": bool(getattr(cfg, "mcp_ws_enabled", False)),
        "sse": bool(getattr(cfg, "mcp_sse_enabled", False)),
    }
    click.echo(json.dumps(flags))


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
