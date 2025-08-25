from __future__ import annotations

import click

from ..discovery.client import discover
from ..config import Config
import httpx


@click.group("discovery")
def discovery() -> None:
    """Agent discovery commands."""


@discovery.command("list")
def list_cmd() -> None:
    items = discover()
    if not items:
        click.echo("No agents discovered.")
        return
    for e in items:
        click.echo(f"- {e.name} ({e.agent_id}) caps={','.join(e.capabilities)} {e.transport}:{e.endpoint}")


@discovery.command("handshake")
def handshake_cmd() -> None:
    """Attempt a basic coordination handshake with discovered agents (AD4)."""
    items = discover()
    if not items:
        click.echo("No agents discovered.")
        return
    cfg = Config.load()
    my = {
        "agent_id": "agentsmcp-local",
        "name": "agentsmcp",
        "capabilities": list(cfg.agents.keys()),
        "endpoint": f"http://{cfg.server.host}:{cfg.server.port}",
        "token": getattr(cfg, "discovery_token", None),
    }
    for e in items:
        if e.transport != "http" or not e.endpoint:
            click.echo(f"Skip {e.name}: unsupported transport")
            continue
        try:
            with httpx.Client(timeout=5) as client:
                pr = client.get(f"{e.endpoint}/coord/ping")
                ok = pr.status_code == 200
                hr = client.post(f"{e.endpoint}/coord/handshake", json=my)
                click.echo(f"{e.name} ping={ok} handshake={hr.status_code}")
        except Exception as ex:
            click.echo(f"{e.name} error: {ex}")
