"""CLI entry point for AgentsMCP."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from pathlib import Path
from typing import Optional

import click

from . import __version__
from .agent_manager import AgentManager
from .config import Config
from .logging_config import configure_logging
from .settings import AppSettings
from .commands.mcp import mcp as mcp_group
from .commands.chat import chat as chat_cmd

PID_FILE = Path(os.path.expanduser("~/.agentsmcp/.agentsmcp.pid"))


def _load_config(config_path: Optional[str]) -> Config:
    env = AppSettings()
    base: Config
    if config_path and Path(config_path).exists():
        base = Config.from_file(Path(config_path))
    else:
        # Prefer per-user config under ~/.agentsmcp
        user_cfg = Config.default_config_path()
        if user_cfg.exists():
            base = Config.from_file(user_cfg)
        elif Path("agentsmcp.yaml").exists():
            base = Config.from_file(Path("agentsmcp.yaml"))
        else:
            base = Config()
    return env.to_runtime_config(base)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="agentsmcp")
@click.option("--log-level", default=None, help="Log level (override env)")
@click.option("--log-format", default=None, type=click.Choice(["json", "text"]))
@click.option("--config", "config_path", default=None, help="Path to YAML config")
def main(
    log_level: Optional[str], log_format: Optional[str], config_path: Optional[str]
) -> None:
    """AgentsMCP CLI."""
    env = AppSettings()
    configure_logging(
        level=log_level or env.log_level, fmt=log_format or env.log_format
    )
    # Stash resolved config path for subcommands via env var
    if config_path:
        os.environ["AGENTSMCP_CONFIG"] = config_path
    # Register extra command groups
    main.add_command(mcp_group, name="mcp")
    main.add_command(chat_cmd, name="chat")


@main.group()
def agent() -> None:
    """Manage agents and jobs."""


@agent.command("list")
@click.option("--config", "config_path", default=None, help="Path to YAML config")
def agent_list(config_path: Optional[str]) -> None:
    cfg = _load_config(config_path)
    rows = [
        f"- {name} (model={ac.model}, tools={','.join(ac.tools)})"
        for name, ac in cfg.agents.items()
    ]
    click.echo("Available agents:\n" + "\n".join(rows))


@agent.command("spawn")
@click.argument("type")
@click.argument("task")
@click.option("--timeout", default=300, type=int)
@click.option("--model", "model_override", default=None, help="Override model for this run")
@click.option("--provider", "provider_override", default=None, type=click.Choice(["openai", "openrouter", "ollama", "custom"]))
@click.option("--api-base", "api_base_override", default=None, help="Override API base URL for this run")
@click.option("--config", "config_path", default=None)
def agent_spawn(
    type: str,
    task: str,
    timeout: int,
    model_override: Optional[str],
    provider_override: Optional[str],
    api_base_override: Optional[str],
    config_path: Optional[str],
) -> None:
    """Spawn a new agent job."""
    cfg = _load_config(config_path)
    if type not in cfg.agents:
        raise click.BadParameter(f"Unknown agent type: {type}")

    # Apply ephemeral overrides for this invocation
    ac = cfg.agents[type]
    if model_override:
        ac.model = model_override
    if provider_override:
        from .config import ProviderType
        ac.provider = ProviderType(provider_override)
    if api_base_override:
        ac.api_base = api_base_override

    mgr = AgentManager(cfg)

    async def _run():
        job_id = await mgr.spawn_agent(type, task, timeout=timeout)
        click.echo(job_id)

    asyncio.run(_run())


@agent.command("status")
@click.argument("job_id")
@click.option("--config", "config_path", default=None)
def agent_status(job_id: str, config_path: Optional[str]) -> None:
    """Fetch status of a job."""
    cfg = _load_config(config_path)
    mgr = AgentManager(cfg)

    async def _run():
        status = await mgr.get_job_status(job_id)
        if not status:
            raise click.ClickException("Job not found")
        click.echo(
            f"state={status.state.value} created_at={status.created_at.isoformat()} updated_at={status.updated_at.isoformat()}\n"
            f"output={status.output or ''}\nerror={status.error or ''}"
        )

    asyncio.run(_run())


@agent.command("cancel")
@click.argument("job_id")
@click.option("--config", "config_path", default=None)
def agent_cancel(job_id: str, config_path: Optional[str]) -> None:
    """Cancel a job by id."""
    cfg = _load_config(config_path)
    mgr = AgentManager(cfg)

    async def _run():
        ok = await mgr.cancel_job(job_id)
        if not ok:
            raise click.ClickException("Job not found or already finished")
        click.echo("cancelled")

    asyncio.run(_run())


@main.group()
def server() -> None:
    """Manage the API server."""


@server.command("start")
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
@click.option("--background", is_flag=True, help="Run server in background")
@click.option("--config", "config_path", default=None)
def server_start(
    host: Optional[str],
    port: Optional[int],
    background: bool,
    config_path: Optional[str],
) -> None:
    env = AppSettings()
    if host:
        env.server_host = host  # type: ignore[attr-defined]
    if port:
        env.server_port = port  # type: ignore[attr-defined]

    if background:
        # Launch uvicorn as a subprocess using the factory
        cmd = [
            "python",
            "-m",
            "uvicorn",
            "agentsmcp.server:create_app",
            "--factory",
            "--host",
            host or env.server_host,
            "--port",
            str(port or env.server_port),
        ]
        if config_path:
            os.environ["AGENTSMCP_CONFIG"] = config_path
        proc = subprocess.Popen(cmd)
        PID_FILE.write_text(str(proc.pid))
        click.echo(f"Server started in background (pid={proc.pid})")
    else:
        # Run blocking in current process
        if config_path:
            os.environ["AGENTSMCP_CONFIG"] = config_path
        import uvicorn

        uvicorn.run(
            "agentsmcp.server:create_app",
            factory=True,
            host=host or env.server_host,
            port=port or env.server_port,
            log_level=env.log_level.lower(),
        )


@server.command("stop")
def server_stop() -> None:
    if not PID_FILE.exists():
        raise click.ClickException(
            "PID file not found; was the server started with --background?"
        )
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        click.echo(f"Sent SIGTERM to pid {pid}")
    except Exception as e:
        raise click.ClickException(f"Failed to stop server: {e}")
    finally:
        try:
            PID_FILE.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
