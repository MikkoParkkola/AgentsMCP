from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from ..agent_manager import AgentManager
from ..config import Config, ProviderType
from ..settings import AppSettings

console = Console()


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


@click.command("chat")
@click.option("--agent", "agent_type", default="codex", help="Agent type to use (codex/claude/ollama)")
@click.option("--model", "model_override", default=None, help="Model override for this session")
@click.option("--provider", "provider_override", default=None, type=click.Choice(["openai", "openrouter", "ollama", "custom"]))
@click.option("--api-base", "api_base_override", default=None, help="API base URL override")
@click.option("--config", "config_path", default=None, help="Path to YAML config")
@click.option("--save-on-exit", is_flag=True, default=False, help="Save session settings to config on exit")
@click.option("--system", "system_prompt", default=None, help="Session system prompt override")
def chat(
    agent_type: str,
    model_override: Optional[str],
    provider_override: Optional[str],
    api_base_override: Optional[str],
    config_path: Optional[str],
    save_on_exit: bool,
    system_prompt: Optional[str],
) -> None:
    """Interactive CLI AI client with configurable session and persistent settings.

    Commands:
      /help                 Show commands
      /model <name>         Set model for this session
      /provider <name>      Set provider (openai|openrouter|ollama|custom)
      /api_base <url>       Set API base URL for this session
      /system               Edit system prompt for this session
      /temp <0..2>          Set temperature
      /new                  Start a new session (keeps settings)
      /mcp                  Manage MCP servers (list/enable/disable)
      /save                 Save current settings to config file
      /quit                 Exit
    """
    cfg = _load_config(config_path)
    if agent_type not in cfg.agents:
        raise click.ClickException(f"Unknown agent type: {agent_type}")

    ac = cfg.agents[agent_type]
    # Apply session overrides
    if model_override:
        ac.model = model_override
    if provider_override:
        ac.provider = ProviderType(provider_override)
    if api_base_override:
        ac.api_base = api_base_override
    if system_prompt:
        ac.system_prompt = system_prompt

    # Manager for ad-hoc task runs
    mgr = AgentManager(cfg)

    def _header() -> Panel:
        prov = getattr(ac, "provider", ProviderType.OPENAI).value
        model = ac.model or (ac.model_priority[0] if ac.model_priority else "(auto)")
        sysline = (ac.system_prompt[:80] + "…") if ac.system_prompt and len(ac.system_prompt) > 80 else (ac.system_prompt or "default")
        return Panel.fit(
            f"Agent: [bold]{agent_type}[/bold]  Provider: [bold]{prov}[/bold]  Model: [bold]{model}[/bold]\nSystem: {sysline}",
            title="AgentsMCP Chat",
        )

    def _help() -> None:
        console.print(_header())
        console.print(
            """
[bold]/help[/bold]                 Show commands
[bold]/model <name>[/bold]         Set model for this session
[bold]/provider <name>[/bold]      Set provider (openai|openrouter|ollama|custom)
[bold]/api_base <url>[/bold]       Set API base URL for this session
[bold]/system[/bold]               Edit system prompt for this session
[bold]/temp <0..2>[/bold]          Set temperature
[bold]/new[/bold]                  Start a new session (keeps settings)
[bold]/mcp[/bold]                  Manage MCP servers (list/enable/disable)
[bold]/save[/bold]                 Save current settings to config file
[bold]/quit[/bold]                 Exit
            """
        )

    async def _ask_once(prompt: str) -> str:
        # Spawn ephemeral job using current settings
        job_id = await mgr.spawn_agent(agent_type, prompt, timeout=ac.timeout)
        status = await mgr.wait_for_completion(job_id)
        return status.output or status.error or "(no output)"

    def _mcp_menu() -> None:
        while True:
            table = Table(title="MCP Servers (toggle with 'enable <name>' / 'disable <name>')")
            table.add_column("Name")
            table.add_column("Enabled")
            table.add_column("Transport")
            table.add_column("Command/URL")
            for s in cfg.mcp or []:
                cmd = " ".join(s.command or []) if s.command else (s.url or "")
                table.add_row(s.name, "yes" if s.enabled else "no", s.transport or "stdio", cmd)
            console.print(table)
            cmd = Prompt.ask("mcp> ", default="back")
            if cmd.strip() in ("back", "exit", "quit"):
                break
            parts = cmd.split()
            if len(parts) >= 2:
                action, name = parts[0], parts[1]
                match action:
                    case "enable":
                        for s in cfg.mcp or []:
                            if s.name == name:
                                s.enabled = True
                                console.print(f"Enabled {name}")
                                break
                        else:
                            console.print(f"Unknown MCP: {name}")
                    case "disable":
                        for s in cfg.mcp or []:
                            if s.name == name:
                                s.enabled = False
                                console.print(f"Disabled {name}")
                                break
                        else:
                            console.print(f"Unknown MCP: {name}")
                    case _:
                        console.print("Unknown action. Use 'enable <name>' or 'disable <name>'.")
            else:
                console.print("Commands: enable <name>, disable <name>, back")

    console.print(_header())
    _help()

    history: list[tuple[str, str]] = []  # (role, text)

    while True:
        try:
            user = Prompt.ask("you")
        except (KeyboardInterrupt, EOFError):
            console.print("\nExiting…")
            break

        if not user.strip():
            continue

        if user.startswith("/"):
            parts = user[1:].split()
            cmd = parts[0]
            args = parts[1:]
            if cmd == "help":
                _help()
                continue
            if cmd == "quit":
                break
            if cmd == "model" and args:
                ac.model = " ".join(args)
                console.print(_header())
                continue
            if cmd == "provider" and args:
                try:
                    ac.provider = ProviderType(args[0])
                except Exception:
                    console.print("Invalid provider. Use openai|openrouter|ollama|custom")
                console.print(_header())
                continue
            if cmd == "api_base" and args:
                ac.api_base = " ".join(args)
                console.print(_header())
                continue
            if cmd == "system":
                ac.system_prompt = Prompt.ask("System prompt", default=ac.system_prompt or "")
                console.print(_header())
                continue
            if cmd == "temp" and args:
                try:
                    t = float(args[0])
                    if 0 <= t <= 2:
                        ac.temperature = t
                        console.print(_header())
                    else:
                        console.print("Temperature must be between 0 and 2")
                except ValueError:
                    console.print("Invalid number")
                continue
            if cmd == "new":
                history.clear()
                console.print(Panel.fit("New session started", title="Session"))
                continue
            if cmd == "mcp":
                _mcp_menu()
                continue
            if cmd == "save":
                path = _save_config(cfg, config_path)
                console.print(f"Saved settings to {path}")
                continue
            console.print("Unknown command. Type /help for options.")
            continue

        # Regular chat turn
        console.print(Panel.fit(user, title="You"))
        try:
            answer = asyncio.run(_ask_once(user))
        except RuntimeError:
            # If running inside event loop (rare), fallback
            answer = "(execution error inside running loop)"
        history.append(("user", user))
        history.append(("assistant", answer))
        console.print(Panel(answer, title=f"{agent_type}"))

    if save_on_exit and Confirm.ask("Save session settings to config?", default=True):
        path = _save_config(cfg, config_path)
        console.print(f"Saved settings to {path}")
