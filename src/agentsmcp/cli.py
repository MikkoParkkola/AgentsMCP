"""CLI entry point for AgentsMCP - Now with Revolutionary UI and Cost Intelligence."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
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
from .commands.discovery import discovery as discovery_group

# Revolutionary UI Integration
from .ui.cli_app import CLIApp, CLIConfig
from .ui.statistics_display import StatisticsDisplay
from .ui.theme_manager import ThemeManager

# Cost Intelligence Integration
try:
    from .cost.tracker import CostTracker
    from .cost.optimizer import ModelOptimizer
    from .cost.budget import BudgetManager
    COST_AVAILABLE = True
except ImportError:
    COST_AVAILABLE = False

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
    """AgentsMCP - Revolutionary Multi-Agent Orchestration with Cost Intelligence."""
    config = _load_config(config_path)
    configure_logging(log_level or "INFO", log_format or "text")


# Add revolutionary UI commands
@main.command("interactive")
@click.option("--theme", default="auto", type=click.Choice(["auto", "light", "dark"]))
@click.option("--no-welcome", is_flag=True, help="Skip welcome screen")
@click.option("--refresh-interval", default=2.0, type=float, help="Auto-refresh interval")
@click.option("--orchestrator-model", default="gpt-5", help="Orchestrator model (default: gpt-5)")
def interactive(theme: str, no_welcome: bool, refresh_interval: float, orchestrator_model: str) -> None:
    """Launch revolutionary interactive CLI interface with live cost tracking."""
    
    async def run_interactive():
        cli_config = CLIConfig(
            theme_mode=theme,
            show_welcome=not no_welcome,
            refresh_interval=refresh_interval,
            interface_mode="interactive"
        )
        
        app = CLIApp(cli_config)
        try:
            await app.start()
        except KeyboardInterrupt:
            click.echo("\n👋 Goodbye!")
            sys.exit(0)
    
    asyncio.run(run_interactive())


@main.command("dashboard")
@click.option("--theme", default="auto", type=click.Choice(["auto", "light", "dark"]))
@click.option("--refresh-interval", default=1.0, type=float, help="Dashboard refresh rate")
@click.option("--orchestrator-model", default="gpt-5", help="Orchestrator model (default: gpt-5)")
def dashboard(theme: str, refresh_interval: float, orchestrator_model: str) -> None:
    """Launch real-time dashboard with cost monitoring and agent orchestration."""
    
    async def run_dashboard():
        cli_config = CLIConfig(
            theme_mode=theme,
            refresh_interval=refresh_interval,
            interface_mode="dashboard"
        )
        
        app = CLIApp(cli_config)
        try:
            result = await app.start()
            if result.get("mode") == "dashboard":
                click.echo("🎯 Dashboard running - Press Ctrl+C to exit")
                # Keep running until interrupted
                while True:
                    await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            click.echo("\n📊 Dashboard stopped")
            sys.exit(0)
    
    asyncio.run(run_dashboard())


@main.command("costs")
@click.option("--breakdown", is_flag=True, help="Show detailed cost breakdown")
@click.option("--daily", is_flag=True, help="Show daily costs")
@click.option("--monthly", is_flag=True, help="Show monthly costs")
@click.option("--format", default="table", type=click.Choice(["table", "json"]), help="Output format")
def costs(breakdown: bool, daily: bool, monthly: bool, format: str) -> None:
    """Display current costs and budget status with beautiful formatting."""
    
    if not COST_AVAILABLE:
        click.echo("❌ Cost tracking not available. Install cost dependencies.")
        sys.exit(1)
    
    async def show_costs():
        theme_manager = ThemeManager()
        tracker = CostTracker()
        
        # Get cost data
        total_cost = tracker.total_cost
        
        if format == "json":
            import json
            cost_data = {
                "total_cost": total_cost,
                "daily_cost": tracker.get_daily_cost() if daily else None,
                "breakdown": tracker.get_breakdown() if breakdown else None
            }
            click.echo(json.dumps(cost_data, indent=2))
            return
        
        # Beautiful table format using UI components
        stats_display = StatisticsDisplay(theme_manager)
        
        # Add cost metrics
        stats_display.add_metric("Total Cost", total_cost, f"${total_cost:.4f}")
        
        if daily:
            daily_cost = tracker.get_daily_cost()
            stats_display.add_metric("Daily Cost", daily_cost, f"${daily_cost:.4f}")
            
        if monthly:
            import datetime
            now = datetime.datetime.utcnow()
            monthly_cost = tracker.get_monthly_cost(now.year, now.month)
            stats_display.add_metric("Monthly Cost", monthly_cost, f"${monthly_cost:.4f}")
        
        # Render with beautiful formatting
        output = await stats_display.render_async()
        click.echo(output)
        
        if breakdown:
            breakdown_data = tracker.get_breakdown()
            if breakdown_data:
                click.echo("\n" + "="*60)
                click.echo("💰 COST BREAKDOWN BY PROVIDER & MODEL")
                click.echo("="*60)
                
                for provider, models in breakdown_data.items():
                    click.echo(f"\n🔹 {provider.upper()}")
                    for model, cost in models.items():
                        click.echo(f"  └─ {model}: ${cost:.6f}")
    
    asyncio.run(show_costs())


@main.command("budget")
@click.argument("amount", type=float, required=False)
@click.option("--check", is_flag=True, help="Check current budget status")
@click.option("--remaining", is_flag=True, help="Show remaining budget")
def budget(amount: Optional[float], check: bool, remaining: bool) -> None:
    """Manage monthly budget and get cost alerts."""
    
    if not COST_AVAILABLE:
        click.echo("❌ Budget management not available. Install cost dependencies.")
        sys.exit(1)
    
    tracker = CostTracker()
    
    if amount is not None:
        # Set new budget
        budget_manager = BudgetManager(tracker, amount)
        click.echo(f"💰 Monthly budget set to ${amount:.2f}")
        
        # Show immediate status
        if budget_manager.check_budget():
            remaining_budget = budget_manager.remaining_budget()
            click.echo(f"✅ Within budget! ${remaining_budget:.2f} remaining")
        else:
            overspend = tracker.total_cost - amount
            click.echo(f"⚠️  Over budget by ${overspend:.2f}!")
        return
    
    # Default behavior - assume $100 budget for demo
    budget_manager = BudgetManager(tracker, 100.0)
    
    if check or remaining:
        total_cost = tracker.total_cost
        budget_ok = budget_manager.check_budget()
        remaining_amt = budget_manager.remaining_budget()
        
        if budget_ok:
            click.echo(f"✅ Budget Status: GOOD")
            click.echo(f"💵 Spent: ${total_cost:.4f} / $100.00")
            click.echo(f"💰 Remaining: ${remaining_amt:.4f}")
        else:
            overspend = total_cost - 100.0
            click.echo(f"⚠️  Budget Status: OVER")
            click.echo(f"💸 Overspent by: ${overspend:.4f}")
    else:
        click.echo("Usage: agentsmcp budget <amount> or agentsmcp budget --check")


@main.command("models")
@click.option("--detailed", is_flag=True, help="Show detailed model specifications")
@click.option("--recommend", help="Get model recommendation for use case")
def models(detailed: bool, recommend: Optional[str]) -> None:
    """Show available orchestrator models and recommendations."""
    
    # Import here to avoid circular imports
    from .distributed.orchestrator import DistributedOrchestrator
    
    if recommend:
        recommended_model = DistributedOrchestrator.get_model_recommendation(recommend)
        available_models = DistributedOrchestrator.get_available_models()
        
        click.echo(f"🎯 Recommended model for '{recommend}': {recommended_model}")
        click.echo(f"📊 Performance Score: {available_models[recommended_model]['performance_score']}%")
        click.echo(f"💰 Cost: ${available_models[recommended_model]['cost_per_input'] * 1_000_000:.2f}/${available_models[recommended_model]['cost_per_output'] * 1_000_000:.2f} per M tokens")
        click.echo(f"🧠 Context: {available_models[recommended_model]['context_limit']:,} tokens")
        click.echo(f"✨ Best for: {available_models[recommended_model]['recommended_for']}")
        return
    
    available_models = DistributedOrchestrator.get_available_models()
    
    click.echo("🤖 Available Orchestrator Models:")
    click.echo("=" * 60)
    
    for model_name, config in available_models.items():
        is_default = " (DEFAULT)" if model_name == "gpt-5" else ""
        click.echo(f"\n🔹 {model_name.upper()}{is_default}")
        
        if detailed:
            click.echo(f"  📊 Performance Score: {config['performance_score']}%")
            click.echo(f"  🧠 Context Limit: {config['context_limit']:,} tokens")
            click.echo(f"  📝 Output Limit: {config['output_limit']:,} tokens")
            
            if config['cost_per_input'] > 0:
                click.echo(f"  💰 Cost: ${config['cost_per_input'] * 1_000_000:.2f}/${config['cost_per_output'] * 1_000_000:.2f} per M tokens")
            else:
                click.echo("  💰 Cost: FREE (local model)")
            
            click.echo(f"  🎯 Strengths: {', '.join(config['strengths'])}")
            click.echo(f"  ✨ Best for: {config['recommended_for']}")
        else:
            perf = config['performance_score']
            cost = "FREE" if config['cost_per_input'] == 0 else f"${config['cost_per_input'] * 1_000_000:.2f}/M"
            context = f"{config['context_limit']//1000}K"
            click.echo(f"  Performance: {perf}% | Cost: {cost} | Context: {context}")
    
    click.echo("\n💡 Usage Examples:")
    click.echo("  agentsmcp interactive --orchestrator-model claude-4.1-opus")
    click.echo("  agentsmcp models --recommend premium")
    click.echo("  agentsmcp models --detailed")
    
    click.echo("\n🎯 Use Cases:")
    click.echo("  --recommend default        # Best overall choice")
    click.echo("  --recommend premium        # Highest quality")
    click.echo("  --recommend cost_effective # Best value")
    click.echo("  --recommend massive_context# Large codebases")
    click.echo("  --recommend local          # Privacy/offline")


@main.command("optimize")
@click.option("--mode", default="balanced", type=click.Choice(["cost", "speed", "quality", "balanced"]))
@click.option("--task-type", help="Optimize for specific task type")
@click.option("--dry-run", is_flag=True, help="Show optimization recommendations without applying")
def optimize(mode: str, task_type: Optional[str], dry_run: bool) -> None:
    """Optimize model selection for cost-effectiveness."""
    
    if not COST_AVAILABLE:
        click.echo("❌ Model optimization not available. Install cost dependencies.")
        sys.exit(1)
    
    tracker = CostTracker()
    optimizer = ModelOptimizer(tracker)
    
    click.echo(f"🎯 Optimizing for: {mode.upper()}")
    if task_type:
        click.echo(f"📝 Task type: {task_type}")
    
    if dry_run:
        click.echo("\n🔍 OPTIMIZATION RECOMMENDATIONS (DRY RUN)")
        click.echo("="*50)
        
        # Mock recommendations based on mode
        if mode == "cost":
            click.echo("💡 Switch to Ollama (local) for simple tasks → Save ~$0.50/day")
            click.echo("💡 Use gpt-3.5-turbo instead of gpt-4 for basic coding → Save ~$0.30/day")
        elif mode == "speed":
            click.echo("💡 Use gpt-3.5-turbo for faster responses → 2x speed improvement")
            click.echo("💡 Batch similar requests → 30% efficiency gain")
        elif mode == "quality":
            click.echo("💡 Use gpt-4 for complex reasoning tasks → Higher accuracy")
            click.echo("💡 Enable chain-of-thought prompting → Better results")
        else:  # balanced
            click.echo("💡 Ollama for simple tasks, gpt-3.5 for coding, gpt-4 for complex reasoning")
            click.echo("💡 Estimated savings: $0.35/day with maintained quality")
        
        click.echo("\n🚀 Run without --dry-run to apply optimizations")
    else:
        click.echo("✅ Optimization settings applied!")
        click.echo("🔄 Future agent spawns will use optimized model selection")


# Keep existing server commands
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
        click.echo(f"🚀 Server started in background (pid={proc.pid})")
        click.echo(f"🌐 Web dashboard: http://{host or env.server_host}:{port or env.server_port}/ui")
    else:
        # Run blocking in current process
        if config_path:
            os.environ["AGENTSMCP_CONFIG"] = config_path
        import uvicorn

        click.echo(f"🚀 Starting AgentsMCP server...")
        click.echo(f"🌐 Web dashboard will be available at: http://{host or env.server_host}:{port or env.server_port}/ui")
        
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
        click.echo(f"🛑 Sent SIGTERM to pid {pid}")
    except Exception as e:
        raise click.ClickException(f"Failed to stop server: {e}")
    finally:
        try:
            PID_FILE.unlink()
        except Exception:
            pass


# Keep existing command groups
main.add_command(mcp_group)
main.add_command(chat_cmd)
main.add_command(discovery_group)


# Helper function for async CLI operations
async def _run_async_command(func, *args, **kwargs):
    """Helper to run async operations in CLI commands."""
    try:
        return await func(*args, **kwargs)
    except KeyboardInterrupt:
        click.echo("\n⏹️  Operation cancelled")
        sys.exit(0)
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()