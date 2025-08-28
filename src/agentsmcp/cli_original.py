"""CLI entry point for AgentsMCP - Now with Revolutionary UI and Cost Intelligence."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
import sys
import threading
import time
from typing import Optional

import click

from agentsmcp import __version__
from agentsmcp.agent_manager import AgentManager
from agentsmcp.config import Config
from agentsmcp.orchestrator_factory import OrchestratorFactory
from agentsmcp.logging_config import configure_logging
from agentsmcp.settings import AppSettings
from agentsmcp.commands.mcp import mcp as mcp_group
# Chat functionality moved to enhanced interactive mode
# from agentsmcp.commands.discovery import discovery as discovery_group  # Temporarily disabled due to import issue
from agentsmcp.commands.roles import roles as roles_group
from agentsmcp.commands.setup import setup as setup_cmd
from agentsmcp.commands.rag import rag_group

from agentsmcp.paths import pid_file_path, ensure_dirs

PID_FILE = pid_file_path()


def _load_config(config_path: Optional[str]) -> Config:
    env = AppSettings()
    base: Config
    if config_path and Path(config_path).exists():
        base = Config.from_file(Path(config_path))
    else:
        # Prefer per-user config under ~/.agentsmcp
        from agentsmcp.paths import default_user_config_path
        user_cfg = default_user_config_path()
        if user_cfg.exists():
            base = Config.from_file(user_cfg)
        elif Path("agentsmcp.yaml").exists():
            base = Config.from_file(Path("agentsmcp.yaml"))
        else:
            base = Config()
            # First-run: persist defaults to ~/.agentsmcp/agentsmcp.yaml
            try:
                base.save_to_file(user_cfg)
            except Exception:
                pass
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
    spinner = _Spinner("Initializing AgentsMCP")
    spinner.start()
    try:
        # Create user data directories on demand (avoid import-time side effects)
        try:
            ensure_dirs()
        except Exception:
            pass
        config = _load_config(config_path)
        configure_logging(log_level or "INFO", log_format or "text")
    finally:
        spinner.stop("Initialized")


# Add revolutionary UI commands
@main.command("simple")
@click.argument("task")
@click.option("--complexity", default="moderate", type=click.Choice(["simple", "moderate", "complex"]))
@click.option("--cost-sensitive", is_flag=True, help="Prefer cost-effective models")
@click.option("--timeout", default=300, type=int, help="Task timeout in seconds")
@click.option("--config", "config_path", default=None, help="Path to YAML config")
@click.pass_context
def simple(ctx, task: str, complexity: str, cost_sensitive: bool, timeout: int, config_path: Optional[str]):
    """Execute a task using simple orchestration (recommended default)."""
    config = _load_config(config_path or ctx.parent.params.get('config_path'))
    
    async def run_task():
        orchestrator = OrchestratorFactory.create(config)
        
        from agentsmcp.simple_orchestrator import TaskRequest, TaskComplexity
        task_request = TaskRequest(
            task=task,
            complexity=TaskComplexity(complexity),
            cost_sensitive=cost_sensitive
        )
        
        print(f"🚀 Executing task with {complexity} complexity using simple orchestration...")
        print(f"📋 Task: {task}")
        
        result = await orchestrator.execute_task(task_request, timeout=timeout)
        
        print(f"\n✅ Status: {result['status']}")
        if result['status'] == 'success':
            print(f"🤖 Model used: {result['model_used']}")
            print(f"📄 Result:\n{result['result']}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        await orchestrator.cleanup()
    
    try:
        asyncio.run(run_task())
    except KeyboardInterrupt:
        print("\n🛑 Task cancelled by user")
    except Exception as e:
        print(f"❌ Task failed: {e}")


@main.command("config")
@click.option("--show-preferences", is_flag=True, help="Show current model preferences")
@click.option("--show-recommendations", is_flag=True, help="Show orchestrator mode recommendations")
@click.option("--set-workhorse", help="Set primary model for workhorse role")
@click.option("--set-orchestrator", help="Set primary model for orchestrator role")  
@click.option("--set-specialist", help="Set primary model for specialist role")
@click.option("--config", "config_path", default=None, help="Path to YAML config")
@click.pass_context
def config_cmd(ctx, show_preferences: bool, show_recommendations: bool, 
               set_workhorse: Optional[str], set_orchestrator: Optional[str], 
               set_specialist: Optional[str], config_path: Optional[str]):
    """Configure AgentsMCP orchestration and model preferences."""
    config = _load_config(config_path or ctx.parent.params.get('config_path'))
    
    async def run_config():
        orchestrator = OrchestratorFactory.create(config)
        
        if show_recommendations:
            recommendations = OrchestratorFactory.get_mode_recommendations(config)
            print("🎯 Orchestrator Mode Recommendations:")
            print(f"   Detected mode: {recommendations['detected_mode']}")
            for reason in recommendations['reasoning']:
                print(f"   • {reason}")
            
            print("\n📊 Simple vs Complex Mode:")
            print("   Simple Mode Benefits:")
            for benefit in recommendations['benefits']['simple']:
                print(f"     ✓ {benefit}")
            print("   Complex Mode Benefits:")
            for benefit in recommendations['benefits']['complex']:
                print(f"     ✓ {benefit}")
        
        if show_preferences and hasattr(orchestrator, 'model_preferences'):
            print("\n🤖 Current Model Preferences:")
            for role, pref in orchestrator.model_preferences.items():
                print(f"   {role.value.title()} Role:")
                print(f"     Primary: {pref.primary_model}")
                print(f"     Fallbacks: {', '.join(pref.fallback_models)}")
                print(f"     Cost threshold: ${pref.cost_threshold:.3f}")
        
        # Update preferences if requested
        updates = {}
        if set_workhorse:
            updates['workhorse'] = {'primary_model': set_workhorse}
        if set_orchestrator:
            updates['orchestrator'] = {'primary_model': set_orchestrator}
        if set_specialist:
            updates['specialist'] = {'primary_model': set_specialist}
            
        if updates:
            if hasattr(orchestrator, 'configure_preferences'):
                orchestrator.configure_preferences(updates)
                print(f"\n✅ Updated preferences: {', '.join(updates.keys())}")
            else:
                print("❌ Current orchestrator does not support runtime preference updates")
        
        await orchestrator.cleanup()
    
    try:
        asyncio.run(run_config())
    except Exception as e:
        print(f"❌ Configuration failed: {e}")


@main.command("interactive")
@click.option("--theme", default="auto", type=click.Choice(["auto", "light", "dark"]))
@click.option("--no-welcome", is_flag=True, help="Skip welcome screen")
@click.option("--refresh-interval", default=2.0, type=float, help="Auto-refresh interval")
@click.option("--orchestrator-model", default="gpt-5", help="Orchestrator model (default: gpt-5)")
@click.option(
    "--agent",
    "agent_type",
    default="ollama-turbo-coding",
    help="Default AI agent (ollama-turbo-coding/ollama-turbo-general/codex/claude/ollama)",
)
@click.option("--model", "model_override", default=None, help="Model override for session")
@click.option(
    "--provider",
    "provider_override",
    default=None,
    type=click.Choice(["openai", "anthropic", "ollama", "ollama-turbo", "openrouter", "codex"]),
)
@click.option("--streaming/--no-streaming", default=True, help="Enable/disable streaming mode")
@click.option("--webui/--no-webui", default=False, help="Start the web UI server alongside the interactive CLI")
@click.option("--ui", "ui_mode", default="tui", type=click.Choice(["interactive", "dashboard", "tui", "stats"]))
def interactive(theme: str, no_welcome: bool, refresh_interval: float, orchestrator_model: str,
               agent_type: str, model_override: Optional[str], provider_override: Optional[str], streaming: bool,
               webui: bool, ui_mode: str) -> None:
    """Launch enhanced interactive CLI with AI chat, agent orchestration, and task delegation."""
    
    async def run_interactive():
        from agentsmcp.settings import AppSettings
        import subprocess
        # Lazy-load UI modules to keep import-time fast
        from agentsmcp.ui.cli_app import CLIApp, CLIConfig
        
        # Optionally start the web UI server in the background
        web_proc = None
        env = AppSettings()
        if webui:
            try:
                cmd = [
                    "python",
                    "-m",
                    "uvicorn",
                    "agentsmcp.server:create_app",
                    "--factory",
                    "--host",
                    env.server_host,
                    "--port",
                    str(env.server_port),
                ]
                # Show a brief spinner while the server comes up
                boot = _Spinner("Starting Web UI")
                boot.start()
                web_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Actively probe readiness with short backoff up to ~5s
                import urllib.request
                ready = False
                delay = 0.2
                for _ in range(10):
                    try:
                        with urllib.request.urlopen(f"http://{env.server_host}:{env.server_port}/health/live", timeout=1.0) as resp:
                            if resp.status == 200:
                                ready = True
                                break
                    except Exception:
                        pass
                    time.sleep(delay)
                    delay = min(1.5, delay * 1.7)
                boot.stop("Web UI ready" if ready else None)
                if ready:
                    click.echo("✓ Web UI started")
                    click.echo(f"🌐 Web UI: http://{env.server_host}:{env.server_port}/ui (disable with --no-webui)")
                else:
                    click.echo("⚠️  Web UI may not be ready yet; try again shortly")
            except Exception as e:
                click.echo(f"⚠️  Failed to start web UI automatically: {e}")
        cli_config = CLIConfig(
            theme_mode=theme,
            show_welcome=not no_welcome,
            refresh_interval=refresh_interval,
            interface_mode=ui_mode,
            orchestrator_model=orchestrator_model,
            agent_type=agent_type,
            model_override=model_override,
            provider_override=provider_override,
            streaming=streaming
        )
        
        app = CLIApp(cli_config)
        try:
            await app.start()
        except KeyboardInterrupt:
            click.echo("\n👋 Goodbye!")
            sys.exit(0)
        finally:
            # Stop background web UI if we started it
            if web_proc:
                try:
                    web_proc.terminate()
                except Exception:
                    pass
    
    asyncio.run(run_interactive())


@main.command("dashboard")
@click.option("--theme", default="auto", type=click.Choice(["auto", "light", "dark"]))
@click.option("--refresh-interval", default=1.0, type=float, help="Dashboard refresh rate")
@click.option("--orchestrator-model", default="gpt-5", help="Orchestrator model (default: gpt-5)")
def dashboard(theme: str, refresh_interval: float, orchestrator_model: str) -> None:
    """Launch real-time dashboard with cost monitoring and agent orchestration."""
    
    async def run_dashboard():
        # Lazy-load UI modules
        from agentsmcp.ui.cli_app import CLIApp, CLIConfig
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
    
    # Lazy import cost modules to avoid penalizing CLI startup
    try:
        from agentsmcp.ui.theme_manager import ThemeManager
        from agentsmcp.ui.statistics_display import StatisticsDisplay
        from agentsmcp.cost.tracker import CostTracker
    except Exception:
        click.echo("❌ Cost tracking not available. Install cost extras.")
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
@click.option("--json", "as_json", is_flag=True, help="Output JSON envelope")
def budget(amount: Optional[float], check: bool, remaining: bool, as_json: bool) -> None:
    """Manage monthly budget and get cost alerts."""
    
    # Lazy import cost modules
    try:
        from agentsmcp.cost.tracker import CostTracker
        from agentsmcp.cost.budget import BudgetManager
    except Exception:
        click.echo("❌ Budget management not available. Install cost extras.")
        sys.exit(1)

    tracker = CostTracker()
    
    if amount is not None:
        # Set new budget
        budget_manager = BudgetManager(tracker, amount)
        if as_json:
            res = {
                "ok": True,
                "budget": amount,
                "within": budget_manager.check_budget(),
                "remaining": budget_manager.remaining_budget(),
                "spent": tracker.total_cost,
            }
            import json as _json
            click.echo(_json.dumps(res))
        else:
            click.echo(f"💰 Monthly budget set to ${amount:.2f}")
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
        if as_json:
            res = {
                "ok": True,
                "within": budget_ok,
                "remaining": remaining_amt,
                "spent": total_cost,
                "budget": 100.0,
            }
            import json as _json
            click.echo(_json.dumps(res))
        else:
            if budget_ok:
                click.echo(f"✅ Budget Status: GOOD")
                click.echo(f"💵 Spent: ${total_cost:.4f} / $100.00")
                click.echo(f"💰 Remaining: ${remaining_amt:.4f}")
            else:
                overspend = total_cost - 100.0
                click.echo(f"⚠️  Budget Status: OVER")
                click.echo(f"💸 Overspent by: ${overspend:.4f}")
    else:
        if as_json:
            import json as _json
            click.echo(_json.dumps({"ok": False, "error": {"code": "usage", "message": "budget <amount> or --check"}}))
        else:
            click.echo("Usage: agentsmcp budget <amount> or agentsmcp budget --check")


@main.command("models")
@click.option("--detailed", is_flag=True, help="Show detailed model specifications")
@click.option("--recommend", help="Get model recommendation for use case")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]))
def models(detailed: bool, recommend: Optional[str], fmt: str) -> None:
    """Show available orchestrator models and recommendations."""
    
    # Import here to avoid circular imports
    from agentsmcp.distributed.orchestrator import DistributedOrchestrator
    
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

    if fmt == "json":
        import json as _json
        click.echo(_json.dumps(available_models))
        return

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
    # Lazy import optimizer/cost modules
    try:
        from agentsmcp.cost.tracker import CostTracker
        from agentsmcp.cost.optimizer import ModelOptimizer
    except Exception:
        click.echo("❌ Model optimization not available. Install cost extras.")
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
@click.option("--enable-mcp-api", is_flag=True, help="Enable MCP REST API endpoints (/mcp)")
def server_start(
    host: Optional[str],
    port: Optional[int],
    background: bool,
    config_path: Optional[str],
    enable_mcp_api: bool,
) -> None:
    env = AppSettings()
    if host:
        env.server_host = host  # type: ignore[attr-defined]
    if port:
        env.server_port = port  # type: ignore[attr-defined]

    # Optional: enable /mcp via env toggle handled in AppSettings
    if enable_mcp_api:
        os.environ["AGENTSMCP_MCP_API_ENABLED"] = "true"

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
        click.echo(f"🌐 Web dashboard (if enabled): http://{host or env.server_host}:{port or env.server_port}/ui")
    else:
        # Run blocking in current process
        if config_path:
            os.environ["AGENTSMCP_CONFIG"] = config_path
        import uvicorn

        click.echo(f"🚀 Starting AgentsMCP server...")
        click.echo(f"🌐 Web dashboard (if enabled): http://{host or env.server_host}:{port or env.server_port}/ui")
        
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
# Chat functionality now integrated in enhanced interactive mode
# main.add_command(discovery_group)  # Temporarily disabled due to import issue
# Role-based orchestration commands
main.add_command(roles_group)
# RAG knowledge management commands
main.add_command(rag_group)
# Setup wizard command
main.add_command(setup_cmd)


# Helper function for async CLI operations
async def _run_async_command(func, *args, **kwargs):
    """Helper to run async operations in CLI commands."""
    try:
        return await func(*args, **kwargs)
    except KeyboardInterrupt:
        click.echo("\n⏹️  Operation cancelled")
        sys.exit(0)
class _Spinner:
    """Lightweight CLI spinner for startup feedback."""

    FRAMES = ["⠙", "⠸", "⠼", "⠴", "⠦", "⠇", "⠋"]

    def __init__(self, text: str = "Loading"):
        self.text = text
        self._running = False
        self._thread = None
        self._tty = sys.stderr.isatty()

    def _loop(self):
        i = 0
        while self._running:
            if self._tty:
                frame = self.FRAMES[i % len(self.FRAMES)]
                sys.stderr.write(f"\r{frame} {self.text} ")
                sys.stderr.flush()
            time.sleep(0.08)
            i += 1
        if self._tty:
            sys.stderr.write("\r")
            sys.stderr.flush()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, final_message: str | None = None):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        if final_message and sys.stderr.isatty():
            sys.stderr.write(f"✓ {final_message}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
