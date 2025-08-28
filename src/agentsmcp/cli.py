"""CLI entry point for AgentsMCP with hierarchical command structure."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
import threading
import time
from typing import Optional

import click

from agentsmcp import __version__
from agentsmcp.errors import (
    AgentsMCPError,
    ConfigError,
    NetworkError,
    TaskExecutionError,
    AuthenticationError,
    MissingParameterError,
    require_option
)
from agentsmcp.cli_enhanced import (
    EnhancedAgentsMCPCLI,
    handle_common_errors,
    validate_task_input,
    check_config_exists,
    with_intelligent_suggestions
)
from agentsmcp.progressive_disclosure import (
    ProgressiveDisclosureGroup,
    advanced_option,
    SmartDefaults
)
from agentsmcp.onboarding import detect_and_run_onboarding
from agentsmcp.agent_manager import AgentManager
from agentsmcp.config import Config
from agentsmcp.orchestrator_factory import OrchestratorFactory
from agentsmcp.logging_config import configure_logging
from agentsmcp.settings import AppSettings
from agentsmcp.commands.mcp import mcp as mcp_group
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

class _Spinner:
    def __init__(self, msg: str):
        self.message = msg
        self._running = False
        self._thread = None
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
    
    def _spin(self):
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while self._running:
            print(f"\r{chars[i % len(chars)]} {self.message}", end="", flush=True)
            time.sleep(0.1)
            i += 1
    
    def stop(self, final_msg=""):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        print(f"\r✅ {final_msg or self.message}")

# =====================================================================
# Root CLI Group
# =====================================================================

class AgentsMCPProgressiveGroup(ProgressiveDisclosureGroup, EnhancedAgentsMCPCLI):
    """Combined progressive disclosure and enhanced error handling group."""
    pass

@click.group(
    cls=AgentsMCPProgressiveGroup,
    context_settings={"help_option_names": ["-h", "--help"]}
)
@click.version_option(version=__version__, prog_name="agentsmcp")
@click.option("--log-level", default=None, help="Log level (override env)")
@click.option("--log-format", default=None, type=click.Choice(["json", "text"]))
@click.option("--config", "config_path", default=None, help="Path to YAML config")
@click.option("--debug", is_flag=True, hidden=True, help="Enable debug mode")
def main(
    log_level: Optional[str], log_format: Optional[str], config_path: Optional[str], debug: bool
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

# =====================================================================
# 1️⃣ INIT GROUP - Getting Started
# =====================================================================

@main.group(name="init")
def init_group():
    """Getting started – discovery & first-time configuration."""
    pass

@init_group.command("setup")
@click.option("--mode", type=click.Choice(["interactive", "quick", "advanced"]), 
              default="interactive", help="Setup mode")
@click.option("--force", is_flag=True, help="Force onboarding even if config exists")
@click.pass_context
@with_intelligent_suggestions
def init_setup(ctx, mode: str, force: bool, advanced: bool = False):
    """🛠️ Interactive wizard that creates a valid AgentsMCP configuration."""
    from agentsmcp.onboarding import OnboardingWizard
    
    wizard = OnboardingWizard(mode=mode)
    
    if force or wizard.should_run_onboarding():
        success = wizard.run()
        if not success:
            ctx.exit(1)
    else:
        click.echo(click.style("✅ Configuration already exists and is valid.", fg='green'))
        click.echo(f"Use {click.style('--force', fg='cyan')} to run setup anyway.")
        click.echo(f"Edit existing config: {click.style('agentsmcp config edit', fg='cyan')}")

@init_group.command("onboarding")
@click.option("--mode", type=click.Choice(["interactive", "quick", "advanced"]), 
              default="interactive", help="Onboarding mode")
@click.option("--force", is_flag=True, help="Force onboarding even if config exists")
def init_onboarding(mode: str, force: bool, advanced: bool = False):
    """🚀 Run the first-time onboarding wizard."""
    from agentsmcp.onboarding import OnboardingWizard
    
    wizard = OnboardingWizard(mode=mode)
    
    if force or wizard.should_run_onboarding():
        success = wizard.run()
        if not success:
            click.echo(click.style("❌ Onboarding failed or was cancelled.", fg='red'))
            raise click.Abort()
    else:
        click.echo(click.style("✅ Configuration already exists and is valid.", fg='green'))
        click.echo(f"Use {click.style('--force', fg='cyan')} to run onboarding anyway.")

@init_group.command("config")
@click.option("--show-preferences", is_flag=True, help="Show current model preferences")
@click.option("--show-recommendations", is_flag=True, help="Show orchestrator mode recommendations")
@click.option("--set-workhorse", help="Set primary model for workhorse role")
@click.option("--set-orchestrator", help="Set primary model for orchestrator role")  
@click.option("--set-specialist", help="Set primary model for specialist role")
@click.pass_context
def init_config(ctx, show_preferences: bool, show_recommendations: bool, 
               set_workhorse: Optional[str], set_orchestrator: Optional[str], 
               set_specialist: Optional[str]):
    """Configure AgentsMCP orchestration and model preferences."""
    config_path = ctx.parent.parent.params.get('config_path')
    config = _load_config(config_path)
    
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

# =====================================================================
# 2️⃣ RUN GROUP - Core Execution
# =====================================================================

@main.group(name="run")
def run_group():
    """Run the core AgentsMCP workflows."""
    pass

@run_group.command("simple")
@click.argument("task")
@advanced_option("--complexity", default=SmartDefaults.get_complexity_default(), 
                type=click.Choice(["simple", "moderate", "complex"]), 
                advanced=False, help="Task complexity level")
@advanced_option("--cost-sensitive", is_flag=True, advanced=False,
                help="Prefer cost-effective models")
@advanced_option("--timeout", default=SmartDefaults.get_timeout_default(), 
                type=int, advanced=True, help="Task timeout in seconds")
@advanced_option("--max-retries", default=3, type=int, advanced=True,
                help="Maximum retry attempts on failure")
@advanced_option("--enable-debug", is_flag=True, advanced=True,
                help="Enable verbose debug output")
@click.pass_context
@handle_common_errors
@with_intelligent_suggestions
def run_simple(ctx, task: str, complexity: str, cost_sensitive: bool, timeout: int,
              max_retries: int, enable_debug: bool, advanced: bool = False):
    """Execute a task using simple orchestration (recommended default)."""
    # Validate task input
    task = validate_task_input(task)
    
    config_path = ctx.parent.parent.params.get('config_path')
    try:
        config = _load_config(config_path)
    except Exception as e:
        raise ConfigError(f"Failed to load configuration: {e}")
    
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
        
        try:
            result = await orchestrator.execute_task(task_request, timeout=timeout)
        except Exception as e:
            await orchestrator.cleanup()
            if "authentication" in str(e).lower() or "api" in str(e).lower():
                raise AuthenticationError("AI service")
            elif "rate" in str(e).lower() or "limit" in str(e).lower():
                raise RateLimitError("AI service")
            else:
                raise TaskExecutionError(str(e), task)
        
        print(f"\n✅ Status: {result['status']}")
        if result['status'] == 'success':
            print(f"🤖 Model used: {result['model_used']}")
            print(f"📄 Result:\n{result['result']}")
        else:
            error_msg = result.get('error', 'Unknown error')
            await orchestrator.cleanup()
            raise TaskExecutionError(error_msg, task)
        
        await orchestrator.cleanup()
    
    asyncio.run(run_task())

@run_group.command("interactive")
@advanced_option("--theme", default="auto", type=click.Choice(["auto", "light", "dark"]),
                advanced=False, help="Interface theme")
@advanced_option("--no-welcome", is_flag=True, advanced=True,
                help="Skip welcome screen")
@advanced_option("--refresh-interval", default=2.0, type=float, advanced=True,
                help="Auto-refresh interval")
@advanced_option("--orchestrator-model", default="gpt-5", advanced=True,
                help="Orchestrator model (default: gpt-5)")
@advanced_option("--agent", "agent_type", default="ollama-turbo-coding", advanced=True,
                help="Default AI agent (ollama-turbo-coding/ollama-turbo-general/codex/claude/ollama)")
@click.option(
    "--legacy",
    is_flag=True,
    hidden=True,
    help="Run the legacy basic interactive mode (for power-users).",
)
@click.pass_context
def run_interactive(ctx, theme: str, no_welcome: bool, refresh_interval: float, 
                   orchestrator_model: str, agent_type: str, legacy: bool = False, advanced: bool = False):
    """Launch the interactive interface.

    By default this launches the **modern world-class TUI**. If the hidden
    ``--legacy`` flag is supplied the historic, line-oriented interactive mode
    (previous behaviour) is used. All other advanced options are forwarded to
    the UI layer.
    """
    config_path = ctx.parent.parent.params.get('config_path')
    config = _load_config(config_path)
    
    # Import here to avoid startup overhead
    from agentsmcp.ui.cli_app import CLIApp, CLIConfig
    import logging
    
    # Determine which internal mode the UI should run.
    # "tui" → the new Modern TUI (default)
    # "interactive" → legacy basic REPL, kept for backward compatibility.
    ui_mode = "interactive" if legacy else "tui"
    
    try:
        # Create CLI configuration with parameters
        cli_config = CLIConfig(
            theme_mode=theme,
            show_welcome=not no_welcome,
            refresh_interval=refresh_interval,
            orchestrator_model=orchestrator_model,
            agent_type=agent_type,
        )
        
        # Pass the mode to CLIApp so it knows which interface to launch
        app = CLIApp(config=cli_config, mode=ui_mode)
        
        # Run the app asynchronously
        async def run_app():
            return await app.start()
        
        asyncio.run(run_app())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as exc:
        logging.getLogger(__name__).exception(
            "Failed to start the %s interactive UI", ui_mode
        )
        print(f"❌ Interactive mode failed: {exc}")

# =====================================================================
# 3️⃣ MONITOR GROUP - Observability & Control
# =====================================================================

@main.group(name="monitor")
def monitor_group():
    """Observability, cost-tracking and budget alerts."""
    pass

@monitor_group.command("costs")
@advanced_option("--detailed", is_flag=True, advanced=False,
                help="Show detailed cost breakdown")
@advanced_option("--format", "fmt", default="table", 
                type=click.Choice(["table", "json"]), advanced=True,
                help="Output format")
@advanced_option("--days", default=7, type=int, advanced=True,
                help="Number of days to include in cost analysis")
@with_intelligent_suggestions
def monitor_costs(detailed: bool, fmt: str, days: int, advanced: bool = False):
    """Display current costs and budget status with beautiful formatting."""
    # Lazy import cost modules
    try:
        from agentsmcp.cost.tracker import CostTracker
        from agentsmcp.cost.display import CostDisplay
    except Exception:
        click.echo("❌ Cost tracking not available. Install cost extras.")
        sys.exit(1)

    tracker = CostTracker()
    display = CostDisplay(tracker)
    
    if fmt == "json":
        import json as _json
        cost_data = {
            "total_cost": tracker.total_cost,
            "today_cost": tracker.get_today_cost(),
            "sessions": len(tracker.get_recent_sessions(7)),
            "detailed": detailed
        }
        if detailed:
            cost_data["breakdown"] = tracker.get_cost_breakdown()
        click.echo(_json.dumps(cost_data))
    else:
        display.show_costs(detailed=detailed)

@monitor_group.command("dashboard")
@click.option("--port", default=8000, help="Port for dashboard server")
@click.option("--bind", default="127.0.0.1", help="Bind address")
@click.option("--open-browser", is_flag=True, help="Open dashboard in browser")
def monitor_dashboard(port: int, bind: str, open_browser: bool, advanced: bool = False):
    """Launch real-time dashboard with cost monitoring and agent orchestration."""
    
    # Check if server is already running
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((bind, port))
    sock.close()
    
    if result == 0:
        click.echo(f"🌐 Dashboard already running at http://{bind}:{port}")
        if open_browser:
            import webbrowser
            webbrowser.open(f"http://{bind}:{port}")
        return
    
    if PID_FILE.exists():
        click.echo("🔄 Stopping existing server...")
        try:
            with open(PID_FILE) as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)  # Allow graceful shutdown
        except (OSError, ValueError, ProcessLookupError):
            pass
        
        if PID_FILE.exists():
            PID_FILE.unlink()
    
    # Start server
    click.echo(f"🚀 Starting dashboard server on {bind}:{port}")
    if open_browser:
        click.echo("🌐 Opening dashboard in browser...")
    
    try:
        from agentsmcp.server import create_app
        import uvicorn
        
        # Store PID for management
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        
        # Open browser after a delay if requested
        if open_browser:
            def open_browser_delayed():
                time.sleep(2)  # Wait for server to start
                import webbrowser
                webbrowser.open(f"http://{bind}:{port}")
            
            browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
            browser_thread.start()
        
        app = create_app()
        uvicorn.run(app, host=bind, port=port, log_level="info")
        
    except KeyboardInterrupt:
        click.echo("\n🛑 Dashboard stopped")
    except Exception as e:
        click.echo(f"❌ Failed to start dashboard: {e}")
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()

@monitor_group.command("budget")
@click.argument("amount", type=float, required=False)
@click.option("--check", is_flag=True, help="Check current budget status")
@click.option("--remaining", is_flag=True, help="Show remaining budget")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@handle_common_errors
def monitor_budget(amount: Optional[float], check: bool, remaining: bool, as_json: bool):
    """Manage monthly budget and get cost alerts."""
    
    # Lazy import cost modules with better error handling
    try:
        from agentsmcp.cost.tracker import CostTracker
        from agentsmcp.cost.budget import BudgetManager
    except ImportError:
        raise ConfigError("Budget management not available. Install with: pip install agentsmcp[cost]")

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
            click.echo("Usage: agentsmcp monitor budget <amount> or agentsmcp monitor budget --check")

# =====================================================================
# 4️⃣ KNOWLEDGE GROUP - Data & Intelligence  
# =====================================================================

@main.group(name="knowledge")
def knowledge_group():
    """Knowledge-base management and model selection."""
    pass

@knowledge_group.command("rag")
@click.pass_context
def knowledge_rag(ctx):
    """📚 Manage RAG knowledge base for enhanced agent responses."""
    # Delegate to existing rag group
    ctx.forward(rag_group)

@knowledge_group.command("models")
@click.option("--detailed", is_flag=True, help="Show detailed model specifications")
@click.option("--recommend", help="Get model recommendation for use case")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]))
def knowledge_models(detailed: bool, recommend: Optional[str], fmt: str, advanced: bool = False):
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
        score = config['performance_score']
        status = "🟢" if score >= 85 else "🟡" if score >= 70 else "🔴"
        
        click.echo(f"\n{status} {model_name}")
        click.echo(f"   📊 Performance: {score}% | 💰 Cost: ${config['cost_per_input']*1_000_000:.2f}/${config['cost_per_output']*1_000_000:.2f}/M")
        click.echo(f"   🧠 Context: {config['context_limit']:,} tokens")
        
        if detailed:
            click.echo(f"   ✨ Best for: {config['recommended_for']}")
            if config.get('limitations'):
                click.echo(f"   ⚠️  Limitations: {', '.join(config['limitations'])}")

@knowledge_group.command("optimize")
@click.option("--cost-target", type=float, help="Target cost per operation")
@click.option("--performance-min", type=int, help="Minimum performance score")
def knowledge_optimize(cost_target: Optional[float], performance_min: Optional[int]):
    """Optimize model selection for cost-effectiveness."""
    
    from agentsmcp.distributed.orchestrator import DistributedOrchestrator
    
    available_models = DistributedOrchestrator.get_available_models()
    
    # Filter models based on criteria
    filtered_models = {}
    for name, config in available_models.items():
        if cost_target and config['cost_per_input'] > cost_target:
            continue
        if performance_min and config['performance_score'] < performance_min:
            continue
        filtered_models[name] = config
    
    if not filtered_models:
        click.echo("❌ No models meet the specified criteria")
        return
    
    # Sort by cost-effectiveness (performance / cost ratio)
    sorted_models = sorted(
        filtered_models.items(),
        key=lambda x: x[1]['performance_score'] / (x[1]['cost_per_input'] * 1_000_000),
        reverse=True
    )
    
    click.echo("💡 Optimized Model Recommendations:")
    click.echo("=" * 50)
    
    for i, (name, config) in enumerate(sorted_models[:5], 1):  # Top 5
        ratio = config['performance_score'] / (config['cost_per_input'] * 1_000_000)
        click.echo(f"{i}. {name}")
        click.echo(f"   📊 Performance: {config['performance_score']}%")
        click.echo(f"   💰 Cost: ${config['cost_per_input']*1_000_000:.2f}/M tokens")
        click.echo(f"   🎯 Cost-effectiveness: {ratio:.1f}")

# =====================================================================
# 5️⃣ SERVER GROUP - Infrastructure & Integration
# =====================================================================

@main.group(name="server")
def server_group():
    """Server lifecycle and integration utilities."""
    pass

# Server commands would need to be implemented based on existing server logic
@server_group.command("start")
@click.option("--port", default=8000, help="Port to run server on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
def server_start(port, host):
    """Start the AgentsMCP API server."""
    click.echo(f"🚀 Starting AgentsMCP server on {host}:{port}")
    # Implementation would start the FastAPI server

@server_group.command("stop")
def server_stop():
    """Stop the AgentsMCP API server."""
    click.echo("🛑 Stopping AgentsMCP server")

@server_group.command("status")
def server_status():
    """Show API server status."""
    click.echo("📊 AgentsMCP Server Status")

@server_group.command("mcp")
@click.pass_context
def server_mcp(ctx):
    """Manage MCP servers in configuration."""
    # Don't forward the advanced parameter since mcp_group doesn't expect it
    ctx.invoke(mcp_group)

@server_group.command("roles")
@click.pass_context
def server_roles(ctx):
    """🛠️ Role‑based orchestration commands."""
    ctx.forward(roles_group)

# =====================================================================
# 6️⃣ CONFIG GROUP - Advanced Configuration
# =====================================================================

@main.group(name="config")
def config_group():
    """Advanced configuration utilities."""
    pass

@config_group.command("show")
@click.option("--format", "fmt", default="yaml", type=click.Choice(["yaml", "json"]))
def config_show(fmt):
    """Display current configuration."""
    click.echo("📋 Current AgentsMCP Configuration:")
    if fmt == "json":
        click.echo('{"status": "configuration display not yet implemented"}')
    else:
        click.echo("# Configuration display not yet implemented")

@config_group.command("edit")
def config_edit():
    """Edit configuration files."""
    click.echo("✏️  Opening configuration editor...")
    click.echo("💡 Use: agentsmcp init setup for guided configuration")

@config_group.command("validate")
def config_validate():
    """Validate configuration."""
    click.echo("✅ Configuration validation not yet implemented")

# =====================================================================
# BACKWARD COMPATIBILITY ALIASES
# =====================================================================

# Add flat command aliases for backward compatibility
@main.command("simple", hidden=True)
@click.argument("task")
@advanced_option("--complexity", default=SmartDefaults.get_complexity_default(), 
                type=click.Choice(["simple", "moderate", "complex"]), 
                advanced=False, help="Task complexity level")
@advanced_option("--cost-sensitive", is_flag=True, advanced=False,
                help="Prefer cost-effective models")
@advanced_option("--timeout", default=SmartDefaults.get_timeout_default(), 
                type=int, advanced=True, help="Task timeout in seconds")
@advanced_option("--max-retries", default=3, type=int, advanced=True,
                help="Maximum retry attempts on failure")
@advanced_option("--enable-debug", is_flag=True, advanced=True,
                help="Enable verbose debug output")
@click.pass_context
def simple_alias(ctx, task: str, complexity: str, cost_sensitive: bool, timeout: int,
                max_retries: int, enable_debug: bool, advanced: bool = False):
    """[ALIAS] Execute a task using simple orchestration."""
    # Forward to the hierarchical command with all parameters
    ctx.forward(run_simple, task=task, complexity=complexity, cost_sensitive=cost_sensitive,
                timeout=timeout, max_retries=max_retries, enable_debug=enable_debug, advanced=advanced)

@main.command("interactive", hidden=True)
@click.option("--theme", default="auto", type=click.Choice(["auto", "light", "dark"]))
@click.option("--no-welcome", is_flag=True, help="Skip welcome screen")
@click.option("--refresh-interval", default=2.0, type=float, help="Auto-refresh interval")
@click.option("--orchestrator-model", default="gpt-5", help="Orchestrator model")
@click.option("--agent", "agent_type", default="ollama-turbo-coding", help="Default AI agent")
@click.pass_context
def interactive_alias(ctx, theme: str, no_welcome: bool, refresh_interval: float, 
                     orchestrator_model: str, agent_type: str):
    """[ALIAS] Launch enhanced interactive CLI."""
    ctx.forward(run_interactive)

@main.command("costs", hidden=True)
@click.option("--detailed", is_flag=True, help="Show detailed cost breakdown")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]))
@click.pass_context
def costs_alias(ctx, detailed: bool, fmt: str):
    """[ALIAS] Display current costs and budget status."""
    ctx.forward(monitor_costs)

@main.command("budget", hidden=True)
@click.argument("amount", type=float, required=False)
@click.option("--check", is_flag=True, help="Check current budget status")
@click.option("--remaining", is_flag=True, help="Show remaining budget")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def budget_alias(ctx, amount: Optional[float], check: bool, remaining: bool, as_json: bool):
    """[ALIAS] Manage monthly budget and get cost alerts."""
    ctx.forward(monitor_budget)

@main.command("dashboard", hidden=True)
@click.option("--port", default=8000, help="Port for dashboard server")
@click.option("--bind", default="127.0.0.1", help="Bind address")
@click.option("--open-browser", is_flag=True, help="Open dashboard in browser")
@click.pass_context
def dashboard_alias(ctx, port: int, bind: str, open_browser: bool):
    """[ALIAS] Launch real-time dashboard."""
    ctx.forward(monitor_dashboard)

@main.command("models", hidden=True)
@click.option("--detailed", is_flag=True, help="Show detailed model specifications")
@click.option("--recommend", help="Get model recommendation for use case")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]))
@click.pass_context
def models_alias(ctx, detailed: bool, recommend: Optional[str], fmt: str):
    """[ALIAS] Show available orchestrator models."""
    ctx.forward(knowledge_models)

@main.command("optimize", hidden=True)
@click.option("--cost-target", type=float, help="Target cost per operation")
@click.option("--performance-min", type=int, help="Minimum performance score")
@click.pass_context
def optimize_alias(ctx, cost_target: Optional[float], performance_min: Optional[int]):
    """[ALIAS] Optimize model selection for cost-effectiveness."""
    ctx.forward(knowledge_optimize)

@main.command("setup", hidden=True)
@click.pass_context
def setup_alias(ctx):
    """[ALIAS] Interactive wizard for AgentsMCP configuration."""
    ctx.forward(init_setup)

## Removed conflicting top-level 'config' alias to allow the 'config' group
## (with subcommands like 'edit', 'show', 'validate') to work as expected.

@main.command("first-run", hidden=False)
@click.pass_context
def first_run_alias(ctx):
    """Guided first‑run onboarding (alias to 'init onboarding')."""
    ctx.forward(init_onboarding)

# =====================================================================
# 🔮 SUGGESTIONS - Context-aware intelligent suggestions
# =====================================================================

@main.command("suggest")
@click.option("--all", is_flag=True, help="Show all available suggestions")
@click.option("--after", help="Show suggestions after a specific command")
@click.pass_context
def suggest(ctx, all: bool, after: Optional[str]):
    """💡 Get intelligent suggestions for what to do next."""
    from agentsmcp.intelligent_suggestions import get_suggestion_system, display_suggestions
    
    suggestion_system = get_suggestion_system()
    
    if all:
        # Show all types of suggestions
        suggestions = suggestion_system.suggest_next_actions()
        display_suggestions(suggestions, "💡 All Suggestions")
        
        # Also show frequently used commands
        frequently_used = suggestion_system.usage_tracker.get_frequently_used(5)
        if frequently_used:
            click.echo(f"\n{click.style('📊 Frequently Used Commands', fg='blue', bold=True)}")
            click.echo("─" * 25)
            for i, (cmd, count) in enumerate(frequently_used, 1):
                click.echo(f"{i}. {click.style(cmd, fg='cyan')} ({count} times)")
    
    elif after:
        # Show suggestions after a specific command
        suggestions = suggestion_system.suggest_next_actions(after)
        display_suggestions(suggestions, f"💡 After '{after}'")
    
    else:
        # Show contextual suggestions
        suggestions = suggestion_system.suggest_next_actions()
        if suggestions:
            display_suggestions(suggestions[:5], "💡 Suggestions")
        else:
            click.echo("💡 No specific suggestions right now.")
            click.echo("   Try running --all to see all available options.")

# Add group aliases that forward to the group commands
@main.command("rag", hidden=True)
@click.pass_context
def rag_alias(ctx):
    """[ALIAS] Manage RAG knowledge base."""
    ctx.forward(knowledge_rag)

@main.command("mcp", hidden=True) 
@click.pass_context
def mcp_alias(ctx):
    """[ALIAS] Manage MCP servers in configuration."""
    ctx.forward(server_mcp)

@main.command("roles", hidden=True)
@click.pass_context
def roles_alias(ctx):
    """[ALIAS] Role‑based orchestration commands."""
    ctx.forward(server_roles)

if __name__ == "__main__":
    main()
