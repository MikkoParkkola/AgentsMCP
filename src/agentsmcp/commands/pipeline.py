"""
Multi-Agent CI Pipeline CLI Commands

Provides a complete Click-based CLI for managing AgentsMCP pipelines including
creating from templates, running with beautiful Rich UI, validation, and log viewing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from pydantic import ValidationError
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

# Import pipeline components
from agentsmcp.pipeline.schema import PipelineSpec
from agentsmcp.config.pipeline_config import (
    PipelineConfig,
    PipelineConfigError,
    load_pipeline_config,
    create_default_pipeline_config
)
from agentsmcp.pipeline.core import PipelineEngine
from agentsmcp.pipeline.monitor import create_monitor, MonitoringConfig
from agentsmcp.templates.manager import (
    TemplateManager,
    TemplateNotFoundError,
    TemplateRenderingError, 
    TemplateValidationError
)

logger = logging.getLogger(__name__)
console = Console(highlight=False)

# Template manager for loading pipeline templates
_template_manager = None

def get_template_manager():
    """Get singleton template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager


@click.group(name="pipeline", help="🛠️  Multi-Agent CI Pipeline management")
def pipeline():
    """Multi-Agent CI Pipeline commands for orchestrating AI agents in CI/CD workflows."""
    pass


@pipeline.command()
@click.option(
    "-t", "--template", 
    required=True,
    help="Template to scaffold from (use --list to see available templates)"
)
@click.option(
    "-o", "--output", 
    type=click.Path(dir_okay=False, writable=True), 
    default="agentsmcp.pipeline.yml",
    help="Output file path"
)
@click.option(
    "--var",
    multiple=True,
    help="Template variables in KEY=VALUE format (can be repeated)"
)
@click.option(
    "--list",
    is_flag=True,
    help="List available templates and exit"
)
@click.option(
    "--no-ui",
    is_flag=True,
    help="Output raw JSON instead of Rich formatting"
)
def create(template: str, output: str, var: tuple, list: bool, no_ui: bool):
    """🛠️  Create a new pipeline from a built-in template."""
    
    template_mgr = get_template_manager()
    
    # Handle --list option
    if list:
        templates = template_mgr.list_templates()
        if no_ui:
            template_info = []
            for tmpl_name in templates:
                info = template_mgr.get_template_info(tmpl_name)
                template_info.append(info)
            console.print_json(data=template_info)
        else:
            console.print(Panel(
                "📦 Available Pipeline Templates",
                title="Templates",
                border_style="bright_blue"
            ))
            
            table = Table(box=box.ROUNDED)
            table.add_column("Template", style="cyan", no_wrap=True)
            table.add_column("Description", style="blue")
            
            for tmpl_name in templates:
                info = template_mgr.get_template_info(tmpl_name)
                table.add_row(tmpl_name, info.get("description", "No description"))
            
            console.print(table)
            console.print("\n💡 Usage:")
            console.print("   agentsmcp pipeline create -t python-package -o my-pipeline.yml")
            console.print("   agentsmcp pipeline create -t node-app --var repository=myorg/myapp")
        return
    
    if not template:
        console.print("[red]❌ Template name is required. Use --list to see available templates.[/red]")
        sys.exit(1)
    
    try:
        # Parse template variables
        variables = {}
        for var_pair in var:
            if "=" not in var_pair:
                console.print(f"[red]❌ Invalid variable format: {var_pair}. Use KEY=VALUE format.[/red]")
                sys.exit(1)
            key, value = var_pair.split("=", 1)
            # Try to parse as JSON for booleans, numbers, etc.
            try:
                import json
                variables[key] = json.loads(value)
            except json.JSONDecodeError:
                variables[key] = value  # Keep as string
        
        # Load and render template
        pipeline_config = template_mgr.create_pipeline_from_template(
            template, output, variables=variables
        )
        
        result = {
            "status": "created",
            "template": template,
            "pipeline_name": pipeline_config["name"],
            "output_path": str(output),
            "stages": len(pipeline_config.get("stages", []))
        }
        
        if no_ui:
            console.print_json(data=result)
        else:
            # Beautiful Rich output
            table = Table(title="📋 Pipeline Configuration", box=box.ROUNDED)
            table.add_column("Property", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")
            
            table.add_row("Template", template)
            table.add_row("Pipeline Name", pipeline_config["name"])
            table.add_row("Stages", str(len(pipeline_config.get("stages", []))))
            table.add_row("Output Path", str(output))
            
            if variables:
                vars_text = ", ".join(f"{k}={v}" for k, v in variables.items())
                table.add_row("Variables", vars_text)
            
            console.print(Panel(
                table,
                title="✅ Pipeline Created Successfully",
                border_style="green"
            ))
            
            console.print(f"💡 Next steps:")
            console.print(f"   • Edit [cyan]{output}[/] to customize your pipeline")
            console.print(f"   • Run [cyan]agentsmcp pipeline validate {output}[/] to verify")
            console.print(f"   • Execute with [cyan]agentsmcp pipeline run {output}[/]")
            
    except TemplateNotFoundError as e:
        error_msg = {"status": "error", "type": "template_not_found", "message": str(e)}
        if no_ui:
            console.print_json(data=error_msg)
        else:
            console.print(f"[red]❌ {e}[/red]")
            console.print("Use [cyan]--list[/] to see available templates.")
        sys.exit(1)
    except TemplateRenderingError as e:
        error_msg = {"status": "error", "type": "rendering_error", "message": str(e)}
        if no_ui:
            console.print_json(data=error_msg)
        else:
            console.print(f"[red]❌ Template rendering failed: {e}[/red]")
        sys.exit(1)
    except TemplateValidationError as e:
        error_msg = {"status": "error", "type": "validation_error", "message": str(e)}
        if no_ui:
            console.print_json(data=error_msg)
        else:
            console.print(f"[red]❌ Template validation failed: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        error_msg = {"status": "error", "message": str(e)}
        if no_ui:
            console.print_json(data=error_msg)
        else:
            console.print(f"[red]❌ Failed to create pipeline: {e}[/red]")
        sys.exit(1)


@pipeline.command()
@click.option(
    "--detailed",
    is_flag=True,
    help="Show detailed template information"
)
@click.option(
    "--no-ui",
    is_flag=True,
    help="Output raw JSON instead of Rich formatting"
)
def list(detailed: bool, no_ui: bool):
    """📄  List available pipeline templates."""
    
    template_mgr = get_template_manager()
    templates = template_mgr.list_templates()
    
    if no_ui:
        template_info = []
        for tmpl_name in templates:
            info = template_mgr.get_template_info(tmpl_name)
            if detailed:
                template_info.append(info)
            else:
                template_info.append({
                    "name": tmpl_name,
                    "description": info.get("description", "No description"),
                    "available": info.get("available", False)
                })
        console.print_json(data=template_info)
        return
    
    # Beautiful Rich table
    table = Table(
        title="🧩 Available Pipeline Templates",
        box=box.ROUNDED,
        header_style="bold magenta"
    )
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="blue")
    
    if detailed:
        table.add_column("Stages", justify="right", style="yellow")
        table.add_column("Status", style="green")
    
    for tmpl_name in templates:
        info = template_mgr.get_template_info(tmpl_name)
        if detailed:
            status = "✅ Available" if info.get("available", False) else "❌ Error"
            table.add_row(
                tmpl_name,
                info.get("description", "No description"),
                str(info.get("stages", 0)),
                status
            )
        else:
            table.add_row(
                tmpl_name,
                info.get("description", "No description")
            )
    
    console.print(Panel(
        table,
        title="📦 Pipeline Templates",
        border_style="bright_blue"
    ))
    
    console.print("\n💡 Usage:")
    console.print("   agentsmcp pipeline create -t python-package -o my-pipeline.yml")
    console.print("   agentsmcp pipeline create -t node-app --var repository=myorg/myapp")


@pipeline.command()
@click.argument(
    "pipeline_file",
    type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option(
    "--no-ui",
    is_flag=True,
    help="Output raw JSON instead of Rich formatting"
)
def validate(pipeline_file: str, no_ui: bool):
    """🔍  Validate a pipeline configuration file."""
    
    file_path = Path(pipeline_file)
    
    try:
        # Load and validate the pipeline configuration
        config = load_pipeline_config(file_path)
        
        # Additional validation
        config.validate()
        
        result = {
            "status": "valid",
            "file": str(file_path),
            "pipeline_name": config.spec.name,
            "version": config.spec.version,
            "stages": len(config.spec.stages),
            "agent_types": list(config.get_agent_types()),
            "stage_names": config.get_stage_names()
        }
        
        if no_ui:
            console.print_json(data=result)
        else:
            # Beautiful validation report
            status_text = Text()
            status_text.append("✅ Pipeline: ", style="bold")
            status_text.append(config.spec.name, style="bold green")
            status_text.append(f" (v{config.spec.version})")
            
            details = [
                f"📄 [bold]File:[/] {file_path.name}",
                f"🏗️  [bold]Stages:[/] {len(config.spec.stages)}",
                f"🤖 [bold]Agent Types:[/] {', '.join(config.get_agent_types())}",
            ]
            
            if config.spec.description:
                details.insert(1, f"📝 [bold]Description:[/] {config.spec.description}")
            
            console.print(Panel(
                "\n".join(details),
                title="✅ Validation Successful",
                border_style="green"
            ))
            
            # Show stage breakdown
            stage_table = Table(title="📋 Pipeline Stages", box=box.SIMPLE)
            stage_table.add_column("Stage", style="cyan")
            stage_table.add_column("Agents", style="yellow")
            stage_table.add_column("Parallel", justify="center", style="green")
            
            for stage in config.spec.stages:
                agents_text = f"{len(stage.agents)} agents" if stage.agents else "inherit defaults"
                parallel_emoji = "✅" if stage.parallel else "❌"
                stage_table.add_row(stage.name, agents_text, parallel_emoji)
            
            console.print(stage_table)
            
    except ValidationError as e:
        # Pydantic validation error
        error_details = []
        for error in e.errors():
            location = " → ".join(str(loc) for loc in error["loc"])
            error_details.append(f"[red]•[/red] {location}: {error['msg']}")
        
        result = {
            "status": "invalid",
            "file": str(file_path),
            "errors": [{"location": " → ".join(str(loc) for loc in err["loc"]), "message": err["msg"]} for err in e.errors()]
        }
        
        if no_ui:
            console.print_json(data=result)
            sys.exit(1)
        else:
            error_panel = Panel(
                "\n".join(error_details),
                title="❌ Validation Errors",
                border_style="red"
            )
            console.print(error_panel)
            console.print(f"\n💡 Fix the errors above in [cyan]{file_path}[/] and try again")
            sys.exit(1)
            
    except PipelineConfigError as e:
        result = {"status": "error", "file": str(file_path), "message": str(e)}
        
        if no_ui:
            console.print_json(data=result)
            sys.exit(1)
        else:
            console.print(f"[red]❌ Configuration Error: {e}[/red]")
            sys.exit(1)
            
    except Exception as e:
        result = {"status": "error", "file": str(file_path), "message": str(e)}
        
        if no_ui:
            console.print_json(data=result)
            sys.exit(1)
        else:
            console.print(f"[red]❌ Unexpected error: {e}[/red]")
            sys.exit(1)


@pipeline.command()
@click.argument(
    "pipeline_file",
    type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option(
    "-s", "--stage",
    help="Run only a specific stage"
)
@click.option(
    "--no-ui",
    is_flag=True,
    help="Output raw JSON instead of Rich live UI"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate and show execution plan without running"
)
def run(pipeline_file: str, stage: Optional[str], no_ui: bool, dry_run: bool):
    """🚀  Execute a pipeline with beautiful real-time progress tracking."""
    asyncio.run(_run_async(pipeline_file, stage, no_ui, dry_run))


async def _run_async(pipeline_file: str, stage: Optional[str], no_ui: bool, dry_run: bool):
    """Async implementation of pipeline execution."""
    file_path = Path(pipeline_file)
    
    try:
        # Load and validate pipeline
        config = load_pipeline_config(file_path)
        
        # Apply defaults to get final execution spec
        execution_spec = config.spec.apply_defaults()
        
        # Create filtered spec if single stage requested
        if stage:
            stage_spec = execution_spec.get_stage(stage)
            if not stage_spec:
                available_stages = [s.name for s in execution_spec.stages]
                if no_ui:
                    error = {"status": "error", "message": f"Stage '{stage}' not found. Available: {available_stages}"}
                    console.print_json(data=error)
                    sys.exit(1)
                else:
                    console.print(f"[red]❌ Stage '{stage}' not found[/red]")
                    console.print(f"Available stages: {', '.join(available_stages)}")
                    sys.exit(1)
            # Create filtered spec with single stage
            execution_spec = execution_spec.copy(update={"stages": [stage_spec]})
        
        stages_to_run = execution_spec.stages
        
        if dry_run:
            # Show execution plan
            plan = {
                "pipeline": execution_spec.name,
                "version": execution_spec.version,
                "stages_to_run": len(stages_to_run),
                "total_agents": sum(len(s.agents) for s in stages_to_run),
                "execution_plan": [
                    {
                        "stage": s.name,
                        "agents": len(s.agents),
                        "parallel": s.parallel,
                        "agent_types": [a.type for a in s.agents]
                    }
                    for s in stages_to_run
                ]
            }
            
            if no_ui:
                console.print_json(data=plan)
            else:
                console.print(Panel(
                    f"🎯 [bold]Pipeline:[/] {execution_spec.name}\n"
                    f"📋 [bold]Stages to run:[/] {len(stages_to_run)}\n"
                    f"🤖 [bold]Total agents:[/] {sum(len(s.agents) for s in stages_to_run)}",
                    title="📊 Execution Plan (Dry Run)",
                    border_style="yellow"
                ))
                
                for stage_spec in stages_to_run:
                    agent_details = [f"• {a.type} ({a.model}) - {a.task}" for a in stage_spec.agents]
                    stage_panel = Panel(
                        "\n".join(agent_details),
                        title=f"🏗️  Stage: {stage_spec.name} ({'parallel' if stage_spec.parallel else 'sequential'})",
                        border_style="cyan"
                    )
                    console.print(stage_panel)
            return
        
        # Execute pipeline with real engine and monitoring
        run_id = f"{execution_spec.name}-{int(time.time())}"
        
        # Create pipeline engine
        engine = PipelineEngine(config)
        
        # Create monitor with appropriate configuration
        monitor_config = MonitoringConfig(
            enable_ui=not no_ui,
            json_output=no_ui,
            json_interval=1.0 if no_ui else 0.25
        )
        monitor = create_monitor(
            engine.tracker, 
            execution_spec.name,
            enable_ui=not no_ui,
            json_output=no_ui
        )
        
        if not no_ui:
            # Show execution header
            console.rule(f"[bold bright_green]🚀 Pipeline Execution: {execution_spec.name}")
            
            header_info = [
                f"[cyan]Pipeline:[/] {execution_spec.name} (v{execution_spec.version})",
                f"[cyan]Run ID:[/] {run_id}",
                f"[cyan]Stages:[/] {len(stages_to_run)}",
                f"[cyan]Total Agents:[/] {sum(len(s.agents) for s in stages_to_run)}"
            ]
            
            if execution_spec.description:
                header_info.insert(1, f"[cyan]Description:[/] {execution_spec.description}")
            
            console.print(Panel(
                "\n".join(header_info),
                title="📋 Execution Info",
                border_style="blue"
            ))
            console.print()
        
        # Set up Ctrl+C handler
        interrupted = False
        def signal_handler(sig, frame):
            nonlocal interrupted
            interrupted = True
            if not no_ui:
                console.print("\n[yellow]⚠️  Stopping pipeline execution...[/]")
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Start monitoring
            async with monitor:
                # Execute the pipeline
                pipeline_task = asyncio.create_task(engine.run_async(execution_spec))
                
                # Wait for completion or interruption
                while not pipeline_task.done():
                    if interrupted:
                        pipeline_task.cancel()
                        try:
                            await pipeline_task
                        except asyncio.CancelledError:
                            pass
                        break
                    await asyncio.sleep(0.1)
                
                if not interrupted:
                    # Get results
                    result = await pipeline_task
                    
                    if no_ui:
                        # JSON output mode
                        output = {
                            "run_id": run_id,
                            "pipeline": execution_spec.name,
                            "status": "completed" if result.get("pipeline_success") else "failed",
                            "duration": result.get("duration", 0.0),
                            "stages_completed": result.get("stages_completed", 0),
                            "total_stages": result.get("total_stages", 0),
                            "result": result
                        }
                        console.print_json(data=output)
                    else:
                        # Rich UI summary
                        success = result.get("pipeline_success", False)
                        if success:
                            console.print(Panel(
                                f"✅ [bold green]Pipeline completed successfully![/]\n\n"
                                f"📊 [bold]Summary:[/]\n"
                                f"   • Stages completed: {result.get('stages_completed', 0)}/{result.get('total_stages', 0)}\n"
                                f"   • Duration: {result.get('duration', 0.0):.2f}s\n"
                                f"   • Run ID: {run_id}",
                                title="🎉 Execution Complete",
                                border_style="bright_green"
                            ))
                        else:
                            console.print(Panel(
                                f"❌ [bold red]Pipeline completed with failures![/]\n\n"
                                f"📊 [bold]Summary:[/]\n"
                                f"   • Stages completed: {result.get('stages_completed', 0)}/{result.get('total_stages', 0)}\n"
                                f"   • Duration: {result.get('duration', 0.0):.2f}s\n"
                                f"   • Run ID: {run_id}",
                                title="❌ Execution Failed",
                                border_style="red"
                            ))
                else:
                    if no_ui:
                        console.print_json(data={"status": "interrupted", "run_id": run_id})
                    else:
                        console.print(Panel(
                            "Pipeline execution was interrupted by user",
                            title="⚠️  Interrupted",
                            border_style="yellow"
                        ))
                        
        except Exception as execution_error:
            if no_ui:
                console.print_json(data={
                    "status": "error",
                    "run_id": run_id,
                    "error": str(execution_error)
                })
            else:
                console.print(f"[red]❌ Pipeline execution failed: {execution_error}[/red]")
            sys.exit(1)
        
    except Exception as e:
        error_result = {"status": "error", "pipeline": str(file_path), "message": str(e)}
        
        if no_ui:
            console.print_json(data=error_result)
            sys.exit(1)
        else:
            console.print(f"[red]❌ Pipeline setup failed: {e}[/red]")
            sys.exit(1)


@pipeline.command()
@click.argument(
    "run_id_or_pipeline",
    required=False
)
@click.option(
    "--follow", "-f",
    is_flag=True,
    help="Continuously follow log output (like tail -f)"
)
@click.option(
    "--lines", "-n",
    type=int,
    default=50,
    help="Number of recent lines to show"
)
@click.option(
    "--no-ui",
    is_flag=True,
    help="Output raw JSON log entries"
)
def logs(run_id_or_pipeline: Optional[str], follow: bool, lines: int, no_ui: bool):
    """📜  View pipeline execution logs."""
    
    # Placeholder implementation - in real system would read from log files
    if not run_id_or_pipeline:
        if no_ui:
            console.print_json(data={"error": "No run ID or pipeline file specified"})
            sys.exit(1)
        else:
            console.print("[red]❌ Please specify a run ID or pipeline file[/red]")
            console.print("Usage: agentsmcp pipeline logs <run-id>")
            console.print("       agentsmcp pipeline logs my-pipeline.yml")
            sys.exit(1)
    
    # Mock log entries for demonstration
    mock_logs = [
        {"timestamp": time.time() - 300, "level": "INFO", "stage": "install-deps", "agent": "ollama-turbo", "message": "Installing requirements.txt"},
        {"timestamp": time.time() - 250, "level": "INFO", "stage": "install-deps", "agent": "ollama-turbo", "message": "Dependencies installed successfully"},
        {"timestamp": time.time() - 200, "level": "INFO", "stage": "test", "agent": "claude", "message": "Running pytest suite"},
        {"timestamp": time.time() - 150, "level": "WARN", "stage": "test", "agent": "claude", "message": "2 tests skipped due to missing fixtures"},
        {"timestamp": time.time() - 100, "level": "INFO", "stage": "test", "agent": "claude", "message": "Tests completed: 15 passed, 2 skipped"},
        {"timestamp": time.time() - 50, "level": "INFO", "stage": "build", "agent": "codex", "message": "Building package wheel"},
        {"timestamp": time.time() - 10, "level": "INFO", "stage": "build", "agent": "codex", "message": "Build completed successfully"}
    ]
    
    if no_ui:
        # Show recent logs as JSON
        recent_logs = mock_logs[-lines:] if lines < len(mock_logs) else mock_logs
        console.print_json(data=recent_logs)
        return
    
    # Beautiful Rich log display
    console.rule(f"[bold bright_blue]📜 Pipeline Logs: {run_id_or_pipeline}")
    
    def format_log_entry(entry):
        timestamp = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
        level_colors = {
            "INFO": "green",
            "WARN": "yellow", 
            "ERROR": "red",
            "DEBUG": "dim"
        }
        level_color = level_colors.get(entry["level"], "white")
        
        return Panel(
            f"[{level_color}]{entry['level']}[/] {entry['message']}",
            title=f"🏗️  {entry['stage']} → 🤖 {entry['agent']} [{timestamp}]",
            border_style=level_color,
            padding=(0, 1)
        )
    
    # Show recent logs
    recent_logs = mock_logs[-lines:] if lines < len(mock_logs) else mock_logs
    for log_entry in recent_logs:
        console.print(format_log_entry(log_entry))
    
    if follow:
        console.print(Panel(
            "👀 Following logs... Press Ctrl+C to stop",
            border_style="dim"
        ))
        
        try:
            # Simulate live log following
            while True:
                time.sleep(2)
                # In real implementation, would read new log entries
                new_entry = {
                    "timestamp": time.time(),
                    "level": "INFO", 
                    "stage": "monitor",
                    "agent": "system",
                    "message": "Pipeline monitoring active"
                }
                console.print(format_log_entry(new_entry))
        except KeyboardInterrupt:
            console.print("\n👋 Stopped following logs")


# Export the command group for integration with main CLI
__all__ = ["pipeline"]