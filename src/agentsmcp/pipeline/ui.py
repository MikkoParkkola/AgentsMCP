"""
Rich UI components for the AgentsMCP pipeline.

Provides beautiful, real-time UI components for monitoring pipeline execution:
- PipelineMonitor: Main UI orchestrator with Rich Live display
- StageProgressTracker: Individual stage progress bars with timing
- AgentStatusPanel: Real-time agent execution status table
- ErrorFormatter: Beautiful error displays with context and suggestions

All components follow AgentsMCP design patterns and integrate seamlessly
with the existing Rich-based CLI ecosystem.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Any, Dict, List, Optional, Mapping
from dataclasses import dataclass

from rich import box
from rich.align import Align
from rich.console import Console, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .core import ExecutionTracker

# State styling and emojis for pipeline states
STATE_STYLES = {
    "pending": "dim",
    "running": "bold yellow", 
    "completed": "bold green",
    "successful": "bold green",
    "failed": "bold red",
    "skipped": "bold cyan",
    "cancelled": "bold magenta"
}

STATE_EMOJIS = {
    "pending": "⏳",
    "running": "🚀",
    "completed": "✅", 
    "successful": "✅",
    "failed": "❌",
    "skipped": "⏭️",
    "cancelled": "🛑"
}


def get_state_emoji(state: str) -> str:
    """Get emoji for a pipeline state."""
    return STATE_EMOJIS.get(state, "⚪")


def get_state_style(state: str) -> str:
    """Get Rich style for a pipeline state."""
    return STATE_STYLES.get(state, "")


@dataclass
class PipelineStats:
    """Pipeline execution statistics."""
    total_stages: int = 0
    completed_stages: int = 0
    failed_stages: int = 0
    total_agents: int = 0
    running_agents: int = 0
    completed_agents: int = 0
    failed_agents: int = 0
    start_time: Optional[dt.datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_stages == 0:
            return 0.0
        return (self.completed_stages / self.total_stages) * 100
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if not self.start_time:
            return 0.0
        return (dt.datetime.now() - self.start_time).total_seconds()


class ErrorFormatter:
    """Formats errors and exceptions for beautiful display."""
    
    @staticmethod
    def format_error(error: Any, stage_name: str = "", agent_name: str = "") -> Panel:
        """Format an error as a Rich Panel with context."""
        title = "❌ Error"
        if stage_name:
            title += f" in {stage_name}"
        if agent_name:
            title += f" ({agent_name})"
            
        if isinstance(error, Exception):
            content = Text()
            content.append(f"{type(error).__name__}: ", style="bold red")
            content.append(str(error), style="red")
        elif isinstance(error, dict) and "error" in error:
            content = Text()
            content.append("Error: ", style="bold red") 
            content.append(str(error["error"]), style="red")
        else:
            content = Text(str(error), style="red")
            
        return Panel(
            content,
            title=title,
            border_style="red",
            expand=False
        )
    
    @staticmethod
    def format_suggestions(error: Any) -> Optional[Panel]:
        """Format helpful suggestions based on error type."""
        suggestions = []
        
        error_str = str(error).lower()
        if "timeout" in error_str:
            suggestions.append("• Try increasing the timeout_seconds value")
            suggestions.append("• Check if the agent service is responsive")
        elif "connection" in error_str or "network" in error_str:
            suggestions.append("• Verify network connectivity")
            suggestions.append("• Check if the agent service is running")
        elif "authentication" in error_str or "auth" in error_str:
            suggestions.append("• Check API keys and credentials")
            suggestions.append("• Verify agent configuration")
        elif "rate limit" in error_str or "quota" in error_str:
            suggestions.append("• Add delays between agent calls")
            suggestions.append("• Check API usage limits")
            
        if not suggestions:
            return None
            
        content = Text()
        content.append("💡 Suggestions:\n", style="bold cyan")
        for suggestion in suggestions:
            content.append(f"{suggestion}\n", style="cyan")
            
        return Panel(
            content,
            title="Troubleshooting",
            border_style="cyan",
            expand=False
        )


class StageProgressTracker:
    """Tracks and displays progress for a single pipeline stage."""
    
    def __init__(self, stage_name: str, agent_count: int):
        self.stage_name = stage_name
        self.agent_count = max(agent_count, 1)  # Avoid division by zero
        self.state = "pending"
        self.completed_agents = 0
        self.failed_agents = 0
        
        # Create Rich Progress with beautiful columns
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            expand=True
        )
        
        # Add the main task
        self.task_id = self.progress.add_task(
            f"{get_state_emoji('pending')} {stage_name}",
            total=self.agent_count
        )
    
    def update_state(self, state: str, completed: int = None, failed: int = None) -> None:
        """Update stage state and progress."""
        self.state = state
        if completed is not None:
            self.completed_agents = completed
        if failed is not None:
            self.failed_agents = failed
            
        # Update task description with emoji and state
        emoji = get_state_emoji(state)
        description = f"{emoji} {self.stage_name}"
        
        # Calculate total completed (successful + failed)
        total_completed = self.completed_agents + self.failed_agents
        
        self.progress.update(
            self.task_id,
            completed=total_completed,
            description=description
        )
    
    def start(self) -> None:
        """Start the progress display."""
        self.progress.start_task(self.task_id)
    
    def stop(self) -> None:
        """Stop the progress display."""
        self.progress.stop()
        
    def render(self) -> RenderableType:
        """Render the progress bar."""
        return self.progress


class AgentStatusPanel:
    """Displays a live table of agent statuses."""
    
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        
    def update_agent(self, agent_id: str, stage: str, state: str, 
                    start_time: Optional[dt.datetime] = None, 
                    error: Optional[str] = None) -> None:
        """Update agent status information."""
        if agent_id not in self.agents:
            self.agents[agent_id] = {}
            
        self.agents[agent_id].update({
            "stage": stage,
            "state": state,
            "start_time": start_time or self.agents[agent_id].get("start_time"),
            "error": error
        })
    
    def _format_duration(self, start_time: Optional[dt.datetime]) -> str:
        """Format duration since start time."""
        if not start_time:
            return "--"
        duration = (dt.datetime.now() - start_time).total_seconds()
        if duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            return f"{duration/60:.1f}m"
        else:
            return f"{duration/3600:.1f}h"
    
    def render(self) -> RenderableType:
        """Render the agent status table."""
        if not self.agents:
            return Panel(
                Text("No agents running", style="dim"),
                title="🤖 Agent Status",
                border_style="blue"
            )
        
        table = Table(
            title="🤖 Agent Status",
            box=box.ROUNDED,
            header_style="bold cyan",
            expand=True
        )
        
        table.add_column("Agent", style="cyan", no_wrap=True)
        table.add_column("Stage", style="green") 
        table.add_column("Status", style="yellow")
        table.add_column("Duration", justify="right", style="blue")
        table.add_column("Details", style="dim")
        
        for agent_id, info in self.agents.items():
            state = info.get("state", "unknown")
            emoji = get_state_emoji(state)
            style = get_state_style(state)
            
            # Format status with emoji
            status_text = Text()
            status_text.append(f"{emoji} ", style=style)
            status_text.append(state, style=style)
            
            # Details column for errors or additional info
            details = ""
            if info.get("error"):
                details = f"❌ {str(info['error'])[:50]}..."
            elif state == "running":
                details = "🔄 Processing..."
                
            table.add_row(
                agent_id,
                info.get("stage", "--"),
                status_text,
                self._format_duration(info.get("start_time")),
                details
            )
        
        return table


class PipelineMonitor:
    """Main UI orchestrator for pipeline monitoring with Rich Live display."""
    
    def __init__(self, tracker: ExecutionTracker, pipeline_name: str = "Pipeline"):
        self.tracker = tracker
        self.pipeline_name = pipeline_name
        self.console = Console()
        self.live: Optional[Live] = None
        self.stage_trackers: Dict[str, StageProgressTracker] = {}
        self.agent_panel = AgentStatusPanel()
        self.stats = PipelineStats(start_time=dt.datetime.now())
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        """Start the live monitoring display."""
        if self.live is not None:
            return  # Already started
            
        # Create initial layout
        layout = self._build_layout()
        
        # Start Rich Live display
        self.live = Live(
            layout,
            console=self.console,
            refresh_per_second=4,
            screen=False
        )
        self.live.start()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
    def stop(self) -> None:
        """Stop the live monitoring display."""
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
            
        if self.live:
            self.live.stop()
            self.live = None
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop that updates the display."""
        while not self._stop_event.is_set():
            try:
                self._update_display()
                time.sleep(0.25)  # 4 FPS
            except Exception as e:
                self.console.print(f"[red]Monitor error: {e}[/red]")
                break
    
    def _update_display(self) -> None:
        """Update the display with current status."""
        if not self.live:
            return
            
        # Get current status from tracker
        status = self.tracker.get_status_summary()
        
        # Update statistics
        self._update_stats(status)
        
        # Update stage trackers
        self._update_stage_trackers(status)
        
        # Update agent panel
        self._update_agent_panel(status) 
        
        # Rebuild and update layout
        layout = self._build_layout()
        self.live.update(layout)
    
    def _update_stats(self, status: Dict[str, Any]) -> None:
        """Update pipeline statistics from status."""
        stages = status.get("stages", {})
        self.stats.total_stages = len(stages)
        self.stats.completed_stages = sum(
            1 for s in stages.values() 
            if s in ["completed", "successful"]
        )
        self.stats.failed_stages = sum(
            1 for s in stages.values()
            if s == "failed"
        )
        
        # Count agents from stage trackers
        total_agents = 0
        running_agents = 0
        completed_agents = 0
        failed_agents = 0
        
        for agent_id, info in self.agent_panel.agents.items():
            total_agents += 1
            state = info.get("state", "pending")
            if state == "running":
                running_agents += 1
            elif state in ["completed", "successful"]:
                completed_agents += 1
            elif state == "failed":
                failed_agents += 1
        
        self.stats.total_agents = total_agents
        self.stats.running_agents = running_agents
        self.stats.completed_agents = completed_agents
        self.stats.failed_agents = failed_agents
    
    def _update_stage_trackers(self, status: Dict[str, Any]) -> None:
        """Update stage progress trackers."""
        stages = status.get("stages", {})
        agent_results = status.get("agent_results", {})
        
        for stage_name, stage_status in stages.items():
            if stage_name not in self.stage_trackers:
                # Count agents for this stage
                stage_agents = [
                    result for results_list in agent_results.values()
                    for result in results_list
                    if getattr(result, 'stage_name', '') == stage_name
                ]
                agent_count = max(len(stage_agents), 1)
                
                self.stage_trackers[stage_name] = StageProgressTracker(
                    stage_name, agent_count
                )
            
            tracker = self.stage_trackers[stage_name]
            
            # Count completed and failed agents for this stage
            completed = 0
            failed = 0
            for results_list in agent_results.values():
                for result in results_list:
                    if getattr(result, 'stage_name', '') == stage_name:
                        if getattr(result, 'success', False):
                            completed += 1
                        else:
                            failed += 1
            
            tracker.update_state(stage_status, completed, failed)
    
    def _update_agent_panel(self, status: Dict[str, Any]) -> None:
        """Update agent status panel."""
        agent_results = status.get("agent_results", {})
        
        for stage_name, results_list in agent_results.items():
            for result in results_list:
                agent_id = getattr(result, 'agent_name', 'unknown')
                state = "completed" if getattr(result, 'success', False) else "failed"
                error = getattr(result, 'error', None)
                
                self.agent_panel.update_agent(
                    agent_id=agent_id,
                    stage=stage_name,
                    state=state,
                    error=error
                )
    
    def _build_layout(self) -> Table:
        """Build the main layout for the display."""
        layout = Table.grid(expand=True)
        
        # Header with pipeline info
        header = self._build_header()
        layout.add_row(header)
        layout.add_row("")  # Spacing
        
        # Stage progress bars
        for tracker in self.stage_trackers.values():
            layout.add_row(tracker.render())
        
        if self.stage_trackers:
            layout.add_row("")  # Spacing
        
        # Agent status panel
        layout.add_row(self.agent_panel.render())
        layout.add_row("")  # Spacing
        
        # Statistics footer
        footer = self._build_footer()
        layout.add_row(footer)
        
        return layout
    
    def _build_header(self) -> Panel:
        """Build the header panel with pipeline information."""
        content = Text()
        content.append(f"🔧 ", style="bold blue")
        content.append(self.pipeline_name, style="bold blue")
        
        if self.stats.duration > 0:
            content.append(f" • ⏱️ {self.stats.duration:.1f}s", style="dim")
        
        if self.stats.total_stages > 0:
            success_rate = self.stats.success_rate
            if success_rate == 100:
                content.append(f" • ✅ {success_rate:.0f}%", style="green")
            elif success_rate > 50:
                content.append(f" • 🟡 {success_rate:.0f}%", style="yellow")
            else:
                content.append(f" • ❌ {success_rate:.0f}%", style="red")
        
        return Panel(
            Align.center(content),
            border_style="blue",
            padding=(0, 1)
        )
    
    def _build_footer(self) -> Panel:
        """Build the footer with statistics."""
        content = Table.grid(expand=True)
        content.add_column(justify="center")
        content.add_column(justify="center") 
        content.add_column(justify="center")
        content.add_column(justify="center")
        
        # Stages stats
        stages_text = Text()
        stages_text.append("📋 Stages: ", style="bold")
        stages_text.append(f"{self.stats.completed_stages}", style="green")
        stages_text.append("/", style="dim")
        stages_text.append(f"{self.stats.total_stages}", style="blue")
        if self.stats.failed_stages > 0:
            stages_text.append(f" (❌{self.stats.failed_stages})", style="red")
        
        # Agents stats
        agents_text = Text()
        agents_text.append("🤖 Agents: ", style="bold")
        if self.stats.running_agents > 0:
            agents_text.append(f"🚀{self.stats.running_agents} ", style="yellow")
        agents_text.append(f"✅{self.stats.completed_agents}", style="green")
        if self.stats.failed_agents > 0:
            agents_text.append(f" ❌{self.stats.failed_agents}", style="red")
        
        # Timing
        timing_text = Text()
        timing_text.append("⏱️ Duration: ", style="bold")
        timing_text.append(f"{self.stats.duration:.1f}s", style="blue")
        
        # Status
        status_text = Text()
        status_text.append("📊 Status: ", style="bold")
        if self.stats.failed_stages > 0:
            status_text.append("Failed", style="red")
        elif self.stats.completed_stages == self.stats.total_stages and self.stats.total_stages > 0:
            status_text.append("Completed", style="green")
        else:
            status_text.append("Running", style="yellow")
        
        content.add_row(stages_text, agents_text, timing_text, status_text)
        
        return Panel(
            content,
            title="📈 Statistics",
            border_style="dim",
            padding=(0, 1)
        )
    
    def show_final_summary(self) -> None:
        """Show final summary after pipeline completion."""
        if not self.live:
            return
            
        # Build final summary
        summary = Table.grid(expand=True)
        
        # Title
        if self.stats.failed_stages > 0:
            title = Text("💥 Pipeline Completed with Errors", style="bold red")
        else:
            title = Text("🎉 Pipeline Completed Successfully", style="bold green")
        
        summary.add_row(Align.center(title))
        summary.add_row("")
        
        # Stats table
        stats_table = Table(box=box.ROUNDED, expand=True)
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Value", justify="right")
        
        stats_table.add_row(
            "Duration",
            f"{self.stats.duration:.1f}s"
        )
        stats_table.add_row(
            "Stages Completed", 
            f"[green]{self.stats.completed_stages}[/green]/{self.stats.total_stages}"
        )
        if self.stats.failed_stages > 0:
            stats_table.add_row(
                "Stages Failed",
                f"[red]{self.stats.failed_stages}[/red]"
            )
        stats_table.add_row(
            "Agents Completed",
            f"[green]{self.stats.completed_agents}[/green]/{self.stats.total_agents}"
        )
        if self.stats.failed_agents > 0:
            stats_table.add_row(
                "Agents Failed", 
                f"[red]{self.stats.failed_agents}[/red]"
            )
        stats_table.add_row(
            "Success Rate",
            f"{self.stats.success_rate:.1f}%"
        )
        
        summary.add_row(stats_table)
        
        # Update live display with summary
        self.live.update(Panel(summary, title="Final Results", border_style="blue"))
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            # Normal completion - show summary briefly
            self.show_final_summary()
            time.sleep(2)  # Let user see the summary
        else:
            # Exception occurred - show error
            error_panel = ErrorFormatter.format_error(exc_val)
            self.console.print(error_panel)
            
            # Try to show suggestions
            suggestions = ErrorFormatter.format_suggestions(exc_val)
            if suggestions:
                self.console.print(suggestions)
        
        self.stop()


# Export main classes
__all__ = [
    "PipelineMonitor",
    "StageProgressTracker", 
    "AgentStatusPanel",
    "ErrorFormatter",
    "PipelineStats"
]