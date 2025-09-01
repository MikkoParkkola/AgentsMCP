"""Self-Improvement Dashboard for TUI

This module provides a comprehensive dashboard interface for monitoring and
interacting with the AgentsMCP self-improvement system within the TUI.

SECURITY: Secure display of improvement data without exposing internals
PERFORMANCE: Real-time updates with efficient rendering - <100ms refresh
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
import json

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Metrics displayed in the dashboard."""
    
    # System status
    optimizer_active: bool = False
    current_mode: str = "disabled"
    
    # Performance metrics
    total_improvements: int = 0
    successful_implementations: int = 0
    rollbacks_performed: int = 0
    avg_completion_time: float = 0.0
    system_stability: float = 1.0
    
    # Recent activity
    recent_improvements: List[str] = None
    active_optimizations: int = 0
    
    # User feedback
    user_satisfaction: float = 0.0
    feedback_count: int = 0
    
    def __post_init__(self):
        if self.recent_improvements is None:
            self.recent_improvements = []


class ImprovementDashboard:
    """
    Real-time dashboard for AgentsMCP self-improvement system.
    
    Provides comprehensive monitoring of improvement activities, system performance,
    and user feedback with interactive controls.
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.console = Console()
        
        # Dashboard state
        self._running = False
        self._live_display: Optional[Live] = None
        self._last_update = datetime.now()
        self._metrics = DashboardMetrics()
        
        # Update tracking
        self._update_interval = 2.0  # seconds
        self._last_status_check = 0.0
        
        # Display configuration
        self.show_detailed_metrics = True
        self.show_recent_activity = True
        self.show_performance_trends = True
        self.auto_refresh = True
        
        logger.info("ImprovementDashboard initialized")
    
    async def start_dashboard(self) -> None:
        """Start the real-time improvement dashboard."""
        if self._running:
            logger.warning("Dashboard already running")
            return
        
        self._running = True
        
        try:
            with Live(
                self._create_dashboard_layout(),
                refresh_per_second=0.5,
                screen=False
            ) as live:
                self._live_display = live
                
                while self._running:
                    # Update metrics
                    await self._update_metrics()
                    
                    # Update display
                    live.update(self._create_dashboard_layout())
                    
                    await asyncio.sleep(self._update_interval)
                    
        except KeyboardInterrupt:
            logger.info("Dashboard stopped by user")
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
        finally:
            self._running = False
            self._live_display = None
    
    def stop_dashboard(self) -> None:
        """Stop the dashboard."""
        self._running = False
    
    async def _update_metrics(self) -> None:
        """Update dashboard metrics from the orchestrator."""
        current_time = time.time()
        
        # Rate limit status checks
        if current_time - self._last_status_check < 1.0:
            return
        
        self._last_status_check = current_time
        
        if not self.orchestrator or not hasattr(self.orchestrator, 'continuous_optimizer'):
            return
        
        try:
            # Get optimization status
            if self.orchestrator.continuous_optimizer:
                status = await self.orchestrator.get_self_improvement_status()
                
                # Update metrics
                self._metrics.optimizer_active = status.get('optimizer_active', False)
                self._metrics.current_mode = status.get('current_mode', 'disabled')
                
                # Optimization stats
                opt_stats = status.get('optimization_stats', {})
                self._metrics.total_improvements = opt_stats.get('improvements_implemented', 0)
                self._metrics.rollbacks_performed = opt_stats.get('rollbacks_performed', 0)
                
                # Performance trends
                perf_trends = status.get('performance_trends', {})
                system_health = perf_trends.get('system_health', {})
                self._metrics.avg_completion_time = system_health.get('avg_completion_time', 0.0)
                self._metrics.system_stability = system_health.get('avg_stability', 1.0)
                
                # Implementation status
                impl_status = status.get('implementation_status', {})
                self._metrics.active_optimizations = impl_status.get('active_implementations', 0)
                self._metrics.successful_implementations = impl_status.get('total_implementations', 0) or 0
                
                # Recent implementations
                recent_impls = impl_status.get('recent_implementations', [])
                self._metrics.recent_improvements = [
                    f"{impl['status']}: {impl['opportunity_id'][:20]}..." 
                    for impl in recent_impls[-5:]
                ]
                
                # User feedback
                feedback_summary = status.get('feedback_summary', {})
                self._metrics.user_satisfaction = feedback_summary.get('average_rating', 0.0)
                self._metrics.feedback_count = feedback_summary.get('total_feedback', 0)
                
            self._last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to update dashboard metrics: {e}")
    
    def _create_dashboard_layout(self) -> Layout:
        """Create the main dashboard layout."""
        layout = Layout()
        
        # Create sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Split body into columns
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Split left column
        layout["left"].split_column(
            Layout(name="status", size=8),
            Layout(name="metrics", size=12),
            Layout(name="activity")
        )
        
        # Populate sections
        layout["header"] = self._create_header()
        layout["status"] = self._create_status_panel()
        layout["metrics"] = self._create_metrics_panel()
        layout["activity"] = self._create_activity_panel()
        layout["right"] = self._create_controls_panel()
        layout["footer"] = self._create_footer()
        
        return layout
    
    def _create_header(self) -> Panel:
        """Create dashboard header."""
        title = Text("AgentsMCP Self-Improvement Dashboard", style="bold blue")
        subtitle = Text(f"Last Updated: {self._last_update.strftime('%H:%M:%S')}", style="dim")
        
        return Panel(
            Align.center(Group(title, subtitle)),
            style="blue",
            padding=(0, 1)
        )
    
    def _create_status_panel(self) -> Panel:
        """Create system status panel."""
        status_table = Table(show_header=False, box=None, padding=0)
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="bright_white")
        
        # Status indicator
        status_color = "green" if self._metrics.optimizer_active else "red"
        status_text = "ACTIVE" if self._metrics.optimizer_active else "INACTIVE"
        
        status_table.add_row("System Status", f"[{status_color}]{status_text}[/{status_color}]")
        status_table.add_row("Operation Mode", self._metrics.current_mode.upper())
        status_table.add_row("Active Optimizations", str(self._metrics.active_optimizations))
        
        # Performance indicators
        if self._metrics.avg_completion_time > 0:
            completion_color = "red" if self._metrics.avg_completion_time > 5.0 else "green"
            status_table.add_row(
                "Avg Task Time", 
                f"[{completion_color}]{self._metrics.avg_completion_time:.2f}s[/{completion_color}]"
            )
        
        stability_color = "green" if self._metrics.system_stability > 0.8 else "yellow" if self._metrics.system_stability > 0.6 else "red"
        stability_pct = f"{self._metrics.system_stability * 100:.1f}%"
        status_table.add_row("System Stability", f"[{stability_color}]{stability_pct}[/{stability_color}]")
        
        return Panel(
            status_table,
            title="System Status",
            border_style="green" if self._metrics.optimizer_active else "red"
        )
    
    def _create_metrics_panel(self) -> Panel:
        """Create metrics panel."""
        metrics_table = Table(show_header=True, box=None)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right")
        metrics_table.add_column("Trend", justify="center")
        
        # Improvement metrics
        metrics_table.add_row(
            "Total Improvements", 
            str(self._metrics.total_improvements),
            "📈" if self._metrics.total_improvements > 0 else "➖"
        )
        
        # Success rate
        if self._metrics.successful_implementations > 0:
            success_rate = (self._metrics.total_improvements / self._metrics.successful_implementations) * 100
            success_color = "green" if success_rate > 80 else "yellow" if success_rate > 60 else "red"
            metrics_table.add_row(
                "Implementation Success Rate",
                f"[{success_color}]{success_rate:.1f}%[/{success_color}]",
                "🎯" if success_rate > 80 else "⚠️"
            )
        
        metrics_table.add_row(
            "Rollbacks Performed",
            str(self._metrics.rollbacks_performed),
            "🔄" if self._metrics.rollbacks_performed > 0 else "✅"
        )
        
        # User feedback metrics
        if self._metrics.feedback_count > 0:
            satisfaction_color = "green" if self._metrics.user_satisfaction > 4.0 else "yellow" if self._metrics.user_satisfaction > 3.0 else "red"
            metrics_table.add_row(
                "User Satisfaction",
                f"[{satisfaction_color}]{self._metrics.user_satisfaction:.1f}/5.0[/{satisfaction_color}]",
                "😊" if self._metrics.user_satisfaction > 4.0 else "😐" if self._metrics.user_satisfaction > 3.0 else "😞"
            )
            
            metrics_table.add_row(
                "Feedback Count",
                str(self._metrics.feedback_count),
                "💬"
            )
        
        return Panel(
            metrics_table,
            title="Performance Metrics",
            border_style="cyan"
        )
    
    def _create_activity_panel(self) -> Panel:
        """Create recent activity panel."""
        activity_text = Text()
        
        if self._metrics.recent_improvements:
            activity_text.append("Recent Improvements:\n\n", style="bold")
            
            for i, improvement in enumerate(self._metrics.recent_improvements[-5:], 1):
                status = improvement.split(':')[0].lower()
                color = "green" if "success" in status else "yellow" if "progress" in status else "red"
                activity_text.append(f"{i}. ", style="dim")
                activity_text.append(improvement, style=color)
                activity_text.append("\n")
        else:
            activity_text.append("No recent improvement activity", style="dim")
        
        return Panel(
            activity_text,
            title="Recent Activity",
            border_style="yellow"
        )
    
    def _create_controls_panel(self) -> Panel:
        """Create controls and actions panel."""
        controls_text = Text()
        
        controls_text.append("Available Actions:\n\n", style="bold")
        
        # Manual controls
        if self._metrics.optimizer_active:
            controls_text.append("• Manual Optimization Cycle\n", style="green")
            controls_text.append("• View Detailed Report\n", style="green")
            controls_text.append("• Rollback Last Change\n", style="yellow")
            controls_text.append("• Export Metrics\n", style="cyan")
            controls_text.append("• Stop Optimization\n", style="red")
        else:
            controls_text.append("• Start Optimization System\n", style="green")
            controls_text.append("• View System Status\n", style="cyan")
        
        controls_text.append("\n")
        controls_text.append("Hot Keys:\n", style="bold")
        controls_text.append("• [r] Refresh Dashboard\n", style="dim")
        controls_text.append("• [q] Quit Dashboard\n", style="dim")
        controls_text.append("• [m] Manual Optimization\n", style="dim")
        controls_text.append("• [s] Toggle Auto-refresh\n", style="dim")
        
        return Panel(
            controls_text,
            title="Controls",
            border_style="magenta"
        )
    
    def _create_footer(self) -> Panel:
        """Create dashboard footer."""
        footer_text = Text()
        
        # System info
        footer_text.append(f"Mode: {self._metrics.current_mode} | ", style="dim")
        footer_text.append(f"Auto-refresh: {'ON' if self.auto_refresh else 'OFF'} | ", style="dim")
        footer_text.append(f"Update interval: {self._update_interval}s", style="dim")
        
        return Panel(
            Align.center(footer_text),
            style="dim"
        )
    
    def show_improvement_summary(self) -> None:
        """Show detailed improvement summary."""
        if not self.orchestrator or not self.orchestrator.continuous_optimizer:
            self.console.print("[red]Self-improvement system not available[/red]")
            return
        
        # This would show a detailed modal/overlay with comprehensive improvement data
        self.console.print("\n[bold blue]Improvement System Summary[/bold blue]")
        self.console.print("=" * 50)
        
        # Display detailed metrics
        summary_table = Table(title="Current Status")
        summary_table.add_column("Component", style="cyan")
        summary_table.add_column("Status", justify="center")
        summary_table.add_column("Details")
        
        summary_table.add_row(
            "Performance Analyzer",
            "🟢 Active" if self._metrics.optimizer_active else "🔴 Inactive",
            f"Tracking {self._metrics.total_improvements} improvements"
        )
        
        summary_table.add_row(
            "Improvement Detector", 
            "🟢 Monitoring" if self._metrics.optimizer_active else "🔴 Stopped",
            f"Success rate: {(self._metrics.successful_implementations / max(self._metrics.total_improvements, 1)) * 100:.1f}%"
        )
        
        summary_table.add_row(
            "User Feedback System",
            "🟢 Collecting" if self._metrics.feedback_count > 0 else "🟡 Limited data",
            f"{self._metrics.feedback_count} feedback entries, avg: {self._metrics.user_satisfaction:.1f}/5"
        )
        
        self.console.print(summary_table)
        self.console.print("\n[dim]Press any key to continue...[/dim]")
    
    async def trigger_manual_optimization(self) -> bool:
        """Trigger manual optimization cycle."""
        if not self.orchestrator or not self.orchestrator.continuous_optimizer:
            self.console.print("[red]Self-improvement system not available[/red]")
            return False
        
        try:
            self.console.print("[yellow]Triggering manual optimization cycle...[/yellow]")
            
            result = await self.orchestrator.trigger_manual_optimization()
            
            if result.get('success'):
                self.console.print("[green]✓ Manual optimization completed successfully[/green]")
                return True
            else:
                error = result.get('error', 'Unknown error')
                self.console.print(f"[red]✗ Manual optimization failed: {error}[/red]")
                return False
                
        except Exception as e:
            self.console.print(f"[red]✗ Error triggering optimization: {e}[/red]")
            return False
    
    def export_dashboard_data(self, filepath: str = None) -> str:
        """Export current dashboard data."""
        if not filepath:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'/tmp/agentsmcp_dashboard_export_{timestamp}.json'
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'dashboard_metrics': {
                'optimizer_active': self._metrics.optimizer_active,
                'current_mode': self._metrics.current_mode,
                'total_improvements': self._metrics.total_improvements,
                'successful_implementations': self._metrics.successful_implementations,
                'rollbacks_performed': self._metrics.rollbacks_performed,
                'avg_completion_time': self._metrics.avg_completion_time,
                'system_stability': self._metrics.system_stability,
                'user_satisfaction': self._metrics.user_satisfaction,
                'feedback_count': self._metrics.feedback_count,
                'recent_improvements': self._metrics.recent_improvements
            },
            'dashboard_config': {
                'show_detailed_metrics': self.show_detailed_metrics,
                'show_recent_activity': self.show_recent_activity,
                'show_performance_trends': self.show_performance_trends,
                'auto_refresh': self.auto_refresh,
                'update_interval': self._update_interval
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.console.print(f"[green]✓ Dashboard data exported to: {filepath}[/green]")
            return filepath
            
        except Exception as e:
            self.console.print(f"[red]✗ Export failed: {e}[/red]")
            return ""
    
    async def handle_key_input(self, key: str) -> bool:
        """Handle keyboard input for dashboard controls."""
        if key.lower() == 'q':
            self.stop_dashboard()
            return False
            
        elif key.lower() == 'r':
            await self._update_metrics()
            self.console.print("[green]Dashboard refreshed[/green]")
            
        elif key.lower() == 'm':
            await self.trigger_manual_optimization()
            
        elif key.lower() == 's':
            self.auto_refresh = not self.auto_refresh
            self.console.print(f"[yellow]Auto-refresh {'enabled' if self.auto_refresh else 'disabled'}[/yellow]")
            
        elif key.lower() == 'e':
            self.export_dashboard_data()
            
        elif key.lower() == 'h':
            self.show_improvement_summary()
        
        return True


# Standalone dashboard runner
async def run_improvement_dashboard(orchestrator=None):
    """Run the improvement dashboard as standalone application."""
    dashboard = ImprovementDashboard(orchestrator)
    
    try:
        await dashboard.start_dashboard()
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop_dashboard()


if __name__ == "__main__":
    # For testing without orchestrator
    asyncio.run(run_improvement_dashboard())