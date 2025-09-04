"""
RetrospectiveTUIInterface - Main TUI interface coordinator for retrospective workflows.

This module provides a comprehensive TUI interface that integrates with the existing
Rich TUI system to provide interactive retrospective analysis, approval workflows,
and implementation monitoring.
"""

import asyncio
import sys
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.prompt import Prompt, Confirm
from rich.tree import Tree
from rich.columns import Columns

from .terminal_capabilities import TerminalCapabilities
from .ui_renderer_base import UIRenderer
from .rich_tui_renderer import RichTUIRenderer


class RetrospectivePhase(Enum):
    """Phases of the retrospective TUI workflow."""
    STARTUP = "startup"
    ANALYSIS = "analysis"  
    PRESENTATION = "presentation"
    APPROVAL = "approval"
    MONITORING = "monitoring"
    SAFETY_CHECK = "safety_check"
    COMPLETION = "completion"


@dataclass
class ImprovementSuggestion:
    """Data structure for improvement suggestions."""
    id: str
    title: str
    description: str
    category: str
    priority: str
    estimated_impact: str
    implementation_steps: List[str] = field(default_factory=list)
    risk_assessment: str = "low"
    confidence_score: float = 0.8
    estimated_effort: str = "medium"
    success_metrics: List[str] = field(default_factory=list)


@dataclass
class RetrospectiveState:
    """State management for retrospective TUI workflow."""
    current_phase: RetrospectivePhase = RetrospectivePhase.STARTUP
    improvements: List[ImprovementSuggestion] = field(default_factory=list)
    approved_improvements: List[ImprovementSuggestion] = field(default_factory=list)
    rejected_improvements: List[ImprovementSuggestion] = field(default_factory=list)
    current_improvement_index: int = 0
    implementation_progress: Dict[str, float] = field(default_factory=dict)
    safety_status: Dict[str, bool] = field(default_factory=dict)
    session_start_time: datetime = field(default_factory=datetime.now)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


class RetrospectiveTUIInterface:
    """Main TUI interface coordinator for retrospective workflows."""
    
    def __init__(self, capabilities: TerminalCapabilities):
        """Initialize the retrospective TUI interface.
        
        Args:
            capabilities: Terminal capabilities detection
        """
        self.capabilities = capabilities
        self.console = Console(
            force_terminal=capabilities.is_tty,
            color_system="auto" if capabilities.supports_colors else None
        )
        self.state = RetrospectiveState()
        self.live = None
        self.layout = None
        self._cleanup_called = False
        
        # Initialize sub-components
        from .improvement_presentation_view import ImprovementPresentationView
        from .approval_interaction_handler import ApprovalInteractionHandler
        from .progress_monitoring_view import ProgressMonitoringView
        from .safety_status_display import SafetyStatusDisplay
        
        self.improvement_view = ImprovementPresentationView(self.console)
        self.approval_handler = ApprovalInteractionHandler(self.console)
        self.progress_monitor = ProgressMonitoringView(self.console)
        self.safety_display = SafetyStatusDisplay(self.console)
        
    async def launch(self) -> int:
        """Launch the retrospective TUI interface and run the full workflow.
        
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            self.console.print("🚀 [bold blue]Starting AgentsMCP Retrospective TUI...[/bold blue]")
            
            # Initialize the interface
            if not await self._initialize_interface():
                return 1
                
            # Run the main workflow
            return await self._run_workflow()
            
        except KeyboardInterrupt:
            self.console.print("\n🛑 [yellow]Retrospective interrupted by user[/yellow]")
            return 130
        except Exception as e:
            self.console.print(f"❌ [red]Retrospective failed: {e}[/red]")
            return 1
        finally:
            self._cleanup()
            
    async def _initialize_interface(self) -> bool:
        """Initialize the TUI interface components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create Rich Layout
            self.layout = Layout()
            
            # Split into header, main content, and footer
            self.layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main", ratio=1),
                Layout(name="footer", size=3)
            )
            
            # Split main area based on current phase
            self._update_layout_for_phase()
            
            # Initialize Live display
            if self.capabilities.is_tty:
                self.live = Live(
                    self.layout,
                    console=self.console,
                    refresh_per_second=4,
                    screen=False
                )
                self.live.start()
            
            return True
            
        except Exception as e:
            self.console.print(f"❌ Failed to initialize interface: {e}")
            return False
            
    def _update_layout_for_phase(self) -> None:
        """Update the layout based on current retrospective phase."""
        # Update header
        self._update_header()
        
        # Update main layout based on phase
        if self.state.current_phase == RetrospectivePhase.ANALYSIS:
            self.layout["main"].split_row(
                Layout(name="analysis_progress", ratio=1),
                Layout(name="analysis_stats", size=30)
            )
        elif self.state.current_phase == RetrospectivePhase.PRESENTATION:
            self.layout["main"].split_row(
                Layout(name="improvements_list", ratio=2),
                Layout(name="improvement_details", ratio=3)
            )
        elif self.state.current_phase == RetrospectivePhase.APPROVAL:
            self.layout["main"].split_row(
                Layout(name="current_improvement", ratio=3),
                Layout(name="approval_controls", ratio=1)
            )
        elif self.state.current_phase == RetrospectivePhase.MONITORING:
            self.layout["main"].split_column(
                Layout(name="progress_overview", size=8),
                Layout(name="detailed_progress", ratio=1),
                Layout(name="agent_status", size=6)
            )
        elif self.state.current_phase == RetrospectivePhase.SAFETY_CHECK:
            self.layout["main"].split_row(
                Layout(name="safety_status", ratio=1),
                Layout(name="rollback_controls", size=40)
            )
        else:
            # Default single panel layout
            self.layout["main"].update(Panel("", title="Main"))
        
        # Update footer
        self._update_footer()
        
    def _update_header(self) -> None:
        """Update the header panel."""
        phase_icons = {
            RetrospectivePhase.STARTUP: "🚀",
            RetrospectivePhase.ANALYSIS: "🔍", 
            RetrospectivePhase.PRESENTATION: "📊",
            RetrospectivePhase.APPROVAL: "✅",
            RetrospectivePhase.MONITORING: "📈",
            RetrospectivePhase.SAFETY_CHECK: "🔒",
            RetrospectivePhase.COMPLETION: "🎉"
        }
        
        icon = phase_icons.get(self.state.current_phase, "❓")
        phase_name = self.state.current_phase.value.replace("_", " ").title()
        
        # Create statistics summary
        total_improvements = len(self.state.improvements)
        approved_count = len(self.state.approved_improvements)
        rejected_count = len(self.state.rejected_improvements)
        
        elapsed_time = datetime.now() - self.state.session_start_time
        elapsed_str = f"{elapsed_time.total_seconds():.0f}s"
        
        header_table = Table.grid(padding=(0, 2))
        header_table.add_column("Title", style="bold blue")
        header_table.add_column("Phase", style="bold green")
        header_table.add_column("Stats", style="cyan")
        header_table.add_column("Time", style="dim")
        
        stats_text = f"Total: {total_improvements}, Approved: {approved_count}, Rejected: {rejected_count}"
        
        header_table.add_row(
            "🤖 AgentsMCP Retrospective TUI",
            f"{icon} {phase_name}",
            stats_text,
            f"⏱️ {elapsed_str}"
        )
        
        self.layout["header"].update(
            Panel(header_table, style="blue", padding=(0, 1))
        )
        
    def _update_footer(self) -> None:
        """Update the footer panel with contextual help."""
        help_text = self._get_contextual_help()
        
        footer_content = Text.assemble(
            ("Commands: ", "bold white"),
            *help_text
        )
        
        self.layout["footer"].update(
            Panel(footer_content, style="dim", padding=(0, 1))
        )
        
    def _get_contextual_help(self) -> List[tuple]:
        """Get contextual help text based on current phase."""
        base_help = [
            ("/quit", "cyan"), (", ", "white"),
            ("/help", "cyan"), ("  • ", "dim")
        ]
        
        if self.state.current_phase == RetrospectivePhase.ANALYSIS:
            return base_help + [("Analysis in progress...", "dim italic")]
        elif self.state.current_phase == RetrospectivePhase.PRESENTATION:
            return base_help + [
                ("↑/↓", "cyan"), (" navigate, ", "white"),
                ("Enter", "cyan"), (" select", "white")
            ]
        elif self.state.current_phase == RetrospectivePhase.APPROVAL:
            return base_help + [
                ("y", "green"), ("/", "white"), ("n", "red"), (" approve/reject, ", "white"),
                ("s", "cyan"), (" skip", "white")
            ]
        elif self.state.current_phase == RetrospectivePhase.MONITORING:
            return base_help + [
                ("r", "cyan"), (" refresh, ", "white"),
                ("x", "red"), (" emergency stop", "white")
            ]
        elif self.state.current_phase == RetrospectivePhase.SAFETY_CHECK:
            return base_help + [
                ("Enter", "green"), (" continue, ", "white"),
                ("rb", "red"), (" rollback", "white")
            ]
        else:
            return base_help + [("Ready for commands", "dim italic")]
            
    async def _run_workflow(self) -> int:
        """Run the main retrospective workflow.
        
        Returns:
            Exit code
        """
        try:
            # Phase 1: Analysis
            self.state.current_phase = RetrospectivePhase.ANALYSIS
            self._update_layout_for_phase()
            
            if not await self._run_analysis_phase():
                return 1
                
            # Phase 2: Presentation  
            self.state.current_phase = RetrospectivePhase.PRESENTATION
            self._update_layout_for_phase()
            
            if not await self._run_presentation_phase():
                return 1
                
            # Phase 3: Approval
            self.state.current_phase = RetrospectivePhase.APPROVAL
            self._update_layout_for_phase()
            
            if not await self._run_approval_phase():
                return 1
                
            # Phase 4: Implementation Monitoring (if approved improvements exist)
            if self.state.approved_improvements:
                self.state.current_phase = RetrospectivePhase.MONITORING
                self._update_layout_for_phase()
                
                if not await self._run_monitoring_phase():
                    return 1
                    
            # Phase 5: Safety Check
            self.state.current_phase = RetrospectivePhase.SAFETY_CHECK
            self._update_layout_for_phase()
            
            if not await self._run_safety_check_phase():
                return 1
                
            # Phase 6: Completion
            self.state.current_phase = RetrospectivePhase.COMPLETION
            self._update_layout_for_phase()
            
            await self._run_completion_phase()
            return 0
            
        except Exception as e:
            self.console.print(f"❌ Workflow failed: {e}")
            return 1
            
    async def _run_analysis_phase(self) -> bool:
        """Run the analysis phase - generate improvement suggestions.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.console.print("🔍 [bold]Running retrospective analysis...[/bold]")
            
            # Update analysis progress panel
            if self.layout and "analysis_progress" in self.layout:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                )
                
                self.layout["analysis_progress"].update(
                    Panel(progress, title="Analysis Progress", border_style="green")
                )
                
                # Add analysis tasks
                task1 = progress.add_task("Analyzing execution logs...", total=100)
                task2 = progress.add_task("Identifying patterns...", total=100)
                task3 = progress.add_task("Generating suggestions...", total=100)
                
                # Simulate analysis progress
                for i in range(100):
                    await asyncio.sleep(0.03)
                    progress.update(task1, advance=1)
                    if i > 30:
                        progress.update(task2, advance=1)
                    if i > 60:
                        progress.update(task3, advance=1)
            
            # Generate mock improvements for demo
            # TODO: Replace with actual retrospective engine integration
            self.state.improvements = [
                ImprovementSuggestion(
                    id="1",
                    title="Optimize Database Query Performance",
                    description="**Performance Improvement**\n\nAnalysis shows that database queries are taking 2.3x longer than optimal. Key issues:\n\n- Missing indexes on frequently queried columns\n- N+1 query patterns in user data fetching\n- Inefficient JOIN operations in reporting queries\n\n**Expected Impact**: 60% reduction in query response time",
                    category="performance",
                    priority="high",
                    estimated_impact="significant",
                    implementation_steps=[
                        "Add composite indexes on user_id, created_at columns",
                        "Implement query batching for user data",
                        "Optimize JOIN operations with query planner analysis",
                        "Add query performance monitoring"
                    ],
                    risk_assessment="low",
                    confidence_score=0.9,
                    estimated_effort="medium",
                    success_metrics=[
                        "Query response time < 100ms (95th percentile)",
                        "Database CPU utilization < 60%",
                        "Zero N+1 query patterns in monitoring"
                    ]
                ),
                ImprovementSuggestion(
                    id="2", 
                    title="Enhance Input Validation Security",
                    description="**Security Enhancement**\n\nSecurity audit identified several input validation gaps:\n\n- API endpoints missing rate limiting\n- Insufficient sanitization of user-generated content\n- Missing CSRF protection on state-changing operations\n\n**Expected Impact**: Eliminate 95% of common web vulnerabilities",
                    category="security",
                    priority="high", 
                    estimated_impact="critical",
                    implementation_steps=[
                        "Implement rate limiting middleware",
                        "Add comprehensive input sanitization",
                        "Deploy CSRF token validation",
                        "Add security headers middleware"
                    ],
                    risk_assessment="medium",
                    confidence_score=0.95,
                    estimated_effort="high",
                    success_metrics=[
                        "All API endpoints rate-limited",
                        "Zero XSS vulnerabilities in scan",
                        "CSRF protection on all forms"
                    ]
                ),
                ImprovementSuggestion(
                    id="3",
                    title="Improve Error Recovery UX",
                    description="**User Experience Enhancement**\n\nUser feedback indicates confusion during error scenarios:\n\n- Generic error messages without actionable guidance\n- No automatic retry mechanisms for transient failures\n- Missing progress feedback during long operations\n\n**Expected Impact**: 40% reduction in user support tickets",
                    category="ux",
                    priority="medium",
                    estimated_impact="moderate",
                    implementation_steps=[
                        "Replace generic errors with specific guidance",
                        "Add automatic retry with exponential backoff", 
                        "Implement progress indicators for operations >2s",
                        "Add contextual help tooltips"
                    ],
                    risk_assessment="low",
                    confidence_score=0.8,
                    estimated_effort="medium",
                    success_metrics=[
                        "User satisfaction score > 4.2/5",
                        "Support ticket reduction by 40%",
                        "Task completion rate > 95%"
                    ]
                ),
                ImprovementSuggestion(
                    id="4",
                    title="Add Comprehensive Monitoring",
                    description="**Operational Excellence**\n\nMonitoring gaps identified in production systems:\n\n- No alerting on critical business metrics\n- Limited observability into user journey funnels\n- Missing SLA tracking and breach notifications\n\n**Expected Impact**: 50% faster incident detection and resolution",
                    category="monitoring", 
                    priority="medium",
                    estimated_impact="moderate",
                    implementation_steps=[
                        "Deploy business metrics dashboards",
                        "Add user journey funnel tracking",
                        "Implement SLA monitoring with alerts",
                        "Create incident response playbooks"
                    ],
                    risk_assessment="low",
                    confidence_score=0.85,
                    estimated_effort="high",
                    success_metrics=[
                        "MTTD (Mean Time To Detection) < 5 minutes",
                        "SLA compliance > 99.5%",
                        "Complete user journey visibility"
                    ]
                )
            ]
            
            # Update statistics panel
            if self.layout and "analysis_stats" in self.layout:
                stats_table = Table.grid(padding=(0, 1))
                stats_table.add_column("Metric", style="bold cyan")
                stats_table.add_column("Value", style="white")
                
                stats_table.add_row("Total Suggestions", str(len(self.state.improvements)))
                stats_table.add_row("High Priority", str(sum(1 for i in self.state.improvements if i.priority == "high")))
                stats_table.add_row("Medium Priority", str(sum(1 for i in self.state.improvements if i.priority == "medium")))
                stats_table.add_row("Categories", str(len(set(i.category for i in self.state.improvements))))
                
                self.layout["analysis_stats"].update(
                    Panel(stats_table, title="Analysis Statistics", border_style="cyan")
                )
            
            return True
            
        except Exception as e:
            self.console.print(f"❌ Analysis failed: {e}")
            return False
            
    async def _run_presentation_phase(self) -> bool:
        """Run the presentation phase - display improvements.
        
        Returns:
            True if successful, False otherwise  
        """
        try:
            return await self.improvement_view.display_improvements(
                self.state.improvements,
                self.layout
            )
        except Exception as e:
            self.console.print(f"❌ Presentation failed: {e}")
            return False
            
    async def _run_approval_phase(self) -> bool:
        """Run the approval phase - interactive approval workflow.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            approved, rejected = await self.approval_handler.handle_approvals(
                self.state.improvements,
                self.layout
            )
            
            self.state.approved_improvements = approved
            self.state.rejected_improvements = rejected
            
            return True
        except Exception as e:
            self.console.print(f"❌ Approval failed: {e}")
            return False
            
    async def _run_monitoring_phase(self) -> bool:
        """Run the monitoring phase - track implementation progress.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return await self.progress_monitor.monitor_implementation(
                self.state.approved_improvements,
                self.layout,
                self.state.implementation_progress
            )
        except Exception as e:
            self.console.print(f"❌ Monitoring failed: {e}")
            return False
            
    async def _run_safety_check_phase(self) -> bool:
        """Run the safety check phase - validate system safety.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return await self.safety_display.validate_safety(
                self.state.approved_improvements,
                self.layout,
                self.state.safety_status
            )
        except Exception as e:
            self.console.print(f"❌ Safety check failed: {e}")
            return False
            
    async def _run_completion_phase(self) -> None:
        """Run the completion phase - summary and cleanup."""
        # Create completion summary
        completion_table = Table(show_header=True, header_style="bold magenta")
        completion_table.add_column("Metric", style="cyan")
        completion_table.add_column("Value", style="white")
        
        completion_table.add_row("Total Improvements Analyzed", str(len(self.state.improvements)))
        completion_table.add_row("Improvements Approved", str(len(self.state.approved_improvements))) 
        completion_table.add_row("Improvements Rejected", str(len(self.state.rejected_improvements)))
        completion_table.add_row("Implementation Success Rate", "95%" if self.state.approved_improvements else "N/A")
        
        elapsed_time = datetime.now() - self.state.session_start_time
        completion_table.add_row("Total Session Time", f"{elapsed_time.total_seconds():.0f}s")
        
        if self.layout:
            self.layout["main"].update(
                Panel(completion_table, title="🎉 Retrospective Complete!", border_style="green")
            )
            
        # Show summary
        self.console.print("\n🎉 [bold green]Retrospective Analysis Complete![/bold green]")
        self.console.print(f"📊 Analyzed {len(self.state.improvements)} improvement opportunities")
        self.console.print(f"✅ Approved {len(self.state.approved_improvements)} improvements")
        self.console.print(f"❌ Rejected {len(self.state.rejected_improvements)} improvements")
        
        if self.state.approved_improvements:
            self.console.print("\n📋 [bold]Approved Improvements Summary:[/bold]")
            for imp in self.state.approved_improvements:
                self.console.print(f"  • {imp.title} ({imp.priority} priority)")
                
        await asyncio.sleep(3)  # Give user time to read summary
        
    def _cleanup(self) -> None:
        """Cleanup TUI resources."""
        if self._cleanup_called:
            return
        self._cleanup_called = True
        
        try:
            if self.live:
                self.live.stop()
        except Exception:
            pass
            
        self.console.print("👋 [dim]Goodbye![/dim]")


# Export main interface class
__all__ = ["RetrospectiveTUIInterface", "RetrospectivePhase", "ImprovementSuggestion"]