"""
ApprovalInteractionHandler - Interactive approval/rejection interface for improvement suggestions.

This module provides a comprehensive interface for users to review, approve, or reject
improvement suggestions with rich visual feedback and batch operation support.
"""

import asyncio
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, BarColumn, TextColumn, TaskID
from rich.prompt import Prompt, Confirm
from rich.align import Align
from rich.columns import Columns
from rich.tree import Tree


class ApprovalDecision(Enum):
    """Approval decision options."""
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    PENDING = "pending"


@dataclass
class ApprovalSession:
    """Tracks state during an approval session."""
    decisions: Dict[str, ApprovalDecision] = field(default_factory=dict)
    current_index: int = 0
    batch_mode: bool = False
    auto_advance: bool = True
    show_detailed_view: bool = True
    approval_reasons: Dict[str, str] = field(default_factory=dict)
    session_notes: str = ""
    risk_threshold: str = "medium"  # low, medium, high
    

class ApprovalInteractionHandler:
    """Interactive approval/rejection interface with comprehensive decision tracking."""
    
    def __init__(self, console: Console):
        """Initialize the approval interaction handler.
        
        Args:
            console: Rich console for rendering
        """
        self.console = console
        self.session = ApprovalSession()
        
    async def handle_approvals(self, improvements: List, layout: Layout) -> Tuple[List, List]:
        """Handle interactive approval process for improvements.
        
        Args:
            improvements: List of improvement suggestions to review
            layout: Rich layout for UI updates
            
        Returns:
            Tuple of (approved_improvements, rejected_improvements)
        """
        if not improvements:
            return [], []
            
        try:
            # Initialize session
            self.session = ApprovalSession()
            
            # Show approval workflow introduction
            await self._show_approval_introduction(improvements, layout)
            
            # Run approval workflow
            await self._run_approval_workflow(improvements, layout)
            
            # Process final decisions
            approved, rejected = self._process_final_decisions(improvements)
            
            # Show approval summary
            await self._show_approval_summary(approved, rejected, layout)
            
            return approved, rejected
            
        except Exception as e:
            self.console.print(f"❌ Approval process failed: {e}")
            return [], []
            
    async def _show_approval_introduction(self, improvements: List, layout: Layout) -> None:
        """Show introduction to the approval process.
        
        Args:
            improvements: List of improvements to review
            layout: Layout for UI updates
        """
        intro_content = []
        
        # Title
        title_text = Text()
        title_text.append("✅ ", style="green")
        title_text.append("Improvement Approval Workflow", style="bold green")
        intro_content.append(title_text)
        intro_content.append("")
        
        # Summary stats
        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_column("Metric", style="bold cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Total Improvements:", str(len(improvements)))
        stats_table.add_row("High Priority:", str(sum(1 for i in improvements if i.priority == "high")))
        stats_table.add_row("Critical Impact:", str(sum(1 for i in improvements if getattr(i, 'estimated_impact', '') == "critical")))
        
        categories = set(i.category for i in improvements)
        stats_table.add_row("Categories:", ", ".join(categories))
        
        intro_content.append(stats_table)
        intro_content.append("")
        
        # Instructions
        instructions = Text()
        instructions.append("📋 Instructions:", style="bold yellow")
        instructions.append("\n• Review each improvement carefully")
        instructions.append("\n• Consider impact, effort, and risk factors")
        instructions.append("\n• Use 'y' to approve, 'n' to reject, 's' to skip")
        instructions.append("\n• Type 'batch' for batch operations")
        instructions.append("\n• Type 'help' for more options")
        intro_content.append(instructions)
        
        from rich.console import Group
        intro_group = Group(*intro_content)
        
        if layout and "current_improvement" in layout:
            layout["current_improvement"].update(
                Panel(intro_group, title="🚀 Approval Process", border_style="green")
            )
        else:
            self.console.print(Panel(intro_group, title="🚀 Approval Process", border_style="green"))
            
        # Brief pause for user to read
        await asyncio.sleep(2)
        
    async def _run_approval_workflow(self, improvements: List, layout: Layout) -> None:
        """Run the main approval workflow loop.
        
        Args:
            improvements: List of improvements to process
            layout: Layout for UI updates
        """
        while self.session.current_index < len(improvements):
            current_improvement = improvements[self.session.current_index]
            
            # Update UI with current improvement
            self._update_current_improvement_panel(current_improvement, layout)
            self._update_approval_controls_panel(improvements, layout)
            
            # Get user decision
            decision = await self._get_approval_decision(current_improvement)
            
            # Process decision
            if decision == "quit":
                break
            elif decision == "batch":
                await self._handle_batch_operations(improvements, layout)
            elif decision == "help":
                await self._show_approval_help()
            elif decision == "back":
                self.session.current_index = max(0, self.session.current_index - 1)
            elif decision == "summary":
                await self._show_current_summary(improvements)
            elif decision in ["approve", "reject", "skip"]:
                # Record decision
                decision_enum = {
                    "approve": ApprovalDecision.APPROVED,
                    "reject": ApprovalDecision.REJECTED,
                    "skip": ApprovalDecision.SKIPPED
                }[decision]
                
                self.session.decisions[current_improvement.id] = decision_enum
                
                # Request reason for major decisions
                if decision in ["approve", "reject"] and current_improvement.priority == "high":
                    reason = await self._get_decision_reason(decision, current_improvement)
                    if reason:
                        self.session.approval_reasons[current_improvement.id] = reason
                
                # Advance to next improvement
                self.session.current_index += 1
                
    def _update_current_improvement_panel(self, improvement, layout: Layout) -> None:
        """Update the current improvement panel with detailed view.
        
        Args:
            improvement: Current improvement being reviewed
            layout: Layout to update
        """
        content_sections = []
        
        # Header with improvement info
        header_table = Table.grid(padding=(0, 2))
        header_table.add_column("Field", style="bold cyan", width=18)
        header_table.add_column("Value", style="white")
        
        header_table.add_row("ID:", improvement.id)
        header_table.add_row("Category:", improvement.category.title())
        
        # Priority with color coding
        priority_color = {"high": "red", "medium": "yellow", "low": "green"}.get(improvement.priority, "white")
        header_table.add_row("Priority:", f"[{priority_color}]{improvement.priority.upper()}[/{priority_color}]")
        
        # Impact with color coding
        impact_color = {
            "critical": "bold red", "significant": "bold yellow",
            "moderate": "yellow", "minor": "dim white"
        }.get(getattr(improvement, 'estimated_impact', 'moderate'), "white")
        header_table.add_row("Impact:", f"[{impact_color}]{getattr(improvement, 'estimated_impact', 'moderate').upper()}[/{impact_color}]")
        
        # Risk assessment
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(
            getattr(improvement, 'risk_assessment', 'medium'), "yellow"
        )
        header_table.add_row("Risk:", f"[{risk_color}]{getattr(improvement, 'risk_assessment', 'medium').upper()}[/{risk_color}]")
        
        header_table.add_row("Effort:", getattr(improvement, 'estimated_effort', 'unknown').title())
        header_table.add_row("Confidence:", f"{getattr(improvement, 'confidence_score', 0.8):.1%}")
        
        content_sections.append(Panel(header_table, title="📊 Overview", border_style="cyan"))
        
        # Title and description
        title_text = Text()
        title_text.append("🎯 ", style="blue") 
        title_text.append(improvement.title, style="bold white")
        content_sections.append(Panel(title_text, border_style="blue"))
        
        # Description with markdown support
        if hasattr(improvement, 'description') and improvement.description:
            try:
                description_md = Markdown(improvement.description, code_theme="monokai")
                content_sections.append(
                    Panel(description_md, title="📝 Description", border_style="green")
                )
            except Exception:
                content_sections.append(
                    Panel(
                        Text(improvement.description),
                        title="📝 Description",
                        border_style="green"
                    )
                )
        
        # Implementation preview (first 3 steps)
        if hasattr(improvement, 'implementation_steps') and improvement.implementation_steps:
            steps_preview = []
            for i, step in enumerate(improvement.implementation_steps[:3], 1):
                step_text = Text()
                step_text.append(f"{i}. ", style="cyan")
                step_text.append(step[:80] + ("..." if len(step) > 80 else ""))
                steps_preview.append(step_text)
                
            if len(improvement.implementation_steps) > 3:
                more_text = Text()
                more_text.append(f"... and {len(improvement.implementation_steps) - 3} more steps", style="dim italic")
                steps_preview.append(more_text)
            
            from rich.console import Group
            steps_group = Group(*steps_preview)
            content_sections.append(
                Panel(steps_group, title="📋 Implementation Preview", border_style="yellow")
            )
        
        # Approval status
        current_decision = self.session.decisions.get(improvement.id, ApprovalDecision.PENDING)
        status_color = {
            ApprovalDecision.APPROVED: "green",
            ApprovalDecision.REJECTED: "red", 
            ApprovalDecision.SKIPPED: "yellow",
            ApprovalDecision.PENDING: "dim"
        }[current_decision]
        
        status_text = Text()
        status_text.append("Status: ", style="bold")
        status_text.append(current_decision.value.title(), style=status_color)
        content_sections.append(Panel(status_text, border_style=status_color))
        
        # Combine all sections
        from rich.console import Group
        combined_content = Group(*content_sections)
        
        if layout and "current_improvement" in layout:
            layout["current_improvement"].update(
                Panel(
                    combined_content,
                    title=f"[bold]Reviewing Improvement {self.session.current_index + 1}/{len(improvements) if 'improvements' in locals() else '?'}[/bold]",
                    border_style="magenta"
                )
            )
            
    def _update_approval_controls_panel(self, improvements: List, layout: Layout) -> None:
        """Update the approval controls panel with options and progress.
        
        Args:
            improvements: List of all improvements
            layout: Layout to update
        """
        controls_content = []
        
        # Progress indicator
        progress_text = Text()
        progress_text.append("Progress: ", style="bold white")
        progress_text.append(f"{self.session.current_index + 1}/{len(improvements)}", style="cyan")
        
        completed = len([d for d in self.session.decisions.values() if d != ApprovalDecision.PENDING])
        progress_text.append(f" • Decided: {completed}", style="green" if completed > 0 else "dim")
        
        controls_content.append(progress_text)
        controls_content.append("")
        
        # Decision summary
        if self.session.decisions:
            summary_table = Table.grid(padding=(0, 1))
            summary_table.add_column("Decision", style="bold")
            summary_table.add_column("Count", style="white")
            
            approved_count = len([d for d in self.session.decisions.values() if d == ApprovalDecision.APPROVED])
            rejected_count = len([d for d in self.session.decisions.values() if d == ApprovalDecision.REJECTED])
            skipped_count = len([d for d in self.session.decisions.values() if d == ApprovalDecision.SKIPPED])
            
            summary_table.add_row("[green]Approved:", str(approved_count))
            summary_table.add_row("[red]Rejected:", str(rejected_count))
            summary_table.add_row("[yellow]Skipped:", str(skipped_count))
            
            controls_content.append(summary_table)
            controls_content.append("")
        
        # Control options
        controls_text = Text()
        controls_text.append("Commands:", style="bold yellow")
        controls_text.append("\n[green]y[/green] - Approve")
        controls_text.append("\n[red]n[/red] - Reject") 
        controls_text.append("\n[yellow]s[/yellow] - Skip")
        controls_text.append("\n[cyan]b[/cyan] - Previous")
        controls_text.append("\n[magenta]batch[/magenta] - Batch ops")
        controls_text.append("\n[blue]help[/blue] - More options")
        controls_text.append("\n[dim]q[/dim] - Quit")
        
        controls_content.append(controls_text)
        
        # Risk warning for high-risk items
        if hasattr(improvements[self.session.current_index], 'risk_assessment'):
            risk = getattr(improvements[self.session.current_index], 'risk_assessment', 'medium')
            if risk == "high":
                controls_content.append("")
                warning_text = Text()
                warning_text.append("⚠️  ", style="red")
                warning_text.append("HIGH RISK IMPROVEMENT", style="bold red")
                warning_text.append("\nRequires careful consideration", style="dim red")
                controls_content.append(warning_text)
        
        from rich.console import Group
        controls_group = Group(*controls_content)
        
        if layout and "approval_controls" in layout:
            layout["approval_controls"].update(
                Panel(controls_group, title="🎛️ Approval Controls", border_style="yellow")
            )
            
    async def _get_approval_decision(self, improvement) -> str:
        """Get approval decision from user input.
        
        Args:
            improvement: Current improvement being reviewed
            
        Returns:
            User decision as string
        """
        try:
            # Contextual prompt based on improvement priority
            if improvement.priority == "high":
                prompt_style = "bold red"
                prompt_prefix = "🔴 HIGH PRIORITY"
            elif improvement.priority == "medium":
                prompt_style = "bold yellow" 
                prompt_prefix = "🟡 MEDIUM PRIORITY"
            else:
                prompt_style = "bold green"
                prompt_prefix = "🟢 LOW PRIORITY"
                
            prompt_text = f"[{prompt_style}]{prompt_prefix}[/{prompt_style}] Decision"
            
            decision = Prompt.ask(
                prompt_text,
                choices=["y", "n", "s", "b", "batch", "help", "summary", "q", ""],
                default=""
            ).lower().strip()
            
            # Map user input to actions
            decision_mapping = {
                "y": "approve",
                "yes": "approve",
                "n": "reject", 
                "no": "reject",
                "s": "skip",
                "skip": "skip",
                "b": "back",
                "back": "back",
                "batch": "batch",
                "help": "help",
                "summary": "summary",
                "q": "quit",
                "quit": "quit",
                "": "skip"  # Default to skip if no input
            }
            
            return decision_mapping.get(decision, "skip")
            
        except (EOFError, KeyboardInterrupt):
            return "quit"
        except Exception as e:
            self.console.print(f"❌ Input error: {e}")
            return "skip"
            
    async def _get_decision_reason(self, decision: str, improvement) -> Optional[str]:
        """Get reason for approval/rejection decision.
        
        Args:
            decision: The decision made (approve/reject)
            improvement: The improvement being decided on
            
        Returns:
            Reason string or None
        """
        try:
            action = "approval" if decision == "approve" else "rejection"
            
            reason = Prompt.ask(
                f"Reason for {action} (optional)",
                default=""
            ).strip()
            
            return reason if reason else None
            
        except (EOFError, KeyboardInterrupt):
            return None
        except Exception:
            return None
            
    async def _handle_batch_operations(self, improvements: List, layout: Layout) -> None:
        """Handle batch approval operations.
        
        Args:
            improvements: List of improvements
            layout: Layout for UI updates
        """
        try:
            self.console.print("\n🔄 [bold]Batch Operations[/bold]")
            
            batch_choice = Prompt.ask(
                "Batch operation",
                choices=["approve_all", "reject_all", "approve_high", "reject_low", "clear", "cancel"],
                default="cancel"
            )
            
            if batch_choice == "approve_all":
                for imp in improvements[self.session.current_index:]:
                    self.session.decisions[imp.id] = ApprovalDecision.APPROVED
                self.console.print("✅ [green]Approved all remaining improvements[/green]")
                self.session.current_index = len(improvements)
                
            elif batch_choice == "reject_all":
                for imp in improvements[self.session.current_index:]:
                    self.session.decisions[imp.id] = ApprovalDecision.REJECTED
                self.console.print("❌ [red]Rejected all remaining improvements[/red]")
                self.session.current_index = len(improvements)
                
            elif batch_choice == "approve_high":
                count = 0
                for imp in improvements[self.session.current_index:]:
                    if imp.priority == "high":
                        self.session.decisions[imp.id] = ApprovalDecision.APPROVED
                        count += 1
                self.console.print(f"✅ [green]Approved {count} high-priority improvements[/green]")
                
            elif batch_choice == "reject_low":
                count = 0
                for imp in improvements[self.session.current_index:]:
                    if imp.priority == "low":
                        self.session.decisions[imp.id] = ApprovalDecision.REJECTED
                        count += 1
                self.console.print(f"❌ [red]Rejected {count} low-priority improvements[/red]")
                
            elif batch_choice == "clear":
                # Clear all decisions
                self.session.decisions.clear()
                self.session.approval_reasons.clear()
                self.console.print("🔄 [yellow]Cleared all decisions[/yellow]")
                
        except (EOFError, KeyboardInterrupt):
            pass
        except Exception as e:
            self.console.print(f"❌ Batch operation failed: {e}")
            
    async def _show_approval_help(self) -> None:
        """Show detailed help for approval process."""
        try:
            help_content = []
            
            # Commands section
            commands_table = Table(show_header=True, header_style="bold magenta")
            commands_table.add_column("Command", style="cyan", width=15)
            commands_table.add_column("Description", style="white")
            
            commands_table.add_row("y, yes", "Approve the current improvement")
            commands_table.add_row("n, no", "Reject the current improvement")
            commands_table.add_row("s, skip", "Skip (neutral - will be reviewed later)")
            commands_table.add_row("b, back", "Go back to previous improvement")
            commands_table.add_row("batch", "Batch operations menu")
            commands_table.add_row("summary", "Show current decision summary")
            commands_table.add_row("help", "Show this help")
            commands_table.add_row("q, quit", "Exit approval process")
            
            help_content.append(commands_table)
            
            # Decision guidelines
            guidelines_text = Text()
            guidelines_text.append("\n📋 Decision Guidelines:", style="bold yellow")
            guidelines_text.append("\n• Consider impact vs effort ratio")
            guidelines_text.append("\n• High-risk improvements need stronger justification")
            guidelines_text.append("\n• Critical security improvements should be prioritized")
            guidelines_text.append("\n• Performance improvements with >80% confidence are usually safe")
            guidelines_text.append("\n• When in doubt, skip for later review")
            
            help_content.append(guidelines_text)
            
            from rich.console import Group
            help_group = Group(*help_content)
            
            help_panel = Panel(help_group, title="📚 Approval Help", border_style="blue")
            self.console.print(help_panel)
            
            input("\nPress Enter to continue...")
            
        except (EOFError, KeyboardInterrupt):
            pass
        except Exception as e:
            self.console.print(f"❌ Help display failed: {e}")
            
    async def _show_current_summary(self, improvements: List) -> None:
        """Show current approval session summary.
        
        Args:
            improvements: List of all improvements
        """
        try:
            summary_content = []
            
            # Overall stats
            stats_table = Table.grid(padding=(0, 2))
            stats_table.add_column("Metric", style="bold cyan")
            stats_table.add_column("Value", style="white")
            
            approved_count = len([d for d in self.session.decisions.values() if d == ApprovalDecision.APPROVED])
            rejected_count = len([d for d in self.session.decisions.values() if d == ApprovalDecision.REJECTED])
            skipped_count = len([d for d in self.session.decisions.values() if d == ApprovalDecision.SKIPPED])
            remaining_count = len(improvements) - len(self.session.decisions)
            
            stats_table.add_row("Total Improvements:", str(len(improvements)))
            stats_table.add_row("Approved:", f"[green]{approved_count}[/green]")
            stats_table.add_row("Rejected:", f"[red]{rejected_count}[/red]")
            stats_table.add_row("Skipped:", f"[yellow]{skipped_count}[/yellow]")
            stats_table.add_row("Remaining:", f"[cyan]{remaining_count}[/cyan]")
            
            summary_content.append(stats_table)
            
            # Decision breakdown by category
            if self.session.decisions:
                category_breakdown = {}
                for imp in improvements:
                    decision = self.session.decisions.get(imp.id, ApprovalDecision.PENDING)
                    if decision != ApprovalDecision.PENDING:
                        if imp.category not in category_breakdown:
                            category_breakdown[imp.category] = {"approved": 0, "rejected": 0, "skipped": 0}
                        category_breakdown[imp.category][decision.value] += 1
                
                if category_breakdown:
                    summary_content.append(Text("\n📊 Decisions by Category:", style="bold yellow"))
                    
                    category_table = Table(show_header=True, header_style="bold magenta")
                    category_table.add_column("Category", style="cyan")
                    category_table.add_column("Approved", style="green")
                    category_table.add_column("Rejected", style="red")
                    category_table.add_column("Skipped", style="yellow")
                    
                    for category, counts in category_breakdown.items():
                        category_table.add_row(
                            category.title(),
                            str(counts["approved"]),
                            str(counts["rejected"]),
                            str(counts["skipped"])
                        )
                    
                    summary_content.append(category_table)
            
            from rich.console import Group
            summary_group = Group(*summary_content)
            
            summary_panel = Panel(summary_group, title="📈 Approval Summary", border_style="green")
            self.console.print(summary_panel)
            
            input("\nPress Enter to continue...")
            
        except (EOFError, KeyboardInterrupt):
            pass
        except Exception as e:
            self.console.print(f"❌ Summary display failed: {e}")
            
    def _process_final_decisions(self, improvements: List) -> Tuple[List, List]:
        """Process final approval decisions and return approved/rejected lists.
        
        Args:
            improvements: List of all improvements
            
        Returns:
            Tuple of (approved_improvements, rejected_improvements)
        """
        approved = []
        rejected = []
        
        for improvement in improvements:
            decision = self.session.decisions.get(improvement.id, ApprovalDecision.PENDING)
            
            if decision == ApprovalDecision.APPROVED:
                approved.append(improvement)
            elif decision == ApprovalDecision.REJECTED:
                rejected.append(improvement)
            # Skipped and pending items are not included in either list
            
        return approved, rejected
        
    async def _show_approval_summary(self, approved: List, rejected: List, layout: Layout) -> None:
        """Show final approval summary.
        
        Args:
            approved: List of approved improvements
            rejected: List of rejected improvements
            layout: Layout for UI updates
        """
        summary_content = []
        
        # Title
        title_text = Text()
        title_text.append("✅ ", style="green")
        title_text.append("Approval Process Complete!", style="bold green")
        summary_content.append(title_text)
        summary_content.append("")
        
        # Results summary
        results_table = Table.grid(padding=(0, 2))
        results_table.add_column("Result", style="bold")
        results_table.add_column("Count", style="white")
        results_table.add_column("Details", style="dim")
        
        results_table.add_row(
            "[green]✅ Approved:",
            f"[green]{len(approved)}[/green]",
            "Ready for implementation" if approved else "None selected"
        )
        results_table.add_row(
            "[red]❌ Rejected:",
            f"[red]{len(rejected)}[/red]",
            "Will not be implemented" if rejected else "None rejected"
        )
        
        summary_content.append(results_table)
        
        # Approved improvements list
        if approved:
            summary_content.append("")
            approved_text = Text()
            approved_text.append("🚀 Approved for Implementation:", style="bold green")
            summary_content.append(approved_text)
            
            for i, improvement in enumerate(approved, 1):
                imp_text = Text()
                imp_text.append(f"  {i}. ", style="green")
                imp_text.append(improvement.title, style="white")
                imp_text.append(f" ({improvement.priority} priority)", style="dim")
                summary_content.append(imp_text)
        
        # Next steps
        if approved:
            summary_content.append("")
            next_steps_text = Text()
            next_steps_text.append("📋 Next Steps:", style="bold cyan")
            next_steps_text.append("\n• Implementation monitoring will begin")
            next_steps_text.append("\n• Progress tracking will be enabled")
            next_steps_text.append("\n• Safety validation will be performed")
            summary_content.append(next_steps_text)
        
        from rich.console import Group
        summary_group = Group(*summary_content)
        
        if layout and "current_improvement" in layout:
            layout["current_improvement"].update(
                Panel(summary_group, title="🎉 Final Results", border_style="green")
            )
        else:
            self.console.print(Panel(summary_group, title="🎉 Final Results", border_style="green"))
            
        # Brief pause for user to read summary
        await asyncio.sleep(3)


# Export main class
__all__ = ["ApprovalInteractionHandler", "ApprovalDecision", "ApprovalSession"]