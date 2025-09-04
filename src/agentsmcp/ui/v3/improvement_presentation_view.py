"""
ImprovementPresentationView - Rich improvement display with beautiful markdown rendering.

This module provides a sophisticated presentation interface for improvement suggestions,
leveraging Rich's markdown rendering capabilities to create beautiful, interactive displays.
"""

import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich.columns import Columns
from rich.tree import Tree
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.prompt import Prompt
from rich.align import Align


@dataclass 
class PresentationConfig:
    """Configuration for improvement presentation."""
    items_per_page: int = 5
    auto_advance_delay: float = 3.0
    show_detailed_metrics: bool = True
    enable_filtering: bool = True
    markdown_width: int = 80


class ImprovementPresentationView:
    """Rich improvement display with beautiful markdown formatting."""
    
    def __init__(self, console: Console):
        """Initialize the presentation view.
        
        Args:
            console: Rich console for rendering
        """
        self.console = console
        self.config = PresentationConfig()
        self.current_page = 0
        self.selected_improvement = 0
        self.filter_category = None
        self.filter_priority = None
        
    async def display_improvements(self, improvements: List, layout: Layout) -> bool:
        """Display improvements with interactive navigation and beautiful formatting.
        
        Args:
            improvements: List of improvement suggestions
            layout: Rich layout to update
            
        Returns:
            True if display completed successfully, False otherwise
        """
        try:
            if not improvements:
                await self._show_no_improvements_message(layout)
                return True
                
            # Create presentation workflow
            return await self._run_presentation_workflow(improvements, layout)
            
        except Exception as e:
            self.console.print(f"❌ Presentation display failed: {e}")
            return False
            
    async def _run_presentation_workflow(self, improvements: List, layout: Layout) -> bool:
        """Run the interactive presentation workflow.
        
        Args:
            improvements: List of improvement suggestions
            layout: Rich layout to update
            
        Returns:
            True if workflow completed successfully
        """
        filtered_improvements = self._apply_filters(improvements)
        
        while True:
            # Update the improvements list panel
            self._update_improvements_list_panel(filtered_improvements, layout)
            
            # Update the improvement details panel  
            if filtered_improvements:
                current_improvement = filtered_improvements[self.selected_improvement]
                self._update_improvement_details_panel(current_improvement, layout)
            
            # Show navigation prompt
            action = await self._get_user_navigation_action(filtered_improvements)
            
            if action == "quit":
                return True
            elif action == "next_improvement":
                self.selected_improvement = min(
                    self.selected_improvement + 1, 
                    len(filtered_improvements) - 1
                )
            elif action == "prev_improvement":
                self.selected_improvement = max(self.selected_improvement - 1, 0)
            elif action == "next_page":
                self.current_page = min(
                    self.current_page + 1,
                    (len(filtered_improvements) - 1) // self.config.items_per_page
                )
                self.selected_improvement = self.current_page * self.config.items_per_page
            elif action == "prev_page":
                self.current_page = max(self.current_page - 1, 0)
                self.selected_improvement = self.current_page * self.config.items_per_page
            elif action == "filter":
                await self._handle_filtering()
                filtered_improvements = self._apply_filters(improvements)
                self.selected_improvement = 0
                self.current_page = 0
            elif action == "details":
                await self._show_detailed_improvement_view(
                    filtered_improvements[self.selected_improvement]
                )
            elif action == "continue":
                return True
                
    def _apply_filters(self, improvements: List) -> List:
        """Apply category and priority filters to improvements.
        
        Args:
            improvements: List of all improvements
            
        Returns:
            Filtered list of improvements
        """
        filtered = improvements
        
        if self.filter_category:
            filtered = [imp for imp in filtered if imp.category == self.filter_category]
            
        if self.filter_priority:
            filtered = [imp for imp in filtered if imp.priority == self.filter_priority]
            
        return filtered
        
    def _update_improvements_list_panel(self, improvements: List, layout: Layout) -> None:
        """Update the improvements list panel with current page.
        
        Args:
            improvements: List of improvements to display
            layout: Layout to update
        """
        if not improvements:
            layout["improvements_list"].update(
                Panel("No improvements found.", title="Improvements", border_style="yellow")
            )
            return
            
        # Calculate pagination
        start_idx = self.current_page * self.config.items_per_page
        end_idx = min(start_idx + self.config.items_per_page, len(improvements))
        page_improvements = improvements[start_idx:end_idx]
        
        # Create improvements table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Priority", width=8)
        table.add_column("Category", width=12)
        table.add_column("Title", style="bold")
        table.add_column("Impact", width=12)
        table.add_column("Effort", width=8)
        table.add_column("Confidence", width=10)
        
        for i, improvement in enumerate(page_improvements):
            row_idx = start_idx + i
            
            # Highlight selected improvement
            row_style = "bold blue" if row_idx == self.selected_improvement else ""
            selection_marker = "→" if row_idx == self.selected_improvement else " "
            
            # Priority styling
            priority_color = {
                "high": "red",
                "medium": "yellow", 
                "low": "green"
            }.get(improvement.priority, "white")
            
            # Impact styling  
            impact_color = {
                "critical": "bold red",
                "significant": "bold yellow",
                "moderate": "yellow",
                "minor": "dim white"
            }.get(improvement.estimated_impact, "white")
            
            table.add_row(
                f"{selection_marker}{row_idx + 1}",
                f"[{priority_color}]{improvement.priority.upper()}[/{priority_color}]",
                improvement.category.title(),
                improvement.title,
                f"[{impact_color}]{improvement.estimated_impact}[/{impact_color}]",
                improvement.estimated_effort,
                f"{improvement.confidence_score:.1%}",
                style=row_style
            )
        
        # Add pagination info
        total_pages = (len(improvements) - 1) // self.config.items_per_page + 1 if improvements else 0
        pagination_text = f"Page {self.current_page + 1}/{total_pages} • {len(improvements)} total"
        
        # Add filter info
        filter_info = []
        if self.filter_category:
            filter_info.append(f"Category: {self.filter_category}")
        if self.filter_priority:
            filter_info.append(f"Priority: {self.filter_priority}")
            
        filter_text = " • Filters: " + ", ".join(filter_info) if filter_info else ""
        
        title = f"Improvements - {pagination_text}{filter_text}"
        
        layout["improvements_list"].update(
            Panel(table, title=title, border_style="cyan")
        )
        
    def _update_improvement_details_panel(self, improvement, layout: Layout) -> None:
        """Update the improvement details panel with rich markdown formatting.
        
        Args:
            improvement: Selected improvement to display
            layout: Layout to update
        """
        # Create detailed improvement display with markdown support
        details_content = []
        
        # Title and basic info
        title_text = Text()
        title_text.append("🔍 ", style="blue")
        title_text.append(improvement.title, style="bold white")
        details_content.append(title_text)
        details_content.append("")
        
        # Priority and impact badges
        priority_colors = {"high": "red", "medium": "yellow", "low": "green"}
        impact_colors = {
            "critical": "bold red", "significant": "bold yellow", 
            "moderate": "yellow", "minor": "dim"
        }
        
        badges = Text()
        badges.append("🏷️  ", style="dim")
        badges.append(f"[{priority_colors.get(improvement.priority, 'white')}]"
                     f"{improvement.priority.upper()} PRIORITY[/]  ")
        badges.append(f"[{impact_colors.get(improvement.estimated_impact, 'white')}]"
                     f"{improvement.estimated_impact.upper()} IMPACT[/]")
        details_content.append(badges)
        details_content.append("")
        
        # Render description as markdown for beautiful formatting
        if hasattr(improvement, 'description') and improvement.description:
            try:
                # Create markdown with proper width constraints
                markdown_content = Markdown(
                    improvement.description,
                    code_theme="monokai"
                )
                details_content.append(markdown_content)
            except Exception:
                # Fallback to plain text if markdown fails
                details_content.append(Text(improvement.description))
        else:
            details_content.append(Text("No description available.", style="dim italic"))
            
        details_content.append("")
        
        # Implementation steps (if available)
        if hasattr(improvement, 'implementation_steps') and improvement.implementation_steps:
            steps_text = Text()
            steps_text.append("📋 Implementation Steps:", style="bold green")
            details_content.append(steps_text)
            
            for i, step in enumerate(improvement.implementation_steps, 1):
                step_text = Text()
                step_text.append(f"  {i}. ", style="cyan")
                step_text.append(step)
                details_content.append(step_text)
                
            details_content.append("")
        
        # Success metrics (if available)
        if hasattr(improvement, 'success_metrics') and improvement.success_metrics:
            metrics_text = Text()
            metrics_text.append("🎯 Success Metrics:", style="bold yellow")
            details_content.append(metrics_text)
            
            for metric in improvement.success_metrics:
                metric_text = Text()
                metric_text.append("  • ", style="yellow")
                metric_text.append(metric)
                details_content.append(metric_text)
                
            details_content.append("")
        
        # Risk assessment and confidence
        risk_info = Table.grid(padding=(0, 2))
        risk_info.add_column("Label", style="bold")
        risk_info.add_column("Value")
        
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(
            getattr(improvement, 'risk_assessment', 'medium'), "yellow"
        )
        
        risk_info.add_row(
            "Risk Level:",
            f"[{risk_color}]{getattr(improvement, 'risk_assessment', 'medium').upper()}[/{risk_color}]"
        )
        risk_info.add_row(
            "Confidence:",
            f"{getattr(improvement, 'confidence_score', 0.8):.1%}"
        )
        risk_info.add_row(
            "Effort Estimate:",
            getattr(improvement, 'estimated_effort', 'unknown').title()
        )
        risk_info.add_row(
            "Category:",
            getattr(improvement, 'category', 'general').title()
        )
        
        details_content.append(risk_info)
        
        # Combine all content
        from rich.console import Group
        combined_content = Group(*details_content)
        
        layout["improvement_details"].update(
            Panel(
                combined_content,
                title=f"[bold]Improvement Details[/bold]",
                border_style="green"
            )
        )
        
    async def _get_user_navigation_action(self, improvements: List) -> str:
        """Get user navigation action through input prompt.
        
        Args:
            improvements: List of improvements for context
            
        Returns:
            User action as string
        """
        if not improvements:
            return "continue"
            
        try:
            # Show navigation help
            help_text = Text()
            help_text.append("Navigation: ", style="bold white")
            help_text.append("↑/↓", style="cyan")
            help_text.append(" select, ", style="dim")
            help_text.append("PgUp/PgDn", style="cyan") 
            help_text.append(" page, ", style="dim")
            help_text.append("f", style="cyan")
            help_text.append(" filter, ", style="dim")
            help_text.append("d", style="cyan")
            help_text.append(" details, ", style="dim")
            help_text.append("c", style="cyan")
            help_text.append(" continue, ", style="dim")
            help_text.append("q", style="cyan")
            help_text.append(" quit", style="dim")
            
            self.console.print(help_text)
            
            # Get user input
            action = Prompt.ask(
                "Action",
                choices=["↑", "↓", "up", "down", "pgup", "pgdn", "f", "d", "c", "q", ""],
                default=""
            ).lower()
            
            # Map actions
            action_mapping = {
                "↑": "prev_improvement",
                "up": "prev_improvement", 
                "↓": "next_improvement",
                "down": "next_improvement",
                "pgup": "prev_page",
                "pgdn": "next_page",
                "f": "filter",
                "d": "details",
                "c": "continue",
                "q": "quit",
                "": "continue"
            }
            
            return action_mapping.get(action, "continue")
            
        except (EOFError, KeyboardInterrupt):
            return "quit"
        except Exception:
            return "continue"
            
    async def _handle_filtering(self) -> None:
        """Handle interactive filtering options."""
        try:
            self.console.print("\n🔍 [bold]Filter Options[/bold]")
            
            # Category filter
            category_choice = Prompt.ask(
                "Filter by category",
                choices=["all", "performance", "security", "ux", "monitoring", "cost", "reliability"],
                default="all"
            )
            
            self.filter_category = None if category_choice == "all" else category_choice
            
            # Priority filter  
            priority_choice = Prompt.ask(
                "Filter by priority",
                choices=["all", "high", "medium", "low"],
                default="all"
            )
            
            self.filter_priority = None if priority_choice == "all" else priority_choice
            
            # Confirm filters
            filters_applied = []
            if self.filter_category:
                filters_applied.append(f"Category: {self.filter_category}")
            if self.filter_priority:
                filters_applied.append(f"Priority: {self.filter_priority}")
                
            if filters_applied:
                self.console.print(f"✅ Filters applied: {', '.join(filters_applied)}")
            else:
                self.console.print("✅ All filters cleared")
                
        except (EOFError, KeyboardInterrupt):
            pass
        except Exception as e:
            self.console.print(f"❌ Filter setup failed: {e}")
            
    async def _show_detailed_improvement_view(self, improvement) -> None:
        """Show a detailed full-screen view of an improvement.
        
        Args:
            improvement: Improvement to display in detail
        """
        try:
            self.console.clear()
            
            # Title
            title_text = Text()
            title_text.append("🔍 ", style="blue")
            title_text.append("DETAILED VIEW: ", style="bold blue")
            title_text.append(improvement.title, style="bold white")
            
            title_panel = Panel(
                Align.center(title_text),
                style="blue",
                padding=(1, 2)
            )
            self.console.print(title_panel)
            
            # Create comprehensive details layout
            detail_sections = []
            
            # Overview section
            overview_table = Table.grid(padding=(0, 2))
            overview_table.add_column("Field", style="bold cyan", width=20)
            overview_table.add_column("Value", style="white")
            
            overview_table.add_row("ID:", getattr(improvement, 'id', 'N/A'))
            overview_table.add_row("Category:", getattr(improvement, 'category', 'general').title())
            overview_table.add_row("Priority:", getattr(improvement, 'priority', 'medium').title())
            overview_table.add_row("Estimated Impact:", getattr(improvement, 'estimated_impact', 'unknown').title())
            overview_table.add_row("Estimated Effort:", getattr(improvement, 'estimated_effort', 'unknown').title())
            overview_table.add_row("Risk Assessment:", getattr(improvement, 'risk_assessment', 'medium').title())
            overview_table.add_row("Confidence Score:", f"{getattr(improvement, 'confidence_score', 0.8):.1%}")
            
            detail_sections.append(Panel(overview_table, title="📊 Overview", border_style="cyan"))
            
            # Description section with markdown
            if hasattr(improvement, 'description') and improvement.description:
                try:
                    description_md = Markdown(improvement.description, code_theme="monokai")
                    detail_sections.append(
                        Panel(description_md, title="📝 Description", border_style="green")
                    )
                except Exception:
                    detail_sections.append(
                        Panel(
                            Text(improvement.description),
                            title="📝 Description", 
                            border_style="green"
                        )
                    )
            
            # Implementation steps
            if hasattr(improvement, 'implementation_steps') and improvement.implementation_steps:
                steps_content = []
                for i, step in enumerate(improvement.implementation_steps, 1):
                    step_text = Text()
                    step_text.append(f"{i}. ", style="bold cyan")
                    step_text.append(step)
                    steps_content.append(step_text)
                
                from rich.console import Group
                steps_group = Group(*steps_content)
                detail_sections.append(
                    Panel(steps_group, title="📋 Implementation Steps", border_style="yellow")
                )
            
            # Success metrics
            if hasattr(improvement, 'success_metrics') and improvement.success_metrics:
                metrics_content = []
                for metric in improvement.success_metrics:
                    metric_text = Text()
                    metric_text.append("🎯 ", style="yellow")
                    metric_text.append(metric)
                    metrics_content.append(metric_text)
                    
                from rich.console import Group
                metrics_group = Group(*metrics_content)
                detail_sections.append(
                    Panel(metrics_group, title="🎯 Success Metrics", border_style="magenta")
                )
            
            # Display all sections
            for section in detail_sections:
                self.console.print(section)
                self.console.print()  # Add spacing
            
            # Wait for user input
            input("\nPress Enter to return to improvement list...")
            
        except (EOFError, KeyboardInterrupt):
            pass
        except Exception as e:
            self.console.print(f"❌ Detailed view failed: {e}")
            input("Press Enter to continue...")
            
    async def _show_no_improvements_message(self, layout: Layout) -> None:
        """Show message when no improvements are found.
        
        Args:
            layout: Layout to update
        """
        message = Text()
        message.append("🎉 ", style="green")
        message.append("No improvement opportunities found!\n\n", style="bold green")
        message.append("This indicates that the system is already performing optimally ", style="white")
        message.append("according to current analysis parameters.", style="dim white")
        
        no_improvements_panel = Panel(
            Align.center(message),
            title="Analysis Complete",
            border_style="green"
        )
        
        if layout and "improvements_list" in layout:
            layout["improvements_list"].update(no_improvements_panel)
        else:
            self.console.print(no_improvements_panel)
            
        # Wait briefly for user to read message
        await asyncio.sleep(3)


# Export main class
__all__ = ["ImprovementPresentationView", "PresentationConfig"]