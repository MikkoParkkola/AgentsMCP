"""
Abstract interfaces and implementations for different approval methods.

This module provides the interface abstractions and concrete implementations
for CLI, TUI, and automated approval interfaces.
"""

from __future__ import annotations

import asyncio
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import logging

from .approval_config import ApprovalConfig, UserPreferences
from .approval_decision import (
    ApprovalDecision, ApprovalStatus, ApprovalMode, ApprovalInterfaceType,
    ApprovalContext, BatchApprovalDecision, RejectionReason
)
from ..data_models import SelfImprovementAction, ImprovementCategory, PriorityLevel

logger = logging.getLogger(__name__)


@dataclass
class ApprovalPromptData:
    """Data structure for approval prompt display."""
    
    improvement: SelfImprovementAction
    impact_estimate: Optional[str] = None
    risk_analysis: Optional[str] = None
    implementation_details: Optional[str] = None
    similar_approvals: List[ApprovalDecision] = None
    estimated_duration: Optional[str] = None
    confidence_level: float = 1.0
    
    def __post_init__(self):
        if self.similar_approvals is None:
            self.similar_approvals = []


class ApprovalInterface(ABC):
    """Abstract base class for approval interfaces."""
    
    def __init__(self, config: ApprovalConfig):
        self.config = config
        self.preferences = config.user_preferences
        self._session_stats = {
            "approvals": 0,
            "rejections": 0,
            "timeouts": 0,
            "start_time": time.time()
        }
    
    @abstractmethod
    async def request_approval(self, 
                             improvements: List[SelfImprovementAction],
                             context: ApprovalContext) -> BatchApprovalDecision:
        """Request approval for a list of improvements."""
        pass
    
    @abstractmethod
    async def request_single_approval(self,
                                    improvement: SelfImprovementAction,
                                    prompt_data: ApprovalPromptData,
                                    timeout_seconds: Optional[int] = None) -> ApprovalDecision:
        """Request approval for a single improvement."""
        pass
    
    @abstractmethod
    def display_progress(self, message: str, progress: float = 0.0) -> None:
        """Display progress information."""
        pass
    
    @abstractmethod
    def display_summary(self, batch_decision: BatchApprovalDecision) -> None:
        """Display approval session summary."""
        pass
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        elapsed = time.time() - self._session_stats["start_time"]
        return {
            **self._session_stats,
            "elapsed_seconds": elapsed,
            "items_per_minute": (self._session_stats["approvals"] + 
                               self._session_stats["rejections"]) / max(elapsed / 60, 1)
        }


class AutoApprovalInterface(ApprovalInterface):
    """Automated approval interface that requires no user interaction."""
    
    def __init__(self, config: ApprovalConfig):
        super().__init__(config)
        self.auto_approve_all = config.approval_mode == ApprovalMode.AUTO
    
    async def request_approval(self,
                             improvements: List[SelfImprovementAction],
                             context: ApprovalContext) -> BatchApprovalDecision:
        """Automatically approve or reject improvements based on configuration."""
        batch = BatchApprovalDecision(
            session_id=context.session_id,
            batch_action=ApprovalMode.AUTO,
            approved_by="system"
        )
        
        logger.info(f"Auto-processing {len(improvements)} improvements")
        
        for improvement in improvements:
            decision = await self._make_auto_decision(improvement, context)
            batch.add_decision(decision)
            
            if decision.status == ApprovalStatus.APPROVED:
                self._session_stats["approvals"] += 1
            else:
                self._session_stats["rejections"] += 1
        
        batch.complete_batch("system")
        return batch
    
    async def request_single_approval(self,
                                    improvement: SelfImprovementAction,
                                    prompt_data: ApprovalPromptData,
                                    timeout_seconds: Optional[int] = None) -> ApprovalDecision:
        """Auto-approve or reject a single improvement."""
        context = ApprovalContext()
        context.add_improvement(improvement)
        return await self._make_auto_decision(improvement, context)
    
    async def _make_auto_decision(self, 
                                improvement: SelfImprovementAction,
                                context: ApprovalContext) -> ApprovalDecision:
        """Make an automated approval decision."""
        decision = ApprovalDecision(
            improvement_id=improvement.action_id,
            approval_mode_used=ApprovalMode.AUTO,
            interface_type_used=ApprovalInterfaceType.AUTO,
            session_id=context.session_id
        )
        
        # Check if this category/priority should be auto-approved
        if (self.auto_approve_all or 
            self.config.should_auto_approve_category(improvement.category, improvement.priority)):
            
            decision.approve("system", "Auto-approved based on configuration")
            logger.debug(f"Auto-approved improvement: {improvement.title}")
        else:
            # Auto-reject with appropriate reason
            reason = self._determine_rejection_reason(improvement)
            decision.reject("system", reason, "Auto-rejected based on configuration")
            logger.debug(f"Auto-rejected improvement: {improvement.title} (reason: {reason.value})")
        
        return decision
    
    def _determine_rejection_reason(self, improvement: SelfImprovementAction) -> RejectionReason:
        """Determine appropriate rejection reason for auto-rejection."""
        # Simple heuristics for rejection reasons
        if improvement.priority == PriorityLevel.LOW:
            return RejectionReason.LOW_IMPACT
        elif improvement.category == ImprovementCategory.SECURITY:
            return RejectionReason.TOO_RISKY
        else:
            return RejectionReason.POLICY_VIOLATION
    
    def display_progress(self, message: str, progress: float = 0.0) -> None:
        """Display progress (minimal for auto interface)."""
        logger.info(f"Auto approval: {message} ({progress:.1%})")
    
    def display_summary(self, batch_decision: BatchApprovalDecision) -> None:
        """Display minimal summary for auto approval."""
        logger.info(f"Auto approval completed: {batch_decision.approved_count} approved, "
                   f"{batch_decision.rejected_count} rejected")


class CLIApprovalInterface(ApprovalInterface):
    """Command-line interface for approval decisions."""
    
    def __init__(self, config: ApprovalConfig):
        super().__init__(config)
        self._current_session: Optional[str] = None
    
    async def request_approval(self,
                             improvements: List[SelfImprovementAction],
                             context: ApprovalContext) -> BatchApprovalDecision:
        """Request approval via CLI prompts."""
        self._current_session = context.session_id
        
        print(f"\n🔍 Approval Required for {len(improvements)} Improvements")
        print("=" * 50)
        
        # Show batch options
        batch_choice = self._get_batch_choice(len(improvements))
        
        batch = BatchApprovalDecision(
            session_id=context.session_id,
            batch_action=batch_choice,
            approved_by="user"
        )
        
        if batch_choice in [ApprovalMode.BATCH_APPROVE, ApprovalMode.BATCH_REJECT]:
            # Handle batch operations
            await self._handle_batch_operation(batch_choice, improvements, batch, context)
        else:
            # Individual review
            await self._handle_individual_review(improvements, batch, context)
        
        batch.complete_batch("user")
        return batch
    
    def _get_batch_choice(self, improvement_count: int) -> ApprovalMode:
        """Get user's choice for batch operations."""
        print(f"\nHow would you like to review {improvement_count} improvements?")
        print("1. Review each individually")
        print("2. Approve all")
        print("3. Reject all") 
        print("4. Interactive (choose as we go)")
        
        while True:
            try:
                choice = input("\nEnter choice (1-4): ").strip()
                if choice == "1":
                    return ApprovalMode.MANUAL
                elif choice == "2":
                    return ApprovalMode.BATCH_APPROVE
                elif choice == "3":
                    return ApprovalMode.BATCH_REJECT
                elif choice == "4":
                    return ApprovalMode.INTERACTIVE
                else:
                    print("Invalid choice. Please enter 1-4.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled.")
                return ApprovalMode.BATCH_REJECT
    
    async def _handle_batch_operation(self,
                                    batch_choice: ApprovalMode,
                                    improvements: List[SelfImprovementAction],
                                    batch: BatchApprovalDecision,
                                    context: ApprovalContext) -> None:
        """Handle batch approve or reject operations."""
        action = "approve" if batch_choice == ApprovalMode.BATCH_APPROVE else "reject"
        
        if self.preferences.confirm_batch_operations:
            print(f"\n⚠️  Are you sure you want to {action} all {len(improvements)} improvements?")
            confirm = input("Type 'yes' to confirm: ").strip().lower()
            if confirm != "yes":
                print("Batch operation cancelled. Switching to individual review.")
                await self._handle_individual_review(improvements, batch, context)
                return
        
        # Process all improvements with the same decision
        for i, improvement in enumerate(improvements, 1):
            self.display_progress(f"Processing {i}/{len(improvements)}", i / len(improvements))
            
            decision = ApprovalDecision(
                improvement_id=improvement.action_id,
                approval_mode_used=batch_choice,
                interface_type_used=ApprovalInterfaceType.CLI,
                session_id=context.session_id
            )
            
            if batch_choice == ApprovalMode.BATCH_APPROVE:
                decision.approve("user", "Batch approved")
                self._session_stats["approvals"] += 1
            else:
                decision.reject("user", RejectionReason.USER_PREFERENCE, "Batch rejected")
                self._session_stats["rejections"] += 1
            
            batch.add_decision(decision)
    
    async def _handle_individual_review(self,
                                      improvements: List[SelfImprovementAction],
                                      batch: BatchApprovalDecision,
                                      context: ApprovalContext) -> None:
        """Handle individual review of improvements."""
        for i, improvement in enumerate(improvements, 1):
            print(f"\n--- Improvement {i}/{len(improvements)} ---")
            
            prompt_data = ApprovalPromptData(improvement=improvement)
            decision = await self.request_single_approval(
                improvement, prompt_data, self.config.timeouts.individual_item_timeout
            )
            
            batch.add_decision(decision)
            
            if decision.status == ApprovalStatus.APPROVED:
                self._session_stats["approvals"] += 1
            elif decision.status == ApprovalStatus.REJECTED:
                self._session_stats["rejections"] += 1
            else:
                self._session_stats["timeouts"] += 1
    
    async def request_single_approval(self,
                                    improvement: SelfImprovementAction,
                                    prompt_data: ApprovalPromptData,
                                    timeout_seconds: Optional[int] = None) -> ApprovalDecision:
        """Request approval for a single improvement via CLI."""
        decision = ApprovalDecision(
            improvement_id=improvement.action_id,
            approval_mode_used=ApprovalMode.MANUAL,
            interface_type_used=ApprovalInterfaceType.CLI
        )
        
        # Display improvement details
        self._display_improvement_details(improvement, prompt_data)
        
        # Get user decision
        while True:
            try:
                print("\nOptions:")
                print("  a - Approve")
                print("  r - Reject") 
                print("  s - Skip (for now)")
                print("  d - Show more details")
                print("  q - Quit session")
                
                choice = input("Your choice (a/r/s/d/q): ").strip().lower()
                
                if choice == "a":
                    notes = input("Optional approval notes: ").strip()
                    decision.approve("user", notes)
                    break
                elif choice == "r":
                    reason = self._get_rejection_reason()
                    notes = input("Optional rejection notes: ").strip()
                    decision.reject("user", reason, notes)
                    break
                elif choice == "s":
                    # Keep as pending
                    break
                elif choice == "d":
                    self._display_detailed_info(improvement, prompt_data)
                elif choice == "q":
                    decision.reject("user", RejectionReason.USER_PREFERENCE, "Session cancelled")
                    break
                else:
                    print("Invalid choice. Please enter a, r, s, d, or q.")
                    
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled.")
                decision.reject("user", RejectionReason.USER_PREFERENCE, "Interrupted by user")
                break
        
        return decision
    
    def _display_improvement_details(self, 
                                   improvement: SelfImprovementAction,
                                   prompt_data: ApprovalPromptData) -> None:
        """Display improvement details in CLI format."""
        print(f"\n📋 Title: {improvement.title}")
        print(f"🏷️  Category: {improvement.category.value.title()}")
        print(f"⭐ Priority: {improvement.priority.value.title()}")
        
        if improvement.description:
            print(f"📝 Description: {improvement.description}")
        
        if improvement.expected_benefit:
            print(f"💰 Expected Benefit: {improvement.expected_benefit}")
        
        if improvement.estimated_effort:
            print(f"⏱️  Estimated Effort: {improvement.estimated_effort}")
        
        if prompt_data.impact_estimate and self.preferences.show_impact_estimates:
            print(f"📊 Impact Estimate: {prompt_data.impact_estimate}")
        
        if prompt_data.risk_analysis and self.preferences.show_risk_analysis:
            print(f"⚠️  Risk Analysis: {prompt_data.risk_analysis}")
    
    def _display_detailed_info(self,
                              improvement: SelfImprovementAction,
                              prompt_data: ApprovalPromptData) -> None:
        """Display detailed improvement information."""
        print("\n" + "="*60)
        print("DETAILED INFORMATION")
        print("="*60)
        
        if improvement.implementation_notes:
            print(f"🔧 Implementation Notes:\n{improvement.implementation_notes}")
        
        if prompt_data.implementation_details:
            print(f"\n🏗️  Implementation Details:\n{prompt_data.implementation_details}")
        
        if prompt_data.similar_approvals:
            print(f"\n📚 Similar Previous Approvals: {len(prompt_data.similar_approvals)}")
            for similar in prompt_data.similar_approvals[:3]:  # Show top 3
                print(f"  - {similar.status.value} ({similar.user_notes[:50]}...)")
        
        if prompt_data.confidence_level:
            print(f"\n🎯 Confidence Level: {prompt_data.confidence_level:.1%}")
        
        print("\n" + "="*60)
    
    def _get_rejection_reason(self) -> RejectionReason:
        """Get rejection reason from user."""
        reasons = list(RejectionReason)
        
        print("\nRejection reasons:")
        for i, reason in enumerate(reasons, 1):
            print(f"  {i}. {reason.value.replace('_', ' ').title()}")
        
        while True:
            try:
                choice = input(f"Select reason (1-{len(reasons)}): ").strip()
                index = int(choice) - 1
                if 0 <= index < len(reasons):
                    return reasons[index]
                else:
                    print(f"Invalid choice. Please enter 1-{len(reasons)}.")
            except (ValueError, KeyboardInterrupt, EOFError):
                print("Invalid input or operation cancelled. Using 'User Preference'.")
                return RejectionReason.USER_PREFERENCE
    
    def display_progress(self, message: str, progress: float = 0.0) -> None:
        """Display progress in CLI format."""
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r{message} [{bar}] {progress:.1%}", end="", flush=True)
        if progress >= 1.0:
            print()  # New line when complete
    
    def display_summary(self, batch_decision: BatchApprovalDecision) -> None:
        """Display approval session summary in CLI format."""
        print("\n" + "="*50)
        print("APPROVAL SESSION SUMMARY")
        print("="*50)
        
        total = len(batch_decision.individual_decisions)
        print(f"📊 Total items reviewed: {total}")
        print(f"✅ Approved: {batch_decision.approved_count}")
        print(f"❌ Rejected: {batch_decision.rejected_count}")
        print(f"📈 Approval rate: {batch_decision.get_approval_rate():.1%}")
        
        if batch_decision.batch_notes:
            print(f"📝 Session notes: {batch_decision.batch_notes}")
        
        elapsed = (batch_decision.completed_at - batch_decision.started_at).total_seconds()
        print(f"⏱️  Session duration: {elapsed:.1f} seconds")
        
        print("="*50)


class TUIApprovalInterface(ApprovalInterface):
    """Rich Terminal UI interface for approval decisions."""
    
    def __init__(self, config: ApprovalConfig):
        super().__init__(config)
        self._rich_available = self._check_rich_availability()
        
        if not self._rich_available:
            logger.warning("Rich library not available, falling back to CLI interface")
    
    def _check_rich_availability(self) -> bool:
        """Check if Rich library is available for TUI."""
        try:
            import rich
            return True
        except ImportError:
            return False
    
    async def request_approval(self,
                             improvements: List[SelfImprovementAction],
                             context: ApprovalContext) -> BatchApprovalDecision:
        """Request approval via Rich TUI (fallback to CLI if Rich not available)."""
        if not self._rich_available:
            # Fallback to CLI interface
            cli_interface = CLIApprovalInterface(self.config)
            return await cli_interface.request_approval(improvements, context)
        
        # Rich TUI implementation would go here
        # For now, implement a basic version that delegates to CLI
        return await self._basic_tui_approval(improvements, context)
    
    async def _basic_tui_approval(self,
                                improvements: List[SelfImprovementAction],
                                context: ApprovalContext) -> BatchApprovalDecision:
        """Basic TUI implementation."""
        try:
            from rich.console import Console
            from rich.prompt import Prompt, Confirm
            from rich.table import Table
            from rich.panel import Panel
            from rich.progress import Progress, TaskID
            
            console = Console()
            
            # Display header
            console.print(Panel(
                f"[bold cyan]Approval Required for {len(improvements)} Improvements[/bold cyan]",
                title="Retrospective Improvement Approval",
                border_style="cyan"
            ))
            
            # Create batch decision
            batch = BatchApprovalDecision(
                session_id=context.session_id,
                batch_action=ApprovalMode.INTERACTIVE,
                approved_by="user"
            )
            
            # Process each improvement
            with Progress() as progress:
                task = progress.add_task("[cyan]Processing improvements...", total=len(improvements))
                
                for i, improvement in enumerate(improvements):
                    # Display improvement in a nice panel
                    self._display_rich_improvement(console, improvement, i + 1, len(improvements))
                    
                    # Get decision
                    decision = await self._get_rich_decision(console, improvement, context)
                    batch.add_decision(decision)
                    
                    if decision.status == ApprovalStatus.APPROVED:
                        self._session_stats["approvals"] += 1
                    elif decision.status == ApprovalStatus.REJECTED:
                        self._session_stats["rejections"] += 1
                    
                    progress.update(task, advance=1)
            
            batch.complete_batch("user")
            self._display_rich_summary(console, batch)
            return batch
            
        except ImportError:
            # Fallback to CLI if Rich import fails
            cli_interface = CLIApprovalInterface(self.config)
            return await cli_interface.request_approval(improvements, context)
    
    def _display_rich_improvement(self, console, improvement: SelfImprovementAction, 
                                current: int, total: int) -> None:
        """Display improvement using Rich formatting."""
        try:
            from rich.table import Table
            from rich.panel import Panel
            
            table = Table(show_header=False, box=None, pad_edge=False)
            table.add_column("Field", style="bold")
            table.add_column("Value")
            
            table.add_row("Title", f"[bold]{improvement.title}[/bold]")
            table.add_row("Category", f"[yellow]{improvement.category.value.title()}[/yellow]")
            table.add_row("Priority", f"[red]{improvement.priority.value.title()}[/red]")
            
            if improvement.description:
                table.add_row("Description", improvement.description)
            
            if improvement.expected_benefit:
                table.add_row("Expected Benefit", f"[green]{improvement.expected_benefit}[/green]")
            
            if improvement.estimated_effort:
                table.add_row("Estimated Effort", f"[blue]{improvement.estimated_effort}[/blue]")
            
            console.print(Panel(
                table,
                title=f"[bold]Improvement {current}/{total}[/bold]",
                border_style="blue"
            ))
            
        except ImportError:
            # Fallback to basic display
            console.print(f"\n=== Improvement {current}/{total} ===")
            console.print(f"Title: {improvement.title}")
            console.print(f"Category: {improvement.category.value}")
            console.print(f"Priority: {improvement.priority.value}")
    
    async def _get_rich_decision(self, console, improvement: SelfImprovementAction,
                               context: ApprovalContext) -> ApprovalDecision:
        """Get approval decision using Rich prompts."""
        try:
            from rich.prompt import Prompt
            
            decision = ApprovalDecision(
                improvement_id=improvement.action_id,
                approval_mode_used=ApprovalMode.INTERACTIVE,
                interface_type_used=ApprovalInterfaceType.TUI,
                session_id=context.session_id
            )
            
            choice = Prompt.ask(
                "\n[bold]Decision[/bold]",
                choices=["approve", "reject", "skip"],
                default="approve"
            )
            
            if choice == "approve":
                notes = Prompt.ask("[dim]Approval notes (optional)[/dim]", default="")
                decision.approve("user", notes)
            elif choice == "reject":
                reason_choice = Prompt.ask(
                    "[bold]Rejection reason[/bold]",
                    choices=["too_risky", "low_impact", "resource_constraints", "user_preference"],
                    default="user_preference"
                )
                reason = RejectionReason(reason_choice)
                notes = Prompt.ask("[dim]Rejection notes (optional)[/dim]", default="")
                decision.reject("user", reason, notes)
            # else: keep as pending (skip)
            
            return decision
            
        except ImportError:
            # Fallback to basic input
            choice = input("Decision (approve/reject/skip): ").strip().lower()
            decision = ApprovalDecision(
                improvement_id=improvement.action_id,
                approval_mode_used=ApprovalMode.INTERACTIVE,
                interface_type_used=ApprovalInterfaceType.TUI,
                session_id=context.session_id
            )
            
            if choice == "approve":
                decision.approve("user", "")
            elif choice == "reject":
                decision.reject("user", RejectionReason.USER_PREFERENCE, "")
            
            return decision
    
    def _display_rich_summary(self, console, batch: BatchApprovalDecision) -> None:
        """Display summary using Rich formatting."""
        try:
            from rich.panel import Panel
            from rich.table import Table
            
            table = Table(show_header=False)
            table.add_column("Metric", style="bold")
            table.add_column("Value")
            
            total = len(batch.individual_decisions)
            table.add_row("Total Items", str(total))
            table.add_row("✅ Approved", f"[green]{batch.approved_count}[/green]")
            table.add_row("❌ Rejected", f"[red]{batch.rejected_count}[/red]")
            table.add_row("📈 Approval Rate", f"[cyan]{batch.get_approval_rate():.1%}[/cyan]")
            
            console.print(Panel(
                table,
                title="[bold green]Approval Session Complete[/bold green]",
                border_style="green"
            ))
            
        except ImportError:
            console.print(f"\nSession Complete: {batch.approved_count} approved, "
                         f"{batch.rejected_count} rejected")
    
    async def request_single_approval(self,
                                    improvement: SelfImprovementAction,
                                    prompt_data: ApprovalPromptData,
                                    timeout_seconds: Optional[int] = None) -> ApprovalDecision:
        """Request single approval via TUI (fallback to CLI if needed)."""
        if not self._rich_available:
            cli_interface = CLIApprovalInterface(self.config)
            return await cli_interface.request_single_approval(improvement, prompt_data, timeout_seconds)
        
        # Implement rich single approval
        return await self._rich_single_approval(improvement, prompt_data, timeout_seconds)
    
    async def _rich_single_approval(self,
                                  improvement: SelfImprovementAction,
                                  prompt_data: ApprovalPromptData,
                                  timeout_seconds: Optional[int] = None) -> ApprovalDecision:
        """Single approval using Rich UI."""
        try:
            from rich.console import Console
            
            console = Console()
            self._display_rich_improvement(console, improvement, 1, 1)
            return await self._get_rich_decision(console, improvement, ApprovalContext())
            
        except ImportError:
            # Fallback implementation
            decision = ApprovalDecision(
                improvement_id=improvement.action_id,
                interface_type_used=ApprovalInterfaceType.TUI
            )
            decision.approve("user", "Fallback approval")
            return decision
    
    def display_progress(self, message: str, progress: float = 0.0) -> None:
        """Display progress using Rich if available."""
        if self._rich_available:
            try:
                from rich.console import Console
                console = Console()
                console.print(f"[cyan]{message}[/cyan] {progress:.1%}")
            except ImportError:
                print(f"{message} {progress:.1%}")
        else:
            print(f"{message} {progress:.1%}")
    
    def display_summary(self, batch_decision: BatchApprovalDecision) -> None:
        """Display summary using Rich if available."""
        if self._rich_available:
            try:
                from rich.console import Console
                console = Console()
                self._display_rich_summary(console, batch_decision)
                return
            except ImportError:
                pass
        
        # Fallback to basic display
        print(f"\nSession Complete: {batch_decision.approved_count} approved, "
              f"{batch_decision.rejected_count} rejected")


def create_approval_interface(config: ApprovalConfig) -> ApprovalInterface:
    """Factory function to create appropriate approval interface."""
    interface_type = config.interface_type
    
    if interface_type == ApprovalInterfaceType.AUTO:
        return AutoApprovalInterface(config)
    elif interface_type == ApprovalInterfaceType.CLI:
        return CLIApprovalInterface(config)
    elif interface_type == ApprovalInterfaceType.TUI:
        return TUIApprovalInterface(config)
    else:
        logger.warning(f"Unknown interface type: {interface_type}, defaulting to AUTO")
        return AutoApprovalInterface(config)