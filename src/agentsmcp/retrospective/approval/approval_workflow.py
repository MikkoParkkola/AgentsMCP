"""
Main workflow orchestration for approval decisions.

This module provides the core workflow engine that orchestrates the approval process,
coordinating between criteria evaluation, user interfaces, and decision persistence.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import logging

from .approval_config import ApprovalConfig
from .approval_decision import (
    ApprovalDecision, ApprovalStatus, ApprovalMode, ApprovalInterfaceType,
    ApprovalContext, BatchApprovalDecision, ApprovalHistory, RejectionReason
)
from .approval_interfaces import ApprovalInterface, create_approval_interface, ApprovalPromptData
from .approval_criteria import ApprovalCriteriaEngine, FilterAction
from ..data_models import SelfImprovementAction, ImprovementCategory, PriorityLevel

logger = logging.getLogger(__name__)


class WorkflowState(Enum):
    """States of the approval workflow."""
    INITIALIZING = "initializing"
    EVALUATING_CRITERIA = "evaluating_criteria"
    REQUESTING_USER_INPUT = "requesting_user_input"
    PROCESSING_DECISIONS = "processing_decisions"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowMetrics:
    """Metrics collected during workflow execution."""
    
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    
    # Timing metrics
    criteria_evaluation_time: float = 0.0
    user_interaction_time: float = 0.0
    decision_processing_time: float = 0.0
    total_processing_time: float = 0.0
    
    # Decision metrics
    total_improvements: int = 0
    auto_approved: int = 0
    auto_rejected: int = 0
    manually_approved: int = 0
    manually_rejected: int = 0
    timed_out: int = 0
    errors: int = 0
    
    # Efficiency metrics
    items_per_second: float = 0.0
    user_engagement_score: float = 0.0  # 0.0 to 1.0
    
    def finalize(self) -> None:
        """Finalize metrics calculations."""
        self.end_time = datetime.now(timezone.utc)
        self.total_processing_time = (self.end_time - self.start_time).total_seconds()
        
        if self.total_processing_time > 0:
            self.items_per_second = self.total_improvements / self.total_processing_time
        
        # Calculate user engagement (how much user was involved vs auto-processing)
        manual_decisions = self.manually_approved + self.manually_rejected
        if self.total_improvements > 0:
            self.user_engagement_score = manual_decisions / self.total_improvements
    
    def get_approval_rate(self) -> float:
        """Get overall approval rate."""
        total_decisions = (self.auto_approved + self.auto_rejected + 
                         self.manually_approved + self.manually_rejected)
        if total_decisions == 0:
            return 0.0
        
        total_approved = self.auto_approved + self.manually_approved
        return total_approved / total_decisions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "criteria_evaluation_time": self.criteria_evaluation_time,
            "user_interaction_time": self.user_interaction_time,
            "decision_processing_time": self.decision_processing_time,
            "total_processing_time": self.total_processing_time,
            "total_improvements": self.total_improvements,
            "auto_approved": self.auto_approved,
            "auto_rejected": self.auto_rejected,
            "manually_approved": self.manually_approved,
            "manually_rejected": self.manually_rejected,
            "timed_out": self.timed_out,
            "errors": self.errors,
            "items_per_second": self.items_per_second,
            "user_engagement_score": self.user_engagement_score,
            "approval_rate": self.get_approval_rate()
        }


class ApprovalWorkflow:
    """Main workflow orchestrator for approval decisions."""
    
    def __init__(self, 
                 config: ApprovalConfig,
                 criteria_engine: Optional[ApprovalCriteriaEngine] = None,
                 interface: Optional[ApprovalInterface] = None):
        self.config = config
        self.criteria_engine = criteria_engine or ApprovalCriteriaEngine()
        self.interface = interface or create_approval_interface(config)
        
        # Workflow state
        self.state = WorkflowState.INITIALIZING
        self.metrics = WorkflowMetrics()
        self.current_context: Optional[ApprovalContext] = None
        
        # Results
        self.current_batch: Optional[BatchApprovalDecision] = None
        self.approval_history: List[BatchApprovalDecision] = []
        
        # Configuration for this workflow instance
        self.timeout_seconds = config.timeouts.interactive_timeout
        self.max_concurrent_approvals = config.max_concurrent_approvals
        
        # Setup default criteria if engine is empty
        if not self.criteria_engine.filters and not self.criteria_engine.global_criteria:
            self.criteria_engine.create_default_filters()
    
    async def process_improvements(self, 
                                 improvements: List[SelfImprovementAction],
                                 context: Optional[ApprovalContext] = None) -> BatchApprovalDecision:
        """Main entry point for processing a batch of improvements."""
        self.state = WorkflowState.INITIALIZING
        self.metrics = WorkflowMetrics()
        self.metrics.total_improvements = len(improvements)
        
        # Create or use provided context
        if context is None:
            context = ApprovalContext()
            for improvement in improvements:
                context.add_improvement(improvement)
        
        self.current_context = context
        
        try:
            logger.info(f"Starting approval workflow for {len(improvements)} improvements")
            
            # Step 1: Evaluate criteria for filtering and pre-processing
            self.state = WorkflowState.EVALUATING_CRITERIA
            criteria_start = time.time()
            
            categorized_improvements = await self._evaluate_criteria(improvements, context)
            
            self.metrics.criteria_evaluation_time = time.time() - criteria_start
            
            # Step 2: Process improvements based on criteria results
            self.state = WorkflowState.PROCESSING_DECISIONS
            decision_start = time.time()
            
            batch_decision = await self._process_categorized_improvements(
                categorized_improvements, context
            )
            
            self.metrics.decision_processing_time = time.time() - decision_start
            
            # Step 3: Finalize workflow
            self.state = WorkflowState.FINALIZING
            self.current_batch = batch_decision
            self.approval_history.append(batch_decision)
            
            # Update metrics
            self._update_metrics_from_batch(batch_decision)
            self.metrics.finalize()
            
            self.state = WorkflowState.COMPLETED
            logger.info(f"Approval workflow completed: {batch_decision.approved_count} approved, "
                       f"{batch_decision.rejected_count} rejected")
            
            # Display summary via interface
            self.interface.display_summary(batch_decision)
            
            return batch_decision
            
        except asyncio.TimeoutError:
            logger.error("Approval workflow timed out")
            self.state = WorkflowState.FAILED
            self.metrics.errors += 1
            
            # Return partial results if available
            if self.current_batch:
                return self.current_batch
            
            # Create emergency batch with all rejections
            return self._create_emergency_batch(improvements, context, "Workflow timeout")
            
        except Exception as e:
            logger.error(f"Approval workflow failed: {e}", exc_info=True)
            self.state = WorkflowState.FAILED
            self.metrics.errors += 1
            
            return self._create_emergency_batch(improvements, context, f"Workflow error: {e}")
    
    async def _evaluate_criteria(self, 
                               improvements: List[SelfImprovementAction],
                               context: ApprovalContext) -> Dict[str, List[SelfImprovementAction]]:
        """Evaluate criteria and categorize improvements."""
        categorized = {
            "auto_approve": [],
            "auto_reject": [],
            "manual_review": [],
            "flagged": []
        }
        
        for improvement in improvements:
            try:
                # Get criteria evaluation
                evaluation = self.criteria_engine.evaluate_improvement(improvement, {
                    "context": context,
                    "workflow_config": self.config
                })
                
                recommended_action = evaluation.get("recommended_action")
                confidence = evaluation.get("confidence", 1.0)
                
                # Categorize based on recommendation and confidence
                if recommended_action == FilterAction.AUTO_APPROVE and confidence >= 0.8:
                    categorized["auto_approve"].append(improvement)
                elif recommended_action == FilterAction.AUTO_REJECT and confidence >= 0.8:
                    categorized["auto_reject"].append(improvement)
                elif recommended_action in [FilterAction.FLAG_FOR_REVIEW, FilterAction.ESCALATE]:
                    categorized["flagged"].append(improvement)
                else:
                    categorized["manual_review"].append(improvement)
                
                # Store evaluation results in improvement metadata for later use
                if not hasattr(improvement, '_approval_metadata'):
                    improvement._approval_metadata = {}
                improvement._approval_metadata['criteria_evaluation'] = evaluation
                
            except Exception as e:
                logger.warning(f"Failed to evaluate criteria for improvement {improvement.action_id}: {e}")
                categorized["manual_review"].append(improvement)
        
        logger.info(f"Criteria evaluation complete: "
                   f"{len(categorized['auto_approve'])} auto-approve, "
                   f"{len(categorized['auto_reject'])} auto-reject, "
                   f"{len(categorized['manual_review'])} manual review, "
                   f"{len(categorized['flagged'])} flagged")
        
        return categorized
    
    async def _process_categorized_improvements(self,
                                             categorized: Dict[str, List[SelfImprovementAction]],
                                             context: ApprovalContext) -> BatchApprovalDecision:
        """Process improvements based on their categorization."""
        batch = BatchApprovalDecision(
            session_id=context.session_id,
            batch_action=self.config.approval_mode,
            approved_by="workflow"
        )
        
        # Process auto-approvals
        for improvement in categorized["auto_approve"]:
            decision = ApprovalDecision(
                improvement_id=improvement.action_id,
                approval_mode_used=ApprovalMode.AUTO,
                interface_type_used=ApprovalInterfaceType.AUTO,
                session_id=context.session_id
            )
            
            evaluation = getattr(improvement, '_approval_metadata', {}).get('criteria_evaluation', {})
            reasoning = "; ".join(evaluation.get('reasoning', ['Auto-approved by criteria']))
            
            decision.approve("system", f"Criteria-based approval: {reasoning}")
            batch.add_decision(decision)
            self.metrics.auto_approved += 1
        
        # Process auto-rejections
        for improvement in categorized["auto_reject"]:
            decision = ApprovalDecision(
                improvement_id=improvement.action_id,
                approval_mode_used=ApprovalMode.AUTO,
                interface_type_used=ApprovalInterfaceType.AUTO,
                session_id=context.session_id
            )
            
            evaluation = getattr(improvement, '_approval_metadata', {}).get('criteria_evaluation', {})
            reasoning = "; ".join(evaluation.get('reasoning', ['Auto-rejected by criteria']))
            
            decision.reject("system", RejectionReason.POLICY_VIOLATION, f"Criteria-based rejection: {reasoning}")
            batch.add_decision(decision)
            self.metrics.auto_rejected += 1
        
        # Process manual reviews and flagged items
        manual_items = categorized["manual_review"] + categorized["flagged"]
        
        if manual_items and self.config.approval_mode != ApprovalMode.AUTO:
            user_start = time.time()
            
            try:
                # Set timeout for user interaction
                timeout = self.config.get_timeout_for_context(len(manual_items))
                
                self.state = WorkflowState.REQUESTING_USER_INPUT
                self.interface.display_progress("Requesting user approval...", 0.0)
                
                # Request user approval with timeout
                user_batch = await asyncio.wait_for(
                    self.interface.request_approval(manual_items, context),
                    timeout=timeout
                )
                
                # Merge user decisions into main batch
                for decision in user_batch.individual_decisions:
                    batch.add_decision(decision)
                    
                    if decision.status == ApprovalStatus.APPROVED:
                        self.metrics.manually_approved += 1
                    elif decision.status == ApprovalStatus.REJECTED:
                        self.metrics.manually_rejected += 1
                    else:
                        self.metrics.timed_out += 1
                
                self.metrics.user_interaction_time = time.time() - user_start
                
            except asyncio.TimeoutError:
                logger.warning(f"User interaction timed out after {timeout} seconds")
                self.metrics.user_interaction_time = time.time() - user_start
                
                # Handle timeout by rejecting pending items
                for improvement in manual_items:
                    decision = ApprovalDecision(
                        improvement_id=improvement.action_id,
                        approval_mode_used=self.config.approval_mode,
                        interface_type_used=self.config.interface_type,
                        session_id=context.session_id
                    )
                    decision.reject("system", RejectionReason.TIMING_ISSUES, "User interaction timeout")
                    batch.add_decision(decision)
                    self.metrics.timed_out += 1
        
        else:
            # Auto-reject manual items if in pure auto mode
            for improvement in manual_items:
                decision = ApprovalDecision(
                    improvement_id=improvement.action_id,
                    approval_mode_used=ApprovalMode.AUTO,
                    interface_type_used=ApprovalInterfaceType.AUTO,
                    session_id=context.session_id
                )
                decision.reject("system", RejectionReason.POLICY_VIOLATION, "Auto mode - manual review required")
                batch.add_decision(decision)
                self.metrics.auto_rejected += 1
        
        batch.complete_batch("workflow")
        return batch
    
    def _update_metrics_from_batch(self, batch: BatchApprovalDecision) -> None:
        """Update metrics from batch decision results."""
        # Metrics are updated during processing, but we can validate here
        total_from_batch = batch.approved_count + batch.rejected_count
        total_from_metrics = (self.metrics.auto_approved + self.metrics.auto_rejected +
                            self.metrics.manually_approved + self.metrics.manually_rejected +
                            self.metrics.timed_out)
        
        if total_from_batch != total_from_metrics:
            logger.warning(f"Metric mismatch: batch={total_from_batch}, metrics={total_from_metrics}")
    
    def _create_emergency_batch(self, 
                              improvements: List[SelfImprovementAction],
                              context: ApprovalContext,
                              reason: str) -> BatchApprovalDecision:
        """Create emergency batch when workflow fails."""
        batch = BatchApprovalDecision(
            session_id=context.session_id,
            batch_action=ApprovalMode.BATCH_REJECT,
            approved_by="emergency_system",
            batch_notes=f"Emergency batch due to workflow failure: {reason}"
        )
        
        for improvement in improvements:
            decision = ApprovalDecision(
                improvement_id=improvement.action_id,
                approval_mode_used=ApprovalMode.AUTO,
                interface_type_used=ApprovalInterfaceType.AUTO,
                session_id=context.session_id
            )
            decision.reject("emergency_system", RejectionReason.OTHER, reason)
            batch.add_decision(decision)
        
        batch.complete_batch("emergency_system")
        return batch
    
    async def process_single_improvement(self, 
                                       improvement: SelfImprovementAction,
                                       context: Optional[Dict[str, Any]] = None) -> ApprovalDecision:
        """Process a single improvement for approval."""
        workflow_context = ApprovalContext()
        workflow_context.add_improvement(improvement)
        
        # Evaluate criteria
        evaluation = self.criteria_engine.evaluate_improvement(improvement, context or {})
        recommended_action = evaluation.get("recommended_action")
        confidence = evaluation.get("confidence", 1.0)
        
        decision = ApprovalDecision(
            improvement_id=improvement.action_id,
            approval_mode_used=self.config.approval_mode,
            interface_type_used=self.config.interface_type,
            session_id=workflow_context.session_id
        )
        
        # Handle based on recommendation and confidence
        if recommended_action == FilterAction.AUTO_APPROVE and confidence >= 0.8:
            reasoning = "; ".join(evaluation.get('reasoning', ['Auto-approved by criteria']))
            decision.approve("system", f"Criteria-based approval: {reasoning}")
            
        elif recommended_action == FilterAction.AUTO_REJECT and confidence >= 0.8:
            reasoning = "; ".join(evaluation.get('reasoning', ['Auto-rejected by criteria']))
            decision.reject("system", RejectionReason.POLICY_VIOLATION, f"Criteria-based rejection: {reasoning}")
            
        elif self.config.approval_mode == ApprovalMode.AUTO:
            # Default auto approval if no strong recommendation
            decision.approve("system", "Auto-approved (default)")
            
        else:
            # Request user input
            prompt_data = ApprovalPromptData(
                improvement=improvement,
                impact_estimate=evaluation.get('metadata', {}).get('impact_estimate'),
                risk_analysis=evaluation.get('metadata', {}).get('risk_analysis')
            )
            
            try:
                timeout = self.config.timeouts.individual_item_timeout
                decision = await asyncio.wait_for(
                    self.interface.request_single_approval(improvement, prompt_data, timeout),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                decision.reject("system", RejectionReason.TIMING_ISSUES, "Single approval timeout")
        
        return decision
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get comprehensive workflow summary."""
        return {
            "state": self.state.value,
            "metrics": self.metrics.to_dict(),
            "config": {
                "approval_mode": self.config.approval_mode.value,
                "interface_type": self.config.interface_type.value,
                "timeout_seconds": self.timeout_seconds
            },
            "criteria_engine_stats": self.criteria_engine.get_statistics(),
            "history_count": len(self.approval_history),
            "current_batch": {
                "session_id": self.current_batch.session_id if self.current_batch else None,
                "approved_count": self.current_batch.approved_count if self.current_batch else 0,
                "rejected_count": self.current_batch.rejected_count if self.current_batch else 0
            } if self.current_batch else None
        }
    
    def cancel_workflow(self, reason: str = "User cancellation") -> None:
        """Cancel the current workflow."""
        self.state = WorkflowState.CANCELLED
        logger.info(f"Approval workflow cancelled: {reason}")
        
        if self.current_batch:
            self.current_batch.batch_notes = f"Cancelled: {reason}"
    
    def reset_workflow(self) -> None:
        """Reset workflow for reuse."""
        self.state = WorkflowState.INITIALIZING
        self.metrics = WorkflowMetrics()
        self.current_context = None
        self.current_batch = None
        # Keep approval history for learning
        
        logger.info("Approval workflow reset")


class WorkflowFactory:
    """Factory for creating configured approval workflows."""
    
    @staticmethod
    def create_auto_workflow(config: Optional[ApprovalConfig] = None) -> ApprovalWorkflow:
        """Create workflow configured for automatic approvals."""
        if config is None:
            config = ApprovalConfig()
        
        config.approval_mode = ApprovalMode.AUTO
        config.interface_type = ApprovalInterfaceType.AUTO
        
        return ApprovalWorkflow(config)
    
    @staticmethod
    def create_interactive_workflow(config: Optional[ApprovalConfig] = None) -> ApprovalWorkflow:
        """Create workflow configured for interactive approvals."""
        if config is None:
            config = ApprovalConfig()
        
        config.approval_mode = ApprovalMode.INTERACTIVE
        config.interface_type = ApprovalInterfaceType.TUI
        
        return ApprovalWorkflow(config)
    
    @staticmethod
    def create_cli_workflow(config: Optional[ApprovalConfig] = None) -> ApprovalWorkflow:
        """Create workflow configured for CLI approvals."""
        if config is None:
            config = ApprovalConfig()
        
        config.approval_mode = ApprovalMode.MANUAL
        config.interface_type = ApprovalInterfaceType.CLI
        
        return ApprovalWorkflow(config)
    
    @staticmethod
    def create_custom_workflow(
        approval_mode: ApprovalMode,
        interface_type: ApprovalInterfaceType,
        criteria_engine: Optional[ApprovalCriteriaEngine] = None,
        config: Optional[ApprovalConfig] = None
    ) -> ApprovalWorkflow:
        """Create workflow with custom configuration."""
        if config is None:
            config = ApprovalConfig()
        
        config.approval_mode = approval_mode
        config.interface_type = interface_type
        
        return ApprovalWorkflow(config, criteria_engine)