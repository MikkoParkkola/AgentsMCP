"""
Approval orchestrator that coordinates the approval process with the safety framework.

This module provides the main orchestrator that integrates the approval system
with the existing retrospective safety framework and manages the complete flow
from improvement generation to safety validation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
import logging

from .approval_config import ApprovalConfig, load_approval_config
from .approval_decision import (
    ApprovalDecision, ApprovalStatus, ApprovalMode, ApprovalInterfaceType,
    ApprovalContext, BatchApprovalDecision, ApprovalHistory
)
from .approval_workflow import ApprovalWorkflow, WorkflowFactory, WorkflowState
from .approval_criteria import ApprovalCriteriaEngine
from ..data_models import SelfImprovementAction, ComprehensiveRetrospectiveReport
from ..safety.safety_orchestrator import SafetyOrchestrator
from ..safety.safety_config import SafetyConfig

logger = logging.getLogger(__name__)


class OrchestrationPhase(Enum):
    """Phases of the approval orchestration process."""
    INITIALIZATION = "initialization"
    PRE_APPROVAL_ANALYSIS = "pre_approval_analysis"
    APPROVAL_WORKFLOW = "approval_workflow"
    POST_APPROVAL_PROCESSING = "post_approval_processing"
    SAFETY_INTEGRATION = "safety_integration"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OrchestrationResult:
    """Result of the complete approval orchestration."""
    
    # Approval results
    batch_decision: BatchApprovalDecision
    approved_improvements: List[SelfImprovementAction] = field(default_factory=list)
    rejected_improvements: List[SelfImprovementAction] = field(default_factory=list)
    
    # Safety integration results
    safety_validated_improvements: List[SelfImprovementAction] = field(default_factory=list)
    safety_rejected_improvements: List[SelfImprovementAction] = field(default_factory=list)
    
    # Orchestration metadata
    phase: OrchestrationPhase = OrchestrationPhase.COMPLETED
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    total_processing_time: float = 0.0
    
    # Quality metrics
    approval_accuracy: float = 0.0  # How well approvals aligned with safety validation
    safety_compliance_rate: float = 0.0
    user_satisfaction_score: Optional[float] = None
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def finalize(self) -> None:
        """Finalize the orchestration result."""
        self.end_time = datetime.now(timezone.utc)
        self.total_processing_time = (self.end_time - self.start_time).total_seconds()
        
        # Calculate quality metrics
        total_approved = len(self.approved_improvements)
        if total_approved > 0:
            self.safety_compliance_rate = len(self.safety_validated_improvements) / total_approved
        
        # Calculate approval accuracy (approved items that passed safety)
        if total_approved > 0:
            self.approval_accuracy = len(self.safety_validated_improvements) / total_approved
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "phase": self.phase.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_processing_time": self.total_processing_time,
            "approved_count": len(self.approved_improvements),
            "rejected_count": len(self.rejected_improvements),
            "safety_validated_count": len(self.safety_validated_improvements),
            "safety_rejected_count": len(self.safety_rejected_improvements),
            "approval_accuracy": self.approval_accuracy,
            "safety_compliance_rate": self.safety_compliance_rate,
            "user_satisfaction_score": self.user_satisfaction_score,
            "errors": self.errors,
            "warnings": self.warnings,
            "batch_decision_id": self.batch_decision.batch_id
        }


class ApprovalOrchestrator:
    """Main orchestrator for the complete approval and safety validation process."""
    
    def __init__(self,
                 approval_config: Optional[ApprovalConfig] = None,
                 safety_config: Optional[SafetyConfig] = None,
                 approval_history: Optional[ApprovalHistory] = None):
        
        # Load configurations
        self.approval_config = approval_config or load_approval_config()
        self.safety_config = safety_config or SafetyConfig()
        self.approval_history = approval_history or ApprovalHistory()
        
        # Initialize components
        self.criteria_engine = ApprovalCriteriaEngine()
        self.workflow: Optional[ApprovalWorkflow] = None
        self.safety_orchestrator: Optional[SafetyOrchestrator] = None
        
        # State tracking
        self.current_phase = OrchestrationPhase.INITIALIZATION
        self.active_sessions: Set[str] = set()
        self.orchestration_history: List[OrchestrationResult] = []
        
        # Performance tracking
        self._metrics = {
            "total_orchestrations": 0,
            "successful_orchestrations": 0,
            "failed_orchestrations": 0,
            "average_processing_time": 0.0,
            "total_improvements_processed": 0,
            "total_approved": 0,
            "total_safety_validated": 0
        }
        
        # Setup components
        self._setup_criteria_engine()
        self._setup_safety_integration()
    
    def _setup_criteria_engine(self) -> None:
        """Setup the criteria engine with default filters."""
        self.criteria_engine.create_default_filters()
        
        # Add orchestrator-specific criteria if needed
        # This could be enhanced based on historical approval patterns
        if self.approval_history and len(self.approval_history.session_history) > 0:
            self._apply_learned_criteria()
    
    def _apply_learned_criteria(self) -> None:
        """Apply criteria learned from approval history."""
        # Analyze historical patterns
        for category in self.approval_history.auto_approve_categories:
            logger.info(f"Learned preference: auto-approve {category.value} improvements")
        
        for category in self.approval_history.auto_reject_categories:
            logger.info(f"Learned preference: auto-reject {category.value} improvements")
        
        # This could be enhanced to create custom criteria based on patterns
    
    def _setup_safety_integration(self) -> None:
        """Setup safety framework integration."""
        if self.approval_config.require_safety_validation:
            self.safety_orchestrator = SafetyOrchestrator(self.safety_config)
        else:
            logger.info("Safety validation disabled in approval configuration")
    
    async def orchestrate_approval_process(self,
                                         improvements: List[SelfImprovementAction],
                                         context: Optional[Dict[str, Any]] = None) -> OrchestrationResult:
        """Main orchestration method for the complete approval process."""
        result = OrchestrationResult(
            batch_decision=BatchApprovalDecision(),  # Will be replaced
            phase=OrchestrationPhase.INITIALIZATION
        )
        
        try:
            self._metrics["total_orchestrations"] += 1
            self._metrics["total_improvements_processed"] += len(improvements)
            
            logger.info(f"Starting approval orchestration for {len(improvements)} improvements")
            
            # Phase 1: Initialization and pre-analysis
            result.phase = OrchestrationPhase.PRE_APPROVAL_ANALYSIS
            self.current_phase = result.phase
            
            await self._pre_approval_analysis(improvements, context, result)
            
            # Phase 2: Run approval workflow
            result.phase = OrchestrationPhase.APPROVAL_WORKFLOW
            self.current_phase = result.phase
            
            approval_context = await self._create_approval_context(improvements, context)
            batch_decision = await self._execute_approval_workflow(improvements, approval_context, result)
            result.batch_decision = batch_decision
            
            # Phase 3: Post-approval processing
            result.phase = OrchestrationPhase.POST_APPROVAL_PROCESSING
            self.current_phase = result.phase
            
            await self._post_approval_processing(batch_decision, result)
            
            # Phase 4: Safety integration (if enabled)
            if self.approval_config.require_safety_validation and self.safety_orchestrator:
                result.phase = OrchestrationPhase.SAFETY_INTEGRATION
                self.current_phase = result.phase
                
                await self._integrate_with_safety_framework(result)
            else:
                logger.info("Skipping safety integration (disabled)")
                # All approved items are considered safety validated
                result.safety_validated_improvements = result.approved_improvements.copy()
            
            # Phase 5: Finalization
            result.phase = OrchestrationPhase.FINALIZATION
            self.current_phase = result.phase
            
            await self._finalize_orchestration(result)
            
            result.phase = OrchestrationPhase.COMPLETED
            self.current_phase = result.phase
            
            result.finalize()
            self.orchestration_history.append(result)
            
            self._metrics["successful_orchestrations"] += 1
            self._metrics["total_approved"] += len(result.approved_improvements)
            self._metrics["total_safety_validated"] += len(result.safety_validated_improvements)
            
            logger.info(f"Approval orchestration completed successfully: "
                       f"{len(result.safety_validated_improvements)} improvements ready for implementation")
            
            return result
            
        except Exception as e:
            logger.error(f"Approval orchestration failed: {e}", exc_info=True)
            result.phase = OrchestrationPhase.FAILED
            result.errors.append(f"Orchestration failed: {e}")
            result.finalize()
            
            self._metrics["failed_orchestrations"] += 1
            
            return result
    
    async def _pre_approval_analysis(self,
                                   improvements: List[SelfImprovementAction],
                                   context: Optional[Dict[str, Any]],
                                   result: OrchestrationResult) -> None:
        """Perform pre-approval analysis and preparation."""
        logger.info("Performing pre-approval analysis")
        
        # Analyze improvement complexity and relationships
        complexity_scores = []
        for improvement in improvements:
            # Simple complexity analysis based on content length and category
            complexity = len(improvement.description or "") / 100.0  # Normalize by description length
            if improvement.category.value == "security":
                complexity *= 1.5  # Security changes are more complex
            elif improvement.category.value == "reliability":
                complexity *= 1.2
            
            complexity_scores.append(min(complexity, 1.0))
        
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0.0
        
        # Adjust timeout based on complexity
        if avg_complexity > 0.7:
            self.approval_config.timeouts.interactive_timeout = int(
                self.approval_config.timeouts.interactive_timeout * 1.5
            )
            result.warnings.append(f"Increased timeout due to high complexity ({avg_complexity:.2f})")
        
        # Check for duplicate or similar improvements
        similar_pairs = self._find_similar_improvements(improvements)
        if similar_pairs:
            result.warnings.append(f"Found {len(similar_pairs)} pairs of potentially similar improvements")
        
        logger.info(f"Pre-approval analysis complete: avg complexity {avg_complexity:.2f}")
    
    def _find_similar_improvements(self, 
                                 improvements: List[SelfImprovementAction]) -> List[tuple]:
        """Find potentially similar improvements."""
        similar_pairs = []
        
        for i, imp1 in enumerate(improvements):
            for j, imp2 in enumerate(improvements[i+1:], i+1):
                # Simple similarity check based on title and category
                if (imp1.category == imp2.category and 
                    self._similarity_score(imp1.title, imp2.title) > 0.7):
                    similar_pairs.append((i, j))
        
        return similar_pairs
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two texts."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _create_approval_context(self,
                                     improvements: List[SelfImprovementAction],
                                     context: Optional[Dict[str, Any]]) -> ApprovalContext:
        """Create approval context for the workflow."""
        approval_context = ApprovalContext()
        
        for improvement in improvements:
            approval_context.add_improvement(improvement)
        
        # Add system state information
        approval_context.system_load = context.get("system_load", 0.0) if context else 0.0
        approval_context.environment = context.get("environment", "production") if context else "production"
        
        # Set timeout based on configuration and context
        approval_context.timeout_seconds = self.approval_config.get_timeout_for_context(
            len(improvements)
        )
        
        return approval_context
    
    async def _execute_approval_workflow(self,
                                       improvements: List[SelfImprovementAction],
                                       approval_context: ApprovalContext,
                                       result: OrchestrationResult) -> BatchApprovalDecision:
        """Execute the approval workflow."""
        logger.info("Executing approval workflow")
        
        # Create workflow based on configuration
        self.workflow = ApprovalWorkflow(
            config=self.approval_config,
            criteria_engine=self.criteria_engine
        )
        
        # Track active session
        self.active_sessions.add(approval_context.session_id)
        
        try:
            batch_decision = await self.workflow.process_improvements(
                improvements, approval_context
            )
            
            logger.info(f"Approval workflow completed: {batch_decision.approved_count} approved, "
                       f"{batch_decision.rejected_count} rejected")
            
            return batch_decision
            
        finally:
            # Clean up active session
            self.active_sessions.discard(approval_context.session_id)
    
    async def _post_approval_processing(self,
                                      batch_decision: BatchApprovalDecision,
                                      result: OrchestrationResult) -> None:
        """Process approval results and categorize improvements."""
        logger.info("Performing post-approval processing")
        
        # Categorize improvements based on decisions
        approved_ids = set()
        rejected_ids = set()
        
        for decision in batch_decision.individual_decisions:
            if decision.status == ApprovalStatus.APPROVED:
                approved_ids.add(decision.improvement_id)
            else:
                rejected_ids.add(decision.improvement_id)
        
        # Find corresponding improvements
        improvement_map = {imp.action_id: imp for imp in 
                          (result.approved_improvements + result.rejected_improvements)}
        
        # If not yet populated, get from batch context
        if not improvement_map and hasattr(batch_decision, '_source_improvements'):
            improvement_map = {imp.action_id: imp for imp in batch_decision._source_improvements}
        
        # Populate result lists
        for decision in batch_decision.individual_decisions:
            improvement_id = decision.improvement_id
            # In a real implementation, we'd need to map back to the original improvements
            # For now, we'll create placeholder improvements or need to pass them through
            pass
        
        # Update approval history
        self.approval_history.add_batch_decision(batch_decision)
        
        logger.info(f"Post-approval processing complete: "
                   f"{len(approved_ids)} approved, {len(rejected_ids)} rejected")
    
    async def _integrate_with_safety_framework(self, result: OrchestrationResult) -> None:
        """Integrate approved improvements with the safety framework."""
        if not self.safety_orchestrator:
            logger.warning("Safety orchestrator not available")
            return
        
        logger.info(f"Integrating {len(result.approved_improvements)} approved improvements with safety framework")
        
        try:
            # For each approved improvement, validate with safety framework
            for improvement in result.approved_improvements:
                try:
                    # This would use the actual safety validation logic
                    # For now, we'll simulate the integration
                    safety_result = await self._validate_improvement_safety(improvement)
                    
                    if safety_result["passed"]:
                        result.safety_validated_improvements.append(improvement)
                        logger.debug(f"Safety validation passed for: {improvement.title}")
                    else:
                        result.safety_rejected_improvements.append(improvement)
                        result.warnings.append(f"Safety validation failed for: {improvement.title} - {safety_result.get('reason', 'Unknown')}")
                        
                except Exception as e:
                    logger.error(f"Safety validation error for {improvement.action_id}: {e}")
                    result.errors.append(f"Safety validation error: {e}")
                    result.safety_rejected_improvements.append(improvement)
            
            logger.info(f"Safety integration complete: "
                       f"{len(result.safety_validated_improvements)} validated, "
                       f"{len(result.safety_rejected_improvements)} rejected by safety framework")
        
        except Exception as e:
            logger.error(f"Safety framework integration failed: {e}")
            result.errors.append(f"Safety integration failed: {e}")
            # In case of failure, don't validate any improvements
            result.safety_rejected_improvements.extend(result.approved_improvements)
            result.safety_validated_improvements.clear()
    
    async def _validate_improvement_safety(self, improvement: SelfImprovementAction) -> Dict[str, Any]:
        """Validate a single improvement with the safety framework."""
        # This is a simplified implementation
        # In practice, this would use the actual safety orchestrator
        
        safety_score = 0.8  # Base safety score
        
        # Reduce safety score for high-risk categories
        if improvement.category.value == "security":
            safety_score -= 0.3
        elif improvement.category.value == "reliability":
            safety_score -= 0.2
        
        # Check for risky keywords
        risky_keywords = ["delete", "remove", "disable", "experimental", "breaking"]
        content = f"{improvement.title} {improvement.description}".lower()
        
        for keyword in risky_keywords:
            if keyword in content:
                safety_score -= 0.2
                break
        
        passed = safety_score >= self.safety_config.thresholds.min_safety_score if hasattr(self.safety_config, 'thresholds') else safety_score >= 0.6
        
        return {
            "passed": passed,
            "safety_score": safety_score,
            "reason": "Automated safety validation" if passed else "Failed safety threshold"
        }
    
    async def _finalize_orchestration(self, result: OrchestrationResult) -> None:
        """Finalize the orchestration process."""
        logger.info("Finalizing approval orchestration")
        
        # Update performance metrics
        if len(self.orchestration_history) > 0:
            total_time = sum(r.total_processing_time for r in self.orchestration_history)
            self._metrics["average_processing_time"] = total_time / len(self.orchestration_history)
        
        # Save configuration if auto-save is enabled
        if self.approval_config.auto_save_config:
            self.approval_config.save_to_file()
        
        # Log final statistics
        logger.info(f"Orchestration finalized: {len(result.safety_validated_improvements)} improvements ready for implementation")
        
        # Generate recommendations for future orchestrations
        await self._generate_improvement_recommendations(result)
    
    async def _generate_improvement_recommendations(self, result: OrchestrationResult) -> None:
        """Generate recommendations for improving the approval process."""
        recommendations = []
        
        # Analyze approval accuracy
        if result.approval_accuracy < 0.8:
            recommendations.append("Consider refining approval criteria to improve safety compliance")
        
        # Analyze processing time
        if result.total_processing_time > 600:  # 10 minutes
            recommendations.append("Consider enabling more auto-approval criteria to reduce processing time")
        
        # Analyze user engagement
        if result.batch_decision and len(result.batch_decision.individual_decisions) > 0:
            manual_decisions = sum(1 for d in result.batch_decision.individual_decisions 
                                 if d.approval_mode_used != ApprovalMode.AUTO)
            engagement_rate = manual_decisions / len(result.batch_decision.individual_decisions)
            
            if engagement_rate > 0.8:
                recommendations.append("High manual review rate - consider expanding auto-approval criteria")
            elif engagement_rate < 0.2:
                recommendations.append("Low manual review rate - ensure important decisions still get human oversight")
        
        if recommendations:
            logger.info(f"Orchestration recommendations: {'; '.join(recommendations)}")
    
    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration statistics."""
        return {
            "metrics": self._metrics.copy(),
            "active_sessions": len(self.active_sessions),
            "orchestration_history_count": len(self.orchestration_history),
            "current_phase": self.current_phase.value,
            "approval_config": {
                "mode": self.approval_config.approval_mode.value,
                "interface": self.approval_config.interface_type.value,
                "safety_validation_enabled": self.approval_config.require_safety_validation
            },
            "recent_results": [
                {
                    "start_time": r.start_time.isoformat(),
                    "processing_time": r.total_processing_time,
                    "approved_count": len(r.approved_improvements),
                    "safety_validated_count": len(r.safety_validated_improvements),
                    "approval_accuracy": r.approval_accuracy
                }
                for r in self.orchestration_history[-5:]  # Last 5 orchestrations
            ]
        }
    
    async def shutdown(self) -> None:
        """Shutdown the orchestrator gracefully."""
        logger.info("Shutting down approval orchestrator")
        
        # Cancel any active workflows
        if self.workflow and self.workflow.state not in [WorkflowState.COMPLETED, WorkflowState.FAILED]:
            self.workflow.cancel_workflow("Orchestrator shutdown")
        
        # Save final configuration
        if self.approval_config.auto_save_config:
            self.approval_config.save_to_file()
        
        # Clear active sessions
        self.active_sessions.clear()
        
        logger.info("Approval orchestrator shutdown complete")


# Factory functions for common orchestrator configurations

def create_production_orchestrator(safety_validation: bool = True) -> ApprovalOrchestrator:
    """Create orchestrator configured for production use."""
    config = load_approval_config()
    config.approval_mode = ApprovalMode.INTERACTIVE
    config.interface_type = ApprovalInterfaceType.TUI
    config.require_safety_validation = safety_validation
    
    return ApprovalOrchestrator(approval_config=config)


def create_development_orchestrator() -> ApprovalOrchestrator:
    """Create orchestrator configured for development use."""
    config = load_approval_config()
    config.approval_mode = ApprovalMode.AUTO
    config.interface_type = ApprovalInterfaceType.AUTO
    config.require_safety_validation = False
    
    return ApprovalOrchestrator(approval_config=config)


def create_testing_orchestrator() -> ApprovalOrchestrator:
    """Create orchestrator configured for testing."""
    config = ApprovalConfig()  # Use defaults
    config.approval_mode = ApprovalMode.AUTO
    config.interface_type = ApprovalInterfaceType.AUTO
    config.require_safety_validation = True  # Test safety integration
    
    return ApprovalOrchestrator(approval_config=config)