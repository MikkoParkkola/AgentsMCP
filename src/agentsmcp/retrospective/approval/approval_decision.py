"""
Data structures for approval decisions and outcomes.

This module defines the core data structures used throughout the approval system
to represent approval decisions, outcomes, and related metadata.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from ..data_models import SelfImprovementAction, ImprovementCategory, PriorityLevel


class ApprovalStatus(Enum):
    """Status of an approval decision."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


class ApprovalMode(Enum):
    """Approval mode configuration."""
    AUTO = "auto"  # Auto-approve all improvements
    MANUAL = "manual"  # User reviews each improvement individually
    INTERACTIVE = "interactive"  # User can choose batch operations or individual review
    BATCH_APPROVE = "batch_approve"  # Approve all with confirmation
    BATCH_REJECT = "batch_reject"  # Reject all with confirmation


class ApprovalInterfaceType(Enum):
    """Type of approval interface to use."""
    CLI = "cli"  # Command line prompts for headless environments
    TUI = "tui"  # Rich terminal UI with improvement previews
    AUTO = "auto"  # No user interaction, uses configured defaults


class RejectionReason(Enum):
    """Standard rejection reasons."""
    TOO_RISKY = "too_risky"
    LOW_IMPACT = "low_impact"
    RESOURCE_CONSTRAINTS = "resource_constraints"
    TIMING_ISSUES = "timing_issues"
    USER_PREFERENCE = "user_preference"
    POLICY_VIOLATION = "policy_violation"
    DUPLICATE = "duplicate"
    SUPERSEDED = "superseded"
    OTHER = "other"


@dataclass
class ApprovalDecision:
    """Represents a single approval decision for an improvement."""
    
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    improvement_id: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None  # User ID or system identifier
    decision_timestamp: Optional[datetime] = None
    
    # Decision rationale
    rejection_reason: Optional[RejectionReason] = None
    user_notes: str = ""
    confidence_level: float = 1.0  # 0.0 to 1.0
    
    # Context information
    approval_mode_used: Optional[ApprovalMode] = None
    interface_type_used: Optional[ApprovalInterfaceType] = None
    session_id: Optional[str] = None
    
    # Timing information
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def approve(self, approved_by: str, notes: str = "") -> None:
        """Mark decision as approved."""
        self.status = ApprovalStatus.APPROVED
        self.approved_by = approved_by
        self.user_notes = notes
        self.decision_timestamp = datetime.now(timezone.utc)
    
    def reject(self, rejected_by: str, reason: RejectionReason, notes: str = "") -> None:
        """Mark decision as rejected."""
        self.status = ApprovalStatus.REJECTED
        self.approved_by = rejected_by
        self.rejection_reason = reason
        self.user_notes = notes
        self.decision_timestamp = datetime.now(timezone.utc)
    
    def is_expired(self) -> bool:
        """Check if the decision has expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def mark_expired(self) -> None:
        """Mark the decision as expired."""
        if self.status == ApprovalStatus.PENDING:
            self.status = ApprovalStatus.EXPIRED
            self.decision_timestamp = datetime.now(timezone.utc)


@dataclass
class BatchApprovalDecision:
    """Represents a batch approval decision for multiple improvements."""
    
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    improvement_ids: List[str] = field(default_factory=list)
    
    # Batch decision
    batch_action: ApprovalMode = ApprovalMode.INTERACTIVE
    approved_count: int = 0
    rejected_count: int = 0
    
    # Decision details
    individual_decisions: List[ApprovalDecision] = field(default_factory=list)
    batch_notes: str = ""
    approved_by: Optional[str] = None
    
    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    # Context
    approval_criteria_applied: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_decision(self, decision: ApprovalDecision) -> None:
        """Add an individual decision to the batch."""
        self.individual_decisions.append(decision)
        self.improvement_ids.append(decision.improvement_id)
        
        if decision.status == ApprovalStatus.APPROVED:
            self.approved_count += 1
        elif decision.status == ApprovalStatus.REJECTED:
            self.rejected_count += 1
    
    def complete_batch(self, completed_by: str) -> None:
        """Mark the batch as completed."""
        self.completed_at = datetime.now(timezone.utc)
        self.approved_by = completed_by
    
    def get_approval_rate(self) -> float:
        """Get the approval rate for this batch."""
        total = len(self.individual_decisions)
        if total == 0:
            return 0.0
        return self.approved_count / total


@dataclass
class ApprovalHistory:
    """Historical record of approval decisions."""
    
    user_id: str = ""
    session_history: List[BatchApprovalDecision] = field(default_factory=list)
    
    # Statistics
    total_approvals: int = 0
    total_rejections: int = 0
    most_common_rejection_reasons: List[RejectionReason] = field(default_factory=list)
    
    # Preferences learned from history
    auto_approve_categories: Set[ImprovementCategory] = field(default_factory=set)
    auto_reject_categories: Set[ImprovementCategory] = field(default_factory=set)
    preferred_approval_mode: Optional[ApprovalMode] = None
    preferred_interface_type: Optional[ApprovalInterfaceType] = None
    
    # Timing preferences
    typical_session_duration: Optional[float] = None  # seconds
    preferred_batch_size: Optional[int] = None
    
    def add_batch_decision(self, batch: BatchApprovalDecision) -> None:
        """Add a batch decision to history."""
        self.session_history.append(batch)
        self.total_approvals += batch.approved_count
        self.total_rejections += batch.rejected_count
        
        # Update statistics
        self._update_preferences()
    
    def get_approval_rate(self) -> float:
        """Get overall approval rate."""
        total = self.total_approvals + self.total_rejections
        if total == 0:
            return 0.0
        return self.total_approvals / total
    
    def get_category_preference(self, category: ImprovementCategory) -> Optional[ApprovalStatus]:
        """Get learned preference for a specific category."""
        if category in self.auto_approve_categories:
            return ApprovalStatus.APPROVED
        elif category in self.auto_reject_categories:
            return ApprovalStatus.REJECTED
        return None
    
    def _update_preferences(self) -> None:
        """Update learned preferences based on history."""
        if not self.session_history:
            return
        
        # Analyze category preferences
        category_stats: Dict[ImprovementCategory, Dict[str, int]] = {}
        
        for batch in self.session_history[-10:]:  # Look at last 10 batches
            for decision in batch.individual_decisions:
                # We would need to access the improvement to get its category
                # This is a simplified implementation
                pass
        
        # Update preferred modes based on recent usage
        recent_modes = [batch.batch_action for batch in self.session_history[-5:]]
        if recent_modes:
            self.preferred_approval_mode = max(set(recent_modes), key=recent_modes.count)


@dataclass
class ApprovalContext:
    """Context information for approval decisions."""
    
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    
    # Improvements being considered
    improvements: List[SelfImprovementAction] = field(default_factory=list)
    improvement_count: int = 0
    
    # Categorization
    categories_present: Set[ImprovementCategory] = field(default_factory=set)
    priority_levels_present: Set[PriorityLevel] = field(default_factory=set)
    
    # Timing constraints
    timeout_seconds: Optional[int] = None
    deadline: Optional[datetime] = None
    
    # System state
    system_load: float = 0.0  # 0.0 to 1.0
    safety_level_required: str = "standard"
    
    # Context metadata
    retrospective_id: Optional[str] = None
    triggering_event: Optional[str] = None
    environment: str = "production"
    
    def add_improvement(self, improvement: SelfImprovementAction) -> None:
        """Add an improvement to the context."""
        self.improvements.append(improvement)
        self.improvement_count += 1
        self.categories_present.add(improvement.category)
        self.priority_levels_present.add(improvement.priority)
    
    def get_complexity_score(self) -> float:
        """Calculate a complexity score for this approval context."""
        score = 0.0
        
        # Base complexity from count
        score += min(self.improvement_count * 0.1, 1.0)
        
        # Category diversity
        score += len(self.categories_present) * 0.1
        
        # Priority spread
        score += len(self.priority_levels_present) * 0.05
        
        # System constraints
        if self.timeout_seconds and self.timeout_seconds < 300:  # 5 minutes
            score += 0.2
        
        return min(score, 1.0)