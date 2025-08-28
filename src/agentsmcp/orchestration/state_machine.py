"""
Task state management for orchestrated execution.

Manages task lifecycle states and transitions following AGENTS.md v2 patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from ..models import TaskEnvelopeV1, EnvelopeStatus


@dataclass
class TaskState:
    """Represents the state of a task in the coordination system."""
    task_id: str
    envelope: TaskEnvelopeV1
    status: EnvelopeStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_role: Optional[str] = None
    assigned_agent: Optional[str] = None
    retry_count: int = 0
    error_message: Optional[str] = None


class TaskStateMachine:
    """
    Manages task state transitions and lifecycle.
    
    Ensures proper state transitions and maintains task history
    for debugging and monitoring.
    """
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        EnvelopeStatus.PENDING: [EnvelopeStatus.PENDING],  # Can stay pending
        EnvelopeStatus.SUCCESS: [],  # Terminal state
        EnvelopeStatus.ERROR: [EnvelopeStatus.PENDING],  # Can retry
    }
    
    def __init__(self):
        """Initialize state machine."""
        self.transition_history = {}
    
    async def transition(
        self,
        task_state: TaskState,
        new_status: EnvelopeStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Transition task to new status if valid.
        
        Returns True if transition was successful, False otherwise.
        """
        current_status = task_state.status
        
        # Check if transition is valid
        if not self._is_valid_transition(current_status, new_status):
            return False
        
        # Record transition
        now = datetime.now(timezone.utc)
        self._record_transition(task_state, current_status, new_status, now)
        
        # Update task state
        task_state.status = new_status
        
        # Update timestamps
        if new_status == EnvelopeStatus.PENDING and current_status == EnvelopeStatus.PENDING:
            if not task_state.started_at:
                task_state.started_at = now
        elif new_status in [EnvelopeStatus.SUCCESS, EnvelopeStatus.ERROR]:
            task_state.completed_at = now
        
        # Handle error state
        if new_status == EnvelopeStatus.ERROR:
            task_state.error_message = error_message
        
        return True
    
    def can_retry(self, task_state: TaskState, max_retries: int = 3) -> bool:
        """Check if task can be retried."""
        return (
            task_state.status == EnvelopeStatus.ERROR and
            task_state.retry_count < max_retries
        )
    
    def prepare_retry(self, task_state: TaskState) -> None:
        """Prepare task for retry attempt."""
        task_state.retry_count += 1
        task_state.status = EnvelopeStatus.PENDING
        task_state.started_at = None
        task_state.completed_at = None
        task_state.error_message = None
    
    def get_task_duration(self, task_state: TaskState) -> Optional[float]:
        """Get task execution duration in seconds."""
        if not task_state.started_at:
            return None
        
        end_time = task_state.completed_at or datetime.now(timezone.utc)
        return (end_time - task_state.started_at).total_seconds()
    
    def get_transition_history(self, task_id: str) -> list:
        """Get transition history for a task."""
        return self.transition_history.get(task_id, [])
    
    def _is_valid_transition(
        self,
        current_status: EnvelopeStatus,
        new_status: EnvelopeStatus
    ) -> bool:
        """Check if state transition is valid."""
        valid_next_states = self.VALID_TRANSITIONS.get(current_status, [])
        return new_status in valid_next_states or current_status == new_status
    
    def _record_transition(
        self,
        task_state: TaskState,
        from_status: EnvelopeStatus,
        to_status: EnvelopeStatus,
        timestamp: datetime
    ) -> None:
        """Record state transition for history."""
        if task_state.task_id not in self.transition_history:
            self.transition_history[task_state.task_id] = []
        
        self.transition_history[task_state.task_id].append({
            "from_status": from_status.value,
            "to_status": to_status.value,
            "timestamp": timestamp,
            "retry_count": task_state.retry_count
        })