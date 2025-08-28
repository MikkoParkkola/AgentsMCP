"""
Quality gates and validation logic for orchestrated task execution.

Implements checkpoints and validation rules following AGENTS.md v2 quality patterns.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta

from ..models import ResultEnvelopeV1, EnvelopeStatus
from .state_machine import TaskState


class QualityGate:
    """Represents a single quality gate with validation logic."""
    
    def __init__(
        self,
        name: str,
        validator: Callable[[TaskState, ResultEnvelopeV1], bool],
        description: str,
        required: bool = True
    ):
        self.name = name
        self.validator = validator
        self.description = description
        self.required = required
    
    async def evaluate(self, task_state: TaskState, result: ResultEnvelopeV1) -> bool:
        """Evaluate this gate against task and result."""
        try:
            return self.validator(task_state, result)
        except Exception as e:
            # Gate evaluation failure - assume gate failed
            return False


class QualityGateManager:
    """
    Manages quality gates and validation for task execution.
    
    Enforces quality checkpoints at key phases of task execution
    and provides pass/fail feedback for coordination decisions.
    """
    
    def __init__(self):
        """Initialize with default quality gates."""
        self.gates: Dict[str, QualityGate] = {}
        self.gate_results: Dict[str, Dict[str, bool]] = {}
        
        # Register default quality gates
        self._register_default_gates()
    
    def register_gate(self, gate: QualityGate) -> None:
        """Register a new quality gate."""
        self.gates[gate.name] = gate
    
    async def evaluate(self, task_state: TaskState, result: ResultEnvelopeV1) -> Dict[str, bool]:
        """
        Evaluate all applicable quality gates for a task.
        
        Returns dict mapping gate names to pass/fail status.
        """
        gate_results = {}
        
        for gate_name, gate in self.gates.items():
            try:
                passed = await gate.evaluate(task_state, result)
                gate_results[gate_name] = passed
            except Exception as e:
                # Gate evaluation error - assume failure
                gate_results[gate_name] = False
        
        # Store results for history
        self.gate_results[task_state.task_id] = gate_results
        
        return gate_results
    
    def get_gate_results(self, task_id: str) -> Optional[Dict[str, bool]]:
        """Get gate evaluation results for a task."""
        return self.gate_results.get(task_id)
    
    def all_gates_passed(self, task_id: str) -> bool:
        """Check if all required gates passed for a task."""
        results = self.gate_results.get(task_id, {})
        
        for gate_name, passed in results.items():
            gate = self.gates.get(gate_name)
            if gate and gate.required and not passed:
                return False
        
        return True
    
    def get_failed_gates(self, task_id: str) -> List[str]:
        """Get list of failed gate names for a task."""
        results = self.gate_results.get(task_id, {})
        return [name for name, passed in results.items() if not passed]
    
    def _register_default_gates(self) -> None:
        """Register default quality gates."""
        
        # Success status gate
        self.register_gate(QualityGate(
            name="success_status",
            validator=lambda task, result: result.status == EnvelopeStatus.SUCCESS,
            description="Task completed successfully",
            required=True
        ))
        
        # Minimum confidence gate
        self.register_gate(QualityGate(
            name="confidence_threshold",
            validator=lambda task, result: (result.confidence or 0) >= 0.6,
            description="Result confidence >= 60%",
            required=True
        ))
        
        # Execution time gate (reasonable completion time)
        self.register_gate(QualityGate(
            name="execution_time",
            validator=self._validate_execution_time,
            description="Task completed within reasonable time",
            required=False
        ))
        
        # Output quality gate (has meaningful output)
        self.register_gate(QualityGate(
            name="output_quality",
            validator=self._validate_output_quality,
            description="Result contains meaningful output",
            required=True
        ))
        
        # Error handling gate (proper error reporting)
        self.register_gate(QualityGate(
            name="error_handling",
            validator=self._validate_error_handling,
            description="Errors are properly reported and actionable",
            required=False
        ))
    
    def _validate_execution_time(self, task_state: TaskState, result: ResultEnvelopeV1) -> bool:
        """Validate task completed within reasonable time."""
        if not task_state.started_at or not task_state.completed_at:
            return True  # Can't validate without timestamps
        
        duration = task_state.completed_at - task_state.started_at
        
        # Reasonable limits based on task complexity
        max_duration = timedelta(minutes=15)  # 15 minute max for any task
        
        return duration <= max_duration
    
    def _validate_output_quality(self, task_state: TaskState, result: ResultEnvelopeV1) -> bool:
        """Validate result has meaningful output."""
        if result.status == EnvelopeStatus.ERROR:
            # For error status, just need error info
            return bool(result.notes)
        
        # For success, need artifacts or meaningful notes
        has_artifacts = bool(result.artifacts)
        has_notes = bool(result.notes and len(result.notes.strip()) > 10)
        
        return has_artifacts or has_notes
    
    def _validate_error_handling(self, task_state: TaskState, result: ResultEnvelopeV1) -> bool:
        """Validate proper error handling and reporting."""
        if result.status == EnvelopeStatus.SUCCESS:
            return True  # Success doesn't need error handling
        
        # For failures, should have clear error information
        has_error_notes = bool(result.notes and "error" in result.notes.lower())
        has_metrics = bool(result.metrics)
        
        return has_error_notes or has_metrics