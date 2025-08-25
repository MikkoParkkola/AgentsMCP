"""
State management for AgentsMCP chain composition.

This module defines the state models and validation logic for chain execution,
providing type-safe state passing between chain steps with proper serialization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

__all__ = [
    "ChainState",
    "StepResult", 
    "ChainContext",
    "StateValidator",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Core State Models
# --------------------------------------------------------------------------- #

@dataclass
class StepResult:
    """
    Result of a single step in a chain execution.
    
    Attributes
    ----------
    step_id : str
        Unique identifier for the step.
    agent_id : str
        ID of the agent that executed this step.
    success : bool
        Whether the step completed successfully.
    output : Any
        The output data from the step.
    error : Optional[str]
        Error message if step failed.
    execution_time : float
        Time taken to execute the step in seconds.
    cost : Optional[float]
        Cost of executing this step.
    metadata : Dict[str, Any]
        Additional metadata about the execution.
    """
    step_id: str
    agent_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass 
class ChainContext:
    """
    Context information for chain execution.
    
    Attributes
    ----------
    chain_id : str
        Unique identifier for the chain execution.
    user_id : Optional[str]
        ID of the user who initiated the chain.
    budget : Optional[float]
        Budget limit for the entire chain execution.
    timeout : Optional[float]
        Timeout for the entire chain in seconds.
    retry_config : Dict[str, Any]
        Configuration for retry logic.
    preferences : Dict[str, Any]
        User preferences for agent selection.
    """
    chain_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    budget: Optional[float] = None
    timeout: Optional[float] = None
    retry_config: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ChainState:
    """
    Complete state for a chain execution.
    
    This is the primary state object that flows through the LangGraph
    state machine, containing all data and context needed for execution.
    """
    context: ChainContext
    data: Dict[str, Any] = field(default_factory=dict)
    step_results: List[StepResult] = field(default_factory=list)
    current_step: int = 0
    total_cost: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step_result(self, result: StepResult) -> None:
        """Add a step result and update state."""
        self.step_results.append(result)
        if result.cost:
            self.total_cost += result.cost
        if not result.success and result.error:
            self.errors.append(f"Step {result.step_id}: {result.error}")
        
        logger.debug(
            f"Added step result {result.step_id}: "
            f"success={result.success}, cost={result.cost}"
        )

    def get_step_output(self, step_id: str) -> Any:
        """Get output from a specific step."""
        for result in self.step_results:
            if result.step_id == step_id:
                return result.output
        raise KeyError(f"Step {step_id} not found in results")

    def has_errors(self) -> bool:
        """Check if any steps have failed."""
        return len(self.errors) > 0

    def get_last_success_output(self) -> Any:
        """Get output from the last successful step."""
        for result in reversed(self.step_results):
            if result.success:
                return result.output
        return None

    def is_budget_exceeded(self) -> bool:
        """Check if budget limit has been exceeded."""
        if self.context.budget is None:
            return False
        return self.total_cost > self.context.budget

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for persistence."""
        return {
            "context": {
                "chain_id": self.context.chain_id,
                "user_id": self.context.user_id,
                "budget": self.context.budget,
                "timeout": self.context.timeout,
                "retry_config": self.context.retry_config,
                "preferences": self.context.preferences,
                "created_at": self.context.created_at.isoformat(),
            },
            "data": self.data,
            "step_results": [
                {
                    "step_id": r.step_id,
                    "agent_id": r.agent_id,
                    "success": r.success,
                    "output": r.output,
                    "error": r.error,
                    "execution_time": r.execution_time,
                    "cost": r.cost,
                    "metadata": r.metadata,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self.step_results
            ],
            "current_step": self.current_step,
            "total_cost": self.total_cost,
            "errors": self.errors,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChainState:
        """Deserialize state from dictionary."""
        context_data = data["context"]
        context = ChainContext(
            chain_id=context_data["chain_id"],
            user_id=context_data.get("user_id"),
            budget=context_data.get("budget"),
            timeout=context_data.get("timeout"),
            retry_config=context_data.get("retry_config", {}),
            preferences=context_data.get("preferences", {}),
            created_at=datetime.fromisoformat(context_data["created_at"]),
        )

        step_results = [
            StepResult(
                step_id=r["step_id"],
                agent_id=r["agent_id"],
                success=r["success"],
                output=r["output"],
                error=r.get("error"),
                execution_time=r.get("execution_time", 0.0),
                cost=r.get("cost"),
                metadata=r.get("metadata", {}),
                timestamp=datetime.fromisoformat(r["timestamp"]),
            )
            for r in data.get("step_results", [])
        ]

        return cls(
            context=context,
            data=data.get("data", {}),
            step_results=step_results,
            current_step=data.get("current_step", 0),
            total_cost=data.get("total_cost", 0.0),
            errors=data.get("errors", []),
            metadata=data.get("metadata", {}),
        )

# --------------------------------------------------------------------------- #
# State Validation
# --------------------------------------------------------------------------- #

class StateValidator:
    """Validates chain state for consistency and constraints."""
    
    def __init__(self, max_budget: Optional[float] = None):
        self.max_budget = max_budget
        self.logger = logging.getLogger(__name__)

    def validate_state(self, state: ChainState) -> List[str]:
        """
        Validate chain state and return list of validation errors.
        
        Parameters
        ----------
        state : ChainState
            State to validate.
            
        Returns
        -------
        List[str]
            List of validation error messages.
        """
        errors = []

        # Budget validation
        if self.max_budget and state.total_cost > self.max_budget:
            errors.append(f"Total cost {state.total_cost} exceeds maximum {self.max_budget}")

        if state.context.budget and state.total_cost > state.context.budget:
            errors.append(f"Total cost {state.total_cost} exceeds chain budget {state.context.budget}")

        # Step validation
        if state.current_step < 0:
            errors.append("Current step cannot be negative")

        if state.current_step > len(state.step_results):
            errors.append("Current step exceeds number of completed steps")

        # Data validation
        if not isinstance(state.data, dict):
            errors.append("State data must be a dictionary")

        # Check for duplicate step IDs
        step_ids = [r.step_id for r in state.step_results]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step IDs found in results")

        return errors

    def validate_context(self, context: ChainContext) -> List[str]:
        """Validate chain context."""
        errors = []

        if context.budget is not None and context.budget < 0:
            errors.append("Budget cannot be negative")

        if context.timeout is not None and context.timeout <= 0:
            errors.append("Timeout must be positive")

        return errors

    def validate_step_result(self, result: StepResult) -> List[str]:
        """Validate individual step result."""
        errors = []

        if not result.step_id:
            errors.append("Step ID cannot be empty")

        if not result.agent_id:
            errors.append("Agent ID cannot be empty")

        if result.cost is not None and result.cost < 0:
            errors.append("Cost cannot be negative")

        if result.execution_time < 0:
            errors.append("Execution time cannot be negative")

        return errors

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def serialize_state(state: ChainState) -> str:
    """Serialize chain state to JSON string."""
    try:
        return json.dumps(state.to_dict(), indent=2, default=str)
    except Exception as exc:
        logger.error(f"Failed to serialize state: {exc}")
        raise

def deserialize_state(data: str) -> ChainState:
    """Deserialize chain state from JSON string."""
    try:
        parsed = json.loads(data)
        return ChainState.from_dict(parsed)
    except Exception as exc:
        logger.error(f"Failed to deserialize state: {exc}")
        raise

# --------------------------------------------------------------------------- #
# Test and demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create test state
    context = ChainContext(user_id="test-user", budget=100.0)
    state = ChainState(context=context)
    
    # Add some step results
    result1 = StepResult(
        step_id="step1",
        agent_id="agent1", 
        success=True,
        output="Hello from step 1",
        cost=5.0,
    )
    state.add_step_result(result1)
    
    result2 = StepResult(
        step_id="step2",
        agent_id="agent2",
        success=False,
        output=None,
        error="Step failed",
        cost=2.0,
    )
    state.add_step_result(result2)
    
    # Test serialization
    serialized = serialize_state(state)
    print("Serialized state:")
    print(serialized)
    
    # Test deserialization
    restored = deserialize_state(serialized)
    print(f"\nRestored state has {len(restored.step_results)} step results")
    print(f"Total cost: ${restored.total_cost}")
    print(f"Has errors: {restored.has_errors()}")
    
    # Test validation
    validator = StateValidator(max_budget=50.0)
    errors = validator.validate_state(restored)
    if errors:
        print(f"\nValidation errors: {errors}")
    else:
        print("\nState validation passed")