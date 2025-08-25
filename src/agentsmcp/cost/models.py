"""Core data models for cost tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any


@dataclass(frozen=True)
class CostRecord:
    """Record of a single AI model call with cost information."""
    
    call_id: str
    provider: str
    model: str
    task: str
    input_tokens: int
    output_tokens: int
    cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used in this call."""
        return self.input_tokens + self.output_tokens
    
    @property
    def cost_per_token(self) -> float:
        """Cost per token for this call."""
        if self.total_tokens == 0:
            return 0.0
        return self.cost / self.total_tokens


@dataclass
class BudgetAlert:
    """Budget threshold alert."""
    
    threshold: float
    current_cost: float
    budget_limit: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def percentage_used(self) -> float:
        """Percentage of budget used."""
        if self.budget_limit == 0:
            return 0.0
        return (self.current_cost / self.budget_limit) * 100