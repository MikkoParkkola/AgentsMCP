"""Data models for benchmarking system."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional


class TaskCategory(Enum):
    """Categories of AI tasks for benchmarking."""
    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    GENERAL = "general"


@dataclass
class BenchmarkResult:
    """Result of a model benchmark."""
    
    model: str
    provider: str
    task_category: TaskCategory
    quality_score: float
    speed_seconds: float
    cost: float
    timestamp: datetime
    
    @property
    def cost_effectiveness(self) -> float:
        """Calculate cost-effectiveness score (quality/cost)."""
        if self.cost == 0:
            return float('inf')  # Free models have infinite cost-effectiveness
        return self.quality_score / self.cost