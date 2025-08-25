"""Thread-safe cost tracking for AI model calls."""

import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

from .models import CostRecord, BudgetAlert


class CostTracker:
    """Thread-safe cost tracker for AI model usage."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._records: List[CostRecord] = []
        self._total_cost: float = 0.0
        
    def record_call(self, record: CostRecord, token_price: float = 0.0) -> None:
        """Record a new AI model call with cost."""
        with self._lock:
            # Update cost if not already set
            if record.cost == 0.0 and token_price > 0.0:
                cost = record.total_tokens * token_price
                record = CostRecord(
                    call_id=record.call_id,
                    provider=record.provider,
                    model=record.model,
                    task=record.task,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    cost=cost,
                    timestamp=record.timestamp
                )
            
            self._records.append(record)
            self._total_cost += record.cost
    
    @property
    def total_cost(self) -> float:
        """Get total cost across all calls."""
        with self._lock:
            return self._total_cost
    
    def get_daily_cost(self, date: Optional[datetime] = None) -> float:
        """Get cost for a specific day."""
        if date is None:
            date = datetime.utcnow()
        
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        with self._lock:
            daily_cost = sum(
                record.cost for record in self._records
                if start_of_day <= record.timestamp < end_of_day
            )
            return daily_cost
    
    def get_monthly_cost(self, year: int, month: int) -> float:
        """Get cost for a specific month."""
        with self._lock:
            monthly_cost = sum(
                record.cost for record in self._records
                if record.timestamp.year == year and record.timestamp.month == month
            )
            return monthly_cost
    
    def get_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get cost breakdown by provider and model."""
        breakdown = defaultdict(lambda: defaultdict(float))
        
        with self._lock:
            for record in self._records:
                breakdown[record.provider][record.model] += record.cost
        
        # Convert to regular dict
        return {
            provider: dict(models)
            for provider, models in breakdown.items()
        }
    
    def get_recent_calls(self, limit: int = 10) -> List[CostRecord]:
        """Get most recent calls."""
        with self._lock:
            return sorted(self._records, key=lambda r: r.timestamp, reverse=True)[:limit]
    
    def clear(self) -> None:
        """Clear all tracking data."""
        with self._lock:
            self._records.clear()
            self._total_cost = 0.0