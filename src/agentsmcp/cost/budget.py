"""Budget management for cost control."""

from typing import Optional, List
from datetime import datetime

from .tracker import CostTracker
from .models import BudgetAlert


class BudgetManager:
    """Manages budget limits and alerts."""
    
    def __init__(self, cost_tracker: CostTracker, monthly_limit: float = 100.0):
        self.cost_tracker = cost_tracker
        self.monthly_limit = monthly_limit
        self.alerts: List[BudgetAlert] = []
    
    def check_budget(self) -> bool:
        """Check if we're within budget."""
        now = datetime.utcnow()
        monthly_cost = self.cost_tracker.get_monthly_cost(now.year, now.month)
        return monthly_cost <= self.monthly_limit
    
    def remaining_budget(self) -> float:
        """Get remaining budget for current month."""
        now = datetime.utcnow()
        monthly_cost = self.cost_tracker.get_monthly_cost(now.year, now.month)
        return max(0.0, self.monthly_limit - monthly_cost)
    
    def usage_percentage(self) -> float:
        """Get budget usage percentage."""
        now = datetime.utcnow()
        monthly_cost = self.cost_tracker.get_monthly_cost(now.year, now.month)
        if self.monthly_limit == 0:
            return 0.0
        return (monthly_cost / self.monthly_limit) * 100
    
    def check_alert_threshold(self, threshold: float = 80.0) -> Optional[BudgetAlert]:
        """Check if we've exceeded alert threshold."""
        usage = self.usage_percentage()
        
        if usage >= threshold:
            now = datetime.utcnow()
            monthly_cost = self.cost_tracker.get_monthly_cost(now.year, now.month)
            
            alert = BudgetAlert(
                threshold=threshold,
                current_cost=monthly_cost,
                budget_limit=self.monthly_limit
            )
            
            self.alerts.append(alert)
            return alert
        
        return None