"""
Cost optimization subsystem for AgentsMCP routing.

The CostOptimizer provides intelligent cost optimization algorithms that analyze
historical performance and cost data to make optimal routing decisions.
"""

from __future__ import annotations

import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import statistics

from .tracker import MetricsTracker, RequestMetrics
from .selector import ModelSelector, TaskSpec
from .models import ModelDB, Model

__all__ = [
    "CostOptimizer",
    "OptimizationResult",
    "CostOptimizerError",
]

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Exception Classes
# --------------------------------------------------------------------------- #
class CostOptimizerError(Exception):
    """Base exception for cost optimizer errors."""
    pass

# --------------------------------------------------------------------------- #
# Result Classes
# --------------------------------------------------------------------------- #
@dataclass
class OptimizationResult:
    """Result of an optimization analysis."""
    model_id: str
    expected_cost: float
    expected_performance: float
    confidence: float
    reasoning: str
    alternatives: List[Tuple[str, float, str]]  # (model_id, cost, reason)

# --------------------------------------------------------------------------- #
# Main Optimizer Class
# --------------------------------------------------------------------------- #
class CostOptimizer:
    """
    Intelligent cost optimizer that analyzes historical data to make
    optimal routing decisions balancing cost, performance, and quality.

    The optimizer integrates with MetricsTracker for historical analysis,
    ModelSelector for routing decisions, and ModelDB for model capabilities.
    """

    def __init__(self,
                 metrics_tracker: MetricsTracker,
                 model_selector: ModelSelector,
                 model_db: ModelDB):
        """
        Initialize the cost optimizer.

        Parameters
        ----------
        metrics_tracker : MetricsTracker
            Instance for accessing historical performance and cost data.
        model_selector : ModelSelector
            Instance for model selection logic.
        model_db : ModelDB
            Instance for model capabilities and pricing data.
        """
        self.metrics_tracker = metrics_tracker
        self.model_selector = model_selector
        self.model_db = model_db
        
        # Cache for expensive computations
        self._cost_trends_cache: Optional[Dict[str, Any]] = None
        self._efficiency_cache: Optional[Dict[str, float]] = None
        
        logger.info("CostOptimizer initialized")

    # ----------------------------------------------------------------------- #
    # Cost Trend Analysis
    # ----------------------------------------------------------------------- #
    def analyze_cost_trends(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze spending patterns and detect anomalies over the specified period.

        Parameters
        ----------
        days_back : int
            Number of days to analyze (default: 30)

        Returns
        -------
        Dict[str, Any]
            Analysis results including trends, anomalies, and recommendations
        """
        if not self.metrics_tracker._metrics:
            return {
                "total_cost": 0.0,
                "daily_average": 0.0,
                "trends": {},
                "anomalies": [],
                "recommendations": ["No historical data available for analysis"]
            }

        # Get recent metrics
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        recent_metrics = [
            m for m in self.metrics_tracker._metrics
            if start_date <= m.request_ts <= end_date
        ]

        if not recent_metrics:
            return {
                "total_cost": 0.0,
                "daily_average": 0.0,
                "trends": {},
                "anomalies": [],
                "recommendations": ["No recent data available for analysis"]
            }

        # Calculate basic statistics
        total_cost = sum(m.cost for m in recent_metrics)
        daily_average = total_cost / days_back
        
        # Group by model for trend analysis
        model_costs = defaultdict(list)
        for m in recent_metrics:
            day = m.request_ts.date().isoformat()
            model_costs[m.model].append((day, m.cost))

        # Calculate trends per model
        trends = {}
        for model, costs in model_costs.items():
            daily_sums = defaultdict(float)
            for day, cost in costs:
                daily_sums[day] += cost
            
            if len(daily_sums) > 1:
                costs_list = list(daily_sums.values())
                trend_direction = "increasing" if costs_list[-1] > costs_list[0] else "decreasing"
                volatility = statistics.stdev(costs_list) if len(costs_list) > 1 else 0.0
            else:
                trend_direction = "stable"
                volatility = 0.0
                
            trends[model] = {
                "direction": trend_direction,
                "volatility": volatility,
                "total_cost": sum(costs_list) if 'costs_list' in locals() else 0.0
            }

        # Detect anomalies (days with unusually high cost)
        daily_totals = defaultdict(float)
        for m in recent_metrics:
            day = m.request_ts.date().isoformat()
            daily_totals[day] += m.cost

        costs = list(daily_totals.values())
        anomalies = []
        if len(costs) > 2:
            mean_cost = statistics.mean(costs)
            std_cost = statistics.stdev(costs)
            
            for day, cost in daily_totals.items():
                if cost > mean_cost + 2 * std_cost:  # 2 sigma threshold
                    anomalies.append({
                        "date": day,
                        "cost": cost,
                        "deviation": (cost - mean_cost) / std_cost
                    })

        # Generate recommendations
        recommendations = self._generate_cost_recommendations(trends, total_cost, daily_average)

        result = {
            "total_cost": total_cost,
            "daily_average": daily_average,
            "period_days": days_back,
            "trends": trends,
            "anomalies": anomalies,
            "recommendations": recommendations
        }
        
        self._cost_trends_cache = result
        return result

    def _generate_cost_recommendations(self, 
                                     trends: Dict[str, Dict], 
                                     total_cost: float, 
                                     daily_average: float) -> List[str]:
        """Generate cost optimization recommendations based on trends."""
        recommendations = []
        
        # Find most expensive models
        model_costs = [(model, data["total_cost"]) for model, data in trends.items()]
        model_costs.sort(key=lambda x: x[1], reverse=True)
        
        if model_costs:
            most_expensive = model_costs[0]
            if most_expensive[1] > total_cost * 0.3:  # >30% of total cost
                recommendations.append(
                    f"Model '{most_expensive[0]}' accounts for "
                    f"{most_expensive[1]/total_cost*100:.1f}% of total cost. "
                    f"Consider cheaper alternatives for appropriate tasks."
                )

        # Check for volatile models
        for model, data in trends.items():
            if data["volatility"] > daily_average * 0.5:
                recommendations.append(
                    f"Model '{model}' shows high cost volatility. "
                    f"Consider implementing usage quotas or budget alerts."
                )

        # Check for increasing trends
        increasing_models = [
            model for model, data in trends.items()
            if data["direction"] == "increasing"
        ]
        if len(increasing_models) > len(trends) / 2:
            recommendations.append(
                "Multiple models show increasing cost trends. "
                "Review usage patterns and consider cost controls."
            )

        if not recommendations:
            recommendations.append("Cost trends appear stable. Continue monitoring.")

        return recommendations

    # ----------------------------------------------------------------------- #
    # Budget Allocation
    # ----------------------------------------------------------------------- #
    def suggest_budget_allocation(self, total_budget: float) -> Dict[str, Any]:
        """
        Suggest optimal budget allocation across models and task types.

        Parameters
        ----------
        total_budget : float
            Total budget to allocate (in USD)

        Returns
        -------
        Dict[str, Any]
            Budget allocation recommendations
        """
        if not self.metrics_tracker._metrics:
            return {
                "error": "No historical data available",
                "allocations": {}
            }

        # Analyze current usage patterns
        model_usage = Counter()
        model_costs = defaultdict(float)
        task_usage = Counter()
        task_costs = defaultdict(float)

        for m in self.metrics_tracker._metrics:
            model_usage[m.model] += 1
            model_costs[m.model] += m.cost
            if m.task_type:
                task_usage[m.task_type] += 1
                task_costs[m.task_type] += m.cost

        total_historical_cost = sum(model_costs.values())
        if total_historical_cost == 0:
            return {
                "error": "No cost data available",
                "allocations": {}
            }

        # Allocate budget proportionally to historical usage with efficiency adjustments
        model_allocations = {}
        for model in model_usage:
            historical_proportion = model_costs[model] / total_historical_cost
            base_allocation = total_budget * historical_proportion
            
            # Adjust for efficiency (cost per success)
            model_metrics = [m for m in self.metrics_tracker._metrics if m.model == model]
            success_rate = sum(1 for m in model_metrics if m.success) / len(model_metrics)
            efficiency_multiplier = 0.8 + (success_rate * 0.4)  # 0.8 to 1.2 range
            
            adjusted_allocation = base_allocation * efficiency_multiplier
            model_allocations[model] = {
                "allocation": adjusted_allocation,
                "historical_cost": model_costs[model],
                "success_rate": success_rate,
                "efficiency_score": efficiency_multiplier
            }

        # Task-based allocations
        task_allocations = {}
        for task in task_usage:
            historical_proportion = task_costs[task] / total_historical_cost
            task_allocations[task] = {
                "allocation": total_budget * historical_proportion,
                "historical_cost": task_costs[task],
                "request_count": task_usage[task]
            }

        return {
            "total_budget": total_budget,
            "model_allocations": model_allocations,
            "task_allocations": task_allocations,
            "recommendations": self._generate_allocation_recommendations(model_allocations)
        }

    def _generate_allocation_recommendations(self, allocations: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on budget allocation analysis."""
        recommendations = []
        
        # Find inefficient models
        inefficient = [
            model for model, data in allocations.items()
            if data["efficiency_score"] < 0.9
        ]
        if inefficient:
            recommendations.append(
                f"Models with low efficiency scores: {', '.join(inefficient)}. "
                f"Consider reducing their budget allocation or improving their usage patterns."
            )

        # Find high-cost models
        sorted_by_allocation = sorted(
            allocations.items(),
            key=lambda x: x[1]["allocation"],
            reverse=True
        )
        if sorted_by_allocation:
            top_model = sorted_by_allocation[0]
            if top_model[1]["allocation"] > sum(a["allocation"] for a in allocations.values()) * 0.5:
                recommendations.append(
                    f"Model '{top_model[0]}' consumes {top_model[1]['allocation']/sum(a['allocation'] for a in allocations.values())*100:.1f}% of budget. "
                    f"Consider diversifying model usage."
                )

        return recommendations

    # ----------------------------------------------------------------------- #
    # Model Selection Optimization
    # ----------------------------------------------------------------------- #
    def optimize_model_selection(self, task_spec: TaskSpec) -> OptimizationResult:
        """
        Optimize model selection for a given task specification.

        Parameters
        ----------
        task_spec : TaskSpec
            Task specification including requirements and constraints

        Returns
        -------
        OptimizationResult
            Optimized model selection with reasoning
        """
        # Get candidate models from selector
        candidates = self.model_db.filter(
            category=task_spec.task_type,
            min_performance=task_spec.min_performance_tier,
            max_cost_per_input=task_spec.max_cost_per_1k_tokens,
            custom_filters=[
                lambda m: (
                    m.context_length is None or 
                    task_spec.required_context_length is None or
                    m.context_length >= task_spec.required_context_length
                )
            ] if task_spec.required_context_length else None
        )

        if not candidates:
            raise CostOptimizerError(
                f"No models available matching task specification: {task_spec}"
            )

        # Score candidates based on historical performance and cost
        scored_candidates = []
        for model in candidates:
            score = self._calculate_optimization_score(model, task_spec)
            scored_candidates.append((model, score))

        # Sort by score (higher is better)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        best_model, best_score = scored_candidates[0]
        
        # Calculate expected cost and performance
        expected_cost = self._estimate_request_cost(best_model, task_spec)
        expected_performance = best_model.performance_tier
        
        # Generate alternatives
        alternatives = []
        for model, score in scored_candidates[1:4]:  # Top 3 alternatives
            alt_cost = self._estimate_request_cost(model, task_spec)
            reason = f"Score: {score:.2f}, Cost: ${alt_cost:.4f}"
            alternatives.append((model.id, alt_cost, reason))

        # Generate reasoning
        reasoning = self._generate_selection_reasoning(best_model, task_spec, best_score)

        return OptimizationResult(
            model_id=best_model.id,
            expected_cost=expected_cost,
            expected_performance=expected_performance,
            confidence=min(best_score / 10.0, 1.0),  # Normalize to 0-1
            reasoning=reasoning,
            alternatives=alternatives
        )

    def _calculate_optimization_score(self, model: Model, task_spec: TaskSpec) -> float:
        """Calculate optimization score for a model given task requirements."""
        score = 0.0
        
        # Performance score (0-5 points)
        score += model.performance_tier
        
        # Cost efficiency (0-3 points, inverse relationship)
        if task_spec.max_cost_per_1k_tokens:
            if model.cost_per_input_token <= task_spec.max_cost_per_1k_tokens:
                efficiency = task_spec.max_cost_per_1k_tokens / max(model.cost_per_input_token, 0.001)
                score += min(efficiency, 3.0)
        else:
            # Reward lower cost models
            score += max(0, 3.0 - model.cost_per_input_token)
        
        # Historical performance bonus (0-2 points)
        historical_bonus = self._get_historical_performance_bonus(model.id, task_spec.task_type)
        score += historical_bonus
        
        # Provider preference (0-1 points)
        preferred_provider = task_spec.preferences.get("preferred_provider")
        if preferred_provider and model.provider.lower() == preferred_provider.lower():
            score += 1.0
            
        return score

    def _get_historical_performance_bonus(self, model_id: str, task_type: Optional[str]) -> float:
        """Get bonus score based on historical performance."""
        if not self.metrics_tracker._metrics:
            return 0.0
            
        # Find relevant historical metrics
        relevant_metrics = [
            m for m in self.metrics_tracker._metrics
            if m.model == model_id and (task_type is None or m.task_type == task_type)
        ]
        
        if not relevant_metrics:
            return 0.0
            
        # Calculate success rate bonus
        success_rate = sum(1 for m in relevant_metrics if m.success) / len(relevant_metrics)
        return success_rate * 2.0  # Up to 2 points for perfect success rate

    def _estimate_request_cost(self, model: Model, task_spec: TaskSpec) -> float:
        """Estimate the cost for a single request."""
        # Simple estimation based on expected token usage
        estimated_input_tokens = task_spec.required_context_length or 1000
        estimated_output_tokens = 500  # Default assumption
        
        input_cost = (estimated_input_tokens / 1000) * model.cost_per_input_token
        output_cost = (estimated_output_tokens / 1000) * model.cost_per_output_token
        
        return input_cost + output_cost

    def _generate_selection_reasoning(self, model: Model, task_spec: TaskSpec, score: float) -> str:
        """Generate human-readable reasoning for model selection."""
        reasons = []
        
        reasons.append(f"Selected {model.name} by {model.provider}")
        reasons.append(f"Performance tier: {model.performance_tier}/5")
        reasons.append(f"Cost: ${model.cost_per_input_token}/1k input, ${model.cost_per_output_token}/1k output")
        
        if task_spec.max_cost_per_1k_tokens:
            if model.cost_per_input_token <= task_spec.max_cost_per_1k_tokens:
                reasons.append("Within budget constraints")
            
        if model.context_length:
            reasons.append(f"Context length: {model.context_length:,} tokens")
            
        reasons.append(f"Optimization score: {score:.2f}")
        
        return "; ".join(reasons)

    # ----------------------------------------------------------------------- #
    # Cost Forecasting
    # ----------------------------------------------------------------------- #
    def predict_monthly_cost(self, months_ahead: int = 3) -> Dict[str, Any]:
        """
        Predict monthly costs based on current usage patterns.

        Parameters
        ----------
        months_ahead : int
            Number of months to forecast

        Returns
        -------
        Dict[str, Any]
            Cost forecast with confidence intervals
        """
        if not self.metrics_tracker._metrics:
            return {
                "error": "No historical data available for prediction",
                "forecasts": []
            }

        # Analyze recent monthly costs
        monthly_costs = defaultdict(float)
        for m in self.metrics_tracker._metrics:
            month_key = m.request_ts.strftime("%Y-%m")
            monthly_costs[month_key] += m.cost

        if len(monthly_costs) < 2:
            return {
                "error": "Insufficient historical data (need at least 2 months)",
                "forecasts": []
            }

        # Simple trend analysis
        sorted_months = sorted(monthly_costs.items())
        costs = [cost for _, cost in sorted_months]
        
        # Calculate trend
        if len(costs) > 1:
            recent_avg = statistics.mean(costs[-3:]) if len(costs) >= 3 else statistics.mean(costs)
            growth_rate = (costs[-1] / costs[0]) ** (1 / len(costs)) - 1 if costs[0] > 0 else 0
        else:
            recent_avg = costs[0]
            growth_rate = 0

        # Generate forecasts
        forecasts = []
        current_month = datetime.utcnow()
        base_cost = recent_avg

        for i in range(1, months_ahead + 1):
            forecast_month = current_month + timedelta(days=30 * i)
            predicted_cost = base_cost * ((1 + growth_rate) ** i)
            
            # Add uncertainty (±20% confidence interval)
            confidence_interval = predicted_cost * 0.2
            
            forecasts.append({
                "month": forecast_month.strftime("%Y-%m"),
                "predicted_cost": predicted_cost,
                "confidence_low": predicted_cost - confidence_interval,
                "confidence_high": predicted_cost + confidence_interval,
                "growth_rate": growth_rate * 100
            })

        return {
            "historical_months": len(monthly_costs),
            "recent_average": recent_avg,
            "growth_rate_percent": growth_rate * 100,
            "forecasts": forecasts
        }

    # ----------------------------------------------------------------------- #
    # Comprehensive Cost Report
    # ----------------------------------------------------------------------- #
    def generate_cost_report(self, include_recommendations: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive cost analysis report.

        Parameters
        ----------
        include_recommendations : bool
            Whether to include optimization recommendations

        Returns
        -------
        Dict[str, Any]
            Comprehensive cost report
        """
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {}
        }

        # Overall summary
        if self.metrics_tracker._metrics:
            total_cost = self.metrics_tracker._total_cost
            total_requests = self.metrics_tracker._total_requests
            avg_cost_per_request = total_cost / max(total_requests, 1)
            
            report["summary"] = {
                "total_cost": total_cost,
                "total_requests": total_requests,
                "average_cost_per_request": avg_cost_per_request,
                "cost_per_successful_request": (
                    total_cost / max(sum(self.metrics_tracker._success_counter.values()), 1)
                )
            }
        else:
            report["summary"] = {
                "total_cost": 0.0,
                "total_requests": 0,
                "average_cost_per_request": 0.0,
                "cost_per_successful_request": 0.0
            }

        # Add component analyses
        report["cost_trends"] = self.analyze_cost_trends()
        report["budget_allocation"] = self.suggest_budget_allocation(10000.0)  # $10K example
        report["cost_forecast"] = self.predict_monthly_cost()

        # Add recommendations if requested
        if include_recommendations:
            report["recommendations"] = self._generate_comprehensive_recommendations(report)

        return report

    def _generate_comprehensive_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate comprehensive cost optimization recommendations."""
        recommendations = []
        
        # From cost trends
        if "cost_trends" in report and "recommendations" in report["cost_trends"]:
            recommendations.extend(report["cost_trends"]["recommendations"])

        # From budget allocation
        if "budget_allocation" in report and "recommendations" in report["budget_allocation"]:
            recommendations.extend(report["budget_allocation"]["recommendations"])

        # High-level recommendations
        summary = report.get("summary", {})
        avg_cost = summary.get("average_cost_per_request", 0)
        
        if avg_cost > 0.05:  # $0.05 per request threshold
            recommendations.append(
                f"Average cost per request (${avg_cost:.4f}) is high. "
                f"Consider using cheaper models for routine tasks."
            )

        success_rate = (
            summary.get("cost_per_successful_request", 0) / max(avg_cost, 0.001)
            if avg_cost > 0 else 1.0
        )
        
        if success_rate < 0.9:
            recommendations.append(
                f"Success rate appears low ({success_rate*100:.1f}%). "
                f"Review error patterns and model selection criteria."
            )

        return list(set(recommendations))  # Remove duplicates


# --------------------------------------------------------------------------- #
# Demo and Testing
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from .models import ModelDB
    from .selector import ModelSelector
    from .tracker import MetricsTracker
    
    # Create sample components (normally these would be real instances)
    model_db = ModelDB()
    tracker = MetricsTracker()
    selector = ModelSelector(model_db)
    
    # Initialize optimizer
    optimizer = CostOptimizer(tracker, selector, model_db)
    
    print("=== Cost Optimizer Demo ===")
    print("Note: This demo requires historical data in the tracker for meaningful results.")
    
    # Generate empty report (no historical data)
    report = optimizer.generate_cost_report()
    print(f"\nSample report structure: {list(report.keys())}")
    
    print("\nCost Optimizer initialized successfully!")