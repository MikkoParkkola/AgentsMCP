"""
ImprovementPrioritizer - Advanced ranking and selection logic for improvements.

This component provides sophisticated prioritization algorithms that go beyond
basic priority scoring, incorporating business value, resource constraints,
and strategic considerations.
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from .improvement_generator import (
    ImprovementOpportunity,
    ImprovementType,
    ImplementationEffort,
    RiskLevel
)


class PrioritizationStrategy(Enum):
    """Available prioritization strategies."""
    EXPECTED_VALUE = "expected_value"  # Impact × Probability / Effort
    RISK_ADJUSTED_VALUE = "risk_adjusted_value"  # EV adjusted for risk
    RESOURCE_CONSTRAINED = "resource_constrained"  # Optimize for resource limits
    STRATEGIC_ALIGNMENT = "strategic_alignment"  # Align with strategic goals
    QUICK_WINS_FIRST = "quick_wins_first"  # Low effort, high impact first
    BALANCED_PORTFOLIO = "balanced_portfolio"  # Balanced risk/reward mix


@dataclass 
class ResourceConstraints:
    """Resource constraints for improvement prioritization."""
    
    # Time constraints
    total_time_budget: Optional[timedelta] = None
    time_per_sprint: Optional[timedelta] = None
    
    # Team capacity
    available_developers: int = 1
    developer_skill_levels: Dict[str, float] = field(default_factory=dict)  # skill -> level (0-1)
    
    # Technical constraints
    deployment_windows: List[Tuple[datetime, datetime]] = field(default_factory=list)
    required_approvals: Set[str] = field(default_factory=set)
    
    # Budget constraints
    max_cost_budget: Optional[float] = None
    cost_per_developer_hour: float = 100.0


@dataclass
class StrategicGoals:
    """Strategic goals for improvement prioritization."""
    
    # Performance targets
    target_response_time_ms: Optional[float] = None
    target_error_rate: Optional[float] = None
    target_availability: Optional[float] = None
    
    # User experience targets
    target_user_satisfaction: Optional[float] = None
    target_task_completion_rate: Optional[float] = None
    
    # Business goals
    priority_components: Set[str] = field(default_factory=set)
    strategic_initiatives: Dict[str, float] = field(default_factory=dict)  # initiative -> weight
    
    # Technical debt goals
    max_technical_debt_score: Optional[float] = None
    modernization_priorities: List[str] = field(default_factory=list)


@dataclass
class PrioritizationResult:
    """Result of improvement prioritization."""
    
    prioritized_improvements: List[ImprovementOpportunity]
    prioritization_strategy: PrioritizationStrategy
    total_expected_value: float
    resource_utilization: Dict[str, float]
    strategic_alignment_score: float
    risk_distribution: Dict[RiskLevel, int]
    effort_distribution: Dict[ImplementationEffort, int]
    rationale: List[str]


class ImprovementPrioritizer:
    """
    Advanced prioritization engine for improvement opportunities.
    
    Provides multiple prioritization strategies that consider business value,
    resource constraints, strategic alignment, and risk management.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Strategy implementations
        self._strategies = {
            PrioritizationStrategy.EXPECTED_VALUE: self._prioritize_by_expected_value,
            PrioritizationStrategy.RISK_ADJUSTED_VALUE: self._prioritize_by_risk_adjusted_value,
            PrioritizationStrategy.RESOURCE_CONSTRAINED: self._prioritize_by_resource_constraints,
            PrioritizationStrategy.STRATEGIC_ALIGNMENT: self._prioritize_by_strategic_alignment,
            PrioritizationStrategy.QUICK_WINS_FIRST: self._prioritize_quick_wins,
            PrioritizationStrategy.BALANCED_PORTFOLIO: self._prioritize_balanced_portfolio
        }
        
        # Default weights for multi-criteria decision making
        self._default_weights = {
            'impact': 0.4,
            'confidence': 0.2,
            'effort': 0.2,
            'risk': 0.1,
            'strategic_alignment': 0.1
        }
    
    async def prioritize_improvements(
        self,
        improvements: List[ImprovementOpportunity],
        strategy: PrioritizationStrategy = PrioritizationStrategy.EXPECTED_VALUE,
        resource_constraints: Optional[ResourceConstraints] = None,
        strategic_goals: Optional[StrategicGoals] = None,
        max_selections: Optional[int] = None
    ) -> PrioritizationResult:
        """
        Prioritize improvements using the specified strategy.
        
        Args:
            improvements: List of improvement opportunities
            strategy: Prioritization strategy to use
            resource_constraints: Optional resource constraints
            strategic_goals: Optional strategic goals
            max_selections: Maximum number of improvements to select
            
        Returns:
            Prioritization result with ranked improvements
        """
        try:
            self.logger.info(f"Prioritizing {len(improvements)} improvements using {strategy.value}")
            
            if not improvements:
                return PrioritizationResult(
                    prioritized_improvements=[],
                    prioritization_strategy=strategy,
                    total_expected_value=0.0,
                    resource_utilization={},
                    strategic_alignment_score=0.0,
                    risk_distribution={},
                    effort_distribution={},
                    rationale=["No improvements to prioritize"]
                )
            
            # Execute prioritization strategy
            strategy_func = self._strategies[strategy]
            prioritized_improvements = await strategy_func(
                improvements,
                resource_constraints,
                strategic_goals
            )
            
            # Apply selection limits
            if max_selections:
                prioritized_improvements = prioritized_improvements[:max_selections]
            
            # Calculate result metrics
            result = await self._calculate_prioritization_metrics(
                prioritized_improvements,
                strategy,
                resource_constraints,
                strategic_goals
            )
            
            self.logger.info(
                f"Prioritization complete: {len(result.prioritized_improvements)} improvements selected, "
                f"total EV: {result.total_expected_value:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prioritization failed: {e}")
            return PrioritizationResult(
                prioritized_improvements=improvements,  # Fallback to original order
                prioritization_strategy=strategy,
                total_expected_value=0.0,
                resource_utilization={},
                strategic_alignment_score=0.0,
                risk_distribution={},
                effort_distribution={},
                rationale=[f"Prioritization failed: {e}"]
            )
    
    async def _prioritize_by_expected_value(
        self,
        improvements: List[ImprovementOpportunity],
        resource_constraints: Optional[ResourceConstraints],
        strategic_goals: Optional[StrategicGoals]
    ) -> List[ImprovementOpportunity]:
        """Prioritize by expected value (impact × confidence / effort)."""
        
        def calculate_expected_value(improvement: ImprovementOpportunity) -> float:
            # Calculate impact score from expected benefits
            impact_score = sum(improvement.expected_benefits.values()) / max(len(improvement.expected_benefits), 1)
            
            # Calculate effort penalty
            effort_penalty = self._get_effort_penalty(improvement.effort)
            
            # Expected value formula
            expected_value = (impact_score * improvement.confidence) / effort_penalty
            
            return expected_value
        
        # Calculate expected values
        for improvement in improvements:
            improvement.priority_score = calculate_expected_value(improvement)
        
        # Sort by expected value (descending)
        return sorted(improvements, key=lambda x: x.priority_score, reverse=True)
    
    async def _prioritize_by_risk_adjusted_value(
        self,
        improvements: List[ImprovementOpportunity],
        resource_constraints: Optional[ResourceConstraints],
        strategic_goals: Optional[StrategicGoals]
    ) -> List[ImprovementOpportunity]:
        """Prioritize by risk-adjusted expected value."""
        
        def calculate_risk_adjusted_value(improvement: ImprovementOpportunity) -> float:
            # Base expected value
            impact_score = sum(improvement.expected_benefits.values()) / max(len(improvement.expected_benefits), 1)
            effort_penalty = self._get_effort_penalty(improvement.effort)
            base_value = (impact_score * improvement.confidence) / effort_penalty
            
            # Risk adjustment factor
            risk_penalty = self._get_risk_penalty(improvement.risk)
            
            # Risk-adjusted value
            return base_value / risk_penalty
        
        # Calculate risk-adjusted values
        for improvement in improvements:
            improvement.priority_score = calculate_risk_adjusted_value(improvement)
        
        return sorted(improvements, key=lambda x: x.priority_score, reverse=True)
    
    async def _prioritize_by_resource_constraints(
        self,
        improvements: List[ImprovementOpportunity],
        resource_constraints: Optional[ResourceConstraints],
        strategic_goals: Optional[StrategicGoals]
    ) -> List[ImprovementOpportunity]:
        """Prioritize considering resource constraints (knapsack optimization)."""
        if not resource_constraints:
            # Fall back to expected value if no constraints
            return await self._prioritize_by_expected_value(improvements, resource_constraints, strategic_goals)
        
        # Estimate resource requirements for each improvement
        for improvement in improvements:
            improvement.priority_score = self._calculate_resource_efficiency(improvement, resource_constraints)
        
        # Select improvements that fit within constraints
        selected_improvements = []
        remaining_time = resource_constraints.total_time_budget or timedelta(days=30)
        
        # Sort by resource efficiency
        sorted_improvements = sorted(improvements, key=lambda x: x.priority_score, reverse=True)
        
        for improvement in sorted_improvements:
            estimated_time = self._estimate_implementation_time(improvement, resource_constraints)
            
            if estimated_time <= remaining_time:
                selected_improvements.append(improvement)
                remaining_time -= estimated_time
        
        return selected_improvements
    
    async def _prioritize_by_strategic_alignment(
        self,
        improvements: List[ImprovementOpportunity],
        resource_constraints: Optional[ResourceConstraints],
        strategic_goals: Optional[StrategicGoals]
    ) -> List[ImprovementOpportunity]:
        """Prioritize by strategic alignment."""
        if not strategic_goals:
            return await self._prioritize_by_expected_value(improvements, resource_constraints, strategic_goals)
        
        def calculate_strategic_alignment(improvement: ImprovementOpportunity) -> float:
            alignment_score = 0.0
            
            # Component alignment
            if strategic_goals.priority_components:
                component_overlap = len(
                    set(improvement.affected_components) & strategic_goals.priority_components
                )
                alignment_score += component_overlap * 2.0
            
            # Initiative alignment
            for initiative, weight in strategic_goals.strategic_initiatives.items():
                if initiative.lower() in improvement.description.lower():
                    alignment_score += weight
            
            # Performance target alignment
            benefits = improvement.expected_benefits
            if strategic_goals.target_response_time_ms and 'response_time_improvement_percent' in benefits:
                alignment_score += benefits['response_time_improvement_percent'] * 0.1
            
            if strategic_goals.target_error_rate and 'error_rate_reduction_percent' in benefits:
                alignment_score += benefits['error_rate_reduction_percent'] * 0.1
            
            # Base expected value
            impact_score = sum(benefits.values()) / max(len(benefits), 1)
            effort_penalty = self._get_effort_penalty(improvement.effort)
            base_value = (impact_score * improvement.confidence) / effort_penalty
            
            # Combined score
            return base_value * (1 + alignment_score * 0.2)
        
        for improvement in improvements:
            improvement.priority_score = calculate_strategic_alignment(improvement)
        
        return sorted(improvements, key=lambda x: x.priority_score, reverse=True)
    
    async def _prioritize_quick_wins(
        self,
        improvements: List[ImprovementOpportunity],
        resource_constraints: Optional[ResourceConstraints],
        strategic_goals: Optional[StrategicGoals]
    ) -> List[ImprovementOpportunity]:
        """Prioritize quick wins (low effort, high impact)."""
        
        def calculate_quick_win_score(improvement: ImprovementOpportunity) -> float:
            # Heavily favor low effort
            effort_bonus = {
                ImplementationEffort.AUTOMATIC: 10.0,
                ImplementationEffort.MINIMAL: 8.0,
                ImplementationEffort.LOW: 6.0,
                ImplementationEffort.MEDIUM: 3.0,
                ImplementationEffort.HIGH: 1.0,
                ImplementationEffort.COMPLEX: 0.5
            }
            
            # Favor high impact
            impact_score = sum(improvement.expected_benefits.values()) / max(len(improvement.expected_benefits), 1)
            
            # Quick win score
            return impact_score * improvement.confidence * effort_bonus.get(improvement.effort, 1.0)
        
        for improvement in improvements:
            improvement.priority_score = calculate_quick_win_score(improvement)
        
        return sorted(improvements, key=lambda x: x.priority_score, reverse=True)
    
    async def _prioritize_balanced_portfolio(
        self,
        improvements: List[ImprovementOpportunity],
        resource_constraints: Optional[ResourceConstraints],
        strategic_goals: Optional[StrategicGoals]
    ) -> List[ImprovementOpportunity]:
        """Create balanced portfolio of improvements across risk/effort levels."""
        
        # Group improvements by risk/effort combinations
        portfolios = {}
        for improvement in improvements:
            key = (improvement.risk, improvement.effort)
            if key not in portfolios:
                portfolios[key] = []
            portfolios[key].append(improvement)
        
        # Select best from each portfolio bucket
        balanced_selection = []
        
        # Target distribution (rough guidelines)
        target_counts = {
            (RiskLevel.LOW, ImplementationEffort.LOW): 3,      # Safe quick wins
            (RiskLevel.LOW, ImplementationEffort.MEDIUM): 2,   # Safe improvements
            (RiskLevel.MEDIUM, ImplementationEffort.LOW): 2,   # Quick wins with some risk
            (RiskLevel.MEDIUM, ImplementationEffort.MEDIUM): 2, # Balanced improvements
            (RiskLevel.HIGH, ImplementationEffort.LOW): 1,     # High-risk quick win
        }
        
        for (risk, effort), count in target_counts.items():
            if (risk, effort) in portfolios:
                # Sort by expected value within bucket
                bucket_improvements = portfolios[(risk, effort)]
                bucket_improvements.sort(
                    key=lambda x: sum(x.expected_benefits.values()) * x.confidence,
                    reverse=True
                )
                balanced_selection.extend(bucket_improvements[:count])
        
        # Fill remaining slots with highest expected value
        remaining_improvements = [
            imp for imp in improvements if imp not in balanced_selection
        ]
        remaining_improvements.sort(
            key=lambda x: sum(x.expected_benefits.values()) * x.confidence,
            reverse=True
        )
        
        # Add up to 10 total improvements
        max_additional = max(0, 10 - len(balanced_selection))
        balanced_selection.extend(remaining_improvements[:max_additional])
        
        return balanced_selection
    
    def _get_effort_penalty(self, effort: ImplementationEffort) -> float:
        """Get effort penalty factor."""
        penalties = {
            ImplementationEffort.AUTOMATIC: 1.0,
            ImplementationEffort.MINIMAL: 1.2,
            ImplementationEffort.LOW: 1.5,
            ImplementationEffort.MEDIUM: 2.0,
            ImplementationEffort.HIGH: 3.0,
            ImplementationEffort.COMPLEX: 5.0
        }
        return penalties.get(effort, 2.0)
    
    def _get_risk_penalty(self, risk: RiskLevel) -> float:
        """Get risk penalty factor."""
        penalties = {
            RiskLevel.MINIMAL: 1.0,
            RiskLevel.LOW: 1.1,
            RiskLevel.MEDIUM: 1.3,
            RiskLevel.HIGH: 1.8,
            RiskLevel.CRITICAL: 3.0
        }
        return penalties.get(risk, 1.3)
    
    def _calculate_resource_efficiency(
        self,
        improvement: ImprovementOpportunity,
        resource_constraints: ResourceConstraints
    ) -> float:
        """Calculate resource efficiency score."""
        # Estimate time requirement
        estimated_time = self._estimate_implementation_time(improvement, resource_constraints)
        time_hours = estimated_time.total_seconds() / 3600
        
        # Calculate value per hour
        impact_score = sum(improvement.expected_benefits.values()) / max(len(improvement.expected_benefits), 1)
        expected_value = impact_score * improvement.confidence
        
        efficiency = expected_value / max(time_hours, 1)
        return efficiency
    
    def _estimate_implementation_time(
        self,
        improvement: ImprovementOpportunity,
        resource_constraints: ResourceConstraints
    ) -> timedelta:
        """Estimate implementation time based on effort and team capacity."""
        base_hours = {
            ImplementationEffort.AUTOMATIC: 1,
            ImplementationEffort.MINIMAL: 4,
            ImplementationEffort.LOW: 16,
            ImplementationEffort.MEDIUM: 40,
            ImplementationEffort.HIGH: 80,
            ImplementationEffort.COMPLEX: 160
        }
        
        hours = base_hours.get(improvement.effort, 40)
        
        # Adjust for team skill level
        if resource_constraints.developer_skill_levels:
            avg_skill = statistics.mean(resource_constraints.developer_skill_levels.values())
            hours = hours / max(avg_skill, 0.5)  # More skilled = faster
        
        # Adjust for team size (with diminishing returns)
        if resource_constraints.available_developers > 1:
            # Brooks' law: adding people to a late project makes it later
            efficiency = min(resource_constraints.available_developers, 3) * 0.8
            hours = hours / efficiency
        
        return timedelta(hours=hours)
    
    async def _calculate_prioritization_metrics(
        self,
        prioritized_improvements: List[ImprovementOpportunity],
        strategy: PrioritizationStrategy,
        resource_constraints: Optional[ResourceConstraints],
        strategic_goals: Optional[StrategicGoals]
    ) -> PrioritizationResult:
        """Calculate metrics for prioritization result."""
        
        # Total expected value
        total_expected_value = sum(
            sum(imp.expected_benefits.values()) * imp.confidence
            for imp in prioritized_improvements
        )
        
        # Resource utilization
        resource_utilization = {}
        if resource_constraints:
            total_time_needed = sum(
                self._estimate_implementation_time(imp, resource_constraints)
                for imp in prioritized_improvements
            )
            if resource_constraints.total_time_budget:
                resource_utilization['time'] = (
                    total_time_needed.total_seconds() / 
                    resource_constraints.total_time_budget.total_seconds()
                )
        
        # Strategic alignment score
        strategic_alignment_score = 0.0
        if strategic_goals and strategic_goals.priority_components:
            aligned_improvements = 0
            for improvement in prioritized_improvements:
                if any(comp in strategic_goals.priority_components 
                      for comp in improvement.affected_components):
                    aligned_improvements += 1
            strategic_alignment_score = aligned_improvements / len(prioritized_improvements)
        
        # Risk distribution
        risk_distribution = {}
        for improvement in prioritized_improvements:
            risk_distribution[improvement.risk] = risk_distribution.get(improvement.risk, 0) + 1
        
        # Effort distribution
        effort_distribution = {}
        for improvement in prioritized_improvements:
            effort_distribution[improvement.effort] = effort_distribution.get(improvement.effort, 0) + 1
        
        # Generate rationale
        rationale = [
            f"Used {strategy.value} prioritization strategy",
            f"Selected {len(prioritized_improvements)} improvements",
            f"Total expected value: {total_expected_value:.2f}",
            f"Strategic alignment: {strategic_alignment_score:.1%}"
        ]
        
        if resource_utilization.get('time'):
            rationale.append(f"Time utilization: {resource_utilization['time']:.1%}")
        
        return PrioritizationResult(
            prioritized_improvements=prioritized_improvements,
            prioritization_strategy=strategy,
            total_expected_value=total_expected_value,
            resource_utilization=resource_utilization,
            strategic_alignment_score=strategic_alignment_score,
            risk_distribution=risk_distribution,
            effort_distribution=effort_distribution,
            rationale=rationale
        )