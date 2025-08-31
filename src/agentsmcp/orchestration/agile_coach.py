"""Agile coach integration for planning, retrospectives, and team optimization.

This module provides comprehensive agile coaching capabilities including:
- Planning guidance for complex tasks
- Post-task retrospective facilitation 
- Team performance analysis and improvement suggestions
- Ceremony scheduling (daily standups, retrospectives, planning)
- Historical pattern recognition for process improvement
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import uuid

from pydantic import BaseModel, Field

from .models import (
    TaskClassification,
    TeamComposition,
    TeamPerformanceMetrics,
    TaskType,
    ComplexityLevel,
    RiskLevel,
    CoordinationStrategy,
    AgentSpec
)


class AgilePhase(str, Enum):
    """Agile ceremony phases."""
    PLANNING = "planning"
    DAILY_STANDUP = "daily_standup"
    RETROSPECTIVE = "retrospective" 
    POSTMORTEM = "postmortem"
    REVIEW = "review"
    REFINEMENT = "refinement"


class CeremonyType(str, Enum):
    """Types of agile ceremonies."""
    SPRINT_PLANNING = "sprint_planning"
    DAILY_STANDUP = "daily_standup"
    SPRINT_REVIEW = "sprint_review"
    SPRINT_RETROSPECTIVE = "sprint_retrospective"
    BACKLOG_REFINEMENT = "backlog_refinement"
    POSTMORTEM = "postmortem"


class ImprovementPriority(str, Enum):
    """Priority levels for improvement suggestions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CoachActions:
    """Actions recommended by the agile coach for task planning."""
    recommended_approach: str
    risk_mitigations: List[str]
    coordination_strategy: CoordinationStrategy
    suggested_milestones: List[str]
    team_adjustments: List[str]
    estimated_velocity: float
    confidence_score: float


@dataclass
class RetrospectiveReport:
    """Retrospective analysis and recommendations."""
    summary: str
    what_went_well: List[str]
    what_could_improve: List[str]
    action_items: List[str]
    team_health_score: float
    velocity_analysis: Dict[str, float]
    improvement_suggestions: List[str]
    next_sprint_recommendations: List[str]


class CeremonySchedule(BaseModel):
    """Schedule for agile ceremonies."""
    phase: AgilePhase = Field(..., description="Current agile phase")
    upcoming_ceremonies: List[Dict[str, Any]] = Field(default_factory=list)
    recommended_participants: List[str] = Field(default_factory=list)
    estimated_duration: int = Field(default=30, description="Duration in minutes")
    next_ceremony_time: Optional[datetime] = None


class ImprovementSuggestion(BaseModel):
    """Team improvement suggestion."""
    category: str = Field(..., description="Category of improvement")
    description: str = Field(..., description="Detailed suggestion")
    priority: ImprovementPriority = Field(..., description="Implementation priority")
    impact: str = Field(..., description="Expected impact")
    effort: str = Field(..., description="Implementation effort")
    success_criteria: List[str] = Field(default_factory=list)
    owner: Optional[str] = None


class TeamMetrics(BaseModel):
    """Comprehensive team performance metrics."""
    velocity_trend: List[float] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    collaboration_score: float = Field(default=0.0, ge=0.0, le=1.0)
    delivery_predictability: float = Field(default=0.0, ge=0.0, le=1.0)
    cycle_time_avg: float = Field(default=0.0)
    defect_rate: float = Field(default=0.0)
    team_satisfaction: float = Field(default=0.0, ge=0.0, le=1.0)
    learning_velocity: float = Field(default=0.0)


class AgileCoachIntegration:
    """Integration class for agile coaching capabilities."""
    
    def __init__(self):
        self.coaching_history: List[Dict[str, Any]] = []
        self.team_patterns: Dict[str, Any] = {}
        self.improvement_tracking: Dict[str, Any] = {}
    
    async def coach_planning(
        self, 
        task_classification: TaskClassification,
        team_composition: TeamComposition
    ) -> CoachActions:
        """Provide coaching guidance for task planning.
        
        Args:
            task_classification: Analysis of the task to be planned
            team_composition: Proposed team structure
            
        Returns:
            CoachActions with planning recommendations
        """
        # Analyze task complexity and team fit
        complexity_factor = self._assess_complexity_factor(task_classification.complexity)
        risk_factor = self._assess_risk_factor(task_classification.risk_level)
        team_size = len(team_composition.primary_team)
        
        # Generate coaching recommendations
        recommended_approach = self._determine_approach(
            task_classification.task_type,
            complexity_factor,
            team_size
        )
        
        risk_mitigations = self._generate_risk_mitigations(
            task_classification.risk_level,
            task_classification.task_type,
            team_composition.coordination_strategy
        )
        
        # Adjust coordination strategy based on task and team
        coordination_strategy = self._optimize_coordination_strategy(
            task_classification,
            team_composition
        )
        
        suggested_milestones = self._generate_milestones(
            task_classification.task_type,
            task_classification.estimated_effort
        )
        
        team_adjustments = self._suggest_team_adjustments(
            task_classification,
            team_composition
        )
        
        # Calculate estimated velocity based on historical data and team composition
        estimated_velocity = self._calculate_velocity_estimate(
            task_classification,
            team_composition
        )
        
        confidence_score = self._calculate_planning_confidence(
            task_classification,
            team_composition,
            risk_mitigations
        )
        
        # Store for future pattern recognition
        planning_record = {
            "timestamp": datetime.now(timezone.utc),
            "task_type": task_classification.task_type,
            "complexity": task_classification.complexity,
            "team_size": team_size,
            "coordination_strategy": coordination_strategy,
            "estimated_velocity": estimated_velocity
        }
        self.coaching_history.append(planning_record)
        
        return CoachActions(
            recommended_approach=recommended_approach,
            risk_mitigations=risk_mitigations,
            coordination_strategy=coordination_strategy,
            suggested_milestones=suggested_milestones,
            team_adjustments=team_adjustments,
            estimated_velocity=estimated_velocity,
            confidence_score=confidence_score
        )
    
    async def coach_retrospective(
        self,
        execution_results: Dict[str, Any],
        performance_metrics: TeamPerformanceMetrics,
        team_composition: TeamComposition
    ) -> RetrospectiveReport:
        """Facilitate post-task retrospective analysis.
        
        Args:
            execution_results: Results from task execution
            performance_metrics: Team performance data
            team_composition: Team structure that executed the task
            
        Returns:
            RetrospectiveReport with analysis and recommendations
        """
        # Analyze execution outcomes
        success_indicators = self._analyze_success_indicators(execution_results)
        performance_analysis = self._analyze_performance_metrics(performance_metrics)
        
        # Generate retrospective insights
        what_went_well = self._identify_successes(
            execution_results,
            success_indicators,
            team_composition
        )
        
        what_could_improve = self._identify_improvements(
            execution_results,
            performance_analysis,
            team_composition
        )
        
        action_items = self._generate_action_items(
            what_could_improve,
            performance_analysis
        )
        
        team_health_score = self._calculate_team_health_score(
            performance_metrics,
            execution_results
        )
        
        velocity_analysis = self._analyze_velocity_trends(performance_metrics)
        
        improvement_suggestions = self._generate_improvement_suggestions(
            what_could_improve,
            velocity_analysis,
            team_composition
        )
        
        next_sprint_recommendations = self._generate_next_sprint_recommendations(
            performance_analysis,
            team_health_score,
            improvement_suggestions
        )
        
        summary = self._generate_retrospective_summary(
            what_went_well,
            what_could_improve,
            team_health_score
        )
        
        # Update patterns for future coaching
        self._update_team_patterns(
            team_composition,
            performance_metrics,
            what_went_well,
            what_could_improve
        )
        
        return RetrospectiveReport(
            summary=summary,
            what_went_well=what_went_well,
            what_could_improve=what_could_improve,
            action_items=action_items,
            team_health_score=team_health_score,
            velocity_analysis=velocity_analysis,
            improvement_suggestions=improvement_suggestions,
            next_sprint_recommendations=next_sprint_recommendations
        )
    
    async def schedule_ceremonies(
        self,
        phase: AgilePhase,
        team: List[AgentSpec]
    ) -> CeremonySchedule:
        """Schedule appropriate agile ceremonies.
        
        Args:
            phase: Current phase in the agile process
            team: Team members for the ceremonies
            
        Returns:
            CeremonySchedule with ceremony details
        """
        upcoming_ceremonies = []
        recommended_participants = [agent.role for agent in team]
        estimated_duration = 30  # Default duration
        
        # Generate ceremony schedule based on phase
        if phase == AgilePhase.PLANNING:
            upcoming_ceremonies = [
                {
                    "type": CeremonyType.SPRINT_PLANNING,
                    "purpose": "Plan upcoming work items",
                    "duration": 60,
                    "required_participants": recommended_participants
                },
                {
                    "type": CeremonyType.BACKLOG_REFINEMENT,
                    "purpose": "Refine and estimate backlog items",
                    "duration": 30,
                    "required_participants": [p for p in recommended_participants if 'architect' in p.lower() or 'coder' in p.lower()]
                }
            ]
            estimated_duration = 60
            
        elif phase == AgilePhase.DAILY_STANDUP:
            upcoming_ceremonies = [
                {
                    "type": CeremonyType.DAILY_STANDUP,
                    "purpose": "Synchronize team progress and identify blockers",
                    "duration": 15,
                    "required_participants": recommended_participants
                }
            ]
            estimated_duration = 15
            
        elif phase == AgilePhase.RETROSPECTIVE:
            upcoming_ceremonies = [
                {
                    "type": CeremonyType.SPRINT_RETROSPECTIVE,
                    "purpose": "Reflect on team performance and identify improvements",
                    "duration": 45,
                    "required_participants": recommended_participants
                },
                {
                    "type": CeremonyType.SPRINT_REVIEW,
                    "purpose": "Demonstrate completed work and gather feedback",
                    "duration": 30,
                    "required_participants": recommended_participants
                }
            ]
            estimated_duration = 45
            
        elif phase == AgilePhase.POSTMORTEM:
            upcoming_ceremonies = [
                {
                    "type": CeremonyType.POSTMORTEM,
                    "purpose": "Analyze significant incidents or failures",
                    "duration": 60,
                    "required_participants": recommended_participants
                }
            ]
            estimated_duration = 60
        
        # Calculate next ceremony time
        next_ceremony_time = self._calculate_next_ceremony_time(phase, upcoming_ceremonies)
        
        return CeremonySchedule(
            phase=phase,
            upcoming_ceremonies=upcoming_ceremonies,
            recommended_participants=recommended_participants,
            estimated_duration=estimated_duration,
            next_ceremony_time=next_ceremony_time
        )
    
    async def suggest_improvements(
        self,
        team_metrics: TeamMetrics
    ) -> List[ImprovementSuggestion]:
        """Generate team improvement suggestions based on metrics.
        
        Args:
            team_metrics: Current team performance metrics
            
        Returns:
            List of prioritized improvement suggestions
        """
        suggestions = []
        
        # Analyze velocity trends
        if len(team_metrics.velocity_trend) >= 2:
            recent_velocity = sum(team_metrics.velocity_trend[-3:]) / min(3, len(team_metrics.velocity_trend))
            older_velocity = sum(team_metrics.velocity_trend[:-3] or team_metrics.velocity_trend) / max(1, len(team_metrics.velocity_trend[:-3] or team_metrics.velocity_trend))
            
            if recent_velocity < older_velocity * 0.8:
                suggestions.append(ImprovementSuggestion(
                    category="velocity",
                    description="Team velocity has declined. Consider investigating blockers, workload distribution, or technical debt.",
                    priority=ImprovementPriority.HIGH,
                    impact="Restore team productivity and delivery capability",
                    effort="Medium - requires investigation and targeted interventions",
                    success_criteria=["Velocity returns to historical average", "Cycle time improves", "Team reports fewer blockers"]
                ))
        
        # Quality score analysis
        if team_metrics.quality_score < 0.7:
            suggestions.append(ImprovementSuggestion(
                category="quality",
                description="Quality metrics indicate room for improvement. Consider enhanced testing, code reviews, or technical practices.",
                priority=ImprovementPriority.HIGH,
                impact="Reduce defects and rework, improve customer satisfaction",
                effort="Medium - may require process changes and tooling",
                success_criteria=["Quality score above 0.8", "Reduced defect rate", "Faster resolution times"]
            ))
        
        # Collaboration score analysis
        if team_metrics.collaboration_score < 0.6:
            suggestions.append(ImprovementSuggestion(
                category="collaboration",
                description="Team collaboration could be enhanced. Consider improving communication channels, pair programming, or knowledge sharing.",
                priority=ImprovementPriority.MEDIUM,
                impact="Better knowledge sharing, reduced silos, improved team cohesion",
                effort="Low to Medium - focus on practices and culture",
                success_criteria=["Collaboration score above 0.75", "Increased cross-team knowledge", "Better communication metrics"]
            ))
        
        # Delivery predictability analysis
        if team_metrics.delivery_predictability < 0.7:
            suggestions.append(ImprovementSuggestion(
                category="predictability",
                description="Delivery predictability needs improvement. Focus on better estimation, task breakdown, or scope management.",
                priority=ImprovementPriority.MEDIUM,
                impact="More reliable delivery commitments and better planning",
                effort="Medium - requires process refinement and team training",
                success_criteria=["Predictability score above 0.8", "Better sprint completion rates", "Improved estimation accuracy"]
            ))
        
        # Cycle time analysis
        if team_metrics.cycle_time_avg > 5.0:  # Assuming days
            suggestions.append(ImprovementSuggestion(
                category="flow",
                description="Cycle time is high. Consider reducing work-in-progress limits, eliminating bottlenecks, or improving handoffs.",
                priority=ImprovementPriority.MEDIUM,
                impact="Faster feedback loops and improved flow",
                effort="Medium - may require process and workflow changes",
                success_criteria=["Cycle time reduced by 25%", "Improved flow metrics", "Reduced waiting times"]
            ))
        
        # Team satisfaction analysis
        if team_metrics.team_satisfaction < 0.7:
            suggestions.append(ImprovementSuggestion(
                category="satisfaction",
                description="Team satisfaction is below optimal. Consider addressing workload, autonomy, skill development, or team dynamics.",
                priority=ImprovementPriority.HIGH,
                impact="Improved retention, motivation, and overall performance",
                effort="Variable - depends on root causes",
                success_criteria=["Team satisfaction above 0.8", "Improved retention metrics", "Better engagement scores"]
            ))
        
        # Learning velocity analysis
        if team_metrics.learning_velocity < 0.5:
            suggestions.append(ImprovementSuggestion(
                category="learning",
                description="Team learning velocity could be improved. Consider dedicated learning time, skill development programs, or knowledge sharing sessions.",
                priority=ImprovementPriority.MEDIUM,
                impact="Enhanced team capabilities and adaptability",
                effort="Medium - requires time allocation and structured approach",
                success_criteria=["Learning velocity above 0.7", "Increased skill assessments", "More knowledge sharing activities"]
            ))
        
        # Sort suggestions by priority
        priority_order = {
            ImprovementPriority.CRITICAL: 0,
            ImprovementPriority.HIGH: 1,
            ImprovementPriority.MEDIUM: 2,
            ImprovementPriority.LOW: 3
        }
        
        suggestions.sort(key=lambda x: priority_order[x.priority])
        
        return suggestions
    
    # Helper methods
    
    def _assess_complexity_factor(self, complexity: ComplexityLevel) -> float:
        """Convert complexity level to numeric factor."""
        factors = {
            ComplexityLevel.TRIVIAL: 0.2,
            ComplexityLevel.LOW: 0.4,
            ComplexityLevel.MEDIUM: 0.6,
            ComplexityLevel.HIGH: 0.8,
            ComplexityLevel.CRITICAL: 1.0
        }
        return factors.get(complexity, 0.6)
    
    def _assess_risk_factor(self, risk_level: RiskLevel) -> float:
        """Convert risk level to numeric factor."""
        factors = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 1.0
        }
        return factors.get(risk_level, 0.5)
    
    def _determine_approach(self, task_type: TaskType, complexity_factor: float, team_size: int) -> str:
        """Determine recommended approach based on task characteristics."""
        if complexity_factor > 0.8:
            return "Incremental approach with frequent checkpoints and risk mitigation"
        elif task_type in [TaskType.IMPLEMENTATION, TaskType.REFACTORING]:
            return "Test-driven development with continuous integration"
        elif task_type == TaskType.DESIGN:
            return "Collaborative design sessions with rapid prototyping"
        elif team_size > 5:
            return "Structured coordination with clear role definitions"
        else:
            return "Agile collaboration with daily synchronization"
    
    def _generate_risk_mitigations(
        self, 
        risk_level: RiskLevel, 
        task_type: TaskType,
        coordination_strategy: CoordinationStrategy
    ) -> List[str]:
        """Generate risk mitigation strategies."""
        mitigations = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            mitigations.extend([
                "Implement frequent checkpoint reviews",
                "Create rollback plans for critical changes",
                "Establish escalation procedures"
            ])
        
        if task_type == TaskType.IMPLEMENTATION:
            mitigations.extend([
                "Implement comprehensive testing strategy",
                "Use feature flags for gradual rollout"
            ])
        
        if coordination_strategy == CoordinationStrategy.PARALLEL:
            mitigations.append("Define clear interface contracts to avoid integration issues")
        
        return mitigations
    
    def _optimize_coordination_strategy(
        self,
        task_classification: TaskClassification,
        team_composition: TeamComposition
    ) -> CoordinationStrategy:
        """Optimize coordination strategy based on task and team characteristics."""
        if task_classification.complexity == ComplexityLevel.CRITICAL:
            return CoordinationStrategy.HIERARCHICAL
        elif len(team_composition.primary_team) > 6:
            return CoordinationStrategy.HIERARCHICAL
        elif task_classification.risk_level == RiskLevel.HIGH:
            return CoordinationStrategy.COLLABORATIVE
        else:
            return team_composition.coordination_strategy
    
    def _generate_milestones(self, task_type: TaskType, estimated_effort: int) -> List[str]:
        """Generate suggested milestones based on task type and effort."""
        milestones = []
        
        if task_type == TaskType.DESIGN:
            milestones = [
                "Requirements analysis complete",
                "High-level design approved",
                "Detailed design finalized",
                "Design review passed"
            ]
        elif task_type == TaskType.IMPLEMENTATION:
            milestones = [
                "Development environment setup",
                "Core functionality implemented",
                "Testing complete",
                "Code review approved"
            ]
        elif task_type == TaskType.TESTING:
            milestones = [
                "Test plan approved",
                "Test cases implemented",
                "Testing execution complete",
                "Results analyzed and reported"
            ]
        else:
            milestones = [
                f"25% completion milestone",
                f"50% completion milestone", 
                f"75% completion milestone",
                f"Final deliverables complete"
            ]
        
        return milestones
    
    def _suggest_team_adjustments(
        self,
        task_classification: TaskClassification,
        team_composition: TeamComposition
    ) -> List[str]:
        """Suggest team composition adjustments."""
        adjustments = []
        
        if task_classification.complexity == ComplexityLevel.CRITICAL and len(team_composition.primary_team) < 3:
            adjustments.append("Consider adding senior architect for critical complexity")
        
        if task_classification.risk_level == RiskLevel.HIGH and not any('qa' in agent.role.lower() for agent in team_composition.primary_team):
            adjustments.append("Add dedicated QA engineer for high-risk tasks")
        
        if len(team_composition.primary_team) > 8:
            adjustments.append("Consider splitting into smaller sub-teams to improve coordination")
        
        return adjustments
    
    def _calculate_velocity_estimate(
        self,
        task_classification: TaskClassification,
        team_composition: TeamComposition
    ) -> float:
        """Calculate estimated team velocity."""
        base_velocity = len(team_composition.primary_team) * 0.7  # Base assumption
        
        # Adjust for complexity
        complexity_multiplier = {
            ComplexityLevel.TRIVIAL: 1.5,
            ComplexityLevel.LOW: 1.2,
            ComplexityLevel.MEDIUM: 1.0,
            ComplexityLevel.HIGH: 0.8,
            ComplexityLevel.CRITICAL: 0.6
        }.get(task_classification.complexity, 1.0)
        
        # Adjust for coordination strategy
        coordination_multiplier = {
            CoordinationStrategy.SEQUENTIAL: 0.8,
            CoordinationStrategy.PARALLEL: 1.1,
            CoordinationStrategy.HIERARCHICAL: 0.9,
            CoordinationStrategy.COLLABORATIVE: 1.0,
            CoordinationStrategy.PIPELINE: 1.2
        }.get(team_composition.coordination_strategy, 1.0)
        
        return base_velocity * complexity_multiplier * coordination_multiplier
    
    def _calculate_planning_confidence(
        self,
        task_classification: TaskClassification,
        team_composition: TeamComposition,
        risk_mitigations: List[str]
    ) -> float:
        """Calculate confidence score for planning decisions."""
        base_confidence = 0.7
        
        # Adjust based on classification confidence
        classification_factor = task_classification.confidence * 0.3
        
        # Adjust based on team composition confidence
        composition_factor = team_composition.confidence_score * 0.3
        
        # Adjust based on risk mitigation coverage
        mitigation_factor = min(len(risk_mitigations) / 5.0, 1.0) * 0.4
        
        return min(base_confidence + classification_factor + composition_factor + mitigation_factor, 1.0)
    
    def _analyze_success_indicators(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze execution results for success indicators."""
        indicators = {
            "completion_rate": execution_results.get("completion_rate", 0.0),
            "quality_metrics": execution_results.get("quality_metrics", {}),
            "timeline_adherence": execution_results.get("timeline_adherence", 0.0),
            "stakeholder_satisfaction": execution_results.get("stakeholder_satisfaction", 0.0)
        }
        return indicators
    
    def _analyze_performance_metrics(self, performance_metrics: TeamPerformanceMetrics) -> Dict[str, Any]:
        """Analyze team performance metrics."""
        return {
            "success_rate_trend": "improving" if performance_metrics.success_rate > 0.8 else "needs_attention",
            "duration_efficiency": "good" if performance_metrics.average_duration and performance_metrics.average_duration < 3600 else "slow",
            "cost_efficiency": "good" if performance_metrics.average_cost and performance_metrics.average_cost < 10.0 else "high"
        }
    
    def _identify_successes(
        self,
        execution_results: Dict[str, Any],
        success_indicators: Dict[str, Any],
        team_composition: TeamComposition
    ) -> List[str]:
        """Identify what went well during execution."""
        successes = []
        
        if success_indicators.get("completion_rate", 0) > 0.9:
            successes.append("High completion rate achieved")
        
        if success_indicators.get("timeline_adherence", 0) > 0.8:
            successes.append("Good adherence to timeline")
        
        if team_composition.coordination_strategy == CoordinationStrategy.COLLABORATIVE:
            successes.append("Effective collaborative approach")
        
        if execution_results.get("defects", 0) < 2:
            successes.append("Low defect rate maintained")
        
        return successes or ["Task completion achieved"]
    
    def _identify_improvements(
        self,
        execution_results: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        team_composition: TeamComposition
    ) -> List[str]:
        """Identify areas that could be improved."""
        improvements = []
        
        if performance_analysis.get("duration_efficiency") == "slow":
            improvements.append("Execution time longer than expected - investigate bottlenecks")
        
        if performance_analysis.get("cost_efficiency") == "high":
            improvements.append("Cost higher than budgeted - review resource allocation")
        
        if execution_results.get("rework_count", 0) > 2:
            improvements.append("Multiple rework cycles - improve initial analysis and design")
        
        if len(team_composition.primary_team) > 5 and performance_analysis.get("success_rate_trend") == "needs_attention":
            improvements.append("Large team coordination challenges - consider team structure")
        
        return improvements
    
    def _generate_action_items(
        self,
        what_could_improve: List[str],
        performance_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate specific action items from improvement areas."""
        action_items = []
        
        for improvement in what_could_improve:
            if "bottleneck" in improvement.lower():
                action_items.append("Conduct bottleneck analysis and process optimization")
            elif "cost" in improvement.lower():
                action_items.append("Review and optimize resource allocation strategies")
            elif "rework" in improvement.lower():
                action_items.append("Implement enhanced upfront analysis and validation processes")
            elif "coordination" in improvement.lower():
                action_items.append("Refine team structure and communication protocols")
        
        return action_items
    
    def _calculate_team_health_score(
        self,
        performance_metrics: TeamPerformanceMetrics,
        execution_results: Dict[str, Any]
    ) -> float:
        """Calculate overall team health score."""
        success_rate_score = performance_metrics.success_rate
        timeline_score = execution_results.get("timeline_adherence", 0.7)
        quality_score = 1.0 - min(execution_results.get("defects", 0) / 10.0, 1.0)
        
        return (success_rate_score + timeline_score + quality_score) / 3.0
    
    def _analyze_velocity_trends(self, performance_metrics: TeamPerformanceMetrics) -> Dict[str, float]:
        """Analyze velocity and performance trends."""
        return {
            "current_success_rate": performance_metrics.success_rate,
            "avg_duration": performance_metrics.average_duration or 0.0,
            "avg_cost": performance_metrics.average_cost or 0.0,
            "total_executions": float(performance_metrics.total_executions)
        }
    
    def _generate_improvement_suggestions(
        self,
        what_could_improve: List[str],
        velocity_analysis: Dict[str, float],
        team_composition: TeamComposition
    ) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        for improvement in what_could_improve:
            if "time" in improvement.lower() or "duration" in improvement.lower():
                suggestions.append("Implement time-boxing and focus techniques")
            elif "cost" in improvement.lower():
                suggestions.append("Optimize team composition for cost efficiency")
            elif "quality" in improvement.lower():
                suggestions.append("Enhance testing and review processes")
        
        if velocity_analysis["avg_duration"] > 7200:  # 2 hours
            suggestions.append("Break down tasks into smaller, more manageable chunks")
        
        return suggestions
    
    def _generate_next_sprint_recommendations(
        self,
        performance_analysis: Dict[str, Any],
        team_health_score: float,
        improvement_suggestions: List[str]
    ) -> List[str]:
        """Generate recommendations for the next sprint."""
        recommendations = []
        
        if team_health_score < 0.7:
            recommendations.append("Focus on team health and process improvements before taking on complex tasks")
        
        if performance_analysis.get("success_rate_trend") == "needs_attention":
            recommendations.append("Reduce scope and focus on execution quality")
        
        if len(improvement_suggestions) > 3:
            recommendations.append("Prioritize top 2-3 improvement initiatives to avoid overload")
        
        recommendations.append("Continue regular retrospectives and team health monitoring")
        
        return recommendations
    
    def _generate_retrospective_summary(
        self,
        what_went_well: List[str],
        what_could_improve: List[str],
        team_health_score: float
    ) -> str:
        """Generate overall retrospective summary."""
        health_status = "excellent" if team_health_score > 0.8 else "good" if team_health_score > 0.6 else "needs attention"
        
        return f"""Retrospective Summary:
        
Team Health: {health_status} ({team_health_score:.2f})
Successes: {len(what_went_well)} key achievements identified
Improvement Areas: {len(what_could_improve)} areas for focus

The team demonstrated strong execution in several areas while identifying specific opportunities for enhancement. Focus on systematic improvement while maintaining current strengths."""
    
    def _update_team_patterns(
        self,
        team_composition: TeamComposition,
        performance_metrics: TeamPerformanceMetrics,
        what_went_well: List[str],
        what_could_improve: List[str]
    ) -> None:
        """Update historical patterns for future coaching."""
        pattern_key = f"{team_composition.coordination_strategy}_{len(team_composition.primary_team)}"
        
        if pattern_key not in self.team_patterns:
            self.team_patterns[pattern_key] = {
                "success_patterns": [],
                "challenge_patterns": [],
                "performance_history": []
            }
        
        self.team_patterns[pattern_key]["success_patterns"].extend(what_went_well)
        self.team_patterns[pattern_key]["challenge_patterns"].extend(what_could_improve)
        self.team_patterns[pattern_key]["performance_history"].append({
            "timestamp": datetime.now(timezone.utc),
            "success_rate": performance_metrics.success_rate,
            "duration": performance_metrics.average_duration,
            "cost": performance_metrics.average_cost
        })
    
    def _calculate_next_ceremony_time(
        self,
        phase: AgilePhase,
        upcoming_ceremonies: List[Dict[str, Any]]
    ) -> Optional[datetime]:
        """Calculate when the next ceremony should occur."""
        now = datetime.now(timezone.utc)
        
        if phase == AgilePhase.DAILY_STANDUP:
            # Next working day at 9 AM
            tomorrow = now + timedelta(days=1)
            return tomorrow.replace(hour=9, minute=0, second=0, microsecond=0)
        elif phase == AgilePhase.PLANNING:
            # Start of next week
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            next_monday = now + timedelta(days=days_until_monday)
            return next_monday.replace(hour=10, minute=0, second=0, microsecond=0)
        elif phase == AgilePhase.RETROSPECTIVE:
            # End of current week (Friday)
            days_until_friday = (4 - now.weekday()) % 7
            if days_until_friday == 0 and now.hour >= 16:
                days_until_friday = 7
            friday = now + timedelta(days=days_until_friday)
            return friday.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Default to next day for other phases
        return now + timedelta(days=1)