"""Retrospective engine for continuous improvement and learning.

This module provides comprehensive retrospective capabilities including:
- Post-task retrospective facilitation with all participating agents
- Performance analysis and improvement identification
- Actionable improvement actions generation
- Team pattern updates based on learnings
- Integration with agile coach for facilitation
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .models import (
    TeamComposition,
    TeamPerformanceMetrics,
    TaskClassification,
    AgentSpec,
    ComplexityLevel,
    RiskLevel,
)
from .execution_engine import TeamExecution, ExecutionStatus
from .feedback_collector import FeedbackCollector, AgentFeedback
from .agile_coach import RetrospectiveReport, ImprovementSuggestion, ImprovementPriority


class RetrospectiveType(str, Enum):
    """Types of retrospectives that can be conducted."""
    POST_TASK = "post_task"
    SPRINT = "sprint"
    INCIDENT = "incident"
    CONTINUOUS = "continuous"


class RetrospectiveScope(str, Enum):
    """Scope of retrospective analysis."""
    TASK_SPECIFIC = "task_specific"
    TEAM_FOCUSED = "team_focused"
    PROCESS_FOCUSED = "process_focused"
    COMPREHENSIVE = "comprehensive"


@dataclass
class RetrospectiveFacilitationConfig:
    """Configuration for retrospective facilitation."""
    timeout_seconds: int = 30
    require_all_agents: bool = False
    anonymize_feedback: bool = True
    include_performance_metrics: bool = True
    generate_action_items: bool = True
    auto_update_patterns: bool = True
    parallel_collection: bool = True


@dataclass
class ImprovementAction:
    """Specific improvement action item from retrospective."""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    category: str = ""
    priority: ImprovementPriority = ImprovementPriority.MEDIUM
    assigned_to: Optional[str] = None
    estimated_effort_hours: Optional[float] = None
    expected_impact: str = ""
    success_criteria: List[str] = field(default_factory=list)
    due_date: Optional[datetime] = None
    status: str = "open"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TeamPatternUpdate:
    """Update to team patterns based on retrospective learnings."""
    pattern_type: str
    old_pattern: Dict[str, Any]
    new_pattern: Dict[str, Any]
    confidence_change: float
    evidence: List[str]
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EnhancedRetrospectiveReport:
    """Enhanced retrospective report with detailed analysis."""
    retrospective_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    retrospective_type: RetrospectiveType = RetrospectiveType.POST_TASK
    scope: RetrospectiveScope = RetrospectiveScope.COMPREHENSIVE
    
    # Basic retrospective data
    execution_summary: str = ""
    what_went_well: List[str] = field(default_factory=list)
    what_could_improve: List[str] = field(default_factory=list)
    blockers_encountered: List[str] = field(default_factory=list)
    
    # Performance analysis
    performance_analysis: Dict[str, Any] = field(default_factory=dict)
    team_health_score: float = 0.0
    coordination_effectiveness: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    
    # Improvement items
    improvement_actions: List[ImprovementAction] = field(default_factory=list)
    pattern_updates: List[TeamPatternUpdate] = field(default_factory=list)
    
    # Feedback aggregation
    agent_feedback_summary: Dict[str, Any] = field(default_factory=dict)
    feedback_themes: List[str] = field(default_factory=list)
    
    # Metrics and trends
    velocity_impact: float = 0.0
    quality_impact: float = 0.0
    collaboration_score: float = 0.0
    learning_outcomes: List[str] = field(default_factory=list)
    
    # Future recommendations
    next_task_recommendations: List[str] = field(default_factory=list)
    team_composition_suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    facilitated_by: str = "retrospective_engine"
    participants: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class RetrospectiveEngine:
    """Main engine for conducting retrospectives and continuous improvement."""
    
    def __init__(
        self,
        feedback_collector: Optional[FeedbackCollector] = None,
        facilitation_config: Optional[RetrospectiveFacilitationConfig] = None,
    ):
        self.log = logging.getLogger(__name__)
        self.feedback_collector = feedback_collector or FeedbackCollector()
        self.facilitation_config = facilitation_config or RetrospectiveFacilitationConfig()
        
        # State tracking
        self.retrospective_history: List[EnhancedRetrospectiveReport] = []
        self.team_patterns: Dict[str, Dict[str, Any]] = {}
        self.improvement_tracking: Dict[str, List[ImprovementAction]] = {}
        
        self.log.info("RetrospectiveEngine initialized")
    
    async def conduct_retrospective(
        self,
        execution_results: Dict[str, Any],
        performance_metrics: TeamPerformanceMetrics,
        team_composition: TeamComposition,
        task_classification: TaskClassification,
        retrospective_type: RetrospectiveType = RetrospectiveType.POST_TASK,
        scope: RetrospectiveScope = RetrospectiveScope.COMPREHENSIVE,
    ) -> EnhancedRetrospectiveReport:
        """Conduct a comprehensive retrospective session.
        
        Args:
            execution_results: Results from task execution
            performance_metrics: Team performance data
            team_composition: Team that executed the task
            task_classification: Classification of the executed task
            retrospective_type: Type of retrospective to conduct
            scope: Scope of analysis
            
        Returns:
            EnhancedRetrospectiveReport with findings and recommendations
        """
        start_time = datetime.now(timezone.utc)
        
        self.log.info("Starting %s retrospective with %s scope", 
                     retrospective_type.value, scope.value)
        
        # Initialize retrospective report
        report = EnhancedRetrospectiveReport(
            retrospective_type=retrospective_type,
            scope=scope,
            participants=[agent.role for agent in team_composition.primary_team],
        )
        
        try:
            # Phase 1: Collect agent feedback
            if self.facilitation_config.parallel_collection:
                feedback_data = await self._collect_feedback_parallel(
                    team_composition.primary_team, execution_results
                )
            else:
                feedback_data = await self._collect_feedback_sequential(
                    team_composition.primary_team, execution_results
                )
            
            # Phase 2: Analyze execution and performance
            performance_analysis = await self._analyze_performance(
                execution_results, performance_metrics, task_classification
            )
            report.performance_analysis = performance_analysis
            
            # Phase 3: Synthesize feedback and identify themes
            feedback_synthesis = await self._synthesize_feedback(feedback_data)
            report.agent_feedback_summary = feedback_synthesis["summary"]
            report.feedback_themes = feedback_synthesis["themes"]
            
            # Phase 4: Extract what went well and improvements
            report.what_went_well = await self._extract_successes(
                feedback_data, performance_analysis
            )
            report.what_could_improve = await self._extract_improvements(
                feedback_data, performance_analysis
            )
            report.blockers_encountered = await self._extract_blockers(feedback_data)
            
            # Phase 5: Generate improvement actions
            if self.facilitation_config.generate_action_items:
                report.improvement_actions = await self._generate_improvement_actions(
                    report, task_classification
                )
            
            # Phase 6: Update team patterns if enabled
            if self.facilitation_config.auto_update_patterns:
                report.pattern_updates = await self._update_team_patterns(
                    team_composition, performance_analysis, feedback_data
                )
            
            # Phase 7: Calculate scores and metrics
            report.team_health_score = await self._calculate_team_health_score(
                feedback_data, performance_analysis
            )
            report.coordination_effectiveness = await self._calculate_coordination_effectiveness(
                team_composition, feedback_data
            )
            
            # Phase 8: Generate future recommendations
            report.next_task_recommendations = await self._generate_task_recommendations(
                performance_analysis, task_classification
            )
            report.team_composition_suggestions = await self._generate_team_suggestions(
                team_composition, feedback_data, performance_analysis
            )
            
        except Exception as e:
            self.log.error("Retrospective failed: %s", e)
            report.execution_summary = f"Retrospective failed: {str(e)}"
            raise
        
        finally:
            # Finalize report
            end_time = datetime.now(timezone.utc)
            report.completed_at = end_time
            report.duration_seconds = (end_time - start_time).total_seconds()
            
            # Store in history
            self.retrospective_history.append(report)
            
            self.log.info("Retrospective completed in %.2fs with %d improvement actions",
                         report.duration_seconds, len(report.improvement_actions))
        
        return report
    
    async def _collect_feedback_parallel(
        self,
        agent_specs: List[AgentSpec],
        execution_results: Dict[str, Any],
    ) -> Dict[str, AgentFeedback]:
        """Collect feedback from all agents in parallel."""
        
        self.log.debug("Collecting feedback from %d agents in parallel", len(agent_specs))
        
        # Create tasks for parallel feedback collection
        feedback_tasks = []
        for agent_spec in agent_specs:
            task = asyncio.create_task(
                self.feedback_collector.collect_agent_feedback(
                    [agent_spec], execution_results
                )
            )
            feedback_tasks.append((agent_spec.role, task))
        
        # Wait for all feedback with timeout
        feedback_data = {}
        for role, task in feedback_tasks:
            try:
                result = await asyncio.wait_for(
                    task, timeout=self.facilitation_config.timeout_seconds
                )
                if result:
                    feedback_data.update(result)
            except asyncio.TimeoutError:
                self.log.warning("Feedback collection timeout for agent: %s", role)
                if self.facilitation_config.require_all_agents:
                    raise
            except Exception as e:
                self.log.error("Feedback collection failed for agent %s: %s", role, e)
                if self.facilitation_config.require_all_agents:
                    raise
        
        return feedback_data
    
    async def _collect_feedback_sequential(
        self,
        agent_specs: List[AgentSpec],
        execution_results: Dict[str, Any],
    ) -> Dict[str, AgentFeedback]:
        """Collect feedback from agents sequentially."""
        
        self.log.debug("Collecting feedback from %d agents sequentially", len(agent_specs))
        
        feedback_data = {}
        for agent_spec in agent_specs:
            try:
                result = await asyncio.wait_for(
                    self.feedback_collector.collect_agent_feedback([agent_spec], execution_results),
                    timeout=self.facilitation_config.timeout_seconds
                )
                if result:
                    feedback_data.update(result)
            except asyncio.TimeoutError:
                self.log.warning("Feedback collection timeout for agent: %s", agent_spec.role)
                if self.facilitation_config.require_all_agents:
                    raise
            except Exception as e:
                self.log.error("Feedback collection failed for agent %s: %s", agent_spec.role, e)
                if self.facilitation_config.require_all_agents:
                    raise
        
        return feedback_data
    
    async def _analyze_performance(
        self,
        execution_results: Dict[str, Any],
        performance_metrics: TeamPerformanceMetrics,
        task_classification: TaskClassification,
    ) -> Dict[str, Any]:
        """Analyze team performance and execution results."""
        
        analysis = {
            "execution_status": execution_results.get("status", "unknown"),
            "completion_time": execution_results.get("duration_seconds", 0.0),
            "success_rate": performance_metrics.success_rate,
            "cost_efficiency": 0.0,
            "quality_indicators": {},
            "bottlenecks": [],
            "resource_usage": execution_results.get("resource_usage", {}),
        }
        
        # Calculate cost efficiency
        if performance_metrics.average_cost and performance_metrics.average_cost > 0:
            expected_cost = self._estimate_expected_cost(task_classification)
            analysis["cost_efficiency"] = min(1.0, expected_cost / performance_metrics.average_cost)
        
        # Identify bottlenecks
        if "task_timings" in execution_results:
            analysis["bottlenecks"] = await self._identify_bottlenecks(
                execution_results["task_timings"]
            )
        
        # Quality indicators
        analysis["quality_indicators"] = {
            "error_rate": len(execution_results.get("errors", [])) / max(1, execution_results.get("total_tasks", 1)),
            "completion_rate": execution_results.get("completed_tasks", 0) / max(1, execution_results.get("total_tasks", 1)),
            "team_coordination": self._assess_team_coordination(execution_results),
        }
        
        return analysis
    
    async def _synthesize_feedback(
        self, feedback_data: Dict[str, AgentFeedback]
    ) -> Dict[str, Any]:
        """Synthesize feedback from all agents to identify common themes."""
        
        themes = set()
        positive_feedback = []
        negative_feedback = []
        suggestions = []
        
        for role, feedback in feedback_data.items():
            # Extract themes from feedback text analysis
            if hasattr(feedback, 'themes'):
                themes.update(feedback.themes)
            
            # Collect positive and negative feedback
            if hasattr(feedback, 'what_went_well'):
                positive_feedback.extend(feedback.what_went_well)
            if hasattr(feedback, 'what_could_improve'):
                negative_feedback.extend(feedback.what_could_improve)
            if hasattr(feedback, 'suggestions'):
                suggestions.extend(feedback.suggestions)
        
        return {
            "summary": {
                "total_responses": len(feedback_data),
                "response_rate": len(feedback_data) / max(1, len(feedback_data)),
                "avg_satisfaction": sum(
                    getattr(f, 'satisfaction_score', 0.5) for f in feedback_data.values()
                ) / len(feedback_data) if feedback_data else 0.0,
            },
            "themes": list(themes),
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "suggestions": suggestions,
        }
    
    async def _extract_successes(
        self,
        feedback_data: Dict[str, AgentFeedback],
        performance_analysis: Dict[str, Any],
    ) -> List[str]:
        """Extract what went well from feedback and performance data."""
        
        successes = []
        
        # From performance analysis
        if performance_analysis.get("success_rate", 0) > 0.8:
            successes.append("High task success rate achieved")
        
        if performance_analysis.get("cost_efficiency", 0) > 0.9:
            successes.append("Excellent cost efficiency")
        
        if performance_analysis["quality_indicators"].get("completion_rate", 0) > 0.9:
            successes.append("Strong task completion rate")
        
        # From agent feedback
        for role, feedback in feedback_data.items():
            if hasattr(feedback, 'what_went_well'):
                successes.extend([f"{role}: {item}" for item in feedback.what_went_well])
        
        return successes
    
    async def _extract_improvements(
        self,
        feedback_data: Dict[str, AgentFeedback],
        performance_analysis: Dict[str, Any],
    ) -> List[str]:
        """Extract improvement areas from feedback and performance data."""
        
        improvements = []
        
        # From performance analysis
        if performance_analysis.get("cost_efficiency", 1.0) < 0.7:
            improvements.append("Improve cost efficiency through better resource utilization")
        
        if performance_analysis["quality_indicators"].get("error_rate", 0) > 0.1:
            improvements.append("Reduce error rate through better validation and testing")
        
        if performance_analysis.get("bottlenecks"):
            improvements.append("Address identified bottlenecks in execution flow")
        
        # From agent feedback
        for role, feedback in feedback_data.items():
            if hasattr(feedback, 'what_could_improve'):
                improvements.extend([f"{role}: {item}" for item in feedback.what_could_improve])
        
        return improvements
    
    async def _extract_blockers(
        self, feedback_data: Dict[str, AgentFeedback]
    ) -> List[str]:
        """Extract blockers and impediments from agent feedback."""
        
        blockers = []
        for role, feedback in feedback_data.items():
            if hasattr(feedback, 'blockers'):
                blockers.extend([f"{role}: {blocker}" for blocker in feedback.blockers])
        
        return blockers
    
    async def _generate_improvement_actions(
        self,
        report: EnhancedRetrospectiveReport,
        task_classification: TaskClassification,
    ) -> List[ImprovementAction]:
        """Generate specific, actionable improvement items."""
        
        actions = []
        
        # Generate actions from improvement areas
        for improvement in report.what_could_improve:
            action = ImprovementAction(
                title=f"Address: {improvement[:50]}...",
                description=improvement,
                category=self._categorize_improvement(improvement),
                priority=self._prioritize_improvement(improvement, task_classification),
                estimated_effort_hours=self._estimate_effort(improvement),
                expected_impact="Improve team performance and delivery quality",
            )
            actions.append(action)
        
        # Generate actions from blockers
        for blocker in report.blockers_encountered:
            action = ImprovementAction(
                title=f"Remove blocker: {blocker[:50]}...",
                description=f"Address and prevent: {blocker}",
                category="blocker_removal",
                priority=ImprovementPriority.HIGH,
                estimated_effort_hours=2.0,
                expected_impact="Remove impediment to team velocity",
            )
            actions.append(action)
        
        return actions
    
    async def _update_team_patterns(
        self,
        team_composition: TeamComposition,
        performance_analysis: Dict[str, Any],
        feedback_data: Dict[str, AgentFeedback],
    ) -> List[TeamPatternUpdate]:
        """Update team patterns based on retrospective learnings."""
        
        updates = []
        team_key = self._generate_team_key(team_composition)
        
        # Update success patterns
        if performance_analysis.get("success_rate", 0) > 0.8:
            old_pattern = self.team_patterns.get(f"{team_key}_success", {})
            new_pattern = {
                **old_pattern,
                "coordination_strategy": team_composition.coordination_strategy.value,
                "team_size": len(team_composition.primary_team),
                "roles": [agent.role for agent in team_composition.primary_team],
                "success_count": old_pattern.get("success_count", 0) + 1,
                "last_success": datetime.now(timezone.utc).isoformat(),
            }
            
            update = TeamPatternUpdate(
                pattern_type="success_pattern",
                old_pattern=old_pattern,
                new_pattern=new_pattern,
                confidence_change=0.1,
                evidence=[f"Success rate: {performance_analysis['success_rate']}"],
            )
            updates.append(update)
            self.team_patterns[f"{team_key}_success"] = new_pattern
        
        return updates
    
    async def _calculate_team_health_score(
        self,
        feedback_data: Dict[str, AgentFeedback],
        performance_analysis: Dict[str, Any],
    ) -> float:
        """Calculate overall team health score."""
        
        scores = []
        
        # Performance-based scores
        scores.append(performance_analysis.get("success_rate", 0.0))
        scores.append(performance_analysis.get("cost_efficiency", 0.0))
        scores.append(performance_analysis["quality_indicators"].get("completion_rate", 0.0))
        
        # Feedback-based scores
        if feedback_data:
            avg_satisfaction = sum(
                getattr(f, 'satisfaction_score', 0.5) for f in feedback_data.values()
            ) / len(feedback_data)
            scores.append(avg_satisfaction)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    async def _calculate_coordination_effectiveness(
        self,
        team_composition: TeamComposition,
        feedback_data: Dict[str, AgentFeedback],
    ) -> float:
        """Calculate how effectively the team coordinated."""
        
        effectiveness_indicators = []
        
        # Strategy alignment
        if team_composition.coordination_strategy in [
            "parallel", "collaborative"
        ]:
            effectiveness_indicators.append(0.8)
        else:
            effectiveness_indicators.append(0.6)
        
        # Feedback on coordination
        coordination_feedback = 0.0
        for feedback in feedback_data.values():
            if hasattr(feedback, 'coordination_rating'):
                coordination_feedback += feedback.coordination_rating
        
        if feedback_data:
            coordination_feedback /= len(feedback_data)
            effectiveness_indicators.append(coordination_feedback)
        
        return sum(effectiveness_indicators) / len(effectiveness_indicators)
    
    async def _generate_task_recommendations(
        self,
        performance_analysis: Dict[str, Any],
        task_classification: TaskClassification,
    ) -> List[str]:
        """Generate recommendations for future similar tasks."""
        
        recommendations = []
        
        if performance_analysis.get("cost_efficiency", 1.0) < 0.8:
            recommendations.append("Consider simpler coordination strategy for cost optimization")
        
        if performance_analysis["quality_indicators"].get("error_rate", 0) > 0.1:
            recommendations.append("Add more validation checkpoints for quality assurance")
        
        if task_classification.complexity == ComplexityLevel.HIGH:
            recommendations.append("Break down complex tasks into smaller, manageable chunks")
        
        return recommendations
    
    async def _generate_team_suggestions(
        self,
        team_composition: TeamComposition,
        feedback_data: Dict[str, AgentFeedback],
        performance_analysis: Dict[str, Any],
    ) -> List[str]:
        """Generate suggestions for team composition improvements."""
        
        suggestions = []
        
        if len(team_composition.primary_team) > 5:
            suggestions.append("Consider smaller team size for better coordination")
        
        if performance_analysis.get("coordination_effectiveness", 0) < 0.7:
            suggestions.append("Review coordination strategy effectiveness")
        
        # Role-specific suggestions from feedback
        role_performance = {}
        for role, feedback in feedback_data.items():
            if hasattr(feedback, 'performance_self_assessment'):
                role_performance[role] = feedback.performance_self_assessment
        
        if role_performance:
            low_performers = [role for role, score in role_performance.items() if score < 0.6]
            if low_performers:
                suggestions.append(f"Consider additional support or training for roles: {', '.join(low_performers)}")
        
        return suggestions
    
    # Helper methods
    
    def _estimate_expected_cost(self, task_classification: TaskClassification) -> float:
        """Estimate expected cost based on task classification."""
        base_cost = 1.0
        complexity_multiplier = {
            ComplexityLevel.TRIVIAL: 0.2,
            ComplexityLevel.LOW: 0.5,
            ComplexityLevel.MEDIUM: 1.0,
            ComplexityLevel.HIGH: 2.0,
            ComplexityLevel.CRITICAL: 3.0,
        }
        return base_cost * complexity_multiplier.get(task_classification.complexity, 1.0)
    
    async def _identify_bottlenecks(self, task_timings: Dict[str, float]) -> List[str]:
        """Identify bottlenecks from task execution timings."""
        if not task_timings:
            return []
        
        avg_time = sum(task_timings.values()) / len(task_timings)
        bottlenecks = []
        
        for task, duration in task_timings.items():
            if duration > avg_time * 1.5:
                bottlenecks.append(f"Task '{task}' took {duration:.2f}s (avg: {avg_time:.2f}s)")
        
        return bottlenecks
    
    def _assess_team_coordination(self, execution_results: Dict[str, Any]) -> float:
        """Assess team coordination quality from execution results."""
        # Simple heuristic based on task completion patterns
        completed_tasks = execution_results.get("completed_tasks", 0)
        total_tasks = execution_results.get("total_tasks", 1)
        failed_tasks = execution_results.get("failed_tasks", 0)
        
        if total_tasks == 0:
            return 0.0
        
        completion_rate = completed_tasks / total_tasks
        failure_rate = failed_tasks / total_tasks
        
        # Good coordination = high completion, low failures
        coordination_score = completion_rate * (1.0 - failure_rate)
        return min(1.0, max(0.0, coordination_score))
    
    def _categorize_improvement(self, improvement: str) -> str:
        """Categorize improvement based on content."""
        improvement_lower = improvement.lower()
        
        if any(word in improvement_lower for word in ["cost", "resource", "efficient"]):
            return "cost_optimization"
        elif any(word in improvement_lower for word in ["quality", "error", "test"]):
            return "quality_improvement"
        elif any(word in improvement_lower for word in ["coordination", "communication", "team"]):
            return "team_coordination"
        elif any(word in improvement_lower for word in ["process", "workflow", "procedure"]):
            return "process_improvement"
        else:
            return "general_improvement"
    
    def _prioritize_improvement(
        self, improvement: str, task_classification: TaskClassification
    ) -> ImprovementPriority:
        """Determine priority of improvement based on content and context."""
        improvement_lower = improvement.lower()
        
        # High-risk tasks get higher priority improvements
        if task_classification.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            if any(word in improvement_lower for word in ["error", "failure", "critical"]):
                return ImprovementPriority.CRITICAL
        
        if any(word in improvement_lower for word in ["blocker", "critical", "urgent"]):
            return ImprovementPriority.HIGH
        elif any(word in improvement_lower for word in ["cost", "efficiency", "optimize"]):
            return ImprovementPriority.MEDIUM
        else:
            return ImprovementPriority.LOW
    
    def _estimate_effort(self, improvement: str) -> float:
        """Estimate effort required for improvement in hours."""
        improvement_lower = improvement.lower()
        
        if any(word in improvement_lower for word in ["refactor", "redesign", "overhaul"]):
            return 8.0
        elif any(word in improvement_lower for word in ["implement", "create", "develop"]):
            return 4.0
        elif any(word in improvement_lower for word in ["adjust", "tweak", "modify"]):
            return 2.0
        else:
            return 1.0
    
    def _generate_team_key(self, team_composition: TeamComposition) -> str:
        """Generate a key for team pattern tracking."""
        roles = sorted([agent.role for agent in team_composition.primary_team])
        return f"{'-'.join(roles)}_{team_composition.coordination_strategy.value}"
    
    # Public API methods
    
    def get_retrospective_history(self, limit: Optional[int] = None) -> List[EnhancedRetrospectiveReport]:
        """Get retrospective history with optional limit."""
        if limit:
            return self.retrospective_history[-limit:]
        return self.retrospective_history.copy()
    
    def get_team_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get current team patterns."""
        return self.team_patterns.copy()
    
    def get_improvement_tracking(self) -> Dict[str, List[ImprovementAction]]:
        """Get improvement action tracking."""
        return self.improvement_tracking.copy()
    
    async def update_improvement_action_status(
        self, action_id: str, new_status: str, notes: Optional[str] = None
    ) -> bool:
        """Update the status of an improvement action."""
        for team_actions in self.improvement_tracking.values():
            for action in team_actions:
                if action.action_id == action_id:
                    action.status = new_status
                    if notes:
                        action.description += f"\n\nUpdate: {notes}"
                    self.log.info("Updated improvement action %s to status: %s", action_id, new_status)
                    return True
        return False