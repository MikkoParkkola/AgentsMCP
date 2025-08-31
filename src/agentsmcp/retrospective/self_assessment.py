"""Agent self-assessment system for structured performance evaluation.

This module provides comprehensive self-assessment capabilities for agents to evaluate
their own performance, decisions, tool usage, and identify learning opportunities.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..roles.base import RoleName
from .data_models import (
    PerformanceAssessment,
    DecisionPoint,
    ToolUsage,
    Challenge,
    SelfImprovementAction,
    ImprovementCategory,
    PriorityLevel,
)


class SelfAssessmentError(Exception):
    """Raised when self-assessment process fails."""
    pass


class DecisionHistoryIncomplete(Exception):
    """Raised when decision history is incomplete or invalid."""
    pass


class ToolUsageAnalysisFailure(Exception):
    """Raised when tool usage analysis fails."""
    pass


class AgentSelfAssessmentSystem:
    """System for conducting structured agent self-assessments."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log = logger or logging.getLogger(__name__)
        
        # Assessment weights by role (can be customized)
        self.role_weights = {
            RoleName.ARCHITECT: {
                "design_quality": 0.3,
                "decision_effectiveness": 0.3,
                "collaboration": 0.2,
                "learning": 0.2,
            },
            RoleName.CODER: {
                "code_quality": 0.4,
                "efficiency": 0.3,
                "tool_usage": 0.2,
                "learning": 0.1,
            },
            RoleName.QA: {
                "validation_thoroughness": 0.4,
                "issue_detection": 0.3,
                "process_adherence": 0.2,
                "learning": 0.1,
            },
        }
        
        self.log.info("AgentSelfAssessmentSystem initialized")
    
    async def conduct_performance_assessment(
        self,
        agent_role: RoleName,
        task_execution_data: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> PerformanceAssessment:
        """Conduct comprehensive performance self-assessment.
        
        Args:
            agent_role: The role of the agent being assessed
            task_execution_data: Execution metrics and outputs
            execution_context: Additional context about execution environment
            
        Returns:
            PerformanceAssessment: Structured self-evaluation
            
        Raises:
            SelfAssessmentError: If assessment process fails
        """
        try:
            self.log.debug("Starting performance assessment for role: %s", agent_role.value)
            
            assessment = PerformanceAssessment()
            
            # Calculate component scores based on role and execution data
            component_scores = await self._calculate_component_scores(
                agent_role, task_execution_data, execution_context
            )
            
            # Set component scores
            assessment.efficiency_score = component_scores.get("efficiency", 0.7)
            assessment.quality_score = component_scores.get("quality", 0.7)
            assessment.collaboration_score = component_scores.get("collaboration", 0.7)
            assessment.learning_score = component_scores.get("learning", 0.7)
            
            # Calculate overall score using role-specific weights
            assessment.overall_score = await self._calculate_overall_score(
                agent_role, component_scores
            )
            
            # Generate assessment notes and insights
            assessment.self_assessment_notes = await self._generate_assessment_notes(
                agent_role, component_scores, task_execution_data
            )
            
            # Identify strengths and improvement areas
            assessment.areas_of_strength = await self._identify_strengths(
                component_scores, agent_role
            )
            assessment.areas_for_improvement = await self._identify_improvement_areas(
                component_scores, agent_role
            )
            
            self.log.debug(
                "Performance assessment completed for %s with overall score: %.2f",
                agent_role.value,
                assessment.overall_score
            )
            
            return assessment
            
        except Exception as e:
            self.log.error("Performance assessment failed for %s: %s", agent_role.value, e)
            raise SelfAssessmentError(f"Assessment failed: {str(e)}")
    
    async def analyze_decisions(
        self,
        agent_role: RoleName,
        decision_history: List[Dict[str, Any]],
        task_outcome: Dict[str, Any],
    ) -> List[DecisionPoint]:
        """Analyze major decisions made during task execution.
        
        Args:
            agent_role: The role of the decision-making agent
            decision_history: History of decisions made
            task_outcome: Final outcome of the task
            
        Returns:
            List[DecisionPoint]: Analysis of decisions made
            
        Raises:
            DecisionHistoryIncomplete: If decision history is invalid
        """
        if not decision_history:
            raise DecisionHistoryIncomplete("Decision history is empty or missing")
        
        self.log.debug("Analyzing %d decisions for role: %s", len(decision_history), agent_role.value)
        
        analyzed_decisions = []
        
        for i, decision_data in enumerate(decision_history):
            try:
                decision = DecisionPoint(
                    timestamp=self._parse_timestamp(decision_data.get("timestamp")),
                    decision_description=decision_data.get("description", f"Decision #{i+1}"),
                    options_considered=decision_data.get("options", []),
                    chosen_option=decision_data.get("chosen_option", "Unknown"),
                    rationale=decision_data.get("rationale", "No rationale provided"),
                    confidence_level=float(decision_data.get("confidence", 0.5)),
                )
                
                # Analyze decision outcome based on task result
                decision.outcome = await self._analyze_decision_outcome(
                    decision, task_outcome, agent_role
                )
                
                analyzed_decisions.append(decision)
                
            except Exception as e:
                self.log.warning("Failed to analyze decision %d: %s", i, e)
                continue
        
        return analyzed_decisions
    
    async def analyze_tool_usage(
        self,
        agent_role: RoleName,
        tool_usage_log: List[Dict[str, Any]],
        task_metrics: Dict[str, Any],
    ) -> List[ToolUsage]:
        """Analyze tool usage patterns and effectiveness.
        
        Args:
            agent_role: The role of the agent using tools
            tool_usage_log: Log of tool usage during execution
            task_metrics: Metrics about overall task performance
            
        Returns:
            List[ToolUsage]: Analysis of tool usage patterns
            
        Raises:
            ToolUsageAnalysisFailure: If analysis fails
        """
        if not tool_usage_log:
            self.log.info("No tool usage data available for analysis")
            return []
        
        try:
            self.log.debug("Analyzing tool usage for role: %s", agent_role.value)
            
            # Aggregate tool usage by tool name
            tool_aggregates = {}
            
            for usage_entry in tool_usage_log:
                tool_name = usage_entry.get("tool_name", "unknown")
                
                if tool_name not in tool_aggregates:
                    tool_aggregates[tool_name] = {
                        "count": 0,
                        "successes": 0,
                        "total_time": 0.0,
                        "outcomes": [],
                    }
                
                aggregate = tool_aggregates[tool_name]
                aggregate["count"] += 1
                
                if usage_entry.get("success", True):
                    aggregate["successes"] += 1
                
                aggregate["total_time"] += usage_entry.get("duration", 0.0)
                
                if "outcome" in usage_entry:
                    aggregate["outcomes"].append(usage_entry["outcome"])
            
            # Create ToolUsage objects with analysis
            tool_usage_analysis = []
            
            for tool_name, aggregate in tool_aggregates.items():
                success_rate = aggregate["successes"] / aggregate["count"] if aggregate["count"] > 0 else 0.0
                
                tool_usage = ToolUsage(
                    tool_name=tool_name,
                    usage_count=aggregate["count"],
                    success_rate=success_rate,
                    total_time_seconds=aggregate["total_time"],
                    outcomes=aggregate["outcomes"],
                )
                
                # Generate efficiency assessment
                tool_usage.efficiency_assessment = await self._assess_tool_efficiency(
                    tool_usage, agent_role, task_metrics
                )
                
                tool_usage_analysis.append(tool_usage)
            
            return sorted(tool_usage_analysis, key=lambda t: t.usage_count, reverse=True)
            
        except Exception as e:
            self.log.error("Tool usage analysis failed: %s", e)
            raise ToolUsageAnalysisFailure(f"Analysis failed: {str(e)}")
    
    async def identify_challenges(
        self,
        agent_role: RoleName,
        execution_log: List[Dict[str, Any]],
        task_context: Dict[str, Any],
    ) -> List[Challenge]:
        """Identify challenges and obstacles encountered during execution.
        
        Args:
            agent_role: The role of the agent
            execution_log: Log of execution events
            task_context: Context about the task being executed
            
        Returns:
            List[Challenge]: Identified challenges with analysis
        """
        self.log.debug("Identifying challenges for role: %s", agent_role.value)
        
        challenges = []
        
        # Scan execution log for challenge indicators
        for entry in execution_log:
            entry_type = entry.get("type", "")
            
            if entry_type in ["error", "failure", "timeout", "retry"]:
                challenge = Challenge(
                    description=entry.get("description", "Unknown challenge"),
                    category=self._categorize_challenge(entry),
                    impact_level=self._assess_challenge_impact(entry),
                    resolution_attempted=entry.get("resolution_attempt"),
                    resolution_successful=entry.get("resolution_successful", False),
                )
                
                # Extract lessons learned if available
                if "lessons" in entry:
                    challenge.lessons_learned = entry["lessons"]
                
                challenges.append(challenge)
        
        # Add role-specific challenge detection
        role_challenges = await self._detect_role_specific_challenges(
            agent_role, task_context, execution_log
        )
        challenges.extend(role_challenges)
        
        return challenges[:10]  # Limit to most significant challenges
    
    async def generate_success_factors(
        self,
        agent_role: RoleName,
        task_execution_data: Dict[str, Any],
        performance_assessment: PerformanceAssessment,
    ) -> List[str]:
        """Generate list of factors that contributed to success.
        
        Args:
            agent_role: The role of the agent
            task_execution_data: Execution metrics and data
            performance_assessment: Performance assessment results
            
        Returns:
            List[str]: Factors that contributed to success
        """
        success_factors = []
        
        # Analyze performance scores for success indicators
        if performance_assessment.overall_score > 0.8:
            success_factors.append("Strong overall performance execution")
        
        if performance_assessment.efficiency_score > 0.8:
            success_factors.append("Efficient task execution approach")
        
        if performance_assessment.quality_score > 0.8:
            success_factors.append("High quality outputs delivered")
        
        # Check task execution metrics
        if task_execution_data.get("completion_time", 0) < task_execution_data.get("estimated_time", float('inf')):
            success_factors.append("Completed task ahead of schedule")
        
        if task_execution_data.get("error_count", 0) == 0:
            success_factors.append("Error-free execution achieved")
        
        # Add role-specific success factors
        role_factors = await self._get_role_specific_success_factors(
            agent_role, task_execution_data, performance_assessment
        )
        success_factors.extend(role_factors)
        
        return success_factors[:5]  # Top 5 success factors
    
    # Private helper methods
    
    async def _calculate_component_scores(
        self,
        agent_role: RoleName,
        task_execution_data: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate component scores for performance assessment."""
        
        scores = {
            "efficiency": 0.7,  # Default neutral scores
            "quality": 0.7,
            "collaboration": 0.7,
            "learning": 0.7,
        }
        
        # Calculate efficiency score
        completion_time = task_execution_data.get("completion_time", 0)
        estimated_time = task_execution_data.get("estimated_time", completion_time or 1)
        
        if completion_time > 0 and estimated_time > 0:
            time_efficiency = min(1.0, estimated_time / completion_time)
            scores["efficiency"] = time_efficiency * 0.7 + 0.3  # 0.3 to 1.0 range
        
        # Calculate quality score based on outputs
        error_count = task_execution_data.get("error_count", 0)
        total_outputs = task_execution_data.get("output_count", 1)
        
        error_rate = error_count / max(1, total_outputs)
        scores["quality"] = max(0.0, 1.0 - (error_rate * 2))  # Penalize errors
        
        # Collaboration score (basic heuristic)
        if execution_context and "team_interactions" in execution_context:
            interaction_count = len(execution_context["team_interactions"])
            if interaction_count > 0:
                scores["collaboration"] = min(1.0, 0.5 + (interaction_count * 0.1))
        
        # Learning score (based on new patterns or techniques used)
        if task_execution_data.get("new_techniques_used", 0) > 0:
            scores["learning"] = min(1.0, 0.7 + (task_execution_data["new_techniques_used"] * 0.1))
        
        return scores
    
    async def _calculate_overall_score(
        self,
        agent_role: RoleName,
        component_scores: Dict[str, float],
    ) -> float:
        """Calculate overall performance score using role-specific weights."""
        
        # Get role-specific weights or use defaults
        weights = self.role_weights.get(agent_role, {
            "efficiency": 0.3,
            "quality": 0.3,
            "collaboration": 0.2,
            "learning": 0.2,
        })
        
        # Map component scores to weight categories
        score_mapping = {
            "efficiency": component_scores.get("efficiency", 0.7),
            "quality": component_scores.get("quality", 0.7),
            "collaboration": component_scores.get("collaboration", 0.7),
            "learning": component_scores.get("learning", 0.7),
        }
        
        # Calculate weighted average
        weighted_sum = sum(score * weights.get(category, 0.25) 
                          for category, score in score_mapping.items())
        
        return min(1.0, max(0.0, weighted_sum))
    
    async def _generate_assessment_notes(
        self,
        agent_role: RoleName,
        component_scores: Dict[str, float],
        task_execution_data: Dict[str, Any],
    ) -> str:
        """Generate narrative assessment notes."""
        
        overall_score = await self._calculate_overall_score(agent_role, component_scores)
        
        if overall_score > 0.8:
            return f"Excellent performance by {agent_role.value} with strong scores across all dimensions"
        elif overall_score > 0.6:
            return f"Good performance by {agent_role.value} with some areas for optimization"
        else:
            return f"Performance by {agent_role.value} requires attention and improvement"
    
    async def _identify_strengths(
        self,
        component_scores: Dict[str, float],
        agent_role: RoleName,
    ) -> List[str]:
        """Identify areas of strength based on component scores."""
        
        strengths = []
        
        for component, score in component_scores.items():
            if score > 0.8:
                strengths.append(f"Strong {component} performance")
        
        # Add role-specific strengths
        if agent_role == RoleName.CODER and component_scores.get("quality", 0) > 0.8:
            strengths.append("High-quality code generation")
        elif agent_role == RoleName.QA and component_scores.get("quality", 0) > 0.8:
            strengths.append("Thorough validation and testing")
        elif agent_role == RoleName.ARCHITECT and component_scores.get("quality", 0) > 0.8:
            strengths.append("Comprehensive architectural design")
        
        return strengths[:3]  # Top 3 strengths
    
    async def _identify_improvement_areas(
        self,
        component_scores: Dict[str, float],
        agent_role: RoleName,
    ) -> List[str]:
        """Identify areas needing improvement based on component scores."""
        
        improvements = []
        
        for component, score in component_scores.items():
            if score < 0.6:
                improvements.append(f"Improve {component} through focused practice")
        
        # Add role-specific improvement suggestions
        if agent_role == RoleName.CODER and component_scores.get("efficiency", 0) < 0.7:
            improvements.append("Optimize coding approach for better efficiency")
        elif agent_role == RoleName.QA and component_scores.get("quality", 0) < 0.7:
            improvements.append("Enhance testing thoroughness and coverage")
        elif agent_role == RoleName.ARCHITECT and component_scores.get("collaboration", 0) < 0.7:
            improvements.append("Improve stakeholder communication and collaboration")
        
        return improvements[:3]  # Top 3 improvement areas
    
    def _parse_timestamp(self, timestamp_data: Any) -> datetime:
        """Parse timestamp from various formats."""
        if isinstance(timestamp_data, datetime):
            return timestamp_data
        elif isinstance(timestamp_data, str):
            try:
                return datetime.fromisoformat(timestamp_data.replace('Z', '+00:00'))
            except:
                return datetime.now(timezone.utc)
        else:
            return datetime.now(timezone.utc)
    
    async def _analyze_decision_outcome(
        self,
        decision: DecisionPoint,
        task_outcome: Dict[str, Any],
        agent_role: RoleName,
    ) -> str:
        """Analyze the outcome of a specific decision."""
        
        task_success = task_outcome.get("success", True)
        task_quality = task_outcome.get("quality_score", 0.7)
        
        if task_success and task_quality > 0.8:
            return f"Decision contributed to successful task completion"
        elif task_success:
            return f"Decision led to task completion with acceptable quality"
        else:
            return f"Decision may have contributed to suboptimal outcome"
    
    async def _assess_tool_efficiency(
        self,
        tool_usage: ToolUsage,
        agent_role: RoleName,
        task_metrics: Dict[str, Any],
    ) -> str:
        """Assess efficiency of tool usage."""
        
        if tool_usage.success_rate > 0.9 and tool_usage.usage_count > 3:
            return "Highly efficient tool usage with consistent success"
        elif tool_usage.success_rate > 0.7:
            return "Good tool usage with room for optimization"
        else:
            return "Tool usage needs improvement - consider alternative approaches"
    
    def _categorize_challenge(self, challenge_entry: Dict[str, Any]) -> str:
        """Categorize a challenge based on its characteristics."""
        
        entry_type = challenge_entry.get("type", "").lower()
        description = challenge_entry.get("description", "").lower()
        
        if "timeout" in entry_type or "timeout" in description:
            return "performance"
        elif "error" in entry_type or "failure" in entry_type:
            return "technical"
        elif "coordination" in description or "communication" in description:
            return "coordination"
        else:
            return "general"
    
    def _assess_challenge_impact(self, challenge_entry: Dict[str, Any]) -> str:
        """Assess the impact level of a challenge."""
        
        if challenge_entry.get("severity") == "critical":
            return "critical"
        elif challenge_entry.get("blocked_execution", False):
            return "high"
        elif challenge_entry.get("caused_retry", False):
            return "medium"
        else:
            return "low"
    
    async def _detect_role_specific_challenges(
        self,
        agent_role: RoleName,
        task_context: Dict[str, Any],
        execution_log: List[Dict[str, Any]],
    ) -> List[Challenge]:
        """Detect challenges specific to the agent role."""
        
        challenges = []
        
        # Role-specific challenge detection logic
        if agent_role == RoleName.CODER:
            # Look for code compilation or validation issues
            for entry in execution_log:
                if "compilation" in entry.get("description", "").lower():
                    challenges.append(Challenge(
                        description="Code compilation challenges encountered",
                        category="technical",
                        impact_level="medium",
                    ))
        
        elif agent_role == RoleName.QA:
            # Look for test failures or validation issues
            for entry in execution_log:
                if "test" in entry.get("description", "").lower() and entry.get("type") == "failure":
                    challenges.append(Challenge(
                        description="Test validation challenges identified",
                        category="quality_assurance",
                        impact_level="high",
                    ))
        
        return challenges
    
    async def _get_role_specific_success_factors(
        self,
        agent_role: RoleName,
        task_execution_data: Dict[str, Any],
        performance_assessment: PerformanceAssessment,
    ) -> List[str]:
        """Get success factors specific to the agent role."""
        
        factors = []
        
        if agent_role == RoleName.CODER:
            if task_execution_data.get("code_lines_generated", 0) > 100:
                factors.append("Generated substantial code implementation")
            if performance_assessment.quality_score > 0.8:
                factors.append("Maintained high code quality standards")
        
        elif agent_role == RoleName.QA:
            if task_execution_data.get("tests_created", 0) > 5:
                factors.append("Created comprehensive test coverage")
            if task_execution_data.get("issues_found", 0) > 0:
                factors.append("Successfully identified quality issues")
        
        elif agent_role == RoleName.ARCHITECT:
            if task_execution_data.get("design_artifacts", 0) > 0:
                factors.append("Produced detailed architectural designs")
            if performance_assessment.collaboration_score > 0.8:
                factors.append("Effective stakeholder collaboration")
        
        return factors