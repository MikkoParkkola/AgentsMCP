"""Individual agent retrospective framework.

This module provides the core framework for individual agents to perform
self-retrospective analysis after task completion. It includes structured
logging, self-assessment capabilities, and integration with the agent lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..models import TaskEnvelopeV1, ResultEnvelopeV1
from ..roles.base import RoleName
from .data_models import (
    IndividualRetrospective,
    IndividualRetrospectiveConfig,
    PerformanceAssessment,
    DecisionPoint,
    ToolUsage,
    Challenge,
    SelfImprovementAction,
    ImprovementCategory,
    PriorityLevel,
    RetrospectiveType,
)


class RetrospectiveTimeoutError(Exception):
    """Raised when retrospective process times out."""
    pass


class SelfAssessmentFailure(Exception):
    """Raised when self-assessment process fails."""
    pass


class InvalidTaskContext(Exception):
    """Raised when task context is invalid for retrospective."""
    pass


class IndividualRetrospectiveFramework:
    """Framework for conducting individual agent retrospectives."""
    
    def __init__(
        self,
        config: Optional[IndividualRetrospectiveConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or IndividualRetrospectiveConfig()
        self.log = logger or logging.getLogger(__name__)
        
        # Internal state
        self._active_retrospectives: Dict[str, IndividualRetrospective] = {}
        self._retrospective_history: List[IndividualRetrospective] = []
        
        self.log.info("IndividualRetrospectiveFramework initialized")
    
    async def conduct_retrospective(
        self,
        agent_role: RoleName,
        task_context: TaskEnvelopeV1,
        execution_results: ResultEnvelopeV1,
        agent_state: Optional[Dict[str, Any]] = None,
    ) -> IndividualRetrospective:
        """Conduct a complete individual retrospective for an agent.
        
        Args:
            agent_role: The role of the agent conducting retrospective
            task_context: Original task data
            execution_results: Task execution results
            agent_state: Optional agent internal state during execution
            
        Returns:
            IndividualRetrospective: Complete retrospective data
            
        Raises:
            RetrospectiveTimeoutError: If retrospective times out
            SelfAssessmentFailure: If self-assessment fails
            InvalidTaskContext: If task context is invalid
        """
        start_time = datetime.now(timezone.utc)
        
        # Validate inputs
        if not task_context or not task_context.objective:
            raise InvalidTaskContext("Task context missing or invalid")
        
        if not execution_results:
            raise InvalidTaskContext("Execution results missing")
        
        self.log.info(
            "Starting individual retrospective for %s on task %s",
            agent_role.value,
            task_context.inputs.get("task_id", "unknown")
        )
        
        # Initialize retrospective
        retrospective = IndividualRetrospective(
            agent_role=agent_role,
            task_id=task_context.inputs.get("task_id", str(task_context.telemetry.trace_id)),
            retrospective_type=RetrospectiveType.INDIVIDUAL,
        )
        
        # Track active retrospective
        self._active_retrospectives[retrospective.retrospective_id] = retrospective
        
        try:
            # Execute retrospective phases with timeout
            await asyncio.wait_for(
                self._execute_retrospective_phases(retrospective, task_context, execution_results, agent_state),
                timeout=self.config.timeout_seconds
            )
            
        except asyncio.TimeoutError:
            self.log.warning("Retrospective timed out for agent %s", agent_role.value)
            raise RetrospectiveTimeoutError(f"Retrospective timed out after {self.config.timeout_seconds}s")
        
        except Exception as e:
            self.log.error("Retrospective failed for agent %s: %s", agent_role.value, e)
            raise SelfAssessmentFailure(f"Retrospective failed: {str(e)}")
        
        finally:
            # Finalize retrospective
            end_time = datetime.now(timezone.utc)
            retrospective.duration_seconds = (end_time - start_time).total_seconds()
            
            # Store in history and remove from active
            self._retrospective_history.append(retrospective)
            self._active_retrospectives.pop(retrospective.retrospective_id, None)
            
            self.log.info(
                "Individual retrospective completed for %s in %.2fs",
                agent_role.value,
                retrospective.duration_seconds
            )
        
        return retrospective
    
    async def _execute_retrospective_phases(
        self,
        retrospective: IndividualRetrospective,
        task_context: TaskEnvelopeV1,
        execution_results: ResultEnvelopeV1,
        agent_state: Optional[Dict[str, Any]],
    ) -> None:
        """Execute all phases of the individual retrospective process."""
        
        # Phase 1: Basic retrospective analysis
        await self._analyze_execution_outcomes(retrospective, task_context, execution_results)
        
        # Phase 2: Performance self-assessment
        if self.config.performance_assessment_required:
            await self._conduct_performance_assessment(retrospective, execution_results, agent_state)
        
        # Phase 3: Decision analysis (if enabled)
        if self.config.include_decision_analysis:
            await self._analyze_decisions(retrospective, agent_state)
        
        # Phase 4: Tool usage analysis (if enabled)  
        if self.config.include_tool_usage_analysis:
            await self._analyze_tool_usage(retrospective, agent_state)
        
        # Phase 5: Collaboration feedback (if enabled)
        if self.config.include_collaboration_feedback:
            await self._gather_collaboration_feedback(retrospective, agent_state)
        
        # Phase 6: Learning extraction
        await self._extract_learnings(retrospective, task_context, execution_results)
        
        # Phase 7: Generate improvement actions
        await self._generate_improvement_actions(retrospective)
    
    async def _analyze_execution_outcomes(
        self,
        retrospective: IndividualRetrospective,
        task_context: TaskEnvelopeV1,
        execution_results: ResultEnvelopeV1,
    ) -> None:
        """Analyze task execution outcomes for basic retrospective data."""
        
        # Generate execution summary
        status = execution_results.status.value if execution_results.status else "unknown"
        objective = task_context.objective[:100] + "..." if len(task_context.objective) > 100 else task_context.objective
        
        retrospective.execution_summary = (
            f"Task '{objective}' completed with status: {status}. "
            f"Confidence: {execution_results.confidence:.2f}"
        )
        
        # Analyze what went well
        successes = []
        if execution_results.status and execution_results.status.value == "ok":
            successes.append("Task completed successfully")
        
        if execution_results.confidence and execution_results.confidence > 0.8:
            successes.append("High confidence in results achieved")
        
        if execution_results.artifacts:
            successes.append(f"Generated {len(execution_results.artifacts)} artifacts")
        
        # Add role-specific success indicators
        role_successes = self._get_role_specific_successes(retrospective.agent_role, execution_results)
        successes.extend(role_successes)
        
        retrospective.what_went_well = successes[:5]  # Limit to top 5
        
        # Analyze improvement areas
        improvements = []
        if execution_results.confidence and execution_results.confidence < 0.7:
            improvements.append("Improve result confidence through better validation")
        
        if execution_results.metrics and execution_results.metrics.get("retries", 0) > 0:
            retry_count = execution_results.metrics["retries"]
            improvements.append(f"Reduce retries (had {retry_count}) through better initial approach")
        
        # Add role-specific improvement areas
        role_improvements = self._get_role_specific_improvements(retrospective.agent_role, execution_results)
        improvements.extend(role_improvements)
        
        retrospective.what_could_improve = improvements[:5]  # Limit to top 5
        
        # Identify challenges from execution results
        challenges = []
        if execution_results.status and execution_results.status.value == "error":
            error_msgs = execution_results.artifacts.get("error_details", ["Unknown error"])
            for error in error_msgs[:self.config.max_challenges]:
                challenge = Challenge(
                    description=f"Execution error: {error}",
                    category="technical",
                    impact_level="high",
                    resolution_successful=False,
                )
                challenges.append(challenge)
        
        retrospective.challenges_encountered = challenges
    
    async def _conduct_performance_assessment(
        self,
        retrospective: IndividualRetrospective,
        execution_results: ResultEnvelopeV1,
        agent_state: Optional[Dict[str, Any]],
    ) -> None:
        """Conduct performance self-assessment."""
        
        assessment = PerformanceAssessment()
        
        # Calculate overall score based on execution results
        base_score = 0.6  # Neutral baseline
        
        # Adjust based on status
        if execution_results.status and execution_results.status.value == "ok":
            base_score += 0.2
        elif execution_results.status and execution_results.status.value == "error":
            base_score -= 0.3
        
        # Adjust based on confidence
        if execution_results.confidence:
            confidence_bonus = (execution_results.confidence - 0.5) * 0.4
            base_score += confidence_bonus
        
        # Ensure score is within bounds
        assessment.overall_score = max(0.0, min(1.0, base_score))
        
        # Set component scores (simplified heuristic)
        assessment.efficiency_score = assessment.overall_score * 0.9  # Slightly lower
        assessment.quality_score = assessment.overall_score * 1.1  # Slightly higher
        assessment.collaboration_score = 0.7  # Default neutral
        assessment.learning_score = 0.8  # Default positive
        
        # Generate assessment notes
        if assessment.overall_score > 0.8:
            assessment.self_assessment_notes = "Strong performance with good results achieved"
            assessment.areas_of_strength.extend(["Task completion", "Result quality"])
        elif assessment.overall_score > 0.6:
            assessment.self_assessment_notes = "Satisfactory performance with room for improvement"
            assessment.areas_for_improvement.append("Optimize approach for better efficiency")
        else:
            assessment.self_assessment_notes = "Performance below expectations, requires attention"
            assessment.areas_for_improvement.extend(["Review methodology", "Improve validation"])
        
        retrospective.performance_assessment = assessment
    
    async def _analyze_decisions(
        self,
        retrospective: IndividualRetrospective,
        agent_state: Optional[Dict[str, Any]],
    ) -> None:
        """Analyze major decisions made during task execution."""
        
        decisions = []
        
        # Extract decision data from agent state if available
        if agent_state and "decisions" in agent_state:
            for decision_data in agent_state["decisions"]:
                decision = DecisionPoint(
                    decision_description=decision_data.get("description", "Unknown decision"),
                    chosen_option=decision_data.get("chosen_option", "Unknown"),
                    rationale=decision_data.get("rationale", "No rationale provided"),
                    confidence_level=decision_data.get("confidence", 0.5),
                )
                decisions.append(decision)
        else:
            # Generate placeholder decision analysis
            decision = DecisionPoint(
                decision_description="Approach selection for task execution",
                chosen_option="Standard implementation approach",
                rationale="Based on task requirements and available tools",
                confidence_level=0.7,
            )
            decisions.append(decision)
        
        retrospective.decisions_made = decisions[:5]  # Limit to important decisions
    
    async def _analyze_tool_usage(
        self,
        retrospective: IndividualRetrospective,
        agent_state: Optional[Dict[str, Any]],
    ) -> None:
        """Analyze tool usage patterns and effectiveness."""
        
        tools = []
        
        # Extract tool usage from agent state if available
        if agent_state and "tool_usage" in agent_state:
            for tool_name, usage_data in agent_state["tool_usage"].items():
                tool_usage = ToolUsage(
                    tool_name=tool_name,
                    usage_count=usage_data.get("count", 1),
                    success_rate=usage_data.get("success_rate", 1.0),
                    total_time_seconds=usage_data.get("total_time", 0.0),
                    outcomes=usage_data.get("outcomes", []),
                )
                
                # Add efficiency assessment
                if tool_usage.success_rate > 0.9:
                    tool_usage.efficiency_assessment = "Highly effective tool usage"
                elif tool_usage.success_rate > 0.7:
                    tool_usage.efficiency_assessment = "Effective tool usage with minor issues"
                else:
                    tool_usage.efficiency_assessment = "Tool usage needs optimization"
                
                tools.append(tool_usage)
        
        retrospective.tools_used = tools
    
    async def _gather_collaboration_feedback(
        self,
        retrospective: IndividualRetrospective,
        agent_state: Optional[Dict[str, Any]],
    ) -> None:
        """Gather feedback on collaboration and team dynamics."""
        
        # Default collaboration assessment
        retrospective.collaboration_feedback = "Limited collaboration data available for assessment"
        retrospective.communication_effectiveness = 0.7  # Default neutral
        
        # Extract collaboration data if available
        if agent_state and "collaboration" in agent_state:
            collab_data = agent_state["collaboration"]
            
            retrospective.collaboration_feedback = collab_data.get(
                "feedback", 
                "Collaboration proceeded according to established patterns"
            )
            
            retrospective.communication_effectiveness = collab_data.get("communication_score", 0.7)
            
            retrospective.team_dynamics_observations = collab_data.get(
                "observations", 
                ["Standard team coordination observed"]
            )
    
    async def _extract_learnings(
        self,
        retrospective: IndividualRetrospective,
        task_context: TaskEnvelopeV1,
        execution_results: ResultEnvelopeV1,
    ) -> None:
        """Extract key learnings and knowledge gaps from the task execution."""
        
        learnings = []
        knowledge_gaps = []
        
        # Generate learnings based on task outcome
        if execution_results.status and execution_results.status.value == "ok":
            learnings.append(f"Successfully completed {task_context.objective[:50]}...")
            learnings.append("Validated approach effectiveness for similar tasks")
        else:
            learnings.append("Identified failure patterns to avoid in future tasks")
            knowledge_gaps.append("Need better error recovery strategies")
        
        # Role-specific learnings
        role_learnings = self._get_role_specific_learnings(retrospective.agent_role, execution_results)
        learnings.extend(role_learnings)
        
        # Limit to configured maximum
        retrospective.key_learnings = learnings[:self.config.max_learnings]
        retrospective.knowledge_gaps_identified = knowledge_gaps[:5]
    
    async def _generate_improvement_actions(
        self,
        retrospective: IndividualRetrospective,
    ) -> None:
        """Generate specific improvement actions based on retrospective analysis."""
        
        actions = []
        
        # Generate actions from improvement areas
        for improvement in retrospective.what_could_improve:
            action = SelfImprovementAction(
                title=f"Address: {improvement[:40]}...",
                description=improvement,
                category=self._categorize_improvement(improvement),
                priority=self._prioritize_improvement(improvement),
                estimated_effort="1-2 hours",
                expected_benefit="Improved task execution effectiveness",
            )
            actions.append(action)
        
        # Generate actions from challenges
        for challenge in retrospective.challenges_encountered:
            if not challenge.resolution_successful:
                action = SelfImprovementAction(
                    title=f"Resolve: {challenge.description[:40]}...",
                    description=f"Develop solution for: {challenge.description}",
                    category=ImprovementCategory.PROCESS,
                    priority=PriorityLevel.HIGH if challenge.impact_level == "high" else PriorityLevel.MEDIUM,
                    estimated_effort="2-4 hours",
                    expected_benefit="Prevent similar challenges in future tasks",
                )
                actions.append(action)
        
        # Generate actions from knowledge gaps
        for gap in retrospective.knowledge_gaps_identified:
            action = SelfImprovementAction(
                title=f"Learn: {gap[:40]}...",
                description=f"Address knowledge gap: {gap}",
                category=ImprovementCategory.LEARNING,
                priority=PriorityLevel.MEDIUM,
                estimated_effort="1-3 hours",
                expected_benefit="Enhanced capability for future tasks",
            )
            actions.append(action)
        
        # Limit to configured maximum
        retrospective.self_improvement_actions = actions[:self.config.max_improvement_actions]
    
    # Helper methods
    
    def _get_role_specific_successes(self, role: RoleName, results: ResultEnvelopeV1) -> List[str]:
        """Get role-specific success indicators."""
        successes = []
        
        if role == RoleName.CODER:
            if results.artifacts and any("code" in str(k).lower() for k in results.artifacts.keys()):
                successes.append("Generated code artifacts successfully")
        elif role == RoleName.QA:
            if results.artifacts and any("test" in str(k).lower() for k in results.artifacts.keys()):
                successes.append("Created test artifacts and validation")
        elif role == RoleName.ARCHITECT:
            if results.artifacts and any("design" in str(k).lower() or "plan" in str(k).lower() for k in results.artifacts.keys()):
                successes.append("Produced architectural design artifacts")
        
        return successes
    
    def _get_role_specific_improvements(self, role: RoleName, results: ResultEnvelopeV1) -> List[str]:
        """Get role-specific improvement suggestions."""
        improvements = []
        
        if role == RoleName.CODER:
            improvements.append("Enhance code documentation and comments")
            improvements.append("Improve error handling in generated code")
        elif role == RoleName.QA:
            improvements.append("Expand test coverage for edge cases")
            improvements.append("Improve test automation and validation")
        elif role == RoleName.ARCHITECT:
            improvements.append("Provide more detailed implementation guidance")
            improvements.append("Include more comprehensive risk analysis")
        
        return improvements
    
    def _get_role_specific_learnings(self, role: RoleName, results: ResultEnvelopeV1) -> List[str]:
        """Get role-specific learning insights."""
        learnings = []
        
        if role == RoleName.CODER:
            learnings.append("Refined understanding of code implementation patterns")
        elif role == RoleName.QA:
            learnings.append("Enhanced knowledge of testing strategies and validation")
        elif role == RoleName.ARCHITECT:
            learnings.append("Improved architectural decision-making process")
        
        return learnings
    
    def _categorize_improvement(self, improvement: str) -> ImprovementCategory:
        """Categorize improvement action based on content."""
        improvement_lower = improvement.lower()
        
        if any(word in improvement_lower for word in ["performance", "speed", "efficient"]):
            return ImprovementCategory.PERFORMANCE
        elif any(word in improvement_lower for word in ["coordination", "team", "collaborate"]):
            return ImprovementCategory.COORDINATION
        elif any(word in improvement_lower for word in ["tool", "usage", "technique"]):
            return ImprovementCategory.TOOL_USAGE
        elif any(word in improvement_lower for word in ["decision", "choice", "approach"]):
            return ImprovementCategory.DECISION_MAKING
        elif any(word in improvement_lower for word in ["communication", "feedback"]):
            return ImprovementCategory.COMMUNICATION
        elif any(word in improvement_lower for word in ["process", "workflow", "method"]):
            return ImprovementCategory.PROCESS
        else:
            return ImprovementCategory.LEARNING
    
    def _prioritize_improvement(self, improvement: str) -> PriorityLevel:
        """Determine priority of improvement action."""
        improvement_lower = improvement.lower()
        
        if any(word in improvement_lower for word in ["critical", "urgent", "blocker"]):
            return PriorityLevel.CRITICAL
        elif any(word in improvement_lower for word in ["error", "failure", "fix"]):
            return PriorityLevel.HIGH
        elif any(word in improvement_lower for word in ["optimize", "improve", "enhance"]):
            return PriorityLevel.MEDIUM
        else:
            return PriorityLevel.LOW
    
    # Public API methods
    
    def get_retrospective_history(self, agent_role: Optional[RoleName] = None, limit: Optional[int] = None) -> List[IndividualRetrospective]:
        """Get retrospective history with optional filtering."""
        history = self._retrospective_history
        
        if agent_role:
            history = [r for r in history if r.agent_role == agent_role]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_active_retrospectives(self) -> Dict[str, IndividualRetrospective]:
        """Get currently active retrospectives."""
        return self._active_retrospectives.copy()
    
    async def cancel_retrospective(self, retrospective_id: str) -> bool:
        """Cancel an active retrospective."""
        if retrospective_id in self._active_retrospectives:
            retrospective = self._active_retrospectives.pop(retrospective_id)
            retrospective.execution_summary = "Retrospective cancelled"
            self.log.warning("Cancelled retrospective %s", retrospective_id)
            return True
        return False