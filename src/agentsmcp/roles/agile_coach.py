"""Agile Coach role for process facilitation, retrospectives, and team optimization.

This role specializes in:
- Facilitating agile ceremonies and processes
- Conducting retrospectives and team health assessments
- Providing improvement recommendations
- Optimizing team dynamics and workflow
- Pattern recognition for continuous improvement
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from .base import BaseRole, RoleName, TaskEnvelope, ResultEnvelope, EnvelopeStatus


class AgileCoachRole(BaseRole):
    """Agile coach role providing comprehensive process facilitation and team optimization.
    
    This role integrates with the AgileCoachIntegration to provide:
    - Planning guidance and risk mitigation strategies  
    - Retrospective facilitation and analysis
    - Team performance monitoring and improvement suggestions
    - Ceremony scheduling and process optimization
    - Historical pattern recognition for continuous improvement
    """

    _preferred_agent_type = "claude"  # Prefer Claude for complex reasoning and analysis
    
    def __init__(self):
        super().__init__()
        # Import here to avoid circular import
        from ..orchestration.agile_coach import AgileCoachIntegration
        self._coach_integration = AgileCoachIntegration()

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.PROCESS_COACH

    @classmethod
    def responsibilities(cls) -> List[str]:
        return [
            "Agile process facilitation",
            "Retrospective analysis and facilitation",
            "Team performance optimization",
            "Continuous improvement recommendations",
            "Ceremony planning and scheduling",
            "Risk mitigation guidance",
            "Team health monitoring",
            "Historical pattern recognition"
        ]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return [
            "Process improvement recommendations",
            "Team composition adjustments",
            "Ceremony scheduling and format",
            "Coaching intervention strategies",
            "Performance improvement priorities",
            "Risk mitigation approaches"
        ]

    @classmethod
    def apply(cls, task: TaskEnvelope) -> ResultEnvelope:
        """Apply agile coaching logic to the given task.
        
        Args:
            task: Task envelope containing coaching request
            
        Returns:
            ResultEnvelope with coaching recommendations and decisions
        """
        coach = cls()
        
        # Determine the type of coaching request
        coaching_type = task.payload.get("coaching_type", "general")
        
        try:
            if coaching_type == "planning":
                result = coach._handle_planning_coaching(task)
            elif coaching_type == "retrospective":
                result = coach._handle_retrospective_coaching(task)
            elif coaching_type == "ceremony_scheduling":
                result = coach._handle_ceremony_scheduling(task)
            elif coaching_type == "team_improvement":
                result = coach._handle_team_improvement(task)
            elif coaching_type == "health_assessment":
                result = coach._handle_health_assessment(task)
            else:
                result = coach._handle_general_coaching(task)
                
            return ResultEnvelope(
                id=task.id,
                role=cls.name(),
                status=EnvelopeStatus.SUCCESS,
                model_assigned=(task.requested_agent_type or cls._preferred_agent_type),
                decisions=result.get("decisions", []),
                risks=result.get("risks", []),
                followups=result.get("followups", []),
                outputs=result.get("outputs", {}),
                errors=[]
            )
            
        except Exception as e:
            return ResultEnvelope(
                id=task.id,
                role=cls.name(),
                status=EnvelopeStatus.ERROR,
                model_assigned=(task.requested_agent_type or cls._preferred_agent_type),
                decisions=[],
                risks=[f"Coaching analysis failed: {str(e)}"],
                followups=["Review coaching request and retry"],
                outputs={},
                errors=[f"Agile coaching error: {str(e)}"]
            )

    def _handle_planning_coaching(self, task: TaskEnvelope) -> Dict[str, Any]:
        """Handle planning phase coaching requests."""
        payload = task.payload
        
        # Extract task classification and team composition
        task_classification = self._extract_task_classification(payload)
        team_composition = self._extract_team_composition(payload)
        
        # Generate coaching recommendations using async simulation
        # In real implementation, this would use await coach_planning()
        coach_actions = self._simulate_coach_planning(task_classification, team_composition)
        
        decisions = [
            f"Recommended approach: {coach_actions['recommended_approach']}",
            f"Coordination strategy: {coach_actions['coordination_strategy']}",
            f"Estimated velocity: {coach_actions['estimated_velocity']:.2f}"
        ]
        
        risks = coach_actions['risk_mitigations']
        
        followups = [
            "Implement suggested milestones and checkpoints",
            "Monitor team velocity and adjust as needed",
            "Review risk mitigations before task execution"
        ]
        
        outputs = {
            "coaching_type": "planning",
            "coach_actions": coach_actions,
            "planning_confidence": coach_actions['confidence_score'],
            "recommended_milestones": coach_actions['suggested_milestones'],
            "team_adjustments": coach_actions['team_adjustments']
        }
        
        return {
            "decisions": decisions,
            "risks": risks,
            "followups": followups,
            "outputs": outputs
        }

    def _handle_retrospective_coaching(self, task: TaskEnvelope) -> Dict[str, Any]:
        """Handle retrospective coaching requests."""
        payload = task.payload
        
        # Extract retrospective data
        execution_results = payload.get("execution_results", {})
        performance_metrics = self._extract_performance_metrics(payload)
        team_composition = self._extract_team_composition(payload)
        
        # Generate retrospective report using async simulation
        retrospective_report = self._simulate_retrospective_coaching(
            execution_results, performance_metrics, team_composition
        )
        
        decisions = [
            f"Team health score: {retrospective_report['team_health_score']:.2f}",
            f"Identified {len(retrospective_report['what_went_well'])} successes",
            f"Identified {len(retrospective_report['what_could_improve'])} improvement areas"
        ]
        
        risks = [
            item for item in retrospective_report['what_could_improve'] 
            if any(keyword in item.lower() for keyword in ['risk', 'critical', 'blocker'])
        ]
        
        followups = retrospective_report['action_items']
        
        outputs = {
            "coaching_type": "retrospective",
            "retrospective_report": retrospective_report,
            "team_health_score": retrospective_report['team_health_score'],
            "improvement_suggestions": retrospective_report['improvement_suggestions'],
            "next_sprint_recommendations": retrospective_report['next_sprint_recommendations']
        }
        
        return {
            "decisions": decisions,
            "risks": risks,
            "followups": followups,
            "outputs": outputs
        }

    def _handle_ceremony_scheduling(self, task: TaskEnvelope) -> Dict[str, Any]:
        """Handle ceremony scheduling requests."""
        payload = task.payload
        
        from ..orchestration.agile_coach import AgilePhase
        phase = AgilePhase(payload.get("phase", AgilePhase.PLANNING))
        team = self._extract_team_specs(payload)
        
        # Generate ceremony schedule using async simulation
        ceremony_schedule = self._simulate_ceremony_scheduling(phase, team)
        
        decisions = [
            f"Scheduled {len(ceremony_schedule['upcoming_ceremonies'])} ceremonies",
            f"Recommended duration: {ceremony_schedule['estimated_duration']} minutes",
            f"Phase: {ceremony_schedule['phase']}"
        ]
        
        risks = []
        if len(ceremony_schedule['upcoming_ceremonies']) > 3:
            risks.append("High ceremony load may impact productivity")
        
        followups = [
            "Send calendar invites to recommended participants",
            "Prepare ceremony agendas and materials",
            "Confirm availability of all participants"
        ]
        
        outputs = {
            "coaching_type": "ceremony_scheduling",
            "ceremony_schedule": ceremony_schedule,
            "upcoming_ceremonies": ceremony_schedule['upcoming_ceremonies'],
            "next_ceremony_time": ceremony_schedule.get('next_ceremony_time')
        }
        
        return {
            "decisions": decisions,
            "risks": risks,
            "followups": followups,
            "outputs": outputs
        }

    def _handle_team_improvement(self, task: TaskEnvelope) -> Dict[str, Any]:
        """Handle team improvement coaching requests."""
        payload = task.payload
        
        team_metrics = self._extract_team_metrics(payload)
        
        # Generate improvement suggestions using async simulation
        improvement_suggestions = self._simulate_improvement_suggestions(team_metrics)
        
        from ..orchestration.agile_coach import ImprovementPriority
        # Prioritize suggestions
        critical_suggestions = [s for s in improvement_suggestions if s['priority'] == ImprovementPriority.CRITICAL]
        high_priority_suggestions = [s for s in improvement_suggestions if s['priority'] == ImprovementPriority.HIGH]
        
        decisions = [
            f"Identified {len(improvement_suggestions)} improvement opportunities",
            f"Critical priority items: {len(critical_suggestions)}",
            f"High priority items: {len(high_priority_suggestions)}"
        ]
        
        risks = [
            s['description'] for s in critical_suggestions
        ]
        
        followups = [
            "Prioritize critical and high-priority improvements",
            "Assign owners for each improvement initiative",
            "Set success criteria and timelines",
            "Schedule follow-up assessment"
        ]
        
        outputs = {
            "coaching_type": "team_improvement",
            "improvement_suggestions": [self._serialize_improvement_suggestion(s) for s in improvement_suggestions],
            "critical_items": len(critical_suggestions),
            "high_priority_items": len(high_priority_suggestions),
            "total_suggestions": len(improvement_suggestions)
        }
        
        return {
            "decisions": decisions,
            "risks": risks,
            "followups": followups,
            "outputs": outputs
        }

    def _handle_health_assessment(self, task: TaskEnvelope) -> Dict[str, Any]:
        """Handle team health assessment requests."""
        payload = task.payload
        
        team_metrics = self._extract_team_metrics(payload)
        
        # Calculate health indicators
        health_score = self._calculate_overall_health_score(team_metrics)
        health_category = self._categorize_health_score(health_score)
        
        # Generate health insights
        health_insights = self._generate_health_insights(team_metrics, health_score)
        
        decisions = [
            f"Overall team health: {health_category} ({health_score:.2f})",
            f"Velocity trend: {self._analyze_velocity_trend(team_metrics.velocity_trend)}",
            f"Quality assessment: {self._assess_quality_level(team_metrics.quality_score)}"
        ]
        
        risks = health_insights.get('risks', [])
        
        followups = health_insights.get('recommendations', [])
        
        outputs = {
            "coaching_type": "health_assessment",
            "health_score": health_score,
            "health_category": health_category,
            "health_insights": health_insights,
            "key_metrics": {
                "velocity_trend": team_metrics.velocity_trend,
                "quality_score": team_metrics.quality_score,
                "collaboration_score": team_metrics.collaboration_score,
                "team_satisfaction": team_metrics.team_satisfaction
            }
        }
        
        return {
            "decisions": decisions,
            "risks": risks,
            "followups": followups,
            "outputs": outputs
        }

    def _handle_general_coaching(self, task: TaskEnvelope) -> Dict[str, Any]:
        """Handle general coaching requests."""
        decisions = [
            "Provide general agile coaching guidance",
            "Assess current team state",
            "Recommend process improvements"
        ]
        
        followups = [
            "Conduct team assessment",
            "Review current processes",
            "Identify improvement opportunities",
            "Schedule regular coaching sessions"
        ]
        
        outputs = {
            "coaching_type": "general",
            "general_guidance": {
                "focus_areas": [
                    "Team collaboration and communication",
                    "Process efficiency and flow",
                    "Quality and technical excellence",
                    "Continuous learning and adaptation"
                ],
                "recommended_practices": [
                    "Regular retrospectives and health checks",
                    "Clear definition of done and acceptance criteria",
                    "Time-boxed iterations with frequent feedback",
                    "Cross-functional team structure and skills"
                ]
            }
        }
        
        return {
            "decisions": decisions,
            "risks": [],
            "followups": followups,
            "outputs": outputs
        }

    # Helper methods for data extraction and simulation

    def _extract_task_classification(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract task classification from payload."""
        from ..orchestration.models import TaskType, ComplexityLevel, RiskLevel
        classification_data = payload.get("task_classification", {})
        
        return {
            "task_type": TaskType(classification_data.get("task_type", TaskType.IMPLEMENTATION)),
            "complexity": ComplexityLevel(classification_data.get("complexity", ComplexityLevel.MEDIUM)),
            "risk_level": RiskLevel(classification_data.get("risk_level", RiskLevel.MEDIUM)),
            "estimated_effort": classification_data.get("estimated_effort", 50),
            "confidence": classification_data.get("confidence", 0.7)
        }

    def _extract_team_composition(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract team composition from payload."""
        from ..orchestration.models import CoordinationStrategy
        
        composition_data = payload.get("team_composition", {})
        
        primary_team = []
        for agent_data in composition_data.get("primary_team", []):
            primary_team.append({
                "role": agent_data.get("role", "coder"),
                "model_assignment": agent_data.get("model_assignment", "ollama"),
                "priority": agent_data.get("priority", 1)
            })
        
        return {
            "primary_team": primary_team,
            "coordination_strategy": CoordinationStrategy(
                composition_data.get("coordination_strategy", CoordinationStrategy.COLLABORATIVE)
            ),
            "confidence_score": composition_data.get("confidence_score", 0.7)
        }

    def _extract_performance_metrics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics from payload."""
        metrics_data = payload.get("performance_metrics", {})
        
        return {
            "success_rate": metrics_data.get("success_rate", 0.8),
            "average_duration": metrics_data.get("average_duration", 3600),
            "average_cost": metrics_data.get("average_cost", 5.0),
            "total_executions": metrics_data.get("total_executions", 10)
        }

    def _extract_team_specs(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract team specifications from payload."""
        team_data = payload.get("team", [])
        
        return [
            {
                "role": member.get("role", "team_member"),
                "model_assignment": member.get("model_assignment", "ollama"),
                "priority": member.get("priority", 1)
            }
            for member in team_data
        ]

    def _extract_team_metrics(self, payload: Dict[str, Any]):
        """Extract team metrics from payload."""
        from ..orchestration.agile_coach import TeamMetrics
        metrics_data = payload.get("team_metrics", {})
        
        return TeamMetrics(
            velocity_trend=metrics_data.get("velocity_trend", [0.7, 0.8, 0.75]),
            quality_score=metrics_data.get("quality_score", 0.8),
            collaboration_score=metrics_data.get("collaboration_score", 0.75),
            delivery_predictability=metrics_data.get("delivery_predictability", 0.8),
            cycle_time_avg=metrics_data.get("cycle_time_avg", 3.5),
            defect_rate=metrics_data.get("defect_rate", 0.05),
            team_satisfaction=metrics_data.get("team_satisfaction", 0.8),
            learning_velocity=metrics_data.get("learning_velocity", 0.6)
        )

    # Simulation methods (in real implementation, these would be async calls)

    def _simulate_coach_planning(
        self, 
        task_classification: Dict[str, Any],
        team_composition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate coaching planning analysis."""
        complexity = task_classification["complexity"]
        team_size = len(team_composition["primary_team"])
        
        # Import enums locally to avoid circular import
        from ..orchestration.models import ComplexityLevel
        
        # Simulate planning recommendations
        if complexity == ComplexityLevel.CRITICAL:
            approach = "Incremental approach with frequent checkpoints and risk mitigation"
            risk_mitigations = [
                "Implement frequent checkpoint reviews",
                "Create rollback plans for critical changes",
                "Establish escalation procedures"
            ]
        elif complexity == ComplexityLevel.HIGH:
            approach = "Structured approach with clear milestones and quality gates"
            risk_mitigations = [
                "Define clear acceptance criteria",
                "Implement comprehensive testing strategy"
            ]
        else:
            approach = "Agile collaborative approach with regular synchronization"
            risk_mitigations = ["Maintain regular communication and feedback loops"]
        
        suggested_milestones = [
            "Initial analysis and planning complete",
            "Core implementation milestone",
            "Testing and validation complete",
            "Final delivery and handoff"
        ]
        
        team_adjustments = []
        if team_size < 2 and complexity in [ComplexityLevel.HIGH, ComplexityLevel.CRITICAL]:
            team_adjustments.append("Consider adding additional team members for complex work")
        
        estimated_velocity = team_size * 0.7 * (1.0 if complexity == ComplexityLevel.MEDIUM else 0.8)
        
        return {
            "recommended_approach": approach,
            "risk_mitigations": risk_mitigations,
            "coordination_strategy": team_composition["coordination_strategy"],
            "suggested_milestones": suggested_milestones,
            "team_adjustments": team_adjustments,
            "estimated_velocity": estimated_velocity,
            "confidence_score": 0.8
        }

    def _simulate_retrospective_coaching(
        self,
        execution_results: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        team_composition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate retrospective coaching analysis."""
        completion_rate = execution_results.get("completion_rate", 0.9)
        success_rate = performance_metrics.get("success_rate", 0.8)
        
        what_went_well = []
        what_could_improve = []
        
        if completion_rate > 0.9:
            what_went_well.append("High completion rate achieved")
        if success_rate > 0.8:
            what_went_well.append("Good success rate maintained")
        
        if performance_metrics.get("average_duration", 0) > 7200:  # 2 hours
            what_could_improve.append("Execution time longer than expected")
        if performance_metrics.get("average_cost", 0) > 10.0:
            what_could_improve.append("Cost higher than budgeted")
        
        action_items = [
            "Review and optimize slow processes",
            "Continue successful practices",
            "Monitor key metrics closely"
        ]
        
        team_health_score = (success_rate + completion_rate) / 2.0
        
        velocity_analysis = {
            "current_success_rate": success_rate,
            "avg_duration": performance_metrics.get("average_duration", 0),
            "avg_cost": performance_metrics.get("average_cost", 0)
        }
        
        improvement_suggestions = [
            "Implement time-boxing techniques",
            "Enhance testing and quality processes",
            "Optimize resource allocation"
        ]
        
        next_sprint_recommendations = [
            "Focus on identified improvement areas",
            "Maintain current successful practices",
            "Monitor team health metrics"
        ]
        
        return {
            "summary": f"Team health: {'good' if team_health_score > 0.7 else 'needs attention'}",
            "what_went_well": what_went_well,
            "what_could_improve": what_could_improve,
            "action_items": action_items,
            "team_health_score": team_health_score,
            "velocity_analysis": velocity_analysis,
            "improvement_suggestions": improvement_suggestions,
            "next_sprint_recommendations": next_sprint_recommendations
        }

    def _simulate_ceremony_scheduling(
        self,
        phase,  # AgilePhase type - import locally
        team: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Simulate ceremony scheduling."""
        from ..orchestration.agile_coach import AgilePhase
        
        upcoming_ceremonies = []
        recommended_participants = [member["role"] for member in team]
        estimated_duration = 30
        
        if phase == AgilePhase.PLANNING:
            upcoming_ceremonies = [
                {
                    "type": "sprint_planning",
                    "purpose": "Plan upcoming work items",
                    "duration": 60,
                    "required_participants": recommended_participants
                }
            ]
            estimated_duration = 60
        elif phase == AgilePhase.RETROSPECTIVE:
            upcoming_ceremonies = [
                {
                    "type": "sprint_retrospective",
                    "purpose": "Reflect on team performance",
                    "duration": 45,
                    "required_participants": recommended_participants
                }
            ]
            estimated_duration = 45
        
        return {
            "phase": phase,
            "upcoming_ceremonies": upcoming_ceremonies,
            "recommended_participants": recommended_participants,
            "estimated_duration": estimated_duration,
            "next_ceremony_time": datetime.now(timezone.utc).isoformat()
        }

    def _simulate_improvement_suggestions(self, team_metrics) -> List[Dict[str, Any]]:
        """Simulate improvement suggestions generation."""
        from ..orchestration.agile_coach import ImprovementPriority
        
        suggestions = []
        
        if team_metrics.quality_score < 0.7:
            suggestions.append({
                "category": "quality",
                "description": "Quality metrics indicate room for improvement",
                "priority": ImprovementPriority.HIGH,
                "impact": "Reduce defects and rework",
                "effort": "Medium",
                "success_criteria": ["Quality score above 0.8"]
            })
        
        if team_metrics.collaboration_score < 0.6:
            suggestions.append({
                "category": "collaboration",
                "description": "Team collaboration could be enhanced",
                "priority": ImprovementPriority.MEDIUM,
                "impact": "Better knowledge sharing and team cohesion",
                "effort": "Low to Medium",
                "success_criteria": ["Collaboration score above 0.75"]
            })
        
        if team_metrics.cycle_time_avg > 5.0:
            suggestions.append({
                "category": "flow",
                "description": "Cycle time is high - consider process optimization",
                "priority": ImprovementPriority.MEDIUM,
                "impact": "Faster feedback loops",
                "effort": "Medium",
                "success_criteria": ["Cycle time reduced by 25%"]
            })
        
        return suggestions

    # Utility methods

    def _serialize_improvement_suggestion(self, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize improvement suggestion for output."""
        return {
            "category": suggestion["category"],
            "description": suggestion["description"],
            "priority": suggestion["priority"].value if hasattr(suggestion["priority"], 'value') else str(suggestion["priority"]),
            "impact": suggestion["impact"],
            "effort": suggestion["effort"],
            "success_criteria": suggestion["success_criteria"]
        }

    def _calculate_overall_health_score(self, team_metrics) -> float:
        """Calculate overall team health score."""
        scores = [
            team_metrics.quality_score,
            team_metrics.collaboration_score,
            team_metrics.delivery_predictability,
            team_metrics.team_satisfaction
        ]
        return sum(scores) / len(scores)

    def _categorize_health_score(self, health_score: float) -> str:
        """Categorize health score into descriptive levels."""
        if health_score >= 0.8:
            return "excellent"
        elif health_score >= 0.7:
            return "good"
        elif health_score >= 0.6:
            return "fair"
        else:
            return "needs attention"

    def _analyze_velocity_trend(self, velocity_trend: List[float]) -> str:
        """Analyze velocity trend pattern."""
        if len(velocity_trend) < 2:
            return "insufficient data"
        
        recent_avg = sum(velocity_trend[-2:]) / 2
        older_avg = sum(velocity_trend[:-2] or velocity_trend) / max(1, len(velocity_trend[:-2] or velocity_trend))
        
        if recent_avg > older_avg * 1.1:
            return "improving"
        elif recent_avg < older_avg * 0.9:
            return "declining"
        else:
            return "stable"

    def _assess_quality_level(self, quality_score: float) -> str:
        """Assess quality level from score."""
        if quality_score >= 0.9:
            return "excellent"
        elif quality_score >= 0.8:
            return "good"
        elif quality_score >= 0.7:
            return "acceptable"
        else:
            return "needs improvement"

    def _generate_health_insights(
        self, 
        team_metrics, 
        health_score: float
    ) -> Dict[str, List[str]]:
        """Generate insights based on team health assessment."""
        risks = []
        recommendations = []
        
        if health_score < 0.6:
            risks.append("Overall team health is low - requires immediate attention")
            recommendations.append("Conduct detailed team assessment and intervention")
        
        if team_metrics.team_satisfaction < 0.6:
            risks.append("Low team satisfaction may lead to turnover")
            recommendations.append("Address team satisfaction issues through surveys and one-on-ones")
        
        if team_metrics.quality_score < 0.7:
            recommendations.append("Implement quality improvement initiatives")
        
        if team_metrics.learning_velocity < 0.5:
            recommendations.append("Invest in team learning and skill development")
        
        return {
            "risks": risks,
            "recommendations": recommendations
        }