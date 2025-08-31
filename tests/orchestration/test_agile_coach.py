"""Tests for agile coach integration and role classes."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch

from agentsmcp.orchestration.agile_coach import (
    AgileCoachIntegration,
    AgilePhase,
    CeremonyType,
    ImprovementPriority,
    CoachActions,
    RetrospectiveReport,
    CeremonySchedule,
    ImprovementSuggestion,
    TeamMetrics
)
from agentsmcp.orchestration.models import (
    TaskClassification,
    TeamComposition,
    TeamPerformanceMetrics,
    AgentSpec,
    TaskType,
    ComplexityLevel,
    RiskLevel,
    CoordinationStrategy,
    TechnologyStack
)
from agentsmcp.roles.agile_coach import AgileCoachRole
from agentsmcp.roles.base import RoleName, TaskEnvelope, EnvelopeStatus
from agentsmcp.models import EnvelopeMeta


class TestAgileCoachIntegration:
    """Test AgileCoachIntegration class."""

    @pytest.fixture
    def coach_integration(self):
        """Create AgileCoachIntegration instance for testing."""
        return AgileCoachIntegration()

    @pytest.fixture
    def sample_task_classification(self):
        """Create sample task classification for testing."""
        return TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.HIGH,
            required_roles=["coder", "qa"],
            optional_roles=["architect"],
            technologies=[TechnologyStack.PYTHON, TechnologyStack.API],
            estimated_effort=70,
            risk_level=RiskLevel.HIGH,
            keywords=["implement", "api", "python"],
            confidence=0.8
        )

    @pytest.fixture
    def sample_team_composition(self):
        """Create sample team composition for testing."""
        return TeamComposition(
            primary_team=[
                AgentSpec(role="architect", model_assignment="claude", priority=1),
                AgentSpec(role="coder", model_assignment="codex", priority=2),
                AgentSpec(role="qa", model_assignment="ollama", priority=3)
            ],
            load_order=["architect", "coder", "qa"],
            coordination_strategy=CoordinationStrategy.COLLABORATIVE,
            confidence_score=0.85,
            estimated_cost=15.0
        )

    @pytest.fixture
    def sample_performance_metrics(self):
        """Create sample performance metrics for testing."""
        return TeamPerformanceMetrics(
            team_id="test_team_001",
            task_type=TaskType.IMPLEMENTATION,
            success_rate=0.85,
            average_duration=3600.0,
            average_cost=12.5,
            total_executions=25
        )

    @pytest.fixture
    def sample_team_metrics(self):
        """Create sample team metrics for testing."""
        return TeamMetrics(
            velocity_trend=[0.6, 0.7, 0.8, 0.75, 0.9],
            quality_score=0.8,
            collaboration_score=0.75,
            delivery_predictability=0.8,
            cycle_time_avg=3.5,
            defect_rate=0.05,
            team_satisfaction=0.8,
            learning_velocity=0.6
        )

    @pytest.mark.asyncio
    async def test_coach_planning_high_complexity(
        self, 
        coach_integration, 
        sample_task_classification, 
        sample_team_composition
    ):
        """Test planning coaching for high complexity tasks."""
        # Modify to high complexity and critical risk
        sample_task_classification.complexity = ComplexityLevel.CRITICAL
        sample_task_classification.risk_level = RiskLevel.CRITICAL
        
        coach_actions = await coach_integration.coach_planning(
            sample_task_classification,
            sample_team_composition
        )
        
        assert isinstance(coach_actions, CoachActions)
        assert "incremental" in coach_actions.recommended_approach.lower()
        assert len(coach_actions.risk_mitigations) >= 2
        assert coach_actions.confidence_score > 0.0
        assert coach_actions.estimated_velocity > 0.0
        
        # Verify specific risk mitigations for critical tasks
        assert any("checkpoint" in rm.lower() for rm in coach_actions.risk_mitigations)
        assert any("rollback" in rm.lower() or "escalation" in rm.lower() for rm in coach_actions.risk_mitigations)

    @pytest.mark.asyncio
    async def test_coach_planning_small_team(
        self,
        coach_integration,
        sample_task_classification,
        sample_team_composition
    ):
        """Test planning coaching with small team for complex task."""
        # Reduce team size to single member
        sample_team_composition.primary_team = [
            AgentSpec(role="coder", model_assignment="codex", priority=1)
        ]
        sample_team_composition.load_order = ["coder"]
        
        coach_actions = await coach_integration.coach_planning(
            sample_task_classification,
            sample_team_composition
        )
        
        assert isinstance(coach_actions, CoachActions)
        # Should suggest team adjustments for high complexity with small team
        assert len(coach_actions.team_adjustments) > 0
        
        # Velocity should be adjusted for single person team
        assert coach_actions.estimated_velocity < 2.0

    @pytest.mark.asyncio
    async def test_coach_retrospective_analysis(
        self,
        coach_integration,
        sample_performance_metrics,
        sample_team_composition
    ):
        """Test retrospective coaching analysis."""
        execution_results = {
            "completion_rate": 0.95,
            "timeline_adherence": 0.85,
            "defects": 1,
            "rework_count": 2,
            "stakeholder_satisfaction": 0.9
        }
        
        retrospective_report = await coach_integration.coach_retrospective(
            execution_results,
            sample_performance_metrics,
            sample_team_composition
        )
        
        assert isinstance(retrospective_report, RetrospectiveReport)
        assert len(retrospective_report.what_went_well) > 0
        assert retrospective_report.team_health_score > 0.0
        assert retrospective_report.team_health_score <= 1.0
        assert len(retrospective_report.velocity_analysis) > 0
        assert retrospective_report.summary
        
        # High completion rate should be in successes
        assert any("completion" in success.lower() for success in retrospective_report.what_went_well)

    @pytest.mark.asyncio
    async def test_coach_retrospective_poor_performance(
        self,
        coach_integration,
        sample_performance_metrics,
        sample_team_composition
    ):
        """Test retrospective with poor performance metrics."""
        execution_results = {
            "completion_rate": 0.6,
            "timeline_adherence": 0.5,
            "defects": 8,
            "rework_count": 5,
            "stakeholder_satisfaction": 0.4
        }
        
        # Modify performance metrics to be poor
        sample_performance_metrics.success_rate = 0.5
        sample_performance_metrics.average_duration = 10800.0  # 3 hours
        sample_performance_metrics.average_cost = 25.0
        
        retrospective_report = await coach_integration.coach_retrospective(
            execution_results,
            sample_performance_metrics,
            sample_team_composition
        )
        
        assert isinstance(retrospective_report, RetrospectiveReport)
        assert len(retrospective_report.what_could_improve) > 0
        assert len(retrospective_report.action_items) > 0
        assert retrospective_report.team_health_score < 0.8
        
        # Should identify time and cost issues
        assert any("time" in improvement.lower() or "duration" in improvement.lower() 
                  for improvement in retrospective_report.what_could_improve)

    @pytest.mark.asyncio
    async def test_schedule_ceremonies_planning_phase(
        self,
        coach_integration,
        sample_team_composition
    ):
        """Test ceremony scheduling for planning phase."""
        team_specs = sample_team_composition.primary_team
        
        ceremony_schedule = await coach_integration.schedule_ceremonies(
            AgilePhase.PLANNING,
            team_specs
        )
        
        assert isinstance(ceremony_schedule, CeremonySchedule)
        assert ceremony_schedule.phase == AgilePhase.PLANNING
        assert len(ceremony_schedule.upcoming_ceremonies) > 0
        assert ceremony_schedule.estimated_duration > 30  # Planning takes longer
        
        # Should include sprint planning ceremony
        ceremony_types = [c.get("type") for c in ceremony_schedule.upcoming_ceremonies]
        assert any("planning" in str(ct).lower() for ct in ceremony_types)

    @pytest.mark.asyncio
    async def test_schedule_ceremonies_retrospective_phase(
        self,
        coach_integration,
        sample_team_composition
    ):
        """Test ceremony scheduling for retrospective phase."""
        team_specs = sample_team_composition.primary_team
        
        ceremony_schedule = await coach_integration.schedule_ceremonies(
            AgilePhase.RETROSPECTIVE,
            team_specs
        )
        
        assert isinstance(ceremony_schedule, CeremonySchedule)
        assert ceremony_schedule.phase == AgilePhase.RETROSPECTIVE
        assert len(ceremony_schedule.upcoming_ceremonies) > 0
        
        # Should include retrospective ceremonies
        ceremony_types = [c.get("type") for c in ceremony_schedule.upcoming_ceremonies]
        assert any("retrospective" in str(ct).lower() for ct in ceremony_types)

    @pytest.mark.asyncio
    async def test_schedule_ceremonies_daily_standup(
        self,
        coach_integration,
        sample_team_composition
    ):
        """Test ceremony scheduling for daily standup."""
        team_specs = sample_team_composition.primary_team
        
        ceremony_schedule = await coach_integration.schedule_ceremonies(
            AgilePhase.DAILY_STANDUP,
            team_specs
        )
        
        assert isinstance(ceremony_schedule, CeremonySchedule)
        assert ceremony_schedule.phase == AgilePhase.DAILY_STANDUP
        assert ceremony_schedule.estimated_duration == 15  # Standups are short
        
        # Should have all team members as participants
        assert len(ceremony_schedule.recommended_participants) == len(sample_team_composition.primary_team)

    @pytest.mark.asyncio
    async def test_suggest_improvements_quality_issues(
        self,
        coach_integration,
        sample_team_metrics
    ):
        """Test improvement suggestions for quality issues."""
        # Modify metrics to show quality problems
        sample_team_metrics.quality_score = 0.6
        sample_team_metrics.defect_rate = 0.15
        
        suggestions = await coach_integration.suggest_improvements(sample_team_metrics)
        
        assert len(suggestions) > 0
        assert all(isinstance(s, ImprovementSuggestion) for s in suggestions)
        
        # Should include quality improvement suggestion
        quality_suggestions = [s for s in suggestions if s.category == "quality"]
        assert len(quality_suggestions) > 0
        assert quality_suggestions[0].priority in [ImprovementPriority.HIGH, ImprovementPriority.CRITICAL]

    @pytest.mark.asyncio
    async def test_suggest_improvements_collaboration_issues(
        self,
        coach_integration,
        sample_team_metrics
    ):
        """Test improvement suggestions for collaboration issues."""
        # Modify metrics to show collaboration problems
        sample_team_metrics.collaboration_score = 0.5
        sample_team_metrics.team_satisfaction = 0.6
        
        suggestions = await coach_integration.suggest_improvements(sample_team_metrics)
        
        assert len(suggestions) > 0
        
        # Should include collaboration improvement suggestions
        collaboration_suggestions = [s for s in suggestions if s.category == "collaboration"]
        assert len(collaboration_suggestions) > 0
        
        # Should also address satisfaction issues
        satisfaction_suggestions = [s for s in suggestions if s.category == "satisfaction"]
        assert len(satisfaction_suggestions) > 0

    @pytest.mark.asyncio
    async def test_suggest_improvements_velocity_decline(
        self,
        coach_integration,
        sample_team_metrics
    ):
        """Test improvement suggestions for declining velocity."""
        # Show declining velocity trend
        sample_team_metrics.velocity_trend = [0.9, 0.8, 0.7, 0.6, 0.5]
        
        suggestions = await coach_integration.suggest_improvements(sample_team_metrics)
        
        assert len(suggestions) > 0
        
        # Should include velocity improvement suggestion
        velocity_suggestions = [s for s in suggestions if s.category == "velocity"]
        assert len(velocity_suggestions) > 0
        assert velocity_suggestions[0].priority == ImprovementPriority.HIGH

    @pytest.mark.asyncio 
    async def test_suggest_improvements_flow_issues(
        self,
        coach_integration,
        sample_team_metrics
    ):
        """Test improvement suggestions for flow issues."""
        # Show high cycle time and low predictability
        sample_team_metrics.cycle_time_avg = 8.0
        sample_team_metrics.delivery_predictability = 0.6
        
        suggestions = await coach_integration.suggest_improvements(sample_team_metrics)
        
        assert len(suggestions) > 0
        
        # Should include flow and predictability improvements
        flow_suggestions = [s for s in suggestions if s.category == "flow"]
        predictability_suggestions = [s for s in suggestions if s.category == "predictability"]
        
        assert len(flow_suggestions) > 0 or len(predictability_suggestions) > 0

    def test_helper_methods_complexity_assessment(self, coach_integration):
        """Test helper methods for complexity assessment."""
        assert coach_integration._assess_complexity_factor(ComplexityLevel.TRIVIAL) == 0.2
        assert coach_integration._assess_complexity_factor(ComplexityLevel.CRITICAL) == 1.0
        assert coach_integration._assess_complexity_factor(ComplexityLevel.MEDIUM) == 0.6

    def test_helper_methods_risk_assessment(self, coach_integration):
        """Test helper methods for risk assessment."""
        assert coach_integration._assess_risk_factor(RiskLevel.LOW) == 0.2
        assert coach_integration._assess_risk_factor(RiskLevel.CRITICAL) == 1.0
        assert coach_integration._assess_risk_factor(RiskLevel.MEDIUM) == 0.5

    def test_generate_milestones_design_task(self, coach_integration):
        """Test milestone generation for design tasks."""
        milestones = coach_integration._generate_milestones(TaskType.DESIGN, 60)
        
        assert len(milestones) > 0
        assert any("design" in m.lower() for m in milestones)
        assert any("requirements" in m.lower() for m in milestones)

    def test_generate_milestones_implementation_task(self, coach_integration):
        """Test milestone generation for implementation tasks."""
        milestones = coach_integration._generate_milestones(TaskType.IMPLEMENTATION, 80)
        
        assert len(milestones) > 0
        assert any("implementation" in m.lower() or "development" in m.lower() for m in milestones)
        assert any("testing" in m.lower() for m in milestones)

    def test_pattern_tracking(self, coach_integration, sample_team_composition, sample_performance_metrics):
        """Test that coaching patterns are tracked for learning."""
        initial_pattern_count = len(coach_integration.team_patterns)
        
        # Simulate pattern update
        coach_integration._update_team_patterns(
            sample_team_composition,
            sample_performance_metrics,
            ["Good collaboration"],
            ["Need better estimation"]
        )
        
        # Should have recorded patterns
        pattern_key = f"{sample_team_composition.coordination_strategy}_{len(sample_team_composition.primary_team)}"
        assert pattern_key in coach_integration.team_patterns
        
        pattern_data = coach_integration.team_patterns[pattern_key]
        assert "Good collaboration" in pattern_data["success_patterns"]
        assert "Need better estimation" in pattern_data["challenge_patterns"]
        assert len(pattern_data["performance_history"]) > 0


class TestAgileCoachRole:
    """Test AgileCoachRole class."""

    @pytest.fixture
    def agile_coach_role(self):
        """Create AgileCoachRole instance for testing."""
        return AgileCoachRole()

    @pytest.fixture
    def sample_task_envelope(self):
        """Create sample task envelope for testing."""
        return TaskEnvelope(
            id="test_task_001",
            title="Test Agile Coaching",
            description="Test agile coaching functionality",
            payload={
                "coaching_type": "planning",
                "task_classification": {
                    "task_type": "implementation",
                    "complexity": "high",
                    "risk_level": "medium",
                    "estimated_effort": 60,
                    "confidence": 0.8
                },
                "team_composition": {
                    "primary_team": [
                        {"role": "architect", "model_assignment": "claude", "priority": 1},
                        {"role": "coder", "model_assignment": "codex", "priority": 2}
                    ],
                    "coordination_strategy": "collaborative",
                    "confidence_score": 0.85
                }
            },
            meta=EnvelopeMeta()
        )

    def test_role_basic_properties(self, agile_coach_role):
        """Test basic role properties."""
        assert agile_coach_role.name() == RoleName.PROCESS_COACH
        assert len(agile_coach_role.responsibilities()) > 0
        assert len(agile_coach_role.decision_rights()) > 0
        assert agile_coach_role.preferred_agent_type() == "claude"

    def test_role_responsibilities(self, agile_coach_role):
        """Test role responsibilities are comprehensive."""
        responsibilities = agile_coach_role.responsibilities()
        
        # Check for key coaching responsibilities
        responsibility_text = " ".join(responsibilities).lower()
        assert "retrospective" in responsibility_text
        assert "performance" in responsibility_text
        assert "improvement" in responsibility_text
        assert "process" in responsibility_text

    def test_role_decision_rights(self, agile_coach_role):
        """Test role decision rights are appropriate."""
        decision_rights = agile_coach_role.decision_rights()
        
        # Check for key decision areas
        decision_text = " ".join(decision_rights).lower()
        assert "improvement" in decision_text or "process" in decision_text
        assert "team" in decision_text or "coaching" in decision_text

    def test_apply_planning_coaching(self, agile_coach_role, sample_task_envelope):
        """Test applying planning coaching."""
        sample_task_envelope.payload["coaching_type"] = "planning"
        
        result = agile_coach_role.apply(sample_task_envelope)
        
        assert result.status == EnvelopeStatus.SUCCESS
        assert result.role == RoleName.PROCESS_COACH
        assert len(result.decisions) > 0
        assert "coaching_type" in result.outputs
        assert result.outputs["coaching_type"] == "planning"
        
        # Should have coaching actions in output
        assert "coach_actions" in result.outputs
        coach_actions = result.outputs["coach_actions"]
        assert "recommended_approach" in coach_actions
        assert "risk_mitigations" in coach_actions

    def test_apply_retrospective_coaching(self, agile_coach_role, sample_task_envelope):
        """Test applying retrospective coaching."""
        sample_task_envelope.payload.update({
            "coaching_type": "retrospective",
            "execution_results": {
                "completion_rate": 0.9,
                "timeline_adherence": 0.8,
                "defects": 2
            },
            "performance_metrics": {
                "success_rate": 0.85,
                "average_duration": 3600,
                "average_cost": 10.0,
                "total_executions": 15
            }
        })
        
        result = agile_coach_role.apply(sample_task_envelope)
        
        assert result.status == EnvelopeStatus.SUCCESS
        assert result.outputs["coaching_type"] == "retrospective"
        
        # Should have retrospective report in output
        assert "retrospective_report" in result.outputs
        retrospective_report = result.outputs["retrospective_report"]
        assert "what_went_well" in retrospective_report
        assert "what_could_improve" in retrospective_report
        assert "team_health_score" in retrospective_report

    def test_apply_ceremony_scheduling(self, agile_coach_role, sample_task_envelope):
        """Test applying ceremony scheduling coaching."""
        sample_task_envelope.payload.update({
            "coaching_type": "ceremony_scheduling",
            "phase": "planning",
            "team": [
                {"role": "architect", "model_assignment": "claude"},
                {"role": "coder", "model_assignment": "codex"},
                {"role": "qa", "model_assignment": "ollama"}
            ]
        })
        
        result = agile_coach_role.apply(sample_task_envelope)
        
        assert result.status == EnvelopeStatus.SUCCESS
        assert result.outputs["coaching_type"] == "ceremony_scheduling"
        
        # Should have ceremony schedule in output
        assert "ceremony_schedule" in result.outputs
        ceremony_schedule = result.outputs["ceremony_schedule"]
        assert "upcoming_ceremonies" in ceremony_schedule
        assert "recommended_participants" in ceremony_schedule

    def test_apply_team_improvement(self, agile_coach_role, sample_task_envelope):
        """Test applying team improvement coaching."""
        sample_task_envelope.payload.update({
            "coaching_type": "team_improvement",
            "team_metrics": {
                "velocity_trend": [0.7, 0.8, 0.6],
                "quality_score": 0.65,  # Below threshold
                "collaboration_score": 0.75,
                "delivery_predictability": 0.8,
                "cycle_time_avg": 4.0,
                "defect_rate": 0.08,
                "team_satisfaction": 0.75,
                "learning_velocity": 0.6
            }
        })
        
        result = agile_coach_role.apply(sample_task_envelope)
        
        assert result.status == EnvelopeStatus.SUCCESS
        assert result.outputs["coaching_type"] == "team_improvement"
        
        # Should have improvement suggestions
        assert "improvement_suggestions" in result.outputs
        suggestions = result.outputs["improvement_suggestions"]
        assert len(suggestions) > 0
        
        # Should identify quality issue
        quality_suggestions = [s for s in suggestions if s["category"] == "quality"]
        assert len(quality_suggestions) > 0

    def test_apply_health_assessment(self, agile_coach_role, sample_task_envelope):
        """Test applying health assessment coaching."""
        sample_task_envelope.payload.update({
            "coaching_type": "health_assessment",
            "team_metrics": {
                "velocity_trend": [0.8, 0.85, 0.9],
                "quality_score": 0.85,
                "collaboration_score": 0.8,
                "delivery_predictability": 0.85,
                "cycle_time_avg": 3.0,
                "defect_rate": 0.03,
                "team_satisfaction": 0.85,
                "learning_velocity": 0.7
            }
        })
        
        result = agile_coach_role.apply(sample_task_envelope)
        
        assert result.status == EnvelopeStatus.SUCCESS
        assert result.outputs["coaching_type"] == "health_assessment"
        
        # Should have health assessment results
        assert "health_score" in result.outputs
        assert "health_category" in result.outputs
        health_score = result.outputs["health_score"]
        assert 0.0 <= health_score <= 1.0
        
        # With good metrics, should be healthy
        health_category = result.outputs["health_category"]
        assert health_category in ["good", "excellent"]

    def test_apply_general_coaching(self, agile_coach_role, sample_task_envelope):
        """Test applying general coaching."""
        sample_task_envelope.payload["coaching_type"] = "general"
        
        result = agile_coach_role.apply(sample_task_envelope)
        
        assert result.status == EnvelopeStatus.SUCCESS
        assert result.outputs["coaching_type"] == "general"
        
        # Should have general guidance
        assert "general_guidance" in result.outputs
        guidance = result.outputs["general_guidance"]
        assert "focus_areas" in guidance
        assert "recommended_practices" in guidance

    def test_apply_unknown_coaching_type(self, agile_coach_role, sample_task_envelope):
        """Test applying with unknown coaching type falls back to general."""
        sample_task_envelope.payload["coaching_type"] = "unknown_type"
        
        result = agile_coach_role.apply(sample_task_envelope)
        
        assert result.status == EnvelopeStatus.SUCCESS
        # Should fall back to general coaching
        assert result.outputs["coaching_type"] == "general"

    def test_apply_error_handling(self, agile_coach_role):
        """Test error handling in apply method."""
        # Create task with bad enum value that will cause actual error
        bad_task = TaskEnvelope(
            id="bad_task",
            title="Bad Task",
            description="Task with bad enum value",
            payload={
                "coaching_type": "planning", 
                "task_classification": {
                    "task_type": "invalid_task_type",  # This will cause ValueError in enum conversion
                    "complexity": "invalid_complexity",
                    "risk_level": "invalid_risk"
                }
            },
            meta=EnvelopeMeta()
        )
        
        result = agile_coach_role.apply(bad_task)
        
        assert result.status == EnvelopeStatus.ERROR
        assert len(result.errors) > 0
        assert "error" in result.errors[0].lower()

    # Test helper methods

    def test_extract_task_classification(self, agile_coach_role):
        """Test task classification extraction."""
        payload = {
            "task_classification": {
                "task_type": "implementation",
                "complexity": "high",
                "risk_level": "medium",
                "estimated_effort": 75,
                "confidence": 0.9
            }
        }
        
        classification = agile_coach_role._extract_task_classification(payload)
        
        assert classification["task_type"] == TaskType.IMPLEMENTATION
        assert classification["complexity"] == ComplexityLevel.HIGH
        assert classification["risk_level"] == RiskLevel.MEDIUM
        assert classification["estimated_effort"] == 75
        assert classification["confidence"] == 0.9

    def test_extract_task_classification_defaults(self, agile_coach_role):
        """Test task classification extraction with defaults."""
        payload = {"task_classification": {}}
        
        classification = agile_coach_role._extract_task_classification(payload)
        
        # Should have default values
        assert classification["task_type"] == TaskType.IMPLEMENTATION
        assert classification["complexity"] == ComplexityLevel.MEDIUM
        assert classification["risk_level"] == RiskLevel.MEDIUM

    def test_extract_team_composition(self, agile_coach_role):
        """Test team composition extraction."""
        payload = {
            "team_composition": {
                "primary_team": [
                    {"role": "architect", "model_assignment": "claude", "priority": 1},
                    {"role": "coder", "model_assignment": "codex", "priority": 2}
                ],
                "coordination_strategy": "hierarchical",
                "confidence_score": 0.9
            }
        }
        
        composition = agile_coach_role._extract_team_composition(payload)
        
        assert len(composition["primary_team"]) == 2
        assert composition["primary_team"][0]["role"] == "architect"
        assert composition["coordination_strategy"] == CoordinationStrategy.HIERARCHICAL
        assert composition["confidence_score"] == 0.9

    def test_extract_team_metrics(self, agile_coach_role):
        """Test team metrics extraction."""
        payload = {
            "team_metrics": {
                "velocity_trend": [0.7, 0.8, 0.9],
                "quality_score": 0.85,
                "collaboration_score": 0.8,
                "team_satisfaction": 0.9,
                "cycle_time_avg": 2.5
            }
        }
        
        metrics = agile_coach_role._extract_team_metrics(payload)
        
        assert isinstance(metrics, TeamMetrics)
        assert metrics.velocity_trend == [0.7, 0.8, 0.9]
        assert metrics.quality_score == 0.85
        assert metrics.collaboration_score == 0.8
        assert metrics.team_satisfaction == 0.9
        assert metrics.cycle_time_avg == 2.5

    def test_utility_methods(self, agile_coach_role):
        """Test utility methods."""
        # Test velocity trend analysis
        assert agile_coach_role._analyze_velocity_trend([0.7, 0.8, 0.9]) == "improving"
        assert agile_coach_role._analyze_velocity_trend([0.9, 0.8, 0.7]) == "declining"
        assert agile_coach_role._analyze_velocity_trend([0.8, 0.8, 0.8]) == "stable"
        assert agile_coach_role._analyze_velocity_trend([0.8]) == "insufficient data"
        
        # Test quality assessment
        assert agile_coach_role._assess_quality_level(0.95) == "excellent"
        assert agile_coach_role._assess_quality_level(0.85) == "good"
        assert agile_coach_role._assess_quality_level(0.75) == "acceptable"
        assert agile_coach_role._assess_quality_level(0.65) == "needs improvement"
        
        # Test health score categorization
        assert agile_coach_role._categorize_health_score(0.85) == "excellent"
        assert agile_coach_role._categorize_health_score(0.75) == "good"
        assert agile_coach_role._categorize_health_score(0.65) == "fair"
        assert agile_coach_role._categorize_health_score(0.55) == "needs attention"

    def test_integration_with_coach_integration(self, agile_coach_role):
        """Test that role integrates properly with AgileCoachIntegration."""
        # Verify that the role has a coach integration instance
        assert hasattr(agile_coach_role, '_coach_integration')
        assert isinstance(agile_coach_role._coach_integration, AgileCoachIntegration)


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.fixture
    def coach_integration(self):
        """Create AgileCoachIntegration instance for testing."""
        return AgileCoachIntegration()

    @pytest.mark.asyncio
    async def test_empty_velocity_trend(self, coach_integration):
        """Test handling of empty velocity trend."""
        team_metrics = TeamMetrics(
            velocity_trend=[],  # Empty trend
            quality_score=0.8,
            collaboration_score=0.75,
            delivery_predictability=0.8,
            cycle_time_avg=3.5,
            defect_rate=0.05,
            team_satisfaction=0.8,
            learning_velocity=0.6
        )
        
        suggestions = await coach_integration.suggest_improvements(team_metrics)
        
        # Should still provide suggestions based on other metrics
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_all_metrics_excellent(self, coach_integration):
        """Test behavior when all metrics are excellent."""
        team_metrics = TeamMetrics(
            velocity_trend=[0.9, 0.95, 1.0],
            quality_score=0.95,
            collaboration_score=0.95,
            delivery_predictability=0.95,
            cycle_time_avg=1.5,
            defect_rate=0.01,
            team_satisfaction=0.95,
            learning_velocity=0.9
        )
        
        suggestions = await coach_integration.suggest_improvements(team_metrics)
        
        # Should have few or no suggestions when everything is excellent
        assert len(suggestions) <= 2  # Maybe learning or minor optimizations

    @pytest.mark.asyncio
    async def test_ceremony_scheduling_postmortem(self, coach_integration):
        """Test ceremony scheduling for postmortem phase."""
        team_specs = [
            AgentSpec(role="architect", model_assignment="claude", priority=1),
            AgentSpec(role="coder", model_assignment="codex", priority=2)
        ]
        
        ceremony_schedule = await coach_integration.schedule_ceremonies(
            AgilePhase.POSTMORTEM,
            team_specs
        )
        
        assert ceremony_schedule.phase == AgilePhase.POSTMORTEM
        assert ceremony_schedule.estimated_duration == 60  # Postmortems take longer
        
        # Should schedule postmortem ceremony
        ceremony_types = [c.get("type") for c in ceremony_schedule.upcoming_ceremonies]
        assert any("postmortem" in str(ct).lower() for ct in ceremony_types)