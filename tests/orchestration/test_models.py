"""Tests for orchestration models."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from agentsmcp.orchestration.models import (
    TaskClassification,
    TeamComposition,
    AgentSpec,
    ResourceConstraints,
    TaskResult,
    TeamPerformanceMetrics,
    ClassificationCache,
    TaskType,
    ComplexityLevel,
    RiskLevel,
    TechnologyStack,
    CoordinationStrategy,
    TaskClassificationError,
    InvalidObjective,
    InsufficientContext,
    UnsupportedTaskType,
    TeamCompositionError,
    InsufficientResources,
    NoSuitableAgents,
)


class TestTaskClassification:
    """Test TaskClassification model."""

    def test_valid_task_classification(self):
        """Test creating a valid task classification."""
        classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.MEDIUM,
            required_roles=["coder", "qa"],
            optional_roles=["architect"],
            technologies=[TechnologyStack.PYTHON, TechnologyStack.API],
            estimated_effort=45,
            risk_level=RiskLevel.MEDIUM,
            keywords=["implement", "api", "python"],
            confidence=0.85
        )
        
        assert classification.task_type == TaskType.IMPLEMENTATION
        assert classification.complexity == ComplexityLevel.MEDIUM
        assert classification.required_roles == ["coder", "qa"]
        assert classification.optional_roles == ["architect"]
        assert TechnologyStack.PYTHON in classification.technologies
        assert classification.estimated_effort == 45
        assert classification.risk_level == RiskLevel.MEDIUM
        assert "implement" in classification.keywords
        assert classification.confidence == 0.85

    def test_effort_validation(self):
        """Test effort validation (must be 1-100)."""
        # Valid effort
        TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.LOW,
            estimated_effort=50,
            risk_level=RiskLevel.LOW,
            confidence=0.8
        )
        
        # Invalid effort - too low
        with pytest.raises(ValidationError):
            TaskClassification(
                task_type=TaskType.IMPLEMENTATION,
                complexity=ComplexityLevel.LOW,
                estimated_effort=0,
                risk_level=RiskLevel.LOW,
                confidence=0.8
            )
        
        # Invalid effort - too high
        with pytest.raises(ValidationError):
            TaskClassification(
                task_type=TaskType.IMPLEMENTATION,
                complexity=ComplexityLevel.LOW,
                estimated_effort=101,
                risk_level=RiskLevel.LOW,
                confidence=0.8
            )

    def test_confidence_validation(self):
        """Test confidence validation (must be 0.0-1.0)."""
        # Valid confidence
        TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.LOW,
            estimated_effort=50,
            risk_level=RiskLevel.LOW,
            confidence=0.5
        )
        
        # Invalid confidence - too low
        with pytest.raises(ValidationError):
            TaskClassification(
                task_type=TaskType.IMPLEMENTATION,
                complexity=ComplexityLevel.LOW,
                estimated_effort=50,
                risk_level=RiskLevel.LOW,
                confidence=-0.1
            )
        
        # Invalid confidence - too high
        with pytest.raises(ValidationError):
            TaskClassification(
                task_type=TaskType.IMPLEMENTATION,
                complexity=ComplexityLevel.LOW,
                estimated_effort=50,
                risk_level=RiskLevel.LOW,
                confidence=1.1
            )

    def test_roles_validation(self):
        """Test role list validation and cleaning."""
        classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.LOW,
            required_roles=["coder", "", "   ", "qa", "coder"],  # Empty and duplicate entries
            optional_roles=["architect", "", "   "],  # Only valid string types with empty entries
            estimated_effort=50,
            risk_level=RiskLevel.LOW,
            confidence=0.8
        )
        
        # Should clean up empty strings and duplicates
        assert "coder" in classification.required_roles
        assert "qa" in classification.required_roles
        assert "" not in classification.required_roles
        assert "   " not in classification.required_roles
        assert "architect" in classification.optional_roles

    def test_keywords_validation(self):
        """Test keyword list validation and normalization."""
        classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.LOW,
            keywords=["IMPLEMENT", "  API  ", "", "Python"],  # Mixed case, no invalid types
            estimated_effort=50,
            risk_level=RiskLevel.LOW,
            confidence=0.8
        )
        
        # Should normalize to lowercase and strip whitespace
        assert "implement" in classification.keywords
        assert "api" in classification.keywords
        assert "python" in classification.keywords
        assert "" not in classification.keywords


class TestAgentSpec:
    """Test AgentSpec model."""

    def test_valid_agent_spec(self):
        """Test creating a valid agent specification."""
        spec = AgentSpec(
            role="coder",
            model_assignment="codex",
            priority=2,
            resource_requirements={"memory": 1024, "cpu": 2},
            specializations=["python", "api_development"]
        )
        
        assert spec.role == "coder"
        assert spec.model_assignment == "codex"
        assert spec.priority == 2
        assert spec.resource_requirements["memory"] == 1024
        assert "python" in spec.specializations

    def test_role_validation(self):
        """Test role validation (cannot be empty)."""
        # Valid role
        AgentSpec(role="coder", model_assignment="codex")
        
        # Invalid role - empty string
        with pytest.raises(ValidationError):
            AgentSpec(role="", model_assignment="codex")
        
        # Invalid role - whitespace only
        with pytest.raises(ValidationError):
            AgentSpec(role="   ", model_assignment="codex")

    def test_priority_validation(self):
        """Test priority validation (must be 1-10)."""
        # Valid priority
        AgentSpec(role="coder", model_assignment="codex", priority=5)
        
        # Invalid priority - too low
        with pytest.raises(ValidationError):
            AgentSpec(role="coder", model_assignment="codex", priority=0)
        
        # Invalid priority - too high
        with pytest.raises(ValidationError):
            AgentSpec(role="coder", model_assignment="codex", priority=11)


class TestResourceConstraints:
    """Test ResourceConstraints model."""

    def test_valid_resource_constraints(self):
        """Test creating valid resource constraints."""
        constraints = ResourceConstraints(
            max_agents=8,
            memory_limit=2048,
            time_budget=600,
            cost_budget=5.0,
            parallel_limit=4
        )
        
        assert constraints.max_agents == 8
        assert constraints.memory_limit == 2048
        assert constraints.time_budget == 600
        assert constraints.cost_budget == 5.0
        assert constraints.parallel_limit == 4

    def test_agent_limits_validation(self):
        """Test agent count limits validation."""
        # Valid limits
        ResourceConstraints(max_agents=5, parallel_limit=3)
        
        # Invalid max_agents - too low
        with pytest.raises(ValidationError):
            ResourceConstraints(max_agents=0)
        
        # Invalid max_agents - too high
        with pytest.raises(ValidationError):
            ResourceConstraints(max_agents=21)
        
        # Invalid parallel_limit - too low
        with pytest.raises(ValidationError):
            ResourceConstraints(parallel_limit=0)

    def test_cost_budget_validation(self):
        """Test cost budget validation (cannot be negative)."""
        # Valid cost budget
        ResourceConstraints(cost_budget=10.0)
        ResourceConstraints(cost_budget=0.0)  # Zero is allowed
        
        # Invalid cost budget - negative
        with pytest.raises(ValidationError):
            ResourceConstraints(cost_budget=-1.0)


class TestTeamComposition:
    """Test TeamComposition model."""

    def test_valid_team_composition(self):
        """Test creating a valid team composition."""
        primary_agents = [
            AgentSpec(role="coder", model_assignment="codex"),
            AgentSpec(role="qa", model_assignment="claude")
        ]
        
        composition = TeamComposition(
            primary_team=primary_agents,
            load_order=["coder", "qa"],
            coordination_strategy=CoordinationStrategy.SEQUENTIAL,
            confidence_score=0.9
        )
        
        assert len(composition.primary_team) == 2
        assert composition.load_order == ["coder", "qa"]
        assert composition.coordination_strategy == CoordinationStrategy.SEQUENTIAL
        assert composition.confidence_score == 0.9

    def test_primary_team_validation(self):
        """Test primary team validation (cannot be empty)."""
        agents = [AgentSpec(role="coder", model_assignment="codex")]
        
        # Valid team
        TeamComposition(
            primary_team=agents,
            load_order=["coder"],
            coordination_strategy=CoordinationStrategy.SEQUENTIAL,
            confidence_score=0.8
        )
        
        # Invalid team - empty
        with pytest.raises(ValidationError):
            TeamComposition(
                primary_team=[],
                load_order=[],
                coordination_strategy=CoordinationStrategy.SEQUENTIAL,
                confidence_score=0.8
            )

    def test_load_order_validation(self):
        """Test load order validation (must match primary team roles)."""
        agents = [
            AgentSpec(role="coder", model_assignment="codex"),
            AgentSpec(role="qa", model_assignment="claude")
        ]
        
        # Valid load order
        TeamComposition(
            primary_team=agents,
            load_order=["coder", "qa"],
            coordination_strategy=CoordinationStrategy.SEQUENTIAL,
            confidence_score=0.8
        )
        
        # Invalid load order - role not in primary team
        with pytest.raises(ValidationError):
            TeamComposition(
                primary_team=agents,
                load_order=["coder", "architect"],  # architect not in primary team
                coordination_strategy=CoordinationStrategy.SEQUENTIAL,
                confidence_score=0.8
            )


class TestTaskResult:
    """Test TaskResult model."""

    def test_valid_task_result(self):
        """Test creating a valid task result."""
        result = TaskResult(
            task_id="task-123",
            status="completed",
            artifacts=["file1.py", "file2.py"],
            metrics={"duration": 120, "lines_of_code": 500},
            errors=[]
        )
        
        assert result.task_id == "task-123"
        assert result.status == "completed"
        assert len(result.artifacts) == 2
        assert result.metrics["duration"] == 120
        assert len(result.errors) == 0
        assert isinstance(result.completed_at, datetime)

    def test_status_validation(self):
        """Test status validation (must be valid status)."""
        valid_statuses = ['pending', 'running', 'completed', 'failed', 'cancelled']
        
        # Test valid statuses
        for status in valid_statuses:
            result = TaskResult(task_id="test", status=status)
            assert result.status == status.lower()
        
        # Test case insensitive
        TaskResult(task_id="test", status="COMPLETED")
        
        # Invalid status
        with pytest.raises(ValidationError):
            TaskResult(task_id="test", status="invalid_status")


class TestTeamPerformanceMetrics:
    """Test TeamPerformanceMetrics model."""

    def test_valid_performance_metrics(self):
        """Test creating valid performance metrics."""
        metrics = TeamPerformanceMetrics(
            team_id="team-123",
            task_type=TaskType.IMPLEMENTATION,
            success_rate=0.85,
            average_duration=300.5,
            average_cost=2.5,
            total_executions=20
        )
        
        assert metrics.team_id == "team-123"
        assert metrics.task_type == TaskType.IMPLEMENTATION
        assert metrics.success_rate == 0.85
        assert metrics.average_duration == 300.5
        assert metrics.average_cost == 2.5
        assert metrics.total_executions == 20
        assert isinstance(metrics.last_updated, datetime)

    def test_success_rate_validation(self):
        """Test success rate validation (must be 0.0-1.0)."""
        # Valid success rate
        TeamPerformanceMetrics(
            team_id="test",
            task_type=TaskType.IMPLEMENTATION,
            success_rate=0.5,
            total_executions=10
        )
        
        # Invalid success rate - too low
        with pytest.raises(ValidationError):
            TeamPerformanceMetrics(
                team_id="test",
                task_type=TaskType.IMPLEMENTATION,
                success_rate=-0.1,
                total_executions=10
            )
        
        # Invalid success rate - too high
        with pytest.raises(ValidationError):
            TeamPerformanceMetrics(
                team_id="test",
                task_type=TaskType.IMPLEMENTATION,
                success_rate=1.1,
                total_executions=10
            )

    def test_total_executions_validation(self):
        """Test total executions validation (cannot be negative)."""
        # Valid executions
        TeamPerformanceMetrics(
            team_id="test",
            task_type=TaskType.IMPLEMENTATION,
            success_rate=0.8,
            total_executions=0  # Zero is valid
        )
        
        # Invalid executions - negative
        with pytest.raises(ValidationError):
            TeamPerformanceMetrics(
                team_id="test",
                task_type=TaskType.IMPLEMENTATION,
                success_rate=0.8,
                total_executions=-1
            )


class TestClassificationCache:
    """Test ClassificationCache model."""

    def test_valid_classification_cache(self):
        """Test creating valid classification cache."""
        classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.LOW,
            estimated_effort=30,
            risk_level=RiskLevel.LOW,
            confidence=0.8
        )
        
        cache = ClassificationCache(
            objective_hash="abc123",
            context_hash="def456",
            classification=classification,
            hit_count=5
        )
        
        assert cache.objective_hash == "abc123"
        assert cache.context_hash == "def456"
        assert cache.classification == classification
        assert cache.hit_count == 5
        assert isinstance(cache.created_at, datetime)

    def test_is_expired_property(self):
        """Test cache expiration check."""
        classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.LOW,
            estimated_effort=30,
            risk_level=RiskLevel.LOW,
            confidence=0.8
        )
        
        # Fresh cache entry
        cache = ClassificationCache(
            objective_hash="abc123",
            context_hash="def456",
            classification=classification
        )
        
        # Should not be expired with default TTL
        assert not cache.is_expired(3600)  # 1 hour
        
        # Should be expired with very short TTL
        assert cache.is_expired(0)  # 0 seconds

    def test_hit_count_validation(self):
        """Test hit count validation (must be >= 1)."""
        classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.LOW,
            estimated_effort=30,
            risk_level=RiskLevel.LOW,
            confidence=0.8
        )
        
        # Valid hit count
        ClassificationCache(
            objective_hash="abc",
            context_hash="def",
            classification=classification,
            hit_count=1
        )
        
        # Invalid hit count - too low
        with pytest.raises(ValidationError):
            ClassificationCache(
                objective_hash="abc",
                context_hash="def",
                classification=classification,
                hit_count=0
            )


class TestExceptions:
    """Test custom exception classes."""

    def test_task_classification_exceptions(self):
        """Test task classification exception hierarchy."""
        # Base exception
        base_exc = TaskClassificationError("base error")
        assert str(base_exc) == "base error"
        
        # Specific exceptions
        invalid_obj = InvalidObjective("invalid objective")
        assert isinstance(invalid_obj, TaskClassificationError)
        
        insufficient_ctx = InsufficientContext("insufficient context")
        assert isinstance(insufficient_ctx, TaskClassificationError)
        
        unsupported_type = UnsupportedTaskType("unsupported type")
        assert isinstance(unsupported_type, TaskClassificationError)

    def test_team_composition_exceptions(self):
        """Test team composition exception hierarchy."""
        # Base exception
        base_exc = TeamCompositionError("base error")
        assert str(base_exc) == "base error"
        
        # Specific exceptions
        insufficient_res = InsufficientResources("insufficient resources")
        assert isinstance(insufficient_res, TeamCompositionError)
        
        no_agents = NoSuitableAgents("no suitable agents")
        assert isinstance(no_agents, TeamCompositionError)


# Edge case tests
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_lists_and_none_values(self):
        """Test handling of empty lists and None values."""
        # TaskClassification with minimal required fields
        classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.LOW,
            estimated_effort=30,
            risk_level=RiskLevel.LOW,
            confidence=0.8,
            required_roles=[],  # Empty list
            optional_roles=[],  # Empty list
            technologies=[],   # Empty list
            keywords=[]        # Empty list
        )
        
        assert classification.required_roles == []
        assert classification.optional_roles == []
        assert classification.technologies == []
        assert classification.keywords == []

    def test_boundary_values(self):
        """Test boundary values for numeric fields."""
        # Minimum effort and confidence
        TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.TRIVIAL,
            estimated_effort=1,  # Minimum
            risk_level=RiskLevel.LOW,
            confidence=0.0  # Minimum
        )
        
        # Maximum effort and confidence
        TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.CRITICAL,
            estimated_effort=100,  # Maximum
            risk_level=RiskLevel.CRITICAL,
            confidence=1.0  # Maximum
        )
        
        # Resource constraints boundaries
        ResourceConstraints(
            max_agents=1,      # Minimum
            parallel_limit=1   # Minimum
        )
        
        ResourceConstraints(
            max_agents=20,     # Maximum
            parallel_limit=20  # No explicit max, but practical limit
        )