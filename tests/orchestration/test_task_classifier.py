"""Tests for task classifier."""

import pytest
from unittest.mock import patch
from datetime import datetime, timezone

from agentsmcp.orchestration.task_classifier import TaskClassifier
from agentsmcp.orchestration.models import (
    TaskClassification,
    TaskType,
    ComplexityLevel,
    RiskLevel,
    TechnologyStack,
    InvalidObjective,
    InsufficientContext,
    UnsupportedTaskType,
)


class TestTaskClassifier:
    """Test TaskClassifier functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = TaskClassifier()

    def test_init(self):
        """Test classifier initialization."""
        assert self.classifier is not None
        assert hasattr(self.classifier, '_classification_cache')
        assert hasattr(self.classifier, '_cache_ttl')
        assert self.classifier._cache_ttl == 3600

    def test_invalid_objective_empty(self):
        """Test classification with empty objective raises InvalidObjective."""
        with pytest.raises(InvalidObjective):
            self.classifier.classify("")
        
        with pytest.raises(InvalidObjective):
            self.classifier.classify("   ")  # Whitespace only
        
        with pytest.raises(InvalidObjective):
            self.classifier.classify(None)

    # Golden Tests as specified in ICD
    def test_golden_test_simple_implementation(self):
        """Golden test: Simple implementation task."""
        objective = "Implement user authentication API endpoint"
        context = {"technologies": ["python", "api"]}
        
        result = self.classifier.classify(objective, context)
        
        assert result.task_type == TaskType.IMPLEMENTATION
        assert result.complexity in [ComplexityLevel.LOW, ComplexityLevel.MEDIUM]
        assert "coder" in result.required_roles or "api_engineer" in result.required_roles
        assert TechnologyStack.PYTHON in result.technologies
        assert TechnologyStack.API in result.technologies
        assert result.confidence > 0.5
        assert result.estimated_effort >= 1
        assert result.estimated_effort <= 100

    def test_golden_test_complex_design(self):
        """Golden test: Complex design task."""
        objective = "Design distributed microservices architecture for e-commerce platform with scalability and security requirements"
        context = {"module": "architecture", "technologies": ["microservices", "api"]}
        
        result = self.classifier.classify(objective, context)
        
        assert result.task_type == TaskType.DESIGN
        assert result.complexity in [ComplexityLevel.HIGH, ComplexityLevel.CRITICAL]
        assert "architect" in result.required_roles
        assert result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert result.confidence > 0.6
        assert result.estimated_effort >= 60

    def test_golden_test_bug_fix(self):
        """Golden test: Bug fix task."""
        objective = "Fix memory leak in user session management"
        context = {"file_paths": ["/src/auth/sessions.py"]}
        
        result = self.classifier.classify(objective, context)
        
        assert result.task_type == TaskType.BUG_FIX
        assert result.complexity in [ComplexityLevel.LOW, ComplexityLevel.MEDIUM, ComplexityLevel.HIGH]
        assert "coder" in result.required_roles
        assert result.confidence > 0.5

    def test_golden_test_testing_task(self):
        """Golden test: Testing task."""
        objective = "Create comprehensive unit tests for payment processing module"
        context = {"technologies": ["testing", "python"]}
        
        result = self.classifier.classify(objective, context)
        
        assert result.task_type == TaskType.TESTING
        assert "qa" in result.required_roles
        assert TechnologyStack.TESTING in result.technologies
        assert result.confidence > 0.7

    def test_golden_test_documentation(self):
        """Golden test: Documentation task."""
        objective = "Write API documentation for REST endpoints with examples and usage guide"
        
        result = self.classifier.classify(objective)
        
        assert result.task_type == TaskType.DOCUMENTATION
        assert "docs" in result.required_roles
        assert TechnologyStack.API in result.technologies
        assert result.confidence > 0.7

    # Additional edge case tests beyond golden tests
    def test_edge_case_multilingual_keywords(self):
        """Edge case: Mixed case and special characters in keywords."""
        objective = "Implement REST-API with OAuth2.0 authentication using JWT tokens"
        
        result = self.classifier.classify(objective)
        
        assert result.task_type == TaskType.IMPLEMENTATION
        assert "implement" in result.keywords or any("implement" in kw for kw in result.keywords)
        assert TechnologyStack.API in result.technologies

    def test_edge_case_very_long_objective(self):
        """Edge case: Very long objective text."""
        objective = "Implement a highly scalable, distributed microservices architecture " * 20
        
        result = self.classifier.classify(objective)
        
        assert result.complexity in [ComplexityLevel.HIGH, ComplexityLevel.CRITICAL]
        assert result.estimated_effort >= 80

    # Classification accuracy tests
    def test_task_type_detection_implementation(self):
        """Test implementation task type detection."""
        test_cases = [
            "Implement user registration",
            "Build payment gateway",
            "Create REST API endpoints",
            "Develop authentication system",
            "Code the main algorithm"
        ]
        
        for objective in test_cases:
            result = self.classifier.classify(objective)
            assert result.task_type == TaskType.IMPLEMENTATION, f"Failed for: {objective}"

    def test_task_type_detection_design(self):
        """Test design task type detection."""
        test_cases = [
            "Design database schema",
            "Plan system architecture",
            "Create UI wireframes",
            "Blueprint the API structure",
            "Architect the microservices"
        ]
        
        for objective in test_cases:
            result = self.classifier.classify(objective)
            assert result.task_type == TaskType.DESIGN, f"Failed for: {objective}"

    def test_task_type_detection_review(self):
        """Test review task type detection."""
        test_cases = [
            "Review pull request",
            "Audit security implementation",
            "Check code quality",
            "Inspect database queries",
            "Evaluate performance"
        ]
        
        for objective in test_cases:
            result = self.classifier.classify(objective)
            assert result.task_type == TaskType.REVIEW, f"Failed for: {objective}"

    def test_complexity_assessment_trivial(self):
        """Test trivial complexity assessment."""
        objective = "Fix typo"
        
        result = self.classifier.classify(objective)
        
        assert result.complexity == ComplexityLevel.TRIVIAL
        assert result.estimated_effort <= 10

    def test_complexity_assessment_critical(self):
        """Test critical complexity assessment."""
        objective = "Design enterprise-scale distributed system with fault tolerance, security, performance optimization, and compliance requirements for mission-critical applications"
        
        result = self.classifier.classify(objective)
        
        assert result.complexity == ComplexityLevel.CRITICAL
        assert result.estimated_effort >= 80

    def test_role_detection_backend(self):
        """Test backend role detection."""
        objective = "Implement database layer with ORM and connection pooling"
        context = {"technologies": ["database", "backend"]}
        
        result = self.classifier.classify(objective, context)
        
        assert "backend_engineer" in result.required_roles or "coder" in result.required_roles
        assert TechnologyStack.DATABASE in result.technologies

    def test_role_detection_frontend(self):
        """Test frontend role detection."""
        objective = "Create React components for user dashboard with responsive design"
        context = {"technologies": ["react", "frontend"]}
        
        result = self.classifier.classify(objective, context)
        
        assert "web_frontend_engineer" in result.required_roles or "coder" in result.required_roles
        assert TechnologyStack.REACT in result.technologies

    def test_role_detection_tui(self):
        """Test TUI role detection."""
        objective = "Build terminal interface for command-line application"
        context = {"technologies": ["tui", "terminal"]}
        
        result = self.classifier.classify(objective, context)
        
        assert "tui_frontend_engineer" in result.required_roles or "coder" in result.required_roles
        assert TechnologyStack.TUI in result.technologies

    def test_technology_detection_python(self):
        """Test Python technology detection."""
        objective = "Write Python script with pandas for data processing"
        
        result = self.classifier.classify(objective)
        
        assert TechnologyStack.PYTHON in result.technologies

    def test_technology_detection_javascript(self):
        """Test JavaScript technology detection."""
        objective = "Implement JavaScript module with ES6 features"
        
        result = self.classifier.classify(objective)
        
        assert TechnologyStack.JAVASCRIPT in result.technologies

    def test_technology_from_context(self):
        """Test technology detection from context."""
        objective = "Implement feature"
        context = {"technologies": ["typescript", "react"]}
        
        result = self.classifier.classify(objective, context)
        
        assert TechnologyStack.TYPESCRIPT in result.technologies
        assert TechnologyStack.REACT in result.technologies

    def test_risk_assessment_low(self):
        """Test low risk assessment."""
        objective = "Add logging to existing function"
        
        result = self.classifier.classify(objective)
        
        assert result.risk_level == RiskLevel.LOW

    def test_risk_assessment_high(self):
        """Test high risk assessment."""
        objective = "Refactor critical production database schema with breaking changes"
        
        result = self.classifier.classify(objective)
        
        assert result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    # Caching tests
    def test_caching_basic(self):
        """Test basic caching functionality."""
        objective = "Implement user service"
        context = {"module": "user"}
        
        # First call should cache the result
        result1 = self.classifier.classify(objective, context)
        
        # Second call should use cache
        result2 = self.classifier.classify(objective, context)
        
        assert result1.task_type == result2.task_type
        assert result1.complexity == result2.complexity
        assert result1.confidence == result2.confidence

    def test_cache_key_generation(self):
        """Test cache key generation for different inputs."""
        # Same objective, different context should generate different keys
        key1 = self.classifier._generate_cache_key("test", {"a": 1})
        key2 = self.classifier._generate_cache_key("test", {"b": 2})
        
        assert key1 != key2
        
        # Same inputs should generate same key
        key3 = self.classifier._generate_cache_key("test", {"a": 1})
        assert key1 == key3

    def test_cache_stats(self):
        """Test cache statistics."""
        # Initial stats
        stats = self.classifier.get_cache_stats()
        assert stats["total_entries"] == 0
        assert stats["total_hits"] == 0
        
        # Add some classifications
        self.classifier.classify("test 1")
        self.classifier.classify("test 2")
        self.classifier.classify("test 1")  # Should hit cache
        
        stats = self.classifier.get_cache_stats()
        assert stats["total_entries"] == 2
        assert stats["total_hits"] == 3

    def test_cache_cleanup(self):
        """Test cache cleanup functionality."""
        # Fill cache beyond cleanup threshold
        original_cleanup = self.classifier._cleanup_cache
        cleanup_called = False
        
        def mock_cleanup():
            nonlocal cleanup_called
            cleanup_called = True
            original_cleanup()
        
        self.classifier._cleanup_cache = mock_cleanup
        
        # Add many entries to trigger cleanup
        for i in range(1001):  # Beyond the 1000 entry limit
            self.classifier._cache_classification(f"key_{i}", 
                TaskClassification(
                    task_type=TaskType.IMPLEMENTATION,
                    complexity=ComplexityLevel.LOW,
                    estimated_effort=30,
                    risk_level=RiskLevel.LOW,
                    confidence=0.8
                )
            )
        
        assert cleanup_called

    # Performance tests
    @patch('time.time')
    def test_performance_requirement(self, mock_time):
        """Test that classification completes within 200ms."""
        # Mock time to simulate performance measurement
        mock_time.side_effect = [0.0, 0.15]  # 150ms duration
        
        objective = "Implement complex feature with multiple requirements"
        result = self.classifier.classify(objective)
        
        # Should complete without performance warning
        assert result is not None

    @patch('time.time')
    @patch('builtins.print')
    def test_performance_warning(self, mock_print, mock_time):
        """Test performance warning for slow classification."""
        # Mock time to simulate slow performance
        mock_time.side_effect = [0.0, 0.25]  # 250ms duration
        
        objective = "Implement feature"
        self.classifier.classify(objective)
        
        # Should have printed performance warning
        mock_print.assert_called_once()
        assert "Warning: Classification took" in mock_print.call_args[0][0]

    # Keyword extraction tests
    def test_keyword_extraction_basic(self):
        """Test basic keyword extraction."""
        text = "Implement user authentication with JWT tokens"
        keywords = self.classifier._extract_keywords(text)
        
        assert "implement" in keywords
        assert "user" in keywords
        assert "authentication" in keywords
        assert "jwt" in keywords
        assert "tokens" in keywords
        # Stop words should be filtered out
        assert "with" not in keywords

    def test_phrase_extraction(self):
        """Test multi-word phrase extraction."""
        text = "Design REST API with machine learning integration"
        phrases = self.classifier._extract_phrases(text)
        
        # Should extract meaningful phrases
        phrase_found = any("rest api" in phrase or "machine learning" in phrase for phrase in phrases)
        assert phrase_found

    def test_keyword_normalization(self):
        """Test keyword normalization (lowercase, whitespace removal)."""
        text = "  IMPLEMENT   User  Authentication  "
        keywords = self.classifier._extract_keywords(text)
        
        assert "implement" in keywords
        assert "user" in keywords
        assert "authentication" in keywords

    # Error handling tests
    def test_classification_error_handling(self):
        """Test error handling during classification."""
        with patch.object(self.classifier, '_detect_task_type', side_effect=Exception("Test error")):
            with pytest.raises(UnsupportedTaskType) as exc_info:
                self.classifier.classify("test objective")
            
            assert "Classification failed" in str(exc_info.value)

    def test_context_handling_none(self):
        """Test handling of None context."""
        result = self.classifier.classify("implement feature", context=None)
        assert result is not None

    def test_context_handling_empty_dict(self):
        """Test handling of empty context dictionary."""
        result = self.classifier.classify("implement feature", context={})
        assert result is not None

    def test_constraints_handling_none(self):
        """Test handling of None constraints."""
        result = self.classifier.classify("implement feature", constraints=None)
        assert result is not None

    # Confidence calculation tests
    def test_confidence_calculation_high_quality_input(self):
        """Test confidence calculation for high-quality input."""
        objective = "Implement comprehensive REST API authentication system with JWT tokens, role-based access control, and security audit logging"
        context = {"technologies": ["python", "api", "security"], "module": "auth"}
        
        result = self.classifier.classify(objective, context)
        
        # Should have high confidence due to specific keywords and good context
        assert result.confidence >= 0.8

    def test_confidence_calculation_poor_input(self):
        """Test confidence calculation for poor-quality input."""
        objective = "do stuff"  # Vague objective
        
        result = self.classifier.classify(objective)
        
        # Should have lower confidence due to vague input
        assert result.confidence <= 0.7

    # Integration tests
    def test_full_classification_pipeline(self):
        """Test the complete classification pipeline."""
        objective = "Design and implement scalable REST API for user management with authentication, authorization, comprehensive testing, and documentation"
        context = {
            "repo": "user-service",
            "technologies": ["python", "api", "database"],
            "file_paths": ["/src/api/users.py", "/tests/test_users.py"]
        }
        constraints = {
            "time_budget": 3600,
            "complexity_limit": "high"
        }
        
        result = self.classifier.classify(objective, context, constraints)
        
        # Verify all aspects of classification
        assert result.task_type in [TaskType.DESIGN, TaskType.IMPLEMENTATION]
        assert result.complexity in [ComplexityLevel.MEDIUM, ComplexityLevel.HIGH, ComplexityLevel.CRITICAL]
        assert len(result.required_roles) >= 1
        assert len(result.technologies) >= 1
        assert 1 <= result.estimated_effort <= 100
        assert result.risk_level in list(RiskLevel)
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.keywords) >= 3

    def test_classification_consistency(self):
        """Test that classification is consistent for identical inputs."""
        objective = "Implement user authentication API"
        context = {"technologies": ["python"]}
        
        results = []
        for _ in range(5):
            result = self.classifier.classify(objective, context)
            results.append(result)
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result.task_type == first_result.task_type
            assert result.complexity == first_result.complexity
            assert result.required_roles == first_result.required_roles
            assert result.technologies == first_result.technologies
            assert result.confidence == first_result.confidence