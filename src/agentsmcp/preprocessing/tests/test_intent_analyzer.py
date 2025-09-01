"""
Tests for Intent Analyzer component.

Tests the NLP-based intent extraction including:
- Intent classification accuracy
- Technical domain detection
- Entity and keyword extraction
- Ambiguity detection
- Performance benchmarks
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from ..intent_analyzer import (
    IntentAnalyzer, IntentType, TechnicalDomain, UrgencyLevel, 
    IntentAnalysis, ExtractedEntity
)


class TestIntentAnalyzer:
    """Test suite for IntentAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create intent analyzer instance."""
        return IntentAnalyzer()
    
    @pytest.mark.asyncio
    async def test_basic_intent_analysis(self, analyzer):
        """Test basic intent analysis functionality."""
        result = await analyzer.analyze_intent("Create a Python web application")
        
        assert isinstance(result, IntentAnalysis)
        assert result.primary_intent == IntentType.TASK_EXECUTION
        assert result.technical_domain == TechnicalDomain.WEB_DEVELOPMENT
        assert result.confidence > 0.5
        assert "create" in result.action_verbs
        assert "python" in result.technologies
    
    @pytest.mark.asyncio
    async def test_information_seeking_intent(self, analyzer):
        """Test classification of information seeking requests."""
        test_cases = [
            "What is machine learning?",
            "How do I set up Docker?", 
            "Explain REST APIs",
            "Tell me about Python decorators"
        ]
        
        for prompt in test_cases:
            result = await analyzer.analyze_intent(prompt)
            assert result.primary_intent == IntentType.INFORMATION_SEEKING
            assert result.confidence > 0.6
    
    @pytest.mark.asyncio
    async def test_task_execution_intent(self, analyzer):
        """Test classification of task execution requests."""
        test_cases = [
            "Create a REST API in Flask",
            "Build a React component",
            "Implement user authentication",
            "Write a Python script to process CSV files"
        ]
        
        for prompt in test_cases:
            result = await analyzer.analyze_intent(prompt)
            assert result.primary_intent == IntentType.TASK_EXECUTION
            assert result.confidence > 0.7
    
    @pytest.mark.asyncio
    async def test_problem_solving_intent(self, analyzer):
        """Test classification of problem solving requests."""
        test_cases = [
            "Fix the memory leak in my application",
            "Debug this Python error",
            "My Docker container won't start",
            "Resolve authentication issues"
        ]
        
        for prompt in test_cases:
            result = await analyzer.analyze_intent(prompt)
            assert result.primary_intent == IntentType.PROBLEM_SOLVING
            assert result.confidence > 0.6
    
    @pytest.mark.asyncio
    async def test_analysis_review_intent(self, analyzer):
        """Test classification of analysis/review requests.""" 
        test_cases = [
            "Review my code for security issues",
            "Analyze the performance of this algorithm",
            "Evaluate the architecture design",
            "Assess the quality of this codebase"
        ]
        
        for prompt in test_cases:
            result = await analyzer.analyze_intent(prompt)
            assert result.primary_intent == IntentType.ANALYSIS_REVIEW
            assert result.confidence > 0.7
    
    @pytest.mark.asyncio
    async def test_social_conversational_intent(self, analyzer):
        """Test classification of social/conversational requests."""
        test_cases = [
            "Hello there!",
            "Good morning",
            "Thanks for your help",
            "How are you doing?"
        ]
        
        for prompt in test_cases:
            result = await analyzer.analyze_intent(prompt)
            assert result.primary_intent == IntentType.SOCIAL_CONVERSATIONAL
            assert result.confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_technical_domain_detection(self, analyzer):
        """Test technical domain classification."""
        domain_tests = [
            ("Create a React web app", TechnicalDomain.WEB_DEVELOPMENT),
            ("Build a machine learning model", TechnicalDomain.DATA_SCIENCE),
            ("Set up Kubernetes cluster", TechnicalDomain.DEVOPS_INFRASTRUCTURE),
            ("Implement OAuth authentication", TechnicalDomain.SECURITY_PRIVACY),
            ("Write a Python class", TechnicalDomain.SOFTWARE_DEVELOPMENT)
        ]
        
        for prompt, expected_domain in domain_tests:
            result = await analyzer.analyze_intent(prompt)
            assert result.technical_domain == expected_domain
    
    @pytest.mark.asyncio
    async def test_complexity_assessment(self, analyzer):
        """Test complexity level assessment."""
        complexity_tests = [
            ("Fix typo", "low"),
            ("Create a simple function", "medium"),
            ("Build a microservices architecture", "high"),
            ("Implement enterprise-grade distributed system", "very_high")
        ]
        
        for prompt, expected_complexity in complexity_tests:
            result = await analyzer.analyze_intent(prompt)
            assert result.complexity_level == expected_complexity
    
    @pytest.mark.asyncio
    async def test_urgency_detection(self, analyzer):
        """Test urgency level detection."""
        urgency_tests = [
            ("URGENT: System is down", UrgencyLevel.CRITICAL),
            ("High priority bug fix needed", UrgencyLevel.HIGH),
            ("Please help when you can", UrgencyLevel.MEDIUM),
            ("Consider this for future", UrgencyLevel.LOW)
        ]
        
        for prompt, expected_urgency in urgency_tests:
            result = await analyzer.analyze_intent(prompt)
            assert result.urgency == expected_urgency
    
    @pytest.mark.asyncio
    async def test_entity_extraction(self, analyzer):
        """Test entity extraction from prompts."""
        result = await analyzer.analyze_intent(
            "Create a Python Flask API with PostgreSQL database and JWT authentication"
        )
        
        # Check for extracted technologies
        extracted_techs = [e.text.lower() for e in result.entities if e.entity_type == "technology"]
        assert any("python" in tech for tech in extracted_techs)
        assert any("flask" in tech for tech in extracted_techs)
        
        # Check for extracted concepts
        extracted_concepts = [e.text.lower() for e in result.entities if e.entity_type == "concept"]
        assert any("api" in concept for concept in extracted_concepts)
    
    @pytest.mark.asyncio
    async def test_ambiguity_detection(self, analyzer):
        """Test detection of ambiguous terms."""
        result = await analyzer.analyze_intent("Fix it and make this better")
        
        assert len(result.ambiguous_terms) > 0
        assert any(term in ["it", "this"] for term in result.ambiguous_terms)
    
    @pytest.mark.asyncio
    async def test_missing_context_detection(self, analyzer):
        """Test detection of missing context.""" 
        result = await analyzer.analyze_intent(
            "Update the file with the new requirements", 
            context={}  # No context provided
        )
        
        assert len(result.missing_context) > 0
        assert "incomplete_specification" in result.missing_context
    
    @pytest.mark.asyncio
    async def test_success_criteria_extraction(self, analyzer):
        """Test extraction of success criteria."""
        result = await analyzer.analyze_intent(
            "Build an API so that users can authenticate securely and access their data"
        )
        
        assert len(result.success_criteria) > 0
        criteria_text = " ".join(result.success_criteria)
        assert "authenticate" in criteria_text.lower()
    
    @pytest.mark.asyncio
    async def test_keyword_extraction(self, analyzer):
        """Test keyword extraction quality."""
        result = await analyzer.analyze_intent(
            "Develop a React application with user authentication and data visualization"
        )
        
        expected_keywords = ["react", "application", "user", "authentication", "data", "visualization"]
        found_keywords = [kw for kw in expected_keywords if kw in result.keywords]
        
        assert len(found_keywords) >= 4  # Should find most key terms
    
    @pytest.mark.asyncio
    async def test_technology_extraction(self, analyzer):
        """Test technology-specific extraction."""
        result = await analyzer.analyze_intent(
            "Use Python Django with PostgreSQL and Redis for caching"
        )
        
        expected_techs = ["python", "django", "postgresql", "redis"]
        found_techs = [tech for tech in expected_techs if tech in result.technologies]
        
        assert len(found_techs) >= 3  # Should identify most technologies
    
    @pytest.mark.asyncio
    async def test_constraint_analysis(self, analyzer):
        """Test constraint and requirement analysis."""
        result = await analyzer.analyze_intent(
            "Build a secure API within budget constraints and deliver by Friday"
        )
        
        assert result.has_constraints
        assert "budget" in result.constraint_types
        assert "time" in result.constraint_types
    
    @pytest.mark.asyncio
    async def test_performance_with_long_input(self, analyzer):
        """Test performance with long input texts."""
        long_input = " ".join(["Create a comprehensive web application"] * 50)
        
        import time
        start_time = time.time()
        result = await analyzer.analyze_intent(long_input)
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert processing_time < 1.0  # Less than 1 second
        assert result.processing_time_ms < 1000
        assert result.raw_input_length == len(long_input)
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, analyzer):
        """Test handling of empty or invalid input."""
        result = await analyzer.analyze_intent("")
        
        assert isinstance(result, IntentAnalysis)
        assert result.confidence < 0.5  # Should have low confidence
        assert result.raw_input_length == 0
    
    @pytest.mark.asyncio
    async def test_special_character_handling(self, analyzer):
        """Test handling of special characters and encoding."""
        special_inputs = [
            "Create an API with €1000 budget",
            "Fix the 'undefined' error in JavaScript",
            "Build app with <form> validation"
        ]
        
        for prompt in special_inputs:
            result = await analyzer.analyze_intent(prompt)
            assert isinstance(result, IntentAnalysis)
            assert result.confidence > 0.3  # Should still process reasonably
    
    @pytest.mark.asyncio
    async def test_multilingual_detection(self, analyzer):
        """Test basic multilingual input detection."""
        result = await analyzer.analyze_intent("Créer une application web")
        
        # Should still attempt to process, even if not English
        assert isinstance(result, IntentAnalysis)
        # Language detection might indicate non-English
        # (This would be enhanced with proper language detection library)
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, analyzer):
        """Test concurrent intent analysis requests."""
        prompts = [
            "Create a web app",
            "Fix the bug",
            "Analyze the performance",
            "What is React?",
            "Build an API"
        ]
        
        # Run analyses concurrently
        tasks = [analyzer.analyze_intent(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(prompts)
        for result in results:
            assert isinstance(result, IntentAnalysis)
            assert result.confidence > 0.3
    
    def test_statistics_tracking(self, analyzer):
        """Test that statistics are properly tracked."""
        initial_count = analyzer.analyses_performed
        
        # Run sync analysis
        result = analyzer._classify_task_sync("test input", {})
        
        assert analyzer.analyses_performed == initial_count + 1
        assert isinstance(result, type(analyzer._classify_task_sync("", {})))
    
    def test_get_analysis_stats(self, analyzer):
        """Test analysis statistics retrieval."""
        stats = analyzer.get_analysis_stats()
        
        assert isinstance(stats, dict)
        assert "total_analyses" in stats
        assert "intent_distribution" in stats
        assert isinstance(stats["intent_distribution"], dict)
    
    @pytest.mark.asyncio
    async def test_context_influence(self, analyzer):
        """Test how context influences analysis."""
        base_prompt = "Create a function"
        
        # Without context
        result_no_context = await analyzer.analyze_intent(base_prompt)
        
        # With technical context
        result_with_context = await analyzer.analyze_intent(
            base_prompt,
            context={
                "previous_subjects": ["machine learning", "data processing"],
                "project_type": "data_science"
            }
        )
        
        # Results should be different based on context
        assert result_no_context.technical_domain != result_with_context.technical_domain or \
               result_no_context.confidence != result_with_context.confidence
    
    @pytest.mark.asyncio 
    async def test_confidence_calibration(self, analyzer):
        """Test confidence score calibration."""
        # High confidence cases
        high_confidence_prompts = [
            "Create a Python Flask web application",
            "Fix the TypeError in line 42",
            "Hello! How are you today?"
        ]
        
        for prompt in high_confidence_prompts:
            result = await analyzer.analyze_intent(prompt)
            assert result.confidence >= 0.7
        
        # Low confidence cases
        low_confidence_prompts = [
            "Fix it",
            "Make this better",
            "Do something"
        ]
        
        for prompt in low_confidence_prompts:
            result = await analyzer.analyze_intent(prompt) 
            assert result.confidence <= 0.5


@pytest.mark.integration
class TestIntentAnalyzerIntegration:
    """Integration tests for IntentAnalyzer."""
    
    @pytest.mark.asyncio
    async def test_real_world_scenarios(self):
        """Test with real-world user prompts."""
        analyzer = IntentAnalyzer()
        
        scenarios = [
            {
                "prompt": "I'm building a e-commerce website and need help implementing user authentication with JWT tokens",
                "expected_intent": IntentType.TASK_EXECUTION,
                "expected_domain": TechnicalDomain.WEB_DEVELOPMENT,
                "expected_technologies": ["jwt"]
            },
            {
                "prompt": "My Python script is throwing a 'KeyError' when processing JSON data from an API",
                "expected_intent": IntentType.PROBLEM_SOLVING,
                "expected_domain": TechnicalDomain.SOFTWARE_DEVELOPMENT,
                "expected_technologies": ["python", "json"]
            },
            {
                "prompt": "Can you review my machine learning model's performance and suggest improvements?",
                "expected_intent": IntentType.ANALYSIS_REVIEW,
                "expected_domain": TechnicalDomain.DATA_SCIENCE,
                "expected_keywords": ["machine", "learning", "performance"]
            }
        ]
        
        for scenario in scenarios:
            result = await analyzer.analyze_intent(scenario["prompt"])
            
            assert result.primary_intent == scenario["expected_intent"]
            assert result.technical_domain == scenario["expected_domain"]
            assert result.confidence > 0.6
            
            if "expected_technologies" in scenario:
                for tech in scenario["expected_technologies"]:
                    assert tech in result.technologies
            
            if "expected_keywords" in scenario:
                for keyword in scenario["expected_keywords"]:
                    assert keyword in result.keywords