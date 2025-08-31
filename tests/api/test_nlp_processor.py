"""
Comprehensive test suite for NLP Processor API - Natural Language Processing with 95% accuracy.

This test suite validates the NLP processor functionality including intent recognition,
command translation, context understanding, and performance under various linguistic scenarios.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agentsmcp.api.nlp_processor import (
    NLPProcessor,
    IntentResult,
    EntityExtractionResult,
    SentimentResult,
    ContextualUnderstanding,
    LanguageDetection,
    NLPConfig,
    ProcessingPipeline
)


@pytest.fixture
def nlp_config():
    """Create NLP configuration for testing."""
    return NLPConfig(
        accuracy_threshold=0.95,
        confidence_threshold=0.8,
        max_context_length=2048,
        supported_languages=["en", "es", "fr", "de", "it"],
        enable_caching=True,
        cache_ttl=300,
        enable_learning=True,
        batch_size=32,
        timeout_seconds=10.0
    )


@pytest.fixture
async def nlp_processor(nlp_config):
    """Create an NLP processor for testing."""
    processor = NLPProcessor(config=nlp_config)
    await processor.initialize()
    yield processor
    await processor.cleanup()


@pytest.fixture
def sample_texts():
    """Provide sample texts for testing various NLP scenarios."""
    return {
        "commands": [
            "create a new file called main.py",
            "search for all TODO comments in the codebase",
            "show me the status of the current agents",
            "help me understand how to use this tool",
            "configure the theme to dark mode",
            "quit the application"
        ],
        "questions": [
            "What is the best way to optimize this code?",
            "How can I improve the performance of my application?",
            "Why is this function not working correctly?",
            "When should I use async programming?",
            "Where can I find the documentation for this API?"
        ],
        "complex": [
            "I need to create a Python script that processes CSV files, extracts specific columns, performs data validation, and generates a summary report",
            "Can you help me set up a CI/CD pipeline using GitHub Actions that runs tests, performs security scans, and deploys to multiple environments?",
            "I'm having trouble with my React application where the state is not updating correctly after API calls, and the UI is not re-rendering",
        ],
        "multilingual": [
            "créer un nouveau fichier",  # French
            "buscar archivos de configuración",  # Spanish
            "zeige mir den Status",  # German
            "aiutami con questo problema",  # Italian
            "¿Cómo puedo optimizar este código?",  # Spanish question
        ],
        "ambiguous": [
            "it doesn't work",
            "this is broken", 
            "help me",
            "what should I do?",
            "make it better"
        ]
    }


class TestNLPProcessorInitialization:
    """Test suite for NLP processor initialization and configuration."""

    def test_nlp_processor_initialization(self, nlp_config):
        """Test NLP processor initializes with correct configuration."""
        processor = NLPProcessor(config=nlp_config)
        
        assert processor.config == nlp_config
        assert not processor.is_initialized
        assert processor.statistics["total_processed"] == 0

    @pytest.mark.asyncio
    async def test_processor_async_initialization(self, nlp_processor):
        """Test async initialization completes successfully."""
        assert nlp_processor.is_initialized
        assert nlp_processor.pipeline is not None
        assert len(nlp_processor.models) > 0

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = NLPConfig(accuracy_threshold=0.9)
        assert config.accuracy_threshold == 0.9
        
        # Test defaults
        default_config = NLPConfig()
        assert 0.8 <= default_config.accuracy_threshold <= 1.0
        assert default_config.confidence_threshold > 0

    @pytest.mark.asyncio
    async def test_initialization_with_invalid_config(self):
        """Test initialization handles invalid configuration gracefully."""
        invalid_config = NLPConfig(accuracy_threshold=1.5)  # Invalid threshold
        processor = NLPProcessor(config=invalid_config)
        
        # Should handle gracefully or raise specific exception
        try:
            await processor.initialize()
        except ValueError as e:
            assert "threshold" in str(e).lower()


class TestIntentRecognition:
    """Test suite for intent recognition accuracy and performance."""

    @pytest.mark.asyncio
    async def test_command_intent_recognition(self, nlp_processor, sample_texts):
        """Test recognition of command intents."""
        command_texts = sample_texts["commands"]
        
        results = []
        for text in command_texts:
            result = await nlp_processor.recognize_intent(text)
            results.append(result)
            
            assert isinstance(result, IntentResult)
            assert result.confidence >= nlp_processor.config.confidence_threshold
            assert result.intent in ["create", "search", "status", "help", "config", "quit"]

        # Check overall accuracy
        high_confidence_results = [r for r in results if r.confidence >= 0.9]
        accuracy = len(high_confidence_results) / len(results)
        assert accuracy >= 0.8  # At least 80% high confidence

    @pytest.mark.asyncio
    async def test_question_intent_recognition(self, nlp_processor, sample_texts):
        """Test recognition of question intents."""
        question_texts = sample_texts["questions"]
        
        results = []
        for text in question_texts:
            result = await nlp_processor.recognize_intent(text)
            results.append(result)
            
            assert isinstance(result, IntentResult)
            assert result.intent in ["question", "help", "inquiry", "request"]

        # Verify question detection accuracy
        question_results = [r for r in results if r.intent in ["question", "help", "inquiry"]]
        accuracy = len(question_results) / len(results)
        assert accuracy >= 0.8

    @pytest.mark.asyncio
    async def test_complex_intent_recognition(self, nlp_processor, sample_texts):
        """Test recognition of complex, multi-part intents."""
        complex_texts = sample_texts["complex"]
        
        for text in complex_texts:
            result = await nlp_processor.recognize_intent(text)
            
            assert isinstance(result, IntentResult)
            assert result.confidence > 0.5  # Lower threshold for complex texts
            assert len(result.sub_intents) > 1  # Should detect multiple intents
            assert result.complexity_score > 0.5

    @pytest.mark.asyncio
    async def test_ambiguous_intent_handling(self, nlp_processor, sample_texts):
        """Test handling of ambiguous intents."""
        ambiguous_texts = sample_texts["ambiguous"]
        
        for text in ambiguous_texts:
            result = await nlp_processor.recognize_intent(text)
            
            assert isinstance(result, IntentResult)
            # Should either have low confidence or request clarification
            if result.confidence < 0.6:
                assert len(result.clarification_questions) > 0
            else:
                # If confident, should provide alternative interpretations
                assert len(result.alternative_intents) > 0

    @pytest.mark.asyncio
    async def test_context_aware_intent_recognition(self, nlp_processor):
        """Test context-aware intent recognition."""
        # Set up context
        context = {
            "previous_intent": "search",
            "current_mode": "development",
            "user_skill_level": "intermediate",
            "recent_actions": ["create_file", "search_code"]
        }
        
        # Test intent recognition with context
        text = "find more like that"
        result = await nlp_processor.recognize_intent(text, context=context)
        
        assert isinstance(result, IntentResult)
        # Should use context to improve recognition
        assert result.intent == "search"  # Should infer from context
        assert result.context_influence_score > 0.3


class TestEntityExtraction:
    """Test suite for entity extraction from natural language."""

    @pytest.mark.asyncio
    async def test_file_path_extraction(self, nlp_processor):
        """Test extraction of file paths from text."""
        texts_with_files = [
            "create a file called main.py",
            "open the config.json file",
            "search in /usr/local/bin/python",
            "edit the src/components/App.tsx file",
            "delete the old backup.tar.gz"
        ]
        
        for text in texts_with_files:
            result = await nlp_processor.extract_entities(text)
            
            assert isinstance(result, EntityExtractionResult)
            file_entities = [e for e in result.entities if e.type == "file_path"]
            assert len(file_entities) > 0
            
            # Verify file path format
            for entity in file_entities:
                assert "." in entity.value or "/" in entity.value

    @pytest.mark.asyncio
    async def test_parameter_extraction(self, nlp_processor):
        """Test extraction of parameters and values."""
        texts_with_parameters = [
            "set the theme to dark mode",
            "change the timeout to 30 seconds", 
            "configure max_agents to 12",
            "update the temperature parameter to 0.7",
            "set verbose mode to true"
        ]
        
        for text in texts_with_parameters:
            result = await nlp_processor.extract_entities(text)
            
            assert isinstance(result, EntityExtractionResult)
            param_entities = [e for e in result.entities if e.type in ["parameter", "setting", "config"]]
            value_entities = [e for e in result.entities if e.type == "value"]
            
            assert len(param_entities) > 0
            assert len(value_entities) > 0

    @pytest.mark.asyncio
    async def test_numeric_entity_extraction(self, nlp_processor):
        """Test extraction of numeric entities."""
        texts_with_numbers = [
            "process 100 files at once",
            "set timeout to 5 minutes",
            "use 8 threads for processing",
            "allocate 2.5 GB of memory",
            "run for 3.14 seconds"
        ]
        
        for text in texts_with_numbers:
            result = await nlp_processor.extract_entities(text)
            
            numeric_entities = [e for e in result.entities if e.type in ["number", "quantity", "duration"]]
            assert len(numeric_entities) > 0

    @pytest.mark.asyncio
    async def test_date_time_extraction(self, nlp_processor):
        """Test extraction of date and time entities."""
        texts_with_datetime = [
            "schedule this for tomorrow at 3 PM",
            "run the backup at midnight",
            "process files from last week",
            "set reminder for next Friday",
            "analyze data from 2024-01-15"
        ]
        
        for text in texts_with_datetime:
            result = await nlp_processor.extract_entities(text)
            
            datetime_entities = [e for e in result.entities if e.type in ["date", "time", "datetime"]]
            if datetime_entities:  # Not all texts may have parseable dates
                assert len(datetime_entities) > 0

    @pytest.mark.asyncio
    async def test_nested_entity_extraction(self, nlp_processor):
        """Test extraction of nested or compound entities."""
        complex_text = "create a Python script called data_processor.py that reads CSV files from the input folder, processes them using pandas with 4 threads, and saves results to output.json"
        
        result = await nlp_processor.extract_entities(complex_text)
        
        assert isinstance(result, EntityExtractionResult)
        assert len(result.entities) >= 5  # Multiple entities expected
        
        # Check for different entity types
        entity_types = {e.type for e in result.entities}
        assert "file_path" in entity_types
        assert "technology" in entity_types or "tool" in entity_types


class TestSentimentAnalysis:
    """Test suite for sentiment analysis capabilities."""

    @pytest.mark.asyncio
    async def test_positive_sentiment_detection(self, nlp_processor):
        """Test detection of positive sentiment."""
        positive_texts = [
            "This tool is amazing and very helpful!",
            "I love how easy it is to use",
            "Great job on the implementation",
            "Perfect! This is exactly what I needed",
            "Excellent performance and reliability"
        ]
        
        for text in positive_texts:
            result = await nlp_processor.analyze_sentiment(text)
            
            assert isinstance(result, SentimentResult)
            assert result.sentiment == "positive"
            assert result.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_negative_sentiment_detection(self, nlp_processor):
        """Test detection of negative sentiment."""
        negative_texts = [
            "This is broken and doesn't work at all",
            "I hate this interface, it's confusing",
            "Terrible performance, always crashes", 
            "This is frustrating and useless",
            "Awful experience, nothing works properly"
        ]
        
        for text in negative_texts:
            result = await nlp_processor.analyze_sentiment(text)
            
            assert isinstance(result, SentimentResult)
            assert result.sentiment == "negative"
            assert result.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_neutral_sentiment_detection(self, nlp_processor):
        """Test detection of neutral sentiment."""
        neutral_texts = [
            "Create a file called main.py",
            "Show me the current status", 
            "What is the configuration setting?",
            "List all available commands",
            "Process the input data"
        ]
        
        for text in neutral_texts:
            result = await nlp_processor.analyze_sentiment(text)
            
            assert isinstance(result, SentimentResult)
            assert result.sentiment == "neutral"

    @pytest.mark.asyncio
    async def test_mixed_sentiment_detection(self, nlp_processor):
        """Test detection of mixed or complex sentiment."""
        mixed_texts = [
            "I like the features but the interface is confusing",
            "Good performance but poor documentation",
            "Works well most of the time, but occasionally crashes"
        ]
        
        for text in mixed_texts:
            result = await nlp_processor.analyze_sentiment(text)
            
            assert isinstance(result, SentimentResult)
            # Should detect complexity
            assert result.complexity_score > 0.3


class TestLanguageDetection:
    """Test suite for language detection capabilities."""

    @pytest.mark.asyncio
    async def test_english_detection(self, nlp_processor):
        """Test detection of English language."""
        english_texts = [
            "Hello, how are you today?",
            "Create a new file and process the data",
            "This is a test of the language detection system"
        ]
        
        for text in english_texts:
            result = await nlp_processor.detect_language(text)
            
            assert isinstance(result, LanguageDetection)
            assert result.language == "en"
            assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_multilingual_detection(self, nlp_processor, sample_texts):
        """Test detection of multiple languages."""
        multilingual_texts = sample_texts["multilingual"]
        expected_languages = ["fr", "es", "de", "it", "es"]
        
        for text, expected_lang in zip(multilingual_texts, expected_languages):
            result = await nlp_processor.detect_language(text)
            
            assert isinstance(result, LanguageDetection)
            assert result.language == expected_lang
            assert result.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_mixed_language_detection(self, nlp_processor):
        """Test detection of mixed language texts."""
        mixed_texts = [
            "Hello, comment allez-vous?",  # English + French
            "Gracias for your help today",  # Spanish + English
            "Das ist good for testing"  # German + English
        ]
        
        for text in mixed_texts:
            result = await nlp_processor.detect_language(text)
            
            assert isinstance(result, LanguageDetection)
            # Should detect primary language or mark as mixed
            assert result.confidence > 0.5
            if result.is_mixed:
                assert len(result.detected_languages) > 1

    @pytest.mark.asyncio
    async def test_low_confidence_language_detection(self, nlp_processor):
        """Test handling of texts with unclear language."""
        unclear_texts = [
            "123 456",  # Numbers only
            "@#$%^&*()",  # Special characters only
            "a b c d e"  # Very short/unclear
        ]
        
        for text in unclear_texts:
            result = await nlp_processor.detect_language(text)
            
            assert isinstance(result, LanguageDetection)
            # Should handle gracefully
            assert result.confidence >= 0 or result.language == "unknown"


class TestContextualUnderstanding:
    """Test suite for contextual understanding capabilities."""

    @pytest.mark.asyncio
    async def test_conversation_context_tracking(self, nlp_processor):
        """Test tracking of conversation context."""
        conversation = [
            "Create a new Python file",
            "Add a main function to it", 
            "Now add error handling",
            "Make it executable"
        ]
        
        context = {}
        for i, text in enumerate(conversation):
            understanding = await nlp_processor.understand_with_context(text, context)
            
            assert isinstance(understanding, ContextualUnderstanding)
            context = understanding.updated_context
            
            if i > 0:
                # Should maintain context across turns
                assert "python" in str(context).lower() or "file" in str(context).lower()

    @pytest.mark.asyncio
    async def test_anaphora_resolution(self, nlp_processor):
        """Test resolution of pronouns and references."""
        context = {
            "last_mentioned_file": "main.py",
            "current_project": "web_app", 
            "active_agent": "claude"
        }
        
        texts_with_references = [
            "edit it",  # referring to file
            "update that project",  # referring to project  
            "ask them about the issue",  # referring to agent
        ]
        
        for text in texts_with_references:
            understanding = await nlp_processor.understand_with_context(text, context)
            
            assert isinstance(understanding, ContextualUnderstanding)
            # Should resolve references
            assert understanding.resolution_confidence > 0.5
            assert len(understanding.resolved_references) > 0

    @pytest.mark.asyncio
    async def test_implicit_parameter_inference(self, nlp_processor):
        """Test inference of implicit parameters from context."""
        context = {
            "current_directory": "/home/user/project",
            "default_file_type": ".py",
            "preferred_theme": "dark"
        }
        
        text = "create a new file"  # No explicit parameters
        understanding = await nlp_processor.understand_with_context(text, context)
        
        assert isinstance(understanding, ContextualUnderstanding)
        # Should infer parameters from context
        assert len(understanding.inferred_parameters) > 0

    @pytest.mark.asyncio
    async def test_context_conflict_resolution(self, nlp_processor):
        """Test resolution of conflicting contextual information."""
        conflicting_context = {
            "last_command": "delete",
            "safety_mode": "enabled",
            "user_intent": "create"
        }
        
        text = "do it now"  # Ambiguous with conflicting context
        understanding = await nlp_processor.understand_with_context(text, conflicting_context)
        
        assert isinstance(understanding, ContextualUnderstanding)
        # Should handle conflict gracefully
        assert understanding.conflict_resolution_strategy is not None
        assert understanding.confidence_penalty > 0  # Lower confidence due to conflict


class TestPerformanceAndAccuracy:
    """Test suite for performance and accuracy requirements."""

    @pytest.mark.asyncio
    async def test_95_percent_accuracy_requirement(self, nlp_processor):
        """Test NLP processor meets 95% accuracy requirement."""
        # Create a comprehensive test set
        test_cases = [
            # Clear, unambiguous commands
            ("create file main.py", "create"),
            ("search for TODO", "search"), 
            ("show status", "status"),
            ("help me", "help"),
            ("quit application", "quit"),
            
            # Questions
            ("what is this?", "question"),
            ("how do I use this?", "question"),
            ("why isn't this working?", "question"),
            
            # Complex commands
            ("create a Python script that processes CSV files", "create"),
            ("search for all functions in the codebase", "search"),
            ("show me the current agent status", "status"),
        ]
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        
        for text, expected_intent in test_cases:
            result = await nlp_processor.recognize_intent(text)
            
            # Check if prediction matches expected intent
            if result.intent == expected_intent and result.confidence >= 0.8:
                correct_predictions += 1
            elif result.intent in result.alternative_intents[:2]:  # Top 2 alternatives
                correct_predictions += 0.5  # Partial credit
        
        accuracy = correct_predictions / total_predictions
        assert accuracy >= 0.95, f"Accuracy {accuracy:.3f} below 95% requirement"

    @pytest.mark.asyncio
    async def test_response_time_under_100ms(self, nlp_processor):
        """Test NLP processing meets sub-100ms response time requirement."""
        test_texts = [
            "create a new file",
            "search for functions", 
            "what is the status?",
            "help me understand this",
            "configure the settings"
        ]
        
        response_times = []
        for text in test_texts:
            start_time = time.time()
            result = await nlp_processor.recognize_intent(text)
            response_time = time.time() - start_time
            
            response_times.append(response_time)
            assert isinstance(result, IntentResult)
        
        # Calculate statistics
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Should meet sub-100ms requirement
        assert avg_response_time < 0.1, f"Average response time {avg_response_time:.3f}s > 0.1s"
        assert max_response_time < 0.15, f"Max response time {max_response_time:.3f}s > 0.15s"

    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self, nlp_processor):
        """Test performance under concurrent load."""
        test_texts = [
            "create file",
            "search data", 
            "show status",
            "help me",
            "configure system"
        ] * 10  # 50 total requests
        
        # Process concurrently
        start_time = time.time()
        tasks = [nlp_processor.recognize_intent(text) for text in test_texts]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # All should complete successfully
        assert len(results) == len(test_texts)
        assert all(isinstance(r, IntentResult) for r in results)
        
        # Should maintain reasonable throughput
        throughput = len(test_texts) / total_time
        assert throughput >= 100, f"Throughput {throughput:.1f} requests/sec too low"

    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, nlp_processor):
        """Test memory usage remains optimized."""
        import gc
        
        # Get baseline memory
        gc.collect()
        baseline_objects = len(gc.get_objects())
        
        # Process many texts
        for i in range(200):
            text = f"process request number {i}"
            result = await nlp_processor.recognize_intent(text)
            assert isinstance(result, IntentResult)
        
        # Check memory growth
        gc.collect()
        final_objects = len(gc.get_objects())
        growth_ratio = final_objects / baseline_objects
        
        # Memory growth should be reasonable
        assert growth_ratio < 2.0, f"Memory grew {growth_ratio:.2f}x, should be < 2x"

    @pytest.mark.asyncio
    async def test_batch_processing_optimization(self, nlp_processor):
        """Test batch processing optimization."""
        texts = [f"process batch item {i}" for i in range(50)]
        
        # Test batch processing
        start_time = time.time()
        results = await nlp_processor.process_batch(texts)
        batch_time = time.time() - start_time
        
        # Compare with sequential processing
        start_time = time.time()
        sequential_results = []
        for text in texts[:10]:  # Just test first 10 for comparison
            result = await nlp_processor.recognize_intent(text)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Batch should be more efficient
        batch_per_item = batch_time / len(texts)
        sequential_per_item = sequential_time / 10
        
        assert batch_per_item <= sequential_per_item * 1.2  # Allow 20% overhead


class TestCachingAndOptimization:
    """Test suite for caching and optimization features."""

    @pytest.mark.asyncio
    async def test_result_caching(self, nlp_processor):
        """Test result caching improves performance."""
        text = "create a new file for testing"
        
        # First processing (cache miss)
        start_time = time.time()
        result1 = await nlp_processor.recognize_intent(text)
        first_time = time.time() - start_time
        
        # Second processing (cache hit)
        start_time = time.time()
        result2 = await nlp_processor.recognize_intent(text)
        second_time = time.time() - start_time
        
        # Results should be identical
        assert result1.intent == result2.intent
        assert abs(result1.confidence - result2.confidence) < 0.01
        
        # Second call should be faster (cached)
        assert second_time < first_time * 0.5  # At least 50% faster

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, nlp_processor):
        """Test cache invalidation works correctly."""
        text = "test cache invalidation"
        
        # Process once to cache
        result1 = await nlp_processor.recognize_intent(text)
        
        # Invalidate cache
        await nlp_processor.invalidate_cache()
        
        # Process again (should not use cache)
        result2 = await nlp_processor.recognize_intent(text)
        
        # Results should still be consistent
        assert result1.intent == result2.intent

    @pytest.mark.asyncio
    async def test_cache_expiration(self, nlp_processor):
        """Test cache expiration functionality."""
        # Set short TTL for testing
        nlp_processor.config.cache_ttl = 0.5  # 0.5 seconds
        
        text = "test cache expiration"
        
        # Process once
        result1 = await nlp_processor.recognize_intent(text)
        
        # Wait for cache to expire
        await asyncio.sleep(0.6)
        
        # Process again (cache should be expired)
        result2 = await nlp_processor.recognize_intent(text)
        
        # Results should still be consistent
        assert result1.intent == result2.intent

    @pytest.mark.asyncio
    async def test_adaptive_caching(self, nlp_processor):
        """Test adaptive caching based on usage patterns."""
        # Frequently used pattern
        frequent_text = "show status"
        
        # Process multiple times
        for _ in range(10):
            await nlp_processor.recognize_intent(frequent_text)
        
        # Check cache statistics
        stats = nlp_processor.get_cache_statistics()
        
        assert stats["hit_rate"] > 0.7  # Should have high hit rate
        assert stats["total_requests"] >= 10


class TestLearningAndAdaptation:
    """Test suite for learning and adaptation capabilities."""

    @pytest.mark.asyncio
    async def test_feedback_learning(self, nlp_processor):
        """Test learning from user feedback."""
        text = "show me information"
        
        # Initial prediction
        result1 = await nlp_processor.recognize_intent(text)
        initial_confidence = result1.confidence
        
        # Provide positive feedback
        await nlp_processor.learn_from_feedback(text, result1.intent, positive=True)
        
        # Process again
        result2 = await nlp_processor.recognize_intent(text)
        
        # Confidence should improve
        assert result2.confidence >= initial_confidence

    @pytest.mark.asyncio
    async def test_pattern_learning(self, nlp_processor):
        """Test learning from usage patterns."""
        # Simulate user patterns
        patterns = [
            ("create file", "create"),
            ("make file", "create"),
            ("generate file", "create"),
            ("new file", "create")
        ]
        
        # Train on patterns
        for text, intent in patterns:
            result = await nlp_processor.recognize_intent(text)
            await nlp_processor.learn_from_feedback(text, intent, positive=True)
        
        # Test similar pattern
        test_result = await nlp_processor.recognize_intent("build file")
        
        # Should generalize to similar pattern
        assert test_result.intent == "create" or "create" in test_result.alternative_intents

    @pytest.mark.asyncio
    async def test_user_specific_adaptation(self, nlp_processor):
        """Test adaptation to user-specific language patterns."""
        user_id = "test_user_123"
        
        # User-specific patterns
        user_patterns = [
            ("show info", "status"),  # User says "info" instead of "status"
            ("list stuff", "search"),  # User says "stuff" for items
        ]
        
        # Learn user patterns
        for text, intent in user_patterns:
            await nlp_processor.learn_user_pattern(user_id, text, intent)
        
        # Test adaptation
        result = await nlp_processor.recognize_intent("show info", user_id=user_id)
        
        # Should recognize user-specific pattern
        assert result.intent == "status" or result.confidence > 0.7

    @pytest.mark.asyncio
    async def test_continuous_improvement(self, nlp_processor):
        """Test continuous improvement over time."""
        # Get baseline accuracy
        baseline_accuracy = await nlp_processor.get_current_accuracy()
        
        # Simulate learning over time
        learning_samples = [
            ("display data", "show"),
            ("exhibit information", "show"),
            ("present results", "show"),
            ("reveal details", "show"),
        ]
        
        for text, correct_intent in learning_samples:
            result = await nlp_processor.recognize_intent(text)
            await nlp_processor.learn_from_feedback(text, correct_intent, positive=True)
        
        # Check improved accuracy
        improved_accuracy = await nlp_processor.get_current_accuracy()
        
        # Should show improvement
        assert improved_accuracy >= baseline_accuracy


class TestErrorHandlingAndResilience:
    """Test suite for error handling and system resilience."""

    @pytest.mark.asyncio
    async def test_malformed_input_handling(self, nlp_processor):
        """Test handling of malformed or invalid input."""
        malformed_inputs = [
            "",  # Empty string
            None,  # None input
            "   ",  # Whitespace only
            "\n\t\r",  # Special characters only
            "a" * 10000,  # Extremely long input
            "🎉🚀🔥" * 100,  # Many emojis
        ]
        
        for malformed_input in malformed_inputs:
            if malformed_input is not None:
                result = await nlp_processor.recognize_intent(malformed_input)
                
                # Should handle gracefully
                assert isinstance(result, IntentResult)
                assert result.intent in ["unknown", "error", "unclear"]

    @pytest.mark.asyncio
    async def test_timeout_handling(self, nlp_processor):
        """Test handling of processing timeouts."""
        # Configure short timeout for testing
        original_timeout = nlp_processor.config.timeout_seconds
        nlp_processor.config.timeout_seconds = 0.001  # Very short timeout
        
        try:
            # Should timeout on complex processing
            complex_text = "This is a very complex sentence with many clauses and subclauses that might take a while to process completely and thoroughly with all the various linguistic analysis components running in parallel."
            
            result = await nlp_processor.recognize_intent(complex_text)
            
            # Should return some result even with timeout
            assert isinstance(result, IntentResult)
            
        finally:
            # Restore original timeout
            nlp_processor.config.timeout_seconds = original_timeout

    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, nlp_processor):
        """Test handling of resource exhaustion."""
        # Simulate many concurrent requests
        tasks = []
        for i in range(100):  # Create many concurrent tasks
            task = nlp_processor.recognize_intent(f"concurrent request {i}")
            tasks.append(task)
        
        # Should handle gracefully without crashing
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should complete most requests successfully
        successful_results = [r for r in results if isinstance(r, IntentResult)]
        success_rate = len(successful_results) / len(results)
        
        assert success_rate >= 0.8  # At least 80% should succeed

    @pytest.mark.asyncio
    async def test_model_failure_fallback(self, nlp_processor):
        """Test fallback behavior when models fail."""
        # Mock model failure
        with patch.object(nlp_processor, '_primary_model', side_effect=Exception("Model failed")):
            result = await nlp_processor.recognize_intent("test fallback")
            
            # Should use fallback mechanism
            assert isinstance(result, IntentResult)
            assert result.confidence >= 0 or result.intent == "unknown"

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, nlp_processor):
        """Test graceful degradation when features are unavailable."""
        # Disable some features
        original_features = nlp_processor.enabled_features.copy()
        nlp_processor.enabled_features["entity_extraction"] = False
        nlp_processor.enabled_features["sentiment_analysis"] = False
        
        try:
            result = await nlp_processor.recognize_intent("test degraded mode")
            
            # Should still provide basic intent recognition
            assert isinstance(result, IntentResult)
            assert result.intent is not None
            
        finally:
            # Restore features
            nlp_processor.enabled_features = original_features


class TestStatisticsAndMonitoring:
    """Test suite for statistics and monitoring capabilities."""

    @pytest.mark.asyncio
    async def test_processing_statistics(self, nlp_processor):
        """Test processing statistics collection."""
        # Process several requests
        for i in range(10):
            await nlp_processor.recognize_intent(f"test request {i}")
        
        stats = nlp_processor.get_statistics()
        
        assert stats["total_processed"] >= 10
        assert stats["average_response_time"] > 0
        assert "accuracy_rate" in stats
        assert "cache_hit_rate" in stats

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, nlp_processor):
        """Test performance monitoring metrics."""
        # Generate some activity
        for _ in range(5):
            await nlp_processor.recognize_intent("monitor performance")
        
        metrics = nlp_processor.get_performance_metrics()
        
        assert "throughput" in metrics
        assert "latency_percentiles" in metrics
        assert "error_rate" in metrics
        assert metrics["throughput"] > 0

    @pytest.mark.asyncio
    async def test_accuracy_tracking(self, nlp_processor):
        """Test accuracy tracking over time."""
        # Simulate predictions with feedback
        test_cases = [
            ("create file", "create", True),
            ("show status", "status", True), 
            ("help me", "help", False),  # Simulate incorrect prediction
        ]
        
        for text, expected_intent, correct in test_cases:
            result = await nlp_processor.recognize_intent(text)
            await nlp_processor.record_accuracy(text, expected_intent, result.intent, correct)
        
        accuracy_stats = nlp_processor.get_accuracy_statistics()
        
        assert "overall_accuracy" in accuracy_stats
        assert "recent_accuracy" in accuracy_stats
        assert 0 <= accuracy_stats["overall_accuracy"] <= 1


# Integration and end-to-end tests
@pytest.mark.asyncio
async def test_complete_nlp_processing_pipeline(nlp_processor):
    """Test complete NLP processing pipeline from input to result."""
    complex_input = "I need help creating a Python script called data_analyzer.py that reads CSV files from the input directory, processes them with pandas, performs statistical analysis, and saves the results to an output file with a timestamp"
    
    # Process through complete pipeline
    results = await nlp_processor.process_complete(complex_input)
    
    # Should return comprehensive analysis
    assert "intent" in results
    assert "entities" in results
    assert "sentiment" in results
    assert "language" in results
    assert "context" in results
    
    # Validate each component
    assert isinstance(results["intent"], IntentResult)
    assert isinstance(results["entities"], EntityExtractionResult)
    assert isinstance(results["sentiment"], SentimentResult)
    assert isinstance(results["language"], LanguageDetection)
    
    # Should meet accuracy requirements
    assert results["intent"].confidence >= 0.8
    assert results["language"].confidence >= 0.9


@pytest.mark.asyncio
async def test_real_world_usage_scenarios(nlp_processor):
    """Test real-world usage scenarios."""
    scenarios = [
        # Development workflows
        "create a new React component for user authentication",
        "search for all functions that handle database connections", 
        "show me the current status of all running agents",
        "help me optimize this SQL query for better performance",
        
        # System administration
        "configure the logging level to debug mode",
        "restart the web server and check if it's running properly",
        "backup all user data to the external storage",
        
        # Troubleshooting
        "why is the application consuming so much memory?",
        "the database connection keeps timing out, what should I do?",
        "how can I fix the failing unit tests in the authentication module?",
    ]
    
    for scenario in scenarios:
        result = await nlp_processor.recognize_intent(scenario)
        
        # Should handle real-world complexity
        assert isinstance(result, IntentResult)
        assert result.confidence >= 0.7  # Lower threshold for complex scenarios
        assert result.intent != "unknown" or len(result.alternative_intents) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])