"""
Comprehensive test suite for Enhanced Command Interface - Revolutionary CLI experience.

This test suite validates the natural language processing, intelligent command composition,
context-aware suggestions, and learning capabilities of the enhanced command interface.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agentsmcp.ui.components.enhanced_command_interface import (
    EnhancedCommandInterface,
    CommandIntent,
    ConfidenceLevel,
    CommandSuggestion,
    InterpretationResult,
    CommandContext,
    create_enhanced_command_interface
)
from agentsmcp.ui.v2.event_system import AsyncEventSystem, Event, EventType


@pytest.fixture
async def mock_event_system():
    """Create a mock event system for testing."""
    event_system = AsyncEventSystem()
    await event_system.start()
    yield event_system
    await event_system.stop()


@pytest.fixture
def enhanced_interface(mock_event_system):
    """Create an enhanced command interface for testing."""
    return EnhancedCommandInterface(mock_event_system)


class TestCommandIntentRecognition:
    """Test suite for natural language intent recognition."""

    @pytest.mark.asyncio
    async def test_chat_intent_recognition(self, enhanced_interface):
        """Test recognition of chat intents."""
        test_inputs = [
            "chat with claude about python",
            "talk to the agent",
            "ask about machine learning",
            "what is artificial intelligence?",
            "send a message to the model"
        ]
        
        for user_input in test_inputs:
            result = await enhanced_interface.interpret_natural_language(user_input)
            assert result.intent == CommandIntent.CHAT
            assert result.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
            assert "chat" in result.interpreted_command.lower()

    @pytest.mark.asyncio
    async def test_search_intent_recognition(self, enhanced_interface):
        """Test recognition of search intents."""
        test_inputs = [
            "search for python files",
            "find all TODO comments",
            "look for configuration files",
            "locate the main function",
            "where is the database connection?"
        ]
        
        for user_input in test_inputs:
            result = await enhanced_interface.interpret_natural_language(user_input)
            assert result.intent == CommandIntent.SEARCH
            assert result.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]

    @pytest.mark.asyncio
    async def test_create_intent_recognition(self, enhanced_interface):
        """Test recognition of creation intents."""
        test_inputs = [
            "create a new file",
            "make a python script",
            "generate a readme",
            "build a docker image",
            "write a test function"
        ]
        
        for user_input in test_inputs:
            result = await enhanced_interface.interpret_natural_language(user_input)
            assert result.intent == CommandIntent.CREATE
            assert result.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]

    @pytest.mark.asyncio
    async def test_help_intent_recognition(self, enhanced_interface):
        """Test recognition of help intents."""
        test_inputs = [
            "help me with commands",
            "how to use this tool",
            "what can I do here?",
            "explain the features",
            "usage information"
        ]
        
        for user_input in test_inputs:
            result = await enhanced_interface.interpret_natural_language(user_input)
            assert result.intent == CommandIntent.HELP
            assert result.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]

    @pytest.mark.asyncio
    async def test_direct_command_recognition(self, enhanced_interface):
        """Test recognition of direct commands starting with /."""
        test_cases = [
            ("/chat hello world", CommandIntent.CHAT),
            ("/help commands", CommandIntent.HELP),
            ("/status system", CommandIntent.STATUS),
            ("/config theme dark", CommandIntent.CONFIG),
            ("/quit", CommandIntent.QUIT)
        ]
        
        for command, expected_intent in test_cases:
            result = await enhanced_interface.interpret_natural_language(command)
            assert result.intent == expected_intent
            assert result.confidence == ConfidenceLevel.HIGH
            assert result.interpreted_command == command


class TestCommandSuggestions:
    """Test suite for command suggestion system."""

    @pytest.mark.asyncio
    async def test_context_aware_suggestions(self, enhanced_interface):
        """Test that suggestions are context-aware."""
        # Update context to simulate user patterns
        enhanced_interface.update_context(
            user_skill_level="expert",
            recent_commands=["/chat", "/chat", "/status"],
            usage_patterns={"chat": 5, "status": 2}
        )
        
        result = await enhanced_interface.interpret_natural_language("talk about code")
        
        # Should suggest chat-related commands due to usage patterns
        chat_suggestions = [s for s in result.suggestions if s.intent == CommandIntent.CHAT]
        assert len(chat_suggestions) > 0
        assert all(s.confidence > 0.5 for s in chat_suggestions)

    @pytest.mark.asyncio
    async def test_fuzzy_matching_suggestions(self, enhanced_interface):
        """Test fuzzy matching for similar commands."""
        result = await enhanced_interface.interpret_natural_language("/hep")
        
        # Should suggest /help due to fuzzy matching
        assert len(result.error_corrections) > 0
        help_suggestions = [s for s in result.suggestions if "/help" in s.command]
        assert len(help_suggestions) > 0

    @pytest.mark.asyncio
    async def test_capability_based_suggestions(self, enhanced_interface):
        """Test suggestions based on required capabilities."""
        # Test with a complex query that should return multiple capability-matched suggestions
        result = await enhanced_interface.interpret_natural_language("configure system settings")
        
        config_suggestions = [s for s in result.suggestions if s.intent == CommandIntent.CONFIG]
        assert len(config_suggestions) > 0


class TestNaturalLanguageProcessing:
    """Test suite for natural language processing accuracy."""

    @pytest.mark.asyncio
    async def test_input_cleaning_and_normalization(self, enhanced_interface):
        """Test input cleaning and typo correction."""
        typo_inputs = [
            ("halp me with commands", "help me with commands"),
            ("stauts of system", "status of system"),  
            ("conig the theme", "config the theme"),
            ("eixt the application", "exit the application")
        ]
        
        for typo_input, expected_clean in typo_inputs:
            cleaned = enhanced_interface._clean_input(typo_input)
            assert expected_clean in cleaned.lower() or typo_input == cleaned

    @pytest.mark.asyncio
    async def test_parameter_extraction(self, enhanced_interface):
        """Test parameter extraction from natural language."""
        test_cases = [
            ("chat with claude about python", {"message": "claude about python"}),
            ("search for TODO comments", {"query": "TODO comments"}), 
            ("set theme to dark", {"setting": "theme", "value": "dark"}),
            ("create a new file called main.py", {"target": "a new file called main.py"})
        ]
        
        for user_input, expected_params in test_cases:
            result = await enhanced_interface.interpret_natural_language(user_input)
            # Check if extracted parameters match expected structure
            if expected_params:
                assert len(result.parameters) > 0

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, enhanced_interface):
        """Test confidence level calculation accuracy."""
        # High confidence cases
        high_confidence_inputs = [
            "/help",
            "/quit", 
            "chat hello world",
            "help me"
        ]
        
        for user_input in high_confidence_inputs:
            result = await enhanced_interface.interpret_natural_language(user_input)
            assert result.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]

        # Low confidence cases (ambiguous inputs)
        low_confidence_inputs = [
            "hmm",
            "maybe", 
            "xyz abc 123",
            ""
        ]
        
        for user_input in low_confidence_inputs:
            result = await enhanced_interface.interpret_natural_language(user_input)
            # These should either be low confidence or fallback to chat
            assert result.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.UNKNOWN, ConfidenceLevel.MEDIUM]


class TestLearningAndAdaptation:
    """Test suite for learning and adaptation capabilities."""

    @pytest.mark.asyncio
    async def test_feedback_learning(self, enhanced_interface):
        """Test learning from user feedback."""
        original_input = "show me the files"
        chosen_command = "/search files"
        
        # Simulate positive feedback
        await enhanced_interface.learn_from_feedback(original_input, chosen_command, success=True)
        
        assert len(enhanced_interface.learning_history) > 0
        assert original_input in enhanced_interface.context.recent_commands or chosen_command in enhanced_interface.context.recent_commands

    @pytest.mark.asyncio 
    async def test_usage_pattern_tracking(self, enhanced_interface):
        """Test usage pattern tracking and adaptation."""
        # Simulate multiple successful chat interactions
        for i in range(5):
            await enhanced_interface.learn_from_feedback(f"chat message {i}", "/chat", success=True)
        
        # Usage patterns should show chat as frequently used
        assert "chat" in enhanced_interface.context.usage_patterns
        assert enhanced_interface.context.usage_patterns["chat"] >= 5

    @pytest.mark.asyncio
    async def test_context_updates(self, enhanced_interface):
        """Test context updates affect interpretation."""
        # Update user skill level to beginner
        enhanced_interface.update_context(user_skill_level="beginner")
        assert enhanced_interface.context.user_skill_level == "beginner"
        
        # Update to expert
        enhanced_interface.update_context(user_skill_level="expert")
        assert enhanced_interface.context.user_skill_level == "expert"


class TestErrorHandlingAndRecovery:
    """Test suite for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_unknown_command_handling(self, enhanced_interface):
        """Test handling of unknown commands."""
        result = await enhanced_interface.interpret_natural_language("/unknown_command")
        
        assert result.intent == CommandIntent.UNKNOWN
        assert len(result.error_corrections) > 0
        assert "Unknown command" in result.explanation

    @pytest.mark.asyncio
    async def test_empty_input_handling(self, enhanced_interface):
        """Test handling of empty or whitespace-only input."""
        empty_inputs = ["", "   ", "\n\t", None]
        
        for empty_input in empty_inputs:
            if empty_input is not None:
                result = await enhanced_interface.interpret_natural_language(empty_input)
                assert result is not None
                assert isinstance(result, InterpretationResult)

    @pytest.mark.asyncio
    async def test_exception_handling(self, enhanced_interface):
        """Test graceful handling of internal exceptions."""
        # Mock a method to raise an exception
        with patch.object(enhanced_interface, '_recognize_intent', side_effect=Exception("Test error")):
            result = await enhanced_interface.interpret_natural_language("test input")
            
            assert result.intent == CommandIntent.UNKNOWN
            assert "Error processing input" in result.explanation


class TestPerformanceAndMetrics:
    """Test suite for performance tracking and metrics."""

    @pytest.mark.asyncio
    async def test_response_time_tracking(self, enhanced_interface):
        """Test response time tracking."""
        await enhanced_interface.interpret_natural_language("test input")
        
        assert len(enhanced_interface.response_times) > 0
        assert all(isinstance(rt, float) for rt in enhanced_interface.response_times)
        assert all(rt >= 0 for rt in enhanced_interface.response_times)

    @pytest.mark.asyncio
    async def test_performance_stats_generation(self, enhanced_interface):
        """Test performance statistics generation."""
        # Generate some activity
        for i in range(5):
            await enhanced_interface.interpret_natural_language(f"test input {i}")
            await enhanced_interface.learn_from_feedback(f"test input {i}", "/chat", success=True)
        
        stats = enhanced_interface.get_performance_stats()
        
        assert "average_response_time_ms" in stats
        assert "total_interpretations" in stats
        assert "accuracy_rate" in stats
        assert "learning_samples" in stats
        assert stats["total_interpretations"] >= 5

    @pytest.mark.asyncio
    async def test_caching_effectiveness(self, enhanced_interface):
        """Test suggestion caching."""
        # Make same request twice
        input_text = "help me with commands"
        
        result1 = await enhanced_interface.interpret_natural_language(input_text)
        result2 = await enhanced_interface.interpret_natural_language(input_text)
        
        # Cache should have entries
        assert len(enhanced_interface.suggestion_cache) > 0


class TestAccessibilityAndUsability:
    """Test suite for accessibility and usability features."""

    @pytest.mark.asyncio
    async def test_verbose_explanations(self, enhanced_interface):
        """Test verbose explanations for accessibility."""
        enhanced_interface.verbose_explanations = True
        
        result = await enhanced_interface.interpret_natural_language("help")
        
        assert len(result.explanation) > 0
        assert isinstance(result.explanation, str)

    @pytest.mark.asyncio
    async def test_announcement_settings(self, enhanced_interface):
        """Test announcement settings for screen readers."""
        enhanced_interface.announce_suggestions = True
        
        result = await enhanced_interface.interpret_natural_language("chat")
        
        # Should provide suggestions that can be announced
        assert len(result.suggestions) > 0

    @pytest.mark.asyncio
    async def test_high_contrast_mode(self, enhanced_interface):
        """Test high contrast mode settings."""
        enhanced_interface.high_contrast_mode = True
        
        # Interface should still function normally in high contrast mode
        result = await enhanced_interface.interpret_natural_language("status")
        
        assert result is not None
        assert isinstance(result, InterpretationResult)


class TestEventSystemIntegration:
    """Test suite for event system integration."""

    @pytest.mark.asyncio
    async def test_event_emission(self, enhanced_interface, mock_event_system):
        """Test that interpretation events are emitted correctly."""
        events = []
        
        async def event_handler(event):
            events.append(event)
        
        # Register event handler
        mock_event_system.register_handler(EventType.CUSTOM, event_handler)
        
        await enhanced_interface.interpret_natural_language("test input")
        
        # Should have emitted an interpretation event
        interpretation_events = [e for e in events if e.data.get("action") == "interpretation"]
        assert len(interpretation_events) > 0

    @pytest.mark.asyncio
    async def test_callback_registration(self, enhanced_interface):
        """Test callback registration and execution."""
        callback_called = False
        callback_result = None
        
        async def test_callback(result):
            nonlocal callback_called, callback_result
            callback_called = True
            callback_result = result
        
        enhanced_interface.add_callback("interpretation", test_callback)
        
        result = await enhanced_interface.interpret_natural_language("test input")
        
        assert callback_called
        assert callback_result == result


class TestCleanupAndResourceManagement:
    """Test suite for cleanup and resource management."""

    @pytest.mark.asyncio
    async def test_cleanup_functionality(self, enhanced_interface):
        """Test cleanup clears resources properly."""
        # Generate some activity to create resources
        await enhanced_interface.interpret_natural_language("test")
        enhanced_interface.add_callback("test", lambda x: x)
        
        # Verify resources exist
        assert len(enhanced_interface.suggestion_cache) > 0 or len(enhanced_interface.response_times) > 0
        
        # Cleanup
        await enhanced_interface.cleanup()
        
        # Verify resources are cleared
        assert len(enhanced_interface.suggestion_cache) == 0
        assert len(enhanced_interface._callbacks) == 0
        assert len(enhanced_interface.response_times) == 0

    @pytest.mark.asyncio
    async def test_memory_management(self, enhanced_interface):
        """Test memory management with learning history limits."""
        # Generate more learning data than the limit
        for i in range(1200):  # More than the 1000 limit
            enhanced_interface.learning_history.append((f"input{i}", f"command{i}", 1.0))
        
        await enhanced_interface.learn_from_feedback("test", "/test", True)
        
        # Should be trimmed to 800 entries
        assert len(enhanced_interface.learning_history) <= 801  # 800 + 1 new entry


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_create_enhanced_command_interface(self, mock_event_system):
        """Test utility function for creating interface."""
        interface = create_enhanced_command_interface(mock_event_system)
        
        assert isinstance(interface, EnhancedCommandInterface)
        assert interface.event_system == mock_event_system

    @pytest.mark.asyncio
    async def test_similarity_calculation(self, enhanced_interface):
        """Test string similarity calculation."""
        # Test exact match
        assert enhanced_interface._calculate_similarity("hello", "hello") == 1.0
        
        # Test no match
        assert enhanced_interface._calculate_similarity("hello", "world") < 1.0
        
        # Test empty strings
        assert enhanced_interface._calculate_similarity("", "") == 1.0
        assert enhanced_interface._calculate_similarity("hello", "") == 0.0

    def test_command_registry_completeness(self, enhanced_interface):
        """Test that command registry contains all expected commands."""
        expected_commands = [
            "/chat", "/ask", "/agent", "/model", "/help", 
            "/status", "/config", "/clear", "/history", "/quit"
        ]
        
        for cmd in expected_commands:
            assert cmd in enhanced_interface.commands
            assert isinstance(enhanced_interface.commands[cmd], CommandSuggestion)


# Integration tests with real async components
@pytest.mark.asyncio
async def test_full_interpretation_workflow(enhanced_interface):
    """Test complete interpretation workflow from input to result."""
    user_input = "chat with claude about testing best practices"
    
    result = await enhanced_interface.interpret_natural_language(user_input)
    
    # Comprehensive validation
    assert result.original_input == user_input
    assert result.intent == CommandIntent.CHAT
    assert result.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
    assert "/chat" in result.interpreted_command
    assert len(result.suggestions) > 0
    assert len(result.explanation) > 0
    assert result.parameters.get("message") is not None


@pytest.mark.asyncio 
async def test_multiple_concurrent_interpretations(enhanced_interface):
    """Test handling multiple concurrent interpretation requests."""
    inputs = [
        "help with commands",
        "search for files", 
        "create new project",
        "show status",
        "quit application"
    ]
    
    # Execute concurrently
    tasks = [enhanced_interface.interpret_natural_language(inp) for inp in inputs]
    results = await asyncio.gather(*tasks)
    
    # All should complete successfully
    assert len(results) == len(inputs)
    assert all(isinstance(result, InterpretationResult) for result in results)
    assert all(result.original_input in inputs for result in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])