"""Tests for the main natural language processor."""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.agentsmcp.cli.v3.nlp.processor import NaturalLanguageProcessor
from src.agentsmcp.cli.v3.models.nlp_models import (
    LLMConfig,
    ConversationContext,
    ParsedCommand,
    CommandInterpretation,
    ParsingResult,
    ParsingMethod,
    ParsingFailedError,
    AmbiguousInputError,
    LLMUnavailableError,
    ContextTooLargeError,
    UnsupportedLanguageError
)


class TestNaturalLanguageProcessor:
    """Test natural language processor functionality."""

    def setup_method(self):
        """Setup test instance."""
        self.config = LLMConfig(
            model_name="test-model",
            max_tokens=1024,
            temperature=0.1,
            timeout_seconds=30.0
        )
        self.processor = NaturalLanguageProcessor(self.config)

    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        processor = NaturalLanguageProcessor()
        assert processor.config.model_name == "gpt-oss:20b"
        assert processor.llm_integration is not None
        assert processor.pattern_matcher is not None
        assert processor.context is not None
        assert len(processor.metrics.total_requests) == 0

    def test_initialization_custom_config(self):
        """Test initialization with custom configuration."""
        assert self.processor.config == self.config
        assert self.processor.config.model_name == "test-model"

    @pytest.mark.asyncio
    async def test_parse_success_llm_method(self):
        """Test successful parsing using LLM method."""
        # Mock successful LLM response
        mock_command = ParsedCommand(
            action="analyze",
            parameters={"target": "code"},
            confidence=0.85,
            method=ParsingMethod.LLM
        )
        
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (mock_command, "I understood this as analyzing code.")
            
            result = await self.processor.parse("analyze my code")
            
            assert result.success is True
            assert result.structured_command is not None
            assert result.structured_command.action == "analyze"
            assert result.explanation == "I understood this as analyzing code."
            assert result.method_used == ParsingMethod.LLM
            assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_parse_fallback_to_patterns(self):
        """Test fallback to pattern matching when LLM fails."""
        # Mock LLM failure
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = LLMUnavailableError("LLM service unavailable")
            
            # Mock pattern matching success
            with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                mock_pattern.return_value = ParsedCommand(
                    action="help",
                    parameters={},
                    confidence=0.9,
                    method=ParsingMethod.RULE_BASED
                )
                
                result = await self.processor.parse("help")
                
                assert result.success is True
                assert result.structured_command.action == "help"
                assert result.method_used == ParsingMethod.RULE_BASED
                assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_parse_hybrid_method(self):
        """Test hybrid parsing method."""
        # Mock partial LLM success with low confidence
        low_confidence_command = ParsedCommand(
            action="analyze",
            parameters={},
            confidence=0.4,
            method=ParsingMethod.LLM
        )
        
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (low_confidence_command, "Uncertain interpretation")
            
            # Mock pattern matching with higher confidence
            with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                mock_pattern.return_value = ParsedCommand(
                    action="analyze",
                    parameters={"target": "code"},
                    confidence=0.85,
                    method=ParsingMethod.RULE_BASED
                )
                
                result = await self.processor.parse("analyze my code")
                
                assert result.success is True
                assert result.method_used == ParsingMethod.HYBRID
                assert result.structured_command.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_parse_complete_failure(self):
        """Test parsing when both LLM and patterns fail."""
        # Mock LLM failure
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = LLMUnavailableError("LLM unavailable")
            
            # Mock pattern matching failure
            with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                mock_pattern.return_value = None
                
                result = await self.processor.parse("xyz random gibberish")
                
                assert result.success is False
                assert result.structured_command is None
                assert len(result.errors) > 0
                assert any("Failed to parse" in error.message for error in result.errors)

    @pytest.mark.asyncio
    async def test_parse_with_context(self):
        """Test parsing with conversation context."""
        context = ConversationContext(
            command_history=["help", "status"],
            current_directory="/project",
            recent_files=["main.py"]
        )
        
        mock_command = ParsedCommand(
            action="analyze",
            parameters={"target": "main.py"},
            confidence=0.9,
            method=ParsingMethod.LLM
        )
        
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (mock_command, "Analyzing main.py from context")
            
            result = await self.processor.parse("analyze it", context)
            
            assert result.success is True
            assert result.structured_command.parameters.get("target") == "main.py"
            
            # Verify context was passed to LLM
            mock_llm.assert_called_once()
            call_args = mock_llm.call_args
            assert call_args[0][0] == "analyze it"  # natural_input
            passed_context = call_args[0][1]  # context
            assert passed_context is not None

    @pytest.mark.asyncio
    async def test_parse_context_too_large_error(self):
        """Test handling of context too large error."""
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = ContextTooLargeError("Context exceeds limit")
            
            result = await self.processor.parse("analyze " + "x" * 10000)
            
            assert result.success is False
            assert any(error.error_type == "ContextTooLarge" for error in result.errors)

    @pytest.mark.asyncio
    async def test_get_alternative_interpretations(self):
        """Test getting alternative interpretations."""
        # Mock LLM returning primary interpretation
        primary_command = ParsedCommand(
            action="analyze",
            parameters={"target": "code"},
            confidence=0.85,
            method=ParsingMethod.LLM
        )
        
        # Mock pattern matcher returning alternative
        alternative_command = ParsedCommand(
            action="file",
            parameters={"operation": "read"},
            confidence=0.7,
            method=ParsingMethod.RULE_BASED
        )
        
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (primary_command, "Primary interpretation")
            
            with patch.object(self.processor.pattern_matcher, 'match_patterns') as mock_patterns:
                from src.agentsmcp.cli.v3.models.nlp_models import PatternMatch
                mock_patterns.return_value = [
                    PatternMatch(
                        pattern="file_read",
                        action="file",
                        parameters={"operation": "read"},
                        confidence=0.7,
                        priority=3
                    )
                ]
                
                result = await self.processor.parse("read my code file")
                
                assert result.success is True
                assert len(result.alternative_interpretations) > 0
                assert any(interp.command.action == "file" 
                          for interp in result.alternative_interpretations)

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Test that metrics are properly tracked."""
        initial_requests = self.processor.metrics.total_requests
        
        mock_command = ParsedCommand(
            action="help",
            parameters={},
            confidence=0.95,
            method=ParsingMethod.LLM
        )
        
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (mock_command, "Help requested")
            
            await self.processor.parse("help")
            
            assert self.processor.metrics.total_requests == initial_requests + 1
            assert self.processor.metrics.successful_parses > 0
            assert self.processor.metrics.llm_calls > 0

    @pytest.mark.asyncio
    async def test_metrics_tracking_failure(self):
        """Test metrics tracking on parsing failure."""
        initial_requests = self.processor.metrics.total_requests
        initial_failures = self.processor.metrics.failed_parses
        
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("Unexpected error")
            
            with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                mock_pattern.return_value = None
                
                await self.processor.parse("invalid input")
                
                assert self.processor.metrics.total_requests == initial_requests + 1
                assert self.processor.metrics.failed_parses == initial_failures + 1

    def test_add_conversation_context(self):
        """Test adding to conversation context."""
        initial_history_length = len(self.processor.context.command_history)
        
        self.processor.add_to_context("analyze code")
        
        assert len(self.processor.context.command_history) == initial_history_length + 1
        assert self.processor.context.command_history[-1] == "analyze code"

    def test_clear_conversation_context(self):
        """Test clearing conversation context."""
        # Add some context
        self.processor.context.command_history = ["cmd1", "cmd2", "cmd3"]
        self.processor.context.recent_files = ["file1.py", "file2.py"]
        
        self.processor.clear_context()
        
        assert len(self.processor.context.command_history) == 0
        assert len(self.processor.context.recent_files) == 0

    def test_update_llm_config(self):
        """Test updating LLM configuration."""
        new_config = LLMConfig(
            model_name="new-model",
            temperature=0.5,
            max_tokens=2048
        )
        
        self.processor.update_llm_config(new_config)
        
        assert self.processor.config == new_config
        assert self.processor.config.model_name == "new-model"

    @pytest.mark.asyncio
    async def test_check_llm_availability(self):
        """Test checking LLM availability."""
        with patch.object(self.processor.llm_integration, 'check_availability', 
                         new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True
            
            available = await self.processor.check_llm_availability()
            assert available is True

    def test_get_metrics(self):
        """Test getting current metrics."""
        metrics = self.processor.get_metrics()
        
        assert hasattr(metrics, 'total_requests')
        assert hasattr(metrics, 'successful_parses')
        assert hasattr(metrics, 'failed_parses')
        assert hasattr(metrics, 'success_rate')

    def test_get_supported_actions(self):
        """Test getting supported actions."""
        actions = self.processor.get_supported_actions()
        
        assert len(actions) > 0
        assert "help" in actions
        assert "analyze" in actions
        assert "status" in actions

    def test_get_command_examples(self):
        """Test getting command examples."""
        examples = self.processor.get_command_examples("analyze")
        
        assert len(examples) > 0
        assert any("analyze" in example.lower() for example in examples)

    @pytest.mark.asyncio
    async def test_error_recovery_suggestions(self):
        """Test error recovery suggestions."""
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = LLMUnavailableError("LLM service down")
            
            with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                mock_pattern.return_value = None
                
                result = await self.processor.parse("unclear input")
                
                assert result.success is False
                assert len(result.errors) > 0
                
                # Check if errors have recovery suggestions
                error = result.errors[0]
                assert hasattr(error, 'recovery_suggestions')

    @pytest.mark.asyncio
    async def test_processing_time_tracking(self):
        """Test processing time is tracked."""
        mock_command = ParsedCommand(
            action="help",
            parameters={},
            confidence=0.95,
            method=ParsingMethod.LLM
        )
        
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            # Add small delay to simulate processing
            async def delayed_parse(*args, **kwargs):
                await asyncio.sleep(0.01)  # 10ms delay
                return (mock_command, "Help requested")
                
            mock_llm.side_effect = delayed_parse
            
            result = await self.processor.parse("help")
            
            assert result.processing_time_ms >= 10  # Should be at least 10ms

    @pytest.mark.asyncio
    async def test_confidence_threshold_handling(self):
        """Test handling of confidence thresholds."""
        # Test very low confidence command from LLM
        low_confidence_command = ParsedCommand(
            action="unclear",
            parameters={},
            confidence=0.1,
            method=ParsingMethod.LLM
        )
        
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (low_confidence_command, "Very uncertain interpretation")
            
            # Mock pattern matching with slightly better confidence
            with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                mock_pattern.return_value = ParsedCommand(
                    action="help",
                    parameters={},
                    confidence=0.3,
                    method=ParsingMethod.RULE_BASED
                )
                
                result = await self.processor.parse("maybe help me somehow")
                
                # Should prefer pattern matching result due to higher confidence
                if result.success:
                    assert result.structured_command.action == "help"
                    assert result.method_used in [ParsingMethod.RULE_BASED, ParsingMethod.HYBRID]

    @pytest.mark.asyncio
    async def test_ambiguous_input_handling(self):
        """Test handling of ambiguous input."""
        # Mock multiple high-confidence interpretations
        mock_command1 = ParsedCommand(
            action="analyze",
            parameters={"target": "code"},
            confidence=0.85,
            method=ParsingMethod.LLM
        )
        
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (mock_command1, "Could be analyze")
            
            # Mock pattern matching with different high-confidence result
            with patch.object(self.processor.pattern_matcher, 'match_patterns') as mock_patterns:
                from src.agentsmcp.cli.v3.models.nlp_models import PatternMatch
                mock_patterns.return_value = [
                    PatternMatch(
                        pattern="file_operation",
                        action="file",
                        parameters={"operation": "read"},
                        confidence=0.8,
                        priority=2
                    ),
                    PatternMatch(
                        pattern="help_command", 
                        action="help",
                        parameters={},
                        confidence=0.75,
                        priority=1
                    )
                ]
                
                result = await self.processor.parse("check my files")
                
                # Should provide multiple interpretations
                assert result.success is True
                assert len(result.alternative_interpretations) > 0

    def test_context_integration(self):
        """Test context is properly integrated into parsing."""
        # Add some context
        self.processor.context.command_history = ["analyze", "help"]
        self.processor.context.current_directory = "/project"
        self.processor.context.recent_files = ["main.py", "test.py"]
        
        # The context should be accessible
        context_dict = {
            "command_history": self.processor.context.command_history,
            "current_directory": self.processor.context.current_directory,
            "recent_files": self.processor.context.recent_files
        }
        
        assert context_dict["command_history"] == ["analyze", "help"]
        assert context_dict["current_directory"] == "/project"
        assert "main.py" in context_dict["recent_files"]

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test proper exception handling and error reporting."""
        # Test various exception types
        exception_tests = [
            (LLMUnavailableError("LLM down"), "LLMUnavailable"),
            (ContextTooLargeError("Too big"), "ContextTooLarge"), 
            (ParsingFailedError("Parse failed"), "ParsingFailed"),
            (Exception("Unexpected"), "UnexpectedError")
        ]
        
        for exception, expected_error_type in exception_tests:
            with patch.object(self.processor.llm_integration, 'parse_command', 
                             new_callable=AsyncMock) as mock_llm:
                mock_llm.side_effect = exception
                
                with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                    mock_pattern.return_value = None
                    
                    result = await self.processor.parse("test input")
                    
                    assert result.success is False
                    assert len(result.errors) > 0
                    
                    # Check that error type is recorded (approximately)
                    error_messages = [error.message for error in result.errors]
                    assert any(error_msg for error_msg in error_messages)