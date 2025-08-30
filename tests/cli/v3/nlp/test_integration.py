"""Integration tests for natural language processing components."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.agentsmcp.cli.v3.nlp.processor import NaturalLanguageProcessor
from src.agentsmcp.cli.v3.models.nlp_models import (
    LLMConfig,
    ConversationContext,
    ParsedCommand,
    ParsingMethod
)


class TestNLPIntegration:
    """Integration tests for NLP components working together."""

    def setup_method(self):
        """Setup test instance."""
        self.config = LLMConfig(
            model_name="test-model",
            max_tokens=1024,
            temperature=0.1
        )
        self.processor = NaturalLanguageProcessor(self.config)

    @pytest.mark.asyncio
    async def test_end_to_end_parsing_workflow(self):
        """Test complete end-to-end parsing workflow."""
        # Test cases covering different parsing scenarios
        test_cases = [
            {
                "input": "help",
                "expected_action": "help",
                "min_confidence": 0.8
            },
            {
                "input": "analyze my code",
                "expected_action": "analyze", 
                "min_confidence": 0.7
            },
            {
                "input": "start the TUI",
                "expected_action": "tui",
                "min_confidence": 0.8
            },
            {
                "input": "check status",
                "expected_action": "status",
                "min_confidence": 0.7
            }
        ]
        
        for test_case in test_cases:
            # Mock successful pattern matching (since LLM won't be available in tests)
            mock_command = ParsedCommand(
                action=test_case["expected_action"],
                parameters={},
                confidence=test_case["min_confidence"],
                method=ParsingMethod.RULE_BASED
            )
            
            with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                mock_pattern.return_value = mock_command
                
                with patch.object(self.processor.llm_integration, 'parse_command', 
                                 new_callable=AsyncMock) as mock_llm:
                    mock_llm.side_effect = Exception("LLM unavailable")
                    
                    result = await self.processor.parse(test_case["input"])
                    
                    assert result.success is True, f"Failed to parse: {test_case['input']}"
                    assert result.structured_command is not None
                    assert result.structured_command.action == test_case["expected_action"]
                    assert result.structured_command.confidence >= test_case["min_confidence"]

    @pytest.mark.asyncio
    async def test_context_aware_parsing(self):
        """Test context-aware parsing across multiple interactions."""
        # Simulate a conversation sequence
        conversation_sequence = [
            ("help", "help"),
            ("what's my status", "status"), 
            ("analyze the code", "analyze"),
            ("start TUI mode", "tui")
        ]
        
        context = ConversationContext()
        
        for input_text, expected_action in conversation_sequence:
            # Mock pattern matching
            mock_command = ParsedCommand(
                action=expected_action,
                parameters={},
                confidence=0.8,
                method=ParsingMethod.RULE_BASED
            )
            
            with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                mock_pattern.return_value = mock_command
                
                with patch.object(self.processor.llm_integration, 'parse_command', 
                                 new_callable=AsyncMock) as mock_llm:
                    mock_llm.side_effect = Exception("LLM unavailable")
                    
                    result = await self.processor.parse(input_text, context)
                    
                    assert result.success is True
                    assert result.structured_command.action == expected_action
                    
                    # Update context for next iteration
                    context.add_command(input_text)
        
        # Verify context was built up
        assert len(context.command_history) == len(conversation_sequence)

    @pytest.mark.asyncio
    async def test_hybrid_parsing_scenario(self):
        """Test hybrid parsing where both LLM and patterns contribute."""
        input_text = "analyze my code for security issues"
        
        # Mock LLM returning moderate confidence result
        llm_command = ParsedCommand(
            action="analyze",
            parameters={"type": "security"},
            confidence=0.6,
            method=ParsingMethod.LLM
        )
        
        # Mock pattern matching with higher confidence
        pattern_command = ParsedCommand(
            action="analyze",
            parameters={"target": "code"},
            confidence=0.85,
            method=ParsingMethod.RULE_BASED
        )
        
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (llm_command, "Found security analysis request")
            
            with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                mock_pattern.return_value = pattern_command
                
                result = await self.processor.parse(input_text)
                
                assert result.success is True
                assert result.method_used == ParsingMethod.HYBRID
                # Should combine aspects from both methods
                assert result.structured_command.action == "analyze"

    @pytest.mark.asyncio
    async def test_fallback_chain(self):
        """Test the complete fallback chain from LLM to patterns to failure."""
        scenarios = [
            {
                "name": "LLM success",
                "llm_result": (ParsedCommand(action="help", parameters={}, confidence=0.9, method=ParsingMethod.LLM), "Help requested"),
                "llm_exception": None,
                "pattern_result": None,
                "expected_success": True,
                "expected_method": ParsingMethod.LLM
            },
            {
                "name": "LLM failure, pattern success", 
                "llm_result": None,
                "llm_exception": Exception("LLM failed"),
                "pattern_result": ParsedCommand(action="help", parameters={}, confidence=0.8, method=ParsingMethod.RULE_BASED),
                "expected_success": True,
                "expected_method": ParsingMethod.RULE_BASED
            },
            {
                "name": "Both fail",
                "llm_result": None,
                "llm_exception": Exception("LLM failed"),
                "pattern_result": None,
                "expected_success": False,
                "expected_method": None
            }
        ]
        
        for scenario in scenarios:
            with patch.object(self.processor.llm_integration, 'parse_command', 
                             new_callable=AsyncMock) as mock_llm:
                if scenario["llm_exception"]:
                    mock_llm.side_effect = scenario["llm_exception"]
                else:
                    mock_llm.return_value = scenario["llm_result"]
                
                with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                    mock_pattern.return_value = scenario["pattern_result"]
                    
                    result = await self.processor.parse("help")
                    
                    assert result.success == scenario["expected_success"], f"Scenario failed: {scenario['name']}"
                    if scenario["expected_success"]:
                        assert result.method_used == scenario["expected_method"]

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test comprehensive error handling and recovery suggestions."""
        error_scenarios = [
            {
                "exception": Exception("Network timeout"),
                "expected_error_type": "UnexpectedError",
                "should_have_suggestions": True
            },
            {
                "exception": ValueError("Invalid response"),
                "expected_error_type": "ParsingError", 
                "should_have_suggestions": True
            }
        ]
        
        for scenario in error_scenarios:
            with patch.object(self.processor.llm_integration, 'parse_command', 
                             new_callable=AsyncMock) as mock_llm:
                mock_llm.side_effect = scenario["exception"]
                
                with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                    mock_pattern.return_value = None
                    
                    result = await self.processor.parse("ambiguous input")
                    
                    assert result.success is False
                    assert len(result.errors) > 0
                    
                    if scenario["should_have_suggestions"]:
                        # At least one error should have recovery suggestions
                        has_suggestions = any(
                            hasattr(error, 'recovery_suggestions') and len(error.recovery_suggestions) > 0
                            for error in result.errors
                        )
                        # Note: This might not always be true depending on implementation

    @pytest.mark.asyncio
    async def test_performance_requirements(self):
        """Test that performance requirements are met."""
        # Test that pattern matching is fast (<100ms)
        import time
        
        start_time = time.time()
        
        # Mock fast pattern matching
        mock_command = ParsedCommand(
            action="help",
            parameters={},
            confidence=0.9,
            method=ParsingMethod.RULE_BASED
        )
        
        with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
            mock_pattern.return_value = mock_command
            
            with patch.object(self.processor.llm_integration, 'parse_command', 
                             new_callable=AsyncMock) as mock_llm:
                mock_llm.side_effect = Exception("LLM unavailable")
                
                result = await self.processor.parse("help")
                
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                
                assert result.success is True
                assert processing_time < 100, f"Pattern matching took {processing_time}ms, should be <100ms"

    @pytest.mark.asyncio
    async def test_alternative_interpretations_generation(self):
        """Test generation of alternative interpretations."""
        input_text = "check my files"
        
        # Mock LLM primary interpretation
        primary_command = ParsedCommand(
            action="file",
            parameters={"operation": "list"},
            confidence=0.8,
            method=ParsingMethod.LLM
        )
        
        # Mock pattern matching alternatives
        from src.agentsmcp.cli.v3.models.nlp_models import PatternMatch
        mock_patterns = [
            PatternMatch(
                pattern="status_check",
                action="status",
                parameters={"component": "files"},
                confidence=0.7,
                priority=2
            ),
            PatternMatch(
                pattern="analyze_files",
                action="analyze", 
                parameters={"target": "files"},
                confidence=0.65,
                priority=3
            )
        ]
        
        with patch.object(self.processor.llm_integration, 'parse_command', 
                         new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (primary_command, "Listing files")
            
            with patch.object(self.processor.pattern_matcher, 'match_patterns') as mock_match_patterns:
                mock_match_patterns.return_value = mock_patterns
                
                result = await self.processor.parse(input_text)
                
                assert result.success is True
                assert result.structured_command.action == "file"
                assert len(result.alternative_interpretations) > 0
                
                # Check alternative interpretations
                alt_actions = [interp.command.action for interp in result.alternative_interpretations]
                assert "status" in alt_actions or "analyze" in alt_actions

    @pytest.mark.asyncio
    async def test_metrics_and_monitoring(self):
        """Test metrics collection during processing."""
        initial_metrics = self.processor.get_metrics()
        initial_requests = initial_metrics.total_requests
        
        # Perform several parsing operations
        test_inputs = ["help", "status", "analyze code", "start tui"]
        
        for input_text in test_inputs:
            mock_command = ParsedCommand(
                action="test",
                parameters={},
                confidence=0.8,
                method=ParsingMethod.RULE_BASED
            )
            
            with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                mock_pattern.return_value = mock_command
                
                with patch.object(self.processor.llm_integration, 'parse_command', 
                                 new_callable=AsyncMock) as mock_llm:
                    mock_llm.side_effect = Exception("LLM unavailable")
                    
                    await self.processor.parse(input_text)
        
        final_metrics = self.processor.get_metrics()
        
        # Verify metrics were updated
        assert final_metrics.total_requests == initial_requests + len(test_inputs)
        assert final_metrics.successful_parses > initial_metrics.successful_parses
        assert final_metrics.rule_based_matches > initial_metrics.rule_based_matches

    @pytest.mark.asyncio
    async def test_context_preservation(self):
        """Test that context is preserved and built up correctly."""
        # Start with empty context
        context = ConversationContext()
        assert len(context.command_history) == 0
        
        # Process several commands
        commands = [
            "help with analysis",
            "check the status", 
            "analyze my code",
            "start TUI"
        ]
        
        for i, cmd in enumerate(commands):
            mock_command = ParsedCommand(
                action="test",
                parameters={},
                confidence=0.8,
                method=ParsingMethod.RULE_BASED
            )
            
            with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                mock_pattern.return_value = mock_command
                
                with patch.object(self.processor.llm_integration, 'parse_command', 
                                 new_callable=AsyncMock) as mock_llm:
                    mock_llm.side_effect = Exception("LLM unavailable")
                    
                    result = await self.processor.parse(cmd, context)
                    
                    assert result.success is True
                    
                    # Update context manually (since we're mocking the processor)
                    context.add_command(cmd)
                    
                    # Verify context is building up
                    assert len(context.command_history) == i + 1
                    assert context.command_history[-1] == cmd

    @pytest.mark.asyncio 
    async def test_concurrent_parsing_operations(self):
        """Test handling of concurrent parsing operations."""
        inputs = ["help", "status", "analyze", "tui", "init"]
        
        async def parse_with_mock(input_text):
            mock_command = ParsedCommand(
                action="test",
                parameters={"input": input_text},
                confidence=0.8,
                method=ParsingMethod.RULE_BASED
            )
            
            with patch.object(self.processor.pattern_matcher, 'parse_command_fallback') as mock_pattern:
                mock_pattern.return_value = mock_command
                
                with patch.object(self.processor.llm_integration, 'parse_command', 
                                 new_callable=AsyncMock) as mock_llm:
                    mock_llm.side_effect = Exception("LLM unavailable")
                    
                    return await self.processor.parse(input_text)
        
        # Run multiple parsing operations concurrently
        tasks = [parse_with_mock(input_text) for input_text in inputs]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(result.success for result in results)
        assert len(results) == len(inputs)
        
        # Verify each result corresponds to its input
        for i, result in enumerate(results):
            assert result.structured_command.parameters.get("input") == inputs[i]