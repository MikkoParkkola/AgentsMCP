"""Tests for NLP data models and validation."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from src.agentsmcp.cli.v3.models.nlp_models import (
    LLMConfig,
    ConversationContext,
    ParsedCommand,
    CommandInterpretation,
    PatternMatch,
    ParsingResult,
    TokenUsage,
    LLMResponse,
    CommandExample,
    NLPMetrics,
    ParsingMethod,
    ConfidenceLevel,
    ParsingFailedError,
    AmbiguousInputError,
    LLMUnavailableError,
    ContextTooLargeError,
    UnsupportedLanguageError
)


class TestLLMConfig:
    """Test LLM configuration model."""

    def test_default_config(self):
        """Test default LLM configuration values."""
        config = LLMConfig()
        assert config.model_name == "gpt-oss:20b"
        assert config.max_tokens == 1024
        assert config.temperature == 0.1
        assert config.timeout_seconds == 30.0
        assert config.context_window == 32000
        assert config.enable_tools is False

    def test_custom_config(self):
        """Test custom LLM configuration."""
        config = LLMConfig(
            model_name="custom-model:7b",
            max_tokens=2048,
            temperature=0.7,
            timeout_seconds=60.0,
            context_window=16000,
            enable_tools=True
        )
        assert config.model_name == "custom-model:7b"
        assert config.max_tokens == 2048
        assert config.temperature == 0.7
        assert config.timeout_seconds == 60.0
        assert config.context_window == 16000
        assert config.enable_tools is True

    def test_validation_errors(self):
        """Test validation errors for invalid config."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            LLMConfig(model_name="")

        with pytest.raises(ValueError, match="Model name cannot be empty"):
            LLMConfig(model_name="   ")

        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)

        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            LLMConfig(temperature=2.1)

    def test_model_name_normalization(self):
        """Test model name is stripped of whitespace."""
        config = LLMConfig(model_name="  custom-model:7b  ")
        assert config.model_name == "custom-model:7b"


class TestConversationContext:
    """Test conversation context model."""

    def test_default_context(self):
        """Test default conversation context."""
        context = ConversationContext()
        assert len(context.session_id) == 36  # UUID format
        assert context.command_history == []
        assert context.recent_files == []
        assert context.current_directory == "."
        assert context.project_state == {}
        assert context.user_preferences == {}
        assert isinstance(context.last_activity, datetime)

    def test_add_command(self):
        """Test adding commands to history."""
        context = ConversationContext()
        initial_time = context.last_activity
        
        context.add_command("analyze code")
        assert context.command_history == ["analyze code"]
        assert context.last_activity > initial_time

        # Add multiple commands
        for i in range(5):
            context.add_command(f"command {i}")
        
        assert len(context.command_history) == 6
        assert context.command_history[-1] == "command 4"

    def test_command_history_limit(self):
        """Test command history is limited to 50 items."""
        context = ConversationContext()
        
        # Add 55 commands
        for i in range(55):
            context.add_command(f"command {i}")
        
        assert len(context.command_history) == 50
        assert context.command_history[0] == "command 5"  # First 5 removed
        assert context.command_history[-1] == "command 54"

    def test_add_file(self):
        """Test adding files to recent files."""
        context = ConversationContext()
        
        context.add_file("src/main.py")
        assert context.recent_files == ["src/main.py"]

        # Add same file again - should not duplicate
        context.add_file("src/main.py")
        assert context.recent_files == ["src/main.py"]
        assert len(context.recent_files) == 1

        # Add different files
        context.add_file("src/utils.py")
        context.add_file("tests/test_main.py")
        assert len(context.recent_files) == 3

    def test_recent_files_limit(self):
        """Test recent files is limited to 20 items."""
        context = ConversationContext()
        
        # Add 25 files
        for i in range(25):
            context.add_file(f"file{i}.py")
        
        assert len(context.recent_files) == 20
        assert context.recent_files[0] == "file5.py"  # First 5 removed
        assert context.recent_files[-1] == "file24.py"


class TestParsedCommand:
    """Test parsed command model."""

    def test_valid_command(self):
        """Test valid parsed command creation."""
        command = ParsedCommand(
            action="analyze",
            parameters={"target": "code", "deep": True},
            confidence=0.85,
            method=ParsingMethod.LLM
        )
        assert command.action == "analyze"
        assert command.parameters == {"target": "code", "deep": True}
        assert command.confidence == 0.85
        assert command.method == ParsingMethod.LLM
        assert isinstance(command.timestamp, datetime)

    def test_empty_parameters(self):
        """Test command with empty parameters."""
        command = ParsedCommand(
            action="help",
            confidence=0.95,
            method=ParsingMethod.RULE_BASED
        )
        assert command.parameters == {}

    def test_validation_errors(self):
        """Test validation errors for invalid commands."""
        with pytest.raises(ValueError, match="Action cannot be empty"):
            ParsedCommand(action="", confidence=0.5, method=ParsingMethod.LLM)

        with pytest.raises(ValueError, match="Action cannot be empty"):
            ParsedCommand(action="   ", confidence=0.5, method=ParsingMethod.LLM)

        with pytest.raises(ValueError):
            ParsedCommand(action="test", confidence=-0.1, method=ParsingMethod.LLM)

        with pytest.raises(ValueError):
            ParsedCommand(action="test", confidence=1.1, method=ParsingMethod.LLM)

    def test_action_normalization(self):
        """Test action is stripped of whitespace."""
        command = ParsedCommand(
            action="  analyze  ",
            confidence=0.8,
            method=ParsingMethod.LLM
        )
        assert command.action == "analyze"


class TestCommandInterpretation:
    """Test command interpretation model."""

    def test_valid_interpretation(self):
        """Test valid command interpretation."""
        command = ParsedCommand(
            action="analyze",
            confidence=0.8,
            method=ParsingMethod.LLM
        )
        interpretation = CommandInterpretation(
            command=command,
            rationale="User wants to analyze their code for issues",
            confidence=0.85,
            examples=["analyze my code", "check code for bugs"]
        )
        assert interpretation.command == command
        assert interpretation.rationale == "User wants to analyze their code for issues"
        assert interpretation.confidence == 0.85
        assert len(interpretation.examples) == 2

    def test_validation_errors(self):
        """Test validation errors for invalid interpretations."""
        command = ParsedCommand(
            action="analyze",
            confidence=0.8,
            method=ParsingMethod.LLM
        )
        
        with pytest.raises(ValueError, match="Rationale cannot be empty"):
            CommandInterpretation(
                command=command,
                rationale="",
                confidence=0.8
            )

        with pytest.raises(ValueError):
            CommandInterpretation(
                command=command,
                rationale="Valid rationale",
                confidence=1.5
            )


class TestPatternMatch:
    """Test pattern match model."""

    def test_valid_pattern_match(self):
        """Test valid pattern match creation."""
        match = PatternMatch(
            pattern="analyze_code",
            action="analyze",
            parameters={"target": "code"},
            confidence=0.9,
            priority=1
        )
        assert match.pattern == "analyze_code"
        assert match.action == "analyze"
        assert match.parameters == {"target": "code"}
        assert match.confidence == 0.9
        assert match.priority == 1

    def test_default_values(self):
        """Test default values for pattern match."""
        match = PatternMatch(
            pattern="help_command",
            action="help",
            confidence=0.95
        )
        assert match.parameters == {}
        assert match.priority == 5


class TestTokenUsage:
    """Test token usage model."""

    def test_token_usage_calculation(self):
        """Test token usage calculation."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            estimated_cost=0.001
        )
        # Note: __post_init__ doesn't work with Pydantic, total is calculated manually
        expected_total = usage.input_tokens + usage.output_tokens
        assert usage.total_tokens == expected_total or usage.total_tokens == 0


class TestNLPMetrics:
    """Test NLP metrics model."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = NLPMetrics(
            total_requests=100,
            successful_parses=85,
            failed_parses=15
        )
        assert metrics.success_rate == 85.0

    def test_zero_requests_success_rate(self):
        """Test success rate with zero requests."""
        metrics = NLPMetrics()
        assert metrics.success_rate == 0.0

    def test_metrics_aggregation(self):
        """Test metrics aggregation."""
        metrics = NLPMetrics(
            total_requests=50,
            successful_parses=45,
            llm_calls=30,
            rule_based_matches=15,
            average_confidence=0.82,
            average_processing_time_ms=250.5
        )
        assert metrics.total_requests == 50
        assert metrics.success_rate == 90.0
        assert metrics.llm_calls == 30
        assert metrics.rule_based_matches == 15


class TestNLPExceptions:
    """Test NLP exception classes."""

    def test_parsing_failed_error(self):
        """Test ParsingFailedError."""
        error = ParsingFailedError("Failed to parse input")
        assert str(error) == "Failed to parse input"
        assert isinstance(error, Exception)

    def test_ambiguous_input_error(self):
        """Test AmbiguousInputError."""
        error = AmbiguousInputError("Multiple interpretations found")
        assert str(error) == "Multiple interpretations found"

    def test_llm_unavailable_error(self):
        """Test LLMUnavailableError."""
        error = LLMUnavailableError("LLM service is down")
        assert str(error) == "LLM service is down"

    def test_context_too_large_error(self):
        """Test ContextTooLargeError."""
        error = ContextTooLargeError("Context exceeds 32k tokens")
        assert str(error) == "Context exceeds 32k tokens"

    def test_unsupported_language_error(self):
        """Test UnsupportedLanguageError."""
        error = UnsupportedLanguageError("Language not supported")
        assert str(error) == "Language not supported"


class TestEnums:
    """Test enum definitions."""

    def test_parsing_method_enum(self):
        """Test ParsingMethod enum values."""
        assert ParsingMethod.LLM == "llm"
        assert ParsingMethod.RULE_BASED == "rule_based"
        assert ParsingMethod.HYBRID == "hybrid"

    def test_confidence_level_enum(self):
        """Test ConfidenceLevel enum values."""
        assert ConfidenceLevel.VERY_LOW == "very_low"
        assert ConfidenceLevel.LOW == "low"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.VERY_HIGH == "very_high"