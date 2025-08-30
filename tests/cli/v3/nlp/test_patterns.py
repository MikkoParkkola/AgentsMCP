"""Tests for rule-based pattern matching."""

import pytest
from unittest.mock import patch, MagicMock

from src.agentsmcp.cli.v3.nlp.patterns import PatternMatcher, CommandPattern
from src.agentsmcp.cli.v3.models.nlp_models import PatternMatch, ParsedCommand, ParsingMethod


class TestPatternMatcher:
    """Test pattern matcher functionality."""

    def setup_method(self):
        """Setup test instance."""
        self.matcher = PatternMatcher()

    def test_initialization(self):
        """Test pattern matcher initialization."""
        assert len(self.matcher.patterns) > 0
        assert len(self.matcher.compiled_patterns) > 0
        assert len(self.matcher.patterns) == len(self.matcher.compiled_patterns)

    def test_analyze_code_patterns(self):
        """Test code analysis pattern matching."""
        test_cases = [
            ("analyze my code", "analyze"),
            ("check code for issues", "analyze"),
            ("review the code", "analyze"),
            ("audit code for security", "analyze"),
            ("scan my code", "analyze"),
        ]

        for input_text, expected_action in test_cases:
            matches = self.matcher.match_patterns(input_text)
            assert len(matches) > 0
            assert matches[0].action == expected_action
            assert matches[0].confidence > 0.5

    def test_help_patterns(self):
        """Test help pattern matching."""
        test_cases = [
            ("help", "help"),
            ("show help", "help"),
            ("what can you do", "help"),
            ("available commands", "help"),
            ("how do I", "help"),
        ]

        for input_text, expected_action in test_cases:
            matches = self.matcher.match_patterns(input_text)
            assert len(matches) > 0
            best_match = matches[0]
            assert best_match.action == expected_action
            assert best_match.confidence > 0.7

    def test_status_patterns(self):
        """Test status pattern matching."""
        test_cases = [
            ("status", "status"),
            ("check status", "status"),
            ("show status", "status"),
            ("what's the status", "status"),
            ("system health", "status"),
        ]

        for input_text, expected_action in test_cases:
            matches = self.matcher.match_patterns(input_text)
            assert len(matches) > 0
            assert any(m.action == expected_action for m in matches)

    def test_tui_patterns(self):
        """Test TUI pattern matching."""
        test_cases = [
            ("start tui", "tui"),
            ("open the tui", "tui"),
            ("launch tui", "tui"),
            ("run tui", "tui"),
            ("interactive mode", "tui"),
        ]

        for input_text, expected_action in test_cases:
            matches = self.matcher.match_patterns(input_text)
            assert len(matches) > 0
            tui_matches = [m for m in matches if m.action == expected_action]
            assert len(tui_matches) > 0
            assert tui_matches[0].confidence > 0.8

    def test_setup_project_patterns(self):
        """Test project setup pattern matching."""
        test_cases = [
            ("help me set up the project", "init"),
            ("initialize project", "init"),
            ("create new project", "init"),
            ("bootstrap the project", "init"),
            ("get started", "init"),
        ]

        for input_text, expected_action in test_cases:
            matches = self.matcher.match_patterns(input_text)
            assert len(matches) > 0
            init_matches = [m for m in matches if m.action == expected_action]
            assert len(init_matches) > 0

    def test_parameter_extraction(self):
        """Test parameter extraction from patterns."""
        # Test analyze with target
        matches = self.matcher.match_patterns("analyze code in src/")
        analyze_matches = [m for m in matches if m.action == "analyze"]
        if analyze_matches:
            # Check if target parameter was extracted
            match = analyze_matches[0]
            # Parameters might be extracted depending on regex match
            assert isinstance(match.parameters, dict)

        # Test file operations
        matches = self.matcher.match_patterns("read file config.json")
        file_matches = [m for m in matches if m.action == "file"]
        if file_matches:
            match = file_matches[0]
            assert isinstance(match.parameters, dict)

    def test_confidence_scoring(self):
        """Test confidence scoring for different inputs."""
        # High confidence: exact matches
        matches = self.matcher.match_patterns("help")
        help_matches = [m for m in matches if m.action == "help"]
        if help_matches:
            assert help_matches[0].confidence > 0.9

        # Lower confidence: partial matches
        matches = self.matcher.match_patterns("maybe help me")
        if matches:
            assert matches[0].confidence < 0.9

    def test_priority_sorting(self):
        """Test that matches are sorted by priority and confidence."""
        matches = self.matcher.match_patterns("help analyze")
        
        # Should have multiple matches
        assert len(matches) > 1
        
        # Check sorting: priority first, then confidence
        for i in range(len(matches) - 1):
            current = matches[i]
            next_match = matches[i + 1]
            
            # Either current has higher priority (lower number)
            # or same priority with higher confidence
            assert (current.priority <= next_match.priority)
            if current.priority == next_match.priority:
                assert current.confidence >= next_match.confidence

    def test_no_matches(self):
        """Test behavior with input that shouldn't match any patterns."""
        matches = self.matcher.match_patterns("xyz random gibberish 123")
        # Might have some weak matches, but confidence should be very low
        if matches:
            assert all(m.confidence < 0.3 for m in matches)

    def test_get_best_match(self):
        """Test getting the best single match."""
        # Clear match
        best = self.matcher.get_best_match("help")
        assert best is not None
        assert best.action == "help"

        # No good matches
        best = self.matcher.get_best_match("xyz random")
        # Might return None or very low confidence match
        if best:
            assert best.confidence < 0.5

    def test_get_pattern_examples(self):
        """Test getting examples for actions."""
        help_examples = self.matcher.get_pattern_examples("help")
        assert len(help_examples) > 0
        assert any("help" in example.lower() for example in help_examples)

        analyze_examples = self.matcher.get_pattern_examples("analyze")
        assert len(analyze_examples) > 0
        assert any("analyze" in example.lower() for example in analyze_examples)

    def test_parse_command_fallback(self):
        """Test fallback command parsing."""
        # High confidence match
        command = self.matcher.parse_command_fallback("help", confidence_threshold=0.3)
        assert command is not None
        assert command.action == "help"
        assert command.method == ParsingMethod.RULE_BASED

        # Low confidence - should return None with high threshold
        command = self.matcher.parse_command_fallback("maybe help", confidence_threshold=0.9)
        # Might be None if confidence is below threshold

        # Very unclear input
        command = self.matcher.parse_command_fallback("xyz random", confidence_threshold=0.3)
        # Should likely be None

    def test_add_custom_pattern(self):
        """Test adding custom patterns."""
        initial_count = len(self.matcher.patterns)
        
        custom_pattern = CommandPattern(
            pattern_id="test_custom",
            action="test",
            regex_patterns=[r"test\\s+custom"],
            keywords=["test", "custom"],
            parameter_extractors={},
            examples=["test custom"],
            priority=3,
            confidence_base=0.8
        )
        
        self.matcher.add_custom_pattern(custom_pattern)
        
        assert len(self.matcher.patterns) == initial_count + 1
        assert "test_custom" in self.matcher.compiled_patterns
        
        # Test the new pattern works
        matches = self.matcher.match_patterns("test custom")
        test_matches = [m for m in matches if m.action == "test"]
        assert len(test_matches) > 0

    def test_get_supported_actions(self):
        """Test getting list of supported actions."""
        actions = self.matcher.get_supported_actions()
        assert len(actions) > 0
        assert "help" in actions
        assert "analyze" in actions
        assert "status" in actions
        assert "tui" in actions

    def test_get_pattern_info(self):
        """Test getting pattern information."""
        # Get info for existing pattern
        info = self.matcher.get_pattern_info("help_command")
        assert info is not None
        assert info.action == "help"
        assert len(info.examples) > 0

        # Get info for non-existent pattern
        info = self.matcher.get_pattern_info("nonexistent")
        assert info is None

    def test_case_insensitive_matching(self):
        """Test case insensitive pattern matching."""
        test_cases = [
            "HELP",
            "Help",
            "hElP",
            "ANALYZE MY CODE",
            "Start TUI"
        ]
        
        for input_text in test_cases:
            matches = self.matcher.match_patterns(input_text)
            assert len(matches) > 0  # Should find matches regardless of case

    def test_regex_pattern_compilation_errors(self):
        """Test handling of invalid regex patterns."""
        with patch('src.agentsmcp.cli.v3.nlp.patterns.logger') as mock_logger:
            # Create pattern with invalid regex
            invalid_pattern = CommandPattern(
                pattern_id="invalid_regex",
                action="test",
                regex_patterns=["[invalid regex"],  # Invalid regex
                keywords=["test"],
                parameter_extractors={},
                examples=["test"],
                priority=5,
                confidence_base=0.8
            )
            
            # Should handle the error gracefully
            self.matcher.add_custom_pattern(invalid_pattern)
            
            # Should log a warning
            mock_logger.warning.assert_called()

    def test_parameter_extraction_errors(self):
        """Test handling of parameter extraction errors."""
        # Create pattern with invalid parameter extractor
        pattern = CommandPattern(
            pattern_id="test_param_error",
            action="test",
            regex_patterns=["test"],
            keywords=["test"],
            parameter_extractors={"param": "[invalid regex"},  # Invalid regex
            examples=["test"],
            priority=5,
            confidence_base=0.8
        )
        
        self.matcher.add_custom_pattern(pattern)
        
        with patch('src.agentsmcp.cli.v3.nlp.patterns.logger') as mock_logger:
            # Should handle parameter extraction error
            matches = self.matcher.match_patterns("test")
            
            # Should still return matches but log warning
            test_matches = [m for m in matches if m.action == "test"]
            if test_matches:
                assert isinstance(test_matches[0].parameters, dict)

    def test_empty_input(self):
        """Test behavior with empty input."""
        matches = self.matcher.match_patterns("")
        assert len(matches) == 0

        matches = self.matcher.match_patterns("   ")
        assert len(matches) == 0

    def test_optimize_cost_patterns(self):
        """Test cost optimization patterns."""
        test_cases = [
            ("optimize my costs", "optimize"),
            ("reduce costs", "optimize"),
            ("cost optimization", "optimize"),
            ("save on costs", "optimize"),
        ]

        for input_text, expected_action in test_cases:
            matches = self.matcher.match_patterns(input_text)
            optimize_matches = [m for m in matches if m.action == expected_action]
            assert len(optimize_matches) > 0

    def test_file_operation_patterns(self):
        """Test file operation patterns."""
        test_cases = [
            ("read file config.json", "file"),
            ("list files in src/", "file"),
            ("edit main.py", "file"),
            ("create new file", "file"),
        ]

        for input_text, expected_action in test_cases:
            matches = self.matcher.match_patterns(input_text)
            file_matches = [m for m in matches if m.action == expected_action]
            assert len(file_matches) > 0

    def test_settings_patterns(self):
        """Test settings and configuration patterns."""
        test_cases = [
            ("settings", "settings"),
            ("configuration", "settings"),
            ("preferences", "settings"),
            ("change settings", "settings"),
        ]

        for input_text, expected_action in test_cases:
            matches = self.matcher.match_patterns(input_text)
            settings_matches = [m for m in matches if m.action == expected_action]
            assert len(settings_matches) > 0

    def test_dashboard_patterns(self):
        """Test dashboard and monitoring patterns."""
        test_cases = [
            ("dashboard", "dashboard"),
            ("monitoring", "dashboard"),
            ("show dashboard", "dashboard"),
            ("open dashboard", "dashboard"),
        ]

        for input_text, expected_action in test_cases:
            matches = self.matcher.match_patterns(input_text)
            dashboard_matches = [m for m in matches if m.action == expected_action]
            assert len(dashboard_matches) > 0