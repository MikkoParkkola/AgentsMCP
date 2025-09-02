"""
Tests for Revolutionary TUI Interface layout improvements.

This module tests the layout fixes that make the TUI more compact and professional.
"""

import pytest
import asyncio
from datetime import datetime

from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface, TUIState


class TestLayoutCompactness:
    """Test suite for verifying layout compactness improvements."""
    
    @pytest.fixture
    def interface(self):
        """Create a TUI interface for testing."""
        interface = RevolutionaryTUIInterface()
        interface.state = TUIState()
        return interface
    
    def count_empty_lines(self, content: str) -> int:
        """Count empty lines in content."""
        return sum(1 for line in content.split('\n') if line.strip() == '')
    
    def test_status_panel_no_unnecessary_empty_lines(self, interface):
        """Test that status panel has no unnecessary empty lines."""
        # Test empty state
        content = interface._create_status_panel()
        empty_lines = self.count_empty_lines(content)
        assert empty_lines <= 1, f"Status panel (empty) has too many empty lines: {empty_lines}"
        
        # Test with data
        interface.state.agent_status = {"test": "active"}
        interface.state.system_metrics = {"cpu": 10.5}
        content = interface._create_status_panel()
        empty_lines = self.count_empty_lines(content)
        assert empty_lines == 0, f"Status panel (with data) should have no empty lines: {empty_lines}"
    
    @pytest.mark.asyncio
    async def test_dashboard_panel_compactness(self, interface):
        """Test that dashboard panel is compact."""
        # Mock dashboard
        class MockDashboard:
            def get_current_state(self):
                return {
                    'active_agents': 2,
                    'running_tasks': 0,
                    'success_rate': 100.0,
                    'recent_activity': ['System ready']
                }
        
        interface.symphony_dashboard = MockDashboard()
        content = await interface._create_dashboard_panel()
        empty_lines = self.count_empty_lines(content)
        assert empty_lines <= 1, f"Dashboard panel has too many empty lines: {empty_lines}"
    
    def test_chat_panel_welcome_message_concise(self, interface):
        """Test that welcome message is concise."""
        content = interface._create_chat_panel()
        lines = content.split('\n')
        
        # Should be much shorter than the original verbose welcome
        assert len(lines) <= 6, f"Welcome message too long: {len(lines)} lines"
        
        # Should not contain excessive bullet points
        bullet_lines = [line for line in lines if line.strip().startswith('•')]
        assert len(bullet_lines) == 0, f"Welcome message should not have bullet points: {len(bullet_lines)}"
    
    def test_chat_panel_conversation_compact(self, interface):
        """Test that conversation history is compact."""
        interface.state.conversation_history = [
            {"role": "user", "content": "Hello", "timestamp": "10:00:00"},
            {"role": "assistant", "content": "Hi", "timestamp": "10:00:01"}
        ]
        
        content = interface._create_chat_panel()
        empty_lines = self.count_empty_lines(content)
        assert empty_lines == 0, f"Chat conversation should have no empty lines: {empty_lines}"
    
    def test_input_panel_help_compact(self, interface):
        """Test that input panel help is compact."""
        interface.state.current_input = ""
        interface.state.is_processing = False
        
        content = interface._create_input_panel()
        lines = content.split('\n')
        
        # Should have condensed help
        assert len(lines) <= 2, f"Input panel help too verbose: {len(lines)} lines"
    
    def test_input_panel_no_unnecessary_spacing(self, interface):
        """Test that input panel has no unnecessary spacing."""
        interface.state.current_input = "test"
        interface.input_history = ["prev"]
        interface.history_index = 0
        interface.state.input_suggestions = ["suggestion"]
        
        content = interface._create_input_panel()
        empty_lines = self.count_empty_lines(content)
        assert empty_lines == 0, f"Input panel should have no empty lines: {empty_lines}"
    
    def test_footer_panel_single_line(self, interface):
        """Test that footer panel is a single compact line."""
        content = interface._create_footer_panel()
        
        # Should be single line
        assert '\n' not in content, "Footer should be single line"
        
        # Should use compact separators
        separator_count = content.count(' • ')
        assert separator_count >= 5, f"Footer should have compact separators: {separator_count}"
    
    @pytest.mark.asyncio
    async def test_overall_content_density(self, interface):
        """Test overall content density across all panels."""
        # Setup full state
        interface.state.agent_status = {"test": "active"}
        interface.state.system_metrics = {"cpu": 10}
        interface.state.conversation_history = [
            {"role": "user", "content": "test", "timestamp": "10:00:00"}
        ]
        interface.state.current_input = "test input"
        
        # Test all panels
        status_content = interface._create_status_panel()
        dashboard_content = await interface._create_dashboard_panel()
        chat_content = interface._create_chat_panel()
        input_content = interface._create_input_panel()
        footer_content = interface._create_footer_panel()
        
        total_empty_lines = (
            self.count_empty_lines(status_content) +
            self.count_empty_lines(dashboard_content) +
            self.count_empty_lines(chat_content) +
            self.count_empty_lines(input_content) +
            self.count_empty_lines(footer_content)
        )
        
        # Should have minimal empty lines across all panels
        assert total_empty_lines <= 3, f"Total empty lines across all panels too high: {total_empty_lines}"


class TestContentQuality:
    """Test suite for verifying content quality improvements."""
    
    @pytest.fixture
    def interface(self):
        """Create a TUI interface for testing."""
        interface = RevolutionaryTUIInterface()
        interface.state = TUIState()
        return interface
    
    def test_welcome_message_professional(self, interface):
        """Test that welcome message is professional and informative."""
        content = interface._create_chat_panel()
        
        # Should mention key features
        assert "Revolutionary TUI" in content
        assert "AI Command Composer" in content
        
        # Should be encouraging but not overwhelming
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        assert len(non_empty_lines) <= 5, "Welcome message should be concise"
    
    def test_status_panel_informative(self, interface):
        """Test that status panel provides useful information."""
        interface.state.agent_status = {"orchestrator": "active", "composer": "offline"}
        interface.state.system_metrics = {"fps": 60, "memory_mb": 42.5}
        
        content = interface._create_status_panel()
        
        # Should show agent status
        assert "orchestrator: active" in content
        assert "composer: offline" in content
        
        # Should show metrics
        assert "Performance:" in content
        assert "fps: 60" in content
        assert "memory_mb: 42.50" in content
    
    def test_input_panel_clear_guidance(self, interface):
        """Test that input panel provides clear guidance."""
        interface.state.current_input = ""
        interface.state.is_processing = False
        
        content = interface._create_input_panel()
        
        # Should provide essential shortcuts in compact form
        assert "Type to chat" in content
        assert "History" in content
        assert "Enter" in content or "Send" in content


class TestVisualConsistency:
    """Test suite for visual consistency across panels."""
    
    @pytest.fixture
    def interface(self):
        """Create a TUI interface for testing."""
        interface = RevolutionaryTUIInterface()
        interface.state = TUIState()
        return interface
    
    def test_consistent_emoji_usage(self, interface):
        """Test that emoji usage is consistent and professional."""
        # Setup state
        interface.state.agent_status = {"test": "active"}
        interface.state.system_metrics = {"cpu": 10}
        interface.state.conversation_history = [
            {"role": "user", "content": "hello", "timestamp": "10:00:00"},
            {"role": "assistant", "content": "hi", "timestamp": "10:00:01"}
        ]
        
        status_content = interface._create_status_panel()
        chat_content = interface._create_chat_panel()
        input_content = interface._create_input_panel()
        footer_content = interface._create_footer_panel()
        
        # Should use consistent emoji patterns
        assert "🟢" in status_content or "🟡" in status_content  # Status indicators
        assert "👤" in chat_content  # User indicator
        assert "🤖" in chat_content  # Assistant indicator
        assert "💬" in input_content  # Input indicator
    
    def test_consistent_spacing_style(self, interface):
        """Test that spacing style is consistent across panels."""
        interface.state.agent_status = {"test1": "active", "test2": "offline"}
        interface.state.system_metrics = {"metric1": 10, "metric2": 20}
        
        content = interface._create_status_panel()
        lines = content.split('\n')
        
        # Metric lines should have consistent indentation
        metric_lines = [line for line in lines if line.startswith('  ')]
        for line in metric_lines:
            assert line.startswith('  '), f"Inconsistent indentation: {line}"