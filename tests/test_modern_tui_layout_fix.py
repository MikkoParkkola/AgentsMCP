#!/usr/bin/env python3
"""
Unit tests for ModernTUI layout checking fix.
Tests that the _update_section_if_changed method handles missing layout sections gracefully.
"""

import sys
import pytest
from unittest.mock import Mock, patch

# Import the actual module
try:
    from agentsmcp.ui.modern_tui import ModernTUI, TUIMode
except ImportError:
    pytest.skip("ModernTUI not available", allow_module_level=True)


class TestModernTUILayoutFix:
    """Test cases for the layout checking fix."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create minimal mock dependencies
        self.mock_config = Mock()
        self.mock_theme_manager = Mock()
        self.mock_conversation_manager = Mock()
        self.mock_orchestration_manager = Mock()
        
        # Mock the orchestration manager methods that are used in __init__
        self.mock_orchestration_manager.user_settings = {}

    @patch('agentsmcp.ui.modern_tui.Layout')  # Mock Rich Layout
    @patch('agentsmcp.ui.modern_tui.Console')  # Mock Rich Console
    def test_update_section_handles_missing_section(self, mock_console, mock_layout_class):
        """Test that _update_section_if_changed handles missing layout sections without crashing."""
        
        # Create a mock layout that raises KeyError for missing sections
        mock_layout_instance = Mock()
        mock_layout_instance.__getitem__ = Mock(side_effect=KeyError("No layout with name test_section"))
        mock_layout_class.return_value = mock_layout_instance
        
        # Create ModernTUI instance
        tui = ModernTUI(
            config=self.mock_config,
            theme_manager=self.mock_theme_manager,
            conversation_manager=self.mock_conversation_manager,
            orchestration_manager=self.mock_orchestration_manager
        )
        
        # Set the layout manually to control the test
        tui._layout = mock_layout_instance
        tui._render_cache = {}
        
        # Mock Panel class for testing
        mock_content = Mock()
        mock_content.__str__ = Mock(return_value="test content")
        
        # Test accessing a non-existent section - should not raise KeyError
        result = tui._update_section_if_changed("nonexistent_section", mock_content)
        
        # Should return False (no update happened) but not crash
        assert result is False
        
        # Verify the layout access was attempted (and failed safely)
        mock_layout_instance.__getitem__.assert_called_with("nonexistent_section")

    @patch('agentsmcp.ui.modern_tui.Layout')
    @patch('agentsmcp.ui.modern_tui.Console')
    def test_update_section_handles_content_fallback(self, mock_console, mock_layout_class):
        """Test that content section fallback logic works correctly."""
        
        # Create mock layout with main_area but no content section at root level
        mock_layout_instance = Mock()
        mock_main_area = Mock()
        
        def mock_getitem(key):
            if key == "content":
                raise KeyError("No layout with name content")
            elif key == "main_area":
                return mock_main_area
            else:
                raise KeyError(f"No layout with name {key}")
        
        mock_layout_instance.__getitem__ = Mock(side_effect=mock_getitem)
        mock_layout_class.return_value = mock_layout_instance
        
        # Create ModernTUI instance
        tui = ModernTUI(
            config=self.mock_config,
            theme_manager=self.mock_theme_manager,
            conversation_manager=self.mock_conversation_manager,
            orchestration_manager=self.mock_orchestration_manager
        )
        
        # Set up test state
        tui._layout = mock_layout_instance
        tui._render_cache = {}
        tui._sidebar_collapsed = True  # In collapsed mode, content maps to main_area
        
        # Mock content
        mock_content = Mock()
        mock_content.__str__ = Mock(return_value="test content")
        
        # Test accessing content section - should fall back to main_area
        result = tui._update_section_if_changed("content", mock_content)
        
        # Should return True (update happened)
        assert result is True
        
        # Verify main_area.update was called
        mock_main_area.update.assert_called_once_with(mock_content)

    @patch('agentsmcp.ui.modern_tui.Layout')
    @patch('agentsmcp.ui.modern_tui.Console')
    def test_update_section_handles_complete_fallback_failure(self, mock_console, mock_layout_class):
        """Test graceful handling when even fallback fails."""
        
        # Create mock layout that always raises KeyError
        mock_layout_instance = Mock()
        mock_layout_instance.__getitem__ = Mock(side_effect=KeyError("No sections available"))
        mock_layout_class.return_value = mock_layout_instance
        
        # Create ModernTUI instance
        tui = ModernTUI(
            config=self.mock_config,
            theme_manager=self.mock_theme_manager,
            conversation_manager=self.mock_conversation_manager,
            orchestration_manager=self.mock_orchestration_manager
        )
        
        # Set up test state
        tui._layout = mock_layout_instance
        tui._render_cache = {}
        tui._sidebar_collapsed = True
        
        # Mock content
        mock_content = Mock()
        mock_content.__str__ = Mock(return_value="test content")
        
        # Test accessing content section when even main_area doesn't exist
        result = tui._update_section_if_changed("content", mock_content)
        
        # Should return False but not crash
        assert result is False

    @patch('agentsmcp.ui.modern_tui.Layout')
    @patch('agentsmcp.ui.modern_tui.Console')
    def test_update_section_works_with_existing_section(self, mock_console, mock_layout_class):
        """Test that normal operation with existing sections still works."""
        
        # Create mock layout with an existing section
        mock_layout_instance = Mock()
        mock_section = Mock()
        mock_layout_instance.__getitem__ = Mock(return_value=mock_section)
        mock_layout_class.return_value = mock_layout_instance
        
        # Create ModernTUI instance
        tui = ModernTUI(
            config=self.mock_config,
            theme_manager=self.mock_theme_manager,
            conversation_manager=self.mock_conversation_manager,
            orchestration_manager=self.mock_orchestration_manager
        )
        
        # Set up test state
        tui._layout = mock_layout_instance
        tui._render_cache = {}
        
        # Mock content
        mock_content = Mock()
        mock_content.__str__ = Mock(return_value="test content")
        
        # Test accessing existing section
        result = tui._update_section_if_changed("header", mock_content)
        
        # Should return True (update happened)
        assert result is True
        
        # Verify section.update was called
        mock_section.update.assert_called_once_with(mock_content)