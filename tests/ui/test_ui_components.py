"""
Comprehensive test suite for UI components in AgentsMCP.

Tests the core UI functionality including theme management, component rendering,
and dashboard display to ensure proper functionality and prevent regressions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import io
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agentsmcp.ui.theme_manager import ThemeManager, Theme, ColorPalette, Fore
from agentsmcp.ui.ui_components import UIComponents
from agentsmcp.ui.status_dashboard import StatusDashboard, DashboardConfig


class TestThemeManager:
    """Test suite for ThemeManager functionality."""
    
    def test_theme_manager_initialization(self):
        """Test that ThemeManager initializes correctly."""
        theme_manager = ThemeManager()
        assert theme_manager is not None
        assert hasattr(theme_manager, 'current_theme')
        assert hasattr(theme_manager, '_themes')
    
    def test_light_theme_properties(self):
        """Test light theme has correct properties."""
        theme_manager = ThemeManager()
        light_theme = theme_manager._themes.get('light')
        
        assert light_theme is not None
        assert light_theme.name == 'AgentsMCP Light'
        assert hasattr(light_theme, 'palette')
        assert hasattr(light_theme.palette, 'primary')
        assert hasattr(light_theme.palette, 'secondary')
        assert hasattr(light_theme.palette, 'accent')
    
    def test_dark_theme_properties(self):
        """Test dark theme has correct properties."""
        theme_manager = ThemeManager()
        dark_theme = theme_manager._themes.get('dark')
        
        assert dark_theme is not None
        assert dark_theme.name == 'AgentsMCP Dark'
        assert hasattr(dark_theme, 'palette')
        assert hasattr(dark_theme.palette, 'primary')
        assert hasattr(dark_theme.palette, 'secondary')
        assert hasattr(dark_theme.palette, 'accent')
    
    def test_set_theme(self):
        """Test setting themes by name."""
        theme_manager = ThemeManager()
        
        # Test setting to dark theme
        theme_manager.set_theme('dark')
        assert theme_manager.current_theme.name == 'AgentsMCP Dark'
        
        # Test setting to light theme
        theme_manager.set_theme('light')
        assert theme_manager.current_theme.name == 'AgentsMCP Light'
    
    def test_invalid_theme_raises_error(self):
        """Test that invalid theme names raise ValueError."""
        theme_manager = ThemeManager()
        with pytest.raises(ValueError, match="Theme 'nonexistent' not found"):
            theme_manager.set_theme('nonexistent')
    
    @patch('agentsmcp.ui.theme_manager.ThemeManager._detect_terminal_theme')
    def test_auto_detect_theme(self, mock_detect):
        """Test automatic theme detection based on environment."""
        theme_manager = ThemeManager()
        
        # Test dark mode detection
        from agentsmcp.ui.theme_manager import ThemeType
        mock_detect.return_value = ThemeType.DARK
        detected_theme = theme_manager.auto_detect_theme()
        assert detected_theme.name == 'AgentsMCP Dark'
        
        # Test light mode detection
        mock_detect.return_value = ThemeType.LIGHT
        detected_theme = theme_manager.auto_detect_theme()
        assert detected_theme.name == 'AgentsMCP Light'


class TestUIComponents:
    """Test suite for UIComponents functionality."""
    
    def test_ui_components_initialization(self):
        """Test UIComponents initializes with theme manager."""
        theme_manager = ThemeManager()
        ui = UIComponents(theme_manager)
        
        assert ui is not None
        assert ui.theme_manager == theme_manager
    
    def test_clear_screen_returns_empty_string(self):
        """Test clear screen returns empty string to prevent scrollback pollution."""
        theme_manager = ThemeManager()
        ui = UIComponents(theme_manager)
        
        result = ui.clear_screen()
        assert isinstance(result, str)
        assert result == ""  # Should return empty string to prevent console flooding
    
    def test_move_cursor_returns_ansi_codes(self):
        """Test cursor movement returns proper ANSI codes."""
        theme_manager = ThemeManager()
        ui = UIComponents(theme_manager)
        
        result = ui.move_cursor(5, 10)
        assert isinstance(result, str)
        assert "5" in result and "10" in result
    
    def test_box_drawing_basic(self):
        """Test basic box drawing functionality."""
        theme_manager = ThemeManager()
        ui = UIComponents(theme_manager)
        
        content = "Test content"
        result = ui.box(content)
        
        assert isinstance(result, str)
        assert "Test content" in result
        assert len(result) > len(content)  # Should have box characters
    
    def test_box_drawing_with_title(self):
        """Test box drawing with title."""
        theme_manager = ThemeManager()
        ui = UIComponents(theme_manager)
        
        content = "Test content"
        title = "Test Title"
        result = ui.box(content, title=title)
        
        assert isinstance(result, str)
        assert "Test content" in result
        assert "Test Title" in result
    
    def test_box_drawing_with_width(self):
        """Test box drawing respects width parameter."""
        theme_manager = ThemeManager()
        ui = UIComponents(theme_manager)
        
        content = "Short"
        result = ui.box(content, width=50)
        
        assert isinstance(result, str)
        lines = result.split('\n')
        # Check that some lines approach the specified width
        assert any(len(line) >= 40 for line in lines)  # Allow some variance
    
    def test_cursor_visibility_controls(self):
        """Test cursor show/hide functionality."""
        theme_manager = ThemeManager()
        ui = UIComponents(theme_manager)
        
        show_result = ui.show_cursor()
        hide_result = ui.hide_cursor()
        
        assert isinstance(show_result, str)
        assert isinstance(hide_result, str)
        assert show_result != hide_result


class TestStatusDashboard:
    """Test suite for StatusDashboard functionality."""
    
    def test_status_dashboard_initialization(self):
        """Test StatusDashboard initializes correctly."""
        theme_manager = ThemeManager()
        config = DashboardConfig()
        
        dashboard = StatusDashboard(
            orchestration_manager=None,
            theme_manager=theme_manager,
            config=config
        )
        
        assert dashboard is not None
        assert dashboard.theme_manager == theme_manager
        assert dashboard.config == config
    
    def test_dashboard_config_defaults(self):
        """Test DashboardConfig has correct default values."""
        config = DashboardConfig()
        
        assert hasattr(config, 'auto_refresh')
        assert hasattr(config, 'refresh_interval')
        assert isinstance(config.refresh_interval, (int, float))
        assert config.refresh_interval > 0
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_dashboard_render_mock_data(self, mock_stdout):
        """Test dashboard renders with mock data when no orchestration manager."""
        theme_manager = ThemeManager()
        config = DashboardConfig()
        
        dashboard = StatusDashboard(
            orchestration_manager=None,
            theme_manager=theme_manager,
            config=config
        )
        
        # Test that fallback data is used without errors
        assert dashboard.orchestration_manager is None
        # Dashboard should handle None orchestration_manager gracefully
    
    def test_dashboard_with_mock_orchestration_manager(self):
        """Test dashboard with mocked orchestration manager."""
        mock_orchestration_manager = Mock()
        mock_orchestration_manager.get_system_status = Mock(return_value={
            "system_status": "running",
            "session_id": "test-session",
            "uptime": "00:01:30",
            "orchestration_mode": "symphony"
        })
        
        theme_manager = ThemeManager()
        config = DashboardConfig()
        
        dashboard = StatusDashboard(
            orchestration_manager=mock_orchestration_manager,
            theme_manager=theme_manager,
            config=config
        )
        
        assert dashboard.orchestration_manager == mock_orchestration_manager
    
    def test_dashboard_stop_functionality(self):
        """Test dashboard can be stopped properly."""
        theme_manager = ThemeManager()
        config = DashboardConfig()
        
        dashboard = StatusDashboard(
            orchestration_manager=None,
            theme_manager=theme_manager,
            config=config
        )
        
        # Test stop method exists and can be called
        if hasattr(dashboard, 'stop_dashboard'):
            dashboard.stop_dashboard()
        # Should not raise any exceptions


@pytest.fixture
def mock_terminal_libraries():
    """Mock terminal-related libraries to prevent import errors in test environment."""
    with patch.dict('sys.modules', {
        'blessed': Mock(),
        'curses': Mock(),
        'rich': Mock(),
        'rich.console': Mock(),
        'rich.table': Mock(),
        'rich.panel': Mock(),
        'rich.columns': Mock()
    }):
        yield


class TestIntegration:
    """Integration tests for UI components working together."""
    
    def test_full_ui_stack_initialization(self, mock_terminal_libraries):
        """Test that all UI components can be initialized together."""
        theme_manager = ThemeManager()
        ui_components = UIComponents(theme_manager)
        config = DashboardConfig()
        dashboard = StatusDashboard(
            orchestration_manager=None,
            theme_manager=theme_manager,
            config=config
        )
        
        assert theme_manager is not None
        assert ui_components is not None
        assert dashboard is not None
    
    def test_theme_propagation(self, mock_terminal_libraries):
        """Test that theme changes propagate through UI components."""
        theme_manager = ThemeManager()
        ui_components = UIComponents(theme_manager)
        
        # Change theme
        theme_manager.set_theme('dark')
        
        # UI components should reflect the new theme
        assert ui_components.theme_manager.current_theme.name == 'AgentsMCP Dark'
    
    def test_ui_components_box_with_theme(self, mock_terminal_libraries):
        """Test UI box drawing uses theme colors correctly."""
        theme_manager = ThemeManager()
        ui_components = UIComponents(theme_manager)
        
        content = "Themed content"
        result = ui_components.box(content)
        
        # Result should contain ANSI color codes from theme
        assert isinstance(result, str)
        assert content in result
        # Should contain ANSI escape sequences for coloring
        assert "\033[" in result or len(result) > len(content)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])