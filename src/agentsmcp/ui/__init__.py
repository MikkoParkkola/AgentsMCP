"""
Revolutionary CLI UI System for AgentsMCP

Beautiful, adaptive, and intelligent command-line interfaces inspired by:
- Claude Code's elegant interaction patterns
- Codex CLI's responsive design principles  
- Gemini CLI's sophisticated status displays

Features:
- Adaptive dark/light theme detection
- Real-time status and statistics
- Interactive dashboards
- Smooth animations and transitions
- Revolutionary Apple-style user experience
"""

from .theme_manager import ThemeManager, Theme
from .status_dashboard import StatusDashboard
from .command_interface import CommandInterface
from .statistics_display import StatisticsDisplay
from .ui_components import UIComponents
from .cli_app import CLIApp

__all__ = [
    'ThemeManager',
    'Theme', 
    'StatusDashboard',
    'CommandInterface',
    'StatisticsDisplay',
    'UIComponents',
    'CLIApp'
]