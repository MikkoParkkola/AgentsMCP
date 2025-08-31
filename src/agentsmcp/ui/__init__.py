"""UI package with lazy-loading for heavy components.

This module avoids importing large UI submodules at startup. Access classes via
attributes (e.g., ``from agentsmcp.ui import CommandInterface``) to trigger a
lazy import on first use.
"""

from ..lazy_loading import lazy_import

_theme_manager = lazy_import('.theme_manager', __package__)
_status_dashboard = lazy_import('.status_dashboard', __package__)
_command_interface = lazy_import('.command_interface', __package__)
_statistics_display = lazy_import('.statistics_display', __package__)
_ui_components = lazy_import('.ui_components', __package__)
_cli_app = lazy_import('.cli_app', __package__)


def _attr_map():
    return {
        'ThemeManager': _theme_manager.ThemeManager,
        'Theme': _theme_manager.Theme,
        'StatusDashboard': _status_dashboard.StatusDashboard,
        'CommandInterface': _command_interface.CommandInterface,
        'StatisticsDisplay': _statistics_display.StatisticsDisplay,
        'UIComponents': _ui_components.UIComponents,
        'CLIApp': _cli_app.CLIApp,
    }


def __getattr__(name):  # PEP 562 lazy attribute access for modules
    m = _attr_map()
    if name in m:
        return m[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'ThemeManager', 'Theme', 'StatusDashboard', 'CommandInterface',
    'StatisticsDisplay', 'UIComponents', 'CLIApp'
]
