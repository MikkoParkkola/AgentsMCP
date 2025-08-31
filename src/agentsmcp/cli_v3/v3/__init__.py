"""CLI v3 architecture with intelligent command engine and natural language support.

This package provides the revolutionary CLI v3 experience that integrates:
- Natural language command processing
- Progressive disclosure based on user skill level  
- Intelligent command routing and execution
- Cross-modal coordination (CLI/TUI/WebUI)
- Comprehensive command registry and discovery
"""

from .main import CliV3Main, create_cli_v3, main as cli_v3_main

__all__ = ["CliV3Main", "create_cli_v3", "cli_v3_main"]