"""
V4 UI - Persistent Workspace Controller for Agent Orchestration

This module provides the Mission Control Center for AI agents - a persistent TUI interface
that never exits and provides real-time orchestration of agent workflows.

Features:
- Persistent event loop (never exits on EOF)
- Non-blocking keyboard input handling  
- Live agent display with status and progress
- Interactive controls for agent management
- Real-time updates and sequential thinking display
- htop/k9s-style interface for AI agents
"""

from .workspace_controller import WorkspaceController, main

__all__ = ['WorkspaceController', 'main']