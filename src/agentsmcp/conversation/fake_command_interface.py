"""
Fake command interface for testing conversational interface.
Provides minimal stub implementation for ConversationManager tests.
"""

from typing import Dict, List, Tuple, Any, Optional


class FakeCommandInterface:
    """A lightweight stub for CommandInterface used in tests.
    
    Records all calls and returns deterministic responses so tests
    can verify behavior without depending on the real CLI.
    """
    
    def __init__(self):
        self.called: List[Tuple[str, tuple, dict]] = []
        self._status = "All systems nominal - fake status"
        self._theme = "auto"
        
    # Public API used by ConversationManager
    
    async def handle_command(self, command: str, *args, **kwargs) -> str:
        """Handle command execution with fake responses."""
        self.called.append(("handle_command", (command, args, kwargs), {}))
        
        if command == "status":
            return f"🚀 System Status: {self._status}\n   Agents: 3 running\n   Memory: 45% used\n   Network: Connected"
        elif command == "settings":
            return "⚙️  Settings dialog opened (fake)\n   Current provider: ollama-turbo\n   Current model: gpt-oss:120b"
        elif command == "dashboard":
            return "📊 Dashboard started (fake)\n   Real-time monitoring active\n   URL: http://localhost:8000/dashboard"
        elif command == "web":
            return "🌐 Web API endpoints (fake):\n   - GET /api/status\n   - POST /api/agents\n   - GET /api/dashboard"
        elif command.startswith("theme"):
            parts = command.split()
            if len(parts) > 1:
                self._theme = parts[1]
                return f"🎨 Theme changed to '{self._theme}' (fake)"
            else:
                return f"🎨 Current theme: '{self._theme}' (fake)"
        else:
            return f"✅ Executed fake command '{command}' with args {args} {kwargs}"
    
    # Helper methods for tests
    
    def set_status(self, status: str) -> None:
        """Allow tests to control status response."""
        self._status = status
        
    def set_theme(self, theme: str) -> None:
        """Allow tests to control theme response."""
        self._theme = theme
        
    def reset_calls(self) -> None:
        """Clear call history for fresh test state."""
        self.called.clear()
        
    def get_calls(self) -> List[Tuple[str, tuple, dict]]:
        """Get recorded calls for test assertions."""
        return self.called.copy()