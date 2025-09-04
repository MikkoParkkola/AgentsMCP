"""Enhanced Progress Tracking System for Multi-turn LLM Processing."""

import time
import asyncio
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ProcessingPhase(Enum):
    """Different phases of LLM processing."""
    ANALYZING = "🔍 Analyzing your request"
    TOOL_EXECUTION = "🛠️ Executing tool"
    PROCESSING_RESULTS = "🔄 Processing results"
    MULTI_TURN = "📊 Multi-turn processing"
    GENERATING_RESPONSE = "💭 Generating response"
    FINALIZING = "✨ Finalizing response"
    STREAMING = "🎯 Streaming response"


@dataclass
class ToolExecutionInfo:
    """Information about tool being executed."""
    name: str
    description: str
    index: int
    total: int


class ProgressTracker:
    """
    Tracks progress through multi-turn LLM processing with detailed status updates.
    Provides timing information and phase-specific status messages.
    """
    
    # Tool icon mapping for better visibility
    TOOL_ICONS = {
        "search_files": "🔍",
        "find_files": "📁",
        "read_file": "📄",
        "write_file": "✏️",
        "edit_file": "📝",
        "bash_command": "⚡",
        "shell_command": "💻",
        "web_search": "🌐",
        "mcp_tool": "🔧",
        "grep": "🔎",
        "glob": "🌟",
        "task": "📋",
        "memory": "💾",
        "pieces": "🧩",
        "git": "📦",
        "lsp": "🔤",
        "sequential_thinking": "🧠",
        "semgrep": "🛡️",
        "trivy": "🔒",
        "default": "🛠️"
    }
    
    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize progress tracker.
        
        Args:
            progress_callback: Async function to call with status updates
        """
        self.start_time = time.time()
        self.progress_callback = progress_callback
        self.current_phase: Optional[ProcessingPhase] = None
        self.current_turn = 0
        self.max_turns = 0
        self.tools_executed = 0
        
    async def update_phase(self, phase: ProcessingPhase, details: str = "") -> None:
        """Update the current processing phase."""
        self.current_phase = phase
        elapsed = time.time() - self.start_time
        
        if details:
            message = f"{phase.value}: {details} [{elapsed:.1f}s]"
        else:
            message = f"{phase.value} [{elapsed:.1f}s]"
            
        await self._notify(message)
    
    async def update_tool_execution(self, tool_info: ToolExecutionInfo) -> None:
        """Update progress during tool execution."""
        elapsed = time.time() - self.start_time
        icon = self._get_tool_icon(tool_info.name)
        
        message = f"{icon} Executing {tool_info.name}"
        if tool_info.total > 1:
            message += f" ({tool_info.index + 1}/{tool_info.total})"
        if tool_info.description:
            message += f": {tool_info.description}"
        message += f" [{elapsed:.1f}s]"
        
        await self._notify(message)
        self.tools_executed += 1
    
    async def update_multi_turn(self, turn: int, max_turns: int, action: str = "") -> None:
        """Update progress for multi-turn processing."""
        self.current_turn = turn
        self.max_turns = max_turns
        elapsed = time.time() - self.start_time
        
        message = f"📊 Turn {turn}/{max_turns}: {action} [{elapsed:.1f}s]"
        await self._notify(message)
    
    async def update_streaming(self, chunk_count: int = 0) -> None:
        """Update progress during streaming response."""
        elapsed = time.time() - self.start_time
        message = f"🎯 Streaming response"
        if chunk_count > 0:
            message += f" ({chunk_count} chunks)"
        message += f" [{elapsed:.1f}s]"
        await self._notify(message)
    
    async def update_custom_status(self, status: str, icon: str = "⏳") -> None:
        """Update with a custom status message."""
        elapsed = time.time() - self.start_time
        message = f"{icon} {status} [{elapsed:.1f}s]"
        await self._notify(message)
    
    def _get_tool_icon(self, tool_name: str) -> str:
        """Get appropriate icon for tool name."""
        # Try exact match first
        if tool_name in self.TOOL_ICONS:
            return self.TOOL_ICONS[tool_name]
        
        # Try partial matches for complex tool names
        for key, icon in self.TOOL_ICONS.items():
            if key in tool_name.lower():
                return icon
        
        return self.TOOL_ICONS["default"]
    
    async def _notify(self, message: str) -> None:
        """Send progress update to callback if available."""
        if self.progress_callback:
            try:
                if asyncio.iscoroutinefunction(self.progress_callback):
                    await self.progress_callback(message)
                else:
                    self.progress_callback(message)
            except Exception:
                # Ignore callback failures to avoid breaking main processing
                pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        elapsed = time.time() - self.start_time
        return {
            "total_time": elapsed,
            "current_phase": self.current_phase.value if self.current_phase else None,
            "current_turn": self.current_turn,
            "max_turns": self.max_turns,
            "tools_executed": self.tools_executed
        }


class SimpleProgressTracker:
    """Simplified progress tracker for basic scenarios."""
    
    def __init__(self, status_callback: Optional[Callable[[str], None]] = None):
        self.start_time = time.time()
        self.status_callback = status_callback
    
    async def update(self, message: str, icon: str = "⏳") -> None:
        """Update progress with simple message."""
        elapsed = time.time() - self.start_time
        status = f"{icon} {message} [{elapsed:.1f}s]"
        
        if self.status_callback:
            try:
                if asyncio.iscoroutinefunction(self.status_callback):
                    await self.status_callback(status)
                else:
                    self.status_callback(status)
            except Exception:
                # Ignore callback failures
                pass


def create_progress_tracker(callback: Optional[Callable[[str], None]] = None,
                          simple: bool = False) -> ProgressTracker:
    """
    Factory function to create appropriate progress tracker.
    
    Args:
        callback: Progress callback function
        simple: If True, create simplified tracker
        
    Returns:
        Progress tracker instance
    """
    if simple:
        return SimpleProgressTracker(callback)
    else:
        return ProgressTracker(callback)