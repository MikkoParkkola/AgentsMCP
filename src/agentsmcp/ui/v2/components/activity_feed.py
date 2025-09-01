"""
Real-time activity feed component for the TUI.

Displays a scrolling feed of agent activities, task events, and system notifications
with filtering, search, and persistence capabilities.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from ...monitoring.agent_tracker import get_agent_tracker, AgentStatus, TaskPhase, AgentActivity, TaskInfo
from ...monitoring.metrics_collector import get_metrics_collector
from .status_indicators import (
    StatusIndicator, AlertDisplay, get_status_display, get_task_phase_display, format_alert
)

logger = logging.getLogger(__name__)


class ActivityType(Enum):
    """Types of activities that can be displayed in the feed."""
    AGENT_STATUS_CHANGE = "agent_status_change"
    TASK_START = "task_start"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    QUALITY_GATE = "quality_gate"
    SELF_IMPROVEMENT = "self_improvement"
    SYSTEM_ALERT = "system_alert"
    USER_ACTION = "user_action"
    ERROR = "error"


@dataclass
class ActivityEvent:
    """Individual activity event in the feed."""
    timestamp: float
    activity_type: ActivityType
    agent_id: Optional[str]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    level: str = "info"  # info, warning, error, success
    task_id: Optional[str] = None
    
    def __post_init__(self):
        """Ensure timestamp is set."""
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class ActivityFeedConfig:
    """Configuration for the activity feed."""
    max_events: int = 1000
    auto_scroll: bool = True
    show_timestamps: bool = True
    show_agent_ids: bool = True
    show_task_ids: bool = False
    filter_levels: Set[str] = field(default_factory=lambda: {"info", "warning", "error", "success"})
    filter_types: Set[ActivityType] = field(default_factory=set)
    update_interval: float = 0.1
    compact_mode: bool = False
    max_message_length: int = 80


class ActivityFeed:
    """
    Real-time activity feed showing system events and agent activities.
    
    Provides a scrolling log of all system activities with filtering,
    search, and persistence capabilities.
    """
    
    def __init__(self, config: ActivityFeedConfig = None):
        """Initialize activity feed."""
        self.config = config or ActivityFeedConfig()
        
        # Event storage
        self._events: deque[ActivityEvent] = deque(maxlen=self.config.max_events)
        self._filtered_events: List[ActivityEvent] = []
        
        # Display state
        self._scroll_position = 0
        self._cached_display = ""
        self._last_update = 0
        self._search_query = ""
        self._highlighted_terms: Set[str] = set()
        
        # Components
        self.status_indicator = StatusIndicator()
        self.alert_display = AlertDisplay()
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._event_callbacks: List[Callable[[ActivityEvent], None]] = []
        self._display_callbacks: List[Callable[[str], None]] = []
        
        # Set up monitoring
        self.agent_tracker = get_agent_tracker()
        self.metrics_collector = get_metrics_collector()
        
        # Register for agent events
        self.agent_tracker.add_status_listener(self._on_agent_status_change)
        self.agent_tracker.add_task_listener(self._on_agent_task_change)
        
        logger.debug("ActivityFeed initialized")
    
    def start(self):
        """Start the activity feed."""
        if self._running:
            return
        
        self._running = True
        
        # Start update task
        try:
            loop = asyncio.get_event_loop()
            self._update_task = loop.create_task(self._update_loop())
            logger.info("ActivityFeed started")
        except RuntimeError:
            logger.warning("No event loop available, activity feed will be manual")
        
        # Add initial system event
        self.add_event(ActivityEvent(
            timestamp=time.time(),
            activity_type=ActivityType.SYSTEM_ALERT,
            agent_id=None,
            message="Activity feed started",
            level="info"
        ))
    
    async def stop(self):
        """Stop the activity feed."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
        
        # Clean up listeners
        self.agent_tracker.remove_status_listener(self._on_agent_status_change)
        self.agent_tracker.remove_task_listener(self._on_agent_task_change)
        
        logger.info("ActivityFeed stopped")
    
    def render(self, width: int = 80, height: int = 20) -> str:
        """
        Render the activity feed.
        
        Args:
            width: Available width for display
            height: Available height for display (number of event rows)
            
        Returns:
            Formatted activity feed string
        """
        with self._lock:
            # Check if we need to update
            current_time = time.time()
            if current_time - self._last_update > self.config.update_interval:
                self._update_display(width, height)
                self._last_update = current_time
            
            return self._cached_display
    
    def add_event(self, event: ActivityEvent):
        """Add a new event to the feed."""
        with self._lock:
            self._events.append(event)
            
            # Update filtered events
            if self._passes_filters(event):
                self._filtered_events.append(event)
                
                # Maintain max size for filtered events
                if len(self._filtered_events) > self.config.max_events:
                    self._filtered_events = self._filtered_events[-self.config.max_events//2:]
            
            # Auto-scroll to bottom if enabled
            if self.config.auto_scroll:
                self._scroll_position = max(0, len(self._filtered_events) - 1)
            
            # Force display update on next render
            self._last_update = 0
            
            # Notify callbacks
            for callback in self._event_callbacks[:]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Event callback error: {e}")
    
    def scroll_up(self, lines: int = 1):
        """Scroll the feed up."""
        with self._lock:
            self._scroll_position = max(0, self._scroll_position - lines)
            self._last_update = 0  # Force update
    
    def scroll_down(self, lines: int = 1):
        """Scroll the feed down."""
        with self._lock:
            max_scroll = max(0, len(self._filtered_events) - 1)
            self._scroll_position = min(max_scroll, self._scroll_position + lines)
            self._last_update = 0  # Force update
    
    def scroll_to_bottom(self):
        """Scroll to the bottom of the feed."""
        with self._lock:
            self._scroll_position = max(0, len(self._filtered_events) - 1)
            self._last_update = 0  # Force update
    
    def set_search(self, query: str):
        """Set search query for filtering events."""
        with self._lock:
            self._search_query = query.lower()
            self._highlighted_terms = set(query.lower().split()) if query else set()
            self._update_filtered_events()
            self._last_update = 0  # Force update
    
    def clear_search(self):
        """Clear search query."""
        self.set_search("")
    
    def set_filter_levels(self, levels: Set[str]):
        """Set filter levels."""
        with self._lock:
            self.config.filter_levels = levels
            self._update_filtered_events()
            self._last_update = 0  # Force update
    
    def set_filter_types(self, types: Set[ActivityType]):
        """Set filter types."""
        with self._lock:
            self.config.filter_types = types
            self._update_filtered_events()
            self._last_update = 0  # Force update
    
    def clear_events(self):
        """Clear all events from the feed."""
        with self._lock:
            self._events.clear()
            self._filtered_events.clear()
            self._scroll_position = 0
            self._last_update = 0  # Force update
    
    def get_event_count(self) -> int:
        """Get total number of events."""
        with self._lock:
            return len(self._events)
    
    def get_filtered_event_count(self) -> int:
        """Get number of filtered events."""
        with self._lock:
            return len(self._filtered_events)
    
    def add_event_callback(self, callback: Callable[[ActivityEvent], None]):
        """Add callback for new events."""
        self._event_callbacks.append(callback)
    
    def add_display_callback(self, callback: Callable[[str], None]):
        """Add callback for display updates."""
        self._display_callbacks.append(callback)
    
    def remove_event_callback(self, callback: Callable[[ActivityEvent], None]):
        """Remove event callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)
    
    def remove_display_callback(self, callback: Callable[[str], None]):
        """Remove display callback."""
        if callback in self._display_callbacks:
            self._display_callbacks.remove(callback)
    
    async def _update_loop(self):
        """Background update loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.update_interval)
                # Force display update on next render
                with self._lock:
                    self._last_update = 0
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in activity feed update loop: {e}")
    
    def _update_display(self, width: int, height: int):
        """Update the cached display."""
        try:
            with self._lock:
                if not self._filtered_events:
                    self._cached_display = "[dim]No activity to display[/dim]"
                    return
                
                # Calculate display window
                start_idx = max(0, self._scroll_position - height + 1)
                end_idx = min(len(self._filtered_events), start_idx + height)
                
                display_events = self._filtered_events[start_idx:end_idx]
                
                # Format events
                lines = []
                for event in display_events:
                    line = self._format_event(event, width)
                    lines.append(line)
                
                # Add scroll indicator if needed
                if len(self._filtered_events) > height:
                    total_events = len(self._filtered_events)
                    current_position = self._scroll_position + 1
                    scroll_info = f"[dim]({current_position}/{total_events})[/dim]"
                    
                    if lines:
                        # Add to last line
                        lines[-1] += f" {scroll_info}"
                    else:
                        lines.append(scroll_info)
                
                self._cached_display = "\n".join(lines)
                
                # Notify display callbacks
                for callback in self._display_callbacks[:]:
                    try:
                        callback(self._cached_display)
                    except Exception as e:
                        logger.error(f"Display callback error: {e}")
                        
        except Exception as e:
            logger.error(f"Error updating activity feed display: {e}")
            self._cached_display = f"[red]Error updating activity feed: {str(e)}[/red]"
    
    def _format_event(self, event: ActivityEvent, width: int) -> str:
        """Format a single event for display."""
        parts = []
        
        # Timestamp
        if self.config.show_timestamps:
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
            parts.append(f"[dim]{timestamp_str}[/dim]")
        
        # Event type indicator
        type_indicator = self._get_type_indicator(event.activity_type)
        parts.append(type_indicator)
        
        # Agent ID
        if self.config.show_agent_ids and event.agent_id:
            agent_str = event.agent_id[:8] + "..." if len(event.agent_id) > 8 else event.agent_id
            parts.append(f"[cyan]{agent_str}[/cyan]")
        
        # Task ID
        if self.config.show_task_ids and event.task_id:
            task_str = event.task_id[:6] + "..." if len(event.task_id) > 6 else event.task_id
            parts.append(f"[dim]{task_str}[/dim]")
        
        # Message
        message = event.message
        if len(message) > self.config.max_message_length:
            message = message[:self.config.max_message_length-3] + "..."
        
        # Highlight search terms
        if self._highlighted_terms:
            for term in self._highlighted_terms:
                if term in message.lower():
                    # Simple highlighting - could be improved
                    message = message.replace(term, f"[yellow]{term}[/yellow]")
        
        # Apply level color
        level_color = self._get_level_color(event.level)
        if level_color:
            message = f"[{level_color}]{message}[/{level_color}]"
        
        parts.append(message)
        
        # Join parts and truncate to width
        line = " ".join(parts)
        return line[:width] if width > 0 else line
    
    def _get_type_indicator(self, activity_type: ActivityType) -> str:
        """Get indicator for activity type."""
        indicators = {
            ActivityType.AGENT_STATUS_CHANGE: "🔄",
            ActivityType.TASK_START: "▶",
            ActivityType.TASK_PROGRESS: "⏳",
            ActivityType.TASK_COMPLETE: "✅",
            ActivityType.QUALITY_GATE: "🛡",
            ActivityType.SELF_IMPROVEMENT: "🔧",
            ActivityType.SYSTEM_ALERT: "🔔",
            ActivityType.USER_ACTION: "👤",
            ActivityType.ERROR: "❌",
        }
        
        unicode_icon = indicators.get(activity_type, "•")
        
        # Fallback to ASCII if needed
        if not self.status_indicator.config.use_unicode:
            ascii_indicators = {
                ActivityType.AGENT_STATUS_CHANGE: "~",
                ActivityType.TASK_START: ">",
                ActivityType.TASK_PROGRESS: ".",
                ActivityType.TASK_COMPLETE: "+",
                ActivityType.QUALITY_GATE: "#",
                ActivityType.SELF_IMPROVEMENT: "=",
                ActivityType.SYSTEM_ALERT: "!",
                ActivityType.USER_ACTION: "@",
                ActivityType.ERROR: "X",
            }
            return ascii_indicators.get(activity_type, "*")
        
        return unicode_icon
    
    def _get_level_color(self, level: str) -> Optional[str]:
        """Get color for event level."""
        colors = {
            "info": None,  # Default color
            "success": "green",
            "warning": "yellow", 
            "error": "red",
            "critical": "bright_red"
        }
        return colors.get(level)
    
    def _passes_filters(self, event: ActivityEvent) -> bool:
        """Check if event passes current filters."""
        # Level filter
        if event.level not in self.config.filter_levels:
            return False
        
        # Type filter
        if self.config.filter_types and event.activity_type not in self.config.filter_types:
            return False
        
        # Search filter
        if self._search_query:
            searchable_text = f"{event.message} {event.agent_id or ''} {event.task_id or ''}".lower()
            if self._search_query not in searchable_text:
                return False
        
        return True
    
    def _update_filtered_events(self):
        """Update the list of filtered events."""
        self._filtered_events = [event for event in self._events if self._passes_filters(event)]
        
        # Adjust scroll position if needed
        if self._scroll_position >= len(self._filtered_events):
            self._scroll_position = max(0, len(self._filtered_events) - 1)
    
    def _on_agent_status_change(self, agent_id: str, old_status: AgentStatus, new_status: AgentStatus):
        """Handle agent status change events."""
        if old_status == new_status:
            return
        
        # Determine event level
        level = "info"
        if new_status == AgentStatus.ERROR:
            level = "error"
        elif new_status in [AgentStatus.WORKING, AgentStatus.COMPLETING]:
            level = "success"
        elif new_status in [AgentStatus.WAITING_RESOURCE, AgentStatus.WAITING_INPUT]:
            level = "warning"
        
        # Create event
        event = ActivityEvent(
            timestamp=time.time(),
            activity_type=ActivityType.AGENT_STATUS_CHANGE,
            agent_id=agent_id,
            message=f"Status: {old_status.value} → {new_status.value}",
            level=level,
            details={
                'old_status': old_status.value,
                'new_status': new_status.value
            }
        )
        
        self.add_event(event)
    
    def _on_agent_task_change(self, agent_id: str, old_task: Optional[TaskInfo], new_task: Optional[TaskInfo]):
        """Handle agent task change events."""
        current_time = time.time()
        
        if old_task and not new_task:
            # Task completed
            event = ActivityEvent(
                timestamp=current_time,
                activity_type=ActivityType.TASK_COMPLETE,
                agent_id=agent_id,
                message=f"Completed: {old_task.description[:40]}",
                level="success",
                task_id=old_task.task_id,
                details={
                    'duration': old_task.elapsed_time,
                    'phase': old_task.phase.value
                }
            )
            self.add_event(event)
        
        elif new_task and not old_task:
            # Task started
            event = ActivityEvent(
                timestamp=current_time,
                activity_type=ActivityType.TASK_START,
                agent_id=agent_id,
                message=f"Started: {new_task.description[:40]}",
                level="info",
                task_id=new_task.task_id
            )
            self.add_event(event)
    
    def add_quality_gate_event(self, gate_type: str, passed: bool, details: Optional[Dict] = None):
        """Add a quality gate event."""
        event = ActivityEvent(
            timestamp=time.time(),
            activity_type=ActivityType.QUALITY_GATE,
            agent_id=None,
            message=f"Quality gate {gate_type}: {'PASS' if passed else 'FAIL'}",
            level="success" if passed else "warning",
            details=details or {}
        )
        self.add_event(event)
    
    def add_self_improvement_event(self, action: str, details: Optional[Dict] = None):
        """Add a self-improvement event."""
        event = ActivityEvent(
            timestamp=time.time(),
            activity_type=ActivityType.SELF_IMPROVEMENT,
            agent_id=None,
            message=f"Self-improvement: {action}",
            level="info",
            details=details or {}
        )
        self.add_event(event)
    
    def add_user_action_event(self, action: str, details: Optional[Dict] = None):
        """Add a user action event."""
        event = ActivityEvent(
            timestamp=time.time(),
            activity_type=ActivityType.USER_ACTION,
            agent_id=None,
            message=f"User: {action}",
            level="info",
            details=details or {}
        )
        self.add_event(event)
    
    def add_error_event(self, error_message: str, agent_id: Optional[str] = None,
                       details: Optional[Dict] = None):
        """Add an error event."""
        event = ActivityEvent(
            timestamp=time.time(),
            activity_type=ActivityType.ERROR,
            agent_id=agent_id,
            message=f"Error: {error_message}",
            level="error",
            details=details or {}
        )
        self.add_event(event)