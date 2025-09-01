"""
Color-coded status indicators and visual components for the TUI.

Provides reusable status indicators, progress bars, and visual elements
with consistent color schemes and accessibility support.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ...monitoring.agent_tracker import AgentStatus, TaskPhase


class StatusColor(Enum):
    """Status color codes for consistent theming."""
    IDLE = "bright_cyan"
    ACTIVE = "bright_green"
    THINKING = "bright_yellow"
    WORKING = "bright_blue"
    WAITING = "yellow"
    ERROR = "bright_red"
    SUCCESS = "green"
    WARNING = "bright_yellow"
    INFO = "cyan"
    DISABLED = "bright_black"


class ProgressBarStyle(Enum):
    """Progress bar visual styles."""
    SOLID = "solid"
    BLOCKS = "blocks"
    DOTS = "dots"
    ARROWS = "arrows"
    SPINNER = "spinner"


@dataclass
class StatusIndicatorConfig:
    """Configuration for status indicators."""
    show_icons: bool = True
    show_colors: bool = True
    use_unicode: bool = True
    animation_enabled: bool = True
    update_interval_ms: int = 250


class StatusIndicator:
    """
    Individual status indicator with color, icon, and animation support.
    
    Provides a consistent way to display status information with
    appropriate colors and visual cues.
    """
    
    # Status icons (Unicode and ASCII fallbacks)
    ICONS = {
        AgentStatus.IDLE: ("⏸", "[]"),
        AgentStatus.STARTING: ("🚀", ">>"),
        AgentStatus.THINKING: ("🤔", "??"),
        AgentStatus.WORKING: ("⚙", "##"),
        AgentStatus.WAITING_RESOURCE: ("⏳", ".."),
        AgentStatus.WAITING_INPUT: ("⌨", ">>"),
        AgentStatus.COMPLETING: ("✅", "OK"),
        AgentStatus.ERROR: ("❌", "XX"),
        AgentStatus.STOPPING: ("🛑", "ST"),
        AgentStatus.STOPPED: ("⏹", "[]"),
    }
    
    # Animated spinner frames
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    SPINNER_FRAMES_ASCII = ["|", "/", "-", "\\"]
    
    def __init__(self, config: StatusIndicatorConfig = None):
        """Initialize status indicator."""
        self.config = config or StatusIndicatorConfig()
        self._spinner_index = 0
        self._last_update = 0
        
    def get_status_display(self, status: AgentStatus, animated: bool = False) -> str:
        """
        Get display string for a status.
        
        Args:
            status: Agent status to display
            animated: Whether to show animated spinner for active states
            
        Returns:
            Formatted status string with color and icon
        """
        # Get icon
        if self.config.show_icons:
            if animated and status in [AgentStatus.THINKING, AgentStatus.WORKING]:
                icon = self._get_spinner_char()
            else:
                unicode_icon, ascii_icon = self.ICONS.get(status, ("?", "?"))
                icon = unicode_icon if self.config.use_unicode else ascii_icon
        else:
            icon = ""
        
        # Get color
        color = self._get_status_color(status)
        
        # Format display string
        if self.config.show_colors:
            return f"[{color}]{icon}[/{color}]" if icon else f"[{color}]●[/{color}]"
        else:
            return icon or "●"
    
    def get_task_phase_display(self, phase: TaskPhase, animated: bool = False) -> str:
        """Get display string for a task phase."""
        phase_icons = {
            TaskPhase.QUEUED: ("⏳", "Q"),
            TaskPhase.ANALYZING: ("🔍", "A"),
            TaskPhase.PLANNING: ("📋", "P"),
            TaskPhase.EXECUTING: ("⚡", "E"),
            TaskPhase.VALIDATING: ("✓", "V"),
            TaskPhase.COMPLETING: ("🏁", "C"),
            TaskPhase.COMPLETED: ("✅", "✓"),
            TaskPhase.FAILED: ("❌", "X"),
        }
        
        if self.config.show_icons:
            if animated and phase == TaskPhase.EXECUTING:
                icon = self._get_spinner_char()
            else:
                unicode_icon, ascii_icon = phase_icons.get(phase, ("?", "?"))
                icon = unicode_icon if self.config.use_unicode else ascii_icon
        else:
            icon = ""
        
        color = self._get_phase_color(phase)
        
        if self.config.show_colors:
            return f"[{color}]{icon}[/{color}]" if icon else f"[{color}]●[/{color}]"
        else:
            return icon or "●"
    
    def _get_status_color(self, status: AgentStatus) -> str:
        """Get color for agent status."""
        color_mapping = {
            AgentStatus.IDLE: StatusColor.IDLE.value,
            AgentStatus.STARTING: StatusColor.INFO.value,
            AgentStatus.THINKING: StatusColor.THINKING.value,
            AgentStatus.WORKING: StatusColor.ACTIVE.value,
            AgentStatus.WAITING_RESOURCE: StatusColor.WAITING.value,
            AgentStatus.WAITING_INPUT: StatusColor.WAITING.value,
            AgentStatus.COMPLETING: StatusColor.SUCCESS.value,
            AgentStatus.ERROR: StatusColor.ERROR.value,
            AgentStatus.STOPPING: StatusColor.WARNING.value,
            AgentStatus.STOPPED: StatusColor.DISABLED.value,
        }
        return color_mapping.get(status, StatusColor.INFO.value)
    
    def _get_phase_color(self, phase: TaskPhase) -> str:
        """Get color for task phase."""
        color_mapping = {
            TaskPhase.QUEUED: StatusColor.WAITING.value,
            TaskPhase.ANALYZING: StatusColor.THINKING.value,
            TaskPhase.PLANNING: StatusColor.THINKING.value,
            TaskPhase.EXECUTING: StatusColor.ACTIVE.value,
            TaskPhase.VALIDATING: StatusColor.INFO.value,
            TaskPhase.COMPLETING: StatusColor.SUCCESS.value,
            TaskPhase.COMPLETED: StatusColor.SUCCESS.value,
            TaskPhase.FAILED: StatusColor.ERROR.value,
        }
        return color_mapping.get(phase, StatusColor.INFO.value)
    
    def _get_spinner_char(self) -> str:
        """Get current spinner character."""
        current_time = time.time() * 1000  # milliseconds
        if current_time - self._last_update > self.config.update_interval_ms:
            self._spinner_index = (self._spinner_index + 1) % len(self.SPINNER_FRAMES)
            self._last_update = current_time
        
        if self.config.use_unicode:
            return self.SPINNER_FRAMES[self._spinner_index]
        else:
            return self.SPINNER_FRAMES_ASCII[self._spinner_index % len(self.SPINNER_FRAMES_ASCII)]


class ProgressBar:
    """
    Configurable progress bar component.
    
    Supports various styles, colors, and customizable appearance
    for displaying task progress and completion status.
    """
    
    def __init__(self, width: int = 20, style: ProgressBarStyle = ProgressBarStyle.SOLID,
                 show_percentage: bool = True, show_time: bool = False):
        """
        Initialize progress bar.
        
        Args:
            width: Width of the progress bar in characters
            style: Visual style of the progress bar
            show_percentage: Whether to show percentage text
            show_time: Whether to show elapsed/remaining time
        """
        self.width = width
        self.style = style
        self.show_percentage = show_percentage
        self.show_time = show_time
        
        # Style characters
        self.style_chars = {
            ProgressBarStyle.SOLID: ("█", "░"),
            ProgressBarStyle.BLOCKS: ("■", "□"),
            ProgressBarStyle.DOTS: ("●", "○"),
            ProgressBarStyle.ARROWS: ("►", "─"),
        }
    
    def render(self, progress: float, estimated_duration: Optional[float] = None,
               elapsed_time: Optional[float] = None, color: str = "green") -> str:
        """
        Render progress bar.
        
        Args:
            progress: Progress as a percentage (0.0 to 100.0)
            estimated_duration: Total estimated duration in seconds
            elapsed_time: Elapsed time in seconds
            color: Color for the filled portion
            
        Returns:
            Formatted progress bar string
        """
        # Clamp progress
        progress = max(0.0, min(100.0, progress))
        
        # Calculate filled characters
        filled_chars = int((progress / 100.0) * self.width)
        empty_chars = self.width - filled_chars
        
        # Get style characters
        if self.style == ProgressBarStyle.SPINNER:
            # Special case for spinner style
            spinner = StatusIndicator()
            spinner_char = spinner._get_spinner_char()
            bar = f"[{color}]{spinner_char}[/{color}]" + "─" * (self.width - 1)
        else:
            filled_char, empty_char = self.style_chars.get(self.style, ("█", "░"))
            filled_part = f"[{color}]{filled_char * filled_chars}[/{color}]"
            empty_part = f"[dim]{empty_char * empty_chars}[/dim]"
            bar = filled_part + empty_part
        
        # Add brackets
        bar = f"[{bar}]"
        
        # Add percentage
        if self.show_percentage:
            bar += f" {progress:5.1f}%"
        
        # Add time information
        if self.show_time and elapsed_time is not None:
            time_info = []
            
            # Elapsed time
            elapsed_str = self._format_duration(elapsed_time)
            time_info.append(f"⏱ {elapsed_str}")
            
            # Remaining time estimate
            if estimated_duration and progress > 0:
                remaining = estimated_duration - elapsed_time
                if remaining > 0:
                    remaining_str = self._format_duration(remaining)
                    time_info.append(f"⏳ {remaining_str}")
            
            if time_info:
                bar += f" ({' / '.join(time_info)})"
        
        return bar
    
    def render_simple(self, progress: float, color: str = "green") -> str:
        """Render a simple progress bar without extras."""
        progress = max(0.0, min(100.0, progress))
        filled_chars = int((progress / 100.0) * self.width)
        empty_chars = self.width - filled_chars
        
        filled_char, empty_char = self.style_chars.get(self.style, ("█", "░"))
        filled_part = f"[{color}]{filled_char * filled_chars}[/{color}]"
        empty_part = f"[dim]{empty_char * empty_chars}[/dim]"
        
        return f"[{filled_part}{empty_part}]"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m{secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes}m"


class MetricDisplay:
    """
    Display component for metrics and statistics.
    
    Formats numerical metrics with appropriate units, colors,
    and trend indicators.
    """
    
    def __init__(self, decimal_places: int = 1, use_colors: bool = True):
        """
        Initialize metric display.
        
        Args:
            decimal_places: Number of decimal places to show
            use_colors: Whether to use colors for trend indicators
        """
        self.decimal_places = decimal_places
        self.use_colors = use_colors
    
    def format_metric(self, value: float, unit: str = "", 
                     previous_value: Optional[float] = None,
                     threshold_good: Optional[float] = None,
                     threshold_warning: Optional[float] = None) -> str:
        """
        Format a metric with optional trend and color coding.
        
        Args:
            value: Current metric value
            unit: Unit string (e.g., "ms", "%", "MB")
            previous_value: Previous value for trend calculation
            threshold_good: Threshold for good (green) values
            threshold_warning: Threshold for warning (yellow) values
            
        Returns:
            Formatted metric string
        """
        # Format the base value
        if abs(value) >= 1000000:
            formatted = f"{value / 1000000:.{self.decimal_places}f}M"
        elif abs(value) >= 1000:
            formatted = f"{value / 1000:.{self.decimal_places}f}k"
        else:
            formatted = f"{value:.{self.decimal_places}f}"
        
        # Add unit
        if unit:
            formatted += unit
        
        # Determine color based on thresholds
        color = None
        if self.use_colors:
            if threshold_good is not None and value <= threshold_good:
                color = "green"
            elif threshold_warning is not None and value <= threshold_warning:
                color = "yellow"
            elif threshold_warning is not None:
                color = "red"
        
        # Add trend indicator
        trend = ""
        if previous_value is not None and previous_value != value:
            if value > previous_value:
                trend = " ↑" if self.use_colors else " ^"
            else:
                trend = " ↓" if self.use_colors else " v"
        
        # Apply color and trend
        if color:
            return f"[{color}]{formatted}{trend}[/{color}]"
        else:
            return f"{formatted}{trend}"
    
    def format_percentage(self, value: float, 
                         previous_value: Optional[float] = None) -> str:
        """Format a percentage value."""
        return self.format_metric(
            value, "%", previous_value,
            threshold_good=80.0,  # Green below 80%
            threshold_warning=95.0  # Yellow below 95%, red above
        )
    
    def format_duration(self, seconds: float,
                       previous_value: Optional[float] = None) -> str:
        """Format a duration value."""
        if seconds < 0.001:
            return self.format_metric(seconds * 1000000, "μs", 
                                    previous_value * 1000000 if previous_value else None)
        elif seconds < 1:
            return self.format_metric(seconds * 1000, "ms",
                                    previous_value * 1000 if previous_value else None,
                                    threshold_good=100,  # Green below 100ms
                                    threshold_warning=1000)  # Yellow below 1s
        else:
            return self.format_metric(seconds, "s", previous_value,
                                    threshold_good=1.0,  # Green below 1s
                                    threshold_warning=5.0)  # Yellow below 5s
    
    def format_count(self, value: int, 
                    previous_value: Optional[int] = None) -> str:
        """Format a count value."""
        return self.format_metric(float(value), "", 
                                float(previous_value) if previous_value is not None else None)
    
    def format_rate(self, value: float, unit: str = "/s",
                   previous_value: Optional[float] = None) -> str:
        """Format a rate value."""
        return self.format_metric(value, unit, previous_value)


class AlertDisplay:
    """
    Display component for alerts and notifications.
    
    Provides consistent formatting for different alert levels
    with appropriate colors and icons.
    """
    
    ALERT_ICONS = {
        "info": ("ℹ", "i"),
        "warning": ("⚠", "!"),
        "error": ("❌", "X"),
        "critical": ("🔥", "!!"),
        "success": ("✅", "✓"),
    }
    
    ALERT_COLORS = {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "critical": "bright_red",
        "success": "green",
    }
    
    def __init__(self, use_unicode: bool = True, show_timestamp: bool = True):
        """
        Initialize alert display.
        
        Args:
            use_unicode: Whether to use Unicode icons
            show_timestamp: Whether to show timestamps
        """
        self.use_unicode = use_unicode
        self.show_timestamp = show_timestamp
    
    def format_alert(self, level: str, message: str, 
                    details: Optional[Dict[str, Any]] = None,
                    timestamp: Optional[float] = None) -> str:
        """
        Format an alert message.
        
        Args:
            level: Alert level (info, warning, error, critical, success)
            message: Alert message
            details: Optional details dictionary
            timestamp: Optional timestamp
            
        Returns:
            Formatted alert string
        """
        # Get icon and color
        unicode_icon, ascii_icon = self.ALERT_ICONS.get(level, ("•", "*"))
        icon = unicode_icon if self.use_unicode else ascii_icon
        color = self.ALERT_COLORS.get(level, "white")
        
        # Format base message
        formatted = f"[{color}]{icon} {message}[/{color}]"
        
        # Add timestamp
        if self.show_timestamp and timestamp:
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            formatted = f"[dim]{time_str}[/dim] {formatted}"
        
        # Add details
        if details:
            detail_parts = []
            for key, value in details.items():
                if key in ['current', 'threshold']:
                    detail_parts.append(f"{key}: {value}")
                else:
                    detail_parts.append(f"{key}: {value}")
            
            if detail_parts:
                formatted += f" [dim]({', '.join(detail_parts)})[/dim]"
        
        return formatted


# Global instances for easy access
_status_indicator = StatusIndicator()
_metric_display = MetricDisplay()
_alert_display = AlertDisplay()


def get_status_display(status: AgentStatus, animated: bool = False) -> str:
    """Get status display using global indicator."""
    return _status_indicator.get_status_display(status, animated)


def get_task_phase_display(phase: TaskPhase, animated: bool = False) -> str:
    """Get task phase display using global indicator."""
    return _status_indicator.get_task_phase_display(phase, animated)


def format_metric(value: float, unit: str = "", **kwargs) -> str:
    """Format metric using global display."""
    return _metric_display.format_metric(value, unit, **kwargs)


def format_alert(level: str, message: str, **kwargs) -> str:
    """Format alert using global display."""
    return _alert_display.format_alert(level, message, **kwargs)