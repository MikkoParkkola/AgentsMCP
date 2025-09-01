"""
Live agent status panel component for the TUI.

Displays real-time status of all active agents including their current tasks,
progress, queue sizes, and resource usage with live updates.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict

from ...monitoring.agent_tracker import get_agent_tracker, AgentStatus, AgentActivity, TaskPhase
from ...monitoring.metrics_collector import get_metrics_collector
from .status_indicators import (
    StatusIndicator, ProgressBar, MetricDisplay, ProgressBarStyle,
    get_status_display, get_task_phase_display, format_metric
)

logger = logging.getLogger(__name__)


@dataclass
class AgentStatusPanelConfig:
    """Configuration for agent status panel."""
    max_agents_displayed: int = 20
    show_idle_agents: bool = True
    show_resource_usage: bool = True
    show_task_progress: bool = True
    show_queue_size: bool = True
    update_interval: float = 0.5
    auto_sort: bool = True
    sort_by: str = "activity"  # "activity", "name", "type", "status"
    compact_mode: bool = False


class AgentStatusPanel:
    """
    Live agent status panel showing real-time agent information.
    
    Displays a sortable, filterable list of all agents with their current
    status, tasks, progress, and resource usage.
    """
    
    def __init__(self, config: AgentStatusPanelConfig = None):
        """Initialize agent status panel."""
        self.config = config or AgentStatusPanelConfig()
        
        # Components
        self.agent_tracker = get_agent_tracker()
        self.metrics_collector = get_metrics_collector()
        self.status_indicator = StatusIndicator()
        self.progress_bar = ProgressBar(width=15, style=ProgressBarStyle.SOLID)
        self.metric_display = MetricDisplay()
        
        # State
        self._last_update = 0
        self._cached_display = ""
        self._filter_status: Optional[AgentStatus] = None
        self._filter_type: Optional[str] = None
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._display_callbacks: List[Callable[[str], None]] = []
        
        logger.debug("AgentStatusPanel initialized")
    
    def start(self):
        """Start the agent status panel updates."""
        if self._running:
            return
        
        self._running = True
        
        # Start update task
        try:
            loop = asyncio.get_event_loop()
            self._update_task = loop.create_task(self._update_loop())
            logger.info("AgentStatusPanel started")
        except RuntimeError:
            logger.warning("No event loop available, status panel will be manual")
    
    async def stop(self):
        """Stop the agent status panel."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
        
        logger.info("AgentStatusPanel stopped")
    
    def render(self, width: int = 80, height: int = 10) -> str:
        """
        Render the agent status panel.
        
        Args:
            width: Available width for display
            height: Available height for display (number of agent rows)
            
        Returns:
            Formatted agent status panel string
        """
        with self._lock:
            # Check if we need to update
            current_time = time.time()
            if current_time - self._last_update > self.config.update_interval:
                self._update_display(width, height)
                self._last_update = current_time
            
            return self._cached_display
    
    def set_filter(self, status: Optional[AgentStatus] = None, 
                   agent_type: Optional[str] = None):
        """Set filters for agent display."""
        with self._lock:
            self._filter_status = status
            self._filter_type = agent_type
            # Force update on next render
            self._last_update = 0
    
    def clear_filter(self):
        """Clear all filters."""
        self.set_filter(None, None)
    
    def add_display_callback(self, callback: Callable[[str], None]):
        """Add callback for display updates."""
        self._display_callbacks.append(callback)
    
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
                logger.error(f"Error in agent status panel update loop: {e}")
    
    def _update_display(self, width: int, height: int):
        """Update the cached display."""
        try:
            # Get agent activities
            activities = self.agent_tracker.get_all_activities()
            
            # Apply filters
            filtered_activities = self._apply_filters(activities)
            
            # Sort agents
            sorted_agents = self._sort_agents(filtered_activities)
            
            # Limit to display count
            display_agents = sorted_agents[:min(height, self.config.max_agents_displayed)]
            
            # Generate display
            if self.config.compact_mode:
                display = self._render_compact(display_agents, width)
            else:
                display = self._render_detailed(display_agents, width)
            
            self._cached_display = display
            
            # Notify callbacks
            for callback in self._display_callbacks[:]:
                try:
                    callback(display)
                except Exception as e:
                    logger.error(f"Display callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating agent status display: {e}")
            self._cached_display = f"[red]Error updating agent status: {str(e)}[/red]"
    
    def _apply_filters(self, activities: Dict[str, AgentActivity]) -> Dict[str, AgentActivity]:
        """Apply filters to agent activities."""
        filtered = {}
        
        for agent_id, activity in activities.items():
            # Status filter
            if self._filter_status and activity.status != self._filter_status:
                continue
            
            # Type filter
            if self._filter_type and activity.agent_type != self._filter_type:
                continue
            
            # Show idle filter
            if not self.config.show_idle_agents and activity.status == AgentStatus.IDLE:
                continue
            
            filtered[agent_id] = activity
        
        return filtered
    
    def _sort_agents(self, activities: Dict[str, AgentActivity]) -> List[tuple[str, AgentActivity]]:
        """Sort agents based on configuration."""
        agent_list = list(activities.items())
        
        if not self.config.auto_sort:
            return agent_list
        
        if self.config.sort_by == "activity":
            # Sort by activity level (working > thinking > waiting > idle)
            activity_priority = {
                AgentStatus.WORKING: 5,
                AgentStatus.THINKING: 4,
                AgentStatus.WAITING_RESOURCE: 3,
                AgentStatus.WAITING_INPUT: 3,
                AgentStatus.STARTING: 2,
                AgentStatus.COMPLETING: 2,
                AgentStatus.IDLE: 1,
                AgentStatus.ERROR: 6,  # Errors at top
                AgentStatus.STOPPING: 0,
                AgentStatus.STOPPED: 0,
            }
            agent_list.sort(key=lambda x: (
                -activity_priority.get(x[1].status, 0),  # Status priority (descending)
                -x[1].last_update,  # Most recently updated first
                x[0]  # Agent ID alphabetically
            ))
        
        elif self.config.sort_by == "name":
            agent_list.sort(key=lambda x: x[0])  # Sort by agent ID
        
        elif self.config.sort_by == "type":
            agent_list.sort(key=lambda x: (x[1].agent_type, x[0]))
        
        elif self.config.sort_by == "status":
            agent_list.sort(key=lambda x: (x[1].status.value, x[0]))
        
        return agent_list
    
    def _render_detailed(self, agents: List[tuple[str, AgentActivity]], width: int) -> str:
        """Render detailed agent status display."""
        if not agents:
            return "[dim]No agents to display[/dim]"
        
        lines = []
        
        # Header
        header = self._format_header(width)
        if header:
            lines.append(header)
            lines.append("─" * width)
        
        # Agent rows
        for agent_id, activity in agents:
            row = self._format_agent_row(agent_id, activity, width)
            lines.append(row)
        
        # Footer with summary
        if len(agents) < len(self.agent_tracker.get_all_activities()):
            total_count = len(self.agent_tracker.get_all_activities())
            lines.append("─" * width)
            lines.append(f"[dim]Showing {len(agents)} of {total_count} agents[/dim]")
        
        return "\n".join(lines)
    
    def _render_compact(self, agents: List[tuple[str, AgentActivity]], width: int) -> str:
        """Render compact agent status display."""
        if not agents:
            return "[dim]No agents[/dim]"
        
        lines = []
        
        # Group by status
        by_status = defaultdict(list)
        for agent_id, activity in agents:
            by_status[activity.status].append((agent_id, activity))
        
        # Show each status group
        for status in [AgentStatus.WORKING, AgentStatus.THINKING, AgentStatus.WAITING_RESOURCE, 
                      AgentStatus.WAITING_INPUT, AgentStatus.IDLE, AgentStatus.ERROR]:
            if status not in by_status:
                continue
            
            status_agents = by_status[status]
            status_display = get_status_display(status, animated=True)
            count = len(status_agents)
            
            # Show count and some agent names
            agent_names = [agent_id.split('_')[0] for agent_id, _ in status_agents[:3]]
            if len(status_agents) > 3:
                agent_names.append(f"+{len(status_agents) - 3}")
            
            line = f"{status_display} {count:2d}: {', '.join(agent_names)}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_header(self, width: int) -> str:
        """Format the header row."""
        if self.config.compact_mode:
            return ""
        
        # Calculate column widths
        id_width = 12
        type_width = 10
        status_width = 8
        task_width = max(20, width - id_width - type_width - status_width - 10)
        
        header = (
            f"{'Agent ID':<{id_width}} "
            f"{'Type':<{type_width}} "
            f"{'Status':<{status_width}} "
            f"{'Task':<{task_width}}"
        )
        
        return header[:width]
    
    def _format_agent_row(self, agent_id: str, activity: AgentActivity, width: int) -> str:
        """Format a single agent row."""
        # Calculate column widths
        id_width = 12
        type_width = 10
        status_width = 8
        remaining = max(20, width - id_width - type_width - status_width - 10)
        
        # Format agent ID (truncated)
        agent_display = agent_id[:id_width-1] if len(agent_id) > id_width else agent_id
        
        # Format type (truncated)
        type_display = activity.agent_type[:type_width-1] if len(activity.agent_type) > type_width else activity.agent_type
        
        # Format status with indicator
        status_display = get_status_display(activity.status, animated=True)
        
        # Format task information
        task_info = self._format_task_info(activity, remaining)
        
        row = (
            f"{agent_display:<{id_width}} "
            f"{type_display:<{type_width}} "
            f"{status_display:<{status_width}} "
            f"{task_info}"
        )
        
        return row[:width]
    
    def _format_task_info(self, activity: AgentActivity, width: int) -> str:
        """Format task information for an agent."""
        if not activity.current_task:
            # Show queue info if no current task
            if activity.task_queue_size > 0:
                return f"[dim]Queued: {activity.task_queue_size}[/dim]"
            else:
                return "[dim]No task[/dim]"
        
        task = activity.current_task
        
        # Task phase indicator
        phase_display = get_task_phase_display(task.phase, animated=True)
        
        # Progress bar if available
        progress_part = ""
        if task.progress_percentage > 0:
            progress_bar = self.progress_bar.render_simple(task.progress_percentage)
            progress_part = f" {progress_bar}"
        
        # Current step
        step_part = ""
        if task.current_step:
            step_text = task.current_step[:15] + "..." if len(task.current_step) > 15 else task.current_step
            step_part = f" {step_text}"
        
        # Time information
        time_part = ""
        if task.elapsed_time > 1:
            if task.estimated_remaining:
                time_part = f" ({task.elapsed_time:.0f}s/{task.estimated_remaining:.0f}s)"
            else:
                time_part = f" ({task.elapsed_time:.0f}s)"
        
        # Combine parts
        task_info = f"{phase_display}{progress_part}{step_part}{time_part}"
        
        return task_info[:width]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for display."""
        activities = self.agent_tracker.get_all_activities()
        
        # Count by status
        status_counts = defaultdict(int)
        for activity in activities.values():
            status_counts[activity.status.value] += 1
        
        # Count by type
        type_counts = defaultdict(int)
        for activity in activities.values():
            type_counts[activity.agent_type] += 1
        
        # Task statistics
        active_tasks = sum(1 for a in activities.values() if a.current_task)
        queued_tasks = sum(a.task_queue_size for a in activities.values())
        
        # Resource usage (if available)
        total_cpu = sum(a.resource_usage.get('cpu_percent', 0) for a in activities.values())
        total_memory = sum(a.resource_usage.get('memory_mb', 0) for a in activities.values())
        avg_cpu = total_cpu / len(activities) if activities else 0
        
        return {
            'total_agents': len(activities),
            'status_counts': dict(status_counts),
            'type_counts': dict(type_counts),
            'active_tasks': active_tasks,
            'queued_tasks': queued_tasks,
            'average_cpu_percent': round(avg_cpu, 1),
            'total_memory_mb': round(total_memory, 1),
            'most_active_type': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        }