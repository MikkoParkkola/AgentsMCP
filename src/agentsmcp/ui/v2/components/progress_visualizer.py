"""
Progress visualization components for the TUI.

Provides various progress visualization elements including bars, charts,
dependency graphs, and time-series visualizations for comprehensive
progress tracking.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
import math

from ...monitoring.agent_tracker import get_agent_tracker, AgentStatus, TaskPhase, TaskInfo
from .status_indicators import ProgressBar, ProgressBarStyle, get_task_phase_display

logger = logging.getLogger(__name__)


@dataclass
class ProgressVisualizerConfig:
    """Configuration for progress visualizer."""
    show_dependency_graph: bool = True
    show_time_series: bool = True
    show_parallel_execution: bool = True
    chart_width: int = 60
    chart_height: int = 10
    update_interval: float = 0.5
    max_history_points: int = 200
    compact_mode: bool = False


@dataclass
class TaskNode:
    """Node in dependency graph representing a task."""
    task_id: str
    agent_id: str
    description: str
    phase: TaskPhase
    progress: float
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    start_time: Optional[float] = None
    estimated_duration: Optional[float] = None


@dataclass
class ProgressSnapshot:
    """Snapshot of system progress at a point in time."""
    timestamp: float
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    active_tasks: int
    overall_progress: float
    agent_utilization: float


class DependencyGraph:
    """Visualizes task dependencies and execution flow."""
    
    def __init__(self, width: int = 60, height: int = 10):
        """Initialize dependency graph."""
        self.width = width
        self.height = height
        self.nodes: Dict[str, TaskNode] = {}
        self.layout_cache: Dict[str, Tuple[int, int]] = {}
    
    def add_task(self, task_id: str, agent_id: str, description: str,
                phase: TaskPhase, progress: float, dependencies: Set[str] = None):
        """Add or update a task in the graph."""
        if task_id not in self.nodes:
            self.nodes[task_id] = TaskNode(
                task_id=task_id,
                agent_id=agent_id,
                description=description,
                phase=phase,
                progress=progress,
                dependencies=dependencies or set()
            )
            # Update dependent relationships
            for dep_id in self.nodes[task_id].dependencies:
                if dep_id in self.nodes:
                    self.nodes[dep_id].dependents.add(task_id)
        else:
            # Update existing node
            node = self.nodes[task_id]
            node.phase = phase
            node.progress = progress
    
    def remove_task(self, task_id: str):
        """Remove a task from the graph."""
        if task_id in self.nodes:
            node = self.nodes[task_id]
            
            # Clean up relationships
            for dep_id in node.dependencies:
                if dep_id in self.nodes:
                    self.nodes[dep_id].dependents.discard(task_id)
            
            for dep_id in node.dependents:
                if dep_id in self.nodes:
                    self.nodes[dep_id].dependencies.discard(task_id)
            
            del self.nodes[task_id]
            self.layout_cache.pop(task_id, None)
    
    def render(self) -> List[str]:
        """Render the dependency graph as ASCII art."""
        if not self.nodes:
            return [" " * self.width] * self.height
        
        # Calculate layout
        layout = self._calculate_layout()
        
        # Create grid
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw connections first
        self._draw_connections(grid, layout)
        
        # Draw nodes
        for task_id, node in self.nodes.items():
            if task_id in layout:
                x, y = layout[task_id]
                if 0 <= x < self.width and 0 <= y < self.height:
                    symbol = self._get_node_symbol(node)
                    grid[y][x] = symbol
        
        # Convert grid to strings
        return ["".join(row) for row in grid]
    
    def _calculate_layout(self) -> Dict[str, Tuple[int, int]]:
        """Calculate node positions for the graph."""
        if not self.nodes:
            return {}
        
        # Use cached layout if valid
        if len(self.layout_cache) == len(self.nodes):
            return self.layout_cache
        
        # Simple layered layout
        layers = self._get_dependency_layers()
        layout = {}
        
        layer_height = max(1, self.height // max(1, len(layers)))
        
        for layer_idx, layer_tasks in enumerate(layers):
            y = min(self.height - 1, layer_idx * layer_height)
            
            if len(layer_tasks) == 1:
                x = self.width // 2
                layout[layer_tasks[0]] = (x, y)
            else:
                # Distribute tasks across width
                step = max(1, self.width // len(layer_tasks))
                for task_idx, task_id in enumerate(layer_tasks):
                    x = min(self.width - 1, task_idx * step + step // 2)
                    layout[task_id] = (x, y)
        
        self.layout_cache = layout
        return layout
    
    def _get_dependency_layers(self) -> List[List[str]]:
        """Get tasks organized in dependency layers."""
        layers = []
        remaining = set(self.nodes.keys())
        
        while remaining:
            # Find tasks with no remaining dependencies
            current_layer = []
            for task_id in remaining:
                deps = self.nodes[task_id].dependencies
                if not (deps & remaining):  # No unprocessed dependencies
                    current_layer.append(task_id)
            
            if not current_layer:
                # Circular dependency - add all remaining
                current_layer = list(remaining)
            
            layers.append(current_layer)
            remaining -= set(current_layer)
        
        return layers
    
    def _draw_connections(self, grid: List[List[str]], layout: Dict[str, Tuple[int, int]]):
        """Draw connections between dependent tasks."""
        for task_id, node in self.nodes.items():
            if task_id not in layout:
                continue
            
            start_x, start_y = layout[task_id]
            
            for dep_id in node.dependencies:
                if dep_id in layout:
                    end_x, end_y = layout[dep_id]
                    self._draw_line(grid, start_x, start_y, end_x, end_y)
    
    def _draw_line(self, grid: List[List[str]], x1: int, y1: int, x2: int, y2: int):
        """Draw a simple line between two points."""
        if x1 == x2:
            # Vertical line
            start_y, end_y = sorted([y1, y2])
            for y in range(start_y + 1, end_y):
                if 0 <= x1 < len(grid[0]) and 0 <= y < len(grid):
                    grid[y][x1] = "|"
        elif y1 == y2:
            # Horizontal line
            start_x, end_x = sorted([x1, x2])
            for x in range(start_x + 1, end_x):
                if 0 <= x < len(grid[0]) and 0 <= y1 < len(grid):
                    grid[y1][x] = "-"
        else:
            # Simple diagonal approximation
            if abs(x2 - x1) > abs(y2 - y1):
                # More horizontal
                if x1 < x2:
                    if y1 < y2:
                        char = "\\"
                    else:
                        char = "/"
                else:
                    if y1 < y2:
                        char = "/"
                    else:
                        char = "\\"
            else:
                # More vertical
                char = "|"
            
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            if 0 <= mid_x < len(grid[0]) and 0 <= mid_y < len(grid):
                grid[mid_y][mid_x] = char
    
    def _get_node_symbol(self, node: TaskNode) -> str:
        """Get symbol for a task node based on its phase."""
        symbols = {
            TaskPhase.QUEUED: "○",
            TaskPhase.ANALYZING: "◐",
            TaskPhase.PLANNING: "◑",
            TaskPhase.EXECUTING: "●",
            TaskPhase.VALIDATING: "◕",
            TaskPhase.COMPLETING: "◉",
            TaskPhase.COMPLETED: "✓",
            TaskPhase.FAILED: "✗",
        }
        return symbols.get(node.phase, "?")


class TimeSeriesChart:
    """Time-series chart for progress visualization."""
    
    def __init__(self, width: int = 60, height: int = 8):
        """Initialize time-series chart."""
        self.width = width
        self.height = height
        self.data_points: deque = deque(maxlen=width)
    
    def add_data_point(self, timestamp: float, value: float):
        """Add a data point to the chart."""
        self.data_points.append((timestamp, value))
    
    def render(self, title: str = "", y_label: str = "", 
               min_val: Optional[float] = None, max_val: Optional[float] = None) -> List[str]:
        """Render the time-series chart."""
        lines = []
        
        # Add title
        if title:
            lines.append(title[:self.width].center(self.width))
            lines.append("─" * self.width)
        
        if not self.data_points:
            # Empty chart
            for _ in range(self.height):
                lines.append(" " * self.width)
            return lines
        
        # Extract values
        values = [point[1] for point in self.data_points]
        
        # Determine range
        if min_val is None:
            min_val = min(values)
        if max_val is None:
            max_val = max(values)
        
        if max_val == min_val:
            max_val = min_val + 1
        
        # Create chart grid
        chart_height = self.height - (2 if title else 0)
        grid = [[" " for _ in range(self.width)] for _ in range(chart_height)]
        
        # Plot data points
        for i, value in enumerate(values):
            if i >= self.width:
                break
            
            # Scale value to chart height
            y = chart_height - 1 - int((value - min_val) / (max_val - min_val) * (chart_height - 1))
            y = max(0, min(chart_height - 1, y))
            
            # Use different characters for the line
            if i > 0 and i < len(values):
                prev_y = chart_height - 1 - int((values[i-1] - min_val) / (max_val - min_val) * (chart_height - 1))
                prev_y = max(0, min(chart_height - 1, prev_y))
                
                if y == prev_y:
                    grid[y][i] = "─"
                elif y < prev_y:
                    grid[y][i] = "╱"
                else:
                    grid[y][i] = "╲"
            else:
                grid[y][i] = "●"
        
        # Add Y-axis labels
        for i, row in enumerate(grid):
            y_val = max_val - (i / (chart_height - 1)) * (max_val - min_val)
            if i == 0:
                label = f"{y_val:4.1f}"
            elif i == chart_height - 1:
                label = f"{y_val:4.1f}"
            elif i == chart_height // 2:
                label = f"{y_val:4.1f}"
            else:
                label = "     "
            
            line = label[:5] + "│" + "".join(row)
            lines.append(line[:self.width])
        
        return lines


class ParallelExecutionView:
    """Visualizes parallel agent execution."""
    
    def __init__(self, width: int = 60, height: int = 8):
        """Initialize parallel execution view."""
        self.width = width
        self.height = height
    
    def render(self, agent_activities: Dict[str, Any]) -> List[str]:
        """Render parallel execution view."""
        lines = []
        
        # Header
        lines.append("Parallel Agent Execution".center(self.width))
        lines.append("─" * self.width)
        
        # Filter to working agents
        working_agents = {
            agent_id: activity for agent_id, activity in agent_activities.items()
            if activity.status in [AgentStatus.WORKING, AgentStatus.THINKING, 
                                 AgentStatus.WAITING_RESOURCE, AgentStatus.WAITING_INPUT]
        }
        
        if not working_agents:
            lines.append("No agents currently active".center(self.width))
            while len(lines) < self.height:
                lines.append("")
            return lines
        
        # Create timeline view
        max_agents = min(self.height - 2, len(working_agents))
        agent_list = list(working_agents.items())[:max_agents]
        
        for agent_id, activity in agent_list:
            line = self._render_agent_timeline(agent_id, activity)
            lines.append(line[:self.width])
        
        # Pad to height
        while len(lines) < self.height:
            lines.append("")
        
        return lines
    
    def _render_agent_timeline(self, agent_id: str, activity: Any) -> str:
        """Render timeline for a single agent."""
        # Agent name (truncated)
        name = agent_id[:12] + "..." if len(agent_id) > 12 else agent_id
        
        # Status indicator
        status_indicator = get_task_phase_display(
            activity.current_task.phase if activity.current_task else TaskPhase.QUEUED,
            animated=True
        )
        
        # Progress bar
        progress_bar = ProgressBar(width=20, style=ProgressBarStyle.SOLID)
        if activity.current_task:
            progress_display = progress_bar.render_simple(
                activity.current_task.progress_percentage
            )
        else:
            progress_display = progress_bar.render_simple(0)
        
        # Current task description
        task_desc = ""
        if activity.current_task:
            task_desc = activity.current_task.current_step or activity.current_task.description
            task_desc = task_desc[:15] + "..." if len(task_desc) > 15 else task_desc
        
        return f"{name:<15} {status_indicator} {progress_display} {task_desc}"


class ProgressVisualizer:
    """
    Comprehensive progress visualization system.
    
    Combines multiple visualization types to provide complete insight
    into system progress, task dependencies, and parallel execution.
    """
    
    def __init__(self, config: ProgressVisualizerConfig = None):
        """Initialize progress visualizer."""
        self.config = config or ProgressVisualizerConfig()
        
        # Components
        self.dependency_graph = DependencyGraph(
            width=self.config.chart_width,
            height=self.config.chart_height
        )
        self.time_series = TimeSeriesChart(
            width=self.config.chart_width,
            height=self.config.chart_height
        )
        self.parallel_view = ParallelExecutionView(
            width=self.config.chart_width,
            height=self.config.chart_height
        )
        
        # State
        self._progress_history: deque = deque(maxlen=self.config.max_history_points)
        self._cached_displays: Dict[str, str] = {}
        self._last_update = 0
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        
        # Data sources
        self.agent_tracker = get_agent_tracker()
        
        logger.debug("ProgressVisualizer initialized")
    
    def start(self):
        """Start the progress visualizer."""
        if self._running:
            return
        
        self._running = True
        
        # Start update task
        try:
            loop = asyncio.get_event_loop()
            self._update_task = loop.create_task(self._update_loop())
            logger.info("ProgressVisualizer started")
        except RuntimeError:
            logger.warning("No event loop available, visualizer will be manual")
    
    async def stop(self):
        """Stop the progress visualizer."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
        
        logger.info("ProgressVisualizer stopped")
    
    def render_dependency_graph(self) -> str:
        """Render the dependency graph view."""
        if not self.config.show_dependency_graph:
            return ""
        
        with self._lock:
            # Update graph with current tasks
            self._update_dependency_graph()
            
            # Render graph
            graph_lines = self.dependency_graph.render()
            
            # Add header and formatting
            result_lines = []
            result_lines.append("Task Dependencies".center(self.config.chart_width))
            result_lines.append("─" * self.config.chart_width)
            result_lines.extend(graph_lines)
            
            return "\n".join(result_lines)
    
    def render_time_series(self) -> str:
        """Render the time-series progress chart."""
        if not self.config.show_time_series:
            return ""
        
        with self._lock:
            chart_lines = self.time_series.render(
                title="Overall Progress Over Time",
                y_label="Progress %",
                min_val=0.0,
                max_val=100.0
            )
            
            return "\n".join(chart_lines)
    
    def render_parallel_execution(self) -> str:
        """Render the parallel execution view."""
        if not self.config.show_parallel_execution:
            return ""
        
        with self._lock:
            agent_activities = self.agent_tracker.get_all_activities()
            execution_lines = self.parallel_view.render(agent_activities)
            
            return "\n".join(execution_lines)
    
    def render_combined(self, width: int = 80, height: int = 30) -> str:
        """Render combined progress visualization."""
        sections = []
        section_height = height // 3
        
        # Dependency graph section
        if self.config.show_dependency_graph:
            sections.append(self.render_dependency_graph())
            sections.append("═" * width)
        
        # Time series section
        if self.config.show_time_series:
            sections.append(self.render_time_series())
            sections.append("═" * width)
        
        # Parallel execution section
        if self.config.show_parallel_execution:
            sections.append(self.render_parallel_execution())
        
        return "\n".join(sections)
    
    async def _update_loop(self):
        """Background update loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.update_interval)
                self._collect_progress_data()
                # Clear cached displays to force refresh
                with self._lock:
                    self._cached_displays.clear()
                    self._last_update = time.time()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in progress visualizer update loop: {e}")
    
    def _collect_progress_data(self):
        """Collect current progress data."""
        try:
            current_time = time.time()
            
            # Get agent and task statistics
            agent_activities = self.agent_tracker.get_all_activities()
            summary = self.agent_tracker.get_system_summary()
            
            # Calculate overall progress metrics
            total_agents = len(agent_activities)
            active_agents = sum(1 for a in agent_activities.values() 
                              if a.status in [AgentStatus.WORKING, AgentStatus.THINKING])
            
            # Calculate task progress
            total_tasks = summary.get('active_tasks', 0) + summary.get('total_tasks_completed', 0)
            completed_tasks = summary.get('total_tasks_completed', 0)
            failed_tasks = summary.get('total_tasks_failed', 0)
            active_tasks = summary.get('active_tasks', 0)
            
            overall_progress = (completed_tasks / max(1, total_tasks)) * 100
            agent_utilization = (active_agents / max(1, total_agents)) * 100
            
            # Create progress snapshot
            snapshot = ProgressSnapshot(
                timestamp=current_time,
                total_tasks=total_tasks,
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                active_tasks=active_tasks,
                overall_progress=overall_progress,
                agent_utilization=agent_utilization
            )
            
            # Add to history
            self._progress_history.append(snapshot)
            
            # Update time series chart
            self.time_series.add_data_point(current_time, overall_progress)
            
        except Exception as e:
            logger.error(f"Error collecting progress data: {e}")
    
    def _update_dependency_graph(self):
        """Update the dependency graph with current tasks."""
        try:
            agent_activities = self.agent_tracker.get_all_activities()
            
            # Clear old tasks
            current_tasks = set()
            
            # Add current tasks
            for agent_id, activity in agent_activities.items():
                if activity.current_task:
                    task = activity.current_task
                    task_id = f"{agent_id}_{task.task_id}"
                    current_tasks.add(task_id)
                    
                    self.dependency_graph.add_task(
                        task_id=task_id,
                        agent_id=agent_id,
                        description=task.description,
                        phase=task.phase,
                        progress=task.progress_percentage
                    )
            
            # Remove completed/cancelled tasks
            existing_tasks = set(self.dependency_graph.nodes.keys())
            for task_id in existing_tasks - current_tasks:
                self.dependency_graph.remove_task(task_id)
                
        except Exception as e:
            logger.error(f"Error updating dependency graph: {e}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of current progress."""
        if not self._progress_history:
            return {}
        
        latest = self._progress_history[-1]
        
        # Calculate trends if we have enough data
        trends = {}
        if len(self._progress_history) >= 10:
            recent_progress = [s.overall_progress for s in list(self._progress_history)[-10:]]
            recent_utilization = [s.agent_utilization for s in list(self._progress_history)[-10:]]
            
            trends = {
                'progress_trend': recent_progress[-1] - recent_progress[0],
                'utilization_trend': recent_utilization[-1] - recent_utilization[0]
            }
        
        return {
            'overall_progress': latest.overall_progress,
            'agent_utilization': latest.agent_utilization,
            'active_tasks': latest.active_tasks,
            'completed_tasks': latest.completed_tasks,
            'failed_tasks': latest.failed_tasks,
            'total_tasks': latest.total_tasks,
            'trends': trends,
            'dependency_count': len(self.dependency_graph.nodes)
        }