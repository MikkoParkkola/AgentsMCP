"""
Metrics dashboard and visualization components for the TUI.

Provides comprehensive metrics display with charts, statistics,
and performance indicators for system monitoring.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics

from ...monitoring.metrics_collector import get_metrics_collector, MetricType, AggregatedMetric
from ...monitoring.performance_monitor import get_performance_monitor, PerformanceMetrics
from ...monitoring.agent_tracker import get_agent_tracker
from .status_indicators import MetricDisplay, ProgressBar, ProgressBarStyle, format_metric

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for metrics dashboard."""
    update_interval: float = 1.0
    chart_width: int = 40
    chart_height: int = 8
    show_trends: bool = True
    show_sparklines: bool = True
    max_history_points: int = 100
    auto_scale_charts: bool = True
    compact_mode: bool = False


class SparklineChart:
    """ASCII sparkline chart for displaying metric trends."""
    
    SPARK_CHARS = "▁▂▃▄▅▆▇█"
    
    def __init__(self, width: int = 20):
        """Initialize sparkline chart."""
        self.width = width
    
    def render(self, values: List[float], min_val: Optional[float] = None,
               max_val: Optional[float] = None) -> str:
        """
        Render sparkline chart.
        
        Args:
            values: List of values to chart
            min_val: Minimum value for scaling (auto-detected if None)
            max_val: Maximum value for scaling (auto-detected if None)
            
        Returns:
            ASCII sparkline string
        """
        if not values:
            return " " * self.width
        
        # Take last N values to fit width
        display_values = values[-self.width:] if len(values) > self.width else values
        
        if len(display_values) == 1:
            return self.SPARK_CHARS[4] + " " * (self.width - 1)
        
        # Determine range
        if min_val is None:
            min_val = min(display_values)
        if max_val is None:
            max_val = max(display_values)
        
        if max_val == min_val:
            return self.SPARK_CHARS[4] * len(display_values) + " " * (self.width - len(display_values))
        
        # Map values to spark characters
        spark_chars = []
        for value in display_values:
            # Normalize to 0-7 range
            normalized = (value - min_val) / (max_val - min_val)
            char_index = min(7, int(normalized * 7))
            spark_chars.append(self.SPARK_CHARS[char_index])
        
        # Pad with spaces if needed
        result = "".join(spark_chars)
        if len(result) < self.width:
            result += " " * (self.width - len(result))
        
        return result


class BarChart:
    """ASCII bar chart for displaying metrics."""
    
    def __init__(self, width: int = 30, height: int = 8):
        """Initialize bar chart."""
        self.width = width
        self.height = height
    
    def render(self, data: Dict[str, float], title: str = "") -> List[str]:
        """
        Render bar chart.
        
        Args:
            data: Dictionary of label -> value pairs
            title: Optional chart title
            
        Returns:
            List of strings representing the chart
        """
        if not data:
            return [" " * self.width] * self.height
        
        lines = []
        
        # Add title
        if title:
            lines.append(title[:self.width].center(self.width))
            lines.append("─" * self.width)
        
        # Calculate scaling
        max_val = max(data.values()) if data.values() else 1
        scale = (self.width - 12) / max_val if max_val > 0 else 1  # Leave space for labels
        
        # Create bars
        for label, value in data.items():
            bar_length = int(value * scale)
            bar = "█" * bar_length
            
            # Format label and value
            label_str = label[:8].ljust(8)
            value_str = f"{value:6.1f}"
            
            line = f"{label_str} {bar} {value_str}"
            lines.append(line[:self.width])
        
        # Pad to height
        while len(lines) < self.height:
            lines.append(" " * self.width)
        
        return lines[:self.height]


class MetricsDashboard:
    """
    Comprehensive metrics dashboard with various visualizations.
    
    Displays system performance, agent status, and resource utilization
    with charts, graphs, and statistical summaries.
    """
    
    def __init__(self, config: DashboardConfig = None):
        """Initialize metrics dashboard."""
        self.config = config or DashboardConfig()
        
        # Components
        self.metrics_collector = get_metrics_collector()
        self.performance_monitor = get_performance_monitor()
        self.agent_tracker = get_agent_tracker()
        self.metric_display = MetricDisplay()
        self.sparkline = SparklineChart(width=20)
        self.bar_chart = BarChart(width=self.config.chart_width, height=self.config.chart_height)
        
        # State
        self._cached_display = ""
        self._last_update = 0
        self._metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_history_points))
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._display_callbacks: List[Callable[[str], None]] = []
        
        logger.debug("MetricsDashboard initialized")
    
    def start(self):
        """Start the metrics dashboard."""
        if self._running:
            return
        
        self._running = True
        
        # Start update task
        try:
            loop = asyncio.get_event_loop()
            self._update_task = loop.create_task(self._update_loop())
            logger.info("MetricsDashboard started")
        except RuntimeError:
            logger.warning("No event loop available, dashboard will be manual")
    
    async def stop(self):
        """Stop the metrics dashboard."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
        
        logger.info("MetricsDashboard stopped")
    
    def render(self, width: int = 80, height: int = 20) -> str:
        """
        Render the metrics dashboard.
        
        Args:
            width: Available width for display
            height: Available height for display
            
        Returns:
            Formatted dashboard string
        """
        with self._lock:
            # Check if we need to update
            current_time = time.time()
            if current_time - self._last_update > self.config.update_interval:
                self._update_display(width, height)
                self._last_update = current_time
            
            return self._cached_display
    
    def render_section(self, section: str, width: int = 80, height: int = 10) -> str:
        """
        Render a specific dashboard section.
        
        Args:
            section: Section name ('performance', 'agents', 'resources', 'quality')
            width: Available width
            height: Available height
            
        Returns:
            Formatted section string
        """
        sections = {
            'performance': self._render_performance_section,
            'agents': self._render_agents_section,
            'resources': self._render_resources_section,
            'quality': self._render_quality_section,
            'overview': self._render_overview_section
        }
        
        renderer = sections.get(section, self._render_overview_section)
        return renderer(width, height)
    
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
                self._collect_metrics()
                # Force display update on next render
                with self._lock:
                    self._last_update = 0
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
    
    def _collect_metrics(self):
        """Collect current metrics for history tracking."""
        current_time = time.time()
        
        # Performance metrics
        try:
            perf_metrics = self.performance_monitor.get_current_metrics()
            self._metric_history['response_time_p95'].append(perf_metrics.response_time_p95)
            self._metric_history['cpu_usage'].append(perf_metrics.cpu_usage_percent)
            self._metric_history['memory_usage'].append(perf_metrics.memory_usage_percent)
            self._metric_history['requests_per_second'].append(perf_metrics.requests_per_second)
            self._metric_history['error_rate'].append(perf_metrics.error_rate)
        except Exception as e:
            logger.debug(f"Error collecting performance metrics: {e}")
        
        # Agent metrics
        try:
            agent_summary = self.agent_tracker.get_system_summary()
            self._metric_history['active_agents'].append(agent_summary.get('agents_by_status', {}).get('working', 0))
            self._metric_history['total_agents'].append(agent_summary.get('total_agents', 0))
            self._metric_history['active_tasks'].append(agent_summary.get('active_tasks', 0))
            self._metric_history['queued_tasks'].append(agent_summary.get('queued_tasks', 0))
        except Exception as e:
            logger.debug(f"Error collecting agent metrics: {e}")
    
    def _update_display(self, width: int, height: int):
        """Update the cached display."""
        try:
            if self.config.compact_mode:
                display = self._render_compact_dashboard(width, height)
            else:
                display = self._render_full_dashboard(width, height)
            
            self._cached_display = display
            
            # Notify callbacks
            for callback in self._display_callbacks[:]:
                try:
                    callback(display)
                except Exception as e:
                    logger.error(f"Display callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating dashboard display: {e}")
            self._cached_display = f"[red]Error updating dashboard: {str(e)}[/red]"
    
    def _render_full_dashboard(self, width: int, height: int) -> str:
        """Render full dashboard with multiple sections."""
        sections = []
        section_height = max(4, height // 4)  # Divide into 4 sections
        
        # Overview section
        sections.append(self._render_overview_section(width, section_height))
        sections.append("─" * width)
        
        # Performance section
        sections.append(self._render_performance_section(width, section_height))
        sections.append("─" * width)
        
        # Agents section
        sections.append(self._render_agents_section(width, section_height))
        sections.append("─" * width)
        
        # Resources section
        sections.append(self._render_resources_section(width, section_height))
        
        return "\n".join(sections)
    
    def _render_compact_dashboard(self, width: int, height: int) -> str:
        """Render compact dashboard with key metrics only."""
        lines = []
        
        # Key performance indicators
        try:
            perf_metrics = self.performance_monitor.get_current_metrics()
            agent_summary = self.agent_tracker.get_system_summary()
            
            # Top line: Response time, CPU, Memory
            kpi1 = (
                f"Response: {self.metric_display.format_duration(perf_metrics.response_time_p95)} | "
                f"CPU: {self.metric_display.format_percentage(perf_metrics.cpu_usage_percent)} | "
                f"Memory: {self.metric_display.format_percentage(perf_metrics.memory_usage_percent)}"
            )
            lines.append(kpi1[:width])
            
            # Second line: Agents, Tasks, Throughput
            kpi2 = (
                f"Agents: {agent_summary.get('total_agents', 0)}/{agent_summary.get('agents_by_status', {}).get('working', 0)} | "
                f"Tasks: {agent_summary.get('active_tasks', 0)}/{agent_summary.get('queued_tasks', 0)} | "
                f"RPS: {perf_metrics.requests_per_second:.1f}"
            )
            lines.append(kpi2[:width])
            
            # Sparklines for trends
            if self.config.show_sparklines:
                spark_response = self.sparkline.render(list(self._metric_history['response_time_p95']))
                spark_cpu = self.sparkline.render(list(self._metric_history['cpu_usage']))
                
                lines.append(f"Response: {spark_response}")
                lines.append(f"CPU:      {spark_cpu}")
            
        except Exception as e:
            lines.append(f"[red]Error: {str(e)}[/red]")
        
        # Pad to height
        while len(lines) < height:
            lines.append("")
        
        return "\n".join(lines[:height])
    
    def _render_overview_section(self, width: int, height: int) -> str:
        """Render overview section with key metrics."""
        try:
            perf_metrics = self.performance_monitor.get_current_metrics()
            agent_summary = self.agent_tracker.get_system_summary()
            health = self.performance_monitor.get_system_health()
            
            lines = []
            lines.append(f"[bold]System Overview[/bold]")
            lines.append("")
            
            # Health score
            health_color = "green" if health['overall_health_score'] >= 90 else "yellow" if health['overall_health_score'] >= 70 else "red"
            lines.append(f"Health Score: [{health_color}]{health['overall_health_score']:.0f}%[/{health_color}] ({health['status']})")
            
            # Key metrics in columns
            col1_width = width // 2
            col2_width = width - col1_width - 3
            
            metrics_data = [
                ("Response Time (p95)", self.metric_display.format_duration(perf_metrics.response_time_p95)),
                ("Requests/sec", f"{perf_metrics.requests_per_second:.1f}"),
                ("Success Rate", f"{perf_metrics.success_rate:.1f}%"),
                ("Active Agents", f"{agent_summary.get('agents_by_status', {}).get('working', 0)}/{agent_summary.get('total_agents', 0)}"),
                ("Active Tasks", f"{agent_summary.get('active_tasks', 0)}"),
                ("Queued Tasks", f"{agent_summary.get('queued_tasks', 0)}"),
            ]
            
            # Display in two columns
            for i in range(0, len(metrics_data), 2):
                left_metric = metrics_data[i]
                right_metric = metrics_data[i+1] if i+1 < len(metrics_data) else ("", "")
                
                left_str = f"{left_metric[0]}: {left_metric[1]}"[:col1_width]
                right_str = f"{right_metric[0]}: {right_metric[1]}"[:col2_width] if right_metric[0] else ""
                
                line = f"{left_str:<{col1_width}} | {right_str}"
                lines.append(line[:width])
            
            # Pad to height
            while len(lines) < height:
                lines.append("")
            
            return "\n".join(lines[:height])
            
        except Exception as e:
            return f"[red]Error rendering overview: {str(e)}[/red]"
    
    def _render_performance_section(self, width: int, height: int) -> str:
        """Render performance metrics section."""
        try:
            perf_metrics = self.performance_monitor.get_current_metrics()
            trends = self.performance_monitor.get_performance_trends(30)
            
            lines = []
            lines.append(f"[bold]Performance Metrics[/bold]")
            lines.append("")
            
            # Response time metrics
            lines.append("Response Times:")
            lines.append(f"  P50: {self.metric_display.format_duration(perf_metrics.response_time_p50)}")
            lines.append(f"  P95: {self.metric_display.format_duration(perf_metrics.response_time_p95)}")
            lines.append(f"  P99: {self.metric_display.format_duration(perf_metrics.response_time_p99)}")
            
            # Throughput
            lines.append(f"Throughput: {perf_metrics.requests_per_second:.1f} req/s")
            lines.append(f"Error Rate: {perf_metrics.error_rate:.1f}%")
            
            # Sparklines if enabled
            if self.config.show_sparklines and height > 8:
                response_spark = self.sparkline.render(list(self._metric_history['response_time_p95']))
                error_spark = self.sparkline.render(list(self._metric_history['error_rate']))
                
                lines.append("")
                lines.append(f"Response Trend: {response_spark}")
                lines.append(f"Error Trend:    {error_spark}")
            
            # Pad to height
            while len(lines) < height:
                lines.append("")
            
            return "\n".join(lines[:height])
            
        except Exception as e:
            return f"[red]Error rendering performance: {str(e)}[/red]"
    
    def _render_agents_section(self, width: int, height: int) -> str:
        """Render agent status section."""
        try:
            agent_summary = self.agent_tracker.get_system_summary()
            
            lines = []
            lines.append(f"[bold]Agent Status[/bold]")
            lines.append("")
            
            # Agent counts by status
            status_counts = agent_summary.get('agents_by_status', {})
            for status, count in status_counts.items():
                if count > 0:
                    lines.append(f"  {status.replace('_', ' ').title()}: {count}")
            
            # Agent types
            type_counts = agent_summary.get('agents_by_type', {})
            if type_counts:
                lines.append("")
                lines.append("By Type:")
                for agent_type, count in sorted(type_counts.items()):
                    lines.append(f"  {agent_type}: {count}")
            
            # Task statistics
            lines.append("")
            lines.append(f"Tasks - Active: {agent_summary.get('active_tasks', 0)}, Queued: {agent_summary.get('queued_tasks', 0)}")
            lines.append(f"Success Rate: {agent_summary.get('average_success_rate', 0):.1f}%")
            
            # Pad to height
            while len(lines) < height:
                lines.append("")
            
            return "\n".join(lines[:height])
            
        except Exception as e:
            return f"[red]Error rendering agents: {str(e)}[/red]"
    
    def _render_resources_section(self, width: int, height: int) -> str:
        """Render system resources section."""
        try:
            perf_metrics = self.performance_monitor.get_current_metrics()
            
            lines = []
            lines.append(f"[bold]System Resources[/bold]")
            lines.append("")
            
            # CPU usage with bar
            cpu_percent = perf_metrics.cpu_usage_percent
            cpu_bar = self._create_usage_bar(cpu_percent, 20)
            lines.append(f"CPU:    {cpu_bar} {cpu_percent:.1f}%")
            
            # Memory usage with bar
            mem_percent = perf_metrics.memory_usage_percent
            mem_bar = self._create_usage_bar(mem_percent, 20)
            lines.append(f"Memory: {mem_bar} {mem_percent:.1f}%")
            lines.append(f"        ({perf_metrics.memory_usage_mb:.0f} MB)")
            
            # Disk usage if available
            if hasattr(perf_metrics, 'disk_usage_percent'):
                disk_percent = perf_metrics.disk_usage_percent
                disk_bar = self._create_usage_bar(disk_percent, 20)
                lines.append(f"Disk:   {disk_bar} {disk_percent:.1f}%")
            
            # Sparklines if enabled
            if self.config.show_sparklines and height > 6:
                cpu_spark = self.sparkline.render(list(self._metric_history['cpu_usage']))
                mem_spark = self.sparkline.render(list(self._metric_history['memory_usage']))
                
                lines.append("")
                lines.append(f"CPU Trend:    {cpu_spark}")
                lines.append(f"Memory Trend: {mem_spark}")
            
            # Pad to height
            while len(lines) < height:
                lines.append("")
            
            return "\n".join(lines[:height])
            
        except Exception as e:
            return f"[red]Error rendering resources: {str(e)}[/red]"
    
    def _render_quality_section(self, width: int, height: int) -> str:
        """Render quality metrics section."""
        try:
            perf_metrics = self.performance_monitor.get_current_metrics()
            
            lines = []
            lines.append(f"[bold]Quality Metrics[/bold]")
            lines.append("")
            
            # Success rates
            lines.append(f"Success Rate: {perf_metrics.success_rate:.1f}%")
            lines.append(f"Error Rate: {perf_metrics.error_rate:.1f}%")
            
            # Quality gate pass rate (if available)
            if hasattr(perf_metrics, 'quality_gate_pass_rate'):
                lines.append(f"Quality Gates: {perf_metrics.quality_gate_pass_rate:.1f}%")
            
            # Recent error sparkline
            if self.config.show_sparklines:
                error_spark = self.sparkline.render(list(self._metric_history['error_rate']))
                lines.append(f"Error Trend: {error_spark}")
            
            # Pad to height
            while len(lines) < height:
                lines.append("")
            
            return "\n".join(lines[:height])
            
        except Exception as e:
            return f"[red]Error rendering quality: {str(e)}[/red]"
    
    def _create_usage_bar(self, percentage: float, width: int = 20) -> str:
        """Create a usage bar for percentages."""
        filled = int((percentage / 100.0) * width)
        empty = width - filled
        
        # Choose color based on usage level
        if percentage < 60:
            color = "green"
        elif percentage < 80:
            color = "yellow"
        else:
            color = "red"
        
        filled_part = f"[{color}]{'█' * filled}[/{color}]"
        empty_part = f"[dim]{'░' * empty}[/dim]"
        
        return f"[{filled_part}{empty_part}]"