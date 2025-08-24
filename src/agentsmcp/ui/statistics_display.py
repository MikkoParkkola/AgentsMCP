"""
Advanced Statistics Display for AgentsMCP CLI

Provides beautiful real-time visualization of orchestration metrics,
system performance, and agent behavior with trend analysis.
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import statistics
import json
import os

from .theme_manager import ThemeManager
from .ui_components import UIComponents


@dataclass
class MetricValue:
    """Container for metric values with timestamp"""
    value: float
    timestamp: datetime
    label: str = ""


@dataclass
class TrendData:
    """Trend analysis data"""
    current: float
    previous: float
    change_percent: float
    direction: str  # 'up', 'down', 'stable'
    spark_line: str


class StatisticsDisplay:
    """Real-time statistics display with beautiful visualizations"""
    
    def __init__(self, theme_manager: ThemeManager = None, config: Dict[str, Any] = None):
        self.theme_manager = theme_manager or ThemeManager()
        self.ui = UIComponents(self.theme_manager)
        self.config = config or {}
        
        # Configuration with intelligent defaults
        self.refresh_interval = self.config.get('refresh_interval', 2.0)  # seconds
        self.history_size = self.config.get('history_size', 100)
        self.show_trends = self.config.get('show_trends', True)
        self.show_sparklines = self.config.get('show_sparklines', True)
        self.auto_scale = self.config.get('auto_scale', True)
        
        # Metric storage with automatic cleanup
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.history_size))
        self.categories = {
            'orchestration': ['active_agents', 'symphony_sessions', 'task_queue_size', 'completion_rate'],
            'performance': ['cpu_usage', 'memory_usage', 'response_time', 'throughput'],
            'agents': ['spawned_agents', 'agent_efficiency', 'error_rate', 'uptime'],
            'system': ['disk_usage', 'network_io', 'api_calls', 'cache_hit_rate']
        }
        
        # Display state
        self.is_running = False
        self.selected_category = 'orchestration'
        self.display_mode = 'overview'  # 'overview', 'detailed', 'trends'
        
        # Spark line characters for beautiful micro-charts
        self.spark_chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        
    def add_metric(self, name: str, value: float, label: str = "", timestamp: datetime = None):
        """Add a metric value with automatic trend calculation"""
        if timestamp is None:
            timestamp = datetime.now()
        
        metric = MetricValue(value, timestamp, label)
        self.metrics[name].append(metric)
        
        # Trigger trend analysis if we have enough data
        if len(self.metrics[name]) >= 2:
            self._update_trend(name)
    
    def _update_trend(self, metric_name: str):
        """Update trend data for a metric"""
        if len(self.metrics[metric_name]) < 2:
            return
        
        values = [m.value for m in list(self.metrics[metric_name])[-10:]]  # Last 10 values
        current = values[-1]
        previous = values[-2] if len(values) >= 2 else current
        
        # Calculate percentage change
        if previous != 0:
            change_percent = ((current - previous) / previous) * 100
        else:
            change_percent = 0
        
        # Determine direction
        if abs(change_percent) < 1:
            direction = 'stable'
        elif change_percent > 0:
            direction = 'up'
        else:
            direction = 'down'
        
        # Generate sparkline
        spark_line = self._generate_sparkline(values) if self.show_sparklines else ""
        
        return TrendData(current, previous, change_percent, direction, spark_line)
    
    def _generate_sparkline(self, values: List[float]) -> str:
        """Generate beautiful sparkline visualization"""
        if len(values) < 2:
            return ""
        
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return self.spark_chars[3] * len(values)  # Flat line
        
        spark_line = ""
        for value in values:
            # Normalize to 0-7 range for spark characters
            normalized = (value - min_val) / (max_val - min_val)
            char_index = min(7, int(normalized * 8))
            spark_line += self.spark_chars[char_index]
        
        return spark_line
    
    async def start_display(self) -> Dict[str, Any]:
        """Start the interactive statistics display"""
        self.is_running = True
        
        print(self.ui.clear_screen())
        print(self.ui.hide_cursor(), end='')
        
        try:
            await self._display_loop()
        finally:
            print(self.ui.show_cursor(), end='')
            self.is_running = False
        
        return {"status": "stopped", "uptime": time.time()}
    
    async def _display_loop(self):
        """Main display loop with real-time updates"""
        start_time = time.time()
        
        while self.is_running:
            try:
                # Clear screen and redraw
                print(self.ui.move_cursor(1, 1), end='')
                
                if self.display_mode == 'overview':
                    await self._render_overview()
                elif self.display_mode == 'detailed':
                    await self._render_detailed()
                elif self.display_mode == 'trends':
                    await self._render_trends()
                
                # Update footer with controls and uptime
                uptime = int(time.time() - start_time)
                footer = self._render_footer(uptime)
                print(footer)
                
                # Sleep with interruption checking
                await asyncio.sleep(self.refresh_interval)
                
            except KeyboardInterrupt:
                self.is_running = False
                break
            except Exception as e:
                # Graceful error handling
                error_msg = f"Display error: {str(e)}"
                print(self.theme_manager.current_theme.colors["error"] + error_msg + 
                      self.theme_manager.current_theme.colors["reset"])
                await asyncio.sleep(1)
    
    async def _render_overview(self):
        """Render overview dashboard with key metrics"""
        theme = self.theme_manager.current_theme
        
        # Header
        header = self.ui.box(
            "🚀 AgentsMCP Statistics Dashboard",
            title="Real-time Overview",
            style='heavy',
            width=80
        )
        print(header)
        
        # Create multi-column layout
        columns = []
        
        for category, metric_names in self.categories.items():
            category_data = []
            for metric_name in metric_names[:4]:  # Show top 4 metrics per category
                if metric_name in self.metrics and self.metrics[metric_name]:
                    latest = list(self.metrics[metric_name])[-1]
                    trend = self._update_trend(metric_name) if len(self.metrics[metric_name]) >= 2 else None
                    
                    # Format value with appropriate units
                    formatted_value = self._format_metric_value(metric_name, latest.value)
                    
                    # Add trend indicator
                    trend_indicator = ""
                    if trend and self.show_trends:
                        if trend.direction == 'up':
                            trend_indicator = f" {theme.colors['success']}↗{theme.colors['reset']} "
                        elif trend.direction == 'down':
                            trend_indicator = f" {theme.colors['error']}↘{theme.colors['reset']} "
                        else:
                            trend_indicator = f" {theme.colors['warning']}→{theme.colors['reset']} "
                    
                    # Add sparkline if enabled
                    sparkline = ""
                    if trend and trend.spark_line and self.show_sparklines:
                        sparkline = f" {theme.colors['accent']}{trend.spark_line}{theme.colors['reset']}"
                    
                    display_name = metric_name.replace('_', ' ').title()
                    category_data.append(f"{display_name}: {formatted_value}{trend_indicator}{sparkline}")
                else:
                    display_name = metric_name.replace('_', ' ').title()
                    category_data.append(f"{display_name}: {theme.colors['muted']}No data{theme.colors['reset']}")
            
            # Create category box
            category_title = category.replace('_', ' ').title()
            category_box = self.ui.box(
                "\n".join(category_data),
                title=f"📊 {category_title}",
                style='light',
                padding=1,
                width=35
            )
            columns.append(category_box)
        
        # Display in 2x2 grid
        for i in range(0, len(columns), 2):
            if i + 1 < len(columns):
                print(self.ui.side_by_side([columns[i], columns[i + 1]]))
            else:
                print(columns[i])
            print()  # Spacing between rows
    
    async def _render_detailed(self):
        """Render detailed view for selected category"""
        theme = self.theme_manager.current_theme
        
        category_metrics = self.categories.get(self.selected_category, [])
        category_title = self.selected_category.replace('_', ' ').title()
        
        header = self.ui.box(
            f"📈 Detailed {category_title} Metrics",
            title="Detailed View",
            style='heavy',
            width=80
        )
        print(header)
        
        for metric_name in category_metrics:
            if metric_name in self.metrics and self.metrics[metric_name]:
                await self._render_metric_detail(metric_name)
            else:
                print(f"\n{theme.colors['muted']}No data available for {metric_name}{theme.colors['reset']}")
    
    async def _render_metric_detail(self, metric_name: str):
        """Render detailed view for a single metric"""
        theme = self.theme_manager.current_theme
        metric_data = list(self.metrics[metric_name])
        
        if not metric_data:
            return
        
        # Calculate statistics
        values = [m.value for m in metric_data]
        latest_value = values[-1]
        avg_value = statistics.mean(values)
        min_value = min(values)
        max_value = max(values)
        
        # Generate trend analysis
        trend = self._update_trend(metric_name) if len(values) >= 2 else None
        
        # Create metric summary
        display_name = metric_name.replace('_', ' ').title()
        summary_lines = [
            f"Current: {self._format_metric_value(metric_name, latest_value)}",
            f"Average: {self._format_metric_value(metric_name, avg_value)}",
            f"Range: {self._format_metric_value(metric_name, min_value)} - {self._format_metric_value(metric_name, max_value)}",
        ]
        
        if trend:
            change_color = theme.colors['success'] if trend.direction == 'up' else (
                theme.colors['error'] if trend.direction == 'down' else theme.colors['warning']
            )
            summary_lines.append(f"Change: {change_color}{trend.change_percent:+.1f}%{theme.colors['reset']}")
            
            if trend.spark_line:
                summary_lines.append(f"Trend: {theme.colors['accent']}{trend.spark_line}{theme.colors['reset']}")
        
        # Create detailed box
        metric_box = self.ui.box(
            "\n".join(summary_lines),
            title=f"📊 {display_name}",
            style='light',
            padding=1,
            width=50
        )
        print(metric_box)
    
    async def _render_trends(self):
        """Render trend analysis view"""
        theme = self.theme_manager.current_theme
        
        header = self.ui.box(
            "📈 Trend Analysis Dashboard",
            title="Trend Analysis",
            style='heavy',
            width=80
        )
        print(header)
        
        # Collect all metrics with trends
        trending_metrics = []
        for metric_name in self.metrics:
            if len(self.metrics[metric_name]) >= 2:
                trend = self._update_trend(metric_name)
                if trend:
                    trending_metrics.append((metric_name, trend))
        
        # Sort by absolute change percentage
        trending_metrics.sort(key=lambda x: abs(x[1].change_percent), reverse=True)
        
        # Display top trending metrics
        for i, (metric_name, trend) in enumerate(trending_metrics[:10]):
            display_name = metric_name.replace('_', ' ').title()
            
            # Color based on trend direction
            if trend.direction == 'up':
                trend_color = theme.colors['success']
                arrow = "↗"
            elif trend.direction == 'down':
                trend_color = theme.colors['error']
                arrow = "↘"
            else:
                trend_color = theme.colors['warning']
                arrow = "→"
            
            trend_line = f"{i+1:2d}. {display_name:25} {trend_color}{arrow} {trend.change_percent:+6.1f}%{theme.colors['reset']}"
            
            if trend.spark_line:
                trend_line += f" {theme.colors['accent']}{trend.spark_line}{theme.colors['reset']}"
            
            print(f"  {trend_line}")
        
        if not trending_metrics:
            print(f"  {theme.colors['muted']}No trend data available yet{theme.colors['reset']}")
    
    def _render_footer(self, uptime: int) -> str:
        """Render footer with controls and status"""
        theme = self.theme_manager.current_theme
        
        # Format uptime
        uptime_str = f"{uptime//3600:02d}:{(uptime%3600)//60:02d}:{uptime%60:02d}"
        
        # Control hints
        controls = [
            "1-4: Switch Category",
            "O: Overview",
            "D: Detailed",
            "T: Trends",
            "Q: Quit"
        ]
        
        footer_content = f"Uptime: {uptime_str} | " + " | ".join(controls)
        
        return self.ui.box(
            footer_content,
            style='light',
            padding=0,
            width=80
        )
    
    def _format_metric_value(self, metric_name: str, value: float) -> str:
        """Format metric value with appropriate units and precision"""
        # Define units and formatting for different metric types
        formatters = {
            'cpu_usage': lambda v: f"{v:.1f}%",
            'memory_usage': lambda v: f"{v:.1f}%",
            'disk_usage': lambda v: f"{v:.1f}%",
            'response_time': lambda v: f"{v:.2f}ms",
            'throughput': lambda v: f"{v:.1f}/s",
            'completion_rate': lambda v: f"{v:.1f}%",
            'agent_efficiency': lambda v: f"{v:.1f}%",
            'error_rate': lambda v: f"{v:.2f}%",
            'cache_hit_rate': lambda v: f"{v:.1f}%",
            'uptime': lambda v: f"{v/3600:.1f}h",
            'network_io': lambda v: self._format_bytes(v),
        }
        
        # Use specific formatter or default to integer/decimal
        if metric_name in formatters:
            return formatters[metric_name](value)
        elif isinstance(value, float):
            return f"{value:.2f}" if value < 100 else f"{value:.0f}"
        else:
            return str(int(value))
    
    def _format_bytes(self, bytes_value: float) -> str:
        """Format bytes in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f}PB"
    
    def switch_category(self, category: str):
        """Switch to different metric category"""
        if category in self.categories:
            self.selected_category = category
    
    def switch_display_mode(self, mode: str):
        """Switch display mode"""
        if mode in ['overview', 'detailed', 'trends']:
            self.display_mode = mode
    
    def stop_display(self):
        """Stop the statistics display"""
        self.is_running = False
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of all current metrics"""
        summary = {}
        for metric_name, metric_data in self.metrics.items():
            if metric_data:
                values = [m.value for m in metric_data]
                latest = values[-1]
                trend = self._update_trend(metric_name) if len(values) >= 2 else None
                
                summary[metric_name] = {
                    'current': latest,
                    'average': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values),
                    'trend': {
                        'direction': trend.direction if trend else 'unknown',
                        'change_percent': trend.change_percent if trend else 0,
                        'sparkline': trend.spark_line if trend else ""
                    } if trend else None
                }
        
        return summary
    
    async def simulate_metrics(self, duration: int = 60):
        """Simulate realistic metrics for testing (remove in production)"""
        import random
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # Simulate orchestration metrics
            self.add_metric('active_agents', random.randint(5, 25))
            self.add_metric('symphony_sessions', random.randint(1, 8))
            self.add_metric('task_queue_size', random.randint(0, 50))
            self.add_metric('completion_rate', random.uniform(85, 99))
            
            # Simulate performance metrics
            self.add_metric('cpu_usage', random.uniform(20, 80))
            self.add_metric('memory_usage', random.uniform(30, 75))
            self.add_metric('response_time', random.uniform(50, 500))
            self.add_metric('throughput', random.uniform(100, 1000))
            
            # Simulate agent metrics
            self.add_metric('spawned_agents', random.randint(10, 100))
            self.add_metric('agent_efficiency', random.uniform(75, 95))
            self.add_metric('error_rate', random.uniform(0, 5))
            self.add_metric('uptime', time.time() - start_time)
            
            # Simulate system metrics
            self.add_metric('disk_usage', random.uniform(40, 85))
            self.add_metric('network_io', random.uniform(1024*1024, 100*1024*1024))
            self.add_metric('api_calls', random.randint(100, 2000))
            self.add_metric('cache_hit_rate', random.uniform(60, 95))
            
            await asyncio.sleep(2)  # Update every 2 seconds


async def main():
    """Test the statistics display"""
    stats = StatisticsDisplay()
    
    # Start metrics simulation
    asyncio.create_task(stats.simulate_metrics(300))  # 5 minutes of data
    
    # Start the display
    await stats.start_display()


if __name__ == "__main__":
    asyncio.run(main())