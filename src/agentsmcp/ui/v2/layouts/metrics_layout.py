"""
Specialized metrics-focused layout for comprehensive monitoring display.

Provides a dedicated layout optimized for metrics visualization and monitoring,
with emphasis on detailed analytics and performance tracking.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Static

from ..components.metrics_dashboard import MetricsDashboard
from ..components.activity_feed import ActivityFeed
from ..components.progress_visualizer import (
    DependencyGraph,
    TimeSeriesChart,
    ParallelExecutionView
)
from ..components.agent_status_panel import AgentStatusPanel
from ...monitoring.metrics_collector import MetricsCollector
from ...monitoring.agent_tracker import AgentTracker
from ...monitoring.performance_monitor import PerformanceMonitor


@dataclass
class MetricsLayoutConfig:
    """Configuration for metrics-focused layout."""
    
    # Panel visibility
    show_dependency_graph: bool = True
    show_timeseries: bool = True
    show_parallel_view: bool = True
    show_agent_details: bool = True
    show_activity_feed: bool = True
    
    # Layout ratios (percentages)
    main_metrics_ratio: int = 40
    visualization_ratio: int = 35
    status_ratio: int = 25
    
    # Update intervals
    metrics_update_ms: int = 1000
    visualization_update_ms: int = 2000
    status_update_ms: int = 500
    
    # Display options
    show_historical_data: bool = True
    show_performance_alerts: bool = True
    compact_agent_view: bool = False
    max_activity_items: int = 50


class MetricsLayout(Widget):
    """
    Specialized layout for comprehensive metrics and monitoring display.
    
    Optimized for detailed analytics, performance tracking, and system monitoring.
    Provides multiple visualization panels with configurable layout ratios.
    """
    
    def __init__(
        self,
        config: Optional[MetricsLayoutConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        agent_tracker: Optional[AgentTracker] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.config = config or MetricsLayoutConfig()
        self.metrics_collector = metrics_collector
        self.agent_tracker = agent_tracker
        self.performance_monitor = performance_monitor
        
        # Panel widgets
        self._metrics_dashboard: Optional[MetricsDashboard] = None
        self._dependency_graph: Optional[DependencyGraph] = None
        self._timeseries_chart: Optional[TimeSeriesChart] = None
        self._parallel_view: Optional[ParallelExecutionView] = None
        self._agent_status: Optional[AgentStatusPanel] = None
        self._activity_feed: Optional[ActivityFeed] = None
    
    def compose(self) -> ComposeResult:
        """Compose the metrics-focused layout."""
        
        with Vertical(id="metrics-layout-container"):
            # Top section: Main metrics dashboard
            with Horizontal(id="metrics-main-section"):
                if self.metrics_collector:
                    self._metrics_dashboard = MetricsDashboard(
                        metrics_collector=self.metrics_collector,
                        performance_monitor=self.performance_monitor,
                        update_interval=self.config.metrics_update_ms / 1000.0,
                        show_alerts=self.config.show_performance_alerts,
                        show_historical=self.config.show_historical_data,
                        id="main-metrics-dashboard"
                    )
                    yield self._metrics_dashboard
                else:
                    yield Static("Metrics Dashboard\n(No metrics collector)", 
                               id="metrics-placeholder")
            
            # Middle section: Visualization panels
            with Horizontal(id="metrics-visualization-section"):
                # Left: Dependency graph and timeline
                with Vertical(id="graph-timeline-panel"):
                    if self.config.show_dependency_graph and self.agent_tracker:
                        self._dependency_graph = DependencyGraph(
                            agent_tracker=self.agent_tracker,
                            update_interval=self.config.visualization_update_ms / 1000.0,
                            id="dependency-graph"
                        )
                        yield self._dependency_graph
                    
                    if self.config.show_timeseries and self.metrics_collector:
                        self._timeseries_chart = TimeSeriesChart(
                            metrics_collector=self.metrics_collector,
                            update_interval=self.config.visualization_update_ms / 1000.0,
                            max_points=100,
                            id="timeseries-chart"
                        )
                        yield self._timeseries_chart
                
                # Right: Parallel execution view
                if self.config.show_parallel_view and self.agent_tracker:
                    self._parallel_view = ParallelExecutionView(
                        agent_tracker=self.agent_tracker,
                        update_interval=self.config.visualization_update_ms / 1000.0,
                        max_agents=20,
                        id="parallel-execution-view"
                    )
                    yield self._parallel_view
            
            # Bottom section: Status and activity
            with Horizontal(id="metrics-status-section"):
                # Left: Agent status
                if self.config.show_agent_details and self.agent_tracker:
                    self._agent_status = AgentStatusPanel(
                        agent_tracker=self.agent_tracker,
                        update_interval=self.config.status_update_ms / 1000.0,
                        compact_mode=self.config.compact_agent_view,
                        show_resource_usage=True,
                        show_task_history=True,
                        id="agent-status-panel"
                    )
                    yield self._agent_status
                
                # Right: Activity feed
                if self.config.show_activity_feed:
                    self._activity_feed = ActivityFeed(
                        agent_tracker=self.agent_tracker,
                        max_items=self.config.max_activity_items,
                        show_timestamps=True,
                        show_severity=True,
                        enable_search=True,
                        id="activity-feed"
                    )
                    yield self._activity_feed
    
    def on_mount(self) -> None:
        """Initialize layout styling and update timers."""
        self._apply_layout_styles()
        self._start_update_timers()
    
    def _apply_layout_styles(self) -> None:
        """Apply CSS styles for metrics layout."""
        
        # Calculate section heights based on ratios
        main_height = f"{self.config.main_metrics_ratio}%"
        viz_height = f"{self.config.visualization_ratio}%"
        status_height = f"{self.config.status_ratio}%"
        
        # Apply styles to layout sections
        styles = {
            "#metrics-layout-container": {
                "layout": "vertical",
                "height": "100%",
                "width": "100%"
            },
            "#metrics-main-section": {
                "layout": "horizontal",
                "height": main_height,
                "border": "solid $accent",
                "margin": 1
            },
            "#metrics-visualization-section": {
                "layout": "horizontal", 
                "height": viz_height,
                "border": "solid $primary",
                "margin": 1
            },
            "#metrics-status-section": {
                "layout": "horizontal",
                "height": status_height,
                "border": "solid $secondary",
                "margin": 1
            },
            "#graph-timeline-panel": {
                "layout": "vertical",
                "width": "50%"
            },
            "#main-metrics-dashboard": {
                "width": "100%",
                "padding": 1
            },
            "#dependency-graph": {
                "height": "50%",
                "border": "solid $warning"
            },
            "#timeseries-chart": {
                "height": "50%",
                "border": "solid $success"
            },
            "#parallel-execution-view": {
                "width": "50%",
                "border": "solid $error"
            },
            "#agent-status-panel": {
                "width": "60%",
                "border": "solid $surface"
            },
            "#activity-feed": {
                "width": "40%",
                "border": "solid $muted"
            }
        }
        
        # Apply styles to components
        for selector, style_dict in styles.items():
            try:
                widget = self.query_one(selector)
                for property_name, value in style_dict.items():
                    setattr(widget.styles, property_name, value)
            except Exception:
                # Selector not found, continue
                continue
    
    def _start_update_timers(self) -> None:
        """Start background update timers for all panels."""
        
        # Set update intervals for each component
        if self._metrics_dashboard:
            self.set_timer(
                self.config.metrics_update_ms / 1000.0,
                self._update_metrics_dashboard,
                repeat=True
            )
        
        if self._dependency_graph or self._timeseries_chart or self._parallel_view:
            self.set_timer(
                self.config.visualization_update_ms / 1000.0,
                self._update_visualizations,
                repeat=True
            )
        
        if self._agent_status or self._activity_feed:
            self.set_timer(
                self.config.status_update_ms / 1000.0,
                self._update_status_panels,
                repeat=True
            )
    
    async def _update_metrics_dashboard(self) -> None:
        """Update the main metrics dashboard."""
        if self._metrics_dashboard and self._metrics_dashboard.is_mounted:
            await self._metrics_dashboard.refresh_metrics()
    
    async def _update_visualizations(self) -> None:
        """Update visualization panels."""
        if self._dependency_graph and self._dependency_graph.is_mounted:
            await self._dependency_graph.refresh_graph()
        
        if self._timeseries_chart and self._timeseries_chart.is_mounted:
            await self._timeseries_chart.refresh_chart()
        
        if self._parallel_view and self._parallel_view.is_mounted:
            await self._parallel_view.refresh_view()
    
    async def _update_status_panels(self) -> None:
        """Update status and activity panels."""
        if self._agent_status and self._agent_status.is_mounted:
            await self._agent_status.refresh_status()
        
        if self._activity_feed and self._activity_feed.is_mounted:
            await self._activity_feed.refresh_feed()
    
    def get_layout_info(self) -> Dict[str, any]:
        """Get current layout information."""
        return {
            "type": "metrics_layout",
            "config": self.config,
            "panels": {
                "metrics_dashboard": self._metrics_dashboard is not None,
                "dependency_graph": self._dependency_graph is not None,
                "timeseries_chart": self._timeseries_chart is not None,
                "parallel_view": self._parallel_view is not None,
                "agent_status": self._agent_status is not None,
                "activity_feed": self._activity_feed is not None
            },
            "update_intervals": {
                "metrics": self.config.metrics_update_ms,
                "visualization": self.config.visualization_update_ms,
                "status": self.config.status_update_ms
            }
        }
    
    async def reconfigure(self, new_config: MetricsLayoutConfig) -> None:
        """Reconfigure the layout with new settings."""
        self.config = new_config
        
        # Re-apply styles and restart timers
        self._apply_layout_styles()
        self._start_update_timers()
        
        # Update component configurations
        if self._metrics_dashboard:
            await self._metrics_dashboard.reconfigure(
                update_interval=new_config.metrics_update_ms / 1000.0,
                show_alerts=new_config.show_performance_alerts,
                show_historical=new_config.show_historical_data
            )
        
        if self._agent_status:
            await self._agent_status.reconfigure(
                update_interval=new_config.status_update_ms / 1000.0,
                compact_mode=new_config.compact_agent_view
            )
        
        if self._activity_feed:
            await self._activity_feed.reconfigure(
                max_items=new_config.max_activity_items
            )
    
    def get_terminal_size_requirements(self) -> Tuple[int, int]:
        """Get minimum terminal size requirements for metrics layout."""
        # Metrics layout requires larger terminal due to multiple panels
        return (120, 40)  # width, height