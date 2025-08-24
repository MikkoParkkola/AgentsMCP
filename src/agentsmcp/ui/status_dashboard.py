"""
Revolutionary Status Dashboard - Real-time Orchestration Monitoring

Beautiful, interactive dashboard inspired by:
- Claude Code's clean status displays
- Codex CLI's real-time monitoring
- Gemini CLI's sophisticated metrics visualization

Features:
- Real-time agent status monitoring
- Symphony mode visualization
- Predictive spawning analytics
- Emotional intelligence metrics
- Performance trend analysis
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from .theme_manager import ThemeManager
from .ui_components import UIComponents
from ..orchestration.orchestration_manager import OrchestrationManager

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Configuration for the status dashboard"""
    refresh_interval: float = 1.0  # seconds
    max_history_points: int = 100
    show_animations: bool = True
    compact_mode: bool = False
    auto_refresh: bool = True

@dataclass
class MetricPoint:
    """Single metric data point with timestamp"""
    timestamp: datetime
    value: float
    label: str = ""

class StatusDashboard:
    """
    Revolutionary Status Dashboard
    
    Provides real-time monitoring and visualization of the complete
    AgentsMCP orchestration system with beautiful, adaptive interfaces.
    """
    
    def __init__(self, orchestration_manager: OrchestrationManager,
                 theme_manager: Optional[ThemeManager] = None,
                 config: Optional[DashboardConfig] = None):
        self.orchestration_manager = orchestration_manager
        self.theme_manager = theme_manager or ThemeManager()
        self.ui = UIComponents(self.theme_manager)
        self.config = config or DashboardConfig()
        
        # Dashboard state
        self.is_running = False
        self.last_update = datetime.now()
        self.update_count = 0
        
        # Metrics history
        self.metrics_history: Dict[str, List[MetricPoint]] = {
            'task_completion_rate': [],
            'agent_count': [],
            'average_performance': [],
            'system_load': [],
            'emotional_wellness': [],
            'prediction_accuracy': []
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_updates': 0,
            'average_update_time': 0.0,
            'max_update_time': 0.0,
            'last_update_time': 0.0
        }
    
    async def start_dashboard(self) -> Dict[str, Any]:
        """
        Start the interactive dashboard
        
        Returns:
            Dashboard startup information
        """
        logger.info("🚀 Starting Revolutionary Status Dashboard")
        
        self.is_running = True
        self.last_update = datetime.now()
        
        # Clear screen and hide cursor
        print(self.ui.clear_screen())
        print(self.ui.hide_cursor(), end='')
        
        try:
            if self.config.auto_refresh:
                await self._run_auto_refresh_loop()
            else:
                # Single update mode
                await self._update_dashboard()
        finally:
            # Restore cursor
            print(self.ui.show_cursor(), end='')
        
        return {
            "dashboard_started": True,
            "start_time": self.last_update.isoformat(),
            "refresh_interval": self.config.refresh_interval,
            "auto_refresh": self.config.auto_refresh
        }
    
    async def _run_auto_refresh_loop(self):
        """Run the auto-refresh loop"""
        logger.info("🔄 Starting auto-refresh loop")
        
        try:
            while self.is_running:
                update_start = time.time()
                
                # Move cursor to top
                print(self.ui.move_cursor(1, 1), end='')
                
                # Update dashboard content
                await self._update_dashboard()
                
                # Track performance
                update_time = time.time() - update_start
                self._update_performance_stats(update_time)
                
                # Wait for next refresh
                await asyncio.sleep(self.config.refresh_interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Dashboard interrupted by user")
            self.is_running = False
        except Exception as e:
            logger.error(f"❌ Dashboard error: {e}")
            raise
    
    async def _update_dashboard(self):
        """Update dashboard content"""
        self.update_count += 1
        current_time = datetime.now()
        
        # Get system status
        try:
            system_status = await self.orchestration_manager.get_system_status()
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            system_status = {"error": str(e)}
        
        # Build dashboard layout
        dashboard_content = await self._build_dashboard_layout(system_status)
        
        # Display dashboard
        print(dashboard_content)
        
        # Update metrics history
        await self._update_metrics_history(system_status)
        
        self.last_update = current_time
    
    async def _build_dashboard_layout(self, system_status: Dict[str, Any]) -> str:
        """Build the complete dashboard layout"""
        sections = []
        
        # Header section
        header = self._build_header_section(system_status)
        sections.append(header)
        
        # System overview section
        overview = self._build_overview_section(system_status)
        sections.append(overview)
        
        # Agent status section
        agent_status = await self._build_agent_status_section(system_status)
        sections.append(agent_status)
        
        # Performance metrics section
        metrics = self._build_metrics_section(system_status)
        sections.append(metrics)
        
        # Components status section
        components = self._build_components_section(system_status)
        sections.append(components)
        
        if not self.config.compact_mode:
            # Trends section
            trends = self._build_trends_section()
            sections.append(trends)
        
        # Footer section
        footer = self._build_footer_section()
        sections.append(footer)
        
        return '\n\n'.join(sections)
    
    def _build_header_section(self, system_status: Dict[str, Any]) -> str:
        """Build dashboard header with title and key status"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Main title
        title = self.ui.heading("🎼 AgentsMCP Orchestration Dashboard", level=1, centered=True)
        
        # Status indicator
        if system_status.get("system_status") == "running":
            status = self.ui.status_indicator("active", "System Active")
        else:
            status = self.ui.status_indicator("error", "System Error")
        
        # Time and refresh info
        time_info = f"Last Updated: {current_time}"
        if self.config.auto_refresh:
            time_info += f" | Refresh: {self.config.refresh_interval}s"
        
        time_styled = self.theme_manager.colorize(time_info, 'text_muted')
        
        # Combine header components
        header_content = f"{title}\n\n{status}    {time_styled}"
        
        return header_content
    
    def _build_overview_section(self, system_status: Dict[str, Any]) -> str:
        """Build system overview section"""
        session_id = system_status.get("session_id", "N/A")
        uptime = system_status.get("uptime", "0:00:00")
        mode = system_status.get("orchestration_mode", "unknown")
        
        overview_data = [
            ("Session ID", session_id[:8] if session_id != "N/A" else "N/A"),
            ("Orchestration Mode", mode.title()),
            ("System Uptime", str(uptime)),
            ("Update Count", str(self.update_count))
        ]
        
        # Create two-column layout
        left_column = []
        right_column = []
        
        for i, (label, value) in enumerate(overview_data):
            metric_display = self.ui.metric_display(label, value)
            if i % 2 == 0:
                left_column.append(metric_display)
            else:
                right_column.append(metric_display)
        
        left_content = '\n'.join(left_column)
        right_content = '\n'.join(right_column)
        
        columns_layout = self.ui.multi_column_layout([
            (left_content, 40),
            (right_content, 40)
        ], gap=4)
        
        return self.ui.card("📊 System Overview", columns_layout, status="info")
    
    async def _build_agent_status_section(self, system_status: Dict[str, Any]) -> str:
        """Build agent status section with detailed metrics"""
        component_status = system_status.get("component_status", {})
        
        # Extract agent information
        total_agents = 0
        active_agents = 0
        agent_details = []
        
        # Symphony mode agents
        symphony_status = component_status.get("symphony_mode", {})
        if isinstance(symphony_status, dict):
            symphony_active = symphony_status.get("active_agents", 0)
            total_agents += symphony_active
            if symphony_status.get("is_conducting", False):
                active_agents += symphony_active
        
        # Predictive spawner agents  
        spawner_status = component_status.get("predictive_spawner", {})
        if isinstance(spawner_status, dict):
            spawner_active = spawner_status.get("active_agents", 0)
            total_agents += spawner_active
            active_agents += spawner_active
        
        # Build agent metrics
        agent_metrics = [
            self.ui.metric_display("Total Agents", str(total_agents)),
            self.ui.metric_display("Active Agents", str(active_agents)),
            self.ui.metric_display("Max Capacity", str(system_status.get("max_agents", 50))),
        ]
        
        # Add utilization percentage
        max_agents = system_status.get("max_agents", 50)
        if max_agents > 0:
            utilization = (total_agents / max_agents) * 100
            utilization_color = "success" if utilization < 80 else "warning" if utilization < 95 else "error"
            utilization_display = self.theme_manager.colorize(f"{utilization:.1f}%", utilization_color)
            agent_metrics.append(f"Utilization: {utilization_display}")
        
        agent_content = '\n'.join(agent_metrics)
        
        # Add agent status visualization if not compact
        if not self.config.compact_mode and total_agents > 0:
            progress = min(1.0, total_agents / max_agents)
            progress_bar = self.ui.progress_bar(progress, label="Capacity")
            agent_content += f"\n\n{progress_bar}"
        
        return self.ui.card("🤖 Agent Status", agent_content, status="success" if active_agents > 0 else "inactive")
    
    def _build_metrics_section(self, system_status: Dict[str, Any]) -> str:
        """Build performance metrics section"""
        performance_metrics = system_status.get("performance_metrics", {})
        
        metrics = [
            ("Tasks Completed", str(performance_metrics.get("total_tasks_completed", 0))),
            ("Avg Quality Score", f"{performance_metrics.get('average_quality_score', 0.0):.2f}"),
            ("Avg Completion Time", f"{performance_metrics.get('average_completion_time', 0.0):.1f}s"),
            ("Agent Satisfaction", f"{performance_metrics.get('agent_satisfaction', 0.0):.2f}")
        ]
        
        # Create metrics grid
        left_metrics = []
        right_metrics = []
        
        for i, (label, value) in enumerate(metrics):
            metric_display = self.ui.metric_display(label, value)
            if i % 2 == 0:
                left_metrics.append(metric_display)
            else:
                right_metrics.append(metric_display)
        
        left_content = '\n'.join(left_metrics)
        right_content = '\n'.join(right_metrics)
        
        metrics_layout = self.ui.multi_column_layout([
            (left_content, 35),
            (right_content, 35)
        ], gap=6)
        
        return self.ui.card("📈 Performance Metrics", metrics_layout, status="info")
    
    def _build_components_section(self, system_status: Dict[str, Any]) -> str:
        """Build orchestration components status section"""
        component_status = system_status.get("component_status", {})
        
        components = [
            ("Seamless Coordinator", component_status.get("seamless_coordinator", "unknown")),
            ("Emotional Orchestrator", component_status.get("emotional_orchestrator", "unknown")),
            ("Symphony Mode", component_status.get("symphony_mode", "unknown")),
            ("Predictive Spawner", component_status.get("predictive_spawner", "unknown"))
        ]
        
        component_items = []
        for name, status in components:
            if isinstance(status, dict):
                # Extract status from dict
                if status.get("is_conducting", False):
                    status_text = "conducting"
                elif status.get("active_agents", 0) > 0:
                    status_text = "active"
                else:
                    status_text = "ready"
            else:
                status_text = str(status)
            
            # Map status to indicator
            if status_text in ["active", "running", "conducting"]:
                indicator_status = "success"
            elif status_text in ["ready", "inactive"]:
                indicator_status = "inactive"
            else:
                indicator_status = "warning"
            
            item = self.ui.list_item(f"{name}: {status_text.title()}", status=indicator_status)
            component_items.append(item)
        
        components_content = '\n'.join(component_items)
        
        return self.ui.card("⚙️ Components Status", components_content, status="info")
    
    def _build_trends_section(self) -> str:
        """Build trends and analytics section"""
        if not any(self.metrics_history.values()):
            return self.ui.card("📊 Trends", "No historical data available yet...", status="info")
        
        trend_items = []
        
        # Analyze trends for each metric
        for metric_name, history in self.metrics_history.items():
            if len(history) >= 2:
                # Calculate trend
                recent_values = [point.value for point in history[-10:]]  # Last 10 points
                if len(recent_values) >= 2:
                    change = ((recent_values[-1] - recent_values[0]) / recent_values[0]) * 100 if recent_values[0] != 0 else 0
                    latest_value = recent_values[-1]
                    
                    # Format metric name for display
                    display_name = metric_name.replace('_', ' ').title()
                    
                    trend_display = self.ui.metric_display(
                        display_name, 
                        f"{latest_value:.2f}", 
                        change=change
                    )
                    trend_items.append(trend_display)
        
        if not trend_items:
            trends_content = "Collecting trend data..."
        else:
            trends_content = '\n'.join(trend_items)
        
        return self.ui.card("📈 Performance Trends", trends_content, status="info")
    
    def _build_footer_section(self) -> str:
        """Build dashboard footer with controls and status"""
        footer_items = []
        
        # Performance info
        perf_info = (
            f"Updates: {self.performance_stats['total_updates']} | "
            f"Avg Time: {self.performance_stats['average_update_time']:.3f}s"
        )
        footer_items.append(self.theme_manager.colorize(perf_info, 'text_muted'))
        
        # Controls info
        if self.config.auto_refresh:
            controls = "Press Ctrl+C to exit | Auto-refresh enabled"
        else:
            controls = "Dashboard updated once | Use auto_refresh=True for continuous updates"
        
        footer_items.append(self.theme_manager.colorize(controls, 'text_muted'))
        
        # Separator
        separator = self.ui.separator(width=80)
        
        footer_content = f"{separator}\n{' | '.join(footer_items)}"
        
        return footer_content
    
    async def _update_metrics_history(self, system_status: Dict[str, Any]):
        """Update metrics history for trend analysis"""
        current_time = datetime.now()
        
        # Extract metrics from system status
        performance_metrics = system_status.get("performance_metrics", {})
        component_status = system_status.get("component_status", {})
        
        # Update task completion rate
        total_tasks = performance_metrics.get("total_tasks_completed", 0)
        if hasattr(self, '_last_total_tasks'):
            completion_rate = total_tasks - self._last_total_tasks
        else:
            completion_rate = 0
        self._last_total_tasks = total_tasks
        
        self._add_metric_point("task_completion_rate", current_time, completion_rate)
        
        # Update agent count
        total_agents = 0
        symphony_status = component_status.get("symphony_mode", {})
        if isinstance(symphony_status, dict):
            total_agents += symphony_status.get("active_agents", 0)
        
        spawner_status = component_status.get("predictive_spawner", {})
        if isinstance(spawner_status, dict):
            total_agents += spawner_status.get("active_agents", 0)
        
        self._add_metric_point("agent_count", current_time, total_agents)
        
        # Update other metrics
        self._add_metric_point("average_performance", current_time, performance_metrics.get("average_quality_score", 0.0))
        
        # System load (simplified calculation)
        max_agents = system_status.get("max_agents", 50)
        system_load = (total_agents / max_agents) * 100 if max_agents > 0 else 0
        self._add_metric_point("system_load", current_time, system_load)
        
        # Emotional wellness (placeholder - would come from emotional orchestrator)
        self._add_metric_point("emotional_wellness", current_time, performance_metrics.get("agent_satisfaction", 0.0) * 100)
        
        # Prediction accuracy (placeholder - would come from predictive spawner)
        if isinstance(spawner_status, dict):
            prediction_accuracy = spawner_status.get("prediction_accuracy", 0.7) * 100
        else:
            prediction_accuracy = 70.0
        self._add_metric_point("prediction_accuracy", current_time, prediction_accuracy)
    
    def _add_metric_point(self, metric_name: str, timestamp: datetime, value: float):
        """Add a metric point to history"""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        point = MetricPoint(timestamp=timestamp, value=value)
        self.metrics_history[metric_name].append(point)
        
        # Trim history if too long
        if len(self.metrics_history[metric_name]) > self.config.max_history_points:
            self.metrics_history[metric_name].pop(0)
    
    def _update_performance_stats(self, update_time: float):
        """Update dashboard performance statistics"""
        self.performance_stats['total_updates'] += 1
        self.performance_stats['last_update_time'] = update_time
        self.performance_stats['max_update_time'] = max(
            self.performance_stats['max_update_time'], update_time
        )
        
        # Update running average
        total = self.performance_stats['total_updates']
        current_avg = self.performance_stats['average_update_time']
        self.performance_stats['average_update_time'] = (
            (current_avg * (total - 1) + update_time) / total
        )
    
    def stop_dashboard(self):
        """Stop the dashboard"""
        logger.info("🛑 Stopping Status Dashboard")
        self.is_running = False
        
        # Restore cursor
        print(self.ui.show_cursor(), end='')
    
    async def get_dashboard_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current dashboard data"""
        try:
            system_status = await self.orchestration_manager.get_system_status()
        except Exception as e:
            system_status = {"error": str(e)}
        
        return {
            "timestamp": datetime.now().isoformat(),
            "update_count": self.update_count,
            "system_status": system_status,
            "metrics_history_length": {
                name: len(history) for name, history in self.metrics_history.items()
            },
            "performance_stats": self.performance_stats.copy(),
            "dashboard_config": {
                "refresh_interval": self.config.refresh_interval,
                "auto_refresh": self.config.auto_refresh,
                "compact_mode": self.config.compact_mode
            }
        }
    
    def configure_dashboard(self, **kwargs):
        """Update dashboard configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Dashboard config updated: {key} = {value}")
            else:
                logger.warning(f"Unknown config option: {key}")
    
    def export_metrics_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Export metrics history for analysis"""
        exported = {}
        
        for metric_name, history in self.metrics_history.items():
            exported[metric_name] = [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "value": point.value,
                    "label": point.label
                }
                for point in history
            ]
        
        return exported