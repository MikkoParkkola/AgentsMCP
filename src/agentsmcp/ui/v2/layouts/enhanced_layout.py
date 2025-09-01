"""
Enhanced TUI layout with integrated progress panels and metrics display.

Provides a comprehensive layout that combines chat interface with real-time
agent status, activity feed, metrics dashboard, and progress visualization.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

from ..components.agent_status_panel import AgentStatusPanel, AgentStatusPanelConfig
from ..components.activity_feed import ActivityFeed, ActivityFeedConfig
from ..components.metrics_dashboard import MetricsDashboard, DashboardConfig
from ..components.progress_visualizer import ProgressVisualizer, ProgressVisualizerConfig

logger = logging.getLogger(__name__)


class LayoutMode(Enum):
    """Layout modes for different use cases."""
    COMPACT = "compact"           # Minimal space usage
    BALANCED = "balanced"         # Equal space distribution
    DETAILED = "detailed"         # Maximum information display
    METRICS_FOCUSED = "metrics"   # Emphasize metrics and charts
    CHAT_FOCUSED = "chat"        # Emphasize chat interface
    MONITORING = "monitoring"     # Full monitoring dashboard


@dataclass
class PanelDimensions:
    """Dimensions and position for a panel."""
    x: int
    y: int
    width: int
    height: int


@dataclass
class EnhancedLayoutConfig:
    """Configuration for enhanced layout."""
    mode: LayoutMode = LayoutMode.BALANCED
    show_agent_panel: bool = True
    show_activity_feed: bool = True
    show_metrics_dashboard: bool = True
    show_progress_visualizer: bool = True
    
    # Panel size preferences (ratios)
    chat_ratio: float = 0.4          # 40% for chat
    agent_panel_ratio: float = 0.2   # 20% for agent status
    activity_ratio: float = 0.15     # 15% for activity feed
    metrics_ratio: float = 0.25      # 25% for metrics/progress
    
    # Update settings
    update_interval: float = 0.5
    auto_adjust_layout: bool = True
    min_panel_size: int = 10
    
    # Visual settings
    show_panel_borders: bool = True
    show_panel_titles: bool = True
    use_color_themes: bool = True


class EnhancedLayout:
    """
    Enhanced TUI layout with integrated progress and metrics panels.
    
    Manages the positioning and rendering of multiple panels including
    chat interface, agent status, activity feed, metrics dashboard,
    and progress visualization.
    """
    
    def __init__(self, config: EnhancedLayoutConfig = None):
        """Initialize enhanced layout."""
        self.config = config or EnhancedLayoutConfig()
        
        # Panel components
        self.agent_panel: Optional[AgentStatusPanel] = None
        self.activity_feed: Optional[ActivityFeed] = None
        self.metrics_dashboard: Optional[MetricsDashboard] = None
        self.progress_visualizer: Optional[ProgressVisualizer] = None
        
        # Layout state
        self._terminal_size: Tuple[int, int] = (80, 24)  # width, height
        self._panel_dimensions: Dict[str, PanelDimensions] = {}
        self._cached_layout = ""
        self._last_update = 0
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._layout_callbacks: List[Callable[[str], None]] = []
        
        # Initialize components based on config
        self._initialize_components()
        
        logger.debug(f"EnhancedLayout initialized in {self.config.mode.value} mode")
    
    def start(self):
        """Start the enhanced layout system."""
        if self._running:
            return
        
        self._running = True
        
        # Start components
        if self.agent_panel:
            self.agent_panel.start()
        if self.activity_feed:
            self.activity_feed.start()
        if self.metrics_dashboard:
            self.metrics_dashboard.start()
        if self.progress_visualizer:
            self.progress_visualizer.start()
        
        # Start layout update task
        try:
            loop = asyncio.get_event_loop()
            self._update_task = loop.create_task(self._update_loop())
            logger.info("EnhancedLayout started")
        except RuntimeError:
            logger.warning("No event loop available, layout will be manual")
    
    async def stop(self):
        """Stop the enhanced layout system."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
        
        # Stop components
        if self.agent_panel:
            await self.agent_panel.stop()
        if self.activity_feed:
            await self.activity_feed.stop()
        if self.metrics_dashboard:
            await self.metrics_dashboard.stop()
        if self.progress_visualizer:
            await self.progress_visualizer.stop()
        
        logger.info("EnhancedLayout stopped")
    
    def set_terminal_size(self, width: int, height: int):
        """Update terminal size and recalculate layout."""
        with self._lock:
            self._terminal_size = (width, height)
            self._calculate_panel_dimensions()
            self._last_update = 0  # Force layout update
    
    def render(self, chat_content: str = "", force_update: bool = False) -> str:
        """
        Render the complete enhanced layout.
        
        Args:
            chat_content: Content for the chat panel
            force_update: Force update even if not time yet
            
        Returns:
            Complete layout string ready for terminal display
        """
        with self._lock:
            current_time = time.time()
            
            if (force_update or 
                current_time - self._last_update > self.config.update_interval or
                not self._cached_layout):
                
                self._update_layout(chat_content)
                self._last_update = current_time
            
            return self._cached_layout
    
    def render_panel(self, panel_name: str) -> str:
        """
        Render a specific panel.
        
        Args:
            panel_name: Name of panel ('chat', 'agents', 'activity', 'metrics', 'progress')
            
        Returns:
            Panel content string
        """
        if panel_name not in self._panel_dimensions:
            return ""
        
        dims = self._panel_dimensions[panel_name]
        
        if panel_name == "agents" and self.agent_panel:
            return self.agent_panel.render(dims.width, dims.height)
        elif panel_name == "activity" and self.activity_feed:
            return self.activity_feed.render(dims.width, dims.height)
        elif panel_name == "metrics" and self.metrics_dashboard:
            return self.metrics_dashboard.render(dims.width, dims.height)
        elif panel_name == "progress" and self.progress_visualizer:
            return self.progress_visualizer.render_combined(dims.width, dims.height)
        else:
            return ""
    
    def get_chat_dimensions(self) -> Optional[PanelDimensions]:
        """Get dimensions for the chat panel."""
        return self._panel_dimensions.get("chat")
    
    def get_panel_dimensions(self, panel_name: str) -> Optional[PanelDimensions]:
        """Get dimensions for a specific panel."""
        return self._panel_dimensions.get(panel_name)
    
    def set_layout_mode(self, mode: LayoutMode):
        """Change the layout mode."""
        with self._lock:
            if mode != self.config.mode:
                self.config.mode = mode
                self._adjust_ratios_for_mode()
                self._calculate_panel_dimensions()
                self._last_update = 0  # Force update
                logger.info(f"Layout mode changed to {mode.value}")
    
    def toggle_panel(self, panel_name: str, visible: bool):
        """Toggle visibility of a specific panel."""
        with self._lock:
            if panel_name == "agents":
                self.config.show_agent_panel = visible
            elif panel_name == "activity":
                self.config.show_activity_feed = visible
            elif panel_name == "metrics":
                self.config.show_metrics_dashboard = visible
            elif panel_name == "progress":
                self.config.show_progress_visualizer = visible
            
            self._calculate_panel_dimensions()
            self._last_update = 0  # Force update
    
    def add_layout_callback(self, callback: Callable[[str], None]):
        """Add callback for layout updates."""
        self._layout_callbacks.append(callback)
    
    def remove_layout_callback(self, callback: Callable[[str], None]):
        """Remove layout callback."""
        if callback in self._layout_callbacks:
            self._layout_callbacks.remove(callback)
    
    async def _update_loop(self):
        """Background layout update loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.update_interval)
                # Force layout update on next render
                with self._lock:
                    self._last_update = 0
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in enhanced layout update loop: {e}")
    
    def _initialize_components(self):
        """Initialize panel components based on configuration."""
        if self.config.show_agent_panel:
            agent_config = AgentStatusPanelConfig(
                compact_mode=(self.config.mode == LayoutMode.COMPACT)
            )
            self.agent_panel = AgentStatusPanel(agent_config)
        
        if self.config.show_activity_feed:
            activity_config = ActivityFeedConfig(
                compact_mode=(self.config.mode == LayoutMode.COMPACT)
            )
            self.activity_feed = ActivityFeed(activity_config)
        
        if self.config.show_metrics_dashboard:
            dashboard_config = DashboardConfig(
                compact_mode=(self.config.mode == LayoutMode.COMPACT)
            )
            self.metrics_dashboard = MetricsDashboard(dashboard_config)
        
        if self.config.show_progress_visualizer:
            visualizer_config = ProgressVisualizerConfig(
                compact_mode=(self.config.mode == LayoutMode.COMPACT)
            )
            self.progress_visualizer = ProgressVisualizer(visualizer_config)
    
    def _calculate_panel_dimensions(self):
        """Calculate dimensions for all panels based on terminal size and configuration."""
        width, height = self._terminal_size
        
        # Reserve space for borders if enabled
        border_space = 2 if self.config.show_panel_borders else 0
        usable_width = width - border_space
        usable_height = height - border_space
        
        # Count enabled panels
        enabled_panels = []
        if self.config.show_agent_panel:
            enabled_panels.append("agents")
        if self.config.show_activity_feed:
            enabled_panels.append("activity")
        if self.config.show_metrics_dashboard:
            enabled_panels.append("metrics")
        if self.config.show_progress_visualizer:
            enabled_panels.append("progress")
        
        # Always include chat
        enabled_panels.insert(0, "chat")
        
        # Calculate layout based on mode
        if self.config.mode == LayoutMode.COMPACT:
            self._calculate_compact_layout(usable_width, usable_height, enabled_panels)
        elif self.config.mode == LayoutMode.DETAILED:
            self._calculate_detailed_layout(usable_width, usable_height, enabled_panels)
        elif self.config.mode == LayoutMode.METRICS_FOCUSED:
            self._calculate_metrics_focused_layout(usable_width, usable_height, enabled_panels)
        elif self.config.mode == LayoutMode.CHAT_FOCUSED:
            self._calculate_chat_focused_layout(usable_width, usable_height, enabled_panels)
        elif self.config.mode == LayoutMode.MONITORING:
            self._calculate_monitoring_layout(usable_width, usable_height, enabled_panels)
        else:  # BALANCED
            self._calculate_balanced_layout(usable_width, usable_height, enabled_panels)
    
    def _calculate_balanced_layout(self, width: int, height: int, panels: List[str]):
        """Calculate balanced layout with equal emphasis."""
        self._panel_dimensions.clear()
        
        if len(panels) <= 2:
            # Simple vertical split
            chat_height = int(height * 0.6)
            panel_height = height - chat_height
            
            self._panel_dimensions["chat"] = PanelDimensions(0, 0, width, chat_height)
            if len(panels) > 1:
                panel_name = panels[1]
                self._panel_dimensions[panel_name] = PanelDimensions(0, chat_height, width, panel_height)
        
        elif len(panels) <= 4:
            # Two columns
            left_width = int(width * 0.5)
            right_width = width - left_width
            
            # Left column: Chat takes most space
            chat_height = int(height * 0.7)
            left_panel_height = height - chat_height
            
            self._panel_dimensions["chat"] = PanelDimensions(0, 0, left_width, chat_height)
            
            if "agents" in panels:
                self._panel_dimensions["agents"] = PanelDimensions(0, chat_height, left_width, left_panel_height)
            
            # Right column: Split remaining panels
            right_panels = [p for p in panels[2:] if p != "agents"]
            if right_panels:
                panel_height = height // len(right_panels)
                for i, panel in enumerate(right_panels):
                    y = i * panel_height
                    h = panel_height if i < len(right_panels) - 1 else height - y
                    self._panel_dimensions[panel] = PanelDimensions(left_width, y, right_width, h)
        
        else:
            # More complex grid layout
            self._calculate_grid_layout(width, height, panels)
    
    def _calculate_compact_layout(self, width: int, height: int, panels: List[str]):
        """Calculate compact layout minimizing space usage."""
        self._panel_dimensions.clear()
        
        # Chat gets 60% of height, others share remaining
        chat_height = int(height * 0.6)
        remaining_height = height - chat_height
        
        self._panel_dimensions["chat"] = PanelDimensions(0, 0, width, chat_height)
        
        # Other panels share remaining space horizontally
        other_panels = panels[1:]
        if other_panels:
            panel_width = width // len(other_panels)
            for i, panel in enumerate(other_panels):
                x = i * panel_width
                w = panel_width if i < len(other_panels) - 1 else width - x
                self._panel_dimensions[panel] = PanelDimensions(x, chat_height, w, remaining_height)
    
    def _calculate_detailed_layout(self, width: int, height: int, panels: List[str]):
        """Calculate detailed layout maximizing information display."""
        self._panel_dimensions.clear()
        
        # More space for side panels in detailed mode
        left_width = int(width * 0.45)
        right_width = width - left_width
        
        # Chat takes left column
        self._panel_dimensions["chat"] = PanelDimensions(0, 0, left_width, height)
        
        # Right column divided among other panels
        other_panels = panels[1:]
        if other_panels:
            panel_height = height // len(other_panels)
            for i, panel in enumerate(other_panels):
                y = i * panel_height
                h = panel_height if i < len(other_panels) - 1 else height - y
                self._panel_dimensions[panel] = PanelDimensions(left_width, y, right_width, h)
    
    def _calculate_metrics_focused_layout(self, width: int, height: int, panels: List[str]):
        """Calculate layout emphasizing metrics and monitoring."""
        self._panel_dimensions.clear()
        
        # Chat gets smaller space
        chat_width = int(width * 0.3)
        metrics_width = width - chat_width
        
        self._panel_dimensions["chat"] = PanelDimensions(0, 0, chat_width, height)
        
        # Prioritize metrics and progress panels
        metrics_panels = [p for p in panels[1:] if p in ["metrics", "progress"]]
        other_panels = [p for p in panels[1:] if p not in ["metrics", "progress"]]
        
        # Metrics panels get upper part of right column
        if metrics_panels:
            metrics_height = int(height * 0.7)
            panel_height = metrics_height // len(metrics_panels)
            
            for i, panel in enumerate(metrics_panels):
                y = i * panel_height
                h = panel_height if i < len(metrics_panels) - 1 else metrics_height - y
                self._panel_dimensions[panel] = PanelDimensions(chat_width, y, metrics_width, h)
            
            # Other panels share remaining space
            if other_panels:
                other_height = height - metrics_height
                other_panel_height = other_height // len(other_panels)
                
                for i, panel in enumerate(other_panels):
                    y = metrics_height + i * other_panel_height
                    h = other_panel_height if i < len(other_panels) - 1 else height - y
                    self._panel_dimensions[panel] = PanelDimensions(chat_width, y, metrics_width, h)
    
    def _calculate_chat_focused_layout(self, width: int, height: int, panels: List[str]):
        """Calculate layout emphasizing chat interface."""
        self._panel_dimensions.clear()
        
        # Chat gets majority of space
        chat_height = int(height * 0.8)
        panel_height = height - chat_height
        
        self._panel_dimensions["chat"] = PanelDimensions(0, 0, width, chat_height)
        
        # Other panels share bottom strip
        other_panels = panels[1:]
        if other_panels:
            panel_width = width // len(other_panels)
            for i, panel in enumerate(other_panels):
                x = i * panel_width
                w = panel_width if i < len(other_panels) - 1 else width - x
                self._panel_dimensions[panel] = PanelDimensions(x, chat_height, w, panel_height)
    
    def _calculate_monitoring_layout(self, width: int, height: int, panels: List[str]):
        """Calculate full monitoring dashboard layout."""
        self._panel_dimensions.clear()
        
        # Remove chat or minimize it for monitoring
        if "chat" in panels and len(panels) > 1:
            # Minimize chat to small corner
            chat_width = int(width * 0.25)
            chat_height = int(height * 0.3)
            self._panel_dimensions["chat"] = PanelDimensions(0, 0, chat_width, chat_height)
            
            # Use remaining space for monitoring panels
            remaining_width = width - chat_width
            other_panels = [p for p in panels if p != "chat"]
            
            if other_panels:
                if len(other_panels) == 1:
                    # Single panel takes remaining space
                    panel = other_panels[0]
                    self._panel_dimensions[panel] = PanelDimensions(chat_width, 0, remaining_width, height)
                else:
                    # Create monitoring grid
                    rows = 2
                    cols = (len(other_panels) + rows - 1) // rows
                    panel_width = remaining_width // cols
                    panel_height = height // rows
                    
                    for i, panel in enumerate(other_panels):
                        row = i // cols
                        col = i % cols
                        x = chat_width + col * panel_width
                        y = row * panel_height
                        w = panel_width if col < cols - 1 else width - x
                        h = panel_height if row < rows - 1 else height - y
                        self._panel_dimensions[panel] = PanelDimensions(x, y, w, h)
            
            # Fill remaining space below chat with another panel if available
            remaining_y = chat_height
            remaining_height = height - chat_height
            if remaining_height > self.config.min_panel_size and other_panels:
                # Use agents panel if available
                if "agents" in other_panels and "agents" not in self._panel_dimensions:
                    self._panel_dimensions["agents"] = PanelDimensions(0, remaining_y, chat_width, remaining_height)
    
    def _calculate_grid_layout(self, width: int, height: int, panels: List[str]):
        """Calculate grid layout for many panels."""
        self._panel_dimensions.clear()
        
        # Create a grid that accommodates all panels
        panel_count = len(panels)
        
        # Determine grid size
        if panel_count <= 4:
            rows, cols = 2, 2
        elif panel_count <= 6:
            rows, cols = 2, 3
        elif panel_count <= 9:
            rows, cols = 3, 3
        else:
            rows = int((panel_count ** 0.5) + 0.5)
            cols = (panel_count + rows - 1) // rows
        
        panel_width = width // cols
        panel_height = height // rows
        
        for i, panel in enumerate(panels):
            row = i // cols
            col = i % cols
            
            x = col * panel_width
            y = row * panel_height
            w = panel_width if col < cols - 1 else width - x
            h = panel_height if row < rows - 1 else height - y
            
            self._panel_dimensions[panel] = PanelDimensions(x, y, w, h)
    
    def _adjust_ratios_for_mode(self):
        """Adjust panel ratios based on layout mode."""
        if self.config.mode == LayoutMode.COMPACT:
            self.config.chat_ratio = 0.6
            self.config.agent_panel_ratio = 0.15
            self.config.activity_ratio = 0.1
            self.config.metrics_ratio = 0.15
        
        elif self.config.mode == LayoutMode.DETAILED:
            self.config.chat_ratio = 0.35
            self.config.agent_panel_ratio = 0.25
            self.config.activity_ratio = 0.2
            self.config.metrics_ratio = 0.2
        
        elif self.config.mode == LayoutMode.METRICS_FOCUSED:
            self.config.chat_ratio = 0.25
            self.config.agent_panel_ratio = 0.15
            self.config.activity_ratio = 0.1
            self.config.metrics_ratio = 0.5
        
        elif self.config.mode == LayoutMode.CHAT_FOCUSED:
            self.config.chat_ratio = 0.7
            self.config.agent_panel_ratio = 0.1
            self.config.activity_ratio = 0.1
            self.config.metrics_ratio = 0.1
        
        elif self.config.mode == LayoutMode.MONITORING:
            self.config.chat_ratio = 0.1
            self.config.agent_panel_ratio = 0.3
            self.config.activity_ratio = 0.3
            self.config.metrics_ratio = 0.3
        
        else:  # BALANCED
            self.config.chat_ratio = 0.4
            self.config.agent_panel_ratio = 0.2
            self.config.activity_ratio = 0.15
            self.config.metrics_ratio = 0.25
    
    def _update_layout(self, chat_content: str):
        """Update the complete layout."""
        try:
            # Recalculate dimensions if needed
            if self.config.auto_adjust_layout:
                self._calculate_panel_dimensions()
            
            # Render each panel
            layout_sections = []
            
            # Get all panels in render order
            panels_to_render = []
            
            if "chat" in self._panel_dimensions:
                panels_to_render.append(("chat", chat_content))
            
            if "agents" in self._panel_dimensions and self.agent_panel:
                dims = self._panel_dimensions["agents"]
                content = self.agent_panel.render(dims.width, dims.height)
                panels_to_render.append(("agents", content))
            
            if "activity" in self._panel_dimensions and self.activity_feed:
                dims = self._panel_dimensions["activity"]
                content = self.activity_feed.render(dims.width, dims.height)
                panels_to_render.append(("activity", content))
            
            if "metrics" in self._panel_dimensions and self.metrics_dashboard:
                dims = self._panel_dimensions["metrics"]
                content = self.metrics_dashboard.render(dims.width, dims.height)
                panels_to_render.append(("metrics", content))
            
            if "progress" in self._panel_dimensions and self.progress_visualizer:
                dims = self._panel_dimensions["progress"]
                content = self.progress_visualizer.render_combined(dims.width, dims.height)
                panels_to_render.append(("progress", content))
            
            # Combine panels into complete layout
            self._cached_layout = self._combine_panels(panels_to_render)
            
            # Notify callbacks
            for callback in self._layout_callbacks[:]:
                try:
                    callback(self._cached_layout)
                except Exception as e:
                    logger.error(f"Layout callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating enhanced layout: {e}")
            self._cached_layout = f"[red]Layout Error: {str(e)}[/red]"
    
    def _combine_panels(self, panels: List[Tuple[str, str]]) -> str:
        """Combine panel contents into complete layout."""
        if not panels:
            return ""
        
        # For simple implementation, just concatenate with separators
        # In a real implementation, this would handle proper positioning
        sections = []
        
        for panel_name, content in panels:
            if self.config.show_panel_titles:
                title = f"=== {panel_name.upper()} ==="
                sections.append(title)
            
            if content:
                sections.append(content)
            
            if self.config.show_panel_borders:
                sections.append("─" * min(80, self._terminal_size[0]))
        
        return "\n".join(sections)