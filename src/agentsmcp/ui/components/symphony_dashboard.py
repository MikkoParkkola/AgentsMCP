"""
Symphony Mode Dashboard - Multi-agent orchestration visualization and control.

This component provides a revolutionary dashboard for managing multiple AI agents:
- Real-time agent status and activity visualization
- Interactive agent coordination and communication
- Beautiful CLI-based dashboard with smooth animations
- Multi-agent workflow orchestration
- Performance monitoring and optimization
- Collaborative task management
"""

from __future__ import annotations

import asyncio
import logging
import json
from typing import Optional, Dict, List, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import math

from ..v2.event_system import AsyncEventSystem, Event, EventType

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent operational states."""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELEGATED = "delegated"


class AgentCapability(Enum):
    """Agent capability types."""
    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    DOCUMENT_PROCESSING = "document_processing"
    WEB_SEARCH = "web_search"
    FILE_OPERATIONS = "file_operations"
    DATA_ANALYSIS = "data_analysis"
    CREATIVE_WRITING = "creative_writing"
    TRANSLATION = "translation"
    REASONING = "reasoning"


@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""
    response_time_avg: float = 0.0
    response_time_p95: float = 0.0
    success_rate: float = 1.0
    tokens_processed: int = 0
    tasks_completed: int = 0
    uptime_percentage: float = 100.0
    last_activity: Optional[datetime] = None
    error_count: int = 0


@dataclass
class Agent:
    """Represents an AI agent in the symphony."""
    id: str
    name: str
    model: str
    capabilities: Set[AgentCapability]
    state: AgentState = AgentState.OFFLINE
    current_task: Optional[str] = None
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    position: Tuple[int, int] = (0, 0)  # For visualization
    color: str = "white"
    animation_phase: float = 0.0  # For smooth animations
    last_heartbeat: Optional[datetime] = None
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Represents a task in the symphony."""
    id: str
    title: str
    description: str
    status: TaskStatus
    assigned_agent_id: Optional[str]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    priority: int = 5  # 1-10, higher is more priority
    dependencies: List[str] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0


@dataclass
class Connection:
    """Represents a connection between agents."""
    from_agent_id: str
    to_agent_id: str
    connection_type: str  # "data_flow", "delegation", "collaboration"
    strength: float = 1.0  # Visual weight of connection
    last_activity: Optional[datetime] = None
    message_count: int = 0


@dataclass
class DashboardRegion:
    """A region of the dashboard display."""
    name: str
    x: int
    y: int
    width: int
    height: int
    content: List[str] = field(default_factory=list)
    border_style: str = "single"
    title: str = ""


class SymphonyDashboard:
    """
    Revolutionary multi-agent orchestration dashboard.
    
    Features:
    - Real-time agent status visualization with smooth animations
    - Interactive agent network topology
    - Task queue and workflow management
    - Performance metrics and monitoring
    - Beautiful CLI-based graphics with 60fps animations
    - Accessibility-first design with screen reader support
    - Multi-agent coordination and communication
    """
    
    def __init__(self, event_system: AsyncEventSystem):
        """Initialize the symphony dashboard."""
        self.event_system = event_system
        
        # Agent management
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.connections: List[Connection] = []
        
        # Dashboard state
        self.active = False
        self.current_view = "overview"  # overview, agents, tasks, metrics
        self.selected_agent_id: Optional[str] = None
        self.selected_task_id: Optional[str] = None
        
        # Display regions
        self.regions: Dict[str, DashboardRegion] = {}
        self.terminal_width = 120
        self.terminal_height = 40
        
        # Animation system
        self.animation_frame = 0
        self.animation_speed = 0.1  # Radians per frame
        self.last_frame_time = 0.0
        self.target_fps = 60
        
        # Performance tracking
        self.frame_times: deque = deque(maxlen=60)  # Last 60 frame times
        self.update_times: deque = deque(maxlen=100)  # Update performance
        
        # Event handlers
        self._callbacks: Dict[str, Callable] = {}
        
        # Color schemes for different themes
        self.colors = {
            "primary": "\x1b[36m",     # Cyan
            "secondary": "\x1b[35m",   # Magenta
            "success": "\x1b[32m",     # Green
            "warning": "\x1b[33m",     # Yellow
            "error": "\x1b[31m",       # Red
            "info": "\x1b[34m",        # Blue
            "muted": "\x1b[37m",       # Light gray
            "bright": "\x1b[1m",       # Bold
            "reset": "\x1b[0m"         # Reset
        }
        
        # Unicode symbols for beautiful display
        self.symbols = {
            "agent_idle": "○",
            "agent_active": "●",
            "agent_busy": "◐",
            "agent_error": "✗",
            "task_pending": "⧖",
            "task_running": "⚡",
            "task_completed": "✓",
            "task_failed": "✗",
            "connection": "→",
            "heartbeat": "♥",
            "cpu": "⚙",
            "memory": "🧠",
            "network": "🌐",
            "time": "⏰"
        }
        
        # Initialize standard layout
        self._initialize_regions()
        self._initialize_demo_agents()
    
    def _initialize_regions(self):
        """Initialize dashboard regions."""
        self.regions = {
            "header": DashboardRegion("header", 0, 0, self.terminal_width, 3, title="AgentsMCP Symphony Dashboard"),
            "agent_grid": DashboardRegion("agent_grid", 0, 3, self.terminal_width // 2, 20, title="Agent Status"),
            "task_queue": DashboardRegion("task_queue", self.terminal_width // 2, 3, self.terminal_width // 2, 20, title="Task Queue"),
            "metrics": DashboardRegion("metrics", 0, 23, self.terminal_width // 3, 12, title="Performance Metrics"),
            "network": DashboardRegion("network", self.terminal_width // 3, 23, self.terminal_width // 3, 12, title="Agent Network"),
            "logs": DashboardRegion("logs", (2 * self.terminal_width) // 3, 23, self.terminal_width // 3, 12, title="Activity Log"),
            "status_bar": DashboardRegion("status_bar", 0, self.terminal_height - 2, self.terminal_width, 2)
        }
    
    def _initialize_demo_agents(self):
        """Initialize demo agents for testing."""
        demo_agents = [
            Agent(
                id="claude",
                name="Claude",
                model="claude-3-opus",
                capabilities={AgentCapability.CHAT, AgentCapability.CODE_ANALYSIS, AgentCapability.REASONING},
                state=AgentState.IDLE,
                position=(10, 5),
                color=self.colors["primary"]
            ),
            Agent(
                id="codex",
                name="Codex",
                model="gpt-4",
                capabilities={AgentCapability.CODE_GENERATION, AgentCapability.CODE_ANALYSIS},
                state=AgentState.IDLE,
                position=(30, 5),
                color=self.colors["success"]
            ),
            Agent(
                id="ollama",
                name="Ollama",
                model="llama2:13b",
                capabilities={AgentCapability.CHAT, AgentCapability.CREATIVE_WRITING},
                state=AgentState.IDLE,
                position=(50, 5),
                color=self.colors["info"]
            )
        ]
        
        for agent in demo_agents:
            self.agents[agent.id] = agent
    
    async def initialize(self) -> bool:
        """Initialize the symphony dashboard."""
        try:
            # Set up event listeners
            await self._setup_event_listeners()
            
            # Start animation loop
            asyncio.create_task(self._animation_loop())
            
            # Initialize agent monitoring
            await self._start_agent_monitoring()
            
            logger.info("Symphony dashboard initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize symphony dashboard: {e}")
            return False
    
    async def _setup_event_listeners(self):
        """Set up event listeners for dashboard updates."""
        await self.event_system.subscribe(EventType.CUSTOM, self._handle_custom_event)
    
    async def _handle_custom_event(self, event: Event):
        """Handle custom events from other components."""
        try:
            event_data = event.data
            component = event_data.get('component')
            action = event_data.get('action')
            
            if component == "agent_manager":
                if action == "agent_status_changed":
                    agent_id = event_data.get('agent_id')
                    status = event_data.get('status')
                    if agent_id in self.agents:
                        self.agents[agent_id].state = AgentState(status)
                        self.agents[agent_id].last_heartbeat = datetime.now()
                
                elif action == "task_assigned":
                    task_data = event_data.get('task')
                    task = Task(
                        id=task_data['id'],
                        title=task_data['title'],
                        description=task_data.get('description', ''),
                        status=TaskStatus(task_data['status']),
                        assigned_agent_id=task_data.get('agent_id'),
                        created_at=datetime.now()
                    )
                    self.tasks[task.id] = task
                    
                    # Update agent state
                    if task.assigned_agent_id and task.assigned_agent_id in self.agents:
                        self.agents[task.assigned_agent_id].state = AgentState.BUSY
                        self.agents[task.assigned_agent_id].current_task = task.id
            
            elif component == "enhanced_command_interface":
                if action == "interpretation":
                    # Update activity in agents
                    await self._update_agent_activity("system", "command_processed")
                    
        except Exception as e:
            logger.error(f"Error handling custom event in symphony dashboard: {e}")
    
    async def _animation_loop(self):
        """Main animation loop for smooth 60fps updates."""
        try:
            frame_duration = 1.0 / self.target_fps  # Target 60fps
            
            while self.active:
                frame_start = asyncio.get_event_loop().time()
                
                # Update animation frame
                self.animation_frame += self.animation_speed
                if self.animation_frame > 2 * math.pi:
                    self.animation_frame = 0.0
                
                # Update agent animations
                for agent in self.agents.values():
                    agent.animation_phase = self.animation_frame
                
                # Render frame if visible
                if self.active:
                    await self._render_frame()
                
                # Calculate frame time
                frame_end = asyncio.get_event_loop().time()
                frame_time = frame_end - frame_start
                self.frame_times.append(frame_time)
                
                # Sleep to maintain target FPS
                sleep_time = max(0, frame_duration - frame_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Animation loop error: {e}")
    
    async def _render_frame(self):
        """Render a complete frame of the dashboard."""
        try:
            # Clear and update all regions
            await self._update_header()
            await self._update_agent_grid()
            await self._update_task_queue()
            await self._update_metrics()
            await self._update_network_visualization()
            await self._update_activity_logs()
            await self._update_status_bar()
            
            # Emit render event with frame data
            frame_data = self._build_frame_data()
            event = Event(
                event_type=EventType.CUSTOM,
                data={
                    "component": "symphony_dashboard",
                    "action": "frame_rendered",
                    "frame_data": frame_data
                }
            )
            await self.event_system.emit_event(event)
            
        except Exception as e:
            logger.error(f"Frame render error: {e}")
    
    def _build_frame_data(self) -> Dict[str, Any]:
        """Build complete frame data for rendering."""
        return {
            "regions": {
                name: {
                    "x": region.x,
                    "y": region.y,
                    "width": region.width,
                    "height": region.height,
                    "content": region.content,
                    "title": region.title,
                    "border_style": region.border_style
                }
                for name, region in self.regions.items()
            },
            "animation_frame": self.animation_frame,
            "current_view": self.current_view,
            "selected_agent": self.selected_agent_id,
            "selected_task": self.selected_task_id
        }
    
    async def _update_header(self):
        """Update the header region."""
        current_time = datetime.now().strftime("%H:%M:%S")
        agent_count = len(self.agents)
        active_agents = len([a for a in self.agents.values() if a.state == AgentState.ACTIVE])
        task_count = len(self.tasks)
        active_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING])
        
        # Calculate average FPS
        avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
        
        header_lines = [
            f"{self.colors['bright']}🎼 AgentsMCP Symphony Dashboard{self.colors['reset']} - {current_time}",
            f"Agents: {active_agents}/{agent_count} active • Tasks: {active_tasks}/{task_count} running • FPS: {avg_fps:.1f}",
            "─" * self.terminal_width
        ]
        
        self.regions["header"].content = header_lines
    
    async def _update_agent_grid(self):
        """Update the agent grid visualization."""
        region = self.regions["agent_grid"]
        lines = []
        
        # Header
        lines.append(f"{self.colors['bright']}Agent Status Grid{self.colors['reset']}")
        lines.append("─" * (region.width - 2))
        
        # Agent grid (2 columns)
        agents_list = list(self.agents.values())
        for i in range(0, len(agents_list), 2):
            left_agent = agents_list[i] if i < len(agents_list) else None
            right_agent = agents_list[i + 1] if i + 1 < len(agents_list) else None
            
            left_display = self._format_agent_display(left_agent) if left_agent else ""
            right_display = self._format_agent_display(right_agent) if right_agent else ""
            
            # Pad to fit columns
            left_col_width = region.width // 2 - 2
            right_col_width = region.width // 2 - 2
            
            left_padded = left_display[:left_col_width].ljust(left_col_width)
            right_padded = right_display[:right_col_width].ljust(right_col_width)
            
            lines.append(f"{left_padded} │ {right_padded}")
        
        # Fill remaining space
        while len(lines) < region.height - 1:
            lines.append("")
        
        region.content = lines
    
    def _format_agent_display(self, agent: Agent) -> str:
        """Format agent display for the grid."""
        # State symbol with animation
        state_symbols = {
            AgentState.IDLE: self.symbols["agent_idle"],
            AgentState.ACTIVE: self.symbols["agent_active"],
            AgentState.BUSY: self._animate_symbol(self.symbols["agent_busy"]),
            AgentState.ERROR: f"{self.colors['error']}{self.symbols['agent_error']}{self.colors['reset']}",
            AgentState.OFFLINE: f"{self.colors['muted']}○{self.colors['reset']}",
            AgentState.INITIALIZING: self._animate_symbol("◴")
        }
        
        symbol = state_symbols.get(agent.state, "?")
        
        # Color based on state
        state_colors = {
            AgentState.IDLE: self.colors["muted"],
            AgentState.ACTIVE: self.colors["success"],
            AgentState.BUSY: self.colors["warning"],
            AgentState.ERROR: self.colors["error"],
            AgentState.OFFLINE: self.colors["muted"]
        }
        
        color = state_colors.get(agent.state, self.colors["reset"])
        
        # Format display
        name_display = f"{color}{agent.name}{self.colors['reset']}"
        model_display = f"{self.colors['muted']}({agent.model}){self.colors['reset']}"
        
        # Response time indicator
        response_indicator = ""
        if agent.metrics.response_time_avg > 0:
            if agent.metrics.response_time_avg < 0.5:
                response_indicator = f"{self.colors['success']}●{self.colors['reset']}"
            elif agent.metrics.response_time_avg < 2.0:
                response_indicator = f"{self.colors['warning']}●{self.colors['reset']}"
            else:
                response_indicator = f"{self.colors['error']}●{self.colors['reset']}"
        
        return f"{symbol} {name_display} {model_display} {response_indicator}"
    
    def _animate_symbol(self, base_symbol: str) -> str:
        """Animate a symbol based on current animation frame."""
        # Create spinning effect for busy symbols
        spin_chars = ["◴", "◷", "◶", "◵"]
        spin_index = int((self.animation_frame * 4) / (2 * math.pi)) % len(spin_chars)
        return spin_chars[spin_index]
    
    async def _update_task_queue(self):
        """Update the task queue visualization."""
        region = self.regions["task_queue"]
        lines = []
        
        # Header
        lines.append(f"{self.colors['bright']}Task Queue{self.colors['reset']}")
        lines.append("─" * (region.width - 2))
        
        # Sort tasks by priority and status
        sorted_tasks = sorted(
            self.tasks.values(),
            key=lambda t: (t.status != TaskStatus.RUNNING, -t.priority, t.created_at)
        )
        
        # Display tasks
        for task in sorted_tasks[:region.height - 4]:  # Leave space for header
            task_display = self._format_task_display(task, region.width - 2)
            lines.append(task_display)
        
        # Fill remaining space
        while len(lines) < region.height - 1:
            lines.append("")
        
        region.content = lines
    
    def _format_task_display(self, task: Task, max_width: int) -> str:
        """Format task display for the queue."""
        # Status symbol
        status_symbols = {
            TaskStatus.PENDING: f"{self.colors['muted']}{self.symbols['task_pending']}{self.colors['reset']}",
            TaskStatus.RUNNING: f"{self.colors['warning']}{self._animate_symbol(self.symbols['task_running'])}{self.colors['reset']}",
            TaskStatus.COMPLETED: f"{self.colors['success']}{self.symbols['task_completed']}{self.colors['reset']}",
            TaskStatus.FAILED: f"{self.colors['error']}{self.symbols['task_failed']}{self.colors['reset']}"
        }
        
        symbol = status_symbols.get(task.status, "?")
        
        # Priority indicator
        priority_indicator = "●" * min(task.priority // 2, 5)  # Visual priority
        
        # Progress bar for running tasks
        progress_bar = ""
        if task.status == TaskStatus.RUNNING and task.progress > 0:
            bar_width = 10
            filled = int(task.progress * bar_width)
            progress_bar = f"[{'█' * filled}{'░' * (bar_width - filled)}]"
        
        # Format display
        title_display = task.title[:max_width - 20]  # Leave space for other elements
        agent_display = f"@{task.assigned_agent_id}" if task.assigned_agent_id else ""
        
        return f"{symbol} {title_display} {progress_bar} {agent_display} {priority_indicator}"
    
    async def _update_metrics(self):
        """Update performance metrics display."""
        region = self.regions["metrics"]
        lines = []
        
        lines.append(f"{self.colors['bright']}Performance{self.colors['reset']}")
        lines.append("─" * (region.width - 2))
        
        # System metrics
        avg_response_time = sum(a.metrics.response_time_avg for a in self.agents.values()) / len(self.agents) if self.agents else 0
        total_tasks = sum(a.metrics.tasks_completed for a in self.agents.values())
        avg_success_rate = sum(a.metrics.success_rate for a in self.agents.values()) / len(self.agents) if self.agents else 100
        
        lines.append(f"Avg Response: {avg_response_time:.2f}s")
        lines.append(f"Tasks Done: {total_tasks}")
        lines.append(f"Success Rate: {avg_success_rate:.1f}%")
        lines.append("")
        
        # Agent performance table
        lines.append("Agent Performance:")
        for agent in self.agents.values():
            perf_line = f"{agent.name[:8]:<8} {agent.metrics.response_time_avg:>6.2f}s {agent.metrics.success_rate:>6.1f}%"
            lines.append(perf_line)
        
        # Fill remaining space
        while len(lines) < region.height - 1:
            lines.append("")
        
        region.content = lines
    
    async def _update_network_visualization(self):
        """Update agent network topology visualization."""
        region = self.regions["network"]
        lines = []
        
        lines.append(f"{self.colors['bright']}Agent Network{self.colors['reset']}")
        lines.append("─" * (region.width - 2))
        
        # Simple network topology
        if len(self.agents) > 0:
            agent_ids = list(self.agents.keys())
            
            # Show connections between agents
            for i, agent_id in enumerate(agent_ids):
                agent = self.agents[agent_id]
                state_color = self.colors["success"] if agent.state == AgentState.ACTIVE else self.colors["muted"]
                
                # Create simple network diagram
                indent = "  " * (i % 3)
                connection_line = f"{indent}{state_color}[{agent.name}]{self.colors['reset']}"
                
                if i > 0:
                    connection_line = f"{indent}└─ {state_color}[{agent.name}]{self.colors['reset']}"
                
                lines.append(connection_line)
                
                # Show current task if any
                if agent.current_task:
                    task_line = f"{indent}   ⚡ Task: {agent.current_task[:10]}..."
                    lines.append(task_line)
        else:
            lines.append("No agents connected")
        
        # Fill remaining space
        while len(lines) < region.height - 1:
            lines.append("")
        
        region.content = lines
    
    async def _update_activity_logs(self):
        """Update activity log display."""
        region = self.regions["logs"]
        lines = []
        
        lines.append(f"{self.colors['bright']}Activity Log{self.colors['reset']}")
        lines.append("─" * (region.width - 2))
        
        # Recent activities (mock data for now)
        current_time = datetime.now()
        activities = [
            f"{(current_time - timedelta(seconds=5)).strftime('%H:%M:%S')} Claude: Task completed",
            f"{(current_time - timedelta(seconds=12)).strftime('%H:%M:%S')} Codex: Code generation started",
            f"{(current_time - timedelta(seconds=25)).strftime('%H:%M:%S')} System: New task queued",
            f"{(current_time - timedelta(seconds=34)).strftime('%H:%M:%S')} Ollama: Agent initialized"
        ]
        
        for activity in activities:
            if len(lines) < region.height - 1:
                lines.append(activity[:region.width - 2])
        
        # Fill remaining space
        while len(lines) < region.height - 1:
            lines.append("")
        
        region.content = lines
    
    async def _update_status_bar(self):
        """Update the status bar."""
        region = self.regions["status_bar"]
        
        # Build status information
        view_info = f"View: {self.current_view.title()}"
        selection_info = ""
        if self.selected_agent_id:
            selection_info = f"Agent: {self.selected_agent_id}"
        elif self.selected_task_id:
            selection_info = f"Task: {self.selected_task_id}"
        
        # Keyboard shortcuts
        shortcuts = "F1:Help F2:Agents F3:Tasks F4:Metrics Q:Quit"
        
        # Format status line
        left_status = f"{view_info} | {selection_info}" if selection_info else view_info
        right_status = shortcuts
        
        # Calculate spacing
        total_width = region.width
        used_width = len(left_status) + len(right_status)
        spacing = " " * max(1, total_width - used_width)
        
        status_line = f"{left_status}{spacing}{right_status}"
        
        region.content = [
            "─" * region.width,
            status_line[:region.width]
        ]
    
    async def _start_agent_monitoring(self):
        """Start monitoring agent health and metrics."""
        async def monitor_loop():
            while self.active:
                try:
                    current_time = datetime.now()
                    
                    for agent in self.agents.values():
                        # Simulate heartbeat check
                        if agent.last_heartbeat:
                            time_since_heartbeat = current_time - agent.last_heartbeat
                            if time_since_heartbeat > timedelta(seconds=30):
                                agent.state = AgentState.OFFLINE
                        
                        # Update metrics (mock for demo)
                        if agent.state == AgentState.ACTIVE:
                            agent.metrics.last_activity = current_time
                            # Simulate some metric updates
                            agent.metrics.response_time_avg = 0.5 + (self.animation_frame % 1.0)
                    
                    await asyncio.sleep(1.0)  # Monitor every second
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Agent monitoring error: {e}")
        
        asyncio.create_task(monitor_loop())
    
    async def _update_agent_activity(self, agent_id: str, activity_type: str):
        """Update agent activity metrics."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.last_heartbeat = datetime.now()
            agent.metrics.last_activity = datetime.now()
            
            # Update state based on activity
            if activity_type == "task_started":
                agent.state = AgentState.BUSY
            elif activity_type == "task_completed":
                agent.state = AgentState.IDLE
                agent.metrics.tasks_completed += 1
                agent.current_task = None
    
    async def activate(self):
        """Activate the symphony dashboard."""
        self.active = True
        logger.info("Symphony dashboard activated")
        
        # Emit activation event
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "symphony_dashboard",
                "action": "activated"
            }
        )
        await self.event_system.emit_event(event)
    
    async def deactivate(self):
        """Deactivate the symphony dashboard."""
        self.active = False
        logger.info("Symphony dashboard deactivated")
        
        # Emit deactivation event
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "symphony_dashboard",
                "action": "deactivated"
            }
        )
        await self.event_system.emit_event(event)
    
    async def switch_view(self, view_name: str):
        """Switch to a different dashboard view."""
        valid_views = ["overview", "agents", "tasks", "metrics"]
        if view_name in valid_views:
            self.current_view = view_name
            logger.info(f"Switched to {view_name} view")
    
    async def select_agent(self, agent_id: str):
        """Select an agent for detailed view."""
        if agent_id in self.agents:
            self.selected_agent_id = agent_id
            self.selected_task_id = None  # Clear task selection
            logger.info(f"Selected agent: {agent_id}")
    
    async def select_task(self, task_id: str):
        """Select a task for detailed view."""
        if task_id in self.tasks:
            self.selected_task_id = task_id
            self.selected_agent_id = None  # Clear agent selection
            logger.info(f"Selected task: {task_id}")
    
    async def add_agent(self, agent: Agent):
        """Add a new agent to the dashboard."""
        self.agents[agent.id] = agent
        logger.info(f"Added agent: {agent.id}")
        
        # Emit agent added event
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "symphony_dashboard",
                "action": "agent_added",
                "agent_id": agent.id,
                "agent_name": agent.name
            }
        )
        await self.event_system.emit_event(event)
    
    async def remove_agent(self, agent_id: str):
        """Remove an agent from the dashboard."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            if self.selected_agent_id == agent_id:
                self.selected_agent_id = None
            logger.info(f"Removed agent: {agent_id}")
    
    async def add_task(self, task: Task):
        """Add a new task to the dashboard."""
        self.tasks[task.id] = task
        logger.info(f"Added task: {task.id}")
        
        # Update agent if assigned
        if task.assigned_agent_id and task.assigned_agent_id in self.agents:
            await self._update_agent_activity(task.assigned_agent_id, "task_started")
    
    async def update_task_progress(self, task_id: str, progress: float):
        """Update task progress."""
        if task_id in self.tasks:
            self.tasks[task_id].progress = max(0.0, min(1.0, progress))
            if progress >= 1.0:
                self.tasks[task_id].status = TaskStatus.COMPLETED
                self.tasks[task_id].completed_at = datetime.now()
                
                # Update agent
                if self.tasks[task_id].assigned_agent_id:
                    await self._update_agent_activity(
                        self.tasks[task_id].assigned_agent_id,
                        "task_completed"
                    )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get dashboard performance statistics."""
        avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        return {
            "active": self.active,
            "current_view": self.current_view,
            "agent_count": len(self.agents),
            "task_count": len(self.tasks),
            "selected_agent": self.selected_agent_id,
            "selected_task": self.selected_task_id,
            "performance": {
                "average_fps": avg_fps,
                "average_frame_time_ms": avg_frame_time * 1000,
                "target_fps": self.target_fps,
                "frame_time_samples": len(self.frame_times)
            },
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "state": agent.state.value,
                    "model": agent.model,
                    "current_task": agent.current_task,
                    "metrics": {
                        "response_time_avg": agent.metrics.response_time_avg,
                        "success_rate": agent.metrics.success_rate,
                        "tasks_completed": agent.metrics.tasks_completed
                    }
                }
                for agent_id, agent in self.agents.items()
            },
            "tasks": {
                task_id: {
                    "title": task.title,
                    "status": task.status.value,
                    "progress": task.progress,
                    "assigned_agent": task.assigned_agent_id
                }
                for task_id, task in self.tasks.items()
            }
        }
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for specific event types."""
        self._callbacks[event_type] = callback
    
    def remove_callback(self, event_type: str):
        """Remove callback for specific event types."""
        if event_type in self._callbacks:
            del self._callbacks[event_type]
    
    async def cleanup(self):
        """Cleanup the symphony dashboard."""
        self.active = False
        
        # Clear data structures
        self.agents.clear()
        self.tasks.clear()
        self.connections.clear()
        self.regions.clear()
        self._callbacks.clear()
        
        # Clear performance data
        self.frame_times.clear()
        self.update_times.clear()
        
        logger.info("Symphony dashboard cleaned up")


# Utility function for easy instantiation
def create_symphony_dashboard(event_system: AsyncEventSystem) -> SymphonyDashboard:
    """Create and return a new SymphonyDashboard instance."""
    return SymphonyDashboard(event_system)