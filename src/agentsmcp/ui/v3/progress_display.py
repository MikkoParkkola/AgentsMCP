"""Progress display system with real-time timing for AgentsMCP TUI.

This module provides comprehensive progress visualization including progress bars,
timing information, and agent status tracking for transparent AI operations.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class AgentStatus(Enum):
    """Status types for agents."""
    IDLE = "idle"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentProgress:
    """Progress information for a single agent."""
    agent_id: str
    agent_name: str
    status: AgentStatus
    current_step: Optional[str] = None
    progress_percentage: float = 0.0
    estimated_total_duration_ms: int = 0
    actual_duration_ms: int = 0
    started_at: Optional[float] = None
    last_update: float = field(default_factory=time.time)
    
    def start(self) -> None:
        """Mark the agent as started."""
        self.started_at = time.time()
        self.status = AgentStatus.IN_PROGRESS
        self.last_update = self.started_at
    
    def update_progress(self, percentage: float, current_step: str = None) -> None:
        """Update agent progress."""
        self.progress_percentage = min(100.0, max(0.0, percentage))
        if current_step:
            self.current_step = current_step
        self.last_update = time.time()
    
    def complete(self) -> None:
        """Mark the agent as completed."""
        self.status = AgentStatus.COMPLETED
        self.progress_percentage = 100.0
        self.last_update = time.time()
        if self.started_at:
            self.actual_duration_ms = int((time.time() - self.started_at) * 1000)
    
    def set_error(self, error_message: str = None) -> None:
        """Mark the agent as errored."""
        self.status = AgentStatus.ERROR
        self.current_step = error_message or "Error occurred"
        self.last_update = time.time()
    
    @property
    def elapsed_time_ms(self) -> int:
        """Get elapsed time since start."""
        if not self.started_at:
            return 0
        return int((time.time() - self.started_at) * 1000)
    
    @property
    def estimated_completion_ms(self) -> int:
        """Get estimated time to completion."""
        if self.progress_percentage <= 0:
            return self.estimated_total_duration_ms
        
        elapsed = self.elapsed_time_ms
        if self.progress_percentage >= 100:
            return 0
        
        # Calculate estimated remaining time based on current progress
        estimated_total = int((elapsed / self.progress_percentage) * 100)
        return max(0, estimated_total - elapsed)
    
    def format_progress_bar(self, width: int = 20) -> str:
        """Format a progress bar string."""
        filled = int((self.progress_percentage / 100) * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def format_status_icon(self) -> str:
        """Get status icon for the agent."""
        status_icons = {
            AgentStatus.IDLE: "⏸️",
            AgentStatus.PLANNING: "🎯",
            AgentStatus.IN_PROGRESS: "🟢",
            AgentStatus.WAITING: "🟡",
            AgentStatus.BLOCKED: "🔴",
            AgentStatus.COMPLETED: "✅",
            AgentStatus.ERROR: "❌"
        }
        return status_icons.get(self.status, "❓")


@dataclass
class TaskTiming:
    """Timing information for a task."""
    task_id: str
    task_name: str
    started_at: float
    estimated_duration_ms: int
    phases: Dict[str, float] = field(default_factory=dict)
    
    def add_phase(self, phase_name: str) -> None:
        """Add a timing phase."""
        self.phases[phase_name] = time.time()
    
    @property
    def elapsed_time_ms(self) -> int:
        """Get total elapsed time."""
        return int((time.time() - self.started_at) * 1000)
    
    @property
    def estimated_remaining_ms(self) -> int:
        """Get estimated remaining time."""
        return max(0, self.estimated_duration_ms - self.elapsed_time_ms)
    
    def format_duration(self, duration_ms: int) -> str:
        """Format duration in human-readable format."""
        if duration_ms < 1000:
            return f"{duration_ms}ms"
        elif duration_ms < 60000:
            return f"{duration_ms / 1000:.1f}s"
        else:
            minutes = duration_ms // 60000
            seconds = (duration_ms % 60000) // 1000
            return f"{minutes}m {seconds}s"
    
    def format_elapsed(self) -> str:
        """Format elapsed time."""
        return self.format_duration(self.elapsed_time_ms)
    
    def format_estimated_remaining(self) -> str:
        """Format estimated remaining time."""
        return self.format_duration(self.estimated_remaining_ms)


class ProgressDisplay:
    """
    Real-time progress display system for AgentsMCP operations.
    
    Provides progress bars, timing information, and status updates for
    orchestrator and agent operations with live updates.
    """
    
    def __init__(self, update_callback: Optional[Callable[[str], None]] = None):
        self.update_callback = update_callback
        self.agents: Dict[str, AgentProgress] = {}
        self.task_timing: Optional[TaskTiming] = None
        self.orchestrator_status = "Ready"
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Performance tracking
        self.total_tasks = 0
        self.completed_tasks = 0
        self.average_task_duration_ms = 0
    
    def start_task(self, task_id: str, task_name: str, estimated_duration_ms: int = 60000) -> None:
        """Start tracking a new task."""
        with self.lock:
            self.task_timing = TaskTiming(
                task_id=task_id,
                task_name=task_name,
                started_at=time.time(),
                estimated_duration_ms=estimated_duration_ms
            )
            self.total_tasks += 1
            self.is_running = True
            
            # Start the update thread if not already running
            if not self.update_thread or not self.update_thread.is_alive():
                self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
                self.update_thread.start()
    
    def complete_task(self) -> None:
        """Mark the current task as completed and ensure background thread stops."""
        with self.lock:
            if self.task_timing:
                elapsed = self.task_timing.elapsed_time_ms
                
                # Update performance tracking
                self.completed_tasks += 1
                if self.completed_tasks == 1:
                    self.average_task_duration_ms = elapsed
                else:
                    self.average_task_duration_ms = int(
                        (self.average_task_duration_ms * (self.completed_tasks - 1) + elapsed) / self.completed_tasks
                    )
                
                # Complete all active agents
                for agent in self.agents.values():
                    if agent.status in [AgentStatus.IN_PROGRESS, AgentStatus.PLANNING, AgentStatus.WAITING]:
                        agent.complete()
            
            # Stop the background update thread
            self.is_running = False
        
        # Wait for background thread to finish
        if self.update_thread and self.update_thread.is_alive():
            try:
                self.update_thread.join(timeout=1.0)  # Wait up to 1 second
            except Exception:
                pass  # Ignore join errors
    
    def add_agent(self, agent_id: str, agent_name: str, estimated_duration_ms: int = 30000) -> None:
        """Add a new agent to track."""
        with self.lock:
            self.agents[agent_id] = AgentProgress(
                agent_id=agent_id,
                agent_name=agent_name,
                status=AgentStatus.IDLE,
                estimated_total_duration_ms=estimated_duration_ms
            )
    
    def update_agent_progress(self, agent_id: str, percentage: float, current_step: str = None) -> None:
        """Update progress for an agent."""
        with self.lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                agent.update_progress(percentage, current_step)
                
                # Auto-start if not started yet
                if not agent.started_at and percentage > 0:
                    agent.start()
    
    def start_agent(self, agent_id: str, initial_step: str = None) -> None:
        """Start an agent."""
        with self.lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                agent.start()
                if initial_step:
                    agent.current_step = initial_step
    
    def complete_agent(self, agent_id: str) -> None:
        """Mark an agent as completed."""
        with self.lock:
            if agent_id in self.agents:
                self.agents[agent_id].complete()
    
    def set_agent_error(self, agent_id: str, error_message: str = None) -> None:
        """Mark an agent as errored."""
        with self.lock:
            if agent_id in self.agents:
                self.agents[agent_id].set_error(error_message)
    
    def update_orchestrator_status(self, status: str) -> None:
        """Update orchestrator status."""
        with self.lock:
            self.orchestrator_status = status
    
    def add_task_phase(self, phase_name: str) -> None:
        """Add a task phase for timing."""
        with self.lock:
            if self.task_timing:
                self.task_timing.add_phase(phase_name)
    
    def format_progress_display(self, include_timing: bool = True) -> str:
        """Format the complete progress display with timestamps."""
        import datetime
        
        with self.lock:
            lines = []
            timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
            
            # Task timing information
            if self.task_timing and include_timing:
                elapsed = self.task_timing.format_elapsed()
                remaining = self.task_timing.format_estimated_remaining()
                lines.append(f"{timestamp} ⏱️  Total Task Duration: {elapsed}")
                if self.is_running and self.task_timing.estimated_remaining_ms > 0:
                    lines.append(f"{timestamp} 📅 Estimated Remaining: {remaining}")
            
            # Orchestrator status
            lines.append(f"{timestamp} 🎯 Orchestrator: {self.orchestrator_status}")
            
            # Agent progress bars
            if self.agents:
                lines.append("")  # Separator
                for agent in self.agents.values():
                    progress_bar = agent.format_progress_bar(20)
                    icon = agent.format_status_icon()
                    percentage = f"{agent.progress_percentage:.0f}%"
                    
                    agent_line = f"{timestamp} {icon} {agent.agent_name:<15} {progress_bar} {percentage:>4}"
                    
                    if agent.current_step:
                        step_display = agent.current_step[:30] + "..." if len(agent.current_step) > 30 else agent.current_step
                        agent_line += f" - {step_display}"
                    
                    if agent.status == AgentStatus.IN_PROGRESS and agent.started_at:
                        duration_str = TaskTiming("", "", 0, 0).format_duration(agent.elapsed_time_ms)
                        agent_line += f" ({duration_str})"
                    
                    lines.append(agent_line)
            
            # Performance summary
            if self.completed_tasks > 0:
                lines.append("")  # Separator
                success_rate = (self.completed_tasks / max(1, self.total_tasks)) * 100
                avg_duration = TaskTiming("", "", 0, 0).format_duration(self.average_task_duration_ms)
                timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
                lines.append(f"{timestamp} 📊 Tasks: {self.completed_tasks}/{self.total_tasks} | "
                           f"Success: {success_rate:.1f}% | "
                           f"Avg: {avg_duration}")
            
            return "\n".join(lines)
    
    def format_status_line(self) -> str:
        """Format a compact status line for status bars with timestamps."""
        import datetime
        
        with self.lock:
            timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
            active_agents = sum(1 for a in self.agents.values() 
                              if a.status in [AgentStatus.IN_PROGRESS, AgentStatus.PLANNING])
            completed_agents = sum(1 for a in self.agents.values() 
                                 if a.status == AgentStatus.COMPLETED)
            
            status_parts = [timestamp]
            
            # Task timing
            if self.task_timing and self.is_running:
                elapsed = self.task_timing.format_elapsed()
                status_parts.append(f"⏱️ {elapsed}")
            
            # Agent summary
            if self.agents:
                status_parts.append(f"🤖 {active_agents} active, {completed_agents} done")
            
            # Overall status
            status_parts.append(f"🎯 {self.orchestrator_status}")
            
            return " | ".join(status_parts)
    
    def _update_loop(self) -> None:
        """Background thread loop for real-time updates."""
        while self.is_running:
            try:
                if self.update_callback:
                    display_text = self.format_progress_display()
                    self.update_callback(display_text)
                
                # Update every second
                time.sleep(1.0)
                
            except Exception as e:
                # Log error but continue
                print(f"Progress display update error: {e}")
                time.sleep(1.0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics and timing analysis."""
        with self.lock:
            # Calculate advanced timing metrics
            running_agents = [a for a in self.agents.values() 
                            if a.status in [AgentStatus.IN_PROGRESS, AgentStatus.PLANNING]]
            completed_agents = [a for a in self.agents.values() if a.status == AgentStatus.COMPLETED]
            
            # Average completion times
            avg_completion_time_ms = 0
            if completed_agents:
                total_completion_time = sum(a.actual_duration_ms for a in completed_agents if a.actual_duration_ms > 0)
                avg_completion_time_ms = total_completion_time / len(completed_agents)
            
            # Efficiency metrics
            estimated_vs_actual_ratio = 0
            if completed_agents:
                valid_agents = [a for a in completed_agents 
                              if a.estimated_total_duration_ms > 0 and a.actual_duration_ms > 0]
                if valid_agents:
                    total_estimated = sum(a.estimated_total_duration_ms for a in valid_agents)
                    total_actual = sum(a.actual_duration_ms for a in valid_agents)
                    estimated_vs_actual_ratio = total_actual / total_estimated if total_estimated > 0 else 0
            
            # Current task timing
            current_task_elapsed_ms = 0
            current_task_estimated_remaining_ms = 0
            if self.task_timing and self.is_running:
                current_task_elapsed_ms = self.task_timing.elapsed_time_ms
                current_task_estimated_remaining_ms = self.task_timing.estimated_remaining_ms
            
            return {
                # Basic metrics
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "success_rate": (self.completed_tasks / max(1, self.total_tasks)) * 100,
                "average_task_duration_ms": self.average_task_duration_ms,
                "is_running": self.is_running,
                
                # Agent metrics
                "active_agents": len(running_agents),
                "total_agents": len(self.agents),
                "completed_agents": len(completed_agents),
                
                # Advanced timing analysis
                "average_agent_completion_time_ms": avg_completion_time_ms,
                "estimation_accuracy_ratio": estimated_vs_actual_ratio,
                "estimation_accuracy_percentage": (1 - abs(1 - estimated_vs_actual_ratio)) * 100 if estimated_vs_actual_ratio > 0 else 0,
                
                # Current task metrics
                "current_task_elapsed_ms": current_task_elapsed_ms,
                "current_task_estimated_remaining_ms": current_task_estimated_remaining_ms,
                "current_task_progress_percentage": self._calculate_overall_progress(),
                
                # Performance categories
                "fast_agents": len([a for a in completed_agents if a.actual_duration_ms < 15000]),  # < 15 seconds
                "normal_agents": len([a for a in completed_agents if 15000 <= a.actual_duration_ms < 60000]),  # 15-60 seconds
                "slow_agents": len([a for a in completed_agents if a.actual_duration_ms >= 60000])  # >= 60 seconds
            }
    
    def _calculate_overall_progress(self) -> float:
        """Calculate overall progress across all agents."""
        if not self.agents:
            return 0.0
        
        total_progress = sum(agent.progress_percentage for agent in self.agents.values())
        return total_progress / len(self.agents)
    
    def get_timing_analysis_report(self) -> str:
        """Generate a detailed timing analysis report."""
        stats = self.get_performance_stats()
        
        report_lines = [
            "📊 **Performance Timing Analysis**",
            f"**Overall Success Rate:** {stats['success_rate']:.1f}%",
            f"**Average Task Duration:** {TaskTiming('', '', 0, 0).format_duration(int(stats['average_task_duration_ms']))}",
            "",
            "**Agent Performance Distribution:**",
            f"  🟢 Fast Agents (< 15s): {stats['fast_agents']}",
            f"  🟡 Normal Agents (15-60s): {stats['normal_agents']}",
            f"  🔴 Slow Agents (> 60s): {stats['slow_agents']}",
            "",
            f"**Estimation Accuracy:** {stats['estimation_accuracy_percentage']:.1f}%",
        ]
        
        # Add current task info if running
        if stats['is_running']:
            elapsed = TaskTiming('', '', 0, 0).format_duration(stats['current_task_elapsed_ms'])
            remaining = TaskTiming('', '', 0, 0).format_duration(stats['current_task_estimated_remaining_ms'])
            report_lines.extend([
                "",
                "**Current Task:**",
                f"  ⏱️ Elapsed: {elapsed}",
                f"  📅 Est. Remaining: {remaining}",
                f"  📈 Overall Progress: {stats['current_task_progress_percentage']:.1f}%"
            ])
        
        return "\n".join(report_lines)
    
    def cleanup_completed_agents(self, max_age_minutes: int = 30) -> int:
        """Clean up old completed agents."""
        with self.lock:
            current_time = time.time()
            max_age_seconds = max_age_minutes * 60
            
            agents_to_remove = []
            for agent_id, agent in self.agents.items():
                if (agent.status in [AgentStatus.COMPLETED, AgentStatus.ERROR] and
                    (current_time - agent.last_update) > max_age_seconds):
                    agents_to_remove.append(agent_id)
            
            for agent_id in agents_to_remove:
                del self.agents[agent_id]
            
            return len(agents_to_remove)
    
    def reset(self) -> None:
        """Reset the progress display."""
        with self.lock:
            self.agents.clear()
            self.task_timing = None
            self.orchestrator_status = "Ready"
            self.is_running = False
    
    def cleanup(self) -> None:
        """Clean up all resources and stop background threads."""
        with self.lock:
            self.is_running = False
            
        # Wait for background thread to finish
        if self.update_thread and self.update_thread.is_alive():
            try:
                self.update_thread.join(timeout=2.0)  # Wait up to 2 seconds
            except Exception:
                pass
                
        # Clear all data
        with self.lock:
            self.agents.clear()
            self.task_timing = None
            self.update_callback = None