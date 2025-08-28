"""
Real-time monitoring and status collection for AgentsMCP pipelines.

Provides monitoring infrastructure that bridges the core pipeline engine
with the Rich UI components, supporting both interactive and JSON modes.

Key Components:
- LiveMonitor: Async-compatible pipeline monitor with UI management
- StatusCollector: Thread-safe status aggregation and snapshot creation
- MonitoringConfig: Configuration for monitoring behavior and output modes
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime

from .core import ExecutionTracker


@dataclass
class StatusSnapshot:
    """Immutable snapshot of current pipeline execution status."""
    pipeline_name: str
    global_state: str
    total_stages: int
    completed_stages: int
    failed_stages: int
    stage_details: Dict[str, Dict[str, Any]]
    agent_statuses: Dict[str, Dict[str, Any]]
    start_time: Optional[datetime]
    duration: float
    success_rate: float
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO format
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        return data
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert snapshot to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass 
class MonitoringConfig:
    """Configuration for pipeline monitoring."""
    enable_ui: bool = True
    json_output: bool = False
    json_interval: float = 1.0
    refresh_rate: float = 0.25
    show_agent_details: bool = True
    show_stage_progress: bool = True
    compact_mode: bool = False


class StatusCollector:
    """Thread-safe collector that aggregates pipeline status from ExecutionTracker."""
    
    def __init__(self, tracker: ExecutionTracker, pipeline_name: str = "Pipeline"):
        self.tracker = tracker
        self.pipeline_name = pipeline_name
        self._lock = threading.Lock()
        self._start_time = datetime.now()
    
    async def collect_status(self) -> StatusSnapshot:
        """Collect current status and create a snapshot."""
        # Get status from tracker (should be thread-safe)
        try:
            status_summary = await self.tracker.get_status_summary()
        except Exception:
            # Fallback for synchronous tracker
            status_summary = self._get_sync_status()
        
        with self._lock:
            return self._create_snapshot(status_summary)
    
    def _get_sync_status(self) -> Dict[str, Any]:
        """Get status synchronously as fallback."""
        try:
            # Try to get basic status info
            return {
                "stages": getattr(self.tracker, "_stage_status", {}),
                "agent_results": getattr(self.tracker, "_agent_results", {}),
                "elapsed_time": (datetime.now() - self._start_time).total_seconds()
            }
        except Exception:
            return {"stages": {}, "agent_results": {}, "elapsed_time": 0.0}
    
    def _create_snapshot(self, status_data: Dict[str, Any]) -> StatusSnapshot:
        """Create a status snapshot from raw status data."""
        stages = status_data.get("stages", {})
        agent_results = status_data.get("agent_results", {})
        elapsed_time = status_data.get("elapsed_time", 0.0)
        
        # Calculate stage statistics
        total_stages = len(stages)
        completed_stages = sum(1 for s in stages.values() if s in ["completed", "successful"])
        failed_stages = sum(1 for s in stages.values() if s == "failed")
        
        # Determine global state
        if failed_stages > 0:
            global_state = "failed"
        elif completed_stages == total_stages and total_stages > 0:
            global_state = "completed"
        elif any(s == "running" for s in stages.values()):
            global_state = "running"
        else:
            global_state = "pending"
        
        # Calculate success rate
        success_rate = (completed_stages / max(total_stages, 1)) * 100
        
        # Build stage details
        stage_details = {}
        for stage_name, stage_status in stages.items():
            # Count agents for this stage
            stage_agents = []
            for results_list in agent_results.values():
                if isinstance(results_list, list):
                    stage_agents.extend([
                        r for r in results_list 
                        if getattr(r, 'stage_name', '') == stage_name
                    ])
            
            completed_agents = sum(1 for a in stage_agents if getattr(a, 'success', False))
            failed_agents = len(stage_agents) - completed_agents
            
            stage_details[stage_name] = {
                "status": stage_status,
                "total_agents": len(stage_agents),
                "completed_agents": completed_agents,
                "failed_agents": failed_agents
            }
        
        # Build agent statuses
        agent_statuses = {}
        for stage_name, results_list in agent_results.items():
            if isinstance(results_list, list):
                for result in results_list:
                    agent_name = getattr(result, 'agent_name', 'unknown')
                    agent_statuses[agent_name] = {
                        "stage": stage_name,
                        "success": getattr(result, 'success', False),
                        "duration": getattr(result, 'duration', 0.0),
                        "error": getattr(result, 'error', None)
                    }
        
        return StatusSnapshot(
            pipeline_name=self.pipeline_name,
            global_state=global_state,
            total_stages=total_stages,
            completed_stages=completed_stages,
            failed_stages=failed_stages,
            stage_details=stage_details,
            agent_statuses=agent_statuses,
            start_time=self._start_time,
            duration=elapsed_time,
            success_rate=success_rate
        )


class LiveMonitor:
    """Async-compatible live monitor that manages UI and JSON output."""
    
    def __init__(self, tracker: ExecutionTracker, config: MonitoringConfig, 
                 pipeline_name: str = "Pipeline"):
        self.tracker = tracker
        self.config = config
        self.pipeline_name = pipeline_name
        self.collector = StatusCollector(tracker, pipeline_name)
        
        # UI components (lazy loaded)
        self._ui_monitor = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._json_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
        # Callbacks
        self.on_status_update: Optional[Callable[[StatusSnapshot], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
    
    async def start(self) -> None:
        """Start the live monitoring."""
        if self._monitor_task is not None:
            return  # Already started
        
        # Start UI if enabled
        if self.config.enable_ui:
            await self._start_ui()
        
        # Start JSON output if enabled
        if self.config.json_output:
            self._json_task = asyncio.create_task(self._json_output_loop())
        
        # Start main monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self) -> None:
        """Stop the live monitoring."""
        # Signal stop
        self._stop_event.set()
        
        # Cancel tasks
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        
        if self._json_task:
            self._json_task.cancel()
            try:
                await self._json_task
            except asyncio.CancelledError:
                pass
            self._json_task = None
        
        # Stop UI
        if self._ui_monitor:
            self._ui_monitor.stop()
            self._ui_monitor = None
    
    async def _start_ui(self) -> None:
        """Start the Rich UI monitor."""
        try:
            from .ui import PipelineMonitor
            self._ui_monitor = PipelineMonitor(self.tracker, self.pipeline_name)
            self._ui_monitor.start()
        except ImportError:
            # UI components not available
            self.config.enable_ui = False
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Collect current status
                snapshot = await self.collector.collect_status()
                
                # Call status update callback if provided
                if self.on_status_update:
                    try:
                        self.on_status_update(snapshot)
                    except Exception as e:
                        if self.on_error:
                            self.on_error(e)
                
                # Check if pipeline is finished
                if snapshot.global_state in ["completed", "failed"]:
                    break
                
                await asyncio.sleep(self.config.refresh_rate)
                
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
                break
    
    async def _json_output_loop(self) -> None:
        """JSON output loop for --no-ui mode."""
        while not self._stop_event.is_set():
            try:
                snapshot = await self.collector.collect_status()
                
                # Output JSON snapshot
                print(snapshot.to_json(indent=None))
                
                # Check if finished
                if snapshot.global_state in ["completed", "failed"]:
                    break
                
                await asyncio.sleep(self.config.json_interval)
                
            except Exception as e:
                # Output error as JSON
                error_json = json.dumps({
                    "error": str(e),
                    "type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                })
                print(error_json)
                break
    
    async def get_current_status(self) -> StatusSnapshot:
        """Get current status snapshot."""
        return await self.collector.collect_status()
    
    def set_status_callback(self, callback: Callable[[StatusSnapshot], None]) -> None:
        """Set callback for status updates."""
        self.on_status_update = callback
    
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for errors.""" 
        self.on_error = callback
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class ProgressCalculator:
    """Utility class for calculating pipeline progress metrics."""
    
    @staticmethod
    def calculate_stage_progress(stage_name: str, snapshot: StatusSnapshot) -> Dict[str, Any]:
        """Calculate progress metrics for a specific stage."""
        stage_details = snapshot.stage_details.get(stage_name, {})
        
        total_agents = stage_details.get("total_agents", 0)
        completed_agents = stage_details.get("completed_agents", 0)
        failed_agents = stage_details.get("failed_agents", 0)
        
        progress = (completed_agents + failed_agents) / max(total_agents, 1) * 100
        success_rate = completed_agents / max(total_agents, 1) * 100
        
        return {
            "total_agents": total_agents,
            "completed_agents": completed_agents,
            "failed_agents": failed_agents,
            "progress": progress,
            "success_rate": success_rate,
            "status": stage_details.get("status", "pending")
        }
    
    @staticmethod
    def calculate_overall_progress(snapshot: StatusSnapshot) -> Dict[str, Any]:
        """Calculate overall pipeline progress metrics."""
        total_agents = sum(
            details.get("total_agents", 0) 
            for details in snapshot.stage_details.values()
        )
        completed_agents = sum(
            details.get("completed_agents", 0)
            for details in snapshot.stage_details.values()
        )
        failed_agents = sum(
            details.get("failed_agents", 0)
            for details in snapshot.stage_details.values()
        )
        
        overall_progress = 0.0
        if total_agents > 0:
            overall_progress = ((completed_agents + failed_agents) / total_agents) * 100
        
        return {
            "total_stages": snapshot.total_stages,
            "completed_stages": snapshot.completed_stages,
            "failed_stages": snapshot.failed_stages,
            "total_agents": total_agents,
            "completed_agents": completed_agents,
            "failed_agents": failed_agents,
            "overall_progress": overall_progress,
            "success_rate": snapshot.success_rate,
            "duration": snapshot.duration,
            "global_state": snapshot.global_state
        }
    
    @staticmethod
    def estimate_remaining_time(snapshot: StatusSnapshot) -> Optional[float]:
        """Estimate remaining time based on current progress."""
        if snapshot.duration == 0 or snapshot.success_rate == 0:
            return None
        
        # Simple estimation based on current success rate
        total_progress = snapshot.success_rate
        if total_progress >= 100:
            return 0.0
        
        time_per_percent = snapshot.duration / total_progress
        remaining_percent = 100 - total_progress
        
        return time_per_percent * remaining_percent


def create_monitor(tracker: ExecutionTracker, 
                  pipeline_name: str = "Pipeline",
                  enable_ui: bool = True,
                  json_output: bool = False,
                  json_interval: float = 1.0) -> LiveMonitor:
    """Convenience function to create a configured LiveMonitor."""
    config = MonitoringConfig(
        enable_ui=enable_ui,
        json_output=json_output,
        json_interval=json_interval
    )
    return LiveMonitor(tracker, config, pipeline_name)


# Export main classes
__all__ = [
    "LiveMonitor",
    "StatusCollector", 
    "StatusSnapshot",
    "MonitoringConfig",
    "ProgressCalculator",
    "create_monitor"
]