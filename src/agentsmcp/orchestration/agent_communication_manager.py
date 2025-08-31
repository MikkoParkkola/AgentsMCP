"""
Agent Communication Manager - Implements orchestrator-only communication pattern.

This module ensures that all agent communication flows through the orchestrator,
capturing reasoning, decisions, and providing centralized communication control.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class CommunicationType(Enum):
    """Types of communication events."""
    AGENT_REASONING = "agent_reasoning"
    AGENT_DECISION = "agent_decision"
    AGENT_OUTPUT = "agent_output"
    AGENT_ERROR = "agent_error"
    AGENT_COLLABORATION = "agent_collaboration"
    ORCHESTRATOR_COMMAND = "orchestrator_command"
    USER_INPUT = "user_input"


@dataclass
class CommunicationEvent:
    """A single communication event in the system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    communication_type: CommunicationType = CommunicationType.AGENT_OUTPUT
    source_agent: Optional[str] = None
    target_agent: Optional[str] = None
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning_steps: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None
    session_id: Optional[str] = None


@dataclass 
class AgentReasoningLog:
    """Detailed reasoning log for an agent."""
    agent_id: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    reasoning_chain: List[str] = field(default_factory=list)
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    confidence_levels: List[float] = field(default_factory=list)
    communication_events: List[CommunicationEvent] = field(default_factory=list)
    final_output: Optional[str] = None
    success: Optional[bool] = None
    error_details: Optional[str] = None


class AgentCommunicationManager:
    """
    Manages all communication between agents and the orchestrator.
    
    Implements the orchestrator-only communication pattern where:
    - Users communicate only with the orchestrator
    - Agents communicate only with the orchestrator
    - All communication is logged for retrospective analysis
    - Reasoning and decision-making processes are captured
    """
    
    def __init__(self):
        """Initialize the communication manager."""
        self.communication_history: List[CommunicationEvent] = []
        self.agent_reasoning_logs: Dict[str, AgentReasoningLog] = {}
        self.active_sessions: Set[str] = set()
        self.communication_callbacks: List[Callable[[CommunicationEvent], None]] = []
        self._max_history_size = 10000
        self._lock = asyncio.Lock()
    
    def register_communication_callback(
        self, 
        callback: Callable[[CommunicationEvent], None]
    ) -> None:
        """Register a callback for communication events."""
        self.communication_callbacks.append(callback)
    
    async def start_agent_session(
        self, 
        agent_id: str, 
        session_id: str
    ) -> None:
        """Start a new agent reasoning session."""
        async with self._lock:
            self.active_sessions.add(session_id)
            self.agent_reasoning_logs[agent_id] = AgentReasoningLog(
                agent_id=agent_id,
                session_id=session_id,
                start_time=datetime.now(timezone.utc)
            )
            
            # Log session start
            event = CommunicationEvent(
                communication_type=CommunicationType.ORCHESTRATOR_COMMAND,
                source_agent="orchestrator",
                target_agent=agent_id,
                content=f"Session {session_id} started",
                session_id=session_id,
                metadata={"action": "session_start"}
            )
            await self._record_communication_event(event)
    
    async def end_agent_session(
        self,
        agent_id: str,
        session_id: str,
        success: bool = True,
        final_output: Optional[str] = None,
        error_details: Optional[str] = None
    ) -> AgentReasoningLog:
        """End an agent reasoning session and return the complete log."""
        async with self._lock:
            if agent_id in self.agent_reasoning_logs:
                log = self.agent_reasoning_logs[agent_id]
                log.end_time = datetime.now(timezone.utc)
                log.success = success
                log.final_output = final_output
                log.error_details = error_details
                
                # Log session end
                event = CommunicationEvent(
                    communication_type=CommunicationType.ORCHESTRATOR_COMMAND,
                    source_agent="orchestrator",
                    target_agent=agent_id,
                    content=f"Session {session_id} ended with success={success}",
                    session_id=session_id,
                    metadata={
                        "action": "session_end",
                        "success": success,
                        "output_length": len(final_output or ""),
                        "duration_seconds": (
                            (log.end_time - log.start_time).total_seconds()
                            if log.end_time else 0
                        )
                    }
                )
                await self._record_communication_event(event)
                
                # Remove from active sessions
                self.active_sessions.discard(session_id)
                
                return log
            else:
                logger.warning(f"No reasoning log found for agent {agent_id}")
                return AgentReasoningLog(
                    agent_id=agent_id,
                    session_id=session_id,
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc),
                    success=success,
                    error_details=error_details or "No reasoning log found"
                )
    
    async def log_agent_reasoning(
        self,
        agent_id: str,
        session_id: str,
        reasoning_step: str,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an agent's reasoning step."""
        async with self._lock:
            if agent_id in self.agent_reasoning_logs:
                log = self.agent_reasoning_logs[agent_id]
                log.reasoning_chain.append(reasoning_step)
                if confidence is not None:
                    log.confidence_levels.append(confidence)
                
                # Create communication event
                event = CommunicationEvent(
                    communication_type=CommunicationType.AGENT_REASONING,
                    source_agent=agent_id,
                    target_agent="orchestrator",
                    content=reasoning_step,
                    session_id=session_id,
                    confidence_score=confidence,
                    metadata=metadata or {}
                )
                
                log.communication_events.append(event)
                await self._record_communication_event(event)
    
    async def log_agent_decision(
        self,
        agent_id: str,
        session_id: str,
        decision: str,
        decision_data: Dict[str, Any],
        confidence: Optional[float] = None
    ) -> None:
        """Log an agent's decision."""
        async with self._lock:
            if agent_id in self.agent_reasoning_logs:
                log = self.agent_reasoning_logs[agent_id]
                decision_entry = {
                    "decision": decision,
                    "data": decision_data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "confidence": confidence
                }
                log.decisions_made.append(decision_entry)
                
                # Create communication event
                event = CommunicationEvent(
                    communication_type=CommunicationType.AGENT_DECISION,
                    source_agent=agent_id,
                    target_agent="orchestrator",
                    content=decision,
                    session_id=session_id,
                    confidence_score=confidence,
                    metadata=decision_data
                )
                
                log.communication_events.append(event)
                await self._record_communication_event(event)
    
    async def log_agent_tool_use(
        self,
        agent_id: str,
        session_id: str,
        tool_name: str,
        tool_parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an agent's tool usage."""
        async with self._lock:
            if agent_id in self.agent_reasoning_logs:
                log = self.agent_reasoning_logs[agent_id]
                if tool_name not in log.tools_used:
                    log.tools_used.append(tool_name)
                
                # Create communication event
                event = CommunicationEvent(
                    communication_type=CommunicationType.AGENT_OUTPUT,
                    source_agent=agent_id,
                    target_agent="orchestrator",
                    content=f"Using tool: {tool_name}",
                    session_id=session_id,
                    metadata={
                        "action": "tool_use",
                        "tool_name": tool_name,
                        "parameters": tool_parameters or {}
                    }
                )
                
                log.communication_events.append(event)
                await self._record_communication_event(event)
    
    async def log_agent_output(
        self,
        agent_id: str,
        session_id: str,
        output: str,
        output_type: str = "response",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log agent output."""
        async with self._lock:
            # Create communication event
            event = CommunicationEvent(
                communication_type=CommunicationType.AGENT_OUTPUT,
                source_agent=agent_id,
                target_agent="orchestrator",
                content=output,
                session_id=session_id,
                metadata={
                    "output_type": output_type,
                    **(metadata or {})
                }
            )
            
            # Add to agent log if exists
            if agent_id in self.agent_reasoning_logs:
                log = self.agent_reasoning_logs[agent_id]
                log.communication_events.append(event)
            
            await self._record_communication_event(event)
    
    async def log_agent_error(
        self,
        agent_id: str,
        session_id: str,
        error: str,
        error_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log agent error."""
        async with self._lock:
            event = CommunicationEvent(
                communication_type=CommunicationType.AGENT_ERROR,
                source_agent=agent_id,
                target_agent="orchestrator",
                content=error,
                session_id=session_id,
                metadata=error_details or {}
            )
            
            # Add to agent log if exists
            if agent_id in self.agent_reasoning_logs:
                log = self.agent_reasoning_logs[agent_id]
                log.communication_events.append(event)
            
            await self._record_communication_event(event)
    
    async def get_agent_reasoning_log(
        self, 
        agent_id: str
    ) -> Optional[AgentReasoningLog]:
        """Get the reasoning log for a specific agent."""
        async with self._lock:
            return self.agent_reasoning_logs.get(agent_id)
    
    async def get_all_reasoning_logs(self) -> Dict[str, AgentReasoningLog]:
        """Get all agent reasoning logs."""
        async with self._lock:
            return self.agent_reasoning_logs.copy()
    
    async def get_communication_history(
        self,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        communication_type: Optional[CommunicationType] = None,
        limit: Optional[int] = None
    ) -> List[CommunicationEvent]:
        """Get communication history with optional filtering."""
        async with self._lock:
            events = self.communication_history.copy()
            
            # Apply filters
            if session_id:
                events = [e for e in events if e.session_id == session_id]
            if agent_id:
                events = [e for e in events if e.source_agent == agent_id or e.target_agent == agent_id]
            if communication_type:
                events = [e for e in events if e.communication_type == communication_type]
            
            # Apply limit
            if limit:
                events = events[-limit:]
            
            return events
    
    async def generate_retrospective_report(
        self,
        session_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate a retrospective report of all communications."""
        async with self._lock:
            report = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_events": len(self.communication_history),
                "active_sessions": len(self.active_sessions),
                "total_agents": len(self.agent_reasoning_logs),
                "session_summary": {},
                "agent_performance": {},
                "communication_patterns": {}
            }
            
            # Filter by session IDs if provided
            events = self.communication_history
            if session_ids:
                events = [e for e in events if e.session_id in session_ids]
            
            # Analyze communication patterns
            event_types = {}
            for event in events:
                event_type = event.communication_type.value
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            report["communication_patterns"]["event_distribution"] = event_types
            
            # Analyze agent performance
            for agent_id, log in self.agent_reasoning_logs.items():
                if session_ids and log.session_id not in session_ids:
                    continue
                    
                duration = None
                if log.end_time and log.start_time:
                    duration = (log.end_time - log.start_time).total_seconds()
                
                report["agent_performance"][agent_id] = {
                    "session_id": log.session_id,
                    "success": log.success,
                    "duration_seconds": duration,
                    "reasoning_steps": len(log.reasoning_chain),
                    "decisions_made": len(log.decisions_made),
                    "tools_used": len(log.tools_used),
                    "communication_events": len(log.communication_events),
                    "average_confidence": (
                        sum(log.confidence_levels) / len(log.confidence_levels)
                        if log.confidence_levels else None
                    )
                }
            
            return report
    
    async def _record_communication_event(self, event: CommunicationEvent) -> None:
        """Record a communication event and trigger callbacks."""
        self.communication_history.append(event)
        
        # Trigger callbacks
        for callback in self.communication_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Communication callback failed: {e}")
        
        # Maintain history size limit
        if len(self.communication_history) > self._max_history_size:
            self.communication_history = self.communication_history[-self._max_history_size:]
    
    async def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old session data."""
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        cleaned_count = 0
        
        async with self._lock:
            # Clean communication history
            original_count = len(self.communication_history)
            self.communication_history = [
                event for event in self.communication_history
                if event.timestamp.timestamp() > cutoff
            ]
            cleaned_count += original_count - len(self.communication_history)
            
            # Clean reasoning logs
            old_logs = [
                agent_id for agent_id, log in self.agent_reasoning_logs.items()
                if log.start_time.timestamp() < cutoff
            ]
            for agent_id in old_logs:
                del self.agent_reasoning_logs[agent_id]
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old communication records")
        return cleaned_count


# Global instance for the orchestration system
_communication_manager = AgentCommunicationManager()


def get_communication_manager() -> AgentCommunicationManager:
    """Get the global communication manager instance."""
    return _communication_manager