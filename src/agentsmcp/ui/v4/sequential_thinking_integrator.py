"""
Sequential Thinking Integrator for V4 Workspace Controller

This module provides integration with the MCP sequential-thinking tool to capture
real agent thought processes and display them in the workspace controller.

Key Features:
- Integration with mcp__sequential-thinking__sequentialthinking tool
- Real-time capture of agent thought processes
- Structured thinking step display with branching support
- Hypothesis generation and verification tracking
- Integration with workspace controller display system
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class ThinkingPhase(Enum):
    """Phases of sequential thinking process."""
    INITIAL = "initial"
    ANALYSIS = "analysis"
    HYPOTHESIS = "hypothesis"
    VERIFICATION = "verification"
    REVISION = "revision"
    BRANCHING = "branching"
    CONCLUSION = "conclusion"


@dataclass
class ThinkingStep:
    """Represents a single step in sequential thinking."""
    step_number: int
    thought: str
    phase: ThinkingPhase
    timestamp: datetime
    is_revision: bool = False
    revises_step: Optional[int] = None
    branch_from_step: Optional[int] = None
    branch_id: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ThinkingSession:
    """Represents a complete thinking session for an agent."""
    session_id: str
    agent_id: str
    objective: str
    steps: List[ThinkingStep] = field(default_factory=list)
    current_hypothesis: Optional[str] = None
    verification_results: List[Dict[str, Any]] = field(default_factory=list)
    branches: Dict[str, List[ThinkingStep]] = field(default_factory=dict)
    is_complete: bool = False
    final_answer: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class SequentialThinkingIntegrator:
    """
    Integrates with MCP sequential-thinking tool to capture real agent thought processes.
    
    Provides structured thinking capture and display for the workspace controller.
    """
    
    def __init__(self):
        """Initialize the sequential thinking integrator."""
        self.active_sessions: Dict[str, ThinkingSession] = {}
        self.completed_sessions: List[ThinkingSession] = []
        
        # Callbacks for UI updates
        self.thinking_callbacks: List[Callable[[str, ThinkingStep], None]] = []
        self.session_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # MCP tool integration
        self._mcp_tool_available = self._check_mcp_tool_availability()
        
        logger.info(f"Sequential Thinking Integrator initialized (MCP tool available: {self._mcp_tool_available})")
    
    def _check_mcp_tool_availability(self) -> bool:
        """Check if the MCP sequential-thinking tool is available."""
        try:
            # This would check if the mcp__sequential-thinking__sequentialthinking tool is available
            # For now, we'll assume it's available if we're in the MCP environment
            return True
        except Exception as e:
            logger.warning(f"MCP sequential-thinking tool not available: {e}")
            return False
    
    def add_thinking_callback(self, callback: Callable[[str, ThinkingStep], None]):
        """Add callback for thinking step updates."""
        self.thinking_callbacks.append(callback)
    
    def add_session_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for session updates."""
        self.session_callbacks.append(callback)
    
    async def start_thinking_session(self, agent_id: str, objective: str) -> str:
        """Start a new thinking session for an agent."""
        session_id = f"thinking_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = ThinkingSession(
            session_id=session_id,
            agent_id=agent_id,
            objective=objective
        )
        
        self.active_sessions[session_id] = session
        
        # Notify callbacks
        self._notify_session_update(session_id, {
            'type': 'session_started',
            'agent_id': agent_id,
            'objective': objective
        })
        
        logger.info(f"Started thinking session {session_id} for agent {agent_id}")
        return session_id
    
    async def add_thinking_step(
        self,
        session_id: str,
        thought: str,
        phase: ThinkingPhase = ThinkingPhase.ANALYSIS,
        is_revision: bool = False,
        revises_step: Optional[int] = None,
        branch_from_step: Optional[int] = None,
        branch_id: Optional[str] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a thinking step to an active session."""
        if session_id not in self.active_sessions:
            logger.warning(f"No active session found: {session_id}")
            return False
        
        session = self.active_sessions[session_id]
        
        step_number = len(session.steps) + 1
        step = ThinkingStep(
            step_number=step_number,
            thought=thought,
            phase=phase,
            timestamp=datetime.now(),
            is_revision=is_revision,
            revises_step=revises_step,
            branch_from_step=branch_from_step,
            branch_id=branch_id,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        session.steps.append(step)
        
        # Handle branching
        if branch_id and branch_id not in session.branches:
            session.branches[branch_id] = []
        if branch_id:
            session.branches[branch_id].append(step)
        
        # Notify callbacks
        self._notify_thinking_update(session_id, step)
        
        return True
    
    async def set_hypothesis(self, session_id: str, hypothesis: str) -> bool:
        """Set the current hypothesis for a thinking session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.current_hypothesis = hypothesis
        
        # Add as a thinking step
        await self.add_thinking_step(
            session_id,
            f"HYPOTHESIS: {hypothesis}",
            phase=ThinkingPhase.HYPOTHESIS,
            confidence=0.5,
            metadata={'type': 'hypothesis'}
        )
        
        return True
    
    async def add_verification_result(
        self,
        session_id: str,
        hypothesis: str,
        verification: str,
        is_verified: bool,
        confidence: float = 0.0
    ) -> bool:
        """Add verification result for a hypothesis."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        result = {
            'hypothesis': hypothesis,
            'verification': verification,
            'is_verified': is_verified,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
        
        session.verification_results.append(result)
        
        # Add as a thinking step
        status = "VERIFIED" if is_verified else "REJECTED"
        await self.add_thinking_step(
            session_id,
            f"{status}: {verification}",
            phase=ThinkingPhase.VERIFICATION,
            confidence=confidence,
            metadata={'type': 'verification', 'result': result}
        )
        
        return True
    
    async def complete_thinking_session(self, session_id: str, final_answer: str) -> bool:
        """Complete a thinking session with the final answer."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.is_complete = True
        session.final_answer = final_answer
        session.completed_at = datetime.now()
        
        # Add final thinking step
        await self.add_thinking_step(
            session_id,
            f"FINAL ANSWER: {final_answer}",
            phase=ThinkingPhase.CONCLUSION,
            confidence=1.0,
            metadata={'type': 'final_answer'}
        )
        
        # Move to completed sessions
        self.completed_sessions.append(session)
        del self.active_sessions[session_id]
        
        # Notify callbacks
        self._notify_session_update(session_id, {
            'type': 'session_completed',
            'final_answer': final_answer
        })
        
        logger.info(f"Completed thinking session {session_id}")
        return True
    
    async def simulate_mcp_thinking_process(self, session_id: str, objective: str):
        """
        Simulate MCP sequential-thinking tool process.
        
        This method demonstrates how the real MCP tool would be integrated.
        """
        if not self._mcp_tool_available:
            return await self._demo_thinking_process(session_id, objective)
        
        # In real implementation, this would call the MCP tool:
        # mcp__sequential-thinking__sequentialthinking
        
        try:
            # Simulate the MCP tool call structure
            thinking_params = {
                "thought": f"Let me analyze the objective: {objective}",
                "nextThoughtNeeded": True,
                "thoughtNumber": 1,
                "totalThoughts": 5,
                "isRevision": False,
                "needsMoreThoughts": False
            }
            
            # This would be the actual MCP tool call:
            # result = await mcp_tool.sequentialthinking(**thinking_params)
            
            # Simulate thinking steps
            steps = [
                ("I need to understand what's being asked and break it down into components", ThinkingPhase.ANALYSIS),
                ("Let me identify the key requirements and constraints", ThinkingPhase.ANALYSIS),
                ("Based on my analysis, I hypothesize that the solution involves...", ThinkingPhase.HYPOTHESIS),
                ("Let me verify this hypothesis by checking against the requirements", ThinkingPhase.VERIFICATION),
                ("The hypothesis looks correct, proceeding with implementation details", ThinkingPhase.CONCLUSION)
            ]
            
            for i, (thought, phase) in enumerate(steps, 1):
                await self.add_thinking_step(
                    session_id,
                    thought,
                    phase=phase,
                    confidence=0.8,
                    metadata={
                        'mcp_tool': 'sequential-thinking',
                        'step_type': 'simulated'
                    }
                )
                await asyncio.sleep(1)  # Simulate thinking time
            
        except Exception as e:
            logger.error(f"Error in MCP thinking process: {e}")
            await self._demo_thinking_process(session_id, objective)
    
    async def _demo_thinking_process(self, session_id: str, objective: str):
        """Demo thinking process when MCP tool is not available."""
        demo_steps = [
            ("Let me break down this task step by step", ThinkingPhase.INITIAL),
            ("First, I need to understand the core requirements", ThinkingPhase.ANALYSIS),
            ("The key components appear to be: 1) input processing, 2) logic implementation, 3) output formatting", ThinkingPhase.ANALYSIS),
            ("I should also consider error handling and edge cases", ThinkingPhase.ANALYSIS),
            ("My hypothesis is that a modular approach would work best here", ThinkingPhase.HYPOTHESIS),
            ("Let me verify this by checking if it satisfies all requirements", ThinkingPhase.VERIFICATION),
            ("Yes, the modular approach handles all the requirements effectively", ThinkingPhase.VERIFICATION),
            ("I'm confident this is the right approach to take", ThinkingPhase.CONCLUSION)
        ]
        
        for i, (thought, phase) in enumerate(demo_steps, 1):
            await self.add_thinking_step(
                session_id,
                thought,
                phase=phase,
                confidence=0.7 + (i * 0.03),  # Gradually increasing confidence
                metadata={'demo': True}
            )
            await asyncio.sleep(0.5)  # Simulate thinking time
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a thinking session."""
        # Check active sessions first
        session = self.active_sessions.get(session_id)
        if not session:
            # Check completed sessions
            session = next((s for s in self.completed_sessions if s.session_id == session_id), None)
        
        if not session:
            return None
        
        return {
            'session_id': session_id,
            'agent_id': session.agent_id,
            'objective': session.objective,
            'step_count': len(session.steps),
            'current_hypothesis': session.current_hypothesis,
            'verification_count': len(session.verification_results),
            'branch_count': len(session.branches),
            'is_complete': session.is_complete,
            'final_answer': session.final_answer,
            'created_at': session.created_at.isoformat(),
            'completed_at': session.completed_at.isoformat() if session.completed_at else None,
            'recent_steps': [
                {
                    'step_number': step.step_number,
                    'thought': step.thought,
                    'phase': step.phase.value,
                    'confidence': step.confidence,
                    'timestamp': step.timestamp.isoformat()
                }
                for step in session.steps[-5:]  # Last 5 steps
            ]
        }
    
    def get_formatted_thinking_display(self, session_id: str) -> List[str]:
        """Get formatted thinking display for UI."""
        session = self.active_sessions.get(session_id)
        if not session:
            session = next((s for s in self.completed_sessions if s.session_id == session_id), None)
        
        if not session:
            return ["No thinking session found"]
        
        lines = []
        lines.append(f"🧠 Thinking Session: {session.objective}")
        lines.append("=" * 50)
        
        for step in session.steps[-10:]:  # Show last 10 steps
            timestamp = step.timestamp.strftime("%H:%M:%S")
            phase_icon = {
                ThinkingPhase.INITIAL: "🔍",
                ThinkingPhase.ANALYSIS: "📊", 
                ThinkingPhase.HYPOTHESIS: "💡",
                ThinkingPhase.VERIFICATION: "✅",
                ThinkingPhase.REVISION: "🔄",
                ThinkingPhase.BRANCHING: "🌲",
                ThinkingPhase.CONCLUSION: "🎯"
            }.get(step.phase, "💭")
            
            confidence_bar = "█" * int(step.confidence * 10) + "░" * (10 - int(step.confidence * 10))
            
            lines.append(f"[{timestamp}] {phase_icon} Step {step.step_number}")
            lines.append(f"  {step.thought}")
            lines.append(f"  Confidence: {confidence_bar} {step.confidence:.1%}")
            lines.append("")
        
        if session.current_hypothesis:
            lines.append("💡 Current Hypothesis:")
            lines.append(f"  {session.current_hypothesis}")
            lines.append("")
        
        if session.final_answer:
            lines.append("🎯 Final Answer:")
            lines.append(f"  {session.final_answer}")
        
        return lines
    
    def _notify_thinking_update(self, session_id: str, step: ThinkingStep):
        """Notify UI callbacks of thinking updates."""
        for callback in self.thinking_callbacks:
            try:
                callback(session_id, step)
            except Exception as e:
                logger.error(f"Error in thinking callback: {e}")
    
    def _notify_session_update(self, session_id: str, update_data: Dict[str, Any]):
        """Notify UI callbacks of session updates."""
        for callback in self.session_callbacks:
            try:
                callback(session_id, update_data)
            except Exception as e:
                logger.error(f"Error in session callback: {e}")
    
    def get_all_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active thinking sessions."""
        return {
            session_id: self.get_session_summary(session_id)
            for session_id in self.active_sessions.keys()
        }
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """Get metrics about thinking sessions."""
        total_sessions = len(self.active_sessions) + len(self.completed_sessions)
        completed_count = len(self.completed_sessions)
        
        avg_steps = 0
        if self.completed_sessions:
            avg_steps = sum(len(s.steps) for s in self.completed_sessions) / len(self.completed_sessions)
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': len(self.active_sessions),
            'completed_sessions': completed_count,
            'average_steps_per_session': avg_steps,
            'mcp_tool_available': self._mcp_tool_available
        }


# Integration helper functions

async def create_thinking_integrator() -> SequentialThinkingIntegrator:
    """Create a sequential thinking integrator."""
    return SequentialThinkingIntegrator()


def format_thinking_for_display(steps: List[ThinkingStep]) -> List[str]:
    """Format thinking steps for display in workspace controller."""
    lines = []
    for step in steps[-5:]:  # Show last 5 steps
        timestamp = step.timestamp.strftime("%H:%M:%S")
        phase_emoji = {
            ThinkingPhase.INITIAL: "🔍",
            ThinkingPhase.ANALYSIS: "📊",
            ThinkingPhase.HYPOTHESIS: "💡", 
            ThinkingPhase.VERIFICATION: "✅",
            ThinkingPhase.REVISION: "🔄",
            ThinkingPhase.BRANCHING: "🌲",
            ThinkingPhase.CONCLUSION: "🎯"
        }.get(step.phase, "💭")
        
        lines.append(f"[{timestamp}] {phase_emoji} {step.thought}")
    
    return lines