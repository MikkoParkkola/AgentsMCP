"""
Delegation engine for routing tasks to appropriate agents via envelopes.

Implements the Tier 2 stateless agent function pattern, bridging roles
to concrete agent implementations.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from ..models import TaskEnvelopeV1, ResultEnvelopeV1, EnvelopeStatus
from ..roles import get_role, RoleName


class DelegationEngine:
    """
    Routes tasks to appropriate agents based on role assignments and capabilities.
    
    Acts as the bridge between Tier 1 coordination and Tier 2 agent functions,
    ensuring proper envelope serialization and agent communication.
    """
    
    def __init__(self, agent_manager):
        """Initialize with agent manager for concrete agent execution."""
        self.agent_manager = agent_manager
        
        # Agent type mapping for role preferences
        self.agent_type_mapping = {
            "codex": "codex",
            "claude": "claude", 
            "ollama": "ollama"
        }
    
    async def execute_task(
        self,
        task: TaskEnvelopeV1,
        role_name: RoleName,
        timeout: Optional[int] = None,
        max_retries: int = 2
    ) -> ResultEnvelopeV1:
        """
        Execute task via role-based agent delegation.
        
        Implements retry logic and fallback agent selection following
        AGENTS.md v2 patterns.
        """
        role = get_role(role_name)
        
        # Get preferred agent type from role
        preferred_agent = role.preferred_agent_type()
        
        # Build agent selection list (primary + fallbacks)
        agent_candidates = self._build_agent_candidates(preferred_agent)
        
        last_error = None
        
        for attempt in range(max_retries):
            for agent_type in agent_candidates:
                try:
                    result = await self._execute_with_agent(
                        task, role, agent_type, timeout
                    )
                    
                    if result.status == EnvelopeStatus.SUCCESS:
                        return result
                    else:
                        last_error = Exception(result.notes or "Agent execution failed")
                        
                except Exception as e:
                    last_error = e
                    continue
        
        # All attempts failed
        return ResultEnvelopeV1(
            status=EnvelopeStatus.ERROR,
            notes=f"All delegation attempts failed. Last error: {str(last_error)}",
            confidence=0.0,
            metrics={"attempts": max_retries * len(agent_candidates)}
        )
    
    async def _execute_with_agent(
        self,
        task: TaskEnvelopeV1,
        role,
        agent_type: str,
        timeout: Optional[int]
    ) -> ResultEnvelopeV1:
        """Execute task with specific agent type."""
        
        # Use role's async execute method if available (BaseRole)
        if hasattr(role, 'execute'):
            try:
                return await role.execute(
                    task, 
                    self.agent_manager,
                    timeout=timeout,
                    max_retries=1
                )
            except Exception as e:
                return ResultEnvelopeV1(
                    status=EnvelopeStatus.ERROR,
                    notes=f"Role execution failed: {str(e)}",
                    confidence=0.0
                )
        
        # Fallback: use agent manager directly
        return await self._execute_via_agent_manager(
            task, role, agent_type, timeout
        )
    
    async def _execute_via_agent_manager(
        self,
        task: TaskEnvelopeV1,
        role,
        agent_type: str,
        timeout: Optional[int]
    ) -> ResultEnvelopeV1:
        """Execute via AgentManager with envelope serialization."""
        
        # Build prompt from task envelope
        prompt = self._build_prompt_from_task(task, role)
        
        try:
            # Spawn agent job
            job_id = await self.agent_manager.spawn_agent(
                agent_type=agent_type,
                task=prompt,
                timeout=timeout or 300
            )
            
            # Wait for completion
            status = await self.agent_manager.wait_for_completion(job_id)
            
            # Convert AgentsMCP status to ResultEnvelope
            if status.state.value == "completed":
                return ResultEnvelopeV1(
                    status=EnvelopeStatus.SUCCESS,
                    artifacts={"output": status.output or "", "job_id": job_id},
                    confidence=0.8,
                    notes=f"Completed by {agent_type} agent"
                )
            else:
                return ResultEnvelopeV1(
                    status=EnvelopeStatus.ERROR,
                    artifacts={"job_id": job_id, "agent_error": status.error},
                    confidence=0.0,
                    notes=f"Agent job {status.state.value}: {status.error}"
                )
                
        except Exception as e:
            return ResultEnvelopeV1(
                status=EnvelopeStatus.ERROR,
                notes=f"Agent manager error: {str(e)}",
                confidence=0.0
            )
    
    def _build_agent_candidates(self, preferred_agent: str) -> list[str]:
        """Build ordered list of agent candidates with fallbacks."""
        candidates = [preferred_agent]
        
        # Add fallbacks based on AGENTS.md v2 routing strategy
        if preferred_agent == "codex":
            candidates.extend(["claude", "ollama"])
        elif preferred_agent == "claude":
            candidates.extend(["codex", "ollama"])
        elif preferred_agent == "ollama":
            candidates.extend(["codex", "claude"])
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(candidates))
    
    def _build_prompt_from_task(self, task: TaskEnvelopeV1, role) -> str:
        """Convert TaskEnvelope to structured prompt for agents."""
        prompt_parts = [
            f"Role: {role.name().value}",
            f"Objective: {task.objective}"
        ]
        
        if task.bounded_context:
            prompt_parts.append(f"Context: {task.bounded_context}")
        
        if task.constraints:
            prompt_parts.append(f"Constraints: {'; '.join(task.constraints)}")
        
        if task.inputs:
            prompt_parts.append(f"Inputs: {task.inputs}")
        
        if task.output_schema:
            prompt_parts.append(f"Expected output format: {task.output_schema}")
        
        # Add role-specific guidance
        if hasattr(role, 'responsibilities'):
            responsibilities = role.responsibilities()
            if responsibilities:
                prompt_parts.append(f"Responsibilities: {'; '.join(responsibilities)}")
        
        prompt_parts.append("Provide detailed, production-ready results.")
        
        return "\n".join(prompt_parts)