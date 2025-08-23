import asyncio
import json
from typing import Dict, Any

from .base import BaseAgent
from ..config import AgentConfig, Config


class CodexAgent(BaseAgent):
    """Codex agent for code generation and analysis tasks."""
    
    def __init__(self, agent_config: AgentConfig, global_config: Config):
        super().__init__(agent_config, global_config)
        self.session_id = None
    
    async def execute_task(self, task: str) -> str:
        """Execute a task using Codex via MCP."""
        try:
            # Import MCP codex module
            from mcp import codex
            
            # Use Codex with the configured parameters
            config = {
                "model": self.get_model(),
                "base-instructions": self.get_system_prompt(),
                "approval-policy": "on-request",
                "sandbox": "workspace-write"
            }
            
            if self.session_id:
                # Continue existing session
                result = await codex.codex_reply({
                    "sessionId": self.session_id,
                    "prompt": task
                })
            else:
                # Start new session
                result = await codex.codex({
                    "prompt": task,
                    **config
                })
                
                # Extract session ID for follow-up
                if "sessionId" in result:
                    self.session_id = result["sessionId"]
            
            return result.get("response", "Task completed successfully")
            
        except ImportError:
            # Fallback if MCP codex is not available
            return await self._simulate_codex_response(task)
        except Exception as e:
            raise Exception(f"Codex agent error: {str(e)}")
    
    async def _simulate_codex_response(self, task: str) -> str:
        """Simulate a Codex response for development/testing."""
        return f"[Codex Simulation] Task: {task}\nResponse: This would be handled by the Codex agent with model {self.get_model()}"
    
    async def cleanup(self):
        """Clean up Codex session."""
        self.session_id = None