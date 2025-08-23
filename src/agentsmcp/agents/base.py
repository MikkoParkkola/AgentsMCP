from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..config import AgentConfig, Config


class BaseAgent(ABC):
    """Base class for all MCP agents."""
    
    def __init__(self, agent_config: AgentConfig, global_config: Config):
        self.agent_config = agent_config
        self.global_config = global_config
        self.tools = []
        
        # Initialize available tools
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize tools based on agent configuration."""
        available_tools = {}
        
        # Load all configured tools
        for tool_config in self.global_config.tools:
            if tool_config.enabled:
                available_tools[tool_config.name] = tool_config
        
        # Filter to only tools this agent can use
        self.tools = [
            available_tools[tool_name] 
            for tool_name in self.agent_config.tools 
            if tool_name in available_tools
        ]
    
    @abstractmethod
    async def execute_task(self, task: str) -> str:
        """Execute a task and return the result."""
        pass
    
    async def cleanup(self):
        """Clean up any resources used by the agent."""
        pass
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return self.agent_config.system_prompt or "You are a helpful AI assistant."
    
    def get_max_tokens(self) -> int:
        """Get the maximum tokens for this agent."""
        return self.agent_config.max_tokens
    
    def get_temperature(self) -> float:
        """Get the temperature for this agent."""
        return self.agent_config.temperature
    
    def get_model(self) -> Optional[str]:
        """Get the model name for this agent."""
        return self.agent_config.model