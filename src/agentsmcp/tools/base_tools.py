"""Base tools for OpenAI Agents SDK integration."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseTool(ABC):
    """Base class for all AgentsMCP tools compatible with OpenAI Agents SDK."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def execute(self, *args, **kwargs) -> str:
        """Execute the tool with given parameters."""
        pass

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema(),
            },
        }

    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool parameters."""
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters against schema."""
        # Basic validation - can be extended
        return True


class ToolRegistry:
    """Registry for managing tools across different agents."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self.logger = logging.getLogger(__name__)

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        self.logger.debug(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_tools_for_agent(self, agent_type: str) -> List[BaseTool]:
        """Get tools suitable for a specific agent type."""
        # For now, return all tools - can be extended with agent-specific filtering
        return self.get_all_tools()

    def to_openai_functions(
        self, agent_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Convert registered tools to OpenAI functions format."""
        tools = (
            self.get_tools_for_agent(agent_type) if agent_type else self.get_all_tools()
        )
        return [tool.to_openai_function() for tool in tools]


# Global tool registry
tool_registry = ToolRegistry()
