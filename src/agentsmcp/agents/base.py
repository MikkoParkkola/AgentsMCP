import logging
import os
from abc import ABC
from typing import Any, List, Optional

from agents import Agent, Runner
from openai import OpenAI

from ..config import AgentConfig, Config, ProviderType
from ..tools import tool_registry
from ..tools.mcp_tool import MCPCallTool


class BaseAgent(ABC):
    """Base class for all MCP agents using OpenAI Agents SDK."""

    def __init__(self, agent_config: AgentConfig, global_config: Config):
        self.agent_config = agent_config
        self.global_config = global_config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize OpenAI-compatible client (supports custom base URLs)
        self.client = OpenAI(
            api_key=self._get_api_key(),
            base_url=self._get_api_base(),
        )

        # Initialize the OpenAI Agent
        self.openai_agent = self._create_openai_agent()

        # Initialize available tools
        self.tools = self._initialize_tools()

    def _get_api_key(self) -> str:
        """Resolve API key based on provider with sensible fallbacks."""
        # Agent-specific env override wins
        if self.agent_config.api_key_env and os.getenv(self.agent_config.api_key_env):
            return os.getenv(self.agent_config.api_key_env)  # type: ignore[return-value]

        prov = getattr(self.agent_config, "provider", ProviderType.OPENAI)
        # Per-agent type override env
        per_agent_env = os.getenv(f"AGENTSMCP_{self.agent_config.type.upper()}_API_KEY")
        if per_agent_env:
            return per_agent_env

        # Global providers config
        try:
            prov_cfg = self.global_config.providers.get(prov.value)  # type: ignore[attr-defined]
        except Exception:
            prov_cfg = None
        if prov_cfg and getattr(prov_cfg, "api_key", None):
            return prov_cfg.api_key  # type: ignore[return-value]

        if prov == ProviderType.OPENAI:
            key = os.getenv("OPENAI_API_KEY")
        elif prov == ProviderType.OPENROUTER:
            # OpenRouter uses an OpenAI-compatible API surface
            # prefer OPENROUTER_API_KEY but allow OPENAI_API_KEY as fallback in dev
            key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        elif prov == ProviderType.OLLAMA:
            # Local default requires no key
            key = os.getenv("OLLAMA_API_KEY", "local")
        else:
            key = os.getenv("OPENAI_API_KEY")

        if not key:
            raise ValueError(
                f"No API key found for provider={prov.value}. Set OPENAI_API_KEY/OPENROUTER_API_KEY or configure api_key_env."
            )
        return key

    def _get_api_base(self) -> str | None:
        prov = getattr(self.agent_config, "provider", ProviderType.OPENAI)
        if self.agent_config.api_base:
            return self.agent_config.api_base
        # Global providers config
        try:
            prov_cfg = self.global_config.providers.get(prov.value)  # type: ignore[attr-defined]
        except Exception:
            prov_cfg = None
        if prov_cfg and getattr(prov_cfg, "api_base", None):
            return prov_cfg.api_base
        if prov == ProviderType.OPENROUTER:
            return "https://openrouter.ai/api/v1"
        # Default None => client default base
        return None

    def _create_openai_agent(self) -> Agent:
        """Create the OpenAI Agent instance."""
        return Agent(
            name=self.agent_config.type.title(),
            instructions=self.get_system_prompt(),
            model=self.get_model() or "gpt-4",
            tools=self._get_agent_tools(),
        )

    def _initialize_tools(self) -> List[Any]:
        """Initialize tools based on agent configuration."""
        available_tools = {}

        # Load all configured tools
        for tool_config in self.global_config.tools:
            if tool_config.enabled:
                available_tools[tool_config.name] = tool_config

        # Filter to only tools this agent can use
        return [
            available_tools[tool_name]
            for tool_name in self.agent_config.tools
            if tool_name in available_tools
        ]

    def _get_agent_tools(self) -> List[Any]:
        """Get tools for the OpenAI Agent. Combines registry tools with agent-specific tools."""
        tools = []

        # Get tools from the global registry
        registry_tools = tool_registry.get_tools_for_agent(self.agent_config.type)
        tools.extend(registry_tools)

        # Add any agent-specific tools (implemented in subclasses)
        agent_tools = self._get_custom_agent_tools()
        tools.extend(agent_tools)

        # If MCP servers are configured, expose the generic mcp_call tool.
        try:
            configured_servers = [s.name for s in getattr(self.global_config, "mcp", [])]
        except Exception:
            configured_servers = []

        if configured_servers:
            # If the agent has an explicit allowlist, respect it; otherwise allow all enabled servers.
            allowed = self.agent_config.mcp if getattr(self.agent_config, "mcp", None) else configured_servers
            tools.append(MCPCallTool(self.global_config, allowed_servers=allowed))

        self.logger.debug(
            f"Loaded {len(tools)} tools for {self.agent_config.type} agent"
        )
        return tools

    def _get_custom_agent_tools(self) -> List[Any]:
        """Get custom tools specific to this agent type. Override in subclasses."""
        return []

    async def execute_task(self, task: str) -> str:
        """Execute a task using the OpenAI Agents SDK."""
        # Test/Mock mode: avoid network calls and return deterministic output for local/CI tests
        if os.getenv("AGENTSMCP_TEST_MODE") == "1":
            return await self._simulate(task)
        try:
            self.logger.info(f"Executing task with {self.agent_config.type} agent")

            # Run the agent with the task
            result = await Runner.run(self.openai_agent, task)

            # Extract the final output
            output = getattr(result, "final_output", str(result))

            self.logger.info("Task completed successfully")
            return output

        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            raise Exception(f"{self.agent_config.type} agent error: {str(e)}")

    async def cleanup(self):
        """Clean up any resources used by the agent."""
        self.logger.debug(f"Cleaning up {self.agent_config.type} agent resources")
        # OpenAI Agents SDK handles cleanup automatically
        pass

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return self.agent_config.system_prompt or self._get_default_instructions()

    def _get_default_instructions(self) -> str:
        """Get default instructions for this agent type."""
        return "You are a helpful AI assistant."

    def get_max_tokens(self) -> int:
        """Get the maximum tokens for this agent."""
        return self.agent_config.max_tokens

    def get_temperature(self) -> float:
        """Get the temperature for this agent."""
        return self.agent_config.temperature

    def get_model(self) -> Optional[str]:
        """Get the model name for this agent."""
        # Explicit override on spawn takes precedence
        if self.agent_config.model:
            return self.agent_config.model
        # Otherwise pick from priority list when present
        if getattr(self.agent_config, "model_priority", None):
            return self.agent_config.model_priority[0]
        return None

    async def _simulate(self, task: str) -> str:
        """Default simulation output in test mode. Subclasses should override for specificity."""
        model = self.get_model() or "unknown-model"
        return f"{self.agent_config.type.title()} Agent Simulation: {task} (model {model})"
