import logging
import os
from typing import Optional

from ..config import AgentConfig, Config
from ..agents.base import BaseAgent
from ..conversation.llm_client import LLMClient
from ..events import EventBus
from datetime import datetime


class SelfAgent(BaseAgent):
    """Lightweight agent that calls the built-in LLM client directly.

    - Ignores the OpenAI Agents SDK path used by BaseAgent.
    - Honors per-agent provider/model from AgentConfig.
    - Restricts providers to the configured allowlist (default: ollama-turbo).
    """

    def __init__(self, agent_config: AgentConfig, global_config: Config):
        # Do not call BaseAgent.__init__ to avoid OpenAI Agents SDK path
        self.agent_config = agent_config
        self.global_config = global_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.event_bus: EventBus | None = None

    async def execute_task(self, task: str) -> str:
        # Construct an LLM client and force provider/model + allowlist
        llm = LLMClient()
        # Force only the allowed provider
        try:
            if isinstance(llm.config, dict):
                llm.config["providers_enabled"] = ["ollama-turbo"]
        except Exception:
            pass
        # Override provider/model from agent config
        try:
            llm.provider = getattr(self.agent_config, "provider", None) or "ollama-turbo"
        except Exception:
            llm.provider = "ollama-turbo"
        try:
            llm.model = self.agent_config.model or "gpt-oss:120b"
        except Exception:
            llm.model = "gpt-oss:120b"
        # Ensure API key available via env (OLLAMA_API_KEY) or config
        if llm.provider == "ollama-turbo" and not os.getenv("OLLAMA_API_KEY"):
            # Attempt to read from global_config.providers if present
            try:
                prov = self.global_config.providers.get("ollama-turbo")
                if prov and prov.api_key:
                    os.environ.setdefault("OLLAMA_API_KEY", prov.api_key)
            except Exception:
                pass
        prompt = self.get_system_prompt() + "\n\n" + task
        return await llm.send_message(prompt)

    async def cleanup(self):
        # Nothing to cleanup
        pass
