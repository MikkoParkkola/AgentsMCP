"""
Conversational interface module for AgentsMCP.
Provides natural language interaction with the AgentsMCP platform.
"""

from .conversation import ConversationManager, CommandIntent
from .llm_client import LLMClient, ConversationMessage

__all__ = ['ConversationManager', 'CommandIntent', 'LLMClient', 'ConversationMessage']