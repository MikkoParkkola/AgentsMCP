"""
User Prompt Pre-processor System for AgentsMCP

This module provides comprehensive user prompt preprocessing capabilities including:
- Intent analysis and extraction
- Clarification question generation
- Prompt optimization and enhancement
- Conversation context management
- Confidence scoring and threshold-based processing

The preprocessor acts as the first layer in the orchestrator workflow,
ensuring high-quality, unambiguous prompts before task delegation.
"""

from .intent_analyzer import IntentAnalyzer
from .clarification_engine import ClarificationEngine
from .prompt_optimizer import PromptOptimizer
from .conversation_context import ConversationContext
from .preprocessor import UserPromptPreprocessor

__all__ = [
    'IntentAnalyzer',
    'ClarificationEngine', 
    'PromptOptimizer',
    'ConversationContext',
    'UserPromptPreprocessor'
]

__version__ = "1.0.0"