"""Context window management for LLM conversations."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import time

logger = logging.getLogger(__name__)


class ContextStatus(Enum):
    """Context window usage status levels."""
    GREEN = "green"  # <70%
    YELLOW = "yellow"  # 70-85%
    RED = "red"  # 85%+


@dataclass
class ProviderContextLimits:
    """Context window limits for different model providers."""
    
    # Anthropic Claude models
    ANTHROPIC_CLAUDE_3_5_SONNET = 200_000
    ANTHROPIC_CLAUDE_3_OPUS = 200_000
    ANTHROPIC_CLAUDE_3_HAIKU = 200_000
    ANTHROPIC_CLAUDE_3_SONNET = 200_000
    
    # OpenAI models
    OPENAI_GPT_4O = 128_000
    OPENAI_GPT_4O_MINI = 128_000
    OPENAI_GPT_4_TURBO = 128_000
    OPENAI_GPT_4 = 32_000
    OPENAI_GPT_3_5_TURBO = 16_000
    
    # OpenRouter (varies by provider)
    OPENROUTER_ANTHROPIC_CLAUDE_3_5_SONNET = 200_000
    OPENROUTER_OPENAI_GPT_4O = 128_000
    
    # Ollama local models (typical ranges)
    OLLAMA_GPT_OSS_20B = 8_000
    OLLAMA_LLAMA_2_70B = 4_000
    OLLAMA_CODELLAMA_34B = 16_000
    OLLAMA_MISTRAL_7B = 8_000
    
    @classmethod
    def get_provider_limits(cls) -> Dict[str, int]:
        """Get all provider context limits as a dictionary."""
        return {
            # Anthropic
            "anthropic/claude-3.5-sonnet": cls.ANTHROPIC_CLAUDE_3_5_SONNET,
            "anthropic/claude-3-opus": cls.ANTHROPIC_CLAUDE_3_OPUS,
            "anthropic/claude-3-haiku": cls.ANTHROPIC_CLAUDE_3_HAIKU,
            "anthropic/claude-3-sonnet": cls.ANTHROPIC_CLAUDE_3_SONNET,
            
            # OpenAI
            "openai/gpt-4o": cls.OPENAI_GPT_4O,
            "openai/gpt-4o-mini": cls.OPENAI_GPT_4O_MINI,
            "openai/gpt-4-turbo": cls.OPENAI_GPT_4_TURBO,
            "openai/gpt-4": cls.OPENAI_GPT_4,
            "openai/gpt-3.5-turbo": cls.OPENAI_GPT_3_5_TURBO,
            
            # OpenRouter
            "openrouter/anthropic/claude-3.5-sonnet": cls.OPENROUTER_ANTHROPIC_CLAUDE_3_5_SONNET,
            "openrouter/openai/gpt-4o": cls.OPENROUTER_OPENAI_GPT_4O,
            
            # Ollama
            "ollama/gpt-oss:20b": cls.OLLAMA_GPT_OSS_20B,
            "ollama/llama2:70b": cls.OLLAMA_LLAMA_2_70B,
            "ollama/codellama:34b": cls.OLLAMA_CODELLAMA_34B,
            "ollama/mistral:7b": cls.OLLAMA_MISTRAL_7B,
        }


@dataclass
class ContextUsage:
    """Context window usage information."""
    current_tokens: int
    max_tokens: int
    percentage: float
    status: ContextStatus
    provider: str
    model: str
    
    @classmethod
    def from_tokens(cls, current_tokens: int, max_tokens: int, provider: str, model: str) -> 'ContextUsage':
        """Create context usage from token counts."""
        percentage = (current_tokens / max_tokens) * 100 if max_tokens > 0 else 0
        
        if percentage < 70:
            status = ContextStatus.GREEN
        elif percentage < 85:
            status = ContextStatus.YELLOW
        else:
            status = ContextStatus.RED
            
        return cls(
            current_tokens=current_tokens,
            max_tokens=max_tokens,
            percentage=percentage,
            status=status,
            provider=provider,
            model=model
        )
    
    def format_usage(self) -> str:
        """Format usage as human-readable string."""
        return f"Context: {self.percentage:.1f}% ({self.current_tokens:,}/{self.max_tokens:,} tokens)"
    
    def format_detailed(self) -> str:
        """Format detailed usage information."""
        status_emoji = {
            ContextStatus.GREEN: "🟢",
            ContextStatus.YELLOW: "🟡", 
            ContextStatus.RED: "🔴"
        }
        
        return (
            f"{status_emoji[self.status]} {self.provider}/{self.model}: "
            f"{self.percentage:.1f}% ({self.current_tokens:,}/{self.max_tokens:,} tokens)"
        )


@dataclass
class CompactionEvent:
    """Information about a context compaction event."""
    timestamp: float
    messages_summarized: int
    tokens_saved: int
    summary: str
    trigger_percentage: float


class ContextManager:
    """Manages context window usage, limits detection, and compaction."""
    
    def __init__(self):
        self.provider_limits = ProviderContextLimits.get_provider_limits()
        self.compaction_events: List[CompactionEvent] = []
        self.compaction_threshold = 80.0  # Trigger compaction at 80%
        self.preserve_recent_messages = 10  # Keep last 10 messages uncompacted
        
    def detect_context_limit(self, provider: str, model: str) -> int:
        """Detect context window limit for a provider/model combination."""
        # Normalize provider/model string
        provider_key = f"{provider.lower()}/{model.lower()}"
        
        # Try exact match first
        if provider_key in self.provider_limits:
            return self.provider_limits[provider_key]
        
        # Try partial matches
        for key, limit in self.provider_limits.items():
            if model.lower() in key.lower() and provider.lower() in key.lower():
                return limit
        
        # Fallbacks based on provider
        provider_lower = provider.lower()
        model_lower = model.lower()
        
        if "anthropic" in provider_lower or "claude" in model_lower:
            return 200_000  # Default for Claude models
        elif "openai" in provider_lower or "gpt" in model_lower:
            return 128_000  # Default for modern GPT models
        elif "ollama" in provider_lower:
            return 8_000   # Conservative default for local models
        
        # Very conservative fallback
        logger.warning(f"Unknown provider/model: {provider}/{model}, using conservative limit")
        return 4_000
    
    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough approximation: 1 token ≈ 4 characters for English text
        # This is conservative and works reasonably for most models
        return len(text) // 4
    
    def calculate_usage(self, messages: List[Any], provider: str, model: str) -> ContextUsage:
        """Calculate current context usage."""
        max_tokens = self.detect_context_limit(provider, model)
        
        # Estimate current token usage
        total_tokens = 0
        for message in messages:
            if hasattr(message, 'content'):
                total_tokens += self.estimate_token_count(str(message.content))
            elif isinstance(message, dict) and 'content' in message:
                total_tokens += self.estimate_token_count(str(message['content']))
            elif isinstance(message, str):
                total_tokens += self.estimate_token_count(message)
        
        return ContextUsage.from_tokens(total_tokens, max_tokens, provider, model)
    
    def should_compact(self, usage: ContextUsage) -> bool:
        """Determine if context should be compacted."""
        return usage.percentage >= self.compaction_threshold
    
    def create_compaction_summary(self, messages: List[Any]) -> str:
        """Create a summary of messages for compaction."""
        # Extract content from messages for summarization
        contents = []
        for msg in messages:
            if hasattr(msg, 'content'):
                contents.append(str(msg.content))
            elif isinstance(msg, dict) and 'content' in msg:
                contents.append(str(msg['content']))
            elif isinstance(msg, str):
                contents.append(msg)
        
        # Create summary (this would ideally use an LLM for better summarization)
        all_content = "\n".join(contents)
        
        # Simple extractive summary for now
        lines = all_content.split('\n')
        important_lines = [
            line for line in lines 
            if any(keyword in line.lower() for keyword in [
                'decision', 'task', 'requirement', 'error', 'important', 
                'critical', 'issue', 'solution', 'result'
            ])
        ]
        
        if len(important_lines) > 20:
            important_lines = important_lines[:20]
        
        summary = "Previous conversation summary:\n" + "\n".join(important_lines)
        
        # Limit summary length
        if len(summary) > 2000:
            summary = summary[:1997] + "..."
            
        return summary
    
    def compact_context(self, messages: List[Any], usage: ContextUsage) -> Tuple[List[Any], CompactionEvent]:
        """Compact context by summarizing older messages."""
        if len(messages) <= self.preserve_recent_messages:
            # Not enough messages to compact
            raise ValueError("Not enough messages to compact")
        
        # Split messages: older (to summarize) and recent (to preserve)
        messages_to_summarize = messages[:-self.preserve_recent_messages]
        recent_messages = messages[-self.preserve_recent_messages:]
        
        # Create summary
        summary = self.create_compaction_summary(messages_to_summarize)
        
        # Calculate tokens saved
        original_tokens = sum(
            self.estimate_token_count(str(getattr(msg, 'content', msg)))
            for msg in messages_to_summarize
        )
        summary_tokens = self.estimate_token_count(summary)
        tokens_saved = original_tokens - summary_tokens
        
        # Create summary message object (adapt based on your message structure)
        if messages and hasattr(messages[0], 'role'):
            # Assuming ChatMessage-like structure
            summary_message_class = type(messages[0])
            if hasattr(messages[0], 'role'):
                summary_msg = summary_message_class(
                    role=getattr(messages[0].role.__class__, 'SYSTEM', 'system'),
                    content=summary,
                    timestamp=time.time(),
                    metadata={'compacted': True, 'original_count': len(messages_to_summarize)}
                )
            else:
                summary_msg = {'role': 'system', 'content': summary, 'metadata': {'compacted': True}}
        else:
            summary_msg = {'role': 'system', 'content': summary, 'metadata': {'compacted': True}}
        
        # Create compacted message list
        compacted_messages = [summary_msg] + recent_messages
        
        # Record compaction event
        compaction_event = CompactionEvent(
            timestamp=time.time(),
            messages_summarized=len(messages_to_summarize),
            tokens_saved=tokens_saved,
            summary=summary,
            trigger_percentage=usage.percentage
        )
        
        self.compaction_events.append(compaction_event)
        
        return compacted_messages, compaction_event
    
    def get_all_provider_limits(self) -> Dict[str, int]:
        """Get all known provider context limits."""
        return self.provider_limits.copy()
    
    def get_compaction_history(self) -> List[CompactionEvent]:
        """Get history of compaction events."""
        return self.compaction_events.copy()
    
    def format_context_status(self, usage: ContextUsage) -> str:
        """Format context status for display."""
        return usage.format_detailed()
    
    def get_context_recommendations(self, usage: ContextUsage) -> List[str]:
        """Get recommendations based on context usage."""
        recommendations = []
        
        if usage.status == ContextStatus.RED:
            recommendations.extend([
                "⚠️  Context usage is critical (85%+)",
                "Consider using /context compact to summarize older messages",
                "Switch to a model with larger context window if available",
                "Break down complex tasks into smaller conversations"
            ])
        elif usage.status == ContextStatus.YELLOW:
            recommendations.extend([
                "⚡ Context usage is high (70-85%)",
                "Monitor usage closely",
                "Prepare for automatic compaction if needed"
            ])
        else:
            recommendations.append("✅ Context usage is healthy (<70%)")
            
        return recommendations