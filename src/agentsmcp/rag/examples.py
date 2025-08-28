"""
RAG integration examples for AgentsMCP.

This module provides concrete examples of how to integrate RAG functionality
with different types of agents in the AgentsMCP ecosystem.
"""

from __future__ import annotations

import logging
from typing import Optional, Callable, Any

from agentsmcp.rag import RAGClient, RAGQueryInterface, RAGAgentMixin, RAGQueryConfig, enrich_with_rag

log = logging.getLogger(__name__)


class SimpleRAGAgent:
    """Example of explicit RAG integration without using the mixin.
    
    This approach gives you full control over when and how RAG is applied.
    """
    
    def __init__(self, llm_callable: Callable[[str], str], rag_enabled: bool = True):
        """
        Parameters
        ----------
        llm_callable:
            Function that takes a prompt string and returns an LLM response
        rag_enabled:
            Whether to enable RAG functionality
        """
        self.llm = llm_callable
        self.rag_enabled = rag_enabled
        
        if rag_enabled:
            # Initialize RAG components
            self.rag_client = RAGClient()
            config = RAGQueryConfig(
                enabled=True,
                top_k=5,
                relevance_threshold=0.2,
                expand_query=True
            )
            self.rag_interface = RAGQueryInterface(self.rag_client, config)
        else:
            self.rag_interface = None
    
    def ask(self, question: str) -> str:
        """Ask a question, optionally enriched with RAG context."""
        if self.rag_enabled and self.rag_interface:
            enriched_question = self.rag_interface.enrich_prompt(question)
            log.info("RAG enrichment: %d -> %d chars", len(question), len(enriched_question))
        else:
            enriched_question = question
        
        return self.llm(enriched_question)
    
    def ask_without_rag(self, question: str) -> str:
        """Ask a question without RAG enrichment (bypass RAG even if enabled)."""
        return self.llm(question)


class ChatAgent(RAGAgentMixin):
    """Example of RAG integration using the mixin pattern.
    
    This is the recommended approach for most use cases as it handles
    the RAG lifecycle automatically.
    """
    
    def __init__(self, llm_callable: Callable[[str], str], rag_enabled: bool = True, **kwargs):
        # Initialize the mixin with RAG configuration
        rag_config = RAGQueryConfig(
            enabled=rag_enabled,
            top_k=3,
            relevance_threshold=0.1
        )
        
        super().__init__(
            rag=rag_enabled,
            rag_config=rag_config,
            **kwargs
        )
        
        self.llm = llm_callable
        self.conversation_history = []
    
    def chat(self, message: str) -> str:
        """Send a message and get a response, with automatic RAG enrichment."""
        # The mixin automatically enriches the prompt if RAG is enabled
        response = self.enrich_and_send(message, self.llm)
        
        # Store conversation history
        self.conversation_history.append({"user": message, "assistant": response})
        
        return response
    
    def get_context_aware_response(self, message: str) -> str:
        """Get a response that considers both RAG context and conversation history."""
        # Build context from conversation history
        context_prompt = self._build_conversation_context(message)
        
        # RAG enrichment happens automatically in enrich_and_send
        return self.enrich_and_send(context_prompt, self.llm)
    
    def _build_conversation_context(self, current_message: str) -> str:
        """Build a prompt that includes recent conversation context."""
        if not self.conversation_history:
            return current_message
        
        # Include last 2 conversation turns for context
        recent_context = []
        for turn in self.conversation_history[-2:]:
            recent_context.append(f"User: {turn['user']}")
            recent_context.append(f"Assistant: {turn['assistant']}")
        
        context = "\\n".join(recent_context)
        return f"Previous conversation:\\n{context}\\n\\nCurrent message: {current_message}"


def demonstrate_simple_integration():
    """Example of one-off RAG integration without classes."""
    
    def mock_llm(prompt: str) -> str:
        return f"Mock response to: {prompt[:50]}..."
    
    # Simple function-based approach
    user_question = "How do neural networks learn?"
    
    # Option 1: Use the convenience function
    enriched_prompt = enrich_with_rag(user_question)
    response = mock_llm(enriched_prompt)
    print(f"Response with RAG: {response}")
    
    # Option 2: Manual integration
    rag_client = RAGClient()
    if rag_client._config.enabled:  # Check if RAG is configured
        search_results = rag_client.search(user_question, k=3)
        if search_results:
            context = "\\n\\n".join([result["text"] for result in search_results[:2]])
            manual_enriched = f"{user_question}\\n\\nRelevant context:\\n{context}"
            response = mock_llm(manual_enriched)
            print(f"Response with manual RAG: {response}")
    else:
        print("RAG not enabled - using direct LLM response")
        response = mock_llm(user_question)
        print(f"Direct response: {response}")


class CodeAssistantAgent(RAGAgentMixin):
    """Specialized agent for code-related queries with RAG.
    
    This example shows how to customize RAG behavior for specific domains.
    """
    
    def __init__(self, llm_callable: Callable[[str], str], **kwargs):
        # Configure RAG specifically for code assistance
        rag_config = RAGQueryConfig(
            enabled=True,
            top_k=5,  # More context for code examples
            relevance_threshold=0.15,  # Lower threshold for code snippets
            expand_query=True  # Help find related code patterns
        )
        
        super().__init__(
            rag=True,
            rag_config=rag_config,
            **kwargs
        )
        
        self.llm = llm_callable
    
    def explain_code(self, code_snippet: str) -> str:
        """Explain a code snippet with relevant documentation context."""
        prompt = f"Please explain this code snippet:\\n\\n```\\n{code_snippet}\\n```"
        return self.enrich_and_send(prompt, self.llm)
    
    def suggest_improvements(self, code_snippet: str, language: str = "python") -> str:
        """Suggest improvements with best practices from documentation."""
        prompt = f"Suggest improvements for this {language} code:\\n\\n```{language}\\n{code_snippet}\\n```\\n\\nConsider performance, readability, and best practices."
        return self.enrich_and_send(prompt, self.llm)
    
    def debug_help(self, error_message: str, code_context: str = "") -> str:
        """Get debugging help with relevant troubleshooting context."""
        prompt = f"Help debug this error:\\n\\nError: {error_message}"
        if code_context:
            prompt += f"\\n\\nCode context:\\n```\\n{code_context}\\n```"
        
        return self.enrich_and_send(prompt, self.llm)


# Usage examples
if __name__ == "__main__":
    # Mock LLM function for demonstration
    def mock_llm_function(prompt: str) -> str:
        return f"Mock LLM response (prompt length: {len(prompt)} chars)"
    
    print("=== SimpleRAGAgent Example ===")
    simple_agent = SimpleRAGAgent(mock_llm_function, rag_enabled=True)
    result1 = simple_agent.ask("What is machine learning?")
    print(f"RAG-enhanced response: {result1}")
    
    print("\\n=== ChatAgent Example ===")
    chat_agent = ChatAgent(mock_llm_function, rag_enabled=True)
    result2 = chat_agent.chat("Explain transformers in AI")
    print(f"Chat response: {result2}")
    
    print("\\n=== Function-based Example ===")
    demonstrate_simple_integration()
    
    print("\\n=== CodeAssistantAgent Example ===")
    code_agent = CodeAssistantAgent(mock_llm_function)
    code_example = "def fibonacci(n):\\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
    result3 = code_agent.explain_code(code_example)
    print(f"Code explanation: {result3}")