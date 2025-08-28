"""
RAG (Retrieval-Augmented Generation) core package.

Provides a small, well-documented facade that ties together the
ingestion, chunking, embedding, and vector-storage layers while
respecting the configuration system installed in *AgentsMCP*.

Typical usage:

    from agentsmcp.rag import RAGClient
    client = RAGClient()
    client.ingest("/path/to/docs")

For agent integration:

    from agentsmcp.rag import RAGQueryInterface, enrich_with_rag
    
    # Explicit integration
    interface = RAGQueryInterface()
    enriched_prompt = interface.enrich_prompt("How do transformers work?")
    
    # One-off function
    enriched_prompt = enrich_with_rag("How do transformers work?")
"""

from .client import RAGClient, RAGError      # Main public API
from .interface import RAGQueryInterface, RAGAgentMixin, RAGQueryConfig, enrich_with_rag

__all__ = ["RAGClient", "RAGError", "RAGQueryInterface", "RAGAgentMixin", "RAGQueryConfig", "enrich_with_rag"]