"""RAG query interface for the :mod:`agentsmcp` framework.

This module adds a thin, well documented, bridge between the generic RAG
components (client, vector store, embeddings, etc.) and agents that
participate in a conversation.  The goal is to provide a *single point of
integration* that is optional – when the ``RAG`` feature flag is disabled the
extra overhead goes away and the agent interacts as if the feature does not
exist.

The interface can be used in two ways:

* **Explicit integration** – the user creates an instance of
  :class:`RAGQueryInterface` and manually calls :meth:`enrich_prompt` before
  passing a prompt to an LLM.
* **Mixin integration** – agents can inherit from
  :class:`RAGAgentMixin`.  The mixin automatically enriches messages before
  they're sent to a language model.

Both paths expose the same tiny, documented surface area which keeps the
implementation future‑proof and testable.

The implementation takes advantage of :mod:`cachetools` for short‑lived
caching and relies on the existing :class:`~agentsmcp.rag.client.RAGClient`
class for the heavy lifting of document retrieval.

Example::

    from agentsmcp.rag import RAGClient
    from agentsmcp.rag.interface import RAGQueryInterface, RAGConfig

    rag_client = RAGClient()  # loads default configuration
    config = RAGConfig(top_k=5, relevance_threshold=0.3)

    interface = RAGQueryInterface(rag_client, config)

    # Enrich a user prompt before sending to an LLM
    enriched_prompt = interface.enrich_prompt("Explain how a transformer works")

In a real agent::

    from agentsmcp.core.agent import BaseAgent
    from agentsmcp.rag.interface import RAGAgentMixin


    class MyAgent(RAGAgentMixin, BaseAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(rag=True, **kwargs)
            # ``self.llm`` should already exist from BaseAgent

        def answer(self, question: str) -> str:
            return self.enrich_and_send(question, self.llm.generate)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Any

try:
    from cachetools import TTLCache
    from cachetools.keys import hashkey
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    # Simple fallback cache
    class TTLCache:
        def __init__(self, maxsize: int, ttl: int):
            self._cache = {}
            self.maxsize = maxsize
            self.ttl = ttl
        
        def __contains__(self, key):
            return key in self._cache
        
        def __getitem__(self, key):
            return self._cache[key]
        
        def __setitem__(self, key, value):
            if len(self._cache) >= self.maxsize:
                # Simple eviction - remove first item
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = value
    
    def hashkey(*args):
        return hash(args)

from .client import RAGClient
from agentsmcp.config import get_config

__all__ = ["RAGQueryInterface", "RAGAgentMixin", "RAGQueryConfig"]

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RAGQueryConfig:
    """Configuration for the query interface.

    Attributes
    ----------
    enabled:
        Flag controlling whether RAG is active.
    expand_query:
        If *True* the original prompt will be fed through a simple query
        expansion step.  The base implementation is trivial but this flag
        allows for later extensions (e.g. synonym generation).
    top_k:
        Number of documents retrieved from the vector store.
    relevance_threshold:
        A lower bound on the *score* returned by the vector store.  Documents
        below this threshold are discarded.  The score is expected to be in
        ``[0, 1]``.
    cache_ttl:
        Time‑to‑live for cached search results, in seconds.
    cache_maxsize:
        Max number of cached queries.
    """

    enabled: bool = True
    expand_query: bool = False
    top_k: int = 3
    relevance_threshold: float = 0.0
    cache_ttl: int = 600
    cache_maxsize: int = 128


class RAGQueryInterface:
    """High‑level wrapper around :class:`~agentsmcp.rag.client.RAGClient`.

    The class is deliberately lightweight; its API is:

    ``enrich_prompt(prompt: str) -> str``
        Performs the full round‑trip: optional query expansion, search, filter,
        context formatting and returns a new prompt that contains a *Relevant
        Context* block.

    The constructor optionally accepts a :class:`RAGClient` instance.  If
    ``None`` a default client will be created.
    """

    def __init__(self, rag_client: Optional[RAGClient] = None, config: Optional[RAGQueryConfig] = None):
        self.client = rag_client or RAGClient()
        
        # Use config from system if not provided
        if config is None:
            try:
                system_config = get_config()
                if hasattr(system_config, 'rag') and hasattr(system_config.rag, 'query_interface'):
                    rag_config = system_config.rag.query_interface
                    config = RAGQueryConfig(
                        enabled=rag_config.enabled,
                        expand_query=rag_config.expand_query,
                        top_k=rag_config.top_k,
                        relevance_threshold=rag_config.relevance_threshold,
                        cache_ttl=rag_config.cache_ttl,
                        cache_maxsize=rag_config.cache_maxsize
                    )
                else:
                    config = RAGQueryConfig()
            except Exception:
                # Fallback to defaults if config loading fails
                log.debug("Failed to load system config for RAG query interface, using defaults")
                config = RAGQueryConfig()
        
        self.cfg = config
        self._cache = TTLCache(maxsize=self.cfg.cache_maxsize, ttl=self.cfg.cache_ttl)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def enrich_prompt(self, prompt: str) -> str:
        """Return a prompt enriched with a context block.

        Parameters
        ----------
        prompt:
            Prompt submitted by a user or an LLM.  If the query interface is
            disabled the prompt is returned unchanged.
        """

        if not self.cfg.enabled:
            return prompt

        query = self._maybe_expand(prompt)
        docs = self._search(query, self.cfg.top_k)
        if not docs:
            return prompt
        context = self._format_context(docs)
        enriched = f"{prompt}\\n\\nRelevant Context:\\n{context}"
        return enriched

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _maybe_expand(self, prompt: str) -> str:
        """Optionally perform a simple query‑expansion step.

        The base class does not implement any sophisticated expansion
        strategy but the function exists to keep the public API stable.
        """

        if not self.cfg.expand_query:
            return prompt
        # Simple placeholder – in an actual implementation you could use a
        # language model or a synonym library.
        return f"{prompt} (expanded)"

    def _search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Execute a vector‑store search with result filtering.

        The method caches results to avoid expensive repeated queries.  The
        cache key is a hash of the query string and the ``k`` value.
        """

        key = hashkey(query, k)
        if key in self._cache:
            log.debug("RAG cache hit for key %s", key)
            return self._cache[key]
        try:
            documents = self.client.search(query, k)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("RAG search failed: %s", exc)
            documents = []
        # Filter on relevance score
        if self.cfg.relevance_threshold > 0:
            documents = [d for d in documents if d.get("score", 0) >= self.cfg.relevance_threshold]
        self._cache[key] = documents
        return documents

    def _format_context(self, docs: Iterable[Dict[str, Any]]) -> str:
        """Return a formatted context string for a list of documents.

        Each document contains ``text`` and optional metadata.
        The default formatting is plain text with a small header.
        """

        formatted = []
        for idx, doc in enumerate(docs, 1):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            header = f"Document {idx}:"
            if metadata:
                # Include relevant metadata like source file
                source = metadata.get("source", "")
                if source:
                    import pathlib
                    header += f" (from {pathlib.Path(source).name})"
                
                score = doc.get("score", 0)
                if score > 0:
                    header += f" [relevance: {score:.3f}]"
            
            formatted.append(f"{header}\\n{text}")
        return "\\n\\n".join(formatted)


class RAGAgentMixin:
    """Mixin that automatically enriches prompts before sending to an LLM.

    The mixin expects the host class to implement a ``generate`` method
    (or a compatible callable) that takes a string and returns a string.
    The example implementation below demonstrates how you might combine a
    base agent with the mixin.

    Usage example::

        class AgentWithRAG(RAGAgentMixin, BaseAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(rag=True, **kwargs)
                # ``self.llm`` is supplied by BaseAgent

            def ask(self, question: str) -> str:
                return self.enrich_and_send(question, self.llm.generate)
    """

    def __init__(self, *, rag: bool = False, rag_config: Optional[RAGQueryConfig] = None, rag_client: Optional[RAGClient] = None, **kwargs):  # pragma: no cover - delegation to super
        super().__init__(**kwargs)
        self._rag_enabled = rag
        self._rag_interface: Optional[RAGQueryInterface] = None
        if rag:
            interface = RAGQueryInterface(rag_client, rag_config)
            self._rag_interface = interface
        elif rag_client is not None:
            log.warning("rag_client supplied but rag flag not enabled – ignoring")

    def enrich_and_send(self, prompt: str, llm_callable) -> str:
        """Enrich *prompt* with context and forward it to *llm_callable*.

        Parameters
        ----------
        prompt:
            Original user query.
        llm_callable:
            Callable accepting a string and returning the LLM's response.
        """

        if self._rag_enabled and self._rag_interface:
            enriched = self._rag_interface.enrich_prompt(prompt)
            log.debug("RAG enriched prompt length: %d -> %d", len(prompt), len(enriched))
        else:
            enriched = prompt
        return llm_callable(enriched)


# Convenience function for non-mixin integration
def enrich_with_rag(prompt: str, rag_client: Optional[RAGClient] = None, config: Optional[RAGQueryConfig] = None) -> str:
    """Standalone function to enrich a prompt with RAG context.
    
    This is useful for one-off integrations where you don't want to use the mixin pattern.
    
    Parameters
    ----------
    prompt:
        Original user prompt
    rag_client:
        Optional RAG client instance (creates default if None)
    config:
        Optional query configuration (uses defaults if None)
        
    Returns
    -------
    str:
        Enriched prompt with relevant context, or original prompt if RAG is disabled
        
    Example
    -------
    >>> enriched = enrich_with_rag("How do transformers work?")
    >>> response = my_llm.generate(enriched)
    """
    interface = RAGQueryInterface(rag_client, config)
    return interface.enrich_prompt(prompt)