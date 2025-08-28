"""
RAGClient – high‑level public API.

All user‑facing operations route through this class.  The client
loads the configuration lazily, constructs sub‑components, and
offers helper operations that map nicely to the CLI commands.

A disabled RAG stack is handled gracefully: any attempt to
perform an operation will raise a nicely‑worded exception that
guides the user to enable the feature in the configuration file.
"""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Iterable, Sequence

from agentsmcp.config import get_config

from .ingestion import IngestionPipeline, Document
from .chunking import Chunker
from .embeddings import EmbeddingManager
from .vector_store import VectorStore, FAISSVectorStore

log = logging.getLogger(__name__)

__all__ = ["RAGClient", "RAGError"]


class RAGError(RuntimeError):
    """Raised when RAG functionality is unavailable or mis‑configured."""
    pass


@dataclass(frozen=True)
class _RAGConfig:
    enabled: bool
    embedder: dict
    vector_store: dict
    ingestion: dict
    freshness_policy: dict


class RAGClient:
    """Primary user interface for RAG operations."""

    def __init__(self, config_path: str | None = None):
        """
        Args:
            config_path: Optional custom path to the AgentsMCP
                configuration file – useful in tests.

        Raises:
            RAGError: If the configuration cannot be loaded or parsed.
        """
        log.debug("Initialising RAGClient")
        self._config_loader = get_config()

        # Load the RAG section; fall back to defaults if missing.
        cfg_raw = getattr(self._config_loader, 'rag', None)
        if cfg_raw is None:
            cfg_raw = {'enabled': False}
        
        if hasattr(cfg_raw, 'dict'):
            cfg_dict = cfg_raw.model_dump()
        else:
            cfg_dict = cfg_raw if isinstance(cfg_raw, dict) else {'enabled': False}

        # Pull out the most important knobs.
        self._config = _RAGConfig(
            enabled=cfg_dict.get("enabled", False),
            embedder=cfg_dict.get("embedder", {}),
            vector_store=cfg_dict.get("vector_store", {}),
            ingestion=cfg_dict.get("ingestion", {}),
            freshness_policy=cfg_dict.get("freshness_policy", {}),
        )

        if not self._config.enabled:
            log.info("RAG system disabled in configuration")
            self._ingestor = None
            self._chunker = None
            self._embedder = None
            self._vector_store = None
            return

        log.info("RAG system enabled – building internal components")

        # --- Core sub‑components ------------------------------------
        self._ingestor = IngestionPipeline(self._config.ingestion)
        self._chunker = Chunker(self._config.ingestion)  # Use ingestion config for chunking params
        self._embedder = EmbeddingManager(self._config.embedder)
        self._vector_store = self._build_vector_store()

    # ----------------------------------------------------------------
    # Public API helpers
    # ----------------------------------------------------------------
    def ingest(self, path: str | pathlib.Path | Iterable[str]) -> None:
        """
        Ingest content from one or multiple source paths.

        Parameters
        ----------
        path : str | pathlib.Path | Iterable[str]
            A file, a folder or a list of any of both.  If a URL is
            provided the worker will attempt an HTTP GET.

        Notes
        -----
        This method is idempotent – ingesting the same file multiple
        times will just replace the vector data with new embeddings.
        """
        if not self._config.enabled:
            raise RAGError("Cannot ingest while RAG is disabled")

        paths = path if isinstance(path, Iterable) and not isinstance(path, (str, pathlib.Path)) else [path]
        docs: list[Document] = []

        for src in paths:
            docs.extend(self._ingestor.ingest(src))

        log.debug(f"Ingested {len(docs)} documents")

        # Chunk the incoming text.
        chunks: list[tuple[str, dict]] = []
        for doc in docs:
            for chunk_text, meta in self._chunker.chunk(doc.text, doc.metadata):
                # Augment metadata with the source ID.
                meta.setdefault("source", doc.metadata.get("source"))
                chunks.append((chunk_text, meta))

        # Generate embeddings in batches.
        batch_texts = [text for text, _ in chunks]
        batch_embeddings = self._embedder.embed_texts(batch_texts)

        # Persist to vector store.
        self._vector_store.add_embeddings(
            embeddings=batch_embeddings,
            metadatas=[meta for _, meta in chunks],
        )
        log.info(f"Persisted {len(chunks)} chunks → vector index")

    def list(self) -> Sequence[dict]:
        """
        Return a lightweight representation of the vector index.

        Each record contains at least:
            * id              – internal identifier
            * metadata        – stored during ingestion
            * vector_norm_sq  – optional size optimisation

        Returns
        -------
        Sequence[dict]
        """
        if not self._config.enabled:
            raise RAGError("Cannot list vectors while RAG is disabled")
        return self._vector_store.list()

    def remove(self, vector_id: int | str) -> None:
        """
        Delete a single vector (and its metadata) from the
        underlying storage.

        Parameters
        ----------
        vector_id : int | str
            The internal identifier (index or UUID) returned
            by :func:`list`.
        """
        if not self._config.enabled:
            raise RAGError("Cannot remove vectors while RAG is disabled")
        self._vector_store.delete(vector_id)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Search for relevant documents based on query text.
        
        Parameters
        ----------
        query : str
            Search query text
        k : int
            Number of top results to return
            
        Returns
        -------
        list[dict]
            List of matching documents with metadata and relevance scores
        """
        if not self._config.enabled:
            return []  # Return empty results when RAG disabled
            
        try:
            # Get query embedding
            query_embedding = self._embedder.embed_texts([query])[0]
            
            # Search vector store
            results = self._vector_store.query(query_embedding, k=k)
            
            # Format results
            formatted_results = []
            for idx, score, metadata in results:
                formatted_results.append({
                    'text': metadata.get('text', ''),
                    'metadata': metadata,
                    'score': score,
                    'id': idx
                })
            
            return formatted_results
        except Exception as e:
            log.warning(f"RAG search failed: {e}")
            return []

    # ----------------------------------------------------------------
    # Helper utilities
    # ----------------------------------------------------------------
    def _build_vector_store(self) -> VectorStore:
        """Instantiate the configured vector store implementation."""
        impl = self._config.vector_store.get("backend", "faiss").lower()

        if impl == "faiss":
            # Persist to directory `$HOME/.agentsmcp/rag/faiss`
            data_dir = os.path.expandvars(
                os.path.expanduser(self._config.vector_store.get("path", "~/.agentsmcp/knowl_base/faiss.index"))
            )
            os.makedirs(os.path.dirname(data_dir), exist_ok=True)
            return FAISSVectorStore(os.path.dirname(data_dir))

        # Future: LanceDB / Pinecone / other providers
        raise RAGError(f"Unsupported vector store backend: {impl}")

    # ----------------------------------------------------------------
    # Config toggles; these change the running instance, not the file!
    # ----------------------------------------------------------------
    def enable(self) -> None:
        """Enable the RAG subsystem (in‑memory for the current run)."""
        if self._config.enabled:
            log.warning("RAG already enabled")
            return
        self.__init__()  # Re‑initialise with enabled flag

    def disable(self) -> None:
        """Disable RAG – all future operations will abort."""
        if not self._config.enabled:
            log.warning("RAG already disabled")
            return
        self._config = _RAGConfig(
            enabled=False,
            embedder={},
            vector_store={},
            ingestion={},
            freshness_policy={},
        )
        self._ingestor = None
        self._chunker = None
        self._embedder = None
        self._vector_store = None