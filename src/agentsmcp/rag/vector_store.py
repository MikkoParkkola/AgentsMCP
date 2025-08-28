"""
Vector storage backends - simplified version with FAISS support.

Provides abstract interface and FAISS implementation for similarity search.
"""

from __future__ import annotations

import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple, Any

log = logging.getLogger(__name__)

__all__ = ["VectorStore", "FAISSVectorStore"]


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    def add_embeddings(self, embeddings: List[List[float]], metadatas: List[dict]) -> None:
        """Add embeddings with metadata to the store."""
        pass

    @abstractmethod
    def query(self, query_embedding: List[float], k: int = 5) -> List[Tuple[int, float, dict]]:
        """Query for similar embeddings.
        
        Returns:
            List of (index, similarity_score, metadata) tuples
        """
        pass

    @abstractmethod
    def delete(self, vector_id: int | str) -> None:
        """Delete a vector by ID."""
        pass

    @abstractmethod
    def list(self) -> List[dict]:
        """List all stored vectors with metadata."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector storage with persistence."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.index_path = os.path.join(data_dir, "faiss.index")
        self.metadata_path = os.path.join(data_dir, "metadata.pkl")
        
        self._index = None
        self._metadatas = []
        self._dimension = None
        
        # Ensure directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing data
        self._load_index()

    def add_embeddings(self, embeddings: List[List[float]], metadatas: List[dict]) -> None:
        """Add embeddings with metadata to FAISS index."""
        if not embeddings:
            return
            
        try:
            import faiss
            import numpy as np
        except ImportError:
            log.warning("FAISS not available, falling back to simple storage")
            return self._add_simple(embeddings, metadatas)
        
        # Initialize index if needed
        if self._index is None:
            self._dimension = len(embeddings[0])
            self._index = faiss.IndexFlatL2(self._dimension)
            log.info(f"Created FAISS index with dimension {self._dimension}")
        
        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        
        # Add to index
        self._index.add(vectors)
        
        # Store metadata
        self._metadatas.extend(metadatas)
        
        # Persist to disk
        self._save_index()
        
        log.debug(f"Added {len(embeddings)} vectors to FAISS index")

    def query(self, query_embedding: List[float], k: int = 5) -> List[Tuple[int, float, dict]]:
        """Query FAISS index for similar vectors."""
        if self._index is None or self._index.ntotal == 0:
            return []
            
        try:
            import faiss
            import numpy as np
        except ImportError:
            log.warning("FAISS not available for querying")
            return []
        
        # Convert query to numpy array
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Search
        k = min(k, self._index.ntotal)  # Don't ask for more than we have
        distances, indices = self._index.search(query_vector, k)
        
        # Format results
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self._metadatas):
                # Convert L2 distance to similarity score (lower distance = higher similarity)
                similarity = 1.0 / (1.0 + dist)
                results.append((int(idx), float(similarity), self._metadatas[idx]))
        
        return results

    def delete(self, vector_id: int | str) -> None:
        """Delete vector - FAISS doesn't support deletion, so we mark as deleted."""
        try:
            idx = int(vector_id)
            if 0 <= idx < len(self._metadatas):
                # Mark as deleted in metadata
                self._metadatas[idx]["_deleted"] = True
                self._save_index()
                log.debug(f"Marked vector {idx} as deleted")
        except (ValueError, IndexError) as e:
            log.warning(f"Failed to delete vector {vector_id}: {e}")

    def list(self) -> List[dict]:
        """List all vectors with metadata."""
        results = []
        for i, metadata in enumerate(self._metadatas):
            if not metadata.get("_deleted", False):
                results.append({
                    "id": i,
                    "metadata": metadata,
                    "vector_norm_sq": metadata.get("_norm_sq", 0.0)
                })
        return results

    def _add_simple(self, embeddings: List[List[float]], metadatas: List[dict]) -> None:
        """Fallback storage without FAISS."""
        # Just store metadata - no vector operations possible
        self._metadatas.extend(metadatas)
        self._save_index()
        log.info(f"Added {len(embeddings)} vectors to simple storage (no similarity search)")

    def _load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        try:
            import faiss
        except ImportError:
            log.debug("FAISS not available, using simple storage")
            return self._load_simple()
        
        # Load FAISS index
        if os.path.exists(self.index_path):
            try:
                self._index = faiss.read_index(self.index_path)
                self._dimension = self._index.d
                log.debug(f"Loaded FAISS index with {self._index.ntotal} vectors")
            except Exception as e:
                log.warning(f"Failed to load FAISS index: {e}")
                self._index = None
        
        # Load metadata
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    self._metadatas = pickle.load(f)
                log.debug(f"Loaded {len(self._metadatas)} metadata entries")
            except Exception as e:
                log.warning(f"Failed to load metadata: {e}")
                self._metadatas = []

    def _load_simple(self) -> None:
        """Load metadata without FAISS."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    self._metadatas = pickle.load(f)
                log.debug(f"Loaded {len(self._metadatas)} metadata entries (simple mode)")
            except Exception as e:
                log.warning(f"Failed to load metadata: {e}")
                self._metadatas = []

    def _save_index(self) -> None:
        """Persist FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            if self._index is not None:
                try:
                    import faiss
                    faiss.write_index(self._index, self.index_path)
                    log.debug(f"Saved FAISS index to {self.index_path}")
                except ImportError:
                    pass  # FAISS not available
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self._metadatas, f)
            log.debug(f"Saved {len(self._metadatas)} metadata entries")
        except Exception as e:
            log.warning(f"Failed to save index: {e}")
