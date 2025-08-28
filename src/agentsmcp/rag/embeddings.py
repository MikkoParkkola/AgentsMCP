"""
Embedding manager – simplified version with fallbacks.
"""

from __future__ import annotations

import logging
import random
from typing import Iterable, List

log = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding generation with optional dependencies."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_name = cfg.get("model", "all-MiniLM-L6-v2")
        self.batch_size = cfg.get("batch_size", 128)
        self._model = None

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Return list of vectors matching input ordering."""
        text_list = list(texts)
        if not text_list:
            return []

        # Try sentence-transformers first
        try:
            return self._embed_sentence_transformers(text_list)
        except ImportError:
            log.warning("sentence-transformers not available, using random embeddings")
            return self._embed_random(text_list)
        except Exception as e:
            log.warning(f"Embedding failed: {e}, using random embeddings")
            return self._embed_random(text_list)

    def _embed_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """Use sentence-transformers for embeddings."""
        from sentence_transformers import SentenceTransformer
        
        if self._model is None:
            log.info("Loading sentence‑transformers model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)

        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=False,
        )
        return [vec.tolist() for vec in vectors]

    def _embed_random(self, texts: List[str]) -> List[List[float]]:
        """Fallback: generate random embeddings for testing."""
        # Generate deterministic random embeddings based on text hash
        embeddings = []
        for text in texts:
            random.seed(hash(text) % (2**32))
            embedding = [random.uniform(-1, 1) for _ in range(384)]  # Standard embedding size
            embeddings.append(embedding)
        return embeddings

    def close(self) -> None:
        """Release any expensive resources."""
        self._model = None