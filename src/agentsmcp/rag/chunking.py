"""
Chunker – turns long documents into manageable pieces - simplified version.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)


class Chunker:
    """Chunk a large string into small, annotated pieces."""

    def __init__(self, cfg: dict):
        self.size = cfg.get("chunk_size", 1000)
        self.overlap = cfg.get("overlap", 200)
        log.debug("Chunker configured: size=%d, overlap=%d", self.size, self.overlap)

    def chunk(self, text: str, base_metadata: dict | None = None) -> List[Tuple[str, dict]]:
        """
        Return a list of (chunk_text, metadata) tuples.
        """
        if not text:
            return []

        base_metadata = base_metadata or {}
        words = text.split()
        chunks = []
        
        # Simple word-based chunking with overlap
        step = max(1, self.size - self.overlap)
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.size]
            if not chunk_words:
                break
                
            chunk_text = ' '.join(chunk_words)
            chunk_meta = {
                **base_metadata,
                'chunk_nr': len(chunks),
                'parent': base_metadata.get('source'),
                'text': chunk_text  # Store original text in metadata
            }
            chunks.append((chunk_text, chunk_meta))
        
        return chunks if chunks else [(text, {**base_metadata, 'chunk_nr': 0, 'text': text})]