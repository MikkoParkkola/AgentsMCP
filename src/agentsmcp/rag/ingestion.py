"""
Document ingestion engine - simplified version.

Supports basic file reading and directory traversal.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import pathlib
from dataclasses import dataclass
from typing import Iterable, List

log = logging.getLogger(__name__)

__all__ = ["Document", "IngestionPipeline"]


@dataclass
class Document:
    """Represents a user supplied text file."""
    text: str           # Raw text after basic post‑processing
    metadata: dict      # Source information + timestamps


class IngestionPipeline:
    """Ingestion driver – resolves a path to its constituent documents."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def ingest(self, path: str | pathlib.Path | Iterable[str]) -> List[Document]:
        """
        Accept one or many paths and return Document objects.
        """
        items = path if isinstance(path, Iterable) and not isinstance(path, (str, pathlib.Path)) else [path]
        docs: List[Document] = []

        for entry in items:
            entry_path = pathlib.Path(os.path.expanduser(str(entry)))
            log.debug("Ingesting %s", entry_path)

            if entry_path.is_file():
                docs.extend(self._ingest_file(entry_path))
            elif entry_path.is_dir():
                docs.extend(self._ingest_directory(entry_path))
            else:
                log.warning("Skipping unknown path: %s", entry_path)

        return docs

    def _ingest_file(self, path: pathlib.Path) -> List[Document]:
        """Ingest a single file."""
        try:
            # Skip large files
            max_size = self.cfg.get("max_file_bytes", 5_000_000)
            if path.stat().st_size > max_size:
                log.info("Skipping large file %s", path)
                return []

            # Read text files
            if path.suffix.lower() in {'.txt', '.md', '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h'}:
                text = path.read_text(encoding='utf-8', errors='ignore')
                metadata = {
                    'source': str(path.resolve()),
                    'mtime': path.stat().st_mtime,
                    'size': path.stat().st_size,
                    'media_type': mimetypes.guess_type(str(path))[0] or 'text/plain'
                }
                return [Document(text, metadata)]
            else:
                log.debug("No handler for %s – skipping", path)
                return []
                
        except Exception as e:
            log.warning("Failed to ingest %s: %s", path, e)
            return []

    def _ingest_directory(self, dir_path: pathlib.Path) -> List[Document]:
        """Recursively ingest all files in directory."""
        docs: List[Document] = []
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                docs.extend(self._ingest_file(file_path))
        return docs