from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import Config


@dataclass
class Document:
    """A document in the RAG system."""

    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    """Result from document retrieval."""

    document: Document
    score: float


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.rag_config = config.rag
        self.documents: Dict[str, Document] = {}
        self.embedder = None

    async def initialize(self):
        """Initialize the RAG pipeline."""
        try:
            # Try to import sentence transformers for embeddings
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(self.rag_config.embedding_model)
        except ImportError:
            # Fallback to simple text matching
            self.embedder = None

    async def add_document(
        self, doc_id: str, content: str, metadata: Dict[str, Any] = None
    ) -> Document:
        """Add a document to the RAG system."""
        if metadata is None:
            metadata = {}

        # Create document
        document = Document(id=doc_id, content=content, metadata=metadata)

        # Generate embedding if embedder is available
        if self.embedder:
            document.embedding = self.embedder.encode(content).tolist()

        # Store document
        self.documents[doc_id] = document
        return document

    async def add_documents_from_directory(
        self, directory_path: str, patterns: List[str] = None
    ):
        """Add all documents from a directory."""
        import glob
        import os
        from pathlib import Path

        if patterns is None:
            patterns = ["*.txt", "*.md", "*.py", "*.js", "*.ts"]

        for pattern in patterns:
            for file_path in glob.glob(
                os.path.join(directory_path, "**", pattern), recursive=True
            ):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Create document ID from file path
                    doc_id = str(Path(file_path).relative_to(directory_path))

                    metadata = {
                        "file_path": file_path,
                        "file_type": Path(file_path).suffix,
                        "size": os.path.getsize(file_path),
                    }

                    await self.add_document(doc_id, content, metadata)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    async def retrieve(
        self, query: str, max_results: int = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        if max_results is None:
            max_results = self.rag_config.max_results

        if not self.documents:
            return []

        if self.embedder:
            return await self._embedding_retrieval(query, max_results)
        else:
            return await self._keyword_retrieval(query, max_results)

    async def _embedding_retrieval(
        self, query: str, max_results: int
    ) -> List[RetrievalResult]:
        """Retrieve documents using embedding similarity."""
        import math
        
        # Optional numpy import for enhanced calculations
        try:
            import numpy as np
            HAS_NUMPY = True
        except ImportError:
            HAS_NUMPY = False

        # Generate query embedding
        query_embedding = self.embedder.encode(query)

        results = []
        for document in self.documents.values():
            if document.embedding:
                # Calculate cosine similarity
                if HAS_NUMPY:
                    doc_embedding = np.array(document.embedding)
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                else:
                    # Fallback without numpy
                    doc_embedding = document.embedding
                    dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                    norm_query = math.sqrt(sum(a * a for a in query_embedding))
                    norm_doc = math.sqrt(sum(b * b for b in doc_embedding))
                    similarity = dot_product / (norm_query * norm_doc) if norm_query * norm_doc > 0 else 0.0

                if similarity >= self.rag_config.similarity_threshold:
                    results.append(
                        RetrievalResult(document=document, score=float(similarity))
                    )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    async def _keyword_retrieval(
        self, query: str, max_results: int
    ) -> List[RetrievalResult]:
        """Retrieve documents using keyword matching (fallback)."""
        query_words = set(query.lower().split())

        results = []
        for document in self.documents.values():
            content_words = set(document.content.lower().split())

            # Calculate simple word overlap score
            overlap = len(query_words.intersection(content_words))
            score = overlap / len(query_words) if query_words else 0

            if score > 0:
                results.append(RetrievalResult(document=document, score=score))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    async def generate_context(self, query: str) -> str:
        """Generate context from retrieved documents."""
        retrieved = await self.retrieve(query)

        if not retrieved:
            return ""

        context_parts = []
        for result in retrieved:
            # Chunk the document if it's too long
            chunks = self._chunk_text(result.document.content)

            for chunk in chunks[:2]:  # Limit chunks per document
                context_parts.append(f"Source: {result.document.id}\n{chunk}")

        return "\n\n---\n\n".join(context_parts)

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        words = text.split()
        chunks = []

        chunk_size = self.rag_config.chunk_size
        chunk_overlap = self.rag_config.chunk_overlap

        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i : i + chunk_size]
            chunks.append(" ".join(chunk_words))

        return chunks

    async def search(self, query: str) -> Dict[str, Any]:
        """Perform a full RAG search with context generation."""
        retrieved_docs = await self.retrieve(query)
        context = await self.generate_context(query)

        return {
            "query": query,
            "retrieved_documents": [
                {
                    "id": result.document.id,
                    "score": result.score,
                    "metadata": result.document.metadata,
                }
                for result in retrieved_docs
            ],
            "context": context,
            "document_count": len(self.documents),
        }
