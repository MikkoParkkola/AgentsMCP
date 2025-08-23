import pytest
import asyncio
from pathlib import Path
import tempfile

from agentsmcp.config import Config, RAGConfig
from agentsmcp.rag.pipeline import RAGPipeline, Document


@pytest.fixture
def config():
    """Create test configuration."""
    config = Config()
    config.rag = RAGConfig(
        chunk_size=100,
        chunk_overlap=10,
        max_results=5,
        similarity_threshold=0.5
    )
    return config


@pytest.fixture
async def rag_pipeline(config):
    """Create RAG pipeline for testing."""
    pipeline = RAGPipeline(config)
    await pipeline.initialize()
    return pipeline


@pytest.mark.asyncio
async def test_add_document(rag_pipeline):
    """Test adding a document to the pipeline."""
    doc = await rag_pipeline.add_document(
        "test-1", 
        "This is a test document about machine learning.",
        {"type": "test"}
    )
    
    assert doc.id == "test-1"
    assert "machine learning" in doc.content
    assert doc.metadata["type"] == "test"
    assert "test-1" in rag_pipeline.documents


@pytest.mark.asyncio
async def test_retrieve_documents(rag_pipeline):
    """Test retrieving documents."""
    # Add test documents
    await rag_pipeline.add_document("doc-1", "Python programming tutorial")
    await rag_pipeline.add_document("doc-2", "Machine learning with Python") 
    await rag_pipeline.add_document("doc-3", "JavaScript web development")
    
    # Test retrieval
    results = await rag_pipeline.retrieve("Python programming")
    
    assert len(results) > 0
    # Should find Python-related documents
    found_python = any("Python" in result.document.content for result in results)
    assert found_python


@pytest.mark.asyncio
async def test_generate_context(rag_pipeline):
    """Test context generation."""
    await rag_pipeline.add_document("doc-1", "Python is a programming language")
    await rag_pipeline.add_document("doc-2", "Machine learning uses Python")
    
    context = await rag_pipeline.generate_context("Python programming")
    
    assert isinstance(context, str)
    if context:  # Only check if context was generated
        assert "Python" in context


@pytest.mark.asyncio
async def test_search(rag_pipeline):
    """Test full search functionality."""
    await rag_pipeline.add_document("doc-1", "Python programming guide")
    
    result = await rag_pipeline.search("Python")
    
    assert "query" in result
    assert "retrieved_documents" in result
    assert "context" in result
    assert "document_count" in result
    assert result["query"] == "Python"
    assert result["document_count"] == 1


def test_chunk_text(rag_pipeline):
    """Test text chunking."""
    text = " ".join([f"word{i}" for i in range(200)])  # 200 words
    
    chunks = rag_pipeline._chunk_text(text)
    
    assert len(chunks) > 1
    # First chunk should be around chunk_size words
    first_chunk_words = len(chunks[0].split())
    assert first_chunk_words <= rag_pipeline.rag_config.chunk_size


def test_document_creation():
    """Test Document dataclass."""
    doc = Document(
        id="test-1",
        content="Test content",
        metadata={"source": "test"},
        embedding=[0.1, 0.2, 0.3]
    )
    
    assert doc.id == "test-1"
    assert doc.content == "Test content"
    assert doc.metadata["source"] == "test"
    assert doc.embedding == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_empty_pipeline(rag_pipeline):
    """Test operations on empty pipeline."""
    results = await rag_pipeline.retrieve("test query")
    assert results == []
    
    context = await rag_pipeline.generate_context("test query")
    assert context == ""