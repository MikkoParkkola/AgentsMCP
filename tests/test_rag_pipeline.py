import tempfile
from pathlib import Path

import pytest

from agentsmcp.config import Config, RAGConfig
from agentsmcp.rag.pipeline import Document, RAGPipeline


@pytest.fixture
async def rag_pipeline():
    """Return a RAGPipeline instance."""
    config = Config(rag=RAGConfig())
    pipeline = RAGPipeline(config)
    await pipeline.initialize()
    return pipeline


@pytest.mark.asyncio
async def test_add_document(rag_pipeline):
    """Test adding a document to the pipeline."""
    pipeline = await rag_pipeline
    doc = await pipeline.add_document(
        "test-1",
        "This is a test document about machine learning.",
        {"type": "test"}
    )
    assert isinstance(doc, Document)
    assert doc.id == "test-1"
    assert "machine learning" in doc.content
    assert pipeline.documents["test-1"] == doc


@pytest.mark.asyncio
async def test_retrieve_documents(rag_pipeline):
    """Test retrieving documents."""
    pipeline = await rag_pipeline
    # Add test documents
    await pipeline.add_document("doc-1", "Python programming tutorial")
    await pipeline.add_document("doc-2", "Advanced Java development")
    
    # Retrieve documents
    results = await pipeline.retrieve("python")
    assert len(results) == 1
    assert results[0].document.id == "doc-1"


@pytest.mark.asyncio
async def test_generate_context(rag_pipeline):
    """Test context generation."""
    pipeline = await rag_pipeline
    await pipeline.add_document("doc-1", "Python is a programming language")
    
    context = await pipeline.generate_context("python language")
    assert "Source: doc-1" in context
    assert "Python is a programming language" in context


@pytest.mark.asyncio
async def test_search(rag_pipeline):
    """Test full search functionality."""
    pipeline = await rag_pipeline
    await pipeline.add_document("doc-1", "Python programming guide")
    
    result = await pipeline.search("python guide")
    assert result["query"] == "python guide"
    assert len(result["retrieved_documents"]) == 1
    assert result["retrieved_documents"][0]["id"] == "doc-1"


@pytest.mark.asyncio
async def test_chunk_text(rag_pipeline):
    """Test text chunking."""
    pipeline = await rag_pipeline
    text = " ".join([f"word{i}" for i in range(200)])  # 200 words
    
    chunks = pipeline._chunk_text(text)
    assert len(chunks) > 0
    assert chunks[0].startswith("word0")
    
    # Test with smaller chunk size
    pipeline.rag_config.chunk_size = 50
    pipeline.rag_config.chunk_overlap = 10
    chunks = pipeline._chunk_text(text)
    assert len(chunks) > 1


@pytest.mark.asyncio
async def test_add_documents_from_directory(rag_pipeline):
    """Test adding documents from a directory."""
    pipeline = await rag_pipeline
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        (dir_path / "test1.txt").write_text("This is a test file.")
        (dir_path / "test2.md").write_text("Another test file.")
        
        await pipeline.add_documents_from_directory(str(dir_path))
        
        assert len(pipeline.documents) == 2
        assert "test1.txt" in pipeline.documents
        assert "test2.md" in pipeline.documents


@pytest.mark.asyncio
async def test_empty_pipeline(rag_pipeline):
    """Test operations on empty pipeline."""
    pipeline = await rag_pipeline
    results = await pipeline.retrieve("test query")
    assert len(results) == 0
    
    context = await pipeline.generate_context("test query")
    assert context == ""
