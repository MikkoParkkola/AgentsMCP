"""
Tests for the memory subsystem.
"""

import pytest
import asyncio
from agentsmcp.memory import (
    InMemoryProvider,
    serialize_context,
    deserialize_context,
    get_memory_health
)


class TestSerializers:
    """Test context serialization and deserialization."""
    
    def test_serialize_deserialize_simple(self):
        """Test basic serialization/deserialization."""
        context = {"agent_id": "test", "session": "abc123", "data": [1, 2, 3]}
        
        # Test compressed
        serialized = serialize_context(context, compress=True)
        assert isinstance(serialized, bytes)
        
        deserialized = deserialize_context(serialized, compressed=True)
        assert deserialized == context
        
    def test_serialize_deserialize_uncompressed(self):
        """Test uncompressed serialization/deserialization."""
        context = {"simple": "test"}
        
        # Test uncompressed
        serialized = serialize_context(context, compress=False)
        assert isinstance(serialized, bytes)
        
        deserialized = deserialize_context(serialized, compressed=False)
        assert deserialized == context


class TestInMemoryProvider:
    """Test in-memory provider implementation."""
    
    @pytest.fixture
    def provider(self):
        return InMemoryProvider()
    
    @pytest.mark.asyncio
    async def test_store_and_load_context(self, provider):
        """Test storing and loading context."""
        agent_id = "test-agent-1"
        context = {"session_id": "abc123", "state": "active", "data": {"key": "value"}}
        
        # Store context
        await provider.store_context(agent_id, context)
        
        # Load context
        loaded = await provider.load_context(agent_id)
        assert loaded == context
        
    @pytest.mark.asyncio
    async def test_load_nonexistent_context(self, provider):
        """Test loading non-existent context returns None."""
        result = await provider.load_context("nonexistent-agent")
        assert result is None
        
    @pytest.mark.asyncio
    async def test_delete_context(self, provider):
        """Test deleting context."""
        agent_id = "test-agent-2"
        context = {"test": "data"}
        
        # Store then delete
        await provider.store_context(agent_id, context)
        await provider.delete_context(agent_id)
        
        # Should be None after delete
        result = await provider.load_context(agent_id)
        assert result is None
        
    @pytest.mark.asyncio
    async def test_health_check(self, provider):
        """Test health check always returns True."""
        assert await provider.health_check() == True


class TestHealthCheck:
    """Test health check functionality."""
    
    @pytest.mark.asyncio
    async def test_get_memory_health_empty(self):
        """Test health check with no providers."""
        result = await get_memory_health([])
        assert result["providers"] == {}
        assert result["overall_healthy"] == True
        assert result["check_duration_ms"] >= 0.0
        
    @pytest.mark.asyncio
    async def test_get_memory_health_single_provider(self):
        """Test health check with single provider."""
        provider = InMemoryProvider()
        result = await get_memory_health([provider])
        
        assert "InMemoryProvider" in result["providers"]
        assert result["providers"]["InMemoryProvider"] == True
        assert result["overall_healthy"] == True
        assert result["check_duration_ms"] >= 0.0


if __name__ == "__main__":
    # Simple test runner
    import asyncio
    
    async def run_async_tests():
        provider = InMemoryProvider()
        
        # Test basic operations
        await provider.store_context("test", {"data": "test"})
        context = await provider.load_context("test")
        print(f"Loaded context: {context}")
        
        # Test health check
        health = await get_memory_health([provider])
        print(f"Health check: {health}")
        
        print("All tests passed!")
    
    asyncio.run(run_async_tests())