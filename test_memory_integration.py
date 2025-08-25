#!/usr/bin/env python3
"""
Simple integration test for the memory subsystem.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_memory_subsystem():
    """Test memory subsystem integration."""
    print("🧠 Testing AgentsMCP Memory Subsystem Integration")
    print("=" * 60)
    
    # Test 1: Import memory subsystem
    try:
        from agentsmcp.memory import (
            InMemoryProvider, 
            serialize_context, 
            deserialize_context,
            get_memory_health
        )
        print("✅ Memory subsystem import successful")
    except ImportError as e:
        print(f"❌ Failed to import memory subsystem: {e}")
        return False

    # Test 2: Test InMemoryProvider
    try:
        provider = InMemoryProvider()
        test_context = {
            "agent_id": "test-agent",
            "session": "test-session",
            "data": {"key": "value", "number": 42}
        }
        
        # Store context
        await provider.store_context("test-agent", test_context)
        
        # Load context
        loaded = await provider.load_context("test-agent") 
        assert loaded == test_context
        
        # Health check
        is_healthy = await provider.health_check()
        assert is_healthy == True
        
        print("✅ InMemoryProvider operations successful")
    except Exception as e:
        print(f"❌ InMemoryProvider test failed: {e}")
        return False
    
    # Test 3: Test serialization
    try:
        test_data = {"complex": [1, 2, {"nested": "data"}]}
        
        # Test compressed serialization
        serialized = serialize_context(test_data, compress=True)
        deserialized = deserialize_context(serialized, compressed=True)
        assert deserialized == test_data
        
        # Test uncompressed serialization  
        serialized_uncompressed = serialize_context(test_data, compress=False)
        deserialized_uncompressed = deserialize_context(serialized_uncompressed, compressed=False)
        assert deserialized_uncompressed == test_data
        
        print("✅ Context serialization/deserialization successful")
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        return False
    
    # Test 4: Test health monitoring
    try:
        providers = [InMemoryProvider(), InMemoryProvider()]
        health = await get_memory_health(providers)
        
        assert "providers" in health
        assert "overall_healthy" in health
        assert "check_duration_ms" in health
        assert health["overall_healthy"] == True
        # Note: Multiple providers of same type get same name, so only 1 entry
        assert "InMemoryProvider" in health["providers"]
        
        print("✅ Memory health monitoring successful")
    except Exception as e:
        print(f"❌ Health monitoring test failed: {e}")
        return False
    
    # Test 5: Test web server integration (if available)
    try:
        # Simple import test
        from agentsmcp.web import app
        print("✅ Web server integration available")
    except ImportError as e:
        print(f"⚠️  Web server integration not available: {e}")
        # This is not a failure since web deps are optional
    except Exception as e:
        print(f"⚠️  Web server integration issue: {e}")
        # This is not a failure since web deps are optional

    print("=" * 60)
    print("🎉 All memory subsystem tests passed!")
    print()
    
    # Show summary
    print("📊 Memory Subsystem Summary:")
    print(f"   • InMemoryProvider: ✅ Working")
    print(f"   • Context Serialization: ✅ Working")
    print(f"   • Health Monitoring: ✅ Working") 
    print(f"   • Web Integration: ⚠️  Optional")
    print()
    print("🚀 Ready to proceed with Redis integration!")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_memory_subsystem())
    exit(0 if success else 1)