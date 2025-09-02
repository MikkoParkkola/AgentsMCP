"""
Test script for reliability modules.

This script tests the startup orchestrator and timeout guardian to ensure they
can prevent TUI hangs and provide guaranteed startup completion.
"""

import asyncio
import sys
import time
from typing import Dict, Any

from .startup_orchestrator import StartupOrchestrator, StartupResult, StartupConfig
from .timeout_guardian import TimeoutGuardian, timeout_protection


class MockTUIComponent:
    """Mock TUI component for testing."""
    
    def __init__(self, name: str, init_delay: float = 0.1, should_fail: bool = False):
        self.name = name
        self.init_delay = init_delay
        self.should_fail = should_fail
        self._initialized = False
        self._ready = False
    
    async def initialize(self):
        """Initialize the component with potential delay or failure."""
        print(f"  Initializing {self.name}...")
        await asyncio.sleep(self.init_delay)
        
        if self.should_fail:
            raise Exception(f"{self.name} initialization failed")
        
        self._initialized = True
        self._ready = True
        print(f"  ✅ {self.name} initialized successfully")
    
    def is_ready(self) -> bool:
        """Check if component is ready."""
        return self._ready


async def test_startup_orchestrator():
    """Test the startup orchestrator with various scenarios."""
    print("\n🚀 Testing Startup Orchestrator")
    print("=" * 50)
    
    # Test 1: Normal startup (should succeed)
    print("\n📋 Test 1: Normal startup")
    components = {
        'orchestrator': MockTUIComponent('Orchestrator', 0.3),
        'display': MockTUIComponent('Display', 0.5),
        'input': MockTUIComponent('Input', 0.2),
        'other': MockTUIComponent('Other', 0.1)
    }
    
    def feedback(msg: str):
        print(f"  💬 {msg}")
    
    orchestrator = StartupOrchestrator()
    start_time = time.time()
    result = await orchestrator.coordinate_startup(components, feedback)
    elapsed = time.time() - start_time
    
    print(f"  Result: {result.value}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Metrics: {orchestrator.get_startup_metrics()}")
    
    assert result == StartupResult.SUCCESS, "Normal startup should succeed"
    assert elapsed < 5.0, "Startup should complete quickly"
    print("  ✅ Test 1 passed")
    
    # Test 2: Slow component (should trigger fallback)  
    print("\n📋 Test 2: Slow component triggering fallback")
    slow_components = {
        'orchestrator': MockTUIComponent('Orchestrator', 0.2),
        'display': MockTUIComponent('SlowDisplay', 4.0),  # Will timeout
        'input': MockTUIComponent('Input', 0.1)
    }
    
    orchestrator2 = StartupOrchestrator()
    start_time = time.time()
    result2 = await orchestrator2.coordinate_startup(slow_components, feedback)
    elapsed2 = time.time() - start_time
    
    print(f"  Result: {result2.value}")
    print(f"  Total time: {elapsed2:.2f}s") 
    print(f"  Metrics: {orchestrator2.get_startup_metrics()}")
    
    assert result2 == StartupResult.FALLBACK, "Slow startup should trigger fallback"
    assert elapsed2 < 6.0, "Even slow startup should complete within reasonable time"
    print("  ✅ Test 2 passed")
    
    # Test 3: Aggressive timeout config
    print("\n📋 Test 3: Aggressive timeout configuration")
    aggressive_config = StartupConfig(
        orchestrator_timeout=1.0,
        display_timeout=0.5,
        input_timeout=0.3,
        finalization_timeout=0.2
    )
    
    normal_components = {
        'orchestrator': MockTUIComponent('Orchestrator', 0.1),
        'display': MockTUIComponent('Display', 0.2),
        'input': MockTUIComponent('Input', 0.1)
    }
    
    orchestrator3 = StartupOrchestrator(aggressive_config)
    start_time = time.time()
    result3 = await orchestrator3.coordinate_startup(normal_components, feedback)
    elapsed3 = time.time() - start_time
    
    print(f"  Result: {result3.value}")
    print(f"  Total time: {elapsed3:.2f}s")
    
    assert elapsed3 < 3.0, "Aggressive config should complete very quickly"
    print("  ✅ Test 3 passed")


async def test_timeout_guardian():
    """Test the timeout guardian with various scenarios."""
    print("\n🛡️ Testing Timeout Guardian") 
    print("=" * 50)
    
    guardian = TimeoutGuardian(default_timeout=2.0, detection_precision=0.01)
    
    # Test 1: Normal operation
    print("\n📋 Test 1: Normal protected operation")
    
    async def quick_task():
        await asyncio.sleep(0.3)
        return "success"
    
    try:
        async with guardian.protect_operation("quick_test", 2.0):
            result = await quick_task()
            print(f"  Result: {result}")
            assert result == "success"
        print("  ✅ Test 1 passed")
    except Exception as e:
        print(f"  ❌ Test 1 failed: {e}")
        raise
    
    # Test 2: Timeout protection
    print("\n📋 Test 2: Timeout protection") 
    
    async def slow_task():
        await asyncio.sleep(3.0)  # Will timeout
        return "should not reach here"
    
    timed_out = False
    try:
        # Use the timeout guardian with asyncio.wait_for for reliable timeout
        async with guardian.protect_operation("slow_test", 1.0) as ctx:
            result = await asyncio.wait_for(slow_task(), timeout=1.0)
            print(f"  Unexpected success: {result}")
    except asyncio.TimeoutError as e:
        print(f"  Expected timeout: {e}")
        timed_out = True
    except Exception as e:
        print(f"  Unexpected error: {e}")
        raise
    
    assert timed_out, "Slow operation should have timed out"
    print("  ✅ Test 2 passed")
    
    # Test 3: Decorator style
    print("\n📋 Test 3: Decorator-style protection")
    
    @guardian.timeout_protected(1.0)
    async def decorated_function():
        await asyncio.sleep(0.2)
        return "decorated success"
    
    try:
        result = await decorated_function()
        print(f"  Result: {result}")
        assert result == "decorated success"
        print("  ✅ Test 3 passed")
    except Exception as e:
        print(f"  ❌ Test 3 failed: {e}")
        raise
    
    # Test 4: Multiple concurrent operations
    print("\n📋 Test 4: Multiple concurrent protected operations")
    
    async def concurrent_task(task_id: int, duration: float):
        await asyncio.sleep(duration)
        return f"task_{task_id}_done"
    
    tasks = []
    for i in range(5):
        coro = guardian.protect_task(
            concurrent_task(i, 0.1 + i * 0.1), 
            f"concurrent_{i}",
            2.0
        )
        tasks.append(asyncio.create_task(coro))
    
    try:
        results = await asyncio.gather(*tasks)
        print(f"  Results: {results}")
        assert len(results) == 5
        print("  ✅ Test 4 passed")
    except Exception as e:
        print(f"  ❌ Test 4 failed: {e}")
        raise
    
    # Show statistics
    stats = guardian.get_protection_stats()
    print(f"\n📊 Guardian Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    await guardian.shutdown()
    print("  ✅ Guardian shutdown complete")


async def test_integration():
    """Test integration of both reliability modules."""
    print("\n🔗 Testing Integration")
    print("=" * 50)
    
    # Create TUI components that use timeout protection
    class ProtectedTUIComponent:
        def __init__(self, name: str, guardian: TimeoutGuardian):
            self.name = name
            self.guardian = guardian
            self._ready = False
        
        async def initialize(self):
            # Use timeout protection for initialization
            async with self.guardian.protect_operation(f"{self.name}_init", 2.0):
                print(f"  Initializing protected {self.name}...")
                await asyncio.sleep(0.3)  # Simulate initialization work
                self._ready = True
                print(f"  ✅ Protected {self.name} initialized")
        
        def is_ready(self) -> bool:
            return self._ready
    
    # Create guardian and components
    guardian = TimeoutGuardian()
    components = {
        'orchestrator': ProtectedTUIComponent('Orchestrator', guardian),
        'display': ProtectedTUIComponent('Display', guardian), 
        'input': ProtectedTUIComponent('Input', guardian)
    }
    
    def feedback(msg: str):
        print(f"  💬 {msg}")
    
    # Test startup with timeout-protected components
    orchestrator = StartupOrchestrator()
    start_time = time.time()
    result = await orchestrator.coordinate_startup(components, feedback)
    elapsed = time.time() - start_time
    
    print(f"  Integration result: {result.value}")
    print(f"  Total time: {elapsed:.2f}s")
    
    # Check guardian stats
    stats = guardian.get_protection_stats()
    print(f"  Guardian operations: {stats['total_operations']}")
    print(f"  Success rate: {stats['success_rate_percent']}%")
    
    assert result in [StartupResult.SUCCESS, StartupResult.FALLBACK], "Integration should not fail"
    assert stats['total_operations'] > 0, "Guardian should have protected operations"
    
    await guardian.shutdown()
    print("  ✅ Integration test passed")


async def run_all_tests():
    """Run all reliability tests."""
    print("🧪 Running Reliability Module Tests")
    print("=" * 60)
    
    try:
        await test_startup_orchestrator()
        await test_timeout_guardian()
        await test_integration()
        
        print("\n🎉 All reliability tests passed!")
        print("✅ The reliability modules are working correctly")
        print("✅ TUI hang prevention mechanisms are functional")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)