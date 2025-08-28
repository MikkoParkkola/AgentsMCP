#!/usr/bin/env python3
"""
Test individual components of ModernTUI to find the hanging issue.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_rich_live_basic():
    """Test basic Rich Live functionality."""
    print("🎨 Testing Rich Live basic functionality...")
    
    try:
        from rich.live import Live
        from rich.panel import Panel
        from rich.console import Console
        
        console = Console(force_terminal=True)
        
        def render():
            return Panel("Test content", title="Test")
        
        print("Starting Live context...")
        with Live(render(), console=console) as live:
            print("Live context started successfully")
            await asyncio.sleep(1)  # Short test
            print("Live context working")
        
        print("✅ Rich Live basic test passed")
        return True
        
    except Exception as e:
        print(f"❌ Rich Live basic test failed: {e}")
        traceback.print_exc()
        return False

async def test_input_queue_basic():
    """Test basic asyncio queue for input handling."""
    print("📥 Testing asyncio queue for input handling...")
    
    try:
        queue = asyncio.Queue()
        
        async def producer():
            await asyncio.sleep(0.1)
            await queue.put("test message")
        
        async def consumer():
            message = await asyncio.wait_for(queue.get(), timeout=1.0)
            return message
            
        # Run producer and consumer
        producer_task = asyncio.create_task(producer())
        message = await consumer()
        await producer_task
        
        if message == "test message":
            print("✅ Input queue basic test passed")
            return True
        else:
            print(f"❌ Wrong message received: {message}")
            return False
            
    except Exception as e:
        print(f"❌ Input queue basic test failed: {e}")
        traceback.print_exc()
        return False

async def test_async_wait_pattern():
    """Test the asyncio.wait pattern used in ModernTUI."""
    print("⏳ Testing asyncio.wait pattern...")
    
    try:
        queue = asyncio.Queue()
        refresh_event = asyncio.Event()
        
        # Simulate putting something in queue after delay
        async def put_in_queue():
            await asyncio.sleep(0.5)
            await queue.put("test input")
        
        # Start the producer
        producer_task = asyncio.create_task(put_in_queue())
        
        # Test the wait pattern
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(queue.get()),
                asyncio.create_task(refresh_event.wait())
            ],
            timeout=2.0,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cleanup
        for task in pending:
            task.cancel()
        await producer_task
        
        if done:
            result = list(done)[0].result()
            if result == "test input":
                print("✅ asyncio.wait pattern test passed")
                return True
            else:
                print(f"❌ Unexpected result: {result}")
                return False
        else:
            print("❌ No tasks completed")
            return False
            
    except Exception as e:
        print(f"❌ asyncio.wait pattern test failed: {e}")
        traceback.print_exc()
        return False

async def test_keyboard_input_module():
    """Test the keyboard input module if available."""
    print("⌨️  Testing keyboard input module...")
    
    try:
        from agentsmcp.ui.keyboard_input import KeyboardInput
        
        keyboard_input = KeyboardInput()
        print(f"📊 Is interactive: {keyboard_input.is_interactive}")
        print(f"📊 Is windows: {keyboard_input.is_windows}")
        print(f"📊 Is unix: {keyboard_input.is_unix}")
        
        if not keyboard_input.is_interactive:
            print("ℹ️  Keyboard input not available (normal for non-TTY)")
            return True
        
        print("✅ Keyboard input module test passed")
        return True
        
    except ImportError:
        print("ℹ️  Keyboard input module not available")
        return True
    except Exception as e:
        print(f"❌ Keyboard input module test failed: {e}")
        traceback.print_exc()
        return False

async def test_conversation_manager_basic():
    """Test basic conversation manager functionality."""
    print("💬 Testing conversation manager...")
    
    try:
        from agentsmcp.ui.command_interface import CommandInterface
        from agentsmcp.ui.theme_manager import ThemeManager
        
        theme_manager = ThemeManager()
        
        # Mock orchestration manager
        class MockOrchestrationManager:
            async def get_system_status(self):
                return {"status": "mock"}
                
        orchestration_manager = MockOrchestrationManager()
        
        command_interface = CommandInterface(
            orchestration_manager=orchestration_manager,
            theme_manager=theme_manager,
            agent_manager=None,
            app_config=None
        )
        
        conversation_manager = command_interface.conversation_manager
        
        if conversation_manager:
            print("✅ Conversation manager created successfully")
            return True
        else:
            print("❌ Conversation manager is None")
            return False
            
    except Exception as e:
        print(f"❌ Conversation manager test failed: {e}")
        traceback.print_exc()
        return False

async def test_simplified_tui_run():
    """Test a simplified version of the TUI run method."""
    print("🏃 Testing simplified TUI run method...")
    
    try:
        from rich.live import Live
        from rich.panel import Panel
        from rich.console import Console
        
        console = Console(force_terminal=True)
        running = True
        input_queue = asyncio.Queue()
        refresh_event = asyncio.Event()
        
        def render():
            return Panel("Simplified TUI Test", title="Test")
        
        # Simulate user quitting after delay
        async def auto_quit():
            await asyncio.sleep(1)
            await input_queue.put("/quit")
        
        quit_task = asyncio.create_task(auto_quit())
        
        with Live(render(), console=console) as live:
            while running:
                try:
                    # Wait for input with timeout
                    done, pending = await asyncio.wait(
                        [
                            asyncio.create_task(input_queue.get()),
                            asyncio.create_task(refresh_event.wait())
                        ],
                        timeout=2.0,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        
                    if done:
                        task = list(done)[0]
                        result = task.result()
                        if result == "/quit":
                            print("Received quit command")
                            running = False
                            break
                    else:
                        # Timeout
                        print("Timeout in simplified TUI")
                        break
                        
                except Exception as e:
                    print(f"Error in simplified TUI loop: {e}")
                    break
        
        await quit_task
        print("✅ Simplified TUI run test passed")
        return True
        
    except Exception as e:
        print(f"❌ Simplified TUI run test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all component tests."""
    print("🔬 ModernTUI Component Analysis")
    print("=" * 50)
    
    tests = [
        ("Rich Live Basic", test_rich_live_basic),
        ("Input Queue Basic", test_input_queue_basic), 
        ("Asyncio Wait Pattern", test_async_wait_pattern),
        ("Keyboard Input Module", test_keyboard_input_module),
        ("Conversation Manager", test_conversation_manager_basic),
        ("Simplified TUI Run", test_simplified_tui_run),
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n🧪 Running: {name}")
        try:
            result = await asyncio.wait_for(test_func(), timeout=10.0)
            results[name] = result
        except asyncio.TimeoutError:
            print(f"⏰ {name} timed out")
            results[name] = False
        except Exception as e:
            print(f"💥 {name} crashed: {e}")
            results[name] = False
    
    print("\n📊 Test Results Summary:")
    print("-" * 30)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
    
    failed_tests = [name for name, passed in results.items() if not passed]
    if failed_tests:
        print(f"\n🔍 Failed components: {', '.join(failed_tests)}")
        print("💡 These components likely cause the hanging issue")
    else:
        print("\n🎉 All components passed! Issue may be in integration")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted")
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        traceback.print_exc()