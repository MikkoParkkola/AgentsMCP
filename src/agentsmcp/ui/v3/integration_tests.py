"""Integration tests for the complete v3 TUI system."""

import asyncio
import sys
import io
from unittest.mock import Mock, patch, MagicMock
from .tui_launcher import TUILauncher, launch_tui
from .terminal_capabilities import detect_terminal_capabilities
from .chat_engine import ChatEngine


class MockTerminal:
    """Mock terminal for testing."""
    
    def __init__(self, width=80, height=24, is_tty=True, supports_colors=True):
        self.width = width
        self.height = height 
        self.is_tty = is_tty
        self.supports_colors = supports_colors
        self.output = []
        
    def write(self, text):
        self.output.append(text)
        
    def read(self, size=1):
        return "\n"  # Simulate Enter key


async def test_tui_launcher_initialization():
    """Test TUI launcher initialization."""
    print("🧪 Testing TUI launcher initialization...")
    
    launcher = TUILauncher()
    
    # Test initialization
    success = launcher.initialize()
    print(f"  Initialization: {'✅ PASS' if success else '❌ FAIL'}")
    
    # Test components
    has_capabilities = launcher.capabilities is not None
    has_renderer = launcher.current_renderer is not None  
    has_chat_engine = launcher.chat_engine is not None
    
    print(f"  Capabilities detected: {'✅ PASS' if has_capabilities else '❌ FAIL'}")
    print(f"  Renderer selected: {'✅ PASS' if has_renderer else '❌ FAIL'}")
    print(f"  Chat engine ready: {'✅ PASS' if has_chat_engine else '❌ FAIL'}")
    
    # Test callback setup
    callbacks_set = all([
        launcher.chat_engine._status_callback is not None,
        launcher.chat_engine._message_callback is not None,
        launcher.chat_engine._error_callback is not None
    ])
    print(f"  Callbacks configured: {'✅ PASS' if callbacks_set else '❌ FAIL'}")
    
    # Clean up
    launcher._cleanup()
    
    return success and has_capabilities and has_renderer and has_chat_engine and callbacks_set


async def test_progressive_renderer_selection():
    """Test progressive renderer selection logic."""
    print("\n🧪 Testing progressive renderer selection...")
    
    from .terminal_capabilities import TerminalCapabilities
    from .ui_renderer_base import ProgressiveRenderer
    from .plain_cli_renderer import PlainCLIRenderer, SimpleTUIRenderer
    from .rich_tui_renderer import RichTUIRenderer
    
    # Test different capability scenarios
    scenarios = [
        {
            'name': 'Rich-capable terminal',
            'caps': TerminalCapabilities(
                is_tty=True, width=120, height=40, 
                supports_colors=True, supports_unicode=True, supports_rich=True,
                is_fast_terminal=True, max_refresh_rate=60,
                force_plain=False, force_simple=False
            ),
            'expected': 'RichTUIRenderer'
        },
        {
            'name': 'Simple TUI terminal', 
            'caps': TerminalCapabilities(
                is_tty=True, width=80, height=24,
                supports_colors=True, supports_unicode=False, supports_rich=False,
                is_fast_terminal=False, max_refresh_rate=30,
                force_plain=False, force_simple=False
            ),
            'expected': 'SimpleTUIRenderer'
        },
        {
            'name': 'Plain text only',
            'caps': TerminalCapabilities(
                is_tty=False, width=80, height=24,
                supports_colors=False, supports_unicode=False, supports_rich=False, 
                is_fast_terminal=False, max_refresh_rate=30,
                force_plain=True, force_simple=False
            ),
            'expected': 'PlainCLIRenderer'
        }
    ]
    
    all_passed = True
    
    for scenario in scenarios:
        print(f"  Testing: {scenario['name']}")
        
        progressive = ProgressiveRenderer(scenario['caps'])
        progressive.register_renderer("rich", RichTUIRenderer, priority=30)
        progressive.register_renderer("simple", SimpleTUIRenderer, priority=20)
        progressive.register_renderer("plain", PlainCLIRenderer, priority=10)
        
        try:
            selected = progressive.select_best_renderer()
            
            if selected:
                renderer_name = selected.__class__.__name__
                expected = scenario['expected']
                passed = renderer_name == expected
                print(f"    Selected: {renderer_name} {'✅' if passed else '❌'}")
                if not passed:
                    print(f"    Expected: {expected}")
                all_passed &= passed
                selected.cleanup()
            else:
                print(f"    No renderer selected ❌")
                all_passed = False
                
        except Exception as e:
            print(f"    Error: {e} ❌")
            all_passed = False
            
        progressive.cleanup()
    
    return all_passed


async def test_chat_engine_integration():
    """Test chat engine integration with UI."""
    print("\n🧪 Testing chat engine integration...")
    
    # Create launcher and initialize
    launcher = TUILauncher()
    success = launcher.initialize()
    
    if not success:
        print("  Initialization failed ❌")
        return False
    
    # Test message processing
    test_commands = [
        ("/help", True, "Should show help"),
        ("hello world", True, "Should process chat message"),
        ("/status", True, "Should show status"),
        ("/quit", False, "Should quit")
    ]
    
    all_passed = True
    
    for command, should_continue, description in test_commands[:-1]:  # Skip quit for now
        print(f"  Testing: {description}")
        
        try:
            result = await launcher.chat_engine.process_input(command)
            passed = result == should_continue
            print(f"    Result: {'✅ PASS' if passed else '❌ FAIL'}")
            all_passed &= passed
            
        except Exception as e:
            print(f"    Error: {e} ❌")
            all_passed = False
    
    # Clean up
    launcher._cleanup()
    
    return all_passed


async def test_end_to_end_simulation():
    """Test end-to-end simulation without actual user interaction."""
    print("\n🧪 Testing end-to-end simulation...")
    
    try:
        # Mock sys.stdin for non-interactive testing
        with patch('sys.stdin') as mock_stdin:
            mock_stdin.read.return_value = '\n'
            mock_stdin.isatty.return_value = False
            
            launcher = TUILauncher()
            
            # Initialize
            init_success = launcher.initialize()
            print(f"  Initialization: {'✅ PASS' if init_success else '❌ FAIL'}")
            
            if not init_success:
                return False
            
            # Test components are working
            renderer_active = launcher.current_renderer and launcher.current_renderer.is_active()
            engine_ready = launcher.chat_engine is not None
            
            print(f"  Renderer active: {'✅ PASS' if renderer_active else '❌ FAIL'}")
            print(f"  Engine ready: {'✅ PASS' if engine_ready else '❌ FAIL'}")
            
            # Test callback system with mock message
            callback_test = True
            try:
                launcher._on_status_change("Test status")
                launcher._on_error("Test error")
                print(f"  Callbacks working: ✅ PASS")
            except Exception as e:
                print(f"  Callbacks working: ❌ FAIL ({e})")
                callback_test = False
            
            # Clean up
            launcher._cleanup()
            
            return init_success and renderer_active and engine_ready and callback_test
            
    except Exception as e:
        print(f"  End-to-end test error: {e} ❌")
        return False


async def run_all_integration_tests():
    """Run all integration tests."""
    print("🚀 Running V3 TUI Integration Tests")
    print("=" * 50)
    
    tests = [
        test_tui_launcher_initialization,
        test_progressive_renderer_selection,
        test_chat_engine_integration, 
        test_end_to_end_simulation
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"  Test failed with exception: {e}")
            results.append(False)
    
    print("\n📊 Integration Test Results")
    print("=" * 30)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "TUI Launcher Initialization",
        "Progressive Renderer Selection",
        "Chat Engine Integration",
        "End-to-End Simulation"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests PASSED! V3 architecture is ready.")
        return True
    else:
        print(f"⚠️ {total-passed} tests FAILED. Review issues above.")
        return False


if __name__ == "__main__":
    asyncio.run(run_all_integration_tests())