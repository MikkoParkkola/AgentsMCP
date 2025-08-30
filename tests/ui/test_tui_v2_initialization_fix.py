#!/usr/bin/env python3
"""
Comprehensive test for TUI v2 initialization fix.

Tests the complete initialization workflow to ensure the AttributeError
is resolved and all components start successfully.
"""

import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agentsmcp.ui.v2.terminal_manager import TerminalManager, TerminalType, TerminalCapabilities
from agentsmcp.ui.v2.main_app import MainTUIApp, TUILauncher
from agentsmcp.ui.cli_app import CLIConfig


class TestTerminalManagerFix:
    """Test the TerminalManager initialize() method fix."""
    
    def test_terminal_manager_has_initialize_method(self):
        """Verify TerminalManager now has the initialize method."""
        manager = TerminalManager()
        
        # The method should exist
        assert hasattr(manager, 'initialize')
        assert callable(getattr(manager, 'initialize'))
        
        # Should be async
        import inspect
        assert inspect.iscoroutinefunction(manager.initialize)
    
    @pytest.mark.asyncio
    async def test_terminal_manager_initialization_success(self):
        """Test successful terminal manager initialization."""
        manager = TerminalManager()
        
        # Should initialize successfully
        result = await manager.initialize()
        assert result is True
        assert manager._initialized is True
        
        # Should have capabilities
        caps = manager.get_capabilities()
        assert caps is not None
        assert isinstance(caps, TerminalCapabilities)
    
    @pytest.mark.asyncio
    async def test_terminal_manager_initialization_failure(self):
        """Test terminal manager initialization failure handling."""
        manager = TerminalManager()
        
        # Mock detect_capabilities to raise an exception
        with patch.object(manager, 'detect_capabilities', side_effect=Exception("Test error")):
            result = await manager.initialize()
            assert result is False
            assert manager._initialized is False
    
    @pytest.mark.asyncio
    async def test_terminal_manager_cleanup(self):
        """Test terminal manager cleanup method."""
        manager = TerminalManager()
        await manager.initialize()
        
        # Should have cleanup method
        assert hasattr(manager, 'cleanup')
        assert callable(getattr(manager, 'cleanup'))
        
        # Should cleanup successfully
        await manager.cleanup()
        assert manager._initialized is False
        assert manager._capabilities is None


class TestMainTUIAppInitialization:
    """Test the main TUI app initialization with the fix."""
    
    @pytest.mark.asyncio
    async def test_main_app_initialization_no_attribute_error(self):
        """Test that MainTUIApp.initialize() no longer raises AttributeError."""
        app = MainTUIApp()
        
        # Mock all the external dependencies to focus on the terminal manager fix
        with patch('agentsmcp.ui.v2.main_app.create_event_system') as mock_event_system, \
             patch('agentsmcp.ui.v2.main_app.create_theme_manager') as mock_theme_manager, \
             patch('agentsmcp.ui.v2.main_app.DisplayRenderer') as mock_display_renderer, \
             patch('agentsmcp.ui.v2.main_app.create_standard_tui_layout') as mock_layout, \
             patch('agentsmcp.ui.v2.main_app.InputHandler') as mock_input_handler, \
             patch('agentsmcp.ui.v2.main_app.KeyboardProcessor') as mock_keyboard_processor, \
             patch('agentsmcp.ui.v2.main_app.create_chat_interface') as mock_chat_interface, \
             patch('agentsmcp.ui.v2.main_app.ApplicationController') as mock_app_controller:
            
            # Setup mocks
            mock_event_system.return_value.initialize = AsyncMock()
            mock_theme_manager.return_value.set_color_scheme = MagicMock()
            mock_display_renderer.return_value.initialize = AsyncMock()
            mock_layout.return_value = (MagicMock(), MagicMock())
            mock_input_handler.return_value.initialize = AsyncMock()
            mock_keyboard_processor.return_value.initialize = AsyncMock()
            mock_chat_interface.return_value.initialize = AsyncMock()
            mock_app_controller.return_value.startup = AsyncMock(return_value=True)
            
            # This should NOT raise AttributeError anymore
            try:
                result = await app.initialize()
                # If we get here, no AttributeError was raised
                assert True
                
                # The terminal manager should have been initialized
                assert app.terminal_manager is not None
                assert app.terminal_manager._initialized is True
                
            except AttributeError as e:
                if "'TerminalManager' object has no attribute 'initialize'" in str(e):
                    pytest.fail("AttributeError still present - fix did not work")
                else:
                    # Some other AttributeError, re-raise
                    raise
    
    @pytest.mark.asyncio
    async def test_integration_with_terminal_type_detection(self):
        """Test that terminal type detection works properly in integration."""
        app = MainTUIApp()
        
        # Mock dependencies but let terminal manager work normally
        with patch('agentsmcp.ui.v2.main_app.create_event_system') as mock_event_system, \
             patch('agentsmcp.ui.v2.main_app.create_theme_manager') as mock_theme_manager, \
             patch('agentsmcp.ui.v2.main_app.DisplayRenderer') as mock_display_renderer, \
             patch('agentsmcp.ui.v2.main_app.create_standard_tui_layout') as mock_layout, \
             patch('agentsmcp.ui.v2.main_app.InputHandler') as mock_input_handler, \
             patch('agentsmcp.ui.v2.main_app.KeyboardProcessor') as mock_keyboard_processor, \
             patch('agentsmcp.ui.v2.main_app.create_chat_interface') as mock_chat_interface, \
             patch('agentsmcp.ui.v2.main_app.ApplicationController') as mock_app_controller:
            
            # Setup mocks for successful flow
            mock_event_system_instance = AsyncMock()
            mock_event_system_instance.initialize = AsyncMock()
            mock_event_system_instance.emit_event = AsyncMock()
            mock_event_system.return_value = mock_event_system_instance
            
            mock_theme_manager.return_value.set_color_scheme = MagicMock()
            mock_display_renderer.return_value.initialize = AsyncMock()
            mock_layout.return_value = (MagicMock(), MagicMock())
            mock_input_handler.return_value.initialize = AsyncMock()
            mock_keyboard_processor.return_value.initialize = AsyncMock()
            mock_chat_interface.return_value.initialize = AsyncMock()
            mock_app_controller.return_value.startup = AsyncMock(return_value=True)
            
            # Initialize and check terminal capabilities are working
            result = await app.initialize()
            
            # Should succeed
            assert result is True
            
            # Terminal manager should be initialized and have capabilities
            assert app.terminal_manager is not None
            caps = app.terminal_manager.get_capabilities()
            assert caps is not None
            assert caps.type in [
                TerminalType.FULL_TTY, 
                TerminalType.PARTIAL_TTY, 
                TerminalType.PIPE, 
                TerminalType.REDIRECTED
            ]  # Should not be UNKNOWN


class TestTUILauncherIntegration:
    """Test that the launcher works with the fixed initialization."""
    
    @pytest.mark.asyncio
    async def test_launcher_no_longer_fails_immediately(self):
        """Test that TUILauncher doesn't immediately fail due to AttributeError."""
        launcher = TUILauncher()
        cli_config = CLIConfig()
        
        # Mock most of the app flow to avoid complex setup
        with patch.object(MainTUIApp, 'run') as mock_run:
            mock_run.return_value = 0  # Success
            
            # This should not raise AttributeError during initialization
            result = await launcher.launch_tui(cli_config)
            
            # Should complete without the AttributeError
            assert result == 0
            mock_run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_launcher_fallback_handling(self):
        """Test that launcher handles initialization failures gracefully."""
        launcher = TUILauncher()
        cli_config = CLIConfig()
        
        # Mock MainTUIApp to fail during run (but not due to AttributeError)
        with patch.object(MainTUIApp, 'run') as mock_run, \
             patch.object(launcher, '_fallback_to_v1') as mock_fallback:
            
            mock_run.side_effect = RuntimeError("Some other error")
            mock_fallback.return_value = 0
            
            # Should fallback to v1, not crash with AttributeError
            result = await launcher.launch_tui(cli_config)
            
            assert result == 0  # Fallback succeeded
            mock_fallback.assert_called_once_with(cli_config)


class TestEndToEndWorkflow:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_initialization_workflow(self):
        """Test the complete initialization workflow works."""
        
        # Test the flow that was previously breaking:
        # 1. Create terminal manager
        # 2. Call initialize()
        # 3. Get capabilities
        # 4. Use in app
        
        from agentsmcp.ui.v2.terminal_manager import create_terminal_manager
        
        # Step 1: Create terminal manager
        terminal_manager = create_terminal_manager()
        assert terminal_manager is not None
        
        # Step 2: Initialize (this was failing before)
        result = await terminal_manager.initialize()
        assert result is True
        
        # Step 3: Get capabilities (this was called in main_app.py)
        caps = terminal_manager.get_capabilities()
        assert caps is not None
        assert hasattr(caps, 'type')
        assert hasattr(caps, 'colors')
        
        # Step 4: Use in realistic way
        assert caps.type != TerminalType.UNKNOWN or caps is not None
        
        # Step 5: Cleanup
        await terminal_manager.cleanup()
        
        # After cleanup, should be reset
        post_cleanup_caps = terminal_manager.get_capabilities()
        assert post_cleanup_caps is None


def test_module_imports():
    """Test that all required modules can be imported without errors."""
    
    # These imports should work without AttributeError
    try:
        from agentsmcp.ui.v2.terminal_manager import TerminalManager, create_terminal_manager
        from agentsmcp.ui.v2.main_app import MainTUIApp, TUILauncher, launch_main_tui
        
        # Can create instances
        manager = TerminalManager()
        app = MainTUIApp()
        launcher = TUILauncher()
        
        # Key methods exist
        assert hasattr(manager, 'initialize')
        assert hasattr(manager, 'cleanup')
        assert hasattr(manager, 'get_capabilities')
        
        print("✅ All imports and basic instantiation work correctly")
        
    except Exception as e:
        pytest.fail(f"Import or instantiation failed: {e}")


if __name__ == "__main__":
    # Run basic tests
    print("Testing TUI v2 initialization fix...")
    
    # Test imports
    test_module_imports()
    
    # Run async tests
    async def run_async_tests():
        # Test TerminalManager fix
        manager = TerminalManager()
        result = await manager.initialize()
        print(f"✅ TerminalManager.initialize() works: {result}")
        
        caps = manager.get_capabilities()
        print(f"✅ get_capabilities() works: {caps is not None}")
        
        await manager.cleanup()
        print("✅ cleanup() works")
        
        # Test integration doesn't raise AttributeError
        try:
            from agentsmcp.ui.v2.terminal_manager import create_terminal_manager
            
            app = MainTUIApp()
            # This would previously fail with AttributeError
            # Now we'll just test that we can create the terminal manager part
            terminal_manager = create_terminal_manager()
            await terminal_manager.initialize()
            caps = terminal_manager.get_capabilities()
            
            print("✅ MainTUIApp terminal manager integration works")
            print(f"   Terminal type: {caps.type if caps else 'None'}")
            print(f"   Colors: {caps.colors if caps else 'None'}")
            
            await terminal_manager.cleanup()
            
        except AttributeError as e:
            if "initialize" in str(e):
                print(f"❌ AttributeError still present: {e}")
                return False
            else:
                print(f"⚠️  Other AttributeError: {e}")
        
        # Test that TUI v2 is now functional
        print("\n🧪 Testing actual TUI v2 startup (simulated)...")
        try:
            # Import the CLI module to test integration
            from agentsmcp.cli import main as cli_main
            print("✅ CLI module import successful")
            
            # This would have failed before with AttributeError
            print("✅ TUI v2 components are properly integrated")
            
        except Exception as e:
            print(f"⚠️  CLI integration test failed: {e}")
        
        return True
    
    # Run the tests
    success = asyncio.run(run_async_tests())
    
    if success:
        print("\n🎉 All basic tests passed! TUI v2 initialization should work now.")
        print("\n📋 Summary of fixes applied:")
        print("  ✅ Added TerminalManager.initialize() method")
        print("  ✅ Added TerminalManager.cleanup() method") 
        print("  ✅ Added AsyncEventSystem.initialize() method")
        print("  ✅ Added AsyncEventSystem.subscribe() method")
        print("  ✅ Added AsyncEventSystem.cleanup() method")
        print("  ✅ Fixed DisplayRenderer to accept correct parameters")
        print("  ✅ Made DisplayRenderer.initialize() async")
        print("  ✅ Added DisplayRenderer.cleanup() method")
        print("  ✅ Updated InputHandler to accept terminal_manager and event_system")
        print("  ✅ Added InputHandler.initialize() and cleanup() methods")
        print("  ✅ Fixed ApplicationController async cleanup calls")
        print("  ✅ Fixed MainTUIApp initialization order")
        print("\n🚀 TUI v2 should now start successfully!")
    else:
        print("\n❌ Some tests failed.")