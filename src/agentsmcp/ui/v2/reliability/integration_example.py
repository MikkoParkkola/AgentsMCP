"""
Integration Example - How to use reliability modules to fix TUI hangs.

This example demonstrates how to integrate the startup orchestrator and timeout guardian
with the Revolutionary TUI Interface to prevent the hanging issue.

The key changes are:
1. Wrap TUI initialization with StartupOrchestrator 
2. Use TimeoutGuardian to protect all async operations
3. Provide guaranteed completion within 10 seconds
4. Fallback to basic mode if any component hangs
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from .startup_orchestrator import StartupOrchestrator, StartupResult, StartupConfig
from .timeout_guardian import TimeoutGuardian, get_global_guardian

logger = logging.getLogger(__name__)


class ReliableTUIInterface:
    """
    A wrapper for the Revolutionary TUI Interface that prevents hangs.
    
    This class demonstrates how to integrate the reliability modules
    with the existing TUI to solve the hanging issue.
    """
    
    def __init__(self, original_tui_class, cli_config=None):
        """
        Initialize the reliable TUI wrapper.
        
        Args:
            original_tui_class: The original RevolutionaryTUIInterface class
            cli_config: CLI configuration
        """
        self.original_tui_class = original_tui_class
        self.cli_config = cli_config
        self.tui_instance = None
        self.startup_orchestrator = None
        self.timeout_guardian = get_global_guardian()
        
    async def run_with_reliability(self) -> int:
        """
        Run the TUI with full reliability protection.
        
        This is the main entry point that prevents hangs and ensures
        the TUI either starts successfully or falls back gracefully.
        """
        print("🚀 Starting Revolutionary TUI with reliability protection...")
        
        try:
            # Create startup configuration with aggressive timeouts
            startup_config = StartupConfig(
                orchestrator_timeout=3.0,   # 3s max for orchestrator
                display_timeout=2.0,        # 2s max for display setup
                input_timeout=1.0,          # 1s max for input setup
                finalization_timeout=1.0,   # 1s max for finalization
                feedback_delay=0.5,         # Show feedback within 500ms
                enable_progressive_fallback=True,
                retry_failed_components=False  # Don't retry to prevent hangs
            )
            
            # Create startup orchestrator
            self.startup_orchestrator = StartupOrchestrator(startup_config)
            
            # Initialize TUI with reliability protection
            result = await self._initialize_tui_with_protection()
            
            if result == StartupResult.SUCCESS:
                print("✅ TUI initialized successfully - starting main loop")
                return await self._run_protected_main_loop()
                
            elif result == StartupResult.FALLBACK:
                print("⚠️ TUI fell back to basic mode due to timeout - continuing with reduced functionality")
                return await self._run_fallback_mode()
                
            else:  # StartupResult.FAILURE
                print("❌ TUI initialization failed - using emergency mode")
                return await self._run_emergency_mode()
                
        except Exception as e:
            logger.error(f"Critical error in reliable TUI: {e}")
            print(f"💥 Critical TUI error: {e}")
            return 1
            
        finally:
            await self._cleanup()
    
    async def _initialize_tui_with_protection(self) -> StartupResult:
        """Initialize TUI components with startup orchestration."""
        
        def feedback_callback(message: str):
            """Show startup feedback to user."""
            print(f"  {message}")
        
        # Create mock components for the TUI initialization
        # In the real implementation, these would be the actual TUI components
        components = {
            'orchestrator': self._create_orchestrator_component(),
            'display': self._create_display_component(), 
            'input': self._create_input_component(),
            'enhancements': self._create_enhancements_component()
        }
        
        # Use startup orchestrator to coordinate initialization
        result = await self.startup_orchestrator.coordinate_startup(
            components, 
            feedback_callback
        )
        
        if result == StartupResult.SUCCESS:
            # Create the actual TUI instance with initialized components
            self.tui_instance = self.original_tui_class(
                cli_config=self.cli_config,
                revolutionary_components=components
            )
        
        return result
    
    def _create_orchestrator_component(self):
        """Create orchestrator component with timeout protection."""
        class ProtectedOrchestrator:
            def __init__(self):
                self._initialized = False
                
            async def initialize(self):
                # Use timeout guardian for initialization
                async with get_global_guardian().protect_operation("orchestrator_init", 2.0):
                    await asyncio.wait_for(self._do_initialize(), timeout=2.0)
                    
            async def _do_initialize(self):
                # Simulate orchestrator initialization
                print("    🎯 Initializing orchestrator...")
                await asyncio.sleep(0.3)  # Simulate work
                self._initialized = True
                print("    ✅ Orchestrator ready")
                
            def is_ready(self) -> bool:
                return self._initialized
        
        return ProtectedOrchestrator()
    
    def _create_display_component(self):
        """Create display component with timeout protection."""
        class ProtectedDisplay:
            def __init__(self):
                self._initialized = False
                
            async def initialize(self):
                # Use timeout guardian for initialization  
                async with get_global_guardian().protect_operation("display_init", 1.5):
                    await asyncio.wait_for(self._do_initialize(), timeout=1.5)
                    
            async def _do_initialize(self):
                # Simulate display initialization
                print("    🖥️ Setting up display...")
                await asyncio.sleep(0.5)  # Simulate work
                self._initialized = True
                print("    ✅ Display ready")
                
            def is_ready(self) -> bool:
                return self._initialized
        
        return ProtectedDisplay()
    
    def _create_input_component(self):
        """Create input component with timeout protection."""
        class ProtectedInput:
            def __init__(self):
                self._initialized = False
                
            async def initialize(self):
                # Use timeout guardian for initialization
                async with get_global_guardian().protect_operation("input_init", 1.0):
                    await asyncio.wait_for(self._do_initialize(), timeout=1.0)
                    
            async def _do_initialize(self):
                # Simulate input initialization
                print("    ⌨️ Setting up input handling...")
                await asyncio.sleep(0.2)  # Simulate work  
                self._initialized = True
                print("    ✅ Input ready")
                
            def is_ready(self) -> bool:
                return self._initialized
        
        return ProtectedInput()
    
    def _create_enhancements_component(self):
        """Create enhancements component with timeout protection.""" 
        class ProtectedEnhancements:
            def __init__(self):
                self._initialized = False
                
            async def initialize(self):
                # Use timeout guardian for initialization
                async with get_global_guardian().protect_operation("enhancements_init", 1.0):
                    await asyncio.wait_for(self._do_initialize(), timeout=1.0)
                    
            async def _do_initialize(self):
                # Simulate enhancements initialization
                print("    ✨ Loading enhancements...")
                await asyncio.sleep(0.1)  # Simulate work
                self._initialized = True
                print("    ✅ Enhancements ready")
                
            def is_ready(self) -> bool:
                return self._initialized
        
        return ProtectedEnhancements()
    
    async def _run_protected_main_loop(self) -> int:
        """Run the main TUI loop with timeout protection."""
        print("🎯 Running TUI main loop with timeout protection...")
        
        try:
            # If we have a real TUI instance, run it
            if self.tui_instance and hasattr(self.tui_instance, 'run'):
                # Protect the main TUI run method with timeout
                async with self.timeout_guardian.protect_operation("tui_main_loop", 300.0):  # 5 minutes
                    return await asyncio.wait_for(self.tui_instance.run(), timeout=300.0)
            else:
                # Mock main loop for demonstration
                return await self._run_mock_tui_loop()
                
        except asyncio.TimeoutError:
            print("⚠️ TUI main loop timed out - switching to basic mode")
            return await self._run_fallback_mode()
        except Exception as e:
            print(f"❌ TUI main loop failed: {e}")
            return 1
    
    async def _run_mock_tui_loop(self) -> int:
        """Mock TUI loop for demonstration."""
        print("🎮 Mock TUI is running...")
        print("  Type 'quit' to exit, or wait 10 seconds for automatic exit")
        
        # Simulate TUI interaction
        for i in range(10):
            await asyncio.sleep(1)
            print(f"  🕒 TUI active... ({i+1}/10)")
            
            # In a real implementation, this would handle user input
            # For demo, we'll just count down
            
        print("  👋 Mock TUI demo completed")
        return 0
    
    async def _run_fallback_mode(self) -> int:
        """Run in fallback mode with reduced functionality."""
        print("🔄 Running in fallback mode...")
        print("  ⚠️ Some TUI features may be unavailable")
        print("  📟 Using basic terminal interface")
        
        # Simple fallback interface
        try:
            print("\n💬 Fallback Interface Ready")
            print("  Available commands: help, status, quit")
            
            for i in range(5):  # Reduced demo time
                await asyncio.sleep(1)
                print(f"  📡 Fallback mode active... ({i+1}/5)")
            
            print("  ✅ Fallback mode demo completed")
            return 0
            
        except Exception as e:
            print(f"❌ Fallback mode failed: {e}")
            return 1
    
    async def _run_emergency_mode(self) -> int:
        """Run in emergency mode with minimal functionality."""
        print("🚨 Running in emergency mode...")
        print("  ⛑️ Minimal functionality only")
        print("  📋 Basic status available")
        
        try:
            await asyncio.sleep(1)
            print("  💾 Emergency mode: System is operational but TUI failed to start")
            print("  🔧 Suggestion: Check system resources and try again")
            print("  📞 If problem persists, contact support")
            return 0
            
        except Exception as e:
            print(f"💥 Emergency mode failed: {e}")
            return 2  # Critical failure
    
    async def _cleanup(self):
        """Cleanup resources."""
        try:
            if self.tui_instance and hasattr(self.tui_instance, 'cleanup'):
                await self.tui_instance.cleanup()
            
            if self.timeout_guardian:
                await self.timeout_guardian.shutdown()
                
            print("🧹 Cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


# Example usage
async def run_reliable_tui_demo():
    """
    Demonstrate how to use the reliable TUI wrapper.
    
    This shows how to integrate the reliability modules with an existing
    TUI to prevent hangs and ensure graceful fallback.
    """
    
    # Mock original TUI class for demonstration
    class MockRevolutionaryTUI:
        def __init__(self, cli_config=None, revolutionary_components=None):
            self.cli_config = cli_config
            self.components = revolutionary_components or {}
            
        async def run(self) -> int:
            print("🎭 Mock Revolutionary TUI running...")
            await asyncio.sleep(2)  # Simulate TUI operation
            print("🎭 Mock Revolutionary TUI completed")
            return 0
            
        async def cleanup(self):
            print("🧹 Mock TUI cleanup")
    
    # Create and run reliable TUI
    reliable_tui = ReliableTUIInterface(MockRevolutionaryTUI)
    return await reliable_tui.run_with_reliability()


# Integration instructions for existing TUI
def integration_instructions():
    """
    Print instructions for integrating reliability modules with existing TUI.
    """
    instructions = """
    🔧 INTEGRATION INSTRUCTIONS FOR EXISTING TUI:
    
    To fix the TUI hanging issue, follow these steps:
    
    1. Import the reliability modules:
       from .reliability import StartupOrchestrator, TimeoutGuardian, StartupResult
    
    2. Wrap the TUI initialization:
       
       async def run_tui():
           orchestrator = StartupOrchestrator()
           components = {'display': display_manager, 'input': input_handler, ...}
           
           result = await orchestrator.coordinate_startup(components)
           
           if result == StartupResult.SUCCESS:
               return await run_main_loop()
           elif result == StartupResult.FALLBACK:  
               return await run_fallback_mode()
           else:
               return await run_emergency_mode()
    
    3. Protect async operations:
       
       from .reliability import timeout_protection
       
       async def risky_operation():
           async with timeout_protection("operation_name", 5.0):
               await potentially_hanging_operation()
    
    4. Use in Revolutionary TUI Interface:
       
       # In revolutionary_tui_interface.py
       from .reliability import coordinate_tui_startup, get_global_guardian
       
       async def initialize(self):
           components = {
               'orchestrator': self.orchestrator,
               'display': self.display_manager,  
               'input': self.input_handler
           }
           
           result = await coordinate_tui_startup(components)
           return result == StartupResult.SUCCESS
    
    ✅ This will prevent the TUI from hanging and ensure startup completes within 10 seconds.
    """
    
    print(instructions)


if __name__ == "__main__":
    print("🧪 Running Reliable TUI Integration Demo")
    print("=" * 60)
    
    # Run the demo
    result = asyncio.run(run_reliable_tui_demo())
    
    print(f"\n📊 Demo completed with exit code: {result}")
    
    # Show integration instructions
    integration_instructions()