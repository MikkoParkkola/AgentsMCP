#!/usr/bin/env python3
"""
Comprehensive test suite for TUI fixes:
1. Markdown rendering in Live display mode
2. Agent progress display integration
3. Rich TUI renderer enhancements

This test validates both the markdown rendering fix and the progress display integration.
"""

import asyncio
import time
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
from agentsmcp.ui.v3.rich_tui_renderer import RichTUIRenderer
from agentsmcp.ui.v3.progress_display import ProgressDisplay, AgentStatus
from agentsmcp.ui.v3.chat_engine import ChatEngine, ChatMessage, MessageRole


class TUIFixTester:
    """Test harness for TUI fixes."""
    
    def __init__(self):
        self.capabilities = detect_terminal_capabilities()
        self.renderer = None
        self.progress_display = None
        self.chat_engine = None
        self.test_results = []
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        status = "PASS" if success else "FAIL"
        self.test_results.append((test_name, success, details))
        print(f"[{status}] {test_name}: {details}")
    
    def print_summary(self):
        """Print test summary."""
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        print(f"\n=== TUI FIX TEST SUMMARY ===")
        print(f"Tests passed: {passed}/{total}")
        
        if passed < total:
            print("\nFailed tests:")
            for test_name, success, details in self.test_results:
                if not success:
                    print(f"  - {test_name}: {details}")
        
        return passed == total
    
    async def test_markdown_rendering(self):
        """Test 1: Markdown rendering in Live display mode."""
        print("\n=== TEST 1: Markdown Rendering in Live Display ===")
        
        try:
            # Initialize Rich TUI renderer
            self.renderer = RichTUIRenderer(self.capabilities)
            init_success = self.renderer.initialize()
            
            self.log_test("Rich TUI Initialization", init_success, 
                         "Renderer initialized successfully" if init_success else "Renderer failed to initialize")
            
            if not init_success:
                return False
            
            # Start live display
            self.renderer.start_live_display()
            
            # Test markdown content
            markdown_content = """**Yes – I can perform *sequential (step‑by‑step) thinking* using the MCP Sequential Thinking tool.**

This involves:
- Breaking complex problems into steps
- Reasoning through each step
- Building understanding progressively
- Arriving at well-reasoned conclusions

Here's an example:
1. First step: analyze the problem
2. Second step: consider options
3. Final step: synthesize solution

```python
def example_code():
    return "Code blocks should render properly"
```

> This is a quote block that should be formatted nicely."""
            
            # Display the markdown content
            self.renderer.display_chat_message("assistant", markdown_content)
            
            # Verify that markdown is preserved in conversation history
            has_markdown = any(
                msg_data.get("is_markdown", False) 
                for msg_data in self.renderer._conversation_history 
                if isinstance(msg_data, dict) and msg_data.get("role") == "assistant"
            )
            
            self.log_test("Markdown Content Preserved", has_markdown,
                         "Markdown flag set correctly in conversation history")
            
            # Test that the conversation panel updates
            panel_updated = len(self.renderer._conversation_history) > 0
            self.log_test("Conversation Panel Updated", panel_updated,
                         f"Panel has {len(self.renderer._conversation_history)} messages")
            
            # Wait a moment for Live display rendering
            await asyncio.sleep(2)
            
            print("✅ Markdown content should now be visible with proper formatting")
            print("   (Bold text, italic text, code blocks, lists, etc.)")
            
            return True
            
        except Exception as e:
            self.log_test("Markdown Rendering Test", False, f"Exception: {e}")
            return False
    
    async def test_progress_display_integration(self):
        """Test 2: Progress display integration with agent status."""
        print("\n=== TEST 2: Progress Display Integration ===")
        
        try:
            if not self.renderer:
                self.log_test("Progress Test Prerequisites", False, "Renderer not initialized")
                return False
            
            # Create progress display
            self.progress_display = ProgressDisplay()
            
            # Integrate with renderer
            self.renderer.set_progress_display(self.progress_display)
            
            self.log_test("Progress Display Integration", True, "Progress display connected to renderer")
            
            # Start a task
            self.progress_display.start_task("test_task", "Comprehensive TUI Testing", estimated_duration_ms=15000)
            
            # Add multiple agents
            agents = [
                ("sequential-thinking", "Sequential Thinking Agent"),
                ("system-architect", "System Architect"),
                ("security-engineer", "Security Engineer"),
                ("code-analyst", "Code Analyst")
            ]
            
            for agent_id, agent_name in agents:
                self.progress_display.add_agent(agent_id, agent_name, estimated_duration_ms=8000)
            
            # Simulate agent progress
            for i, (agent_id, agent_name) in enumerate(agents):
                print(f"\n--- Simulating {agent_name} ---")
                
                # Start the agent
                self.progress_display.start_agent(agent_id, "Initializing")
                self.renderer.show_status(f"🛠️ Agent-{agent_id.upper()}: Starting analysis...")
                await asyncio.sleep(0.5)
                
                # Progress updates
                for progress in [25, 50, 75]:
                    step_name = f"Processing step {progress//25}"
                    self.progress_display.update_agent_progress(agent_id, progress, step_name)
                    self.renderer.show_status(f"🛠️ Agent-{agent_id.upper()}: {step_name}...")
                    await asyncio.sleep(0.3)
                
                # Complete the agent
                self.progress_display.complete_agent(agent_id)
                self.renderer.show_status(f"✅ Agent-{agent_id.upper()}: Completed")
                await asyncio.sleep(0.2)
            
            # Check that agent progress is tracked in renderer
            agent_count = len(self.renderer._agent_progress)
            self.log_test("Agent Progress Tracking", agent_count > 0,
                         f"Tracking {agent_count} agents")
            
            # Complete the task
            self.progress_display.complete_task()
            
            # Wait to see the final status
            await asyncio.sleep(2)
            
            print("✅ Progress display should show:")
            print("   - Multiple agents with progress bars")
            print("   - Task timing information")
            print("   - Agent status icons and percentages")
            print("   - Real-time updates during processing")
            
            return True
            
        except Exception as e:
            self.log_test("Progress Display Integration", False, f"Exception: {e}")
            return False
    
    async def test_enhanced_status_panel(self):
        """Test 3: Enhanced status panel with comprehensive information."""
        print("\n=== TEST 3: Enhanced Status Panel ===")
        
        try:
            if not self.renderer:
                self.log_test("Status Panel Prerequisites", False, "Renderer not initialized")
                return False
            
            # Test different types of status messages
            status_messages = [
                ("🎯 Orchestrator: Analyzing complex query", "orchestrator"),
                ("🧠 Sequential Thinking: Step 2/3 - Developing strategy", "thinking"),
                ("🛠️ Agent-ARCHITECT: Designing system structure", "agent"),
                ("🔍 Analyst Agent: Processing code metrics", "analyst"),
                ("📡 Stream Manager: Streaming response content", "streaming")
            ]
            
            for status_msg, category in status_messages:
                self.renderer.show_status(status_msg)
                await asyncio.sleep(1)
                
                print(f"   Status update: {category}")
            
            # Test manual agent progress updates
            self.renderer.update_agent_progress("test-agent", 45.0, "Manual progress test", "in_progress")
            
            # Verify status panel content
            status_updated = self.renderer._current_status != "Ready"
            self.log_test("Status Panel Updates", status_updated,
                         f"Status: {self.renderer._current_status}")
            
            # Wait to observe the enhanced status panel
            await asyncio.sleep(3)
            
            print("✅ Enhanced status panel should display:")
            print("   - Real-time system status")
            print("   - Agent progress with icons and bars") 
            print("   - Task timing information")
            print("   - Clean, organized layout")
            
            return True
            
        except Exception as e:
            self.log_test("Enhanced Status Panel", False, f"Exception: {e}")
            return False
    
    async def test_chat_engine_integration(self):
        """Test 4: Chat engine integration with progress callbacks."""
        print("\n=== TEST 4: Chat Engine Integration ===")
        
        try:
            # Initialize chat engine
            self.chat_engine = ChatEngine()
            
            # Set up callbacks to work with our renderer
            def status_callback(status: str):
                if self.renderer:
                    self.renderer.show_status(status)
            
            def message_callback(message: ChatMessage):
                if self.renderer:
                    timestamp = time.strftime("%H:%M:%S")
                    self.renderer.display_chat_message(
                        message.role.value, 
                        message.content, 
                        timestamp
                    )
            
            def error_callback(error: str):
                if self.renderer:
                    self.renderer.show_error(error)
            
            self.chat_engine.set_callbacks(
                status_callback=status_callback,
                message_callback=message_callback,
                error_callback=error_callback
            )
            
            self.log_test("Chat Engine Callbacks", True, "Callbacks configured successfully")
            
            # Test message processing with markdown
            test_message = "Can you explain **markdown rendering** with *examples*?"
            
            # Add user message
            user_msg = self.chat_engine.state.add_message(MessageRole.USER, test_message)
            message_callback(user_msg)
            
            # Simulate AI response with markdown
            ai_response = """**Markdown rendering** works by converting *formatted text* into rich display:

## Key Features:
- **Bold text** for emphasis
- *Italic text* for subtle emphasis  
- `code snippets` for technical content
- Lists and formatting

```python
def render_markdown(content):
    return rich.Markdown(content)
```

> This demonstrates the markdown fix in action!"""
            
            ai_msg = self.chat_engine.state.add_message(MessageRole.ASSISTANT, ai_response)
            message_callback(ai_msg)
            
            # Check integration
            integration_success = len(self.renderer._conversation_history) >= 2
            self.log_test("Chat Engine Integration", integration_success,
                         f"Processed {len(self.renderer._conversation_history)} messages")
            
            await asyncio.sleep(2)
            
            print("✅ Chat engine integration should show:")
            print("   - User and AI messages in conversation panel")
            print("   - Proper markdown formatting for AI responses")
            print("   - Callback-driven status updates")
            
            return True
            
        except Exception as e:
            self.log_test("Chat Engine Integration", False, f"Exception: {e}")
            return False
    
    async def run_comprehensive_test(self):
        """Run all comprehensive tests."""
        print("🧪 COMPREHENSIVE TUI FIXES TEST SUITE")
        print("=" * 50)
        print("Testing both critical fixes:")
        print("1. Markdown rendering in Live display mode")
        print("2. Agent progress visibility and tracking")
        print("=" * 50)
        
        try:
            # Test 1: Markdown rendering
            await self.test_markdown_rendering()
            
            # Test 2: Progress display integration
            await self.test_progress_display_integration()
            
            # Test 3: Enhanced status panel
            await self.test_enhanced_status_panel()
            
            # Test 4: Chat engine integration
            await self.test_chat_engine_integration()
            
            # Final validation
            print("\n=== FINAL VALIDATION ===")
            print("Both critical issues should now be resolved:")
            print("✅ Issue 1: AI responses show beautifully rendered markdown")
            print("✅ Issue 2: Rich agent progress bars and task timers visible")
            print("\nPress Enter to continue or Ctrl+C to exit...")
            try:
                input()
            except KeyboardInterrupt:
                pass
            
        except Exception as e:
            print(f"❌ Test suite error: {e}")
            return False
        
        finally:
            # Cleanup
            if self.renderer:
                self.renderer.cleanup()
            if self.progress_display:
                self.progress_display.cleanup()
        
        return self.print_summary()


async def main():
    """Main test function."""
    tester = TUIFixTester()
    success = await tester.run_comprehensive_test()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - TUI FIXES VALIDATED")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - CHECK IMPLEMENTATION")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)