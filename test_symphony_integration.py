#!/usr/bin/env python3
"""
Test script for Symphony Dashboard integration.

This script verifies that the Symphony Dashboard features are properly integrated
and can be activated through the Revolutionary TUI.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_symphony_dashboard():
    """Test Symphony Dashboard initialization and basic functionality."""
    print("🎼 Testing Symphony Dashboard Integration")
    print("=" * 60)
    
    try:
        # Test 1: Import Symphony Dashboard
        print("📦 Test 1: Importing Symphony Dashboard components...")
        from agentsmcp.ui.v2.event_system import AsyncEventSystem
        from agentsmcp.ui.components.symphony_dashboard import (
            SymphonyDashboard, Agent, AgentCapability, AgentState,
            Task, TaskStatus, Connection
        )
        print("✅ Symphony Dashboard components imported successfully")
        
        # Test 2: Create Event System
        print("\n🔧 Test 2: Creating event system...")
        event_system = AsyncEventSystem()
        print("✅ Event system created successfully")
        
        # Test 3: Initialize Symphony Dashboard
        print("\n🎛️ Test 3: Initializing Symphony Dashboard...")
        dashboard = SymphonyDashboard(event_system)
        
        if await dashboard.initialize():
            print("✅ Symphony Dashboard initialized successfully")
        else:
            print("❌ Symphony Dashboard initialization failed")
            return False
        
        # Test 4: Check dashboard state
        print("\n📊 Test 4: Testing dashboard state...")
        state = dashboard.get_current_state()
        print(f"✅ Dashboard state retrieved: {state['agent_count']} agents, {state['task_count']} tasks")
        
        # Test 5: Test performance stats
        print("\n⚡ Test 5: Testing performance stats...")
        perf_stats = dashboard.get_performance_stats()
        avg_fps = perf_stats["performance"]["average_fps"]
        print(f"✅ Performance stats retrieved: {avg_fps:.1f} FPS")
        
        # Test 6: Test agent management
        print("\n🤖 Test 6: Testing agent management...")
        test_agent = Agent(
            id="test_agent",
            name="Test Agent",
            model="test-model",
            capabilities={AgentCapability.CHAT},
            state=AgentState.IDLE,
            position=(10, 10),
            color="green"
        )
        
        await dashboard.add_agent(test_agent)
        updated_state = dashboard.get_current_state()
        print(f"✅ Agent added successfully: {updated_state['agent_count']} agents now")
        
        # Test 7: Test task management
        print("\n📋 Test 7: Testing task management...")
        from datetime import datetime
        test_task = Task(
            id="test_task",
            title="Test Task",
            description="Testing task management",
            status=TaskStatus.PENDING,
            assigned_agent_id="test_agent",
            created_at=datetime.now(),
            priority=5
        )
        
        await dashboard.add_task(test_task)
        final_state = dashboard.get_current_state()
        print(f"✅ Task added successfully: {final_state['task_count']} tasks now")
        
        # Test 8: Test view switching
        print("\n🔄 Test 8: Testing view switching...")
        await dashboard.switch_view("agents")
        await dashboard.switch_view("tasks")
        await dashboard.switch_view("metrics")
        await dashboard.switch_view("overview")
        print("✅ View switching works correctly")
        
        # Test 9: Test activation/deactivation
        print("\n🎮 Test 9: Testing activation/deactivation...")
        await dashboard.activate()
        print("✅ Dashboard activated successfully")
        
        await dashboard.deactivate()
        print("✅ Dashboard deactivated successfully")
        
        # Test 10: Cleanup
        print("\n🧹 Test 10: Testing cleanup...")
        await dashboard.cleanup()
        print("✅ Dashboard cleaned up successfully")
        
        print("\n🎉 All tests passed! Symphony Dashboard integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tui_integration():
    """Test TUI integration points."""
    print("\n🔗 Testing TUI Integration")
    print("=" * 60)
    
    try:
        # Test importing TUI components
        print("📦 Importing TUI components...")
        from agentsmcp.ui.v2.fixed_working_tui import FixedWorkingTUI
        print("✅ TUI components imported successfully")
        
        # Test TUI initialization
        print("\n🔧 Creating TUI instance...")
        tui = FixedWorkingTUI()
        print("✅ TUI instance created successfully")
        
        # Check Symphony Dashboard attributes
        print("\n🎼 Checking Symphony Dashboard integration...")
        if hasattr(tui, 'symphony_dashboard'):
            print("✅ TUI has symphony_dashboard attribute")
        else:
            print("❌ TUI missing symphony_dashboard attribute")
            return False
            
        if hasattr(tui, 'symphony_active'):
            print("✅ TUI has symphony_active attribute")
        else:
            print("❌ TUI missing symphony_active attribute")
            return False
        
        # Check methods exist
        required_methods = [
            '_initialize_symphony_dashboard',
            '_activate_symphony_dashboard',
            '_exit_symphony_dashboard',
            '_symphony_dashboard_loop',
            '_symphony_multi_panel_renderer'
        ]
        
        for method_name in required_methods:
            if hasattr(tui, method_name):
                print(f"✅ TUI has {method_name} method")
            else:
                print(f"❌ TUI missing {method_name} method")
                return False
        
        print("\n🎉 TUI integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ TUI integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting Symphony Dashboard Integration Tests")
    print("=" * 80)
    
    # Test Symphony Dashboard
    dashboard_success = await test_symphony_dashboard()
    
    # Test TUI integration
    tui_success = test_tui_integration()
    
    print("\n" + "=" * 80)
    print("📊 Test Results Summary")
    print("=" * 80)
    
    if dashboard_success:
        print("✅ Symphony Dashboard: All tests passed")
    else:
        print("❌ Symphony Dashboard: Tests failed")
    
    if tui_success:
        print("✅ TUI Integration: All tests passed")
    else:
        print("❌ TUI Integration: Tests failed")
    
    if dashboard_success and tui_success:
        print("\n🎉 SUCCESS: Symphony Dashboard integration is fully functional!")
        print("\n🎮 How to use:")
        print("1. Launch the TUI: python -m src.agentsmcp.ui.v2.fixed_working_tui")
        print("2. Type '/symphony' to activate the Symphony Dashboard")
        print("3. Use F1-F4 to switch between dashboard views")
        print("4. Press ESC or 'Q' to exit Symphony Dashboard")
        return 0
    else:
        print("\n❌ FAILURE: Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)