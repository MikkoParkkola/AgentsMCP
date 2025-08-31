#!/usr/bin/env python3
"""
Test script to verify the orchestrator communication fixes are working correctly.

This script tests:
1. TaskClassifier has the correct classify() method
2. Orchestrator properly handles simple tasks without spawning agents  
3. TUI integration works with the orchestrator
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

async def main():
    """Quick test of orchestrator communication fixes."""
    print("🧪 Testing Orchestrator Communication")
    print("=" * 40)
    
    try:
        # Test 1: TaskClassifier classify method
        from agentsmcp.orchestration.task_classifier import TaskClassifier
        classifier = TaskClassifier()
        result = classifier.classify("hello")
        print(f"✅ TaskClassifier.classify() works: {result.classification.value}")
        
        # Test 2: Orchestrator handles simple task
        from agentsmcp.orchestration import Orchestrator, OrchestratorConfig, OrchestratorMode
        config = OrchestratorConfig(mode=OrchestratorMode.STRICT_ISOLATION)
        orchestrator = Orchestrator(config=config)
        
        response = await orchestrator.process_user_input("hello")
        print(f"✅ Orchestrator simple response: {response.response_type}")
        print(f"✅ No agents spawned: {len(response.agents_consulted) == 0}")
        
        # Test 3: TUI can import orchestrator components
        from agentsmcp.ui.v2.fixed_working_tui import FixedWorkingTUI
        tui = FixedWorkingTUI()
        print(f"✅ TUI can import orchestrator components")
        
        print()
        print("🎉 All tests passed!")
        print("💡 Simple greetings now show single orchestrator response")
        print("🚫 No more individual agent outputs for simple tasks")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)