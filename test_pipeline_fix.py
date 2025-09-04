#!/usr/bin/env python3
"""
Test script to verify the critical execution pipeline fix.
This script tests that preprocessing optimization reaches the LLM execution layer.
"""

import asyncio
import logging
import tempfile
import os
import sys

# Add the project root to sys.path
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

# Configure logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/agentsmcp_pipeline_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

async def test_pipeline_fix():
    """Test that optimized prompts reach the LLM execution layer."""
    print("🧪 Testing AgentsMCP Execution Pipeline Fix")
    print("=" * 50)
    
    # Set TUI mode to prevent console contamination
    os.environ['AGENTSMCP_TUI_MODE'] = '1'
    
    try:
        # Import the chat engine
        from agentsmcp.ui.v3.chat_engine import ChatEngine
        
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize chat engine
            engine = ChatEngine(launch_directory=temp_dir)
            
            # Test complex request that should trigger preprocessing
            complex_request = "Do a comprehensive assessment of this project and tell me why it is better than the competition"
            
            print(f"📝 Testing complex request: {complex_request}")
            print("🔍 Looking for pipeline debug logs...")
            
            # Process the input
            result = await engine.process_input(complex_request)
            
            print(f"✅ Test completed. Result: {result}")
            print("\n📋 Check the log file at: /tmp/agentsmcp_pipeline_test.log")
            print("🔍 Look for [PIPELINE-DEBUG] entries to trace the prompt flow")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner."""
    print("🚀 Starting pipeline fix test...")
    success = await test_pipeline_fix()
    
    if success:
        print("\n✅ Pipeline test completed successfully!")
        print("📝 Review the debug logs to verify optimized prompts are reaching the LLM")
    else:
        print("\n❌ Pipeline test failed!")
        print("🔍 Check the error logs for debugging information")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)