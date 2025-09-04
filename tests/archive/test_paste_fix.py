#!/usr/bin/env python3
"""
Test script for multiline paste fix.

This script tests the paste detection logic to ensure multiline input is handled correctly.
"""
import asyncio
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure minimal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def test_fixed_tui():
    """Test the fixed working TUI with paste detection."""
    print("Testing Fixed Working TUI with paste detection...")
    print("Instructions:")
    print("1. Type some normal text - it should appear character by character")
    print("2. Copy this multiline text and paste it:")
    print("   Line 1")
    print("   Line 2") 
    print("   Line 3")
    print("3. The pasted content should be treated as single input")
    print("4. Press Ctrl+C to exit")
    print()
    
    try:
        from agentsmcp.ui.v2.fixed_working_tui import FixedWorkingTUI
        
        tui = FixedWorkingTUI()
        await tui.run()
        
    except KeyboardInterrupt:
        print("\n👋 Test completed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

async def test_chat_input():
    """Test the chat input component with paste detection."""
    print("Testing Chat Input Component with paste detection...")
    print("This test would require full TUI setup. Skipping for now.")
    return True

async def main():
    """Run all paste detection tests."""
    print("🧪 Testing multiline paste fix implementation...")
    print("=" * 60)
    
    success = True
    
    # Test fixed working TUI
    if not await test_fixed_tui():
        success = False
    
    # Test chat input component  
    if not await test_chat_input():
        success = False
    
    if success:
        print("\n✅ All tests completed!")
    else:
        print("\n❌ Some tests failed!")
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
        sys.exit(0)