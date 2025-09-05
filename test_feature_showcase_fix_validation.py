#!/usr/bin/env python3
"""
Simple validation test to demonstrate that the feature showcase truncation issue is fixed.

This test validates:
1. No more truncation from 385 chars to 121 chars
2. Full message flow works end-to-end
3. Rich formatting renders correctly
"""

import sys
import os
import time
from unittest.mock import Mock

# Add the source directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine, ChatMessage, MessageRole
from agentsmcp.ui.v3.console_message_formatter import ConsoleMessageFormatter
from agentsmcp.capabilities.feature_detector import FeatureDetector, FeatureDetectionResult


def test_showcase_fix():
    """Validate that the showcase truncation issue is fixed."""
    print("FEATURE SHOWCASE FIX VALIDATION")
    print("=" * 50)
    
    # 1. Create a realistic showcase message
    result = FeatureDetectionResult(
        exists=True,
        feature_type="cli_flag",
        detection_method="cli_help_analysis", 
        evidence=["Found '--version' in help output", "Direct testing confirmed"],
        usage_examples=["./agentsmcp --version", "./agentsmcp -v"],
        related_features=["help", "verbose", "debug"],
        confidence=0.95
    )
    
    detector = FeatureDetector()
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        showcase_message = loop.run_until_complete(
            detector.generate_feature_showcase(result)
        )
    finally:
        loop.close()
    
    print(f"1. Generated showcase message: {len(showcase_message)} characters")
    print(f"   Preview: {repr(showcase_message[:100])}...")
    
    # 2. Test ChatEngine processing (the key fix)
    captured_messages = []
    captured_statuses = []
    
    def capture_message(msg):
        captured_messages.append(msg)
        print(f"   ✅ Message captured: role={msg.role.value}, length={len(msg.content)}")
    
    def capture_status(status):
        captured_statuses.append(status)
        print(f"   ❌ Unexpected status: {repr(status[:50])}...")
    
    chat_engine = ChatEngine()
    chat_engine.set_callbacks(
        message_callback=capture_message,
        status_callback=capture_status
    )
    
    # 3. Process the FEATURE_SHOWCASE message
    tasktracker_message = f"FEATURE_SHOWCASE:{showcase_message}"
    print(f"\n2. TaskTracker sends: {len(tasktracker_message)} characters")
    
    chat_engine._notify_status(tasktracker_message)
    
    # 4. Validate results
    print(f"\n3. Results:")
    print(f"   Messages captured: {len(captured_messages)}")
    print(f"   Statuses captured: {len(captured_statuses)} (should be 0)")
    
    if len(captured_messages) == 1 and len(captured_statuses) == 0:
        message = captured_messages[0]
        print(f"   ✅ SUCCESS: Message routed correctly")
        print(f"   ✅ Role: {message.role.value}")
        print(f"   ✅ Content length: {len(message.content)} chars")
        print(f"   ✅ Has correct prefix: {message.content.startswith('FEATURE_SHOWCASE_FORMAT:')}")
        
        # Extract showcase content
        extracted = message.content[24:]  # Remove prefix
        if extracted == showcase_message:
            print(f"   ✅ Content integrity: Perfect match")
        else:
            print(f"   ❌ Content integrity: Mismatch")
            return False
        
    else:
        print(f"   ❌ FAILED: Wrong number of messages/statuses")
        return False
    
    # 5. Test Rich formatting
    print(f"\n4. Testing Rich formatting...")
    from rich.console import Console
    from io import StringIO
    
    output_capture = StringIO()
    console = Console(file=output_capture, width=100, force_terminal=True)
    formatter = ConsoleMessageFormatter(console)
    
    try:
        formatter.format_feature_showcase(showcase_message)
        rich_output = output_capture.getvalue()
        print(f"   ✅ Rich output generated: {len(rich_output)} characters")
        
        # Check for key elements
        checks = [
            ("Feature Already Available" in rich_output, "Title present"),
            ("cli_flag already exists" in rich_output, "Main message present"),
            ("agentsmcp" in rich_output and "--version" in rich_output, "Usage example present"),
            ("Related features" in rich_output, "Related features present"),
            ("Detection evidence" in rich_output, "Evidence present"),
            (len(rich_output) > 2000, "Substantial output (>2000 chars)")
        ]
        
        all_good = True
        for check, desc in checks:
            if check:
                print(f"   ✅ {desc}")
            else:
                print(f"   ❌ {desc}")
                all_good = False
        
        if not all_good:
            return False
        
    except Exception as e:
        print(f"   ❌ Rich formatting failed: {e}")
        return False
    
    print(f"\n5. OVERALL RESULT: ✅ FEATURE SHOWCASE FIX VALIDATED")
    print("   - No truncation (385 -> 121 chars) anymore")
    print("   - Message routing works correctly")
    print("   - Rich formatting renders complete content")
    print("   - Full content preserved through pipeline")
    
    return True


def main():
    """Run the validation test."""
    try:
        success = test_showcase_fix()
        if success:
            print("\n🎉 The feature showcase truncation issue has been FIXED!")
            return 0
        else:
            print("\n❌ The fix is not working correctly.")
            return 1
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())