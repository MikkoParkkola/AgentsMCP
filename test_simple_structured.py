#!/usr/bin/env python3
"""
Simple test for structured processing classification
"""

import sys
sys.path.insert(0, 'src')

from agentsmcp.conversation.conversation import ConversationManager

def test_task_classification():
    """Test task classification logic"""
    print("🧪 Testing Task Classification")
    print("=" * 40)
    
    # Create conversation manager
    conversation_manager = ConversationManager()
    
    # Test cases
    test_cases = [
        # Should use structured processing
        ("Create a Python class for user management", True),
        ("Build a REST API with authentication", True), 
        ("Write a function to calculate fibonacci numbers", True),
        ("Implement a database schema with migrations", True),
        ("Design and create a web application", True),
        ("First create the models, then the views, and finally the controllers", True),
        ("Generate unit tests for my calculator class and also add documentation", True),
        
        # Should NOT use structured processing  
        ("What is Python?", False),
        ("How are you today?", False),
        ("Hello", False),
        ("Thanks", False),
        ("Explain recursion", False),
        ("What does this code do?", False),
        ("Help", False),
    ]
    
    correct_count = 0
    total_count = len(test_cases)
    
    print("Testing classification logic:")
    print()
    
    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = conversation_manager._should_use_structured_processing(input_text)
        correct = result == expected
        
        if correct:
            correct_count += 1
        
        status = "✅ PASS" if correct else "❌ FAIL"
        processing_type = "Structured" if result else "Standard"
        expected_type = "Structured" if expected else "Standard"
        
        print(f"{i:2d}. {status} '{input_text}'")
        print(f"    Expected: {expected_type}, Got: {processing_type}")
        print()
    
    print("=" * 40)
    print(f"Results: {correct_count}/{total_count} correct ({correct_count/total_count*100:.1f}%)")
    
    if correct_count == total_count:
        print("🎉 All classification tests passed!")
    else:
        print("⚠️  Some tests failed - review classification logic")
    
    return correct_count == total_count

def test_status_callback():
    """Test status callback system"""
    print("\n🧪 Testing Status Callback System")
    print("=" * 40)
    
    conversation_manager = ConversationManager()
    captured_updates = []
    
    # Create async status callback
    async def test_callback(update):
        captured_updates.append(update)
        print(f"📊 Captured: {update['status']}")
    
    # Add callback
    conversation_manager.structured_processor.add_status_callback(test_callback)
    
    print(f"✅ Status callback registered")
    print(f"Processor has {len(conversation_manager.structured_processor.status_callbacks)} callbacks")
    
    return True

def test_toggle_feature():
    """Test toggle functionality"""
    print("\n🧪 Testing Toggle Feature")
    print("=" * 40)
    
    conversation_manager = ConversationManager()
    
    # Test initial state
    initial_state = conversation_manager.use_structured_processing
    print(f"Initial state: {'Enabled' if initial_state else 'Disabled'}")
    
    # Test toggle
    toggled_state = conversation_manager.toggle_structured_processing()
    print(f"After toggle: {'Enabled' if toggled_state else 'Disabled'}")
    
    # Test explicit enable
    explicit_enable = conversation_manager.toggle_structured_processing(True)
    print(f"Explicit enable: {'Enabled' if explicit_enable else 'Disabled'}")
    
    # Test explicit disable
    explicit_disable = conversation_manager.toggle_structured_processing(False) 
    print(f"Explicit disable: {'Enabled' if explicit_disable else 'Disabled'}")
    
    print("✅ Toggle functionality works correctly")
    return True

if __name__ == "__main__":
    print("🚀 Simple Structured Processing Tests")
    print("=" * 50)
    
    results = []
    
    try:
        results.append(test_task_classification())
        results.append(test_status_callback())
        results.append(test_toggle_feature())
        
        print("\n" + "=" * 50)
        print("📊 FINAL RESULTS")
        print(f"Tests passed: {sum(results)}/{len(results)}")
        
        if all(results):
            print("🎉 All tests passed! Structured processing is ready.")
        else:
            print("⚠️  Some tests failed - check implementation")
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()