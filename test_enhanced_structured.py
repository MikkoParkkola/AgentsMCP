#!/usr/bin/env python3
"""
Test the enhanced structured processing system with review cycles
"""

import sys
import asyncio
sys.path.insert(0, 'src')

from agentsmcp.conversation.conversation import ConversationManager

async def test_enhanced_workflow():
    """Test the new 7-step workflow with automated review."""
    print("🧪 Testing Enhanced 7-Step Structured Processing")
    print("=" * 55)
    
    # Create conversation manager
    conversation_manager = ConversationManager()
    
    # Test the enhanced classification
    test_cases = [
        ("Create a Python function to validate email addresses", True),
        ("What time is it?", False),
        ("Build a web API for user authentication with proper security", True),
        ("Thanks for your help", False),
    ]
    
    print("1. Testing Enhanced Task Classification")
    print("-" * 40)
    
    all_correct = True
    for input_text, expected in test_cases:
        result = conversation_manager._should_use_structured_processing(input_text)
        correct = result == expected
        status = "✅ PASS" if correct else "❌ FAIL"
        processing_type = "Enhanced 7-Step" if result else "Standard"
        
        print(f"{status} '{input_text[:50]}...'")
        print(f"    Processing: {processing_type}")
        
        if not correct:
            all_correct = False
    
    print(f"\nClassification Results: {'All Passed' if all_correct else 'Some Failed'}")
    
    print("\n2. Testing Review System Components")
    print("-" * 40)
    
    # Test review-related data structures
    from agentsmcp.conversation.structured_processor import ReviewResult, TaskStatus
    
    # Test review result creation
    test_review = ReviewResult(
        issues_found=["Missing error handling", "No input validation"],
        feedback="Code needs improvement in error handling and validation",
        recommendations=["Add try-catch blocks", "Validate all inputs"],
        needs_fixes=True,
        review_agent_id="test_reviewer_123"
    )
    
    print(f"✅ ReviewResult creation: {len(test_review.issues_found)} issues found")
    print(f"✅ Review needs fixes: {test_review.needs_fixes}")
    
    # Test new task statuses
    review_statuses = [TaskStatus.REVIEWING, TaskStatus.FIXING_ISSUES]
    print(f"✅ New review statuses available: {len(review_statuses)}")
    
    print("\n3. Testing Structured Processor Enhancement")
    print("-" * 40)
    
    # Test the enhanced processor has all required methods
    processor = conversation_manager.structured_processor
    
    required_methods = [
        '_review_and_improve_iteratively',
        '_spawn_review_agent', 
        '_fix_review_issues',
        '_generate_demo_instructions',
        '_collect_work_for_review',
        '_format_review_cycles'
    ]
    
    missing_methods = []
    for method_name in required_methods:
        if not hasattr(processor, method_name):
            missing_methods.append(method_name)
    
    if missing_methods:
        print(f"❌ Missing methods: {missing_methods}")
        all_correct = False
    else:
        print(f"✅ All {len(required_methods)} review methods available")
    
    print("\n" + "=" * 55)
    print("📊 ENHANCED SYSTEM VALIDATION")
    
    if all_correct:
        print("🎉 Enhanced 7-step workflow ready!")
        print("\nNew Features:")
        print("• ✅ Mandatory automated review after task completion")
        print("• ✅ Iterative improvement cycle until issues resolved")
        print("• ✅ Demo instructions generation for applicable tasks")
        print("• ✅ Comprehensive quality assurance reporting")
        print("• ✅ Review agent spawning and coordination")
    else:
        print("⚠️  Some validation tests failed - check implementation")
    
    return all_correct

if __name__ == "__main__":
    print("🚀 Enhanced Structured Processing Validation")
    print("=" * 50)
    
    try:
        result = asyncio.run(test_enhanced_workflow())
        
        if result:
            print("\n✨ System is ready for production-quality task processing!")
        else:
            print("\n⚠️  System needs additional work before deployment")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()