#!/usr/bin/env python3
"""
Test the new structured processing system
"""

import sys
import asyncio
sys.path.insert(0, 'src')

from agentsmcp.config import Config
from agentsmcp.conversation.conversation import ConversationManager
from agentsmcp.conversation.structured_processor import StructuredProcessor

async def test_structured_processing():
    """Test the structured processing workflow"""
    print("🧪 Testing Structured Processing System")
    print("=" * 50)
    
    # Create conversation manager with structured processing
    conversation_manager = ConversationManager()
    
    # Test tasks of varying complexity
    test_tasks = [
        {
            "input": "Create a Python class for managing users",
            "should_use_structured": True,
            "description": "Complex task - should use structured processing"
        },
        {
            "input": "What is 2 + 2?",
            "should_use_structured": False,
            "description": "Simple question - should NOT use structured processing"
        },
        {
            "input": "Build a REST API with FastAPI that includes user authentication, database integration, and API documentation",
            "should_use_structured": True,
            "description": "Very complex multi-step task - should use structured processing"
        },
        {
            "input": "How are you doing today?",
            "should_use_structured": False,
            "description": "Casual conversation - should NOT use structured processing"
        },
        {
            "input": "First, create a database schema, then implement the models, and finally create the API endpoints",
            "should_use_structured": True,
            "description": "Multi-step task - should use structured processing"
        }
    ]
    
    print("1. Testing Task Classification")
    print("-" * 30)
    
    all_correct = True
    for i, task in enumerate(test_tasks, 1):
        should_use = conversation_manager._should_use_structured_processing(task["input"])
        expected = task["should_use_structured"]
        
        status = "✅ CORRECT" if should_use == expected else "❌ WRONG"
        if should_use != expected:
            all_correct = False
            
        print(f"{i}. {status}")
        print(f"   Input: {task['input']}")
        print(f"   Expected: {'Structured' if expected else 'Standard'}")
        print(f"   Got: {'Structured' if should_use else 'Standard'}")
        print(f"   Reason: {task['description']}")
        print()
    
    if all_correct:
        print("✅ All task classification tests passed!")
    else:
        print("❌ Some classification tests failed!")
    
    print("\n" + "=" * 50)
    print("2. Testing Structured Processing Workflow")
    print("-" * 30)
    
    # Test one complex task with the full workflow
    complex_task = "Create a Python calculator class with add, subtract, multiply, divide methods and proper error handling"
    
    print(f"Processing: {complex_task}")
    print()
    
    try:
        # Process the task
        result = await conversation_manager.process_input(complex_task)
        
        print("Response received:")
        print("-" * 20)
        # Show first 500 characters of response
        preview = result[:500] + "..." if len(result) > 500 else result
        print(preview)
        print("-" * 20)
        
        # Check if response contains structured elements
        structured_indicators = ["TASK ANALYSIS", "EXECUTION DETAILS", "SUMMARY", "Step"]
        found_indicators = [indicator for indicator in structured_indicators if indicator in result]
        
        if found_indicators:
            print(f"✅ Structured processing detected! Found: {', '.join(found_indicators)}")
        else:
            print("⚠️  Standard processing was used")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
    
    print("\n" + "=" * 50)
    print("3. Testing Status Updates")
    print("-" * 30)
    
    # Test status callback system
    status_updates = []
    
    async def capture_status(update):
        status_updates.append(update)
        print(f"📊 Status: {update['status']} - {update.get('details', 'No details')}")
    
    # Add status callback
    conversation_manager.structured_processor.add_status_callback(capture_status)
    
    try:
        simple_task = "Write a simple hello world function"
        await conversation_manager.structured_processor.process_task(simple_task)
        
        print(f"✅ Captured {len(status_updates)} status updates")
        for i, update in enumerate(status_updates, 1):
            print(f"  {i}. {update['status']}: {update.get('details', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Status update test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Structured Processing Tests Complete!")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_structured_processing())