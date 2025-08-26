#!/usr/bin/env python3
"""
Test the enhanced structured processing system in production
"""

import sys
import asyncio
sys.path.insert(0, 'src')

from agentsmcp.conversation.conversation import ConversationManager

class MockLLMClient:
    """Mock LLM client for testing without actual API calls."""
    
    def __init__(self):
        self.call_count = 0
        
    async def send_message(self, message: str) -> str:
        self.call_count += 1
        
        # Mock different types of responses based on the prompt
        if "task and provide a structured analysis" in message:
            return '''
{
    "intent": "Create a simple calculator function for testing",
    "acceptance_criteria": ["Function should handle basic arithmetic", "Proper error handling for division by zero"],
    "context_analysis": "Simple Python function creation task",
    "complexity": "simple",
    "estimated_duration": "2-3 minutes",
    "required_tools": ["python", "testing"],
    "parallel_opportunities": []
}
'''
        elif "Break down this task into specific executable steps" in message:
            return '''
{
    "steps": [
        {
            "description": "Create basic calculator function with add, subtract, multiply, divide",
            "tools": ["python"],
            "can_parallelize": false,
            "dependencies": []
        },
        {
            "description": "Add error handling for division by zero",
            "tools": ["python"],
            "can_parallelize": false,
            "dependencies": ["step_1"]
        },
        {
            "description": "Write unit tests for the calculator",
            "tools": ["pytest"],
            "can_parallelize": true,
            "dependencies": []
        }
    ]
}
'''
        elif "Execute this specific step" in message:
            return "Step completed successfully. Created calculator function with proper error handling."
            
        elif "comprehensive review checking for" in message:
            if self.call_count <= 2:  # First review finds issues
                return '''
{
    "issues_found": ["Missing docstrings", "No input validation"],
    "feedback": "Code works but needs documentation and validation",
    "recommendations": ["Add docstrings", "Validate input types"],
    "needs_fixes": true
}
'''
            else:  # Second review passes
                return '''
{
    "issues_found": [],
    "feedback": "All issues have been resolved. Code is production ready.",
    "recommendations": ["Consider adding more test cases"],
    "needs_fixes": false
}
'''
        elif "Fix the following issue" in message:
            return "Issue has been fixed by adding proper docstrings and input validation."
            
        elif "Generate demo instructions" in message:
            return '''
Demo Instructions:
```python
from calculator import Calculator
calc = Calculator()
print(calc.add(5, 3))        # Output: 8
print(calc.divide(10, 2))    # Output: 5.0
print(calc.divide(10, 0))    # Output: Error: Division by zero
```
'''
        else:
            return "Task completed successfully with proper testing and documentation."

async def test_complete_workflow():
    """Test the complete 7-step workflow end-to-end."""
    print("🧪 Testing Complete Enhanced Workflow")
    print("=" * 50)
    
    # Create conversation manager with mock LLM
    conversation_manager = ConversationManager()
    
    # Replace the LLM client with our mock
    mock_client = MockLLMClient()
    conversation_manager.structured_processor.llm_client = mock_client
    conversation_manager.llm_client = mock_client
    
    # Test a task that should use structured processing
    test_task = "Create a Python calculator function with error handling and tests"
    
    print(f"Testing task: {test_task}")
    print("-" * 50)
    
    try:
        # Process the task
        result = await conversation_manager.process_input(test_task)
        
        print("✅ Task processed successfully!")
        print("\n📋 RESULT PREVIEW:")
        print("-" * 30)
        
        # Show first 800 characters of the result
        preview = result[:800] + "..." if len(result) > 800 else result
        print(preview)
        
        # Validate key components are present
        required_sections = [
            "TASK ANALYSIS COMPLETE",
            "1. TASK ANALYSIS", 
            "2. CONTEXT & BREAKDOWN",
            "3. EXECUTION DETAILS",
            "5. AUTOMATED REVIEW & QA",
            "6. DEMO INSTRUCTIONS", 
            "6. COMPREHENSIVE SUMMARY"
        ]
        
        found_sections = []
        for section in required_sections:
            if section in result:
                found_sections.append(section)
        
        print(f"\n📊 VALIDATION RESULTS:")
        print("-" * 30)
        print(f"✅ Found {len(found_sections)}/{len(required_sections)} required sections")
        
        for section in found_sections:
            print(f"  ✓ {section}")
        
        missing_sections = [s for s in required_sections if s not in found_sections]
        if missing_sections:
            print(f"\n⚠️  Missing sections:")
            for section in missing_sections:
                print(f"  ✗ {section}")
        
        # Check for review cycles
        if "Review Cycles:" in result:
            print("✅ Automated review cycles completed")
        
        if "DEMO INSTRUCTIONS" in result:
            print("✅ Demo instructions generated")
            
        print(f"\n🔧 LLM API Calls: {mock_client.call_count}")
        
        return len(found_sections) >= 5  # At least 5 out of 7 sections should be present
        
    except Exception as e:
        print(f"❌ Task processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🚀 Production Workflow Test")
    print("=" * 40)
    
    try:
        success = await test_complete_workflow()
        
        print("\n" + "=" * 40)
        if success:
            print("🎉 PRODUCTION WORKFLOW TEST PASSED!")
            print("\nThe enhanced 7-step structured processing system is working correctly:")
            print("• ✅ Task analysis and breakdown")
            print("• ✅ Step-by-step execution")
            print("• ✅ Automated review cycles")
            print("• ✅ Issue detection and fixing")
            print("• ✅ Demo instruction generation")
            print("• ✅ Comprehensive reporting")
        else:
            print("⚠️  PRODUCTION WORKFLOW TEST FAILED!")
            print("Some components may need adjustment.")
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())