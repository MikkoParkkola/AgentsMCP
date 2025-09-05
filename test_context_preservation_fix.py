#!/usr/bin/env python3
"""
TEST: Context Preservation Fix Verification

This test verifies that the ChatEngine now preserves orchestration context for 
follow-up requests, addressing the user's issue #3:
"seems like the context is lost when I ask it to run and complete the plan"
"""

import asyncio
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine, ChatMessage

async def test_context_preservation():
    """Test that follow-up requests maintain orchestration context."""
    print("🧪 CONTEXT PRESERVATION FIX VERIFICATION")
    print("=" * 60)
    print("🎯 Goal: Verify follow-up requests maintain orchestration context")
    print("=" * 60)
    
    try:
        # Create ChatEngine with current directory
        current_dir = os.getcwd()
        print("✓ Creating ChatEngine instance...")
        chat_engine = ChatEngine(current_dir)
        print("✓ ChatEngine created successfully")
        
        # Test scenario: Initial orchestration query followed by "go ahead"
        print(f"\n🧪 TESTING CONTEXT PRESERVATION SCENARIO:")
        print("-" * 60)
        
        # STEP 1: Initial query that should trigger orchestration
        initial_query = "Design a secure user authentication system with proper database design"
        print(f"📤 Step 1 - Initial Query: '{initial_query}'")
        
        # Simulate the first message to build conversation history
        initial_message = ChatMessage(role="user", content=initial_query)
        chat_engine.state.messages.append(initial_message)
        
        # Add a simulated assistant response with orchestration indicators
        orchestration_response = """Agent delegation completed: 2 specialists provided analysis.
[security-engineer] SECURITY ANALYSIS: Authentication requires secure password handling, session management, and input validation. Implement HTTPS, secure cookies, rate limiting, and consider MFA. Audit for OWASP Top 10 vulnerabilities including injection attacks and broken authentication.
[system-architect] ARCHITECTURAL ANALYSIS: Recommended approach involves defining clear service boundaries, API contracts, and data flow patterns. Consider scalability, maintainability, and deployment strategy."""
        
        response_message = ChatMessage(role="assistant", content=orchestration_response)
        chat_engine.state.messages.append(response_message)
        print("✓ Simulated initial orchestration response added to conversation history")
        
        # STEP 2: Test routing decision for follow-up queries
        follow_up_queries = [
            "go ahead",
            "continue",
            "proceed with implementation", 
            "do it",
            "execute the plan"
        ]
        
        print(f"\n🔍 TESTING ROUTING DECISIONS:")
        print("-" * 40)
        
        results = []
        for query in follow_up_queries:
            print(f"\n📋 Testing: '{query}'")
            
            try:
                # Test the routing logic
                route, word_count = chat_engine._route_input(query)
                
                print(f"   Word count: {word_count}")
                print(f"   Route decision: {route}")
                
                # Verify this is routed to orchestration
                expected_route = "preprocessed"  # Should maintain orchestration
                success = (route == expected_route)
                
                if success:
                    print(f"   ✅ SUCCESS: Correctly routed to '{route}' (maintaining orchestration)")
                else:
                    print(f"   ❌ FAILED: Incorrectly routed to '{route}' (should be '{expected_route}')")
                
                results.append({
                    "query": query,
                    "word_count": word_count,
                    "route": route,
                    "success": success
                })
                
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                results.append({
                    "query": query,
                    "word_count": 0,
                    "route": "error",
                    "success": False
                })
        
        # STEP 3: Test control case - new standalone short query (should go direct)
        print(f"\n🔍 TESTING CONTROL CASE (no orchestration context):")
        print("-" * 50)
        
        # Clear conversation history to remove orchestration context  
        chat_engine.state.messages.clear()
        
        control_query = "hello there"
        print(f"📋 Control Query: '{control_query}' (no orchestration context)")
        
        route, word_count = chat_engine._route_input(control_query)
        print(f"   Word count: {word_count}")
        print(f"   Route decision: {route}")
        
        control_success = (route == "direct")  # Should go direct without context
        if control_success:
            print(f"   ✅ SUCCESS: Correctly routed to 'direct' (no orchestration context)")
        else:
            print(f"   ❌ FAILED: Incorrectly routed to '{route}' (should be 'direct')")
        
        # OVERALL RESULTS
        print(f"\n📊 OVERALL RESULTS:")
        print("-" * 60)
        successful_follow_ups = sum(1 for r in results if r["success"])
        print(f"✅ Successful follow-up routings: {successful_follow_ups}/{len(results)}")
        print(f"✅ Control case success: {'Yes' if control_success else 'No'}")
        
        overall_success = (successful_follow_ups == len(results)) and control_success
        
        if overall_success:
            print(f"\n🎉 CONTEXT PRESERVATION FIX VERIFIED!")
            print("   - Follow-up requests now maintain orchestration context")
            print("   - Short standalone queries still route to direct mode")
            print("   - Context-aware routing working as expected")
        else:
            print(f"\n⚠️ PARTIAL SUCCESS:")
            for result in results:
                if not result["success"]:
                    print(f"   Failed: '{result['query']}' -> {result['route']}")
            if not control_success:
                print(f"   Control case failed: {control_query} should go direct")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Context Preservation Fix Test")
    print("🎯 This test verifies that follow-up requests maintain orchestration context")
    print("   while standalone short queries still route to direct mode")
    print()
    
    success = asyncio.run(test_context_preservation())
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 CONTEXT PRESERVATION FIX VERIFIED!")
        print("   Follow-up requests like 'go ahead' now maintain orchestration.")
    else:
        print("❌ CONTEXT PRESERVATION FIX NEEDS MORE WORK")
        print("   Some routing decisions are not working as expected.")
    
    print("=" * 60)
    sys.exit(0 if success else 1)