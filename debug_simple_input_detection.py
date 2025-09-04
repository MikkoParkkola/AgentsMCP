#!/usr/bin/env python3
"""Debug script to test simple input detection logic."""

def _is_simple_input(user_input: str) -> bool:
    """Check if input is a simple greeting or basic query that doesn't need task tracking."""
    input_lower = user_input.lower().strip()
    
    # First check: if input is too long, it's not simple (prevents misclassification)
    print(f"Input: '{user_input}'")
    print(f"Lowercased: '{input_lower}'")
    print(f"Length: {len(input_lower)}")
    
    if len(input_lower) > 50:
        print("❌ FAIL: Too long (>50 characters)")
        return False
    
    # Enhanced simple patterns to prevent infinite loops on basic interactions
    simple_patterns = [
        "hello", "hi", "hey", "howdy", "greetings",
        "thanks", "thank you", "thx", "ty", 
        "bye", "goodbye", "see you", "cya", "farewell",
        "ok", "okay", "yes", "no", "sure", "please", "help",
        "how are you", "what's up", "whats up", "sup", "how's it going",
        "good morning", "good afternoon", "good evening", "good night",
        "nice", "cool", "awesome", "great", "perfect",
        "who are you", "what are you", "are you there"
    ]
    
    # Simple question words that often lead to infinite loops
    simple_question_words = ["who", "what", "when", "where", "why", "how"]
    
    words = input_lower.split()
    print(f"Word count: {len(words)}")
    print(f"Words: {words}")
    
    # Check for exact matches with simple patterns
    stripped_input = input_lower.rstrip('?!.,')
    print(f"Stripped input: '{stripped_input}'")
    
    for pattern in simple_patterns:
        if pattern == stripped_input:
            print(f"✅ MATCH: Exact match with pattern '{pattern}'")
            return True
    
    print("❌ No exact pattern matches")
    
    # Check for short patterns that are mostly simple (stricter matching)
    if len(words) <= 3:
        print(f"Checking short pattern matches (≤3 words)")
        for pattern in simple_patterns:
            if pattern in input_lower and len(input_lower) <= 25:
                print(f"✅ MATCH: Pattern '{pattern}' found in short input")
                return True
        print("❌ No short pattern matches")
        
    # Single question word queries (like "what?", "how?")
    if len(words) == 1 and any(word.rstrip('?!.,') in simple_question_words for word in words):
        print(f"✅ MATCH: Single question word")
        return True
        
    # Very short inputs are likely simple (but not too short to be empty)
    if len(input_lower) <= 15 and len(input_lower.strip()) > 0:
        print(f"✅ MATCH: Very short input (≤15 chars)")
        return True
    
    print("❌ FAIL: No matches found")
    return False

# Test cases
test_cases = [
    "hello",
    "how are you today?",
    "how are you",
    "what's up",
    "thanks",
    "bye"
]

print("🔍 DEBUGGING SIMPLE INPUT DETECTION")
print("=" * 50)

for test_input in test_cases:
    print(f"\n🧪 Testing: '{test_input}'")
    print("-" * 30)
    result = _is_simple_input(test_input)
    print(f"Result: {'✅ SIMPLE' if result else '❌ COMPLEX'}")
    print()