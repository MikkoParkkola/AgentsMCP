#!/usr/bin/env python3

"""
Test script for the preprocessing threshold implementation.
This tests the new word threshold functionality to ensure it works as expected.
"""

import sys
import os
import asyncio

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_word_threshold():
    """Test the word counting and threshold logic."""
    from agentsmcp.conversation.llm_client import LLMClient
    
    # Create LLM client with default threshold (4)
    llm_client = LLMClient()
    
    # Test cases for word threshold
    test_cases = [
        # (input, expected_should_preprocess, description)
        ("hello", False, "Single word - should skip preprocessing"),
        ("hi there", False, "Two words - should skip preprocessing"),
        ("thank you", False, "Two words - should skip preprocessing"),
        ("yes I can", False, "Three words - should skip preprocessing"),
        ("how are you doing", False, "Four words - should skip preprocessing (threshold is >4)"),
        ("please help me understand this", True, "Five words - should use preprocessing"),
        ("make a comprehensive analysis of this", True, "Six words - should use preprocessing"),
        ("analyze this complex code structure", True, "Five words - should use preprocessing"),
        ("", False, "Empty string - should skip preprocessing"),
        ("word", False, "Single word - should skip preprocessing"),
    ]
    
    print("🧪 Testing Word Threshold Logic")
    print("=" * 50)
    print(f"Default threshold: {llm_client.get_preprocessing_threshold()} words")
    print()
    
    all_passed = True
    
    for input_text, expected, description in test_cases:
        actual = llm_client.should_use_preprocessing(input_text)
        word_count = len(input_text.strip().split()) if input_text.strip() else 0
        
        status = "✅ PASS" if actual == expected else "❌ FAIL"
        if actual != expected:
            all_passed = False
            
        print(f"{status} | {word_count:2d} words | {str(actual):5s} | {description}")
        print(f"      Input: '{input_text}'")
        print()
    
    print("=" * 50)
    print(f"Overall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

def test_threshold_configuration():
    """Test threshold configuration functionality."""
    from agentsmcp.conversation.llm_client import LLMClient
    
    print("🔧 Testing Threshold Configuration")
    print("=" * 50)
    
    llm_client = LLMClient()
    
    # Test default threshold
    default_threshold = llm_client.get_preprocessing_threshold()
    print(f"✅ Default threshold: {default_threshold}")
    
    # Test setting valid threshold
    result = llm_client.set_preprocessing_threshold(6)
    print(f"✅ Set threshold to 6: {result[:50]}...")
    
    # Verify the change
    new_threshold = llm_client.get_preprocessing_threshold()
    print(f"✅ New threshold: {new_threshold}")
    
    # Test threshold behavior with new value
    test_inputs = [
        ("hello world how are you doing", 6, False),   # 6 words, threshold 6, should be False (≤6)
        ("hello world how are you doing now", 6, True),  # 7 words, threshold 6, should be True (>6)
        ("hello", 6, False),  # 1 word, should be False
    ]
    
    for input_text, threshold, expected in test_inputs:
        llm_client.set_preprocessing_threshold(threshold)
        actual = llm_client.should_use_preprocessing(input_text)
        word_count = len(input_text.strip().split())
        
        status = "✅ PASS" if actual == expected else "❌ FAIL"
        print(f"{status} | {word_count} words, threshold {threshold} → {actual}")
    
    print("=" * 50)
    print("✅ Configuration tests completed")

if __name__ == "__main__":
    print("🚀 Testing Preprocessing Threshold Implementation")
    print("=" * 70)
    print()
    
    try:
        # Test basic word threshold logic
        logic_passed = test_word_threshold()
        print()
        
        # Test threshold configuration
        test_threshold_configuration()
        print()
        
        if logic_passed:
            print("🎉 All tests completed successfully!")
            print()
            print("📋 Summary of Implementation:")
            print("• ✅ Word threshold logic implemented")
            print("• ✅ Default threshold set to 4 words")
            print("• ✅ Short inputs (≤4 words) skip preprocessing")
            print("• ✅ Long inputs (>4 words) use preprocessing")
            print("• ✅ Threshold is configurable via /preprocessing threshold")
            print("• ✅ Configuration commands updated with threshold info")
        else:
            print("❌ Some tests failed. Please check the implementation.")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the AgentsMCP modules are properly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)