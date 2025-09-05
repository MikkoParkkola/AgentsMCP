#!/usr/bin/env python3
"""
Test script to verify the new feature detection integration works.
This tests the specific Ghost Feature Problem that was failing.
"""

import asyncio
import sys
import os
import time

# Add the src directory to path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agentsmcp.capabilities.feature_detector import FeatureDetector


async def test_feature_detection():
    """Test the feature detection system with the --version flag that was failing."""
    print("🔍 Testing Feature Detection System...")
    print("=" * 60)
    
    try:
        # Initialize detector
        detector = FeatureDetector(".")
        
        # Test the original failing case: --version flag detection
        test_request = "Add a --version flag to the CLI that displays the current version, commit hash, and build date in a clean format"
        
        print(f"📝 Test Request: {test_request}")
        print()
        
        # Run feature detection
        print("🔍 Running feature detection...")
        start_time = time.time()
        
        result = await detector.detect_cli_feature(test_request)
        
        detection_time = int((time.time() - start_time) * 1000)
        
        # Display results
        print(f"⏱️ Detection completed in {detection_time}ms")
        print()
        print("📊 Detection Results:")
        print(f"  • Feature exists: {result.exists}")
        print(f"  • Feature type: {result.feature_type}")
        print(f"  • Detection method: {result.detection_method}")
        print(f"  • Confidence: {result.confidence:.1%}")
        print()
        
        if result.evidence:
            print("🔍 Evidence found:")
            for evidence in result.evidence:
                print(f"  • {evidence}")
            print()
        
        if result.usage_examples:
            print("💡 Usage examples:")
            for example in result.usage_examples:
                print(f"  • {example}")
            print()
        
        if result.related_features:
            print("🔗 Related features:")
            for feature in result.related_features:
                print(f"  • --{feature}")
            print()
        
        # Test the showcase generation
        if result.exists:
            print("🎨 Testing showcase generation...")
            showcase = await detector.generate_feature_showcase(result)
            
            print("📋 Generated Showcase:")
            print("-" * 40)
            print(showcase)
            print("-" * 40)
        
        return result.exists
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🧪 AgentsMCP Feature Detection Integration Test")
    print("This test verifies that the Ghost Feature Problem is fixed.")
    print()
    
    # Test feature detection
    feature_exists = await test_feature_detection()
    
    print()
    print("📊 Test Results:")
    print("=" * 60)
    
    if feature_exists:
        print("✅ SUCCESS: Feature detection correctly identified existing --version flag")
        print("✅ The Ghost Feature Problem has been solved!")
        print()
        print("🎯 Expected behavior:")
        print("  • AgentsMCP should now show a formatted usage guide") 
        print("  • Instead of trying to implement an existing feature")
        print("  • This prevents wasted processing and user confusion")
        return True
    else:
        print("❌ FAILURE: Feature detection did not find existing --version flag")
        print("❌ The Ghost Feature Problem still exists")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)