#!/usr/bin/env python3
"""
Test script to verify accessibility features in TUI.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src directory to Python path
repo_root = Path(__file__).parent
src_dir = repo_root / "src"
sys.path.insert(0, str(src_dir))

async def test_accessibility_features():
    """Test that accessibility features work correctly."""
    try:
        from agentsmcp.ui.modern_tui import ModernTUI
        from rich.console import Console
        
        console = Console()
        
        # Create TUI app with mock parameters
        app = ModernTUI(
            config=None,  # Mock config
            theme_manager=None,  # Mock theme manager
            conversation_manager=None,  # Mock conversation manager
            orchestration_manager=None,  # Mock orchestration manager
            theme="auto",
            no_welcome=True
        )
        
        print("✅ ModernTUI initialized")
        
        # Test 1: Initial accessibility state
        high_contrast = app._accessibility_config.get("high_contrast", False)
        reduce_motion = app._accessibility_config.get("reduce_motion", False)
        increase_spacing = app._accessibility_config.get("increase_spacing", False)
        
        print(f"✅ Initial state - High contrast: {high_contrast}, Motion: {reduce_motion}, Spacing: {increase_spacing}")
        
        # Test 2: Toggle high contrast
        app.set_accessibility_option("high_contrast", True)
        new_high_contrast = app._accessibility_config.get("high_contrast", False)
        if new_high_contrast:
            print("✅ High contrast toggle works")
        else:
            print("❌ High contrast toggle failed")
            return False
        
        # Test 3: Get accessible color
        accessible_green = app._get_accessible_color("green")
        if accessible_green in ["bright_green", "black on bright_green"]:
            print(f"✅ Accessible color mapping works: green -> {accessible_green}")
        else:
            print(f"❌ Accessible color mapping failed: green -> {accessible_green}")
            return False
        
        # Test 4: Toggle motion reduction
        app.set_accessibility_option("reduce_motion", True)
        new_reduce_motion = app._accessibility_config.get("reduce_motion", False)
        if new_reduce_motion:
            print("✅ Motion reduction toggle works")
        else:
            print("❌ Motion reduction toggle failed")
            return False
        
        # Test 5: Get motion style
        motion_style_with_animation = app._get_motion_style(has_animation=True)
        motion_style_no_animation = app._get_motion_style(has_animation=False)
        
        # When reduce_motion is enabled and has_animation=True, should return {"no_animation": True}
        expected_with_animation = {"no_animation": True}
        expected_no_animation = {}
        
        if motion_style_with_animation == expected_with_animation and motion_style_no_animation == expected_no_animation:
            print("✅ Motion style configuration works")
        else:
            print(f"❌ Motion style configuration failed: with_animation={motion_style_with_animation}, without_animation={motion_style_no_animation}")
            return False
        
        # Test 6: Toggle increased spacing
        app.set_accessibility_option("increase_spacing", True)
        new_increase_spacing = app._accessibility_config.get("increase_spacing", False)
        if new_increase_spacing:
            print("✅ Increased spacing toggle works")
        else:
            print("❌ Increased spacing toggle failed")
            return False
        
        # Test 7: Test that hybrid header rendering method exists (integration test)
        if hasattr(app, '_render_hybrid_header'):
            print("✅ Header rendering method available with accessibility integration")
        else:
            print("❌ Header rendering method missing")
            return False
        
        # Test 8: Reset all options
        app.set_accessibility_option("high_contrast", False)
        app.set_accessibility_option("reduce_motion", False)
        app.set_accessibility_option("increase_spacing", False)
        
        final_high_contrast = app._accessibility_config.get("high_contrast", False)
        final_reduce_motion = app._accessibility_config.get("reduce_motion", False)
        final_increase_spacing = app._accessibility_config.get("increase_spacing", False)
        
        if not final_high_contrast and not final_reduce_motion and not final_increase_spacing:
            print("✅ Accessibility options reset successfully")
        else:
            print("❌ Accessibility options reset failed")
            return False
        
        print("🎉 All accessibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_accessibility_features())
    sys.exit(0 if success else 1)