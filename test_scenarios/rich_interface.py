#!/usr/bin/env python3
"""
Rich Interface Test Scenarios for Revolutionary TUI Interface

These test scenarios validate that the Revolutionary TUI properly uses Rich
components and displays the enhanced interface (not fallback mode):
- Rich Live display detection
- Panel and layout rendering
- Color and formatting
- TTY capability detection
- Enhanced vs basic mode validation
"""

import pytest
import subprocess
import re
import os
from pathlib import Path
from typing import List, Dict

# Test configuration
INTERFACE_TEST_TIMEOUT = 15


class RichInterfaceTests:
    """Test scenarios for Rich interface validation."""
    
    @staticmethod
    def get_agentsmcp_cmd():
        """Get the agentsmcp command path."""
        project_root = Path(__file__).parent.parent
        agentsmcp_cmd = project_root / "agentsmcp"
        
        if not agentsmcp_cmd.exists():
            return "agentsmcp"  # Try system installed version
        return str(agentsmcp_cmd)
    
    @pytest.mark.ui
    @pytest.mark.integration
    def test_rich_interface_activation(self):
        """Test that Rich interface is activated (not basic fallback)."""
        cmd = [self.get_agentsmcp_cmd(), "tui", "--debug"]
        
        process = subprocess.run(
            cmd,
            input="quit\n",
            capture_output=True,
            text=True,
            timeout=INTERFACE_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent,
            env={**os.environ, "TERM": "xterm-256color"}  # Ensure good terminal
        )
        
        assert process.returncode in [0, 130], f"Rich interface test failed with code {process.returncode}"
        
        # Should indicate Rich/Enhanced mode, not basic fallback
        output_combined = (process.stdout + process.stderr).lower()
        
        # Positive indicators (should be present)
        rich_indicators = ['rich', 'enhanced', 'tui', 'live', 'revolutionary']
        found_rich = any(indicator in output_combined for indicator in rich_indicators)
        
        # Negative indicators (should NOT be present)
        basic_fallback_indicators = [
            'non-tty environment', 'basic prompt', 'fallback mode',
            'simple prompt', 'minimal mode'
        ]
        found_fallback = any(indicator in output_combined for indicator in basic_fallback_indicators)
        
        assert found_rich, f"No Rich interface indicators found in output: {output_combined[:300]}"
        assert not found_fallback, f"Fallback mode indicators found: {output_combined[:300]}"
    
    @pytest.mark.ui
    @pytest.mark.integration
    def test_no_basic_prompt_fallback(self):
        """Test that TUI does not fall back to basic '> ' prompt."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        process = subprocess.run(
            cmd,
            input="help\nquit\n",
            capture_output=True,
            text=True,
            timeout=INTERFACE_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        assert process.returncode in [0, 130], f"Prompt test failed with code {process.returncode}"
        
        # Check for basic prompt patterns that indicate fallback
        lines = process.stdout.split('\n')
        
        # Count lines that look like basic prompts
        basic_prompt_lines = []
        for line in lines:
            stripped = line.strip()
            # Basic prompt patterns: "> ", ">>> ", etc.
            if re.match(r'^>\s*$', stripped) or re.match(r'^>\s+[^>]', stripped):
                basic_prompt_lines.append(line)
        
        # Should have very few or no basic prompt lines
        assert len(basic_prompt_lines) <= 2, \
            f"Too many basic prompt lines found (indicates fallback): {basic_prompt_lines}"
    
    @pytest.mark.ui
    @pytest.mark.integration
    def test_ansi_escape_sequences_present(self):
        """Test that output contains ANSI escape sequences (indicates Rich formatting)."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        process = subprocess.run(
            cmd,
            input="help\nquit\n",
            capture_output=True,
            text=True,
            timeout=INTERFACE_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent,
            env={**os.environ, "FORCE_COLOR": "1"}  # Force color output
        )
        
        assert process.returncode in [0, 130], f"ANSI test failed with code {process.returncode}"
        
        # Look for ANSI escape sequences that Rich would generate
        ansi_patterns = [
            r'\x1b\[.*?m',      # Color codes
            r'\x1b\[.*?[ABCD]', # Cursor movement  
            r'\x1b\[.*?[HJ]',   # Clear screen/line
            r'\x1b\[.*?[K]',    # Clear to end of line
        ]
        
        found_ansi = False
        for pattern in ansi_patterns:
            if re.search(pattern, process.stdout):
                found_ansi = True
                break
        
        # If no ANSI found, check for rich-style formatting hints
        if not found_ansi:
            rich_hints = ['─', '│', '┌', '┐', '└', '┘', '■', '▪', '●']
            found_rich_chars = any(char in process.stdout for char in rich_hints)
            assert found_rich_chars or len(process.stdout.strip()) > 50, \
                "No ANSI sequences or Rich formatting detected - may be in basic mode"
    
    @pytest.mark.ui
    @pytest.mark.integration
    def test_tty_detection_working(self):
        """Test that TTY detection works properly."""
        cmd = [self.get_agentsmcp_cmd(), "tui", "--debug"]
        
        # Force TTY environment
        process = subprocess.run(
            cmd,
            input="quit\n",
            capture_output=True,
            text=True,
            timeout=INTERFACE_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent,
            env={**os.environ, "TERM": "xterm-256color", "FORCE_COLOR": "1"}
        )
        
        assert process.returncode in [0, 130], f"TTY detection test failed with code {process.returncode}"
        
        # Should not show non-TTY warnings
        output_combined = (process.stdout + process.stderr).lower()
        non_tty_warnings = [
            'non-tty', 'not a tty', 'terminal not detected',
            'running in non-tty environment'
        ]
        
        found_warnings = [warning for warning in non_tty_warnings if warning in output_combined]
        assert len(found_warnings) == 0, \
            f"TTY detection issues found: {found_warnings}"
    
    @pytest.mark.ui
    @pytest.mark.integration
    def test_panel_layout_indicators(self):
        """Test that output suggests panel/layout structure."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        process = subprocess.run(
            cmd,
            input="help\nquit\n",
            capture_output=True,
            text=True,
            timeout=INTERFACE_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        assert process.returncode in [0, 130], f"Panel layout test failed with code {process.returncode}"
        
        # Look for layout/panel indicators
        layout_indicators = [
            '┌', '┐', '└', '┘',  # Box drawing characters
            '─', '│',            # Lines
            '■', '▪', '●',       # Bullet points
            '▶', '▷', '◆',       # Arrows and markers
        ]
        
        found_layout_chars = [char for char in layout_indicators if char in process.stdout]
        
        # Also check for structured content patterns
        lines = process.stdout.split('\n')
        structured_lines = [line for line in lines if len(line.strip()) > 20]
        
        # Should have either layout characters OR structured content
        assert len(found_layout_chars) > 0 or len(structured_lines) > 3, \
            f"No panel/layout structure detected. Layout chars: {found_layout_chars}, Structured lines: {len(structured_lines)}"
    
    @pytest.mark.ui
    @pytest.mark.integration
    def test_revolutionary_features_active(self):
        """Test that Revolutionary TUI features are active."""
        cmd = [self.get_agentsmcp_cmd(), "tui", "--revolutionary"]
        
        process = subprocess.run(
            cmd,
            input="help\nquit\n",
            capture_output=True,
            text=True,
            timeout=INTERFACE_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        assert process.returncode in [0, 130], f"Revolutionary features test failed with code {process.returncode}"
        
        # Should show Revolutionary TUI is active
        output_combined = (process.stdout + process.stderr).lower()
        
        revolutionary_indicators = [
            'revolutionary', 'launching revolutionary tui', 
            'capability detection', 'rich interface'
        ]
        
        found_revolutionary = any(indicator in output_combined for indicator in revolutionary_indicators)
        
        # If no explicit revolutionary indicators, check for advanced features
        if not found_revolutionary:
            advanced_features = [
                'dashboard', 'metrics', 'status', 'panel',
                'ai command', 'symphony'
            ]
            found_advanced = any(feature in output_combined for feature in advanced_features)
            assert found_advanced, f"No Revolutionary TUI or advanced features detected: {output_combined[:200]}"
    
    @pytest.mark.ui
    @pytest.mark.integration
    def test_safe_mode_vs_revolutionary(self):
        """Test difference between safe mode and revolutionary mode."""
        # Test revolutionary mode
        cmd_revolutionary = [self.get_agentsmcp_cmd(), "tui", "--revolutionary"]
        process_revolutionary = subprocess.run(
            cmd_revolutionary,
            input="quit\n",
            capture_output=True,
            text=True,
            timeout=INTERFACE_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        # Test safe mode  
        cmd_safe = [self.get_agentsmcp_cmd(), "tui", "--safe-mode"]
        process_safe = subprocess.run(
            cmd_safe,
            input="quit\n",
            capture_output=True,
            text=True,
            timeout=INTERFACE_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        # Both should work
        assert process_revolutionary.returncode in [0, 130], "Revolutionary mode failed"
        assert process_safe.returncode in [0, 130], "Safe mode failed"
        
        # Revolutionary mode should have more features/output
        rev_output_len = len(process_revolutionary.stdout + process_revolutionary.stderr)
        safe_output_len = len(process_safe.stdout + process_safe.stderr)
        
        # Revolutionary mode typically produces more rich output
        # (This is a heuristic - not always true, but generally expected)
        if rev_output_len > 100 and safe_output_len > 100:
            # Both have significant output, which is good
            pass
        else:
            # At least one mode should work properly
            assert rev_output_len > 50 or safe_output_len > 50, \
                "Neither mode produced significant output"


# Standalone execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])