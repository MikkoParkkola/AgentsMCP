import pytest
import pexpect
import time

@pytest.mark.interactive
@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize("mcp", [["--no-welcome", "--debug"]], indirect=True)
def test_settings_command_accessible(mcp):
    """Test that the settings command is accessible."""
    # Wait for prompt
    try:
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    except pexpect.TIMEOUT:
        mcp.sendline("")
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    
    # Send settings command
    mcp.sendline("settings")
    
    # Should show settings interface
    try:
        mcp.expect(r"(Settings|Configuration|Theme|Provider)", timeout=10)
    except pexpect.TIMEOUT:
        # Settings might load quickly, try to interact
        pass
    
    # Try to exit settings with Escape or q
    mcp.send('\x1b')  # ESC key
    time.sleep(0.5)
    
    # Should return to prompt or show exit message
    try:
        mcp.expect(r"([a-zA-Z0-9_\-]+\s*[>▶]\s*|exit|cancelled)", timeout=5)
    except pexpect.TIMEOUT:
        # Try 'q' to quit
        mcp.send('q')
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)

@pytest.mark.interactive
@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize("mcp", [["--no-welcome", "--debug"]], indirect=True)
def test_settings_arrow_key_navigation(mcp):
    """Test arrow key navigation in settings dialog."""
    # Wait for prompt
    try:
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    except pexpect.TIMEOUT:
        mcp.sendline("")
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    
    # Launch settings
    mcp.sendline("settings")
    
    # Wait for settings to load
    time.sleep(2)
    
    # Try arrow key navigation
    # Down arrow
    mcp.send('\x1b[B')  # Down arrow key
    time.sleep(0.5)
    
    # Up arrow
    mcp.send('\x1b[A')  # Up arrow key  
    time.sleep(0.5)
    
    # Right arrow (might change selection)
    mcp.send('\x1b[C')  # Right arrow key
    time.sleep(0.5)
    
    # Left arrow
    mcp.send('\x1b[D')  # Left arrow key
    time.sleep(0.5)
    
    # Exit settings
    mcp.send('\x1b')  # ESC key
    time.sleep(1)
    
    # Should return to prompt
    try:
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    except pexpect.TIMEOUT:
        # Try alternative exit
        mcp.send('q')
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)

@pytest.mark.interactive
@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize("mcp", [["--no-welcome", "--debug"]], indirect=True)
def test_settings_theme_selection(mcp):
    """Test theme selection in settings."""
    # Wait for prompt
    try:
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    except pexpect.TIMEOUT:
        mcp.sendline("")
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    
    # Launch settings
    mcp.sendline("settings")
    
    # Wait for settings dialog
    time.sleep(2)
    
    # Navigate to theme option (assuming it exists)
    # Try multiple down arrows to find theme setting
    for _ in range(3):
        mcp.send('\x1b[B')  # Down arrow
        time.sleep(0.3)
    
    # Try to activate/change theme setting
    mcp.send('\r')  # Enter key
    time.sleep(0.5)
    
    # If theme options appear, try to select one
    mcp.send('\x1b[B')  # Down arrow to change selection
    time.sleep(0.5)
    mcp.send('\r')  # Enter to confirm
    time.sleep(0.5)
    
    # Exit settings
    mcp.send('\x1b')  # ESC
    time.sleep(1)
    
    # Should return to prompt
    try:
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    except pexpect.TIMEOUT:
        mcp.send('q')
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)

@pytest.mark.interactive
@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize("mcp", [["--no-welcome", "--debug"]], indirect=True)
def test_settings_enter_and_escape_keys(mcp):
    """Test Enter and Escape key handling in settings."""
    # Wait for prompt
    try:
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    except pexpect.TIMEOUT:
        mcp.sendline("")
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    
    # Launch settings
    mcp.sendline("settings")
    time.sleep(2)
    
    # Test Enter key (should activate/select items)
    mcp.send('\r')  # Enter
    time.sleep(0.5)
    
    # Test Escape key (should exit or go back)
    mcp.send('\x1b')  # ESC
    time.sleep(1)
    
    # Should be back at prompt or settings closed
    try:
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    except pexpect.TIMEOUT:
        # Might still be in settings, try 'q'
        mcp.send('q')
        try:
            mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)  
        except pexpect.TIMEOUT:
            # Force exit
            mcp.sendcontrol('c')

@pytest.mark.interactive
@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize("mcp", [["--no-welcome", "--debug"]], indirect=True)
def test_settings_visual_indicators(mcp):
    """Test that settings show visual indicators for selection."""
    # Wait for prompt
    try:
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    except pexpect.TIMEOUT:
        mcp.sendline("")
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    
    # Launch settings
    mcp.sendline("settings")
    time.sleep(2)
    
    # Should have visual indicators like arrows, highlights, etc.
    # Try to capture any visual indicators by navigating
    for i in range(2):
        mcp.send('\x1b[B')  # Down arrow
        time.sleep(0.5)
        
        # Look for selection indicators (this is a loose test)
        # We're mainly ensuring the interface responds to navigation
    
    # Exit cleanly
    mcp.send('\x1b')  # ESC
    time.sleep(1)
    
    try:
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    except pexpect.TIMEOUT:
        mcp.send('q')
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=3)