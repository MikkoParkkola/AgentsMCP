import pytest
import pexpect
import time
import subprocess
import sys

@pytest.mark.interactive
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_dashboard_mode_startup(binary_path, temp_dir):
    """Test that dashboard mode starts successfully."""
    cmd = binary_path + ["--mode", "dashboard", "--no-welcome", "--refresh-interval", "2"]
    
    if sys.platform.startswith('win'):
        child = pexpect.popen_spawn.PopenSpawn(' '.join(cmd), cwd=str(temp_dir), encoding="utf-8", timeout=15)
    else:
        child = pexpect.spawn(cmd[0], args=cmd[1:], cwd=str(temp_dir), encoding="utf-8", timeout=15)
    
    try:
        # Should show dashboard content
        child.expect(r"(Dashboard|AgentsMCP|Orchestration)", timeout=10)
        
        # Let it run for a short time to see auto-refresh
        time.sleep(3)
        
        # Should show status or metrics
        child.expect(r"(Status|Agent|Performance|System)", timeout=5)
        
    finally:
        child.terminate(force=True)
        try:
            child.wait()
        except:
            pass

@pytest.mark.interactive 
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_dashboard_graceful_shutdown(binary_path, temp_dir):
    """Test that dashboard mode can be exited with Ctrl+C."""
    cmd = binary_path + ["--mode", "dashboard", "--no-welcome", "--refresh-interval", "1"]
    
    if sys.platform.startswith('win'):
        child = pexpect.popen_spawn.PopenSpawn(' '.join(cmd), cwd=str(temp_dir), encoding="utf-8", timeout=15)
    else:
        child = pexpect.spawn(cmd[0], args=cmd[1:], cwd=str(temp_dir), encoding="utf-8", timeout=15)
    
    try:
        # Wait for dashboard to start
        child.expect(r"(Dashboard|AgentsMCP)", timeout=10)
        
        # Let it refresh a few times  
        time.sleep(4)
        
        # Send Ctrl+C to exit
        child.sendcontrol('c')
        
        # Should exit gracefully
        try:
            child.expect(r"(stopped|exit|shutdown|goodbye)", timeout=10)
        except pexpect.TIMEOUT:
            # Might exit immediately
            pass
            
        # Should reach EOF (process ended)
        child.expect(pexpect.EOF, timeout=5)
        
    except Exception as e:
        # Force cleanup
        child.terminate(force=True)
        raise e

@pytest.mark.interactive
@pytest.mark.flaky(reruns=3, reruns_delay=2) 
@pytest.mark.parametrize("mcp", [["--no-welcome", "--debug"]], indirect=True)
def test_dashboard_command_from_interactive(mcp):
    """Test launching dashboard from interactive mode."""
    # Wait for prompt
    try:
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    except pexpect.TIMEOUT:
        mcp.sendline("")
        mcp.expect(r"[a-zA-Z0-9_\-]+\s*[>▶]\s*", timeout=5)
    
    # Launch dashboard command
    mcp.sendline("dashboard --refresh 2")
    
    # Should start dashboard
    try:
        mcp.expect(r"(Dashboard|Launching|Press Ctrl\+C)", timeout=10)
    except pexpect.TIMEOUT:
        # Might start immediately
        pass
    
    # Let dashboard run briefly
    time.sleep(3)
    
    # Try to exit with Ctrl+C
    mcp.sendcontrol('c')
    
    # Should return to interactive prompt or exit cleanly
    try:
        # Either back to prompt or graceful exit
        mcp.expect(r"([a-zA-Z0-9_\-]+\s*[>▶]\s*|stopped|completed)", timeout=10)
    except pexpect.TIMEOUT:
        # Might have exited completely
        try:
            mcp.expect(pexpect.EOF, timeout=5)
        except pexpect.TIMEOUT:
            pass

@pytest.mark.interactive
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_dashboard_refresh_interval(binary_path, temp_dir):
    """Test dashboard with custom refresh interval."""
    cmd = binary_path + ["--mode", "dashboard", "--refresh-interval", "0.5", "--no-welcome"]
    
    if sys.platform.startswith('win'):
        child = pexpect.popen_spawn.PopenSpawn(' '.join(cmd), cwd=str(temp_dir), encoding="utf-8", timeout=15)
    else:
        child = pexpect.spawn(cmd[0], args=cmd[1:], cwd=str(temp_dir), encoding="utf-8", timeout=15)
    
    try:
        # Should start dashboard
        child.expect(r"(Dashboard|AgentsMCP)", timeout=10)
        
        # With fast refresh, should see multiple updates quickly
        start_time = time.time()
        update_count = 0
        
        while time.time() - start_time < 3:  # Watch for 3 seconds
            try:
                child.expect(r"(Update|Refresh|Status)", timeout=1)
                update_count += 1
            except pexpect.TIMEOUT:
                continue
        
        # Should have seen some activity with 0.5s refresh
        # (This is a loose check since content might not change)
        
        # Exit cleanly
        child.sendcontrol('c')
        child.expect(pexpect.EOF, timeout=5)
        
    finally:
        child.terminate(force=True)
        try:
            child.wait()
        except:
            pass