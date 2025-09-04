"""
Test suite for ReliableTUIInterface.run() method fixes.

This test suite verifies that the critical fix for the run() method lifecycle
prevents immediate shutdown and properly waits for TUI completion.

GOLDEN TESTS (specified in ICD):
1. test_run_method_stays_active_until_user_exit
2. test_finally_block_only_runs_on_actual_completion  
3. test_main_loop_delegates_correctly_to_original_tui

EDGE CASES (additional coverage):
4. test_run_method_keyboard_interrupt_handling
5. test_run_method_fallback_mode_delegation
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import logging

# Import the classes we're testing
from src.agentsmcp.ui.v2.reliability.integration_layer import ReliableTUIInterface
from src.agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface


class TestReliableTUIRunMethod:
    """Test suite for ReliableTUIInterface.run() method lifecycle fixes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_orchestrator = MagicMock()
        self.mock_ui_state = MagicMock()
        
        # Create ReliableTUIInterface instance
        self.reliable_tui = ReliableTUIInterface(
            agent_orchestrator=self.mock_orchestrator,
            agent_state=self.mock_ui_state
        )
        
        # Mock the original TUI
        self.mock_original_tui = AsyncMock(spec=RevolutionaryTUIInterface)
        self.reliable_tui._original_tui = self.mock_original_tui
        self.reliable_tui._startup_completed = True
        
    @pytest.mark.asyncio
    async def test_run_method_stays_active_until_user_exit(self):
        """
        GOLDEN TEST 1: Verify run() method stays active until user exits.
        
        The run() method should not complete immediately after startup,
        but should wait for the actual TUI completion (user exit).
        """
        # Mock the _wait_for_tui_completion method to simulate long-running TUI
        completion_event = asyncio.Event()
        
        async def mock_wait_for_completion():
            # Simulate TUI running for some time before user exits
            await asyncio.sleep(0.1)  # Brief simulation of active TUI
            completion_event.set()
            
        with patch.object(self.reliable_tui, '_wait_for_tui_completion', side_effect=mock_wait_for_completion):
            with patch.object(self.reliable_tui, 'start', return_value=True):
                with patch.object(self.reliable_tui, 'stop', new_callable=AsyncMock) as mock_stop:
                    
                    # Start the run method
                    start_time = asyncio.get_event_loop().time()
                    result = await self.reliable_tui.run()
                    end_time = asyncio.get_event_loop().time()
                    
                    # Verify the run method waited for completion
                    assert completion_event.is_set(), "TUI should have completed"
                    assert end_time - start_time >= 0.1, "Run method should have waited for TUI completion"
                    assert result == 0, "Should return success code"
                    
                    # Verify stop was called in finally block
                    mock_stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_finally_block_only_runs_on_actual_completion(self):
        """
        GOLDEN TEST 2: Verify finally block only runs when TUI actually completes.
        
        The finally block should not run prematurely while TUI is still active.
        """
        stop_called_event = asyncio.Event()
        completion_started_event = asyncio.Event()
        
        async def mock_wait_for_completion():
            completion_started_event.set()
            # Simulate active TUI that runs for a while
            await asyncio.sleep(0.2)
            
        with patch.object(self.reliable_tui, '_wait_for_tui_completion', side_effect=mock_wait_for_completion):
            with patch.object(self.reliable_tui, 'start', return_value=True):
                with patch.object(self.reliable_tui, 'stop', new_callable=AsyncMock) as mock_stop:
                    
                    async def stop_wrapper():
                        stop_called_event.set()
                        
                    mock_stop.side_effect = stop_wrapper
                    
                    # Start the run method
                    task = asyncio.create_task(self.reliable_tui.run())
                    
                    # Wait for completion to start
                    await completion_started_event.wait()
                    
                    # Verify stop hasn't been called yet (TUI still active)
                    await asyncio.sleep(0.1)  # Give time for premature stop to occur
                    assert not stop_called_event.is_set(), "Stop should not be called while TUI is active"
                    
                    # Wait for actual completion
                    result = await task
                    
                    # Now stop should have been called
                    assert stop_called_event.is_set(), "Stop should be called after TUI completion"
                    assert result == 0
    
    @pytest.mark.asyncio 
    async def test_main_loop_delegates_correctly_to_original_tui(self):
        """
        GOLDEN TEST 3: Verify main loop delegates correctly to original TUI.
        
        The run method should properly delegate to the original TUI's _run_main_loop
        method and wait for its completion.
        """
        original_run_main_loop_called = False
        
        async def mock_run_main_loop(*args, **kwargs):
            nonlocal original_run_main_loop_called
            original_run_main_loop_called = True
            # Simulate the original main loop running
            await asyncio.sleep(0.1)
            
        # Mock the original TUI's _run_main_loop method
        self.mock_original_tui._run_main_loop = mock_run_main_loop
        
        with patch.object(self.reliable_tui, 'start', return_value=True):
            with patch.object(self.reliable_tui, 'stop', new_callable=AsyncMock):
                
                result = await self.reliable_tui.run()
                
                # Verify delegation occurred
                assert original_run_main_loop_called, "Original TUI's _run_main_loop should be called"
                assert result == 0, "Should return success code"
    
    @pytest.mark.asyncio
    async def test_run_method_keyboard_interrupt_handling(self):
        """
        EDGE CASE 1: Verify proper KeyboardInterrupt handling.
        
        When user interrupts with Ctrl+C, the run method should handle it gracefully
        and return exit code 0.
        """
        async def mock_wait_for_completion():
            # Simulate user pressing Ctrl+C
            raise KeyboardInterrupt("User interrupted")
            
        with patch.object(self.reliable_tui, '_wait_for_tui_completion', side_effect=mock_wait_for_completion):
            with patch.object(self.reliable_tui, 'start', return_value=True):
                with patch.object(self.reliable_tui, 'stop', new_callable=AsyncMock) as mock_stop:
                    
                    result = await self.reliable_tui.run()
                    
                    # Should handle KeyboardInterrupt gracefully
                    assert result == 0, "KeyboardInterrupt should result in success exit code"
                    mock_stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_method_fallback_mode_delegation(self):
        """
        EDGE CASE 2: Verify proper fallback mode delegation.
        
        When running in fallback mode, the run method should delegate to
        the original TUI's run() method and wait for its completion.
        """
        # Set up fallback mode
        self.reliable_tui._fallback_mode = True
        
        original_run_called = False
        
        async def mock_original_run():
            nonlocal original_run_called
            original_run_called = True
            # Simulate original TUI run method
            await asyncio.sleep(0.1)
            return 0
            
        self.mock_original_tui.run = mock_original_run
        
        with patch.object(self.reliable_tui, 'start', return_value=True):
            with patch.object(self.reliable_tui, 'stop', new_callable=AsyncMock) as mock_stop:
                
                result = await self.reliable_tui.run()
                
                # Verify fallback delegation
                assert original_run_called, "Original TUI's run() method should be called in fallback mode"
                assert result == 0, "Should return success code"
                mock_stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_method_startup_failure_handling(self):
        """
        EDGE CASE 3: Verify proper handling of startup failures.
        
        If startup fails, run method should return error code 1 immediately
        without waiting for completion.
        """
        with patch.object(self.reliable_tui, 'start', return_value=False):
            with patch.object(self.reliable_tui, 'stop', new_callable=AsyncMock) as mock_stop:
                
                result = await self.reliable_tui.run()
                
                # Should return error code for startup failure
                assert result == 1, "Startup failure should result in error exit code"
                mock_stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_method_exception_handling_with_recovery(self):
        """
        EDGE CASE 4: Verify exception handling with recovery attempt.
        
        If main loop fails but recovery is enabled, should attempt recovery
        and retry the main loop.
        """
        # Set up recovery manager
        mock_recovery_manager = AsyncMock()
        mock_recovery_result = MagicMock()
        mock_recovery_result.success = True
        mock_recovery_manager.manual_recovery.return_value = mock_recovery_result
        
        self.reliable_tui._recovery_manager = mock_recovery_manager
        self.reliable_tui._config = MagicMock()
        self.reliable_tui._config.enable_automatic_recovery = True
        
        call_count = 0
        
        async def mock_wait_for_completion():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call fails
                raise Exception("Main loop failed")
            else:
                # Second call (after recovery) succeeds
                await asyncio.sleep(0.1)
                
        with patch.object(self.reliable_tui, '_wait_for_tui_completion', side_effect=mock_wait_for_completion):
            with patch.object(self.reliable_tui, 'start', return_value=True):
                with patch.object(self.reliable_tui, 'stop', new_callable=AsyncMock):
                    
                    result = await self.reliable_tui.run()
                    
                    # Should succeed after recovery
                    assert result == 0, "Should succeed after recovery"
                    assert call_count == 2, "Should retry after recovery"
                    mock_recovery_manager.manual_recovery.assert_called_once()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])