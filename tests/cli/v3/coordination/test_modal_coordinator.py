"""Tests for modal coordinator functionality."""

import asyncio
import pytest
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from src.agentsmcp.cli.v3.coordination.modal_coordinator import ModalCoordinator
from src.agentsmcp.cli.v3.models.coordination_models import (
    InterfaceMode,
    ModalSwitchRequest,
    StateSync,
    CapabilityQuery,
    ConflictResolution,
    TransitionResult,
    SharedState,
    SessionContext,
    ModeNotSupportedError,
    StateLossError
)


@pytest.fixture
async def coordinator():
    """Create a modal coordinator for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        coord = ModalCoordinator(storage_path=tmpdir)
        await coord.initialize()
        yield coord
        await coord.shutdown()


@pytest.fixture
def sample_session(coordinator):
    """Create a sample session for testing."""
    return {
        'session_id': 'test-session-123',
        'user_id': 'test-user',
        'interface_mode': InterfaceMode.CLI
    }


class TestModalCoordinator:
    """Test modal coordinator functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, coordinator):
        """Test coordinator initialization."""
        # Check that coordinator is initialized
        assert coordinator.state_synchronizer is not None
        assert coordinator.capability_manager is not None
        
        # Check supported interfaces
        supported_interfaces = coordinator.get_supported_interfaces()
        assert InterfaceMode.CLI in supported_interfaces
        # TUI, WebUI, API availability depends on environment
    
    @pytest.mark.asyncio
    async def test_create_session(self, coordinator, sample_session):
        """Test session creation."""
        state = await coordinator.create_session(
            user_id=sample_session['user_id'],
            initial_interface=sample_session['interface_mode'],
            session_id=sample_session['session_id']
        )
        
        assert state.session_id == sample_session['session_id']
        assert state.active_context.user_id == sample_session['user_id']
        assert state.active_context.current_interface == sample_session['interface_mode']
        assert coordinator.get_current_interface(sample_session['session_id']) == sample_session['interface_mode']
    
    @pytest.mark.asyncio
    async def test_mode_switch_cli_to_tui(self, coordinator, sample_session):
        """Test switching from CLI to TUI mode."""
        # Create initial session
        await coordinator.create_session(
            user_id=sample_session['user_id'],
            initial_interface=InterfaceMode.CLI,
            session_id=sample_session['session_id']
        )
        
        # Check if TUI is available
        supported_interfaces = coordinator.get_supported_interfaces()
        if InterfaceMode.TUI not in supported_interfaces:
            pytest.skip("TUI interface not available in test environment")
        
        # Create switch request
        switch_request = ModalSwitchRequest(
            session_id=sample_session['session_id'],
            from_mode=InterfaceMode.CLI,
            to_mode=InterfaceMode.TUI,
            requested_by=sample_session['user_id']
        )
        
        # Perform switch
        result = await coordinator.switch_mode(switch_request)
        
        assert result.success
        assert result.from_mode == InterfaceMode.CLI
        assert result.to_mode == InterfaceMode.TUI
        assert result.duration_ms > 0
        assert coordinator.get_current_interface(sample_session['session_id']) == InterfaceMode.TUI
    
    @pytest.mark.asyncio
    async def test_mode_switch_with_context_preservation(self, coordinator, sample_session):
        """Test mode switching with context preservation."""
        # Create session
        state = await coordinator.create_session(
            user_id=sample_session['user_id'],
            initial_interface=InterfaceMode.CLI,
            session_id=sample_session['session_id']
        )
        
        # Add some context to preserve
        state.user_preferences['theme'] = 'dark'
        state.interface_states[InterfaceMode.CLI] = {
            'command_history': ['ls', 'pwd', 'cd /tmp'],
            'current_directory': '/home/user'
        }
        
        # Mock TUI availability
        coordinator._transition_handlers[InterfaceMode.TUI] = Mock()
        
        # Create switch request
        switch_request = ModalSwitchRequest(
            session_id=sample_session['session_id'],
            from_mode=InterfaceMode.CLI,
            to_mode=InterfaceMode.TUI,
            preserve_state=True,
            transfer_context=True,
            requested_by=sample_session['user_id']
        )
        
        # Perform switch
        result = await coordinator.switch_mode(switch_request)
        
        assert result.success
        assert 'command_history' in result.preserved_context or len(result.lost_context) > 0
    
    @pytest.mark.asyncio
    async def test_mode_switch_to_unsupported_interface(self, coordinator, sample_session):
        """Test switching to unsupported interface mode."""
        # Create session
        await coordinator.create_session(
            user_id=sample_session['user_id'],
            initial_interface=InterfaceMode.CLI,
            session_id=sample_session['session_id']
        )
        
        # Remove WebUI handler to simulate unavailability
        coordinator._transition_handlers.pop(InterfaceMode.WEB_UI, None)
        
        # Create switch request
        switch_request = ModalSwitchRequest(
            session_id=sample_session['session_id'],
            from_mode=InterfaceMode.CLI,
            to_mode=InterfaceMode.WEB_UI,
            requested_by=sample_session['user_id']
        )
        
        # Perform switch - should fail
        result = await coordinator.switch_mode(switch_request)
        
        assert not result.success
        assert len(result.errors) > 0
        assert any("not available" in error for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_session_recovery(self, coordinator, sample_session):
        """Test session recovery from storage."""
        # Create and cleanup session
        original_state = await coordinator.create_session(
            user_id=sample_session['user_id'],
            initial_interface=sample_session['interface_mode'],
            session_id=sample_session['session_id']
        )
        
        original_state.user_preferences['test_key'] = 'test_value'
        await coordinator.cleanup_session(sample_session['session_id'], preserve_state=True)
        
        # Recover session
        recovered_state = await coordinator.recover_session(
            sample_session['session_id'],
            sample_session['user_id']
        )
        
        assert recovered_state is not None
        assert recovered_state.session_id == sample_session['session_id']
        assert recovered_state.user_preferences.get('test_key') == 'test_value'
    
    @pytest.mark.asyncio
    async def test_capability_query(self, coordinator):
        """Test querying interface capabilities."""
        # Query CLI capabilities
        query = CapabilityQuery(
            interface=InterfaceMode.CLI,
            check_permissions=False
        )
        
        capability_info = await coordinator.query_mode_capabilities(query)
        
        assert capability_info.interface == InterfaceMode.CLI
        assert len(capability_info.available_features) > 0
        assert capability_info.performance_profile is not None
    
    @pytest.mark.asyncio
    async def test_get_all_mode_capabilities(self, coordinator):
        """Test getting capabilities for all modes."""
        capabilities = await coordinator.get_all_mode_capabilities()
        
        assert InterfaceMode.CLI in capabilities
        assert len(capabilities[InterfaceMode.CLI]) > 0
        
        # Check that each feature has required fields
        for feature in capabilities[InterfaceMode.CLI]:
            assert feature.name
            assert feature.capability_type
    
    @pytest.mark.asyncio
    async def test_state_synchronization(self, coordinator, sample_session):
        """Test state synchronization across interfaces."""
        # Create session
        state = await coordinator.create_session(
            user_id=sample_session['user_id'],
            initial_interface=sample_session['interface_mode'],
            session_id=sample_session['session_id']
        )
        
        # Create sync request
        sync_request = StateSync(
            session_id=sample_session['session_id'],
            interface_mode=InterfaceMode.CLI,
            current_version=state.version,
            conflict_resolution=ConflictResolution.LAST_WRITE_WINS
        )
        
        # Sync state
        result = await coordinator.sync_state(sync_request)
        
        assert result.success
        assert result.new_version >= state.version
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, coordinator, sample_session):
        """Test graceful degradation of features."""
        # Create session
        await coordinator.create_session(
            user_id=sample_session['user_id'],
            initial_interface=sample_session['interface_mode'],
            session_id=sample_session['session_id']
        )
        
        # Test graceful degradation
        features = await coordinator.graceful_degradation(
            sample_session['session_id'],
            InterfaceMode.CLI,
            ['text_output', 'command_line_args', 'nonexistent_feature']
        )
        
        assert isinstance(features, list)
        assert 'text_output' in features
        assert 'command_line_args' in features
        # nonexistent_feature should be handled gracefully
    
    @pytest.mark.asyncio
    async def test_transition_metrics(self, coordinator, sample_session):
        """Test transition metrics collection."""
        # Create session
        await coordinator.create_session(
            user_id=sample_session['user_id'],
            initial_interface=InterfaceMode.CLI,
            session_id=sample_session['session_id']
        )
        
        # Mock TUI availability for testing
        coordinator._transition_handlers[InterfaceMode.TUI] = Mock()
        
        # Perform multiple transitions to generate metrics
        for i in range(3):
            switch_request = ModalSwitchRequest(
                session_id=sample_session['session_id'],
                from_mode=InterfaceMode.CLI if i % 2 == 0 else InterfaceMode.TUI,
                to_mode=InterfaceMode.TUI if i % 2 == 0 else InterfaceMode.CLI,
                requested_by=sample_session['user_id']
            )
            await coordinator.switch_mode(switch_request)
        
        # Check metrics
        metrics = coordinator.get_transition_metrics()
        
        assert 'transition_duration_ms' in metrics
        assert metrics['transition_duration_ms']['count'] > 0
        assert metrics['transition_duration_ms']['average'] > 0
    
    @pytest.mark.asyncio
    async def test_cleanup_session(self, coordinator, sample_session):
        """Test session cleanup."""
        # Create session
        await coordinator.create_session(
            user_id=sample_session['user_id'],
            initial_interface=sample_session['interface_mode'],
            session_id=sample_session['session_id']
        )
        
        # Verify session exists
        assert coordinator.get_current_interface(sample_session['session_id']) is not None
        
        # Cleanup session
        await coordinator.cleanup_session(sample_session['session_id'], preserve_state=False)
        
        # Verify session is cleaned up
        assert coordinator.get_current_interface(sample_session['session_id']) is None
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, coordinator, sample_session):
        """Test that performance requirements are met."""
        # Create session
        await coordinator.create_session(
            user_id=sample_session['user_id'],
            initial_interface=sample_session['interface_mode'],
            session_id=sample_session['session_id']
        )
        
        # Mock TUI handler
        coordinator._transition_handlers[InterfaceMode.TUI] = Mock()
        
        # Test transition performance
        start_time = datetime.now(timezone.utc)
        
        switch_request = ModalSwitchRequest(
            session_id=sample_session['session_id'],
            from_mode=InterfaceMode.CLI,
            to_mode=InterfaceMode.TUI,
            requested_by=sample_session['user_id']
        )
        
        result = await coordinator.switch_mode(switch_request)
        
        end_time = datetime.now(timezone.utc)
        actual_duration = (end_time - start_time).total_seconds() * 1000
        
        # Performance requirement: transitions < 300ms
        assert actual_duration < 300, f"Transition took {actual_duration}ms, expected < 300ms"
        assert result.duration_ms < 300, f"Reported duration {result.duration_ms}ms, expected < 300ms"


class TestErrorHandling:
    """Test error handling in modal coordinator."""
    
    @pytest.mark.asyncio
    async def test_invalid_session_id(self, coordinator):
        """Test handling of invalid session ID."""
        switch_request = ModalSwitchRequest(
            session_id='nonexistent-session',
            from_mode=InterfaceMode.CLI,
            to_mode=InterfaceMode.TUI,
            requested_by='test-user'
        )
        
        result = await coordinator.switch_mode(switch_request)
        
        assert not result.success
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_same_mode_transition(self):
        """Test validation prevents switching to same mode."""
        with pytest.raises(ValueError, match="Target mode must be different"):
            ModalSwitchRequest(
                session_id='test-session',
                from_mode=InterfaceMode.CLI,
                to_mode=InterfaceMode.CLI,  # Same as from_mode
                requested_by='test-user'
            )
    
    @pytest.mark.asyncio
    async def test_unauthorized_access(self, coordinator):
        """Test unauthorized session access."""
        # Create session for user1
        state = await coordinator.create_session(
            user_id='user1',
            initial_interface=InterfaceMode.CLI,
            session_id='test-session'
        )
        
        await coordinator.cleanup_session('test-session', preserve_state=True)
        
        # Try to recover as different user
        with pytest.raises(PermissionError, match="Session access denied"):
            await coordinator.recover_session('test-session', 'user2')


class TestConcurrency:
    """Test concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_transitions(self, coordinator):
        """Test handling of concurrent transition requests."""
        # Create session
        state = await coordinator.create_session(
            user_id='test-user',
            initial_interface=InterfaceMode.CLI,
            session_id='test-session'
        )
        
        # Mock handlers
        coordinator._transition_handlers[InterfaceMode.TUI] = Mock()
        coordinator._transition_handlers[InterfaceMode.API] = Mock()
        
        # Create concurrent transition requests
        request1 = ModalSwitchRequest(
            session_id='test-session',
            from_mode=InterfaceMode.CLI,
            to_mode=InterfaceMode.TUI,
            requested_by='test-user'
        )
        
        request2 = ModalSwitchRequest(
            session_id='test-session',
            from_mode=InterfaceMode.CLI,
            to_mode=InterfaceMode.API,
            requested_by='test-user'
        )
        
        # Execute concurrently
        results = await asyncio.gather(
            coordinator.switch_mode(request1),
            coordinator.switch_mode(request2),
            return_exceptions=True
        )
        
        # At least one should succeed
        successful_results = [r for r in results if isinstance(r, TransitionResult) and r.success]
        assert len(successful_results) >= 1
    
    @pytest.mark.asyncio
    async def test_concurrent_state_sync(self, coordinator):
        """Test concurrent state synchronization."""
        # Create session
        state = await coordinator.create_session(
            user_id='test-user',
            initial_interface=InterfaceMode.CLI,
            session_id='test-session'
        )
        
        # Create multiple sync requests
        sync_requests = []
        for i in range(3):
            sync_request = StateSync(
                session_id='test-session',
                interface_mode=InterfaceMode.CLI,
                current_version=state.version,
                conflict_resolution=ConflictResolution.LAST_WRITE_WINS
            )
            sync_requests.append(sync_request)
        
        # Execute concurrently
        results = await asyncio.gather(
            *[coordinator.sync_state(req) for req in sync_requests],
            return_exceptions=True
        )
        
        # All should complete without exceptions
        for result in results:
            assert not isinstance(result, Exception), f"Unexpected exception: {result}"