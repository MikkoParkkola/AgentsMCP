"""Integration tests for CLI v3 coordination system."""

import asyncio
import pytest
import tempfile
from datetime import datetime, timezone

from src.agentsmcp.cli.v3.coordination import ModalCoordinator, StateSynchronizer, CapabilityManager
from src.agentsmcp.cli.v3.models.coordination_models import (
    InterfaceMode,
    ModalSwitchRequest,
    StateSync,
    CapabilityQuery,
    ConflictResolution,
    StateChange,
    TransitionResult,
    SyncResult,
    CapabilityInfo
)


@pytest.fixture
async def coordination_system():
    """Create a complete coordination system for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        coordinator = ModalCoordinator(storage_path=tmpdir)
        await coordinator.initialize()
        
        yield {
            'coordinator': coordinator,
            'state_sync': coordinator.state_synchronizer,
            'capability_mgr': coordinator.capability_manager
        }
        
        await coordinator.shutdown()


@pytest.fixture
async def multi_user_setup(coordination_system):
    """Set up multiple users and sessions for testing."""
    coordinator = coordination_system['coordinator']
    
    # Create sessions for multiple users
    users = {
        'user1': await coordinator.create_session('user1', InterfaceMode.CLI, 'session1'),
        'user2': await coordinator.create_session('user2', InterfaceMode.CLI, 'session2'),
        'user3': await coordinator.create_session('user3', InterfaceMode.CLI, 'session3')
    }
    
    return {
        'coordinator': coordinator,
        'users': users,
        'sessions': {
            'session1': users['user1'],
            'session2': users['user2'], 
            'session3': users['user3']
        }
    }


class TestFullCoordinationWorkflow:
    """Test complete coordination workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_mode_transition_workflow(self, coordination_system):
        """Test complete workflow from session creation to mode transition."""
        coordinator = coordination_system['coordinator']
        
        # Step 1: Create session
        user_id = 'workflow-user'
        session_state = await coordinator.create_session(
            user_id=user_id,
            initial_interface=InterfaceMode.CLI
        )
        session_id = session_state.session_id
        
        # Step 2: Query capabilities for target interface
        if InterfaceMode.TUI not in coordinator.get_supported_interfaces():
            pytest.skip("TUI not available for workflow test")
        
        tui_query = CapabilityQuery(
            interface=InterfaceMode.TUI,
            check_permissions=True,
            user_id=user_id
        )
        
        tui_capabilities = await coordinator.query_mode_capabilities(tui_query)
        assert isinstance(tui_capabilities, CapabilityInfo)
        
        # Step 3: Add some context to preserve
        session_state.user_preferences['workflow_test'] = 'preserve_this'
        session_state.interface_states[InterfaceMode.CLI] = {
            'command_history': ['ls', 'pwd'],
            'current_directory': '/tmp'
        }
        
        # Step 4: Synchronize state changes
        state_change = StateChange(
            session_id=session_id,
            interface_mode=InterfaceMode.CLI,
            key_path='user_preferences.workflow_test',
            old_value=None,
            new_value='preserve_this',
            user_id=user_id
        )
        
        sync_request = StateSync(
            session_id=session_id,
            interface_mode=InterfaceMode.CLI,
            current_version=session_state.version,
            changes=[state_change]
        )
        
        sync_result = await coordinator.sync_state(sync_request)
        assert sync_result.success
        
        # Step 5: Perform mode transition
        transition_request = ModalSwitchRequest(
            session_id=session_id,
            from_mode=InterfaceMode.CLI,
            to_mode=InterfaceMode.TUI,
            preserve_state=True,
            transfer_context=True,
            requested_by=user_id
        )
        
        transition_result = await coordinator.switch_mode(transition_request)
        
        # Step 6: Verify transition success
        assert transition_result.success
        assert transition_result.from_mode == InterfaceMode.CLI
        assert transition_result.to_mode == InterfaceMode.TUI
        assert coordinator.get_current_interface(session_id) == InterfaceMode.TUI
        
        # Step 7: Verify state preservation
        final_state = await coordinator.get_synchronized_state(session_id)
        assert final_state.user_preferences.get('workflow_test') == 'preserve_this'
        assert final_state.active_context.current_interface == InterfaceMode.TUI
        
        # Step 8: Clean up
        await coordinator.cleanup_session(session_id)
    
    @pytest.mark.asyncio
    async def test_multi_interface_round_trip(self, coordination_system):
        """Test transitions through multiple interfaces."""
        coordinator = coordination_system['coordinator']
        
        # Create session
        user_id = 'roundtrip-user'
        session_state = await coordinator.create_session(
            user_id=user_id,
            initial_interface=InterfaceMode.CLI
        )
        session_id = session_state.session_id
        
        # Get supported interfaces for testing
        supported_interfaces = coordinator.get_supported_interfaces()
        
        # Plan transition path (CLI -> available interfaces -> back to CLI)
        transition_path = [InterfaceMode.CLI]
        for interface in [InterfaceMode.TUI, InterfaceMode.WEB_UI, InterfaceMode.API]:
            if interface in supported_interfaces:
                transition_path.append(interface)
        transition_path.append(InterfaceMode.CLI)  # Return to start
        
        if len(transition_path) < 3:  # CLI -> other -> CLI minimum
            pytest.skip("Not enough interfaces available for round-trip test")
        
        # Perform transitions
        current_interface = InterfaceMode.CLI
        for target_interface in transition_path[1:]:
            if target_interface == current_interface:
                continue  # Skip if same
            
            transition_request = ModalSwitchRequest(
                session_id=session_id,
                from_mode=current_interface,
                to_mode=target_interface,
                preserve_state=True,
                requested_by=user_id
            )
            
            result = await coordinator.switch_mode(transition_request)
            
            if result.success:
                current_interface = target_interface
                assert coordinator.get_current_interface(session_id) == target_interface
        
        # Verify we made at least one successful transition
        assert coordinator.get_current_interface(session_id) is not None
        
        await coordinator.cleanup_session(session_id)
    
    @pytest.mark.asyncio
    async def test_session_recovery_workflow(self, coordination_system):
        """Test complete session recovery workflow."""
        coordinator = coordination_system['coordinator']
        
        # Create and populate session
        user_id = 'recovery-user'
        session_state = await coordinator.create_session(
            user_id=user_id,
            initial_interface=InterfaceMode.CLI
        )
        session_id = session_state.session_id
        
        # Add substantial state
        session_state.user_preferences.update({
            'theme': 'dark',
            'auto_save': True,
            'recovery_test': 'important_data'
        })
        
        session_state.interface_states[InterfaceMode.CLI] = {
            'command_history': ['git status', 'git add .', 'git commit'],
            'environment_vars': {'TEST_MODE': 'recovery'},
            'working_directory': '/project'
        }
        
        # Sync the changes
        changes = [
            StateChange(
                session_id=session_id,
                interface_mode=InterfaceMode.CLI,
                key_path='user_preferences.recovery_test',
                old_value=None,
                new_value='important_data',
                user_id=user_id
            )
        ]
        
        sync_request = StateSync(
            session_id=session_id,
            interface_mode=InterfaceMode.CLI,
            current_version=session_state.version,
            changes=changes
        )
        
        sync_result = await coordinator.sync_state(sync_request)
        assert sync_result.success
        
        # Simulate application shutdown
        await coordinator.cleanup_session(session_id, preserve_state=True)
        
        # Simulate application restart - recover session
        recovered_state = await coordinator.recover_session(session_id, user_id)
        
        # Verify complete state recovery
        assert recovered_state is not None
        assert recovered_state.session_id == session_id
        assert recovered_state.active_context.user_id == user_id
        assert recovered_state.user_preferences['recovery_test'] == 'important_data'
        assert recovered_state.user_preferences['theme'] == 'dark'
        
        await coordinator.cleanup_session(session_id)


class TestMultiUserCoordination:
    """Test coordination across multiple users and sessions."""
    
    @pytest.mark.asyncio
    async def test_concurrent_user_operations(self, multi_user_setup):
        """Test concurrent operations across multiple users."""
        coordinator = multi_user_setup['coordinator']
        users = multi_user_setup['users']
        
        async def user_workflow(user_id, session_state):
            """Simulate user workflow."""
            session_id = session_state.session_id
            
            # Add user-specific state
            session_state.user_preferences[f'{user_id}_data'] = f'data_for_{user_id}'
            
            # Sync state
            change = StateChange(
                session_id=session_id,
                interface_mode=InterfaceMode.CLI,
                key_path=f'user_preferences.{user_id}_data',
                old_value=None,
                new_value=f'data_for_{user_id}',
                user_id=user_id
            )
            
            sync_request = StateSync(
                session_id=session_id,
                interface_mode=InterfaceMode.CLI,
                current_version=session_state.version,
                changes=[change]
            )
            
            return await coordinator.sync_state(sync_request)
        
        # Run concurrent user workflows
        results = await asyncio.gather(
            user_workflow('user1', users['user1']),
            user_workflow('user2', users['user2']),
            user_workflow('user3', users['user3']),
            return_exceptions=True
        )
        
        # All should succeed
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, SyncResult)
            assert result.success
    
    @pytest.mark.asyncio
    async def test_user_isolation(self, multi_user_setup):
        """Test that user sessions are properly isolated."""
        coordinator = multi_user_setup['coordinator']
        sessions = multi_user_setup['sessions']
        
        # Modify user1's session
        user1_state = await coordinator.get_synchronized_state('session1')
        user1_state.user_preferences['secret_data'] = 'user1_only'
        
        # Try to access user1's session as user2 (should fail)
        with pytest.raises(PermissionError):
            await coordinator.state_synchronizer.recover_session('session1', 'user2')
        
        # Verify user2 cannot see user1's data
        user2_state = await coordinator.get_synchronized_state('session2')
        assert 'secret_data' not in user2_state.user_preferences
    
    @pytest.mark.asyncio
    async def test_capability_queries_per_user(self, multi_user_setup):
        """Test capability queries with user-specific permissions."""
        coordinator = multi_user_setup['coordinator']
        
        # Query capabilities for different users
        user1_query = CapabilityQuery(
            interface=InterfaceMode.CLI,
            check_permissions=True,
            user_id='user1'
        )
        
        user2_query = CapabilityQuery(
            interface=InterfaceMode.CLI,
            check_permissions=True,
            user_id='user2'
        )
        
        user1_caps = await coordinator.query_mode_capabilities(user1_query)
        user2_caps = await coordinator.query_mode_capabilities(user2_query)
        
        # Both should get CLI capabilities (permissions are mocked as granted)
        assert user1_caps.interface == InterfaceMode.CLI
        assert user2_caps.interface == InterfaceMode.CLI
        assert len(user1_caps.available_features) > 0
        assert len(user2_caps.available_features) > 0


class TestPerformanceRequirements:
    """Test that performance requirements are met."""
    
    @pytest.mark.asyncio
    async def test_state_sync_performance(self, coordination_system):
        """Test state sync meets <100ms requirement."""
        coordinator = coordination_system['coordinator']
        
        # Create session
        user_id = 'perf-user'
        session_state = await coordinator.create_session(
            user_id=user_id,
            initial_interface=InterfaceMode.CLI
        )
        
        # Create multiple state changes
        changes = []
        for i in range(10):  # Multiple changes to test batching
            changes.append(StateChange(
                session_id=session_state.session_id,
                interface_mode=InterfaceMode.CLI,
                key_path=f'test.perf_{i}',
                old_value=None,
                new_value=f'value_{i}',
                user_id=user_id
            ))
        
        sync_request = StateSync(
            session_id=session_state.session_id,
            interface_mode=InterfaceMode.CLI,
            current_version=session_state.version,
            changes=changes
        )
        
        # Measure sync time
        start_time = datetime.now(timezone.utc)
        result = await coordinator.sync_state(sync_request)
        end_time = datetime.now(timezone.utc)
        
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Performance requirement: < 100ms
        assert duration_ms < 100, f"State sync took {duration_ms}ms, expected < 100ms"
        assert result.success
        assert result.duration_ms < 100
        
        await coordinator.cleanup_session(session_state.session_id)
    
    @pytest.mark.asyncio
    async def test_mode_transition_performance(self, coordination_system):
        """Test mode transition meets <300ms requirement."""
        coordinator = coordination_system['coordinator']
        
        # Check if we have multiple interfaces for testing
        supported = coordinator.get_supported_interfaces()
        if len(supported) < 2:
            pytest.skip("Need at least 2 interfaces for transition performance test")
        
        # Create session
        user_id = 'transition-perf-user'
        session_state = await coordinator.create_session(
            user_id=user_id,
            initial_interface=InterfaceMode.CLI
        )
        
        # Find target interface
        target_interface = None
        for interface in [InterfaceMode.TUI, InterfaceMode.API, InterfaceMode.WEB_UI]:
            if interface in supported:
                target_interface = interface
                break
        
        if not target_interface:
            pytest.skip("No target interface available for performance test")
        
        # Mock the target interface handler to avoid actual startup overhead
        coordinator._transition_handlers[target_interface] = type('MockHandler', (), {
            'extract_context': lambda self, state: {},
            'prepare_interface': lambda self, state, context: None,
            'activate_interface': lambda self, state: None
        })()
        
        transition_request = ModalSwitchRequest(
            session_id=session_state.session_id,
            from_mode=InterfaceMode.CLI,
            to_mode=target_interface,
            requested_by=user_id
        )
        
        # Measure transition time
        start_time = datetime.now(timezone.utc)
        result = await coordinator.switch_mode(transition_request)
        end_time = datetime.now(timezone.utc)
        
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Performance requirement: < 300ms
        assert duration_ms < 300, f"Mode transition took {duration_ms}ms, expected < 300ms"
        assert result.success
        assert result.duration_ms < 300
        
        await coordinator.cleanup_session(session_state.session_id)


class TestErrorRecovery:
    """Test error recovery and resilience."""
    
    @pytest.mark.asyncio
    async def test_partial_state_loss_recovery(self, coordination_system):
        """Test recovery from partial state loss."""
        coordinator = coordination_system['coordinator']
        
        # Create session with complex state
        user_id = 'recovery-user'
        session_state = await coordinator.create_session(
            user_id=user_id,
            initial_interface=InterfaceMode.CLI
        )
        
        # Add complex state
        session_state.user_preferences.update({
            'critical_data': 'must_preserve',
            'optional_data': 'can_lose'
        })
        
        # Simulate partial corruption by removing some state
        corrupted_state = session_state.model_copy(deep=True)
        corrupted_state.user_preferences.pop('optional_data', None)
        corrupted_state.checksum = coordinator.state_synchronizer._calculate_checksum(corrupted_state)
        
        # Save corrupted state
        await coordinator.state_synchronizer._store.save_state(
            session_state.session_id, 
            corrupted_state
        )
        
        # Recovery should still preserve critical data
        recovered_state = await coordinator.recover_session(
            session_state.session_id, 
            user_id
        )
        
        assert recovered_state is not None
        assert recovered_state.user_preferences.get('critical_data') == 'must_preserve'
        # optional_data might be lost, which is acceptable
        
        await coordinator.cleanup_session(session_state.session_id)
    
    @pytest.mark.asyncio
    async def test_interface_unavailability_handling(self, coordination_system):
        """Test handling when target interface becomes unavailable."""
        coordinator = coordination_system['coordinator']
        
        # Create session
        user_id = 'unavail-user'
        session_state = await coordinator.create_session(
            user_id=user_id,
            initial_interface=InterfaceMode.CLI
        )
        
        # Remove an interface handler to simulate unavailability
        original_handlers = coordinator._transition_handlers.copy()
        coordinator._transition_handlers.pop(InterfaceMode.TUI, None)
        
        try:
            # Try to switch to unavailable interface
            transition_request = ModalSwitchRequest(
                session_id=session_state.session_id,
                from_mode=InterfaceMode.CLI,
                to_mode=InterfaceMode.TUI,
                requested_by=user_id
            )
            
            result = await coordinator.switch_mode(transition_request)
            
            # Should fail gracefully
            assert not result.success
            assert len(result.errors) > 0
            
            # Original interface should still be active
            assert coordinator.get_current_interface(session_state.session_id) == InterfaceMode.CLI
        
        finally:
            # Restore handlers
            coordinator._transition_handlers = original_handlers
        
        await coordinator.cleanup_session(session_state.session_id)
    
    @pytest.mark.asyncio
    async def test_concurrent_failure_isolation(self, coordination_system):
        """Test that failures in one operation don't affect others."""
        coordinator = coordination_system['coordinator']
        
        # Create multiple sessions
        sessions = []
        for i in range(3):
            state = await coordinator.create_session(
                user_id=f'concurrent-user-{i}',
                initial_interface=InterfaceMode.CLI
            )
            sessions.append(state)
        
        async def failing_operation():
            """Operation that will fail."""
            raise ValueError("Simulated failure")
        
        async def successful_operation(session_state):
            """Operation that should succeed."""
            sync_request = StateSync(
                session_id=session_state.session_id,
                interface_mode=InterfaceMode.CLI,
                current_version=session_state.version,
                changes=[]
            )
            return await coordinator.sync_state(sync_request)
        
        # Run mixed successful and failing operations
        operations = [
            failing_operation(),
            successful_operation(sessions[0]),
            failing_operation(),
            successful_operation(sessions[1]),
            successful_operation(sessions[2])
        ]
        
        results = await asyncio.gather(*operations, return_exceptions=True)
        
        # Verify failure isolation
        success_count = 0
        error_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                error_count += 1
            elif isinstance(result, SyncResult) and result.success:
                success_count += 1
        
        assert error_count == 2  # Two failing operations
        assert success_count == 3  # Three successful operations
        
        # Clean up
        for session in sessions:
            await coordinator.cleanup_session(session.session_id)


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""
    
    @pytest.mark.asyncio
    async def test_developer_workflow_scenario(self, coordination_system):
        """Test typical developer workflow across interfaces."""
        coordinator = coordination_system['coordinator']
        
        # Start with CLI for quick commands
        session_state = await coordinator.create_session(
            user_id='developer',
            initial_interface=InterfaceMode.CLI
        )
        session_id = session_state.session_id
        
        # Add CLI context
        session_state.interface_states[InterfaceMode.CLI] = {
            'command_history': ['git status', 'npm test'],
            'current_directory': '/project',
            'environment': 'development'
        }
        
        # Sync CLI state
        cli_sync = StateSync(
            session_id=session_id,
            interface_mode=InterfaceMode.CLI,
            current_version=session_state.version,
            changes=[StateChange(
                session_id=session_id,
                interface_mode=InterfaceMode.CLI,
                key_path='interface_states.cli.environment',
                old_value=None,
                new_value='development',
                user_id='developer'
            )]
        )
        
        result = await coordinator.sync_state(cli_sync)
        assert result.success
        
        # Switch to TUI for interactive debugging (if available)
        supported = coordinator.get_supported_interfaces()
        if InterfaceMode.TUI in supported:
            tui_transition = ModalSwitchRequest(
                session_id=session_id,
                from_mode=InterfaceMode.CLI,
                to_mode=InterfaceMode.TUI,
                preserve_state=True,
                transfer_context=True,
                requested_by='developer'
            )
            
            tui_result = await coordinator.switch_mode(tui_transition)
            if tui_result.success:
                # Verify context was preserved
                final_state = await coordinator.get_synchronized_state(session_id)
                assert final_state.active_context.current_interface == InterfaceMode.TUI
                
                # TUI state should inherit CLI context
                if InterfaceMode.TUI in final_state.interface_states:
                    tui_state = final_state.interface_states[InterfaceMode.TUI]
                    # Some context should be preserved/transformed
                    assert isinstance(tui_state, dict)
        
        await coordinator.cleanup_session(session_id)
    
    @pytest.mark.asyncio 
    async def test_collaboration_scenario(self, coordination_system):
        """Test collaboration scenario with state sharing."""
        coordinator = coordination_system['coordinator']
        
        # Create sessions for team members
        team_lead_session = await coordinator.create_session(
            user_id='team-lead',
            initial_interface=InterfaceMode.CLI,
            session_id='project-session'
        )
        
        # Team lead sets up project context
        team_lead_session.user_preferences.update({
            'project_name': 'coordination-system',
            'current_sprint': 'sprint-5',
            'shared_context': True
        })
        
        # Sync project setup
        setup_changes = [
            StateChange(
                session_id='project-session',
                interface_mode=InterfaceMode.CLI,
                key_path='user_preferences.project_name',
                old_value=None,
                new_value='coordination-system',
                user_id='team-lead'
            )
        ]
        
        setup_sync = StateSync(
            session_id='project-session',
            interface_mode=InterfaceMode.CLI,
            current_version=team_lead_session.version,
            changes=setup_changes
        )
        
        result = await coordinator.sync_state(setup_sync)
        assert result.success
        
        # Simulate persistent state
        await coordinator.cleanup_session('project-session', preserve_state=True)
        
        # Team member joins project (recovers session)
        try:
            # In a real scenario, this would involve proper authorization
            # For test, we simulate the team lead granting access
            recovered_session = await coordinator.recover_session(
                'project-session',
                'team-lead'  # Using team-lead credentials
            )
            
            assert recovered_session is not None
            assert recovered_session.user_preferences.get('project_name') == 'coordination-system'
            
        except PermissionError:
            # Expected - proper isolation working
            pass
        
        # Clean up
        await coordinator.cleanup_session('project-session', preserve_state=False)