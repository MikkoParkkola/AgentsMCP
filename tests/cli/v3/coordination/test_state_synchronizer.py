"""Tests for state synchronizer functionality."""

import asyncio
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.agentsmcp.cli.v3.coordination.state_synchronizer import StateSynchronizer, StateStore
from src.agentsmcp.cli.v3.models.coordination_models import (
    InterfaceMode,
    SyncStatus,
    ConflictResolution,
    StateChange,
    StateSync,
    SessionContext,
    SharedState,
    SyncFailedError
)


@pytest.fixture
async def synchronizer():
    """Create a state synchronizer for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sync = StateSynchronizer(storage_path=tmpdir)
        yield sync
        await sync.shutdown()


@pytest.fixture
async def sample_session(synchronizer):
    """Create a sample session for testing."""
    session_id = 'test-session-123'
    user_id = 'test-user'
    
    state = await synchronizer.create_session(
        session_id=session_id,
        user_id=user_id,
        interface_mode=InterfaceMode.CLI
    )
    
    return {
        'session_id': session_id,
        'user_id': user_id,
        'state': state
    }


class TestStateStore:
    """Test encrypted state storage."""
    
    def test_state_store_initialization(self):
        """Test state store initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StateStore(tmpdir)
            assert store.storage_path.exists()
            assert store._cipher is not None
    
    @pytest.mark.asyncio
    async def test_save_and_load_state(self):
        """Test saving and loading state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StateStore(tmpdir)
            
            # Create test state
            context = SessionContext(
                session_id='test-session',
                user_id='test-user',
                current_interface=InterfaceMode.CLI
            )
            
            state = SharedState(
                session_id='test-session',
                active_context=context
            )
            
            # Save state
            await store.save_state('test-session', state)
            
            # Verify file exists
            state_file = Path(tmpdir) / 'test-session.json'
            assert state_file.exists()
            
            # Load state
            loaded_state = await store.load_state('test-session')
            
            assert loaded_state is not None
            assert loaded_state.session_id == 'test-session'
            assert loaded_state.active_context.user_id == 'test-user'
    
    @pytest.mark.asyncio
    async def test_delete_state(self):
        """Test state deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StateStore(tmpdir)
            
            # Create and save state
            context = SessionContext(
                session_id='test-session',
                user_id='test-user',
                current_interface=InterfaceMode.CLI
            )
            
            state = SharedState(
                session_id='test-session',
                active_context=context
            )
            
            await store.save_state('test-session', state)
            
            # Verify file exists
            state_file = Path(tmpdir) / 'test-session.json'
            assert state_file.exists()
            
            # Delete state
            await store.delete_state('test-session')
            
            # Verify file is deleted
            assert not state_file.exists()
    
    @pytest.mark.asyncio
    async def test_list_sessions(self):
        """Test listing stored sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StateStore(tmpdir)
            
            # Create multiple sessions
            for i in range(3):
                context = SessionContext(
                    session_id=f'session-{i}',
                    user_id=f'user-{i}',
                    current_interface=InterfaceMode.CLI
                )
                
                state = SharedState(
                    session_id=f'session-{i}',
                    active_context=context
                )
                
                await store.save_state(f'session-{i}', state)
            
            # List sessions
            sessions = await store.list_sessions()
            
            assert len(sessions) == 3
            assert 'session-0' in sessions
            assert 'session-1' in sessions
            assert 'session-2' in sessions


class TestStateSynchronizer:
    """Test state synchronizer functionality."""
    
    @pytest.mark.asyncio
    async def test_create_session(self, synchronizer):
        """Test session creation."""
        state = await synchronizer.create_session(
            session_id='test-session',
            user_id='test-user',
            interface_mode=InterfaceMode.CLI
        )
        
        assert state.session_id == 'test-session'
        assert state.active_context.user_id == 'test-user'
        assert state.active_context.current_interface == InterfaceMode.CLI
        assert state.version == 1
        assert state.checksum is not None
    
    @pytest.mark.asyncio
    async def test_get_session(self, synchronizer, sample_session):
        """Test getting session state."""
        # Get existing session
        state = await synchronizer.get_session(sample_session['session_id'])
        
        assert state is not None
        assert state.session_id == sample_session['session_id']
        assert state.active_context.user_id == sample_session['user_id']
        
        # Get non-existent session
        missing_state = await synchronizer.get_session('nonexistent-session')
        assert missing_state is None
    
    @pytest.mark.asyncio
    async def test_synchronize_state_simple(self, synchronizer, sample_session):
        """Test simple state synchronization."""
        # Create state change
        state_change = StateChange(
            session_id=sample_session['session_id'],
            interface_mode=InterfaceMode.CLI,
            key_path='user_preferences.theme',
            old_value='default',
            new_value='dark',
            user_id=sample_session['user_id']
        )
        
        # Create sync request
        sync_request = StateSync(
            session_id=sample_session['session_id'],
            interface_mode=InterfaceMode.CLI,
            current_version=sample_session['state'].version,
            changes=[state_change]
        )
        
        # Synchronize
        result = await synchronizer.synchronize_state(sync_request)
        
        assert result.success
        assert result.new_version > sample_session['state'].version
        assert result.changes_applied == 1
        assert result.duration_ms > 0
    
    @pytest.mark.asyncio
    async def test_synchronize_state_with_conflict(self, synchronizer, sample_session):
        """Test state synchronization with conflicts."""
        # Simulate concurrent modification by incrementing version
        current_state = await synchronizer.get_session(sample_session['session_id'])
        current_state.version += 1
        await synchronizer._store.save_state(sample_session['session_id'], current_state)
        
        # Create state change with old version
        state_change = StateChange(
            session_id=sample_session['session_id'],
            interface_mode=InterfaceMode.CLI,
            key_path='user_preferences.theme',
            old_value='default',
            new_value='dark',
            user_id=sample_session['user_id']
        )
        
        # Create sync request with old version (should conflict)
        sync_request = StateSync(
            session_id=sample_session['session_id'],
            interface_mode=InterfaceMode.CLI,
            current_version=sample_session['state'].version,  # Old version
            changes=[state_change],
            force_sync=False
        )
        
        # Synchronize - should detect conflict
        result = await synchronizer.synchronize_state(sync_request)
        
        # Should fail due to conflict detection
        assert not result.success or result.conflicts_remaining > 0
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_last_write_wins(self, synchronizer, sample_session):
        """Test last-write-wins conflict resolution."""
        # Create state change
        state_change = StateChange(
            session_id=sample_session['session_id'],
            interface_mode=InterfaceMode.CLI,
            key_path='user_preferences.theme',
            old_value='default',
            new_value='dark',
            user_id=sample_session['user_id']
        )
        
        # Create sync request with force sync
        sync_request = StateSync(
            session_id=sample_session['session_id'],
            interface_mode=InterfaceMode.CLI,
            current_version=sample_session['state'].version,
            changes=[state_change],
            force_sync=True,
            conflict_resolution=ConflictResolution.LAST_WRITE_WINS
        )
        
        # Synchronize with force
        result = await synchronizer.synchronize_state(sync_request)
        
        assert result.success
        assert result.changes_applied > 0
    
    @pytest.mark.asyncio
    async def test_session_recovery(self, synchronizer):
        """Test session recovery from storage."""
        # Create session
        original_state = await synchronizer.create_session(
            session_id='recovery-test',
            user_id='recovery-user',
            interface_mode=InterfaceMode.CLI
        )
        
        # Modify state
        original_state.user_preferences['test_key'] = 'test_value'
        await synchronizer._store.save_state('recovery-test', original_state)
        
        # Clear from memory
        synchronizer._active_sessions.pop('recovery-test', None)
        
        # Recover session
        recovered_state = await synchronizer.recover_session('recovery-test', 'recovery-user')
        
        assert recovered_state is not None
        assert recovered_state.session_id == 'recovery-test'
        assert recovered_state.user_preferences.get('test_key') == 'test_value'
    
    @pytest.mark.asyncio
    async def test_unauthorized_recovery(self, synchronizer):
        """Test unauthorized session recovery."""
        # Create session for user1
        await synchronizer.create_session(
            session_id='auth-test',
            user_id='user1',
            interface_mode=InterfaceMode.CLI
        )
        
        # Try to recover as different user
        with pytest.raises(PermissionError, match="Session access denied"):
            await synchronizer.recover_session('auth-test', 'user2')
    
    @pytest.mark.asyncio
    async def test_list_user_sessions(self, synchronizer):
        """Test listing sessions for a user."""
        user_id = 'multi-session-user'
        
        # Create multiple sessions for the user
        for i in range(3):
            await synchronizer.create_session(
                session_id=f'session-{i}',
                user_id=user_id,
                interface_mode=InterfaceMode.CLI
            )
        
        # Create session for different user
        await synchronizer.create_session(
            session_id='other-session',
            user_id='other-user',
            interface_mode=InterfaceMode.CLI
        )
        
        # List sessions for the user
        user_sessions = await synchronizer.list_user_sessions(user_id)
        
        assert len(user_sessions) == 3
        assert 'session-0' in user_sessions
        assert 'session-1' in user_sessions
        assert 'session-2' in user_sessions
        assert 'other-session' not in user_sessions
    
    @pytest.mark.asyncio
    async def test_cleanup_session(self, synchronizer, sample_session):
        """Test session cleanup."""
        # Verify session exists
        state = await synchronizer.get_session(sample_session['session_id'])
        assert state is not None
        
        # Cleanup with preservation
        await synchronizer.cleanup_session(sample_session['session_id'], preserve_state=True)
        
        # Should be removed from memory but retrievable from disk
        assert sample_session['session_id'] not in synchronizer._active_sessions
        
        # Should be able to recover
        recovered = await synchronizer.recover_session(
            sample_session['session_id'], 
            sample_session['user_id']
        )
        assert recovered is not None
    
    @pytest.mark.asyncio
    async def test_cleanup_session_no_preservation(self, synchronizer, sample_session):
        """Test session cleanup without preservation."""
        # Cleanup without preservation
        await synchronizer.cleanup_session(sample_session['session_id'], preserve_state=False)
        
        # Should not be recoverable
        recovered = await synchronizer.recover_session(
            sample_session['session_id'], 
            sample_session['user_id']
        )
        assert recovered is None
    
    @pytest.mark.asyncio
    async def test_checksum_integrity(self, synchronizer, sample_session):
        """Test state integrity checking with checksums."""
        # Get current state
        state = await synchronizer.get_session(sample_session['session_id'])
        original_checksum = state.checksum
        
        # Modify state and recalculate checksum
        state.user_preferences['new_key'] = 'new_value'
        new_checksum = synchronizer._calculate_checksum(state)
        
        # Checksums should be different
        assert original_checksum != new_checksum
        
        # Update checksum
        state.checksum = new_checksum
        
        # Verify integrity check passes
        await synchronizer._verify_state_integrity(sample_session['session_id'])
    
    @pytest.mark.asyncio
    async def test_performance_sync_under_100ms(self, synchronizer, sample_session):
        """Test that state sync meets performance requirements (<100ms)."""
        # Create simple state change
        state_change = StateChange(
            session_id=sample_session['session_id'],
            interface_mode=InterfaceMode.CLI,
            key_path='test.performance',
            old_value=None,
            new_value='test_value',
            user_id=sample_session['user_id']
        )
        
        # Create sync request
        sync_request = StateSync(
            session_id=sample_session['session_id'],
            interface_mode=InterfaceMode.CLI,
            current_version=sample_session['state'].version,
            changes=[state_change]
        )
        
        # Measure sync time
        start_time = datetime.now(timezone.utc)
        result = await synchronizer.synchronize_state(sync_request)
        end_time = datetime.now(timezone.utc)
        
        actual_duration = (end_time - start_time).total_seconds() * 1000
        
        # Performance requirement: sync < 100ms
        assert actual_duration < 100, f"Sync took {actual_duration}ms, expected < 100ms"
        assert result.duration_ms < 100, f"Reported duration {result.duration_ms}ms, expected < 100ms"


class TestConcurrency:
    """Test concurrent synchronizer operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_session_creation(self, synchronizer):
        """Test concurrent session creation."""
        async def create_session(session_id, user_id):
            return await synchronizer.create_session(
                session_id=session_id,
                user_id=user_id,
                interface_mode=InterfaceMode.CLI
            )
        
        # Create sessions concurrently
        results = await asyncio.gather(
            create_session('concurrent-1', 'user1'),
            create_session('concurrent-2', 'user2'),
            create_session('concurrent-3', 'user3'),
            return_exceptions=True
        )
        
        # All should succeed
        for result in results:
            assert not isinstance(result, Exception), f"Unexpected exception: {result}"
            assert isinstance(result, SharedState)
    
    @pytest.mark.asyncio
    async def test_concurrent_state_sync(self, synchronizer, sample_session):
        """Test concurrent state synchronization."""
        async def sync_state(key_suffix):
            state_change = StateChange(
                session_id=sample_session['session_id'],
                interface_mode=InterfaceMode.CLI,
                key_path=f'concurrent.key_{key_suffix}',
                old_value=None,
                new_value=f'value_{key_suffix}',
                user_id=sample_session['user_id']
            )
            
            # Get current version
            current_state = await synchronizer.get_session(sample_session['session_id'])
            
            sync_request = StateSync(
                session_id=sample_session['session_id'],
                interface_mode=InterfaceMode.CLI,
                current_version=current_state.version,
                changes=[state_change],
                force_sync=True  # Allow concurrent updates
            )
            
            return await synchronizer.synchronize_state(sync_request)
        
        # Perform concurrent syncs
        results = await asyncio.gather(
            sync_state('1'),
            sync_state('2'),
            sync_state('3'),
            return_exceptions=True
        )
        
        # All should complete (some might have conflicts)
        for result in results:
            assert not isinstance(result, Exception), f"Unexpected exception: {result}"
    
    @pytest.mark.asyncio
    async def test_concurrent_recovery(self, synchronizer):
        """Test concurrent session recovery."""
        # Create and save session
        original_state = await synchronizer.create_session(
            session_id='concurrent-recovery',
            user_id='recovery-user',
            interface_mode=InterfaceMode.CLI
        )
        
        # Clear from memory
        synchronizer._active_sessions.pop('concurrent-recovery', None)
        
        # Attempt concurrent recovery
        results = await asyncio.gather(
            synchronizer.recover_session('concurrent-recovery', 'recovery-user'),
            synchronizer.recover_session('concurrent-recovery', 'recovery-user'),
            synchronizer.recover_session('concurrent-recovery', 'recovery-user'),
            return_exceptions=True
        )
        
        # All should succeed or be handled gracefully
        successful_recoveries = [r for r in results if isinstance(r, SharedState)]
        assert len(successful_recoveries) >= 1  # At least one should succeed


class TestErrorHandling:
    """Test error handling in state synchronizer."""
    
    @pytest.mark.asyncio
    async def test_sync_nonexistent_session(self, synchronizer):
        """Test syncing non-existent session."""
        sync_request = StateSync(
            session_id='nonexistent-session',
            interface_mode=InterfaceMode.CLI,
            current_version=1,
            changes=[]
        )
        
        result = await synchronizer.synchronize_state(sync_request)
        
        assert not result.success
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_invalid_state_changes(self, synchronizer, sample_session):
        """Test handling of invalid state changes."""
        # Create invalid state change (invalid key path)
        state_change = StateChange(
            session_id=sample_session['session_id'],
            interface_mode=InterfaceMode.CLI,
            key_path='',  # Empty key path
            old_value=None,
            new_value='test',
            user_id=sample_session['user_id']
        )
        
        sync_request = StateSync(
            session_id=sample_session['session_id'],
            interface_mode=InterfaceMode.CLI,
            current_version=sample_session['state'].version,
            changes=[state_change]
        )
        
        # Should handle gracefully
        result = await synchronizer.synchronize_state(sync_request)
        
        # Might succeed with 0 changes applied or fail gracefully
        assert isinstance(result.changes_applied, int)
        assert result.changes_applied >= 0