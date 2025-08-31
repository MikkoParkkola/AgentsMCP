"""State synchronizer for cross-modal interface coordination.

This module provides real-time state synchronization across interfaces with
conflict resolution, persistent storage, and session recovery capabilities.
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

from ..models.coordination_models import (
    InterfaceMode,
    SyncStatus,
    ConflictResolution,
    StateChange,
    ConflictInfo,
    SharedState,
    SessionContext,
    StateSync,
    SyncResult,
    SyncFailedError,
    ConflictResolutionError,
    StateLossError
)

logger = logging.getLogger(__name__)


class StateStore:
    """Encrypted persistent storage for state synchronization."""
    
    def __init__(self, storage_path: str, encryption_key: Optional[str] = None):
        """Initialize state store with encryption."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        if encryption_key:
            key_bytes = encryption_key.encode()
        else:
            key_bytes = os.environ.get("AGENTSMCP_STATE_KEY", "default-dev-key").encode()
        
        # Derive encryption key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'agentsmcp_state_salt',  # In production, use random salt per deployment
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(key_bytes))
        self._cipher = Fernet(key)
    
    async def save_state(self, session_id: str, state: SharedState) -> None:
        """Save encrypted state to disk."""
        state_file = self.storage_path / f"{session_id}.json"
        
        try:
            # Serialize state
            state_data = state.model_dump(mode='json')
            state_json = json.dumps(state_data, default=str)
            
            # Encrypt and save
            encrypted_data = self._cipher.encrypt(state_json.encode())
            
            # Atomic write
            temp_file = state_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                f.write(encrypted_data)
            temp_file.replace(state_file)
            
            logger.debug(f"State saved for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save state for session {session_id}: {e}")
            raise SyncFailedError(f"State save failed: {e}")
    
    async def load_state(self, session_id: str) -> Optional[SharedState]:
        """Load and decrypt state from disk."""
        state_file = self.storage_path / f"{session_id}.json"
        
        if not state_file.exists():
            return None
        
        try:
            # Load and decrypt
            with open(state_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self._cipher.decrypt(encrypted_data)
            state_data = json.loads(decrypted_data.decode())
            
            # Reconstruct state object
            return SharedState(**state_data)
            
        except Exception as e:
            logger.error(f"Failed to load state for session {session_id}: {e}")
            return None
    
    async def delete_state(self, session_id: str) -> None:
        """Delete state file."""
        state_file = self.storage_path / f"{session_id}.json"
        
        try:
            if state_file.exists():
                state_file.unlink()
                logger.debug(f"State deleted for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to delete state for session {session_id}: {e}")
    
    async def list_sessions(self) -> List[str]:
        """List all stored sessions."""
        try:
            return [
                f.stem for f in self.storage_path.glob("*.json")
                if not f.name.endswith('.tmp')
            ]
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []


class StateSynchronizer:
    """Real-time state synchronizer with conflict resolution."""
    
    def __init__(self, storage_path: str = "~/.agentsmcp/state", encryption_key: Optional[str] = None):
        """Initialize state synchronizer."""
        self.storage_path = Path(storage_path).expanduser()
        self._store = StateStore(str(self.storage_path), encryption_key)
        self._active_sessions: Dict[str, SharedState] = {}
        self._sync_locks: Dict[str, asyncio.Lock] = {}
        self._change_queues: Dict[str, asyncio.Queue] = {}
        self._background_tasks: set = set()
        
    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create lock for session."""
        if session_id not in self._sync_locks:
            self._sync_locks[session_id] = asyncio.Lock()
        return self._sync_locks[session_id]
    
    def _get_change_queue(self, session_id: str) -> asyncio.Queue:
        """Get or create change queue for session."""
        if session_id not in self._change_queues:
            self._change_queues[session_id] = asyncio.Queue()
        return self._change_queues[session_id]
    
    def _calculate_checksum(self, state: SharedState) -> str:
        """Calculate state checksum for integrity verification."""
        # Create deterministic representation
        state_dict = state.model_dump(mode='json', exclude={'checksum', 'last_modified'})
        state_json = json.dumps(state_dict, sort_keys=True, default=str)
        
        return hashlib.sha256(state_json.encode()).hexdigest()
    
    async def create_session(
        self, 
        session_id: str, 
        user_id: str, 
        interface_mode: InterfaceMode
    ) -> SharedState:
        """Create new synchronized session."""
        logger.info(f"Creating new session {session_id} for user {user_id}")
        
        async with self._get_session_lock(session_id):
            # Check if session already exists
            if session_id in self._active_sessions:
                logger.warning(f"Session {session_id} already exists")
                return self._active_sessions[session_id]
            
            # Create session context
            context = SessionContext(
                session_id=session_id,
                user_id=user_id,
                current_interface=interface_mode
            )
            
            # Create shared state
            shared_state = SharedState(
                session_id=session_id,
                active_context=context
            )
            
            # Calculate checksum
            shared_state.checksum = self._calculate_checksum(shared_state)
            
            # Store in memory and disk
            self._active_sessions[session_id] = shared_state
            await self._store.save_state(session_id, shared_state)
            
            # Start background sync task
            task = asyncio.create_task(self._background_sync_handler(session_id))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            
            logger.info(f"Session {session_id} created successfully")
            return shared_state
    
    async def get_session(self, session_id: str) -> Optional[SharedState]:
        """Get current session state."""
        # Check memory first
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]
        
        # Load from disk
        stored_state = await self._store.load_state(session_id)
        if stored_state:
            self._active_sessions[session_id] = stored_state
            
            # Start background sync task
            task = asyncio.create_task(self._background_sync_handler(session_id))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            
            return stored_state
        
        return None
    
    async def synchronize_state(self, sync_request: StateSync) -> SyncResult:
        """Synchronize state changes with conflict resolution."""
        start_time = datetime.now(timezone.utc)
        session_id = sync_request.session_id
        
        logger.debug(f"Synchronizing state for session {session_id}")
        
        async with self._get_session_lock(session_id):
            try:
                # Get current state
                current_state = await self.get_session(session_id)
                if not current_state:
                    raise SyncFailedError(f"Session {session_id} not found")
                
                # Check for concurrent modifications
                if current_state.version > sync_request.current_version:
                    logger.warning(f"Concurrent modification detected for session {session_id}")
                    
                    if not sync_request.force_sync:
                        # Detect conflicts
                        conflicts = await self._detect_conflicts(
                            current_state, 
                            sync_request.changes
                        )
                        
                        if conflicts:
                            # Add conflicts to state
                            current_state.conflicts.extend(conflicts)
                            current_state.sync_status = SyncStatus.CONFLICT
                            await self._store.save_state(session_id, current_state)
                            
                            return SyncResult(
                                sync_id=sync_request.sync_id,
                                success=False,
                                new_version=current_state.version,
                                conflicts_remaining=len(conflicts),
                                errors=[f"Conflicts detected: {len(conflicts)} items"],
                                duration_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                            )
                
                # Apply changes
                changes_applied = await self._apply_changes(current_state, sync_request.changes)
                
                # Resolve any existing conflicts if requested
                conflicts_resolved = 0
                if current_state.conflicts:
                    conflicts_resolved = await self._resolve_conflicts(
                        current_state, 
                        sync_request.conflict_resolution
                    )
                
                # Update state metadata
                current_state.version += 1
                current_state.last_modified = datetime.now(timezone.utc)
                current_state.checksum = self._calculate_checksum(current_state)
                
                # Update sync status
                if not current_state.conflicts:
                    current_state.sync_status = SyncStatus.SYNCED
                
                # Persist state
                await self._store.save_state(session_id, current_state)
                
                # Update active session
                self._active_sessions[session_id] = current_state
                
                end_time = datetime.now(timezone.utc)
                duration_ms = int((end_time - start_time).total_seconds() * 1000)
                
                logger.info(f"State synchronized for session {session_id}: {changes_applied} changes applied")
                
                return SyncResult(
                    sync_id=sync_request.sync_id,
                    success=True,
                    new_version=current_state.version,
                    conflicts_resolved=conflicts_resolved,
                    conflicts_remaining=len(current_state.conflicts),
                    changes_applied=changes_applied,
                    duration_ms=duration_ms
                )
                
            except Exception as e:
                logger.error(f"State synchronization failed for session {session_id}: {e}")
                duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                
                return SyncResult(
                    sync_id=sync_request.sync_id,
                    success=False,
                    new_version=sync_request.current_version,
                    errors=[str(e)],
                    duration_ms=duration_ms
                )
    
    async def _detect_conflicts(
        self, 
        current_state: SharedState, 
        incoming_changes: List[StateChange]
    ) -> List[ConflictInfo]:
        """Detect conflicts between current state and incoming changes."""
        conflicts = []
        
        for change in incoming_changes:
            # Check if there are conflicting changes to the same key path
            conflicting_changes = []
            
            # Look for recent changes to the same path
            # In a real implementation, this would check a change log
            # For now, we'll simulate by checking if the path exists and has been modified
            path_parts = change.key_path.split('.')
            current_value = self._get_nested_value(current_state.model_dump(), path_parts)
            
            if current_value != change.old_value:
                # Create a synthetic conflicting change
                conflicting_change = StateChange(
                    session_id=current_state.session_id,
                    interface_mode=current_state.active_context.current_interface,
                    key_path=change.key_path,
                    old_value=change.old_value,
                    new_value=current_value,
                    user_id=current_state.active_context.user_id,
                    timestamp=current_state.last_modified
                )
                conflicting_changes.append(conflicting_change)
            
            if conflicting_changes:
                conflict = ConflictInfo(
                    key_path=change.key_path,
                    local_change=change,
                    remote_changes=conflicting_changes
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _get_nested_value(self, data: Dict[str, Any], path_parts: List[str]) -> Any:
        """Get value from nested dictionary using dot notation path."""
        current = data
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
    
    def _set_nested_value(self, data: Dict[str, Any], path_parts: List[str], value: Any) -> None:
        """Set value in nested dictionary using dot notation path."""
        current = data
        for part in path_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[path_parts[-1]] = value
    
    async def _apply_changes(self, state: SharedState, changes: List[StateChange]) -> int:
        """Apply changes to state."""
        changes_applied = 0
        state_dict = state.model_dump()
        
        for change in changes:
            try:
                path_parts = change.key_path.split('.')
                self._set_nested_value(state_dict, path_parts, change.new_value)
                changes_applied += 1
                logger.debug(f"Applied change to {change.key_path}")
            except Exception as e:
                logger.error(f"Failed to apply change to {change.key_path}: {e}")
        
        # Reconstruct state object (simplified - in practice would be more sophisticated)
        if changes_applied > 0:
            # Update relevant fields from the modified dictionary
            if 'active_context' in state_dict:
                state.active_context = SessionContext(**state_dict['active_context'])
            if 'user_preferences' in state_dict:
                state.user_preferences = state_dict['user_preferences']
            if 'interface_states' in state_dict:
                state.interface_states = {
                    InterfaceMode(k): v for k, v in state_dict['interface_states'].items()
                }
        
        return changes_applied
    
    async def _resolve_conflicts(
        self, 
        state: SharedState, 
        resolution_strategy: ConflictResolution
    ) -> int:
        """Resolve conflicts using specified strategy."""
        conflicts_resolved = 0
        resolved_conflicts = []
        
        for conflict in state.conflicts:
            try:
                if resolution_strategy == ConflictResolution.LAST_WRITE_WINS:
                    # Use the change with the latest timestamp
                    latest_change = max(
                        [conflict.local_change] + conflict.remote_changes,
                        key=lambda c: c.timestamp
                    )
                    conflict.resolved_value = latest_change.new_value
                    
                elif resolution_strategy == ConflictResolution.MERGE:
                    # Simple merge strategy - combine values if possible
                    if isinstance(conflict.local_change.new_value, dict):
                        merged_value = conflict.local_change.new_value.copy()
                        for remote_change in conflict.remote_changes:
                            if isinstance(remote_change.new_value, dict):
                                merged_value.update(remote_change.new_value)
                        conflict.resolved_value = merged_value
                    else:
                        # Fall back to last write wins
                        latest_change = max(
                            [conflict.local_change] + conflict.remote_changes,
                            key=lambda c: c.timestamp
                        )
                        conflict.resolved_value = latest_change.new_value
                
                elif resolution_strategy == ConflictResolution.REJECT:
                    # Keep current value, reject all changes
                    conflict.resolved_value = conflict.local_change.old_value
                
                else:  # USER_PROMPT
                    # Mark for user resolution - don't auto-resolve
                    continue
                
                # Apply resolved value
                path_parts = conflict.key_path.split('.')
                state_dict = state.model_dump()
                self._set_nested_value(state_dict, path_parts, conflict.resolved_value)
                
                # Mark as resolved
                conflict.resolution_strategy = resolution_strategy
                conflict.resolved_at = datetime.now(timezone.utc)
                resolved_conflicts.append(conflict)
                conflicts_resolved += 1
                
                logger.debug(f"Resolved conflict for {conflict.key_path}")
                
            except Exception as e:
                logger.error(f"Failed to resolve conflict for {conflict.key_path}: {e}")
        
        # Remove resolved conflicts
        state.conflicts = [c for c in state.conflicts if c not in resolved_conflicts]
        
        return conflicts_resolved
    
    async def _background_sync_handler(self, session_id: str) -> None:
        """Background task to handle periodic sync operations."""
        logger.debug(f"Starting background sync handler for session {session_id}")
        
        change_queue = self._get_change_queue(session_id)
        
        try:
            while session_id in self._active_sessions:
                try:
                    # Wait for changes or timeout after 30 seconds
                    await asyncio.wait_for(change_queue.get(), timeout=30.0)
                    
                    # Process accumulated changes
                    await self._process_background_changes(session_id)
                    
                except asyncio.TimeoutError:
                    # Periodic maintenance - verify state integrity
                    await self._verify_state_integrity(session_id)
                    
                except Exception as e:
                    logger.error(f"Error in background sync for session {session_id}: {e}")
                    await asyncio.sleep(5)  # Brief pause before retrying
                    
        except Exception as e:
            logger.error(f"Background sync handler failed for session {session_id}: {e}")
        finally:
            logger.debug(f"Background sync handler stopped for session {session_id}")
    
    async def _process_background_changes(self, session_id: str) -> None:
        """Process accumulated background changes."""
        # In a real implementation, this would batch and apply pending changes
        # For now, we'll just update the last activity timestamp
        state = self._active_sessions.get(session_id)
        if state:
            state.active_context.last_activity = datetime.now(timezone.utc)
            await self._store.save_state(session_id, state)
    
    async def _verify_state_integrity(self, session_id: str) -> None:
        """Verify state integrity using checksums."""
        state = self._active_sessions.get(session_id)
        if not state:
            return
        
        current_checksum = self._calculate_checksum(state)
        if state.checksum != current_checksum:
            logger.warning(f"State integrity check failed for session {session_id}")
            state.checksum = current_checksum
            await self._store.save_state(session_id, state)
    
    async def cleanup_session(self, session_id: str, preserve_state: bool = True) -> None:
        """Clean up session resources."""
        logger.info(f"Cleaning up session {session_id}")
        
        async with self._get_session_lock(session_id):
            # Save final state if preserving
            if preserve_state and session_id in self._active_sessions:
                await self._store.save_state(session_id, self._active_sessions[session_id])
            elif not preserve_state:
                await self._store.delete_state(session_id)
            
            # Clean up memory
            self._active_sessions.pop(session_id, None)
            self._sync_locks.pop(session_id, None)
            self._change_queues.pop(session_id, None)
    
    async def recover_session(self, session_id: str, user_id: str) -> Optional[SharedState]:
        """Recover session from persistent storage."""
        logger.info(f"Recovering session {session_id} for user {user_id}")
        
        # Load from storage
        state = await self._store.load_state(session_id)
        if not state:
            logger.warning(f"No stored state found for session {session_id}")
            return None
        
        # Verify user ownership
        if state.active_context.user_id != user_id:
            logger.error(f"User {user_id} cannot access session {session_id}")
            raise PermissionError("Session access denied")
        
        # Restore to active sessions
        self._active_sessions[session_id] = state
        
        # Start background sync
        task = asyncio.create_task(self._background_sync_handler(session_id))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        logger.info(f"Session {session_id} recovered successfully")
        return state
    
    async def list_user_sessions(self, user_id: str) -> List[str]:
        """List all sessions for a user."""
        session_ids = []
        
        for session_id in await self._store.list_sessions():
            state = await self._store.load_state(session_id)
            if state and state.active_context.user_id == user_id:
                session_ids.append(session_id)
        
        return session_ids
    
    async def shutdown(self) -> None:
        """Shutdown synchronizer and clean up resources."""
        logger.info("Shutting down state synchronizer")
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Save all active sessions
        for session_id, state in self._active_sessions.items():
            try:
                await self._store.save_state(session_id, state)
            except Exception as e:
                logger.error(f"Failed to save session {session_id} during shutdown: {e}")
        
        # Clear memory
        self._active_sessions.clear()
        self._sync_locks.clear()
        self._change_queues.clear()
        
        logger.info("State synchronizer shutdown complete")