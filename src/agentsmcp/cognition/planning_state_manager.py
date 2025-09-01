"""Planning State Persistence Manager for AgentsMCP

This module provides robust state persistence and recovery capabilities for thinking processes,
allowing planning states to survive system interruptions and be resumed later.

Features:
- State serialization with JSON and pickle support
- Incremental state checkpointing during long operations
- Recovery from partial states with validation
- State compression for efficient storage
- Cleanup policies for state management
- Version compatibility checking
"""

import asyncio
import gzip
import json
import pickle
import logging
import hashlib
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import asdict, is_dataclass
from enum import Enum

from .models import (
    ThinkingResult, ThinkingStep, PlanningState, StateCheckpoint, 
    StateRecoveryInfo, StateMetadata, StatePersistenceConfig,
    PersistenceFormat, CheckpointStrategy, CleanupPolicy
)
from .exceptions import StateManagerError, StateCorruptedError, StateVersionError


logger = logging.getLogger(__name__)


class StateSerializer:
    """Handles serialization/deserialization of planning states."""
    
    @staticmethod
    def serialize_to_json(data: Any, pretty: bool = False) -> str:
        """Serialize data to JSON with dataclass support."""
        def convert_dataclass(obj):
            if is_dataclass(obj):
                return asdict(obj)
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, 'dict'):  # Pydantic-style
                return obj.dict()
            return str(obj)
        
        return json.dumps(data, default=convert_dataclass, indent=2 if pretty else None)
    
    @staticmethod
    def deserialize_from_json(data: str) -> Any:
        """Deserialize data from JSON."""
        return json.loads(data)
    
    @staticmethod
    def serialize_to_pickle(data: Any) -> bytes:
        """Serialize data to pickle format."""
        return pickle.dumps(data)
    
    @staticmethod
    def deserialize_from_pickle(data: bytes) -> Any:
        """Deserialize data from pickle format."""
        return pickle.loads(data)
    
    @staticmethod
    def compress_data(data: Union[str, bytes]) -> bytes:
        """Compress data using gzip."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return gzip.compress(data)
    
    @staticmethod
    def decompress_data(data: bytes) -> bytes:
        """Decompress gzip data."""
        return gzip.decompress(data)


class StateValidator:
    """Validates planning states for corruption and version compatibility."""
    
    STATE_VERSION = "1.0.0"
    REQUIRED_FIELDS = {"state_id", "created_at", "thinking_trace"}
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_state(self, state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate state structure and content."""
        errors = []
        
        # Check required fields
        missing_fields = self.REQUIRED_FIELDS - set(state.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
        
        # Check version compatibility
        state_version = state.get("version", "unknown")
        if not self._is_version_compatible(state_version):
            errors.append(f"Incompatible state version: {state_version} (current: {self.STATE_VERSION})")
        
        # Validate thinking trace
        if "thinking_trace" in state:
            trace_errors = self._validate_thinking_trace(state["thinking_trace"])
            errors.extend(trace_errors)
        
        # Check timestamp validity
        if "created_at" in state:
            try:
                datetime.fromisoformat(state["created_at"])
            except ValueError:
                errors.append("Invalid created_at timestamp format")
        
        return len(errors) == 0, errors
    
    def _is_version_compatible(self, version: str) -> bool:
        """Check if state version is compatible with current version."""
        if version == "unknown" or version == self.STATE_VERSION:
            return True
        
        # Simple major.minor.patch version check
        try:
            current_parts = [int(x) for x in self.STATE_VERSION.split('.')]
            state_parts = [int(x) for x in version.split('.')]
            
            # Compatible if major version matches
            return current_parts[0] == state_parts[0]
        except (ValueError, IndexError):
            return False
    
    def _validate_thinking_trace(self, trace: List[Dict]) -> List[str]:
        """Validate thinking trace structure."""
        errors = []
        
        if not isinstance(trace, list):
            errors.append("Thinking trace must be a list")
            return errors
        
        for i, step in enumerate(trace):
            if not isinstance(step, dict):
                errors.append(f"Step {i} must be a dictionary")
                continue
            
            if "phase" not in step:
                errors.append(f"Step {i} missing required 'phase' field")
            
            if "timestamp" not in step:
                errors.append(f"Step {i} missing required 'timestamp' field")
        
        return errors


class PlanningStateManager:
    """Manages persistence and recovery of planning states."""
    
    def __init__(self, config: StatePersistenceConfig = None):
        """Initialize the state manager with configuration."""
        self.config = config or StatePersistenceConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Components
        self.serializer = StateSerializer()
        self.validator = StateValidator()
        
        # State tracking
        self.active_states: Dict[str, PlanningState] = {}
        self.checkpoint_timers: Dict[str, asyncio.Task] = {}
        
        # Ensure storage directory exists
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"State manager initialized with storage: {self.config.storage_path}")
    
    async def save_state(self, state: PlanningState, force_sync: bool = False) -> StateMetadata:
        """Save planning state to persistent storage."""
        try:
            state_id = state.state_id
            self.logger.debug(f"Saving state: {state_id}")
            
            # Create state data for serialization
            state_data = {
                "state_id": state_id,
                "created_at": state.created_at.isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version": self.validator.STATE_VERSION,
                "thinking_trace": [asdict(step) for step in state.thinking_trace],
                "current_context": state.current_context,
                "config_snapshot": asdict(state.config_snapshot) if state.config_snapshot else None,
                "metadata": state.metadata
            }
            
            # Serialize based on format preference
            if self.config.format == PersistenceFormat.JSON:
                serialized_data = self.serializer.serialize_to_json(state_data, pretty=True)
                file_content = serialized_data.encode('utf-8')
                file_extension = ".json"
            else:  # PICKLE
                serialized_data = self.serializer.serialize_to_pickle(state_data)
                file_content = serialized_data
                file_extension = ".pkl"
            
            # Compress if enabled
            if self.config.compress:
                file_content = self.serializer.compress_data(file_content)
                file_extension += ".gz"
            
            # Write to file
            file_path = self.config.storage_path / f"{state_id}{file_extension}"
            
            if force_sync or not self.config.async_writes:
                # Synchronous write
                with open(file_path, 'wb') as f:
                    f.write(file_content)
            else:
                # Asynchronous write
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: file_path.write_bytes(file_content)
                )
            
            # Create metadata
            metadata = StateMetadata(
                state_id=state_id,
                file_path=file_path,
                size_bytes=len(file_content),
                created_at=state.created_at,
                updated_at=datetime.now(),
                format=self.config.format,
                compressed=self.config.compress,
                checksum=hashlib.sha256(file_content).hexdigest()
            )
            
            # Update active states
            self.active_states[state_id] = state
            
            # Schedule automatic checkpoint if needed
            if self.config.checkpoint_strategy == CheckpointStrategy.TIME_BASED:
                await self._schedule_checkpoint(state_id)
            
            self.logger.info(f"State saved: {state_id} ({metadata.size_bytes} bytes)")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error saving state {state.state_id}: {e}")
            raise StateManagerError(f"Failed to save state: {e}") from e
    
    async def load_state(self, state_id: str) -> Optional[PlanningState]:
        """Load planning state from persistent storage."""
        try:
            self.logger.debug(f"Loading state: {state_id}")
            
            # Find state file
            state_file = await self._find_state_file(state_id)
            if not state_file:
                self.logger.warning(f"State file not found: {state_id}")
                return None
            
            # Read file content
            file_content = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: state_file.read_bytes()
            )
            
            # Decompress if needed
            if state_file.suffix == '.gz':
                file_content = self.serializer.decompress_data(file_content)
            
            # Deserialize based on format
            if state_file.suffix.endswith('.json') or state_file.suffix.endswith('.json.gz'):
                state_data = self.serializer.deserialize_from_json(file_content.decode('utf-8'))
            else:  # pickle format
                state_data = self.serializer.deserialize_from_pickle(file_content)
            
            # Validate state
            is_valid, errors = self.validator.validate_state(state_data)
            if not is_valid:
                raise StateCorruptedError(f"Invalid state data: {errors}")
            
            # Reconstruct PlanningState object
            planning_state = await self._reconstruct_state(state_data)
            
            # Update active states
            self.active_states[state_id] = planning_state
            
            self.logger.info(f"State loaded: {state_id}")
            return planning_state
            
        except StateCorruptedError:
            raise
        except Exception as e:
            self.logger.error(f"Error loading state {state_id}: {e}")
            raise StateManagerError(f"Failed to load state: {e}") from e
    
    async def create_checkpoint(self, state_id: str, step: ThinkingStep) -> StateCheckpoint:
        """Create a checkpoint for incremental state saving."""
        try:
            if state_id not in self.active_states:
                raise StateManagerError(f"State not active: {state_id}")
            
            state = self.active_states[state_id]
            
            # Add step to thinking trace
            state.thinking_trace.append(step)
            state.updated_at = datetime.now()
            
            # Save updated state
            metadata = await self.save_state(state)
            
            # Create checkpoint record
            checkpoint = StateCheckpoint(
                checkpoint_id=f"{state_id}_{len(state.thinking_trace)}",
                state_id=state_id,
                step_index=len(state.thinking_trace) - 1,
                created_at=datetime.now(),
                metadata={"step_phase": step.phase.value, "step_duration": step.duration_ms}
            )
            
            self.logger.debug(f"Checkpoint created: {checkpoint.checkpoint_id}")
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Error creating checkpoint for {state_id}: {e}")
            raise StateManagerError(f"Failed to create checkpoint: {e}") from e
    
    async def recover_state(self, state_id: str, target_step: Optional[int] = None) -> StateRecoveryInfo:
        """Recover state from storage with optional rollback to specific step."""
        try:
            self.logger.info(f"Recovering state: {state_id}")
            
            # Load base state
            state = await self.load_state(state_id)
            if not state:
                return StateRecoveryInfo(
                    state_id=state_id,
                    recovered=False,
                    error_message="State not found"
                )
            
            # Rollback to specific step if requested
            if target_step is not None:
                if 0 <= target_step < len(state.thinking_trace):
                    state.thinking_trace = state.thinking_trace[:target_step + 1]
                    self.logger.info(f"Rolled back to step {target_step}")
                else:
                    self.logger.warning(f"Invalid target step {target_step}, using full trace")
            
            # Validate recovered state
            recovery_info = StateRecoveryInfo(
                state_id=state_id,
                recovered=True,
                recovered_at=datetime.now(),
                steps_recovered=len(state.thinking_trace),
                last_checkpoint_time=state.updated_at,
                recovery_method="full_state_load"
            )
            
            self.logger.info(f"State recovered: {state_id} ({recovery_info.steps_recovered} steps)")
            return recovery_info
            
        except Exception as e:
            self.logger.error(f"Error recovering state {state_id}: {e}")
            return StateRecoveryInfo(
                state_id=state_id,
                recovered=False,
                error_message=str(e),
                recovery_method="failed"
            )
    
    async def delete_state(self, state_id: str) -> bool:
        """Delete state from persistent storage."""
        try:
            self.logger.debug(f"Deleting state: {state_id}")
            
            # Find and remove state file
            state_file = await self._find_state_file(state_id)
            if state_file and state_file.exists():
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: state_file.unlink()
                )
                self.logger.info(f"State file deleted: {state_file}")
            
            # Remove from active states
            if state_id in self.active_states:
                del self.active_states[state_id]
            
            # Cancel checkpoint timer
            if state_id in self.checkpoint_timers:
                self.checkpoint_timers[state_id].cancel()
                del self.checkpoint_timers[state_id]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting state {state_id}: {e}")
            return False
    
    async def list_states(self, include_metadata: bool = False) -> List[Union[str, StateMetadata]]:
        """List all stored states."""
        try:
            states = []
            
            for file_path in self.config.storage_path.iterdir():
                if file_path.is_file():
                    # Extract state ID from filename
                    name_parts = file_path.name.split('.')
                    if name_parts:
                        state_id = name_parts[0]
                        
                        if include_metadata:
                            # Load minimal metadata
                            try:
                                stat = file_path.stat()
                                metadata = StateMetadata(
                                    state_id=state_id,
                                    file_path=file_path,
                                    size_bytes=stat.st_size,
                                    created_at=datetime.fromtimestamp(stat.st_ctime),
                                    updated_at=datetime.fromtimestamp(stat.st_mtime),
                                    format=PersistenceFormat.JSON if 'json' in file_path.suffix else PersistenceFormat.PICKLE,
                                    compressed='.gz' in file_path.suffix
                                )
                                states.append(metadata)
                            except Exception as e:
                                self.logger.warning(f"Error getting metadata for {file_path}: {e}")
                                states.append(state_id)
                        else:
                            states.append(state_id)
            
            return states
            
        except Exception as e:
            self.logger.error(f"Error listing states: {e}")
            return []
    
    async def cleanup_old_states(self) -> int:
        """Clean up old states based on cleanup policy."""
        try:
            if self.config.cleanup_policy == CleanupPolicy.NEVER:
                return 0
            
            self.logger.info("Starting state cleanup")
            cleanup_count = 0
            
            # Calculate cutoff time
            if self.config.cleanup_policy == CleanupPolicy.TIME_BASED:
                cutoff_time = datetime.now() - timedelta(seconds=self.config.max_age_seconds)
            else:  # SIZE_BASED
                # Get all states with metadata
                all_states = await self.list_states(include_metadata=True)
                if len(all_states) <= self.config.max_states:
                    return 0
                
                # Sort by creation time (oldest first)
                all_states.sort(key=lambda x: x.created_at if hasattr(x, 'created_at') else datetime.min)
                states_to_remove = all_states[:-self.config.max_states]
                
                for state_metadata in states_to_remove:
                    if hasattr(state_metadata, 'state_id'):
                        if await self.delete_state(state_metadata.state_id):
                            cleanup_count += 1
                
                self.logger.info(f"Cleaned up {cleanup_count} states (size-based)")
                return cleanup_count
            
            # Time-based cleanup
            for file_path in self.config.storage_path.iterdir():
                if file_path.is_file():
                    try:
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_mtime < cutoff_time:
                            state_id = file_path.name.split('.')[0]
                            if await self.delete_state(state_id):
                                cleanup_count += 1
                    except Exception as e:
                        self.logger.warning(f"Error checking file {file_path}: {e}")
            
            self.logger.info(f"Cleaned up {cleanup_count} old states")
            return cleanup_count
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about state storage."""
        try:
            stats = {
                "total_states": 0,
                "total_size_bytes": 0,
                "active_states": len(self.active_states),
                "storage_path": str(self.config.storage_path),
                "format": self.config.format.value,
                "compressed": self.config.compress
            }
            
            for file_path in self.config.storage_path.iterdir():
                if file_path.is_file():
                    stats["total_states"] += 1
                    stats["total_size_bytes"] += file_path.stat().st_size
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting storage stats: {e}")
            return {"error": str(e)}
    
    async def _find_state_file(self, state_id: str) -> Optional[Path]:
        """Find state file by ID, handling different formats and compression."""
        possible_extensions = ['.json', '.json.gz', '.pkl', '.pkl.gz']
        
        for ext in possible_extensions:
            file_path = self.config.storage_path / f"{state_id}{ext}"
            if file_path.exists():
                return file_path
        
        return None
    
    async def _reconstruct_state(self, state_data: Dict[str, Any]) -> PlanningState:
        """Reconstruct PlanningState object from serialized data."""
        # Reconstruct thinking trace
        thinking_trace = []
        for step_data in state_data.get("thinking_trace", []):
            # This would require proper deserialization of ThinkingStep objects
            # For now, we'll create a minimal reconstruction
            thinking_trace.append(step_data)  # Simplified - should reconstruct ThinkingStep objects
        
        return PlanningState(
            state_id=state_data["state_id"],
            created_at=datetime.fromisoformat(state_data["created_at"]),
            updated_at=datetime.fromisoformat(state_data.get("updated_at", state_data["created_at"])),
            thinking_trace=thinking_trace,
            current_context=state_data.get("current_context", {}),
            config_snapshot=state_data.get("config_snapshot"),
            metadata=state_data.get("metadata", {})
        )
    
    async def _schedule_checkpoint(self, state_id: str):
        """Schedule automatic checkpoint creation."""
        if state_id in self.checkpoint_timers:
            self.checkpoint_timers[state_id].cancel()
        
        async def checkpoint_task():
            await asyncio.sleep(self.config.checkpoint_interval_seconds)
            if state_id in self.active_states:
                try:
                    state = self.active_states[state_id]
                    await self.save_state(state)
                    self.logger.debug(f"Auto-checkpoint completed for {state_id}")
                except Exception as e:
                    self.logger.error(f"Auto-checkpoint failed for {state_id}: {e}")
            
            # Schedule next checkpoint
            if state_id in self.active_states:
                await self._schedule_checkpoint(state_id)
        
        self.checkpoint_timers[state_id] = asyncio.create_task(checkpoint_task())
    
    async def shutdown(self):
        """Shutdown the state manager and clean up resources."""
        self.logger.info("Shutting down state manager")
        
        # Cancel all checkpoint timers
        for timer in self.checkpoint_timers.values():
            timer.cancel()
        self.checkpoint_timers.clear()
        
        # Save all active states
        save_tasks = []
        for state in self.active_states.values():
            save_tasks.append(self.save_state(state, force_sync=True))
        
        if save_tasks:
            await asyncio.gather(*save_tasks, return_exceptions=True)
        
        # Clear active states
        self.active_states.clear()
        
        self.logger.info("State manager shutdown complete")


# Convenience functions
async def create_state_manager(config: Optional[StatePersistenceConfig] = None) -> PlanningStateManager:
    """Create and initialize a state manager with default configuration."""
    return PlanningStateManager(config or StatePersistenceConfig())


async def save_thinking_result(result: ThinkingResult, 
                             storage_path: Optional[Path] = None) -> StateMetadata:
    """Convenience function to save a thinking result to storage."""
    config = StatePersistenceConfig(storage_path=storage_path or Path("./thinking_states"))
    manager = PlanningStateManager(config)
    
    # Create planning state from result
    state = PlanningState(
        state_id=f"thinking_{int(datetime.now().timestamp())}",
        created_at=datetime.now(),
        thinking_trace=result.thinking_trace,
        current_context=result.context or {},
        metadata={"confidence": result.confidence, "duration_ms": result.total_duration_ms}
    )
    
    return await manager.save_state(state)


async def load_thinking_result(state_id: str, 
                             storage_path: Optional[Path] = None) -> Optional[PlanningState]:
    """Convenience function to load a thinking result from storage."""
    config = StatePersistenceConfig(storage_path=storage_path or Path("./thinking_states"))
    manager = PlanningStateManager(config)
    
    return await manager.load_state(state_id)