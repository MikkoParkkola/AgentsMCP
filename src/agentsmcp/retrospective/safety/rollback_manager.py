"""Rollback management for retrospective system improvements."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pickle
import zipfile

from .safety_config import SafetyConfig, RollbackTrigger

logger = logging.getLogger(__name__)


class RollbackState(str, Enum):
    """States of a rollback point."""
    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class ConfigurationSnapshot:
    """Snapshot of system configuration."""
    snapshot_id: str
    timestamp: datetime
    config_files: Dict[str, str]  # path -> content
    environment_variables: Dict[str, str]
    system_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackPoint:
    """A point-in-time snapshot for rollback purposes."""
    rollback_id: str
    name: str
    description: str
    created_at: datetime
    created_by: str
    
    # State information
    state: RollbackState = RollbackState.CREATED
    configuration_snapshot: Optional[ConfigurationSnapshot] = None
    file_backups: Dict[str, str] = field(default_factory=dict)  # original_path -> backup_path
    
    # Rollback metadata
    trigger_reason: Optional[str] = None
    rollback_duration_seconds: Optional[float] = None
    rollback_success: bool = False
    rollback_errors: List[str] = field(default_factory=list)
    
    # Expiration and cleanup
    expires_at: Optional[datetime] = None
    cleanup_completed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert datetime objects to ISO strings
        if self.created_at:
            result['created_at'] = self.created_at.isoformat()
        if self.expires_at:
            result['expires_at'] = self.expires_at.isoformat()
        if self.configuration_snapshot:
            result['configuration_snapshot']['timestamp'] = \
                self.configuration_snapshot.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RollbackPoint:
        """Create from dictionary."""
        # Convert ISO strings back to datetime objects
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'expires_at' in data and data['expires_at']:
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        if 'configuration_snapshot' in data and data['configuration_snapshot']:
            snapshot_data = data['configuration_snapshot']
            snapshot_data['timestamp'] = datetime.fromisoformat(snapshot_data['timestamp'])
            data['configuration_snapshot'] = ConfigurationSnapshot(**snapshot_data)
        
        return cls(**data)


class RollbackManager:
    """Manages rollback points and operations for safety framework."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.storage_path = Path(config.rollback_state_storage_path)
        self.backup_path = Path(config.backup_directory)
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache of active rollback points
        self._rollback_points: Dict[str, RollbackPoint] = {}
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the rollback manager and load existing rollback points."""
        try:
            await self._load_rollback_points()
            await self._cleanup_expired_rollback_points()
            self.logger.info("RollbackManager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RollbackManager: {e}")
            raise
    
    async def create_rollback_point(
        self,
        name: str,
        description: str,
        created_by: str,
        capture_configuration: bool = True,
        capture_files: List[str] = None,
        expires_in_hours: Optional[int] = None
    ) -> RollbackPoint:
        """Create a new rollback point."""
        async with self._lock:
            rollback_id = f"rb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            try:
                # Create rollback point
                rollback_point = RollbackPoint(
                    rollback_id=rollback_id,
                    name=name,
                    description=description,
                    created_at=datetime.now(timezone.utc),
                    created_by=created_by
                )
                
                # Set expiration
                if expires_in_hours:
                    rollback_point.expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)
                
                # Capture configuration snapshot
                if capture_configuration:
                    rollback_point.configuration_snapshot = await self._capture_configuration_snapshot(rollback_id)
                
                # Capture file backups
                if capture_files:
                    rollback_point.file_backups = await self._capture_file_backups(rollback_id, capture_files)
                
                # Save rollback point
                await self._save_rollback_point(rollback_point)
                
                # Add to cache
                self._rollback_points[rollback_id] = rollback_point
                
                # Cleanup old rollback points if needed
                await self._enforce_max_rollback_history()
                
                self.logger.info(f"Created rollback point: {rollback_id}")
                return rollback_point
                
            except Exception as e:
                self.logger.error(f"Failed to create rollback point: {e}")
                raise
    
    async def execute_rollback(
        self,
        rollback_id: str,
        trigger: RollbackTrigger,
        force: bool = False
    ) -> bool:
        """Execute rollback to a specific point."""
        async with self._lock:
            try:
                rollback_point = self._rollback_points.get(rollback_id)
                if not rollback_point:
                    # Try to load from storage
                    rollback_point = await self._load_rollback_point(rollback_id)
                    if not rollback_point:
                        self.logger.error(f"Rollback point not found: {rollback_id}")
                        return False
                
                # Check if rollback is allowed
                if not force and rollback_point.state != RollbackState.CREATED:
                    self.logger.warning(f"Rollback point {rollback_id} is not in created state: {rollback_point.state}")
                    return False
                
                start_time = datetime.now(timezone.utc)
                rollback_point.state = RollbackState.ACTIVE
                rollback_point.trigger_reason = trigger.value
                
                # Save state update
                await self._save_rollback_point(rollback_point)
                
                self.logger.info(f"Starting rollback to point {rollback_id}, trigger: {trigger}")
                
                success = True
                errors = []
                
                # Restore configuration
                if rollback_point.configuration_snapshot:
                    config_success = await self._restore_configuration(rollback_point.configuration_snapshot)
                    if not config_success:
                        success = False
                        errors.append("Failed to restore configuration")
                
                # Restore files
                if rollback_point.file_backups:
                    file_success = await self._restore_files(rollback_point.file_backups)
                    if not file_success:
                        success = False
                        errors.append("Failed to restore files")
                
                # Update rollback point
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                rollback_point.rollback_duration_seconds = duration
                rollback_point.rollback_success = success
                rollback_point.rollback_errors = errors
                rollback_point.state = RollbackState.COMPLETED if success else RollbackState.FAILED
                
                # Save final state
                await self._save_rollback_point(rollback_point)
                
                if success:
                    self.logger.info(f"Rollback completed successfully in {duration:.2f}s")
                else:
                    self.logger.error(f"Rollback failed: {errors}")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Rollback execution failed: {e}")
                
                # Mark rollback as failed
                if rollback_id in self._rollback_points:
                    self._rollback_points[rollback_id].state = RollbackState.FAILED
                    self._rollback_points[rollback_id].rollback_errors.append(str(e))
                    await self._save_rollback_point(self._rollback_points[rollback_id])
                
                return False
    
    async def get_rollback_points(self, active_only: bool = False) -> List[RollbackPoint]:
        """Get list of rollback points."""
        rollback_points = list(self._rollback_points.values())
        
        if active_only:
            rollback_points = [
                rp for rp in rollback_points 
                if rp.state == RollbackState.CREATED
            ]
        
        # Sort by creation time (newest first)
        rollback_points.sort(key=lambda x: x.created_at, reverse=True)
        return rollback_points
    
    async def delete_rollback_point(self, rollback_id: str) -> bool:
        """Delete a rollback point and its associated data."""
        async with self._lock:
            try:
                rollback_point = self._rollback_points.get(rollback_id)
                if not rollback_point:
                    self.logger.warning(f"Rollback point not found for deletion: {rollback_id}")
                    return False
                
                # Remove backups
                await self._cleanup_rollback_point_data(rollback_point)
                
                # Remove from storage
                rollback_file = self.storage_path / f"{rollback_id}.json"
                if rollback_file.exists():
                    rollback_file.unlink()
                
                # Remove from cache
                if rollback_id in self._rollback_points:
                    del self._rollback_points[rollback_id]
                
                self.logger.info(f"Deleted rollback point: {rollback_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to delete rollback point {rollback_id}: {e}")
                return False
    
    async def _capture_configuration_snapshot(self, rollback_id: str) -> ConfigurationSnapshot:
        """Capture current system configuration."""
        snapshot_id = f"{rollback_id}_config"
        
        # Capture configuration files
        config_files = {}
        config_paths = [
            "config.json",
            "settings.yaml", 
            "agent_config.json",
            ".agentsmcp/config.json"
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config_files[config_path] = f.read()
                except Exception as e:
                    self.logger.warning(f"Failed to capture config file {config_path}: {e}")
        
        # Capture environment variables
        env_vars = dict(os.environ)
        
        # Capture system state (basic info)
        system_state = {
            "working_directory": os.getcwd(),
            "python_path": os.sys.path[:5],  # First 5 entries
            "platform": os.name,
        }
        
        return ConfigurationSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(timezone.utc),
            config_files=config_files,
            environment_variables=env_vars,
            system_state=system_state
        )
    
    async def _capture_file_backups(self, rollback_id: str, file_paths: List[str]) -> Dict[str, str]:
        """Create backups of specified files."""
        file_backups = {}
        backup_dir = self.backup_path / rollback_id
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    # Create backup path maintaining directory structure
                    relative_path = os.path.relpath(file_path)
                    backup_file_path = backup_dir / relative_path
                    backup_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(file_path, str(backup_file_path))
                    file_backups[file_path] = str(backup_file_path)
                    
                    self.logger.debug(f"Backed up file: {file_path} -> {backup_file_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to backup file {file_path}: {e}")
        
        return file_backups
    
    async def _restore_configuration(self, snapshot: ConfigurationSnapshot) -> bool:
        """Restore configuration from snapshot."""
        try:
            # Restore configuration files
            for config_path, content in snapshot.config_files.items():
                try:
                    # Create directory if needed
                    os.makedirs(os.path.dirname(config_path), exist_ok=True)
                    
                    # Write configuration file
                    with open(config_path, 'w') as f:
                        f.write(content)
                    
                    self.logger.debug(f"Restored config file: {config_path}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to restore config file {config_path}: {e}")
                    return False
            
            # Note: We don't restore environment variables as they affect the entire process
            # and could cause instability. This would need to be handled at the application level.
            
            self.logger.info("Configuration restored successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore configuration: {e}")
            return False
    
    async def _restore_files(self, file_backups: Dict[str, str]) -> bool:
        """Restore files from backups."""
        try:
            for original_path, backup_path in file_backups.items():
                try:
                    if os.path.exists(backup_path):
                        # Create directory if needed
                        os.makedirs(os.path.dirname(original_path), exist_ok=True)
                        
                        # Restore file
                        shutil.copy2(backup_path, original_path)
                        self.logger.debug(f"Restored file: {backup_path} -> {original_path}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to restore file {original_path}: {e}")
                    return False
            
            self.logger.info("Files restored successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore files: {e}")
            return False
    
    async def _save_rollback_point(self, rollback_point: RollbackPoint):
        """Save rollback point to persistent storage."""
        try:
            rollback_file = self.storage_path / f"{rollback_point.rollback_id}.json"
            
            with open(rollback_file, 'w') as f:
                json.dump(rollback_point.to_dict(), f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save rollback point {rollback_point.rollback_id}: {e}")
            raise
    
    async def _load_rollback_point(self, rollback_id: str) -> Optional[RollbackPoint]:
        """Load rollback point from storage."""
        try:
            rollback_file = self.storage_path / f"{rollback_id}.json"
            
            if not rollback_file.exists():
                return None
            
            with open(rollback_file, 'r') as f:
                data = json.load(f)
            
            return RollbackPoint.from_dict(data)
            
        except Exception as e:
            self.logger.error(f"Failed to load rollback point {rollback_id}: {e}")
            return None
    
    async def _load_rollback_points(self):
        """Load all rollback points from storage."""
        try:
            for rollback_file in self.storage_path.glob("*.json"):
                rollback_id = rollback_file.stem
                rollback_point = await self._load_rollback_point(rollback_id)
                
                if rollback_point:
                    self._rollback_points[rollback_id] = rollback_point
            
            self.logger.info(f"Loaded {len(self._rollback_points)} rollback points")
            
        except Exception as e:
            self.logger.error(f"Failed to load rollback points: {e}")
    
    async def _cleanup_expired_rollback_points(self):
        """Clean up expired rollback points."""
        now = datetime.now(timezone.utc)
        expired_points = []
        
        for rollback_id, rollback_point in self._rollback_points.items():
            if rollback_point.expires_at and rollback_point.expires_at < now:
                expired_points.append(rollback_id)
        
        for rollback_id in expired_points:
            await self.delete_rollback_point(rollback_id)
        
        if expired_points:
            self.logger.info(f"Cleaned up {len(expired_points)} expired rollback points")
    
    async def _enforce_max_rollback_history(self):
        """Enforce maximum rollback history limit."""
        if len(self._rollback_points) <= self.config.max_rollback_history:
            return
        
        # Sort by creation time and remove oldest
        rollback_points = sorted(
            self._rollback_points.values(),
            key=lambda x: x.created_at
        )
        
        excess_count = len(rollback_points) - self.config.max_rollback_history
        for i in range(excess_count):
            await self.delete_rollback_point(rollback_points[i].rollback_id)
        
        self.logger.info(f"Cleaned up {excess_count} old rollback points")
    
    async def _cleanup_rollback_point_data(self, rollback_point: RollbackPoint):
        """Clean up data associated with a rollback point."""
        try:
            # Remove backup directory
            backup_dir = self.backup_path / rollback_point.rollback_id
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            
            rollback_point.cleanup_completed = True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup rollback point data: {e}")
    
    async def create_emergency_rollback_point(self, trigger: RollbackTrigger) -> RollbackPoint:
        """Create an emergency rollback point for immediate use."""
        return await self.create_rollback_point(
            name=f"Emergency Rollback - {trigger.value}",
            description=f"Emergency rollback point created due to {trigger.value}",
            created_by="safety_system",
            capture_configuration=True,
            expires_in_hours=24  # Emergency points expire in 24 hours
        )