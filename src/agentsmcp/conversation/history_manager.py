"""Persistent chat history management for AgentsMCP."""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


@dataclass
class HistoryMessage:
    """A message in the chat history."""
    timestamp: str
    role: str
    content: str
    context_usage: Dict[str, Union[int, float]]
    agent_activities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoryCompaction:
    """Information about a context compaction event."""
    timestamp: str
    messages_summarized: int
    tokens_saved: int
    summary: str
    trigger_percentage: float


@dataclass
class ChatSession:
    """A complete chat session with metadata."""
    session_id: str
    started_at: str
    launch_directory: str
    messages: List[HistoryMessage] = field(default_factory=list)
    context_compactions: List[HistoryCompaction] = field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HistoryManager:
    """Manages persistent chat history storage and retrieval."""
    
    def __init__(self, launch_directory: Optional[str] = None):
        self.launch_directory = launch_directory or os.getcwd()
        self.history_file = Path(self.launch_directory) / ".agentsmcp.log"
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.current_session: Optional[ChatSession] = None
        
        # Initialize or load existing session
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize a new session or load existing one."""
        if self.history_file.exists():
            try:
                self.current_session = self._load_session()
                logger.info(f"Loaded existing session: {self.current_session.session_id}")
            except Exception as e:
                logger.warning(f"Failed to load existing session: {e}")
                self._create_new_session()
        else:
            self._create_new_session()
    
    def _create_new_session(self):
        """Create a new chat session."""
        self.current_session = ChatSession(
            session_id=str(uuid.uuid4()),
            started_at=datetime.now().isoformat(),
            launch_directory=self.launch_directory
        )
        logger.info(f"Created new session: {self.current_session.session_id}")
    
    def _load_session(self) -> ChatSession:
        """Load session from history file."""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert dict data back to dataclass
            session = ChatSession(**data)
            
            # Convert message dicts back to HistoryMessage objects
            session.messages = [
                HistoryMessage(**msg) for msg in session.messages
            ]
            
            # Convert compaction dicts back to HistoryCompaction objects
            session.context_compactions = [
                HistoryCompaction(**comp) for comp in session.context_compactions
            ]
            
            return session
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            raise
    
    def _save_session(self):
        """Save current session to history file."""
        if not self.current_session:
            return
            
        try:
            # Check file size and rotate if needed
            if self.history_file.exists() and self.history_file.stat().st_size > self.max_file_size:
                self._rotate_history_file()
            
            # Convert dataclass to dict for JSON serialization
            session_dict = asdict(self.current_session)
            
            # Ensure directory exists
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first for atomic operation
            temp_file = self.history_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(session_dict, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.rename(self.history_file)
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
    
    def _rotate_history_file(self):
        """Rotate history file when it gets too large."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.history_file.with_suffix(f'.{timestamp}.log')
            shutil.move(str(self.history_file), str(backup_file))
            logger.info(f"Rotated history file to {backup_file}")
        except Exception as e:
            logger.error(f"Error rotating history file: {e}")
    
    def add_message(
        self, 
        role: str, 
        content: str, 
        context_usage: Optional[Dict[str, Union[int, float]]] = None,
        agent_activities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to the current session."""
        if not self.current_session:
            self._create_new_session()
        
        message = HistoryMessage(
            timestamp=datetime.now().isoformat(),
            role=role,
            content=content,
            context_usage=context_usage or {},
            agent_activities=agent_activities or [],
            metadata=metadata or {}
        )
        
        self.current_session.messages.append(message)
        self._save_session()
    
    def add_compaction_event(
        self,
        messages_summarized: int,
        tokens_saved: int,
        summary: str,
        trigger_percentage: float
    ):
        """Add a context compaction event to the history."""
        if not self.current_session:
            return
            
        compaction = HistoryCompaction(
            timestamp=datetime.now().isoformat(),
            messages_summarized=messages_summarized,
            tokens_saved=tokens_saved,
            summary=summary,
            trigger_percentage=trigger_percentage
        )
        
        self.current_session.context_compactions.append(compaction)
        self._save_session()
    
    def get_session_messages(self) -> List[HistoryMessage]:
        """Get all messages from current session."""
        if not self.current_session:
            return []
        return self.current_session.messages.copy()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session."""
        if not self.current_session:
            return {}
        
        messages = self.current_session.messages
        compactions = self.current_session.context_compactions
        
        total_compacted_messages = sum(c.messages_summarized for c in compactions)
        total_tokens_saved = sum(c.tokens_saved for c in compactions)
        
        return {
            'session_id': self.current_session.session_id,
            'started_at': self.current_session.started_at,
            'launch_directory': self.current_session.launch_directory,
            'total_messages': len(messages),
            'total_compactions': len(compactions),
            'total_compacted_messages': total_compacted_messages,
            'total_tokens_saved': total_tokens_saved,
            'provider': self.current_session.provider,
            'model': self.current_session.model
        }
    
    def export_session(self, output_file: Optional[Path] = None) -> Path:
        """Export current session to a file."""
        if not self.current_session:
            raise ValueError("No active session to export")
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"agentsmcp_session_{timestamp}.json")
        
        session_dict = asdict(self.current_session)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(session_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session exported to {output_file}")
        return output_file
    
    def clear_history(self, confirm: bool = False):
        """Clear current session history."""
        if not confirm:
            raise ValueError("Must confirm history clearing with confirm=True")
        
        if self.current_session:
            self.current_session.messages.clear()
            self.current_session.context_compactions.clear()
            self._save_session()
            logger.info("Session history cleared")
    
    def set_provider_model(self, provider: str, model: str):
        """Set the provider and model for the current session."""
        if self.current_session:
            self.current_session.provider = provider
            self.current_session.model = model
            self._save_session()
    
    def get_recent_messages(self, count: int = 10) -> List[HistoryMessage]:
        """Get the most recent messages."""
        if not self.current_session:
            return []
        return self.current_session.messages[-count:]
    
    def search_messages(self, query: str, case_sensitive: bool = False) -> List[HistoryMessage]:
        """Search for messages containing the query string."""
        if not self.current_session:
            return []
        
        matches = []
        search_query = query if case_sensitive else query.lower()
        
        for message in self.current_session.messages:
            content = message.content if case_sensitive else message.content.lower()
            if search_query in content:
                matches.append(message)
        
        return matches
    
    def get_compaction_history(self) -> List[HistoryCompaction]:
        """Get history of context compactions."""
        if not self.current_session:
            return []
        return self.current_session.context_compactions.copy()
    
    def restore_from_backup(self, backup_file: Path):
        """Restore session from a backup file."""
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session = ChatSession(**data)
            session.messages = [HistoryMessage(**msg) for msg in session.messages]
            session.context_compactions = [HistoryCompaction(**comp) for comp in session.context_compactions]
            
            self.current_session = session
            self._save_session()
            
            logger.info(f"Session restored from {backup_file}")
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            raise
    
    def list_backup_files(self) -> List[Path]:
        """List available backup files in the launch directory."""
        backup_pattern = ".agentsmcp.*.log"
        backup_files = list(Path(self.launch_directory).glob(backup_pattern))
        return sorted(backup_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def cleanup_old_backups(self, keep_count: int = 5):
        """Clean up old backup files, keeping only the most recent ones."""
        backup_files = self.list_backup_files()
        
        if len(backup_files) <= keep_count:
            return
        
        files_to_remove = backup_files[keep_count:]
        for backup_file in files_to_remove:
            try:
                backup_file.unlink()
                logger.info(f"Removed old backup: {backup_file}")
            except Exception as e:
                logger.error(f"Error removing backup {backup_file}: {e}")
    
    def get_session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self.current_session.session_id if self.current_session else None
    
    def update_metadata(self, key: str, value: Any):
        """Update session metadata."""
        if self.current_session:
            self.current_session.metadata[key] = value
            self._save_session()