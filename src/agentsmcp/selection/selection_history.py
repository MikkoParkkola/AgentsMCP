"""
Selection History Storage and Retrieval System

Manages historical data for all selection decisions, enabling learning
and performance analysis over time.
"""

import json
import sqlite3
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import asyncio


logger = logging.getLogger(__name__)


@dataclass
class SelectionRecord:
    """Record of a single selection decision and its outcome."""
    
    # Selection context
    selection_id: str
    timestamp: datetime
    selection_type: str  # 'provider', 'model', 'agent', 'tool'
    task_context: Dict[str, Any]
    available_options: List[str]
    
    # Selection decision  
    selected_option: str
    selection_method: str  # 'exploit', 'explore', 'ab_test', 'fallback'
    confidence_score: float  # 0.0 - 1.0
    selection_metadata: Dict[str, Any]
    
    # Outcome metrics
    success: Optional[bool] = None
    completion_time_ms: Optional[int] = None
    quality_score: Optional[float] = None  # 0.0 - 1.0
    cost: Optional[float] = None
    error_message: Optional[str] = None
    user_feedback: Optional[int] = None  # -1, 0, 1
    
    # Additional metrics
    custom_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelectionRecord':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    def get_outcome_score(self) -> float:
        """Calculate composite outcome score (0.0 - 1.0)."""
        if self.success is None:
            return 0.5  # Neutral for incomplete records
        
        if not self.success:
            return 0.0
        
        # Weight different factors
        score = 1.0  # Base success score
        
        if self.quality_score is not None:
            score = 0.7 * score + 0.3 * self.quality_score
        
        if self.user_feedback is not None:
            feedback_score = (self.user_feedback + 1) / 2.0  # Map -1,0,1 to 0,0.5,1
            score = 0.8 * score + 0.2 * feedback_score
        
        return max(0.0, min(1.0, score))


class SelectionHistory:
    """
    Storage and retrieval system for selection decision history.
    
    Provides persistent storage using SQLite with thread-safe operations
    and efficient querying for learning algorithms.
    """
    
    def __init__(self, db_path: str = None, max_records: int = 100000):
        """
        Initialize selection history storage.
        
        Args:
            db_path: Path to SQLite database file
            max_records: Maximum number of records to keep (oldest deleted first)
        """
        self.max_records = max_records
        self.db_path = db_path or str(Path.home() / ".agentsmcp" / "selection_history.db")
        self._db_lock = threading.RLock()
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Performance cache
        self._cache_lock = threading.RLock()
        self._summary_cache: Dict[str, Any] = {}
        self._cache_expiry = datetime.now() + timedelta(minutes=5)
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS selection_records (
                    selection_id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    selection_type TEXT NOT NULL,
                    task_context TEXT NOT NULL,
                    available_options TEXT NOT NULL,
                    selected_option TEXT NOT NULL,
                    selection_method TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    selection_metadata TEXT NOT NULL,
                    success INTEGER,
                    completion_time_ms INTEGER,
                    quality_score REAL,
                    cost REAL,
                    error_message TEXT,
                    user_feedback INTEGER,
                    custom_metrics TEXT NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_timestamp ON selection_records(timestamp);
                CREATE INDEX IF NOT EXISTS idx_selection_type ON selection_records(selection_type);
                CREATE INDEX IF NOT EXISTS idx_selected_option ON selection_records(selected_option);
                CREATE INDEX IF NOT EXISTS idx_success ON selection_records(success);
                CREATE INDEX IF NOT EXISTS idx_composite ON selection_records(selection_type, selected_option, timestamp);
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get thread-safe database connection."""
        with self._db_lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def record_selection(self, record: SelectionRecord) -> None:
        """
        Store a selection record.
        
        Args:
            record: Selection record to store
        """
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO selection_records (
                        selection_id, timestamp, selection_type, task_context,
                        available_options, selected_option, selection_method,
                        confidence_score, selection_metadata, success,
                        completion_time_ms, quality_score, cost, error_message,
                        user_feedback, custom_metrics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.selection_id,
                    record.timestamp.isoformat(),
                    record.selection_type,
                    json.dumps(record.task_context),
                    json.dumps(record.available_options),
                    record.selected_option,
                    record.selection_method,
                    record.confidence_score,
                    json.dumps(record.selection_metadata),
                    record.success,
                    record.completion_time_ms,
                    record.quality_score,
                    record.cost,
                    record.error_message,
                    record.user_feedback,
                    json.dumps(record.custom_metrics)
                ))
                conn.commit()
            
            # Clear cache on new records
            with self._cache_lock:
                self._summary_cache.clear()
                
            # Cleanup old records if over limit
            self._cleanup_old_records()
            
            logger.debug(f"Recorded selection {record.selection_id}")
            
        except Exception as e:
            logger.error(f"Failed to record selection {record.selection_id}: {e}")
    
    def update_outcome(self, 
                      selection_id: str,
                      success: bool,
                      completion_time_ms: int = None,
                      quality_score: float = None,
                      cost: float = None,
                      error_message: str = None,
                      user_feedback: int = None,
                      custom_metrics: Dict[str, float] = None) -> bool:
        """
        Update the outcome for a previously recorded selection.
        
        Args:
            selection_id: ID of selection to update
            success: Whether the selection was successful
            completion_time_ms: Time to complete task
            quality_score: Quality score (0.0 - 1.0)
            cost: Cost of the selection
            error_message: Error message if failed
            user_feedback: User feedback (-1, 0, 1)
            custom_metrics: Additional metrics
            
        Returns:
            True if record was updated successfully
        """
        try:
            with self._get_connection() as conn:
                # Get current custom metrics
                current = conn.execute(
                    "SELECT custom_metrics FROM selection_records WHERE selection_id = ?",
                    (selection_id,)
                ).fetchone()
                
                if not current:
                    logger.warning(f"Selection record {selection_id} not found for outcome update")
                    return False
                
                # Merge custom metrics
                current_metrics = json.loads(current[0] or '{}')
                if custom_metrics:
                    current_metrics.update(custom_metrics)
                
                # Update record
                conn.execute("""
                    UPDATE selection_records SET
                        success = ?, completion_time_ms = ?, quality_score = ?,
                        cost = ?, error_message = ?, user_feedback = ?,
                        custom_metrics = ?
                    WHERE selection_id = ?
                """, (
                    success, completion_time_ms, quality_score,
                    cost, error_message, user_feedback,
                    json.dumps(current_metrics), selection_id
                ))
                conn.commit()
            
            # Clear cache on updates
            with self._cache_lock:
                self._summary_cache.clear()
            
            logger.debug(f"Updated outcome for selection {selection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update outcome for {selection_id}: {e}")
            return False
    
    def get_records(self,
                   selection_type: str = None,
                   selected_option: str = None,
                   since: datetime = None,
                   limit: int = 1000,
                   only_completed: bool = False) -> List[SelectionRecord]:
        """
        Retrieve selection records with filtering.
        
        Args:
            selection_type: Filter by selection type
            selected_option: Filter by selected option
            since: Only records after this timestamp
            limit: Maximum number of records
            only_completed: Only records with outcomes
            
        Returns:
            List of matching selection records
        """
        conditions = []
        params = []
        
        if selection_type:
            conditions.append("selection_type = ?")
            params.append(selection_type)
        
        if selected_option:
            conditions.append("selected_option = ?")
            params.append(selected_option)
        
        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        
        if only_completed:
            conditions.append("success IS NOT NULL")
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT * FROM selection_records 
            {where_clause}
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        params.append(limit)
        
        try:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
            
            records = []
            for row in rows:
                record = SelectionRecord(
                    selection_id=row['selection_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    selection_type=row['selection_type'],
                    task_context=json.loads(row['task_context']),
                    available_options=json.loads(row['available_options']),
                    selected_option=row['selected_option'],
                    selection_method=row['selection_method'],
                    confidence_score=row['confidence_score'],
                    selection_metadata=json.loads(row['selection_metadata']),
                    success=row['success'],
                    completion_time_ms=row['completion_time_ms'],
                    quality_score=row['quality_score'],
                    cost=row['cost'],
                    error_message=row['error_message'],
                    user_feedback=row['user_feedback'],
                    custom_metrics=json.loads(row['custom_metrics'] or '{}')
                )
                records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to retrieve records: {e}")
            return []
    
    def get_performance_summary(self, 
                              selection_type: str = None,
                              days: int = 30) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Args:
            selection_type: Filter by selection type
            days: Number of days to include
            
        Returns:
            Performance summary with success rates, timing, etc.
        """
        cache_key = f"{selection_type}_{days}"
        
        # Check cache
        with self._cache_lock:
            if (cache_key in self._summary_cache and 
                datetime.now() < self._cache_expiry):
                return self._summary_cache[cache_key]
        
        since = datetime.now() - timedelta(days=days)
        conditions = ["timestamp >= ?", "success IS NOT NULL"]
        params = [since.isoformat()]
        
        if selection_type:
            conditions.append("selection_type = ?")
            params.append(selection_type)
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        try:
            with self._get_connection() as conn:
                # Overall statistics
                stats = conn.execute(f"""
                    SELECT 
                        COUNT(*) as total_selections,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                        AVG(completion_time_ms) as avg_completion_time,
                        AVG(quality_score) as avg_quality_score,
                        AVG(cost) as avg_cost,
                        COUNT(DISTINCT selected_option) as unique_options
                    FROM selection_records {where_clause}
                """, params).fetchone()
                
                # Per-option performance
                option_stats = conn.execute(f"""
                    SELECT 
                        selected_option,
                        COUNT(*) as selections,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                        AVG(completion_time_ms) as avg_time,
                        AVG(quality_score) as avg_quality,
                        AVG(cost) as avg_cost
                    FROM selection_records {where_clause}
                    GROUP BY selected_option
                    ORDER BY successes DESC, selections DESC
                """, params).fetchall()
            
            summary = {
                'total_selections': stats['total_selections'],
                'success_rate': stats['successful'] / max(1, stats['total_selections']),
                'avg_completion_time_ms': stats['avg_completion_time'],
                'avg_quality_score': stats['avg_quality_score'], 
                'avg_cost': stats['avg_cost'],
                'unique_options': stats['unique_options'],
                'options': [
                    {
                        'option': row['selected_option'],
                        'selections': row['selections'],
                        'success_rate': row['successes'] / max(1, row['selections']),
                        'avg_completion_time_ms': row['avg_time'],
                        'avg_quality_score': row['avg_quality'],
                        'avg_cost': row['avg_cost']
                    }
                    for row in option_stats
                ],
                'generated_at': datetime.now().isoformat(),
                'period_days': days
            }
            
            # Cache the result
            with self._cache_lock:
                self._summary_cache[cache_key] = summary
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate performance summary: {e}")
            return {}
    
    def get_option_win_rates(self, 
                           selection_type: str,
                           days: int = 7) -> Dict[str, float]:
        """
        Get win rates for each option in head-to-head comparisons.
        
        Args:
            selection_type: Type of selection to analyze
            days: Number of days to include
            
        Returns:
            Dictionary mapping option names to win rates (0.0 - 1.0)
        """
        since = datetime.now() - timedelta(days=days)
        
        try:
            records = self.get_records(
                selection_type=selection_type,
                since=since,
                only_completed=True,
                limit=10000
            )
            
            # Calculate win rates based on outcome scores
            option_scores = {}
            option_counts = {}
            
            for record in records:
                option = record.selected_option
                score = record.get_outcome_score()
                
                if option not in option_scores:
                    option_scores[option] = []
                option_scores[option].append(score)
                option_counts[option] = option_counts.get(option, 0) + 1
            
            # Calculate average scores as win rates
            win_rates = {}
            for option, scores in option_scores.items():
                if len(scores) >= 3:  # Minimum sample size
                    win_rates[option] = sum(scores) / len(scores)
            
            return win_rates
            
        except Exception as e:
            logger.error(f"Failed to calculate win rates: {e}")
            return {}
    
    def _cleanup_old_records(self):
        """Remove oldest records if over limit."""
        try:
            with self._get_connection() as conn:
                count = conn.execute("SELECT COUNT(*) FROM selection_records").fetchone()[0]
                
                if count > self.max_records:
                    # Delete oldest 10% when over limit
                    delete_count = int(0.1 * self.max_records)
                    conn.execute("""
                        DELETE FROM selection_records 
                        WHERE selection_id IN (
                            SELECT selection_id FROM selection_records 
                            ORDER BY timestamp ASC 
                            LIMIT ?
                        )
                    """, (delete_count,))
                    conn.commit()
                    
                    logger.info(f"Cleaned up {delete_count} old selection records")
        
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
    
    def export_data(self, filepath: str, days: int = 30) -> bool:
        """
        Export selection data to JSON file.
        
        Args:
            filepath: Path to export file
            days: Number of days of data to export
            
        Returns:
            True if export successful
        """
        try:
            since = datetime.now() - timedelta(days=days)
            records = self.get_records(since=since, limit=50000)
            
            export_data = {
                'export_date': datetime.now().isoformat(),
                'period_days': days,
                'total_records': len(records),
                'records': [record.to_dict() for record in records]
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(records)} records to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return False


# Utility functions for generating selection IDs
def generate_selection_id(selection_type: str, 
                         task_context: Dict[str, Any],
                         timestamp: datetime = None) -> str:
    """Generate unique selection ID."""
    if timestamp is None:
        timestamp = datetime.now()
    
    # Create hash of context for uniqueness
    context_str = json.dumps(task_context, sort_keys=True)
    context_hash = hashlib.md5(context_str.encode()).hexdigest()[:8]
    
    return f"{selection_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{context_hash}"