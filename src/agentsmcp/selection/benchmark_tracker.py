"""
Benchmark Tracker for Continuous Performance Monitoring

Tracks performance metrics for every provider/model/agent/tool combination,
providing real-time insights for selection optimization algorithms.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
import statistics
import asyncio

from .selection_history import SelectionHistory, SelectionRecord


logger = logging.getLogger(__name__)


@dataclass
class SelectionMetrics:
    """Real-time metrics for a selection option."""
    
    option_name: str
    selection_type: str
    
    # Core performance metrics
    total_selections: int = 0
    successful_selections: int = 0
    success_rate: float = 0.0
    
    # Timing metrics
    avg_completion_time_ms: float = 0.0
    p95_completion_time_ms: float = 0.0
    completion_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Quality metrics
    avg_quality_score: float = 0.0
    quality_scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Cost metrics
    avg_cost: float = 0.0
    total_cost: float = 0.0
    costs: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Error tracking
    error_count: int = 0
    error_rate: float = 0.0
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # User feedback
    positive_feedback: int = 0
    negative_feedback: int = 0
    feedback_score: float = 0.0
    
    # Temporal tracking
    first_seen: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    
    # Performance trends (recent vs historical)
    recent_success_rate: float = 0.0
    trend_direction: str = "stable"  # "improving", "degrading", "stable"
    
    # Confidence metrics
    sample_confidence: float = 0.0  # Based on sample size
    stability_score: float = 0.0    # Based on variance
    
    def update_from_record(self, record: SelectionRecord):
        """Update metrics from a new selection record."""
        now = datetime.now()
        
        # Initialize timestamps
        if self.first_seen is None:
            self.first_seen = now
        self.last_updated = now
        
        # Update counters
        self.total_selections += 1
        
        if record.success is not None:
            if record.success:
                self.successful_selections += 1
                self.last_success = now
            else:
                self.last_failure = now
            
            # Update success rate
            self.success_rate = self.successful_selections / max(1, self.total_selections)
        
        # Update timing metrics
        if record.completion_time_ms is not None:
            self.completion_times.append(record.completion_time_ms)
            self.avg_completion_time_ms = statistics.mean(self.completion_times)
            if len(self.completion_times) >= 20:  # Need enough samples for p95
                self.p95_completion_time_ms = statistics.quantiles(self.completion_times, n=20)[18]
        
        # Update quality metrics
        if record.quality_score is not None:
            self.quality_scores.append(record.quality_score)
            self.avg_quality_score = statistics.mean(self.quality_scores)
        
        # Update cost metrics
        if record.cost is not None:
            self.costs.append(record.cost)
            self.total_cost += record.cost
            self.avg_cost = statistics.mean(self.costs)
        
        # Update error tracking
        if record.error_message:
            self.error_count += 1
            self.recent_errors.append({
                'timestamp': now,
                'message': record.error_message
            })
        
        self.error_rate = self.error_count / max(1, self.total_selections)
        
        # Update user feedback
        if record.user_feedback is not None:
            if record.user_feedback > 0:
                self.positive_feedback += 1
            elif record.user_feedback < 0:
                self.negative_feedback += 1
        
        total_feedback = self.positive_feedback + self.negative_feedback
        if total_feedback > 0:
            self.feedback_score = self.positive_feedback / total_feedback
        
        # Update confidence and stability
        self._update_confidence_metrics()
        self._update_trend_analysis()
    
    def _update_confidence_metrics(self):
        """Update confidence based on sample size and variance."""
        # Sample confidence (more samples = higher confidence)
        if self.total_selections < 10:
            self.sample_confidence = 0.1
        elif self.total_selections < 50:
            self.sample_confidence = 0.5
        elif self.total_selections < 100:
            self.sample_confidence = 0.7
        else:
            self.sample_confidence = min(0.95, 0.7 + 0.005 * (self.total_selections - 100))
        
        # Stability score based on variance in recent performance
        if len(self.completion_times) >= 10:
            cv = statistics.stdev(self.completion_times) / max(1, statistics.mean(self.completion_times))
            self.stability_score = max(0.0, 1.0 - cv)  # Lower variance = higher stability
        else:
            self.stability_score = 0.5  # Neutral for insufficient data
    
    def _update_trend_analysis(self):
        """Analyze performance trend (improving/degrading/stable)."""
        if self.total_selections < 20:
            self.trend_direction = "stable"
            return
        
        # Compare recent 20% of samples with older samples
        recent_size = max(5, int(0.2 * self.total_selections))
        
        if len(self.completion_times) >= recent_size * 2:
            recent_times = list(self.completion_times)[-recent_size:]
            older_times = list(self.completion_times)[:-recent_size][-recent_size:]
            
            recent_avg = statistics.mean(recent_times)
            older_avg = statistics.mean(older_times)
            
            # Performance improvement means lower completion time
            if recent_avg < older_avg * 0.9:  # 10% improvement
                self.trend_direction = "improving"
            elif recent_avg > older_avg * 1.1:  # 10% degradation
                self.trend_direction = "degrading"
            else:
                self.trend_direction = "stable"
        
        # Also consider success rate trends
        if self.total_selections >= 50:
            recent_success_count = sum(1 for _ in range(min(20, self.successful_selections)))
            self.recent_success_rate = recent_success_count / 20.0
            
            # Adjust trend based on success rate
            success_trend = self.recent_success_rate - self.success_rate
            if abs(success_trend) > 0.1:  # Significant difference
                if success_trend > 0:
                    self.trend_direction = "improving"
                else:
                    self.trend_direction = "degrading"
    
    def get_composite_score(self) -> float:
        """Calculate composite performance score (0.0 - 1.0)."""
        weights = {
            'success_rate': 0.4,
            'quality': 0.2,
            'speed': 0.2,
            'cost_efficiency': 0.1,
            'stability': 0.1
        }
        
        # Success rate component
        success_component = self.success_rate
        
        # Quality component
        quality_component = self.avg_quality_score if self.avg_quality_score > 0 else 0.5
        
        # Speed component (inverse of completion time, normalized)
        if self.avg_completion_time_ms > 0:
            # Assume 10 seconds is baseline (1.0), scale accordingly
            baseline_ms = 10000
            speed_component = min(1.0, baseline_ms / self.avg_completion_time_ms)
        else:
            speed_component = 0.5
        
        # Cost efficiency component (inverse of cost, normalized)
        cost_component = 0.5  # Neutral if no cost data
        if self.avg_cost > 0:
            # Assume $0.01 per selection is baseline
            baseline_cost = 0.01
            cost_component = min(1.0, baseline_cost / self.avg_cost)
        
        # Stability component
        stability_component = self.stability_score
        
        # Weighted sum
        composite = (
            weights['success_rate'] * success_component +
            weights['quality'] * quality_component +
            weights['speed'] * speed_component +
            weights['cost_efficiency'] * cost_component +
            weights['stability'] * stability_component
        )
        
        return max(0.0, min(1.0, composite))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'option_name': self.option_name,
            'selection_type': self.selection_type,
            'total_selections': self.total_selections,
            'success_rate': self.success_rate,
            'avg_completion_time_ms': self.avg_completion_time_ms,
            'p95_completion_time_ms': self.p95_completion_time_ms,
            'avg_quality_score': self.avg_quality_score,
            'avg_cost': self.avg_cost,
            'error_rate': self.error_rate,
            'feedback_score': self.feedback_score,
            'trend_direction': self.trend_direction,
            'sample_confidence': self.sample_confidence,
            'stability_score': self.stability_score,
            'composite_score': self.get_composite_score(),
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


class PerformanceWindow(NamedTuple):
    """Time window for performance analysis."""
    start_time: datetime
    end_time: datetime
    metrics: Dict[str, SelectionMetrics]
    sample_count: int


class BenchmarkTracker:
    """
    Continuous performance monitoring system for selection options.
    
    Tracks real-time metrics and provides insights for optimization algorithms.
    """
    
    def __init__(self, 
                 selection_history: SelectionHistory,
                 update_interval_seconds: int = 60,
                 max_windows: int = 168):  # 7 days of hourly windows
        """
        Initialize benchmark tracker.
        
        Args:
            selection_history: Historical data storage
            update_interval_seconds: How often to update metrics
            max_windows: Maximum number of time windows to maintain
        """
        self.selection_history = selection_history
        self.update_interval_seconds = update_interval_seconds
        self.max_windows = max_windows
        
        # Real-time metrics by (selection_type, option_name)
        self.current_metrics: Dict[Tuple[str, str], SelectionMetrics] = {}
        self.metrics_lock = threading.RLock()
        
        # Historical performance windows
        self.performance_windows: deque = deque(maxlen=max_windows)
        self.windows_lock = threading.RLock()
        
        # Background update task
        self._update_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Performance tracking
        self.total_records_processed = 0
        self.last_update_time: Optional[datetime] = None
        
        logger.info("BenchmarkTracker initialized")
    
    async def start(self):
        """Start continuous monitoring."""
        if self._running:
            return
        
        self._running = True
        self._update_task = asyncio.create_task(self._background_update_loop())
        
        # Initialize with recent historical data
        await self._load_recent_data()
        
        logger.info("BenchmarkTracker started")
    
    async def stop(self):
        """Stop continuous monitoring."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("BenchmarkTracker stopped")
    
    def record_selection_outcome(self, record: SelectionRecord):
        """
        Record a new selection outcome and update metrics.
        
        Args:
            record: Complete selection record with outcome
        """
        try:
            key = (record.selection_type, record.selected_option)
            
            with self.metrics_lock:
                if key not in self.current_metrics:
                    self.current_metrics[key] = SelectionMetrics(
                        option_name=record.selected_option,
                        selection_type=record.selection_type
                    )
                
                self.current_metrics[key].update_from_record(record)
                self.total_records_processed += 1
            
            logger.debug(f"Updated metrics for {record.selection_type}:{record.selected_option}")
            
        except Exception as e:
            logger.error(f"Error recording selection outcome: {e}")
    
    def get_metrics(self, 
                   selection_type: str = None,
                   option_name: str = None) -> Dict[Tuple[str, str], SelectionMetrics]:
        """
        Get current metrics with optional filtering.
        
        Args:
            selection_type: Filter by selection type
            option_name: Filter by option name
            
        Returns:
            Dictionary mapping (selection_type, option_name) to metrics
        """
        with self.metrics_lock:
            if selection_type is None and option_name is None:
                return self.current_metrics.copy()
            
            filtered = {}
            for (sel_type, opt_name), metrics in self.current_metrics.items():
                if (selection_type is None or sel_type == selection_type) and \
                   (option_name is None or opt_name == option_name):
                    filtered[(sel_type, opt_name)] = metrics
            
            return filtered
    
    def get_rankings(self, 
                    selection_type: str,
                    metric: str = "composite_score",
                    min_samples: int = 10) -> List[Tuple[str, float]]:
        """
        Get ranked list of options by performance metric.
        
        Args:
            selection_type: Type of selection to rank
            metric: Metric to rank by
            min_samples: Minimum samples required for ranking
            
        Returns:
            List of (option_name, metric_value) tuples, sorted descending
        """
        candidates = []
        
        with self.metrics_lock:
            for (sel_type, option_name), metrics in self.current_metrics.items():
                if (sel_type == selection_type and 
                    metrics.total_selections >= min_samples):
                    
                    if metric == "composite_score":
                        value = metrics.get_composite_score()
                    elif metric == "success_rate":
                        value = metrics.success_rate
                    elif metric == "avg_completion_time_ms":
                        value = -metrics.avg_completion_time_ms  # Negative for ascending order
                    elif metric == "avg_quality_score":
                        value = metrics.avg_quality_score
                    elif metric == "avg_cost":
                        value = -metrics.avg_cost  # Negative for ascending order
                    else:
                        value = 0.0
                    
                    candidates.append((option_name, value))
        
        # Sort by value descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def get_performance_comparison(self, 
                                 selection_type: str,
                                 options: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compare performance across multiple options.
        
        Args:
            selection_type: Type of selection to compare
            options: List of option names to compare
            
        Returns:
            Dictionary mapping option names to their metrics
        """
        comparison = {}
        
        with self.metrics_lock:
            for option in options:
                key = (selection_type, option)
                if key in self.current_metrics:
                    metrics = self.current_metrics[key]
                    comparison[option] = {
                        'success_rate': metrics.success_rate,
                        'avg_completion_time_ms': metrics.avg_completion_time_ms,
                        'avg_quality_score': metrics.avg_quality_score,
                        'avg_cost': metrics.avg_cost,
                        'error_rate': metrics.error_rate,
                        'sample_confidence': metrics.sample_confidence,
                        'composite_score': metrics.get_composite_score(),
                        'total_selections': metrics.total_selections,
                        'trend_direction': metrics.trend_direction
                    }
                else:
                    # No data available
                    comparison[option] = {
                        'success_rate': 0.0,
                        'avg_completion_time_ms': 0.0,
                        'avg_quality_score': 0.0,
                        'avg_cost': 0.0,
                        'error_rate': 0.0,
                        'sample_confidence': 0.0,
                        'composite_score': 0.0,
                        'total_selections': 0,
                        'trend_direction': 'unknown'
                    }
        
        return comparison
    
    def detect_performance_degradation(self, 
                                     selection_type: str = None,
                                     threshold: float = 0.2) -> List[Dict[str, Any]]:
        """
        Detect options with significant performance degradation.
        
        Args:
            selection_type: Filter by selection type
            threshold: Minimum degradation threshold (0.0 - 1.0)
            
        Returns:
            List of degraded options with details
        """
        degraded = []
        
        with self.metrics_lock:
            for (sel_type, option_name), metrics in self.current_metrics.items():
                if selection_type and sel_type != selection_type:
                    continue
                
                if (metrics.trend_direction == "degrading" and
                    metrics.total_selections >= 50 and
                    metrics.sample_confidence > 0.5):
                    
                    # Calculate degradation severity
                    severity = "low"
                    if metrics.recent_success_rate < metrics.success_rate - threshold:
                        severity = "high"
                    elif metrics.recent_success_rate < metrics.success_rate - (threshold / 2):
                        severity = "medium"
                    
                    degraded.append({
                        'selection_type': sel_type,
                        'option_name': option_name,
                        'severity': severity,
                        'current_success_rate': metrics.success_rate,
                        'recent_success_rate': metrics.recent_success_rate,
                        'total_selections': metrics.total_selections,
                        'last_updated': metrics.last_updated
                    })
        
        return degraded
    
    def get_performance_windows(self, 
                              hours: int = 24) -> List[PerformanceWindow]:
        """
        Get historical performance windows.
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            List of performance windows
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self.windows_lock:
            return [
                window for window in self.performance_windows
                if window.start_time >= cutoff
            ]
    
    async def _background_update_loop(self):
        """Background task for periodic metric updates."""
        while self._running:
            try:
                await asyncio.sleep(self.update_interval_seconds)
                await self._update_metrics_from_history()
                self._create_performance_window()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background update loop: {e}")
                await asyncio.sleep(30)  # Back off on errors
    
    async def _load_recent_data(self):
        """Load recent historical data to initialize metrics."""
        try:
            # Load last 7 days of data
            since = datetime.now() - timedelta(days=7)
            records = self.selection_history.get_records(
                since=since,
                only_completed=True,
                limit=10000
            )
            
            logger.info(f"Loading {len(records)} recent records for initialization")
            
            for record in records:
                self.record_selection_outcome(record)
            
            self.last_update_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error loading recent data: {e}")
    
    async def _update_metrics_from_history(self):
        """Update metrics with new records from history."""
        try:
            # Get records since last update
            since = self.last_update_time or (datetime.now() - timedelta(hours=1))
            records = self.selection_history.get_records(
                since=since,
                only_completed=True,
                limit=1000
            )
            
            if records:
                logger.debug(f"Processing {len(records)} new records")
                
                for record in records:
                    self.record_selection_outcome(record)
                
                self.last_update_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating metrics from history: {e}")
    
    def _create_performance_window(self):
        """Create a performance window snapshot."""
        try:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.update_interval_seconds)
            
            with self.metrics_lock:
                window_metrics = {
                    key: {
                        'composite_score': metrics.get_composite_score(),
                        'success_rate': metrics.success_rate,
                        'total_selections': metrics.total_selections,
                        'avg_completion_time_ms': metrics.avg_completion_time_ms
                    }
                    for key, metrics in self.current_metrics.items()
                    if metrics.total_selections > 0
                }
            
            window = PerformanceWindow(
                start_time=window_start,
                end_time=now,
                metrics=window_metrics,
                sample_count=sum(m['total_selections'] for m in window_metrics.values())
            )
            
            with self.windows_lock:
                self.performance_windows.append(window)
            
        except Exception as e:
            logger.error(f"Error creating performance window: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the benchmark tracker."""
        with self.metrics_lock:
            total_options = len(self.current_metrics)
            total_selections = sum(m.total_selections for m in self.current_metrics.values())
            
            if self.current_metrics:
                avg_success_rate = statistics.mean(m.success_rate for m in self.current_metrics.values())
                avg_composite_score = statistics.mean(m.get_composite_score() for m in self.current_metrics.values())
            else:
                avg_success_rate = 0.0
                avg_composite_score = 0.0
            
        with self.windows_lock:
            total_windows = len(self.performance_windows)
        
        return {
            'total_options_tracked': total_options,
            'total_selections_processed': total_selections,
            'records_processed': self.total_records_processed,
            'avg_success_rate': avg_success_rate,
            'avg_composite_score': avg_composite_score,
            'performance_windows_stored': total_windows,
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'is_running': self._running
        }