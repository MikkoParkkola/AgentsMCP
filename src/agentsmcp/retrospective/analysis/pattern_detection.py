"""
Pattern Detection Engine

Advanced algorithms for detecting patterns in agent behavior, performance trends,
and user interactions to identify improvement opportunities with high accuracy.
"""

import asyncio
import logging
import statistics
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

from ..logging.log_schemas import BaseEvent, EventType, EventSeverity


class PatternType(Enum):
    """Types of patterns that can be detected."""
    PERFORMANCE_ANOMALY = "performance_anomaly"
    WORKFLOW_INEFFICIENCY = "workflow_inefficiency"
    USER_BEHAVIOR_PATTERN = "user_behavior_pattern"
    ERROR_CLUSTER = "error_cluster"
    RESOURCE_USAGE_PATTERN = "resource_usage_pattern"
    TEMPORAL_PATTERN = "temporal_pattern"
    SEQUENTIAL_PATTERN = "sequential_pattern"
    CORRELATION_PATTERN = "correlation_pattern"


class PatternSeverity(Enum):
    """Severity levels for detected patterns."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectedPattern:
    """A pattern detected in the execution logs."""
    pattern_id: str
    pattern_type: PatternType
    severity: PatternSeverity
    description: str
    confidence: float
    evidence: Dict[str, Any]
    affected_events: List[str]  # Event IDs
    time_range: Tuple[datetime, datetime]
    suggested_action: Optional[str] = None
    impact_estimate: Optional[str] = None
    frequency: int = 1


@dataclass
class PatternDetectionConfig:
    """Configuration for pattern detection algorithms."""
    accuracy_threshold: float = 0.95
    anomaly_detection_sensitivity: float = 2.0  # Standard deviations
    clustering_min_samples: int = 5
    temporal_window_minutes: int = 30
    correlation_threshold: float = 0.7
    enable_statistical_tests: bool = True
    enable_clustering: bool = True
    enable_time_series_analysis: bool = True
    parallel_processing: bool = True


class PatternDetector:
    """
    Advanced pattern detection engine using statistical analysis,
    machine learning clustering, and correlation analysis.
    """
    
    def __init__(self, config: Optional[PatternDetectionConfig] = None):
        self.config = config or PatternDetectionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Pattern detection state
        self._pattern_cache = {}
        self._event_features = {}
    
    async def detect_patterns(
        self,
        events: List[BaseEvent],
        accuracy_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect patterns in execution events with specified accuracy threshold.
        
        Args:
            events: List of events to analyze
            accuracy_threshold: Minimum accuracy for pattern detection
            
        Returns:
            Dictionary containing detected patterns and analysis metadata
        """
        threshold = accuracy_threshold or self.config.accuracy_threshold
        start_time = datetime.utcnow()
        
        if len(events) < 10:
            return {
                'patterns': [],
                'accuracy': 0.0,
                'confidence': 0.0,
                'events_analyzed': len(events),
                'processing_time_ms': 0,
                'message': 'Insufficient events for pattern detection'
            }
        
        try:
            self.logger.info(f"Starting pattern detection on {len(events)} events")
            
            # Extract features from events
            features = await self._extract_features(events)
            
            # Run different pattern detection algorithms in parallel
            detection_tasks = []
            
            if self.config.enable_statistical_tests:
                detection_tasks.append(self._detect_statistical_anomalies(events, features))
                
            if self.config.enable_clustering:
                detection_tasks.append(self._detect_clusters(events, features))
                
            if self.config.enable_time_series_analysis:
                detection_tasks.append(self._detect_temporal_patterns(events, features))
            
            # Sequential pattern detection
            detection_tasks.append(self._detect_sequential_patterns(events))
            
            # Correlation analysis
            detection_tasks.append(self._detect_correlations(events, features))
            
            # Execute all detection algorithms
            if self.config.parallel_processing:
                detection_results = await asyncio.gather(*detection_tasks, return_exceptions=True)
            else:
                detection_results = []
                for task in detection_tasks:
                    result = await task
                    detection_results.append(result)
            
            # Consolidate patterns from all algorithms
            all_patterns = []
            for result in detection_results:
                if isinstance(result, list):
                    all_patterns.extend(result)
                elif isinstance(result, Exception):
                    self.logger.warning(f"Pattern detection algorithm failed: {result}")
            
            # Filter patterns by confidence and accuracy
            filtered_patterns = [
                pattern for pattern in all_patterns
                if pattern.confidence >= threshold
            ]
            
            # Calculate overall accuracy
            accuracy = self._calculate_pattern_accuracy(filtered_patterns, events)
            
            processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            result = {
                'patterns': [self._pattern_to_dict(p) for p in filtered_patterns],
                'accuracy': accuracy,
                'confidence': statistics.mean([p.confidence for p in filtered_patterns]) if filtered_patterns else 0.0,
                'events_analyzed': len(events),
                'processing_time_ms': processing_time_ms,
                'pattern_types_detected': list(set(p.pattern_type.value for p in filtered_patterns)),
                'total_patterns_found': len(all_patterns),
                'patterns_after_filtering': len(filtered_patterns)
            }
            
            self.logger.info(
                f"Pattern detection complete: {len(filtered_patterns)} patterns found "
                f"(accuracy: {accuracy:.3f}) in {processing_time_ms}ms"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pattern detection failed: {e}")
            return {
                'patterns': [],
                'accuracy': 0.0,
                'confidence': 0.0,
                'events_analyzed': len(events),
                'processing_time_ms': 0,
                'error': str(e)
            }
    
    async def _extract_features(self, events: List[BaseEvent]) -> Dict[str, Any]:
        """Extract numerical features from events for analysis."""
        features = {
            'timestamps': [],
            'event_types': [],
            'severities': [],
            'response_times': [],
            'token_counts': [],
            'memory_usage': [],
            'error_rates': [],
            'session_ids': []
        }
        
        for event in events:
            # Convert timestamp to numerical features
            features['timestamps'].append(event.timestamp.timestamp())
            features['event_types'].append(event.event_type.value)
            features['severities'].append(event.severity.value if hasattr(event, 'severity') else 'info')
            features['session_ids'].append(event.session_id)
            
            # Extract performance metrics
            if hasattr(event, 'response_time_ms') and event.response_time_ms is not None:
                features['response_times'].append(event.response_time_ms)
            
            if hasattr(event, 'token_count') and event.token_count is not None:
                features['token_counts'].append(event.token_count)
                
            if hasattr(event, 'memory_mb') and event.memory_mb is not None:
                features['memory_usage'].append(event.memory_mb)
        
        # Calculate derived features
        features['event_intervals'] = []
        if len(features['timestamps']) > 1:
            for i in range(1, len(features['timestamps'])):
                interval = features['timestamps'][i] - features['timestamps'][i-1]
                features['event_intervals'].append(interval)
        
        return features
    
    async def _detect_statistical_anomalies(
        self, 
        events: List[BaseEvent], 
        features: Dict[str, Any]
    ) -> List[DetectedPattern]:
        """Detect statistical anomalies in performance metrics."""
        patterns = []
        
        # Response time anomalies
        if features['response_times']:
            anomalies = self._detect_outliers(
                features['response_times'],
                sensitivity=self.config.anomaly_detection_sensitivity
            )
            
            if anomalies['outliers']:
                pattern = DetectedPattern(
                    pattern_id=f"perf_anomaly_{datetime.utcnow().strftime('%H%M%S')}",
                    pattern_type=PatternType.PERFORMANCE_ANOMALY,
                    severity=self._classify_anomaly_severity(anomalies),
                    description=f"Response time anomalies detected: {len(anomalies['outliers'])} outliers",
                    confidence=0.85,
                    evidence={
                        'outlier_values': anomalies['outliers'],
                        'mean_response_time': anomalies['mean'],
                        'std_deviation': anomalies['std'],
                        'threshold': anomalies['threshold']
                    },
                    affected_events=[],  # Would need to map back to event IDs
                    time_range=(events[0].timestamp, events[-1].timestamp),
                    suggested_action="Investigate high response time causes",
                    impact_estimate="15-30% performance improvement potential"
                )
                patterns.append(pattern)
        
        # Token usage anomalies
        if features['token_counts']:
            anomalies = self._detect_outliers(features['token_counts'])
            
            if anomalies['outliers']:
                pattern = DetectedPattern(
                    pattern_id=f"token_anomaly_{datetime.utcnow().strftime('%H%M%S')}",
                    pattern_type=PatternType.RESOURCE_USAGE_PATTERN,
                    severity=PatternSeverity.MEDIUM,
                    description=f"Token usage anomalies: {len(anomalies['outliers'])} high-usage events",
                    confidence=0.80,
                    evidence=anomalies,
                    affected_events=[],
                    time_range=(events[0].timestamp, events[-1].timestamp),
                    suggested_action="Optimize context window usage",
                    impact_estimate="10-20% cost reduction potential"
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_clusters(
        self, 
        events: List[BaseEvent], 
        features: Dict[str, Any]
    ) -> List[DetectedPattern]:
        """Detect clusters of similar events using DBSCAN clustering."""
        patterns = []
        
        # Prepare numerical features for clustering
        numerical_features = []
        feature_names = []
        
        if features['response_times']:
            numerical_features.append(features['response_times'])
            feature_names.append('response_time')
            
        if features['token_counts']:
            # Pad or trim to match length
            token_counts = features['token_counts'][:len(features['response_times'])] if features['response_times'] else features['token_counts']
            numerical_features.append(token_counts)
            feature_names.append('token_count')
        
        if len(numerical_features) < 2 or len(numerical_features[0]) < self.config.clustering_min_samples:
            return patterns
        
        # Transpose and standardize features
        feature_matrix = np.array(numerical_features).T
        if feature_matrix.shape[0] < self.config.clustering_min_samples:
            return patterns
        
        try:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(
                eps=0.5,
                min_samples=self.config.clustering_min_samples
            )
            cluster_labels = clustering.fit_predict(scaled_features)
            
            # Analyze clusters
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                    
                cluster_indices = [i for i, l in enumerate(cluster_labels) if l == label]
                cluster_size = len(cluster_indices)
                
                if cluster_size >= self.config.clustering_min_samples:
                    # Calculate cluster statistics
                    cluster_data = feature_matrix[cluster_indices]
                    cluster_mean = np.mean(cluster_data, axis=0)
                    cluster_std = np.std(cluster_data, axis=0)
                    
                    pattern = DetectedPattern(
                        pattern_id=f"cluster_{label}_{datetime.utcnow().strftime('%H%M%S')}",
                        pattern_type=PatternType.USER_BEHAVIOR_PATTERN,
                        severity=PatternSeverity.MEDIUM,
                        description=f"Behavior cluster identified with {cluster_size} similar events",
                        confidence=0.75,
                        evidence={
                            'cluster_size': cluster_size,
                            'cluster_means': cluster_mean.tolist(),
                            'cluster_stds': cluster_std.tolist(),
                            'feature_names': feature_names
                        },
                        affected_events=[],
                        time_range=(events[0].timestamp, events[-1].timestamp),
                        suggested_action="Optimize common usage pattern",
                        frequency=cluster_size
                    )
                    patterns.append(pattern)
        
        except Exception as e:
            self.logger.warning(f"Clustering analysis failed: {e}")
        
        return patterns
    
    async def _detect_temporal_patterns(
        self, 
        events: List[BaseEvent], 
        features: Dict[str, Any]
    ) -> List[DetectedPattern]:
        """Detect time-based patterns and trends."""
        patterns = []
        
        if len(events) < 20:  # Need sufficient data for temporal analysis
            return patterns
        
        # Group events by time windows
        time_windows = self._group_events_by_time_windows(
            events, 
            window_size=timedelta(minutes=self.config.temporal_window_minutes)
        )
        
        # Analyze event frequency patterns
        window_counts = [len(window_events) for window_events in time_windows.values()]
        
        if len(window_counts) >= 3:
            # Detect trends using linear regression
            x = np.arange(len(window_counts))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, window_counts)
            
            if abs(r_value) > 0.7 and p_value < 0.05:  # Significant trend
                trend_type = "increasing" if slope > 0 else "decreasing"
                
                pattern = DetectedPattern(
                    pattern_id=f"temporal_trend_{datetime.utcnow().strftime('%H%M%S')}",
                    pattern_type=PatternType.TEMPORAL_PATTERN,
                    severity=PatternSeverity.MEDIUM,
                    description=f"Event frequency {trend_type} trend detected",
                    confidence=min(0.95, abs(r_value)),
                    evidence={
                        'slope': slope,
                        'correlation': r_value,
                        'p_value': p_value,
                        'trend_type': trend_type,
                        'window_counts': window_counts
                    },
                    affected_events=[],
                    time_range=(events[0].timestamp, events[-1].timestamp),
                    suggested_action=f"Investigate {trend_type} activity pattern"
                )
                patterns.append(pattern)
        
        # Detect periodic patterns
        if len(window_counts) >= 6:
            # Simple periodicity detection using autocorrelation
            autocorr = np.correlate(window_counts, window_counts, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Look for peaks that indicate periodicity
            if len(autocorr) > 2:
                max_autocorr_idx = np.argmax(autocorr[1:]) + 1  # Skip lag 0
                max_autocorr_value = autocorr[max_autocorr_idx]
                
                if max_autocorr_value > 0.6 * autocorr[0]:  # Strong periodicity
                    period = max_autocorr_idx * self.config.temporal_window_minutes
                    
                    pattern = DetectedPattern(
                        pattern_id=f"periodic_pattern_{datetime.utcnow().strftime('%H%M%S')}",
                        pattern_type=PatternType.TEMPORAL_PATTERN,
                        severity=PatternSeverity.LOW,
                        description=f"Periodic pattern detected with ~{period} minute cycle",
                        confidence=0.70,
                        evidence={
                            'period_minutes': period,
                            'autocorr_strength': max_autocorr_value / autocorr[0],
                            'autocorr_data': autocorr[:10].tolist()
                        },
                        affected_events=[],
                        time_range=(events[0].timestamp, events[-1].timestamp),
                        suggested_action="Consider optimizing for detected usage pattern"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_sequential_patterns(self, events: List[BaseEvent]) -> List[DetectedPattern]:
        """Detect sequential patterns in event types and workflows."""
        patterns = []
        
        # Extract event type sequences
        event_sequence = [event.event_type.value for event in events]
        
        # Find common subsequences using sliding window
        sequence_counts = defaultdict(int)
        window_size = 3
        
        for i in range(len(event_sequence) - window_size + 1):
            subsequence = tuple(event_sequence[i:i + window_size])
            sequence_counts[subsequence] += 1
        
        # Identify frequent patterns
        total_windows = len(event_sequence) - window_size + 1
        for sequence, count in sequence_counts.items():
            frequency = count / total_windows
            
            if count >= 3 and frequency > 0.1:  # At least 3 occurrences and 10% frequency
                pattern = DetectedPattern(
                    pattern_id=f"sequential_{hash(sequence)}",
                    pattern_type=PatternType.SEQUENTIAL_PATTERN,
                    severity=PatternSeverity.LOW,
                    description=f"Common sequence pattern: {' → '.join(sequence)}",
                    confidence=min(0.90, frequency * 2),
                    evidence={
                        'sequence': list(sequence),
                        'occurrences': count,
                        'frequency': frequency,
                        'total_windows': total_windows
                    },
                    affected_events=[],
                    time_range=(events[0].timestamp, events[-1].timestamp),
                    suggested_action="Consider optimizing common workflow pattern",
                    frequency=count
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_correlations(
        self, 
        events: List[BaseEvent], 
        features: Dict[str, Any]
    ) -> List[DetectedPattern]:
        """Detect correlations between different metrics and event types."""
        patterns = []
        
        # Correlation between response times and token counts
        if features['response_times'] and features['token_counts']:
            min_length = min(len(features['response_times']), len(features['token_counts']))
            if min_length >= 10:
                response_times = features['response_times'][:min_length]
                token_counts = features['token_counts'][:min_length]
                
                correlation, p_value = stats.pearsonr(response_times, token_counts)
                
                if abs(correlation) > self.config.correlation_threshold and p_value < 0.05:
                    correlation_type = "positive" if correlation > 0 else "negative"
                    
                    pattern = DetectedPattern(
                        pattern_id=f"correlation_rt_tokens_{datetime.utcnow().strftime('%H%M%S')}",
                        pattern_type=PatternType.CORRELATION_PATTERN,
                        severity=PatternSeverity.MEDIUM,
                        description=f"Strong {correlation_type} correlation between response time and token usage",
                        confidence=min(0.95, abs(correlation)),
                        evidence={
                            'correlation_coefficient': correlation,
                            'p_value': p_value,
                            'correlation_type': correlation_type,
                            'sample_size': min_length
                        },
                        affected_events=[],
                        time_range=(events[0].timestamp, events[-1].timestamp),
                        suggested_action="Optimize token usage to improve response times" if correlation > 0 else "Investigate negative correlation cause"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_outliers(
        self, 
        data: List[float], 
        sensitivity: float = 2.0
    ) -> Dict[str, Any]:
        """Detect outliers in numerical data using z-score method."""
        if not data or len(data) < 3:
            return {'outliers': [], 'mean': 0, 'std': 0, 'threshold': 0}
        
        mean_val = statistics.mean(data)
        std_val = statistics.stdev(data)
        
        if std_val == 0:
            return {'outliers': [], 'mean': mean_val, 'std': 0, 'threshold': 0}
        
        threshold = sensitivity * std_val
        outliers = [x for x in data if abs(x - mean_val) > threshold]
        
        return {
            'outliers': outliers,
            'mean': mean_val,
            'std': std_val,
            'threshold': threshold
        }
    
    def _classify_anomaly_severity(self, anomaly_data: Dict[str, Any]) -> PatternSeverity:
        """Classify the severity of detected anomalies."""
        outlier_ratio = len(anomaly_data['outliers']) / (len(anomaly_data['outliers']) + 10)  # Rough estimate
        
        if outlier_ratio > 0.2:
            return PatternSeverity.CRITICAL
        elif outlier_ratio > 0.1:
            return PatternSeverity.HIGH
        elif outlier_ratio > 0.05:
            return PatternSeverity.MEDIUM
        else:
            return PatternSeverity.LOW
    
    def _group_events_by_time_windows(
        self, 
        events: List[BaseEvent], 
        window_size: timedelta
    ) -> Dict[datetime, List[BaseEvent]]:
        """Group events into time windows for temporal analysis."""
        windows = defaultdict(list)
        
        if not events:
            return windows
        
        start_time = events[0].timestamp
        
        for event in events:
            # Calculate which window this event belongs to
            elapsed = event.timestamp - start_time
            window_index = int(elapsed.total_seconds() / window_size.total_seconds())
            window_start = start_time + (window_size * window_index)
            
            windows[window_start].append(event)
        
        return dict(windows)
    
    def _calculate_pattern_accuracy(
        self, 
        patterns: List[DetectedPattern], 
        events: List[BaseEvent]
    ) -> float:
        """Calculate overall accuracy of pattern detection."""
        if not patterns:
            return 0.0
        
        # Use confidence scores as accuracy proxy
        confidence_scores = [pattern.confidence for pattern in patterns]
        
        # Weight by pattern significance (frequency, severity)
        weighted_scores = []
        for pattern in patterns:
            weight = 1.0
            
            # Boost weight for high-severity patterns
            if pattern.severity == PatternSeverity.CRITICAL:
                weight *= 1.5
            elif pattern.severity == PatternSeverity.HIGH:
                weight *= 1.3
            
            # Boost weight for frequent patterns
            if pattern.frequency > 5:
                weight *= 1.2
            
            weighted_scores.append(pattern.confidence * weight)
        
        return min(0.99, statistics.mean(weighted_scores))
    
    def _pattern_to_dict(self, pattern: DetectedPattern) -> Dict[str, Any]:
        """Convert DetectedPattern to dictionary for serialization."""
        return {
            'pattern_id': pattern.pattern_id,
            'pattern_type': pattern.pattern_type.value,
            'severity': pattern.severity.value,
            'description': pattern.description,
            'confidence': pattern.confidence,
            'evidence': pattern.evidence,
            'affected_events_count': len(pattern.affected_events),
            'time_range': {
                'start': pattern.time_range[0].isoformat(),
                'end': pattern.time_range[1].isoformat()
            },
            'suggested_action': pattern.suggested_action,
            'impact_estimate': pattern.impact_estimate,
            'frequency': pattern.frequency
        }