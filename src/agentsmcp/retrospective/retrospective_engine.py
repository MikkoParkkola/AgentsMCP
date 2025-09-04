"""
Retrospective Engine - Main Orchestrator

The main orchestrator that coordinates the entire retrospective analysis process,
providing parallel processing capabilities and high-performance analysis at scale.
"""

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import multiprocessing as mp
import time

from .logging.log_schemas import BaseEvent
from .storage.log_store import LogStore
from .analysis.retrospective_analyzer import RetrospectiveAnalyzer, AnalysisConfiguration, AnalysisResult
from .analysis.pattern_detection import PatternDetector
from .analysis.bottleneck_identification import BottleneckIdentifier
from .analysis.analysis_reporter import AnalysisReporter
from .generation.improvement_generator import ImprovementGenerator, ImprovementOpportunity


@dataclass
class EngineConfiguration:
    """Configuration for the retrospective engine."""
    max_concurrent_sessions: int = 4
    max_concurrent_patterns: int = 2
    max_worker_processes: int = mp.cpu_count()
    max_worker_threads: int = mp.cpu_count() * 2
    analysis_timeout_seconds: int = 300  # 5 minutes
    enable_parallel_processing: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    batch_size: int = 100
    performance_target_events_per_second: int = 1000


class RetrospectiveEngine:
    """
    Main orchestrator for retrospective analysis providing high-performance
    analysis with parallel processing capabilities for scale.
    """
    
    def __init__(
        self,
        log_store: LogStore,
        config: Optional[EngineConfiguration] = None
    ):
        self.log_store = log_store
        self.config = config or EngineConfiguration()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.analyzer = RetrospectiveAnalyzer(log_store)
        self.pattern_detector = PatternDetector()
        self.bottleneck_identifier = BottleneckIdentifier()
        self.improvement_generator = ImprovementGenerator()
        self.reporter = AnalysisReporter()
        
        # Parallel processing resources
        self.process_pool = None
        self.thread_pool = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_sessions_analyzed': 0,
            'total_events_processed': 0,
            'avg_processing_time_ms': 0,
            'cache_hit_rate': 0.0
        }
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_timestamps = {}
        
    async def startup(self):
        """Initialize parallel processing resources."""
        if self.config.enable_parallel_processing:
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.config.max_worker_processes
            )
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.config.max_worker_threads
            )
            
        self.logger.info(
            f"RetrospectiveEngine started with {self.config.max_worker_processes} processes "
            f"and {self.config.max_worker_threads} threads"
        )
    
    async def shutdown(self):
        """Clean up parallel processing resources."""
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        await self.analyzer.cleanup()
        self.logger.info("RetrospectiveEngine shutdown completed")
    
    async def analyze_session_comprehensive(
        self,
        session_id: str,
        analysis_config: Optional[AnalysisConfiguration] = None,
        generate_improvements: bool = True,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive retrospective analysis for a session.
        
        Args:
            session_id: Session to analyze
            analysis_config: Optional custom analysis configuration
            generate_improvements: Whether to generate improvement opportunities
            generate_report: Whether to generate comprehensive report
            
        Returns:
            Complete analysis results including patterns, bottlenecks, improvements, and report
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting comprehensive analysis for session {session_id}")
            
            # Check cache first
            cache_key = f"{session_id}_{hash(str(analysis_config))}"
            if self.config.enable_caching and self._is_cache_valid(cache_key):
                self.logger.info(f"Returning cached analysis for session {session_id}")
                self.performance_metrics['cache_hit_rate'] += 1
                return self.analysis_cache[cache_key]
            
            # Step 1: Core analysis
            analysis_result = await self.analyzer.analyze_session(session_id, analysis_config)
            
            if analysis_result.events_processed == 0:
                return {
                    'status': 'no_data',
                    'message': 'No events found for analysis',
                    'session_id': session_id
                }
            
            # Step 2: Parallel pattern and bottleneck detection
            if self.config.enable_parallel_processing:
                pattern_task = asyncio.create_task(
                    self._detect_patterns_parallel(analysis_result)
                )
                bottleneck_task = asyncio.create_task(
                    self._identify_bottlenecks_parallel(analysis_result)
                )
                
                patterns, bottlenecks = await asyncio.gather(
                    pattern_task, bottleneck_task, return_exceptions=True
                )
                
                # Handle exceptions
                if isinstance(patterns, Exception):
                    self.logger.error(f"Pattern detection failed: {patterns}")
                    patterns = []
                
                if isinstance(bottlenecks, Exception):
                    self.logger.error(f"Bottleneck identification failed: {bottlenecks}")
                    bottlenecks = []
            else:
                # Sequential processing fallback
                patterns = await self._detect_patterns_sequential(analysis_result)
                bottlenecks = await self._identify_bottlenecks_sequential(analysis_result)
            
            # Step 3: Generate improvements if requested
            improvements = []
            if generate_improvements and (patterns or bottlenecks):
                improvements = await self.improvement_generator.generate_improvements(
                    analysis_result, patterns, bottlenecks
                )
            
            # Step 4: Generate comprehensive report if requested
            report = None
            if generate_report:
                report = await self.reporter.generate_comprehensive_report(
                    analysis_result, patterns, bottlenecks, improvements
                )
            
            # Compile comprehensive result
            result = {
                'status': 'success',
                'session_id': session_id,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'analysis_result': analysis_result.to_dict(),
                'patterns_detected': [self._pattern_to_dict(p) for p in patterns],
                'bottlenecks_identified': [self._bottleneck_to_dict(b) for b in bottlenecks],
                'improvement_opportunities': [self._improvement_to_dict(i) for i in improvements],
                'comprehensive_report': report,
                'performance_metrics': {
                    'processing_time_ms': int((time.time() - start_time) * 1000),
                    'events_processed': analysis_result.events_processed,
                    'patterns_found': len(patterns),
                    'bottlenecks_found': len(bottlenecks),
                    'improvements_generated': len(improvements)
                }
            }
            
            # Cache result if caching enabled
            if self.config.enable_caching:
                self.analysis_cache[cache_key] = result
                self.cache_timestamps[cache_key] = datetime.utcnow()
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            self.logger.info(
                f"Comprehensive analysis completed for session {session_id}: "
                f"{len(patterns)} patterns, {len(bottlenecks)} bottlenecks, "
                f"{len(improvements)} improvements in {result['performance_metrics']['processing_time_ms']}ms"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed for session {session_id}: {e}")
            return {
                'status': 'error',
                'session_id': session_id,
                'error': str(e),
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }
    
    async def analyze_multiple_sessions(
        self,
        session_ids: List[str],
        analysis_config: Optional[AnalysisConfiguration] = None,
        max_concurrent: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple sessions in parallel with controlled concurrency.
        
        Args:
            session_ids: List of session IDs to analyze
            analysis_config: Analysis configuration to use for all sessions
            max_concurrent: Maximum concurrent analyses (default from config)
            
        Returns:
            List of analysis results for all sessions
        """
        max_concurrent = max_concurrent or self.config.max_concurrent_sessions
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(session_id: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.analyze_session_comprehensive(session_id, analysis_config)
        
        self.logger.info(f"Analyzing {len(session_ids)} sessions with max concurrency {max_concurrent}")
        
        tasks = [analyze_with_semaphore(session_id) for session_id in session_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    'status': 'error',
                    'session_id': session_ids[i],
                    'error': str(result)
                })
            else:
                final_results.append(result)
        
        return final_results
    
    async def analyze_time_range_batch(
        self,
        start_time: datetime,
        end_time: datetime,
        batch_size: Optional[int] = None,
        analysis_config: Optional[AnalysisConfiguration] = None
    ) -> Dict[str, Any]:
        """
        Analyze events in a time range using batch processing for optimal performance.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            batch_size: Size of batches for processing (default from config)
            analysis_config: Analysis configuration
            
        Returns:
            Aggregated analysis results for the time range
        """
        batch_size = batch_size or self.config.batch_size
        
        self.logger.info(f"Batch analyzing time range {start_time} to {end_time}")
        
        # Fetch all events in time range
        events = await self.log_store.query_events(
            start_time=start_time,
            end_time=end_time
        )
        
        if len(events) < 10:
            return {
                'status': 'insufficient_data',
                'message': f'Only {len(events)} events found in time range',
                'time_range': {'start': start_time.isoformat(), 'end': end_time.isoformat()}
            }
        
        # Group events into batches by session
        session_events = {}
        for event in events:
            if event.session_id not in session_events:
                session_events[event.session_id] = []
            session_events[event.session_id].append(event)
        
        # Create batches of sessions
        session_ids = list(session_events.keys())
        batches = [session_ids[i:i + batch_size] for i in range(0, len(session_ids), batch_size)]
        
        # Process batches in parallel
        batch_results = []
        for batch in batches:
            batch_analysis = await self.analyze_multiple_sessions(batch, analysis_config)
            batch_results.extend(batch_analysis)
        
        # Aggregate results across all batches
        aggregated_result = await self._aggregate_batch_results(
            batch_results, start_time, end_time
        )
        
        return aggregated_result
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for the engine."""
        current_time = datetime.utcnow()
        
        # Calculate cache statistics
        valid_cache_entries = sum(
            1 for timestamp in self.cache_timestamps.values()
            if (current_time - timestamp).total_seconds() < self.config.cache_ttl_seconds
        )
        
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'configuration': {
                'max_worker_processes': self.config.max_worker_processes,
                'max_worker_threads': self.config.max_worker_threads,
                'max_concurrent_sessions': self.config.max_concurrent_sessions,
                'performance_target_events_per_second': self.config.performance_target_events_per_second
            },
            'cache_statistics': {
                'enabled': self.config.enable_caching,
                'total_entries': len(self.analysis_cache),
                'valid_entries': valid_cache_entries,
                'hit_rate_percent': self.performance_metrics.get('cache_hit_rate', 0) * 100
            },
            'resource_utilization': {
                'process_pool_active': self.process_pool is not None,
                'thread_pool_active': self.thread_pool is not None
            }
        }
    
    async def _detect_patterns_parallel(self, analysis_result: AnalysisResult) -> List:
        """Detect patterns using parallel processing."""
        # In a real implementation, this would use the process pool for CPU-intensive pattern detection
        # For now, use the existing pattern detector
        events = await self._get_session_events(analysis_result.session_id)
        pattern_results = await self.pattern_detector.detect_patterns(
            events, analysis_result.configuration.pattern_accuracy_threshold
        )
        
        # Convert pattern results to DetectedPattern objects
        # This is simplified - would need proper conversion logic
        return pattern_results.get('patterns', [])
    
    async def _identify_bottlenecks_parallel(self, analysis_result: AnalysisResult) -> List:
        """Identify bottlenecks using parallel processing."""
        events = await self._get_session_events(analysis_result.session_id)
        bottlenecks = await self.bottleneck_identifier.identify_bottlenecks(events)
        return bottlenecks
    
    async def _detect_patterns_sequential(self, analysis_result: AnalysisResult) -> List:
        """Fallback sequential pattern detection."""
        return await self._detect_patterns_parallel(analysis_result)
    
    async def _identify_bottlenecks_sequential(self, analysis_result: AnalysisResult) -> List:
        """Fallback sequential bottleneck identification."""
        return await self._identify_bottlenecks_parallel(analysis_result)
    
    async def _get_session_events(self, session_id: str) -> List[BaseEvent]:
        """Get events for a specific session."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)  # Default to 24 hours
        
        events = await self.log_store.query_events(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return events
    
    async def _aggregate_batch_results(
        self,
        batch_results: List[Dict[str, Any]],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Aggregate results from batch processing."""
        successful_results = [r for r in batch_results if r.get('status') == 'success']
        
        if not successful_results:
            return {
                'status': 'no_successful_analyses',
                'message': f'No successful analyses in {len(batch_results)} attempts',
                'time_range': {'start': start_time.isoformat(), 'end': end_time.isoformat()}
            }
        
        # Aggregate metrics
        total_events = sum(r['performance_metrics']['events_processed'] for r in successful_results)
        total_patterns = sum(r['performance_metrics']['patterns_found'] for r in successful_results)
        total_bottlenecks = sum(r['performance_metrics']['bottlenecks_found'] for r in successful_results)
        total_improvements = sum(r['performance_metrics']['improvements_generated'] for r in successful_results)
        avg_processing_time = sum(r['performance_metrics']['processing_time_ms'] for r in successful_results) / len(successful_results)
        
        # Collect unique patterns and bottlenecks
        all_patterns = []
        all_bottlenecks = []
        all_improvements = []
        
        for result in successful_results:
            all_patterns.extend(result.get('patterns_detected', []))
            all_bottlenecks.extend(result.get('bottlenecks_identified', []))
            all_improvements.extend(result.get('improvement_opportunities', []))
        
        return {
            'status': 'success',
            'time_range': {'start': start_time.isoformat(), 'end': end_time.isoformat()},
            'sessions_analyzed': len(successful_results),
            'sessions_failed': len(batch_results) - len(successful_results),
            'aggregated_metrics': {
                'total_events_processed': total_events,
                'total_patterns_found': total_patterns,
                'total_bottlenecks_found': total_bottlenecks,
                'total_improvements_generated': total_improvements,
                'average_processing_time_ms': avg_processing_time
            },
            'unique_patterns': len(set(p.get('pattern_id', '') for p in all_patterns)),
            'unique_bottlenecks': len(set(b.get('bottleneck_id', '') for b in all_bottlenecks)),
            'top_improvement_opportunities': sorted(
                all_improvements,
                key=lambda x: x.get('priority_score', 0),
                reverse=True
            )[:10],
            'performance_summary': {
                'events_per_second': total_events / max(avg_processing_time / 1000, 0.001),
                'target_events_per_second': self.config.performance_target_events_per_second,
                'performance_ratio': (total_events / max(avg_processing_time / 1000, 0.001)) / self.config.performance_target_events_per_second
            }
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid."""
        if cache_key not in self.cache_timestamps:
            return False
        
        age = (datetime.utcnow() - self.cache_timestamps[cache_key]).total_seconds()
        return age < self.config.cache_ttl_seconds
    
    def _update_performance_metrics(self, result: Dict[str, Any]):
        """Update performance metrics with result data."""
        self.performance_metrics['total_sessions_analyzed'] += 1
        self.performance_metrics['total_events_processed'] += result['performance_metrics']['events_processed']
        
        # Update average processing time
        current_avg = self.performance_metrics['avg_processing_time_ms']
        new_time = result['performance_metrics']['processing_time_ms']
        sessions_count = self.performance_metrics['total_sessions_analyzed']
        
        self.performance_metrics['avg_processing_time_ms'] = (
            (current_avg * (sessions_count - 1) + new_time) / sessions_count
        )
    
    def _pattern_to_dict(self, pattern) -> Dict[str, Any]:
        """Convert pattern to dictionary for serialization."""
        if hasattr(pattern, 'to_dict'):
            return pattern.to_dict()
        return pattern if isinstance(pattern, dict) else {}
    
    def _bottleneck_to_dict(self, bottleneck) -> Dict[str, Any]:
        """Convert bottleneck to dictionary for serialization."""
        if hasattr(bottleneck, '__dict__'):
            return {
                'bottleneck_id': bottleneck.bottleneck_id,
                'type': bottleneck.bottleneck_type.value,
                'severity': bottleneck.severity.value,
                'description': bottleneck.description,
                'location': bottleneck.location,
                'impact_metrics': bottleneck.impact_metrics,
                'root_cause': bottleneck.root_cause,
                'suggested_remediation': bottleneck.suggested_remediation,
                'confidence': bottleneck.confidence
            }
        return bottleneck if isinstance(bottleneck, dict) else {}
    
    def _improvement_to_dict(self, improvement) -> Dict[str, Any]:
        """Convert improvement to dictionary for serialization."""
        if hasattr(improvement, '__dict__'):
            return {
                'opportunity_id': improvement.opportunity_id,
                'title': improvement.title,
                'description': improvement.description,
                'type': improvement.improvement_type.value,
                'effort': improvement.effort.value,
                'risk': improvement.risk.value,
                'expected_benefits': improvement.expected_benefits,
                'confidence': improvement.confidence,
                'priority_score': improvement.priority_score,
                'implementation_steps': improvement.implementation_steps,
                'success_metrics': improvement.success_metrics
            }
        return improvement if isinstance(improvement, dict) else {}


# Utility function for easy engine setup
async def create_retrospective_engine(
    log_store: LogStore,
    config: Optional[EngineConfiguration] = None
) -> RetrospectiveEngine:
    """Create and initialize a retrospective engine."""
    engine = RetrospectiveEngine(log_store, config)
    await engine.startup()
    return engine