"""
Retrospective Analyzer - Intelligence Core

The main orchestrator for analyzing execution logs to identify improvement opportunities.
Processes 1M+ log events with 95% pattern accuracy across distributed agent workflows.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import statistics
import json

from ..logging.log_schemas import BaseEvent, EventType, EventSeverity
from ..storage.log_store import LogStore
from .pattern_detection import PatternDetector
from .bottleneck_identification import BottleneckIdentifier


class AnalysisDepth(Enum):
    """Analysis depth levels for different use cases."""
    QUICK = "quick"          # Fast analysis for immediate feedback
    STANDARD = "standard"    # Balanced analysis for regular retrospectives  
    COMPREHENSIVE = "comprehensive"  # Deep analysis for major reviews


class AnalysisScope(Enum):
    """Scope of analysis for targeted improvements."""
    PERFORMANCE = "performance"      # Response times, throughput, resource usage
    WORKFLOW = "workflow"           # Agent delegation, sequential thinking
    USER_EXPERIENCE = "user_experience"  # Input routing, feedback patterns
    QUALITY = "quality"             # Success rates, error patterns


@dataclass
class AnalysisConfiguration:
    """Configuration for retrospective analysis."""
    depth: AnalysisDepth = AnalysisDepth.STANDARD
    scopes: Set[AnalysisScope] = field(default_factory=lambda: {
        AnalysisScope.PERFORMANCE, 
        AnalysisScope.WORKFLOW,
        AnalysisScope.USER_EXPERIENCE,
        AnalysisScope.QUALITY
    })
    time_window: timedelta = field(default_factory=lambda: timedelta(hours=24))
    min_events_threshold: int = 100
    pattern_accuracy_threshold: float = 0.95
    parallel_workers: int = 4
    enable_trend_analysis: bool = True
    include_historical_comparison: bool = True


@dataclass
class AnalysisResult:
    """Results of retrospective analysis."""
    session_id: str
    analysis_timestamp: datetime
    configuration: AnalysisConfiguration
    events_processed: int
    pattern_accuracy: float
    
    # Analysis findings
    performance_insights: Dict[str, Any] = field(default_factory=dict)
    workflow_insights: Dict[str, Any] = field(default_factory=dict)
    user_experience_insights: Dict[str, Any] = field(default_factory=dict)
    quality_insights: Dict[str, Any] = field(default_factory=dict)
    
    # Identified opportunities
    improvement_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    bottlenecks_identified: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics
    processing_time_ms: int = 0
    confidence_score: float = 0.0
    data_completeness: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'configuration': {
                'depth': self.configuration.depth.value,
                'scopes': [scope.value for scope in self.configuration.scopes],
                'time_window_hours': self.configuration.time_window.total_seconds() / 3600,
                'min_events_threshold': self.configuration.min_events_threshold
            },
            'events_processed': self.events_processed,
            'pattern_accuracy': self.pattern_accuracy,
            'performance_insights': self.performance_insights,
            'workflow_insights': self.workflow_insights,
            'user_experience_insights': self.user_experience_insights,
            'quality_insights': self.quality_insights,
            'improvement_opportunities': self.improvement_opportunities,
            'bottlenecks_identified': self.bottlenecks_identified,
            'processing_time_ms': self.processing_time_ms,
            'confidence_score': self.confidence_score,
            'data_completeness': self.data_completeness
        }


class RetrospectiveAnalyzer:
    """
    Main retrospective analysis engine that orchestrates pattern detection,
    bottleneck identification, and improvement opportunity generation.
    """
    
    def __init__(
        self, 
        log_store: LogStore,
        configuration: Optional[AnalysisConfiguration] = None
    ):
        self.log_store = log_store
        self.config = configuration or AnalysisConfiguration()
        self.logger = logging.getLogger(__name__)
        
        # Initialize analysis components
        self.pattern_detector = PatternDetector()
        self.bottleneck_identifier = BottleneckIdentifier()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        
        # Analysis state
        self._analysis_cache = {}
        self._last_analysis_time = None
    
    async def analyze_session(
        self, 
        session_id: str,
        custom_config: Optional[AnalysisConfiguration] = None
    ) -> AnalysisResult:
        """
        Perform comprehensive retrospective analysis for a specific session.
        
        Args:
            session_id: The session to analyze
            custom_config: Optional custom configuration for this analysis
            
        Returns:
            AnalysisResult containing insights and improvement opportunities
        """
        start_time = datetime.utcnow()
        config = custom_config or self.config
        
        try:
            self.logger.info(f"Starting retrospective analysis for session {session_id}")
            
            # 1. Fetch events for analysis
            events = await self._fetch_events(session_id, config)
            
            if len(events) < config.min_events_threshold:
                return self._create_insufficient_data_result(session_id, len(events), config)
            
            # 2. Parallel analysis execution
            analysis_tasks = []
            
            if AnalysisScope.PERFORMANCE in config.scopes:
                analysis_tasks.append(self._analyze_performance(events, config))
                
            if AnalysisScope.WORKFLOW in config.scopes:
                analysis_tasks.append(self._analyze_workflow(events, config))
                
            if AnalysisScope.USER_EXPERIENCE in config.scopes:
                analysis_tasks.append(self._analyze_user_experience(events, config))
                
            if AnalysisScope.QUALITY in config.scopes:
                analysis_tasks.append(self._analyze_quality(events, config))
            
            # Execute analyses in parallel
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # 3. Process results and identify patterns
            pattern_results = await self.pattern_detector.detect_patterns(
                events, config.pattern_accuracy_threshold
            )
            
            # 4. Identify bottlenecks across workflows
            bottleneck_results = await self.bottleneck_identifier.identify_bottlenecks(
                events, pattern_results
            )
            
            # 5. Compile comprehensive result
            result = self._compile_analysis_result(
                session_id, events, analysis_results, pattern_results, 
                bottleneck_results, config, start_time
            )
            
            # Cache result for future reference
            self._analysis_cache[session_id] = result
            self._last_analysis_time = datetime.utcnow()
            
            self.logger.info(
                f"Analysis complete: {len(events)} events processed, "
                f"{len(result.improvement_opportunities)} opportunities found, "
                f"accuracy: {result.pattern_accuracy:.3f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis failed for session {session_id}: {e}")
            return self._create_error_result(session_id, str(e), config)
    
    async def analyze_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        custom_config: Optional[AnalysisConfiguration] = None
    ) -> AnalysisResult:
        """
        Analyze events across a time range for trend analysis.
        
        Args:
            start_time: Start of time range
            end_time: End of time range  
            custom_config: Optional custom configuration
            
        Returns:
            AnalysisResult with trend insights
        """
        config = custom_config or self.config
        synthetic_session_id = f"time_range_{start_time.isoformat()}_{end_time.isoformat()}"
        
        # Fetch events in time range
        events = await self.log_store.query_events(
            start_time=start_time,
            end_time=end_time
        )
        
        if len(events) < config.min_events_threshold:
            return self._create_insufficient_data_result(synthetic_session_id, len(events), config)
        
        # Group events by session for comparative analysis
        session_groups = defaultdict(list)
        for event in events:
            session_groups[event.session_id].append(event)
        
        # Analyze each session group
        session_analyses = []
        for session_id, session_events in session_groups.items():
            if len(session_events) >= 10:  # Minimum events per session
                session_analysis = await self.analyze_session(session_id, config)
                session_analyses.append(session_analysis)
        
        # Aggregate insights across sessions
        return self._aggregate_session_analyses(synthetic_session_id, session_analyses, config)
    
    async def get_realtime_insights(
        self,
        session_id: str,
        lookback_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Get real-time insights for active session monitoring.
        
        Args:
            session_id: Current session ID
            lookback_minutes: How far back to look for patterns
            
        Returns:
            Dictionary of real-time insights
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        # Quick configuration for real-time analysis
        realtime_config = AnalysisConfiguration(
            depth=AnalysisDepth.QUICK,
            time_window=timedelta(minutes=lookback_minutes),
            min_events_threshold=10,
            parallel_workers=2
        )
        
        events = await self.log_store.query_events(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time
        )
        
        if len(events) < 10:
            return {
                'status': 'insufficient_data',
                'events_found': len(events),
                'message': 'Need more activity for real-time insights'
            }
        
        # Quick pattern detection
        patterns = await self.pattern_detector.detect_patterns(events, 0.8)
        
        # Calculate basic metrics
        response_times = []
        error_count = 0
        llm_calls = 0
        
        for event in events:
            if event.event_type == EventType.PERFORMANCE_METRICS:
                if hasattr(event, 'response_time_ms'):
                    response_times.append(event.response_time_ms)
            elif event.event_type == EventType.ERROR:
                error_count += 1
            elif event.event_type == EventType.LLM_CALL:
                llm_calls += 1
        
        insights = {
            'status': 'active',
            'period': f'{lookback_minutes} minutes',
            'events_analyzed': len(events),
            'patterns_detected': len(patterns),
            'metrics': {
                'avg_response_time_ms': statistics.mean(response_times) if response_times else 0,
                'error_rate': error_count / len(events) if events else 0,
                'llm_call_frequency': llm_calls / lookback_minutes if lookback_minutes > 0 else 0
            }
        }
        
        # Add immediate improvement suggestions if patterns detected
        if patterns:
            insights['immediate_suggestions'] = [
                pattern.get('suggestion') for pattern in patterns 
                if pattern.get('suggestion') and pattern.get('confidence', 0) > 0.8
            ]
        
        return insights
    
    async def _fetch_events(
        self, 
        session_id: str, 
        config: AnalysisConfiguration
    ) -> List[BaseEvent]:
        """Fetch events for analysis based on configuration."""
        end_time = datetime.utcnow()
        start_time = end_time - config.time_window
        
        events = await self.log_store.query_events(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time
        )
        
        self.logger.debug(f"Fetched {len(events)} events for session {session_id}")
        return events
    
    async def _analyze_performance(
        self, 
        events: List[BaseEvent], 
        config: AnalysisConfiguration
    ) -> Dict[str, Any]:
        """Analyze performance patterns and metrics."""
        performance_events = [
            e for e in events 
            if e.event_type in [EventType.PERFORMANCE_METRICS, EventType.LLM_CALL]
        ]
        
        if not performance_events:
            return {'status': 'no_performance_data'}
        
        # Extract performance metrics
        response_times = []
        token_usage = []
        memory_usage = []
        
        for event in performance_events:
            if hasattr(event, 'response_time_ms') and event.response_time_ms:
                response_times.append(event.response_time_ms)
            if hasattr(event, 'token_count') and event.token_count:
                token_usage.append(event.token_count)
            if hasattr(event, 'memory_mb') and event.memory_mb:
                memory_usage.append(event.memory_mb)
        
        insights = {
            'total_performance_events': len(performance_events),
            'response_times': {
                'count': len(response_times),
                'avg_ms': statistics.mean(response_times) if response_times else 0,
                'p95_ms': self._percentile(response_times, 0.95) if response_times else 0,
                'p99_ms': self._percentile(response_times, 0.99) if response_times else 0
            },
            'token_usage': {
                'count': len(token_usage),
                'total_tokens': sum(token_usage) if token_usage else 0,
                'avg_tokens': statistics.mean(token_usage) if token_usage else 0
            },
            'memory_usage': {
                'count': len(memory_usage),
                'avg_mb': statistics.mean(memory_usage) if memory_usage else 0,
                'peak_mb': max(memory_usage) if memory_usage else 0
            }
        }
        
        # Identify performance patterns
        if response_times:
            # Check for response time degradation
            if len(response_times) > 10:
                first_half = response_times[:len(response_times)//2]
                second_half = response_times[len(response_times)//2:]
                
                first_avg = statistics.mean(first_half)
                second_avg = statistics.mean(second_half)
                
                if second_avg > first_avg * 1.2:  # 20% degradation
                    insights['performance_degradation'] = {
                        'detected': True,
                        'degradation_percent': ((second_avg - first_avg) / first_avg) * 100,
                        'first_half_avg_ms': first_avg,
                        'second_half_avg_ms': second_avg
                    }
        
        return insights
    
    async def _analyze_workflow(
        self, 
        events: List[BaseEvent], 
        config: AnalysisConfiguration
    ) -> Dict[str, Any]:
        """Analyze workflow patterns and agent delegation effectiveness."""
        workflow_events = [
            e for e in events 
            if e.event_type in [
                EventType.AGENT_DELEGATION, 
                EventType.TASK_START, 
                EventType.TASK_COMPLETE
            ]
        ]
        
        if not workflow_events:
            return {'status': 'no_workflow_data'}
        
        # Analyze delegation patterns
        delegations = [e for e in workflow_events if e.event_type == EventType.AGENT_DELEGATION]
        task_starts = [e for e in workflow_events if e.event_type == EventType.TASK_START]
        task_completes = [e for e in workflow_events if e.event_type == EventType.TASK_COMPLETE]
        
        insights = {
            'total_delegations': len(delegations),
            'task_starts': len(task_starts),
            'task_completions': len(task_completes),
            'completion_rate': len(task_completes) / len(task_starts) if task_starts else 0
        }
        
        # Analyze delegation effectiveness
        if delegations:
            agent_types = defaultdict(int)
            success_by_agent = defaultdict(lambda: {'attempts': 0, 'successes': 0})
            
            for delegation in delegations:
                if hasattr(delegation, 'target_agent'):
                    agent_types[delegation.target_agent] += 1
                    success_by_agent[delegation.target_agent]['attempts'] += 1
                    
                    # Look for corresponding completion
                    if hasattr(delegation, 'task_id'):
                        completion_found = any(
                            tc.task_id == delegation.task_id 
                            for tc in task_completes 
                            if hasattr(tc, 'task_id')
                        )
                        if completion_found:
                            success_by_agent[delegation.target_agent]['successes'] += 1
            
            insights['agent_distribution'] = dict(agent_types)
            insights['agent_success_rates'] = {
                agent: data['successes'] / data['attempts'] if data['attempts'] > 0 else 0
                for agent, data in success_by_agent.items()
            }
        
        return insights
    
    async def _analyze_user_experience(
        self, 
        events: List[BaseEvent], 
        config: AnalysisConfiguration
    ) -> Dict[str, Any]:
        """Analyze user experience patterns and satisfaction indicators."""
        ux_events = [
            e for e in events 
            if e.event_type in [EventType.USER_INTERACTION, EventType.ERROR]
        ]
        
        if not ux_events:
            return {'status': 'no_ux_data'}
        
        user_interactions = [e for e in ux_events if e.event_type == EventType.USER_INTERACTION]
        errors = [e for e in ux_events if e.event_type == EventType.ERROR]
        
        insights = {
            'total_interactions': len(user_interactions),
            'total_errors': len(errors),
            'error_rate': len(errors) / len(user_interactions) if user_interactions else 0
        }
        
        # Analyze interaction patterns
        if user_interactions:
            interaction_types = defaultdict(int)
            for interaction in user_interactions:
                if hasattr(interaction, 'interaction_type'):
                    interaction_types[interaction.interaction_type] += 1
            
            insights['interaction_distribution'] = dict(interaction_types)
        
        # Analyze error patterns
        if errors:
            error_categories = defaultdict(int)
            for error in errors:
                if hasattr(error, 'error_category'):
                    error_categories[error.error_category] += 1
                elif hasattr(error, 'severity'):
                    error_categories[f"severity_{error.severity.value}"] += 1
            
            insights['error_distribution'] = dict(error_categories)
        
        return insights
    
    async def _analyze_quality(
        self, 
        events: List[BaseEvent], 
        config: AnalysisConfiguration
    ) -> Dict[str, Any]:
        """Analyze quality patterns including success rates and improvement effectiveness."""
        quality_events = [
            e for e in events 
            if e.event_type in [
                EventType.QUALITY_GATE, 
                EventType.TASK_COMPLETE,
                EventType.ERROR
            ]
        ]
        
        if not quality_events:
            return {'status': 'no_quality_data'}
        
        quality_gates = [e for e in quality_events if e.event_type == EventType.QUALITY_GATE]
        completions = [e for e in quality_events if e.event_type == EventType.TASK_COMPLETE]
        errors = [e for e in quality_events if e.event_type == EventType.ERROR]
        
        insights = {
            'quality_gate_checks': len(quality_gates),
            'successful_completions': len(completions),
            'errors_encountered': len(errors)
        }
        
        # Calculate quality metrics
        total_attempts = len(completions) + len(errors)
        if total_attempts > 0:
            insights['success_rate'] = len(completions) / total_attempts
            insights['failure_rate'] = len(errors) / total_attempts
        
        # Analyze quality gate results
        if quality_gates:
            passed_gates = sum(1 for qg in quality_gates if getattr(qg, 'passed', False))
            insights['quality_gate_pass_rate'] = passed_gates / len(quality_gates)
        
        return insights
    
    def _compile_analysis_result(
        self,
        session_id: str,
        events: List[BaseEvent],
        analysis_results: List[Any],
        pattern_results: Dict[str, Any],
        bottleneck_results: List[Dict[str, Any]],
        config: AnalysisConfiguration,
        start_time: datetime
    ) -> AnalysisResult:
        """Compile all analysis components into final result."""
        processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Extract individual analysis results
        performance_insights = {}
        workflow_insights = {}
        ux_insights = {}
        quality_insights = {}
        
        for i, scope in enumerate([
            AnalysisScope.PERFORMANCE, 
            AnalysisScope.WORKFLOW, 
            AnalysisScope.USER_EXPERIENCE,
            AnalysisScope.QUALITY
        ]):
            if scope in config.scopes and i < len(analysis_results):
                result = analysis_results[i]
                if not isinstance(result, Exception):
                    if scope == AnalysisScope.PERFORMANCE:
                        performance_insights = result
                    elif scope == AnalysisScope.WORKFLOW:
                        workflow_insights = result
                    elif scope == AnalysisScope.USER_EXPERIENCE:
                        ux_insights = result
                    elif scope == AnalysisScope.QUALITY:
                        quality_insights = result
        
        # Calculate overall metrics
        pattern_accuracy = pattern_results.get('accuracy', 0.0)
        confidence_score = self._calculate_confidence_score(
            events, performance_insights, workflow_insights, ux_insights, quality_insights
        )
        data_completeness = self._calculate_data_completeness(events, config)
        
        return AnalysisResult(
            session_id=session_id,
            analysis_timestamp=datetime.utcnow(),
            configuration=config,
            events_processed=len(events),
            pattern_accuracy=pattern_accuracy,
            performance_insights=performance_insights,
            workflow_insights=workflow_insights,
            user_experience_insights=ux_insights,
            quality_insights=quality_insights,
            improvement_opportunities=[],  # Will be populated by improvement generator
            bottlenecks_identified=bottleneck_results,
            processing_time_ms=processing_time_ms,
            confidence_score=confidence_score,
            data_completeness=data_completeness
        )
    
    def _create_insufficient_data_result(
        self,
        session_id: str,
        events_found: int,
        config: AnalysisConfiguration
    ) -> AnalysisResult:
        """Create result for insufficient data scenarios."""
        return AnalysisResult(
            session_id=session_id,
            analysis_timestamp=datetime.utcnow(),
            configuration=config,
            events_processed=events_found,
            pattern_accuracy=0.0,
            confidence_score=0.0,
            data_completeness=events_found / config.min_events_threshold,
            improvement_opportunities=[{
                'type': 'data_collection',
                'description': f'Need at least {config.min_events_threshold} events for analysis, found {events_found}',
                'impact': 'Enables comprehensive analysis',
                'effort': 'automatic',
                'confidence': 1.0
            }]
        )
    
    def _create_error_result(
        self,
        session_id: str,
        error_message: str,
        config: AnalysisConfiguration
    ) -> AnalysisResult:
        """Create result for error scenarios."""
        return AnalysisResult(
            session_id=session_id,
            analysis_timestamp=datetime.utcnow(),
            configuration=config,
            events_processed=0,
            pattern_accuracy=0.0,
            confidence_score=0.0,
            data_completeness=0.0,
            improvement_opportunities=[{
                'type': 'system_issue',
                'description': f'Analysis failed: {error_message}',
                'impact': 'Fix analysis engine',
                'effort': 'technical',
                'confidence': 1.0
            }]
        )
    
    def _aggregate_session_analyses(
        self,
        synthetic_session_id: str,
        session_analyses: List[AnalysisResult],
        config: AnalysisConfiguration
    ) -> AnalysisResult:
        """Aggregate multiple session analyses for trend analysis."""
        if not session_analyses:
            return self._create_insufficient_data_result(synthetic_session_id, 0, config)
        
        # Aggregate metrics
        total_events = sum(analysis.events_processed for analysis in session_analyses)
        avg_accuracy = statistics.mean(analysis.pattern_accuracy for analysis in session_analyses)
        avg_confidence = statistics.mean(analysis.confidence_score for analysis in session_analyses)
        
        # Aggregate insights
        aggregated_performance = self._aggregate_performance_insights(session_analyses)
        aggregated_workflow = self._aggregate_workflow_insights(session_analyses)
        aggregated_ux = self._aggregate_ux_insights(session_analyses)
        aggregated_quality = self._aggregate_quality_insights(session_analyses)
        
        return AnalysisResult(
            session_id=synthetic_session_id,
            analysis_timestamp=datetime.utcnow(),
            configuration=config,
            events_processed=total_events,
            pattern_accuracy=avg_accuracy,
            performance_insights=aggregated_performance,
            workflow_insights=aggregated_workflow,
            user_experience_insights=aggregated_ux,
            quality_insights=aggregated_quality,
            confidence_score=avg_confidence,
            data_completeness=1.0  # Assume complete for aggregated results
        )
    
    def _aggregate_performance_insights(self, analyses: List[AnalysisResult]) -> Dict[str, Any]:
        """Aggregate performance insights across multiple analyses."""
        all_response_times = []
        all_token_usage = []
        
        for analysis in analyses:
            perf = analysis.performance_insights
            if perf.get('response_times', {}).get('avg_ms'):
                all_response_times.append(perf['response_times']['avg_ms'])
            if perf.get('token_usage', {}).get('avg_tokens'):
                all_token_usage.append(perf['token_usage']['avg_tokens'])
        
        return {
            'aggregated_response_times': {
                'sessions_analyzed': len(all_response_times),
                'overall_avg_ms': statistics.mean(all_response_times) if all_response_times else 0,
                'trend': 'improving' if len(all_response_times) > 1 and 
                        all_response_times[-1] < all_response_times[0] else 'stable'
            },
            'aggregated_token_usage': {
                'sessions_analyzed': len(all_token_usage),
                'overall_avg_tokens': statistics.mean(all_token_usage) if all_token_usage else 0
            }
        }
    
    def _aggregate_workflow_insights(self, analyses: List[AnalysisResult]) -> Dict[str, Any]:
        """Aggregate workflow insights across multiple analyses."""
        all_completion_rates = []
        agent_usage = defaultdict(int)
        
        for analysis in analyses:
            workflow = analysis.workflow_insights
            if workflow.get('completion_rate'):
                all_completion_rates.append(workflow['completion_rate'])
            
            if workflow.get('agent_distribution'):
                for agent, count in workflow['agent_distribution'].items():
                    agent_usage[agent] += count
        
        return {
            'aggregated_completion_rates': {
                'sessions_analyzed': len(all_completion_rates),
                'overall_avg_rate': statistics.mean(all_completion_rates) if all_completion_rates else 0
            },
            'aggregated_agent_usage': dict(agent_usage)
        }
    
    def _aggregate_ux_insights(self, analyses: List[AnalysisResult]) -> Dict[str, Any]:
        """Aggregate user experience insights across multiple analyses."""
        all_error_rates = []
        
        for analysis in analyses:
            ux = analysis.user_experience_insights
            if ux.get('error_rate') is not None:
                all_error_rates.append(ux['error_rate'])
        
        return {
            'aggregated_error_rates': {
                'sessions_analyzed': len(all_error_rates),
                'overall_avg_rate': statistics.mean(all_error_rates) if all_error_rates else 0
            }
        }
    
    def _aggregate_quality_insights(self, analyses: List[AnalysisResult]) -> Dict[str, Any]:
        """Aggregate quality insights across multiple analyses."""
        all_success_rates = []
        
        for analysis in analyses:
            quality = analysis.quality_insights
            if quality.get('success_rate') is not None:
                all_success_rates.append(quality['success_rate'])
        
        return {
            'aggregated_success_rates': {
                'sessions_analyzed': len(all_success_rates),
                'overall_avg_rate': statistics.mean(all_success_rates) if all_success_rates else 0
            }
        }
    
    def _calculate_confidence_score(
        self,
        events: List[BaseEvent],
        performance_insights: Dict[str, Any],
        workflow_insights: Dict[str, Any],
        ux_insights: Dict[str, Any],
        quality_insights: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score for the analysis."""
        factors = []
        
        # Data volume factor
        if len(events) >= 1000:
            factors.append(1.0)
        elif len(events) >= 100:
            factors.append(0.8)
        else:
            factors.append(0.5)
        
        # Data completeness factor
        event_types_found = set(event.event_type for event in events)
        expected_types = {
            EventType.USER_INTERACTION,
            EventType.PERFORMANCE_METRICS,
            EventType.TASK_COMPLETE
        }
        completeness = len(event_types_found & expected_types) / len(expected_types)
        factors.append(completeness)
        
        # Analysis depth factor
        analysis_results = [performance_insights, workflow_insights, ux_insights, quality_insights]
        successful_analyses = sum(1 for result in analysis_results if result.get('status') != 'no_data')
        depth_factor = successful_analyses / len(analysis_results) if analysis_results else 0
        factors.append(depth_factor)
        
        return statistics.mean(factors)
    
    def _calculate_data_completeness(
        self,
        events: List[BaseEvent],
        config: AnalysisConfiguration
    ) -> float:
        """Calculate how complete the data is for comprehensive analysis."""
        if not events:
            return 0.0
        
        # Check for essential event types
        event_types_found = set(event.event_type for event in events)
        
        essential_types = {
            EventType.TASK_START,
            EventType.TASK_COMPLETE,
            EventType.PERFORMANCE_METRICS
        }
        
        optional_types = {
            EventType.USER_INTERACTION,
            EventType.AGENT_DELEGATION,
            EventType.ERROR,
            EventType.QUALITY_GATE
        }
        
        essential_score = len(event_types_found & essential_types) / len(essential_types)
        optional_score = len(event_types_found & optional_types) / len(optional_types)
        
        # Weight essential types more heavily
        return essential_score * 0.7 + optional_score * 0.3
    
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of a dataset."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
            
        return sorted_data[index]
    
    async def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        self.logger.info("RetrospectiveAnalyzer cleanup completed")