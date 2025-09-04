"""
Bottleneck Identification Engine

Identifies performance bottlenecks across distributed agent workflows
using flow analysis, queuing theory, and critical path analysis.
"""

import asyncio
import logging
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import networkx as nx

from ..logging.log_schemas import BaseEvent, EventType


class BottleneckType(Enum):
    """Types of bottlenecks that can be identified."""
    RESOURCE_CONTENTION = "resource_contention"
    SEQUENTIAL_DEPENDENCY = "sequential_dependency"
    PROCESSING_DELAY = "processing_delay"
    QUEUE_BUILDUP = "queue_buildup"
    NETWORK_LATENCY = "network_latency"
    CONTEXT_SWITCHING = "context_switching"
    MEMORY_PRESSURE = "memory_pressure"
    AGENT_OVERLOAD = "agent_overload"


class BottleneckSeverity(Enum):
    """Severity levels for identified bottlenecks."""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Bottleneck:
    """A bottleneck identified in the workflow."""
    bottleneck_id: str
    bottleneck_type: BottleneckType
    severity: BottleneckSeverity
    description: str
    location: str  # Where the bottleneck occurs
    impact_metrics: Dict[str, float]
    root_cause: str
    affected_workflows: List[str]
    time_range: Tuple[datetime, datetime]
    evidence: Dict[str, Any]
    suggested_remediation: str
    estimated_improvement: str
    confidence: float


@dataclass
class WorkflowNode:
    """Node in the workflow graph."""
    node_id: str
    node_type: str
    processing_time: float = 0.0
    queue_time: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    throughput: float = 0.0
    error_rate: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)


@dataclass
class WorkflowPath:
    """Path through the workflow graph."""
    path_id: str
    nodes: List[str]
    total_time: float
    critical_path: bool = False
    bottleneck_nodes: List[str] = field(default_factory=list)


class BottleneckIdentifier:
    """
    Advanced bottleneck identification using workflow analysis,
    queuing theory, and critical path analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Analysis state
        self._workflow_graph = nx.DiGraph()
        self._node_metrics = {}
        self._flow_patterns = {}
    
    async def identify_bottlenecks(
        self,
        events: List[BaseEvent],
        pattern_results: Optional[Dict[str, Any]] = None
    ) -> List[Bottleneck]:
        """
        Identify bottlenecks across distributed agent workflows.
        
        Args:
            events: List of execution events to analyze
            pattern_results: Optional pattern detection results for context
            
        Returns:
            List of identified bottlenecks with remediation suggestions
        """
        if len(events) < 20:
            return []
        
        try:
            self.logger.info(f"Identifying bottlenecks in {len(events)} events")
            
            # 1. Build workflow graph from events
            workflow_graph = await self._build_workflow_graph(events)
            
            # 2. Calculate node metrics
            node_metrics = await self._calculate_node_metrics(events, workflow_graph)
            
            # 3. Identify different types of bottlenecks
            bottlenecks = []
            
            # Resource contention bottlenecks
            resource_bottlenecks = await self._identify_resource_bottlenecks(
                events, node_metrics
            )
            bottlenecks.extend(resource_bottlenecks)
            
            # Processing delay bottlenecks
            processing_bottlenecks = await self._identify_processing_bottlenecks(
                events, node_metrics
            )
            bottlenecks.extend(processing_bottlenecks)
            
            # Queue buildup bottlenecks
            queue_bottlenecks = await self._identify_queue_bottlenecks(
                events, workflow_graph
            )
            bottlenecks.extend(queue_bottlenecks)
            
            # Critical path bottlenecks
            critical_path_bottlenecks = await self._identify_critical_path_bottlenecks(
                workflow_graph, node_metrics
            )
            bottlenecks.extend(critical_path_bottlenecks)
            
            # Agent overload bottlenecks
            agent_bottlenecks = await self._identify_agent_overload_bottlenecks(
                events, node_metrics
            )
            bottlenecks.extend(agent_bottlenecks)
            
            # Sort by severity and impact
            bottlenecks.sort(key=lambda b: (
                b.severity.value,
                -b.impact_metrics.get('performance_impact', 0)
            ))
            
            self.logger.info(f"Identified {len(bottlenecks)} bottlenecks")
            return bottlenecks
            
        except Exception as e:
            self.logger.error(f"Bottleneck identification failed: {e}")
            return []
    
    async def _build_workflow_graph(self, events: List[BaseEvent]) -> nx.DiGraph:
        """Build a directed graph representing the workflow from events."""
        graph = nx.DiGraph()
        
        # Group events by session and task
        session_events = defaultdict(list)
        for event in events:
            session_events[event.session_id].append(event)
        
        for session_id, session_events_list in session_events.items():
            # Sort events by timestamp
            session_events_list.sort(key=lambda e: e.timestamp)
            
            # Build workflow nodes and edges
            prev_node = None
            for i, event in enumerate(session_events_list):
                node_id = f"{session_id}_{event.event_type.value}_{i}"
                
                # Add node with basic properties
                graph.add_node(node_id, 
                    event_type=event.event_type.value,
                    timestamp=event.timestamp,
                    session_id=session_id,
                    event_index=i
                )
                
                # Add edge from previous node
                if prev_node:
                    time_diff = (event.timestamp - session_events_list[i-1].timestamp).total_seconds()
                    graph.add_edge(prev_node, node_id, weight=time_diff)
                
                prev_node = node_id
        
        self.logger.debug(f"Built workflow graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph
    
    async def _calculate_node_metrics(
        self, 
        events: List[BaseEvent], 
        workflow_graph: nx.DiGraph
    ) -> Dict[str, WorkflowNode]:
        """Calculate performance metrics for each workflow node."""
        node_metrics = {}
        
        # Group events by type for metric calculation
        events_by_type = defaultdict(list)
        for event in events:
            events_by_type[event.event_type].append(event)
        
        for node_id in workflow_graph.nodes():
            node_data = workflow_graph.nodes[node_id]
            event_type = node_data['event_type']
            
            # Find corresponding events
            matching_events = [
                e for e in events_by_type[EventType(event_type)]
                if e.session_id == node_data['session_id']
            ]
            
            # Calculate metrics
            processing_times = []
            memory_usage = []
            
            for event in matching_events:
                if hasattr(event, 'response_time_ms') and event.response_time_ms:
                    processing_times.append(event.response_time_ms)
                if hasattr(event, 'memory_mb') and event.memory_mb:
                    memory_usage.append(event.memory_mb)
            
            node = WorkflowNode(
                node_id=node_id,
                node_type=event_type,
                processing_time=statistics.mean(processing_times) if processing_times else 0,
                resource_usage={
                    'memory_mb': statistics.mean(memory_usage) if memory_usage else 0
                },
                throughput=len(matching_events),
                error_rate=0.0  # Would be calculated from error events
            )
            
            # Calculate queue time (time waiting for processing)
            predecessors = list(workflow_graph.predecessors(node_id))
            if predecessors:
                edge_weights = [workflow_graph[pred][node_id]['weight'] for pred in predecessors]
                node.queue_time = statistics.mean(edge_weights)
            
            node_metrics[node_id] = node
        
        return node_metrics
    
    async def _identify_resource_bottlenecks(
        self,
        events: List[BaseEvent],
        node_metrics: Dict[str, WorkflowNode]
    ) -> List[Bottleneck]:
        """Identify bottlenecks caused by resource contention."""
        bottlenecks = []
        
        # Analyze memory usage patterns
        memory_usage_by_type = defaultdict(list)
        for node in node_metrics.values():
            memory_usage = node.resource_usage.get('memory_mb', 0)
            if memory_usage > 0:
                memory_usage_by_type[node.node_type].append(memory_usage)
        
        for node_type, memory_values in memory_usage_by_type.items():
            if len(memory_values) >= 3:
                mean_memory = statistics.mean(memory_values)
                max_memory = max(memory_values)
                
                # Check for memory pressure (high variation or consistently high usage)
                if max_memory > mean_memory * 2 and mean_memory > 100:  # >100MB average
                    severity = self._classify_resource_severity(mean_memory, max_memory)
                    
                    bottleneck = Bottleneck(
                        bottleneck_id=f"memory_pressure_{node_type}",
                        bottleneck_type=BottleneckType.MEMORY_PRESSURE,
                        severity=severity,
                        description=f"High memory usage in {node_type} operations",
                        location=node_type,
                        impact_metrics={
                            'avg_memory_mb': mean_memory,
                            'peak_memory_mb': max_memory,
                            'memory_variance': statistics.variance(memory_values)
                        },
                        root_cause=f"Memory-intensive operations in {node_type}",
                        affected_workflows=[node_type],
                        time_range=(events[0].timestamp, events[-1].timestamp),
                        evidence={
                            'memory_samples': len(memory_values),
                            'memory_distribution': {
                                'min': min(memory_values),
                                'mean': mean_memory,
                                'max': max_memory
                            }
                        },
                        suggested_remediation="Optimize memory usage, implement memory pooling, or increase available memory",
                        estimated_improvement="15-30% performance improvement",
                        confidence=0.75
                    )
                    bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    async def _identify_processing_bottlenecks(
        self,
        events: List[BaseEvent],
        node_metrics: Dict[str, WorkflowNode]
    ) -> List[Bottleneck]:
        """Identify bottlenecks caused by slow processing."""
        bottlenecks = []
        
        # Analyze processing times by node type
        processing_times_by_type = defaultdict(list)
        for node in node_metrics.values():
            if node.processing_time > 0:
                processing_times_by_type[node.node_type].append(node.processing_time)
        
        # Calculate overall statistics
        all_processing_times = []
        for times in processing_times_by_type.values():
            all_processing_times.extend(times)
        
        if not all_processing_times:
            return bottlenecks
        
        overall_mean = statistics.mean(all_processing_times)
        overall_p95 = self._percentile(all_processing_times, 0.95)
        
        for node_type, times in processing_times_by_type.items():
            if len(times) >= 3:
                mean_time = statistics.mean(times)
                p95_time = self._percentile(times, 0.95)
                
                # Identify slow processing (significantly above average)
                if mean_time > overall_mean * 1.5 or p95_time > overall_p95 * 1.3:
                    severity = self._classify_processing_severity(mean_time, p95_time, overall_mean)
                    
                    bottleneck = Bottleneck(
                        bottleneck_id=f"processing_delay_{node_type}",
                        bottleneck_type=BottleneckType.PROCESSING_DELAY,
                        severity=severity,
                        description=f"Slow processing in {node_type} operations",
                        location=node_type,
                        impact_metrics={
                            'avg_processing_time_ms': mean_time,
                            'p95_processing_time_ms': p95_time,
                            'slowdown_factor': mean_time / overall_mean
                        },
                        root_cause=f"Inefficient processing in {node_type}",
                        affected_workflows=[node_type],
                        time_range=(events[0].timestamp, events[-1].timestamp),
                        evidence={
                            'processing_samples': len(times),
                            'time_distribution': {
                                'min': min(times),
                                'mean': mean_time,
                                'p95': p95_time,
                                'max': max(times)
                            }
                        },
                        suggested_remediation="Optimize algorithm, add caching, or increase processing resources",
                        estimated_improvement=f"{int((mean_time / overall_mean - 1) * 100)}% faster processing possible",
                        confidence=0.80
                    )
                    bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    async def _identify_queue_bottlenecks(
        self,
        events: List[BaseEvent],
        workflow_graph: nx.DiGraph
    ) -> List[Bottleneck]:
        """Identify bottlenecks caused by queue buildup."""
        bottlenecks = []
        
        # Analyze event intervals to detect queuing
        session_intervals = defaultdict(list)
        session_events = defaultdict(list)
        
        for event in events:
            session_events[event.session_id].append(event)
        
        for session_id, session_event_list in session_events.items():
            session_event_list.sort(key=lambda e: e.timestamp)
            
            for i in range(1, len(session_event_list)):
                interval = (session_event_list[i].timestamp - session_event_list[i-1].timestamp).total_seconds()
                session_intervals[session_id].append(interval)
        
        # Detect queue buildup patterns
        for session_id, intervals in session_intervals.items():
            if len(intervals) >= 5:
                mean_interval = statistics.mean(intervals)
                max_interval = max(intervals)
                
                # Look for large intervals that suggest queuing
                large_intervals = [i for i in intervals if i > mean_interval * 3]
                
                if len(large_intervals) >= 2 and max_interval > 5:  # >5 second delays
                    severity = self._classify_queue_severity(large_intervals, mean_interval)
                    
                    bottleneck = Bottleneck(
                        bottleneck_id=f"queue_buildup_{session_id}",
                        bottleneck_type=BottleneckType.QUEUE_BUILDUP,
                        severity=severity,
                        description=f"Queue buildup detected in session {session_id}",
                        location=f"Session {session_id}",
                        impact_metrics={
                            'avg_interval_s': mean_interval,
                            'max_interval_s': max_interval,
                            'large_intervals_count': len(large_intervals)
                        },
                        root_cause="Processing capacity insufficient for request volume",
                        affected_workflows=[session_id],
                        time_range=(events[0].timestamp, events[-1].timestamp),
                        evidence={
                            'interval_samples': len(intervals),
                            'large_intervals': large_intervals[:5],  # First 5 examples
                            'queue_ratio': len(large_intervals) / len(intervals)
                        },
                        suggested_remediation="Increase processing capacity, implement load balancing, or add queue management",
                        estimated_improvement="20-40% reduction in wait times",
                        confidence=0.70
                    )
                    bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    async def _identify_critical_path_bottlenecks(
        self,
        workflow_graph: nx.DiGraph,
        node_metrics: Dict[str, WorkflowNode]
    ) -> List[Bottleneck]:
        """Identify bottlenecks on the critical path."""
        bottlenecks = []
        
        if workflow_graph.number_of_nodes() < 3:
            return bottlenecks
        
        try:
            # Find longest path (critical path) through the workflow
            if nx.is_directed_acyclic_graph(workflow_graph):
                # For DAGs, we can use topological sort
                topo_sorted = list(nx.topological_sort(workflow_graph))
                
                # Calculate longest paths
                longest_paths = {}
                for node in topo_sorted:
                    predecessors = list(workflow_graph.predecessors(node))
                    if not predecessors:
                        longest_paths[node] = 0
                    else:
                        max_path = max(
                            longest_paths.get(pred, 0) + workflow_graph[pred][node]['weight']
                            for pred in predecessors
                        )
                        longest_paths[node] = max_path
                
                # Find critical path nodes (those with longest accumulated time)
                if longest_paths:
                    max_time = max(longest_paths.values())
                    critical_nodes = [
                        node for node, time in longest_paths.items()
                        if time > max_time * 0.8  # Within 80% of maximum
                    ]
                    
                    # Analyze critical path nodes for bottlenecks
                    for node in critical_nodes:
                        if node in node_metrics:
                            node_metric = node_metrics[node]
                            
                            # Check if this critical path node is particularly slow
                            if node_metric.processing_time > 1000:  # >1 second
                                bottleneck = Bottleneck(
                                    bottleneck_id=f"critical_path_{node}",
                                    bottleneck_type=BottleneckType.SEQUENTIAL_DEPENDENCY,
                                    severity=BottleneckSeverity.HIGH,
                                    description=f"Critical path bottleneck at {node_metric.node_type}",
                                    location=node,
                                    impact_metrics={
                                        'critical_path_time': longest_paths[node],
                                        'node_processing_time': node_metric.processing_time,
                                        'path_contribution': node_metric.processing_time / max_time
                                    },
                                    root_cause=f"Slow processing on critical path in {node_metric.node_type}",
                                    affected_workflows=["critical_path"],
                                    time_range=(datetime.utcnow(), datetime.utcnow()),  # Would need actual timestamps
                                    evidence={
                                        'critical_nodes_count': len(critical_nodes),
                                        'total_critical_path_time': max_time
                                    },
                                    suggested_remediation="Optimize critical path operations, parallelize where possible",
                                    estimated_improvement="Direct impact on overall workflow time",
                                    confidence=0.85
                                )
                                bottlenecks.append(bottleneck)
        
        except Exception as e:
            self.logger.warning(f"Critical path analysis failed: {e}")
        
        return bottlenecks
    
    async def _identify_agent_overload_bottlenecks(
        self,
        events: List[BaseEvent],
        node_metrics: Dict[str, WorkflowNode]
    ) -> List[Bottleneck]:
        """Identify bottlenecks caused by agent overload."""
        bottlenecks = []
        
        # Analyze agent delegation patterns
        agent_events = [e for e in events if e.event_type == EventType.AGENT_DELEGATION]
        
        if not agent_events:
            return bottlenecks
        
        # Count events per agent
        agent_load = defaultdict(int)
        agent_timeframes = defaultdict(list)
        
        for event in agent_events:
            if hasattr(event, 'target_agent'):
                agent_load[event.target_agent] += 1
                agent_timeframes[event.target_agent].append(event.timestamp)
        
        # Analyze load distribution
        total_delegations = sum(agent_load.values())
        mean_load = total_delegations / len(agent_load) if agent_load else 0
        
        for agent, load in agent_load.items():
            if load > mean_load * 2:  # Agent handling 2x average load
                # Calculate load over time
                timestamps = sorted(agent_timeframes[agent])
                if len(timestamps) >= 2:
                    time_span = (timestamps[-1] - timestamps[0]).total_seconds()
                    load_rate = load / time_span if time_span > 0 else 0
                    
                    severity = self._classify_agent_overload_severity(load, mean_load)
                    
                    bottleneck = Bottleneck(
                        bottleneck_id=f"agent_overload_{agent}",
                        bottleneck_type=BottleneckType.AGENT_OVERLOAD,
                        severity=severity,
                        description=f"Agent {agent} is overloaded with {load} delegations",
                        location=f"Agent {agent}",
                        impact_metrics={
                            'delegation_count': load,
                            'load_ratio': load / mean_load,
                            'load_rate_per_second': load_rate
                        },
                        root_cause=f"Excessive delegation to agent {agent}",
                        affected_workflows=[agent],
                        time_range=(timestamps[0], timestamps[-1]),
                        evidence={
                            'total_agents': len(agent_load),
                            'mean_load': mean_load,
                            'time_span_seconds': time_span
                        },
                        suggested_remediation="Redistribute load, add agent instances, or optimize delegation logic",
                        estimated_improvement="10-25% throughput improvement",
                        confidence=0.75
                    )
                    bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _classify_resource_severity(self, mean_memory: float, max_memory: float) -> BottleneckSeverity:
        """Classify severity of resource bottlenecks."""
        if max_memory > 1000 or mean_memory > 500:  # >1GB peak or >500MB average
            return BottleneckSeverity.CRITICAL
        elif max_memory > 500 or mean_memory > 250:
            return BottleneckSeverity.HIGH
        elif max_memory > 200 or mean_memory > 100:
            return BottleneckSeverity.MEDIUM
        else:
            return BottleneckSeverity.LOW
    
    def _classify_processing_severity(
        self, 
        mean_time: float, 
        p95_time: float, 
        overall_mean: float
    ) -> BottleneckSeverity:
        """Classify severity of processing bottlenecks."""
        slowdown_factor = mean_time / overall_mean
        
        if slowdown_factor > 3 or p95_time > 10000:  # 3x slower or >10s p95
            return BottleneckSeverity.CRITICAL
        elif slowdown_factor > 2 or p95_time > 5000:  # 2x slower or >5s p95
            return BottleneckSeverity.HIGH
        elif slowdown_factor > 1.5 or p95_time > 2000:  # 1.5x slower or >2s p95
            return BottleneckSeverity.MEDIUM
        else:
            return BottleneckSeverity.LOW
    
    def _classify_queue_severity(
        self, 
        large_intervals: List[float], 
        mean_interval: float
    ) -> BottleneckSeverity:
        """Classify severity of queue bottlenecks."""
        max_interval = max(large_intervals) if large_intervals else 0
        queue_ratio = len(large_intervals) / 10  # Rough estimate
        
        if max_interval > 30 or queue_ratio > 0.5:  # >30s delays or >50% queue ratio
            return BottleneckSeverity.CRITICAL
        elif max_interval > 15 or queue_ratio > 0.3:  # >15s delays or >30% queue ratio
            return BottleneckSeverity.HIGH
        elif max_interval > 10 or queue_ratio > 0.2:  # >10s delays or >20% queue ratio
            return BottleneckSeverity.MEDIUM
        else:
            return BottleneckSeverity.LOW
    
    def _classify_agent_overload_severity(self, load: int, mean_load: float) -> BottleneckSeverity:
        """Classify severity of agent overload bottlenecks."""
        load_ratio = load / mean_load if mean_load > 0 else 1
        
        if load_ratio > 4:  # 4x average load
            return BottleneckSeverity.CRITICAL
        elif load_ratio > 3:  # 3x average load
            return BottleneckSeverity.HIGH
        elif load_ratio > 2:  # 2x average load
            return BottleneckSeverity.MEDIUM
        else:
            return BottleneckSeverity.LOW
    
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