"""Comprehensive log schemas for AgentsMCP execution tracking.

This module defines all data structures for capturing agent execution data,
including user interactions, agent delegations, LLM calls, performance metrics,
and error conditions. All schemas are designed for both real-time logging and
retrospective analysis.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class EventType(Enum):
    """Types of events that can be logged in the agent execution system."""
    USER_INTERACTION = "user_interaction"
    AGENT_DELEGATION = "agent_delegation"
    LLM_CALL = "llm_call"
    PERFORMANCE_METRICS = "performance_metrics"
    ERROR = "error"
    CONTEXT_CHANGE = "context_change"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    QUALITY_GATE = "quality_gate"
    RETROSPECTIVE_TRIGGER = "retrospective_trigger"


class EventSeverity(Enum):
    """Severity levels for logged events."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    CRITICAL = "critical"


class SanitizationLevel(Enum):
    """Levels of PII sanitization to apply to logged data."""
    NONE = "none"           # No sanitization (dangerous - only for testing)
    MINIMAL = "minimal"     # Remove obvious PII (emails, phone numbers)
    STANDARD = "standard"   # Remove PII + potentially sensitive strings
    STRICT = "strict"       # Aggressive sanitization, may affect functionality
    PARANOID = "paranoid"   # Maximum sanitization, preserves only structure


@dataclass
class BaseEvent:
    """Base class for all logged events with common metadata."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = field(default=EventType.USER_INTERACTION)
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    severity: EventSeverity = EventSeverity.INFO
    
    # Performance tracking
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Context information
    context_metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'session_id': self.session_id,
            'agent_id': self.agent_id,
            'user_id': self.user_id,
            'severity': self.severity.value,
            'duration_ms': self.duration_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'context_metadata': self.context_metadata,
            'tags': self.tags,
        }


@dataclass
class UserInteractionEvent(BaseEvent):
    """Captures user interactions with the system."""
    event_type: EventType = field(default=EventType.USER_INTERACTION, init=False)
    
    user_input: str = ""
    assistant_response: str = ""
    interaction_mode: str = "chat"  # chat, tui, cli, web
    
    # User experience metrics
    response_time_ms: Optional[float] = None
    user_satisfaction_score: Optional[float] = None
    
    # UI context
    ui_component: Optional[str] = None
    screen_dimensions: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with user interaction specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'user_input': self.user_input,
            'assistant_response': self.assistant_response,
            'interaction_mode': self.interaction_mode,
            'response_time_ms': self.response_time_ms,
            'user_satisfaction_score': self.user_satisfaction_score,
            'ui_component': self.ui_component,
            'screen_dimensions': self.screen_dimensions,
        })
        return base_dict


@dataclass
class AgentDelegationEvent(BaseEvent):
    """Captures agent delegation and coordination events."""
    event_type: EventType = field(default=EventType.AGENT_DELEGATION, init=False)
    
    source_agent_id: str = ""
    target_agent_id: str = ""
    delegation_reason: str = ""
    task_description: str = ""
    
    # Delegation flow tracking
    delegation_chain: List[str] = field(default_factory=list)
    parallel_executions: List[str] = field(default_factory=list)
    
    # Success metrics
    delegation_successful: Optional[bool] = None
    failure_reason: Optional[str] = None
    
    # Agent capability matching
    required_capabilities: List[str] = field(default_factory=list)
    agent_confidence_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with agent delegation specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'source_agent_id': self.source_agent_id,
            'target_agent_id': self.target_agent_id,
            'delegation_reason': self.delegation_reason,
            'task_description': self.task_description,
            'delegation_chain': self.delegation_chain,
            'parallel_executions': self.parallel_executions,
            'delegation_successful': self.delegation_successful,
            'failure_reason': self.failure_reason,
            'required_capabilities': self.required_capabilities,
            'agent_confidence_score': self.agent_confidence_score,
        })
        return base_dict


@dataclass
class LLMCallEvent(BaseEvent):
    """Captures LLM API calls and responses."""
    event_type: EventType = field(default=EventType.LLM_CALL, init=False)
    
    model_name: str = ""
    provider: str = ""
    
    # Request details
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Cost tracking
    estimated_cost_usd: Optional[float] = None
    
    # Performance metrics
    latency_ms: Optional[float] = None
    first_token_latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    
    # Quality indicators
    response_truncated: bool = False
    context_limit_hit: bool = False
    retry_count: int = 0
    
    # Content quality (post-sanitization)
    response_length: Optional[int] = None
    response_quality_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with LLM call specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'model_name': self.model_name,
            'provider': self.provider,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'estimated_cost_usd': self.estimated_cost_usd,
            'latency_ms': self.latency_ms,
            'first_token_latency_ms': self.first_token_latency_ms,
            'tokens_per_second': self.tokens_per_second,
            'response_truncated': self.response_truncated,
            'context_limit_hit': self.context_limit_hit,
            'retry_count': self.retry_count,
            'response_length': self.response_length,
            'response_quality_score': self.response_quality_score,
        })
        return base_dict


@dataclass
class PerformanceMetricsEvent(BaseEvent):
    """Captures system performance metrics."""
    event_type: EventType = field(default=EventType.PERFORMANCE_METRICS, init=False)
    
    # System resource usage
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    disk_io_mb_per_sec: Optional[float] = None
    network_io_mb_per_sec: Optional[float] = None
    
    # Agent-specific metrics
    active_agents: int = 0
    queued_tasks: int = 0
    completed_tasks_per_minute: Optional[float] = None
    
    # Throughput metrics
    events_processed_per_second: Optional[float] = None
    average_response_time_ms: Optional[float] = None
    p95_response_time_ms: Optional[float] = None
    
    # Quality metrics
    error_rate_percent: Optional[float] = None
    user_satisfaction_avg: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with performance metrics specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'disk_io_mb_per_sec': self.disk_io_mb_per_sec,
            'network_io_mb_per_sec': self.network_io_mb_per_sec,
            'active_agents': self.active_agents,
            'queued_tasks': self.queued_tasks,
            'completed_tasks_per_minute': self.completed_tasks_per_minute,
            'events_processed_per_second': self.events_processed_per_second,
            'average_response_time_ms': self.average_response_time_ms,
            'p95_response_time_ms': self.p95_response_time_ms,
            'error_rate_percent': self.error_rate_percent,
            'user_satisfaction_avg': self.user_satisfaction_avg,
        })
        return base_dict


@dataclass
class ErrorEvent(BaseEvent):
    """Captures error conditions and exceptions."""
    event_type: EventType = field(default=EventType.ERROR, init=False)
    severity: EventSeverity = field(default=EventSeverity.ERROR)
    
    error_type: str = ""
    error_message: str = ""
    error_code: Optional[str] = None
    
    # Error context
    component: str = ""
    function_name: Optional[str] = None
    line_number: Optional[int] = None
    
    # Stack trace (sanitized)
    stack_trace_hash: Optional[str] = None
    
    # Recovery information
    auto_recovered: bool = False
    recovery_action: Optional[str] = None
    user_impact: str = "unknown"  # none, minimal, moderate, severe
    
    # Related events
    related_event_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with error specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'error_type': self.error_type,
            'error_message': self.error_message,
            'error_code': self.error_code,
            'component': self.component,
            'function_name': self.function_name,
            'line_number': self.line_number,
            'stack_trace_hash': self.stack_trace_hash,
            'auto_recovered': self.auto_recovered,
            'recovery_action': self.recovery_action,
            'user_impact': self.user_impact,
            'related_event_ids': self.related_event_ids,
        })
        return base_dict


@dataclass
class ContextEvent(BaseEvent):
    """Captures context changes and state transitions."""
    event_type: EventType = field(default=EventType.CONTEXT_CHANGE, init=False)
    
    context_type: str = ""  # session, task, conversation, system
    previous_state: Optional[Dict[str, Any]] = None
    new_state: Dict[str, Any] = field(default_factory=dict)
    
    # Change tracking
    changed_fields: List[str] = field(default_factory=list)
    change_trigger: str = ""
    
    # Context quality
    context_completeness: Optional[float] = None
    context_relevance: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with context specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'context_type': self.context_type,
            'previous_state': self.previous_state,
            'new_state': self.new_state,
            'changed_fields': self.changed_fields,
            'change_trigger': self.change_trigger,
            'context_completeness': self.context_completeness,
            'context_relevance': self.context_relevance,
        })
        return base_dict


# Union type for all possible events
AgentEvent = Union[
    UserInteractionEvent,
    AgentDelegationEvent, 
    LLMCallEvent,
    PerformanceMetricsEvent,
    ErrorEvent,
    ContextEvent,
    BaseEvent,  # For generic events
]


@dataclass
class RetentionPolicy:
    """Defines data retention policies for logged events."""
    
    # Retention periods by event type
    user_interactions_days: int = 90
    agent_delegations_days: int = 30
    llm_calls_days: int = 14
    performance_metrics_days: int = 7
    errors_days: int = 180
    context_changes_days: int = 30
    
    # Aggregation settings
    aggregate_after_days: int = 7
    aggregation_granularity: str = "hourly"  # hourly, daily
    
    # Privacy settings
    auto_purge_pii: bool = True
    anonymize_after_days: int = 30
    
    # Compliance settings
    gdpr_compliant: bool = True
    allow_user_data_export: bool = True
    allow_user_data_deletion: bool = True


@dataclass
class LoggingConfig:
    """Configuration for the execution logging system."""
    
    # Performance settings
    enabled: bool = True
    log_level: EventSeverity = EventSeverity.INFO
    max_events_per_second: int = 10000
    buffer_size: int = 1000
    flush_interval_ms: int = 1000
    
    # Storage settings
    storage_backend: str = "file"  # file, database, memory
    storage_path: Optional[str] = None
    encryption_enabled: bool = True
    compression_enabled: bool = True
    
    # PII handling
    sanitization_level: SanitizationLevel = SanitizationLevel.STANDARD
    sanitization_enabled: bool = True
    
    # Retention settings
    retention_policy: RetentionPolicy = field(default_factory=RetentionPolicy)
    
    # Performance monitoring
    track_logging_overhead: bool = True
    max_logging_overhead_percent: float = 2.0
    logging_latency_target_ms: float = 5.0
    
    # Event filtering
    event_type_filters: List[EventType] = field(default_factory=list)
    severity_filter: EventSeverity = EventSeverity.DEBUG
    
    # Integration settings
    enable_retrospective_triggers: bool = True
    retrospective_trigger_threshold: Dict[str, Any] = field(default_factory=dict)