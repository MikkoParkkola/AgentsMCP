"""
Real-time Status Streaming with WebSocket Support

High-performance streaming service for real-time updates of agent status,
task progress, and system metrics with WebSocket and Server-Sent Events support.
"""

import asyncio
import json
import time
import weakref
from typing import Dict, List, Optional, Set, Any, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid
from collections import defaultdict, deque

from .base import APIBase, APIResponse, APIError


class StreamType(str, Enum):
    """Types of real-time streams available."""
    AGENT_STATUS = "agent_status"
    TASK_PROGRESS = "task_progress"
    SYSTEM_METRICS = "system_metrics"
    SYMPHONY_EVENTS = "symphony_events"
    USER_ACTIVITY = "user_activity"
    ERROR_EVENTS = "error_events"
    PERFORMANCE_METRICS = "performance_metrics"


class EventType(str, Enum):
    """Types of events that can be streamed."""
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_STATUS_CHANGED = "agent_status_changed"
    TASK_CREATED = "task_created"
    TASK_UPDATED = "task_updated"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    SYSTEM_ALERT = "system_alert"
    METRIC_UPDATE = "metric_update"
    USER_ACTION = "user_action"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class StreamEvent:
    """Real-time stream event."""
    id: str
    type: EventType
    stream_type: StreamType
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    correlation_id: Optional[str] = None
    priority: int = 0  # Higher numbers = higher priority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "stream_type": self.stream_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id,
            "priority": self.priority
        }


@dataclass
class StreamSubscription:
    """Subscription to a real-time stream."""
    id: str
    user_id: str
    stream_types: Set[StreamType]
    filters: Dict[str, Any]
    callback: Optional[Callable] = None
    last_heartbeat: datetime = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.utcnow()
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class RealtimeStatusStreaming(APIBase):
    """High-performance real-time status streaming service."""
    
    def __init__(self):
        super().__init__("realtime_status_streaming")
        
        # Event storage and streaming
        self.event_buffer: deque = deque(maxlen=10000)  # Keep last 10k events
        self.subscriptions: Dict[str, StreamSubscription] = {}
        self.stream_queues: Dict[str, asyncio.Queue] = {}  # Per-subscription event queues
        
        # Performance tracking
        self.metrics = {
            "events_published": 0,
            "events_delivered": 0,
            "active_subscriptions": 0,
            "average_latency_ms": 0.0,
            "events_per_second": 0.0,
            "buffer_utilization": 0.0
        }
        
        # Stream filters and processors
        self.event_filters: Dict[str, Callable] = {}
        self.event_processors: Dict[EventType, Callable] = {}
        self.rate_limiters: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Initialize streaming system
        asyncio.create_task(self._initialize_streaming_system())
    
    async def _initialize_streaming_system(self):
        """Initialize the real-time streaming system."""
        # Start background tasks
        self.background_tasks.add(
            asyncio.create_task(self._process_event_queue())
        )
        self.background_tasks.add(
            asyncio.create_task(self._monitor_subscriptions())
        )
        self.background_tasks.add(
            asyncio.create_task(self._update_metrics())
        )
        self.background_tasks.add(
            asyncio.create_task(self._cleanup_expired_data())
        )
        
        # Initialize event processors
        self.event_processors = {
            EventType.AGENT_STATUS_CHANGED: self._process_agent_status_event,
            EventType.TASK_UPDATED: self._process_task_progress_event,
            EventType.SYSTEM_ALERT: self._process_system_alert_event,
            EventType.METRIC_UPDATE: self._process_metric_update_event,
        }
        
        self.logger.info("Real-time streaming system initialized")
    
    async def publish_event(
        self,
        event_type: EventType,
        stream_type: StreamType,
        data: Dict[str, Any],
        source: str = "system",
        correlation_id: Optional[str] = None,
        priority: int = 0
    ) -> APIResponse:
        """Publish an event to the real-time stream."""
        return await self._execute_with_metrics(
            "publish_event",
            self._publish_event_internal,
            event_type,
            stream_type,
            data,
            source,
            correlation_id,
            priority
        )
    
    async def _publish_event_internal(
        self,
        event_type: EventType,
        stream_type: StreamType,
        data: Dict[str, Any],
        source: str,
        correlation_id: Optional[str],
        priority: int
    ) -> Dict[str, Any]:
        """Internal event publishing logic."""
        event_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        event = StreamEvent(
            id=event_id,
            type=event_type,
            stream_type=stream_type,
            data=data,
            timestamp=timestamp,
            source=source,
            correlation_id=correlation_id,
            priority=priority
        )
        
        # Add to event buffer
        self.event_buffer.append(event)
        
        # Process event through registered processors
        if event_type in self.event_processors:
            try:
                await self.event_processors[event_type](event)
            except Exception as e:
                self.logger.error(f"Event processing failed for {event_type}: {e}")
        
        # Distribute to subscribers
        await self._distribute_event(event)
        
        # Update metrics
        self.metrics["events_published"] += 1
        
        return {
            "event_id": event_id,
            "timestamp": timestamp.isoformat(),
            "distributed_to": len([s for s in self.subscriptions.values() if stream_type in s.stream_types])
        }
    
    async def _distribute_event(self, event: StreamEvent):
        """Distribute event to relevant subscribers."""
        start_time = time.time()
        distributed_count = 0
        
        for subscription in self.subscriptions.values():
            # Check if subscription is interested in this stream type
            if event.stream_type not in subscription.stream_types:
                continue
            
            # Apply filters
            if not await self._event_passes_filters(event, subscription.filters):
                continue
            
            # Check rate limits
            if not await self._check_rate_limit(subscription.user_id, event.stream_type):
                continue
            
            # Add to subscription queue
            queue = self.stream_queues.get(subscription.id)
            if queue:
                try:
                    await queue.put(event)
                    distributed_count += 1
                except asyncio.QueueFull:
                    self.logger.warning(f"Queue full for subscription {subscription.id}")
        
        # Update metrics
        latency_ms = (time.time() - start_time) * 1000
        self.metrics["events_delivered"] += distributed_count
        self.metrics["average_latency_ms"] = (
            self.metrics["average_latency_ms"] * 0.9 + latency_ms * 0.1
        )
    
    async def _event_passes_filters(
        self, 
        event: StreamEvent, 
        filters: Dict[str, Any]
    ) -> bool:
        """Check if event passes subscription filters."""
        if not filters:
            return True
        
        # Priority filter
        if "min_priority" in filters:
            if event.priority < filters["min_priority"]:
                return False
        
        # Source filter
        if "sources" in filters:
            if event.source not in filters["sources"]:
                return False
        
        # Data filters
        if "data_filters" in filters:
            for key, expected_value in filters["data_filters"].items():
                if key not in event.data or event.data[key] != expected_value:
                    return False
        
        # Custom filter function
        if "custom_filter" in filters:
            filter_name = filters["custom_filter"]
            if filter_name in self.event_filters:
                try:
                    return await self.event_filters[filter_name](event)
                except Exception as e:
                    self.logger.error(f"Custom filter {filter_name} failed: {e}")
                    return False
        
        return True
    
    async def _check_rate_limit(self, user_id: str, stream_type: StreamType) -> bool:
        """Check rate limit for user and stream type."""
        current_time = time.time()
        
        # Default rate limits (events per second)
        rate_limits = {
            StreamType.AGENT_STATUS: 10,
            StreamType.TASK_PROGRESS: 20,
            StreamType.SYSTEM_METRICS: 5,
            StreamType.SYMPHONY_EVENTS: 15,
            StreamType.USER_ACTIVITY: 30,
            StreamType.ERROR_EVENTS: 100,  # High limit for errors
            StreamType.PERFORMANCE_METRICS: 5
        }
        
        limit = rate_limits.get(stream_type, 10)
        window = 1.0  # 1 second window
        
        user_limits = self.rate_limiters[user_id]
        stream_key = stream_type.value
        
        if stream_key in user_limits:
            last_reset = user_limits.get(f"{stream_key}_reset", current_time)
            if current_time - last_reset >= window:
                # Reset window
                user_limits[stream_key] = 0
                user_limits[f"{stream_key}_reset"] = current_time
        else:
            user_limits[stream_key] = 0
            user_limits[f"{stream_key}_reset"] = current_time
        
        if user_limits[stream_key] >= limit:
            return False
        
        user_limits[stream_key] += 1
        return True
    
    async def subscribe(
        self,
        user_id: str,
        stream_types: List[StreamType],
        filters: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """Subscribe to real-time streams."""
        return await self._execute_with_metrics(
            "subscribe",
            self._subscribe_internal,
            user_id,
            stream_types,
            filters or {}
        )
    
    async def _subscribe_internal(
        self,
        user_id: str,
        stream_types: List[StreamType],
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal subscription logic."""
        subscription_id = str(uuid.uuid4())
        
        subscription = StreamSubscription(
            id=subscription_id,
            user_id=user_id,
            stream_types=set(stream_types),
            filters=filters
        )
        
        # Create event queue for subscription
        queue = asyncio.Queue(maxsize=1000)
        
        self.subscriptions[subscription_id] = subscription
        self.stream_queues[subscription_id] = queue
        
        self.logger.info(f"Created subscription {subscription_id} for user {user_id}")
        
        return {
            "subscription_id": subscription_id,
            "stream_types": [st.value for st in stream_types],
            "filters_applied": len(filters) > 0,
            "queue_created": True
        }
    
    async def unsubscribe(self, subscription_id: str) -> APIResponse:
        """Unsubscribe from real-time streams."""
        return await self._execute_with_metrics(
            "unsubscribe",
            self._unsubscribe_internal,
            subscription_id
        )
    
    async def _unsubscribe_internal(self, subscription_id: str) -> Dict[str, Any]:
        """Internal unsubscription logic."""
        if subscription_id not in self.subscriptions:
            raise APIError(f"Subscription {subscription_id} not found", "SUBSCRIPTION_NOT_FOUND", 404)
        
        # Remove subscription
        subscription = self.subscriptions.pop(subscription_id)
        
        # Clean up queue
        if subscription_id in self.stream_queues:
            queue = self.stream_queues.pop(subscription_id)
            # Clear queue
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        self.logger.info(f"Removed subscription {subscription_id}")
        
        return {
            "subscription_id": subscription_id,
            "unsubscribed": True
        }
    
    async def get_events_stream(self, subscription_id: str) -> AsyncGenerator[StreamEvent, None]:
        """Get async generator for event stream."""
        if subscription_id not in self.subscriptions:
            raise APIError(f"Subscription {subscription_id} not found", "SUBSCRIPTION_NOT_FOUND", 404)
        
        queue = self.stream_queues.get(subscription_id)
        if not queue:
            raise APIError(f"Queue not found for subscription {subscription_id}", "QUEUE_NOT_FOUND", 500)
        
        subscription = self.subscriptions[subscription_id]
        
        try:
            while True:
                # Update heartbeat
                subscription.last_heartbeat = datetime.utcnow()
                
                # Get next event (with timeout to allow periodic heartbeat updates)
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield event
                except asyncio.TimeoutError:
                    # Send heartbeat event
                    heartbeat_event = StreamEvent(
                        id=str(uuid.uuid4()),
                        type=EventType.SYSTEM_ALERT,
                        stream_type=StreamType.SYSTEM_METRICS,
                        data={"type": "heartbeat", "subscription_id": subscription_id},
                        timestamp=datetime.utcnow(),
                        source="streaming_service"
                    )
                    yield heartbeat_event
        
        except asyncio.CancelledError:
            # Clean subscription when client disconnects
            await self._unsubscribe_internal(subscription_id)
            raise
        except Exception as e:
            self.logger.error(f"Stream error for subscription {subscription_id}: {e}")
            await self._unsubscribe_internal(subscription_id)
            raise
    
    async def get_events_json_stream(self, subscription_id: str) -> AsyncGenerator[str, None]:
        """Get JSON string stream for WebSocket/SSE."""
        async for event in self.get_events_stream(subscription_id):
            yield json.dumps(event.to_dict())
    
    # Event processors
    async def _process_agent_status_event(self, event: StreamEvent):
        """Process agent status change events."""
        # Add enrichment data
        if "agent_id" in event.data:
            event.data["enriched"] = True
            event.data["timestamp_processed"] = datetime.utcnow().isoformat()
    
    async def _process_task_progress_event(self, event: StreamEvent):
        """Process task progress events."""
        # Calculate estimated completion time if not provided
        if "progress_percent" in event.data and "started_at" in event.data:
            progress = event.data["progress_percent"]
            if progress > 0:
                started_at = datetime.fromisoformat(event.data["started_at"].replace('Z', '+00:00'))
                elapsed = (datetime.utcnow().replace(tzinfo=started_at.tzinfo) - started_at).total_seconds()
                if progress < 100:
                    estimated_total = elapsed * (100 / progress)
                    estimated_remaining = estimated_total - elapsed
                    event.data["estimated_completion_seconds"] = estimated_remaining
    
    async def _process_system_alert_event(self, event: StreamEvent):
        """Process system alert events."""
        # Escalate high priority alerts
        if event.priority >= 8:
            # Create escalated event
            escalated_event = StreamEvent(
                id=str(uuid.uuid4()),
                type=EventType.SYSTEM_ALERT,
                stream_type=StreamType.ERROR_EVENTS,
                data={
                    "type": "escalated_alert",
                    "original_event_id": event.id,
                    "escalation_reason": "high_priority",
                    **event.data
                },
                timestamp=datetime.utcnow(),
                source="alert_processor",
                priority=10
            )
            await self._distribute_event(escalated_event)
    
    async def _process_metric_update_event(self, event: StreamEvent):
        """Process metric update events."""
        # Add trend information if available
        if "metric_name" in event.data and "value" in event.data:
            # This would typically use historical data to calculate trends
            event.data["trend"] = "stable"  # Simplified for now
    
    # Background tasks
    async def _process_event_queue(self):
        """Background task to process event queue."""
        while True:
            try:
                # This task handles any background event processing
                # Currently, event processing is handled synchronously in publish_event
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Event queue processing error: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_subscriptions(self):
        """Background task to monitor subscription health."""
        while True:
            try:
                current_time = datetime.utcnow()
                expired_subscriptions = []
                
                for sub_id, subscription in self.subscriptions.items():
                    # Check for stale subscriptions (no heartbeat in 5 minutes)
                    if (current_time - subscription.last_heartbeat).total_seconds() > 300:
                        expired_subscriptions.append(sub_id)
                
                # Clean up expired subscriptions
                for sub_id in expired_subscriptions:
                    try:
                        await self._unsubscribe_internal(sub_id)
                        self.logger.info(f"Cleaned up expired subscription {sub_id}")
                    except Exception as e:
                        self.logger.error(f"Failed to clean up subscription {sub_id}: {e}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Subscription monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _update_metrics(self):
        """Background task to update streaming metrics."""
        while True:
            try:
                self.metrics["active_subscriptions"] = len(self.subscriptions)
                self.metrics["buffer_utilization"] = len(self.event_buffer) / self.event_buffer.maxlen
                
                # Calculate events per second
                current_time = time.time()
                if not hasattr(self, "_last_metrics_update"):
                    self._last_metrics_update = current_time
                    self._last_event_count = self.metrics["events_published"]
                else:
                    time_diff = current_time - self._last_metrics_update
                    event_diff = self.metrics["events_published"] - self._last_event_count
                    
                    if time_diff > 0:
                        self.metrics["events_per_second"] = event_diff / time_diff
                    
                    self._last_metrics_update = current_time
                    self._last_event_count = self.metrics["events_published"]
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_expired_data(self):
        """Background task to clean up expired data."""
        while True:
            try:
                # Clean up old rate limit data
                current_time = time.time()
                for user_id, limits in list(self.rate_limiters.items()):
                    # Remove rate limit data older than 1 hour
                    keys_to_remove = []
                    for key, value in limits.items():
                        if key.endswith("_reset") and current_time - value > 3600:
                            keys_to_remove.append(key)
                            keys_to_remove.append(key.replace("_reset", ""))
                    
                    for key in keys_to_remove:
                        limits.pop(key, None)
                    
                    # Remove empty user entries
                    if not limits:
                        self.rate_limiters.pop(user_id, None)
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                self.logger.error(f"Data cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def get_streaming_metrics(self) -> APIResponse:
        """Get real-time streaming metrics."""
        return await self._execute_with_metrics(
            "get_streaming_metrics",
            self._get_streaming_metrics_internal
        )
    
    async def _get_streaming_metrics_internal(self) -> Dict[str, Any]:
        """Internal logic for getting streaming metrics."""
        # Calculate queue utilization
        total_queue_size = 0
        max_queue_size = 0
        for queue in self.stream_queues.values():
            total_queue_size += queue.qsize()
            max_queue_size += queue.maxsize
        
        queue_utilization = total_queue_size / max_queue_size if max_queue_size > 0 else 0.0
        
        return {
            **self.metrics,
            "queue_utilization": queue_utilization,
            "total_queue_items": total_queue_size,
            "background_tasks_running": len([t for t in self.background_tasks if not t.done()]),
            "event_buffer_size": len(self.event_buffer),
            "rate_limited_users": len(self.rate_limiters)
        }
    
    async def get_subscription_info(self, subscription_id: str) -> APIResponse:
        """Get information about a specific subscription."""
        return await self._execute_with_metrics(
            "get_subscription_info",
            self._get_subscription_info_internal,
            subscription_id
        )
    
    async def _get_subscription_info_internal(self, subscription_id: str) -> Dict[str, Any]:
        """Internal logic for getting subscription info."""
        if subscription_id not in self.subscriptions:
            raise APIError(f"Subscription {subscription_id} not found", "SUBSCRIPTION_NOT_FOUND", 404)
        
        subscription = self.subscriptions[subscription_id]
        queue = self.stream_queues.get(subscription_id)
        
        return {
            "subscription_id": subscription.id,
            "user_id": subscription.user_id,
            "stream_types": [st.value for st in subscription.stream_types],
            "filters": subscription.filters,
            "created_at": subscription.created_at.isoformat(),
            "last_heartbeat": subscription.last_heartbeat.isoformat(),
            "queue_size": queue.qsize() if queue else 0,
            "queue_maxsize": queue.maxsize if queue else 0
        }
    
    async def close(self):
        """Clean shutdown of streaming service."""
        self.logger.info("Shutting down streaming service...")
        
        # Cancel all background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Clean up all subscriptions
        for subscription_id in list(self.subscriptions.keys()):
            await self._unsubscribe_internal(subscription_id)
        
        self.logger.info("Streaming service shut down complete")