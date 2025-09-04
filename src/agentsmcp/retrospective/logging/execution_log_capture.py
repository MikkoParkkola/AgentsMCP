"""High-performance execution log capture system for AgentsMCP.

This module provides the core logging infrastructure with:
- Asynchronous logging to avoid blocking agent execution
- Configurable performance thresholds (<5ms latency, <2% overhead)
- Automatic buffering and batching for high throughput (10k events/sec)
- Integration with PII sanitization pipeline
- Performance monitoring and adaptive throttling
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import psutil
import queue

from .log_schemas import (
    AgentEvent, EventType, EventSeverity, LoggingConfig, 
    UserInteractionEvent, AgentDelegationEvent, LLMCallEvent,
    PerformanceMetricsEvent, ErrorEvent, ContextEvent
)
from .pii_sanitizer import PIISanitizer, SanitizationLevel
from .storage_adapters import StorageAdapter, FileStorageAdapter

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for the logging system itself."""
    
    # Throughput metrics
    events_logged_total: int = 0
    events_per_second: float = 0.0
    peak_events_per_second: float = 0.0
    
    # Latency metrics
    avg_logging_latency_ms: float = 0.0
    p95_logging_latency_ms: float = 0.0
    p99_logging_latency_ms: float = 0.0
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Error tracking
    failed_events: int = 0
    sanitization_failures: int = 0
    storage_failures: int = 0
    
    # Buffer statistics
    buffer_utilization_percent: float = 0.0
    buffer_overruns: int = 0
    
    # Overhead tracking
    total_overhead_ms: float = 0.0
    overhead_percent: float = 0.0


class AdaptiveThrottling:
    """Adaptive throttling system to maintain performance targets."""
    
    def __init__(
        self, 
        target_overhead_percent: float = 2.0,
        target_latency_ms: float = 5.0,
        sample_window_size: int = 100
    ):
        self.target_overhead_percent = target_overhead_percent
        self.target_latency_ms = target_latency_ms
        self.sample_window_size = sample_window_size
        
        # Circular buffers for moving averages
        self.latency_samples = deque(maxlen=sample_window_size)
        self.overhead_samples = deque(maxlen=sample_window_size)
        
        # Throttling state
        self.throttle_factor = 1.0  # 1.0 = no throttling, 0.5 = 50% throttling
        self.last_adjustment = time.time()
        self.adjustment_interval = 5.0  # seconds
    
    def record_sample(self, latency_ms: float, overhead_percent: float) -> None:
        """Record a performance sample for adaptive adjustment."""
        self.latency_samples.append(latency_ms)
        self.overhead_samples.append(overhead_percent)
        
        # Adjust throttling if needed
        now = time.time()
        if now - self.last_adjustment > self.adjustment_interval:
            self._adjust_throttling()
            self.last_adjustment = now
    
    def _adjust_throttling(self) -> None:
        """Adjust throttling based on recent performance."""
        if not self.latency_samples or not self.overhead_samples:
            return
        
        avg_latency = sum(self.latency_samples) / len(self.latency_samples)
        avg_overhead = sum(self.overhead_samples) / len(self.overhead_samples)
        
        # Determine if we need to throttle more or less
        latency_ratio = avg_latency / self.target_latency_ms
        overhead_ratio = avg_overhead / self.target_overhead_percent
        
        # Use the worse of the two ratios
        performance_ratio = max(latency_ratio, overhead_ratio)
        
        if performance_ratio > 1.2:  # Performance is 20% worse than target
            # Increase throttling
            self.throttle_factor = max(0.1, self.throttle_factor * 0.8)
            logger.warning(f"Increasing logging throttling to {self.throttle_factor:.2f} due to performance")
        elif performance_ratio < 0.8:  # Performance is 20% better than target
            # Decrease throttling
            self.throttle_factor = min(1.0, self.throttle_factor * 1.1)
            if self.throttle_factor < 1.0:
                logger.info(f"Decreasing logging throttling to {self.throttle_factor:.2f}")
    
    def should_log_event(self, event_hash: Optional[str] = None) -> bool:
        """Determine if an event should be logged based on current throttling."""
        if self.throttle_factor >= 1.0:
            return True
        
        # Use deterministic sampling based on event hash or random sampling
        if event_hash:
            # Deterministic sampling - same event always gets same decision
            hash_val = hash(event_hash) % 100
            return (hash_val / 100) < self.throttle_factor
        else:
            # Random sampling
            import random
            return random.random() < self.throttle_factor


class ExecutionLogCapture:
    """High-performance asynchronous execution log capture system.
    
    Provides comprehensive agent execution logging with minimal performance impact
    through asynchronous processing, adaptive throttling, and efficient buffering.
    """
    
    def __init__(
        self,
        config: LoggingConfig,
        storage_adapter: Optional[StorageAdapter] = None,
        pii_sanitizer: Optional[PIISanitizer] = None
    ):
        """Initialize the execution log capture system.
        
        Args:
            config: Logging configuration
            storage_adapter: Storage backend for logs
            pii_sanitizer: PII sanitization pipeline
        """
        self.config = config
        self.storage_adapter = storage_adapter or FileStorageAdapter()
        self.pii_sanitizer = pii_sanitizer or PIISanitizer(
            level=config.sanitization_level,
            preserve_analytics=True
        )
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.throttling = AdaptiveThrottling(
            target_overhead_percent=config.max_logging_overhead_percent,
            target_latency_ms=config.logging_latency_target_ms
        )
        
        # Asynchronous processing
        self.event_queue = asyncio.Queue(maxsize=config.buffer_size)
        self.batch_buffer: List[AgentEvent] = []
        self.flush_task: Optional[asyncio.Task] = None
        self.processing_task: Optional[asyncio.Task] = None
        
        # Thread pool for CPU-intensive operations (sanitization, serialization)
        self.thread_pool = ThreadPoolExecutor(
            max_workers=2, 
            thread_name_prefix="logging-worker"
        )
        
        # Performance tracking
        self.latency_samples = deque(maxlen=1000)
        self.start_time = time.time()
        self.last_metrics_update = time.time()
        
        # Event filtering
        self.event_filters: Set[EventType] = set(config.event_type_filters)
        
        # State management
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"ExecutionLogCapture initialized with buffer size {config.buffer_size}")
    
    async def start(self) -> None:
        """Start the asynchronous logging system."""
        if self.is_running:
            logger.warning("ExecutionLogCapture already running")
            return
        
        self.is_running = True
        self._shutdown_event.clear()
        
        # Start background tasks
        self.processing_task = asyncio.create_task(self._process_events())
        self.flush_task = asyncio.create_task(self._periodic_flush())
        
        # Initialize storage adapter
        await self.storage_adapter.initialize()
        
        logger.info("ExecutionLogCapture started")
    
    async def stop(self) -> None:
        """Stop the logging system and flush remaining events."""
        if not self.is_running:
            return
        
        logger.info("Stopping ExecutionLogCapture...")
        
        self.is_running = False
        self._shutdown_event.set()
        
        # Cancel background tasks
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush any remaining events
        await self._flush_batch()
        
        # Close storage adapter
        await self.storage_adapter.close()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("ExecutionLogCapture stopped")
    
    def log_event(
        self, 
        event: AgentEvent,
        priority: EventSeverity = EventSeverity.INFO
    ) -> bool:
        """Log an event asynchronously.
        
        Args:
            event: The event to log
            priority: Event priority for filtering
            
        Returns:
            True if event was queued for logging, False if throttled/filtered
        """
        if not self.is_running or not self.config.enabled:
            return False
        
        start_time = time.perf_counter()
        
        try:
            # Apply event filtering
            if self._should_filter_event(event, priority):
                return False
            
            # Apply adaptive throttling
            event_hash = getattr(event, 'event_id', None)
            if not self.throttling.should_log_event(event_hash):
                return False
            
            # Try to queue the event (non-blocking)
            try:
                self.event_queue.put_nowait(event)
                self.metrics.events_logged_total += 1
                return True
            except asyncio.QueueFull:
                # Queue is full - record buffer overrun
                self.metrics.buffer_overruns += 1
                logger.warning("Event queue full, dropping event")
                return False
        
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            self.metrics.failed_events += 1
            return False
        
        finally:
            # Track logging latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_samples.append(latency_ms)
            
            # Update performance metrics periodically
            if time.time() - self.last_metrics_update > 5.0:
                self._update_performance_metrics()
                self.last_metrics_update = time.time()
    
    def log_user_interaction(
        self,
        user_input: str,
        assistant_response: str,
        session_id: str,
        interaction_mode: str = "chat",
        response_time_ms: Optional[float] = None,
        **metadata
    ) -> bool:
        """Log a user interaction event."""
        event = UserInteractionEvent(
            user_input=user_input,
            assistant_response=assistant_response,
            session_id=session_id,
            interaction_mode=interaction_mode,
            response_time_ms=response_time_ms,
            context_metadata=metadata
        )
        return self.log_event(event)
    
    def log_agent_delegation(
        self,
        source_agent_id: str,
        target_agent_id: str,
        task_description: str,
        delegation_reason: str,
        session_id: str,
        **metadata
    ) -> bool:
        """Log an agent delegation event."""
        event = AgentDelegationEvent(
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            task_description=task_description,
            delegation_reason=delegation_reason,
            session_id=session_id,
            context_metadata=metadata
        )
        return self.log_event(event)
    
    def log_llm_call(
        self,
        model_name: str,
        provider: str,
        session_id: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        estimated_cost_usd: Optional[float] = None,
        **metadata
    ) -> bool:
        """Log an LLM API call event."""
        event = LLMCallEvent(
            model_name=model_name,
            provider=provider,
            session_id=session_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost_usd,
            context_metadata=metadata
        )
        return self.log_event(event)
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        session_id: str,
        component: str = "",
        severity: EventSeverity = EventSeverity.ERROR,
        **metadata
    ) -> bool:
        """Log an error event."""
        event = ErrorEvent(
            error_type=error_type,
            error_message=error_message,
            session_id=session_id,
            component=component,
            severity=severity,
            context_metadata=metadata
        )
        return self.log_event(event, priority=severity)
    
    async def _process_events(self) -> None:
        """Background task to process events from the queue."""
        while self.is_running:
            try:
                # Wait for events or shutdown signal
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Add event to batch buffer
                self.batch_buffer.append(event)
                
                # Flush batch if it reaches the configured size
                if len(self.batch_buffer) >= 100:  # Configurable batch size
                    await self._flush_batch()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing events: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retrying
        
        logger.debug("Event processing task stopped")
    
    async def _periodic_flush(self) -> None:
        """Background task to periodically flush buffered events."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.flush_interval_ms / 1000.0)
                if self.batch_buffer:
                    await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
        
        logger.debug("Periodic flush task stopped")
    
    async def _flush_batch(self) -> None:
        """Flush the current batch of events to storage."""
        if not self.batch_buffer:
            return
        
        batch = self.batch_buffer.copy()
        self.batch_buffer.clear()
        
        try:
            # Sanitize events in thread pool to avoid blocking
            if self.config.sanitization_enabled:
                loop = asyncio.get_event_loop()
                sanitized_batch = await loop.run_in_executor(
                    self.thread_pool,
                    self._sanitize_batch,
                    batch
                )
            else:
                sanitized_batch = batch
            
            # Store the batch
            await self.storage_adapter.store_batch(sanitized_batch)
            
            logger.debug(f"Flushed batch of {len(batch)} events")
            
        except Exception as e:
            logger.error(f"Error flushing batch: {e}")
            self.metrics.storage_failures += 1
            
            # Re-queue events for retry (with limit to prevent infinite growth)
            if len(self.batch_buffer) < self.config.buffer_size // 2:
                self.batch_buffer.extend(batch[:100])  # Limit retry queue size
    
    def _sanitize_batch(self, events: List[AgentEvent]) -> List[AgentEvent]:
        """Sanitize a batch of events (runs in thread pool)."""
        try:
            return self.pii_sanitizer.sanitize_batch(events)
        except Exception as e:
            logger.error(f"Error sanitizing batch: {e}")
            self.metrics.sanitization_failures += 1
            return events  # Return unsanitized events rather than lose them
    
    def _should_filter_event(self, event: AgentEvent, priority: EventSeverity) -> bool:
        """Determine if an event should be filtered out."""
        # Check event type filters
        if self.event_filters and event.event_type in self.event_filters:
            return True
        
        # Check severity filter
        severity_order = [
            EventSeverity.TRACE,
            EventSeverity.DEBUG, 
            EventSeverity.INFO,
            EventSeverity.WARN,
            EventSeverity.ERROR,
            EventSeverity.CRITICAL
        ]
        
        event_severity_level = severity_order.index(priority)
        config_severity_level = severity_order.index(self.config.severity_filter)
        
        return event_severity_level < config_severity_level
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics for monitoring."""
        try:
            # Calculate throughput
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.metrics.events_per_second = self.metrics.events_logged_total / elapsed
            
            # Calculate latency statistics
            if self.latency_samples:
                sorted_samples = sorted(self.latency_samples)
                n = len(sorted_samples)
                
                self.metrics.avg_logging_latency_ms = sum(sorted_samples) / n
                self.metrics.p95_logging_latency_ms = sorted_samples[int(n * 0.95)]
                self.metrics.p99_logging_latency_ms = sorted_samples[int(n * 0.99)]
            
            # Update system resource usage
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.cpu_usage_percent = process.cpu_percent()
            
            # Calculate buffer utilization
            queue_size = self.event_queue.qsize()
            self.metrics.buffer_utilization_percent = (queue_size / self.config.buffer_size) * 100
            
            # Calculate overhead percentage (approximate)
            total_time = elapsed
            logging_time = self.metrics.total_overhead_ms / 1000
            if total_time > 0:
                self.metrics.overhead_percent = (logging_time / total_time) * 100
            
            # Update adaptive throttling
            self.throttling.record_sample(
                self.metrics.avg_logging_latency_ms,
                self.metrics.overhead_percent
            )
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        self._update_performance_metrics()
        return self.metrics
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status for monitoring."""
        return {
            'queue_size': self.event_queue.qsize(),
            'buffer_size': len(self.batch_buffer),
            'max_queue_size': self.config.buffer_size,
            'buffer_utilization_percent': self.metrics.buffer_utilization_percent,
            'throttle_factor': self.throttling.throttle_factor,
            'is_running': self.is_running,
        }