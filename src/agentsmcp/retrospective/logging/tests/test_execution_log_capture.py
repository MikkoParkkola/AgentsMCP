"""Tests for execution log capture system.

This module tests the high-performance asynchronous logging system including:
- Event queuing and processing
- Performance monitoring and adaptive throttling  
- Error handling and recovery
- Integration with storage adapters
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import List

from ..execution_log_capture import (
    ExecutionLogCapture, PerformanceMetrics, AdaptiveThrottling
)
from ..log_schemas import (
    LoggingConfig, EventSeverity, SanitizationLevel,
    UserInteractionEvent, AgentDelegationEvent, LLMCallEvent, ErrorEvent
)
from ..storage_adapters import MemoryStorageAdapter
from ..pii_sanitizer import PIISanitizer


class TestAdaptiveThrottling:
    """Test adaptive throttling functionality."""
    
    def test_initialization(self):
        """Test throttling system initialization."""
        throttling = AdaptiveThrottling(
            target_overhead_percent=2.0,
            target_latency_ms=5.0
        )
        
        assert throttling.target_overhead_percent == 2.0
        assert throttling.target_latency_ms == 5.0
        assert throttling.throttle_factor == 1.0
    
    def test_performance_tracking(self):
        """Test performance sample recording."""
        throttling = AdaptiveThrottling(sample_window_size=5)
        
        # Record performance samples
        for i in range(3):
            throttling.record_sample(latency_ms=10.0, overhead_percent=1.0)
        
        assert len(throttling.latency_samples) == 3
        assert len(throttling.overhead_samples) == 3
    
    def test_throttling_adjustment(self):
        """Test automatic throttling adjustment."""
        throttling = AdaptiveThrottling(
            target_overhead_percent=2.0,
            target_latency_ms=5.0,
            sample_window_size=3
        )
        
        # Record poor performance samples
        for i in range(3):
            throttling.record_sample(latency_ms=20.0, overhead_percent=5.0)
        
        # Force adjustment
        throttling._adjust_throttling()
        
        # Should increase throttling due to poor performance
        assert throttling.throttle_factor < 1.0
    
    def test_should_log_event_deterministic(self):
        """Test deterministic event logging decisions."""
        throttling = AdaptiveThrottling()
        throttling.throttle_factor = 0.5  # 50% throttling
        
        # Same hash should always give same result
        result1 = throttling.should_log_event("test_event_123")
        result2 = throttling.should_log_event("test_event_123")
        
        assert result1 == result2
    
    def test_should_log_event_no_throttling(self):
        """Test that no throttling allows all events."""
        throttling = AdaptiveThrottling()
        throttling.throttle_factor = 1.0
        
        # Should always return True with no throttling
        for i in range(10):
            assert throttling.should_log_event(f"event_{i}") == True


class TestExecutionLogCapture:
    """Test execution log capture system."""
    
    @pytest.fixture
    async def log_capture(self):
        """Create a test log capture instance."""
        config = LoggingConfig(
            enabled=True,
            buffer_size=100,
            flush_interval_ms=100,
            storage_backend="memory"
        )
        
        storage = MemoryStorageAdapter(max_events=1000)
        sanitizer = PIISanitizer(level=SanitizationLevel.STANDARD)
        
        capture = ExecutionLogCapture(
            config=config,
            storage_adapter=storage,
            pii_sanitizer=sanitizer
        )
        
        await capture.start()
        yield capture
        await capture.stop()
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test log capture initialization."""
        config = LoggingConfig(enabled=True, buffer_size=50)
        storage = MemoryStorageAdapter()
        
        capture = ExecutionLogCapture(
            config=config,
            storage_adapter=storage
        )
        
        assert capture.config.buffer_size == 50
        assert capture.is_running == False
        
        await capture.start()
        assert capture.is_running == True
        
        await capture.stop()
        assert capture.is_running == False
    
    @pytest.mark.asyncio
    async def test_log_user_interaction(self, log_capture):
        """Test logging user interaction events."""
        success = log_capture.log_user_interaction(
            user_input="Hello, world!",
            assistant_response="Hi there! How can I help you?",
            session_id="test_session_123",
            interaction_mode="chat",
            response_time_ms=150.5
        )
        
        assert success == True
        
        # Allow time for processing
        await asyncio.sleep(0.2)
        
        # Check metrics
        metrics = log_capture.get_metrics()
        assert metrics.events_logged_total >= 1
    
    @pytest.mark.asyncio 
    async def test_log_agent_delegation(self, log_capture):
        """Test logging agent delegation events."""
        success = log_capture.log_agent_delegation(
            source_agent_id="orchestrator",
            target_agent_id="coder-agent-1", 
            task_description="Implement user authentication",
            delegation_reason="specialized_capability",
            session_id="test_session_456"
        )
        
        assert success == True
        
        await asyncio.sleep(0.2)
        
        metrics = log_capture.get_metrics()
        assert metrics.events_logged_total >= 1
    
    @pytest.mark.asyncio
    async def test_log_llm_call(self, log_capture):
        """Test logging LLM API call events."""
        success = log_capture.log_llm_call(
            model_name="gpt-4",
            provider="openai",
            session_id="test_session_789",
            prompt_tokens=150,
            completion_tokens=75,
            latency_ms=1200.5,
            estimated_cost_usd=0.0045
        )
        
        assert success == True
        
        await asyncio.sleep(0.2)
        
        metrics = log_capture.get_metrics()
        assert metrics.events_logged_total >= 1
    
    @pytest.mark.asyncio
    async def test_log_error(self, log_capture):
        """Test logging error events."""
        success = log_capture.log_error(
            error_type="ValueError",
            error_message="Invalid input parameter",
            session_id="test_session_error",
            component="user_input_validator",
            severity=EventSeverity.ERROR
        )
        
        assert success == True
        
        await asyncio.sleep(0.2)
        
        metrics = log_capture.get_metrics()
        assert metrics.events_logged_total >= 1
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, log_capture):
        """Test batch processing of multiple events."""
        # Log multiple events quickly
        events_logged = 0
        for i in range(10):
            success = log_capture.log_user_interaction(
                user_input=f"Test message {i}",
                assistant_response=f"Response {i}",
                session_id="batch_test_session"
            )
            if success:
                events_logged += 1
        
        assert events_logged == 10
        
        # Allow time for batch processing
        await asyncio.sleep(0.3)
        
        metrics = log_capture.get_metrics()
        assert metrics.events_logged_total >= 10
    
    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self):
        """Test handling of queue overflow conditions."""
        # Create capture with very small buffer
        config = LoggingConfig(
            enabled=True,
            buffer_size=2,  # Very small buffer
            flush_interval_ms=1000  # Long flush interval
        )
        
        storage = MemoryStorageAdapter()
        capture = ExecutionLogCapture(config=config, storage_adapter=storage)
        
        await capture.start()
        
        try:
            # Fill up the queue beyond capacity
            success_count = 0
            for i in range(10):
                success = capture.log_user_interaction(
                    user_input=f"Overflow test {i}",
                    assistant_response=f"Response {i}",
                    session_id="overflow_test"
                )
                if success:
                    success_count += 1
            
            # Should not be able to log all events due to small buffer
            assert success_count < 10
            
            # Check that buffer overruns are tracked
            metrics = capture.get_metrics()
            assert metrics.buffer_overruns > 0
            
        finally:
            await capture.stop()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, log_capture):
        """Test performance metrics collection."""
        # Log some events to generate metrics
        for i in range(5):
            log_capture.log_user_interaction(
                user_input=f"Performance test {i}",
                assistant_response=f"Response {i}",
                session_id="perf_test"
            )
        
        await asyncio.sleep(0.2)
        
        metrics = log_capture.get_metrics()
        
        # Check that performance metrics are collected
        assert metrics.events_logged_total >= 5
        assert len(log_capture.latency_samples) > 0
        assert metrics.buffer_utilization_percent >= 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_processing(self):
        """Test error handling in event processing."""
        # Create capture with mock storage that fails
        config = LoggingConfig(enabled=True)
        
        mock_storage = AsyncMock()
        mock_storage.store_batch.side_effect = Exception("Storage failure")
        
        capture = ExecutionLogCapture(config=config, storage_adapter=mock_storage)
        
        await capture.start()
        
        try:
            # Log an event
            success = capture.log_user_interaction(
                user_input="Test with failing storage",
                assistant_response="This should handle errors",
                session_id="error_test"
            )
            
            assert success == True  # Should still accept the event
            
            # Allow time for processing and error handling
            await asyncio.sleep(0.3)
            
            # Check that storage failures are tracked
            metrics = capture.get_metrics()
            assert metrics.storage_failures >= 1
            
        finally:
            await capture.stop()
    
    @pytest.mark.asyncio
    async def test_disabled_logging(self):
        """Test that disabled logging doesn't process events."""
        config = LoggingConfig(enabled=False)
        storage = MemoryStorageAdapter()
        
        capture = ExecutionLogCapture(config=config, storage_adapter=storage)
        
        # Should not need to start for disabled logging
        success = capture.log_user_interaction(
            user_input="This should not be logged",
            assistant_response="Disabled response",
            session_id="disabled_test"
        )
        
        assert success == False
        
        metrics = capture.get_metrics()
        assert metrics.events_logged_total == 0
    
    @pytest.mark.asyncio
    async def test_event_filtering(self, log_capture):
        """Test event filtering by type and severity."""
        # Configure to filter out INFO level events
        log_capture.config.severity_filter = EventSeverity.WARN
        
        # Log INFO level event (should be filtered)
        success_info = log_capture.log_error(
            error_type="InfoError",
            error_message="This is just info",
            session_id="filter_test",
            severity=EventSeverity.INFO
        )
        
        # Log ERROR level event (should pass through)
        success_error = log_capture.log_error(
            error_type="CriticalError", 
            error_message="This is critical",
            session_id="filter_test",
            severity=EventSeverity.ERROR
        )
        
        assert success_info == False  # Filtered out
        assert success_error == True  # Allowed through
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, log_capture):
        """Test graceful shutdown with pending events."""
        # Add some events
        for i in range(5):
            log_capture.log_user_interaction(
                user_input=f"Shutdown test {i}",
                assistant_response=f"Response {i}",
                session_id="shutdown_test"
            )
        
        # Should shutdown gracefully and flush remaining events
        await log_capture.stop()
        
        assert log_capture.is_running == False
        
        # Check that events were processed
        metrics = log_capture.get_metrics()
        assert metrics.events_logged_total >= 5
    
    def test_queue_status_monitoring(self, log_capture):
        """Test queue status monitoring functionality."""
        status = log_capture.get_queue_status()
        
        required_keys = [
            'queue_size', 'buffer_size', 'max_queue_size',
            'buffer_utilization_percent', 'throttle_factor', 'is_running'
        ]
        
        for key in required_keys:
            assert key in status
        
        assert isinstance(status['queue_size'], int)
        assert isinstance(status['buffer_utilization_percent'], float)
        assert isinstance(status['is_running'], bool)