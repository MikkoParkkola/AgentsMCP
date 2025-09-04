"""Integration hooks for connecting logging infrastructure to existing AgentsMCP components.

This module provides the integration layer that connects the execution logging
infrastructure to the chat engine, orchestrator, and agent systems without
modifying their core logic.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from functools import wraps
from contextvars import ContextVar

from .log_schemas import (
    EventType, EventSeverity, LoggingConfig,
    UserInteractionEvent, AgentDelegationEvent, LLMCallEvent,
    PerformanceMetricsEvent, ErrorEvent, ContextEvent
)
from .execution_log_capture import ExecutionLogCapture
from ..storage.log_store import LogStore

logger = logging.getLogger(__name__)

# Context variables for tracking current execution context
current_session_id: ContextVar[Optional[str]] = ContextVar('current_session_id', default=None)
current_agent_id: ContextVar[Optional[str]] = ContextVar('current_agent_id', default=None)
current_user_id: ContextVar[Optional[str]] = ContextVar('current_user_id', default=None)


@dataclass
class IntegrationConfig:
    """Configuration for logging integration."""
    
    # Chat engine integration
    log_user_inputs: bool = True
    log_assistant_responses: bool = True
    log_streaming_updates: bool = False
    
    # Agent delegation integration
    log_delegations: bool = True
    log_agent_spawning: bool = True
    log_parallel_execution: bool = True
    
    # LLM call integration
    log_llm_requests: bool = True
    log_llm_responses: bool = True
    log_token_usage: bool = True
    log_costs: bool = True
    
    # Performance monitoring
    log_response_times: bool = True
    log_system_metrics: bool = True
    monitor_memory_usage: bool = True
    
    # Error tracking
    log_exceptions: bool = True
    log_recoveries: bool = True
    track_error_patterns: bool = True
    
    # Context tracking
    log_context_changes: bool = True
    track_session_state: bool = True
    
    # Sampling and filtering
    sample_rate: float = 1.0  # 1.0 = log everything, 0.1 = log 10%
    min_log_level: EventSeverity = EventSeverity.DEBUG


class LoggingIntegration:
    """Central integration point for connecting logging to AgentsMCP components."""
    
    def __init__(
        self,
        log_store: LogStore,
        config: Optional[IntegrationConfig] = None
    ):
        """Initialize the logging integration.
        
        Args:
            log_store: The log store instance for persisting events
            config: Integration configuration
        """
        self.log_store = log_store
        self.config = config or IntegrationConfig()
        
        # Active integrations tracking
        self.active_integrations: Dict[str, bool] = {}
        
        # Performance tracking
        self.integration_stats = {
            'events_logged': 0,
            'integration_errors': 0,
            'total_overhead_ms': 0.0
        }
    
    # Context Management
    
    def set_session_context(self, session_id: str, user_id: Optional[str] = None) -> None:
        """Set the current session context for logging."""
        current_session_id.set(session_id)
        if user_id:
            current_user_id.set(user_id)
    
    def set_agent_context(self, agent_id: str) -> None:
        """Set the current agent context for logging."""
        current_agent_id.set(agent_id)
    
    def clear_context(self) -> None:
        """Clear all context variables."""
        current_session_id.set(None)
        current_agent_id.set(None)
        current_user_id.set(None)
    
    # Chat Engine Integration
    
    def create_chat_engine_wrapper(self, chat_engine_class):
        """Create a wrapper for the chat engine that adds logging."""
        
        class LoggingChatEngineWrapper(chat_engine_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._logging_integration = None
            
            def set_logging_integration(self, integration: 'LoggingIntegration'):
                self._logging_integration = integration
            
            async def process_user_input(self, user_input: str, **kwargs):
                """Override to add input logging."""
                if self._logging_integration and self._logging_integration.config.log_user_inputs:
                    session_id = current_session_id.get() or getattr(self, 'session_id', 'unknown')
                    
                    # Log user interaction start
                    start_time = time.perf_counter()
                    
                    try:
                        # Call original method
                        result = await super().process_user_input(user_input, **kwargs)
                        
                        # Log successful interaction
                        response_time_ms = (time.perf_counter() - start_time) * 1000
                        
                        if hasattr(result, 'content') and self._logging_integration.config.log_assistant_responses:
                            await self._logging_integration.log_user_interaction(
                                user_input=user_input,
                                assistant_response=result.content,
                                session_id=session_id,
                                response_time_ms=response_time_ms,
                                **kwargs
                            )
                        
                        return result
                        
                    except Exception as e:
                        # Log error
                        if self._logging_integration.config.log_exceptions:
                            await self._logging_integration.log_error(
                                error_type=type(e).__name__,
                                error_message=str(e),
                                component="chat_engine",
                                session_id=session_id
                            )
                        raise
                
                return await super().process_user_input(user_input, **kwargs)
        
        return LoggingChatEngineWrapper
    
    # Orchestrator Integration
    
    def create_orchestrator_hooks(self):
        """Create hooks for orchestrator delegation events."""
        
        async def log_delegation_start(
            source_agent: str,
            target_agent: str,
            task_description: str,
            delegation_reason: str = ""
        ):
            """Log the start of an agent delegation."""
            if self.config.log_delegations:
                session_id = current_session_id.get() or "unknown"
                
                await self.log_agent_delegation(
                    source_agent_id=source_agent,
                    target_agent_id=target_agent,
                    task_description=task_description,
                    delegation_reason=delegation_reason,
                    session_id=session_id
                )
        
        async def log_delegation_complete(
            source_agent: str,
            target_agent: str,
            success: bool,
            result_summary: str = "",
            duration_ms: Optional[float] = None
        ):
            """Log the completion of an agent delegation."""
            if self.config.log_delegations:
                session_id = current_session_id.get() or "unknown"
                
                # Create a delegation event with completion info
                event = AgentDelegationEvent(
                    source_agent_id=source_agent,
                    target_agent_id=target_agent,
                    task_description=result_summary,
                    delegation_reason="delegation_complete",
                    session_id=session_id,
                    delegation_successful=success,
                    duration_ms=duration_ms
                )
                
                await self._log_event(event)
        
        return {
            'log_delegation_start': log_delegation_start,
            'log_delegation_complete': log_delegation_complete
        }
    
    # LLM Call Integration
    
    def create_llm_call_wrapper(self, llm_client_method):
        """Create a wrapper for LLM API calls that adds logging."""
        
        @wraps(llm_client_method)
        async def wrapped_llm_call(*args, **kwargs):
            if not self.config.log_llm_requests:
                return await llm_client_method(*args, **kwargs)
            
            start_time = time.perf_counter()
            session_id = current_session_id.get() or "unknown"
            
            # Extract model info from args/kwargs
            model_name = kwargs.get('model', 'unknown')
            provider = kwargs.get('provider', 'unknown')
            
            try:
                # Call original method
                result = await llm_client_method(*args, **kwargs)
                
                # Log successful LLM call
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Extract token usage if available
                prompt_tokens = getattr(result, 'prompt_tokens', None)
                completion_tokens = getattr(result, 'completion_tokens', None)
                estimated_cost = getattr(result, 'estimated_cost', None)
                
                if self.config.log_llm_responses:
                    await self.log_llm_call(
                        model_name=model_name,
                        provider=provider,
                        session_id=session_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        estimated_cost_usd=estimated_cost
                    )
                
                return result
                
            except Exception as e:
                # Log LLM call error
                if self.config.log_exceptions:
                    await self.log_error(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        component="llm_client",
                        session_id=session_id,
                        context_metadata={
                            'model': model_name,
                            'provider': provider,
                            'duration_ms': (time.perf_counter() - start_time) * 1000
                        }
                    )
                raise
        
        return wrapped_llm_call
    
    # Decorator for automatic error logging
    
    def log_errors(self, component: str = "unknown"):
        """Decorator to automatically log errors from methods."""
        
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if self.config.log_exceptions:
                        session_id = current_session_id.get() or "unknown"
                        await self.log_error(
                            error_type=type(e).__name__,
                            error_message=str(e),
                            component=component,
                            session_id=session_id,
                            context_metadata={
                                'function': func.__name__,
                                'args_count': len(args),
                                'kwargs_keys': list(kwargs.keys())
                            }
                        )
                    raise
            return wrapper
        return decorator
    
    # Performance monitoring decorator
    
    def monitor_performance(self, operation_name: str = "unknown"):
        """Decorator to monitor and log performance metrics."""
        
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.config.log_response_times:
                    return await func(*args, **kwargs)
                
                start_time = time.perf_counter()
                memory_before = None
                
                if self.config.monitor_memory_usage:
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_before = process.memory_info().rss / 1024 / 1024
                    except ImportError:
                        pass
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Log performance metrics
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    session_id = current_session_id.get() or "unknown"
                    
                    performance_data = {
                        'operation': operation_name,
                        'duration_ms': duration_ms,
                        'success': True
                    }
                    
                    if memory_before and self.config.monitor_memory_usage:
                        try:
                            import psutil
                            process = psutil.Process()
                            memory_after = process.memory_info().rss / 1024 / 1024
                            performance_data['memory_delta_mb'] = memory_after - memory_before
                        except ImportError:
                            pass
                    
                    event = PerformanceMetricsEvent(
                        session_id=session_id,
                        context_metadata=performance_data
                    )
                    
                    await self._log_event(event)
                    
                    return result
                    
                except Exception as e:
                    # Log failed operation
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    session_id = current_session_id.get() or "unknown"
                    
                    event = PerformanceMetricsEvent(
                        session_id=session_id,
                        context_metadata={
                            'operation': operation_name,
                            'duration_ms': duration_ms,
                            'success': False,
                            'error_type': type(e).__name__
                        }
                    )
                    
                    await self._log_event(event)
                    raise
            
            return wrapper
        return decorator
    
    # High-level logging methods
    
    async def log_user_interaction(
        self,
        user_input: str,
        assistant_response: str,
        session_id: Optional[str] = None,
        interaction_mode: str = "chat",
        response_time_ms: Optional[float] = None,
        **metadata
    ):
        """Log a user interaction event."""
        if not self.config.log_user_inputs:
            return
        
        session_id = session_id or current_session_id.get() or "unknown"
        
        event = UserInteractionEvent(
            user_input=user_input,
            assistant_response=assistant_response,
            session_id=session_id,
            interaction_mode=interaction_mode,
            response_time_ms=response_time_ms,
            context_metadata=metadata
        )
        
        await self._log_event(event)
    
    async def log_agent_delegation(
        self,
        source_agent_id: str,
        target_agent_id: str,
        task_description: str,
        delegation_reason: str = "",
        session_id: Optional[str] = None,
        **metadata
    ):
        """Log an agent delegation event."""
        if not self.config.log_delegations:
            return
        
        session_id = session_id or current_session_id.get() or "unknown"
        
        event = AgentDelegationEvent(
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            task_description=task_description,
            delegation_reason=delegation_reason,
            session_id=session_id,
            context_metadata=metadata
        )
        
        await self._log_event(event)
    
    async def log_llm_call(
        self,
        model_name: str,
        provider: str,
        session_id: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        estimated_cost_usd: Optional[float] = None,
        **metadata
    ):
        """Log an LLM API call event."""
        if not self.config.log_llm_requests:
            return
        
        session_id = session_id or current_session_id.get() or "unknown"
        
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
        
        await self._log_event(event)
    
    async def log_error(
        self,
        error_type: str,
        error_message: str,
        component: str = "unknown",
        session_id: Optional[str] = None,
        severity: EventSeverity = EventSeverity.ERROR,
        **metadata
    ):
        """Log an error event."""
        if not self.config.log_exceptions:
            return
        
        session_id = session_id or current_session_id.get() or "unknown"
        
        event = ErrorEvent(
            error_type=error_type,
            error_message=error_message,
            component=component,
            session_id=session_id,
            severity=severity,
            context_metadata=metadata
        )
        
        await self._log_event(event, priority=severity)
    
    async def log_context_change(
        self,
        context_type: str,
        previous_state: Optional[Dict[str, Any]],
        new_state: Dict[str, Any],
        change_trigger: str = "",
        session_id: Optional[str] = None,
        **metadata
    ):
        """Log a context change event."""
        if not self.config.log_context_changes:
            return
        
        session_id = session_id or current_session_id.get() or "unknown"
        
        # Identify changed fields
        changed_fields = []
        if previous_state:
            for key, value in new_state.items():
                if key not in previous_state or previous_state[key] != value:
                    changed_fields.append(key)
        else:
            changed_fields = list(new_state.keys())
        
        event = ContextEvent(
            context_type=context_type,
            previous_state=previous_state,
            new_state=new_state,
            changed_fields=changed_fields,
            change_trigger=change_trigger,
            session_id=session_id,
            context_metadata=metadata
        )
        
        await self._log_event(event)
    
    # Internal methods
    
    async def _log_event(self, event, priority: EventSeverity = EventSeverity.INFO):
        """Internal method to log events with sampling and filtering."""
        try:
            # Apply sampling
            if self.config.sample_rate < 1.0:
                import random
                if random.random() > self.config.sample_rate:
                    return
            
            # Apply severity filtering
            severity_order = [
                EventSeverity.TRACE, EventSeverity.DEBUG, EventSeverity.INFO,
                EventSeverity.WARN, EventSeverity.ERROR, EventSeverity.CRITICAL
            ]
            
            if severity_order.index(priority) < severity_order.index(self.config.min_log_level):
                return
            
            # Log the event
            start_time = time.perf_counter()
            success = self.log_store.log_capture.log_event(event, priority)
            overhead_ms = (time.perf_counter() - start_time) * 1000
            
            # Update integration stats
            self.integration_stats['total_overhead_ms'] += overhead_ms
            if success:
                self.integration_stats['events_logged'] += 1
            else:
                self.integration_stats['integration_errors'] += 1
                
        except Exception as e:
            logger.error(f"Error in logging integration: {e}")
            self.integration_stats['integration_errors'] += 1
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration performance statistics."""
        return {
            **self.integration_stats,
            'active_integrations': self.active_integrations,
            'config': {
                'sample_rate': self.config.sample_rate,
                'min_log_level': self.config.min_log_level.value,
                'log_user_inputs': self.config.log_user_inputs,
                'log_delegations': self.config.log_delegations,
                'log_llm_requests': self.config.log_llm_requests,
            }
        }


# Utility functions for easy integration

def create_logging_integration(log_store: LogStore, config: Optional[IntegrationConfig] = None) -> LoggingIntegration:
    """Create a logging integration instance with default configuration."""
    return LoggingIntegration(log_store, config)


def integrate_with_chat_engine(chat_engine, integration: LoggingIntegration):
    """Integrate logging with a chat engine instance."""
    if hasattr(chat_engine, 'set_logging_integration'):
        chat_engine.set_logging_integration(integration)
        integration.active_integrations['chat_engine'] = True
        logger.info("Chat engine logging integration activated")
    else:
        logger.warning("Chat engine does not support logging integration")


def integrate_with_orchestrator(orchestrator, integration: LoggingIntegration):
    """Integrate logging with an orchestrator instance."""
    hooks = integration.create_orchestrator_hooks()
    
    # Add hooks to orchestrator if it supports them
    for hook_name, hook_func in hooks.items():
        if hasattr(orchestrator, f'add_{hook_name}'):
            getattr(orchestrator, f'add_{hook_name}')(hook_func)
            integration.active_integrations[f'orchestrator_{hook_name}'] = True
    
    logger.info(f"Orchestrator logging integration activated with {len(hooks)} hooks")


# Context manager for scoped logging
class LoggingContext:
    """Context manager for scoped logging with automatic context management."""
    
    def __init__(
        self,
        integration: LoggingIntegration,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        self.integration = integration
        self.session_id = session_id
        self.agent_id = agent_id
        self.user_id = user_id
        
        # Store previous context
        self.prev_session_id = None
        self.prev_agent_id = None
        self.prev_user_id = None
    
    def __enter__(self):
        # Store current context
        self.prev_session_id = current_session_id.get()
        self.prev_agent_id = current_agent_id.get()
        self.prev_user_id = current_user_id.get()
        
        # Set new context
        if self.session_id:
            current_session_id.set(self.session_id)
        if self.agent_id:
            current_agent_id.set(self.agent_id)
        if self.user_id:
            current_user_id.set(self.user_id)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        current_session_id.set(self.prev_session_id)
        current_agent_id.set(self.prev_agent_id)
        current_user_id.set(self.prev_user_id)
        
        # Log any exceptions that occurred
        if exc_type and self.integration.config.log_exceptions:
            asyncio.create_task(
                self.integration.log_error(
                    error_type=exc_type.__name__,
                    error_message=str(exc_val),
                    component="logging_context"
                )
            )