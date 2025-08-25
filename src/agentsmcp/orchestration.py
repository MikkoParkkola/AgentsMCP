"""
agentsmcp.orchestration
=======================

Lightweight asyncio-based event bus for agent orchestration.
Implements publisher/subscriber pattern with backpressure control.
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Type,
    TypeVar,
)


T = TypeVar("T", bound="Event")


class Event(ABC):
    """Base class for all orchestration events."""
    pass


@dataclass(frozen=True)
class JobStarted(Event):
    """Emitted when a job begins execution."""
    job_id: str
    agent_type: str
    task: str
    timestamp: datetime


@dataclass(frozen=True)
class JobCompleted(Event):
    """Emitted when a job completes successfully."""
    job_id: str
    result: Any
    duration: float
    timestamp: datetime


@dataclass(frozen=True)
class JobFailed(Event):
    """Emitted when a job fails with an error."""
    job_id: str
    error: Exception
    timestamp: datetime


@dataclass(frozen=True)
class AgentSpawned(Event):
    """Emitted when a new agent is spawned."""
    agent_id: str
    agent_type: str
    config: Mapping[str, Any]
    timestamp: datetime


@dataclass(frozen=True)
class AgentTerminated(Event):
    """Emitted when an agent is terminated."""
    agent_id: str
    reason: str
    timestamp: datetime


@dataclass(frozen=True)
class ResourceLimitExceeded(Event):
    """Emitted when resource limits are exceeded."""
    resource_type: str  # cpu, memory, queue
    current_value: float
    limit_value: float
    timestamp: datetime


class EventBus:
    """
    AsyncIO event bus supporting typed events and backpressure.
    
    Subscribers receive events via dedicated bounded queues. When a queue
    reaches capacity, publishing blocks until space becomes available,
    providing flow control.
    """
    
    def __init__(self, *, max_queue_size: int = 100) -> None:
        """
        Initialize the event bus.
        
        Parameters
        ----------
        max_queue_size: Maximum events per subscriber queue (backpressure limit)
        """
        self._max_queue_size = max_queue_size
        self._subscribers: Dict[Type[Event], List[_Subscription]] = {}
        self._lock = asyncio.Lock()
        self._tasks: List[asyncio.Task] = []
        
    async def subscribe(
        self,
        event_type: Type[T],
        handler: Callable[[T], Awaitable[None]],
    ) -> None:
        """
        Register an async handler for the specified event type.
        
        Parameters
        ----------
        event_type: Type of events to handle
        handler: Async function to call for each event
        
        Raises
        ------
        TypeError: If handler is not an async function
        """
        if not inspect.iscoroutinefunction(handler):
            raise TypeError(f"Handler {handler.__name__} must be an async function")
        
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._max_queue_size)
        
        async def _consumer() -> None:
            """Process events from the queue."""
            try:
                while True:
                    event = await queue.get()
                    if event is None:  # Shutdown signal
                        break
                    try:
                        await handler(event)  # type: ignore[arg-type]
                    except Exception as e:
                        # Log but don't crash the consumer
                        print(f"Handler error: {e}")
            except asyncio.CancelledError:
                pass
        
        task = asyncio.create_task(_consumer())
        self._tasks.append(task)
        
        subscription = _Subscription(queue, handler, task)
        
        async with self._lock:
            self._subscribers.setdefault(event_type, []).append(subscription)
    
    async def unsubscribe(
        self,
        event_type: Type[T],
        handler: Callable[[T], Awaitable[None]],
    ) -> None:
        """
        Remove a previously registered handler.
        
        Parameters
        ----------
        event_type: Event type to unsubscribe from
        handler: Handler to remove
        """
        async with self._lock:
            subs = self._subscribers.get(event_type, [])
            for sub in subs:
                if sub.handler is handler:
                    await sub.queue.put(None)  # Signal shutdown
                    sub.task.cancel()
                    subs.remove(sub)
                    break
            
            if not subs:
                self._subscribers.pop(event_type, None)
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event to all matching subscribers.
        
        Blocks if any subscriber queue is full (backpressure).
        
        Parameters
        ----------
        event: Event to publish
        """
        async with self._lock:
            # Find all subscribers for this event type and its supertypes
            queues = []
            for event_type, subs in self._subscribers.items():
                if isinstance(event, event_type):
                    queues.extend(sub.queue for sub in subs)
        
        # Publish to all matching queues (may block on full queues)
        if queues:
            await asyncio.gather(*(q.put(event) for q in queues))
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the event bus, canceling all consumers."""
        async with self._lock:
            # Send shutdown signal to all queues
            for subs in self._subscribers.values():
                for sub in subs:
                    await sub.queue.put(None)
            
            # Cancel all consumer tasks
            for task in self._tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Clear state
            self._subscribers.clear()
            self._tasks.clear()


@dataclass
class _Subscription:
    """Internal subscription tracking."""
    queue: asyncio.Queue[Event]
    handler: Callable[[Event], Awaitable[None]]
    task: asyncio.Task