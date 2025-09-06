"""
Coach Integration Layer

Connects the process coach orchestration system with existing AgentsMCP systems
including the chat engine, UI components, and existing orchestration systems.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

from .process_coach import ProcessCoach, ProcessCoachConfig
from .retrospective_orchestrator import RetrospectiveOrchestrator
from .improvement_coordinator import ImprovementCoordinator, ActionPoint
from .agent_feedback_system import AgentFeedbackSystem
from .continuous_improvement_engine import ContinuousImprovementEngine

logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    """Types of system integrations."""
    CHAT_ENGINE = "chat_engine"
    UI_COMPONENT = "ui_component"
    ORCHESTRATOR = "orchestrator"
    AGENT_SYSTEM = "agent_system"
    MONITORING = "monitoring"
    FEEDBACK = "feedback"

@dataclass
class SystemIntegration:
    """Configuration for a system integration."""
    integration_id: str
    integration_type: IntegrationType
    system_name: str
    callback_handlers: Dict[str, Callable] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    priority: int = 5  # 1-10, higher = more important
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class IntegrationEvent:
    """Events that flow between systems."""
    event_id: str
    event_type: str
    source_system: str
    target_system: Optional[str]
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None

class CoachIntegrationManager:
    """
    Manages integration between the process coach orchestration system
    and existing AgentsMCP systems.
    """
    
    def __init__(self):
        self.process_coach: Optional[ProcessCoach] = None
        self.retrospective_orchestrator: Optional[RetrospectiveOrchestrator] = None
        self.improvement_coordinator: Optional[ImprovementCoordinator] = None
        self.agent_feedback_system: Optional[AgentFeedbackSystem] = None
        self.continuous_improvement_engine: Optional[ContinuousImprovementEngine] = None
        
        self.integrations: Dict[str, SystemIntegration] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.integration_metrics: Dict[str, Dict[str, Any]] = {}
        self._is_initialized = False
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_processor_task: Optional[asyncio.Task] = None
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the coach integration system."""
        try:
            logger.info("Initializing CoachIntegrationManager")
            
            # Initialize core components
            coach_config = ProcessCoachConfig()
            if config and "process_coach" in config:
                # Apply config overrides
                for key, value in config["process_coach"].items():
                    if hasattr(coach_config, key):
                        setattr(coach_config, key, value)
            
            self.process_coach = ProcessCoach(coach_config)
            self.retrospective_orchestrator = RetrospectiveOrchestrator()
            self.improvement_coordinator = ImprovementCoordinator()
            self.agent_feedback_system = AgentFeedbackSystem()
            self.continuous_improvement_engine = ContinuousImprovementEngine()
            
            # Connect components
            await self._connect_core_components()
            
            # Start event processor
            self._event_processor_task = asyncio.create_task(self._process_events())
            
            # Register default integrations
            await self._register_default_integrations()
            
            self._is_initialized = True
            logger.info("CoachIntegrationManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CoachIntegrationManager: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the integration system."""
        logger.info("Shutting down CoachIntegrationManager")
        
        # Stop event processor
        if self._event_processor_task:
            self._event_processor_task.cancel()
            try:
                await self._event_processor_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown continuous improvement engine
        if self.continuous_improvement_engine:
            await self.continuous_improvement_engine.stop_continuous_improvement()
            self.continuous_improvement_engine.cleanup()
        
        self._is_initialized = False
        logger.info("CoachIntegrationManager shutdown complete")
    
    async def register_integration(self, integration: SystemIntegration) -> bool:
        """Register a new system integration."""
        try:
            self.integrations[integration.integration_id] = integration
            self.integration_metrics[integration.integration_id] = {
                "events_processed": 0,
                "errors": 0,
                "last_activity": None,
                "avg_processing_time": 0.0
            }
            
            # Register event handlers
            for event_type, handler in integration.callback_handlers.items():
                if event_type not in self.event_handlers:
                    self.event_handlers[event_type] = []
                self.event_handlers[event_type].append(handler)
            
            logger.info(f"Registered integration: {integration.integration_id} ({integration.system_name})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register integration {integration.integration_id}: {e}")
            return False
    
    async def emit_event(self, event: IntegrationEvent):
        """Emit an event to the integration system."""
        await self._event_queue.put(event)
    
    async def handle_task_completion(self, task_result: Dict[str, Any]) -> str:
        """Handle task completion and trigger process coach workflow."""
        if not self._is_initialized or not self.process_coach:
            logger.warning("CoachIntegrationManager not initialized")
            return ""
        
        try:
            # Emit task completion event
            event = IntegrationEvent(
                event_id=str(uuid.uuid4()),
                event_type="task_completed",
                source_system="chat_engine",
                target_system="process_coach",
                payload=task_result
            )
            await self.emit_event(event)
            
            # Trigger process coach
            trigger_context = {
                "source": "task_completion",
                "task_result": task_result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            cycle_id = await self.process_coach.trigger_improvement_cycle(trigger_context)
            logger.info(f"Started improvement cycle: {cycle_id}")
            return cycle_id
            
        except Exception as e:
            logger.error(f"Error handling task completion: {e}")
            return ""
    
    async def handle_user_feedback(self, feedback: Dict[str, Any]) -> bool:
        """Handle user feedback for system improvement."""
        if not self._is_initialized or not self.agent_feedback_system:
            return False
        
        try:
            # Emit feedback event
            event = IntegrationEvent(
                event_id=str(uuid.uuid4()),
                event_type="user_feedback",
                source_system="ui_component",
                target_system="feedback_system",
                payload=feedback
            )
            await self.emit_event(event)
            
            # Process feedback through agent feedback system
            await self.agent_feedback_system.record_feedback(
                agent_id=feedback.get("agent_id", "system"),
                feedback_type=feedback.get("type", "general"),
                rating=feedback.get("rating"),
                comment=feedback.get("comment", ""),
                metadata=feedback.get("metadata", {})
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling user feedback: {e}")
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self._is_initialized:
            return {"status": "not_initialized"}
        
        status = {
            "status": "active",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {},
            "integrations": {},
            "metrics": self.integration_metrics
        }
        
        # Component status
        if self.process_coach:
            status["components"]["process_coach"] = {
                "active": self.process_coach.is_active,
                "active_cycles": len(getattr(self.process_coach, 'active_cycles', {}))
            }
        
        if self.continuous_improvement_engine:
            engine_status = await self.continuous_improvement_engine.get_system_evolution_status()
            status["components"]["improvement_engine"] = engine_status
        
        # Integration status
        for integration_id, integration in self.integrations.items():
            status["integrations"][integration_id] = {
                "active": integration.is_active,
                "system_name": integration.system_name,
                "type": integration.integration_type.value,
                "priority": integration.priority
            }
        
        return status
    
    async def _connect_core_components(self):
        """Connect the core coach orchestration components."""
        if not all([
            self.process_coach,
            self.retrospective_orchestrator,
            self.improvement_coordinator,
            self.agent_feedback_system,
            self.continuous_improvement_engine
        ]):
            raise ValueError("Core components not initialized")
        
        # Connect process coach to other components
        self.process_coach.retrospective_orchestrator = self.retrospective_orchestrator
        self.process_coach.improvement_coordinator = self.improvement_coordinator
        self.process_coach.agent_feedback_system = self.agent_feedback_system
        
        # Start continuous improvement engine
        await self.continuous_improvement_engine.start_continuous_improvement()
        
        logger.info("Core components connected successfully")
    
    async def _register_default_integrations(self):
        """Register default system integrations."""
        
        # Chat Engine Integration
        chat_integration = SystemIntegration(
            integration_id="chat_engine",
            integration_type=IntegrationType.CHAT_ENGINE,
            system_name="AgentsMCP Chat Engine",
            callback_handlers={
                "task_completed": self._handle_chat_task_completion,
                "conversation_ended": self._handle_conversation_end
            },
            priority=10  # Highest priority
        )
        await self.register_integration(chat_integration)
        
        # UI Component Integration
        ui_integration = SystemIntegration(
            integration_id="ui_components",
            integration_type=IntegrationType.UI_COMPONENT,
            system_name="AgentsMCP UI Components",
            callback_handlers={
                "user_feedback": self._handle_ui_feedback,
                "user_interaction": self._handle_user_interaction
            },
            priority=8
        )
        await self.register_integration(ui_integration)
        
        # Monitoring Integration
        monitoring_integration = SystemIntegration(
            integration_id="monitoring",
            integration_type=IntegrationType.MONITORING,
            system_name="System Monitoring",
            callback_handlers={
                "performance_alert": self._handle_performance_alert,
                "error_event": self._handle_error_event
            },
            priority=9
        )
        await self.register_integration(monitoring_integration)
    
    async def _process_events(self):
        """Process events from the event queue."""
        while True:
            try:
                event = await self._event_queue.get()
                await self._handle_event(event)
                self._event_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_event(self, event: IntegrationEvent):
        """Handle an individual event."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Find handlers for this event type
            handlers = self.event_handlers.get(event.event_type, [])
            
            if not handlers:
                logger.debug(f"No handlers found for event type: {event.event_type}")
                return
            
            # Execute handlers
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Handler error for event {event.event_id}: {e}")
            
            # Update metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._update_metrics(event, processing_time, success=True)
            
        except Exception as e:
            logger.error(f"Error handling event {event.event_id}: {e}")
            self._update_metrics(event, 0, success=False)
    
    def _update_metrics(self, event: IntegrationEvent, processing_time: float, success: bool):
        """Update integration metrics."""
        # Find the integration that generated this event
        integration_id = None
        for int_id, integration in self.integrations.items():
            if integration.system_name.lower().replace(" ", "_") in event.source_system:
                integration_id = int_id
                break
        
        if not integration_id:
            integration_id = "unknown"
        
        if integration_id not in self.integration_metrics:
            self.integration_metrics[integration_id] = {
                "events_processed": 0,
                "errors": 0,
                "last_activity": None,
                "avg_processing_time": 0.0
            }
        
        metrics = self.integration_metrics[integration_id]
        metrics["events_processed"] += 1
        metrics["last_activity"] = event.timestamp.isoformat()
        
        if success:
            # Update average processing time
            current_avg = metrics["avg_processing_time"]
            count = metrics["events_processed"]
            metrics["avg_processing_time"] = ((current_avg * (count - 1)) + processing_time) / count
        else:
            metrics["errors"] += 1
    
    # Default event handlers
    async def _handle_chat_task_completion(self, event: IntegrationEvent):
        """Handle chat engine task completion."""
        await self.handle_task_completion(event.payload)
    
    async def _handle_conversation_end(self, event: IntegrationEvent):
        """Handle conversation end."""
        if self.process_coach:
            # Trigger a retrospective cycle
            trigger_context = {
                "source": "conversation_end",
                "conversation_data": event.payload,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self.process_coach.trigger_improvement_cycle(trigger_context)
    
    async def _handle_ui_feedback(self, event: IntegrationEvent):
        """Handle UI feedback."""
        await self.handle_user_feedback(event.payload)
    
    async def _handle_user_interaction(self, event: IntegrationEvent):
        """Handle user interaction events."""
        if self.agent_feedback_system:
            # Record interaction data for agent performance tracking
            await self.agent_feedback_system.record_interaction(
                agent_id=event.payload.get("agent_id", "system"),
                interaction_type=event.payload.get("interaction_type", "unknown"),
                success=event.payload.get("success", True),
                duration=event.payload.get("duration", 0),
                metadata=event.payload.get("metadata", {})
            )
    
    async def _handle_performance_alert(self, event: IntegrationEvent):
        """Handle performance alerts."""
        if self.improvement_coordinator:
            # Create improvement action for performance issue
            action = ActionPoint(
                action_id=str(uuid.uuid4()),
                title=f"Performance Alert: {event.payload.get('alert_type', 'Unknown')}",
                description=event.payload.get('description', 'Performance alert triggered'),
                category="performance",
                priority=0.9,  # High priority
                assigned_to="system",
                context=event.payload
            )
            await self.improvement_coordinator.add_improvement(action)
    
    async def _handle_error_event(self, event: IntegrationEvent):
        """Handle error events."""
        if self.improvement_coordinator:
            # Create improvement action for error
            action = ActionPoint(
                action_id=str(uuid.uuid4()),
                title=f"Error Event: {event.payload.get('error_type', 'Unknown')}",
                description=event.payload.get('error_message', 'Error event occurred'),
                category="reliability",
                priority=0.8,  # High priority
                assigned_to="system",
                context=event.payload
            )
            await self.improvement_coordinator.add_improvement(action)

# Global integration manager instance
_integration_manager: Optional[CoachIntegrationManager] = None

async def get_integration_manager() -> CoachIntegrationManager:
    """Get the global integration manager instance."""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = CoachIntegrationManager()
        await _integration_manager.initialize()
    return _integration_manager

async def initialize_coach_integration(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the coach integration system."""
    manager = await get_integration_manager()
    return manager._is_initialized

async def shutdown_coach_integration():
    """Shutdown the coach integration system."""
    global _integration_manager
    if _integration_manager:
        await _integration_manager.shutdown()
        _integration_manager = None