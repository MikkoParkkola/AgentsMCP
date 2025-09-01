"""Orchestrator Wrapper with Thinking Integration

This module provides a wrapper around the existing orchestrator that integrates
the thinking framework, enabling deliberative planning before all LLM interactions.

The wrapper intercepts orchestrator requests, applies thinking processes, and
then executes the enhanced plans through the original orchestrator.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, replace
from datetime import datetime

from ..orchestration.orchestrator import Orchestrator, OrchestratorConfig, OrchestratorResponse
from .thinking_framework import ThinkingFramework, ThinkingResult
from .planning_state_manager import PlanningStateManager, PlanningState
from .models import (
    ThinkingConfig, ThinkingPhase, ThinkingStep, PlanningIntegrationConfig,
    OrchestratorIntegrationMode, ThinkingScope, PerformanceProfile
)
from .exceptions import ThinkingIntegrationError


logger = logging.getLogger(__name__)


@dataclass
class ThinkingOrchestratorConfig:
    """Configuration for the thinking-enabled orchestrator."""
    
    # Base orchestrator config
    orchestrator_config: OrchestratorConfig = None
    
    # Thinking integration settings
    integration_mode: OrchestratorIntegrationMode = OrchestratorIntegrationMode.FULL_THINKING
    thinking_scope: ThinkingScope = ThinkingScope.ALL_REQUESTS
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED
    
    # Thinking configuration
    thinking_config: ThinkingConfig = None
    
    # State persistence settings
    enable_state_persistence: bool = True
    state_manager: Optional[PlanningStateManager] = None
    
    # Performance settings
    thinking_timeout_ms: int = 5000  # Max time for thinking process
    simple_request_threshold: int = 50  # Character count for simple requests
    bypass_thinking_for_simple: bool = True
    
    # Integration behavior
    fallback_on_thinking_failure: bool = True
    enhance_orchestrator_responses: bool = True
    preserve_thinking_context: bool = True
    
    # Monitoring and debugging
    log_thinking_steps: bool = False
    include_thinking_metadata: bool = True
    performance_monitoring: bool = True


class ThinkingOrchestrator:
    """Orchestrator enhanced with deliberative thinking capabilities.
    
    This wrapper adds thinking processes to the orchestrator while maintaining
    full compatibility with the existing orchestrator interface.
    """
    
    def __init__(self, config: ThinkingOrchestratorConfig = None):
        """Initialize the thinking-enabled orchestrator."""
        self.config = config or ThinkingOrchestratorConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize base orchestrator
        orch_config = self.config.orchestrator_config or OrchestratorConfig()
        self.orchestrator = Orchestrator(orch_config)
        
        # Initialize thinking components
        thinking_config = self.config.thinking_config or ThinkingConfig()
        self.thinking_framework = ThinkingFramework(thinking_config)
        
        # Initialize state manager if enabled
        if self.config.enable_state_persistence:
            self.state_manager = self.config.state_manager or PlanningStateManager()
        else:
            self.state_manager = None
        
        # Performance monitoring
        self.thinking_metrics = {
            "total_requests": 0,
            "thinking_enabled_requests": 0,
            "thinking_bypassed_requests": 0,
            "thinking_failures": 0,
            "avg_thinking_time_ms": 0,
            "avg_total_time_ms": 0
        }
        
        # Context preservation
        self.active_contexts: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"Thinking orchestrator initialized with mode: {self.config.integration_mode.value}")
    
    async def process_user_input(self, 
                               user_input: str, 
                               context: Dict = None,
                               thinking_override: Optional[ThinkingConfig] = None) -> OrchestratorResponse:
        """Process user input with integrated thinking capabilities.
        
        This method wraps the original orchestrator's process_user_input method,
        adding deliberative planning before execution.
        """
        start_time = time.time()
        self.thinking_metrics["total_requests"] += 1
        context = context or {}
        
        try:
            # Determine if thinking should be applied
            should_think = await self._should_apply_thinking(user_input, context)
            
            if not should_think:
                self.thinking_metrics["thinking_bypassed_requests"] += 1
                self.logger.debug("Bypassing thinking for simple request")
                return await self.orchestrator.process_user_input(user_input, context)
            
            # Apply thinking process
            self.thinking_metrics["thinking_enabled_requests"] += 1
            self.logger.info("Applying thinking process to user input")
            
            # Create thinking context
            thinking_context = await self._create_thinking_context(user_input, context)
            
            # Execute thinking process
            thinking_start = time.time()
            thinking_result = await self._execute_thinking_process(
                user_input, 
                thinking_context, 
                thinking_override
            )
            thinking_duration = (time.time() - thinking_start) * 1000
            
            # Process thinking result through orchestrator
            enhanced_response = await self._process_thinking_result(
                thinking_result, 
                user_input, 
                context
            )
            
            # Enhance response with thinking metadata if enabled
            if self.config.include_thinking_metadata:
                enhanced_response = await self._enhance_response_with_thinking_metadata(
                    enhanced_response,
                    thinking_result,
                    thinking_duration
                )
            
            # Update performance metrics
            total_duration = (time.time() - start_time) * 1000
            self._update_performance_metrics(thinking_duration, total_duration)
            
            # Save state if persistence is enabled
            if self.state_manager and self.config.preserve_thinking_context:
                await self._save_thinking_state(thinking_result, context)
            
            self.logger.info(f"Request processed with thinking in {total_duration:.2f}ms")
            return enhanced_response
            
        except Exception as e:
            self.thinking_metrics["thinking_failures"] += 1
            self.logger.error(f"Error in thinking process: {e}", exc_info=True)
            
            if self.config.fallback_on_thinking_failure:
                self.logger.warning("Falling back to standard orchestrator processing")
                return await self.orchestrator.process_user_input(user_input, context)
            else:
                # Return error response
                return OrchestratorResponse(
                    content="I encountered an issue with my thinking process. Please try rephrasing your request.",
                    response_type="error",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    metadata={"thinking_error": str(e)}
                )
    
    async def run_until_goal(self, 
                           goal: str, 
                           initial_context: Optional[Dict] = None,
                           thinking_session_id: Optional[str] = None) -> OrchestratorResponse:
        """Run with thinking capabilities until goal is achieved.
        
        This method extends the orchestrator's run_until_goal method with
        persistent thinking context across multiple interactions.
        """
        context = initial_context or {}
        
        # Create or restore thinking session
        if thinking_session_id and self.state_manager:
            session_state = await self.state_manager.load_state(thinking_session_id)
            if session_state:
                context.update(session_state.current_context)
                self.logger.info(f"Restored thinking session: {thinking_session_id}")
        
        # Add thinking session tracking
        session_id = thinking_session_id or f"session_{int(time.time())}"
        context["thinking_session_id"] = session_id
        context["goal_tracking"] = {
            "original_goal": goal,
            "steps_taken": [],
            "context_evolution": []
        }
        
        try:
            # Execute with enhanced goal tracking
            response = await self.orchestrator.run_until_goal(goal, context)
            
            # Save final session state
            if self.state_manager and session_id in context:
                await self._finalize_thinking_session(session_id, response, context)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in goal execution with thinking: {e}")
            raise ThinkingIntegrationError(f"Goal execution failed: {e}") from e
    
    async def _should_apply_thinking(self, user_input: str, context: Dict) -> bool:
        """Determine whether thinking should be applied to this request."""
        # Check integration mode
        if self.config.integration_mode == OrchestratorIntegrationMode.DISABLED:
            return False
        
        # Check thinking scope
        if self.config.thinking_scope == ThinkingScope.COMPLEX_ONLY:
            # Simple heuristics for complexity
            if (len(user_input) < self.config.simple_request_threshold and
                not any(word in user_input.lower() for word in 
                       ["implement", "create", "design", "analyze", "plan", "strategy"])):
                return False
        
        # Check if bypass is enabled for simple requests
        if (self.config.bypass_thinking_for_simple and 
            len(user_input) < self.config.simple_request_threshold):
            return False
        
        # Check performance profile constraints
        if self.config.performance_profile == PerformanceProfile.FAST:
            # Apply thinking only to clearly complex requests
            if not any(word in user_input.lower() for word in 
                      ["implement", "create", "design", "complex", "multiple"]):
                return False
        
        return True
    
    async def _create_thinking_context(self, user_input: str, context: Dict) -> Dict[str, Any]:
        """Create enhanced thinking context from orchestrator context."""
        thinking_context = {
            "user_request": user_input,
            "orchestrator_context": context,
            "available_agents": getattr(self.orchestrator, 'active_agents', {}),
            "orchestrator_stats": self.orchestrator.get_stats(),
            "thinking_session_id": context.get("thinking_session_id"),
            "performance_constraints": {
                "max_thinking_time_ms": self.config.thinking_timeout_ms,
                "performance_profile": self.config.performance_profile.value
            }
        }
        
        # Add preserved context if available
        session_id = context.get("thinking_session_id")
        if session_id and session_id in self.active_contexts:
            thinking_context["previous_thinking"] = self.active_contexts[session_id]
        
        return thinking_context
    
    async def _execute_thinking_process(self, 
                                     user_input: str,
                                     thinking_context: Dict[str, Any],
                                     thinking_override: Optional[ThinkingConfig]) -> ThinkingResult:
        """Execute the thinking process with timeout protection."""
        
        # Create progress callback for monitoring
        progress_callback = None
        if self.config.log_thinking_steps:
            progress_callback = self._log_thinking_progress
        
        # Apply thinking configuration override
        config = thinking_override or self.thinking_framework.config
        if self.config.thinking_timeout_ms:
            config = replace(config, timeout_seconds=self.config.thinking_timeout_ms / 1000)
        
        try:
            # Execute thinking with timeout
            thinking_result = await asyncio.wait_for(
                self.thinking_framework.think(
                    user_input,
                    thinking_context,
                    config,
                    progress_callback
                ),
                timeout=self.config.thinking_timeout_ms / 1000
            )
            
            return thinking_result
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Thinking process timed out after {self.config.thinking_timeout_ms}ms")
            # Return minimal thinking result
            return ThinkingResult(
                request=user_input,
                final_approach=None,
                execution_plan=None,
                confidence=0.3,
                thinking_trace=[],
                total_duration_ms=self.config.thinking_timeout_ms,
                context=thinking_context,
                metadata={"timeout": True}
            )
    
    async def _process_thinking_result(self, 
                                    thinking_result: ThinkingResult,
                                    original_input: str,
                                    context: Dict) -> OrchestratorResponse:
        """Process the thinking result through the orchestrator."""
        
        if not thinking_result.execution_plan:
            # No execution plan generated, use original orchestrator
            self.logger.debug("No execution plan generated, using standard processing")
            return await self.orchestrator.process_user_input(original_input, context)
        
        # Extract enhanced request from thinking result
        enhanced_request = self._extract_enhanced_request(thinking_result, original_input)
        
        # Add thinking context to orchestrator context
        enhanced_context = {
            **context,
            "thinking_result": thinking_result,
            "execution_plan": thinking_result.execution_plan,
            "thinking_confidence": thinking_result.confidence,
            "approach_selected": thinking_result.final_approach
        }
        
        # Process through orchestrator
        orchestrator_response = await self.orchestrator.process_user_input(
            enhanced_request, 
            enhanced_context
        )
        
        return orchestrator_response
    
    def _extract_enhanced_request(self, thinking_result: ThinkingResult, original_input: str) -> str:
        """Extract an enhanced request from the thinking result."""
        
        if not thinking_result.execution_plan:
            return original_input
        
        # Build enhanced request based on execution plan
        enhanced_parts = [original_input]
        
        if thinking_result.final_approach:
            enhanced_parts.append(f"Approach: {thinking_result.final_approach.description}")
        
        if hasattr(thinking_result.execution_plan, 'scheduled_tasks'):
            task_descriptions = []
            for task in thinking_result.execution_plan.scheduled_tasks[:3]:  # Limit to avoid overwhelming
                task_descriptions.append(f"- {task.description}")
            
            if task_descriptions:
                enhanced_parts.append("Key tasks:")
                enhanced_parts.extend(task_descriptions)
        
        return "\n".join(enhanced_parts)
    
    async def _enhance_response_with_thinking_metadata(self,
                                                     response: OrchestratorResponse,
                                                     thinking_result: ThinkingResult,
                                                     thinking_duration: float) -> OrchestratorResponse:
        """Enhance orchestrator response with thinking metadata."""
        
        enhanced_metadata = response.metadata or {}
        
        # Add thinking information
        enhanced_metadata.update({
            "thinking_applied": True,
            "thinking_duration_ms": thinking_duration,
            "thinking_confidence": thinking_result.confidence,
            "thinking_phases_completed": len(thinking_result.thinking_trace),
            "approach_selected": thinking_result.final_approach.name if thinking_result.final_approach else None,
            "execution_plan_tasks": len(thinking_result.execution_plan.scheduled_tasks) if thinking_result.execution_plan else 0
        })
        
        # Add thinking trace summary if requested
        if self.config.log_thinking_steps and thinking_result.thinking_trace:
            enhanced_metadata["thinking_summary"] = [
                {
                    "phase": step.phase.value,
                    "duration_ms": step.duration_ms,
                    "success": not hasattr(step, 'error') or not step.error
                }
                for step in thinking_result.thinking_trace[-5:]  # Last 5 steps
            ]
        
        return replace(response, metadata=enhanced_metadata)
    
    async def _save_thinking_state(self, thinking_result: ThinkingResult, context: Dict):
        """Save thinking state for persistence and recovery."""
        if not self.state_manager:
            return
        
        try:
            session_id = context.get("thinking_session_id")
            if not session_id:
                return
            
            # Create planning state
            planning_state = PlanningState(
                state_id=session_id,
                created_at=datetime.now(),
                thinking_trace=thinking_result.thinking_trace,
                current_context=thinking_result.context or {},
                metadata={
                    "confidence": thinking_result.confidence,
                    "duration_ms": thinking_result.total_duration_ms,
                    "approach": thinking_result.final_approach.name if thinking_result.final_approach else None
                }
            )
            
            await self.state_manager.save_state(planning_state)
            
            # Update active contexts
            self.active_contexts[session_id] = thinking_result.context or {}
            
        except Exception as e:
            self.logger.warning(f"Failed to save thinking state: {e}")
    
    async def _finalize_thinking_session(self, 
                                       session_id: str, 
                                       final_response: OrchestratorResponse,
                                       context: Dict):
        """Finalize and clean up thinking session."""
        try:
            if self.state_manager:
                # Update final state
                session_state = await self.state_manager.load_state(session_id)
                if session_state:
                    session_state.metadata.update({
                        "completed": True,
                        "final_response_type": final_response.response_type,
                        "final_confidence": final_response.confidence,
                        "goal_achieved": final_response.response_type == "goal_completed"
                    })
                    await self.state_manager.save_state(session_state)
            
            # Clean up active context
            if session_id in self.active_contexts:
                del self.active_contexts[session_id]
                
        except Exception as e:
            self.logger.warning(f"Failed to finalize thinking session {session_id}: {e}")
    
    def _log_thinking_progress(self, step: ThinkingStep):
        """Log thinking progress for monitoring."""
        self.logger.debug(
            f"Thinking step: {step.phase.value} - "
            f"Duration: {step.duration_ms}ms - "
            f"Success: {not hasattr(step, 'error') or not step.error}"
        )
    
    def _update_performance_metrics(self, thinking_duration: float, total_duration: float):
        """Update performance metrics."""
        # Update running averages
        total_requests = self.thinking_metrics["total_requests"]
        
        prev_thinking_avg = self.thinking_metrics["avg_thinking_time_ms"]
        self.thinking_metrics["avg_thinking_time_ms"] = (
            (prev_thinking_avg * (total_requests - 1) + thinking_duration) / total_requests
        )
        
        prev_total_avg = self.thinking_metrics["avg_total_time_ms"]
        self.thinking_metrics["avg_total_time_ms"] = (
            (prev_total_avg * (total_requests - 1) + total_duration) / total_requests
        )
    
    async def get_thinking_stats(self) -> Dict[str, Any]:
        """Get thinking integration statistics."""
        base_stats = self.orchestrator.get_stats()
        
        thinking_stats = {
            **base_stats,
            "thinking_integration": {
                **self.thinking_metrics,
                "active_contexts": len(self.active_contexts),
                "state_persistence_enabled": self.config.enable_state_persistence,
                "integration_mode": self.config.integration_mode.value,
                "performance_profile": self.config.performance_profile.value
            }
        }
        
        # Add state manager stats if available
        if self.state_manager:
            storage_stats = await self.state_manager.get_storage_stats()
            thinking_stats["thinking_integration"]["storage_stats"] = storage_stats
        
        return thinking_stats
    
    async def cleanup_old_thinking_states(self) -> int:
        """Clean up old thinking states."""
        if not self.state_manager:
            return 0
        
        return await self.state_manager.cleanup_old_states()
    
    async def shutdown(self):
        """Shutdown the thinking orchestrator and clean up resources."""
        self.logger.info("Shutting down thinking orchestrator")
        
        # Shutdown state manager
        if self.state_manager:
            await self.state_manager.shutdown()
        
        # Shutdown base orchestrator
        await self.orchestrator.shutdown()
        
        # Clear active contexts
        self.active_contexts.clear()
        
        self.logger.info("Thinking orchestrator shutdown complete")


# Convenience functions for common use cases
def create_thinking_orchestrator(
    thinking_enabled: bool = True,
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED,
    enable_persistence: bool = True
) -> ThinkingOrchestrator:
    """Create a thinking orchestrator with common configuration."""
    
    config = ThinkingOrchestratorConfig(
        integration_mode=OrchestratorIntegrationMode.FULL_THINKING if thinking_enabled 
                        else OrchestratorIntegrationMode.DISABLED,
        performance_profile=performance_profile,
        enable_state_persistence=enable_persistence,
        thinking_config=ThinkingConfig(
            performance_mode=performance_profile.value,
            enable_lightweight_mode=performance_profile == PerformanceProfile.FAST
        )
    )
    
    return ThinkingOrchestrator(config)


def create_fast_thinking_orchestrator() -> ThinkingOrchestrator:
    """Create a fast thinking orchestrator optimized for speed."""
    return create_thinking_orchestrator(
        performance_profile=PerformanceProfile.FAST,
        enable_persistence=False
    )


def create_comprehensive_thinking_orchestrator() -> ThinkingOrchestrator:
    """Create a comprehensive thinking orchestrator optimized for quality."""
    return create_thinking_orchestrator(
        performance_profile=PerformanceProfile.COMPREHENSIVE,
        enable_persistence=True
    )