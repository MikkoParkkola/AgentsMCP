'''Strict Orchestrator Implementation for AgentsMCP

This orchestrator enforces the architectural principle that ONLY the orchestrator 
communicates directly with users. All agent communications are internal.

Key Principles:
- User <-> Orchestrator ONLY (no direct agent-user communication allowed)
- Orchestrator <-> Agents (internal coordination) 
- Smart task classification to avoid unnecessary agent spawning
- Response synthesis for coherent user experience
- Communication isolation for clean architecture

The orchestrator acts as the single point of contact for users while coordinating
with agents behind the scenes to provide intelligent, consolidated responses.
'''

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .task_classifier import TaskClassifier, TaskClassification, ClassificationResult
from .response_synthesizer import ResponseSynthesizer, SynthesisStrategy
from .communication_interceptor import CommunicationInterceptor
from .intelligent_delegation import get_delegation_system, TaskType
from ..quality import get_quality_gate_system, QualityGateResult
from ..agents import get_agent_loader
from ..self_improvement import ContinuousOptimizer
from ..monitoring import MetricsCollector, AgentTracker, PerformanceMonitor

logger = logging.getLogger(__name__)


class OrchestratorMode(Enum):
    """Orchestrator operation modes."""
    STRICT_ISOLATION = "strict_isolation"  # No direct agent-user communication allowed
    SUPERVISED = "supervised"  # Agent communications monitored and filtered
    TRANSPARENT = "transparent"  # Agent communications visible but orchestrator-mediated


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    mode: OrchestratorMode = OrchestratorMode.STRICT_ISOLATION
    
    # Task classification settings
    enable_smart_classification: bool = True
    simple_task_threshold: float = 0.8
    single_agent_threshold: float = 0.6
    
    # Response synthesis settings
    default_synthesis_strategy: SynthesisStrategy = SynthesisStrategy.SUMMARIZE
    
    # Intelligent delegation settings
    enable_intelligent_delegation: bool = True
    enable_parallel_execution: bool = True
    max_parallel_agents: int = 4
    
    # Quality gate settings
    enable_quality_gates: bool = True
    require_quality_approval_for_critical_files: bool = True
    auto_backup_before_modifications: bool = True
    
    # Enhanced orchestration features
    auto_approve: bool = True
    max_retry_attempts: int = 3
    task_timeout_minutes: int = 5
    synthesis_timeout_ms: int = 2000
    
    # Communication settings
    intercept_all_agent_output: bool = True
    allow_agent_status_messages: bool = False
    consolidate_error_messages: bool = True
    
    # Performance settings
    max_agent_wait_time_ms: int = 30000
    max_parallel_agents: int = 8
    
    # Fallback behavior
    fallback_to_simple_response: bool = True
    orchestrator_persona: str = "helpful AI assistant"

    # -------------------------------------------------
    # NEW: Auto-approve / keep-delegating behavior
    # -------------------------------------------------
    #: When True the orchestrator **never** pauses for a manual
    #: confirmation after a delegated step.  It will automatically
    #: pick the recommended option in a clarification scenario.
    auto_approve: bool = True

    #: Upper bound on the number of internal delegation cycles
    #: before the orchestrator gives up (prevents infinite loops).
    max_delegation_steps: int = 20

    #: How to treat low-confidence classifications:
    #:   * ``auto`` - automatically fall-back to a generic agent.
    #:   * ``ask`` - request clarification from the user.
    low_confidence_policy: str = "auto"   # options: "auto", "ask"
    
    # -------------------------------------------------
    # Self-improvement integration
    # -------------------------------------------------
    #: Enable continuous self-improvement system
    enable_self_improvement: bool = True
    
    #: Self-improvement operation mode
    self_improvement_mode: str = "active"  # "disabled", "passive", "analysis_only", "conservative", "active", "aggressive"


@dataclass 
class OrchestratorResponse:
    """Response from orchestrator processing."""
    content: str
    response_type: str = "normal"  # "normal", "clarification_needed", "goal_completed", "error", "fallback"
    metadata: Optional[Dict] = None
    processing_time_ms: Optional[int] = None
    confidence: Optional[float] = None
    sources: List[str] = field(default_factory=list)
    agents_consulted: List[str] = field(default_factory=list)


class Orchestrator:
    """
    Strict orchestrator that enforces single-point communication architecture.
    
    This is the ONLY interface through which users interact with the agent system.
    All agent communications are intercepted and sanitized before user exposure.
    """
    
    def __init__(self, config: OrchestratorConfig = None):
        """Initialize the orchestrator with configuration."""
        self.config = config or OrchestratorConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Core components
        self.task_classifier = TaskClassifier()
        self.response_synthesizer = ResponseSynthesizer()
        self.communication_interceptor = CommunicationInterceptor()
        
        # Agent management
        self.active_agents = {}
        self.agent_status = {}
        
        # Statistics
        self.total_requests = 0
        self.successful_responses = 0
        self.fallback_responses = 0
        
        # Monitoring system integration
        self.metrics_collector = MetricsCollector()
        self.agent_tracker = AgentTracker()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize agent tracker listeners
        self._setup_monitoring_listeners()
        
        # Self-improvement system
        self.continuous_optimizer = None
        if self.config.enable_self_improvement:
            optimizer_config = {
                'mode': self.config.self_improvement_mode,
                'detailed_logging': False,
                'enable_automatic_rollback': True,
                'telemetry_enabled': True
            }
            self.continuous_optimizer = ContinuousOptimizer(optimizer_config)
        
        self.logger.info(f"Orchestrator initialized in {self.config.mode.value} mode")
        
    def _setup_monitoring_listeners(self):
        """Setup event listeners for monitoring system integration."""
        
        # Agent status change listener
        async def on_agent_status_change(agent_id: str, old_status: str, new_status: str, metadata: Dict):
            self.logger.debug(f"Agent {agent_id} status changed: {old_status} -> {new_status}")
            self.metrics_collector.record_counter(f"agent.{agent_id}.status_changes")
            
            # Record activity in activity feed if available
            if hasattr(self, 'activity_feed'):
                await self.activity_feed.add_event({
                    "type": "agent_status_change",
                    "agent_id": agent_id,
                    "old_status": old_status,
                    "new_status": new_status,
                    "metadata": metadata,
                    "timestamp": time.time()
                })
        
        # Task change listener (adapted to AgentTracker interface)
        def on_task_change(agent_id: str, old_task: Optional[Any], new_task: Optional[Any]):
            # When a task is completed (old_task exists, new_task is None)
            if old_task and not new_task:
                task_id = getattr(old_task, 'task_id', 'unknown')
                status = getattr(old_task, 'phase', 'unknown')
                self.logger.debug(f"Task {task_id} completed by agent {agent_id} with status {status}")
                self.metrics_collector.record_counter(f"agent.{agent_id}.tasks.{status}")
                
                if hasattr(self, 'activity_feed'):
                    # Use asyncio to handle async activity feed in sync context
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.create_task(self.activity_feed.add_event({
                                "type": "task_complete",
                                "agent_id": agent_id,
                                "task_id": task_id,
                                "status": str(status),
                                "timestamp": time.time()
                            }))
                    except Exception:
                        pass  # Ignore activity feed errors in initialization
        
        # Register listeners with agent tracker
        self.agent_tracker.add_status_listener(on_agent_status_change)
        self.agent_tracker.add_task_listener(on_task_change)
    
    async def process_user_input(self, user_input: str, context: Dict = None) -> OrchestratorResponse:
        """
        Primary entry point for user interactions.
        
        This method is the ONLY way users should interact with agents.
        All responses are synthesized and sanitized before returning.
        """
        start_time = time.time()
        self.total_requests += 1
        context = context or {}
        
        # Generate task ID for tracking
        task_id = f"task_{int(time.time())}_{hash(user_input) % 10000}"
        
        # Record metrics and start performance monitoring
        self.metrics_collector.record_counter("orchestrator.requests.total")
        self.metrics_collector.record_gauge("orchestrator.active_tasks", len(self.active_agents))
        performance_timer = self.metrics_collector.start_timer("orchestrator.process_user_input")
        
        # Start self-improvement tracking
        if self.continuous_optimizer:
            await self.continuous_optimizer.on_task_start(task_id, {
                'user_input': user_input[:100],  # Truncated for privacy
                'context': context,
                'start_time': start_time
            })
        
        task_success = False
        error_msg = None
        
        try:
            self.logger.info(f"Processing user input: {user_input[:100]}...")
            
            # Step 1: Classify the task
            classification_result = await self.task_classifier.classify_task(user_input, context)
            
            # Step 2: Handle based on classification
            if classification_result.classification == TaskClassification.SIMPLE_RESPONSE:
                response = await self._handle_simple_task(user_input, classification_result, context)
            elif classification_result.classification == TaskClassification.SINGLE_AGENT_NEEDED:
                response = await self._handle_single_agent_task(user_input, classification_result, context)
            elif classification_result.classification == TaskClassification.MULTI_AGENT_NEEDED:
                response = await self._handle_multi_agent_task(user_input, classification_result, context)
            else:
                response = await self._handle_fallback(user_input, context)
            
            # Step 3: Synthesize and sanitize response
            final_response = await self._synthesize_response(response, classification_result)
            
            processing_time = int((time.time() - start_time) * 1000)
            final_response.processing_time_ms = processing_time
            
            task_success = True
            self.successful_responses += 1
            
            # Record success metrics
            self.metrics_collector.record_counter("orchestrator.requests.success")
            self.metrics_collector.record_histogram("orchestrator.response_time_ms", processing_time)
            self.performance_monitor.record_request(processing_time / 1000.0, success=True)
            
            self.logger.info(f"Successfully processed request in {processing_time}ms")
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error processing user input: {e}", exc_info=True)
            processing_time = int((time.time() - start_time) * 1000)
            task_success = False
            error_msg = str(e)
            
            # Record error metrics
            self.metrics_collector.record_counter("orchestrator.requests.error")
            self.metrics_collector.record_histogram("orchestrator.error_response_time_ms", processing_time)
            
            return OrchestratorResponse(
                content="I encountered an issue processing your request. Please try rephrasing or contact support if this continues.",
                response_type="error",
                processing_time_ms=processing_time,
                metadata={"error": str(e)}
            )
        
        finally:
            # Complete performance monitoring
            if 'performance_timer' in locals():
                performance_timer()  # Call the timer function to record the duration
            
            # Complete self-improvement tracking
            if self.continuous_optimizer:
                await self.continuous_optimizer.on_task_complete(
                    task_id, 
                    success=task_success,
                    error=error_msg,
                    user_feedback=context.get('user_feedback')
                )
    
    async def _handle_simple_task(self, user_input: str, classification: ClassificationResult, context: Dict) -> OrchestratorResponse:
        """Handle tasks that don't require agent delegation."""
        self.logger.debug("Handling simple task without agent delegation")
        
        # Generate simple response based on classification metadata
        response_templates = {
            "greeting": "Hello! How can I help you today?",
            "status": "All systems are running normally.",
            "help": "I can help you with various tasks. What would you like to accomplish?",
        }
        
        # Derive task type from classification reasoning or task complexity
        task_type = "general"  # default
        reasoning_lower = classification.reasoning.lower()
        if "greeting" in reasoning_lower or "hello" in reasoning_lower or "hi" in reasoning_lower:
            task_type = "greeting"
        elif "help" in reasoning_lower:
            task_type = "help"
        elif "status" in reasoning_lower:
            task_type = "status"
        
        content = response_templates.get(task_type, "I understand. Let me help you with that.")
        
        return OrchestratorResponse(
            content=content,
            response_type="normal",
            confidence=classification.confidence
        )
    
    async def _handle_single_agent_task(self, user_input: str, classification: ClassificationResult, context: Dict) -> OrchestratorResponse:
        """Handle tasks requiring single agent delegation."""
        self.logger.debug(f"Delegating to single agent: {classification.required_agents}")
        
        agent_id = classification.required_agents[0] if classification.required_agents else "general"
        
        try:
            # Delegate to agent (implementation would connect to actual agents)
            agent_response = await self._delegate_to_agent(agent_id, user_input, context)
            
            # Intercept and sanitize agent response
            intercepted = self.communication_interceptor.intercept_response(
                agent_id, agent_response, {"task_classification": classification.classification.value}
            )
            
            return OrchestratorResponse(
                content=intercepted.processed_response["sanitized_content"],
                response_type="normal",
                confidence=classification.confidence,
                sources=[agent_id],
                metadata={"agent_used": agent_id, "interception_stats": intercepted.sanitization_applied}
            )
            
        except Exception as e:
            self.logger.error(f"Error delegating to agent {agent_id}: {e}")
            return await self._handle_fallback(user_input, context)
    
    async def _handle_multi_agent_task(self, user_input: str, classification: ClassificationResult, context: Dict) -> OrchestratorResponse:
        """Handle tasks requiring multiple agent coordination using intelligent delegation."""
        self.logger.debug(f"Intelligently coordinating multiple agents for complex task")
        
        try:
            if not self.config.enable_intelligent_delegation:
                # Fallback to simple multi-agent coordination
                return await self._handle_multi_agent_task_simple(user_input, classification, context)
            
            # Step 1: Use intelligent delegation to break down and assign tasks
            delegation_system = get_delegation_system()
            
            # Break down complex task if needed
            if "implement" in user_input.lower() or "create" in user_input.lower():
                task_descriptions = delegation_system.suggest_task_breakdown(user_input)
                self.logger.debug(f"Broke down task into {len(task_descriptions)} subtasks")
            else:
                task_descriptions = [user_input]
            
            # Get optimal agent assignments
            agent_assignments = await delegation_system.delegate_tasks(task_descriptions)
            
            if not agent_assignments:
                self.logger.warning("No agent assignments generated, falling back")
                return await self._handle_fallback(user_input, context)
            
            # Step 2: Execute assignments with conflict resolution
            self.logger.info(f"Executing {len(agent_assignments)} agent assignments")
            
            # Separate parallel and sequential assignments
            parallel_assignments = []
            sequential_assignments = []
            
            for assignment in agent_assignments:
                # Check if agent is available in our runtime config
                if self._is_agent_available(assignment.agent_type):
                    if self.config.enable_parallel_execution and assignment.agent_type not in [a.agent_type for a in sequential_assignments]:
                        parallel_assignments.append(assignment)
                    else:
                        sequential_assignments.append(assignment)
                else:
                    self.logger.warning(f"Agent {assignment.agent_type} not available, using fallback")
                    # Map to available agent
                    fallback_agent = self._get_fallback_agent(assignment.agent_type)
                    if fallback_agent:
                        assignment.agent_type = fallback_agent
                        parallel_assignments.append(assignment)
            
            # Step 3: Execute parallel assignments
            agent_responses = []
            if parallel_assignments:
                parallel_responses = await self._execute_parallel_assignments(parallel_assignments, context)
                agent_responses.extend(parallel_responses)
            
            # Step 4: Execute sequential assignments  
            if sequential_assignments:
                sequential_responses = await self._execute_sequential_assignments(sequential_assignments, context)
                agent_responses.extend(sequential_responses)
            
            if not agent_responses:
                return await self._handle_fallback(user_input, context)
            
            # Step 5: Synthesize results with context awareness
            synthesized = await self._synthesize_intelligent_responses(agent_responses, user_input)
            
            return OrchestratorResponse(
                content=synthesized["content"],
                response_type="normal",
                confidence=classification.confidence,
                sources=[r["agent_id"] for r in agent_responses],
                metadata={"agents_used": len(agent_responses), "synthesis_method": synthesized["method"]}
            )
            
        except Exception as e:
            self.logger.error(f"Error in multi-agent coordination: {e}")
            return await self._handle_fallback(user_input, context)
    
    async def _handle_fallback(self, user_input: str, context: Dict) -> OrchestratorResponse:
        """Handle requests when agent delegation fails or isn't appropriate."""
        self.logger.info("Using fallback response generation")
        self.fallback_responses += 1
        
        if not self.config.fallback_to_simple_response:
            return OrchestratorResponse(
                content="I'm unable to process this request at the moment. Please try again later.",
                response_type="error"
            )
        
        # Generate contextual fallback response
        fallback_content = f"As a {self.config.orchestrator_persona}, I understand you're asking about: {user_input[:100]}{'...' if len(user_input) > 100 else ''}. Let me provide what help I can."
        
        return OrchestratorResponse(
            content=fallback_content,
            response_type="fallback",
            confidence=0.3
        )
    
    async def _delegate_to_agent(self, agent_id: str, user_input: str, context: Dict) -> str:
        """
        Delegate task to specific agent.
        
        This is a placeholder implementation. Real implementation would:
        - Connect to actual agent via MCP or API
        - Handle agent lifecycle (start/stop)
        - Manage agent context and memory
        """
        self.logger.debug(f"Delegating to agent: {agent_id}")
        
        # Placeholder: simulate agent response
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return f"Response from {agent_id} agent for: {user_input}"
    
    async def _synthesize_response(self, response: OrchestratorResponse, classification: ClassificationResult) -> OrchestratorResponse:
        """Apply final response synthesis and safety checks."""
        # Apply any final transformations based on orchestrator mode
        if self.config.mode == OrchestratorMode.STRICT_ISOLATION:
            # Ensure no agent identifiers leak through
            content = response.content
            content = content.replace("agent", "system")
            content = content.replace("Agent", "System")
            response.content = content
        
        return response
    
    async def shutdown(self):
        """Clean shutdown of orchestrator and all managed agents."""
        self.logger.info("Shutting down orchestrator...")
        
        # Cleanup active agents
        for agent_id in list(self.active_agents.keys()):
            try:
                await self._cleanup_agent(agent_id)
            except Exception as e:
                self.logger.error(f"Error cleaning up agent {agent_id}: {e}")
        
        self.active_agents.clear()
        self.agent_status.clear()
        
        logger.info("Orchestrator shutdown complete")

    # ----------------------------------------------------------------------
    # PUBLIC: Run a high-level goal until it is achieved.
    # ----------------------------------------------------------------------
    async def run_until_goal(self, goal: str, initial_context: Optional[Dict] = None) -> OrchestratorResponse:
        """
        Repeatedly invoke ``process_user_input`` using the same *goal* string.
        The loop stops when:
        1. The orchestrator reports ``response_type == "goal_completed"``.
        2. A clarification is required (see ``_maybe_ask_for_clarification``).
        3. ``max_delegation_steps`` is reached.
        """
        context = initial_context or {}
        steps = 0

        while steps < self.config.max_delegation_steps:
            steps += 1
            resp = await self.process_user_input(goal, context)

            # 1️⃣ Goal completed?
            if resp.response_type == "goal_completed":
                self.logger.info(f"Goal satisfied after {steps} delegation step(s).")
                return resp

            # 2️⃣ Clarification needed?
            if resp.response_type == "clarification_needed":
                clarified = await self._maybe_ask_for_clarification(resp, context)
                if clarified is None:                # user chose to stop
                    return resp
                # Use the clarified answer as the new goal string
                goal = clarified
                continue

            # 3️⃣ Normal response – keep looping with the same goal
            continue

        # ------------------------------------------------------------------
        # Too many steps – give up with a friendly message
        # ------------------------------------------------------------------
        return OrchestratorResponse(
            content=(
                "I've been working on this goal for a while but haven't reached a "
                "complete solution yet. Could you give me a bit more detail or "
                "re-frame the request?"
            ),
            response_type="fallback",
            metadata={"steps_exhausted": steps}
        )

    # ----------------------------------------------------------------------
    # PRIVATE: Ask (or auto-pick) a clarification when the orchestrator
    #          cannot decide on its own.
    # ----------------------------------------------------------------------
    async def _maybe_ask_for_clarification(self, resp: OrchestratorResponse,
                                            context: Dict) -> Optional[str]:
        """
        *If* ``auto_approve`` is True -> automatically select the recommended
        option (the orchestrator will have filled ``metadata['options']`` and
        ``metadata['recommended']``).  *Else* -> present the options to the user,
        explain pros/cons & risks, and wait for a textual answer.  The method
        returns the *new* goal string (or ``None`` if the user aborts).
        """
        meta = resp.metadata or {}
        options: List[Dict] = meta.get("options", [])
        recommended: str = meta.get("recommended", "")

        if not options:
            # Nothing to clarify – just return the original content as a new goal.
            return resp.content

        if self.config.auto_approve:
            # Auto-approve path – log and move on
            self.logger.info(f"Auto-approving clarification, picking recommended option: {recommended}")
            return recommended

        # ------------------------------------------------------------------
        # Interactive clarification – build a friendly prompt
        # ------------------------------------------------------------------
        prompt_lines = ["We need a bit more information to continue.", "Please choose one of the options below:"]
        for idx, opt in enumerate(options, start=1):
            label = opt.get("label", f"Option {idx}")
            description = opt.get("description", "")
            pros = opt.get("pros", [])
            cons = opt.get("cons", [])
            risks = opt.get("risks", [])
            prompt_lines.append(f"{idx}. **{label}** – {description}")
            if pros:
                prompt_lines.append(f"   • Pros: {', '.join(pros)}")
            if cons:
                prompt_lines.append(f"   • Cons: {', '.join(cons)}")
            if risks:
                prompt_lines.append(f"   • Risks: {', '.join(risks)}")
        prompt_lines.append(f"\nRecommended choice: **{recommended}**")
        prompt_lines.append("\nEnter the number of your choice (or type 'stop' to abort):")

        # Show the prompt to the user (the orchestrator is the only UI entry-point)
        user_input = input("\n".join(prompt_lines)).strip().lower()
        if user_input in ("stop", "abort", "cancel"):
            self.logger.info("User aborted clarification flow.")
            return None
        try:
            choice_idx = int(user_input) - 1
            if 0 <= choice_idx < len(options):
                chosen = options[choice_idx].get("label", recommended)
                self.logger.info(f"User selected clarification option: {chosen}")
                return chosen
        except ValueError:
            pass
        # Fallback – if we cannot parse the input, just use the recommended one
        self.logger.warning("Could not parse clarification choice – falling back to recommended.")
        return recommended

    async def _cleanup_agent(self, agent_id: str):
        """Cleanup specific agent resources."""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
        if agent_id in self.agent_status:
            del self.agent_status[agent_id]
        
        self.logger.debug(f"Cleaned up agent: {agent_id}")
    
    async def _handle_multi_agent_task_simple(self, user_input: str, classification: ClassificationResult, context: Dict) -> OrchestratorResponse:
        """Fallback simple multi-agent coordination (original implementation)."""
        self.logger.debug(f"Simple multi-agent coordination: {classification.required_agents}")
        
        if not classification.required_agents:
            return await self._handle_fallback(user_input, context)
        
        try:
            # Delegate to multiple agents in parallel
            agent_tasks = []
            for agent_id in classification.required_agents[:self.config.max_parallel_agents]:
                task = self._delegate_to_agent(agent_id, user_input, context)
                agent_tasks.append((agent_id, task))
            
            # Collect responses
            agent_responses = []
            for agent_id, task in agent_tasks:
                try:
                    response = await asyncio.wait_for(
                        task, 
                        timeout=self.config.max_agent_wait_time_ms / 1000
                    )
                    
                    # Intercept each response
                    intercepted = self.communication_interceptor.intercept_response(
                        agent_id, response, {"task_classification": classification.classification.value}
                    )
                    
                    agent_responses.append({
                        "agent_id": agent_id,
                        "content": intercepted.processed_response["sanitized_content"],
                        "metadata": intercepted.processed_response["agent_metadata"]
                    })
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"Agent {agent_id} timed out")
                except Exception as e:
                    self.logger.error(f"Error from agent {agent_id}: {e}")
            
            if not agent_responses:
                return await self._handle_fallback(user_input, context)
            
            # Synthesize multiple responses
            synthesized = await self.response_synthesizer.synthesize_responses(
                agent_responses, self.config.default_synthesis_strategy
            )
            
            return OrchestratorResponse(
                content=synthesized["content"],
                response_type="normal",
                confidence=classification.confidence,
                sources=[r["agent_id"] for r in agent_responses],
                metadata={"agents_used": len(agent_responses)}
            )
            
        except Exception as e:
            self.logger.error(f"Error in simple multi-agent coordination: {e}")
            return await self._handle_fallback(user_input, context)

    def _is_agent_available(self, agent_type: str) -> bool:
        """Check if an agent type is available in runtime config."""
        from ..runtime_config import AGENT_CONFIGS
        return agent_type in AGENT_CONFIGS

    def _get_fallback_agent(self, agent_type: str) -> Optional[str]:
        """Get a fallback agent for unavailable agent types."""
        # Mapping of specialized agents to available fallbacks
        fallback_mapping = {
            "backend_engineer": "backend_developer",
            "web_frontend_engineer": "frontend_developer", 
            "api_engineer": "backend_developer",
            "mobile_engineer": "backend_developer",
            "database_engineer": "backend_developer",
            "chief_qa_engineer": "qa_engineer",
            "security_engineer": "backend_developer",
            "solutions_architect": "backend_developer",
            "backend_qa_engineer": "qa_engineer",
            "web_frontend_qa_engineer": "qa_engineer",
            "tui_frontend_qa_engineer": "qa_engineer", 
            "ux_ui_designer": "product_manager",
            "tui_ux_designer": "product_manager",
            "user_researcher": "market_researcher",
            "data_analyst": "backend_developer",
            "technical_writer": "product_manager",
            "performance_engineer": "backend_developer",
            "site_reliability_engineer": "backend_developer",
        }
        
        fallback = fallback_mapping.get(agent_type)
        if fallback and self._is_agent_available(fallback):
            return fallback
        
        # Final fallback to basic agents
        for basic_agent in ["backend_developer", "frontend_developer", "qa_engineer"]:
            if self._is_agent_available(basic_agent):
                return basic_agent
        
        return None

    async def _execute_parallel_assignments(self, assignments, context: Dict) -> List[Dict]:
        """Execute multiple agent assignments in parallel."""
        self.logger.info(f"Executing {len(assignments)} assignments in parallel")
        
        # Create tasks for parallel execution
        agent_tasks = []
        for assignment in assignments[:self.config.max_parallel_agents]:
            for task in assignment.tasks:
                agent_task = self._delegate_to_agent(
                    assignment.agent_type, 
                    task.description, 
                    {**context, "task_type": task.task_type.value, "task_id": task.id}
                )
                agent_tasks.append((assignment.agent_type, task.id, agent_task))
        
        # Execute all tasks concurrently
        agent_responses = []
        for agent_id, task_id, task in agent_tasks:
            try:
                response = await asyncio.wait_for(
                    task, 
                    timeout=self.config.max_agent_wait_time_ms / 1000
                )
                
                # Intercept and process response
                intercepted = self.communication_interceptor.intercept_response(
                    agent_id, response, {"task_id": task_id, "execution_mode": "parallel"}
                )
                
                agent_responses.append({
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "content": intercepted.processed_response["sanitized_content"],
                    "metadata": intercepted.processed_response["agent_metadata"]
                })
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Agent {agent_id} task {task_id} timed out")
            except Exception as e:
                self.logger.error(f"Error from agent {agent_id} task {task_id}: {e}")
        
        return agent_responses

    async def _execute_sequential_assignments(self, assignments, context: Dict) -> List[Dict]:
        """Execute agent assignments sequentially to resolve conflicts."""
        self.logger.info(f"Executing {len(assignments)} assignments sequentially")
        
        agent_responses = []
        for assignment in assignments:
            for task in assignment.tasks:
                try:
                    response = await asyncio.wait_for(
                        self._delegate_to_agent(
                            assignment.agent_type, 
                            task.description, 
                            {**context, "task_type": task.task_type.value, "task_id": task.id}
                        ),
                        timeout=self.config.max_agent_wait_time_ms / 1000
                    )
                    
                    # Intercept and process response
                    intercepted = self.communication_interceptor.intercept_response(
                        assignment.agent_type, response, {"task_id": task.id, "execution_mode": "sequential"}
                    )
                    
                    agent_responses.append({
                        "agent_id": assignment.agent_type,
                        "task_id": task.id,
                        "content": intercepted.processed_response["sanitized_content"],
                        "metadata": intercepted.processed_response["agent_metadata"]
                    })
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"Agent {assignment.agent_type} task {task.id} timed out")
                except Exception as e:
                    self.logger.error(f"Error from agent {assignment.agent_type} task {task.id}: {e}")
        
        return agent_responses

    async def _synthesize_intelligent_responses(self, agent_responses: List[Dict], user_input: str) -> Dict[str, Any]:
        """Synthesize responses with enhanced context awareness for intelligent delegation."""
        if len(agent_responses) == 1:
            return {
                "content": agent_responses[0]["content"],
                "method": "single_response"
            }
        
        # Group responses by agent type for better synthesis
        grouped_responses = {}
        for response in agent_responses:
            agent_type = response["agent_id"]
            if agent_type not in grouped_responses:
                grouped_responses[agent_type] = []
            grouped_responses[agent_type].append(response)
        
        # Synthesize based on task complexity and agent roles
        if len(grouped_responses) > 1:
            # Multi-agent collaborative response
            synthesis_strategy = SynthesisStrategy.CONSENSUS if "design" in user_input.lower() or "architecture" in user_input.lower() else SynthesisStrategy.COMPREHENSIVE
        else:
            # Single agent type with multiple tasks
            synthesis_strategy = SynthesisStrategy.SUMMARIZE
            
        synthesized = await self.response_synthesizer.synthesize_responses(
            agent_responses, synthesis_strategy
        )
        
        return {
            "content": synthesized["content"],
            "method": f"intelligent_{synthesis_strategy.value}",
            "agents_by_type": list(grouped_responses.keys()),
            "total_responses": len(agent_responses)
        }

    async def start_self_improvement(self) -> None:
        """Start the self-improvement system."""
        if self.continuous_optimizer:
            await self.continuous_optimizer.start()
            self.logger.info("Self-improvement system started")
    
    async def stop_self_improvement(self) -> None:
        """Stop the self-improvement system."""
        if self.continuous_optimizer:
            await self.continuous_optimizer.stop()
            self.logger.info("Self-improvement system stopped")
    
    async def get_self_improvement_status(self) -> Dict[str, Any]:
        """Get self-improvement system status."""
        if not self.continuous_optimizer:
            return {"enabled": False, "status": "disabled"}
        
        return await self.continuous_optimizer.get_optimization_status()
    
    async def trigger_manual_optimization(self) -> Dict[str, Any]:
        """Manually trigger optimization cycle."""
        if not self.continuous_optimizer:
            return {"error": "Self-improvement system not enabled"}
        
        return await self.continuous_optimizer.manual_optimization_cycle()
    
    async def rollback_improvement(self, entry_id: str) -> Dict[str, Any]:
        """Rollback a specific improvement."""
        if not self.continuous_optimizer:
            return {"error": "Self-improvement system not enabled"}
        
        return await self.continuous_optimizer.rollback_improvement(entry_id)
    
    def get_monitoring_components(self) -> Dict[str, Any]:
        """Get monitoring system components for TUI integration."""
        return {
            "metrics_collector": self.metrics_collector,
            "agent_tracker": self.agent_tracker,
            "performance_monitor": self.performance_monitor
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics."""
        success_rate = (self.successful_responses / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_responses": self.successful_responses,
            "fallback_responses": self.fallback_responses,
            "success_rate_percent": round(success_rate, 2),
            "active_agents": len(self.active_agents),
            "mode": self.config.mode.value,
            "interception_stats": self.communication_interceptor.get_interception_stats(),
            "self_improvement_enabled": self.config.enable_self_improvement
        }