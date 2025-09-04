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
        
        # Agent management with enhanced capabilities
        self.active_agents = {}
        self.agent_status = {}
        
        # Enhanced agent management system
        from ..agents.agent_loader import AgentLoader
        from .memory_manager import AgentMemoryManager
        from .agent_health_monitor import AgentHealthMonitor, HealthThresholds
        
        self.agent_loader = AgentLoader()
        self.memory_manager = AgentMemoryManager()
        self.specialist_agents = {}  # Cache for on-demand specialist agents
        
        # Initialize health monitoring system
        health_thresholds = HealthThresholds(
            max_response_time_ms=8000.0,  # Generous for AI agents
            min_success_rate=0.7,
            max_consecutive_failures=3,
            heartbeat_timeout_seconds=45,
            unhealthy_threshold_score=60.0,
            failed_threshold_score=30.0
        )
        
        self.health_monitor = AgentHealthMonitor(
            thresholds=health_thresholds,
            metrics_storage_path=".agentsmcp/health_metrics",
            auto_restart=True,
            monitoring_interval=15.0
        )
        
        # Set up health monitoring callbacks
        self.health_monitor.add_health_change_listener(self._on_agent_health_change)
        self.health_monitor.add_restart_listener(self._on_agent_restart)
        
        # Load all available agent descriptions
        self.available_agent_types = self.agent_loader.load_all_descriptions()
        self.logger.info(f"Loaded {len(self.available_agent_types)} specialist agent types")
        
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
        
        # Planning system state
        self.active_plans = {}
        self.planning_enabled = True
        
        self.logger.info(f"Orchestrator initialized in {self.config.mode.value} mode with "
                        f"{len(self.available_agent_types)} specialist agents available and "
                        f"health monitoring enabled")
        
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
        
        Enhanced with planning phase support to address agent swarm best practices.
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
            
            # Check for planning commands first (addresses agent swarm best practice #1)
            if user_input.strip().startswith('/'):
                command_parts = user_input.strip().split(' ', 1)
                command = command_parts[0].lower()
                task_description = command_parts[1] if len(command_parts) > 1 else ""
                
                # Handle planning commands
                if command in ['/plan', '/spike', '/tech-plan'] or command.startswith('/execute'):
                    self.logger.info(f"Processing planning command: {command}")
                    
                    planning_response = await self.handle_planning_command(command, task_description)
                    
                    processing_time = int((time.time() - start_time) * 1000)
                    task_success = True
                    self.successful_responses += 1
                    
                    # Record success metrics
                    self.metrics_collector.record_counter("orchestrator.requests.planning_command")
                    self.metrics_collector.record_histogram("orchestrator.response_time_ms", processing_time)
                    self.performance_monitor.record_request(processing_time / 1000.0, success=True)
                    
                    return OrchestratorResponse(
                        content=planning_response,
                        response_type="planning_command",
                        processing_time_ms=processing_time,
                        metadata={
                            "command": command,
                            "planning_enabled": True
                        }
                    )
                
                # Handle system commands
                elif command in ['/health', '/status', '/agents']:
                    self.logger.info(f"Processing system command: {command}")
                    
                    if command == '/health':
                        health_summary = self.get_system_health()
                        system_response = (
                            f"🏥 **System Health Report**\n"
                            f"**Overall Health Score:** {health_summary.get('overall_health_score', 0):.1f}/100\n"
                            f"**Total Agents:** {health_summary.get('total_agents', 0)}\n"
                            f"**Healthy:** {health_summary.get('healthy', 0)} | "
                            f"**Degraded:** {health_summary.get('degraded', 0)} | "
                            f"**Unhealthy:** {health_summary.get('unhealthy', 0)} | "
                            f"**Failed:** {health_summary.get('failed', 0)}\n"
                            f"**Monitoring Status:** {'Active' if hasattr(self, 'health_monitor') else 'Inactive'}"
                        )
                    elif command == '/status':
                        planning_summary = self.get_planning_summary()
                        system_response = (
                            f"📊 **Orchestrator Status**\n"
                            f"**Total Requests:** {self.total_requests}\n"
                            f"**Success Rate:** {(self.successful_responses/self.total_requests*100):.1f}%\n"
                            f"**Active Agents:** {len(self.active_agents)}\n"
                            f"**Available Agent Types:** {len(self.available_agent_types)}\n"
                            f"**Planning System:** {planning_summary.get('status', 'inactive')}\n"
                            f"**Active Plans:** {planning_summary.get('active_plans', 0)}"
                        )
                    elif command == '/agents':
                        agent_list = list(self.available_agent_types.keys())[:10]  # Show first 10
                        system_response = (
                            f"🤖 **Available Specialist Agents** ({len(self.available_agent_types)} total)\n"
                            f"**Sample Types:** {', '.join(agent_list)}\n"
                            f"**Active Specialists:** {len(self.specialist_agents)}\n"
                            f"Use `/plan <task>` to create structured execution plans with automatic agent assignment."
                        )
                    
                    processing_time = int((time.time() - start_time) * 1000)
                    task_success = True
                    self.successful_responses += 1
                    
                    return OrchestratorResponse(
                        content=system_response,
                        response_type="system_command",
                        processing_time_ms=processing_time,
                        metadata={"command": command}
                    )
            
            # For complex tasks, suggest using planning phase (best practice implementation)
            complexity_indicators = [
                'implement', 'develop', 'create', 'build', 'design', 'integrate', 
                'deploy', 'migrate', 'refactor', 'optimize', 'analyze', 'research'
            ]
            
            user_lower = user_input.lower()
            is_complex_task = any(indicator in user_lower for indicator in complexity_indicators)
            
            if is_complex_task and len(user_input.split()) > 10 and not user_input.startswith('/'):
                # Suggest planning approach for complex tasks
                suggestion = (
                    f"🎯 **Complex Task Detected**\n\n"
                    f"For best results with complex tasks, consider using the planning phase:\n"
                    f"• `/plan {user_input}` - Create comprehensive execution plan\n"
                    f"• `/tech-plan {user_input}` - Technical implementation plan\n"
                    f"• `/spike {user_input}` - Research and investigation plan\n\n"
                    f"Or I can proceed directly with immediate execution:"
                )
                
                # Store the suggestion but continue with normal processing
                await self.memory_manager.store_agent_memory(
                    agent_type="orchestrator",
                    category="user_interaction",
                    content=f"Suggested planning approach for complex task: {user_input[:100]}",
                    importance=4
                )
            
            # Route to LLM agent for normal processing
            self.logger.debug("Routing to LLM agent for processing")
            
            response = await self._route_to_llm_directly(user_input, context)
            
            # Add planning suggestion to response if applicable
            if is_complex_task and len(user_input.split()) > 10 and not user_input.startswith('/'):
                response.content = suggestion + "\n\n---\n\n" + response.content
                response.metadata = response.metadata or {}
                response.metadata['planning_suggested'] = True
            
            # Record that we successfully got an LLM response
            processing_time = int((time.time() - start_time) * 1000)
            response.processing_time_ms = processing_time
            
            task_success = True
            self.successful_responses += 1
            
            # Record success metrics
            self.metrics_collector.record_counter("orchestrator.requests.success")
            self.metrics_collector.record_histogram("orchestrator.response_time_ms", processing_time)
            self.performance_monitor.record_request(processing_time / 1000.0, success=True)
            
            self.logger.info(f"Successfully processed request via LLM in {processing_time}ms")
            
            return response
            
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
        
        # Analyze user input for intelligent response generation
        user_input_lower = user_input.lower().strip()
        reasoning_lower = classification.reasoning.lower() if classification.reasoning else ""
        
        # Comprehensive response patterns
        response_patterns = {
            # Greetings and social
            ("hello", "hi", "hey", "good morning", "good afternoon", "good evening"): 
                "Hello! I'm your AgentsMCP assistant. How can I help you today?",
            
            # Status and system inquiries
            ("status", "how are you", "are you working", "system status", "health check"):
                "I'm running normally and ready to assist you. All systems are operational. What can I help you with?",
            
            # Help requests
            ("help", "what can you do", "capabilities", "commands", "how do i", "what do you"):
                "I can help you with various tasks including:\n• Managing and coordinating AI agents\n• Software development workflows\n• System monitoring and diagnostics\n• Configuration and settings\n\nWhat specific task would you like help with?",
            
            # Thanks and pleasantries
            ("thank you", "thanks", "appreciate", "great job", "well done"):
                "You're welcome! I'm here to help whenever you need assistance. Is there anything else I can do for you?",
            
            # Farewells
            ("goodbye", "bye", "see you", "quit", "exit", "farewell"):
                "Goodbye! Feel free to return anytime you need assistance. Have a great day!",
            
            # Questions about the system
            ("what is this", "what are you", "who are you", "about"):
                "I'm AgentsMCP, an AI assistant that coordinates multiple specialized agents to help you with various tasks. I can manage complex workflows, delegate tasks to appropriate agents, and provide unified responses. What would you like to explore?",
            
            # Time and date
            ("time", "date", "when", "what time"):
                f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. How can I assist you today?",
            
            # Settings and configuration
            ("settings", "config", "preferences", "options"):
                "I can help you configure various settings including:\n• Theme preferences (light/dark/auto)\n• Agent behavior settings\n• System configurations\n• Display options\n\nWhat would you like to adjust?",
                
            # Weather (redirect to capability)
            ("weather", "temperature", "forecast"):
                "I don't have direct access to weather information, but I can help you set up agents or tools that can provide weather data. Would you like me to help you configure weather services?",
        }
        
        # Find matching pattern
        response_content = None
        for keywords, response in response_patterns.items():
            if any(keyword in user_input_lower for keyword in keywords):
                response_content = response
                break
        
        # If no specific pattern matched, provide intelligent fallback
        if response_content is None:
            # Check if user is asking a question
            if any(question_word in user_input_lower for question_word in ["what", "how", "why", "when", "where", "which", "who", "can you", "do you", "will you", "could you", "would you", "?"]):
                response_content = f"That's an interesting question about {user_input[:50]}{'...' if len(user_input) > 50 else ''}. While I can handle many tasks directly, for more complex queries I can coordinate with specialized agents. Would you like me to explore this further or help you with something else?"
            else:
                # Generic but more helpful fallback
                response_content = f"I can help you with that. For the request '{user_input[:50]}{'...' if len(user_input) > 50 else ''}', I can either handle it directly or coordinate with specialized agents as needed. What specific outcome are you looking for?"
        
        return OrchestratorResponse(
            content=response_content,
            response_type="simple",
            confidence=max(0.8, classification.confidence if classification.confidence else 0.8),
            processing_time_ms=int((time.time() - time.time()) * 1000)  # Minimal processing time
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
        Delegate task to specific agent using real LLM client.
        
        Connects to actual LLM agent via the LLM client to get intelligent responses.
        """
        self.logger.debug(f"Delegating to agent: {agent_id}")
        
        try:
            # Import and create LLM client for real agent responses
            from ..conversation.llm_client import LLMClient
            
            # Create LLM client instance
            llm_client = LLMClient()
            
            # Prepare agent-specific context and prompt
            agent_context = f"Acting as {agent_id} agent, please respond to the following request:\n\n{user_input}"
            
            # Add context information if available
            if context:
                agent_context += f"\n\nContext: {context}"
            
            # Get response from actual LLM
            response = await llm_client.send_message(agent_context, context)
            
            self.logger.debug(f"Agent {agent_id} provided response of length {len(response)}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error connecting to LLM for agent {agent_id}: {e}")
            # Fallback to indicate the issue
            return f"I encountered an issue connecting to the {agent_id} agent. Please check that ollama-turbo is properly configured and running. Error: {str(e)}"
    
    async def _route_to_llm_directly(self, user_input: str, context: Dict) -> OrchestratorResponse:
        """
        Route user input directly to LLM without any task classification.
        
        This is the "pre-processor" approach that ensures ALL user questions
        go to the connected LLM (ollama-turbo) instead of getting template responses.
        """
        self.logger.debug("Routing directly to LLM agent - bypassing all classification")
        
        try:
            # Import and create LLM client for direct LLM responses
            from ..conversation.llm_client import LLMClient
            
            # Create LLM client instance
            llm_client = LLMClient()
            
            # Send user input directly to LLM without any modification or classification
            llm_response = await llm_client.send_message(user_input, context)
            
            # Return the LLM response directly as an orchestrator response
            return OrchestratorResponse(
                content=llm_response,
                response_type="llm_direct",
                confidence=0.9,  # High confidence since it's from the actual LLM
                sources=["ollama-turbo"],
                metadata={"routing_method": "direct_llm_bypass", "bypassed_classification": True}
            )
            
        except Exception as e:
            self.logger.error(f"Error routing directly to LLM: {e}")
            # If LLM fails, give a clear error message
            return OrchestratorResponse(
                content=f"I'm having trouble connecting to the language model. Please check that ollama-turbo is running and accessible. Error: {str(e)}",
                response_type="error",
                confidence=0.0,
                metadata={"error": str(e), "routing_method": "direct_llm_failed"}
            )
    
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

    # ===== ENHANCED AGENT MANAGEMENT METHODS =====
    
    async def spawn_specialist_agent(self, agent_type: str, task_context: str = "") -> Optional[str]:
        """Spawn a specialist agent on-demand based on task requirements.
        
        Args:
            agent_type: Type of specialist agent to spawn
            task_context: Context about the task for memory initialization
            
        Returns:
            Agent ID if successful, None otherwise
        """
        if agent_type not in self.available_agent_types:
            self.logger.warning(f"Unknown specialist agent type: {agent_type}")
            return None
            
        try:
            # Check if agent is already active
            if agent_type in self.specialist_agents:
                self.logger.info(f"Specialist agent {agent_type} already active")
                return self.specialist_agents[agent_type]
            
            # Create agent configuration
            agent_desc = self.available_agent_types[agent_type]
            agent_config = self.agent_loader.create_agent_config(agent_desc)
            
            # Generate unique agent ID
            agent_id = f"{agent_type}_{len(self.active_agents)}"
            
            # Store agent configuration and mark as active
            self.active_agents[agent_id] = agent_config
            self.agent_status[agent_id] = "active"
            self.specialist_agents[agent_type] = agent_id
            
            # Initialize agent memory with task context
            if task_context:
                await self.memory_manager.store_agent_memory(
                    agent_type=agent_type,
                    category="initialization",
                    content=f"Spawned for task: {task_context}",
                    importance=6,
                    tags=["spawn", "initialization"]
                )
            
            self.logger.info(f"Successfully spawned specialist agent {agent_type} with ID {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to spawn specialist agent {agent_type}: {e}")
            return None
    
    async def get_agent_recommendations(self, task_description: str, current_agents: List[str] = None) -> List[str]:
        """Get recommendations for specialist agents based on task description.
        
        Args:
            task_description: Description of the task to be performed
            current_agents: List of currently active agents
            
        Returns:
            List of recommended agent types
        """
        current_agents = current_agents or []
        recommendations = set()
        
        # Keyword-based agent recommendations
        task_lower = task_description.lower()
        
        # Security-related keywords
        if any(word in task_lower for word in ['security', 'vulnerability', 'encrypt', 'auth', 'hack', 'breach']):
            recommendations.add('security-engineer')
            
        # Product management keywords
        if any(word in task_lower for word in ['product', 'feature', 'roadmap', 'requirement', 'user story']):
            recommendations.add('senior-product-manager')
            
        # Architecture keywords
        if any(word in task_lower for word in ['architecture', 'design', 'system', 'scalability', 'performance']):
            recommendations.add('principal-software-architect')
            
        # UX/UI keywords
        if any(word in task_lower for word in ['ui', 'ux', 'design', 'interface', 'user experience']):
            recommendations.add('ux-ui-designer')
            
        # Legal keywords
        if any(word in task_lower for word in ['legal', 'compliance', 'gdpr', 'privacy', 'contract']):
            recommendations.add('legal-counsel')
            
        # Data/analytics keywords
        if any(word in task_lower for word in ['data', 'analytics', 'ml', 'ai', 'model', 'analysis']):
            recommendations.add('data-scientist')
            
        # Marketing keywords
        if any(word in task_lower for word in ['marketing', 'campaign', 'brand', 'customer', 'sales']):
            recommendations.add('marketing-manager')
        
        # Remove already active agents
        recommendations = [agent for agent in recommendations if agent not in current_agents]
        
        return list(recommendations)
    
    async def delegate_to_specialist(self, agent_type: str, task: str, context: Optional[Dict] = None) -> Optional[str]:
        """Delegate a task to a specialist agent, spawning if necessary.
        
        Args:
            agent_type: Type of specialist agent to delegate to
            task: Task description to delegate
            context: Optional context information
            
        Returns:
            Response from specialist agent or None if failed
        """
        try:
            # Spawn agent if not already active
            agent_id = await self.spawn_specialist_agent(agent_type, task)
            if not agent_id:
                return None
            
            # Get contextual memories for the agent
            relevant_memories = await self.memory_manager.get_contextual_memories(
                task_description=task,
                agent_type=agent_type,
                limit=5
            )
            
            # Prepare enhanced context with memories
            enhanced_context = context or {}
            if relevant_memories:
                memory_context = "\n".join([
                    f"Memory ({m.category}): {m.content[:200]}..." if len(m.content) > 200 else f"Memory ({m.category}): {m.content}"
                    for m in relevant_memories
                ])
                enhanced_context["relevant_memories"] = memory_context
            
            # Delegate task to the specialist agent
            response = await self._delegate_to_agent(agent_id, task, enhanced_context)
            
            # Store the interaction in memory
            if response:
                await self.memory_manager.store_agent_memory(
                    agent_type=agent_type,
                    category="interaction",
                    content=f"Task: {task}\nResponse: {response[:500]}{'...' if len(response) > 500 else ''}",
                    importance=7,
                    tags=["delegation", "interaction"]
                )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to delegate to specialist {agent_type}: {e}")
            return None
    
    async def cleanup_inactive_agents(self, max_idle_minutes: int = 30):
        """Clean up specialist agents that have been inactive for too long.
        
        Args:
            max_idle_minutes: Maximum idle time before cleanup
        """
        # This would implement logic to track agent activity and cleanup
        # For now, just log the intention
        self.logger.info(f"Agent cleanup scheduled for agents idle > {max_idle_minutes} minutes")
    
    def get_available_specialist_types(self) -> List[str]:
        """Get list of all available specialist agent types."""
        return list(self.available_agent_types.keys())
    
    def get_active_specialists(self) -> Dict[str, str]:
        """Get mapping of active specialist agent types to their IDs."""
        return dict(self.specialist_agents)
    
    async def get_agent_memory_summary(self, agent_type: str) -> Optional[str]:
        """Get a summary of memories for a specific agent type.
        
        Args:
            agent_type: Type of agent to get memory summary for
            
        Returns:
            Summary of agent memories or None if no memories found
        """
        try:
            memories = await self.memory_manager.retrieve_agent_memories(
                agent_type=agent_type,
                limit=10
            )
            
            if not memories:
                return None
            
            summary_parts = []
            for memory in memories:
                summary_parts.append(f"- {memory.category.title()}: {memory.content[:100]}{'...' if len(memory.content) > 100 else ''}")
            
            return f"Memory Summary for {agent_type}:\n" + "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to get memory summary for {agent_type}: {e}")
            return None

    # Health monitoring callback methods
    def _on_agent_health_change(self, agent_id: str, old_status, new_status):
        """Handle agent health status changes"""
        self.logger.info(f"Agent {agent_id} health changed: {old_status.value} -> {new_status.value}")
        
        # Update agent status tracking
        if agent_id in self.agent_status:
            self.agent_status[agent_id]['health_status'] = new_status.value
            self.agent_status[agent_id]['last_health_change'] = time.time()
        
        # Store health change in memory for learning
        asyncio.create_task(self.memory_manager.store_agent_memory(
            agent_type="orchestrator",
            category="health_monitoring",
            content=f"Agent {agent_id} health changed from {old_status.value} to {new_status.value}",
            importance=7 if new_status.value in ['failed', 'unhealthy'] else 5
        ))
    
    async def _on_agent_restart(self, agent_id: str, restart_count: int):
        """Handle agent restart events"""
        self.logger.warning(f"Agent {agent_id} restarted (restart #{restart_count})")
        
        # Attempt to restart the agent if we have it cached
        if agent_id in self.specialist_agents:
            try:
                # Clear the failed agent
                del self.specialist_agents[agent_id]
                
                # Extract agent type from agent_id (assuming format: type_timestamp)
                agent_type = agent_id.split('_')[0] if '_' in agent_id else agent_id
                
                # Respawn the agent
                await self.spawn_specialist_agent(agent_type, "Agent restarted due to health issues")
                
                self.logger.info(f"Successfully respawned agent {agent_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to respawn agent {agent_id}: {e}")
        
        # Store restart event in memory
        await self.memory_manager.store_agent_memory(
            agent_type="orchestrator", 
            category="agent_management",
            content=f"Agent {agent_id} restarted {restart_count} times due to health issues",
            importance=8
        )
    
    async def start_health_monitoring(self):
        """Start the health monitoring system"""
        await self.health_monitor.start_monitoring()
        self.logger.info("Health monitoring system started")
    
    async def stop_health_monitoring(self):
        """Stop the health monitoring system"""
        await self.health_monitor.stop_monitoring()
        self.logger.info("Health monitoring system stopped")
    
    def get_system_health(self):
        """Get overall system health summary"""
        return self.health_monitor.get_system_health_summary()
    
    def get_unhealthy_agents(self):
        """Get list of unhealthy agents"""
        return self.health_monitor.get_unhealthy_agents()

    
    # Planning system integration
    async def create_execution_plan(self, 
                                   task_description: str,
                                   objective: str,
                                   task_type: Optional[str] = None,
                                   context: Optional[Dict[str, Any]] = None):
        """Create a structured execution plan before task execution"""
        from .planning_system import PlanningSystem
        
        if not hasattr(self, 'planning_system'):
            self.planning_system = PlanningSystem(
                auto_approve_threshold="medium",
                plans_storage_path=".agentsmcp/execution_plans"
            )
        
        plan = await self.planning_system.create_plan(
            task_description=task_description,
            objective=objective, 
            task_type=task_type,
            context=context
        )
        
        # Store plan reference for execution
        self.active_plans[plan.plan_id] = plan
        
        # Log plan creation
        await self.memory_manager.store_agent_memory(
            agent_type="orchestrator",
            category="planning",
            content=f"Created execution plan {plan.plan_id}: {len(plan.steps)} steps, risk level {plan.risk_level}",
            importance=6
        )
        
        return plan
    
    async def execute_with_plan(self, plan_id: str) -> bool:
        """Execute a task using a structured plan"""
        if not hasattr(self, 'planning_system'):
            logger.error("Planning system not initialized")
            return False
        
        plan = self.planning_system.get_plan(plan_id)
        if not plan:
            logger.error(f"Plan {plan_id} not found")
            return False
        
        # Auto-approve if within threshold
        if not await self.planning_system.approve_plan(plan_id):
            logger.warning(f"Plan {plan_id} requires manual approval")
            return False
        
        # Execute plan with step-by-step agent delegation
        async def step_executor(step, plan):
            try:
                # Determine best agent for this step
                if step.assigned_agent:
                    agent_type = step.assigned_agent
                else:
                    # Auto-assign based on step description
                    agent_type = self._determine_step_agent(step.description)
                
                # Record step start in health monitor
                step_agent_id = f"{agent_type}_{int(time.time())}"
                self.health_monitor.register_agent(step_agent_id, agent_type)
                
                # Delegate step to appropriate agent
                start_time = time.time()
                
                if agent_type in self.available_agent_types:
                    result = await self.delegate_to_specialist(agent_type, step.description)
                else:
                    # Fallback to direct LLM execution
                    result = await self._route_to_llm_directly(step.description)
                
                response_time = (time.time() - start_time) * 1000
                
                # Record step completion in health monitor
                self.health_monitor.record_task_completion(
                    step_agent_id, 
                    success=True, 
                    response_time_ms=response_time
                )
                
                return True, str(result)
                
            except Exception as e:
                logger.error(f"Step execution failed: {e}")
                if 'step_agent_id' in locals():
                    self.health_monitor.record_task_completion(
                        step_agent_id,
                        success=False,
                        response_time_ms=(time.time() - start_time) * 1000,
                        error_details={'error': str(e), 'step': step.description}
                    )
                return False, str(e)
        
        success = await self.planning_system.execute_plan(plan_id, step_executor)
        
        if success:
            await self.memory_manager.store_agent_memory(
                agent_type="orchestrator",
                category="planning",
                content=f"Successfully executed plan {plan_id}",
                importance=7
            )
        
        return success
    
    def _determine_step_agent(self, step_description: str) -> str:
        """Determine the best agent type for a plan step"""
        step_lower = step_description.lower()
        
        # Agent selection based on step keywords
        if any(keyword in step_lower for keyword in ['code', 'implement', 'develop', 'program']):
            return 'senior-software-engineer'
        elif any(keyword in step_lower for keyword in ['test', 'qa', 'quality', 'verify']):
            return 'qa-engineer' 
        elif any(keyword in step_lower for keyword in ['design', 'architecture', 'system']):
            return 'principal-software-architect'
        elif any(keyword in step_lower for keyword in ['security', 'scan', 'vulnerability']):
            return 'security-engineer'
        elif any(keyword in step_lower for keyword in ['deploy', 'infrastructure', 'devops']):
            return 'devops-engineer'
        elif any(keyword in step_lower for keyword in ['analyze', 'data', 'metrics', 'report']):
            return 'data-scientist'
        elif any(keyword in step_lower for keyword in ['ui', 'ux', 'interface', 'user']):
            return 'ux-ui-designer'
        elif any(keyword in step_lower for keyword in ['document', 'specification', 'write']):
            return 'technical-writer'
        else:
            return 'senior-software-engineer'  # Default fallback
    
    async def handle_planning_command(self, command: str, task_description: str) -> str:
        """Handle planning-related commands (/plan, /spike, /tech-plan)"""
        try:
            if command == "/plan":
                # Create comprehensive execution plan
                plan = await self.create_execution_plan(
                    task_description=task_description,
                    objective=f"Complete task: {task_description}",
                    context={'complexity': 'medium', 'user_facing': True}
                )
                
                plan_summary = (
                    f"📋 **Execution Plan Created: {plan.plan_id}**\n"
                    f"**Objective:** {plan.objective}\n"
                    f"**Risk Level:** {plan.risk_level}\n"
                    f"**Estimated Duration:** {plan.estimated_total_duration_minutes} minutes\n"
                    f"**Steps:** {len(plan.steps)}\n\n"
                    f"**Plan Steps:**\n"
                )
                
                for i, step in enumerate(plan.steps, 1):
                    plan_summary += f"{i}. {step.description} ({step.estimated_duration_minutes}min)\n"
                
                if plan.identified_risks:
                    plan_summary += f"\n**Identified Risks:** {', '.join(plan.identified_risks)}\n"
                
                if plan.required_agents:
                    plan_summary += f"**Required Agents:** {', '.join(plan.required_agents)}\n"
                
                plan_summary += f"\n✅ Plan ready for execution with `/execute {plan.plan_id}`"
                
                return plan_summary
                
            elif command == "/spike":
                # Create research/investigation plan
                plan = await self.create_execution_plan(
                    task_description=f"Research spike: {task_description}",
                    objective=f"Investigate and document findings for: {task_description}",
                    task_type="data_analysis",
                    context={'complexity': 'low', 'research_focused': True}
                )
                
                return (
                    f"🔍 **Research Spike Plan: {plan.plan_id}**\n"
                    f"Investigation plan created with {len(plan.steps)} research steps.\n"
                    f"Estimated time: {plan.estimated_total_duration_minutes} minutes\n\n"
                    f"Ready to execute with `/execute {plan.plan_id}`"
                )
                
            elif command == "/tech-plan":
                # Create technical implementation plan
                plan = await self.create_execution_plan(
                    task_description=f"Technical implementation: {task_description}",
                    objective=f"Design and implement technical solution for: {task_description}",
                    task_type="code_development",
                    context={'complexity': 'high', 'technical_focus': True}
                )
                
                return (
                    f"⚙️ **Technical Implementation Plan: {plan.plan_id}**\n"
                    f"Technical plan with {len(plan.steps)} implementation steps.\n"
                    f"Risk level: {plan.risk_level}\n"
                    f"Required agents: {', '.join(plan.required_agents)}\n\n"
                    f"Ready to execute with `/execute {plan.plan_id}`"
                )
            
            elif command.startswith("/execute"):
                # Execute a specific plan
                plan_id = command.split(" ")[1] if len(command.split(" ")) > 1 else None
                if not plan_id:
                    return "❌ Please specify plan ID: `/execute <plan_id>`"
                
                success = await self.execute_with_plan(plan_id)
                
                if success:
                    return f"✅ **Plan {plan_id} executed successfully!**"
                else:
                    return f"❌ **Plan {plan_id} execution failed.** Check logs for details."
            
            else:
                return f"❌ Unknown planning command: {command}"
                
        except Exception as e:
            logger.error(f"Planning command error: {e}")
            return f"❌ Planning command failed: {str(e)}"
    
    def get_planning_summary(self) -> Dict[str, Any]:
        """Get summary of planning system status"""
        if not hasattr(self, 'planning_system'):
            return {'status': 'not_initialized'}
        
        return {
            'status': 'active',
            'summary': self.planning_system.get_plan_summary(),
            'active_plans': len(self.active_plans)
        }
