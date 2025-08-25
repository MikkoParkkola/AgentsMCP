"""
Distributed Orchestrator for AgentsMCP

Central coordination hub with 64K-128K context window optimized for:
- Task planning and decomposition
- Worker pool management  
- Cost optimization decisions
- Quality assurance and monitoring
- User interface coordination
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import uuid

from .message_queue import MessageQueue, Task, TaskResult, TaskPriority, TaskStatus
from .mesh_coordinator import (AgentMeshCoordinator, CollaborationRequest, 
                             CollaborationType, TrustLevel, AgentCapability)
from .governance import (GovernanceEngine, GovernancePolicy, AutonomyLevel, 
                        RiskLevel, EscalationRequest, EscalationResponse)
from .context_intelligence_clean import (ContextIntelligenceEngine, ContextItem, 
       ContextPriority, ContextType, ContextBudget)
from .multimodal_engine import (MultiModalEngine, AgentCapabilityProfile, ModalContent, 
                              ModalityType, ProcessingCapability)
from .ollama_turbo_integration import (OllamaHybridOrchestrator, OllamaMode, ModelTier,
                                     OllamaRequest, get_ollama_config_from_env)
from ..cost.tracker import CostTracker
from ..cost.optimizer import ModelOptimizer
from ..cost.budget import BudgetManager

logger = logging.getLogger(__name__)


# Supported orchestrator models with their specifications
ORCHESTRATOR_MODELS = {
    "gpt-5": {
        "context_limit": 400_000,
        "output_limit": 128_000,
        "cost_per_input": 1.25 / 1_000_000,   # $1.25 per M tokens
        "cost_per_output": 10.0 / 1_000_000,  # $10 per M tokens
        "performance_score": 74.9,  # SWE-bench score
        "strengths": ["task_decomposition", "cost_optimization", "systematic_planning"],
        "recommended_for": "Default choice - best performance + cost balance"
    },
    "claude-4.1-opus": {
        "context_limit": 200_000,
        "output_limit": 8_000,
        "cost_per_input": 15.0 / 1_000_000,   # $15 per M tokens
        "cost_per_output": 75.0 / 1_000_000,  # $75 per M tokens
        "performance_score": 74.5,
        "strengths": ["superior_reasoning", "complex_orchestration", "quality_assurance"],
        "recommended_for": "Premium orchestration when cost is not primary concern"
    },
    "claude-4.1-sonnet": {
        "context_limit": 200_000,
        "output_limit": 8_000,
        "cost_per_input": 3.0 / 1_000_000,    # $3 per M tokens
        "cost_per_output": 15.0 / 1_000_000,  # $15 per M tokens
        "performance_score": 72.7,
        "strengths": ["balanced_performance", "good_reasoning", "cost_effective"],
        "recommended_for": "Balanced choice between performance and cost"
    },
    "gemini-2.5-pro": {
        "context_limit": 2_000_000,
        "output_limit": 8_000,
        "cost_per_input": 1.25 / 1_000_000,   # $1.25 per M tokens (up to 200K)
        "cost_per_output": 10.0 / 1_000_000,  # $10 per M tokens
        "performance_score": 63.2,
        "strengths": ["massive_context", "multimodal", "cross_codebase_analysis"],
        "recommended_for": "When massive context windows (>400K tokens) are needed"
    },
    "qwen3-235b-a22b": {
        "context_limit": 1_000_000,
        "output_limit": 32_000,
        "cost_per_input": 0.0,  # Local model
        "cost_per_output": 0.0,  # Local model
        "performance_score": 70.0,  # Estimated
        "strengths": ["local_deployment", "privacy", "zero_api_cost"],
        "recommended_for": "Privacy-first deployments and offline orchestration"
    },
    "qwen3-32b": {
        "context_limit": 128_000,
        "output_limit": 8_000,
        "cost_per_input": 0.0,  # Local model
        "cost_per_output": 0.0,  # Local model
        "performance_score": 65.0,  # Estimated
        "strengths": ["local_deployment", "balanced_local_option", "apache_license"],
        "recommended_for": "Good local orchestrator for moderate complexity tasks"
    },
    "gpt-oss:120b": {
        "context_limit": 32_768,
        "output_limit": 8_000,
        "cost_per_input": 0.0,  # Included in subscription
        "cost_per_output": 0.0,  # Included in subscription
        "performance_score": 76.0,  # Beats many commercial models
        "strengths": ["ultra_performance", "subscription_included", "cloud_powered"],
        "recommended_for": "Premium orchestration with exceptional reasoning - Turbo subscription required",
        "provider": "ollama_turbo"
    },
    "gpt-oss:20b": {
        "context_limit": 32_768,
        "output_limit": 8_000,
        "cost_per_input": 0.0,  # Included in subscription
        "cost_per_output": 0.0,  # Included in subscription
        "performance_score": 71.0,  # Strong performance
        "strengths": ["high_performance", "hybrid_availability", "subscription_included"],
        "recommended_for": "Excellent performance available both locally and via Turbo",
        "provider": "ollama_hybrid"
    }
}


class DistributedOrchestrator:
    """
    Enhanced orchestrator with mesh collaboration capabilities.
    
    Responsibilities:
    - Task decomposition and planning (requires context about full request)
    - Worker pool coordination (lightweight worker status)
    - Cost optimization (aggregate metrics, not detailed history)
    - Quality monitoring (summary metrics, not full results)
    - User session management (current state, not full history)
    - Mesh-based agent collaboration and peer-to-peer task delegation
    - Dynamic trust management and collaboration pattern optimization
    """
    
    def __init__(self, 
                 max_workers: int = 20,
                 context_budget_tokens: int = 64000,
                 cost_budget: float = 100.0,
                 orchestrator_model: str = "gpt-5",
                 enable_mesh: bool = True,
                 max_mesh_size: int = 50,
                 enable_governance: bool = True,
                 governance_policy: GovernancePolicy = None,
                 enable_context_intelligence: bool = True,
                 context_budget: ContextBudget = None,
                 enable_multimodal: bool = True,
                 multimodal_cache_mb: int = 100,
                 max_concurrent_multimodal_tasks: int = 10,
                 enable_ollama_turbo: bool = True,
                 ollama_turbo_api_key: Optional[str] = None):
        
        self.max_workers = max_workers
        self.context_budget_tokens = context_budget_tokens
        self.enable_ollama_turbo = enable_ollama_turbo  # Set early for validation
        
        # Validate and set orchestrator model
        self.orchestrator_model = self._validate_orchestrator_model(orchestrator_model)
        self.model_config = ORCHESTRATOR_MODELS[self.orchestrator_model]
        
        # Adjust context budget based on model limits
        max_model_context = self.model_config["context_limit"]
        if context_budget_tokens > max_model_context:
            logger.warning(
                f"⚠️  Context budget ({context_budget_tokens:,}) exceeds {orchestrator_model} "
                f"limit ({max_model_context:,}). Adjusting to model limit."
            )
            self.context_budget_tokens = max_model_context
        
        # Core components
        self.message_queue = MessageQueue()
        self.cost_tracker = CostTracker()
        self.model_optimizer = ModelOptimizer(self.cost_tracker)
        self.budget_manager = BudgetManager(self.cost_tracker, cost_budget)
        
        # Mesh collaboration (optional)
        self.enable_mesh = enable_mesh
        self.mesh_coordinator = AgentMeshCoordinator(max_mesh_size) if enable_mesh else None
        
        # Governance framework (optional)
        self.enable_governance = enable_governance
        self.governance_engine = GovernanceEngine(governance_policy) if enable_governance else None
        
        # Context intelligence (optional)
        self.enable_context_intelligence = enable_context_intelligence
        self.context_intelligence = ContextIntelligenceEngine(context_budget) if enable_context_intelligence else None
        
        # Multi-modal processing engine (optional)
        self.enable_multimodal = enable_multimodal
        self.multimodal_engine = MultiModalEngine(max_concurrent_multimodal_tasks, multimodal_cache_mb) if enable_multimodal else None
        
        # Ollama Turbo integration (optional) - already set earlier
        self.ollama_orchestrator = None
        if enable_ollama_turbo:
            # Load configuration from environment or use provided key
            config = get_ollama_config_from_env()
            if ollama_turbo_api_key:
                config["turbo_api_key"] = ollama_turbo_api_key
            
            if config["turbo_api_key"]:
                from .ollama_turbo_integration import create_ollama_orchestrator
                self.ollama_orchestrator = create_ollama_orchestrator(
                    mode=config.get("mode", OllamaMode.HYBRID),
                    turbo_api_key=config["turbo_api_key"],
                    prefer_turbo=config.get("prefer_turbo", False)
                )
                logger.info("🦙 Ollama Turbo integration enabled")
            else:
                logger.warning("Ollama Turbo requested but no API key provided - falling back to local only")
                self.enable_ollama_turbo = False
        
        # Orchestrator state (kept lean for context efficiency)
        self.session_id = f"orch_{uuid.uuid4().hex[:8]}"
        self.start_time = datetime.utcnow()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Worker pool state (summarized)
        self.worker_pool = {
            "available_workers": {},
            "worker_capabilities": {},
            "load_balancing": {},
            "performance_summary": {}
        }
        
        # Task orchestration state
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        self.quality_gates: Dict[str, Dict[str, Any]] = {}
        
        # Context management
        self.context_usage = {
            "current_tokens": 0,
            "budget_tokens": context_budget_tokens,
            "state_summary": {},
            "compression_level": 0
        }
        
        # Set up message queue handlers
        self.message_queue.add_task_handler(self._on_task_created)
        self.message_queue.add_result_handler(self._on_task_completed)
        
        # Note: Mesh registration deferred to start() method
        
        # Initialize governance if enabled
        if self.enable_governance:
            self._initialize_governance()
        
        # Initialize context intelligence if enabled
        if self.enable_context_intelligence:
            self._initialize_context_intelligence()
        
        # Initialize multi-modal engine if enabled
        if self.enable_multimodal:
            self._initialize_multimodal_engine()
        
        mesh_status = "enabled" if self.enable_mesh else "disabled"
        governance_status = "enabled" if self.enable_governance else "disabled"
        context_intelligence_status = "enabled" if self.enable_context_intelligence else "disabled"
        multimodal_status = "enabled" if self.enable_multimodal else "disabled"
        ollama_turbo_status = "enabled" if self.enable_ollama_turbo else "disabled"
        logger.info(f"🎯 DistributedOrchestrator initialized - Session: {self.session_id}, Mesh: {mesh_status}, Governance: {governance_status}, Context Intelligence: {context_intelligence_status}, MultiModal: {multimodal_status}, Ollama Turbo: {ollama_turbo_status}")
        
        # Task tracking for proper cleanup
        self._background_tasks = []
        self._started = False
    
    async def start(self) -> Dict[str, Any]:
        """Start the orchestrator system."""
        
        if self._started:
            return {"status": "already_running", "session_id": self.session_id}
            
        logger.info("🚀 Starting DistributedOrchestrator")
        
        # Initialize background tasks
        self._background_tasks.append(
            asyncio.create_task(self._monitor_worker_health())
        )
        self._background_tasks.append(
            asyncio.create_task(self._optimize_costs_periodically())
        )
        self._background_tasks.append(
            asyncio.create_task(self._manage_context_budget())
        )
        self._background_tasks.append(
            asyncio.create_task(self._quality_assurance_loop())
        )
        
        # Initialize mesh optimization if enabled
        if self.enable_mesh:
            self._background_tasks.append(
                asyncio.create_task(self._mesh_optimization_loop())
            )
        
        # Start governance engine if enabled
        if self.governance_engine:
            await self.governance_engine.start()
        
        # Register in mesh if enabled
        if self.enable_mesh and self.mesh_coordinator:
            await self._register_orchestrator_in_mesh()
        
        self._started = True
        
        startup_status = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "max_workers": self.max_workers,
            "context_budget": self.context_budget_tokens,
            "cost_budget": self.budget_manager.monthly_limit,
            "status": "running",
            "capabilities": [
                "task_orchestration",
                "cost_optimization", 
                "quality_assurance",
                "worker_coordination",
                "context_management"
            ]
        }
        
        logger.info("✅ DistributedOrchestrator started successfully")
        return startup_status
    
    async def stop(self) -> Dict[str, Any]:
        """Stop the orchestrator system and clean up resources."""
        
        if not self._started:
            return {"status": "not_running"}
        
        logger.info("🛑 Stopping DistributedOrchestrator")
        
        # Stop governance engine if enabled
        if self.governance_engine:
            await self.governance_engine.stop()
        
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        self._started = False
        
        logger.info("✅ DistributedOrchestrator stopped successfully")
        return {"status": "stopped", "session_id": self.session_id}
    
    async def execute_request(self, user_request: str, session_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a high-level user request by orchestrating workers.
        
        This is where the orchestrator's context window is most valuable -
        understanding the full request, breaking it down intelligently,
        and coordinating the overall execution strategy.
        """
        
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        start_time = datetime.utcnow()
        
        logger.info(f"🎬 Executing request {request_id}: {user_request[:100]}...")
        
        try:
            # 1. Analyze request and plan execution (uses context for understanding)
            execution_plan = await self._analyze_and_plan_request(user_request, session_context)
            
            # 2. Break down into tasks 
            tasks = await self._decompose_into_tasks(execution_plan, request_id)
            
            # 3. Optimize task assignment for cost/quality
            optimized_assignments = await self._optimize_task_assignments(tasks)
            
            # 4. Submit tasks to workers via message queue
            task_ids = []
            for task, assignment in zip(tasks, optimized_assignments):
                task.context.update({
                    "request_id": request_id,
                    "assignment_reasoning": assignment["reasoning"],
                    "cost_limit": assignment["cost_limit"],
                    "quality_target": assignment["quality_target"]
                })
                
                task_id = await self.message_queue.submit_task(task)
                task_ids.append(task_id)
            
            # 5. Monitor execution and coordinate
            execution_result = await self._monitor_request_execution(request_id, task_ids)
            
            # 6. Synthesize results
            final_result = await self._synthesize_results(request_id, execution_result)
            
            # 7. Update session context for future requests
            await self._update_session_context(session_context, final_result)
            
            execution_time = datetime.utcnow() - start_time
            
            return {
                "request_id": request_id,
                "status": "completed",
                "execution_time": str(execution_time),
                "tasks_executed": len(task_ids),
                "total_cost": sum(r.get("cost", 0) for r in execution_result.values()),
                "quality_score": self._calculate_overall_quality(execution_result),
                "result": final_result,
                "orchestration_summary": {
                    "strategy": execution_plan["strategy"],
                    "worker_utilization": await self._get_worker_utilization(),
                    "cost_optimization": execution_plan["cost_optimization"],
                    "context_usage": self._get_context_usage()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Request {request_id} failed: {e}")
            return {
                "request_id": request_id,
                "status": "failed", 
                "error": str(e),
                "execution_time": str(datetime.utcnow() - start_time)
            }
    
    async def _analyze_and_plan_request(self, user_request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze request and create execution plan.
        
        This leverages the orchestrator's context window to understand:
        - Full request complexity and requirements
        - User's historical preferences and patterns
        - Available resources and constraints
        - Optimal execution strategy
        """
        
        # Analyze request complexity
        complexity_indicators = {
            "multi_step": any(word in user_request.lower() for word in 
                            ["then", "after", "next", "following", "phase", "stage"]),
            "requires_coordination": any(word in user_request.lower() for word in 
                                       ["coordinate", "integrate", "combine", "merge"]),
            "technical_depth": any(word in user_request.lower() for word in 
                                 ["architecture", "implementation", "system", "design"]),
            "creative_elements": any(word in user_request.lower() for word in 
                                   ["creative", "design", "ui", "ux", "brand"]),
            "data_processing": any(word in user_request.lower() for word in 
                                 ["analyze", "data", "report", "metrics"])
        }
        
        # Estimate resource requirements
        estimated_complexity = sum(complexity_indicators.values()) / len(complexity_indicators)
        
        # Determine optimal strategy
        if estimated_complexity > 0.6:
            strategy = "parallel_specialized"  # Multiple specialized workers
        elif estimated_complexity > 0.3:
            strategy = "sequential_coordinated"  # Coordinated steps
        else:
            strategy = "single_worker"  # Simple delegation
        
        # Cost optimization strategy
        cost_optimization = self.model_optimizer.recommend_model(
            task_type="orchestration",
            cost_priority=0.7,  # Prioritize cost for orchestration
            quality_priority=0.3
        )
        
        return {
            "strategy": strategy,
            "complexity_score": estimated_complexity,
            "estimated_workers_needed": min(self.max_workers, max(1, int(estimated_complexity * 5))),
            "cost_optimization": cost_optimization.__dict__,
            "requirements_analysis": complexity_indicators,
            "context_size_estimate": len(user_request) // 4,  # Rough token estimate
            "priority": TaskPriority.NORMAL
        }
    
    async def _decompose_into_tasks(self, execution_plan: Dict[str, Any], request_id: str) -> List[Task]:
        """Break down execution plan into specific worker tasks."""
        
        strategy = execution_plan["strategy"]
        complexity = execution_plan["complexity_score"]
        
        tasks = []
        
        if strategy == "single_worker":
            # Simple single task
            task = Task(
                description=f"Execute request {request_id}",
                priority=execution_plan.get("priority", TaskPriority.NORMAL),
                required_capabilities=["general"],
                max_cost=self.budget_manager.remaining_budget() * 0.1  # 10% of remaining budget
            )
            tasks.append(task)
            
        elif strategy == "sequential_coordinated":
            # Break into logical steps
            steps = [
                ("Analysis and Planning", ["analysis", "planning"]),
                ("Core Implementation", ["coding", "implementation"]),
                ("Quality Assurance", ["testing", "qa"]),
                ("Integration and Finalization", ["integration", "coordination"])
            ]
            
            for step_name, capabilities in steps:
                task = Task(
                    description=f"{step_name} for request {request_id}",
                    priority=execution_plan.get("priority", TaskPriority.NORMAL),
                    required_capabilities=capabilities,
                    max_cost=self.budget_manager.remaining_budget() * 0.05  # 5% per step
                )
                tasks.append(task)
                
        else:  # parallel_specialized
            # Create specialized parallel tasks
            specializations = [
                ("Frontend Development", ["frontend", "ui"]),
                ("Backend Development", ["backend", "api"]),
                ("Database Design", ["database", "data"]),
                ("DevOps Setup", ["devops", "deployment"]),
                ("Testing Suite", ["testing", "qa"])
            ]
            
            for spec_name, capabilities in specializations[:execution_plan["estimated_workers_needed"]]:
                task = Task(
                    description=f"{spec_name} for request {request_id}",
                    priority=execution_plan.get("priority", TaskPriority.NORMAL),
                    required_capabilities=capabilities,
                    max_cost=self.budget_manager.remaining_budget() * 0.08  # 8% per specialization
                )
                tasks.append(task)
        
        logger.info(f"📋 Decomposed request into {len(tasks)} tasks using {strategy} strategy")
        return tasks
    
    async def _optimize_task_assignments(self, tasks: List[Task]) -> List[Dict[str, Any]]:
        """Optimize task assignments for cost and quality."""
        
        assignments = []
        
        for task in tasks:
            # Get cost-optimized model recommendation
            recommendation = self.model_optimizer.recommend_model(
                task_type=task.required_capabilities[0] if task.required_capabilities else "general",
                cost_priority=0.6,
                quality_priority=0.4
            )
            
            assignment = {
                "recommended_model": recommendation.model,
                "cost_limit": task.max_cost,
                "quality_target": 0.8,  # Good quality target
                "reasoning": recommendation.reasoning,
                "estimated_cost": recommendation.estimated_cost * 1000,  # Estimate for 1000 tokens
                "estimated_quality": recommendation.estimated_quality
            }
            
            assignments.append(assignment)
        
        return assignments
    
    async def _monitor_request_execution(self, request_id: str, task_ids: List[str]) -> Dict[str, Any]:
        """Monitor execution of all tasks for a request."""
        
        logger.info(f"👀 Monitoring execution of {len(task_ids)} tasks for request {request_id}")
        
        # Wait for all tasks to complete with timeout
        timeout_seconds = 300  # 5 minutes timeout
        start_time = datetime.utcnow()
        
        results = {}
        completed_tasks = set()
        
        while len(completed_tasks) < len(task_ids):
            # Check for timeout
            if (datetime.utcnow() - start_time).total_seconds() > timeout_seconds:
                logger.warning(f"⏰ Request {request_id} execution timeout")
                break
            
            # Check task statuses
            for task_id in task_ids:
                if task_id in completed_tasks:
                    continue
                    
                status = await self.message_queue.get_task_status(task_id)
                if status and status["status"] in ["completed", "failed"]:
                    completed_tasks.add(task_id)
                    results[task_id] = status
            
            # Small delay to avoid busy waiting
            await asyncio.sleep(1.0)
        
        logger.info(f"📊 Request {request_id} monitoring complete: {len(completed_tasks)}/{len(task_ids)} tasks finished")
        return results
    
    async def _synthesize_results(self, request_id: str, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize worker results into final response."""
        
        # Collect successful results
        successful_results = [
            result for result in execution_results.values()
            if result.get("status") == "completed"
        ]
        
        # Calculate metrics
        total_cost = sum(r.get("cost", 0) for r in execution_results.values())
        average_quality = sum(r.get("quality_score", 0) for r in successful_results) / len(successful_results) if successful_results else 0
        
        synthesis = {
            "request_id": request_id,
            "status": "completed" if len(successful_results) > len(execution_results) * 0.5 else "partial_failure",
            "successful_tasks": len(successful_results),
            "total_tasks": len(execution_results),
            "total_cost": total_cost,
            "average_quality": average_quality,
            "execution_summary": "Tasks executed successfully with cost optimization",
            "results": successful_results
        }
        
        return synthesis
    
    async def _update_session_context(self, session_context: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Update session context with execution results for future context."""
        
        if not session_context:
            return
        
        session_id = session_context.get("session_id", "default")
        
        # Keep only essential context to manage token budget
        context_summary = {
            "last_request_cost": result.get("total_cost", 0),
            "last_request_quality": result.get("average_quality", 0),
            "last_execution_strategy": result.get("orchestration_summary", {}).get("strategy"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.active_sessions[session_id] = context_summary
    
    def _calculate_overall_quality(self, execution_results: Dict[str, Any]) -> float:
        """Calculate overall quality score from task results."""
        quality_scores = [r.get("quality_score", 0) for r in execution_results.values() if r.get("quality_score")]
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    async def _get_worker_utilization(self) -> Dict[str, Any]:
        """Get worker utilization summary."""
        queue_status = await self.message_queue.get_queue_status()
        
        return {
            "active_workers": queue_status["active_workers"],
            "total_queued_tasks": queue_status["total_queued"],
            "utilization_percentage": min(100, (queue_status["active_workers"] / self.max_workers) * 100) if self.max_workers > 0 else 0
        }
    
    def _get_context_usage(self) -> Dict[str, Any]:
        """Get current context window usage."""
        # Simplified context usage tracking
        estimated_tokens = len(json.dumps(self.active_sessions)) // 4  # Rough estimate
        
        return {
            "estimated_tokens_used": estimated_tokens,
            "budget_tokens": self.context_budget_tokens,
            "usage_percentage": (estimated_tokens / self.context_budget_tokens) * 100 if self.context_budget_tokens > 0 else 0,
            "compression_needed": estimated_tokens > self.context_budget_tokens * 0.8
        }
    
    # Background monitoring tasks
    
    async def _monitor_worker_health(self):
        """Monitor worker health and availability."""
        while True:
            try:
                queue_status = await self.message_queue.get_queue_status()
                
                # Log worker health summary
                if queue_status["active_workers"] < 2:
                    logger.warning(f"⚠️ Low worker count: {queue_status['active_workers']}")
                
                # Update worker pool summary for context efficiency
                self.worker_pool = {
                    "available_workers": queue_status["active_workers"],
                    "worker_capabilities": {},  # Simplified 
                    "load_balancing": {"strategy": "capability_based"},
                    "performance_summary": {"avg_response_time": "2.3s"}  # Mock
                }
                
            except Exception as e:
                logger.error(f"Worker health monitoring error: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _optimize_costs_periodically(self):
        """Periodically optimize costs and update budget status."""
        while True:
            try:
                # Check budget status
                within_budget = self.budget_manager.check_budget()
                if not within_budget:
                    logger.warning("💸 Over budget! Switching to cost-optimized models")
                
                # Update cost tracking
                total_cost = self.cost_tracker.total_cost
                if total_cost > 0:
                    logger.info(f"💰 Current session cost: ${total_cost:.6f}")
                
            except Exception as e:
                logger.error(f"Cost optimization error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _manage_context_budget(self):
        """Manage context budget and compress state when needed."""
        while True:
            try:
                context_usage = self._get_context_usage()
                
                if context_usage["compression_needed"]:
                    logger.info("🗜️ Compressing context state to stay within budget")
                    await self._compress_context_state()
                
            except Exception as e:
                logger.error(f"Context management error: {e}")
            
            await asyncio.sleep(45)  # Check every 45 seconds
    
    async def _quality_assurance_loop(self):
        """Monitor quality metrics and adjust strategies."""
        while True:
            try:
                # Simplified quality monitoring
                quality_gates = len([g for g in self.quality_gates.values() if g.get("passed", False)])
                total_gates = len(self.quality_gates)
                
                if total_gates > 0:
                    quality_percentage = (quality_gates / total_gates) * 100
                    if quality_percentage < 80:
                        logger.warning(f"⚠️ Quality below threshold: {quality_percentage:.1f}%")
                
            except Exception as e:
                logger.error(f"Quality assurance error: {e}")
            
            await asyncio.sleep(90)  # Check every 90 seconds
    
    async def _mesh_optimization_loop(self):
        """Optimize mesh collaboration patterns."""
        while True:
            try:
                if self.mesh_coordinator:
                    analytics = self.mesh_coordinator.get_collaboration_analytics()
                    if analytics["total_collaborations"] > 10:
                        optimization = await self.mesh_coordinator.optimize_collaboration_patterns()
                        logger.info(f"🕸️ Mesh optimization: {optimization['patterns_optimized']} patterns improved")
                
            except Exception as e:
                logger.error(f"Mesh optimization error: {e}")
            
            await asyncio.sleep(120)  # Check every 2 minutes
    
    async def _compress_context_state(self):
        """Compress orchestrator state to reduce context usage."""
        # Remove old session data
        cutoff_time = datetime.utcnow() - timedelta(minutes=30)
        
        sessions_to_remove = [
            session_id for session_id, session_data in self.active_sessions.items()
            if datetime.fromisoformat(session_data["timestamp"]) < cutoff_time
        ]
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
        
        # Compress workflow history
        old_workflows = [
            workflow_id for workflow_id, workflow in self.active_workflows.items()
            if workflow.get("completed", False)
        ]
        
        for workflow_id in old_workflows[:len(old_workflows)//2]:  # Remove half of old workflows
            del self.active_workflows[workflow_id]
        
        logger.info(f"🗜️ Compressed context: removed {len(sessions_to_remove)} sessions, {len(old_workflows)//2} workflows")
    
    # Mesh collaboration methods
    
    async def _register_orchestrator_in_mesh(self):
        """Register orchestrator as a node in the mesh network."""
        if not self.mesh_coordinator:
            return
        
        # Register orchestrator with special capabilities
        orchestrator_capabilities = {
            AgentCapability.ORCHESTRATION: 1.0,
            AgentCapability.COST_OPTIMIZATION: 0.9,
            AgentCapability.QUALITY_ASSURANCE: 0.8,
            AgentCapability.CONTEXT_MANAGEMENT: 0.9
        }
        
        await self.mesh_coordinator.register_agent(
            self.session_id,
            orchestrator_capabilities,
            {"type": "orchestrator", "model": self.orchestrator_model}
        )
        
        logger.info(f"🕸️ Registered orchestrator {self.session_id} in mesh network")
    
    async def delegate_to_mesh(self, task: Task, preferred_agents: List[str] = None) -> str:
        """Delegate task to mesh network for peer-to-peer execution."""
        if not self.mesh_coordinator:
            raise ValueError("Mesh coordination not enabled")
        
        # Create collaboration request
        collaboration_request = CollaborationRequest(
            requester_id=self.session_id,
            collaboration_type=CollaborationType.TASK_DELEGATION,
            required_capabilities=task.required_capabilities,
            description=task.description,
            priority=task.priority.value,
            max_cost=task.max_cost,
            context=task.context
        )
        
        # Find suitable collaborators
        collaborators = await self.mesh_coordinator.find_collaborators(
            collaboration_request,
            preferred_agents or []
        )
        
        if not collaborators:
            raise ValueError("No suitable collaborators found in mesh")
        
        # Delegate to best collaborator
        best_collaborator = collaborators[0]
        task_id = await self.mesh_coordinator.delegate_task(
            collaboration_request,
            best_collaborator["agent_id"]
        )
        
        logger.info(f"🤝 Delegated task to agent {best_collaborator['agent_id']} via mesh (task_id: {task_id})")
        return task_id
    
    async def get_mesh_analytics(self) -> Dict[str, Any]:
        """Get mesh collaboration analytics."""
        if not self.mesh_coordinator:
            return {"mesh_enabled": False}
        
        return self.mesh_coordinator.get_collaboration_analytics()
    
    # Governance integration methods
    
    def _initialize_governance(self):
        """Initialize governance framework for the orchestrator."""
        if not self.governance_engine:
            return
        
        # Register orchestrator with high autonomy level
        asyncio.create_task(
            self.governance_engine.register_agent(
                self.session_id,
                AutonomyLevel.HIGH,  # Orchestrator has high autonomy
                {"type": "orchestrator", "critical_operations": True}
            )
        )
        
        logger.info(f"⚖️ Initialized governance for orchestrator {self.session_id}")
    
    async def check_governance_approval(self, action: str, context: Dict[str, Any]) -> bool:
        """Check if orchestrator action requires governance approval."""
        if not self.governance_engine:
            return True  # No governance = always approved
        
        approval_result = await self.governance_engine.check_autonomy_permission(
            self.session_id,
            action,
            context
        )
        
        return approval_result["approved"]
    
    async def escalate_to_human(self, issue: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Escalate issue to human oversight."""
        if not self.governance_engine:
            return {"escalated": False, "reason": "No governance framework"}
        
        escalation_request = EscalationRequest(
            agent_id=self.session_id,
            issue_description=issue,
            urgency_level="medium",
            context=context
        )
        
        response = await self.governance_engine.escalate_to_human(escalation_request)
        
        logger.info(f"🚨 Escalated to human: {issue} (escalation_id: {response.escalation_id})")
        return {"escalated": True, "escalation_id": response.escalation_id}
    
    async def get_governance_analytics(self) -> Dict[str, Any]:
        """Get governance framework analytics."""
        if not self.governance_engine:
            return {"governance_enabled": False}
        
        return await self.governance_engine.get_governance_analytics()
    
    # Context intelligence integration methods
    
    def _initialize_context_intelligence(self):
        """Initialize context intelligence engine."""
        if not self.context_intelligence:
            return
        
        # Add initial context about orchestrator capabilities
        asyncio.create_task(self._seed_initial_context())
        
        logger.info(f"🧠 Initialized context intelligence for orchestrator {self.session_id}")
    
    async def _seed_initial_context(self):
        """Seed context intelligence with initial orchestrator context."""
        if not self.context_intelligence:
            return
        
        initial_contexts = [
            ContextItem(
                content=f"Orchestrator {self.session_id} with model {self.orchestrator_model}",
                content_type=ContextType.SYSTEM_STATE,
                priority=ContextPriority.HIGH,
                metadata={"agent_type": "orchestrator", "model": self.orchestrator_model}
            ),
            ContextItem(
                content=f"Budget: ${self.budget_manager.monthly_limit}, Workers: {self.max_workers}",
                content_type=ContextType.SYSTEM_STATE,
                priority=ContextPriority.MEDIUM,
                metadata={"budget": self.budget_manager.monthly_limit, "max_workers": self.max_workers}
            )
        ]
        
        for context_item in initial_contexts:
            await self.context_intelligence.add_context(context_item)
    
    async def add_execution_context(self, context: str, context_type: ContextType, priority: ContextPriority = ContextPriority.MEDIUM):
        """Add execution context to context intelligence."""
        if not self.context_intelligence:
            return
        
        context_item = ContextItem(
            content=context,
            content_type=context_type,
            priority=priority,
            metadata={"source": "orchestrator", "session_id": self.session_id}
        )
        
        await self.context_intelligence.add_context(context_item)
    
    async def get_relevant_context(self, query: str, max_tokens: int = None) -> List[ContextItem]:
        """Get relevant context for orchestration decisions."""
        if not self.context_intelligence:
            return []
        
        return await self.context_intelligence.get_relevant_context(
            query,
            self.session_id,
            max_tokens or self.context_budget_tokens // 4
        )
    
    async def get_context_intelligence_analytics(self) -> Dict[str, Any]:
        """Get context intelligence analytics."""
        if not self.context_intelligence:
            return {"context_intelligence_enabled": False}
        
        return await self.context_intelligence.get_analytics()
    
    # Multi-modal processing integration methods
    
    def _initialize_multimodal_engine(self):
        """Initialize multi-modal processing engine."""
        if not self.multimodal_engine:
            return
        
        # Register orchestrator's multi-modal capabilities
        orchestrator_profile = AgentCapabilityProfile(
            agent_id=self.session_id,
            supported_modalities={ModalityType.TEXT, ModalityType.STRUCTURED},
            capabilities={ProcessingCapability.ANALYSIS, ProcessingCapability.SYNTHESIS, ProcessingCapability.OPTIMIZATION},
            max_content_size={ModalityType.TEXT: 100000, ModalityType.STRUCTURED: 50000},
            processing_speed={ModalityType.TEXT: 1000.0, ModalityType.STRUCTURED: 500.0},
            cost_per_operation={ModalityType.TEXT: 0.001, ModalityType.STRUCTURED: 0.002},
            specializations={"orchestration", "cost_optimization", "task_planning"}
        )
        
        asyncio.create_task(self.multimodal_engine.register_agent(orchestrator_profile))
        
        logger.info(f"🎭 Initialized multi-modal engine for orchestrator {self.session_id}")
    
    async def register_agent_multimodal_profile(self, profile: AgentCapabilityProfile):
        """Register an agent's multi-modal processing capabilities."""
        if not self.multimodal_engine:
            logger.warning("Multi-modal engine not enabled")
            return
        
        await self.multimodal_engine.register_agent(profile)
        logger.info(f"📝 Registered multi-modal profile for agent {profile.agent_id}")
    
    async def process_multimodal_content(self, content: ModalContent, capability: ProcessingCapability, 
                                       agent_id: Optional[str] = None, parameters: Dict[str, Any] = None) -> ModalContent:
        """Process multi-modal content using available agents."""
        if not self.multimodal_engine:
            raise ValueError("Multi-modal processing not enabled")
        
        # Store content first
        content_id = await self.multimodal_engine.store_content(content)
        
        # Process content
        result = await self.multimodal_engine.process_content(
            content_id, capability, agent_id, parameters
        )
        
        # Add context about processing
        if self.context_intelligence:
            await self.add_execution_context(
                f"Processed {content.modality.value} content with {capability.value}",
                ContextType.TASK_EXECUTION,
                ContextPriority.LOW
            )
        
        return result
    
    async def batch_process_multimodal(self, content_list: List[ModalContent], capability: ProcessingCapability,
                                     parameters: Dict[str, Any] = None) -> List[ModalContent]:
        """Batch process multiple pieces of multi-modal content."""
        if not self.multimodal_engine:
            raise ValueError("Multi-modal processing not enabled")
        
        # Store all content first
        content_ids = []
        for content in content_list:
            content_id = await self.multimodal_engine.store_content(content)
            content_ids.append(content_id)
        
        # Batch process
        results = await self.multimodal_engine.batch_process(content_ids, capability, parameters)
        
        # Add context about batch processing
        if self.context_intelligence:
            await self.add_execution_context(
                f"Batch processed {len(content_list)} items with {capability.value}",
                ContextType.TASK_EXECUTION,
                ContextPriority.LOW
            )
        
        return results
    
    async def create_multimodal_pipeline(self, content: ModalContent, 
                                       pipeline: List[Tuple[ProcessingCapability, Optional[str], Optional[Dict[str, Any]]]]) -> List[ModalContent]:
        """Create a processing pipeline for multi-modal content."""
        if not self.multimodal_engine:
            raise ValueError("Multi-modal processing not enabled")
        
        # Store initial content
        content_id = await self.multimodal_engine.store_content(content)
        
        # Execute pipeline
        results = await self.multimodal_engine.create_content_pipeline(content_id, pipeline)
        
        # Add context about pipeline execution
        if self.context_intelligence:
            await self.add_execution_context(
                f"Executed {len(pipeline)}-step multimodal pipeline on {content.modality.value} content",
                ContextType.TASK_EXECUTION,
                ContextPriority.MEDIUM
            )
        
        return results
    
    async def get_multimodal_capability_matrix(self) -> Dict[str, Dict[str, List[str]]]:
        """Get the multi-modal capability matrix showing which agents support what."""
        if not self.multimodal_engine:
            return {}
        
        return await self.multimodal_engine.get_capability_matrix()
    
    async def get_multimodal_analytics(self) -> Dict[str, Any]:
        """Get multi-modal processing analytics."""
        if not self.multimodal_engine:
            return {"multimodal_enabled": False}
        
        stats = await self.multimodal_engine.get_processing_stats()
        optimization = await self.multimodal_engine.optimize_agent_assignments()
        
        return {
            "multimodal_enabled": True,
            "processing_stats": stats,
            "optimization_recommendations": optimization
        }
    
    # Enhanced orchestrator analytics
    
    async def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics from all orchestrator components."""
        analytics = {
            "orchestrator": {
                "session_id": self.session_id,
                "uptime": str(datetime.utcnow() - self.start_time),
                "model": self.orchestrator_model,
                "context_budget": self.context_budget_tokens,
                "active_sessions": len(self.active_sessions),
                "active_workflows": len(self.active_workflows)
            },
            "cost_tracking": {
                "total_cost": self.cost_tracker.total_cost,
                "budget_remaining": self.budget_manager.remaining_budget(),
                "cost_per_input": f"${self.model_config['cost_per_input'] * 1_000_000:.2f}/M tokens",
                "cost_per_output": f"${self.model_config['cost_per_output'] * 1_000_000:.2f}/M tokens"
            }
        }
        
        # Add mesh analytics
        if self.enable_mesh:
            analytics["mesh_collaboration"] = await self.get_mesh_analytics()
        
        # Add governance analytics
        if self.enable_governance:
            analytics["governance_framework"] = await self.get_governance_analytics()
        
        # Add context intelligence analytics
        if self.enable_context_intelligence:
            analytics["context_intelligence"] = await self.get_context_intelligence_analytics()
        
        # Add multi-modal analytics
        if self.enable_multimodal:
            analytics["multimodal_processing"] = await self.get_multimodal_analytics()
        
        return analytics
    
    async def _mesh_optimization_loop(self):
        """Periodically optimize mesh network and collaboration patterns."""
        while True:
            try:
                if self.mesh_coordinator:
                    # Run mesh optimization
                    optimization_report = await self.optimize_mesh_network()
                    
                    # Log key insights
                    if "underperformers" in optimization_report:
                        underperformers = optimization_report["underperformers"]
                        if len(underperformers) > 0:
                            logger.warning(f"⚠️ Mesh underperformers detected: {len(underperformers)} agents")
                    
                    # Adjust collaboration thresholds based on network performance
                    analytics = self.get_mesh_analytics()
                    if analytics.get("average_trust", 0) < 0.5:
                        logger.warning("🚨 Low average mesh trust - adjusting collaboration criteria")
                        # Could implement dynamic trust threshold adjustment here
                    
                    logger.info(f"🕸️ Mesh optimization cycle completed - {analytics.get('total_agents', 0)} agents active")
            
            except Exception as e:
                logger.error(f"Mesh optimization error: {e}")
            
            await asyncio.sleep(600)  # Optimize every 10 minutes
    
    # Governance integration methods
    
    async def check_governance_permission(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if orchestrator has permission to perform an action."""
        if not self.governance_engine:
            return {"permission_granted": True, "governance_enabled": False}
        
        return await self.governance_engine.check_autonomy_permission(
            self.session_id, action, context
        )
    
    async def request_governance_escalation(self, action: str, context: Dict[str, Any]) -> Optional[str]:
        """Request governance escalation for high-risk or high-cost actions."""
        if not self.governance_engine:
            return None
        
        permission = await self.check_governance_permission(action, context)
        
        if permission.get("escalation_required", False):
            escalation_id = permission.get("escalation_request_id")
            logger.info(f"🚨 Governance escalation requested: {escalation_id} for action: {action[:50]}...")
            return escalation_id
        
        return None
    
    async def wait_for_governance_approval(self, escalation_id: str, timeout_minutes: int = 120) -> bool:
        """Wait for governance approval with timeout."""
        if not self.governance_engine or not escalation_id:
            return True
        
        start_time = datetime.utcnow()
        timeout_delta = timedelta(minutes=timeout_minutes)
        
        while (datetime.utcnow() - start_time) < timeout_delta:
            # Check if escalation is still pending
            if escalation_id not in self.governance_engine.pending_escalations:
                # Escalation has been resolved
                logger.info(f"✅ Governance escalation {escalation_id} resolved")
                return True
            
            await asyncio.sleep(30)  # Check every 30 seconds
        
        logger.warning(f"⏰ Governance escalation {escalation_id} timed out")
        return False
    
    def get_governance_analytics(self) -> Dict[str, Any]:
        """Get governance analytics and performance metrics."""
        if not self.governance_engine:
            return {"governance_enabled": False}
        
        analytics = self.governance_engine.get_governance_analytics()
        analytics["governance_enabled"] = True
        return analytics
    
    async def _enhanced_task_execution_with_governance(self, tasks: List[Task]) -> List[Dict[str, Any]]:
        """Enhanced task execution with governance checks."""
        governed_results = []
        
        for task in tasks:
            # Check governance permission before execution
            action = f"execute_task: {task.description}"
            context = {
                "estimated_cost": task.max_cost or 0.0,
                "task_priority": task.priority.value,
                "required_capabilities": task.required_capabilities,
                "task_context": task.context
            }
            
            permission = await self.check_governance_permission(action, context)
            
            if not permission["permission_granted"]:
                # Escalation required
                escalation_id = permission.get("escalation_request_id")
                
                if escalation_id:
                    logger.info(f"🚨 Task requires governance approval: {task.task_id}")
                    
                    # Wait for approval (with shorter timeout for tasks)
                    approved = await self.wait_for_governance_approval(escalation_id, timeout_minutes=30)
                    
                    if not approved:
                        governed_results.append({
                            "task_id": task.task_id,
                            "status": "governance_denied",
                            "governance_check": permission,
                            "escalation_id": escalation_id,
                            "reason": "Governance approval not received within timeout"
                        })
                        continue
                else:
                    # No escalation created, action denied
                    governed_results.append({
                        "task_id": task.task_id,
                        "status": "governance_denied", 
                        "governance_check": permission,
                        "reason": "Action denied by governance policy"
                    })
                    continue
            
            # Permission granted, execute task
            if self.enable_mesh:
                result = await self._enhanced_task_execution_with_mesh([task])
                if result:
                    governed_result = result[0]
                    governed_result["governance_check"] = permission
                    governed_results.append(governed_result)
            else:
                result = await self._execute_standard_task(task)
                result["governance_check"] = permission
                governed_results.append(result)
        
        return governed_results
    
    # Context Intelligence integration methods
    
    async def add_execution_context(self, 
                                  content: Any, 
                                  context_type: ContextType = ContextType.TASK_STATE,
                                  priority: ContextPriority = None) -> Optional[str]:
        """Add execution context to the intelligence system."""
        if not self.context_intelligence:
            return None
        
        context_id = await self.context_intelligence.add_context(
            content=content,
            agent_id=self.session_id,
            content_type=context_type,
            priority=priority,
            tags={"orchestrator", "execution"},
            expiry_minutes=180  # 3 hour default expiry
        )
        
        logger.debug(f"🧠 Added execution context: {context_id}")
        return context_id
    
    async def get_relevant_execution_context(self, query: str, max_tokens: int = None) -> List[ContextItem]:
        """Get relevant context for execution planning."""
        if not self.context_intelligence:
            return []
        
        max_tokens = max_tokens or (self.context_budget_tokens // 4)
        
        relevant_context = await self.context_intelligence.get_relevant_context(
            query=query,
            agent_id=self.session_id,
            max_tokens=max_tokens,
            include_types=[
                ContextType.TASK_STATE,
                ContextType.USER_INTENT,
                ContextType.EXECUTION_HISTORY,
                ContextType.PERFORMANCE_METRICS
            ]
        )
        
        logger.debug(f"🎯 Retrieved {len(relevant_context)} relevant context items")
        return relevant_context
    
    async def share_context_with_worker(self, worker_id: str, context_ids: List[str]) -> int:
        """Share relevant context with a worker agent."""
        if not self.context_intelligence:
            return 0
        
        shared_count = await self.context_intelligence.share_context_between_agents(
            from_agent=self.session_id,
            to_agent=worker_id,
            context_ids=context_ids
        )
        
        logger.info(f"🤝 Shared {shared_count} context items with worker {worker_id}")
        return shared_count
    
    def get_context_intelligence_analytics(self) -> Dict[str, Any]:
        """Get context intelligence analytics."""
        if not self.context_intelligence:
            return {"context_intelligence_enabled": False}
        
        analytics = self.context_intelligence.get_context_analytics()
        analytics["context_intelligence_enabled"] = True
        return analytics
    
    async def _enhanced_task_execution_with_context_intelligence(self, tasks: List[Task]) -> List[Dict[str, Any]]:
        """Enhanced task execution with intelligent context management."""
        if not self.context_intelligence:
            # Fallback to governance-enhanced execution
            if self.enable_governance:
                return await self._enhanced_task_execution_with_governance(tasks)
            elif self.enable_mesh:
                return await self._enhanced_task_execution_with_mesh(tasks)
            else:
                return await self._standard_task_execution(tasks)
        
        context_enhanced_results = []
        
        for task in tasks:
            # Add task context to intelligence system
            task_context_id = await self.add_execution_context(
                content={
                    "task_id": task.task_id,
                    "description": task.description,
                    "priority": task.priority.value,
                    "capabilities": task.required_capabilities,
                    "max_cost": task.max_cost,
                    "context": task.context
                },
                context_type=ContextType.TASK_STATE,
                priority=ContextPriority.HIGH
            )
            
            # Get relevant historical context
            relevant_context = await self.get_relevant_execution_context(
                query=f"task: {task.description}",
                max_tokens=self.context_budget_tokens // 8  # Use 12.5% of budget per task
            )
            
            # Enrich task with relevant context
            if relevant_context:
                historical_insights = [
                    {
                        "type": ctx.content_type.value,
                        "priority": ctx.priority.value,
                        "content": ctx.content,
                        "relevance": ctx.relevance_score
                    }
                    for ctx in relevant_context[:5]  # Top 5 most relevant
                ]
                
                task.context["historical_context"] = historical_insights
                task.context["context_intelligence_used"] = True
            
            # Execute with enhanced context through governance/mesh if available
            if self.enable_governance:
                result = await self._enhanced_task_execution_with_governance([task])
                if result:
                    enhanced_result = result[0]
            elif self.enable_mesh:
                result = await self._enhanced_task_execution_with_mesh([task])
                if result:
                    enhanced_result = result[0]
            else:
                enhanced_result = await self._execute_standard_task(task)
            
            # Add execution result to context intelligence
            await self.add_execution_context(
                content={
                    "task_id": task.task_id,
                    "status": enhanced_result.get("status", "unknown"),
                    "cost": enhanced_result.get("cost", 0),
                    "quality_score": enhanced_result.get("quality_score", 0),
                    "execution_time": enhanced_result.get("execution_time", "unknown"),
                    "collaboration_type": enhanced_result.get("collaboration_type", "standard")
                },
                context_type=ContextType.EXECUTION_HISTORY,
                priority=ContextPriority.MEDIUM
            )
            
            enhanced_result["context_intelligence"] = {
                "task_context_id": task_context_id,
                "relevant_context_count": len(relevant_context),
                "context_tokens_used": sum(ctx.token_estimate for ctx in relevant_context)
            }
            
            context_enhanced_results.append(enhanced_result)
        
        return context_enhanced_results
    
    async def _compress_state(self):
        """Compress orchestrator state to fit within context budget."""
        
        # Keep only most recent sessions
        if len(self.active_sessions) > 10:
            sorted_sessions = sorted(
                self.active_sessions.items(),
                key=lambda x: x[1].get("timestamp", ""),
                reverse=True
            )
            self.active_sessions = dict(sorted_sessions[:10])
        
        # Summarize old workflows
        old_workflows = []
        for workflow_id, workflow in list(self.active_workflows.items()):
            if workflow.get("status") in ["completed", "failed"]:
                old_workflows.append(workflow_id)
        
        for workflow_id in old_workflows[:5]:  # Keep only 5 recent completed workflows
            del self.active_workflows[workflow_id]
        
        logger.info(f"🗜️ State compressed - Sessions: {len(self.active_sessions)}, Workflows: {len(self.active_workflows)}")
    
    # Mesh collaboration methods
    
    
    def _initialize_governance(self):
        """Initialize governance framework for the orchestrator."""
        if not self.governance_engine:
            return
        
        # Set orchestrator-specific governance policy
        orchestrator_policy = GovernancePolicy(
            autonomy_level=AutonomyLevel.SUPERVISED,  # Orchestrator needs some supervision
            cost_limit=self.budget_manager.monthly_limit * 0.1,  # 10% of monthly budget per decision
            risk_threshold=RiskLevel.MEDIUM,
            compliance_checks_required=True,
            audit_trail_required=True
        )
        
        self.governance_engine.set_agent_policy(self.session_id, orchestrator_policy)
        
        # Register orchestrator as a governed agent
        logger.info(f"🏦 Governance framework initialized for orchestrator")
    
    def _initialize_context_intelligence(self):
        """Initialize context intelligence system for enhanced context management."""
        if not self.context_intelligence:
            return
        
        # Create custom context budget if not provided
        if not hasattr(self, 'context_budget') or not self.context_budget:
            self.context_budget = ContextBudget(
                total_tokens=self.context_budget_tokens,
                reserved_tokens=min(8000, self.context_budget_tokens // 8),
                priority_allocation={
                    ContextPriority.CRITICAL: 0.4,    # 40% for critical context
                    ContextPriority.HIGH: 0.3,        # 30% for high priority
                    ContextPriority.MEDIUM: 0.2,      # 20% for medium priority  
                    ContextPriority.LOW: 0.08,        # 8% for low priority
                    ContextPriority.MINIMAL: 0.02     # 2% for minimal priority
                }
            )
        
        logger.info(f"🧠 Context intelligence initialized with {self.context_budget_tokens} token budget")
    
    async def request_mesh_collaboration(self, 
                                       task_description: str,
                                       collaboration_type: CollaborationType = CollaborationType.CONSULTATION,
                                       required_capabilities: List[str] = None,
                                       trust_level: TrustLevel = TrustLevel.MEDIUM) -> Optional[str]:
        """Request collaboration from other agents in the mesh."""
        if not self.mesh_coordinator:
            logger.warning("Mesh collaboration requested but mesh is disabled")
            return None
        
        request = CollaborationRequest(
            requesting_agent=self.session_id,
            collaboration_type=collaboration_type,
            task_description=task_description,
            required_capabilities=required_capabilities or [],
            trust_required=trust_level,
            max_cost=self.budget_manager.remaining_budget() * 0.05,  # 5% of remaining budget
            deadline=datetime.utcnow() + timedelta(minutes=30)
        )
        
        collaborator = await self.mesh_coordinator.request_collaboration(request)
        
        if collaborator:
            logger.info(f"🤝 Mesh collaboration established: {self.session_id} -> {collaborator}")
            return collaborator
        else:
            logger.warning(f"🚫 No suitable mesh collaborator found for: {task_description[:50]}...")
            return None
    
    async def get_collaboration_recommendations(self, task_type: str) -> List[Dict[str, Any]]:
        """Get mesh collaboration recommendations for a specific task type."""
        if not self.mesh_coordinator:
            return []
        
        recommendations = self.mesh_coordinator.get_collaboration_recommendations(
            self.session_id, task_type
        )
        
        logger.info(f"📋 Found {len(recommendations)} mesh collaboration patterns for {task_type}")
        return recommendations
    
    async def update_mesh_trust(self, agent_id: str, performance_score: float, outcome: str):
        """Update trust scores in the mesh network based on collaboration outcomes."""
        if not self.mesh_coordinator:
            return
        
        self.mesh_coordinator.update_trust_score(
            self.session_id, agent_id, performance_score, outcome
        )
        
        logger.info(f"🎯 Updated mesh trust for {agent_id}: {outcome} (score: {performance_score})")
    
    async def optimize_mesh_network(self) -> Dict[str, Any]:
        """Optimize the mesh network configuration and relationships."""
        if not self.mesh_coordinator:
            return {"mesh_enabled": False}
        
        optimization_report = await self.mesh_coordinator.optimize_mesh()
        
        logger.info(f"⚙️ Mesh network optimization completed")
        return optimization_report
    
    def get_mesh_analytics(self) -> Dict[str, Any]:
        """Get analytics about mesh network performance and collaboration patterns."""
        if not self.mesh_coordinator:
            return {"mesh_enabled": False}
        
        analytics = self.mesh_coordinator.get_mesh_analytics()
        analytics["mesh_enabled"] = True
        analytics["orchestrator_id"] = self.session_id
        
        return analytics
    
    async def _enhanced_task_execution_with_mesh(self, tasks: List[Task]) -> List[Dict[str, Any]]:
        """Enhanced task execution that leverages mesh collaboration patterns."""
        if not self.mesh_coordinator:
            # Fallback to standard execution
            return await self._standard_task_execution(tasks)
        
        enhanced_results = []
        
        for task in tasks:
            # Check if mesh collaboration would benefit this task
            task_type = task.required_capabilities[0] if task.required_capabilities else "general"
            recommendations = await self.get_collaboration_recommendations(task_type)
            
            if recommendations and len(recommendations) > 0:
                # Try mesh collaboration first
                best_pattern = max(recommendations, key=lambda x: x.get("confidence", 0))
                
                if best_pattern["confidence"] > 0.7:  # High confidence in collaboration pattern
                    logger.info(f"🕸️ Using mesh collaboration for task: {task.task_id}")
                    
                    collaborator = await self.request_mesh_collaboration(
                        task.description,
                        CollaborationType(best_pattern["type"]) if "type" in best_pattern else CollaborationType.CONSULTATION,
                        task.required_capabilities,
                        TrustLevel.MEDIUM
                    )
                    
                    if collaborator:
                        # Execute with mesh collaboration
                        result = await self._execute_task_with_collaborator(task, collaborator)
                        enhanced_results.append(result)
                        continue
            
            # Fallback to standard execution
            result = await self._execute_standard_task(task)
            enhanced_results.append(result)
        
        return enhanced_results
    
    async def _execute_task_with_collaborator(self, task: Task, collaborator: str) -> Dict[str, Any]:
        """Execute a task with mesh collaboration."""
        start_time = datetime.utcnow()
        
        try:
            # Add collaboration context to task
            task.context["collaboration"] = {
                "collaborator": collaborator,
                "collaboration_type": "mesh_assisted",
                "orchestrator": self.session_id
            }
            
            # Submit to message queue with collaboration info
            task_id = await self.message_queue.submit_task(task)
            
            # Wait for completion with enhanced monitoring
            result = await self._monitor_collaborative_task(task_id, collaborator)
            
            # Update trust based on result
            performance_score = result.get("quality_score", 0.5)
            outcome = "success" if result.get("status") == "completed" else "failure"
            await self.update_mesh_trust(collaborator, performance_score, outcome)
            
            execution_time = datetime.utcnow() - start_time
            
            return {
                "task_id": task_id,
                "status": result.get("status", "completed"),
                "execution_time": str(execution_time),
                "collaboration_type": "mesh_assisted",
                "collaborator": collaborator,
                "cost": result.get("cost", 0),
                "quality_score": performance_score,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Mesh collaboration task failed: {e}")
            await self.update_mesh_trust(collaborator, 0.0, "failure")
            
            return {
                "task_id": task.task_id,
                "status": "failed",
                "error": str(e),
                "collaboration_type": "mesh_failed",
                "collaborator": collaborator
            }
    
    async def _monitor_collaborative_task(self, task_id: str, collaborator: str) -> Dict[str, Any]:
        """Monitor a collaborative task with enhanced tracking."""
        # Enhanced monitoring for collaborative tasks
        timeout = 300  # 5 minutes
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            status = await self.message_queue.get_task_status(task_id)
            
            if status and status["status"] in ["completed", "failed"]:
                return status
            
            # Log progress for collaborative tasks
            if status and "progress" in status:
                logger.info(f"🔄 Collaborative task {task_id} progress: {status['progress']}")
            
            await asyncio.sleep(2.0)  # Check every 2 seconds for collaborative tasks
        
        logger.warning(f"⏰ Collaborative task {task_id} with {collaborator} timed out")
        return {"status": "timeout", "collaborator": collaborator}
    
    async def _execute_standard_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task using standard (non-mesh) approach."""
        task_id = await self.message_queue.submit_task(task)
        
        # Standard monitoring
        timeout = 180  # 3 minutes for standard tasks
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            status = await self.message_queue.get_task_status(task_id)
            
            if status and status["status"] in ["completed", "failed"]:
                return {
                    "task_id": task_id,
                    "status": status["status"],
                    "collaboration_type": "standard",
                    "cost": status.get("cost", 0),
                    "quality_score": status.get("quality_score", 0.5),
                    "result": status
                }
            
            await asyncio.sleep(1.0)
        
        return {
            "task_id": task_id,
            "status": "timeout",
            "collaboration_type": "standard"
        }
    
    async def _standard_task_execution(self, tasks: List[Task]) -> List[Dict[str, Any]]:
        """Standard task execution without mesh collaboration."""
        results = []
        
        for task in tasks:
            result = await self._execute_standard_task(task)
            results.append(result)
        
        return results
    
    # Event handlers
    
    def _on_task_created(self, task: Task):
        """Handle new task creation."""
        logger.debug(f"📝 Task created: {task.task_id}")
    
    def _on_task_completed(self, result: TaskResult):
        """Handle task completion."""
        logger.debug(f"✅ Task completed: {result.task_id} by {result.worker_id}")
        
        # Update cost tracking
        if result.cost > 0:
            from ..cost.models import CostRecord
            record = CostRecord(
                call_id=result.task_id,
                provider="worker",
                model=result.worker_id,
                task="distributed_task",
                input_tokens=result.tokens_used // 2,
                output_tokens=result.tokens_used // 2,
                cost=result.cost
            )
            self.cost_tracker.record_call(record)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        
        queue_status = await self.message_queue.get_queue_status()
        context_usage = self._get_context_usage()
        
        return {
            "session_id": self.session_id,
            "uptime": str(datetime.utcnow() - self.start_time),
            "status": "running",
            "worker_pool": await self._get_worker_utilization(),
            "task_queue": queue_status,
            "cost_tracking": {
                "total_cost": self.cost_tracker.total_cost,
                "daily_cost": self.cost_tracker.get_daily_cost(),
                "within_budget": self.budget_manager.check_budget(),
                "remaining_budget": self.budget_manager.remaining_budget()
            },
            "context_management": context_usage,
            "active_sessions": len(self.active_sessions),
            "active_workflows": len(self.active_workflows),
            "orchestrator_model": {
                "model": self.orchestrator_model,
                "context_limit": self.model_config["context_limit"],
                "performance_score": self.model_config["performance_score"],
                "cost_per_input": f"${self.model_config['cost_per_input'] * 1_000_000:.2f}/M tokens",
                "cost_per_output": f"${self.model_config['cost_per_output'] * 1_000_000:.2f}/M tokens"
            },
            "mesh_collaboration": self.get_mesh_analytics(),
            "governance_framework": self.get_governance_analytics(),
            "context_intelligence": self.get_context_intelligence_analytics()
        }
    
    def _validate_orchestrator_model(self, model: str) -> str:
        """Validate and return the orchestrator model."""
        if model not in ORCHESTRATOR_MODELS:
            logger.warning(
                f"⚠️  Unknown orchestrator model '{model}'. "
                f"Available: {list(ORCHESTRATOR_MODELS.keys())}. "
                "Falling back to default 'gpt-5'."
            )
            return "gpt-5"
        
        logger.info(
            f"🎩 Orchestrator model: {model} "
            f"(Performance: {ORCHESTRATOR_MODELS[model]['performance_score']}%, "
            f"Context: {ORCHESTRATOR_MODELS[model]['context_limit']:,} tokens)"
        )
        return model
    
    @staticmethod
    def get_available_models() -> Dict[str, Dict[str, Any]]:
        """Get all available orchestrator models and their specifications."""
        return ORCHESTRATOR_MODELS.copy()
    
    @staticmethod
    def get_model_recommendation(use_case: str) -> str:
        """Get model recommendation based on use case."""
        recommendations = {
            "default": "gpt-5",
            "cost_effective": "gpt-5",
            "premium": "claude-4.1-opus",
            "balanced": "claude-4.1-sonnet",
            "massive_context": "gemini-2.5-pro",
            "local": "qwen3-235b-a22b",
            "privacy": "qwen3-235b-a22b",
            "offline": "qwen3-32b"
        }
        return recommendations.get(use_case, "gpt-5")
    
    def estimate_orchestration_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for orchestration task."""
        input_cost = input_tokens * self.model_config["cost_per_input"]
        output_cost = output_tokens * self.model_config["cost_per_output"]
        return input_cost + output_cost
    
    # Enhanced Integration Methods for Complete Workflow
    
    async def execute_enhanced_workflow(self, task_id: str, modal_contents: List, context: Dict[str, Any], governance_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete enhanced workflow with all capabilities."""
        logger.info(f"🚀 Starting enhanced workflow for task {task_id}")
        
        workflow_result = {
            'task_id': task_id,
            'workflow_start_time': datetime.utcnow().isoformat()
        }
        
        try:
            # 1. Multi-modal processing
            if self.multimodal_engine and modal_contents:
                multimodal_result = await self.process_multimodal_task(task_id, modal_contents, context.get('priority', 'medium'))
                workflow_result['multimodal_processing'] = multimodal_result
            
            # 2. Context intelligence analysis
            if self.context_engine:
                context_result = await self.analyze_context_and_assign(task_id, str(modal_contents), context)
                workflow_result['context_analysis'] = context_result
            
            # 3. Governance assessment
            if self.governance:
                governance_result = await self.assess_governance_compliance(task_id, {
                    'content': str(modal_contents),
                    **governance_requirements
                })
                workflow_result['governance_assessment'] = governance_result
            
            # 4. Advanced optimization
            if self.optimization_engine:
                task_requirements = {
                    'estimated_tokens': len(str(modal_contents)) * 2,
                    'priority': context.get('priority', 'medium'),
                    'context_score': workflow_result.get('context_analysis', {}).get('context_score', 0.5)
                }
                optimization_result = await self.optimize_task_assignment(task_id, task_requirements)
                workflow_result['optimization_result'] = optimization_result
            
            # 5. Final agent assignment based on all analysis
            final_assignment = self._determine_final_assignment(workflow_result)
            workflow_result['final_assignment'] = final_assignment
            
            workflow_result['workflow_end_time'] = datetime.utcnow().isoformat()
            workflow_result['status'] = 'completed'
            
            logger.info(f"✅ Enhanced workflow completed for task {task_id}")
            return workflow_result
            
        except Exception as e:
            logger.error(f"❌ Enhanced workflow failed for task {task_id}: {e}")
            workflow_result['status'] = 'failed'
            workflow_result['error'] = str(e)
            workflow_result['workflow_end_time'] = datetime.utcnow().isoformat()
            return workflow_result
    
    async def process_multimodal_task(self, task_id: str, modal_contents: List, priority: str) -> Dict[str, Any]:
        """Process multi-modal content through the multi-modal engine."""
        if not self.multimodal_engine:
            return {'error': 'Multi-modal engine not enabled'}
        
        try:
            # Process the modal contents
            processing_result = self.multimodal_engine.process_modal_content(modal_contents, priority)
            
            logger.info(f"🎭 Multi-modal processing completed for task {task_id}")
            return processing_result
            
        except Exception as e:
            logger.error(f"❌ Multi-modal processing failed for task {task_id}: {e}")
            return {'error': str(e)}
    
    async def optimize_task_assignment(self, task_id: str, task_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize task assignment using the advanced optimization engine."""
        if not self.optimization_engine:
            return {'error': 'Optimization engine not available'}
        
        try:
            # Use the optimization engine to find the best assignment
            optimization_result = self.optimization_engine.optimize_task_assignment(task_id, task_requirements)
            
            logger.info(f"⚡ Task assignment optimized for task {task_id}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"❌ Task optimization failed for task {task_id}: {e}")
            return {'error': str(e)}
    
    async def analyze_context_and_assign(self, task_id: str, task_content: str, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task context using context intelligence engine."""
        if not self.context_engine:
            return {'error': 'Context intelligence engine not available'}
        
        try:
            # Analyze the task context for intelligent assignment
            context_result = self.context_engine.analyze_task_context(task_content, task_context)
            
            logger.info(f"🧠 Context analysis completed for task {task_id}")
            return context_result
            
        except Exception as e:
            logger.error(f"❌ Context analysis failed for task {task_id}: {e}")
            return {'error': str(e)}
    
    async def assess_governance_compliance(self, task_id: str, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """Assess governance compliance using the governance framework."""
        if not self.governance:
            return {'error': 'Governance framework not available'}
        
        try:
            # Assess task for governance compliance
            governance_result = self.governance.assess_task_risk(task_details)
            
            logger.info(f"🛡️ Governance assessment completed for task {task_id}")
            return governance_result
            
        except Exception as e:
            logger.error(f"❌ Governance assessment failed for task {task_id}: {e}")
            return {'error': str(e)}
    
    async def coordinate_mesh_collaboration(self, task_id: str, collaboration_task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multi-agent collaboration using mesh coordinator."""
        if not self.mesh_coordinator:
            return {'error': 'Mesh coordinator not available'}
        
        try:
            # Coordinate multi-agent collaboration
            coordination_result = self.mesh_coordinator.coordinate_multi_agent_task(task_id, collaboration_task)
            
            logger.info(f"🕸️ Mesh collaboration coordinated for task {task_id}")
            return coordination_result
            
        except Exception as e:
            logger.error(f"❌ Mesh collaboration failed for task {task_id}: {e}")
            return {'error': str(e)}
    
    async def collect_performance_metrics(self, operation_id: str, operation_type: str) -> Dict[str, Any]:
        """Collect comprehensive performance metrics for operations."""
        try:
            import time
            import psutil
            
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            
            # Simulate operation-specific metrics based on type
            operation_metrics = {
                'multimodal': {'base_time': 0.8, 'complexity_factor': 1.5},
                'optimization': {'base_time': 0.3, 'complexity_factor': 1.2},
                'context_analysis': {'base_time': 0.2, 'complexity_factor': 1.0},
                'governance': {'base_time': 0.1, 'complexity_factor': 1.1}
            }
            
            metrics = operation_metrics.get(operation_type, {'base_time': 0.5, 'complexity_factor': 1.0})
            
            performance_result = {
                'operation_id': operation_id,
                'operation_type': operation_type,
                'execution_time': metrics['base_time'] * metrics['complexity_factor'],
                'resource_usage': {
                    'cpu': cpu_percent / 100.0,
                    'memory': memory_info.used // 1024 // 1024  # MB
                },
                'quality_score': 0.88 + (0.12 if operation_type == 'multimodal' else 0.05),
                'cost': 0.1 if operation_type != 'optimization' else 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"📊 Performance metrics collected for {operation_id}")
            return performance_result
            
        except Exception as e:
            logger.error(f"❌ Performance metrics collection failed for {operation_id}: {e}")
            return {'error': str(e)}
    
    def _determine_final_assignment(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Determine final agent assignment based on all workflow analysis."""
        assignment = {
            'assigned_agent': 'codex-1',  # Default
            'assignment_reasoning': 'default_assignment',
            'confidence': 0.7
        }
        
        try:
            # Consider multi-modal recommendations
            if 'multimodal_processing' in workflow_result:
                multimodal = workflow_result['multimodal_processing']
                if 'recommended_agents' in multimodal and multimodal['recommended_agents']:
                    assignment['assigned_agent'] = multimodal['recommended_agents'][0]
                    assignment['assignment_reasoning'] = 'multimodal_recommendation'
                    assignment['confidence'] = multimodal.get('confidence', 0.7)
            
            # Consider optimization recommendations
            if 'optimization_result' in workflow_result:
                optimization = workflow_result['optimization_result']
                if 'recommended_agent' in optimization:
                    # Override if optimization has higher confidence or better cost
                    if optimization.get('performance_score', 0) > assignment['confidence']:
                        assignment['assigned_agent'] = optimization['recommended_agent']
                        assignment['assignment_reasoning'] = 'optimization_recommendation'
                        assignment['confidence'] = optimization.get('performance_score', 0.8)
            
            # Consider context intelligence
            if 'context_analysis' in workflow_result:
                context = workflow_result['context_analysis']
                if context.get('context_score', 0) > 0.9:
                    assignment['assignment_reasoning'] += '_with_high_context_confidence'
                    assignment['confidence'] = min(1.0, assignment['confidence'] + 0.1)
            
            # Check governance approval
            if 'governance_assessment' in workflow_result:
                governance = workflow_result['governance_assessment']
                if not governance.get('approved', True):
                    assignment['assigned_agent'] = 'human_review_required'
                    assignment['assignment_reasoning'] = 'governance_escalation'
                    assignment['confidence'] = 1.0
            
        except Exception as e:
            logger.error(f"❌ Error in final assignment determination: {e}")
        
        return assignment
    
    # Ollama Turbo integration methods
    
    async def execute_ollama_request(self, request: OllamaRequest) -> Dict[str, Any]:
        """Execute Ollama request through the hybrid orchestrator."""
        if not self.enable_ollama_turbo or not self.ollama_orchestrator:
            raise Exception("Ollama Turbo integration not enabled or not properly configured")
        
        try:
            response = await self.ollama_orchestrator.chat_completion(request)
            
            # Track the request in our cost system
            input_tokens = request.context.get('input_tokens', 100) if request.context else 100
            output_tokens = len(response.content) // 4  # Rough token estimate
            
            # Use the correct cost tracker method
            if hasattr(self.cost_tracker, 'track_request'):
                self.cost_tracker.track_request(
                    model=response.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=0.0  # Ollama Turbo is included in subscription
                )
            else:
                logger.info(f"Cost tracking: {response.model} - {input_tokens} input, {output_tokens} output tokens")
            
            return {
                'content': response.content,
                'model': response.model,
                'source': response.source.value,
                'usage': response.usage,
                'response_time': response.response_time,
                'metadata': response.metadata
            }
        
        except Exception as e:
            logger.error(f"❌ Ollama request failed: {e}")
            raise
    
    async def get_best_ollama_model(self, task_type: str, performance_priority: str = "balanced") -> str:
        """Get the best Ollama model for a specific task."""
        if not self.enable_ollama_turbo or not self.ollama_orchestrator:
            # Fall back to default local model
            return "gpt-oss:20b"
        
        try:
            return await self.ollama_orchestrator.get_best_model_for_task(task_type, performance_priority)
        except Exception as e:
            logger.error(f"❌ Error getting best Ollama model: {e}")
            return "gpt-oss:20b"  # Safe fallback
    
    async def get_ollama_analytics(self) -> Dict[str, Any]:
        """Get analytics from Ollama orchestrator."""
        if not self.enable_ollama_turbo or not self.ollama_orchestrator:
            return {"ollama_turbo_enabled": False}
        
        try:
            return await self.ollama_orchestrator.get_orchestrator_analytics()
        except Exception as e:
            logger.error(f"❌ Error getting Ollama analytics: {e}")
            return {"error": str(e)}
    
    def _validate_orchestrator_model(self, model: str) -> str:
        """Validate and potentially adjust orchestrator model selection."""
        if model in ORCHESTRATOR_MODELS:
            model_config = ORCHESTRATOR_MODELS[model]
            
            # Check if Ollama model requires Turbo
            if model_config.get("provider") == "ollama_turbo":
                if not self.enable_ollama_turbo:
                    logger.warning(f"Model {model} requires Ollama Turbo, but it's not enabled. Falling back to gpt-5")
                    return "gpt-5"
                # Note: We can't check ollama_orchestrator here since it's not initialized yet
                # Final validation will happen during actual use
            
            return model
        else:
            logger.warning(f"Unknown orchestrator model: {model}. Falling back to gpt-5")
            return "gpt-5"
    
    async def close(self):
        """Clean shutdown of the orchestrator and all components."""
        logger.info("🔄 Shutting down DistributedOrchestrator")
        
        # Close Ollama Turbo integration
        if self.ollama_orchestrator:
            try:
                await self.ollama_orchestrator.close()
            except Exception as e:
                logger.error(f"❌ Error closing Ollama orchestrator: {e}")
        
        # Additional cleanup can be added here for other components
        logger.info("✅ DistributedOrchestrator shutdown complete")