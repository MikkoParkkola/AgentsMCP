"""
Dynamic Scaling and Cost-Aware Worker Provisioning

Automatically scales worker pool based on:
- Queue depth and task priority
- Cost constraints and budget limits
- Worker performance and health
- Resource availability and utilization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from .message_queue import MessageQueue, TaskPriority
from .worker import AgentWorker, WorkerCapabilities
from ..cost.tracker import CostTracker
from ..cost.budget import BudgetManager

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    OPTIMIZE = "optimize"


@dataclass
class ScalingDecision:
    """Represents a scaling decision with reasoning."""
    
    action: ScalingAction
    target_workers: int
    reasoning: str
    cost_impact: float
    urgency: float
    capabilities_needed: List[str]


@dataclass
class WorkerProvisioningRule:
    """Rule for worker provisioning based on conditions."""
    
    name: str
    condition: str  # Python expression that can be evaluated
    action: ScalingAction
    target_adjustment: int  # How many workers to add/remove
    cost_priority: float  # 0.0 = cost doesn't matter, 1.0 = cost is critical
    cooldown_seconds: int = 60  # Minimum time between rule triggers


class DynamicScalingManager:
    """
    Manages dynamic scaling of the worker pool with cost awareness.
    
    Features:
    - Queue-based auto-scaling
    - Cost-constrained provisioning
    - Performance-based optimization
    - Health-based worker replacement
    - Capability-aware scaling
    """
    
    def __init__(self,
                 message_queue: MessageQueue,
                 cost_tracker: CostTracker,
                 budget_manager: BudgetManager,
                 min_workers: int = 1,
                 max_workers: int = 20,
                 cost_threshold: float = 10.0):
        
        self.message_queue = message_queue
        self.cost_tracker = cost_tracker
        self.budget_manager = budget_manager
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.cost_threshold = cost_threshold
        
        # Worker pool management
        self.active_workers: Dict[str, AgentWorker] = {}
        self.pending_workers: Dict[str, datetime] = {}  # Workers starting up
        self.scaling_history: List[Dict[str, Any]] = []
        
        # Scaling rules and policies
        self.provisioning_rules = self._initialize_default_rules()
        self.last_rule_triggers: Dict[str, datetime] = {}
        
        # Performance tracking
        self.scaling_metrics = {
            "total_scale_events": 0,
            "scale_up_events": 0,
            "scale_down_events": 0,
            "cost_constrained_events": 0,
            "average_queue_depth": 0.0,
            "average_worker_utilization": 0.0
        }
        
        # Configuration
        self.scaling_config = {
            "queue_depth_threshold": 5,      # Scale up if more than 5 tasks queued
            "utilization_threshold": 0.8,   # Scale up if workers >80% utilized
            "idle_time_threshold": 300,     # Scale down if idle >5 minutes
            "cost_priority": 0.6,           # Balance between cost and performance
            "quality_priority": 0.4,
            "scaling_interval": 30,         # Evaluate scaling every 30 seconds
            "emergency_scaling": True       # Allow emergency scaling for urgent tasks
        }
        
        logger.info("⚖️ DynamicScalingManager initialized")
    
    async def start(self) -> None:
        """Start the scaling manager."""
        
        logger.info("🚀 Starting DynamicScalingManager")
        
        # Start with minimum workers
        await self._ensure_minimum_workers()
        
        # Start background scaling loop
        asyncio.create_task(self._scaling_loop())
        asyncio.create_task(self._health_monitoring_loop())
        asyncio.create_task(self._cost_optimization_loop())
        asyncio.create_task(self._performance_monitoring_loop())
    
    async def _scaling_loop(self):
        """Main scaling evaluation and execution loop."""
        
        while True:
            try:
                # Get current system state
                queue_status = await self.message_queue.get_queue_status()
                worker_metrics = await self._get_worker_metrics()
                cost_metrics = self._get_cost_metrics()
                
                # Make scaling decision
                decision = await self._evaluate_scaling_decision(
                    queue_status, worker_metrics, cost_metrics
                )
                
                # Execute scaling decision
                if decision.action != ScalingAction.MAINTAIN:
                    await self._execute_scaling_decision(decision)
                
                # Update metrics
                await self._update_scaling_metrics(queue_status, worker_metrics)
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
            
            await asyncio.sleep(self.scaling_config["scaling_interval"])
    
    async def _evaluate_scaling_decision(self,
                                       queue_status: Dict[str, Any],
                                       worker_metrics: Dict[str, Any], 
                                       cost_metrics: Dict[str, Any]) -> ScalingDecision:
        """Evaluate whether scaling action is needed."""
        
        current_workers = len(self.active_workers)
        total_queued = queue_status["total_queued"]
        worker_utilization = worker_metrics.get("average_utilization", 0.0)
        
        # Check cost constraints
        within_budget = cost_metrics["within_budget"]
        cost_utilization = cost_metrics["cost_utilization"]
        
        # Priority queue analysis
        urgent_tasks = queue_status["queue_sizes"].get("URGENT", 0)
        high_priority_tasks = queue_status["queue_sizes"].get("HIGH", 0)
        
        logger.debug(f"📊 Scaling evaluation: {current_workers} workers, {total_queued} queued, {worker_utilization:.2f} util")
        
        # Rule-based decision making
        
        # 1. Emergency scaling for urgent tasks
        if urgent_tasks > 0 and current_workers < self.max_workers:
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                target_workers=min(self.max_workers, current_workers + urgent_tasks),
                reasoning=f"Emergency scaling for {urgent_tasks} urgent tasks",
                cost_impact=self._estimate_scaling_cost(urgent_tasks),
                urgency=1.0,
                capabilities_needed=["ollama"]
            )
        
        # 2. Cost-constrained scaling
        if not within_budget or cost_utilization > 0.9:
            if current_workers > self.min_workers:
                return ScalingDecision(
                    action=ScalingAction.SCALE_DOWN,
                    target_workers=max(self.min_workers, current_workers - 1),
                    reasoning="Cost budget exceeded, scaling down",
                    cost_impact=-self._estimate_worker_cost(),
                    urgency=0.8,
                    capabilities_needed=[]
                )
            else:
                return ScalingDecision(
                    action=ScalingAction.MAINTAIN,
                    target_workers=current_workers,
                    reasoning="At minimum workers, cannot scale down despite budget",
                    cost_impact=0.0,
                    urgency=0.5,
                    capabilities_needed=[]
                )
        
        # 3. Queue-based scaling
        if total_queued > self.scaling_config["queue_depth_threshold"]:
            if current_workers < self.max_workers:
                # Calculate optimal worker count
                optimal_workers = min(
                    self.max_workers,
                    current_workers + max(1, total_queued // 3)
                )
                
                return ScalingDecision(
                    action=ScalingAction.SCALE_UP,
                    target_workers=optimal_workers,
                    reasoning=f"Queue depth {total_queued} exceeds threshold",
                    cost_impact=self._estimate_scaling_cost(optimal_workers - current_workers),
                    urgency=0.7,
                    capabilities_needed=self._analyze_queue_capabilities_needed(queue_status)
                )
        
        # 4. Utilization-based scaling
        if worker_utilization > self.scaling_config["utilization_threshold"]:
            if current_workers < self.max_workers:
                return ScalingDecision(
                    action=ScalingAction.SCALE_UP,
                    target_workers=current_workers + 1,
                    reasoning=f"High utilization {worker_utilization:.2f}",
                    cost_impact=self._estimate_worker_cost(),
                    urgency=0.6,
                    capabilities_needed=["ollama"]
                )
        
        # 5. Scale down if underutilized
        if (worker_utilization < 0.2 and 
            total_queued == 0 and 
            current_workers > self.min_workers):
            
            # Check if workers have been idle
            idle_workers = await self._get_idle_workers()
            if idle_workers:
                return ScalingDecision(
                    action=ScalingAction.SCALE_DOWN,
                    target_workers=current_workers - 1,
                    reasoning="Low utilization, removing idle worker",
                    cost_impact=-self._estimate_worker_cost(),
                    urgency=0.3,
                    capabilities_needed=[]
                )
        
        # 6. Cost optimization without scaling
        if cost_utilization > 0.7:
            return ScalingDecision(
                action=ScalingAction.OPTIMIZE,
                target_workers=current_workers,
                reasoning="Optimize existing workers for cost efficiency",
                cost_impact=0.0,
                urgency=0.4,
                capabilities_needed=[]
            )
        
        # 7. No action needed
        return ScalingDecision(
            action=ScalingAction.MAINTAIN,
            target_workers=current_workers,
            reasoning="System balanced, no scaling needed",
            cost_impact=0.0,
            urgency=0.1,
            capabilities_needed=[]
        )
    
    async def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """Execute a scaling decision."""
        
        logger.info(f"⚖️ Executing scaling decision: {decision.action.value} -> {decision.target_workers} workers")
        logger.info(f"   Reasoning: {decision.reasoning}")
        
        current_workers = len(self.active_workers)
        
        if decision.action == ScalingAction.SCALE_UP:
            workers_to_add = decision.target_workers - current_workers
            await self._scale_up_workers(workers_to_add, decision.capabilities_needed)
            
        elif decision.action == ScalingAction.SCALE_DOWN:
            workers_to_remove = current_workers - decision.target_workers
            await self._scale_down_workers(workers_to_remove)
            
        elif decision.action == ScalingAction.OPTIMIZE:
            await self._optimize_existing_workers()
        
        # Record scaling event
        self._record_scaling_event(decision)
        
        # Update metrics
        self.scaling_metrics["total_scale_events"] += 1
        if decision.action == ScalingAction.SCALE_UP:
            self.scaling_metrics["scale_up_events"] += 1
        elif decision.action == ScalingAction.SCALE_DOWN:
            self.scaling_metrics["scale_down_events"] += 1
        
        if decision.reasoning.startswith("Cost"):
            self.scaling_metrics["cost_constrained_events"] += 1
    
    async def _scale_up_workers(self, count: int, capabilities: List[str]) -> None:
        """Add new workers to the pool."""
        
        logger.info(f"📈 Scaling up {count} workers with capabilities: {capabilities}")
        
        for i in range(count):
            # Determine worker capabilities based on needs
            if capabilities:
                worker_caps = capabilities
            else:
                worker_caps = WorkerCapabilities.GENERAL
            
            # Create and start new worker
            worker = AgentWorker(
                capabilities=worker_caps,
                max_context_tokens=32000,  # Standard worker context
                max_concurrent_tasks=3
            )
            
            # Add to pending workers
            self.pending_workers[worker.worker_id] = datetime.utcnow()
            
            # Start worker asynchronously
            asyncio.create_task(self._start_new_worker(worker))
    
    async def _start_new_worker(self, worker: AgentWorker) -> None:
        """Start a new worker and add to active pool."""
        
        try:
            await worker.start(self.message_queue)
            
            # Move from pending to active
            if worker.worker_id in self.pending_workers:
                del self.pending_workers[worker.worker_id]
            
            self.active_workers[worker.worker_id] = worker
            
            logger.info(f"✅ Worker {worker.worker_id} started and added to pool")
            
        except Exception as e:
            logger.error(f"Failed to start worker {worker.worker_id}: {e}")
            
            # Remove from pending
            if worker.worker_id in self.pending_workers:
                del self.pending_workers[worker.worker_id]
    
    async def _scale_down_workers(self, count: int) -> None:
        """Remove workers from the pool."""
        
        logger.info(f"📉 Scaling down {count} workers")
        
        # Select workers to remove (prefer idle workers)
        idle_workers = await self._get_idle_workers()
        workers_to_remove = []
        
        # First, remove idle workers
        for worker_id in list(idle_workers.keys())[:count]:
            workers_to_remove.append(worker_id)
        
        # If not enough idle workers, remove least utilized
        if len(workers_to_remove) < count:
            remaining_needed = count - len(workers_to_remove)
            worker_utilizations = []
            
            for worker_id, worker in self.active_workers.items():
                if worker_id not in workers_to_remove:
                    status = worker.get_status()
                    utilization = status["current_tasks"] / worker.max_concurrent_tasks
                    worker_utilizations.append((utilization, worker_id))
            
            # Sort by utilization and take least utilized
            worker_utilizations.sort()
            for _, worker_id in worker_utilizations[:remaining_needed]:
                workers_to_remove.append(worker_id)
        
        # Stop and remove selected workers
        for worker_id in workers_to_remove:
            if worker_id in self.active_workers:
                worker = self.active_workers[worker_id]
                asyncio.create_task(self._stop_and_remove_worker(worker))
    
    async def _stop_and_remove_worker(self, worker: AgentWorker) -> None:
        """Stop and remove a worker from the pool."""
        
        try:
            logger.info(f"⏹️ Stopping worker {worker.worker_id}")
            
            await worker.stop()
            
            # Remove from active pool
            if worker.worker_id in self.active_workers:
                del self.active_workers[worker.worker_id]
            
            logger.info(f"✅ Worker {worker.worker_id} stopped and removed")
            
        except Exception as e:
            logger.error(f"Failed to stop worker {worker.worker_id}: {e}")
    
    async def _optimize_existing_workers(self) -> None:
        """Optimize existing workers for better cost-effectiveness."""
        
        logger.info("🔧 Optimizing existing workers for cost efficiency")
        
        # This could involve:
        # 1. Switching workers to use cheaper models
        # 2. Adjusting worker capabilities
        # 3. Rebalancing task assignments
        # 4. Context window optimization
        
        # For now, just log the optimization
        # In practice, this might update worker configurations
        logger.info("✅ Worker optimization completed")
    
    async def _get_worker_metrics(self) -> Dict[str, Any]:
        """Get aggregated worker metrics."""
        
        if not self.active_workers:
            return {
                "total_workers": 0,
                "average_utilization": 0.0,
                "total_current_tasks": 0,
                "total_completed_tasks": 0,
                "total_failed_tasks": 0
            }
        
        total_current_tasks = 0
        total_completed_tasks = 0
        total_failed_tasks = 0
        total_utilization = 0.0
        
        for worker in self.active_workers.values():
            status = worker.get_status()
            total_current_tasks += status["current_tasks"]
            total_completed_tasks += status["completed_tasks"]
            total_failed_tasks += status["failed_tasks"]
            
            # Calculate utilization
            utilization = status["current_tasks"] / worker.max_concurrent_tasks
            total_utilization += utilization
        
        average_utilization = total_utilization / len(self.active_workers)
        
        return {
            "total_workers": len(self.active_workers),
            "average_utilization": average_utilization,
            "total_current_tasks": total_current_tasks,
            "total_completed_tasks": total_completed_tasks,
            "total_failed_tasks": total_failed_tasks
        }
    
    def _get_cost_metrics(self) -> Dict[str, Any]:
        """Get cost-related metrics."""
        
        total_cost = self.cost_tracker.total_cost
        daily_cost = self.cost_tracker.get_daily_cost()
        within_budget = self.budget_manager.check_budget()
        remaining_budget = self.budget_manager.remaining_budget()
        
        cost_utilization = 0.0
        if self.budget_manager.monthly_limit > 0:
            monthly_cost = self.cost_tracker.get_monthly_cost(
                datetime.utcnow().year, 
                datetime.utcnow().month
            )
            cost_utilization = monthly_cost / self.budget_manager.monthly_limit
        
        return {
            "total_cost": total_cost,
            "daily_cost": daily_cost,
            "within_budget": within_budget,
            "remaining_budget": remaining_budget,
            "cost_utilization": cost_utilization
        }
    
    async def _get_idle_workers(self) -> Dict[str, AgentWorker]:
        """Get workers that are currently idle."""
        
        idle_workers = {}
        
        for worker_id, worker in self.active_workers.items():
            status = worker.get_status()
            if status["current_tasks"] == 0:
                idle_workers[worker_id] = worker
        
        return idle_workers
    
    def _analyze_queue_capabilities_needed(self, queue_status: Dict[str, Any]) -> List[str]:
        """Analyze what capabilities are needed based on queue contents."""
        
        # This is simplified - in practice, you'd analyze actual tasks
        # to determine what capabilities are most needed
        
        queue_sizes = queue_status["queue_sizes"]
        total_queued = sum(queue_sizes.values())
        
        if total_queued > 10:
            return ["ollama", "coding"]  # High demand scenarios need versatile workers
        elif total_queued > 5:
            return ["ollama"]
        else:
            return ["ollama"]
    
    def _estimate_scaling_cost(self, worker_count: int) -> float:
        """Estimate the cost impact of adding workers."""
        
        # Simple estimation based on average worker cost
        average_worker_cost_per_hour = 2.0  # Estimate based on model usage
        return worker_count * average_worker_cost_per_hour
    
    def _estimate_worker_cost(self) -> float:
        """Estimate the cost of a single worker per hour."""
        return 2.0  # Simplified estimate
    
    def _record_scaling_event(self, decision: ScalingDecision) -> None:
        """Record a scaling event for analysis."""
        
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": decision.action.value,
            "target_workers": decision.target_workers,
            "reasoning": decision.reasoning,
            "cost_impact": decision.cost_impact,
            "urgency": decision.urgency,
            "capabilities_needed": decision.capabilities_needed
        }
        
        self.scaling_history.append(event)
        
        # Keep only recent history to manage memory
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-100:]
    
    async def _update_scaling_metrics(self,
                                    queue_status: Dict[str, Any],
                                    worker_metrics: Dict[str, Any]) -> None:
        """Update scaling metrics."""
        
        # Update averages
        current_avg_queue = self.scaling_metrics["average_queue_depth"]
        current_avg_util = self.scaling_metrics["average_worker_utilization"]
        
        new_queue_depth = queue_status["total_queued"]
        new_utilization = worker_metrics["average_utilization"]
        
        # Simple moving average
        alpha = 0.1  # Smoothing factor
        self.scaling_metrics["average_queue_depth"] = (
            alpha * new_queue_depth + (1 - alpha) * current_avg_queue
        )
        self.scaling_metrics["average_worker_utilization"] = (
            alpha * new_utilization + (1 - alpha) * current_avg_util
        )
    
    async def _ensure_minimum_workers(self) -> None:
        """Ensure minimum number of workers are running."""
        
        current_workers = len(self.active_workers) + len(self.pending_workers)
        
        if current_workers < self.min_workers:
            workers_needed = self.min_workers - current_workers
            await self._scale_up_workers(workers_needed, WorkerCapabilities.GENERAL)
    
    def _initialize_default_rules(self) -> List[WorkerProvisioningRule]:
        """Initialize default provisioning rules."""
        
        return [
            WorkerProvisioningRule(
                name="emergency_urgent_tasks",
                condition="urgent_tasks > 0",
                action=ScalingAction.SCALE_UP,
                target_adjustment=2,
                cost_priority=0.2,  # Cost matters less for urgent tasks
                cooldown_seconds=30
            ),
            WorkerProvisioningRule(
                name="high_queue_depth",
                condition="total_queued > 10",
                action=ScalingAction.SCALE_UP,
                target_adjustment=1,
                cost_priority=0.6,
                cooldown_seconds=60
            ),
            WorkerProvisioningRule(
                name="budget_exceeded",
                condition="not within_budget",
                action=ScalingAction.SCALE_DOWN,
                target_adjustment=-1,
                cost_priority=1.0,  # Cost is critical
                cooldown_seconds=30
            ),
            WorkerProvisioningRule(
                name="low_utilization",
                condition="worker_utilization < 0.1 and total_queued == 0",
                action=ScalingAction.SCALE_DOWN,
                target_adjustment=-1,
                cost_priority=0.8,
                cooldown_seconds=300
            )
        ]
    
    # Background monitoring loops
    
    async def _health_monitoring_loop(self):
        """Monitor worker health and replace unhealthy workers."""
        
        while True:
            try:
                unhealthy_workers = []
                
                for worker_id, worker in self.active_workers.items():
                    status = worker.get_status()
                    
                    # Check for high error rate
                    if (status["error_count"] > 10 or 
                        (status["completed_tasks"] > 0 and 
                         status["failed_tasks"] / (status["completed_tasks"] + status["failed_tasks"]) > 0.5)):
                        
                        unhealthy_workers.append(worker_id)
                
                # Replace unhealthy workers
                for worker_id in unhealthy_workers:
                    logger.warning(f"🏥 Replacing unhealthy worker {worker_id}")
                    
                    worker = self.active_workers[worker_id]
                    capabilities = worker.capabilities
                    
                    # Start replacement first
                    await self._scale_up_workers(1, capabilities)
                    
                    # Then remove unhealthy worker
                    asyncio.create_task(self._stop_and_remove_worker(worker))
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
            
            await asyncio.sleep(120)  # Check every 2 minutes
    
    async def _cost_optimization_loop(self):
        """Periodically optimize for cost efficiency."""
        
        while True:
            try:
                cost_metrics = self._get_cost_metrics()
                
                # If cost utilization is high, optimize
                if cost_metrics["cost_utilization"] > 0.8:
                    logger.info("💰 High cost utilization detected, optimizing workers")
                    await self._optimize_existing_workers()
                
            except Exception as e:
                logger.error(f"Cost optimization error: {e}")
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def _performance_monitoring_loop(self):
        """Monitor and log scaling performance."""
        
        while True:
            try:
                queue_status = await self.message_queue.get_queue_status()
                worker_metrics = await self._get_worker_metrics()
                cost_metrics = self._get_cost_metrics()
                
                # Log scaling status
                logger.info(
                    f"📊 Scaling Status: {len(self.active_workers)} workers, "
                    f"{queue_status['total_queued']} queued, "
                    f"{worker_metrics['average_utilization']:.2f} utilization, "
                    f"${cost_metrics['daily_cost']:.4f} daily cost"
                )
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
            
            await asyncio.sleep(180)  # Log every 3 minutes
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling manager status."""
        
        return {
            "active_workers": len(self.active_workers),
            "pending_workers": len(self.pending_workers),
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "scaling_metrics": self.scaling_metrics.copy(),
            "recent_scaling_events": self.scaling_history[-10:] if self.scaling_history else [],
            "worker_pool_status": [
                {
                    "worker_id": worker.worker_id,
                    "capabilities": worker.capabilities,
                    "status": worker.get_status()
                }
                for worker in self.active_workers.values()
            ]
        }