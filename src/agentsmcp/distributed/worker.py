"""
AgentsMCP Worker - Lightweight task execution with 8K-32K context

Specialized for efficient task execution with minimal context overhead:
- Task-focused processing
- MCP tools integration
- Cost tracking and reporting
- Health monitoring and auto-scaling
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import json
import uuid
import os

from .message_queue import MessageQueue, Task, TaskResult, TaskStatus
from ..cost.models import CostRecord

logger = logging.getLogger(__name__)


class WorkerCapabilities:
    """Define worker capabilities and specializations."""
    
    GENERAL = ["general", "basic_tasks"]
    CODING = ["coding", "implementation", "debugging", "refactoring"]
    FRONTEND = ["frontend", "ui", "react", "vue", "css", "html"]
    BACKEND = ["backend", "api", "database", "server", "microservices"]
    DEVOPS = ["devops", "deployment", "docker", "kubernetes", "ci_cd"]
    DATA = ["data", "analysis", "machine_learning", "statistics"]
    TESTING = ["testing", "qa", "unit_tests", "integration_tests"]
    SECURITY = ["security", "vulnerability", "audit", "penetration_testing"]
    CREATIVE = ["creative", "design", "content", "writing", "ui_ux"]
    
    @classmethod
    def get_all_capabilities(cls) -> List[str]:
        """Get all available capabilities."""
        capabilities = []
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and attr_name.isupper():
                attr_value = getattr(cls, attr_name)
                if isinstance(attr_value, list):
                    capabilities.extend(attr_value)
        return list(set(capabilities))


class AgentWorker:
    """
    Lightweight worker for task execution.
    
    Optimized for:
    - Small context window (8K-32K tokens)
    - Fast task switching
    - Efficient resource utilization
    - Cost tracking integration
    """
    
    def __init__(self,
                 worker_id: str = None,
                 capabilities: List[str] = None,
                 max_context_tokens: int = 32000,
                 max_concurrent_tasks: int = 3):
        
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.capabilities = capabilities or WorkerCapabilities.GENERAL
        self.max_context_tokens = max_context_tokens
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Worker state
        self.start_time = datetime.utcnow()
        self.is_running = False
        self.current_tasks: Dict[str, Task] = {}
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        # Performance tracking
        self.task_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "average_execution_time": 0.0,
            "average_quality_score": 0.0,
            "total_tokens_processed": 0,
            "total_cost": 0.0,
            "success_rate": 1.0
        }
        
        # Context management
        self.context_usage = {
            "current_tokens": 0,
            "max_tokens": max_context_tokens,
            "task_contexts": {}
        }
        
        # Health monitoring
        self.last_health_report = datetime.utcnow()
        self.error_count = 0
        self.max_errors_before_restart = 5
        
        logger.info(f"👷 AgentWorker {self.worker_id} initialized with capabilities: {self.capabilities}")
    
    async def start(self, message_queue: MessageQueue) -> None:
        """Start the worker and begin processing tasks."""
        
        if self.is_running:
            logger.warning(f"Worker {self.worker_id} already running")
            return
        
        self.is_running = True
        self.message_queue = message_queue
        
        logger.info(f"🚀 Starting worker {self.worker_id}")
        
        # Start background tasks
        asyncio.create_task(self._task_processing_loop())
        asyncio.create_task(self._health_monitoring_loop())
        asyncio.create_task(self._context_management_loop())
        
        # Register with message queue
        await self._register_with_orchestrator()
    
    async def stop(self) -> None:
        """Gracefully stop the worker."""
        
        logger.info(f"🛑 Stopping worker {self.worker_id}")
        
        self.is_running = False
        
        # Wait for current tasks to complete (with timeout)
        timeout = 30  # 30 seconds
        start_time = datetime.utcnow()
        
        while self.current_tasks and (datetime.utcnow() - start_time).total_seconds() < timeout:
            await asyncio.sleep(1)
        
        # Force complete remaining tasks
        for task_id in list(self.current_tasks.keys()):
            await self._complete_task_with_error(task_id, "Worker shutdown")
        
        logger.info(f"✅ Worker {self.worker_id} stopped")
    
    async def _task_processing_loop(self):
        """Main task processing loop."""
        
        while self.is_running:
            try:
                # Check if we can take more tasks
                if len(self.current_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(1)
                    continue
                
                # Request next task from queue
                task = await self.message_queue.get_next_task(self.worker_id, self.capabilities)
                
                if task:
                    # Process task asynchronously
                    asyncio.create_task(self._process_task(task))
                else:
                    # No tasks available, wait a bit
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logger.error(f"Task processing loop error: {e}")
                self.error_count += 1
                
                if self.error_count >= self.max_errors_before_restart:
                    logger.error(f"Too many errors, restarting worker {self.worker_id}")
                    await self._restart_worker()
                
                await asyncio.sleep(5)
    
    async def _process_task(self, task: Task) -> None:
        """Process a single task."""
        
        start_time = datetime.utcnow()
        self.current_tasks[task.task_id] = task
        
        logger.info(f"🔄 Processing task {task.task_id}: {task.description[:100]}...")
        
        try:
            # Update context tracking
            estimated_tokens = self._estimate_task_tokens(task)
            self.context_usage["current_tokens"] += estimated_tokens
            self.context_usage["task_contexts"][task.task_id] = estimated_tokens
            
            # Execute the task
            result = await self._execute_task_logic(task)
            
            # Calculate metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create task result
            task_result = TaskResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time_seconds=execution_time,
                tokens_used=estimated_tokens,
                cost=self._calculate_task_cost(task, estimated_tokens),
                quality_score=self._assess_quality_score(result),
                worker_capabilities=self.capabilities.copy()
            )
            
            # Submit result
            await self.message_queue.submit_result(task_result)
            
            # Update performance metrics
            await self._update_performance_metrics(task_result)
            
            # Clean up
            await self._cleanup_task(task.task_id)
            
            self.completed_tasks += 1
            logger.info(f"✅ Task {task.task_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Task {task.task_id} failed: {e}")
            
            # Create failure result
            task_result = TaskResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
                tokens_used=self.context_usage["task_contexts"].get(task.task_id, 0),
                worker_capabilities=self.capabilities.copy()
            )
            
            await self.message_queue.submit_result(task_result)
            await self._cleanup_task(task.task_id)
            
            self.failed_tasks += 1
            self.error_count += 1
    
    async def _execute_task_logic(self, task: Task) -> Dict[str, Any]:
        """
        Execute the actual task logic.
        
        This is where the worker would integrate with:
        - MCP tools for various capabilities
        - AI models (local Ollama or API calls)
        - External services and APIs
        - File system operations
        - etc.
        """
        
        # Simulate task execution based on capabilities
        task_type = self._determine_task_type(task)
        
        if task_type == "coding":
            return await self._execute_coding_task(task)
        elif task_type == "analysis":
            return await self._execute_analysis_task(task)
        elif task_type == "creative":
            return await self._execute_creative_task(task)
        elif task_type == "testing":
            return await self._execute_testing_task(task)
        else:
            return await self._execute_general_task(task)
    
    async def _execute_coding_task(self, task: Task) -> Dict[str, Any]:
        """Execute a coding task."""
        
        # Simulate coding task execution
        await asyncio.sleep(2.0)  # Simulate processing time
        
        return {
            "task_type": "coding",
            "implementation": {
                "files_created": ["main.py", "utils.py"],
                "lines_of_code": 150,
                "functions_implemented": 5,
                "tests_included": True
            },
            "quality_metrics": {
                "code_coverage": 85,
                "complexity_score": "low",
                "security_score": "good"
            },
            "execution_summary": "Coding task completed successfully with best practices"
        }
    
    async def _execute_analysis_task(self, task: Task) -> Dict[str, Any]:
        """Execute an analysis task."""
        
        await asyncio.sleep(1.5)
        
        return {
            "task_type": "analysis",
            "analysis_results": {
                "data_points_processed": 1000,
                "insights_found": 7,
                "confidence_level": 0.89,
                "recommendations": ["Optimize performance", "Improve error handling"]
            },
            "visualization": {
                "charts_generated": 3,
                "reports_created": 1
            },
            "execution_summary": "Analysis completed with actionable insights"
        }
    
    async def _execute_creative_task(self, task: Task) -> Dict[str, Any]:
        """Execute a creative task."""
        
        await asyncio.sleep(3.0)
        
        return {
            "task_type": "creative",
            "creative_output": {
                "design_concepts": 3,
                "color_schemes": 2,
                "typography_choices": 1,
                "user_experience_score": 0.92
            },
            "deliverables": {
                "mockups": 5,
                "style_guide": 1,
                "component_library": 1
            },
            "execution_summary": "Creative task completed with modern design principles"
        }
    
    async def _execute_testing_task(self, task: Task) -> Dict[str, Any]:
        """Execute a testing task."""
        
        await asyncio.sleep(2.5)
        
        return {
            "task_type": "testing",
            "test_results": {
                "test_cases_executed": 45,
                "test_cases_passed": 43,
                "test_cases_failed": 2,
                "coverage_percentage": 92
            },
            "quality_assessment": {
                "bugs_found": 3,
                "performance_issues": 1,
                "security_vulnerabilities": 0
            },
            "execution_summary": "Testing completed with high coverage and few issues"
        }
    
    async def _execute_general_task(self, task: Task) -> Dict[str, Any]:
        """Execute a general task."""
        
        await asyncio.sleep(1.0)
        
        return {
            "task_type": "general",
            "result": {
                "status": "completed",
                "output": f"General task '{task.description}' executed successfully",
                "processing_time": "1.0s",
                "resource_usage": "minimal"
            },
            "execution_summary": "General task completed successfully"
        }
    
    def _determine_task_type(self, task: Task) -> str:
        """Determine the type of task based on description and capabilities."""
        
        description_lower = task.description.lower()
        
        # Check for coding keywords
        if any(word in description_lower for word in ["code", "implement", "function", "class", "api"]):
            return "coding"
        
        # Check for analysis keywords  
        if any(word in description_lower for word in ["analyze", "data", "report", "metrics"]):
            return "analysis"
        
        # Check for creative keywords
        if any(word in description_lower for word in ["design", "ui", "ux", "creative", "visual"]):
            return "creative"
        
        # Check for testing keywords
        if any(word in description_lower for word in ["test", "qa", "quality", "validation"]):
            return "testing"
        
        return "general"
    
    def _estimate_task_tokens(self, task: Task) -> int:
        """Estimate token usage for a task."""
        
        # Base token estimate
        base_tokens = len(task.description) // 4  # Rough character to token ratio
        context_tokens = len(json.dumps(task.context)) // 4
        
        # Add overhead for different task types
        task_type = self._determine_task_type(task)
        overhead_multipliers = {
            "coding": 3.0,      # Code generation needs more tokens
            "analysis": 2.5,    # Data processing and insights
            "creative": 2.0,    # Design and creative work
            "testing": 1.5,     # Test execution and reporting
            "general": 1.0      # Basic tasks
        }
        
        multiplier = overhead_multipliers.get(task_type, 1.0)
        estimated_tokens = int((base_tokens + context_tokens) * multiplier)
        
        # Cap at reasonable limits
        return min(estimated_tokens, self.max_context_tokens // 2)
    
    def _calculate_task_cost(self, task: Task, tokens_used: int) -> float:
        """Calculate the cost for executing a task."""
        
        # Cost depends on the model/provider used
        # This would be determined by the orchestrator's assignment
        assignment = task.context.get("assignment_reasoning", {})
        model = task.context.get("recommended_model", "gpt-oss:20b")  # Default to free local model
        
        # Cost per token by model
        model_costs = {
            "gpt-4": 0.00003,
            "gpt-3.5-turbo": 0.0000015,
            "claude-3-sonnet": 0.000003,
            "gpt-oss:20b": 0.0,  # Free local model
        }
        
        cost_per_token = model_costs.get(model, 0.0)
        return tokens_used * cost_per_token
    
    def _assess_quality_score(self, result: Dict[str, Any]) -> float:
        """Assess quality score based on task result."""
        
        # Base quality score
        base_score = 0.8
        
        # Adjust based on task type and metrics
        task_type = result.get("task_type", "general")
        
        if task_type == "coding":
            # Factor in code quality metrics
            quality_metrics = result.get("quality_metrics", {})
            coverage = quality_metrics.get("code_coverage", 80) / 100.0
            base_score = min(1.0, base_score + (coverage - 0.8) * 0.5)
            
        elif task_type == "testing":
            # Factor in test success rate
            test_results = result.get("test_results", {})
            total_tests = test_results.get("test_cases_executed", 1)
            passed_tests = test_results.get("test_cases_passed", 1)
            success_rate = passed_tests / total_tests
            base_score = success_rate * 0.9 + 0.1  # Scale to 0.1-1.0
            
        elif task_type == "creative":
            # Factor in creativity metrics
            creative_output = result.get("creative_output", {})
            ux_score = creative_output.get("user_experience_score", 0.8)
            base_score = ux_score
            
        return round(base_score, 2)
    
    async def _update_performance_metrics(self, task_result: TaskResult) -> None:
        """Update worker performance metrics."""
        
        total_tasks = self.completed_tasks + self.failed_tasks
        
        if total_tasks > 0:
            # Update success rate
            self.performance_metrics["success_rate"] = self.completed_tasks / total_tasks
            
            # Update average execution time
            current_avg = self.performance_metrics["average_execution_time"]
            new_time = task_result.execution_time_seconds
            self.performance_metrics["average_execution_time"] = (
                (current_avg * (total_tasks - 1) + new_time) / total_tasks
            )
            
            # Update average quality score
            if task_result.quality_score:
                current_quality_avg = self.performance_metrics["average_quality_score"]
                new_quality = task_result.quality_score
                self.performance_metrics["average_quality_score"] = (
                    (current_quality_avg * (total_tasks - 1) + new_quality) / total_tasks
                )
        
        # Update cumulative metrics
        self.performance_metrics["total_tokens_processed"] += task_result.tokens_used
        self.performance_metrics["total_cost"] += task_result.cost
        
        # Keep task history for analysis (limited to save memory)
        task_summary = {
            "task_id": task_result.task_id,
            "execution_time": task_result.execution_time_seconds,
            "tokens_used": task_result.tokens_used,
            "cost": task_result.cost,
            "quality_score": task_result.quality_score,
            "status": task_result.status.value,
            "timestamp": task_result.completed_at.isoformat()
        }
        
        self.task_history.append(task_summary)
        
        # Keep only last 50 tasks to manage memory
        if len(self.task_history) > 50:
            self.task_history = self.task_history[-50:]
    
    async def _cleanup_task(self, task_id: str) -> None:
        """Clean up task resources."""
        
        # Remove from current tasks
        if task_id in self.current_tasks:
            del self.current_tasks[task_id]
        
        # Clean up context tracking
        if task_id in self.context_usage["task_contexts"]:
            tokens_freed = self.context_usage["task_contexts"][task_id]
            self.context_usage["current_tokens"] -= tokens_freed
            del self.context_usage["task_contexts"][task_id]
    
    async def _complete_task_with_error(self, task_id: str, error_message: str) -> None:
        """Complete a task with an error status."""
        
        task = self.current_tasks.get(task_id)
        if not task:
            return
        
        error_result = TaskResult(
            task_id=task_id,
            worker_id=self.worker_id,
            status=TaskStatus.FAILED,
            error=error_message,
            worker_capabilities=self.capabilities.copy()
        )
        
        await self.message_queue.submit_result(error_result)
        await self._cleanup_task(task_id)
    
    async def _register_with_orchestrator(self) -> None:
        """Register this worker with the orchestrator."""
        
        # The registration happens implicitly when we request tasks
        # from the message queue, which updates the orchestrator about
        # our capabilities and availability
        
        logger.info(f"📡 Worker {self.worker_id} registered with orchestrator")
    
    async def _health_monitoring_loop(self):
        """Monitor worker health and report status."""
        
        while self.is_running:
            try:
                now = datetime.utcnow()
                uptime = now - self.start_time
                
                # Health metrics
                health_metrics = {
                    "worker_id": self.worker_id,
                    "uptime": str(uptime),
                    "current_tasks": len(self.current_tasks),
                    "completed_tasks": self.completed_tasks,
                    "failed_tasks": self.failed_tasks,
                    "error_count": self.error_count,
                    "performance_metrics": self.performance_metrics.copy(),
                    "context_usage": self.context_usage.copy(),
                    "last_health_check": now.isoformat()
                }
                
                # Log health status periodically
                if (now - self.last_health_report).total_seconds() > 300:  # Every 5 minutes
                    logger.info(f"💚 Worker {self.worker_id} health: {self.completed_tasks} completed, {self.failed_tasks} failed")
                    self.last_health_report = now
                
                # Reset error count if we've been stable
                if self.error_count > 0 and (now - self.last_health_report).total_seconds() > 300:
                    self.error_count = max(0, self.error_count - 1)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _context_management_loop(self):
        """Manage context window usage and cleanup."""
        
        while self.is_running:
            try:
                current_usage = self.context_usage["current_tokens"]
                max_usage = self.context_usage["max_tokens"]
                usage_percentage = (current_usage / max_usage) * 100 if max_usage > 0 else 0
                
                # If approaching context limit, clean up old task contexts
                if usage_percentage > 80:
                    await self._cleanup_old_contexts()
                    logger.info(f"🧹 Context cleanup performed for worker {self.worker_id}")
                
                # Log context usage periodically
                if usage_percentage > 50:
                    logger.debug(f"📊 Worker {self.worker_id} context usage: {usage_percentage:.1f}%")
                
            except Exception as e:
                logger.error(f"Context management error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _cleanup_old_contexts(self):
        """Clean up old task contexts to free up context space."""
        
        # This is a simplified cleanup - in practice, you might:
        # 1. Summarize completed task contexts
        # 2. Remove contexts for failed tasks
        # 3. Compress frequently accessed information
        
        # For now, just reset to a clean state
        self.context_usage["current_tokens"] = len(self.current_tasks) * 1000  # Estimate for active tasks
        self.context_usage["task_contexts"] = {
            task_id: 1000 for task_id in self.current_tasks.keys()
        }
    
    async def _restart_worker(self):
        """Restart the worker to recover from errors."""
        
        logger.info(f"🔄 Restarting worker {self.worker_id}")
        
        # Save message queue reference
        message_queue = self.message_queue
        
        # Stop current worker
        await self.stop()
        
        # Reset state
        self.error_count = 0
        self.current_tasks = {}
        self.context_usage["current_tokens"] = 0
        self.context_usage["task_contexts"] = {}
        
        # Restart
        await self.start(message_queue)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current worker status."""
        
        uptime = datetime.utcnow() - self.start_time
        
        return {
            "worker_id": self.worker_id,
            "status": "running" if self.is_running else "stopped",
            "uptime": str(uptime),
            "capabilities": self.capabilities,
            "current_tasks": len(self.current_tasks),
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "error_count": self.error_count,
            "performance_metrics": self.performance_metrics.copy(),
            "context_usage": {
                "current_tokens": self.context_usage["current_tokens"],
                "max_tokens": self.context_usage["max_tokens"],
                "usage_percentage": (self.context_usage["current_tokens"] / self.context_usage["max_tokens"]) * 100 if self.context_usage["max_tokens"] > 0 else 0
            }
        }