"""
Core pipeline engine for AgentsMCP.

This module implements the core pipeline execution engine for the multi-agent CI system.
It orchestrates multiple AI agents (ollama-turbo, codex, claude) across different pipeline stages
with support for parallel/sequential execution, comprehensive error handling, and progress tracking.

Key Components:
- PipelineEngine: Main orchestrator for running complete pipelines
- StageRunner: Executes individual stages with agent coordination  
- AgentCoordinator: Manages agent instantiation and execution
- ExecutionTracker: Tracks progress and status across the pipeline
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Mapping, MutableMapping
from dataclasses import dataclass

from .schema import PipelineSpec, StageSpec, AgentAssignment
from ..config.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


# Exception hierarchy
class PipelineError(Exception):
    """Base exception for pipeline-related errors."""
    pass


class StageError(PipelineError):
    """Raised when a stage fails to execute successfully."""
    pass


class AgentError(PipelineError):
    """Raised when an agent fails to complete its task."""
    pass


@dataclass
class AgentResult:
    """Result from executing a single agent."""
    agent_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0


@dataclass 
class StageResult:
    """Aggregated result from executing a stage."""
    stage_name: str
    success: bool
    agent_results: List[AgentResult] 
    duration: float = 0.0


class ExecutionTracker:
    """
    Tracks pipeline execution state and progress.
    
    Thread-safe tracker that maintains state for stages and agents
    throughout pipeline execution.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._stage_status: Dict[str, str] = {}
        self._agent_results: Dict[str, List[AgentResult]] = {}
        self._start_time = time.time()
    
    async def set_stage_status(self, stage_name: str, status: str) -> None:
        """Set the current status of a stage."""
        async with self._lock:
            self._stage_status[stage_name] = status
            logger.debug(f"Stage {stage_name} status: {status}")
    
    async def get_stage_status(self, stage_name: str) -> Optional[str]:
        """Get the current status of a stage."""
        async with self._lock:
            return self._stage_status.get(stage_name)
    
    async def record_agent_result(self, stage_name: str, result: AgentResult) -> None:
        """Record the result of an agent execution."""
        async with self._lock:
            if stage_name not in self._agent_results:
                self._agent_results[stage_name] = []
            self._agent_results[stage_name].append(result)
    
    async def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of current execution status."""
        async with self._lock:
            return {
                "stages": dict(self._stage_status),
                "agent_results": dict(self._agent_results),
                "elapsed_time": time.time() - self._start_time
            }


class AgentCoordinator:
    """
    Coordinates agent instantiation and execution.
    
    Maps agent types to their implementations and handles the execution
    of agents with proper error handling and timeout management.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._agent_cache = {}
    
    def _get_agent_class(self, agent_type: str):
        """Get the agent class for the given agent type."""
        # Try to import the actual agent classes, fall back to stubs
        try:
            if agent_type == "ollama-turbo":
                # Try importing MCP Ollama tool
                return self._get_ollama_agent()
            elif agent_type == "codex":
                # Try importing MCP Codex tool
                return self._get_codex_agent() 
            elif agent_type == "claude":
                # Try importing MCP Claude tool
                return self._get_claude_agent()
            else:
                raise AgentError(f"Unknown agent type: {agent_type}")
        except ImportError as e:
            logger.warning(f"Could not import agent for {agent_type}: {e}")
            # Return stub agent for testing/development
            return self._get_stub_agent(agent_type)
    
    def _get_ollama_agent(self):
        """Get Ollama agent implementation."""
        class OllamaAgent:
            def __init__(self, model: str, **kwargs):
                self.model = model
                self.config = kwargs
            
            async def run(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
                """Run the agent with the given task and payload."""
                # This would use the MCP Ollama tool in real implementation
                await asyncio.sleep(0.1)  # Simulate work
                return {
                    "agent_type": "ollama-turbo", 
                    "model": self.model,
                    "task": task,
                    "result": f"Ollama completed task: {task}",
                    "payload_processed": payload
                }
        
        return OllamaAgent
    
    def _get_codex_agent(self):
        """Get Codex agent implementation."""
        class CodexAgent:
            def __init__(self, model: str, **kwargs):
                self.model = model
                self.config = kwargs
            
            async def run(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
                """Run the agent with the given task and payload."""
                # This would use the MCP Codex tool in real implementation
                await asyncio.sleep(0.2)  # Simulate work
                return {
                    "agent_type": "codex",
                    "model": self.model, 
                    "task": task,
                    "result": f"Codex completed task: {task}",
                    "payload_processed": payload
                }
        
        return CodexAgent
    
    def _get_claude_agent(self):
        """Get Claude agent implementation."""  
        class ClaudeAgent:
            def __init__(self, model: str, **kwargs):
                self.model = model
                self.config = kwargs
            
            async def run(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
                """Run the agent with the given task and payload."""
                # This would use the MCP Claude tool in real implementation  
                await asyncio.sleep(0.15)  # Simulate work
                return {
                    "agent_type": "claude",
                    "model": self.model,
                    "task": task, 
                    "result": f"Claude completed task: {task}",
                    "payload_processed": payload
                }
        
        return ClaudeAgent
    
    def _get_stub_agent(self, agent_type: str):
        """Get stub agent for testing/development."""
        class StubAgent:
            def __init__(self, model: str, **kwargs):
                self.model = model
                self.agent_type = agent_type
                self.config = kwargs
            
            async def run(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
                """Stub implementation for testing."""
                await asyncio.sleep(0.05)
                return {
                    "agent_type": self.agent_type,
                    "model": self.model,
                    "task": task,
                    "result": f"Stub {self.agent_type} completed: {task}",
                    "payload_processed": payload,
                    "stub": True
                }
        
        return StubAgent
    
    async def run_agent(self, assignment: AgentAssignment, context: MutableMapping[str, Any]) -> Dict[str, Any]:
        """
        Execute an agent based on its assignment.
        
        Args:
            assignment: Agent assignment specification
            context: Mutable context that can be updated by the agent
            
        Returns:
            Dict containing agent output
            
        Raises:
            AgentError: If agent execution fails
        """
        start_time = time.time()
        
        try:
            agent_class = self._get_agent_class(assignment.type)
            agent = agent_class(
                model=assignment.model,
                timeout_seconds=assignment.timeout_seconds,
                **assignment.payload.root
            )
            
            logger.info(f"Running {assignment.type} agent: {assignment.task}")
            
            # Execute the agent with the task and payload
            result = await agent.run(assignment.task, assignment.payload.root)
            
            # Update context with agent results
            if isinstance(result, dict):
                context.update(result)
            
            duration = time.time() - start_time
            logger.info(f"Agent {assignment.type} completed in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Agent {assignment.type} failed after {duration:.2f}s: {e}")
            raise AgentError(f"Agent {assignment.type} failed: {e}") from e


class StageRunner:
    """
    Executes individual pipeline stages.
    
    Handles both parallel and sequential execution of agents within a stage,
    with retry logic and proper error handling.
    """
    
    def __init__(self, coordinator: AgentCoordinator, tracker: ExecutionTracker):
        self.coordinator = coordinator
        self.tracker = tracker
    
    async def _run_agent_with_retry(self, assignment: AgentAssignment, context: MutableMapping[str, Any]) -> AgentResult:
        """Run a single agent with retry logic."""
        max_retries = assignment.retries or 1
        attempt = 0
        
        while attempt <= max_retries:
            start_time = time.time()
            
            try:
                result = await self.coordinator.run_agent(assignment, context)
                duration = time.time() - start_time
                
                return AgentResult(
                    agent_name=f"{assignment.type}:{assignment.task}",
                    success=True,
                    output=result,
                    duration=duration
                )
                
            except AgentError as e:
                duration = time.time() - start_time
                attempt += 1
                
                if attempt > max_retries:
                    return AgentResult(
                        agent_name=f"{assignment.type}:{assignment.task}",
                        success=False,
                        error=str(e),
                        duration=duration
                    )
                
                # Exponential backoff
                wait_time = min(2 ** attempt * 0.5, 10)
                logger.warning(f"Agent {assignment.type} attempt {attempt} failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
        
        # This should never be reached due to the loop logic above
        return AgentResult(
            agent_name=f"{assignment.type}:{assignment.task}",
            success=False,
            error="Maximum retries exceeded"
        )
    
    async def run_stage(self, stage: StageSpec, context: MutableMapping[str, Any]) -> StageResult:
        """
        Execute a complete pipeline stage.
        
        Args:
            stage: Stage specification
            context: Mutable execution context
            
        Returns:
            StageResult with aggregated results from all agents
        """
        stage_start = time.time()
        await self.tracker.set_stage_status(stage.name, "running")
        
        logger.info(f"🚀 Starting stage: {stage.name}")
        
        if stage.parallel:
            # Run all agents in parallel
            logger.debug(f"Running {len(stage.agents)} agents in parallel")
            tasks = [self._run_agent_with_retry(agent, context) for agent in stage.agents]
            agent_results = await asyncio.gather(*tasks)
        else:
            # Run agents sequentially
            logger.debug(f"Running {len(stage.agents)} agents sequentially") 
            agent_results = []
            for agent in stage.agents:
                result = await self._run_agent_with_retry(agent, context)
                agent_results.append(result)
                
                # Record each agent result as it completes
                await self.tracker.record_agent_result(stage.name, result)
        
        # Determine stage success
        stage_success = all(result.success for result in agent_results)
        stage_duration = time.time() - stage_start
        
        # Update stage status
        final_status = "completed" if stage_success else "failed"
        await self.tracker.set_stage_status(stage.name, final_status)
        
        if stage_success:
            logger.info(f"✅ Stage {stage.name} completed successfully in {stage_duration:.2f}s")
        else:
            failed_agents = [r.agent_name for r in agent_results if not r.success]
            logger.error(f"❌ Stage {stage.name} failed in {stage_duration:.2f}s. Failed agents: {failed_agents}")
        
        return StageResult(
            stage_name=stage.name,
            success=stage_success,
            agent_results=agent_results,
            duration=stage_duration
        )


class PipelineEngine:
    """
    Main pipeline orchestration engine.
    
    Coordinates the execution of complete pipelines by orchestrating stages
    and handling failure policies, progress tracking, and final results.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.coordinator = AgentCoordinator(config)
        self.tracker = ExecutionTracker()
    
    async def run_async(self, spec: PipelineSpec) -> Dict[str, Any]:
        """
        Execute a pipeline asynchronously.
        
        Args:
            spec: Pipeline specification to execute
            
        Returns:
            Dict containing final pipeline context and results
            
        Raises:
            PipelineError: If pipeline execution fails unrecoverably
        """
        pipeline_start = time.time()
        logger.info(f"🎯 Starting pipeline: {spec.name} ({spec.version})")
        
        # Apply defaults to the spec
        spec = spec.apply_defaults()
        
        # Initialize execution context
        context: Dict[str, Any] = {
            "pipeline_name": spec.name,
            "pipeline_version": spec.version,
            "start_time": pipeline_start
        }
        
        stage_results: List[StageResult] = []
        
        try:
            for i, stage in enumerate(spec.stages, 1):
                logger.info(f"📋 Stage {i}/{len(spec.stages)}: {stage.name}")
                
                runner = StageRunner(self.coordinator, self.tracker)
                stage_result = await runner.run_stage(stage, context)
                stage_results.append(stage_result)
                
                # Handle stage failure according to policy
                if not stage_result.success:
                    failure_policy = stage.on_failure or spec.defaults.on_failure
                    
                    if failure_policy == "abort":
                        logger.error(f"💥 Pipeline aborted due to stage failure: {stage.name}")
                        break
                    elif failure_policy == "skip":
                        logger.warning(f"⏭️ Skipping failed stage: {stage.name}")
                        continue
                    elif failure_policy == "retry":
                        logger.info(f"🔄 Retrying failed stage: {stage.name}")
                        # Simple retry logic - could be enhanced
                        retry_result = await runner.run_stage(stage, context)
                        stage_results[-1] = retry_result  # Replace the failed result
                        if not retry_result.success:
                            logger.error(f"💥 Stage retry failed: {stage.name}")
                            break
            
            # Calculate final results
            pipeline_duration = time.time() - pipeline_start
            successful_stages = sum(1 for r in stage_results if r.success)
            total_stages = len(stage_results)
            
            pipeline_success = successful_stages == total_stages and total_stages == len(spec.stages)
            
            # Final context update
            context.update({
                "pipeline_success": pipeline_success,
                "duration": pipeline_duration,
                "stages_completed": successful_stages,
                "total_stages": len(spec.stages),
                "stage_results": [
                    {
                        "name": r.stage_name,
                        "success": r.success, 
                        "duration": r.duration,
                        "agent_count": len(r.agent_results)
                    }
                    for r in stage_results
                ]
            })
            
            if pipeline_success:
                logger.info(f"🎉 Pipeline {spec.name} completed successfully in {pipeline_duration:.2f}s")
            else:
                logger.error(f"💥 Pipeline {spec.name} completed with failures in {pipeline_duration:.2f}s")
            
            return context
            
        except Exception as e:
            pipeline_duration = time.time() - pipeline_start
            logger.exception(f"💥 Pipeline {spec.name} crashed after {pipeline_duration:.2f}s")
            raise PipelineError(f"Pipeline execution failed: {e}") from e
    
    def run(self, spec: PipelineSpec) -> Dict[str, Any]:
        """
        Synchronous wrapper for pipeline execution.
        
        Args:
            spec: Pipeline specification to execute
            
        Returns:
            Dict containing final pipeline context and results
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(self.run_async(spec))
        finally:
            loop.close()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current pipeline execution status."""
        return await self.tracker.get_status_summary()


# Export main classes
__all__ = [
    "PipelineEngine",
    "StageRunner", 
    "AgentCoordinator",
    "ExecutionTracker",
    "PipelineError",
    "StageError",
    "AgentError",
    "AgentResult",
    "StageResult"
]