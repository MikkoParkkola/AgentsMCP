"""
Chain execution engine with LangGraph integration for AgentsMCP.

This module provides the core execution engine that orchestrates multi-agent
workflows using LangGraph's state graph capabilities, with proper error handling,
monitoring, and integration with the routing system.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from uuid import uuid4

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    raise ImportError("LangGraph is required for chain execution. Install with: pip install langgraph")

from ..routing import ModelSelector, TaskSpec, MetricsTracker, RequestMetrics
from .state import ChainState, StepResult, ChainContext, StateValidator

__all__ = [
    "ChainStep",
    "ChainExecutor", 
    "ExecutionResult",
    "StepExecutionError",
    "ChainExecutionError",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class StepExecutionError(Exception):
    """Raised when a chain step fails to execute."""
    def __init__(self, step_id: str, message: str, original_error: Optional[Exception] = None):
        self.step_id = step_id
        self.original_error = original_error
        super().__init__(f"Step {step_id} failed: {message}")

class ChainExecutionError(Exception):
    """Raised when chain execution fails."""
    def __init__(self, chain_id: str, message: str, step_errors: Optional[List[StepExecutionError]] = None):
        self.chain_id = chain_id
        self.step_errors = step_errors or []
        super().__init__(f"Chain {chain_id} failed: {message}")

# --------------------------------------------------------------------------- #
# Core Types
# --------------------------------------------------------------------------- #

class ChainStep:
    """
    Represents a single step in a chain execution.
    
    Parameters
    ----------
    step_id : str
        Unique identifier for this step.
    task_spec : TaskSpec
        Specification for the task to be performed.
    execute_fn : Callable
        Function to execute for this step.
    retry_count : int
        Number of retries allowed for this step.
    timeout : Optional[float]
        Timeout for step execution in seconds.
    depends_on : List[str]
        List of step IDs this step depends on.
    """
    
    def __init__(
        self,
        step_id: str,
        task_spec: TaskSpec,
        execute_fn: Callable[[ChainState, Any], Any],
        retry_count: int = 3,
        timeout: Optional[float] = None,
        depends_on: Optional[List[str]] = None,
    ):
        self.step_id = step_id
        self.task_spec = task_spec
        self.execute_fn = execute_fn
        self.retry_count = retry_count
        self.timeout = timeout
        self.depends_on = depends_on or []

class ExecutionResult:
    """Result of chain execution."""
    
    def __init__(
        self,
        chain_id: str,
        success: bool,
        final_state: ChainState,
        execution_time: float,
        error: Optional[str] = None,
    ):
        self.chain_id = chain_id
        self.success = success
        self.final_state = final_state
        self.execution_time = execution_time
        self.error = error

# --------------------------------------------------------------------------- #
# Chain Executor
# --------------------------------------------------------------------------- #

class ChainExecutor:
    """
    Core execution engine for multi-agent chains using LangGraph.
    
    Parameters
    ----------
    model_selector : ModelSelector
        Selector for choosing optimal agents for each step.
    metrics_tracker : Optional[MetricsTracker]
        Tracker for monitoring execution metrics.
    state_validator : Optional[StateValidator]
        Validator for chain state consistency.
    """
    
    def __init__(
        self,
        model_selector: ModelSelector,
        metrics_tracker: Optional[MetricsTracker] = None,
        state_validator: Optional[StateValidator] = None,
    ):
        self.model_selector = model_selector
        self.metrics_tracker = metrics_tracker
        self.state_validator = state_validator or StateValidator()
        self.logger = logging.getLogger(__name__)
        
        # Circuit breaker state
        self._circuit_breaker_failures = {}
        self._circuit_breaker_timeout = 60.0  # Reset after 1 minute
        
    def build_graph(self, steps: List[ChainStep]):
        """
        Build a LangGraph from the provided chain steps.
        
        Parameters
        ----------
        steps : List[ChainStep]
            List of steps to include in the chain.
            
        Returns
        -------
        Compiled LangGraph ready for execution.
        """
        # Create state graph
        graph = StateGraph(ChainState)
        
        # Add step execution nodes
        for step in steps:
            node_fn = self._create_step_executor(step)
            graph.add_node(step.step_id, node_fn)
            
        # Add dependencies/edges and identify entry points
        entry_points = []
        all_dependencies = set()
        for step in steps:
            all_dependencies.update(step.depends_on)
            
        for step in steps:
            if step.depends_on:
                for dependency in step.depends_on:
                    graph.add_edge(dependency, step.step_id)
            else:
                # If no dependencies, this is a starting node
                entry_points.append(step.step_id)
        
        # Set entry points
        if entry_points:
            graph.set_entry_point(entry_points[0])
        
        # Find terminal nodes (no outgoing edges) and connect to END
        terminal_nodes = []
        for step in steps:
            if step.step_id not in all_dependencies:
                terminal_nodes.append(step.step_id)
                
        for terminal in terminal_nodes:
            graph.add_edge(terminal, END)
        
        self.logger.info(f"Built LangGraph with {len(steps)} steps")
        return graph.compile()
    
    def _create_step_executor(self, step: ChainStep) -> Callable[[ChainState], ChainState]:
        """Create a LangGraph node function for a step."""
        
        def execute_step(state: ChainState) -> ChainState:
            """Execute a single step and update state."""
            step_start_time = time.time()
            
            try:
                # Check circuit breaker
                if self._is_circuit_breaker_open(step.step_id):
                    raise StepExecutionError(
                        step.step_id,
                        "Circuit breaker is open - too many recent failures"
                    )
                
                # Validate dependencies
                missing_deps = [dep for dep in step.depends_on 
                               if not any(r.step_id == dep and r.success for r in state.step_results)]
                if missing_deps:
                    raise StepExecutionError(
                        step.step_id,
                        f"Missing successful dependencies: {missing_deps}"
                    )
                
                # Select optimal agent for this step
                selection_result = self.model_selector.select_model(step.task_spec)
                selected_model = selection_result.model
                
                self.logger.info(
                    f"Executing step {step.step_id} with agent {selected_model.name}"
                )
                
                # Execute the step with retries
                output = None
                last_error = None
                
                for attempt in range(step.retry_count + 1):
                    try:
                        if step.timeout:
                            output = asyncio.wait_for(
                                self._execute_with_timeout(step, state, selected_model),
                                timeout=step.timeout
                            )
                        else:
                            output = step.execute_fn(state, selected_model)
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < step.retry_count:
                            wait_time = 2 ** attempt  # Exponential backoff
                            self.logger.warning(
                                f"Step {step.step_id} attempt {attempt + 1} failed: {exc}. "
                                f"Retrying in {wait_time}s..."
                            )
                            time.sleep(wait_time)
                        else:
                            self.logger.error(
                                f"Step {step.step_id} failed after {step.retry_count + 1} attempts"
                            )
                
                execution_time = time.time() - step_start_time
                
                if output is not None:
                    # Step succeeded
                    result = StepResult(
                        step_id=step.step_id,
                        agent_id=selected_model.id,
                        success=True,
                        output=output,
                        execution_time=execution_time,
                        cost=self._estimate_cost(selected_model, state.data.get("input", "")),
                        metadata={
                            "model_name": selected_model.name,
                            "provider": selected_model.provider,
                            "attempts": attempt + 1,
                        }
                    )
                    
                    # Reset circuit breaker on success
                    self._reset_circuit_breaker(step.step_id)
                    
                else:
                    # Step failed
                    result = StepResult(
                        step_id=step.step_id,
                        agent_id=selected_model.id,
                        success=False,
                        output=None,
                        error=str(last_error) if last_error else "Unknown error",
                        execution_time=execution_time,
                        metadata={
                            "model_name": selected_model.name,
                            "provider": selected_model.provider,
                            "attempts": step.retry_count + 1,
                        }
                    )
                    
                    # Update circuit breaker
                    self._record_circuit_breaker_failure(step.step_id)
                
                state.add_step_result(result)
                state.current_step += 1
                
                # Record metrics if tracker available
                if self.metrics_tracker:
                    metrics = RequestMetrics(
                        model=selected_model.name,
                        tokens_prompt=len(str(state.data.get("input", "")).split()),
                        tokens_completion=len(str(output).split()) if output else 0,
                        response_time=execution_time,
                        provider=selected_model.provider,
                        success=result.success,
                        cost=result.cost or 0.0,
                    )
                    self.metrics_tracker.record(metrics)
                
                return state
                
            except Exception as exc:
                self.logger.exception(f"Unexpected error in step {step.step_id}: {exc}")
                execution_time = time.time() - step_start_time
                
                result = StepResult(
                    step_id=step.step_id,
                    agent_id="unknown",
                    success=False,
                    output=None,
                    error=str(exc),
                    execution_time=execution_time,
                )
                
                state.add_step_result(result)
                self._record_circuit_breaker_failure(step.step_id)
                return state
        
        return execute_step
    
    async def execute_chain(
        self,
        steps: List[ChainStep],
        initial_data: Dict[str, Any],
        context: Optional[ChainContext] = None,
    ) -> ExecutionResult:
        """
        Execute a chain of steps using LangGraph.
        
        Parameters
        ----------
        steps : List[ChainStep]
            Steps to execute in the chain.
        initial_data : Dict[str, Any]
            Initial data for the chain.
        context : Optional[ChainContext]
            Execution context with constraints and preferences.
            
        Returns
        -------
        ExecutionResult
            Result of the chain execution.
        """
        start_time = time.time()
        
        # Initialize context if not provided
        if context is None:
            context = ChainContext()
        
        # Create initial state
        initial_state = ChainState(context=context, data=initial_data)
        
        # Validate initial state
        validation_errors = self.state_validator.validate_state(initial_state)
        if validation_errors:
            return ExecutionResult(
                chain_id=context.chain_id,
                success=False,
                final_state=initial_state,
                execution_time=0.0,
                error=f"State validation failed: {'; '.join(validation_errors)}",
            )
        
        try:
            # Build and execute graph
            graph = self.build_graph(steps)
            result = await graph.ainvoke(initial_state)
            
            # LangGraph returns the final state directly if it's our ChainState
            if isinstance(result, ChainState):
                final_state = result
            else:
                # Fallback if LangGraph modified the state format
                final_state = initial_state
                self.logger.warning("LangGraph returned unexpected state format")
            
            execution_time = time.time() - start_time
            success = not final_state.has_errors()
            
            self.logger.info(
                f"Chain {context.chain_id} completed: "
                f"success={success}, time={execution_time:.2f}s, cost=${final_state.total_cost:.2f}"
            )
            
            return ExecutionResult(
                chain_id=context.chain_id,
                success=success,
                final_state=final_state,
                execution_time=execution_time,
                error="; ".join(final_state.errors) if final_state.errors else None,
            )
            
        except Exception as exc:
            execution_time = time.time() - start_time
            error_msg = f"Chain execution failed: {exc}"
            
            self.logger.exception(error_msg)
            
            return ExecutionResult(
                chain_id=context.chain_id,
                success=False,
                final_state=initial_state,
                execution_time=execution_time,
                error=error_msg,
            )
    
    async def _execute_with_timeout(self, step: ChainStep, state: ChainState, model: Any) -> Any:
        """Execute step function with timeout handling."""
        return await asyncio.to_thread(step.execute_fn, state, model)
    
    def _estimate_cost(self, model: Any, input_text: str) -> float:
        """Estimate cost for model execution."""
        if not hasattr(model, 'cost_per_input_token'):
            return 0.0
        
        # Simple token estimation (words * 1.3)
        token_count = len(input_text.split()) * 1.3
        return (token_count / 1000.0) * model.cost_per_input_token
    
    # Circuit breaker implementation
    def _is_circuit_breaker_open(self, step_id: str) -> bool:
        """Check if circuit breaker is open for a step."""
        failure_data = self._circuit_breaker_failures.get(step_id)
        if not failure_data:
            return False
            
        failure_count, last_failure_time = failure_data
        
        # Reset if timeout has passed
        if time.time() - last_failure_time > self._circuit_breaker_timeout:
            self._reset_circuit_breaker(step_id)
            return False
            
        # Open circuit if too many failures
        return failure_count >= 5
    
    def _record_circuit_breaker_failure(self, step_id: str) -> None:
        """Record a failure for circuit breaker logic."""
        current_time = time.time()
        if step_id in self._circuit_breaker_failures:
            count, _ = self._circuit_breaker_failures[step_id]
            self._circuit_breaker_failures[step_id] = (count + 1, current_time)
        else:
            self._circuit_breaker_failures[step_id] = (1, current_time)
    
    def _reset_circuit_breaker(self, step_id: str) -> None:
        """Reset circuit breaker for a step."""
        if step_id in self._circuit_breaker_failures:
            del self._circuit_breaker_failures[step_id]
    
    def _should_continue_after_error(self, state: ChainState) -> str:
        """Determine whether to continue chain execution after an error."""
        # Simple logic - could be made more sophisticated
        if state.is_budget_exceeded():
            return "stop"
        if len(state.errors) > 3:  # Stop after too many errors
            return "stop"
        return "continue"

# --------------------------------------------------------------------------- #
# Test and demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    async def demo_execution():
        from ..routing import ModelDB, TaskSpec
        
        # Setup components
        model_db = ModelDB()
        model_selector = ModelSelector(model_db)
        executor = ChainExecutor(model_selector)
        
        # Define test step function
        def test_step(state: ChainState, model: Any) -> str:
            input_text = state.data.get("input", "")
            return f"Processed '{input_text}' with {model.name}"
        
        # Create test steps
        steps = [
            ChainStep(
                step_id="step1",
                task_spec=TaskSpec(task_type="general"),
                execute_fn=test_step,
            ),
            ChainStep(
                step_id="step2", 
                task_spec=TaskSpec(task_type="coding"),
                execute_fn=test_step,
                depends_on=["step1"],
            ),
        ]
        
        # Execute chain
        result = await executor.execute_chain(
            steps=steps,
            initial_data={"input": "test data"},
        )
        
        print(f"Chain execution result:")
        print(f"  Success: {result.success}")
        print(f"  Time: {result.execution_time:.2f}s")
        print(f"  Steps completed: {len(result.final_state.step_results)}")
        print(f"  Total cost: ${result.final_state.total_cost:.2f}")
        if result.error:
            print(f"  Error: {result.error}")
    
    asyncio.run(demo_execution())