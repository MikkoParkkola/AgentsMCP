"""
Chain composition API and builder patterns for AgentsMCP.

This module provides the main chain composition API with fluent builder patterns
for creating complex multi-agent workflows with ease. It integrates with the
routing system for intelligent agent selection.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from ..routing import ModelSelector, TaskSpec
from .executor import ChainExecutor, ChainStep, ExecutionResult
from .state import ChainState, ChainContext

__all__ = [
    "ChainBuilder",
    "ChainComposer", 
    "Step",
    "ConditionalStep",
    "ParallelSteps",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Builder Components
# --------------------------------------------------------------------------- #

class Step:
    """
    Fluent API for defining a single chain step.
    """
    
    def __init__(self, step_id: str, task_type: str = "general"):
        self.step_id = step_id
        self.task_spec = TaskSpec(task_type=task_type)
        self.execute_fn = None
        self.retry_count = 3
        self.timeout = None
        self.depends_on = []
        
    def with_task_type(self, task_type: str) -> Step:
        """Set the task type for agent selection."""
        self.task_spec = TaskSpec(
            task_type=task_type,
            max_cost_per_1k_tokens=self.task_spec.max_cost_per_1k_tokens,
            min_performance_tier=self.task_spec.min_performance_tier,
            required_context_length=self.task_spec.required_context_length,
            preferences=self.task_spec.preferences,
        )
        return self
    
    def with_budget(self, max_cost_per_1k_tokens: float) -> Step:
        """Set budget constraint for this step."""
        self.task_spec = TaskSpec(
            task_type=self.task_spec.task_type,
            max_cost_per_1k_tokens=max_cost_per_1k_tokens,
            min_performance_tier=self.task_spec.min_performance_tier,
            required_context_length=self.task_spec.required_context_length,
            preferences=self.task_spec.preferences,
        )
        return self
        
    def with_performance_tier(self, min_tier: int) -> Step:
        """Set minimum performance requirement."""
        self.task_spec = TaskSpec(
            task_type=self.task_spec.task_type,
            max_cost_per_1k_tokens=self.task_spec.max_cost_per_1k_tokens,
            min_performance_tier=min_tier,
            required_context_length=self.task_spec.required_context_length,
            preferences=self.task_spec.preferences,
        )
        return self
        
    def with_context_length(self, required_length: int) -> Step:
        """Set context length requirement."""
        self.task_spec = TaskSpec(
            task_type=self.task_spec.task_type,
            max_cost_per_1k_tokens=self.task_spec.max_cost_per_1k_tokens,
            min_performance_tier=self.task_spec.min_performance_tier,
            required_context_length=required_length,
            preferences=self.task_spec.preferences,
        )
        return self
        
    def with_preferences(self, preferences: Dict[str, Any]) -> Step:
        """Set agent selection preferences."""
        self.task_spec = TaskSpec(
            task_type=self.task_spec.task_type,
            max_cost_per_1k_tokens=self.task_spec.max_cost_per_1k_tokens,
            min_performance_tier=self.task_spec.min_performance_tier,
            required_context_length=self.task_spec.required_context_length,
            preferences=preferences,
        )
        return self
        
    def execute(self, fn: Callable[[ChainState, Any], Any]) -> Step:
        """Set the execution function for this step."""
        self.execute_fn = fn
        return self
        
    def with_retries(self, retry_count: int) -> Step:
        """Set retry count for this step."""
        self.retry_count = retry_count
        return self
        
    def with_timeout(self, timeout: float) -> Step:
        """Set timeout for this step."""
        self.timeout = timeout
        return self
        
    def depends_on_steps(self, *step_ids: str) -> Step:
        """Set step dependencies."""
        self.depends_on = list(step_ids)
        return self
        
    def to_chain_step(self) -> ChainStep:
        """Convert to ChainStep for execution."""
        if self.execute_fn is None:
            raise ValueError(f"Step {self.step_id} missing execution function")
            
        return ChainStep(
            step_id=self.step_id,
            task_spec=self.task_spec,
            execute_fn=self.execute_fn,
            retry_count=self.retry_count,
            timeout=self.timeout,
            depends_on=self.depends_on,
        )

class ConditionalStep:
    """
    Represents a conditional step that executes based on chain state.
    """
    
    def __init__(self, condition_fn: Callable[[ChainState], bool]):
        self.condition_fn = condition_fn
        self.true_step = None
        self.false_step = None
        
    def if_true(self, step: Step) -> ConditionalStep:
        """Set step to execute if condition is true."""
        self.true_step = step
        return self
        
    def if_false(self, step: Step) -> ConditionalStep:
        """Set step to execute if condition is false."""
        self.false_step = step
        return self
        
    def to_chain_steps(self) -> List[ChainStep]:
        """Convert to list of ChainSteps with conditional logic."""
        steps = []
        
        # Create wrapper functions that check condition
        if self.true_step:
            def conditional_true_fn(state: ChainState, model: Any) -> Any:
                if self.condition_fn(state):
                    return self.true_step.execute_fn(state, model)
                return None
                
            true_step = ChainStep(
                step_id=f"{self.true_step.step_id}_conditional_true",
                task_spec=self.true_step.task_spec,
                execute_fn=conditional_true_fn,
                retry_count=self.true_step.retry_count,
                timeout=self.true_step.timeout,
                depends_on=self.true_step.depends_on,
            )
            steps.append(true_step)
            
        if self.false_step:
            def conditional_false_fn(state: ChainState, model: Any) -> Any:
                if not self.condition_fn(state):
                    return self.false_step.execute_fn(state, model)
                return None
                
            false_step = ChainStep(
                step_id=f"{self.false_step.step_id}_conditional_false",
                task_spec=self.false_step.task_spec,
                execute_fn=conditional_false_fn,
                retry_count=self.false_step.retry_count,
                timeout=self.false_step.timeout,
                depends_on=self.false_step.depends_on,
            )
            steps.append(false_step)
            
        return steps

class ParallelSteps:
    """
    Represents a group of steps that can execute in parallel.
    """
    
    def __init__(self, *steps: Step):
        self.steps = list(steps)
        self.barrier_step_id = None
        
    def with_barrier(self, barrier_step_id: str) -> ParallelSteps:
        """Add a barrier step that waits for all parallel steps to complete."""
        self.barrier_step_id = barrier_step_id
        return self
        
    def to_chain_steps(self) -> List[ChainStep]:
        """Convert to list of ChainSteps for parallel execution."""
        chain_steps = []
        
        # Convert all parallel steps
        for step in self.steps:
            chain_steps.append(step.to_chain_step())
            
        # Add barrier step if specified
        if self.barrier_step_id:
            def barrier_fn(state: ChainState, model: Any) -> str:
                # Collect outputs from all parallel steps
                outputs = []
                for step in self.steps:
                    try:
                        output = state.get_step_output(step.step_id)
                        outputs.append(f"{step.step_id}: {output}")
                    except KeyError:
                        outputs.append(f"{step.step_id}: <no output>")
                return f"Parallel steps completed: {'; '.join(outputs)}"
            
            barrier_step = ChainStep(
                step_id=self.barrier_step_id,
                task_spec=TaskSpec(task_type="general"),
                execute_fn=barrier_fn,
                depends_on=[step.step_id for step in self.steps],
            )
            chain_steps.append(barrier_step)
            
        return chain_steps

# --------------------------------------------------------------------------- #
# Chain Builder
# --------------------------------------------------------------------------- #

class ChainBuilder:
    """
    Fluent API for building complex multi-agent chains.
    """
    
    def __init__(self):
        self.steps = []
        self.context = None
        
    def add_step(self, step: Step) -> ChainBuilder:
        """Add a single step to the chain."""
        self.steps.append(step)
        return self
        
    def add_conditional(self, conditional_step: ConditionalStep) -> ChainBuilder:
        """Add a conditional step to the chain."""
        chain_steps = conditional_step.to_chain_steps()
        self.steps.extend(chain_steps)
        return self
        
    def add_parallel(self, parallel_steps: ParallelSteps) -> ChainBuilder:
        """Add parallel steps to the chain."""
        chain_steps = parallel_steps.to_chain_steps()
        self.steps.extend(chain_steps)
        return self
        
    def with_context(self, context: ChainContext) -> ChainBuilder:
        """Set execution context for the chain."""
        self.context = context
        return self
        
    def with_budget(self, budget: float) -> ChainBuilder:
        """Set budget for the entire chain."""
        if self.context is None:
            self.context = ChainContext()
        self.context.budget = budget
        return self
        
    def with_timeout(self, timeout: float) -> ChainBuilder:
        """Set timeout for the entire chain."""
        if self.context is None:
            self.context = ChainContext()
        self.context.timeout = timeout
        return self
        
    def with_user_id(self, user_id: str) -> ChainBuilder:
        """Set user ID for the chain."""
        if self.context is None:
            self.context = ChainContext()
        self.context.user_id = user_id
        return self
        
    def build(self) -> List[ChainStep]:
        """Build the final list of ChainSteps."""
        chain_steps = []
        
        for step in self.steps:
            if isinstance(step, Step):
                chain_steps.append(step.to_chain_step())
            elif isinstance(step, ChainStep):
                chain_steps.append(step)
                
        return chain_steps

# --------------------------------------------------------------------------- #
# Chain Composer
# --------------------------------------------------------------------------- #

class ChainComposer:
    """
    High-level API for composing and executing multi-agent chains.
    """
    
    def __init__(self, model_selector: ModelSelector):
        self.model_selector = model_selector
        self.executor = ChainExecutor(model_selector)
        self.logger = logging.getLogger(__name__)
        
    def create_builder(self) -> ChainBuilder:
        """Create a new chain builder."""
        return ChainBuilder()
        
    def create_step(self, step_id: str, task_type: str = "general") -> Step:
        """Create a new step with fluent API."""
        return Step(step_id, task_type)
        
    def create_conditional(self, condition_fn: Callable[[ChainState], bool]) -> ConditionalStep:
        """Create a conditional step."""
        return ConditionalStep(condition_fn)
        
    def create_parallel(self, *steps: Step) -> ParallelSteps:
        """Create parallel steps."""
        return ParallelSteps(*steps)
        
    async def execute(
        self,
        builder: ChainBuilder,
        initial_data: Dict[str, Any],
    ) -> ExecutionResult:
        """
        Execute a chain built with ChainBuilder.
        
        Parameters
        ----------
        builder : ChainBuilder
            Built chain definition.
        initial_data : Dict[str, Any]
            Initial data for chain execution.
            
        Returns
        -------
        ExecutionResult
            Result of chain execution.
        """
        steps = builder.build()
        context = builder.context or ChainContext()
        
        self.logger.info(f"Executing chain {context.chain_id} with {len(steps)} steps")
        
        return await self.executor.execute_chain(
            steps=steps,
            initial_data=initial_data,
            context=context,
        )
        
    # Convenience methods for common patterns
    async def execute_sequential(
        self,
        step_functions: List[Callable[[ChainState, Any], Any]],
        task_types: Optional[List[str]] = None,
        initial_data: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute steps in sequence."""
        task_types = task_types or ["general"] * len(step_functions)
        initial_data = initial_data or {}
        
        builder = self.create_builder()
        
        for i, (fn, task_type) in enumerate(zip(step_functions, task_types)):
            step = self.create_step(f"step_{i+1}", task_type).execute(fn)
            if i > 0:
                step.depends_on_steps(f"step_{i}")
            builder.add_step(step)
            
        return await self.execute(builder, initial_data)
        
    async def execute_parallel(
        self,
        step_functions: List[Callable[[ChainState, Any], Any]],
        task_types: Optional[List[str]] = None,
        initial_data: Optional[Dict[str, Any]] = None,
        barrier_step_id: str = "barrier",
    ) -> ExecutionResult:
        """Execute steps in parallel with barrier."""
        task_types = task_types or ["general"] * len(step_functions)
        initial_data = initial_data or {}
        
        builder = self.create_builder()
        
        parallel_steps = []
        for i, (fn, task_type) in enumerate(zip(step_functions, task_types)):
            step = self.create_step(f"parallel_{i+1}", task_type).execute(fn)
            parallel_steps.append(step)
            
        parallel = self.create_parallel(*parallel_steps).with_barrier(barrier_step_id)
        builder.add_parallel(parallel)
        
        return await self.execute(builder, initial_data)

# --------------------------------------------------------------------------- #
# Test and demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    async def demo_composer():
        from ..routing import ModelDB
        
        # Setup
        model_db = ModelDB()
        model_selector = ModelSelector(model_db)
        composer = ChainComposer(model_selector)
        
        # Define step functions
        def analyze_text(state: ChainState, model: Any) -> str:
            text = state.data.get("input_text", "")
            return f"Analysis of '{text}': sentiment=positive, length={len(text)}"
            
        def generate_summary(state: ChainState, model: Any) -> str:
            analysis = state.get_last_success_output()
            return f"Summary based on {analysis}: Key insights identified"
            
        def format_output(state: ChainState, model: Any) -> str:
            summary = state.get_last_success_output()
            return f"Final report: {summary}"
        
        # Build chain using fluent API
        builder = (composer.create_builder()
                  .add_step(
                      composer.create_step("analyze", "reasoning")
                      .execute(analyze_text)
                      .with_performance_tier(3)
                  )
                  .add_step(
                      composer.create_step("summarize", "general") 
                      .execute(generate_summary)
                      .depends_on_steps("analyze")
                  )
                  .add_step(
                      composer.create_step("format", "general")
                      .execute(format_output)
                      .depends_on_steps("summarize")
                  )
                  .with_budget(10.0)
                  .with_user_id("demo-user"))
        
        # Execute chain
        result = await composer.execute(
            builder, 
            {"input_text": "This is a test document for analysis"}
        )
        
        print("Chain Composition Demo Results:")
        print(f"Success: {result.success}")
        print(f"Execution time: {result.execution_time:.2f}s")
        print(f"Total cost: ${result.final_state.total_cost:.4f}")
        print(f"Steps completed: {len(result.final_state.step_results)}")
        
        for i, step_result in enumerate(result.final_state.step_results, 1):
            print(f"  Step {i} ({step_result.step_id}): {step_result.success}")
            if step_result.output:
                print(f"    Output: {step_result.output}")
    
    asyncio.run(demo_composer())