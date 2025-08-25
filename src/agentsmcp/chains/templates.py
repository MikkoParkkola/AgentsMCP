"""
Pre-built chain templates for common multi-agent patterns in AgentsMCP.

This module provides ready-to-use templates for common chain compositions
like sequential processing, parallel execution, map-reduce, and more complex
workflow patterns.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from ..routing import TaskSpec
from .composer import ChainComposer, Step, ParallelSteps
from .executor import ExecutionResult
from .state import ChainState, ChainContext

__all__ = [
    "ChainTemplates",
    "SequentialTemplate",
    "ParallelTemplate", 
    "MapReduceTemplate",
    "ConditionalTemplate",
    "LoopTemplate",
    "PipelineTemplate",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Template Base Class
# --------------------------------------------------------------------------- #

class ChainTemplate:
    """Base class for chain templates."""
    
    def __init__(self, composer: ChainComposer):
        self.composer = composer
        self.logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Sequential Template
# --------------------------------------------------------------------------- #

class SequentialTemplate(ChainTemplate):
    """Template for sequential step execution."""
    
    def create_analysis_pipeline(
        self,
        input_processor: Optional[Callable] = None,
        analyzer: Optional[Callable] = None, 
        summarizer: Optional[Callable] = None,
        formatter: Optional[Callable] = None,
    ) -> List[Step]:
        """
        Create a standard analysis pipeline with preprocessing, analysis, 
        summarization, and formatting steps.
        """
        steps = []
        
        # Input processing step
        if input_processor:
            steps.append(
                self.composer.create_step("preprocess", "general")
                .execute(input_processor)
                .with_performance_tier(2)
            )
        
        # Analysis step
        if analyzer:
            analyze_step = (
                self.composer.create_step("analyze", "reasoning")
                .execute(analyzer)
                .with_performance_tier(4)
            )
            if steps:
                analyze_step.depends_on_steps(steps[-1].step_id)
            steps.append(analyze_step)
        
        # Summarization step
        if summarizer:
            summary_step = (
                self.composer.create_step("summarize", "general")
                .execute(summarizer)
                .with_performance_tier(3)
            )
            if steps:
                summary_step.depends_on_steps(steps[-1].step_id)
            steps.append(summary_step)
        
        # Formatting step
        if formatter:
            format_step = (
                self.composer.create_step("format", "general")
                .execute(formatter)
                .with_performance_tier(2)
            )
            if steps:
                format_step.depends_on_steps(steps[-1].step_id)
            steps.append(format_step)
        
        return steps
    
    def create_coding_pipeline(
        self,
        requirements_analyzer: Callable,
        code_generator: Callable,
        code_reviewer: Callable,
        test_generator: Optional[Callable] = None,
    ) -> List[Step]:
        """Create a coding workflow pipeline."""
        steps = [
            # Analyze requirements
            self.composer.create_step("analyze_requirements", "reasoning")
            .execute(requirements_analyzer)
            .with_performance_tier(4)
            .with_context_length(100000),
            
            # Generate code
            self.composer.create_step("generate_code", "coding")
            .execute(code_generator)
            .depends_on_steps("analyze_requirements")
            .with_performance_tier(5)
            .with_context_length(100000),
            
            # Review code
            self.composer.create_step("review_code", "coding")
            .execute(code_reviewer)
            .depends_on_steps("generate_code")
            .with_performance_tier(4),
        ]
        
        # Optional test generation
        if test_generator:
            steps.append(
                self.composer.create_step("generate_tests", "coding")
                .execute(test_generator)
                .depends_on_steps("review_code")
                .with_performance_tier(4)
            )
        
        return steps

# --------------------------------------------------------------------------- #
# Parallel Template
# --------------------------------------------------------------------------- #

class ParallelTemplate(ChainTemplate):
    """Template for parallel step execution."""
    
    def create_multi_perspective_analysis(
        self,
        analyzers: Dict[str, Callable],
        synthesizer: Optional[Callable] = None,
    ) -> List[Step]:
        """
        Create parallel analysis from multiple perspectives with synthesis.
        
        Parameters
        ----------
        analyzers : Dict[str, Callable]
            Dictionary of perspective_name -> analyzer_function
        synthesizer : Optional[Callable]
            Function to synthesize results from all perspectives
        """
        parallel_steps = []
        
        # Create parallel analysis steps
        for perspective, analyzer in analyzers.items():
            step = (
                self.composer.create_step(f"analyze_{perspective}", "reasoning")
                .execute(analyzer)
                .with_performance_tier(3)
            )
            parallel_steps.append(step)
        
        steps = []
        
        # Add parallel execution
        if parallel_steps:
            parallel = self.composer.create_parallel(*parallel_steps)
            if synthesizer:
                parallel.with_barrier("synthesize")
                
            steps.extend(parallel.to_chain_steps())
            
            # Add synthesis step
            if synthesizer:
                synthesis_step = (
                    self.composer.create_step("synthesize", "reasoning")
                    .execute(synthesizer)
                    .depends_on_steps("synthesize")  # Depends on barrier
                    .with_performance_tier(4)
                )
                steps.append(synthesis_step.to_chain_step())
        
        return steps
    
    def create_multi_agent_research(
        self,
        research_topics: List[str],
        topic_researchers: Dict[str, Callable],
        report_compiler: Optional[Callable] = None,
    ) -> List[Step]:
        """Create parallel research on multiple topics."""
        parallel_steps = []
        
        for topic in research_topics:
            if topic in topic_researchers:
                step = (
                    self.composer.create_step(f"research_{topic}", "general")
                    .execute(topic_researchers[topic])
                    .with_performance_tier(3)
                    .with_context_length(50000)
                )
                parallel_steps.append(step)
        
        steps = []
        
        if parallel_steps:
            parallel = self.composer.create_parallel(*parallel_steps)
            if report_compiler:
                parallel.with_barrier("compile_research")
                
            steps.extend(parallel.to_chain_steps())
            
            if report_compiler:
                compile_step = (
                    self.composer.create_step("compile_research", "general")
                    .execute(report_compiler)
                    .depends_on_steps("compile_research")
                    .with_performance_tier(3)
                )
                steps.append(compile_step.to_chain_step())
        
        return steps

# --------------------------------------------------------------------------- #
# Map-Reduce Template
# --------------------------------------------------------------------------- #

class MapReduceTemplate(ChainTemplate):
    """Template for map-reduce style processing."""
    
    def create_document_processing(
        self,
        document_splitter: Callable,
        chunk_processor: Callable,
        result_aggregator: Callable,
        chunk_size: int = 1000,
    ) -> List[Step]:
        """
        Create map-reduce pipeline for large document processing.
        
        Parameters
        ----------
        document_splitter : Callable
            Function to split document into chunks
        chunk_processor : Callable
            Function to process each chunk
        result_aggregator : Callable
            Function to aggregate chunk results
        chunk_size : int
            Target size for each chunk
        """
        steps = []
        
        # Map phase: split document
        split_step = (
            self.composer.create_step("split_document", "general")
            .execute(document_splitter)
            .with_performance_tier(2)
        )
        steps.append(split_step.to_chain_step())
        
        # Map phase: process chunks in parallel
        # This would need dynamic step generation based on chunk count
        # For now, we'll create a fixed number of parallel processors
        parallel_steps = []
        for i in range(4):  # Max 4 parallel chunks
            chunk_step = (
                self.composer.create_step(f"process_chunk_{i}", "general")
                .execute(self._create_chunk_processor(chunk_processor, i))
                .depends_on_steps("split_document")
                .with_performance_tier(3)
            )
            parallel_steps.append(chunk_step)
        
        parallel = self.composer.create_parallel(*parallel_steps).with_barrier("aggregate")
        steps.extend(parallel.to_chain_steps())
        
        # Reduce phase: aggregate results
        aggregate_step = (
            self.composer.create_step("aggregate", "general")
            .execute(result_aggregator)
            .depends_on_steps("aggregate")  # Depends on barrier
            .with_performance_tier(3)
        )
        steps.append(aggregate_step.to_chain_step())
        
        return steps
    
    def _create_chunk_processor(self, base_processor: Callable, chunk_index: int) -> Callable:
        """Create a chunk-specific processor function."""
        def process_chunk(state: ChainState, model: Any) -> Any:
            # Get chunks from split step
            split_output = state.get_step_output("split_document")
            if isinstance(split_output, list) and chunk_index < len(split_output):
                chunk = split_output[chunk_index]
                return base_processor(chunk, model)
            return None
        
        return process_chunk

# --------------------------------------------------------------------------- #
# Conditional Template
# --------------------------------------------------------------------------- #

class ConditionalTemplate(ChainTemplate):
    """Template for conditional workflow execution."""
    
    def create_quality_gate(
        self,
        quality_checker: Callable,
        success_handler: Callable,
        failure_handler: Callable,
        quality_threshold: float = 0.8,
    ) -> List[Step]:
        """
        Create a quality gate that routes based on quality metrics.
        
        Parameters
        ----------
        quality_checker : Callable
            Function that returns quality score (0-1)
        success_handler : Callable
            Function to execute if quality is acceptable
        failure_handler : Callable
            Function to execute if quality is poor
        quality_threshold : float
            Minimum quality score to pass
        """
        steps = []
        
        # Quality check step
        check_step = (
            self.composer.create_step("quality_check", "general")
            .execute(quality_checker)
            .with_performance_tier(3)
        )
        steps.append(check_step.to_chain_step())
        
        # Conditional execution based on quality
        def quality_condition(state: ChainState) -> bool:
            try:
                quality_score = state.get_step_output("quality_check")
                return float(quality_score) >= quality_threshold
            except (ValueError, KeyError):
                return False
        
        # Success path
        success_step = (
            self.composer.create_step("handle_success", "general")
            .execute(success_handler)
            .depends_on_steps("quality_check")
            .with_performance_tier(2)
        )
        
        # Failure path
        failure_step = (
            self.composer.create_step("handle_failure", "general")
            .execute(failure_handler)
            .depends_on_steps("quality_check")
            .with_performance_tier(2)
        )
        
        conditional = self.composer.create_conditional(quality_condition)
        conditional.if_true(success_step).if_false(failure_step)
        
        steps.extend(conditional.to_chain_steps())
        
        return steps

# --------------------------------------------------------------------------- #
# Loop Template
# --------------------------------------------------------------------------- #

class LoopTemplate(ChainTemplate):
    """Template for iterative processing."""
    
    def create_iterative_refinement(
        self,
        initial_processor: Callable,
        refinement_processor: Callable,
        quality_evaluator: Callable,
        max_iterations: int = 5,
        quality_threshold: float = 0.9,
    ) -> List[Step]:
        """
        Create iterative refinement loop.
        
        Parameters
        ----------
        initial_processor : Callable
            Function for initial processing
        refinement_processor : Callable
            Function for refinement iterations
        quality_evaluator : Callable
            Function to evaluate quality (returns 0-1 score)
        max_iterations : int
            Maximum number of refinement iterations
        quality_threshold : float
            Quality threshold to stop iteration
        """
        steps = []
        
        # Initial processing
        initial_step = (
            self.composer.create_step("initial_process", "general")
            .execute(initial_processor)
            .with_performance_tier(3)
        )
        steps.append(initial_step.to_chain_step())
        
        # Iterative refinement (simulated with fixed steps)
        previous_step = "initial_process"
        
        for i in range(max_iterations):
            # Evaluate quality
            eval_step = (
                self.composer.create_step(f"evaluate_{i}", "general")
                .execute(self._create_evaluator(quality_evaluator, quality_threshold))
                .depends_on_steps(previous_step)
                .with_performance_tier(2)
            )
            steps.append(eval_step.to_chain_step())
            
            # Refine if needed
            refine_step = (
                self.composer.create_step(f"refine_{i}", "general")
                .execute(self._create_conditional_refiner(refinement_processor, f"evaluate_{i}"))
                .depends_on_steps(f"evaluate_{i}")
                .with_performance_tier(3)
            )
            steps.append(refine_step.to_chain_step())
            
            previous_step = f"refine_{i}"
        
        return steps
    
    def _create_evaluator(self, base_evaluator: Callable, threshold: float) -> Callable:
        """Create evaluator with threshold checking."""
        def evaluate(state: ChainState, model: Any) -> Dict[str, Any]:
            score = base_evaluator(state, model)
            return {
                "quality_score": score,
                "meets_threshold": score >= threshold,
                "should_continue": score < threshold
            }
        
        return evaluate
    
    def _create_conditional_refiner(self, base_refiner: Callable, eval_step_id: str) -> Callable:
        """Create refiner that only executes if quality is below threshold."""
        def refine_if_needed(state: ChainState, model: Any) -> Any:
            try:
                eval_result = state.get_step_output(eval_step_id)
                if eval_result.get("should_continue", False):
                    return base_refiner(state, model)
                else:
                    return "Quality threshold met, no refinement needed"
            except KeyError:
                # If evaluation failed, try refinement anyway
                return base_refiner(state, model)
        
        return refine_if_needed

# --------------------------------------------------------------------------- #
# Pipeline Template
# --------------------------------------------------------------------------- #

class PipelineTemplate(ChainTemplate):
    """Template for data processing pipelines."""
    
    def create_etl_pipeline(
        self,
        extractor: Callable,
        transformer: Callable,
        validator: Callable,
        loader: Callable,
    ) -> List[Step]:
        """Create Extract-Transform-Load pipeline."""
        return [
            # Extract
            self.composer.create_step("extract", "general")
            .execute(extractor)
            .with_performance_tier(2)
            .to_chain_step(),
            
            # Transform
            self.composer.create_step("transform", "general")
            .execute(transformer)
            .depends_on_steps("extract")
            .with_performance_tier(3)
            .to_chain_step(),
            
            # Validate
            self.composer.create_step("validate", "general")
            .execute(validator)
            .depends_on_steps("transform")
            .with_performance_tier(2)
            .to_chain_step(),
            
            # Load
            self.composer.create_step("load", "general")
            .execute(loader)
            .depends_on_steps("validate")
            .with_performance_tier(2)
            .to_chain_step(),
        ]

# --------------------------------------------------------------------------- #
# Template Collection
# --------------------------------------------------------------------------- #

class ChainTemplates:
    """
    Collection of all chain templates for easy access.
    """
    
    def __init__(self, composer: ChainComposer):
        self.composer = composer
        self.sequential = SequentialTemplate(composer)
        self.parallel = ParallelTemplate(composer)
        self.map_reduce = MapReduceTemplate(composer)
        self.conditional = ConditionalTemplate(composer)
        self.loop = LoopTemplate(composer)
        self.pipeline = PipelineTemplate(composer)
        
    async def execute_template(
        self,
        template_steps: List[Step],
        initial_data: Dict[str, Any],
        context: Optional[ChainContext] = None,
    ) -> ExecutionResult:
        """Execute a template with given data."""
        builder = self.composer.create_builder()
        
        for step in template_steps:
            if hasattr(step, 'to_chain_step'):
                builder.add_step(step.to_chain_step())
            else:
                builder.steps.append(step)
        
        if context:
            builder.with_context(context)
            
        return await self.composer.execute(builder, initial_data)

# --------------------------------------------------------------------------- #
# Test and demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    async def demo_templates():
        from ..routing import ModelDB, ModelSelector
        
        # Setup
        model_db = ModelDB()
        model_selector = ModelSelector(model_db)
        composer = ChainComposer(model_selector)
        templates = ChainTemplates(composer)
        
        # Demo functions
        def preprocess(state: ChainState, model: Any) -> str:
            return f"Preprocessed: {state.data.get('input', '')}"
            
        def analyze(state: ChainState, model: Any) -> str:
            return f"Analysis complete using {model.name}"
            
        def summarize(state: ChainState, model: Any) -> str:
            return "Summary: Key insights identified"
        
        def format_output(state: ChainState, model: Any) -> str:
            return "Formatted output ready"
        
        print("=== Chain Templates Demo ===\n")
        
        # Test sequential template
        print("1. Sequential Analysis Pipeline:")
        steps = templates.sequential.create_analysis_pipeline(
            input_processor=preprocess,
            analyzer=analyze,
            summarizer=summarize,
            formatter=format_output,
        )
        
        result = await templates.execute_template(
            steps,
            {"input": "Sample document for analysis"}
        )
        
        print(f"   Result: {result.success}, Time: {result.execution_time:.2f}s")
        print(f"   Steps: {len(result.final_state.step_results)}")
        
        # Test parallel template
        print("\n2. Multi-Perspective Analysis:")
        analyzers = {
            "sentiment": lambda state, model: "Sentiment: Positive",
            "topics": lambda state, model: "Topics: Technology, Innovation", 
            "entities": lambda state, model: "Entities: Company A, Product B",
        }
        synthesizer = lambda state, model: "Synthesis: Comprehensive analysis complete"
        
        parallel_steps = templates.parallel.create_multi_perspective_analysis(
            analyzers, synthesizer
        )
        
        result = await templates.execute_template(
            parallel_steps,
            {"input": "Document for multi-perspective analysis"}
        )
        
        print(f"   Result: {result.success}, Time: {result.execution_time:.2f}s")
        print(f"   Steps: {len(result.final_state.step_results)}")
    
    asyncio.run(demo_templates())