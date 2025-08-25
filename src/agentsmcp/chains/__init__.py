"""
agentsmcp.chains
----------------
Multi-agent chain composition system with LangGraph integration.

This package provides a comprehensive system for composing and executing
complex multi-agent workflows using LangGraph's state graph capabilities,
with intelligent agent routing, cost optimization, and robust error handling.

Key Components:

    * `state`    – State management and validation for chain execution
    * `executor` – Core execution engine with LangGraph integration  
    * `composer` – Fluent API for building complex chains
    * `templates`– Pre-built patterns for common workflows

A consumer can simply:

    from agentsmcp.chains import ChainComposer, ChainTemplates
    from agentsmcp.routing import ModelSelector, ModelDB

    # Setup
    model_db = ModelDB()
    model_selector = ModelSelector(model_db)
    composer = ChainComposer(model_selector)
    templates = ChainTemplates(composer)

    # Build and execute a chain
    builder = (composer.create_builder()
              .add_step(composer.create_step("analyze", "reasoning").execute(analyze_fn))
              .add_step(composer.create_step("summarize", "general").execute(summary_fn)
                       .depends_on_steps("analyze")))
    
    result = await composer.execute(builder, {"input": "data"})

For common patterns, use templates:

    steps = templates.sequential.create_analysis_pipeline(
        input_processor=preprocess_fn,
        analyzer=analyze_fn,
        summarizer=summarize_fn,
        formatter=format_fn,
    )
    result = await templates.execute_template(steps, {"input": "data"})
"""

from .state import (
    ChainState,
    StepResult, 
    ChainContext,
    StateValidator,
    serialize_state,
    deserialize_state,
)

from .executor import (
    ChainExecutor,
    ChainStep,
    ExecutionResult,
    StepExecutionError,
    ChainExecutionError,
)

from .composer import (
    ChainComposer,
    ChainBuilder,
    Step,
    ConditionalStep,
    ParallelSteps,
)

from .templates import (
    ChainTemplates,
    SequentialTemplate,
    ParallelTemplate,
    MapReduceTemplate,
    ConditionalTemplate,
    LoopTemplate,
    PipelineTemplate,
)

__version__ = "1.0.0"

# Public API -------------------------------------------------------------- #
__all__ = [
    # State management
    "ChainState",
    "StepResult",
    "ChainContext", 
    "StateValidator",
    "serialize_state",
    "deserialize_state",
    
    # Execution engine
    "ChainExecutor",
    "ChainStep",
    "ExecutionResult",
    "StepExecutionError", 
    "ChainExecutionError",
    
    # Composition API
    "ChainComposer",
    "ChainBuilder",
    "Step",
    "ConditionalStep",
    "ParallelSteps",
    
    # Templates
    "ChainTemplates",
    "SequentialTemplate",
    "ParallelTemplate",
    "MapReduceTemplate", 
    "ConditionalTemplate",
    "LoopTemplate",
    "PipelineTemplate",
]

# --------------------------------------------------------------------------- #
# Quick Start Examples
# --------------------------------------------------------------------------- #

def create_simple_sequential_chain():
    """
    Example: Create a simple sequential analysis chain.
    
    Returns
    -------
    str
        Code example for creating a sequential chain.
    """
    return '''
    from agentsmcp.chains import ChainComposer
    from agentsmcp.routing import ModelDB, ModelSelector
    
    # Setup
    model_db = ModelDB()
    model_selector = ModelSelector(model_db)
    composer = ChainComposer(model_selector)
    
    # Define step functions
    def analyze(state, model):
        return f"Analysis of {state.data['input']} using {model.name}"
    
    def summarize(state, model): 
        analysis = state.get_last_success_output()
        return f"Summary: {analysis[:50]}..."
    
    # Build chain
    builder = (composer.create_builder()
              .add_step(composer.create_step("analyze", "reasoning")
                       .execute(analyze)
                       .with_performance_tier(4))
              .add_step(composer.create_step("summarize", "general")
                       .execute(summarize)
                       .depends_on_steps("analyze"))
              .with_budget(5.0))
    
    # Execute
    result = await composer.execute(builder, {"input": "sample data"})
    '''

def create_parallel_analysis_chain():
    """
    Example: Create parallel analysis with synthesis.
    
    Returns
    -------
    str
        Code example for creating a parallel chain.
    """
    return '''
    from agentsmcp.chains import ChainTemplates
    
    # Define analyzers
    analyzers = {
        "sentiment": lambda state, model: "Sentiment: Positive",
        "entities": lambda state, model: "Entities: [Company, Product]",
        "topics": lambda state, model: "Topics: [Tech, Innovation]",
    }
    
    def synthesize(state, model):
        # Combine results from all parallel steps
        results = [r.output for r in state.step_results if r.success]
        return f"Synthesis: {'; '.join(results)}"
    
    # Create parallel template
    templates = ChainTemplates(composer)
    steps = templates.parallel.create_multi_perspective_analysis(
        analyzers, synthesize
    )
    
    result = await templates.execute_template(steps, {"input": "document"})
    '''

def create_conditional_chain():
    """
    Example: Create chain with conditional execution.
    
    Returns
    -------
    str
        Code example for conditional chain.
    """
    return '''
    # Define quality gate functions
    def check_quality(state, model):
        # Return quality score 0-1
        return 0.85
    
    def handle_success(state, model):
        return "High quality - proceeding with publication"
    
    def handle_failure(state, model):
        return "Low quality - sending for revision"
    
    # Create quality gate template
    steps = templates.conditional.create_quality_gate(
        quality_checker=check_quality,
        success_handler=handle_success, 
        failure_handler=handle_failure,
        quality_threshold=0.8
    )
    
    result = await templates.execute_template(steps, {"input": "content"})
    '''

# --------------------------------------------------------------------------- #
# Integration Examples
# --------------------------------------------------------------------------- #

def integration_with_routing_example():
    """
    Example: Full integration with routing system.
    
    Returns
    -------
    str
        Code example showing routing integration.
    """
    return '''
    from agentsmcp.chains import ChainComposer, ChainTemplates
    from agentsmcp.routing import ModelDB, ModelSelector, MetricsTracker
    
    # Setup with full routing integration
    model_db = ModelDB()
    model_selector = ModelSelector(model_db)
    metrics_tracker = MetricsTracker()
    
    # Create composer with metrics tracking
    composer = ChainComposer(model_selector)
    composer.executor.metrics_tracker = metrics_tracker
    
    # Build chain with specific requirements
    def complex_analysis(state, model):
        # This will automatically use the best model for reasoning tasks
        input_data = state.data.get('input', '')
        return f"Complex analysis by {model.name}: {len(input_data)} chars processed"
    
    builder = (composer.create_builder()
              .add_step(
                  composer.create_step("analyze", "reasoning")
                  .execute(complex_analysis)
                  .with_performance_tier(5)        # Require top-tier model
                  .with_budget(10.0)              # Max $10 per 1K tokens
                  .with_context_length(100000)    # Need large context
                  .with_preferences({"preferred_provider": "openai"})
              )
              .with_budget(50.0)                  # Total chain budget
              .with_timeout(300.0))               # 5 minute timeout
    
    # Execute with full monitoring
    result = await composer.execute(builder, {"input": "large document..."})
    
    # Check metrics
    print(f"Total cost: ${result.final_state.total_cost:.2f}")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    # Get detailed metrics from tracker
    recent_metrics = metrics_tracker.get_recent_metrics(hours=1)
    print(f"Recent requests: {len(recent_metrics)}")
    '''

# --------------------------------------------------------------------------- #
# Advanced Pattern Examples  
# --------------------------------------------------------------------------- #

def advanced_patterns_example():
    """
    Example: Advanced chain composition patterns.
    
    Returns
    -------
    str
        Code example showing advanced patterns.
    """
    return '''
    # Map-Reduce pattern for large document processing
    def split_document(state, model):
        doc = state.data['document']
        chunks = [doc[i:i+1000] for i in range(0, len(doc), 1000)]
        return chunks
    
    def process_chunk(chunk, model):
        return f"Processed chunk: {len(chunk)} chars"
    
    def aggregate_results(state, model):
        # Combine all chunk processing results
        chunk_results = [r.output for r in state.step_results 
                        if r.step_id.startswith('process_chunk')]
        return f"Aggregated {len(chunk_results)} chunk results"
    
    # Create map-reduce chain
    map_reduce_steps = templates.map_reduce.create_document_processing(
        document_splitter=split_document,
        chunk_processor=process_chunk,
        result_aggregator=aggregate_results
    )
    
    # Iterative refinement pattern
    def initial_draft(state, model):
        return "Initial draft of the document"
    
    def refine_draft(state, model):
        previous = state.get_last_success_output()
        return f"Refined: {previous}"
    
    def evaluate_quality(state, model):
        # Return quality score 0-1
        return 0.75
    
    refinement_steps = templates.loop.create_iterative_refinement(
        initial_processor=initial_draft,
        refinement_processor=refine_draft,
        quality_evaluator=evaluate_quality,
        max_iterations=3,
        quality_threshold=0.9
    )
    
    # ETL Pipeline pattern
    def extract_data(state, model):
        return {"records": 1000, "source": "database"}
    
    def transform_data(state, model):
        data = state.get_step_output("extract")
        return {"processed_records": data["records"], "format": "json"}
    
    def validate_data(state, model):
        data = state.get_step_output("transform")
        return {"valid": True, "records": data["processed_records"]}
    
    def load_data(state, model):
        validation = state.get_step_output("validate") 
        return f"Loaded {validation['records']} records successfully"
    
    etl_steps = templates.pipeline.create_etl_pipeline(
        extractor=extract_data,
        transformer=transform_data,
        validator=validate_data,
        loader=load_data
    )
    '''