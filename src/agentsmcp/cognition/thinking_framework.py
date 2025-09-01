"""
Core thinking and planning framework that manages deliberative phases before action execution.

This is the main coordination component that orchestrates all thinking phases
and ensures optimal decision-making quality.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from .models import (
    ThinkingPhase, ThinkingStep, ThinkingResult, PlanningState,
    Approach, RankedApproach, EvaluationCriteria
)
from .config import ThinkingConfig, DEFAULT_THINKING_CONFIG
from .approach_evaluator import ApproachEvaluator
from .task_decomposer import TaskDecomposer
from .execution_planner import ExecutionPlanner
from .metacognitive_monitor import MetacognitiveMonitor
from .planning_state_manager import PlanningStateManager

logger = logging.getLogger(__name__)


class ThinkingTimeout(Exception):
    """Raised when thinking process exceeds timeout."""
    pass


class InvalidPhase(Exception):
    """Raised when invalid phase transition is attempted."""
    pass


class NoViableApproach(Exception):
    """Raised when no viable approaches are found."""
    pass


class PhaseTransitionError(Exception):
    """Raised when phase transition fails."""
    pass


@dataclass
class ThinkingContext:
    """Context for thinking process execution."""
    request: str
    original_context: Dict[str, Any]
    config: ThinkingConfig
    callback: Optional[Callable[[ThinkingStep], None]] = None
    user_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.user_preferences is None:
            self.user_preferences = {}


class ThinkingFramework:
    """
    Core thinking and planning framework managing deliberative phases before execution.
    
    This framework implements a structured thinking process that enhances decision-making
    quality by evaluating multiple approaches and creating optimal execution plans.
    """
    
    def __init__(self, config: Optional[ThinkingConfig] = None):
        """Initialize the thinking framework."""
        self.config = config or DEFAULT_THINKING_CONFIG
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize component systems
        self.approach_evaluator = ApproachEvaluator()
        self.task_decomposer = TaskDecomposer()
        self.execution_planner = ExecutionPlanner()
        self.metacognitive_monitor = MetacognitiveMonitor()
        self.state_manager = PlanningStateManager()
        
        # Active thinking processes
        self._active_processes: Dict[str, PlanningState] = {}
        self._process_locks: Dict[str, asyncio.Lock] = {}
        
        # Performance metrics
        self._total_processes = 0
        self._successful_processes = 0
        self._timeout_count = 0
        
        self.logger.info(f"ThinkingFramework initialized with config: {self.config.enabled_phases}")
    
    async def think(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        config_override: Optional[ThinkingConfig] = None,
        progress_callback: Optional[Callable[[ThinkingStep], None]] = None
    ) -> ThinkingResult:
        """
        Execute complete thinking process for a given request.
        
        Args:
            request: The user's request or task to think about
            context: Additional context for thinking
            config_override: Override default configuration
            progress_callback: Callback for progress updates
        
        Returns:
            ThinkingResult with selected approach and execution plan
            
        Raises:
            ThinkingTimeout: If thinking exceeds configured timeout
            NoViableApproach: If no acceptable approaches are found
        """
        start_time = time.time()
        context = context or {}
        thinking_config = config_override or self.config
        
        # Check if thinking is enabled
        if not thinking_config.enabled:
            return await self._create_fallback_result(request, "Thinking disabled")
        
        # Determine if this should use shortcuts for simple requests
        if thinking_config.enable_thinking_shortcuts and not thinking_config.is_complex_task(request):
            return await self._execute_lightweight_thinking(request, context, thinking_config)
        
        # Create thinking context
        thinking_context = ThinkingContext(
            request=request,
            original_context=context,
            config=thinking_config,
            callback=progress_callback
        )
        
        # Initialize thinking result
        result = ThinkingResult(
            original_request=request,
            metadata={"started_at": datetime.now()}
        )
        
        try:
            self._total_processes += 1
            
            # Create and register planning state
            planning_state = PlanningState(
                request_id=result.request_id,
                current_phase=ThinkingPhase.ANALYZE_REQUEST,
                thinking_result=result
            )
            
            self._active_processes[result.request_id] = planning_state
            self._process_locks[result.request_id] = asyncio.Lock()
            
            # Execute thinking phases with timeout
            timeout_seconds = thinking_config.get_timeout_for_request(request) / 1000
            
            try:
                result = await asyncio.wait_for(
                    self._execute_thinking_phases(thinking_context, planning_state),
                    timeout=timeout_seconds
                )
                
                # Mark as successful
                self._successful_processes += 1
                planning_state.is_complete = True
                
            except asyncio.TimeoutError:
                self._timeout_count += 1
                self.logger.warning(f"Thinking timeout after {timeout_seconds}s for request: {request[:100]}")
                raise ThinkingTimeout(f"Thinking process exceeded {timeout_seconds}s timeout")
            
            # Calculate total thinking time
            total_time_ms = int((time.time() - start_time) * 1000)
            result.total_thinking_time_ms = total_time_ms
            result.metadata["completed_at"] = datetime.now()
            
            # Add quality assessment if enabled
            if thinking_config.enable_quality_monitoring:
                result.quality_assessment = await self.metacognitive_monitor.assess_thinking_quality(
                    result.thinking_trace, result.confidence
                )
            
            self.logger.info(
                f"Thinking completed in {total_time_ms}ms for request: {request[:50]}... "
                f"(confidence: {result.confidence:.2f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in thinking process: {e}", exc_info=True)
            # Create error result with partial information
            error_result = ThinkingResult(
                request_id=result.request_id,
                original_request=request,
                thinking_trace=result.thinking_trace,
                confidence=0.0,
                total_thinking_time_ms=int((time.time() - start_time) * 1000),
                metadata={"error": str(e)}
            )
            return error_result
            
        finally:
            # Clean up process tracking
            if result.request_id in self._active_processes:
                del self._active_processes[result.request_id]
            if result.request_id in self._process_locks:
                del self._process_locks[result.request_id]
    
    async def _execute_thinking_phases(
        self,
        context: ThinkingContext,
        planning_state: PlanningState
    ) -> ThinkingResult:
        """Execute the configured thinking phases in sequence."""
        result = planning_state.thinking_result
        
        # Track phase execution
        phase_start_times = {}
        
        for phase in context.config.enabled_phases:
            phase_start = time.time()
            phase_start_times[phase] = phase_start
            
            # Update planning state
            planning_state.update_phase(phase)
            
            try:
                await self._execute_phase(phase, context, result)
                
            except Exception as e:
                self.logger.error(f"Error in {phase.value} phase: {e}")
                # Add error step and continue if possible
                error_step = ThinkingStep(
                    phase=phase,
                    timestamp=datetime.now(),
                    content=f"Phase error: {str(e)}",
                    confidence=0.0,
                    metadata={"error": True}
                )
                result.add_thinking_step(error_step)
                
                # Call progress callback if provided
                if context.callback:
                    context.callback(error_step)
                
                # Decide whether to continue or abort
                if phase in [ThinkingPhase.ANALYZE_REQUEST, ThinkingPhase.EXPLORE_OPTIONS]:
                    # Critical phases - abort on failure
                    raise PhaseTransitionError(f"Critical phase {phase.value} failed: {e}")
            
            # Record phase duration
            phase_duration = int((time.time() - phase_start) * 1000)
            if result.thinking_trace:
                result.thinking_trace[-1].duration_ms = phase_duration
        
        # Ensure we have a viable result
        if not result.selected_approach:
            raise NoViableApproach("No viable approach selected after thinking")
        
        return result
    
    async def _execute_phase(
        self,
        phase: ThinkingPhase,
        context: ThinkingContext,
        result: ThinkingResult
    ):
        """Execute a specific thinking phase."""
        phase_methods = {
            ThinkingPhase.ANALYZE_REQUEST: self._analyze_request,
            ThinkingPhase.EXPLORE_OPTIONS: self._explore_options,
            ThinkingPhase.EVALUATE_APPROACHES: self._evaluate_approaches,
            ThinkingPhase.SELECT_STRATEGY: self._select_strategy,
            ThinkingPhase.DECOMPOSE_TASKS: self._decompose_tasks,
            ThinkingPhase.PLAN_EXECUTION: self._plan_execution,
            ThinkingPhase.REFLECT_ADJUST: self._reflect_and_adjust
        }
        
        if phase not in phase_methods:
            raise InvalidPhase(f"Unknown thinking phase: {phase}")
        
        await phase_methods[phase](context, result)
    
    async def _analyze_request(self, context: ThinkingContext, result: ThinkingResult):
        """Analyze and understand the request."""
        analysis_content = await self._perform_request_analysis(
            context.request, context.original_context
        )
        
        step = ThinkingStep(
            phase=ThinkingPhase.ANALYZE_REQUEST,
            timestamp=datetime.now(),
            content=analysis_content,
            confidence=0.9,
            metadata={
                "request_length": len(context.request),
                "context_keys": list(context.original_context.keys()),
                "complexity_level": "high" if context.config.is_complex_task(context.request) else "low"
            }
        )
        
        result.add_thinking_step(step)
        
        if context.callback:
            context.callback(step)
    
    async def _explore_options(self, context: ThinkingContext, result: ThinkingResult):
        """Generate multiple approaches to the problem."""
        approaches = await self._generate_approaches(
            context.request, context.original_context, context.config
        )
        
        # Store approaches in result metadata
        result.metadata["generated_approaches"] = approaches
        
        step = ThinkingStep(
            phase=ThinkingPhase.EXPLORE_OPTIONS,
            timestamp=datetime.now(),
            content=f"Generated {len(approaches)} potential approaches",
            confidence=0.8,
            metadata={
                "approach_count": len(approaches),
                "approach_summaries": [a.name for a in approaches]
            }
        )
        
        result.add_thinking_step(step)
        
        if context.callback:
            context.callback(step)
    
    async def _evaluate_approaches(self, context: ThinkingContext, result: ThinkingResult):
        """Evaluate and rank the generated approaches."""
        approaches = result.metadata.get("generated_approaches", [])
        
        if not approaches:
            raise NoViableApproach("No approaches to evaluate")
        
        # Create evaluation criteria
        criteria = EvaluationCriteria(criteria=context.config.evaluation_criteria_weights)
        
        # Evaluate approaches
        ranked_approaches = await self.approach_evaluator.evaluate_approaches(
            approaches, criteria
        )
        
        result.metadata["ranked_approaches"] = ranked_approaches
        
        step = ThinkingStep(
            phase=ThinkingPhase.EVALUATE_APPROACHES,
            timestamp=datetime.now(),
            content=f"Evaluated and ranked {len(ranked_approaches)} approaches",
            confidence=0.85,
            metadata={
                "top_approach": ranked_approaches[0].approach.name if ranked_approaches else None,
                "evaluation_scores": [ra.total_score for ra in ranked_approaches]
            }
        )
        
        result.add_thinking_step(step)
        
        if context.callback:
            context.callback(step)
    
    async def _select_strategy(self, context: ThinkingContext, result: ThinkingResult):
        """Select the best strategy based on evaluation."""
        ranked_approaches = result.metadata.get("ranked_approaches", [])
        
        if not ranked_approaches:
            raise NoViableApproach("No ranked approaches available for selection")
        
        # Select top approach (or apply additional selection logic)
        selected_approach = ranked_approaches[0]
        
        # Validate selection meets minimum confidence threshold
        if selected_approach.total_score < 0.5:  # Configurable threshold
            self.logger.warning(f"Selected approach has low confidence: {selected_approach.total_score}")
        
        result.selected_approach = selected_approach
        result.confidence = selected_approach.total_score
        
        step = ThinkingStep(
            phase=ThinkingPhase.SELECT_STRATEGY,
            timestamp=datetime.now(),
            content=f"Selected strategy: {selected_approach.approach.name}",
            confidence=selected_approach.total_score,
            metadata={
                "selected_approach_id": selected_approach.approach.id,
                "selection_rationale": selected_approach.selection_rationale
            }
        )
        
        result.add_thinking_step(step)
        
        if context.callback:
            context.callback(step)
    
    async def _decompose_tasks(self, context: ThinkingContext, result: ThinkingResult):
        """Break down the selected approach into sub-tasks."""
        if not result.selected_approach:
            raise PhaseTransitionError("No selected approach available for decomposition")
        
        # Decompose the approach into tasks
        subtasks, dependency_graph = await self.task_decomposer.decompose_approach(
            result.selected_approach.approach, context.config
        )
        
        result.metadata["subtasks"] = subtasks
        result.metadata["dependency_graph"] = dependency_graph
        
        step = ThinkingStep(
            phase=ThinkingPhase.DECOMPOSE_TASKS,
            timestamp=datetime.now(),
            content=f"Decomposed approach into {len(subtasks)} sub-tasks",
            confidence=0.8,
            metadata={
                "subtask_count": len(subtasks),
                "dependency_count": len(dependency_graph.edges),
                "parallel_groups": len(dependency_graph.get_parallel_groups())
            }
        )
        
        result.add_thinking_step(step)
        
        if context.callback:
            context.callback(step)
    
    async def _plan_execution(self, context: ThinkingContext, result: ThinkingResult):
        """Create an optimized execution plan."""
        subtasks = result.metadata.get("subtasks", [])
        dependency_graph = result.metadata.get("dependency_graph")
        
        if not subtasks or not dependency_graph:
            raise PhaseTransitionError("No subtasks or dependency graph available for planning")
        
        # Create execution schedule
        execution_schedule = await self.execution_planner.create_schedule(
            subtasks, dependency_graph, context.config
        )
        
        result.execution_plan = execution_schedule
        
        step = ThinkingStep(
            phase=ThinkingPhase.PLAN_EXECUTION,
            timestamp=datetime.now(),
            content=f"Created execution plan with {len(execution_schedule.tasks)} tasks",
            confidence=0.85,
            metadata={
                "execution_order_length": len(execution_schedule.execution_order),
                "parallel_groups": len(execution_schedule.parallel_groups),
                "checkpoints": len(execution_schedule.checkpoints),
                "estimated_duration": str(execution_schedule.estimated_total_duration)
            }
        )
        
        result.add_thinking_step(step)
        
        if context.callback:
            context.callback(step)
    
    async def _reflect_and_adjust(self, context: ThinkingContext, result: ThinkingResult):
        """Perform metacognitive reflection and adjustments."""
        # Assess the quality of our thinking process
        quality_assessment = await self.metacognitive_monitor.assess_thinking_quality(
            result.thinking_trace, result.confidence
        )
        
        result.quality_assessment = quality_assessment
        
        # Generate improvement suggestions if quality is low
        if quality_assessment.overall_quality < context.config.adaptation_threshold:
            adjustments = await self.metacognitive_monitor.suggest_strategy_adjustments(
                result.thinking_trace, quality_assessment
            )
            result.metadata["suggested_adjustments"] = adjustments
        
        step = ThinkingStep(
            phase=ThinkingPhase.REFLECT_ADJUST,
            timestamp=datetime.now(),
            content=f"Reflection complete - quality score: {quality_assessment.overall_quality:.2f}",
            confidence=quality_assessment.overall_quality,
            metadata={
                "quality_score": quality_assessment.overall_quality,
                "improvement_areas": quality_assessment.improvement_areas,
                "adjustments_suggested": len(result.metadata.get("suggested_adjustments", []))
            }
        )
        
        result.add_thinking_step(step)
        
        if context.callback:
            context.callback(step)
    
    async def _execute_lightweight_thinking(
        self,
        request: str,
        context: Dict[str, Any],
        config: ThinkingConfig
    ) -> ThinkingResult:
        """Execute a lightweight thinking process for simple requests."""
        start_time = time.time()
        
        result = ThinkingResult(
            original_request=request,
            confidence=0.7,  # Default confidence for lightweight thinking
            metadata={"lightweight": True}
        )
        
        # Simple analysis step
        analysis_step = ThinkingStep(
            phase=ThinkingPhase.ANALYZE_REQUEST,
            timestamp=datetime.now(),
            content=f"Quick analysis: {request[:100]}",
            confidence=0.7
        )
        result.add_thinking_step(analysis_step)
        
        # Generate single approach
        simple_approach = Approach(
            name="Direct execution",
            description=f"Directly execute: {request}",
            steps=[request]
        )
        
        # Create simple ranked approach
        result.selected_approach = RankedApproach(
            approach=simple_approach,
            total_score=0.7,
            individual_scores={},
            rank=1,
            selection_rationale="Lightweight thinking for simple request"
        )
        
        # Record timing
        result.total_thinking_time_ms = int((time.time() - start_time) * 1000)
        
        return result
    
    async def _create_fallback_result(self, request: str, reason: str) -> ThinkingResult:
        """Create a fallback result when thinking is disabled or fails."""
        return ThinkingResult(
            original_request=request,
            confidence=0.5,
            total_thinking_time_ms=0,
            metadata={"fallback": True, "reason": reason}
        )
    
    async def _perform_request_analysis(self, request: str, context: Dict[str, Any]) -> str:
        """Perform detailed analysis of the request."""
        analysis_parts = []
        
        # Basic request analysis
        analysis_parts.append(f"Request analysis: '{request[:100]}{'...' if len(request) > 100 else ''}'")
        analysis_parts.append(f"Length: {len(request)} characters")
        
        # Context analysis
        if context:
            analysis_parts.append(f"Context provided: {', '.join(context.keys())}")
        
        # Complexity assessment
        complexity_indicators = ['implement', 'create', 'build', 'design', 'multiple', 'system']
        found_indicators = [indicator for indicator in complexity_indicators if indicator in request.lower()]
        if found_indicators:
            analysis_parts.append(f"Complexity indicators: {', '.join(found_indicators)}")
        
        # Intent classification
        if any(word in request.lower() for word in ['how', 'what', 'why', 'when', 'where']):
            analysis_parts.append("Intent: Information seeking")
        elif any(word in request.lower() for word in ['create', 'make', 'build', 'implement']):
            analysis_parts.append("Intent: Creation/Implementation")
        elif any(word in request.lower() for word in ['fix', 'debug', 'solve', 'resolve']):
            analysis_parts.append("Intent: Problem solving")
        else:
            analysis_parts.append("Intent: General task")
        
        return "\n".join(analysis_parts)
    
    async def _generate_approaches(
        self,
        request: str,
        context: Dict[str, Any],
        config: ThinkingConfig
    ) -> List[Approach]:
        """Generate multiple approaches to solving the problem."""
        approaches = []
        
        # Generate approaches based on request type and complexity
        if "implement" in request.lower() or "create" in request.lower():
            # Implementation approaches
            approaches.extend([
                Approach(
                    name="Incremental implementation",
                    description="Build incrementally with testing at each step",
                    steps=["Plan architecture", "Implement core", "Add features", "Test & refine"],
                    benefits=["Lower risk", "Early validation", "Easier debugging"],
                    risks=["Slower initial progress"]
                ),
                Approach(
                    name="Rapid prototype",
                    description="Quick prototype then refactor",
                    steps=["Create MVP", "Test concept", "Refactor for production"],
                    benefits=["Fast validation", "Early feedback"],
                    risks=["Technical debt", "Rework required"]
                ),
                Approach(
                    name="Modular approach",
                    description="Break into independent modules",
                    steps=["Define interfaces", "Implement modules", "Integrate", "Validate"],
                    benefits=["Parallel development", "Reusable components"],
                    risks=["Integration complexity"]
                )
            ])
        elif "analyze" in request.lower() or "investigate" in request.lower():
            # Analysis approaches
            approaches.extend([
                Approach(
                    name="Systematic analysis",
                    description="Thorough systematic investigation",
                    steps=["Define scope", "Collect data", "Analyze patterns", "Generate insights"],
                    benefits=["Comprehensive coverage", "High accuracy"],
                    risks=["Time intensive"]
                ),
                Approach(
                    name="Focused deep-dive",
                    description="Deep analysis of key areas",
                    steps=["Identify key areas", "Deep dive analysis", "Cross-validate findings"],
                    benefits=["Detailed insights", "Efficient use of time"],
                    risks=["May miss broader patterns"]
                )
            ])
        else:
            # General approaches
            approaches.extend([
                Approach(
                    name="Direct execution",
                    description="Execute the request directly",
                    steps=["Parse requirements", "Execute task", "Validate results"],
                    benefits=["Fast execution", "Simple approach"],
                    risks=["May miss nuances"]
                ),
                Approach(
                    name="Planned execution",
                    description="Plan thoroughly then execute",
                    steps=["Analyze requirements", "Create detailed plan", "Execute plan", "Monitor progress"],
                    benefits=["Lower risk", "Better outcomes"],
                    risks=["Slower start"]
                )
            ])
        
        # Limit to configured range
        max_approaches = min(len(approaches), config.max_approaches_to_generate)
        min_approaches = min(max_approaches, config.min_approaches_to_generate)
        
        return approaches[:max(min_approaches, max_approaches)]
    
    async def get_active_processes(self) -> List[PlanningState]:
        """Get list of currently active thinking processes."""
        return list(self._active_processes.values())
    
    async def cancel_process(self, request_id: str) -> bool:
        """Cancel an active thinking process."""
        if request_id in self._active_processes:
            state = self._active_processes[request_id]
            state.error = "Cancelled by user"
            del self._active_processes[request_id]
            return True
        return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the thinking framework."""
        success_rate = (
            self._successful_processes / self._total_processes * 100
            if self._total_processes > 0 else 0
        )
        
        return {
            "total_processes": self._total_processes,
            "successful_processes": self._successful_processes,
            "timeout_count": self._timeout_count,
            "success_rate_percent": round(success_rate, 2),
            "active_processes": len(self._active_processes)
        }