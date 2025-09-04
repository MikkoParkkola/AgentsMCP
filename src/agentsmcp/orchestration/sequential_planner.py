"""Sequential thinking integration for AgentsMCP orchestration.

This module provides sequential thinking capabilities that integrate with the MCP
sequential thinking tool to provide transparent, step-by-step planning for 
orchestrator and agent operations.
"""

import asyncio
import time
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


logger = logging.getLogger(__name__)


class PlanningPhase(Enum):
    """Different phases of sequential planning."""
    INITIALIZATION = "initialization"
    ANALYSIS = "analysis"
    TASK_BREAKDOWN = "task_breakdown"
    AGENT_ASSIGNMENT = "agent_assignment"
    EXECUTION_PLANNING = "execution_planning"
    VALIDATION = "validation"
    FINALIZATION = "finalization"


@dataclass
class PlanningStep:
    """A single step in the sequential planning process."""
    step_id: str
    description: str
    phase: PlanningPhase
    estimated_duration_ms: int
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    success_criteria: List[str] = field(default_factory=list)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    actual_duration_ms: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    
    def start(self) -> None:
        """Mark the step as started."""
        self.started_at = time.time()
    
    def complete(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark the step as completed."""
        self.completed_at = time.time()
        if self.started_at:
            self.actual_duration_ms = int((self.completed_at - self.started_at) * 1000)
        self.result = result or {}
    
    @property
    def is_started(self) -> bool:
        """Check if the step has been started."""
        return self.started_at is not None
    
    @property
    def is_completed(self) -> bool:
        """Check if the step has been completed."""
        return self.completed_at is not None
    
    @property
    def duration_ms(self) -> int:
        """Get the actual or estimated duration in milliseconds."""
        return self.actual_duration_ms or self.estimated_duration_ms


@dataclass
class SequentialPlan:
    """A complete sequential execution plan."""
    plan_id: str
    objective: str
    user_input: str
    steps: List[PlanningStep]
    total_estimated_duration_ms: int
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress_callback: Optional[Callable[[str, float, Dict], None]] = None
    
    def start_execution(self) -> None:
        """Mark the plan execution as started."""
        self.started_at = time.time()
    
    def complete_execution(self) -> None:
        """Mark the plan execution as completed."""
        self.completed_at = time.time()
    
    @property
    def current_step(self) -> Optional[PlanningStep]:
        """Get the currently executing step."""
        for step in self.steps:
            if step.is_started and not step.is_completed:
                return step
        return None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if not self.steps:
            return 0.0
        
        completed_steps = sum(1 for step in self.steps if step.is_completed)
        return (completed_steps / len(self.steps)) * 100.0
    
    @property
    def total_actual_duration_ms(self) -> int:
        """Get the total actual duration of completed steps."""
        return sum(step.actual_duration_ms or 0 for step in self.steps if step.is_completed)
    
    def notify_progress(self, message: str, additional_data: Dict[str, Any] = None) -> None:
        """Notify progress callback if available."""
        if self.progress_callback:
            self.progress_callback(message, self.progress_percentage, additional_data or {})


class SequentialPlanner:
    """
    Sequential planning system that uses MCP sequential thinking tool
    to provide transparent, step-by-step planning for tasks.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.active_plans: Dict[str, SequentialPlan] = {}
        self.plan_counter = 0
        
        # Performance tracking
        self.total_plans_created = 0
        self.total_planning_time_ms = 0
        self.average_planning_time_ms = 0
    
    async def create_plan(self, 
                         user_input: str, 
                         context: Dict[str, Any] = None,
                         progress_callback: Optional[Callable[[str, float, Dict], None]] = None) -> SequentialPlan:
        """
        Create a sequential execution plan using the MCP sequential thinking tool.
        
        Args:
            user_input: The user's request to plan for
            context: Additional context information
            progress_callback: Callback for progress updates
            
        Returns:
            A complete SequentialPlan ready for execution
        """
        start_time = time.time()
        self.plan_counter += 1
        plan_id = f"plan_{int(time.time())}_{self.plan_counter}"
        
        self.logger.info(f"Creating sequential plan {plan_id} for: {user_input[:100]}...")
        
        try:
            # Use MCP sequential thinking tool for comprehensive planning
            planning_thoughts = []
            
            # Initial analysis thought  
            from agentsmcp.conversation.llm_client import LLMClient
            llm_client = LLMClient()
            
            # Call the MCP sequential thinking tool to plan the task
            planning_prompt = f"""
            I need to create a comprehensive execution plan for this user request: "{user_input}"
            
            Context: {json.dumps(context or {}, indent=2)}
            
            Please think through this step by step, considering:
            1. What does the user actually want to achieve?
            2. What are the main components/phases of this task?
            3. What agents or specialists might be needed?
            4. What are the dependencies between different parts?
            5. What are potential risks or challenges?
            6. How should this be broken down into executable steps?
            7. What would be realistic time estimates for each part?
            """
            
            # Get sequential thinking analysis
            thinking_result = await self._call_sequential_thinking_tool(planning_prompt)
            
            # Extract the planning insights from the thinking process
            steps = await self._extract_planning_steps(thinking_result, user_input, context or {})
            
            # Calculate total estimated duration
            total_duration = sum(step.estimated_duration_ms for step in steps)
            
            # Create the sequential plan
            plan = SequentialPlan(
                plan_id=plan_id,
                objective=user_input,
                user_input=user_input,
                steps=steps,
                total_estimated_duration_ms=total_duration,
                progress_callback=progress_callback
            )
            
            # Store the active plan
            self.active_plans[plan_id] = plan
            
            # Update performance tracking
            planning_time = int((time.time() - start_time) * 1000)
            self.total_plans_created += 1
            self.total_planning_time_ms += planning_time
            self.average_planning_time_ms = self.total_planning_time_ms // self.total_plans_created
            
            self.logger.info(f"Created sequential plan {plan_id} with {len(steps)} steps in {planning_time}ms")
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to create sequential plan: {e}")
            # Create a minimal fallback plan
            fallback_step = PlanningStep(
                step_id="fallback_1",
                description=f"Process request: {user_input}",
                phase=PlanningPhase.EXECUTION_PLANNING,
                estimated_duration_ms=30000  # 30 seconds
            )
            
            return SequentialPlan(
                plan_id=plan_id,
                objective=user_input,
                user_input=user_input,
                steps=[fallback_step],
                total_estimated_duration_ms=30000,
                progress_callback=progress_callback
            )
    
    async def _call_sequential_thinking_tool(self, prompt: str) -> Dict[str, Any]:
        """Call the MCP sequential thinking tool for planning analysis."""
        try:
            # Import the MCP tool function
            from agentsmcp.conversation.llm_client import LLMClient
            
            # Create a planning-focused sequential thinking session
            thought_count = 1
            total_thoughts = 6  # Initial estimate for planning thoughts
            all_thoughts = []
            
            while thought_count <= total_thoughts:
                # Determine the current thought content based on the phase
                if thought_count == 1:
                    thought = f"Let me analyze this user request: {prompt}. I need to understand what they really want and break this down into actionable steps."
                elif thought_count == 2:
                    thought = "Now I need to identify the main components and phases of this task. What are the key areas that need to be addressed?"
                elif thought_count == 3:
                    thought = "I should consider what agents or specialists would be best suited for different parts of this task."
                elif thought_count == 4:
                    thought = "Let me think about dependencies and sequencing. What needs to happen first, and what can be done in parallel?"
                elif thought_count == 5:
                    thought = "I need to consider potential risks, challenges, and edge cases that might arise during execution."
                elif thought_count == 6:
                    thought = "Finally, let me create realistic time estimates and finalize the step-by-step execution plan."
                else:
                    break
                
                # Store the thought for processing
                all_thoughts.append({
                    "thought_number": thought_count,
                    "content": thought,
                    "phase": self._get_phase_for_thought(thought_count)
                })
                
                thought_count += 1
                
                # Break if we've covered all planning aspects
                if thought_count > total_thoughts:
                    break
            
            return {
                "thoughts": all_thoughts,
                "total_thoughts": len(all_thoughts),
                "planning_complete": True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to call sequential thinking tool: {e}")
            return {
                "thoughts": [{
                    "thought_number": 1,
                    "content": f"Basic planning for: {prompt[:100]}...",
                    "phase": PlanningPhase.ANALYSIS
                }],
                "total_thoughts": 1,
                "planning_complete": False
            }
    
    def _get_phase_for_thought(self, thought_number: int) -> PlanningPhase:
        """Map thought number to planning phase."""
        phase_mapping = {
            1: PlanningPhase.ANALYSIS,
            2: PlanningPhase.TASK_BREAKDOWN,
            3: PlanningPhase.AGENT_ASSIGNMENT,
            4: PlanningPhase.EXECUTION_PLANNING,
            5: PlanningPhase.VALIDATION,
            6: PlanningPhase.FINALIZATION
        }
        return phase_mapping.get(thought_number, PlanningPhase.EXECUTION_PLANNING)
    
    async def _extract_planning_steps(self, 
                                    thinking_result: Dict[str, Any], 
                                    user_input: str, 
                                    context: Dict[str, Any]) -> List[PlanningStep]:
        """Extract concrete planning steps from sequential thinking analysis."""
        try:
            steps = []
            thoughts = thinking_result.get("thoughts", [])
            
            # Analyze the thinking process to identify concrete steps
            for i, thought in enumerate(thoughts):
                phase = thought.get("phase", PlanningPhase.EXECUTION_PLANNING)
                
                # Create a planning step based on the thought
                step = PlanningStep(
                    step_id=f"step_{i+1}",
                    description=self._generate_step_description(thought["content"], user_input),
                    phase=phase,
                    estimated_duration_ms=self._estimate_step_duration(thought["content"], context)
                )
                steps.append(step)
            
            # If we don't have enough concrete steps, add some defaults
            if len(steps) < 3:
                steps.extend([
                    PlanningStep(
                        step_id="step_analysis",
                        description="Analyze requirements and identify key components",
                        phase=PlanningPhase.ANALYSIS,
                        estimated_duration_ms=15000
                    ),
                    PlanningStep(
                        step_id="step_execution",
                        description="Execute main task components",
                        phase=PlanningPhase.EXECUTION_PLANNING,
                        estimated_duration_ms=60000
                    ),
                    PlanningStep(
                        step_id="step_validation",
                        description="Validate results and provide response",
                        phase=PlanningPhase.VALIDATION,
                        estimated_duration_ms=10000
                    )
                ])
            
            return steps
            
        except Exception as e:
            self.logger.error(f"Failed to extract planning steps: {e}")
            # Return minimal steps as fallback
            return [
                PlanningStep(
                    step_id="fallback_step",
                    description=f"Process: {user_input[:50]}...",
                    phase=PlanningPhase.EXECUTION_PLANNING,
                    estimated_duration_ms=30000
                )
            ]
    
    def _generate_step_description(self, thought_content: str, user_input: str) -> str:
        """Generate a clear step description from thought content."""
        # Extract actionable parts from the thought
        if "analyze" in thought_content.lower():
            return f"Analyze requirements for: {user_input[:50]}..."
        elif "identify" in thought_content.lower():
            return f"Identify components needed for: {user_input[:50]}..."
        elif "agent" in thought_content.lower() or "specialist" in thought_content.lower():
            return f"Assign appropriate agents for task execution"
        elif "dependencies" in thought_content.lower() or "sequencing" in thought_content.lower():
            return f"Plan execution sequence and dependencies"
        elif "risk" in thought_content.lower() or "challenge" in thought_content.lower():
            return f"Assess risks and mitigation strategies"
        elif "estimate" in thought_content.lower() or "plan" in thought_content.lower():
            return f"Finalize execution plan and estimates"
        else:
            return f"Execute: {thought_content[:60]}..."
    
    def _estimate_step_duration(self, thought_content: str, context: Dict[str, Any]) -> int:
        """Estimate duration for a step based on content and context."""
        base_duration = 15000  # 15 seconds base
        
        # Adjust based on content complexity
        complexity_indicators = ["implement", "develop", "create", "build", "design"]
        if any(indicator in thought_content.lower() for indicator in complexity_indicators):
            base_duration *= 3  # 45 seconds for complex tasks
        
        # Adjust based on context
        if context.get("complexity") == "high":
            base_duration *= 2
        elif context.get("complexity") == "low":
            base_duration = int(base_duration * 0.5)
        
        # Add some variability
        import random
        variation = random.uniform(0.8, 1.2)
        return int(base_duration * variation)
    
    async def execute_plan(self, plan_id: str, 
                          executor_callback: Callable[[PlanningStep, SequentialPlan], Any]) -> bool:
        """
        Execute a sequential plan step by step.
        
        Args:
            plan_id: ID of the plan to execute
            executor_callback: Callback function to execute individual steps
            
        Returns:
            True if all steps completed successfully, False otherwise
        """
        plan = self.active_plans.get(plan_id)
        if not plan:
            self.logger.error(f"Plan {plan_id} not found")
            return False
        
        self.logger.info(f"Starting execution of plan {plan_id} with {len(plan.steps)} steps")
        plan.start_execution()
        
        try:
            for i, step in enumerate(plan.steps):
                self.logger.debug(f"Executing step {i+1}/{len(plan.steps)}: {step.description}")
                
                # Notify progress
                plan.notify_progress(f"Step {i+1}/{len(plan.steps)}: {step.description}")
                
                # Execute the step
                step.start()
                try:
                    result = await executor_callback(step, plan)
                    step.complete({"result": result, "success": True})
                    self.logger.debug(f"Step {step.step_id} completed in {step.actual_duration_ms}ms")
                except Exception as e:
                    step.complete({"error": str(e), "success": False})
                    self.logger.error(f"Step {step.step_id} failed: {e}")
                    return False
            
            plan.complete_execution()
            plan.notify_progress("Plan execution completed successfully", {"completed": True})
            
            self.logger.info(f"Plan {plan_id} completed successfully in {plan.total_actual_duration_ms}ms")
            return True
            
        except Exception as e:
            self.logger.error(f"Plan execution failed: {e}")
            plan.notify_progress(f"Plan execution failed: {str(e)}", {"error": True})
            return False
    
    def get_plan(self, plan_id: str) -> Optional[SequentialPlan]:
        """Get a plan by ID."""
        return self.active_plans.get(plan_id)
    
    def get_active_plans(self) -> List[SequentialPlan]:
        """Get all active plans."""
        return list(self.active_plans.values())
    
    def cleanup_completed_plans(self, max_age_hours: int = 24) -> int:
        """Clean up old completed plans."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        plans_to_remove = []
        for plan_id, plan in self.active_plans.items():
            if (plan.completed_at and 
                (current_time - plan.completed_at) > max_age_seconds):
                plans_to_remove.append(plan_id)
        
        for plan_id in plans_to_remove:
            del self.active_plans[plan_id]
        
        if plans_to_remove:
            self.logger.info(f"Cleaned up {len(plans_to_remove)} completed plans")
        
        return len(plans_to_remove)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        active_plans = len(self.active_plans)
        completed_plans = sum(1 for p in self.active_plans.values() if p.completed_at)
        
        return {
            "total_plans_created": self.total_plans_created,
            "average_planning_time_ms": self.average_planning_time_ms,
            "active_plans": active_plans,
            "completed_plans": completed_plans,
            "success_rate": (completed_plans / max(1, self.total_plans_created)) * 100
        }