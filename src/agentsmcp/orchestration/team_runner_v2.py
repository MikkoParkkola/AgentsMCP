"""Dynamic team runner v2 with intelligent orchestration.

This module provides the next-generation team runner that integrates all dynamic
orchestration components including task classification, team composition, agile
coaching, and retrospective analysis. It maintains API compatibility with the
original run_team function while providing enhanced intelligence and capabilities.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Iterable, Callable, Union
from pathlib import Path

from ..agent_manager import AgentManager
from ..events import EventBus
from ..models import TaskEnvelopeV1
from ..runtime_config import Config
from .task_classifier import TaskClassifier
from .team_composer import TeamComposer
from .dynamic_orchestrator import DynamicOrchestrator
from .agile_coach import AgileCoachIntegration, AgilePhase
from .retrospective_engine import RetrospectiveEngine, RetrospectiveType, RetrospectiveScope
from .models import (
    TaskType,
    ComplexityLevel,
    RiskLevel,
    ResourceConstraints,
    TeamComposition,
    CoordinationStrategy,
)

logger = logging.getLogger(__name__)

# Fallback team for backward compatibility
DEFAULT_TEAM: List[str] = [
    "business_analyst",
    "backend_engineer", 
    "api_engineer",
    "web_frontend_engineer",
    "tui_frontend_engineer",
    "backend_qa_engineer",
    "web_frontend_qa_engineer",
    "tui_frontend_qa_engineer",
]


class TeamRunnerV2:
    """Dynamic team runner with intelligent orchestration.
    
    Provides next-generation team running capabilities with:
    - Intelligent task classification and team composition
    - Agile coach integration for complex tasks
    - Dynamic orchestration with fallback mechanisms
    - Comprehensive retrospectives and continuous improvement
    - Full backward compatibility with original API
    """
    
    def __init__(self):
        """Initialize the dynamic team runner."""
        self.task_classifier = TaskClassifier()
        self.team_composer = TeamComposer()
        self.agile_coach = AgileCoachIntegration()
        self.retrospective_engine = RetrospectiveEngine()
        self._orchestrator: Optional[DynamicOrchestrator] = None
        self._communication_manager = None
        self._last_reasoning_logs = {}
        self._last_retrospective_report = {}
        
        # Initialize managed agent loader for controlled agent loading
        from .managed_agent_loader import get_managed_agent_loader
        self.agent_loader = get_managed_agent_loader()
        
    async def run_team(
        self,
        objective: str,
        roles: Iterable[str] | None = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> Dict[str, str]:
        """Run a dynamic team with intelligent orchestration.
        
        This is the main entry point that maintains API compatibility with the
        original run_team function while providing enhanced dynamic capabilities.
        
        Args:
            objective: The task objective/description
            roles: Optional specific roles to use (overrides dynamic composition)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping role names to their output results
        """
        start_time = time.time()
        
        try:
            # Load configuration and setup basic infrastructure
            cfg = Config.load()
            bus = EventBus()
            mgr = AgentManager(cfg, events=bus)
            
            # Step 1: Classify the task to understand its characteristics
            task_classification = await self._classify_task(objective)
            
            # Step 2: Determine if we should use dynamic orchestration or fallback
            use_dynamic = await self._should_use_dynamic_orchestration(
                task_classification, roles
            )
            
            if use_dynamic and roles is None:
                # Use dynamic orchestration for complex tasks without explicit roles
                return await self._run_dynamic_team(
                    objective, task_classification, mgr, progress_callback
                )
            else:
                # Use traditional orchestration for simple tasks or explicit roles
                return await self._run_traditional_team(
                    objective, roles or DEFAULT_TEAM, mgr, progress_callback
                )
                
        except Exception as e:
            logger.error(f"Team execution failed: {e}", exc_info=True)
            if progress_callback:
                try:
                    await progress_callback("team.error", {
                        "error": str(e),
                        "execution_time": time.time() - start_time
                    })
                except Exception:
                    pass
            
            # Fallback to traditional approach on any error
            return await self._run_traditional_team(
                objective, roles or DEFAULT_TEAM, AgentManager(Config.load(), EventBus()), progress_callback
            )
    
    async def _classify_task(self, objective: str) -> Any:
        """Classify the task using the task classifier."""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.task_classifier.classify, objective
            )
        except Exception as e:
            logger.warning(f"Task classification failed: {e}")
            # Return a basic classification for fallback
            return type('TaskClassification', (), {
                'task_type': TaskType.IMPLEMENTATION,
                'complexity': ComplexityLevel.MEDIUM,
                'required_roles': DEFAULT_TEAM,
                'risk_level': RiskLevel.MEDIUM,
                'estimated_effort': 50,
                'confidence': 0.5
            })()
    
    async def _should_use_dynamic_orchestration(
        self, 
        task_classification: Any, 
        explicit_roles: Optional[Iterable[str]]
    ) -> bool:
        """Determine whether to use dynamic orchestration."""
        # Use traditional approach if roles are explicitly specified
        if explicit_roles is not None:
            return False
            
        # Use dynamic orchestration for complex or high-risk tasks
        return (
            task_classification.complexity in [ComplexityLevel.HIGH, ComplexityLevel.CRITICAL] or
            task_classification.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] or
            task_classification.estimated_effort > 70
        )
    
    async def _run_dynamic_team(
        self,
        objective: str,
        task_classification: Any,
        agent_manager: AgentManager,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> Dict[str, str]:
        """Run team using dynamic orchestration."""
        logger.info(f"Running dynamic team for {task_classification.task_type} task")
        
        # Step 1: Agile coach planning for complex tasks
        planning_guidance = None
        if task_classification.complexity in [ComplexityLevel.HIGH, ComplexityLevel.CRITICAL]:
            # We need team composition first for agile coaching
            # For now, create a simple temporary composition
            from .models import AgentSpec, TeamComposition, CoordinationStrategy
            temp_agents = [AgentSpec(role=role, model_assignment="default") for role in (task_classification.required_roles or DEFAULT_TEAM[:3])]
            temp_composition = TeamComposition(
                primary_team=temp_agents,
                load_order=[agent.role for agent in temp_agents],
                coordination_strategy=CoordinationStrategy.PARALLEL,
                confidence_score=0.8
            )
            
            planning_guidance = await self.agile_coach.coach_planning(
                task_classification, temp_composition
            )
            
            if progress_callback:
                try:
                    await progress_callback("planning.completed", {
                        "guidance": planning_guidance.recommended_approach if planning_guidance else None
                    })
                except Exception:
                    pass
        
        # Step 2: Compose optimal team
        try:
            resource_constraints = ResourceConstraints(
                max_agents=len(DEFAULT_TEAM),
                memory_limit_mb=4096,
                time_limit_minutes=60,
                cost_limit=100.0
            )
            
            # Get available roles (use required roles from classification or default team)
            available_roles = (
                task_classification.required_roles 
                if task_classification.required_roles 
                else DEFAULT_TEAM
            )
            
            # Validate that all roles are managed agents
            try:
                validated_roles = self.agent_loader.validate_team_composition(available_roles)
                logger.info(f"Validated {len(validated_roles)} managed agents for dynamic team")
                
                # Preload agents for performance
                self.agent_loader.preload_agents_for_team(validated_roles)
                
                # Use validated roles for team composition
                available_roles = [role.value for role in validated_roles]
                
            except ValueError as e:
                logger.error(f"Team validation failed: {e}")
                # Fallback to traditional approach with default team
                return await self._run_traditional_team(
                    objective, DEFAULT_TEAM, agent_manager, progress_callback
                )
            
            team_composition = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.team_composer.compose_team,
                task_classification,
                available_roles,
                resource_constraints
            )
            
            if progress_callback:
                try:
                    await progress_callback("team.composed", {
                        "agents": [agent.role for agent in team_composition.primary_team],
                        "strategy": team_composition.coordination_strategy.value
                    })
                except Exception:
                    pass
                    
        except Exception as e:
            logger.warning(f"Team composition failed: {e}, falling back to default team")
            # Fallback to traditional approach
            return await self._run_traditional_team(
                objective, DEFAULT_TEAM, agent_manager, progress_callback
            )
        
        # Step 3: Dynamic orchestration with communication management
        try:
            if not self._orchestrator:
                from .agent_communication_manager import get_communication_manager
                self._orchestrator = DynamicOrchestrator(agent_manager=agent_manager)
                self._communication_manager = get_communication_manager()
            
            # Start orchestration session
            session_id = f"team_session_{int(time.time())}_{task_classification.task_type.value}"
            
            # Execute the team with dynamic orchestration
            execution_result = await self._orchestrator.orchestrate_team(
                team_spec=team_composition,
                objective=objective,
                progress_callback=progress_callback
            )
            
            # Convert execution result to the expected format
            results = {}
            if execution_result and hasattr(execution_result, 'agent_results'):
                results = execution_result.agent_results
            else:
                # Fallback: run agents directly
                results = await self._execute_agents_directly(
                    [agent.role for agent in team_composition.primary_team],
                    objective,
                    agent_manager,
                    progress_callback
                )
            
        except Exception as e:
            logger.warning(f"Dynamic orchestration failed: {e}, falling back to direct execution")
            results = await self._execute_agents_directly(
                [agent.role for agent in team_composition.primary_team],
                objective,
                agent_manager,
                progress_callback
            )
        
        # Step 4: Quality review gate (same as original)
        results = await self._run_quality_review_gate(results, agent_manager)
        
        # Step 5: Run retrospective for continuous improvement
        try:
            await self._run_retrospective(
                objective, task_classification, team_composition, results
            )
        except Exception as e:
            logger.warning(f"Retrospective failed: {e}")
        
        # Step 6: Generate and expose reasoning logs for analysis
        try:
            if hasattr(self, '_communication_manager'):
                reasoning_logs = await self._communication_manager.get_all_reasoning_logs()
                retrospective_report = await self._communication_manager.generate_retrospective_report()
                
                # Log summary for immediate feedback
                logger.info(f"Generated reasoning logs for {len(reasoning_logs)} agents")
                logger.info(f"Communication events: {retrospective_report.get('total_events', 0)}")
                
                # Store for external access
                self._last_reasoning_logs = reasoning_logs
                self._last_retrospective_report = retrospective_report
        except Exception as e:
            logger.warning(f"Failed to generate reasoning logs: {e}")
        
        return results
    
    async def _run_traditional_team(
        self,
        objective: str,
        roles: Iterable[str],
        agent_manager: AgentManager,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> Dict[str, str]:
        """Run team using traditional orchestration with managed agent validation."""
        logger.info("Running traditional team orchestration with managed agents")
        
        roles_list = list(roles)
        
        # Validate that all requested roles are managed agents
        try:
            validated_roles = self.agent_loader.validate_team_composition(roles_list)
            logger.info(f"Validated {len(validated_roles)} managed agents for traditional team")
            
            # Preload agents for performance
            self.agent_loader.preload_agents_for_team(validated_roles)
            
            # Convert back to strings for execution
            roles_list = [role.value for role in validated_roles]
            
        except ValueError as e:
            logger.error(f"Traditional team validation failed: {e}")
            # Use only the default managed agents as ultimate fallback
            logger.warning("Falling back to default managed agents only")
            try:
                validated_default = self.agent_loader.validate_team_composition(DEFAULT_TEAM)
                roles_list = [role.value for role in validated_default]
            except Exception as fallback_error:
                logger.critical(f"Even default team validation failed: {fallback_error}")
                raise
        
        results = await self._execute_agents_directly(
            roles_list, objective, agent_manager, progress_callback
        )
        
        # Quality review gate (same as original)
        results = await self._run_quality_review_gate(results, agent_manager)
        
        return results
    
    async def _execute_agents_directly(
        self,
        roles: List[str],
        objective: str,
        agent_manager: AgentManager,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> Dict[str, str]:
        """Execute agents directly with orchestrator-only communication."""
        from .agent_communication_manager import get_communication_manager
        
        results: Dict[str, str] = {}
        comm_manager = get_communication_manager()
        session_id = f"direct_execution_{int(time.time())}"
        
        async def _run_with_communication_tracking(role: str):
            # Start agent session for communication tracking
            await comm_manager.start_agent_session(role, session_id)
            
            try:
                # Log reasoning: Starting agent
                await comm_manager.log_agent_reasoning(
                    role, session_id,
                    f"Starting task: {objective[:100]}...",
                    confidence=0.9,
                    metadata={"action": "task_start"}
                )
                
                # Execute agent task
                task = TaskEnvelopeV1(objective=objective)
                job_id = await agent_manager.spawn_agent(role, task.objective)
                
                # Log tool use
                await comm_manager.log_agent_tool_use(
                    role, session_id, "agent_manager.spawn_agent",
                    {"objective": objective, "job_id": job_id}
                )
                
                if progress_callback:
                    try:
                        await progress_callback("job.spawned", {"agent": role})
                    except Exception:
                        pass
                
                # Log reasoning: Agent spawned
                await comm_manager.log_agent_reasoning(
                    role, session_id,
                    f"Agent spawned with job_id: {job_id}. Waiting for completion...",
                    confidence=0.8
                )
                        
                # Wait for completion
                status = await agent_manager.wait_for_completion(job_id)
                
                # Determine success
                success = bool(status.output and not status.error)
                agent_output = status.output or status.error or ""
                
                # Log final output
                await comm_manager.log_agent_output(
                    role, session_id, agent_output,
                    output_type="final_result"
                )
                
                # Log decision: Task completion
                await comm_manager.log_agent_decision(
                    role, session_id,
                    f"Task {'completed successfully' if success else 'completed with issues'}",
                    {
                        "success": success,
                        "output_length": len(agent_output),
                        "has_error": bool(status.error)
                    },
                    confidence=0.9 if success else 0.6
                )
                
                results[role] = agent_output
                
                if progress_callback:
                    try:
                        await progress_callback("job.completed", {"agent": role})
                    except Exception:
                        pass
                
                # End agent session successfully
                await comm_manager.end_agent_session(
                    role, session_id, success=success, final_output=agent_output
                )
                
            except Exception as e:
                error_msg = str(e)
                
                # Log error
                await comm_manager.log_agent_error(
                    role, session_id, error_msg,
                    {"exception_type": type(e).__name__}
                )
                
                # End agent session with error
                await comm_manager.end_agent_session(
                    role, session_id, success=False, error_details=error_msg
                )
                
                results[role] = f"Error: {error_msg}"
        
        # Execute all agents with communication tracking
        await asyncio.gather(*[_run_with_communication_tracking(r) for r in roles])
        return results
    
    async def _run_quality_review_gate(
        self, 
        results: Dict[str, str], 
        agent_manager: AgentManager
    ) -> Dict[str, str]:
        """Run the quality review gate (same as original implementation)."""
        # Check for staged changes and run QA review
        staged_root = Path.cwd().resolve() / 'build' / 'staging'
        if staged_root.exists() and any(p.is_file() for p in staged_root.rglob('*')):
            review_objective = (
                "Review the staged changes and either approve or discard them. "
                "Use list_staged_changes, git_diff, and approve_changes(commit_message='feat: reviewed changes') if acceptable; "
                "otherwise call discard_staged_changes with a short explanation."
            )
            
            # Run chief QA explicitly with review instruction
            job_id = await agent_manager.spawn_agent('chief_qa_engineer', review_objective)
            status = await agent_manager.wait_for_completion(job_id)
            results['chief_qa_engineer'] = status.output or status.error or ''
        
        return results
    
    async def _run_retrospective(
        self,
        objective: str,
        task_classification: Any,
        team_composition: Any,
        results: Dict[str, str]
    ):
        """Run retrospective analysis for continuous improvement."""
        try:
            # Prepare retrospective context
            execution_context = {
                'objective': objective,
                'task_type': task_classification.task_type,
                'complexity': task_classification.complexity,
                'risk_level': task_classification.risk_level,
                'team_size': len(results),
                'agents_used': list(results.keys()),
                'results_summary': {
                    role: len(output) for role, output in results.items()
                }
            }
            
            # Run retrospective
            retrospective_result = await self.retrospective_engine.conduct_retrospective(
                RetrospectiveType.POST_TASK,
                RetrospectiveScope.COMPREHENSIVE,
                execution_context
            )
            
            logger.info(f"Retrospective completed with {len(retrospective_result.improvement_actions)} improvement actions")
            
        except Exception as e:
            logger.warning(f"Retrospective execution failed: {e}")
    
    async def get_last_reasoning_logs(self) -> Dict[str, Any]:
        """Get the reasoning logs from the last team execution."""
        return self._last_reasoning_logs
    
    async def get_last_retrospective_report(self) -> Dict[str, Any]:
        """Get the retrospective report from the last team execution."""
        return self._last_retrospective_report
    
    async def get_communication_history(
        self,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Any]:
        """Get communication history for analysis."""
        if self._communication_manager:
            return await self._communication_manager.get_communication_history(
                session_id=session_id,
                agent_id=agent_id,
                limit=limit
            )
        return []
    
    async def get_agent_loading_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent loading and performance."""
        return self.agent_loader.get_loading_statistics()


# Create a singleton instance for the module-level function
_team_runner_v2 = TeamRunnerV2()


async def run_team_v2(
    objective: str, 
    roles: Iterable[str] | None = None, 
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
) -> Dict[str, str]:
    """Enhanced team runner with dynamic orchestration.
    
    This function provides the next-generation team running capabilities while
    maintaining full backward compatibility with the original run_team API.
    
    Features:
    - Intelligent task classification and analysis
    - Dynamic team composition based on task requirements
    - Agile coach integration for complex tasks
    - Automated retrospectives for continuous improvement
    - Fallback to traditional orchestration when needed
    - Full backward compatibility with existing callers
    
    Args:
        objective: The task objective/description
        roles: Optional specific roles to use (overrides dynamic composition)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary mapping role names to their output results
        
    Usage:
        # Basic usage (same as original)
        results = await run_team_v2("Build a REST API")
        
        # With specific roles (same as original)
        results = await run_team_v2("Build a REST API", ["backend_engineer", "api_engineer"])
        
        # With progress callback (same as original)
        async def progress(event, data):
            print(f"Progress: {event} - {data}")
        results = await run_team_v2("Build a REST API", progress_callback=progress)
    """
    return await _team_runner_v2.run_team(objective, roles, progress_callback)


# Backward compatibility: export the enhanced function as the default
async def run_team(
    objective: str, 
    roles: Iterable[str] | None = None, 
    progress_callback=None
) -> Dict[str, str]:
    """Run a team of role agents with intelligent orchestration.
    
    This is the enhanced version of the original run_team function that now
    includes intelligent task classification, dynamic team composition, and
    agile coaching while maintaining full backward compatibility.
    
    For complex tasks, this function will:
    1. Classify the task to understand its requirements
    2. Include agile coach for planning and guidance
    3. Compose an optimal team based on task analysis
    4. Use dynamic orchestration for execution
    5. Run retrospectives for continuous improvement
    
    For simple tasks or when explicit roles are provided, it falls back to
    the traditional orchestration approach for reliability.
    
    Args:
        objective: The task objective/description
        roles: Optional specific roles to use (overrides dynamic composition)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary mapping role names to their output results
    """
    return await run_team_v2(objective, roles, progress_callback)


# Convenience functions for accessing reasoning logs and retrospective data
async def get_last_reasoning_logs() -> Dict[str, Any]:
    """Get the reasoning logs from the last team execution."""
    return await _team_runner_v2.get_last_reasoning_logs()


async def get_last_retrospective_report() -> Dict[str, Any]:
    """Get the retrospective report from the last team execution."""
    return await _team_runner_v2.get_last_retrospective_report()


async def get_communication_history(
    session_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Any]:
    """Get communication history for analysis."""
    return await _team_runner_v2.get_communication_history(
        session_id=session_id,
        agent_id=agent_id,
        limit=limit
    )


async def get_agent_loading_statistics() -> Dict[str, Any]:
    """Get statistics about agent loading and performance."""
    return await _team_runner_v2.get_agent_loading_statistics()


# Export both functions for flexibility
__all__ = [
    'TeamRunnerV2',
    'run_team',
    'run_team_v2',
    'get_last_reasoning_logs',
    'get_last_retrospective_report',
    'get_communication_history',
    'get_agent_loading_statistics',
    'DEFAULT_TEAM'
]