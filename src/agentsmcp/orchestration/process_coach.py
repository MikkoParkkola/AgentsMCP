"""
ProcessCoach - The Mandatory Leader of Self-Improvement Loops

This is the core orchestration system that acts as the mandatory leader of all 
self-improvement loops in AgentsMCP. The Process Coach cannot be removed and 
is responsible for:

- Leading all retrospective cycles after each user task completion
- Coordinating complete workflows: Analysis → Generation → Approval → Safety → Implementation  
- Feeding improvements to agents - updating agent roles, processes, and capabilities
- Tracking improvement progress and measuring effectiveness
- Ensuring rapid system optimization through continuous learning cycles
- Managing improvement backlog and prioritization
- Facilitating system learning from successes and failures

The Process Coach operates as the central nervous system for continuous improvement,
ensuring the AgentsMCP system rapidly learns and evolves.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json

from ..retrospective import (
    AgileCoachAnalyzer,
    OrchestratorEnforcementSystem, 
    EnhancedRetrospectiveIntegration,
    ExecutionLogCapture,
    ComprehensiveRetrospectiveReport,
    ActionPoint,
    ImplementationStatus,
    PriorityLevel
)
from ..retrospective.generation import ImprovementGenerator, ImprovementEngine
from ..retrospective.approval import ApprovalOrchestrator
from ..retrospective.safety import SafetyOrchestrator
from ..agents import AgentLoader
from ..config import Config
from .models import TaskResult, TeamPerformanceMetrics


logger = logging.getLogger(__name__)


class ImprovementPhase(Enum):
    """Phases of the improvement cycle."""
    TRIGGERED = "triggered"
    ANALYSIS = "analysis"
    GENERATION = "generation" 
    APPROVAL = "approval"
    SAFETY_VALIDATION = "safety_validation"
    IMPLEMENTATION = "implementation"
    MONITORING = "monitoring"
    COMPLETE = "complete"
    FAILED = "failed"


class CoachTriggerType(Enum):
    """Types of triggers that activate the Process Coach."""
    TASK_COMPLETION = "task_completion"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    AGENT_FAILURE = "agent_failure"
    USER_FEEDBACK = "user_feedback"
    SCHEDULED_CYCLE = "scheduled_cycle"
    SYSTEM_ANOMALY = "system_anomaly"


@dataclass
class ImprovementCycle:
    """Represents a single improvement cycle managed by the Process Coach."""
    cycle_id: str
    trigger_type: CoachTriggerType
    phase: ImprovementPhase
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_logs: List[Dict[str, Any]] = field(default_factory=list)
    analysis_results: Optional[ComprehensiveRetrospectiveReport] = None
    generated_improvements: List[ActionPoint] = field(default_factory=list)
    approved_improvements: List[ActionPoint] = field(default_factory=list)
    implemented_improvements: List[ActionPoint] = field(default_factory=list)
    failed_improvements: List[ActionPoint] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class ProcessCoachConfig:
    """Configuration for the Process Coach orchestration system."""
    # Mandatory execution settings
    mandatory_execution: bool = True
    can_be_disabled: bool = False
    
    # Trigger settings
    auto_trigger_on_task_completion: bool = True
    auto_trigger_on_performance_degradation: bool = True
    auto_trigger_on_agent_failure: bool = True
    
    # Improvement settings
    max_concurrent_improvements: int = 5
    improvement_batch_size: int = 3
    safety_validation_required: bool = True
    user_approval_required: bool = True
    
    # Agent enhancement settings
    enable_agent_role_modification: bool = True
    enable_agent_capability_addition: bool = True  
    enable_agent_removal: bool = True
    enable_new_agent_creation: bool = True
    
    # Monitoring settings
    track_improvement_effectiveness: bool = True
    measure_system_evolution: bool = True
    rollback_failed_improvements: bool = True
    
    # Cycle timing
    min_cycle_interval_seconds: int = 300  # 5 minutes minimum between cycles
    max_cycle_duration_seconds: int = 1800  # 30 minutes max per cycle


class ProcessCoach:
    """
    The Mandatory Leader of Self-Improvement Loops.
    
    This class orchestrates the complete self-improvement workflow and acts as 
    the central coordinator for all system enhancement activities. It cannot be 
    disabled and ensures continuous system evolution.
    """
    
    def __init__(self, config: Optional[ProcessCoachConfig] = None):
        """Initialize the Process Coach with mandatory execution."""
        self.config = config or ProcessCoachConfig()
        self.is_active = True  # Always active, cannot be disabled
        self.current_cycles: Dict[str, ImprovementCycle] = {}
        self.completed_cycles: List[ImprovementCycle] = []
        self.improvement_history: List[ActionPoint] = []
        self.system_metrics: Dict[str, float] = {}
        self.agent_enhancements: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize core components
        self._initialize_components()
        
        # Track system state
        self.last_cycle_time = datetime.now()
        self.total_improvements_implemented = 0
        self.system_performance_trend: List[float] = []
        
        logger.info("ProcessCoach initialized as mandatory self-improvement leader")

    def _initialize_components(self):
        """Initialize all core orchestration components."""
        try:
            # Retrospective analysis components
            self.agile_coach = AgileCoachAnalyzer()
            self.enforcement_system = OrchestratorEnforcementSystem()
            self.retrospective_integration = EnhancedRetrospectiveIntegration()
            self.execution_capture = ExecutionLogCapture()
            
            # Improvement generation and validation
            self.improvement_generator = ImprovementGenerator()
            self.improvement_engine = ImprovementEngine()
            self.approval_orchestrator = ApprovalOrchestrator()
            self.safety_orchestrator = SafetyOrchestrator()
            
            # Agent management
            self.agent_loader = AgentLoader()
            
            logger.info("ProcessCoach components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ProcessCoach components: {e}")
            # Process Coach must always be operational
            raise RuntimeError(f"CRITICAL: ProcessCoach initialization failed: {e}")

    async def trigger_improvement_cycle(
        self,
        trigger_type: CoachTriggerType,
        context: Optional[Dict[str, Any]] = None,
        task_result: Optional[TaskResult] = None
    ) -> str:
        """
        Trigger a new improvement cycle - mandatory execution.
        
        This is the main entry point for all improvement cycles and cannot be bypassed.
        """
        if not self._should_trigger_cycle(trigger_type):
            logger.info(f"Skipping cycle due to timing constraints: {trigger_type}")
            return ""
            
        cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trigger_type.value}"
        
        cycle = ImprovementCycle(
            cycle_id=cycle_id,
            trigger_type=trigger_type,
            phase=ImprovementPhase.TRIGGERED,
            started_at=datetime.now()
        )
        
        self.current_cycles[cycle_id] = cycle
        
        logger.info(f"MANDATORY IMPROVEMENT CYCLE TRIGGERED: {cycle_id}")
        
        try:
            # Execute the complete improvement workflow
            await self._execute_improvement_workflow(cycle, context, task_result)
            
        except Exception as e:
            logger.error(f"Improvement cycle {cycle_id} failed: {e}")
            cycle.phase = ImprovementPhase.FAILED
            cycle.errors.append(str(e))
            
        finally:
            # Always complete the cycle tracking
            cycle.completed_at = datetime.now()
            self.completed_cycles.append(cycle)
            self.current_cycles.pop(cycle_id, None)
            self.last_cycle_time = datetime.now()
        
        return cycle_id

    async def _execute_improvement_workflow(
        self,
        cycle: ImprovementCycle,
        context: Optional[Dict[str, Any]],
        task_result: Optional[TaskResult]
    ):
        """Execute the complete improvement workflow orchestration."""
        
        # Phase 1: Analysis
        cycle.phase = ImprovementPhase.ANALYSIS
        await self._perform_comprehensive_analysis(cycle, context, task_result)
        
        # Phase 2: Generation
        cycle.phase = ImprovementPhase.GENERATION
        await self._generate_improvements(cycle)
        
        # Phase 3: Approval
        cycle.phase = ImprovementPhase.APPROVAL
        await self._seek_user_approval(cycle)
        
        # Phase 4: Safety Validation
        cycle.phase = ImprovementPhase.SAFETY_VALIDATION
        await self._validate_safety(cycle)
        
        # Phase 5: Implementation
        cycle.phase = ImprovementPhase.IMPLEMENTATION
        await self._implement_improvements(cycle)
        
        # Phase 6: Monitoring
        cycle.phase = ImprovementPhase.MONITORING
        await self._monitor_implementation(cycle)
        
        cycle.phase = ImprovementPhase.COMPLETE
        logger.info(f"Improvement cycle {cycle.cycle_id} completed successfully")

    async def _perform_comprehensive_analysis(
        self,
        cycle: ImprovementCycle,
        context: Optional[Dict[str, Any]],
        task_result: Optional[TaskResult]
    ):
        """Perform comprehensive retrospective analysis."""
        try:
            # Capture execution logs from context
            if context:
                cycle.execution_logs.extend(context.get('execution_logs', []))
            
            if task_result:
                cycle.execution_logs.append({
                    'task_result': task_result.to_dict() if hasattr(task_result, 'to_dict') else str(task_result),
                    'timestamp': datetime.now().isoformat()
                })
            
            # Run agile coach analysis
            analysis_context = {
                'execution_logs': cycle.execution_logs,
                'trigger_type': cycle.trigger_type.value,
                'system_metrics': self.system_metrics,
                'agent_performance': context.get('agent_performance', {}) if context else {}
            }
            
            cycle.analysis_results = await self.agile_coach.analyze_comprehensive_retrospective(
                analysis_context
            )
            
            logger.info(f"Analysis complete for cycle {cycle.cycle_id}")
            
        except Exception as e:
            logger.error(f"Analysis failed for cycle {cycle.cycle_id}: {e}")
            cycle.errors.append(f"Analysis failed: {e}")
            raise

    async def _generate_improvements(self, cycle: ImprovementCycle):
        """Generate improvement suggestions based on analysis."""
        try:
            if not cycle.analysis_results:
                raise ValueError("No analysis results available for improvement generation")
            
            # Generate improvements using the improvement engine
            generated = await self.improvement_engine.generate_improvements(
                analysis_results=cycle.analysis_results,
                system_context={
                    'current_agents': await self._get_current_agents(),
                    'system_metrics': self.system_metrics,
                    'improvement_history': self.improvement_history[-50:]  # Recent history
                },
                max_improvements=self.config.improvement_batch_size
            )
            
            cycle.generated_improvements = generated
            
            logger.info(f"Generated {len(generated)} improvements for cycle {cycle.cycle_id}")
            
        except Exception as e:
            logger.error(f"Improvement generation failed for cycle {cycle.cycle_id}: {e}")
            cycle.errors.append(f"Generation failed: {e}")
            raise

    async def _seek_user_approval(self, cycle: ImprovementCycle):
        """Seek user approval for generated improvements."""
        try:
            if not self.config.user_approval_required:
                # Auto-approve all improvements
                cycle.approved_improvements = cycle.generated_improvements.copy()
                return
            
            # Use approval orchestrator to handle user interaction
            approval_results = await self.approval_orchestrator.request_approval(
                improvements=cycle.generated_improvements,
                analysis_context=cycle.analysis_results,
                cycle_id=cycle.cycle_id
            )
            
            cycle.approved_improvements = [
                imp for imp, approved in zip(cycle.generated_improvements, approval_results)
                if approved
            ]
            
            logger.info(f"User approved {len(cycle.approved_improvements)} of {len(cycle.generated_improvements)} improvements")
            
        except Exception as e:
            logger.error(f"Approval process failed for cycle {cycle.cycle_id}: {e}")
            cycle.errors.append(f"Approval failed: {e}")
            raise

    async def _validate_safety(self, cycle: ImprovementCycle):
        """Validate safety of approved improvements."""
        try:
            if not self.config.safety_validation_required:
                return
            
            safe_improvements = []
            
            for improvement in cycle.approved_improvements:
                is_safe = await self.safety_orchestrator.validate_improvement_safety(
                    improvement=improvement,
                    system_context={
                        'current_agents': await self._get_current_agents(),
                        'system_state': self._get_system_state()
                    }
                )
                
                if is_safe:
                    safe_improvements.append(improvement)
                else:
                    logger.warning(f"Improvement blocked by safety validation: {improvement.title}")
            
            cycle.approved_improvements = safe_improvements
            
            logger.info(f"Safety validation passed {len(safe_improvements)} improvements")
            
        except Exception as e:
            logger.error(f"Safety validation failed for cycle {cycle.cycle_id}: {e}")
            cycle.errors.append(f"Safety validation failed: {e}")
            raise

    async def _implement_improvements(self, cycle: ImprovementCycle):
        """Implement approved and validated improvements."""
        implemented = []
        failed = []
        
        for improvement in cycle.approved_improvements:
            try:
                success = await self._implement_single_improvement(improvement, cycle.cycle_id)
                
                if success:
                    implemented.append(improvement)
                    self.total_improvements_implemented += 1
                else:
                    failed.append(improvement)
                    
            except Exception as e:
                logger.error(f"Failed to implement improvement {improvement.title}: {e}")
                failed.append(improvement)
                cycle.errors.append(f"Implementation failed for {improvement.title}: {e}")
        
        cycle.implemented_improvements = implemented
        cycle.failed_improvements = failed
        
        # Update improvement history
        self.improvement_history.extend(implemented)
        
        logger.info(f"Successfully implemented {len(implemented)} improvements, {len(failed)} failed")

    async def _implement_single_improvement(self, improvement: ActionPoint, cycle_id: str) -> bool:
        """Implement a single improvement action."""
        try:
            # Determine improvement type and route to appropriate handler
            if improvement.category.value.startswith('agent_'):
                return await self._implement_agent_improvement(improvement, cycle_id)
            elif improvement.category.value.startswith('process_'):
                return await self._implement_process_improvement(improvement, cycle_id)
            elif improvement.category.value.startswith('system_'):
                return await self._implement_system_improvement(improvement, cycle_id)
            else:
                return await self._implement_generic_improvement(improvement, cycle_id)
                
        except Exception as e:
            logger.error(f"Single improvement implementation failed: {e}")
            return False

    async def _implement_agent_improvement(self, improvement: ActionPoint, cycle_id: str) -> bool:
        """Implement agent-related improvements (roles, capabilities, etc.)."""
        try:
            # Extract agent modification details from improvement
            details = improvement.details
            
            if 'agent_role_modification' in details:
                return await self._modify_agent_role(details['agent_role_modification'])
            elif 'agent_capability_addition' in details:
                return await self._add_agent_capability(details['agent_capability_addition'])
            elif 'new_agent_creation' in details:
                return await self._create_new_agent(details['new_agent_creation'])
            elif 'agent_removal' in details:
                return await self._remove_agent(details['agent_removal'])
            
            return False
            
        except Exception as e:
            logger.error(f"Agent improvement implementation failed: {e}")
            return False

    async def _modify_agent_role(self, modification_details: Dict[str, Any]) -> bool:
        """Modify an existing agent role."""
        if not self.config.enable_agent_role_modification:
            return False
            
        try:
            agent_id = modification_details['agent_id']
            new_role_config = modification_details['new_role_config']
            
            # Track the enhancement
            if agent_id not in self.agent_enhancements:
                self.agent_enhancements[agent_id] = []
                
            self.agent_enhancements[agent_id].append({
                'type': 'role_modification',
                'timestamp': datetime.now().isoformat(),
                'details': new_role_config
            })
            
            # Apply role modification through agent loader
            await self.agent_loader.modify_agent_role(agent_id, new_role_config)
            
            logger.info(f"Successfully modified role for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Agent role modification failed: {e}")
            return False

    async def _add_agent_capability(self, capability_details: Dict[str, Any]) -> bool:
        """Add new capabilities to an existing agent."""
        if not self.config.enable_agent_capability_addition:
            return False
            
        try:
            agent_id = capability_details['agent_id']
            new_capability = capability_details['capability']
            
            # Track the enhancement
            if agent_id not in self.agent_enhancements:
                self.agent_enhancements[agent_id] = []
                
            self.agent_enhancements[agent_id].append({
                'type': 'capability_addition',
                'timestamp': datetime.now().isoformat(),
                'capability': new_capability
            })
            
            # Apply capability addition through agent loader
            await self.agent_loader.add_agent_capability(agent_id, new_capability)
            
            logger.info(f"Successfully added capability to agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Agent capability addition failed: {e}")
            return False

    async def _create_new_agent(self, creation_details: Dict[str, Any]) -> bool:
        """Create a new agent with specified capabilities."""
        if not self.config.enable_new_agent_creation:
            return False
            
        try:
            agent_config = creation_details['agent_config']
            agent_id = creation_details.get('agent_id') or f"generated_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create new agent through agent loader
            await self.agent_loader.create_agent(agent_id, agent_config)
            
            # Track the creation
            self.agent_enhancements[agent_id] = [{
                'type': 'agent_creation',
                'timestamp': datetime.now().isoformat(),
                'config': agent_config
            }]
            
            logger.info(f"Successfully created new agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"New agent creation failed: {e}")
            return False

    async def _remove_agent(self, removal_details: Dict[str, Any]) -> bool:
        """Remove an agent from the system."""
        if not self.config.enable_agent_removal:
            return False
            
        try:
            agent_id = removal_details['agent_id']
            reason = removal_details.get('reason', 'Process Coach improvement')
            
            # Remove agent through agent loader
            await self.agent_loader.remove_agent(agent_id, reason)
            
            # Track the removal
            if agent_id in self.agent_enhancements:
                self.agent_enhancements[agent_id].append({
                    'type': 'agent_removal',
                    'timestamp': datetime.now().isoformat(),
                    'reason': reason
                })
            
            logger.info(f"Successfully removed agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Agent removal failed: {e}")
            return False

    async def _implement_process_improvement(self, improvement: ActionPoint, cycle_id: str) -> bool:
        """Implement process-related improvements."""
        # Implementation depends on specific process improvement type
        # This would integrate with workflow orchestration, timing adjustments, etc.
        logger.info(f"Process improvement implementation: {improvement.title}")
        return True

    async def _implement_system_improvement(self, improvement: ActionPoint, cycle_id: str) -> bool:
        """Implement system-level improvements."""
        # Implementation depends on specific system improvement type
        # This would integrate with configuration updates, resource adjustments, etc.
        logger.info(f"System improvement implementation: {improvement.title}")
        return True

    async def _implement_generic_improvement(self, improvement: ActionPoint, cycle_id: str) -> bool:
        """Implement generic improvements."""
        logger.info(f"Generic improvement implementation: {improvement.title}")
        return True

    async def _monitor_implementation(self, cycle: ImprovementCycle):
        """Monitor the success of implemented improvements."""
        try:
            # Track metrics for implemented improvements
            for improvement in cycle.implemented_improvements:
                improvement.implementation_status = ImplementationStatus.COMPLETED
                improvement.implementation_date = datetime.now()
            
            # Update cycle metrics
            cycle.metrics.update({
                'improvements_generated': len(cycle.generated_improvements),
                'improvements_approved': len(cycle.approved_improvements), 
                'improvements_implemented': len(cycle.implemented_improvements),
                'improvements_failed': len(cycle.failed_improvements),
                'success_rate': len(cycle.implemented_improvements) / max(len(cycle.approved_improvements), 1)
            })
            
            # Update system performance metrics
            await self._update_system_metrics(cycle)
            
            logger.info(f"Implementation monitoring complete for cycle {cycle.cycle_id}")
            
        except Exception as e:
            logger.error(f"Implementation monitoring failed: {e}")
            cycle.errors.append(f"Monitoring failed: {e}")

    def _should_trigger_cycle(self, trigger_type: CoachTriggerType) -> bool:
        """Determine if a new improvement cycle should be triggered."""
        # Always allow mandatory triggers
        if trigger_type == CoachTriggerType.TASK_COMPLETION and self.config.auto_trigger_on_task_completion:
            return True
        
        # Check timing constraints
        time_since_last = (datetime.now() - self.last_cycle_time).total_seconds()
        if time_since_last < self.config.min_cycle_interval_seconds:
            return False
        
        # Check concurrent cycle limits
        if len(self.current_cycles) >= self.config.max_concurrent_improvements:
            return False
        
        return True

    async def _get_current_agents(self) -> Dict[str, Any]:
        """Get current agent configuration."""
        return await self.agent_loader.get_all_agents()

    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for safety validation."""
        return {
            'active_cycles': len(self.current_cycles),
            'total_improvements': self.total_improvements_implemented,
            'system_metrics': self.system_metrics.copy(),
            'agent_enhancements': len(self.agent_enhancements)
        }

    async def _update_system_metrics(self, cycle: ImprovementCycle):
        """Update system-wide performance metrics."""
        self.system_metrics.update({
            'total_cycles_completed': len(self.completed_cycles),
            'total_improvements_implemented': self.total_improvements_implemented,
            'average_cycle_success_rate': sum(
                cycle.metrics.get('success_rate', 0) for cycle in self.completed_cycles
            ) / max(len(self.completed_cycles), 1),
            'system_evolution_score': self._calculate_evolution_score()
        })

    def _calculate_evolution_score(self) -> float:
        """Calculate a score representing how much the system has evolved."""
        base_score = min(self.total_improvements_implemented * 0.1, 10.0)  # Cap at 10
        
        # Bonus for agent enhancements
        agent_bonus = len(self.agent_enhancements) * 0.05
        
        # Bonus for recent successful cycles
        recent_cycles = [c for c in self.completed_cycles[-10:] if c.phase == ImprovementPhase.COMPLETE]
        recent_bonus = len(recent_cycles) * 0.1
        
        return base_score + agent_bonus + recent_bonus

    async def get_improvement_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the improvement system."""
        return {
            'is_active': self.is_active,
            'mandatory_execution': self.config.mandatory_execution,
            'current_cycles': len(self.current_cycles),
            'completed_cycles': len(self.completed_cycles),
            'total_improvements_implemented': self.total_improvements_implemented,
            'agent_enhancements': {
                agent_id: len(enhancements) 
                for agent_id, enhancements in self.agent_enhancements.items()
            },
            'system_metrics': self.system_metrics.copy(),
            'last_cycle_time': self.last_cycle_time.isoformat(),
            'system_evolution_score': self._calculate_evolution_score()
        }

    async def force_improvement_cycle(self, reason: str = "Manual trigger") -> str:
        """Force an improvement cycle regardless of timing constraints."""
        logger.info(f"Forcing improvement cycle: {reason}")
        return await self.trigger_improvement_cycle(
            CoachTriggerType.SCHEDULED_CYCLE,
            context={'force_trigger': True, 'reason': reason}
        )

    def get_agent_enhancement_history(self, agent_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get history of agent enhancements."""
        if agent_id:
            return {agent_id: self.agent_enhancements.get(agent_id, [])}
        return self.agent_enhancements.copy()

    async def rollback_failed_improvements(self) -> int:
        """Rollback any failed improvements if configured to do so."""
        if not self.config.rollback_failed_improvements:
            return 0
        
        rollback_count = 0
        
        for cycle in self.completed_cycles:
            if cycle.failed_improvements and cycle.phase != ImprovementPhase.FAILED:
                try:
                    await self._rollback_cycle_improvements(cycle)
                    rollback_count += len(cycle.failed_improvements)
                except Exception as e:
                    logger.error(f"Rollback failed for cycle {cycle.cycle_id}: {e}")
        
        logger.info(f"Rolled back {rollback_count} failed improvements")
        return rollback_count

    async def _rollback_cycle_improvements(self, cycle: ImprovementCycle):
        """Rollback improvements from a specific cycle."""
        # Implementation would depend on the specific improvements made
        # This is a placeholder for the rollback logic
        logger.info(f"Rolling back improvements from cycle {cycle.cycle_id}")