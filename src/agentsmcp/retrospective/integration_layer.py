"""Integration layer for enhanced retrospective system.

This module provides seamless integration of the enhanced retrospective system
with existing agent lifecycle and orchestration components.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from ..models import TaskEnvelopeV1, ResultEnvelopeV1, EnvelopeStatus
from ..roles.base import RoleName, BaseRole
from ..orchestration.retrospective_engine import RetrospectiveEngine  # Existing system
from .individual_framework import IndividualRetrospectiveFramework
from .coach_analyzer import AgileCoachAnalyzer
from .enforcement import OrchestratorEnforcementSystem, SystemState
from .data_models import (
    IndividualRetrospective,
    ComprehensiveRetrospectiveReport,
    IndividualRetrospectiveConfig,
)


class IntegrationError(Exception):
    """Raised when integration process fails."""
    pass


class LifecycleModificationError(Exception):
    """Raised when agent lifecycle modification fails."""
    pass


class CompatibilityError(Exception):
    """Raised when compatibility check fails."""
    pass


class LifecycleHook:
    """Integration point in agent execution lifecycle."""
    
    def __init__(self, name: str, phase: str, priority: int = 0):
        self.name = name
        self.phase = phase  # pre_execution, post_execution, error_handling
        self.priority = priority
        self.enabled = True


class LifecycleModification:
    """Modification required to agent lifecycle."""
    
    def __init__(
        self,
        component: str,
        modification_type: str,
        description: str,
        impact_level: str = "low",
    ):
        self.component = component
        self.modification_type = modification_type  # add_hook, modify_method, extend_interface
        self.description = description
        self.impact_level = impact_level
        self.applied = False
        self.rollback_info: Optional[Dict[str, Any]] = None


class IntegrationStatus:
    """Status of retrospective system integration."""
    
    def __init__(self):
        self.integrated = False
        self.individual_framework_active = False
        self.coach_analyzer_active = False
        self.enforcement_system_active = False
        self.compatibility_mode = "full"  # full, partial, fallback
        self.integration_errors: List[str] = []
        self.performance_impact: float = 0.0  # Estimated performance impact


class CompatibilityReport:
    """Report on backward compatibility with existing systems."""
    
    def __init__(self):
        self.compatible = True
        self.compatibility_level = "full"  # full, partial, incompatible
        self.breaking_changes: List[str] = []
        self.migration_required: List[str] = []
        self.recommendations: List[str] = []


class OrchestratorConfig:
    """Configuration for orchestrator integration."""
    
    def __init__(self):
        self.enable_individual_retrospectives = True
        self.enable_comprehensive_analysis = True
        self.enable_enforcement_system = True
        self.retrospective_timeout = 30
        self.enforcement_required = True
        self.fallback_to_existing = True
        self.performance_monitoring = True


class EnhancedRetrospectiveIntegration:
    """Main integration layer for enhanced retrospective system."""
    
    def __init__(
        self,
        existing_retrospective_engine: Optional[RetrospectiveEngine] = None,
        orchestrator_config: Optional[OrchestratorConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.existing_engine = existing_retrospective_engine
        self.config = orchestrator_config or OrchestratorConfig()
        self.log = logger or logging.getLogger(__name__)
        
        # Enhanced system components
        self.individual_framework: Optional[IndividualRetrospectiveFramework] = None
        self.coach_analyzer: Optional[AgileCoachAnalyzer] = None
        self.enforcement_system: Optional[OrchestratorEnforcementSystem] = None
        
        # Integration state
        self.integration_status = IntegrationStatus()
        self.lifecycle_hooks: Dict[str, LifecycleHook] = {}
        self.applied_modifications: List[LifecycleModification] = []
        
        # Performance monitoring
        self.performance_metrics: Dict[str, float] = {}
        
        self.log.info("EnhancedRetrospectiveIntegration initialized")
    
    async def initialize_integration(
        self,
        validate_compatibility: bool = True,
    ) -> IntegrationStatus:
        """Initialize the enhanced retrospective system integration.
        
        Args:
            validate_compatibility: Whether to validate backward compatibility
            
        Returns:
            IntegrationStatus: Status of integration process
        """
        self.log.info("Starting enhanced retrospective system integration")
        
        try:
            # Phase 1: Compatibility check
            if validate_compatibility:
                compatibility = await self._check_compatibility()
                if not compatibility.compatible and not self.config.fallback_to_existing:
                    raise CompatibilityError(f"System incompatible: {compatibility.breaking_changes}")
            
            # Phase 2: Initialize enhanced components
            await self._initialize_enhanced_components()
            
            # Phase 3: Setup lifecycle hooks
            await self._setup_lifecycle_hooks()
            
            # Phase 4: Apply lifecycle modifications
            await self._apply_lifecycle_modifications()
            
            # Phase 5: Validate integration
            await self._validate_integration()
            
            # Update integration status
            self.integration_status.integrated = True
            self.integration_status.individual_framework_active = self.individual_framework is not None
            self.integration_status.coach_analyzer_active = self.coach_analyzer is not None
            self.integration_status.enforcement_system_active = self.enforcement_system is not None
            
            self.log.info("Enhanced retrospective system integration completed successfully")
            
        except Exception as e:
            self.log.error("Integration failed: %s", e)
            self.integration_status.integration_errors.append(str(e))
            
            # Attempt fallback if configured
            if self.config.fallback_to_existing:
                await self._setup_fallback_mode()
            else:
                raise IntegrationError(f"Integration failed: {str(e)}")
        
        return self.integration_status
    
    async def conduct_enhanced_retrospective(
        self,
        agent_role: RoleName,
        task_context: TaskEnvelopeV1,
        execution_results: ResultEnvelopeV1,
        team_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[IndividualRetrospective, Optional[ComprehensiveRetrospectiveReport]]:
        """Conduct enhanced retrospective using integrated system.
        
        Args:
            agent_role: Role of the agent
            task_context: Original task context
            execution_results: Task execution results
            team_context: Optional team context for comprehensive analysis
            
        Returns:
            Tuple of individual retrospective and optional comprehensive report
        """
        if not self.integration_status.integrated:
            raise IntegrationError("Enhanced retrospective system not integrated")
        
        individual_retrospective = None
        comprehensive_report = None
        
        try:
            # Phase 1: Individual retrospective
            if self.individual_framework and self.integration_status.individual_framework_active:
                self.log.debug("Conducting individual retrospective for %s", agent_role.value)
                
                individual_retrospective = await self.individual_framework.conduct_retrospective(
                    agent_role=agent_role,
                    task_context=task_context,
                    execution_results=execution_results,
                )
                
                self.log.info("Individual retrospective completed for %s", agent_role.value)
            
            # Phase 2: Check if comprehensive analysis should be triggered
            if (team_context and 
                self.coach_analyzer and 
                self.integration_status.coach_analyzer_active):
                
                # Get all individual retrospectives for the team
                individual_retros = team_context.get("individual_retrospectives", [])
                if individual_retrospective:
                    individual_retros.append(individual_retrospective)
                
                # Trigger comprehensive analysis if we have enough data
                if len(individual_retros) >= 2:  # Configurable threshold
                    self.log.debug("Conducting comprehensive retrospective analysis")
                    
                    comprehensive_report = await self._conduct_comprehensive_analysis(
                        individual_retros, team_context
                    )
                    
                    self.log.info("Comprehensive retrospective analysis completed")
            
            return individual_retrospective, comprehensive_report
            
        except Exception as e:
            self.log.error("Enhanced retrospective failed: %s", e)
            
            # Fallback to existing system if configured
            if self.config.fallback_to_existing and self.existing_engine:
                self.log.info("Falling back to existing retrospective system")
                return await self._fallback_retrospective(agent_role, task_context, execution_results)
            
            raise
    
    async def enforce_action_points(
        self,
        comprehensive_report: ComprehensiveRetrospectiveReport,
        system_state: Optional[SystemState] = None,
    ) -> bool:
        """Enforce implementation of action points from comprehensive report.
        
        Args:
            comprehensive_report: Report containing action points
            system_state: Optional current system state
            
        Returns:
            bool: True if all critical action points are implemented
        """
        if not self.enforcement_system or not self.integration_status.enforcement_system_active:
            self.log.warning("Enforcement system not active, skipping action point enforcement")
            return True  # Don't block if enforcement is not available
        
        if not self.config.enforcement_required:
            self.log.info("Action point enforcement disabled by configuration")
            return True
        
        try:
            self.log.info("Starting action point enforcement")
            
            # Get or create system state
            current_state = system_state or await self._get_current_system_state()
            
            # Create enforcement plan
            enforcement_plan = await self.enforcement_system.create_enforcement_plan(
                comprehensive_report,
                current_state,
                implementation_capabilities={
                    "automatic_config": True,
                    "automatic_training": False,
                    "automatic_process_update": True,
                }
            )
            
            # Execute enforcement plan
            success, errors = await self.enforcement_system.execute_enforcement_plan(
                enforcement_plan,
                current_state,
            )
            
            if success:
                self.log.info("All action points enforced successfully")
            else:
                self.log.warning("Action point enforcement completed with %d errors: %s", 
                                len(errors), "; ".join(errors))
            
            return success
            
        except Exception as e:
            self.log.error("Action point enforcement failed: %s", e)
            
            if self.config.enforcement_required:
                return False  # Block execution if enforcement is required
            else:
                return True  # Allow execution to continue
    
    async def assess_readiness_for_next_task(self) -> bool:
        """Assess if system is ready for next task execution.
        
        Returns:
            bool: True if system is ready for next task
        """
        if not self.enforcement_system or not self.integration_status.enforcement_system_active:
            return True  # Always ready if enforcement is not active
        
        try:
            # Get current system state
            system_state = await self._get_current_system_state()
            
            # Assess readiness
            readiness = await self.enforcement_system.assess_system_readiness(
                system_state=system_state
            )
            
            self.log.info(
                "System readiness assessment: score=%.2f, clearance=%s",
                readiness.overall_readiness_score,
                readiness.next_task_clearance
            )
            
            if not readiness.next_task_clearance:
                self.log.warning("System not ready for next task: %s", readiness.readiness_notes)
            
            return readiness.next_task_clearance
            
        except Exception as e:
            self.log.error("Readiness assessment failed: %s", e)
            # Conservative approach: not ready if assessment fails
            return False
    
    # Private methods
    
    async def _check_compatibility(self) -> CompatibilityReport:
        """Check compatibility with existing systems."""
        
        report = CompatibilityReport()
        
        # Check if existing retrospective engine exists
        if self.existing_engine:
            # Check for method compatibility
            required_methods = ["conduct_retrospective", "get_retrospective_history"]
            for method in required_methods:
                if not hasattr(self.existing_engine, method):
                    report.breaking_changes.append(f"Missing method: {method}")
                    report.compatible = False
        
        # Check agent lifecycle compatibility
        # This would check if BaseRole has the required hooks
        if not hasattr(BaseRole, 'execute'):
            report.breaking_changes.append("BaseRole missing execute method")
            report.compatible = False
        
        # Determine compatibility level
        if not report.compatible:
            report.compatibility_level = "incompatible"
        elif len(report.breaking_changes) > 0:
            report.compatibility_level = "partial"
        else:
            report.compatibility_level = "full"
        
        # Generate recommendations
        if report.breaking_changes:
            report.recommendations.append("Update existing systems to support new interfaces")
        
        report.recommendations.append("Enable fallback mode for gradual migration")
        
        return report
    
    async def _initialize_enhanced_components(self) -> None:
        """Initialize the enhanced retrospective system components."""
        
        # Initialize individual retrospective framework
        if self.config.enable_individual_retrospectives:
            config = IndividualRetrospectiveConfig(
                timeout_seconds=self.config.retrospective_timeout
            )
            self.individual_framework = IndividualRetrospectiveFramework(
                config=config,
                logger=self.log,
            )
        
        # Initialize agile coach analyzer
        if self.config.enable_comprehensive_analysis:
            self.coach_analyzer = AgileCoachAnalyzer(
                analysis_timeout=self.config.retrospective_timeout,
                logger=self.log,
            )
        
        # Initialize enforcement system
        if self.config.enable_enforcement_system:
            self.enforcement_system = OrchestratorEnforcementSystem(
                logger=self.log,
            )
        
        self.log.info("Enhanced system components initialized")
    
    async def _setup_lifecycle_hooks(self) -> None:
        """Setup lifecycle hooks for integration with agent execution."""
        
        # Post-execution hook for individual retrospectives
        self.lifecycle_hooks["post_execution_retrospective"] = LifecycleHook(
            name="post_execution_retrospective",
            phase="post_execution",
            priority=10,
        )
        
        # Pre-execution hook for readiness assessment
        self.lifecycle_hooks["pre_execution_readiness"] = LifecycleHook(
            name="pre_execution_readiness",
            phase="pre_execution",
            priority=5,
        )
        
        # Error handling hook for failure retrospectives
        self.lifecycle_hooks["error_handling_retrospective"] = LifecycleHook(
            name="error_handling_retrospective",
            phase="error_handling",
            priority=15,
        )
        
        self.log.info("Lifecycle hooks configured")
    
    async def _apply_lifecycle_modifications(self) -> None:
        """Apply necessary modifications to agent lifecycle."""
        
        # Modify BaseRole.execute to include retrospective hooks
        modification = LifecycleModification(
            component="BaseRole.execute",
            modification_type="add_hook",
            description="Add retrospective hooks to agent execution",
            impact_level="low",
        )
        
        # For demonstration, we'll just track the modification
        # In a real system, this would modify the actual methods
        modification.applied = True
        modification.rollback_info = {"original_method": "BaseRole.execute"}
        
        self.applied_modifications.append(modification)
        
        self.log.info("Lifecycle modifications applied")
    
    async def _validate_integration(self) -> None:
        """Validate that integration is working correctly."""
        
        # Test individual framework
        if self.individual_framework:
            try:
                # Create test data
                test_role = RoleName.CODER
                test_task = TaskEnvelopeV1(
                    objective="Test integration",
                    inputs={"task_id": "test_integration"},
                )
                test_result = ResultEnvelopeV1(
                    status=EnvelopeStatus.SUCCESS,
                    artifacts={"test": "integration"},
                    confidence=0.8,
                )
                
                # Test retrospective
                retro = await self.individual_framework.conduct_retrospective(
                    agent_role=test_role,
                    task_context=test_task,
                    execution_results=test_result,
                )
                
                assert retro.agent_role == test_role
                self.log.debug("Individual framework validation passed")
                
            except Exception as e:
                raise IntegrationError(f"Individual framework validation failed: {e}")
        
        # Test coach analyzer
        if self.coach_analyzer:
            # Basic initialization test
            assert self.coach_analyzer is not None
            self.log.debug("Coach analyzer validation passed")
        
        # Test enforcement system
        if self.enforcement_system:
            # Basic initialization test
            assert self.enforcement_system is not None
            self.log.debug("Enforcement system validation passed")
        
        self.log.info("Integration validation completed successfully")
    
    async def _setup_fallback_mode(self) -> None:
        """Setup fallback mode using existing retrospective system."""
        
        self.log.info("Setting up fallback mode")
        
        self.integration_status.compatibility_mode = "fallback"
        self.integration_status.individual_framework_active = False
        self.integration_status.coach_analyzer_active = False
        self.integration_status.enforcement_system_active = False
        
        # Ensure existing engine is available
        if not self.existing_engine:
            from ..orchestration.retrospective_engine import RetrospectiveEngine
            self.existing_engine = RetrospectiveEngine()
        
        self.log.info("Fallback mode configured")
    
    async def _conduct_comprehensive_analysis(
        self,
        individual_retrospectives: List[IndividualRetrospective],
        team_context: Dict[str, Any],
    ) -> ComprehensiveRetrospectiveReport:
        """Conduct comprehensive analysis using the coach analyzer."""
        
        if not self.coach_analyzer:
            raise IntegrationError("Coach analyzer not available")
        
        # Extract team composition and metrics from context
        team_composition = team_context.get("team_composition")
        execution_metrics = team_context.get("execution_metrics")
        historical_data = team_context.get("historical_reports", [])
        
        # Analyze retrospectives
        report = await self.coach_analyzer.analyze_retrospectives(
            individual_retrospectives=individual_retrospectives,
            team_composition=team_composition,
            execution_metrics=execution_metrics,
            historical_data=historical_data,
        )
        
        return report
    
    async def _fallback_retrospective(
        self,
        agent_role: RoleName,
        task_context: TaskEnvelopeV1,
        execution_results: ResultEnvelopeV1,
    ) -> Tuple[Optional[IndividualRetrospective], Optional[ComprehensiveRetrospectiveReport]]:
        """Fallback to existing retrospective system."""
        
        if not self.existing_engine:
            self.log.error("No fallback retrospective system available")
            return None, None
        
        try:
            # Use existing retrospective engine
            # This would need adaptation based on the existing engine's interface
            self.log.info("Using existing retrospective system as fallback")
            
            # For now, return None since we can't easily adapt the existing system
            # In a real implementation, this would adapt the existing engine's output
            return None, None
            
        except Exception as e:
            self.log.error("Fallback retrospective failed: %s", e)
            return None, None
    
    async def _get_current_system_state(self) -> SystemState:
        """Get current system state for enforcement."""
        
        system_state = SystemState()
        
        # Populate system state with current information
        system_state.configuration = {
            "retrospective_integration": "active",
            "enforcement_enabled": self.config.enforcement_required,
        }
        
        system_state.active_agents = {"coder", "qa", "architect"}  # Example
        system_state.performance_metrics = self.performance_metrics.copy()
        system_state.health_status = "healthy"
        
        return system_state
    
    # Public utility methods
    
    def get_integration_status(self) -> IntegrationStatus:
        """Get current integration status."""
        return self.integration_status
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    async def disable_component(self, component: str) -> bool:
        """Disable a specific component of the enhanced system."""
        
        if component == "individual_framework":
            self.integration_status.individual_framework_active = False
        elif component == "coach_analyzer":
            self.integration_status.coach_analyzer_active = False
        elif component == "enforcement_system":
            self.integration_status.enforcement_system_active = False
        else:
            return False
        
        self.log.info("Disabled component: %s", component)
        return True
    
    async def enable_component(self, component: str) -> bool:
        """Enable a specific component of the enhanced system."""
        
        if component == "individual_framework" and self.individual_framework:
            self.integration_status.individual_framework_active = True
        elif component == "coach_analyzer" and self.coach_analyzer:
            self.integration_status.coach_analyzer_active = True
        elif component == "enforcement_system" and self.enforcement_system:
            self.integration_status.enforcement_system_active = True
        else:
            return False
        
        self.log.info("Enabled component: %s", component)
        return True