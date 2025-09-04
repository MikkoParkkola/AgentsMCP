"""
Planning Phase System

Implements structured plan-before-execute workflows for agent swarms, addressing
the critical gap in AgentsMCP where agents execute immediately without proper
planning phases.
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    """Plan execution status"""
    DRAFT = "draft"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlanStep(dataclass):
    """Individual step in an execution plan"""
    step_id: str
    description: str
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    estimated_duration_minutes: int = 5
    success_criteria: List[str] = field(default_factory=list)
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Structured execution plan for complex tasks"""
    plan_id: str
    task_description: str
    objective: str
    success_criteria: List[str]
    steps: List[PlanStep]
    status: PlanStatus = PlanStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_total_duration_minutes: int = 30
    actual_duration_minutes: Optional[int] = None
    
    # Risk assessment
    risk_level: str = "medium"  # low, medium, high, critical
    identified_risks: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    # Resource requirements
    required_agents: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    
    # Quality gates
    quality_checks: List[str] = field(default_factory=list)
    rollback_strategy: Optional[str] = None


class PlanningSystem:
    """
    Structured planning system for agent swarms
    
    Features:
    - Plan-before-execute workflows
    - Risk assessment and mitigation
    - Resource requirement analysis
    - Quality gates and success criteria
    - Automatic plan approval for low-risk tasks
    - Human approval required for high-risk tasks
    """
    
    def __init__(self, 
                 auto_approve_threshold: str = "medium",
                 plans_storage_path: Optional[str] = None):
        """
        Initialize the planning system
        
        Args:
            auto_approve_threshold: Risk level threshold for automatic approval
            plans_storage_path: Path to store execution plans
        """
        self.auto_approve_threshold = auto_approve_threshold
        self.active_plans: Dict[str, ExecutionPlan] = {}
        self.completed_plans: Dict[str, ExecutionPlan] = {}
        
        # Storage configuration
        self.plans_storage_path = Path(plans_storage_path) if plans_storage_path else None
        if self.plans_storage_path:
            self.plans_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Planning templates for common task patterns
        self.task_templates = self._initialize_task_templates()
        
        logger.info(f"Planning system initialized with auto-approve threshold: {auto_approve_threshold}")
    
    def _initialize_task_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize templates for common task patterns"""
        return {
            "code_development": {
                "steps": [
                    "Analyze requirements and constraints",
                    "Design architecture and interfaces", 
                    "Implement core functionality",
                    "Write comprehensive tests",
                    "Perform code review and security scan",
                    "Deploy and verify functionality"
                ],
                "required_tools": ["git", "test_framework", "code_analysis"],
                "quality_checks": ["unit_tests_pass", "integration_tests_pass", "security_scan_clean"],
                "risk_factors": ["breaking_changes", "security_implications", "performance_impact"]
            },
            "data_analysis": {
                "steps": [
                    "Define analysis objectives and success metrics",
                    "Collect and validate data sources",
                    "Perform exploratory data analysis",
                    "Apply analytical methods and models",
                    "Validate results and statistical significance",
                    "Generate insights and recommendations"
                ],
                "required_tools": ["data_access", "statistical_analysis", "visualization"],
                "quality_checks": ["data_quality_validated", "statistical_significance_confirmed"],
                "risk_factors": ["data_privacy", "analytical_bias", "result_interpretation"]
            },
            "system_integration": {
                "steps": [
                    "Map system dependencies and interfaces",
                    "Design integration architecture",
                    "Implement integration points",
                    "Test integration scenarios",
                    "Monitor system health and performance",
                    "Document integration patterns"
                ],
                "required_tools": ["system_monitoring", "integration_testing", "documentation"],
                "quality_checks": ["integration_tests_pass", "performance_benchmarks_met"],
                "risk_factors": ["system_downtime", "data_corruption", "performance_degradation"]
            }
        }
    
    async def create_plan(self, 
                         task_description: str,
                         objective: str,
                         task_type: Optional[str] = None,
                         context: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """Create a structured execution plan for a task"""
        plan_id = f"plan_{int(time.time())}_{hash(task_description) % 10000:04d}"
        
        # Analyze task to determine appropriate template and approach
        plan_template = self._select_plan_template(task_description, task_type, context)
        
        # Generate plan steps
        steps = await self._generate_plan_steps(task_description, objective, plan_template, context)
        
        # Assess risk level
        risk_assessment = self._assess_risk_level(task_description, steps, context)
        
        # Estimate duration
        total_duration = sum(step.estimated_duration_minutes for step in steps)
        
        # Create execution plan
        plan = ExecutionPlan(
            plan_id=plan_id,
            task_description=task_description,
            objective=objective,
            success_criteria=self._generate_success_criteria(objective, plan_template),
            steps=steps,
            estimated_total_duration_minutes=total_duration,
            risk_level=risk_assessment['level'],
            identified_risks=risk_assessment['risks'],
            mitigation_strategies=risk_assessment['mitigations'],
            required_agents=self._identify_required_agents(steps),
            required_tools=plan_template.get('required_tools', []),
            quality_checks=plan_template.get('quality_checks', []),
            rollback_strategy=self._generate_rollback_strategy(task_description, risk_assessment)
        )
        
        # Store plan
        self.active_plans[plan_id] = plan
        await self._save_plan(plan)
        
        logger.info(f"Created execution plan {plan_id} with {len(steps)} steps, "
                   f"risk level: {risk_assessment['level']}")
        
        return plan
    
    def _select_plan_template(self, 
                             task_description: str, 
                             task_type: Optional[str],
                             context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Select appropriate plan template based on task characteristics"""
        task_lower = task_description.lower()
        
        # Explicit task type mapping
        if task_type and task_type in self.task_templates:
            return self.task_templates[task_type]
        
        # Keyword-based template selection
        if any(keyword in task_lower for keyword in 
               ['code', 'implement', 'develop', 'program', 'function', 'class']):
            return self.task_templates['code_development']
        
        if any(keyword in task_lower for keyword in 
               ['analyze', 'data', 'statistics', 'report', 'insights', 'metrics']):
            return self.task_templates['data_analysis']
        
        if any(keyword in task_lower for keyword in 
               ['integrate', 'connect', 'system', 'api', 'service', 'deploy']):
            return self.task_templates['system_integration']
        
        # Default template for general tasks
        return {
            "steps": ["Analyze task requirements", "Plan implementation approach", 
                     "Execute implementation", "Verify results and quality"],
            "required_tools": [],
            "quality_checks": ["requirements_met", "quality_verified"],
            "risk_factors": ["complexity", "dependencies"]
        }
    
    async def _generate_plan_steps(self, 
                                  task_description: str,
                                  objective: str, 
                                  template: Dict[str, Any],
                                  context: Optional[Dict[str, Any]]) -> List[PlanStep]:
        """Generate detailed plan steps based on template and task specifics"""
        steps = []
        base_steps = template.get('steps', [])
        
        for i, step_desc in enumerate(base_steps):
            step = PlanStep(
                step_id=f"step_{i+1:02d}",
                description=step_desc,
                estimated_duration_minutes=self._estimate_step_duration(step_desc, context),
                success_criteria=self._generate_step_success_criteria(step_desc, objective),
                dependencies=[f"step_{i:02d}"] if i > 0 else []
            )
            steps.append(step)
        
        return steps
    
    def _estimate_step_duration(self, step_description: str, context: Optional[Dict[str, Any]]) -> int:
        """Estimate duration for a plan step based on complexity indicators"""
        base_duration = 10  # minutes
        
        complexity_keywords = {
            'analyze': 15, 'design': 20, 'implement': 30, 'test': 15, 
            'review': 10, 'deploy': 10, 'document': 10, 'research': 25
        }
        
        step_lower = step_description.lower()
        for keyword, duration in complexity_keywords.items():
            if keyword in step_lower:
                base_duration = duration
                break
        
        # Adjust based on context complexity
        if context and context.get('complexity') == 'high':
            base_duration = int(base_duration * 1.5)
        elif context and context.get('complexity') == 'low':
            base_duration = int(base_duration * 0.7)
        
        return max(5, base_duration)  # Minimum 5 minutes
    
    def _generate_step_success_criteria(self, step_description: str, objective: str) -> List[str]:
        """Generate success criteria for individual steps"""
        step_lower = step_description.lower()
        
        if 'analyze' in step_lower:
            return ["Requirements clearly identified", "Constraints documented", "Approach validated"]
        elif 'design' in step_lower:
            return ["Architecture documented", "Interfaces defined", "Design reviewed"]
        elif 'implement' in step_lower:
            return ["Functionality implemented", "Code follows standards", "Basic testing completed"]
        elif 'test' in step_lower:
            return ["All tests pass", "Coverage requirements met", "Edge cases validated"]
        elif 'review' in step_lower:
            return ["Code reviewed", "Security considerations addressed", "Quality standards met"]
        elif 'deploy' in step_lower:
            return ["Deployment successful", "System functioning", "Health checks pass"]
        else:
            return ["Step objectives completed", "Quality criteria met"]
    
    def _generate_success_criteria(self, objective: str, template: Dict[str, Any]) -> List[str]:
        """Generate overall success criteria for the plan"""
        criteria = [
            "All plan steps completed successfully",
            "Objective achieved as specified",
            "Quality checks passed"
        ]
        
        # Add template-specific criteria
        if template.get('quality_checks'):
            criteria.extend([f"{check} verified" for check in template['quality_checks']])
        
        return criteria
    
    def _assess_risk_level(self, 
                          task_description: str, 
                          steps: List[PlanStep], 
                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risk level and identify potential risks"""
        risks = []
        mitigations = []
        risk_score = 0
        
        task_lower = task_description.lower()
        
        # Risk factor analysis
        high_risk_keywords = ['deploy', 'production', 'database', 'security', 'critical']
        medium_risk_keywords = ['integrate', 'api', 'system', 'user', 'data']
        
        for keyword in high_risk_keywords:
            if keyword in task_lower:
                risk_score += 3
                risks.append(f"High-risk operation: {keyword}")
                mitigations.append(f"Implement careful {keyword} procedures with rollback plan")
        
        for keyword in medium_risk_keywords:
            if keyword in task_lower:
                risk_score += 1
                risks.append(f"Moderate risk: {keyword} operations")
                mitigations.append(f"Test {keyword} operations thoroughly before deployment")
        
        # Context-based risk assessment
        if context:
            if context.get('affects_production', False):
                risk_score += 4
                risks.append("Production system impact")
                mitigations.append("Use staging environment for testing first")
            
            if context.get('user_facing', False):
                risk_score += 2
                risks.append("User experience impact")
                mitigations.append("Plan user communication and fallback options")
        
        # Complexity-based risk
        if len(steps) > 8:
            risk_score += 2
            risks.append("High complexity with many steps")
            mitigations.append("Break down into smaller phases with checkpoints")
        
        # Determine risk level
        if risk_score >= 8:
            level = "critical"
        elif risk_score >= 5:
            level = "high"
        elif risk_score >= 2:
            level = "medium"
        else:
            level = "low"
        
        return {
            'level': level,
            'score': risk_score,
            'risks': risks,
            'mitigations': mitigations
        }
    
    def _identify_required_agents(self, steps: List[PlanStep]) -> List[str]:
        """Identify agent types required for plan execution"""
        agents = set()
        
        for step in steps:
            step_lower = step.description.lower()
            
            if any(keyword in step_lower for keyword in ['code', 'implement', 'develop']):
                agents.add('software-engineer')
            if any(keyword in step_lower for keyword in ['test', 'qa', 'quality']):
                agents.add('qa-engineer')
            if any(keyword in step_lower for keyword in ['design', 'architecture']):
                agents.add('software-architect')
            if any(keyword in step_lower for keyword in ['security', 'scan', 'vulnerability']):
                agents.add('security-engineer')
            if any(keyword in step_lower for keyword in ['deploy', 'infrastructure']):
                agents.add('devops-engineer')
            if any(keyword in step_lower for keyword in ['analyze', 'data', 'metrics']):
                agents.add('data-analyst')
            if any(keyword in step_lower for keyword in ['document', 'specification']):
                agents.add('technical-writer')
        
        return list(agents)
    
    def _generate_rollback_strategy(self, 
                                  task_description: str, 
                                  risk_assessment: Dict[str, Any]) -> Optional[str]:
        """Generate rollback strategy based on task and risk level"""
        if risk_assessment['level'] in ['low']:
            return None
        
        task_lower = task_description.lower()
        
        if 'deploy' in task_lower:
            return "Maintain previous deployment artifacts and automated rollback procedures"
        elif 'database' in task_lower:
            return "Create database backup before changes and test restore procedures"
        elif 'integrate' in task_lower:
            return "Maintain fallback to previous integration version with feature flags"
        else:
            return "Document current state and maintain version control for easy reversion"
    
    async def approve_plan(self, plan_id: str, approver: str = "auto") -> bool:
        """Approve a plan for execution"""
        if plan_id not in self.active_plans:
            logger.error(f"Plan {plan_id} not found")
            return False
        
        plan = self.active_plans[plan_id]
        
        # Check if plan can be auto-approved based on risk level
        risk_levels = ['low', 'medium', 'high', 'critical']
        auto_threshold_index = risk_levels.index(self.auto_approve_threshold)
        plan_risk_index = risk_levels.index(plan.risk_level)
        
        if approver == "auto" and plan_risk_index > auto_threshold_index:
            logger.warning(f"Plan {plan_id} requires manual approval due to {plan.risk_level} risk level")
            return False
        
        plan.status = PlanStatus.APPROVED
        plan.approved_at = datetime.now()
        
        await self._save_plan(plan)
        
        logger.info(f"Plan {plan_id} approved by {approver}")
        return True
    
    async def execute_plan(self, plan_id: str, executor_callback: Optional[callable] = None) -> bool:
        """Execute an approved plan"""
        if plan_id not in self.active_plans:
            logger.error(f"Plan {plan_id} not found")
            return False
        
        plan = self.active_plans[plan_id]
        
        if plan.status != PlanStatus.APPROVED:
            logger.error(f"Plan {plan_id} is not approved for execution")
            return False
        
        plan.status = PlanStatus.EXECUTING
        plan.started_at = datetime.now()
        
        try:
            for step in plan.steps:
                if step.dependencies:
                    # Check if dependencies are completed
                    for dep_id in step.dependencies:
                        dep_step = next((s for s in plan.steps if s.step_id == dep_id), None)
                        if not dep_step or dep_step.status != "completed":
                            logger.error(f"Step {step.step_id} dependency {dep_id} not completed")
                            raise Exception(f"Dependency {dep_id} not completed")
                
                # Execute step
                step.status = "executing"
                step.started_at = datetime.now()
                
                logger.info(f"Executing plan step: {step.description}")
                
                if executor_callback:
                    success, result = await executor_callback(step, plan)
                    if success:
                        step.status = "completed"
                        step.result = result
                        step.completed_at = datetime.now()
                    else:
                        step.status = "failed"
                        step.error = result
                        raise Exception(f"Step failed: {result}")
                else:
                    # Simulate execution for testing
                    await asyncio.sleep(1)
                    step.status = "completed"
                    step.result = f"Step {step.step_id} completed successfully"
                    step.completed_at = datetime.now()
                
                await self._save_plan(plan)
            
            # Plan completed successfully
            plan.status = PlanStatus.COMPLETED
            plan.completed_at = datetime.now()
            
            if plan.started_at:
                duration = (plan.completed_at - plan.started_at).total_seconds() / 60
                plan.actual_duration_minutes = int(duration)
            
            # Move to completed plans
            self.completed_plans[plan_id] = plan
            del self.active_plans[plan_id]
            
            await self._save_plan(plan)
            
            logger.info(f"Plan {plan_id} completed successfully in {plan.actual_duration_minutes} minutes")
            return True
            
        except Exception as e:
            plan.status = PlanStatus.FAILED
            plan.completed_at = datetime.now()
            
            logger.error(f"Plan {plan_id} execution failed: {e}")
            await self._save_plan(plan)
            return False
    
    async def _save_plan(self, plan: ExecutionPlan):
        """Save plan to persistent storage"""
        if not self.plans_storage_path:
            return
        
        try:
            plan_file = self.plans_storage_path / f"{plan.plan_id}.json"
            
            # Convert plan to JSON-serializable format
            plan_data = {
                'plan_id': plan.plan_id,
                'task_description': plan.task_description,
                'objective': plan.objective,
                'success_criteria': plan.success_criteria,
                'status': plan.status.value,
                'risk_level': plan.risk_level,
                'identified_risks': plan.identified_risks,
                'mitigation_strategies': plan.mitigation_strategies,
                'required_agents': plan.required_agents,
                'required_tools': plan.required_tools,
                'quality_checks': plan.quality_checks,
                'rollback_strategy': plan.rollback_strategy,
                'created_at': plan.created_at.isoformat(),
                'approved_at': plan.approved_at.isoformat() if plan.approved_at else None,
                'started_at': plan.started_at.isoformat() if plan.started_at else None,
                'completed_at': plan.completed_at.isoformat() if plan.completed_at else None,
                'estimated_total_duration_minutes': plan.estimated_total_duration_minutes,
                'actual_duration_minutes': plan.actual_duration_minutes,
                'steps': [
                    {
                        'step_id': step.step_id,
                        'description': step.description,
                        'assigned_agent': step.assigned_agent,
                        'dependencies': step.dependencies,
                        'estimated_duration_minutes': step.estimated_duration_minutes,
                        'success_criteria': step.success_criteria,
                        'status': step.status,
                        'started_at': step.started_at.isoformat() if step.started_at else None,
                        'completed_at': step.completed_at.isoformat() if step.completed_at else None,
                        'result': step.result,
                        'error': step.error
                    }
                    for step in plan.steps
                ]
            }
            
            with open(plan_file, 'w') as f:
                json.dump(plan_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save plan {plan.plan_id}: {e}")
    
    def get_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        """Get a plan by ID"""
        return self.active_plans.get(plan_id) or self.completed_plans.get(plan_id)
    
    def list_active_plans(self) -> List[ExecutionPlan]:
        """Get list of active plans"""
        return list(self.active_plans.values())
    
    def list_completed_plans(self) -> List[ExecutionPlan]:
        """Get list of completed plans"""
        return list(self.completed_plans.values())
    
    def get_plan_summary(self) -> Dict[str, Any]:
        """Get summary of all plans"""
        return {
            'active_plans': len(self.active_plans),
            'completed_plans': len(self.completed_plans),
            'plans_by_status': {
                status.value: len([p for p in self.active_plans.values() if p.status == status])
                for status in PlanStatus
            },
            'plans_by_risk_level': {
                'low': len([p for p in list(self.active_plans.values()) + list(self.completed_plans.values()) 
                           if p.risk_level == 'low']),
                'medium': len([p for p in list(self.active_plans.values()) + list(self.completed_plans.values()) 
                              if p.risk_level == 'medium']),
                'high': len([p for p in list(self.active_plans.values()) + list(self.completed_plans.values()) 
                            if p.risk_level == 'high']),
                'critical': len([p for p in list(self.active_plans.values()) + list(self.completed_plans.values()) 
                                if p.risk_level == 'critical'])
            }
        }