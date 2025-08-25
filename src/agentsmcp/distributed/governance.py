"""
Governed Autonomy Framework

Provides intelligent agent autonomy with appropriate human oversight and escalation.
Balances autonomous decision-making with necessary governance controls based on:
- Risk assessment
- Cost thresholds
- Impact analysis
- Trust levels
- Compliance requirements
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)


class AutonomyLevel(Enum):
    """Levels of autonomous operation."""
    FULL_AUTONOMOUS = "full_autonomous"      # No human intervention required
    SUPERVISED = "supervised"               # Periodic human review
    GUIDED = "guided"                      # Human approval for major decisions
    RESTRICTED = "restricted"              # Human approval for all actions
    MANUAL_ONLY = "manual_only"           # No autonomous operation


class RiskLevel(Enum):
    """Risk assessment levels."""
    MINIMAL = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    CRITICAL = 0.9


class EscalationTrigger(Enum):
    """Types of triggers that cause escalation."""
    COST_THRESHOLD = "cost_threshold"
    RISK_THRESHOLD = "risk_threshold"
    QUALITY_DEGRADATION = "quality_degradation"
    SECURITY_CONCERN = "security_concern"
    COMPLIANCE_VIOLATION = "compliance_violation"
    DEADLINE_RISK = "deadline_risk"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TRUST_DEGRADATION = "trust_degradation"
    UNKNOWN_TERRITORY = "unknown_territory"


@dataclass
class GovernancePolicy:
    """Governance policy configuration."""
    autonomy_level: AutonomyLevel = AutonomyLevel.SUPERVISED
    cost_limit: float = 100.0
    risk_threshold: RiskLevel = RiskLevel.MEDIUM
    escalation_triggers: List[EscalationTrigger] = field(default_factory=list)
    human_approval_timeout: timedelta = field(default_factory=lambda: timedelta(hours=2))
    auto_fallback_after_timeout: bool = True
    compliance_checks_required: bool = True
    audit_trail_required: bool = True


@dataclass
class EscalationRequest:
    """Request for human escalation."""
    request_id: str = field(default_factory=lambda: f"esc_{uuid.uuid4().hex[:8]}")
    trigger: EscalationTrigger = EscalationTrigger.UNKNOWN_TERRITORY
    agent_id: str = ""
    task_description: str = ""
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    cost_impact: float = 0.0
    urgency: str = "medium"  # low, medium, high, critical
    context: Dict[str, Any] = field(default_factory=dict)
    requested_decision: str = ""
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EscalationResponse:
    """Response from human oversight."""
    request_id: str
    approved: bool
    decision: str = ""
    modifications: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    new_policy_guidance: Optional[Dict[str, Any]] = None
    escalation_timeout: Optional[timedelta] = None
    responded_at: datetime = field(default_factory=datetime.utcnow)


class GovernanceEngine:
    """
    Core governance engine that manages autonomous decision-making with oversight.
    
    Key features:
    - Risk-based autonomy levels
    - Smart escalation triggers
    - Learning from human decisions
    - Compliance monitoring
    - Audit trail management
    """
    
    def __init__(self, default_policy: GovernancePolicy = None):
        self.default_policy = default_policy or GovernancePolicy()
        self.agent_policies: Dict[str, GovernancePolicy] = {}
        self.escalation_history: Dict[str, List[EscalationRequest]] = {}
        self.pending_escalations: Dict[str, EscalationRequest] = {}
        self.human_response_handlers: List[Callable] = []
        self.policy_learning_enabled = True
        self.compliance_validators: Dict[str, Callable] = {}
        self.audit_trail: List[Dict[str, Any]] = []
        
        # Initialize default escalation triggers
        self._initialize_default_triggers()
        
        # Task handles for cleanup
        self._background_tasks = []
        self._started = False
    
    def _initialize_default_triggers(self):
        """Initialize default escalation triggers based on governance best practices."""
        self.default_policy.escalation_triggers = [
            EscalationTrigger.COST_THRESHOLD,
            EscalationTrigger.RISK_THRESHOLD,
            EscalationTrigger.SECURITY_CONCERN,
            EscalationTrigger.COMPLIANCE_VIOLATION
        ]
    
    def set_agent_policy(self, agent_id: str, policy: GovernancePolicy):
        """Set specific governance policy for an agent."""
        self.agent_policies[agent_id] = policy
        logger.info(f"🏛️ Governance policy set for {agent_id}: {policy.autonomy_level.value}")
    
    def get_agent_policy(self, agent_id: str) -> GovernancePolicy:
        """Get governance policy for an agent."""
        return self.agent_policies.get(agent_id, self.default_policy)
    
    async def check_autonomy_permission(self, 
                                      agent_id: str, 
                                      action: str, 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if an agent has permission to perform an action autonomously.
        
        Returns decision with escalation requirements if needed.
        """
        policy = self.get_agent_policy(agent_id)
        
        # Assess risk for the proposed action
        risk_assessment = await self._assess_action_risk(action, context, agent_id)
        
        # Check cost implications
        cost_impact = context.get("estimated_cost", 0.0)
        
        # Determine if escalation is needed
        escalation_required = False
        escalation_triggers = []
        
        # Check escalation triggers
        if cost_impact > policy.cost_limit:
            escalation_required = True
            escalation_triggers.append(EscalationTrigger.COST_THRESHOLD)
        
        if risk_assessment["overall_risk"] > policy.risk_threshold.value:
            escalation_required = True
            escalation_triggers.append(EscalationTrigger.RISK_THRESHOLD)
        
        # Check compliance requirements
        if policy.compliance_checks_required:
            compliance_issues = await self._check_compliance(action, context)
            if compliance_issues:
                escalation_required = True
                escalation_triggers.append(EscalationTrigger.COMPLIANCE_VIOLATION)
        
        # Check autonomy level
        if policy.autonomy_level in [AutonomyLevel.RESTRICTED, AutonomyLevel.MANUAL_ONLY]:
            escalation_required = True
        elif policy.autonomy_level == AutonomyLevel.GUIDED and risk_assessment["overall_risk"] > 0.3:
            escalation_required = True
        
        decision = {
            "agent_id": agent_id,
            "action": action,
            "permission_granted": not escalation_required,
            "autonomy_level": policy.autonomy_level.value,
            "risk_assessment": risk_assessment,
            "cost_impact": cost_impact,
            "escalation_required": escalation_required,
            "escalation_triggers": [t.value for t in escalation_triggers],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log the decision for audit trail
        self._log_governance_decision(decision)
        
        if escalation_required:
            escalation_request = await self._create_escalation_request(
                agent_id, action, context, escalation_triggers, risk_assessment, cost_impact
            )
            decision["escalation_request_id"] = escalation_request.request_id
        
        return decision
    
    async def _assess_action_risk(self, action: str, context: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        """Assess the risk level of a proposed action."""
        
        # Base risk assessment
        risk_factors = {
            "data_sensitivity": 0.0,
            "system_impact": 0.0,
            "cost_risk": 0.0,
            "reversibility": 0.0,
            "security_implications": 0.0,
            "compliance_risk": 0.0
        }
        
        # Analyze action characteristics
        action_lower = action.lower()
        
        # Data sensitivity risk
        if any(keyword in action_lower for keyword in ["delete", "remove", "drop", "truncate"]):
            risk_factors["data_sensitivity"] = 0.8
            risk_factors["reversibility"] = 0.9
        elif any(keyword in action_lower for keyword in ["modify", "update", "change"]):
            risk_factors["data_sensitivity"] = 0.4
            risk_factors["reversibility"] = 0.6
        
        # System impact risk
        if any(keyword in action_lower for keyword in ["deploy", "install", "configure", "restart"]):
            risk_factors["system_impact"] = 0.7
        elif any(keyword in action_lower for keyword in ["read", "analyze", "report"]):
            risk_factors["system_impact"] = 0.1
        
        # Cost risk assessment
        estimated_cost = context.get("estimated_cost", 0.0)
        if estimated_cost > 50.0:
            risk_factors["cost_risk"] = 0.8
        elif estimated_cost > 10.0:
            risk_factors["cost_risk"] = 0.5
        elif estimated_cost > 1.0:
            risk_factors["cost_risk"] = 0.2
        
        # Security implications
        if any(keyword in action_lower for keyword in ["access", "permission", "auth", "credential"]):
            risk_factors["security_implications"] = 0.6
        
        # Check for compliance-sensitive operations
        if any(keyword in action_lower for keyword in ["data", "personal", "user", "customer"]):
            risk_factors["compliance_risk"] = 0.4
        
        # Historical performance factor (from agent's past performance)
        agent_history = self.escalation_history.get(agent_id, [])
        if len(agent_history) > 0:
            recent_escalations = [e for e in agent_history 
                                if (datetime.utcnow() - e.created_at).days <= 7]
            if len(recent_escalations) > 3:
                # Recent escalation history increases risk
                for factor in risk_factors:
                    risk_factors[factor] = min(1.0, risk_factors[factor] + 0.2)
        
        # Calculate overall risk (weighted average)
        weights = {
            "data_sensitivity": 0.25,
            "system_impact": 0.20,
            "cost_risk": 0.15,
            "reversibility": 0.15,
            "security_implications": 0.15,
            "compliance_risk": 0.10
        }
        
        overall_risk = sum(risk_factors[factor] * weights[factor] 
                          for factor in risk_factors)
        
        return {
            "overall_risk": overall_risk,
            "risk_factors": risk_factors,
            "risk_level": self._categorize_risk_level(overall_risk),
            "mitigation_suggestions": self._generate_risk_mitigation(risk_factors)
        }
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize numeric risk score into risk level."""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        elif risk_score >= 0.2:
            return "low"
        else:
            return "minimal"
    
    def _generate_risk_mitigation(self, risk_factors: Dict[str, float]) -> List[str]:
        """Generate risk mitigation suggestions based on risk factors."""
        suggestions = []
        
        if risk_factors["data_sensitivity"] > 0.5:
            suggestions.append("Consider data backup before proceeding")
            suggestions.append("Implement staged rollout with rollback capability")
        
        if risk_factors["cost_risk"] > 0.5:
            suggestions.append("Set up cost monitoring and alerts")
            suggestions.append("Consider breaking into smaller cost-controlled phases")
        
        if risk_factors["security_implications"] > 0.5:
            suggestions.append("Conduct security review before implementation")
            suggestions.append("Implement least-privilege access controls")
        
        if risk_factors["compliance_risk"] > 0.5:
            suggestions.append("Verify compliance requirements are met")
            suggestions.append("Document compliance considerations")
        
        return suggestions
    
    async def _check_compliance(self, action: str, context: Dict[str, Any]) -> List[str]:
        """Check for compliance violations in the proposed action."""
        violations = []
        
        # Run registered compliance validators
        for validator_name, validator_func in self.compliance_validators.items():
            try:
                result = await validator_func(action, context)
                if result and isinstance(result, list):
                    violations.extend(result)
                elif result and isinstance(result, str):
                    violations.append(result)
            except Exception as e:
                logger.error(f"Compliance validator {validator_name} failed: {e}")
        
        # Basic compliance checks
        action_lower = action.lower()
        
        # Data protection checks
        if "personal" in action_lower or "user" in action_lower:
            if not context.get("gdpr_compliant", False):
                violations.append("GDPR compliance not verified for personal data operation")
        
        # Financial compliance
        cost = context.get("estimated_cost", 0.0)
        if cost > 1000.0 and not context.get("financial_approval", False):
            violations.append("Financial approval required for high-cost operations")
        
        return violations
    
    async def _create_escalation_request(self, 
                                       agent_id: str, 
                                       action: str, 
                                       context: Dict[str, Any],
                                       triggers: List[EscalationTrigger],
                                       risk_assessment: Dict[str, Any],
                                       cost_impact: float) -> EscalationRequest:
        """Create escalation request for human review."""
        
        # Determine urgency based on risk and context
        urgency = "medium"
        if risk_assessment["overall_risk"] > 0.8:
            urgency = "critical"
        elif risk_assessment["overall_risk"] > 0.6:
            urgency = "high"
        elif cost_impact > 500.0:
            urgency = "high"
        
        # Generate alternatives
        alternatives = await self._generate_action_alternatives(action, context, risk_assessment)
        
        escalation_request = EscalationRequest(
            trigger=triggers[0] if triggers else EscalationTrigger.UNKNOWN_TERRITORY,
            agent_id=agent_id,
            task_description=action,
            risk_assessment=risk_assessment,
            cost_impact=cost_impact,
            urgency=urgency,
            context=context,
            requested_decision=f"Approve {action} with estimated cost ${cost_impact:.2f}",
            alternatives=alternatives,
            deadline=datetime.utcnow() + timedelta(hours=2)  # Default 2 hour deadline
        )
        
        # Store pending escalation
        self.pending_escalations[escalation_request.request_id] = escalation_request
        
        # Add to agent's escalation history
        if agent_id not in self.escalation_history:
            self.escalation_history[agent_id] = []
        self.escalation_history[agent_id].append(escalation_request)
        
        # Notify human oversight system
        await self._notify_human_oversight(escalation_request)
        
        logger.info(f"🚨 Escalation created: {escalation_request.request_id} for {agent_id} ({urgency} urgency)")
        
        return escalation_request
    
    async def _generate_action_alternatives(self, 
                                          action: str, 
                                          context: Dict[str, Any],
                                          risk_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative approaches with lower risk/cost."""
        alternatives = []
        
        # Cost reduction alternatives
        estimated_cost = context.get("estimated_cost", 0.0)
        if estimated_cost > 10.0:
            alternatives.append({
                "approach": "phased_implementation",
                "description": "Break into smaller phases to reduce cost per phase",
                "cost_reduction": "60-80%",
                "risk_reduction": "40-60%",
                "trade_offs": "Longer timeline, more coordination required"
            })
        
        # Risk reduction alternatives
        if risk_assessment["overall_risk"] > 0.5:
            alternatives.append({
                "approach": "sandbox_testing",
                "description": "Test in isolated environment first",
                "cost_increase": "20-30%",
                "risk_reduction": "70-90%",
                "trade_offs": "Additional time and setup required"
            })
            
            alternatives.append({
                "approach": "manual_review",
                "description": "Human review at each critical step",
                "cost_increase": "10-15%",
                "risk_reduction": "50-70%",
                "trade_offs": "Requires human availability, slower execution"
            })
        
        # Defer to later alternative
        alternatives.append({
            "approach": "defer_execution",
            "description": "Postpone until better conditions or resources available",
            "cost_change": "0%",
            "risk_reduction": "Varies",
            "trade_offs": "Delayed benefits, potential opportunity cost"
        })
        
        return alternatives
    
    async def _notify_human_oversight(self, request: EscalationRequest):
        """Notify human oversight system of escalation request."""
        # Call registered human response handlers
        for handler in self.human_response_handlers:
            try:
                asyncio.create_task(handler(request))
            except Exception as e:
                logger.error(f"Human oversight notification failed: {e}")
        
        # Log for audit trail
        self._log_governance_event("escalation_created", {
            "request_id": request.request_id,
            "agent_id": request.agent_id,
            "trigger": request.trigger.value,
            "urgency": request.urgency,
            "cost_impact": request.cost_impact
        })
    
    async def respond_to_escalation(self, 
                                  request_id: str, 
                                  response: EscalationResponse) -> bool:
        """Process human response to escalation request."""
        if request_id not in self.pending_escalations:
            logger.warning(f"Unknown escalation request: {request_id}")
            return False
        
        escalation_request = self.pending_escalations[request_id]
        
        # Log response
        self._log_governance_event("escalation_responded", {
            "request_id": request_id,
            "approved": response.approved,
            "response_time": str(response.responded_at - escalation_request.created_at),
            "reasoning": response.reasoning
        })
        
        # Update policy if new guidance provided
        if response.new_policy_guidance:
            await self._update_policy_from_feedback(
                escalation_request.agent_id, 
                response.new_policy_guidance
            )
        
        # Remove from pending escalations
        del self.pending_escalations[request_id]
        
        logger.info(f"✅ Escalation resolved: {request_id} - {'Approved' if response.approved else 'Denied'}")
        
        return True
    
    async def _update_policy_from_feedback(self, agent_id: str, guidance: Dict[str, Any]):
        """Update governance policies based on human feedback."""
        if not self.policy_learning_enabled:
            return
        
        current_policy = self.get_agent_policy(agent_id)
        
        # Adjust cost limits based on feedback
        if "cost_limit" in guidance:
            current_policy.cost_limit = guidance["cost_limit"]
        
        # Adjust risk thresholds
        if "risk_threshold" in guidance:
            risk_level = guidance["risk_threshold"]
            if isinstance(risk_level, str):
                current_policy.risk_threshold = RiskLevel[risk_level.upper()]
        
        # Adjust autonomy level
        if "autonomy_level" in guidance:
            autonomy_level = guidance["autonomy_level"]
            if isinstance(autonomy_level, str):
                current_policy.autonomy_level = AutonomyLevel[autonomy_level.upper()]
        
        # Store updated policy
        self.set_agent_policy(agent_id, current_policy)
        
        logger.info(f"🔄 Policy updated for {agent_id} based on human feedback")
    
    def register_compliance_validator(self, name: str, validator: Callable):
        """Register a custom compliance validator function."""
        self.compliance_validators[name] = validator
        logger.info(f"📋 Compliance validator registered: {name}")
    
    def register_human_response_handler(self, handler: Callable):
        """Register a handler for human oversight notifications."""
        self.human_response_handlers.append(handler)
        logger.info("👤 Human response handler registered")
    
    def _log_governance_decision(self, decision: Dict[str, Any]):
        """Log governance decision for audit trail."""
        if self.get_agent_policy(decision["agent_id"]).audit_trail_required:
            audit_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "governance_decision",
                "data": decision
            }
            self.audit_trail.append(audit_entry)
    
    def _log_governance_event(self, event_type: str, data: Dict[str, Any]):
        """Log governance event for audit trail."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "data": data
        }
        self.audit_trail.append(audit_entry)
    
    async def _escalation_timeout_monitor(self):
        """Monitor escalation requests for timeouts."""
        while True:
            try:
                current_time = datetime.utcnow()
                timed_out_requests = []
                
                for request_id, request in self.pending_escalations.items():
                    if request.deadline and current_time > request.deadline:
                        timed_out_requests.append(request_id)
                
                # Handle timeouts
                for request_id in timed_out_requests:
                    await self._handle_escalation_timeout(request_id)
                
            except Exception as e:
                logger.error(f"Escalation timeout monitoring error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _handle_escalation_timeout(self, request_id: str):
        """Handle escalation request timeout."""
        if request_id not in self.pending_escalations:
            return
        
        request = self.pending_escalations[request_id]
        policy = self.get_agent_policy(request.agent_id)
        
        self._log_governance_event("escalation_timeout", {
            "request_id": request_id,
            "agent_id": request.agent_id,
            "auto_fallback": policy.auto_fallback_after_timeout
        })
        
        if policy.auto_fallback_after_timeout:
            # Auto-approve low-risk requests, deny high-risk ones
            if request.risk_assessment.get("overall_risk", 0.5) < 0.4:
                auto_response = EscalationResponse(
                    request_id=request_id,
                    approved=True,
                    decision="auto_approved_on_timeout",
                    reasoning="Low risk action approved automatically after timeout"
                )
            else:
                auto_response = EscalationResponse(
                    request_id=request_id,
                    approved=False,
                    decision="auto_denied_on_timeout",
                    reasoning="High risk action denied automatically after timeout"
                )
            
            await self.respond_to_escalation(request_id, auto_response)
            logger.warning(f"⏰ Escalation {request_id} auto-resolved on timeout: {auto_response.approved}")
        else:
            logger.warning(f"⏰ Escalation {request_id} timed out with no auto-fallback")
    
    async def _policy_optimization_loop(self):
        """Periodically optimize policies based on escalation patterns."""
        while True:
            try:
                if self.policy_learning_enabled:
                    await self._optimize_policies()
            except Exception as e:
                logger.error(f"Policy optimization error: {e}")
            
            await asyncio.sleep(3600)  # Optimize every hour
    
    async def _optimize_policies(self):
        """Optimize governance policies based on historical data."""
        for agent_id, escalations in self.escalation_history.items():
            if len(escalations) < 5:  # Need sufficient data
                continue
            
            recent_escalations = [e for e in escalations 
                                if (datetime.utcnow() - e.created_at).days <= 30]
            
            if len(recent_escalations) == 0:
                continue
            
            # Analyze escalation patterns
            cost_escalations = [e for e in recent_escalations 
                              if EscalationTrigger.COST_THRESHOLD in [e.trigger]]
            risk_escalations = [e for e in recent_escalations 
                              if EscalationTrigger.RISK_THRESHOLD in [e.trigger]]
            
            current_policy = self.get_agent_policy(agent_id)
            
            # Adjust cost limits if too many cost escalations
            if len(cost_escalations) > len(recent_escalations) * 0.5:
                # More than 50% of escalations are cost-related, might need higher limit
                avg_escalated_cost = sum(e.cost_impact for e in cost_escalations) / len(cost_escalations)
                if avg_escalated_cost * 0.8 > current_policy.cost_limit:
                    new_limit = min(current_policy.cost_limit * 1.5, avg_escalated_cost * 1.2)
                    current_policy.cost_limit = new_limit
                    logger.info(f"📊 Auto-adjusted cost limit for {agent_id}: ${new_limit:.2f}")
            
            # Update policy
            self.set_agent_policy(agent_id, current_policy)
    
    def get_governance_analytics(self) -> Dict[str, Any]:
        """Get analytics about governance performance."""
        total_escalations = sum(len(escalations) for escalations in self.escalation_history.values())
        pending_count = len(self.pending_escalations)
        
        # Calculate average response times from audit trail
        response_times = []
        for entry in self.audit_trail:
            if entry["type"] == "escalation_responded" and "response_time" in entry["data"]:
                try:
                    time_parts = entry["data"]["response_time"].split(":")
                    hours = int(time_parts[0])
                    response_times.append(hours)
                except:
                    continue
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Autonomy effectiveness (fewer escalations = more effective autonomy)
        active_agents = len(self.escalation_history)
        escalations_per_agent = total_escalations / active_agents if active_agents > 0 else 0
        
        return {
            "total_escalations": total_escalations,
            "pending_escalations": pending_count,
            "active_agents": active_agents,
            "escalations_per_agent": round(escalations_per_agent, 2),
            "average_response_time_hours": round(avg_response_time, 2),
            "policy_learning_enabled": self.policy_learning_enabled,
            "compliance_validators": len(self.compliance_validators),
            "human_handlers": len(self.human_response_handlers),
            "audit_trail_entries": len(self.audit_trail),
            "autonomy_effectiveness": max(0, min(100, (10 - escalations_per_agent) * 10))  # Scale 0-100
        }
    
    def export_audit_trail(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """Export audit trail for compliance reporting."""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)  # Default to last 30 days
        if not end_date:
            end_date = datetime.utcnow()
        
        filtered_trail = []
        for entry in self.audit_trail:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if start_date <= entry_time <= end_date:
                filtered_trail.append(entry)
        
        logger.info(f"📊 Audit trail exported: {len(filtered_trail)} entries from {start_date} to {end_date}")
        return filtered_trail
    
    async def start(self):
        """Start background governance tasks."""
        if self._started:
            return
            
        # Start background governance tasks
        self._background_tasks.append(
            asyncio.create_task(self._escalation_timeout_monitor())
        )
        self._background_tasks.append(
            asyncio.create_task(self._policy_optimization_loop())
        )
        self._started = True
    
    async def stop(self):
        """Stop background governance tasks."""
        if not self._started:
            return
            
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        self._started = False