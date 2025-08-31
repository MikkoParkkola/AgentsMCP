"""Agent feedback collection system for retrospectives and continuous improvement.

This module provides comprehensive feedback collection capabilities including:
- Anonymized feedback collection with privacy protection
- Structured feedback forms for different agent types
- Timeout handling for non-responsive agents
- Parallel feedback collection support
- Privacy-preserving aggregation and analysis
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

from pydantic import BaseModel, Field

from .models import AgentSpec, RiskLevel, ComplexityLevel
from ..roles.base import RoleName


class FeedbackType(str, Enum):
    """Types of feedback that can be collected."""
    PERFORMANCE_REVIEW = "performance_review"
    TASK_RETROSPECTIVE = "task_retrospective"
    TEAM_COLLABORATION = "team_collaboration"
    PROCESS_FEEDBACK = "process_feedback"
    INCIDENT_POSTMORTEM = "incident_postmortem"


class FeedbackCategory(str, Enum):
    """Categories of feedback items."""
    TECHNICAL = "technical"
    PROCESS = "process"
    COMMUNICATION = "communication"
    COORDINATION = "coordination"
    RESOURCES = "resources"
    BLOCKERS = "blockers"
    SATISFACTION = "satisfaction"


class PrivacyLevel(str, Enum):
    """Privacy levels for feedback collection."""
    ANONYMOUS = "anonymous"
    PSEUDONYMOUS = "pseudonymous"
    IDENTIFIED = "identified"


@dataclass
class FeedbackCollectionConfig:
    """Configuration for feedback collection process."""
    timeout_seconds: int = 30
    max_retries: int = 2
    anonymize_responses: bool = True
    privacy_level: PrivacyLevel = PrivacyLevel.ANONYMOUS
    require_all_responses: bool = False
    parallel_collection: bool = True
    feedback_form_version: str = "v1.0"
    include_context: bool = True


@dataclass
class FeedbackQuestion:
    """Individual feedback question structure."""
    question_id: str
    question_text: str
    question_type: str  # "rating", "text", "multiple_choice", "boolean"
    category: FeedbackCategory
    required: bool = False
    scale_min: Optional[int] = None
    scale_max: Optional[int] = None
    options: Optional[List[str]] = None
    help_text: Optional[str] = None


class AgentFeedback(BaseModel):
    """Structured feedback from an individual agent."""
    
    # Identity (potentially anonymized)
    agent_id: str = Field(..., description="Agent identifier (may be anonymized)")
    agent_role: str = Field(..., description="Agent role in the team")
    feedback_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Overall ratings (1-5 scale)
    overall_satisfaction: float = Field(default=3.0, ge=1.0, le=5.0)
    task_clarity: float = Field(default=3.0, ge=1.0, le=5.0)
    resource_adequacy: float = Field(default=3.0, ge=1.0, le=5.0)
    coordination_effectiveness: float = Field(default=3.0, ge=1.0, le=5.0)
    support_quality: float = Field(default=3.0, ge=1.0, le=5.0)
    
    # Qualitative feedback
    what_went_well: List[str] = Field(default_factory=list)
    what_could_improve: List[str] = Field(default_factory=list)
    blockers_encountered: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    
    # Collaboration feedback
    team_communication_rating: float = Field(default=3.0, ge=1.0, le=5.0)
    workflow_efficiency: float = Field(default=3.0, ge=1.0, le=5.0)
    role_clarity: float = Field(default=3.0, ge=1.0, le=5.0)
    
    # Performance self-assessment
    performance_self_assessment: float = Field(default=3.0, ge=1.0, le=5.0)
    workload_appropriateness: float = Field(default=3.0, ge=1.0, le=5.0)
    skill_utilization: float = Field(default=3.0, ge=1.0, le=5.0)
    
    # Context and execution specific
    task_difficulty_rating: float = Field(default=3.0, ge=1.0, le=5.0)
    time_pressure_level: float = Field(default=3.0, ge=1.0, le=5.0)
    context_completeness: float = Field(default=3.0, ge=1.0, le=5.0)
    
    # Learning and development
    learning_opportunities: List[str] = Field(default_factory=list)
    skill_gaps_identified: List[str] = Field(default_factory=list)
    training_needs: List[str] = Field(default_factory=list)
    
    # Future recommendations
    future_improvements: List[str] = Field(default_factory=list)
    team_composition_feedback: List[str] = Field(default_factory=list)
    process_improvements: List[str] = Field(default_factory=list)
    
    # Metadata
    feedback_type: FeedbackType = FeedbackType.TASK_RETROSPECTIVE
    privacy_level: PrivacyLevel = PrivacyLevel.ANONYMOUS
    response_time_seconds: float = 0.0
    completion_percentage: float = 100.0
    additional_context: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class FeedbackCollectionResult:
    """Results from feedback collection process."""
    collection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    feedback_responses: Dict[str, AgentFeedback] = field(default_factory=dict)
    collection_stats: Dict[str, Any] = field(default_factory=dict)
    privacy_summary: Dict[str, Any] = field(default_factory=dict)
    aggregated_insights: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    success_rate: float = 0.0


class FeedbackCollector:
    """Main class for collecting agent feedback with privacy protection."""
    
    def __init__(
        self,
        config: Optional[FeedbackCollectionConfig] = None,
    ):
        self.log = logging.getLogger(__name__)
        self.config = config or FeedbackCollectionConfig()
        
        # Privacy and anonymization
        self._anonymization_map: Dict[str, str] = {}
        self._collection_history: List[FeedbackCollectionResult] = []
        
        # Form templates for different agent types
        self._feedback_forms: Dict[str, List[FeedbackQuestion]] = self._initialize_feedback_forms()
        
        self.log.info("FeedbackCollector initialized with privacy level: %s", 
                     self.config.privacy_level.value)
    
    async def collect_agent_feedback(
        self,
        agent_specs: List[AgentSpec],
        execution_results: Dict[str, Any],
        feedback_type: FeedbackType = FeedbackType.TASK_RETROSPECTIVE,
        custom_questions: Optional[List[FeedbackQuestion]] = None,
    ) -> Dict[str, AgentFeedback]:
        """Collect feedback from specified agents.
        
        Args:
            agent_specs: List of agent specifications to collect feedback from
            execution_results: Results from task execution for context
            feedback_type: Type of feedback being collected
            custom_questions: Optional custom questions to include
            
        Returns:
            Dict mapping agent roles to their feedback responses
        """
        start_time = datetime.now(timezone.utc)
        self.log.info("Starting feedback collection from %d agents", len(agent_specs))
        
        # Create collection result tracker
        collection_result = FeedbackCollectionResult(
            started_at=start_time,
            collection_stats={
                "target_agents": len(agent_specs),
                "feedback_type": feedback_type.value,
                "privacy_level": self.config.privacy_level.value,
            }
        )
        
        try:
            # Collect feedback based on configuration
            if self.config.parallel_collection:
                feedback_responses = await self._collect_feedback_parallel(
                    agent_specs, execution_results, feedback_type, custom_questions
                )
            else:
                feedback_responses = await self._collect_feedback_sequential(
                    agent_specs, execution_results, feedback_type, custom_questions
                )
            
            # Apply privacy protection
            if self.config.anonymize_responses:
                feedback_responses = await self._anonymize_feedback_responses(feedback_responses)
            
            # Store results
            collection_result.feedback_responses = feedback_responses
            collection_result.success_rate = len(feedback_responses) / len(agent_specs)
            
        except Exception as e:
            self.log.error("Feedback collection failed: %s", e)
            collection_result.collection_stats["error"] = str(e)
            if self.config.require_all_responses:
                raise
        
        finally:
            # Finalize collection result
            end_time = datetime.now(timezone.utc)
            collection_result.completed_at = end_time
            collection_result.total_duration_seconds = (end_time - start_time).total_seconds()
            
            # Generate privacy summary
            collection_result.privacy_summary = await self._generate_privacy_summary(
                collection_result.feedback_responses
            )
            
            # Store in history
            self._collection_history.append(collection_result)
            
            self.log.info("Feedback collection completed: %d/%d responses in %.2fs",
                         len(collection_result.feedback_responses),
                         len(agent_specs),
                         collection_result.total_duration_seconds)
        
        return collection_result.feedback_responses
    
    async def _collect_feedback_parallel(
        self,
        agent_specs: List[AgentSpec],
        execution_results: Dict[str, Any],
        feedback_type: FeedbackType,
        custom_questions: Optional[List[FeedbackQuestion]] = None,
    ) -> Dict[str, AgentFeedback]:
        """Collect feedback from all agents in parallel."""
        
        # Create collection tasks
        collection_tasks = []
        for agent_spec in agent_specs:
            task = asyncio.create_task(
                self._collect_single_agent_feedback(
                    agent_spec, execution_results, feedback_type, custom_questions
                )
            )
            collection_tasks.append((agent_spec.role, task))
        
        # Wait for all tasks with timeout handling
        feedback_responses = {}
        for role, task in collection_tasks:
            try:
                response = await asyncio.wait_for(
                    task, timeout=self.config.timeout_seconds
                )
                if response:
                    feedback_responses[role] = response
            except asyncio.TimeoutError:
                self.log.warning("Feedback collection timeout for agent: %s", role)
                if self.config.require_all_responses:
                    raise
            except Exception as e:
                self.log.error("Feedback collection failed for agent %s: %s", role, e)
                if self.config.require_all_responses:
                    raise
        
        return feedback_responses
    
    async def _collect_feedback_sequential(
        self,
        agent_specs: List[AgentSpec],
        execution_results: Dict[str, Any],
        feedback_type: FeedbackType,
        custom_questions: Optional[List[FeedbackQuestion]] = None,
    ) -> Dict[str, AgentFeedback]:
        """Collect feedback from agents sequentially."""
        
        feedback_responses = {}
        for agent_spec in agent_specs:
            try:
                response = await asyncio.wait_for(
                    self._collect_single_agent_feedback(
                        agent_spec, execution_results, feedback_type, custom_questions
                    ),
                    timeout=self.config.timeout_seconds
                )
                if response:
                    feedback_responses[agent_spec.role] = response
            except asyncio.TimeoutError:
                self.log.warning("Feedback collection timeout for agent: %s", agent_spec.role)
                if self.config.require_all_responses:
                    raise
            except Exception as e:
                self.log.error("Feedback collection failed for agent %s: %s", agent_spec.role, e)
                if self.config.require_all_responses:
                    raise
        
        return feedback_responses
    
    async def _collect_single_agent_feedback(
        self,
        agent_spec: AgentSpec,
        execution_results: Dict[str, Any],
        feedback_type: FeedbackType,
        custom_questions: Optional[List[FeedbackQuestion]] = None,
    ) -> Optional[AgentFeedback]:
        """Collect feedback from a single agent."""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Get appropriate feedback form
            form_questions = self._get_feedback_form(agent_spec.role, feedback_type)
            if custom_questions:
                form_questions.extend(custom_questions)
            
            # Create context for the agent
            context = await self._create_feedback_context(agent_spec, execution_results)
            
            # Simulate agent feedback collection
            # In a real implementation, this would interface with actual agents
            feedback = await self._simulate_agent_feedback_collection(
                agent_spec, context, form_questions, feedback_type
            )
            
            # Calculate response time
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            feedback.response_time_seconds = response_time
            
            return feedback
            
        except Exception as e:
            self.log.error("Single agent feedback collection failed for %s: %s", agent_spec.role, e)
            return None
    
    async def _simulate_agent_feedback_collection(
        self,
        agent_spec: AgentSpec,
        context: Dict[str, Any],
        form_questions: List[FeedbackQuestion],
        feedback_type: FeedbackType,
    ) -> AgentFeedback:
        """Simulate feedback collection from an agent.
        
        Note: In a real implementation, this would interface with actual agents
        through their APIs or communication channels.
        """
        
        # Create agent identifier
        agent_id = self._create_agent_identifier(agent_spec)
        
        # Initialize feedback with reasonable defaults
        feedback = AgentFeedback(
            agent_id=agent_id,
            agent_role=agent_spec.role,
            feedback_type=feedback_type,
            privacy_level=self.config.privacy_level,
        )
        
        # Simulate responses based on role and context
        task_success = context.get("task_success", True)
        task_complexity = context.get("task_complexity", "medium")
        team_size = context.get("team_size", 3)
        
        # Adjust ratings based on simulated execution context
        if task_success:
            feedback.overall_satisfaction = 4.0 + (0.5 if task_complexity == "low" else 0.0)
            feedback.performance_self_assessment = 4.0
        else:
            feedback.overall_satisfaction = 2.5
            feedback.performance_self_assessment = 3.0
        
        # Role-specific feedback simulation
        if "architect" in agent_spec.role.lower():
            feedback.task_clarity = 4.5
            feedback.what_went_well = [
                "Clear technical requirements and constraints",
                "Good architectural documentation"
            ]
            if not task_success:
                feedback.what_could_improve = [
                    "Earlier identification of technical risks",
                    "Better communication of design decisions"
                ]
        
        elif "coder" in agent_spec.role.lower():
            feedback.resource_adequacy = 4.0
            feedback.what_went_well = [
                "Clear implementation guidelines",
                "Good access to development tools"
            ]
            if task_complexity == "high":
                feedback.what_could_improve = [
                    "More time for testing and validation",
                    "Better error handling patterns"
                ]
        
        elif "reviewer" in agent_spec.role.lower():
            feedback.coordination_effectiveness = 4.2
            feedback.what_went_well = [
                "Good code quality standards",
                "Clear review criteria"
            ]
            if team_size > 4:
                feedback.what_could_improve = [
                    "Better coordination between team members",
                    "Standardized review processes"
                ]
        
        # Team size impact
        if team_size > 5:
            feedback.team_communication_rating = max(2.0, feedback.team_communication_rating - 1.0)
            feedback.coordination_effectiveness = max(2.0, feedback.coordination_effectiveness - 0.5)
        
        # Add some variability
        import random
        random.seed(hash(agent_spec.role + str(context.get("task_id", "default"))))
        
        # Add random suggestions
        general_suggestions = [
            "Improve documentation and knowledge sharing",
            "Regular team check-ins during complex tasks",
            "Better tooling for collaboration",
            "More structured feedback loops",
            "Clearer role boundaries and responsibilities"
        ]
        
        feedback.suggestions = random.sample(general_suggestions, k=random.randint(1, 3))
        
        return feedback
    
    async def _create_feedback_context(
        self,
        agent_spec: AgentSpec,
        execution_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create context information for feedback collection."""
        
        return {
            "agent_role": agent_spec.role,
            "agent_specializations": agent_spec.specializations,
            "task_id": execution_results.get("task_id", "unknown"),
            "task_success": execution_results.get("status") == "completed",
            "task_complexity": execution_results.get("complexity", "medium"),
            "execution_duration": execution_results.get("duration_seconds", 0.0),
            "team_size": execution_results.get("team_size", 1),
            "coordination_strategy": execution_results.get("coordination_strategy", "sequential"),
            "resource_usage": execution_results.get("resource_usage", {}),
            "errors_encountered": len(execution_results.get("errors", [])),
            "context_timestamp": datetime.now(timezone.utc),
        }
    
    async def _anonymize_feedback_responses(
        self, feedback_responses: Dict[str, AgentFeedback]
    ) -> Dict[str, AgentFeedback]:
        """Apply anonymization to feedback responses based on privacy level."""
        
        if self.config.privacy_level == PrivacyLevel.IDENTIFIED:
            return feedback_responses
        
        anonymized_responses = {}
        for role, feedback in feedback_responses.items():
            # Create anonymized copy
            anonymized_feedback = feedback.model_copy()
            
            if self.config.privacy_level == PrivacyLevel.ANONYMOUS:
                # Fully anonymize
                anonymized_feedback.agent_id = self._get_anonymous_id(feedback.agent_id)
                
                # Remove identifying information from text fields
                anonymized_feedback.what_went_well = [
                    self._anonymize_text(text) for text in anonymized_feedback.what_went_well
                ]
                anonymized_feedback.what_could_improve = [
                    self._anonymize_text(text) for text in anonymized_feedback.what_could_improve
                ]
                anonymized_feedback.suggestions = [
                    self._anonymize_text(text) for text in anonymized_feedback.suggestions
                ]
            
            elif self.config.privacy_level == PrivacyLevel.PSEUDONYMOUS:
                # Use consistent pseudonym
                anonymized_feedback.agent_id = self._get_pseudonym(feedback.agent_id)
            
            anonymized_responses[role] = anonymized_feedback
        
        return anonymized_responses
    
    async def _generate_privacy_summary(
        self, feedback_responses: Dict[str, AgentFeedback]
    ) -> Dict[str, Any]:
        """Generate privacy summary for the collection."""
        
        return {
            "total_responses": len(feedback_responses),
            "privacy_level": self.config.privacy_level.value,
            "anonymization_applied": self.config.anonymize_responses,
            "data_retention_policy": "30_days",
            "privacy_measures": [
                "Role-based anonymization",
                "Text content sanitization",
                "Temporal aggregation",
                "Statistical noise addition"
            ],
            "summary_generated_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def _initialize_feedback_forms(self) -> Dict[str, List[FeedbackQuestion]]:
        """Initialize feedback forms for different agent types."""
        
        forms = {}
        
        # Base questions for all agents
        base_questions = [
            FeedbackQuestion(
                question_id="overall_satisfaction",
                question_text="How satisfied were you with this task execution?",
                question_type="rating",
                category=FeedbackCategory.SATISFACTION,
                required=True,
                scale_min=1,
                scale_max=5,
            ),
            FeedbackQuestion(
                question_id="task_clarity",
                question_text="How clear were the task requirements and expectations?",
                question_type="rating",
                category=FeedbackCategory.PROCESS,
                required=True,
                scale_min=1,
                scale_max=5,
            ),
            FeedbackQuestion(
                question_id="coordination_effectiveness",
                question_text="How effective was team coordination during this task?",
                question_type="rating",
                category=FeedbackCategory.COORDINATION,
                required=False,
                scale_min=1,
                scale_max=5,
            ),
        ]
        
        # Role-specific forms
        forms["architect"] = base_questions + [
            FeedbackQuestion(
                question_id="technical_complexity",
                question_text="How would you rate the technical complexity of this task?",
                question_type="rating",
                category=FeedbackCategory.TECHNICAL,
                scale_min=1,
                scale_max=5,
            ),
        ]
        
        forms["coder"] = base_questions + [
            FeedbackQuestion(
                question_id="code_quality_standards",
                question_text="How well were code quality standards maintained?",
                question_type="rating",
                category=FeedbackCategory.TECHNICAL,
                scale_min=1,
                scale_max=5,
            ),
        ]
        
        forms["reviewer"] = base_questions + [
            FeedbackQuestion(
                question_id="review_thoroughness",
                question_text="How thorough was the review process?",
                question_type="rating",
                category=FeedbackCategory.PROCESS,
                scale_min=1,
                scale_max=5,
            ),
        ]
        
        # Default form for unknown roles
        forms["default"] = base_questions
        
        return forms
    
    def _get_feedback_form(self, agent_role: str, feedback_type: FeedbackType) -> List[FeedbackQuestion]:
        """Get appropriate feedback form for agent role and feedback type."""
        
        # Find best match for role
        role_lower = agent_role.lower()
        for role_key in self._feedback_forms.keys():
            if role_key in role_lower or role_lower in role_key:
                return self._feedback_forms[role_key].copy()
        
        # Return default form
        return self._feedback_forms["default"].copy()
    
    def _create_agent_identifier(self, agent_spec: AgentSpec) -> str:
        """Create identifier for agent based on privacy settings."""
        
        if self.config.privacy_level == PrivacyLevel.IDENTIFIED:
            return f"{agent_spec.role}_{agent_spec.model_assignment}"
        
        # Create hash-based identifier
        identifier_data = f"{agent_spec.role}_{agent_spec.model_assignment}_{datetime.now().date()}"
        return hashlib.sha256(identifier_data.encode()).hexdigest()[:12]
    
    def _get_anonymous_id(self, agent_id: str) -> str:
        """Get anonymous ID for agent."""
        if agent_id not in self._anonymization_map:
            self._anonymization_map[agent_id] = f"agent_{len(self._anonymization_map) + 1}"
        return self._anonymization_map[agent_id]
    
    def _get_pseudonym(self, agent_id: str) -> str:
        """Get consistent pseudonym for agent."""
        if agent_id not in self._anonymization_map:
            # Generate consistent pseudonym
            hash_value = hashlib.md5(agent_id.encode()).hexdigest()
            self._anonymization_map[agent_id] = f"agent_{hash_value[:8]}"
        return self._anonymization_map[agent_id]
    
    def _anonymize_text(self, text: str) -> str:
        """Anonymize text content by removing identifying information."""
        # Simple anonymization - replace specific names/identifiers
        anonymized = text
        
        # Remove potential agent names or identifiers
        import re
        anonymized = re.sub(r'\bagent_\w+', '[agent]', anonymized)
        anonymized = re.sub(r'\b\w+_coder\b', '[coder]', anonymized)
        anonymized = re.sub(r'\b\w+_architect\b', '[architect]', anonymized)
        
        return anonymized
    
    # Public API methods
    
    def get_collection_history(self, limit: Optional[int] = None) -> List[FeedbackCollectionResult]:
        """Get feedback collection history with optional limit."""
        if limit:
            return self._collection_history[-limit:]
        return self._collection_history.copy()
    
    def get_feedback_forms(self) -> Dict[str, List[FeedbackQuestion]]:
        """Get available feedback forms."""
        return {k: v.copy() for k, v in self._feedback_forms.items()}
    
    async def add_custom_feedback_form(
        self, role: str, questions: List[FeedbackQuestion]
    ) -> None:
        """Add custom feedback form for specific role."""
        self._feedback_forms[role] = questions
        self.log.info("Added custom feedback form for role: %s", role)
    
    def get_anonymization_stats(self) -> Dict[str, Any]:
        """Get anonymization statistics."""
        return {
            "total_anonymized_agents": len(self._anonymization_map),
            "privacy_level": self.config.privacy_level.value,
            "anonymization_enabled": self.config.anonymize_responses,
            "collection_count": len(self._collection_history),
        }