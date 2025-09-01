"""
Clarification Engine - Question generation and confidence scoring

Generates targeted clarifying questions when user intent is ambiguous or incomplete.
Uses confidence thresholds to determine when clarification is needed and provides
intelligent question ranking and follow-up strategies.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import re

from .intent_analyzer import IntentAnalysis, IntentType, TechnicalDomain

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of clarification questions."""
    CLARIFY_INTENT = "clarify_intent"
    SPECIFY_REQUIREMENTS = "specify_requirements"
    RESOLVE_AMBIGUITY = "resolve_ambiguity"
    DEFINE_SCOPE = "define_scope"
    CONFIRM_ASSUMPTIONS = "confirm_assumptions"
    PRIORITIZE_OPTIONS = "prioritize_options"
    PROVIDE_CONTEXT = "provide_context"


class QuestionPriority(Enum):
    """Priority levels for clarification questions."""
    CRITICAL = "critical"  # Must be answered before proceeding
    HIGH = "high"         # Strongly recommended to answer
    MEDIUM = "medium"     # Helpful but not essential  
    LOW = "low"          # Nice to have


@dataclass
class ClarificationQuestion:
    """A single clarification question."""
    question: str
    question_type: QuestionType
    priority: QuestionPriority
    possible_answers: List[str] = field(default_factory=list)
    suggested_answer: Optional[str] = None
    reasoning: str = ""
    follow_up_questions: List[str] = field(default_factory=list)
    confidence_impact: float = 0.0  # How much answering this improves confidence


@dataclass
class ClarificationSession:
    """A clarification session with multiple questions."""
    session_id: str
    original_input: str
    intent_analysis: IntentAnalysis
    questions: List[ClarificationQuestion]
    current_confidence: float
    target_confidence: float
    answers_received: Dict[str, str] = field(default_factory=dict)
    session_complete: bool = False
    refined_prompt: Optional[str] = None
    iteration_count: int = 0
    max_iterations: int = 3


class ClarificationEngine:
    """
    Intelligent clarification question generation and management.
    
    Generates targeted questions to resolve ambiguity and improve confidence
    in understanding user intent before task delegation.
    """
    
    def __init__(self, confidence_threshold: float = 0.9):
        """Initialize clarification engine."""
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Question templates by type and domain
        self.question_templates = self._build_question_templates()
        self.domain_specific_questions = self._build_domain_questions()
        self.ambiguity_resolvers = self._build_ambiguity_resolvers()
        
        # Active clarification sessions
        self.active_sessions: Dict[str, ClarificationSession] = {}
        
        # Statistics
        self.sessions_created = 0
        self.questions_generated = 0
        self.successful_clarifications = 0
        
        self.logger.info(f"ClarificationEngine initialized with confidence threshold: {confidence_threshold}")
    
    async def assess_clarification_need(self, intent_analysis: IntentAnalysis) -> Tuple[bool, float, List[str]]:
        """
        Assess whether clarification is needed based on intent analysis.
        
        Returns:
            - needs_clarification: bool
            - confidence_gap: float (how much clarification could improve confidence)
            - reasons: List of reasons why clarification is needed
        """
        reasons = []
        confidence_gap = 0.0
        
        # Check base confidence
        if intent_analysis.confidence < self.confidence_threshold:
            confidence_gap += (self.confidence_threshold - intent_analysis.confidence)
            reasons.append(f"Low confidence in intent classification ({intent_analysis.confidence:.2f})")
        
        # Check for ambiguous terms
        if intent_analysis.ambiguous_terms:
            confidence_gap += 0.2
            reasons.append(f"Ambiguous terms detected: {', '.join(intent_analysis.ambiguous_terms[:3])}")
        
        # Check for missing context
        if intent_analysis.missing_context:
            confidence_gap += 0.15
            reasons.append(f"Missing context: {', '.join(intent_analysis.missing_context)}")
        
        # Check for assumptions needed
        if intent_analysis.assumptions_needed:
            confidence_gap += 0.1
            reasons.append(f"Assumptions needed: {', '.join(intent_analysis.assumptions_needed)}")
        
        # Check for vague requirements
        if not intent_analysis.success_criteria and intent_analysis.primary_intent in [
            IntentType.TASK_EXECUTION, IntentType.PROBLEM_SOLVING, IntentType.CREATIVE_GENERATION
        ]:
            confidence_gap += 0.15
            reasons.append("No clear success criteria defined")
        
        # Check for incomplete specifications
        if intent_analysis.primary_intent == IntentType.TASK_EXECUTION and not intent_analysis.technologies:
            confidence_gap += 0.1
            reasons.append("No specific technologies or tools mentioned")
        
        needs_clarification = confidence_gap > 0.1 or len(reasons) >= 2
        
        self.logger.debug(f"Clarification assessment: needed={needs_clarification}, gap={confidence_gap:.2f}, reasons={len(reasons)}")
        
        return needs_clarification, min(confidence_gap, 0.5), reasons
    
    async def create_clarification_session(self, 
                                         user_input: str, 
                                         intent_analysis: IntentAnalysis,
                                         session_id: Optional[str] = None) -> ClarificationSession:
        """Create a new clarification session with targeted questions."""
        import uuid
        
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        
        self.sessions_created += 1
        
        # Generate questions based on analysis
        questions = await self._generate_questions(intent_analysis, user_input)
        
        # Rank questions by priority and impact
        questions = self._rank_questions(questions, intent_analysis)
        
        # Limit to most important questions to avoid overwhelming user
        questions = questions[:5]
        
        session = ClarificationSession(
            session_id=session_id,
            original_input=user_input,
            intent_analysis=intent_analysis,
            questions=questions,
            current_confidence=intent_analysis.confidence,
            target_confidence=self.confidence_threshold
        )
        
        self.active_sessions[session_id] = session
        
        self.logger.info(f"Created clarification session {session_id} with {len(questions)} questions")
        
        return session
    
    async def process_answer(self, session_id: str, question_index: int, answer: str) -> Tuple[ClarificationSession, bool]:
        """
        Process user answer to a clarification question.
        
        Returns:
            - Updated session
            - whether_session_complete: bool
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Clarification session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        if question_index >= len(session.questions):
            raise ValueError(f"Question index {question_index} out of range")
        
        question = session.questions[question_index]
        session.answers_received[question.question] = answer
        
        # Update confidence based on answer
        confidence_improvement = question.confidence_impact
        
        # Adjust confidence based on answer quality
        if self._is_high_quality_answer(answer, question):
            confidence_improvement *= 1.2
        elif self._is_low_quality_answer(answer):
            confidence_improvement *= 0.5
        
        session.current_confidence += confidence_improvement
        session.current_confidence = min(session.current_confidence, 0.99)  # Cap at 99%
        
        # Check if we need follow-up questions
        follow_up_questions = await self._generate_follow_up_questions(question, answer, session)
        
        # Add high-priority follow-up questions to session
        for follow_up in follow_up_questions:
            if follow_up.priority in [QuestionPriority.CRITICAL, QuestionPriority.HIGH]:
                session.questions.append(follow_up)
        
        # Check if session is complete
        unanswered_critical = [
            q for q in session.questions 
            if q.priority == QuestionPriority.CRITICAL and q.question not in session.answers_received
        ]
        
        session_complete = (
            session.current_confidence >= session.target_confidence and
            len(unanswered_critical) == 0
        ) or session.iteration_count >= session.max_iterations
        
        session.session_complete = session_complete
        
        if session_complete:
            session.refined_prompt = await self._generate_refined_prompt(session)
            self.successful_clarifications += 1
            self.logger.info(f"Clarification session {session_id} completed with confidence {session.current_confidence:.2f}")
        
        return session, session_complete
    
    async def get_next_questions(self, session_id: str, limit: int = 3) -> List[ClarificationQuestion]:
        """Get next batch of questions for user."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Clarification session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Find unanswered questions
        unanswered = [
            q for q in session.questions 
            if q.question not in session.answers_received
        ]
        
        # Sort by priority and return top N
        unanswered.sort(key=lambda q: (q.priority.value, -q.confidence_impact))
        
        return unanswered[:limit]
    
    async def _generate_questions(self, intent_analysis: IntentAnalysis, user_input: str) -> List[ClarificationQuestion]:
        """Generate clarification questions based on intent analysis."""
        questions = []
        self.questions_generated += 1
        
        # Generate questions based on identified issues
        
        # 1. Intent clarification
        if intent_analysis.confidence < 0.8:
            intent_questions = await self._generate_intent_questions(intent_analysis)
            questions.extend(intent_questions)
        
        # 2. Ambiguity resolution  
        if intent_analysis.ambiguous_terms:
            ambiguity_questions = await self._generate_ambiguity_questions(intent_analysis.ambiguous_terms, user_input)
            questions.extend(ambiguity_questions)
        
        # 3. Missing context
        if intent_analysis.missing_context:
            context_questions = await self._generate_context_questions(intent_analysis.missing_context)
            questions.extend(context_questions)
        
        # 4. Requirements specification
        if intent_analysis.primary_intent == IntentType.TASK_EXECUTION:
            requirement_questions = await self._generate_requirement_questions(intent_analysis)
            questions.extend(requirement_questions)
        
        # 5. Scope definition
        if intent_analysis.complexity_level in ["high", "very_high"]:
            scope_questions = await self._generate_scope_questions(intent_analysis)
            questions.extend(scope_questions)
        
        # 6. Technology/domain specific questions
        domain_questions = await self._generate_domain_specific_questions(intent_analysis)
        questions.extend(domain_questions)
        
        return questions
    
    async def _generate_intent_questions(self, intent_analysis: IntentAnalysis) -> List[ClarificationQuestion]:
        """Generate questions to clarify user intent."""
        questions = []
        
        # If multiple secondary intents, ask for prioritization
        if len(intent_analysis.secondary_intents) > 1:
            intent_names = [intent.value.replace('_', ' ') for intent in intent_analysis.secondary_intents]
            question = ClarificationQuestion(
                question=f"I see your request could involve multiple types of work: {', '.join(intent_names)}. Which is your primary focus?",
                question_type=QuestionType.CLARIFY_INTENT,
                priority=QuestionPriority.HIGH,
                possible_answers=intent_names,
                reasoning="Multiple possible intents detected",
                confidence_impact=0.2
            )
            questions.append(question)
        
        # If intent is unclear, ask directly
        if intent_analysis.confidence < 0.6:
            question = ClarificationQuestion(
                question="To better understand what you need, could you rephrase your request focusing on your main goal?",
                question_type=QuestionType.CLARIFY_INTENT,
                priority=QuestionPriority.CRITICAL,
                reasoning="Very low confidence in intent classification",
                confidence_impact=0.3
            )
            questions.append(question)
        
        return questions
    
    async def _generate_ambiguity_questions(self, ambiguous_terms: List[str], user_input: str) -> List[ClarificationQuestion]:
        """Generate questions to resolve ambiguous terms."""
        questions = []
        
        for term in ambiguous_terms[:3]:  # Limit to avoid overwhelming
            if term in self.ambiguity_resolvers:
                resolver = self.ambiguity_resolvers[term]
                question = ClarificationQuestion(
                    question=resolver["question_template"].format(term=term),
                    question_type=QuestionType.RESOLVE_AMBIGUITY,
                    priority=QuestionPriority.MEDIUM,
                    possible_answers=resolver.get("common_answers", []),
                    reasoning=f"Ambiguous term '{term}' needs clarification",
                    confidence_impact=0.1
                )
                questions.append(question)
            else:
                # Generic ambiguity question
                question = ClarificationQuestion(
                    question=f"When you mention '{term}', could you be more specific about what you're referring to?",
                    question_type=QuestionType.RESOLVE_AMBIGUITY,
                    priority=QuestionPriority.MEDIUM,
                    reasoning=f"Ambiguous term '{term}' detected",
                    confidence_impact=0.1
                )
                questions.append(question)
        
        return questions
    
    async def _generate_context_questions(self, missing_context: List[str]) -> List[ClarificationQuestion]:
        """Generate questions to provide missing context."""
        questions = []
        
        context_question_map = {
            "unclear_references": "What specific item, project, or system are you referring to when you say 'it', 'this', or 'that'?",
            "incomplete_specification": "Could you provide more details about the specific files, projects, or components you're working with?"
        }
        
        for context_type in missing_context:
            if context_type in context_question_map:
                question = ClarificationQuestion(
                    question=context_question_map[context_type],
                    question_type=QuestionType.PROVIDE_CONTEXT,
                    priority=QuestionPriority.HIGH,
                    reasoning=f"Missing context: {context_type}",
                    confidence_impact=0.15
                )
                questions.append(question)
        
        return questions
    
    async def _generate_requirement_questions(self, intent_analysis: IntentAnalysis) -> List[ClarificationQuestion]:
        """Generate questions about requirements and success criteria."""
        questions = []
        
        # Success criteria
        if not intent_analysis.success_criteria:
            question = ClarificationQuestion(
                question="What would success look like for this task? How will you know when it's completed correctly?",
                question_type=QuestionType.SPECIFY_REQUIREMENTS,
                priority=QuestionPriority.HIGH,
                reasoning="No clear success criteria defined",
                confidence_impact=0.15
            )
            questions.append(question)
        
        # Technology constraints
        if not intent_analysis.technologies and intent_analysis.technical_domain != TechnicalDomain.NON_TECHNICAL:
            question = ClarificationQuestion(
                question="Are there specific technologies, languages, or tools you'd like me to use or avoid?",
                question_type=QuestionType.SPECIFY_REQUIREMENTS,
                priority=QuestionPriority.MEDIUM,
                possible_answers=["No specific requirements", "Use existing project stack", "I'll specify"],
                reasoning="No technology preferences specified",
                confidence_impact=0.1
            )
            questions.append(question)
        
        # Constraints
        if not intent_analysis.has_constraints:
            question = ClarificationQuestion(
                question="Are there any constraints I should be aware of? (time, budget, existing code, team preferences, etc.)",
                question_type=QuestionType.SPECIFY_REQUIREMENTS,
                priority=QuestionPriority.MEDIUM,
                possible_answers=["No constraints", "Time sensitive", "Budget conscious", "Must work with existing code"],
                reasoning="No constraints identified",
                confidence_impact=0.1
            )
            questions.append(question)
        
        return questions
    
    async def _generate_scope_questions(self, intent_analysis: IntentAnalysis) -> List[ClarificationQuestion]:
        """Generate questions to define project scope.""" 
        questions = []
        
        # For complex tasks, clarify scope
        if intent_analysis.complexity_level in ["high", "very_high"]:
            question = ClarificationQuestion(
                question="This seems like a complex task. Should I focus on a specific part first, or provide a comprehensive solution?",
                question_type=QuestionType.DEFINE_SCOPE,
                priority=QuestionPriority.HIGH,
                possible_answers=[
                    "Start with core functionality",
                    "Provide complete solution", 
                    "Create a plan first",
                    "Focus on [specific area]"
                ],
                reasoning="Complex task requires scope clarification",
                confidence_impact=0.2
            )
            questions.append(question)
        
        # For tasks with multiple action verbs, clarify priority
        if len(intent_analysis.action_verbs) > 2:
            verbs = ', '.join(intent_analysis.action_verbs[:3])
            question = ClarificationQuestion(
                question=f"I see multiple actions in your request ({verbs}). What's the most important part to tackle first?",
                question_type=QuestionType.PRIORITIZE_OPTIONS,
                priority=QuestionPriority.MEDIUM,
                reasoning="Multiple actions require prioritization",
                confidence_impact=0.15
            )
            questions.append(question)
        
        return questions
    
    async def _generate_domain_specific_questions(self, intent_analysis: IntentAnalysis) -> List[ClarificationQuestion]:
        """Generate domain-specific clarification questions."""
        questions = []
        domain = intent_analysis.technical_domain
        
        if domain in self.domain_specific_questions:
            domain_templates = self.domain_specific_questions[domain]
            
            # Check which domain-specific questions are relevant
            for template in domain_templates[:2]:  # Limit to 2 per domain
                # Simple relevance check based on keywords
                if any(keyword in intent_analysis.keywords for keyword in template.get("keywords", [])):
                    question = ClarificationQuestion(
                        question=template["question"],
                        question_type=template["type"],
                        priority=template["priority"],
                        possible_answers=template.get("answers", []),
                        reasoning=template["reasoning"],
                        confidence_impact=template.get("confidence_impact", 0.1)
                    )
                    questions.append(question)
        
        return questions
    
    async def _generate_follow_up_questions(self, 
                                          original_question: ClarificationQuestion,
                                          answer: str, 
                                          session: ClarificationSession) -> List[ClarificationQuestion]:
        """Generate follow-up questions based on user's answer."""
        follow_ups = []
        
        # If user gives a very brief answer, ask for more detail
        if len(answer.strip()) < 10 and original_question.priority == QuestionPriority.CRITICAL:
            follow_up = ClarificationQuestion(
                question="Could you provide a bit more detail about that?",
                question_type=original_question.question_type,
                priority=QuestionPriority.HIGH,
                reasoning="Brief answer to critical question",
                confidence_impact=0.1
            )
            follow_ups.append(follow_up)
        
        # Domain-specific follow-ups
        if "technology" in answer.lower() and not session.intent_analysis.technologies:
            follow_up = ClarificationQuestion(
                question="Which specific technologies or versions should I focus on?",
                question_type=QuestionType.SPECIFY_REQUIREMENTS,
                priority=QuestionPriority.MEDIUM,
                reasoning="User mentioned technology but wasn't specific",
                confidence_impact=0.1
            )
            follow_ups.append(follow_up)
        
        return follow_ups
    
    async def _generate_refined_prompt(self, session: ClarificationSession) -> str:
        """Generate a refined, optimized prompt based on clarification answers."""
        # Start with original input
        refined_parts = [session.original_input]
        
        # Add clarified information
        clarifications = []
        for question_text, answer in session.answers_received.items():
            if len(answer.strip()) > 3:  # Skip very short answers
                clarifications.append(f"Clarification: {answer}")
        
        if clarifications:
            refined_parts.append("\nAdditional context:")
            refined_parts.extend(clarifications)
        
        refined_prompt = "\n".join(refined_parts)
        
        # Apply basic optimization
        refined_prompt = self._optimize_prompt_structure(refined_prompt, session.intent_analysis)
        
        return refined_prompt
    
    def _optimize_prompt_structure(self, prompt: str, intent_analysis: IntentAnalysis) -> str:
        """Apply basic prompt optimization."""
        parts = [prompt]
        
        # Add structure based on intent
        if intent_analysis.primary_intent == IntentType.TASK_EXECUTION:
            if "success criteria" not in prompt.lower():
                parts.append("\nPlease ensure the solution meets standard quality and performance requirements.")
        
        elif intent_analysis.primary_intent == IntentType.ANALYSIS_REVIEW:
            if "report" not in prompt.lower():
                parts.append("\nPlease provide a detailed analysis with findings and recommendations.")
        
        return "\n".join(parts)
    
    def _rank_questions(self, questions: List[ClarificationQuestion], intent_analysis: IntentAnalysis) -> List[ClarificationQuestion]:
        """Rank questions by priority and potential impact."""
        def priority_score(q: ClarificationQuestion) -> Tuple[int, float]:
            priority_values = {
                QuestionPriority.CRITICAL: 4,
                QuestionPriority.HIGH: 3,
                QuestionPriority.MEDIUM: 2,
                QuestionPriority.LOW: 1
            }
            return (priority_values[q.priority], q.confidence_impact)
        
        return sorted(questions, key=priority_score, reverse=True)
    
    def _is_high_quality_answer(self, answer: str, question: ClarificationQuestion) -> bool:
        """Check if user provided a high-quality answer."""
        answer = answer.strip().lower()
        
        # Length check
        if len(answer) < 3:
            return False
        
        # If specific answers were suggested, check if user picked one
        if question.possible_answers:
            return any(ans.lower() in answer for ans in question.possible_answers)
        
        # Look for specificity indicators
        specificity_indicators = [
            r'\b\d+\b',  # Numbers
            r'\b[a-z]+\.[a-z]+\b',  # File extensions or domains
            r'\b(version|v)\s*\d',  # Version numbers
            r'\b(using|with|in)\s+\w+\b'  # Technology mentions
        ]
        
        return any(re.search(pattern, answer) for pattern in specificity_indicators)
    
    def _is_low_quality_answer(self, answer: str) -> bool:
        """Check if answer is low quality."""
        answer = answer.strip().lower()
        
        low_quality_patterns = [
            r'^\b(i don\'t know|no|yes|ok|sure|maybe|whatever)\b$',
            r'^[a-z]\s*$',  # Single character
            r'^\.+$',  # Just dots
        ]
        
        return any(re.match(pattern, answer) for pattern in low_quality_patterns)
    
    def close_session(self, session_id: str):
        """Close and cleanup a clarification session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.logger.debug(f"Closed clarification session {session_id}")
    
    # Template and pattern building methods
    def _build_question_templates(self) -> Dict[QuestionType, List[Dict]]:
        """Build question templates by type."""
        return {
            QuestionType.CLARIFY_INTENT: [
                {"template": "What is your main goal with this request?", "context": "unclear_intent"},
                {"template": "Are you looking to {intent1} or {intent2}?", "context": "multiple_intents"}
            ],
            QuestionType.SPECIFY_REQUIREMENTS: [
                {"template": "What specific requirements should I keep in mind?", "context": "general"},
                {"template": "Are there any constraints or limitations?", "context": "constraints"}
            ],
            QuestionType.RESOLVE_AMBIGUITY: [
                {"template": "When you say '{term}', what specifically do you mean?", "context": "ambiguous_term"},
                {"template": "Could you clarify what '{term}' refers to?", "context": "unclear_reference"}
            ],
            QuestionType.DEFINE_SCOPE: [
                {"template": "Should I focus on a specific part or provide a complete solution?", "context": "complex_task"},
                {"template": "What's the priority order for these different aspects?", "context": "multiple_aspects"}
            ],
            QuestionType.PROVIDE_CONTEXT: [
                {"template": "Could you provide more background about your project or situation?", "context": "missing_context"},
                {"template": "What have you tried so far?", "context": "problem_solving"}
            ]
        }
    
    def _build_domain_questions(self) -> Dict[TechnicalDomain, List[Dict]]:
        """Build domain-specific question templates."""
        return {
            TechnicalDomain.SOFTWARE_DEVELOPMENT: [
                {
                    "question": "What programming language and framework should I focus on?",
                    "type": QuestionType.SPECIFY_REQUIREMENTS,
                    "priority": QuestionPriority.HIGH,
                    "keywords": ["code", "program", "develop"],
                    "answers": ["Python", "JavaScript", "Java", "Other"],
                    "reasoning": "Programming language affects implementation approach",
                    "confidence_impact": 0.15
                }
            ],
            TechnicalDomain.WEB_DEVELOPMENT: [
                {
                    "question": "Is this for frontend, backend, or full-stack development?", 
                    "type": QuestionType.DEFINE_SCOPE,
                    "priority": QuestionPriority.HIGH,
                    "keywords": ["web", "website", "app"],
                    "answers": ["Frontend only", "Backend only", "Full-stack", "Not sure"],
                    "reasoning": "Web development scope affects technology choices",
                    "confidence_impact": 0.2
                }
            ],
            TechnicalDomain.DATA_SCIENCE: [
                {
                    "question": "What type of data analysis or modeling do you need?",
                    "type": QuestionType.SPECIFY_REQUIREMENTS,
                    "priority": QuestionPriority.HIGH,
                    "keywords": ["data", "analysis", "model"],
                    "answers": ["Exploratory analysis", "Predictive modeling", "Visualization", "Data cleaning"],
                    "reasoning": "Data science approach depends on specific needs",
                    "confidence_impact": 0.18
                }
            ]
        }
    
    def _build_ambiguity_resolvers(self) -> Dict[str, Dict]:
        """Build ambiguity resolution templates."""
        return {
            "it": {
                "question_template": "What does '{term}' specifically refer to?",
                "common_answers": ["The file", "The project", "The system", "The code"]
            },
            "this": {
                "question_template": "What specific item does '{term}' refer to?", 
                "common_answers": ["This file", "This project", "This feature", "This issue"]
            },
            "system": {
                "question_template": "Which '{term}' are you referring to?",
                "common_answers": ["Operating system", "Application", "Database system", "Web system"]
            },
            "file": {
                "question_template": "Which specific '{term}' do you mean?",
                "common_answers": ["Config file", "Source code file", "Data file", "Documentation"]
            }
        }
    
    def get_engine_stats(self) -> Dict:
        """Get clarification engine statistics."""
        return {
            "sessions_created": self.sessions_created,
            "questions_generated": self.questions_generated,
            "successful_clarifications": self.successful_clarifications,
            "active_sessions": len(self.active_sessions),
            "success_rate": (
                self.successful_clarifications / max(self.sessions_created, 1)
            ) * 100
        }