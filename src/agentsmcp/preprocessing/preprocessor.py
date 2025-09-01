"""
User Prompt Preprocessor - Main orchestration class

Orchestrates the complete user prompt preprocessing pipeline including:
- Intent analysis and extraction
- Clarification question generation when needed
- Prompt optimization and enhancement
- Conversation context management
- Confidence-based processing decisions
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time

from .intent_analyzer import IntentAnalyzer, IntentAnalysis
from .clarification_engine import ClarificationEngine, ClarificationSession
from .prompt_optimizer import PromptOptimizer, OptimizedPrompt, OptimizationLevel
from .conversation_context import ConversationContext, ConversationTurn

logger = logging.getLogger(__name__)


class PreprocessingResult(Enum):
    """Types of preprocessing results."""
    READY_FOR_DELEGATION = "ready_for_delegation"
    NEEDS_CLARIFICATION = "needs_clarification"
    ERROR = "error"
    OPTIMIZED = "optimized"


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessor."""
    confidence_threshold: float = 0.9
    enable_clarification: bool = True
    enable_optimization: bool = True
    enable_context_learning: bool = True
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    max_clarification_iterations: int = 3
    processing_timeout_ms: int = 5000
    preserve_user_style: bool = True
    add_examples: bool = True
    track_conversation_history: bool = True


@dataclass 
class PreprocessedPrompt:
    """Complete preprocessing result."""
    # Input
    original_prompt: str
    session_id: Optional[str]
    
    # Analysis results
    intent_analysis: IntentAnalysis
    confidence: float
    
    # Processing results
    result_type: PreprocessingResult
    final_prompt: str
    optimized_prompt: Optional[OptimizedPrompt] = None
    clarification_session: Optional[ClarificationSession] = None
    
    # Context and learning
    relevant_context: List[Any] = field(default_factory=list)
    conversation_turn: Optional[ConversationTurn] = None
    
    # Metadata
    processing_time_ms: int = 0
    preprocessing_steps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UserPromptPreprocessor:
    """
    Main user prompt preprocessor orchestrating the complete pipeline.
    
    This is the primary interface for the preprocessing system, coordinating
    intent analysis, clarification, optimization, and context management.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the preprocessor with all components."""
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.intent_analyzer = IntentAnalyzer()
        self.clarification_engine = ClarificationEngine(
            confidence_threshold=self.config.confidence_threshold
        )
        self.prompt_optimizer = PromptOptimizer(
            default_level=self.config.optimization_level
        )
        self.conversation_context = ConversationContext()
        
        # Statistics and monitoring
        self.total_processed = 0
        self.clarification_sessions = 0
        self.optimization_requests = 0
        self.ready_delegations = 0
        self.processing_times = []
        
        self.logger.info(f"UserPromptPreprocessor initialized with threshold: {self.config.confidence_threshold}")
    
    async def preprocess_prompt(self, 
                              user_prompt: str,
                              session_id: Optional[str] = None,
                              context: Optional[Dict] = None,
                              user_id: Optional[str] = None,
                              project_context: Optional[str] = None) -> PreprocessedPrompt:
        """
        Main preprocessing pipeline for user prompts.
        
        Args:
            user_prompt: The original user input
            session_id: Optional session identifier
            context: Additional context information
            user_id: Optional user identifier for personalization
            project_context: Optional project context
            
        Returns:
            PreprocessedPrompt with complete analysis and processing results
        """
        start_time = time.time()
        self.total_processed += 1
        preprocessing_steps = []
        
        if context is None:
            context = {}
        
        try:
            self.logger.info(f"Starting preprocessing for prompt: {user_prompt[:100]}...")
            
            # Step 1: Initialize or retrieve session
            if session_id is None and self.config.track_conversation_history:
                session_id = await self.conversation_context.start_session(
                    user_id=user_id, 
                    project_context=project_context
                )
                preprocessing_steps.append("session_created")
            
            # Step 2: Get relevant conversation context
            relevant_context = []
            if session_id and self.config.enable_context_learning:
                relevant_context = await self.conversation_context.get_relevant_context(
                    session_id=session_id,
                    current_input=user_prompt
                )
                preprocessing_steps.append("context_retrieved")
                context["conversation_context"] = relevant_context
            
            # Step 3: Analyze intent
            self.logger.debug("Analyzing user intent...")
            intent_analysis = await self.intent_analyzer.analyze_intent(user_prompt, context)
            preprocessing_steps.append("intent_analyzed")
            
            # Step 4: Check if clarification is needed
            needs_clarification = False
            clarification_session = None
            
            if self.config.enable_clarification and intent_analysis.confidence < self.config.confidence_threshold:
                needs_clarification, confidence_gap, reasons = await self.clarification_engine.assess_clarification_need(
                    intent_analysis
                )
                
                if needs_clarification:
                    self.clarification_sessions += 1
                    clarification_session = await self.clarification_engine.create_clarification_session(
                        user_input=user_prompt,
                        intent_analysis=intent_analysis
                    )
                    preprocessing_steps.append("clarification_created")
                    
                    self.logger.info(f"Clarification needed - confidence: {intent_analysis.confidence:.2f}, gap: {confidence_gap:.2f}")
                    
                    result = PreprocessedPrompt(
                        original_prompt=user_prompt,
                        session_id=session_id,
                        intent_analysis=intent_analysis,
                        confidence=intent_analysis.confidence,
                        result_type=PreprocessingResult.NEEDS_CLARIFICATION,
                        final_prompt=user_prompt,
                        clarification_session=clarification_session,
                        relevant_context=relevant_context,
                        processing_time_ms=int((time.time() - start_time) * 1000),
                        preprocessing_steps=preprocessing_steps,
                        recommendations=reasons,
                        metadata={
                            "confidence_gap": confidence_gap,
                            "clarification_questions": len(clarification_session.questions)
                        }
                    )
                    
                    return result
            
            # Step 5: Optimize prompt if enabled
            optimized_prompt = None
            final_prompt = user_prompt
            
            if self.config.enable_optimization:
                self.optimization_requests += 1
                self.logger.debug("Optimizing prompt...")
                
                optimized_prompt = await self.prompt_optimizer.optimize_prompt(
                    prompt=user_prompt,
                    intent_analysis=intent_analysis,
                    level=self.config.optimization_level,
                    preserve_style=self.config.preserve_user_style,
                    add_examples=self.config.add_examples,
                    context=context
                )
                
                final_prompt = optimized_prompt.optimized_prompt
                preprocessing_steps.append("prompt_optimized")
                
                self.logger.debug(f"Prompt optimized with {len(optimized_prompt.optimizations_applied)} optimizations")
            
            # Step 6: Record conversation turn if session exists
            conversation_turn = None
            if session_id and self.config.track_conversation_history:
                # We don't have the system response yet, so we'll update it later
                conversation_turn = await self.conversation_context.add_conversation_turn(
                    session_id=session_id,
                    user_input=user_prompt,
                    intent_analysis=intent_analysis,
                    system_response="",  # Will be updated after agent response
                    success_indicators={"preprocessing_confidence": intent_analysis.confidence}
                )
                preprocessing_steps.append("conversation_recorded")
            
            # Step 7: Generate recommendations and warnings
            recommendations = await self._generate_recommendations(intent_analysis, context)
            warnings = await self._generate_warnings(intent_analysis, user_prompt)
            
            # Step 8: Finalize result
            self.ready_delegations += 1
            processing_time = int((time.time() - start_time) * 1000)
            self.processing_times.append(processing_time)
            
            result = PreprocessedPrompt(
                original_prompt=user_prompt,
                session_id=session_id,
                intent_analysis=intent_analysis,
                confidence=intent_analysis.confidence,
                result_type=PreprocessingResult.READY_FOR_DELEGATION,
                final_prompt=final_prompt,
                optimized_prompt=optimized_prompt,
                relevant_context=relevant_context,
                conversation_turn=conversation_turn,
                processing_time_ms=processing_time,
                preprocessing_steps=preprocessing_steps,
                recommendations=recommendations,
                warnings=warnings,
                metadata={
                    "optimization_applied": optimized_prompt is not None,
                    "context_entries_used": len(relevant_context),
                    "intent_confidence": intent_analysis.confidence,
                    "complexity_level": intent_analysis.complexity_level
                }
            )
            
            self.logger.info(f"Preprocessing completed successfully in {processing_time}ms - confidence: {intent_analysis.confidence:.2f}")
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"Preprocessing timeout after {self.config.processing_timeout_ms}ms")
            return self._create_error_result(user_prompt, session_id, "Processing timeout", start_time)
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}", exc_info=True)
            return self._create_error_result(user_prompt, session_id, str(e), start_time)
    
    async def handle_clarification_answer(self, 
                                        session_id: str,
                                        question_index: int,
                                        answer: str) -> Tuple[ClarificationSession, bool, Optional[PreprocessedPrompt]]:
        """
        Handle user's answer to a clarification question.
        
        Returns:
            - Updated clarification session
            - Whether clarification is complete
            - Final preprocessed result (if complete)
        """
        try:
            # Process the answer
            clarification_session, is_complete = await self.clarification_engine.process_answer(
                session_id=session_id,
                question_index=question_index,
                answer=answer
            )
            
            if is_complete and clarification_session.refined_prompt:
                # Re-run preprocessing with refined prompt
                refined_result = await self.preprocess_prompt(
                    user_prompt=clarification_session.refined_prompt,
                    session_id=session_id,
                    context={"clarification_completed": True, "original_input": clarification_session.original_input}
                )
                
                self.logger.info(f"Clarification completed for session {session_id}")
                return clarification_session, True, refined_result
            
            return clarification_session, is_complete, None
            
        except Exception as e:
            self.logger.error(f"Error handling clarification answer: {e}", exc_info=True)
            return clarification_session, False, None
    
    async def update_conversation_response(self, 
                                         session_id: str,
                                         turn_id: str,
                                         system_response: str,
                                         success_indicators: Optional[Dict] = None):
        """Update conversation turn with system response."""
        if not session_id or not self.config.track_conversation_history:
            return
        
        # Find and update the conversation turn
        if session_id in self.conversation_context.active_sessions:
            session = self.conversation_context.active_sessions[session_id]
            
            for turn in session.turns:
                if turn.turn_id == turn_id:
                    turn.system_response = system_response
                    if success_indicators:
                        turn.success_indicators.update(success_indicators)
                    break
            
            self.logger.debug(f"Updated conversation turn {turn_id} with system response")
    
    async def provide_user_feedback(self,
                                  session_id: str,
                                  turn_id: str,
                                  feedback: Dict[str, Any]):
        """Process user feedback for learning."""
        if not self.config.enable_context_learning:
            return
        
        await self.conversation_context.learn_from_feedback(
            session_id=session_id,
            turn_id=turn_id,
            feedback=feedback
        )
        
        self.logger.info(f"Processed user feedback for turn {turn_id}")
    
    def _create_error_result(self, user_prompt: str, session_id: Optional[str], error_msg: str, start_time: float) -> PreprocessedPrompt:
        """Create error result for failed preprocessing."""
        processing_time = int((time.time() - start_time) * 1000)
        
        return PreprocessedPrompt(
            original_prompt=user_prompt,
            session_id=session_id,
            intent_analysis=IntentAnalysis(
                primary_intent=self.intent_analyzer._analyze_primary_intent("")[0],
                confidence=0.1
            ),
            confidence=0.1,
            result_type=PreprocessingResult.ERROR,
            final_prompt=user_prompt,
            processing_time_ms=processing_time,
            preprocessing_steps=["error"],
            warnings=[f"Preprocessing failed: {error_msg}"],
            metadata={"error": error_msg}
        )
    
    async def _generate_recommendations(self, intent_analysis: IntentAnalysis, context: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Confidence-based recommendations
        if intent_analysis.confidence < 0.7:
            recommendations.append("Consider providing more specific details about your request")
        
        # Complexity-based recommendations
        if intent_analysis.complexity_level in ["high", "very_high"]:
            recommendations.append("This appears to be a complex task - consider breaking it into smaller parts")
        
        # Domain-specific recommendations
        if intent_analysis.technical_domain.value != "non_technical" and not intent_analysis.technologies:
            recommendations.append("Consider specifying preferred technologies or frameworks")
        
        # Success criteria recommendations
        if not intent_analysis.success_criteria and intent_analysis.primary_intent.value == "task_execution":
            recommendations.append("Define clear success criteria for better results")
        
        # Context-based recommendations
        if intent_analysis.missing_context:
            recommendations.append("Providing additional context would improve the response quality")
        
        return recommendations[:3]  # Limit to top 3
    
    async def _generate_warnings(self, intent_analysis: IntentAnalysis, user_prompt: str) -> List[str]:
        """Generate warnings about potential issues."""
        warnings = []
        
        # Ambiguity warnings
        if len(intent_analysis.ambiguous_terms) > 3:
            warnings.append(f"Multiple ambiguous terms detected: {', '.join(intent_analysis.ambiguous_terms[:3])}")
        
        # Assumption warnings
        if intent_analysis.assumptions_needed:
            warnings.append("Some assumptions may need to be made due to incomplete information")
        
        # Complexity warnings
        if intent_analysis.complexity_level == "very_high" and intent_analysis.confidence < 0.8:
            warnings.append("High complexity task with low confidence - results may need refinement")
        
        # Length warnings
        if len(user_prompt.split()) < 5:
            warnings.append("Very short request - more detail would improve results")
        elif len(user_prompt.split()) > 200:
            warnings.append("Very long request - consider focusing on key priorities")
        
        return warnings
    
    async def close_session(self, session_id: str):
        """Close a conversation session."""
        if self.config.track_conversation_history:
            await self.conversation_context.close_session(session_id)
            self.logger.info(f"Closed preprocessing session {session_id}")
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get comprehensive preprocessing statistics.""" 
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times) 
            if self.processing_times else 0
        )
        
        base_stats = {
            "total_processed": self.total_processed,
            "clarification_sessions": self.clarification_sessions,
            "optimization_requests": self.optimization_requests,
            "ready_delegations": self.ready_delegations,
            "average_processing_time_ms": round(avg_processing_time, 2),
            "clarification_rate": (self.clarification_sessions / max(self.total_processed, 1)) * 100,
            "optimization_rate": (self.optimization_requests / max(self.total_processed, 1)) * 100,
            "success_rate": (self.ready_delegations / max(self.total_processed, 1)) * 100
        }
        
        # Add component stats
        component_stats = {
            "intent_analyzer": self.intent_analyzer.get_analysis_stats(),
            "clarification_engine": self.clarification_engine.get_engine_stats(),
            "prompt_optimizer": self.prompt_optimizer.get_optimizer_stats(),
            "conversation_context": self.conversation_context.get_context_stats()
        }
        
        return {
            **base_stats,
            "components": component_stats,
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "optimization_level": self.config.optimization_level.value,
                "clarification_enabled": self.config.enable_clarification,
                "optimization_enabled": self.config.enable_optimization,
                "context_learning_enabled": self.config.enable_context_learning
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health_status = {"status": "healthy", "components": {}}
        
        # Check each component
        try:
            # Test intent analyzer
            test_analysis = await self.intent_analyzer.analyze_intent("test input")
            health_status["components"]["intent_analyzer"] = "healthy"
        except Exception as e:
            health_status["components"]["intent_analyzer"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        try:
            # Test clarification engine  
            test_session = await self.clarification_engine.create_clarification_session(
                "test", test_analysis
            )
            self.clarification_engine.close_session(test_session.session_id)
            health_status["components"]["clarification_engine"] = "healthy"
        except Exception as e:
            health_status["components"]["clarification_engine"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        try:
            # Test prompt optimizer
            test_optimization = await self.prompt_optimizer.optimize_prompt(
                "test prompt", test_analysis
            )
            health_status["components"]["prompt_optimizer"] = "healthy"
        except Exception as e:
            health_status["components"]["prompt_optimizer"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        try:
            # Test conversation context
            test_session = await self.conversation_context.start_session()
            await self.conversation_context.close_session(test_session)
            health_status["components"]["conversation_context"] = "healthy"
        except Exception as e:
            health_status["components"]["conversation_context"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        health_status["timestamp"] = time.time()
        return health_status