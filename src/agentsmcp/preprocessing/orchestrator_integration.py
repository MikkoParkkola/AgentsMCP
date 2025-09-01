"""
Orchestrator Integration for User Prompt Preprocessing

Integrates the preprocessing system with the existing orchestrator workflow,
ensuring all user prompts are processed through the preprocessing pipeline
before task delegation.
"""

import logging
import asyncio
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

from ..orchestration.orchestrator import Orchestrator, OrchestratorConfig, OrchestratorResponse
from .preprocessor import UserPromptPreprocessor, PreprocessedPrompt, PreprocessingResult
from .config import get_config_manager, PreprocessingSettings

logger = logging.getLogger(__name__)


@dataclass
class EnhancedOrchestratorConfig(OrchestratorConfig):
    """Enhanced orchestrator config with preprocessing options."""
    
    # Preprocessing settings
    enable_preprocessing: bool = True
    preprocessing_confidence_threshold: float = 0.9
    enable_clarification_mode: bool = True
    preprocessing_timeout_ms: int = 5000
    
    # Integration settings  
    preprocessing_failure_fallback: bool = True
    preserve_original_prompt_in_metadata: bool = True
    enable_preprocessing_telemetry: bool = True


class PreprocessingEnabledOrchestrator(Orchestrator):
    """
    Enhanced orchestrator with integrated preprocessing pipeline.
    
    This orchestrator processes all user input through the preprocessing
    system before delegation, providing improved intent understanding,
    clarification capabilities, and optimized prompts.
    """
    
    def __init__(self, config: Optional[EnhancedOrchestratorConfig] = None):
        """Initialize orchestrator with preprocessing capabilities."""
        self.enhanced_config = config or EnhancedOrchestratorConfig()
        
        # Initialize base orchestrator
        super().__init__(self.enhanced_config)
        
        # Initialize preprocessing components
        if self.enhanced_config.enable_preprocessing:
            self.preprocessor = UserPromptPreprocessor(
                config=self._create_preprocessing_config()
            )
            self.logger.info("Preprocessing pipeline enabled")
        else:
            self.preprocessor = None
            self.logger.info("Preprocessing pipeline disabled")
        
        # Preprocessing state management
        self.active_clarification_sessions = {}
        self.preprocessing_stats = {
            "total_preprocessed": 0,
            "clarification_sessions": 0,
            "optimization_applied": 0,
            "preprocessing_failures": 0,
            "average_confidence_improvement": 0.0
        }
    
    def _create_preprocessing_config(self):
        """Create preprocessing config from orchestrator config.""" 
        from .preprocessor import PreprocessingConfig
        from .prompt_optimizer import OptimizationLevel
        
        return PreprocessingConfig(
            confidence_threshold=self.enhanced_config.preprocessing_confidence_threshold,
            enable_clarification=self.enhanced_config.enable_clarification_mode,
            enable_optimization=True,
            enable_context_learning=True,
            optimization_level=OptimizationLevel.STANDARD,
            processing_timeout_ms=self.enhanced_config.preprocessing_timeout_ms,
            track_conversation_history=True
        )
    
    async def process_user_input(self, 
                               user_input: str, 
                               context: Dict = None,
                               session_id: Optional[str] = None,
                               user_id: Optional[str] = None) -> OrchestratorResponse:
        """
        Enhanced user input processing with preprocessing pipeline.
        
        This method integrates preprocessing as the first step, handling
        clarification requests and optimization before standard orchestration.
        """
        start_time = asyncio.get_event_loop().time()
        context = context or {}
        
        try:
            self.logger.info(f"Processing enhanced user input: {user_input[:100]}...")
            
            # Step 1: Preprocess the user input (if enabled)
            if self.preprocessor:
                preprocessing_result = await self._preprocess_user_input(
                    user_input, context, session_id, user_id
                )
                
                # Handle clarification requests
                if preprocessing_result.result_type == PreprocessingResult.NEEDS_CLARIFICATION:
                    return await self._handle_clarification_request(preprocessing_result)
                
                # Handle preprocessing errors  
                if preprocessing_result.result_type == PreprocessingResult.ERROR:
                    if self.enhanced_config.preprocessing_failure_fallback:
                        self.logger.warning("Preprocessing failed, falling back to original orchestration")
                        return await super().process_user_input(user_input, context)
                    else:
                        return self._create_preprocessing_error_response(preprocessing_result)
                
                # Use optimized prompt for delegation
                processed_input = preprocessing_result.final_prompt
                context.update({
                    "preprocessing_applied": True,
                    "original_prompt": user_input,
                    "preprocessing_confidence": preprocessing_result.confidence,
                    "intent_analysis": preprocessing_result.intent_analysis,
                    "optimization_applied": preprocessing_result.optimized_prompt is not None
                })
                
                self.preprocessing_stats["total_preprocessed"] += 1
                if preprocessing_result.optimized_prompt:
                    self.preprocessing_stats["optimization_applied"] += 1
                
            else:
                # No preprocessing - use original input
                processed_input = user_input
            
            # Step 2: Process through standard orchestration pipeline
            orchestrator_response = await super().process_user_input(processed_input, context)
            
            # Step 3: Enhance response with preprocessing metadata
            if self.preprocessor and "preprocessing_applied" in context:
                orchestrator_response = await self._enhance_response_with_preprocessing_data(
                    orchestrator_response, preprocessing_result, context
                )
            
            # Step 4: Update conversation context if available
            if self.preprocessor and preprocessing_result and preprocessing_result.conversation_turn:
                await self.preprocessor.update_conversation_response(
                    session_id=preprocessing_result.session_id,
                    turn_id=preprocessing_result.conversation_turn.turn_id,
                    system_response=orchestrator_response.content,
                    success_indicators={"response_type": orchestrator_response.response_type}
                )
            
            return orchestrator_response
            
        except Exception as e:
            self.logger.error(f"Error in enhanced processing: {e}", exc_info=True)
            
            # Fallback to basic orchestration on error
            if self.enhanced_config.preprocessing_failure_fallback:
                return await super().process_user_input(user_input, context)
            
            processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            return OrchestratorResponse(
                content="I encountered an issue processing your request. Please try rephrasing or contact support.",
                response_type="error",
                processing_time_ms=processing_time,
                metadata={"error": str(e), "preprocessing_enabled": True}
            )
    
    async def _preprocess_user_input(self, 
                                   user_input: str,
                                   context: Dict,
                                   session_id: Optional[str],
                                   user_id: Optional[str]) -> PreprocessedPrompt:
        """Run user input through preprocessing pipeline."""
        try:
            preprocessing_result = await self.preprocessor.preprocess_prompt(
                user_prompt=user_input,
                session_id=session_id,
                context=context,
                user_id=user_id
            )
            
            return preprocessing_result
            
        except asyncio.TimeoutError:
            self.logger.error("Preprocessing timeout")
            self.preprocessing_stats["preprocessing_failures"] += 1
            raise
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            self.preprocessing_stats["preprocessing_failures"] += 1
            raise
    
    async def _handle_clarification_request(self, preprocessing_result: PreprocessedPrompt) -> OrchestratorResponse:
        """Handle clarification requests from preprocessing."""
        if not preprocessing_result.clarification_session:
            return OrchestratorResponse(
                content="I need more information to help you effectively, but there was an error generating clarification questions.",
                response_type="error"
            )
        
        session = preprocessing_result.clarification_session
        self.active_clarification_sessions[session.session_id] = session
        self.preprocessing_stats["clarification_sessions"] += 1
        
        # Get next questions to ask
        questions = await self.preprocessor.clarification_engine.get_next_questions(
            session.session_id, limit=3
        )
        
        if not questions:
            return OrchestratorResponse(
                content="I need more information but couldn't generate specific questions. Could you provide more details about your request?",
                response_type="clarification_needed"
            )
        
        # Format questions for user
        question_text = self._format_clarification_questions(questions)
        
        return OrchestratorResponse(
            content=question_text,
            response_type="clarification_needed",
            confidence=preprocessing_result.confidence,
            metadata={
                "clarification_session_id": session.session_id,
                "questions": [
                    {
                        "question": q.question,
                        "type": q.question_type.value,
                        "priority": q.priority.value,
                        "possible_answers": q.possible_answers
                    } for q in questions
                ],
                "original_prompt": preprocessing_result.original_prompt,
                "confidence": preprocessing_result.confidence,
                "reasons": preprocessing_result.recommendations
            }
        )
    
    async def handle_clarification_response(self, 
                                          session_id: str,
                                          question_index: int,
                                          user_answer: str) -> OrchestratorResponse:
        """
        Handle user's response to clarification questions.
        
        This method should be called when the user provides answers
        to clarification questions.
        """
        if not self.preprocessor or session_id not in self.active_clarification_sessions:
            return OrchestratorResponse(
                content="I couldn't find your clarification session. Please start over with your original request.",
                response_type="error"
            )
        
        try:
            # Process the clarification answer
            session, is_complete, final_result = await self.preprocessor.handle_clarification_answer(
                session_id=session_id,
                question_index=question_index,
                answer=user_answer
            )
            
            if is_complete and final_result:
                # Clarification complete - process the refined prompt
                del self.active_clarification_sessions[session_id]
                
                return await self.process_user_input(
                    user_input=final_result.final_prompt,
                    context={
                        "clarification_completed": True,
                        "original_prompt": session.original_input,
                        "clarification_session_id": session_id
                    }
                )
            else:
                # More questions needed
                next_questions = await self.preprocessor.clarification_engine.get_next_questions(
                    session_id, limit=3
                )
                
                if not next_questions:
                    # No more questions but not complete - use what we have
                    del self.active_clarification_sessions[session_id]
                    refined_prompt = session.refined_prompt or session.original_input
                    
                    return await self.process_user_input(
                        user_input=refined_prompt,
                        context={"partial_clarification": True}
                    )
                
                question_text = self._format_clarification_questions(next_questions)
                
                return OrchestratorResponse(
                    content=question_text,
                    response_type="clarification_needed",
                    metadata={
                        "clarification_session_id": session_id,
                        "questions": [
                            {
                                "question": q.question,
                                "type": q.question_type.value,
                                "priority": q.priority.value
                            } for q in next_questions
                        ],
                        "progress": f"Question {len(session.answers_received) + 1} of {len(session.questions)}"
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Error handling clarification response: {e}")
            if session_id in self.active_clarification_sessions:
                del self.active_clarification_sessions[session_id]
            
            return OrchestratorResponse(
                content="There was an error processing your answer. Please start over with your original request.",
                response_type="error",
                metadata={"error": str(e)}
            )
    
    def _format_clarification_questions(self, questions: List) -> str:
        """Format clarification questions for user presentation."""
        if len(questions) == 1:
            q = questions[0]
            text = f"To better help you, I have a question:\n\n{q.question}"
            
            if q.possible_answers:
                text += f"\n\nSome options to consider: {', '.join(q.possible_answers[:4])}"
            
            return text
        
        # Multiple questions
        text = "To better help you, I have a few questions:\n"
        
        for i, q in enumerate(questions, 1):
            text += f"\n{i}. {q.question}"
            if q.possible_answers and len(q.possible_answers) <= 3:
                text += f" ({', '.join(q.possible_answers)})"
        
        text += "\n\nYou can answer them one by one or all at once."
        return text
    
    def _create_preprocessing_error_response(self, preprocessing_result: PreprocessedPrompt) -> OrchestratorResponse:
        """Create error response for preprocessing failures."""
        return OrchestratorResponse(
            content="I had trouble understanding your request. Could you please rephrase it or provide more specific details?",
            response_type="error",
            processing_time_ms=preprocessing_result.processing_time_ms,
            metadata={
                "preprocessing_error": True,
                "warnings": preprocessing_result.warnings,
                "original_prompt": preprocessing_result.original_prompt if self.enhanced_config.preserve_original_prompt_in_metadata else None
            }
        )
    
    async def _enhance_response_with_preprocessing_data(self, 
                                                      orchestrator_response: OrchestratorResponse,
                                                      preprocessing_result: PreprocessedPrompt,
                                                      context: Dict) -> OrchestratorResponse:
        """Enhance orchestrator response with preprocessing metadata."""
        if not orchestrator_response.metadata:
            orchestrator_response.metadata = {}
        
        # Add preprocessing metadata
        preprocessing_metadata = {
            "preprocessing_applied": True,
            "preprocessing_confidence": preprocessing_result.confidence,
            "intent_detected": preprocessing_result.intent_analysis.primary_intent.value,
            "domain_detected": preprocessing_result.intent_analysis.technical_domain.value,
            "complexity_level": preprocessing_result.intent_analysis.complexity_level,
            "optimization_applied": preprocessing_result.optimized_prompt is not None,
            "preprocessing_time_ms": preprocessing_result.processing_time_ms
        }
        
        if preprocessing_result.optimized_prompt:
            preprocessing_metadata.update({
                "optimization_level": preprocessing_result.optimized_prompt.optimization_level.value,
                "optimizations_applied": preprocessing_result.optimized_prompt.optimizations_applied,
                "estimated_improvement": preprocessing_result.optimized_prompt.estimated_improvement
            })
        
        if preprocessing_result.recommendations:
            preprocessing_metadata["recommendations"] = preprocessing_result.recommendations
        
        if preprocessing_result.warnings:
            preprocessing_metadata["warnings"] = preprocessing_result.warnings
        
        # Preserve original metadata and add preprocessing data
        orchestrator_response.metadata.update(preprocessing_metadata)
        
        # Preserve original prompt if configured
        if self.enhanced_config.preserve_original_prompt_in_metadata:
            orchestrator_response.metadata["original_user_prompt"] = preprocessing_result.original_prompt
        
        return orchestrator_response
    
    async def provide_user_feedback(self,
                                  session_id: str,
                                  turn_id: str,
                                  feedback: Dict[str, Any]):
        """Process user feedback for learning."""
        if self.preprocessor:
            await self.preprocessor.provide_user_feedback(session_id, turn_id, feedback)
    
    async def close_session(self, session_id: str):
        """Close preprocessing and orchestration session.""" 
        if self.preprocessor:
            await self.preprocessor.close_session(session_id)
        
        # Clean up clarification sessions
        if session_id in self.active_clarification_sessions:
            del self.active_clarification_sessions[session_id]
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get comprehensive preprocessing statistics."""
        base_stats = self.preprocessing_stats.copy()
        
        if self.preprocessor:
            component_stats = self.preprocessor.get_preprocessing_stats()
            base_stats["component_details"] = component_stats
        
        base_stats["active_clarification_sessions"] = len(self.active_clarification_sessions)
        base_stats["preprocessing_enabled"] = self.preprocessor is not None
        
        return base_stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check including preprocessing."""
        # Get base orchestrator health
        base_health = {
            "orchestrator": "healthy",
            "total_requests": self.total_requests,
            "success_rate": (self.successful_responses / max(self.total_requests, 1)) * 100
        }
        
        # Check preprocessing health
        if self.preprocessor:
            try:
                preprocessing_health = await self.preprocessor.health_check()
                base_health["preprocessing"] = preprocessing_health
            except Exception as e:
                base_health["preprocessing"] = {"status": "error", "error": str(e)}
        else:
            base_health["preprocessing"] = {"status": "disabled"}
        
        return base_health
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced orchestrator statistics."""
        base_stats = super().get_stats()
        preprocessing_stats = self.get_preprocessing_stats()
        
        return {
            **base_stats,
            "preprocessing": preprocessing_stats,
            "configuration": {
                "preprocessing_enabled": self.enhanced_config.enable_preprocessing,
                "clarification_enabled": self.enhanced_config.enable_clarification_mode,
                "confidence_threshold": self.enhanced_config.preprocessing_confidence_threshold
            }
        }