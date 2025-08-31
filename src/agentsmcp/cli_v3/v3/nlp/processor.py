"""Natural language processor for AgentsMCP CLI v3 architecture.

This module provides the main NaturalLanguageProcessor class that integrates
LLM-based parsing with rule-based fallbacks for robust command interpretation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

from ..models.nlp_models import (
    ConversationContext,
    LLMConfig,
    ParsedCommand,
    CommandInterpretation,
    ParsingResult,
    ParsingMethod,
    NLPError,
    NLPMetrics,
    ParsingFailedError,
    AmbiguousInputError,
    LLMUnavailableError,
    ContextTooLargeError,
    UnsupportedLanguageError
)
from .local_llm_integration import LocalLLMIntegration
from .patterns import PatternMatcher


logger = logging.getLogger(__name__)


class NaturalLanguageProcessor:
    """Main natural language processor implementing the ICD interface.
    
    This processor combines LLM-based parsing with rule-based fallbacks
    to provide robust natural language command interpretation.
    """
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or LLMConfig()
        self.llm_integration = LocalLLMIntegration(self.llm_config)
        self.pattern_matcher = PatternMatcher()
        self.metrics = NLPMetrics()
        
        # Processing settings
        self.enable_llm = True
        self.fallback_threshold = 0.3  # Minimum confidence for rule-based fallback
        self.ambiguity_threshold = 0.7  # Confidence below which we consider ambiguous
        
        logger.info("NaturalLanguageProcessor initialized")
    
    async def parse_command(
        self,
        natural_input: str,
        context: ConversationContext,
        llm_config: Optional[LLMConfig] = None
    ) -> ParsingResult:
        """Parse natural language input into structured commands following ICD.
        
        Args:
            natural_input: User's natural language request
            context: ConversationContext with history, project_state, user_preferences  
            llm_config: Optional LLMConfig with model_name, max_tokens, temperature
            
        Returns:
            ParsingResult with structured_command, alternative_interpretations, explanation
            
        Raises:
            ParsingFailed: When parsing completely fails
            AmbiguousInput: When input has multiple valid interpretations  
            LLMUnavailable: When LLM service is unavailable
            ContextTooLarge: When context exceeds model limits
            UnsupportedLanguage: When input language is not supported
        """
        start_time = time.time()
        request_id = f"nlp_{int(time.time() * 1000)}"
        
        # Update metrics
        self.metrics.total_requests += 1
        
        try:
            # Input validation and preprocessing
            processed_input = self._preprocess_input(natural_input)
            if not processed_input:
                raise ParsingFailedError("Empty or invalid input")
            
            # Use provided config or default
            config = llm_config or self.llm_config
            
            # Update context with current request
            context.add_command(processed_input)
            
            # Try different parsing methods
            parsing_results = await self._attempt_parsing_methods(
                processed_input, context, config
            )
            
            if not parsing_results:
                self.metrics.failed_parses += 1
                raise ParsingFailedError("All parsing methods failed")
            
            # Process results and determine best interpretation
            structured_command, alternatives, method_used, explanation = \
                self._process_parsing_results(parsing_results, processed_input)
            
            # Check for ambiguity
            if structured_command.confidence < self.ambiguity_threshold and alternatives:
                # Consider it ambiguous if we have multiple reasonable interpretations
                confident_alternatives = [alt for alt in alternatives if alt.confidence >= 0.5]
                if len(confident_alternatives) > 1:
                    self.metrics.failed_parses += 1
                    raise AmbiguousInputError(
                        f"Input has {len(confident_alternatives)} possible interpretations"
                    )
            
            # Update success metrics
            self.metrics.successful_parses += 1
            if method_used == ParsingMethod.LLM:
                self.metrics.llm_calls += 1
            elif method_used == ParsingMethod.RULE_BASED:
                self.metrics.rule_based_matches += 1
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Update metrics
            self._update_metrics(structured_command.confidence, processing_time_ms)
            
            return ParsingResult(
                request_id=request_id,
                success=True,
                structured_command=structured_command,
                alternative_interpretations=alternatives,
                explanation=explanation,
                method_used=method_used,
                processing_time_ms=processing_time_ms,
                metadata={
                    "input_length": len(natural_input),
                    "processed_input": processed_input,
                    "context_size": len(context.command_history),
                    "llm_model": config.model_name
                }
            )
            
        except (ParsingFailedError, AmbiguousInputError, LLMUnavailableError, 
                ContextTooLargeError, UnsupportedLanguageError) as e:
            # Known NLP errors - record and re-raise
            self.metrics.failed_parses += 1
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            error = NLPError(
                error_type=type(e).__name__,
                message=str(e),
                recovery_suggestions=self._get_recovery_suggestions(type(e).__name__, natural_input)
            )
            
            return ParsingResult(
                request_id=request_id,
                success=False,
                structured_command=None,
                alternative_interpretations=[],
                explanation=f"Failed to parse: {str(e)}",
                method_used=ParsingMethod.RULE_BASED,  # Default for errors
                processing_time_ms=processing_time_ms,
                errors=[error]
            )
            
        except Exception as e:
            # Unexpected errors
            self.metrics.failed_parses += 1
            logger.error(f"Unexpected error in parse_command: {e}")
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            error = NLPError(
                error_type="UnexpectedError",
                message=f"Internal error: {str(e)}",
                recovery_suggestions=["Try rephrasing your request", "Contact support if issue persists"]
            )
            
            return ParsingResult(
                request_id=request_id,
                success=False,
                structured_command=None,
                alternative_interpretations=[],
                explanation="Internal processing error occurred",
                method_used=ParsingMethod.RULE_BASED,
                processing_time_ms=processing_time_ms,
                errors=[error]
            )
    
    def _preprocess_input(self, natural_input: str) -> str:
        """Preprocess and validate natural language input."""
        if not natural_input or not isinstance(natural_input, str):
            return ""
        
        # Basic cleaning
        processed = natural_input.strip()
        
        # Remove excessive whitespace
        processed = " ".join(processed.split())
        
        # Basic language detection (simple heuristic)
        if not self._is_supported_language(processed):
            raise UnsupportedLanguageError("Input appears to be in an unsupported language")
        
        return processed
    
    def _is_supported_language(self, text: str) -> bool:
        """Simple heuristic to check if text is in supported language (English)."""
        # Very basic check - just ensure we have some English-like words
        english_indicators = [
            'analyze', 'help', 'run', 'start', 'stop', 'show', 'list', 'create', 
            'delete', 'open', 'close', 'the', 'a', 'an', 'and', 'or', 'to',
            'my', 'me', 'i', 'you', 'it', 'this', 'that', 'with', 'for'
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # If text has any common English words, consider it supported
        return any(word in english_indicators for word in words) or len(words) <= 2
    
    async def _attempt_parsing_methods(
        self,
        processed_input: str,
        context: ConversationContext,
        config: LLMConfig
    ) -> List[Tuple[ParsedCommand, str, ParsingMethod]]:
        """Attempt different parsing methods and return results."""
        results = []
        
        # Method 1: Try LLM parsing first (if enabled and available)
        if self.enable_llm:
            try:
                llm_result = await self._try_llm_parsing(processed_input, context, config)
                if llm_result:
                    results.append(llm_result)
            except LLMUnavailableError:
                logger.warning("LLM unavailable, falling back to rule-based parsing")
            except ContextTooLargeError:
                logger.warning("Context too large for LLM, trying rule-based parsing")
            except Exception as e:
                logger.warning(f"LLM parsing failed: {e}")
        
        # Method 2: Rule-based parsing (always attempt as fallback)
        try:
            rule_result = await self._try_rule_based_parsing(processed_input, context)
            if rule_result:
                results.append(rule_result)
        except Exception as e:
            logger.warning(f"Rule-based parsing failed: {e}")
        
        # Method 3: Hybrid approach (combine insights from both if available)
        if len(results) >= 2:
            try:
                hybrid_result = await self._try_hybrid_parsing(results, processed_input)
                if hybrid_result:
                    results.append(hybrid_result)
            except Exception as e:
                logger.warning(f"Hybrid parsing failed: {e}")
        
        return results
    
    async def _try_llm_parsing(
        self,
        processed_input: str,
        context: ConversationContext,
        config: LLMConfig
    ) -> Optional[Tuple[ParsedCommand, str, ParsingMethod]]:
        """Attempt LLM-based parsing."""
        try:
            # Check if LLM is available
            if not await self.llm_integration.check_availability():
                raise LLMUnavailableError("LLM service not available")
            
            # Update integration config if needed
            if config != self.llm_config:
                self.llm_integration.update_config(config)
            
            # Prepare context for LLM
            llm_context = {
                "command_history": context.command_history,
                "current_directory": context.current_directory,
                "recent_files": context.recent_files,
                "user_preferences": context.user_preferences
            }
            
            # Call LLM
            parsed_command, explanation = await self.llm_integration.parse_command(
                processed_input, llm_context
            )
            
            if parsed_command:
                return (parsed_command, explanation, ParsingMethod.LLM)
            
        except (LLMUnavailableError, ContextTooLargeError):
            raise  # Re-raise these specific errors
        except Exception as e:
            logger.error(f"LLM parsing error: {e}")
        
        return None
    
    async def _try_rule_based_parsing(
        self,
        processed_input: str,
        context: ConversationContext
    ) -> Optional[Tuple[ParsedCommand, str, ParsingMethod]]:
        """Attempt rule-based parsing using patterns."""
        try:
            parsed_command = self.pattern_matcher.parse_command_fallback(
                processed_input, self.fallback_threshold
            )
            
            if parsed_command:
                explanation = f"Matched pattern for '{parsed_command.action}' command"
                return (parsed_command, explanation, ParsingMethod.RULE_BASED)
            
        except Exception as e:
            logger.error(f"Rule-based parsing error: {e}")
        
        return None
    
    async def _try_hybrid_parsing(
        self,
        existing_results: List[Tuple[ParsedCommand, str, ParsingMethod]],
        processed_input: str
    ) -> Optional[Tuple[ParsedCommand, str, ParsingMethod]]:
        """Combine insights from LLM and rule-based parsing."""
        if len(existing_results) < 2:
            return None
        
        try:
            llm_result = None
            rule_result = None
            
            for command, explanation, method in existing_results:
                if method == ParsingMethod.LLM:
                    llm_result = (command, explanation)
                elif method == ParsingMethod.RULE_BASED:
                    rule_result = (command, explanation)
            
            if not (llm_result and rule_result):
                return None
            
            llm_command, llm_explanation = llm_result
            rule_command, rule_explanation = rule_result
            
            # If both methods agree on action, create hybrid result
            if llm_command.action == rule_command.action:
                # Merge parameters from both methods
                hybrid_parameters = dict(rule_command.parameters)
                hybrid_parameters.update(llm_command.parameters)
                
                # Use higher confidence
                hybrid_confidence = max(llm_command.confidence, rule_command.confidence) * 0.95
                
                hybrid_command = ParsedCommand(
                    action=llm_command.action,
                    parameters=hybrid_parameters,
                    confidence=hybrid_confidence,
                    method=ParsingMethod.HYBRID
                )
                
                explanation = f"Hybrid parsing: {llm_explanation} (confirmed by pattern matching)"
                
                return (hybrid_command, explanation, ParsingMethod.HYBRID)
            
        except Exception as e:
            logger.warning(f"Hybrid parsing failed: {e}")
        
        return None
    
    def _process_parsing_results(
        self,
        results: List[Tuple[ParsedCommand, str, ParsingMethod]],
        processed_input: str
    ) -> Tuple[ParsedCommand, List[CommandInterpretation], ParsingMethod, str]:
        """Process parsing results to determine best command and alternatives."""
        if not results:
            raise ParsingFailedError("No parsing results available")
        
        # Sort by confidence and method preference
        method_priority = {
            ParsingMethod.HYBRID: 1,
            ParsingMethod.LLM: 2,  
            ParsingMethod.RULE_BASED: 3
        }
        
        sorted_results = sorted(
            results,
            key=lambda x: (method_priority.get(x[2], 4), -x[0].confidence)
        )
        
        # Best result becomes the primary command
        best_command, best_explanation, best_method = sorted_results[0]
        
        # Create alternative interpretations from remaining results
        alternatives = []
        for command, explanation, method in sorted_results[1:]:
            if command.action != best_command.action or command.confidence >= 0.4:
                # Include as alternative if different action or decent confidence
                interpretation = CommandInterpretation(
                    command=command,
                    rationale=explanation,
                    confidence=command.confidence,
                    examples=self.pattern_matcher.get_pattern_examples(command.action)[:2]
                )
                alternatives.append(interpretation)
        
        return best_command, alternatives, best_method, best_explanation
    
    def _update_metrics(self, confidence: float, processing_time_ms: int) -> None:
        """Update internal metrics."""
        # Update averages
        if self.metrics.total_requests > 1:
            # Incremental average update
            weight = 1.0 / self.metrics.total_requests
            self.metrics.average_confidence = (
                (1 - weight) * self.metrics.average_confidence + 
                weight * confidence
            )
            self.metrics.average_processing_time_ms = (
                (1 - weight) * self.metrics.average_processing_time_ms + 
                weight * processing_time_ms
            )
        else:
            self.metrics.average_confidence = confidence
            self.metrics.average_processing_time_ms = float(processing_time_ms)
    
    def _get_recovery_suggestions(self, error_type: str, natural_input: str) -> List[str]:
        """Generate recovery suggestions based on error type."""
        suggestions = []
        
        if error_type == "ParsingFailedError":
            suggestions.extend([
                "Try rephrasing your request more clearly",
                "Use simpler language or break into smaller requests",
                "Try using specific command keywords like 'analyze', 'help', 'status'"
            ])
            
        elif error_type == "AmbiguousInputError":
            suggestions.extend([
                "Be more specific about what you want to do",
                "Add more context to clarify your intent",
                "Choose one specific action and try again"
            ])
            
        elif error_type == "LLMUnavailableError":
            suggestions.extend([
                "LLM service is currently unavailable",
                "Try using direct commands (e.g., 'help', 'status')",
                "Check if Ollama is running locally"
            ])
            
        elif error_type == "ContextTooLargeError":
            suggestions.extend([
                "Your request context is too large for processing",
                "Try starting a new conversation session",
                "Use shorter, more focused requests"
            ])
            
        elif error_type == "UnsupportedLanguageError":
            suggestions.extend([
                "Please use English for commands",
                "Try rephrasing in English",
                "Use simple English words and phrases"
            ])
        
        # Add general suggestions
        suggestions.append("Type 'help' to see available commands")
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    async def get_command_suggestions(
        self, 
        context: ConversationContext,
        limit: int = 5
    ) -> List[str]:
        """Get contextual command suggestions for the user."""
        suggestions = []
        
        # Based on recent command history
        recent_actions = [cmd.split()[0] for cmd in context.command_history[-3:]]
        
        if not recent_actions or "analyze" not in recent_actions:
            suggestions.append("analyze my code")
            
        if "status" not in recent_actions:
            suggestions.append("check status")
            
        if "help" not in recent_actions:
            suggestions.append("show help")
        
        # Based on project state
        if context.recent_files:
            suggestions.append(f"analyze {context.recent_files[-1]}")
        
        # Based on current directory
        if context.current_directory != ".":
            suggestions.append(f"list files in {context.current_directory}")
        
        # Standard suggestions
        suggestions.extend([
            "start the TUI",
            "optimize costs", 
            "open dashboard",
            "configure settings"
        ])
        
        return suggestions[:limit]
    
    def get_metrics(self) -> NLPMetrics:
        """Get current NLP processing metrics."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self.metrics = NLPMetrics()
        logger.info("NLP metrics reset")
    
    def set_llm_enabled(self, enabled: bool) -> None:
        """Enable or disable LLM processing."""
        self.enable_llm = enabled
        logger.info(f"LLM processing {'enabled' if enabled else 'disabled'}")
    
    def update_config(self, new_config: LLMConfig) -> None:
        """Update LLM configuration."""
        self.llm_config = new_config
        self.llm_integration.update_config(new_config)
        logger.info("NLP processor configuration updated")
    
    async def test_system(self) -> Dict[str, Any]:
        """Test all NLP components and return status."""
        results = {
            "llm_integration": await self.llm_integration.test_connection(),
            "pattern_matcher": {
                "available": True,
                "patterns_loaded": len(self.pattern_matcher.patterns),
                "supported_actions": self.pattern_matcher.get_supported_actions()
            },
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": self.metrics.success_rate,
                "average_confidence": self.metrics.average_confidence
            }
        }
        
        return results