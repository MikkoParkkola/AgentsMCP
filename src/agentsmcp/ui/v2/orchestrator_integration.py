"""
TUI Orchestrator Integration

This component integrates the strict orchestrator-only communication architecture
into the existing TUI system. It ensures that:

1. All user inputs are routed through the orchestrator
2. Only orchestrator responses are displayed to users
3. Individual agent communications are completely hidden
4. Simple tasks don't spawn unnecessary agents
5. Complex tasks are handled transparently by orchestrator

Key Integration Points:
- Chat interface process_chat_message method
- Display filtering to hide agent outputs
- Status updates converted to orchestrator perspective
- Error handling through orchestrator fallbacks
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from ...conversation.orchestrated_conversation import OrchestratedConversationManager, create_orchestrated_conversation_manager

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorIntegrationConfig:
    """Configuration for orchestrator TUI integration."""
    enable_strict_isolation: bool = True
    show_orchestrator_stats: bool = False  # For debugging only
    log_agent_interactions: bool = True
    fallback_on_orchestrator_error: bool = True


class OrchestratorTUIIntegration:
    """
    Integration component that enforces orchestrator-only communication in the TUI.
    
    This component acts as a bridge between the existing TUI components and the
    new orchestrated conversation system, ensuring strict communication isolation.
    """
    
    def __init__(self, config: Optional[OrchestratorIntegrationConfig] = None):
        """Initialize the orchestrator TUI integration."""
        self.config = config or OrchestratorIntegrationConfig()
        self.orchestrated_conversation: Optional[OrchestratedConversationManager] = None
        
        # Monitoring components from orchestrator
        self._monitoring_components: Optional[Dict[str, Any]] = None
        
        # Integration state
        self.is_initialized = False
        self.fallback_conversation = None
        
        # Statistics for monitoring
        self.integration_stats = {
            "total_requests": 0,
            "orchestrator_responses": 0,
            "fallback_responses": 0,
            "blocked_agent_messages": 0
        }
        
        logger.info("Orchestrator TUI integration initialized")
    
    async def initialize(self, command_interface=None, theme_manager=None, agent_manager=None):
        """Initialize the orchestrator integration with required dependencies."""
        try:
            # Create orchestrated conversation manager
            self.orchestrated_conversation = create_orchestrated_conversation_manager(
                command_interface=command_interface,
                theme_manager=theme_manager,
                agent_manager=agent_manager
            )
            
            # Keep fallback for emergency cases
            if self.config.fallback_on_orchestrator_error:
                from ...conversation.conversation import ConversationManager
                self.fallback_conversation = ConversationManager(
                    command_interface=command_interface,
                    theme_manager=theme_manager,
                    agent_manager=agent_manager
                )
            
            # Initialize monitoring components
            await self._initialize_monitoring_components()
            
            self.is_initialized = True
            logger.info("Orchestrator TUI integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator TUI integration: {e}")
            raise
    
    async def process_user_input(self, user_input: str) -> str:
        """
        Process user input through orchestrator - MAIN INTEGRATION POINT.
        
        This method should replace all direct calls to conversation managers
        in the TUI components.
        """
        if not self.is_initialized or not self.orchestrated_conversation:
            logger.error("Orchestrator integration not initialized")
            return "I'm not ready to help yet. Please wait a moment and try again."
        
        self.integration_stats["total_requests"] += 1
        
        try:
            # Route through orchestrator - this enforces strict communication isolation
            response = await self.orchestrated_conversation.process_input(user_input)
            
            # This response is guaranteed to be from orchestrator perspective only
            self.integration_stats["orchestrator_responses"] += 1
            
            # Log the interaction type for monitoring
            if self.config.log_agent_interactions:
                self._log_interaction(user_input, response, "orchestrator")
            
            return response
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            
            # Use fallback if configured
            if self.fallback_conversation and self.config.fallback_on_orchestrator_error:
                try:
                    fallback_response = await self.fallback_conversation.process_input(user_input)
                    # Clean the fallback response to remove agent identifiers
                    clean_response = self._sanitize_fallback_response(fallback_response)
                    self.integration_stats["fallback_responses"] += 1
                    
                    if self.config.log_agent_interactions:
                        self._log_interaction(user_input, clean_response, "fallback")
                    
                    return clean_response
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
            
            # Final fallback - orchestrator-style error response
            return ("I apologize, but I'm having trouble processing your request right now. "
                   "Please try again in a moment, or try rephrasing your question.")
    
    async def process_user_input_streaming(self, user_input: str):
        """
        Process user input through orchestrator with streaming support - STREAMING INTEGRATION POINT.
        
        Yields:
            str: Response chunks as they arrive from the orchestrator
        """
        if not self.is_initialized or not self.orchestrated_conversation:
            logger.error("Orchestrator integration not initialized")
            yield "I'm not ready to help yet. Please wait a moment and try again."
            return
        
        self.integration_stats["total_requests"] += 1
        
        try:
            # Check if orchestrated conversation supports streaming
            if hasattr(self.orchestrated_conversation, 'process_input_streaming'):
                async for chunk in self.orchestrated_conversation.process_input_streaming(user_input):
                    yield chunk
            else:
                # Fallback to non-streaming for orchestrator
                response = await self.orchestrated_conversation.process_input(user_input)
                yield response
            
            # This response is guaranteed to be from orchestrator perspective only
            self.integration_stats["orchestrator_responses"] += 1
            
        except Exception as e:
            logger.error(f"Orchestrator streaming error: {e}")
            
            # Use fallback if configured
            if self.fallback_conversation and self.config.fallback_on_orchestrator_error:
                try:
                    if hasattr(self.fallback_conversation, 'llm_client') and hasattr(self.fallback_conversation.llm_client, 'send_message_streaming'):
                        # Use streaming fallback if available
                        full_response = ""
                        async for chunk in self.fallback_conversation.llm_client.send_message_streaming(user_input):
                            full_response += chunk
                            yield self._sanitize_fallback_chunk(chunk)
                        
                        self.integration_stats["fallback_responses"] += 1
                        if self.config.log_agent_interactions:
                            self._log_interaction(user_input, full_response, "fallback_streaming")
                    else:
                        # Non-streaming fallback
                        fallback_response = await self.fallback_conversation.process_input(user_input)
                        clean_response = self._sanitize_fallback_response(fallback_response)
                        self.integration_stats["fallback_responses"] += 1
                        yield clean_response
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback streaming also failed: {fallback_error}")
                    yield ("I apologize, but I'm having trouble processing your request right now. "
                           "Please try again in a moment, or try rephrasing your question.")
            else:
                # Final fallback - orchestrator-style error response
                yield ("I apologize, but I'm having trouble processing your request right now. "
                       "Please try again in a moment, or try rephrasing your question.")
    
    def supports_streaming(self) -> bool:
        """
        Check if the orchestrator integration supports streaming responses.
        """
        if not self.is_initialized or not self.orchestrated_conversation:
            return False
        
        # Check if orchestrator supports streaming
        if hasattr(self.orchestrated_conversation, 'supports_streaming'):
            return self.orchestrated_conversation.supports_streaming()
        
        # Check if fallback LLM client supports streaming
        if (self.fallback_conversation and 
            hasattr(self.fallback_conversation, 'llm_client') and 
            hasattr(self.fallback_conversation.llm_client, 'supports_streaming')):
            return self.fallback_conversation.llm_client.supports_streaming()
        
        return False
    
    def should_display_message(self, message: str, source: str) -> bool:
        """
        Filter messages to ensure only orchestrator messages are displayed.
        
        This method enforces the display isolation part of the architecture.
        """
        if not self.config.enable_strict_isolation:
            return True  # If isolation disabled, allow all messages
        
        # Block messages from individual agents
        blocked_sources = ["agent", "codex", "claude", "ollama"]
        if any(blocked_source in source.lower() for blocked_source in blocked_sources):
            self.integration_stats["blocked_agent_messages"] += 1
            logger.debug(f"Blocked agent message from {source}")
            return False
        
        # Block messages with agent identifiers
        agent_markers = ["🧩", "Agent:", "🤖"]
        if any(marker in message for marker in agent_markers):
            self.integration_stats["blocked_agent_messages"] += 1
            logger.debug("Blocked message with agent identifiers")
            return False
        
        # Allow orchestrator messages
        orchestrator_sources = ["orchestrator", "system", "chat_interface", "main"]
        if any(orch_source in source.lower() for orch_source in orchestrator_sources):
            return True
        
        # Default: allow message (but log for monitoring)
        logger.debug(f"Allowing message from unknown source: {source}")
        return True
    
    def convert_status_message(self, status: str, source: str) -> Optional[str]:
        """
        Convert agent status messages to orchestrator perspective.
        
        Returns None if status should be suppressed entirely.
        """
        if not self.config.enable_strict_isolation:
            return status
        
        # Use the orchestrator's communication interceptor for status conversion
        if (self.orchestrated_conversation and 
            hasattr(self.orchestrated_conversation.orchestrator, 'communication_interceptor')):
            
            return self.orchestrated_conversation.orchestrator.communication_interceptor.intercept_status_message(
                source, status
            )
        
        # Fallback status conversions
        status_conversions = {
            "Agent starting": None,  # Suppress
            "Processing": "Working on your request...",
            "Thinking": "Analyzing your request...",
            "Generating": "Preparing response...", 
            "Complete": "Ready"
        }
        
        for pattern, replacement in status_conversions.items():
            if pattern.lower() in status.lower():
                return replacement
        
        # Default: suppress unknown agent status messages
        return None
    
    def _sanitize_fallback_response(self, response: str) -> str:
        """Sanitize fallback responses to maintain orchestrator perspective."""
        import re
        
        # Remove agent identifiers
        response = re.sub(r'🧩\s*\w+\s*:\s*', '', response)
        response = re.sub(r'Agent\s+\w+\s*:\s*', '', response)
        
        # Convert agent self-references to orchestrator voice
        response = re.sub(r'I am (an? )?\w+ agent', 'I', response, flags=re.IGNORECASE)
        response = re.sub(r'As (an? )?\w+ agent', 'I', response, flags=re.IGNORECASE)
        
        # Clean up formatting
        response = re.sub(r'\s+', ' ', response).strip()
        
        return response
    
    def _sanitize_fallback_chunk(self, chunk: str) -> str:
        """Sanitize streaming fallback response chunks to maintain orchestrator perspective."""
        import re
        
        # Remove agent identifiers from chunks
        chunk = re.sub(r'🧩\s*\w*\s*:?\s*', '', chunk)
        chunk = re.sub(r'Agent\s+\w*\s*:?\s*', '', chunk)
        
        # Convert agent self-references (be more careful with partial chunks)
        chunk = re.sub(r'I am (an? )?\w*\s*agent', 'I', chunk, flags=re.IGNORECASE)
        chunk = re.sub(r'As (an? )?\w*\s*agent', 'I', chunk, flags=re.IGNORECASE)
        
        return chunk
    
    def _log_interaction(self, user_input: str, response: str, interaction_type: str):
        """Log interaction for monitoring (internal use only)."""
        logger.debug(f"Interaction [{interaction_type}]: {user_input[:50]}... -> {response[:50]}...")
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics for monitoring."""
        stats = self.integration_stats.copy()
        
        # Add orchestrator stats if available
        if self.orchestrated_conversation:
            stats["orchestrator_stats"] = self.orchestrated_conversation.get_orchestrator_stats()
            
            if self.config.show_orchestrator_stats:
                stats["task_classification"] = self.orchestrated_conversation.get_task_classification_stats()
                stats["synthesis_stats"] = self.orchestrated_conversation.get_synthesis_stats()
                stats["interception_stats"] = self.orchestrated_conversation.get_interception_stats()
        
        return stats
    
    async def _initialize_monitoring_components(self):
        """Initialize monitoring components from orchestrator."""
        try:
            if (self.orchestrated_conversation and 
                hasattr(self.orchestrated_conversation, 'orchestrator')):
                
                orchestrator = self.orchestrated_conversation.orchestrator
                if hasattr(orchestrator, 'get_monitoring_components'):
                    self._monitoring_components = orchestrator.get_monitoring_components()
                    logger.debug("Monitoring components initialized from orchestrator")
                else:
                    logger.warning("Orchestrator does not support monitoring components")
            else:
                logger.warning("No orchestrator available for monitoring initialization")
                
        except Exception as e:
            logger.error(f"Failed to initialize monitoring components: {e}")
    
    def get_monitoring_components(self) -> Optional[Dict[str, Any]]:
        """Get monitoring components for TUI integration."""
        return self._monitoring_components
    
    def has_monitoring_support(self) -> bool:
        """Check if monitoring components are available."""
        return self._monitoring_components is not None
    
    async def shutdown(self):
        """Shutdown the orchestrator integration."""
        logger.info("Shutting down orchestrator TUI integration...")
        
        if self.orchestrated_conversation:
            await self.orchestrated_conversation.shutdown()
        
        self.is_initialized = False
        logger.info("Orchestrator TUI integration shutdown complete")


# Singleton instance for TUI components to use
_orchestrator_integration: Optional[OrchestratorTUIIntegration] = None


def get_orchestrator_integration() -> OrchestratorTUIIntegration:
    """Get the global orchestrator integration instance."""
    global _orchestrator_integration
    if _orchestrator_integration is None:
        _orchestrator_integration = OrchestratorTUIIntegration()
    return _orchestrator_integration


async def initialize_orchestrator_integration(command_interface=None, theme_manager=None, agent_manager=None):
    """Initialize the global orchestrator integration."""
    integration = get_orchestrator_integration()
    await integration.initialize(command_interface, theme_manager, agent_manager)
    return integration


def is_orchestrator_integration_active() -> bool:
    """Check if orchestrator integration is active."""
    return _orchestrator_integration is not None and _orchestrator_integration.is_initialized


# Compatibility functions for existing TUI code
async def process_user_input_orchestrated(user_input: str) -> str:
    """Process user input through orchestrator (convenience function)."""
    integration = get_orchestrator_integration()
    return await integration.process_user_input(user_input)


async def process_user_input_orchestrated_streaming(user_input: str):
    """Process user input through orchestrator with streaming (convenience function)."""
    integration = get_orchestrator_integration()
    async for chunk in integration.process_user_input_streaming(user_input):
        yield chunk


def supports_streaming_orchestrated() -> bool:
    """Check if orchestrator supports streaming responses (convenience function)."""
    integration = get_orchestrator_integration()
    return integration.supports_streaming()


def should_display_message_orchestrated(message: str, source: str) -> bool:
    """Check if message should be displayed (convenience function)."""
    integration = get_orchestrator_integration()
    return integration.should_display_message(message, source)