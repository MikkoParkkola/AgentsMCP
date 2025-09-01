"""
Orchestrated Conversation Manager

This replaces the direct agent communication in the conversation system with
strict orchestrator-only communication. Users will only see responses from 
the orchestrator perspective, never individual agent outputs.

Key Changes:
- All user inputs routed through orchestrator
- Agent responses intercepted and synthesized 
- Simple tasks handled without agent spawning
- Complex tasks delegated but responses unified
- Complete communication isolation maintained
"""

import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime

from ..orchestration.orchestrator import Orchestrator, OrchestratorConfig, OrchestratorMode
from .conversation import ConversationManager
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class OrchestratedConversationManager:
    """
    Orchestrated conversation manager that enforces strict communication isolation.
    
    This class replaces direct agent communication with orchestrator-mediated
    communication, ensuring users only see unified responses.
    """
    
    def __init__(self, command_interface=None, theme_manager=None, agent_manager=None):
        """Initialize the orchestrated conversation manager."""
        self.command_interface = command_interface
        self.theme_manager = theme_manager
        self.agent_manager = agent_manager
        
        # Initialize orchestrator with strict isolation mode
        orchestrator_config = OrchestratorConfig(
            mode=OrchestratorMode.STRICT_ISOLATION,
            enable_smart_classification=True,
            intercept_all_agent_output=True,
            fallback_to_simple_response=True,
            orchestrator_persona="AgentsMCP assistant"
        )
        
        # Create orchestrator instance
        self.orchestrator = Orchestrator(config=orchestrator_config)
        
        # Conversation context for continuity
        self.conversation_history = []
        self.context_cache = {}
        
        logger.info("Orchestrated conversation manager initialized")
    
    def _create_fallback_conversation_manager(self) -> ConversationManager:
        """Create a fallback conversation manager for agent delegation."""
        # This provides the orchestrator with access to existing agent delegation methods
        # while maintaining communication isolation
        return ConversationManager(
            command_interface=self.command_interface,
            theme_manager=self.theme_manager,
            agent_manager=self.agent_manager
        )
    
    async def process_input(self, user_input: str) -> str:
        """
        Process user input through the orchestrator.
        
        This is the ONLY method that should be called by the TUI.
        All agent interactions happen internally within the orchestrator.
        """
        logger.debug(f"Orchestrated conversation processing: {user_input[:50]}...")
        
        # Build conversation context
        context = self._build_conversation_context(user_input)
        
        try:
            # Route through orchestrator - this enforces communication isolation
            orchestrator_response = await self.orchestrator.process_user_input(
                user_input, context
            )
            
            # Store in conversation history for context
            self.conversation_history.append({
                "input": user_input,
                "response": orchestrator_response.content,
                "response_type": orchestrator_response.response_type,
                "agents_consulted": orchestrator_response.agents_consulted,
                "timestamp": datetime.now(),
                "processing_time_ms": orchestrator_response.processing_time_ms
            })
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-50:]
            
            # Log orchestrator decision for debugging
            logger.info(f"Orchestrator response type: {orchestrator_response.response_type}, "
                       f"agents: {orchestrator_response.agents_consulted}, "
                       f"time: {orchestrator_response.processing_time_ms}ms")
            
            return orchestrator_response.content
            
        except Exception as e:
            logger.error(f"Error in orchestrated conversation processing: {e}")
            
            # Provide orchestrator-style error response
            return ("I apologize, but I encountered an issue processing your request. "
                   "Let me try a different approach. Could you please rephrase what you'd like help with?")
    
    async def process_input_streaming(self, user_input: str):
        """
        Process user input through the orchestrator with streaming support.
        
        Yields response chunks as they arrive from the orchestrator.
        This is the STREAMING method that should be called by the TUI for streaming responses.
        """
        logger.debug(f"Orchestrated streaming conversation processing: {user_input[:50]}...")
        
        # Build conversation context
        context = self._build_conversation_context(user_input)
        
        try:
            # Check if orchestrator supports streaming
            if hasattr(self.orchestrator, 'process_user_input_streaming'):
                full_response = ""
                response_info = None
                
                async for chunk_data in self.orchestrator.process_user_input_streaming(user_input, context):
                    if isinstance(chunk_data, dict) and 'content' in chunk_data:
                        # Extract chunk content and metadata
                        chunk_content = chunk_data['content']
                        if chunk_content:
                            full_response += chunk_content
                            yield chunk_content
                        
                        # Store response metadata from final chunk
                        if chunk_data.get('is_final', False):
                            response_info = chunk_data
                    else:
                        # Simple string chunk
                        chunk_content = str(chunk_data)
                        full_response += chunk_content
                        yield chunk_content
                
                # Store in conversation history for context
                self.conversation_history.append({
                    "input": user_input,
                    "response": full_response,
                    "response_type": response_info.get('response_type', 'streaming') if response_info else 'streaming',
                    "agents_consulted": response_info.get('agents_consulted', []) if response_info else [],
                    "timestamp": datetime.now(),
                    "processing_time_ms": response_info.get('processing_time_ms', 0) if response_info else 0
                })
            else:
                # Fallback to non-streaming orchestrator
                response = await self.process_input(user_input)
                yield response
            
        except Exception as e:
            logger.error(f"Error in orchestrated streaming conversation processing: {e}")
            
            # Provide orchestrator-style error response
            yield ("I apologize, but I encountered an issue processing your request. "
                   "Let me try a different approach. Could you please rephrase what you'd like help with?")
    
    def supports_streaming(self) -> bool:
        """
        Check if the orchestrated conversation manager supports streaming responses.
        """
        return hasattr(self.orchestrator, 'supports_streaming') and self.orchestrator.supports_streaming()
    
    def _build_conversation_context(self, user_input: str) -> Dict[str, Any]:
        """Build context for the orchestrator."""
        context = {
            "conversation_length": len(self.conversation_history),
            "recent_interactions": self.conversation_history[-3:] if self.conversation_history else [],
            "theme": getattr(self.theme_manager, 'current_theme', 'auto') if self.theme_manager else 'auto',
            "available_commands": self._get_available_commands(),
        }
        
        # Add recent task patterns for better classification
        if self.conversation_history:
            recent_types = [entry["response_type"] for entry in self.conversation_history[-5:]]
            context["recent_response_types"] = recent_types
            
            # If recent interactions were simple, bias toward simple responses
            simple_rate = sum(1 for t in recent_types if t == "simple") / len(recent_types)
            context["simple_interaction_bias"] = simple_rate
        
        return context
    
    def _get_available_commands(self) -> Dict[str, str]:
        """Get available commands for context."""
        return {
            "status": "Show system status",
            "settings": "Open settings",
            "help": "Show help information",
            "dashboard": "Start monitoring dashboard",
            "theme": "Change theme (light/dark/auto)",
        }
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get current conversation context (orchestrator perspective only)."""
        orchestrator_stats = self.orchestrator.get_orchestrator_stats()
        
        return {
            # Orchestrator-specific context
            "orchestrator_mode": self.orchestrator.config.mode.value,
            "total_requests": orchestrator_stats["total_requests"],
            "simple_responses": orchestrator_stats["simple_responses"],
            "agent_delegations": orchestrator_stats["agent_delegations"],
            "multi_agent_tasks": orchestrator_stats["multi_agent_tasks"],
            
            # Conversation context
            "conversation_length": len(self.conversation_history),
            "theme": getattr(self.theme_manager, 'current_theme', 'auto') if self.theme_manager else 'auto',
            
            # Performance metrics
            "average_response_time": self._calculate_average_response_time(),
            "communication_isolation_active": True  # Always true for orchestrated mode
        }
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time from recent interactions."""
        if not self.conversation_history:
            return 0.0
        
        recent_times = [
            entry["processing_time_ms"] 
            for entry in self.conversation_history[-10:] 
            if "processing_time_ms" in entry
        ]
        
        return sum(recent_times) / len(recent_times) if recent_times else 0.0
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get detailed orchestrator statistics."""
        return self.orchestrator.get_orchestrator_stats()
    
    def get_task_classification_stats(self) -> Dict[str, Any]:
        """Get task classification statistics."""
        return self.orchestrator.task_classifier.get_classification_stats()
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get response synthesis statistics."""
        return self.orchestrator.response_synthesizer.get_synthesis_stats()
    
    def get_interception_stats(self) -> Dict[str, Any]:
        """Get communication interception statistics."""
        return self.orchestrator.communication_interceptor.get_interception_stats()
    
    async def handle_settings_modification(self, modification_request: str) -> str:
        """Handle settings modification through orchestrator."""
        # Route settings requests through orchestrator for consistency
        return await self.process_input(f"Change settings: {modification_request}")
    
    async def shutdown(self):
        """Shutdown the orchestrated conversation manager."""
        logger.info("Shutting down orchestrated conversation manager...")
        
        if self.orchestrator:
            await self.orchestrator.shutdown()
        
        # Clear conversation state
        self.conversation_history.clear()
        self.context_cache.clear()
        
        logger.info("Orchestrated conversation manager shutdown complete")
    
    # Compatibility methods for existing TUI integration
    @property
    def agent_orchestration_enabled(self) -> bool:
        """Compatibility property - always True for orchestrated mode."""
        return True
    
    @property
    def use_structured_processing(self) -> bool:
        """Compatibility property - orchestrator handles this internally."""
        return True
    
    def _should_use_structured_processing(self, user_input: str) -> bool:
        """Compatibility method - orchestrator handles classification internally."""
        # This method is no longer used since orchestrator handles all routing
        return False


def create_orchestrated_conversation_manager(command_interface=None, theme_manager=None, agent_manager=None) -> OrchestratedConversationManager:
    """Factory function to create orchestrated conversation manager."""
    return OrchestratedConversationManager(
        command_interface=command_interface,
        theme_manager=theme_manager, 
        agent_manager=agent_manager
    )