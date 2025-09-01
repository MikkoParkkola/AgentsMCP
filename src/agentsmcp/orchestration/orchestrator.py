"""
Strict Orchestrator Implementation for AgentsMCP

This orchestrator enforces the architectural principle that ONLY the orchestrator 
communicates directly with users. All agent communications are internal.

Key Principles:
- User ↔ Orchestrator ONLY (no direct agent-user communication)
- Orchestrator ↔ Agents (internal coordination) 
- Smart task classification to avoid unnecessary agent spawning
- Response synthesis for coherent user experience
- Communication isolation for clean architecture

The orchestrator acts as the single point of contact for users while coordinating
with agents behind the scenes to provide intelligent, consolidated responses.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .task_classifier import TaskClassifier, TaskClassification, ClassificationResult
from .response_synthesizer import ResponseSynthesizer, SynthesisStrategy
from .communication_interceptor import CommunicationInterceptor

logger = logging.getLogger(__name__)


class OrchestratorMode(Enum):
    """Orchestrator operation modes."""
    STRICT_ISOLATION = "strict_isolation"  # No direct agent-user communication allowed
    SUPERVISED = "supervised"  # Agent communications monitored and filtered
    TRANSPARENT = "transparent"  # Agent communications visible but orchestrator-mediated


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    mode: OrchestratorMode = OrchestratorMode.STRICT_ISOLATION
    
    # Task classification settings
    enable_smart_classification: bool = True
    simple_task_threshold: float = 0.8
    single_agent_threshold: float = 0.6
    
    # Response synthesis settings
    default_synthesis_strategy: SynthesisStrategy = SynthesisStrategy.SUMMARIZE
    synthesis_timeout_ms: int = 2000
    
    # Communication settings
    intercept_all_agent_output: bool = True
    allow_agent_status_messages: bool = False
    consolidate_error_messages: bool = True
    
    # Performance settings
    max_agent_wait_time_ms: int = 30000
    max_parallel_agents: int = 8
    
    # Fallback behavior
    fallback_to_simple_response: bool = True
    orchestrator_persona: str = "helpful AI assistant"


@dataclass 
class OrchestratorResponse:
    """Response from the orchestrator to the user."""
    content: str
    response_type: str  # "simple", "agent_delegated", "multi_agent", "error"
    agents_consulted: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """
    Strict Orchestrator that enforces single point of communication with users.
    
    This orchestrator ensures that users only ever see responses from the orchestrator
    perspective, while internally coordinating with specialized agents as needed.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None, 
                 conversation_manager=None, agent_manager=None):
        """Initialize the orchestrator."""
        self.config = config or OrchestratorConfig()
        self.conversation_manager = conversation_manager
        self.agent_manager = agent_manager
        
        # Core components
        self.task_classifier = TaskClassifier()
        self.response_synthesizer = ResponseSynthesizer()
        self.communication_interceptor = CommunicationInterceptor()
        
        # State tracking
        self.active_tasks: Dict[str, Any] = {}
        self.agent_pool: Dict[str, Any] = {}
        self.response_cache: Dict[str, OrchestratorResponse] = {}
        
        # Metrics
        self.total_requests = 0
        self.simple_responses = 0
        self.agent_delegations = 0
        self.multi_agent_tasks = 0
        
        logger.info(f"Orchestrator initialized in {self.config.mode.value} mode")
    
    async def process_user_input(self, user_input: str, context: Optional[Dict] = None) -> OrchestratorResponse:
        """
        Process user input through the orchestrator pipeline.
        
        This is the ONLY method that should be called by user interfaces.
        All other communication with agents happens internally.
        """
        start_time = time.time()
        self.total_requests += 1
        task_id = f"task_{self.total_requests}_{int(start_time)}"
        
        try:
            logger.info(f"Orchestrator processing: {user_input[:100]}...")
            
            # Step 1: Classify the task
            classification = await self.task_classifier.classify_task(
                user_input, context or {}
            )
            
            logger.info(f"Task classified as: {classification.classification.value} "
                        f"(confidence: {classification.confidence:.2f}, threshold: {self.config.simple_task_threshold})")
            
            # Step 2: Route based on classification
            if classification.classification == TaskClassification.SIMPLE_RESPONSE:
                logger.info(f"SIMPLE_RESPONSE path: confidence={classification.confidence}, threshold={self.config.simple_task_threshold}")
                # Check confidence - if low confidence simple response, delegate to agent instead
                if classification.confidence < self.config.simple_task_threshold:
                    logger.info(f"Low confidence simple response ({classification.confidence:.2f}), delegating to general agent")
                    # Create a modified classification for single agent
                    modified_classification = ClassificationResult(
                        classification=TaskClassification.SINGLE_AGENT_NEEDED,
                        confidence=0.7,  # Moderate confidence for general role
                        required_agents=["general"],
                        reasoning="Low confidence simple task redirected to general role",
                        task_complexity="moderate",
                        estimated_response_time="moderate"
                    )
                    response = await self._handle_single_agent_task(
                        user_input, context, modified_classification, task_id
                    )
                    self.agent_delegations += 1
                else:
                    response = await self._handle_simple_task(user_input, context, classification)
                    self.simple_responses += 1
                
            elif classification.classification == TaskClassification.SINGLE_AGENT_NEEDED:
                response = await self._handle_single_agent_task(
                    user_input, context, classification, task_id
                )
                self.agent_delegations += 1
                
            elif classification.classification == TaskClassification.MULTI_AGENT_NEEDED:
                response = await self._handle_multi_agent_task(
                    user_input, context, classification, task_id
                )
                self.multi_agent_tasks += 1
                
            else:
                # Fallback to simple response
                response = await self._generate_fallback_response(
                    user_input, "Unknown classification"
                )
            
            # Step 3: Finalize response
            processing_time = int((time.time() - start_time) * 1000)
            response.processing_time_ms = processing_time
            response.confidence_score = classification.confidence
            
            logger.info(f"Orchestrator completed task in {processing_time}ms")
            return response
            
        except Exception as e:
            logger.error(f"Orchestrator error processing input: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            
            return OrchestratorResponse(
                content=f"I encountered an error while processing your request. "
                       f"Let me help you with that in a different way. Could you rephrase your request?",
                response_type="error",
                processing_time_ms=processing_time,
                metadata={"error": str(e), "fallback_attempted": True}
            )
    
    async def _handle_simple_task(self, user_input: str, context: Optional[Dict], 
                                 classification) -> OrchestratorResponse:
        """Handle tasks that don't require agent delegation."""
        logger.debug("Handling simple task without agent spawning")
        
        # Common simple responses
        user_input_lower = user_input.lower().strip()
        
        if any(greeting in user_input_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            content = ("Hello! I'm your AgentsMCP assistant. I can help you with various tasks "
                      "from simple questions to complex project work. What would you like to work on today?")
                      
        elif any(help_word in user_input_lower for help_word in ["help", "what can you do", "capabilities"]):
            content = ("I can help you with:\n"
                      "• Code development and debugging\n"
                      "• Project planning and analysis\n" 
                      "• System configuration and setup\n"
                      "• Documentation and explanations\n"
                      "• Multi-step task coordination\n\n"
                      "Just describe what you'd like to accomplish and I'll coordinate "
                      "the right approach to help you.")
                      
        elif any(status_word in user_input_lower for status_word in ["status", "how are you", "working"]):
            content = (f"I'm running well! System status:\n"
                      f"• Processed {self.total_requests} requests\n"
                      f"• {self.simple_responses} simple responses\n"
                      f"• {self.agent_delegations} agent delegations\n"
                      f"• {self.multi_agent_tasks} multi-agent tasks\n\n"
                      f"Ready to help with your next request!")
                      
        elif any(agent_query in user_input_lower for agent_query in [
            "what agents", "which agents", "available agents", "list agents", "show agents",
            "what roles", "which roles", "available roles", "list roles", "show roles",
            "team", "specialists", "who can help", "what can you do", "capabilities"
        ]):
            content = self._generate_agent_roster_response()
            
        elif any(commands_query in user_input_lower for commands_query in [
            "what commands", "available commands", "list commands", "commands available",
            "what can i do", "how to use", "usage", "instructions"
        ]):
            content = self._generate_commands_help_response()
            
        elif any(project_query in user_input_lower for project_query in [
            "what project", "current project", "project info", "repository", "codebase",
            "what am i working on", "where am i", "project details"
        ]):
            content = self._generate_project_info_response(context)
            
        elif any(system_query in user_input_lower for system_query in [
            "system info", "system status", "configuration", "settings", "config"
        ]):
            content = self._generate_system_info_response()
            
        else:
            # Generic simple response  
            content = ("I understand you're asking about this topic. Let me provide some guidance. "
                      "If you need more detailed assistance, feel free to provide more context "
                      "about what specific help you're looking for.")
        
        return OrchestratorResponse(
            content=content,
            response_type="simple",
            confidence_score=classification.confidence
        )
    
    async def _handle_single_agent_task(self, user_input: str, context: Optional[Dict],
                                       classification, task_id: str) -> OrchestratorResponse:
        """Handle tasks requiring a single agent."""
        logger.info(f"🎯 SINGLE AGENT TASK - Required agents: {classification.required_agents}")
        
        try:
            # Get the recommended agent
            agent_type = classification.required_agents[0] if classification.required_agents else "ollama"
            logger.info(f"🤖 AGENT SELECTION - Using agent: {agent_type}")
            
            # Intercept the agent communication
            agent_response = await self._call_agent_safely(agent_type, user_input, context)
            
            # Synthesize the response from orchestrator perspective
            synthesized = await self.response_synthesizer.synthesize_responses(
                {agent_type: agent_response},
                user_input,
                self.config.default_synthesis_strategy
            )
            
            return OrchestratorResponse(
                content=synthesized.synthesized_response,
                response_type="agent_delegated",
                agents_consulted=[agent_type],
                metadata={
                    "agent_type": agent_type,
                    "synthesis_method": synthesized.synthesis_metadata.get("method_used"),
                    "task_id": task_id
                }
            )
            
        except Exception as e:
            logger.error(f"Single agent task failed: {e}")
            return await self._generate_fallback_response(
                user_input, f"Agent communication error: {str(e)}"
            )
    
    async def _handle_multi_agent_task(self, user_input: str, context: Optional[Dict],
                                      classification, task_id: str) -> OrchestratorResponse:
        """Handle tasks requiring multiple agents."""
        logger.info(f"Coordinating multi-agent task with: {classification.required_agents}")
        
        try:
            # Execute agents in parallel with timeout
            agent_responses = {}
            tasks = []
            
            for agent_type in classification.required_agents:
                task = asyncio.create_task(
                    self._call_agent_safely(agent_type, user_input, context)
                )
                tasks.append((agent_type, task))
            
            # Wait for all agents with timeout
            timeout = self.config.max_agent_wait_time_ms / 1000
            
            for agent_type, task in tasks:
                try:
                    response = await asyncio.wait_for(task, timeout=timeout)
                    agent_responses[agent_type] = response
                except asyncio.TimeoutError:
                    logger.warning(f"Agent {agent_type} timed out")
                    agent_responses[agent_type] = f"Agent {agent_type} response timed out"
                except Exception as e:
                    logger.error(f"Agent {agent_type} failed: {e}")
                    agent_responses[agent_type] = f"Agent {agent_type} encountered an error"
            
            # Synthesize all responses
            synthesized = await self.response_synthesizer.synthesize_responses(
                agent_responses,
                user_input, 
                SynthesisStrategy.COLLABORATIVE
            )
            
            return OrchestratorResponse(
                content=synthesized.synthesized_response,
                response_type="multi_agent",
                agents_consulted=list(agent_responses.keys()),
                metadata={
                    "agent_count": len(agent_responses),
                    "synthesis_method": synthesized.synthesis_metadata.get("method_used"),
                    "task_id": task_id
                }
            )
            
        except Exception as e:
            logger.error(f"Multi-agent task failed: {e}")
            return await self._generate_fallback_response(
                user_input, f"Multi-agent coordination error: {str(e)}"
            )
    
    async def _call_agent_safely(self, agent_type: str, prompt: str, context: Optional[Dict]) -> str:
        """Safely call an agent with interception and error handling."""
        logger.info(f"📞 CALLING AGENT - Type: {agent_type}, Prompt: {prompt[:100]}...")
        try:
            # This would integrate with the existing agent calling mechanism
            # For now, create a placeholder that integrates with existing systems
            if self.conversation_manager and hasattr(self.conversation_manager, '_delegate_to_mcp_agent_with_prompt'):
                logger.info(f"✅ CONVERSATION MANAGER AVAILABLE - Delegating to {agent_type}")
                response = await self.conversation_manager._delegate_to_mcp_agent_with_prompt(
                    agent_type, prompt
                )
            else:
                logger.warning(f"⚠️ NO CONVERSATION MANAGER - Using fallback for {agent_type}")
                # Fallback response
                response = (f"I would help you with that request using my {agent_type} capabilities. "
                          f"However, the agent system is not fully configured right now. "
                          f"Let me provide some general guidance instead: {prompt[:100]}...")
            
            logger.info(f"📥 AGENT RESPONSE - From {agent_type}: {response[:200]}...")
            
            # Intercept the response (remove agent identifiers, clean up)
            intercepted = self.communication_interceptor.intercept_response(
                agent_type, response, {}
            )
            
            logger.info(f"🔄 RESPONSE PROCESSED - Intercepted: {intercepted.processed_response.get('sanitized_content', response)[:200]}...")
            
            return intercepted.processed_response.get("sanitized_content", response)
            
        except Exception as e:
            logger.error(f"Error calling agent {agent_type}: {e}")
            return f"I encountered an issue while processing that request. Let me try a different approach."
    
    def _generate_agent_roster_response(self) -> str:
        """Generate a comprehensive response about available agents and their capabilities."""
        response = "I have access to the following specialized agents:\n\n"
        
        # Import role registry to get actual agent information
        try:
            from ..roles.registry import RoleRegistry
            from ..roles.base import RoleName
            
            registry = RoleRegistry()
            role_classes = registry.ROLE_CLASSES
            
            # Group agents by category for better organization
            categories = {
                "🏗️ **Architecture & Design:**": [
                    (RoleName.ARCHITECT, "System design and technical architecture"),
                ],
                "👨‍💻 **Development:**": [
                    (RoleName.BACKEND_ENGINEER, "Server-side development, databases, APIs"),
                    (RoleName.WEB_FRONTEND_ENGINEER, "React, Vue, web applications"),
                    (RoleName.TUI_FRONTEND_ENGINEER, "Terminal interfaces, CLI tools"),
                    (RoleName.API_ENGINEER, "REST/GraphQL API design and implementation"),
                    (RoleName.DEV_TOOLING_ENGINEER, "Development tools and automation"),
                    (RoleName.CI_CD_ENGINEER, "Continuous integration and deployment"),
                    (RoleName.CODER, "General programming and implementation"),
                ],
                "🔍 **Quality & Testing:**": [
                    (RoleName.BACKEND_QA_ENGINEER, "Server-side testing and validation"),
                    (RoleName.WEB_FRONTEND_QA_ENGINEER, "Web application testing"),
                    (RoleName.TUI_FRONTEND_QA_ENGINEER, "CLI/terminal testing"),
                    (RoleName.CHIEF_QA_ENGINEER, "Overall quality oversight"),
                    (RoleName.QA, "General quality assurance"),
                ],
                "📊 **Analysis & Strategy:**": [
                    (RoleName.BUSINESS_ANALYST, "Requirements analysis, process optimization"),
                    (RoleName.DATA_ANALYST, "Data processing and insights"),
                    (RoleName.DATA_SCIENTIST, "Statistical analysis and modeling"),
                    (RoleName.MARKETING_MANAGER, "Product positioning and strategy"),
                ],
                "🤖 **Machine Learning & AI:**": [
                    (RoleName.ML_SCIENTIST, "Machine learning research and development"),
                    (RoleName.ML_ENGINEER, "ML model deployment and infrastructure"),
                ],
                "⚖️ **Legal & Compliance:**": [
                    (RoleName.IT_LAWYER, "Legal compliance, licensing, privacy (GDPR)"),
                ],
                "🔧 **Operations & Automation:**": [
                    (RoleName.MERGE_BOT, "Code merging and release automation"),
                ],
            }
            
            for category, roles in categories.items():
                response += f"{category}\n"
                for role_name, description in roles:
                    if role_name in role_classes:
                        role_display_name = role_name.value.replace('_', ' ').title()
                        response += f"- {role_display_name} - {description}\n"
                response += "\n"
            
            response += ("**Available Agent Types:**\n"
                        "- Codex (OpenAI) - Best for complex reasoning and analysis\n"
                        "- Claude (Anthropic) - Excellent for large context and detailed work\n"
                        "- Ollama (Local) - Fast local processing for well-defined tasks\n\n"
                        "What type of project are you working on? I can recommend the best agents for your needs.")
            
        except Exception as e:
            # Fallback if role registry fails
            logger.warning(f"Failed to load role registry: {e}")
            response = ("I have access to a comprehensive team of specialized agents including:\n\n"
                      "• **Architecture & Planning** - System design and technical architecture\n"
                      "• **Development** - Backend, frontend, API, and tooling engineers\n"  
                      "• **Quality Assurance** - Testing specialists for all platforms\n"
                      "• **Analysis** - Business analysts, data scientists, and researchers\n"
                      "• **Machine Learning** - ML scientists and engineers\n"
                      "• **Legal & Compliance** - IT lawyers for licensing and privacy\n"
                      "• **Operations** - CI/CD and automation specialists\n\n"
                      "Each agent can work with different AI models (Codex, Claude, Ollama) "
                      "depending on the complexity and requirements of your task.\n\n"
                      "What type of project are you working on? I can recommend the best agents for your needs.")
        
        return response

    def _generate_commands_help_response(self) -> str:
        """Generate response about available commands and usage."""
        return ("**Available Commands:**\n\n"
               "**Chat Commands:**\n"
               "- `help` - Show help information and capabilities\n"
               "- `status` - Check system status and metrics\n"
               "- `agents` or `roles` - List all available specialized agents\n"
               "- `settings` - Configure AgentsMCP preferences\n"
               "- `theme [light|dark|auto]` - Change interface theme\n\n"
               "**Project Commands:**\n"
               "- `analyze` - Analyze current project/repository\n"
               "- `implement [description]` - Delegate complex coding tasks\n"
               "- `test` - Run tests and quality checks\n"
               "- `review` - Code review and suggestions\n\n"
               "**Usage Examples:**\n"
               "- \"What agents do you have?\" - Lists all available specialists\n"
               "- \"Analyze this project for improvements\" - Deep project analysis\n"
               "- \"Implement user authentication\" - Complex feature development\n"
               "- \"Help me optimize performance\" - Performance analysis and fixes\n\n"
               "Just describe what you want to accomplish in natural language, "
               "and I'll coordinate the right approach to help you!")

    def _generate_project_info_response(self, context: Optional[Dict]) -> str:
        """Generate response about current project information."""
        try:
            import os
            from pathlib import Path
            
            current_dir = Path.cwd()
            project_name = current_dir.name
            
            # Try to detect project type
            project_indicators = []
            if (current_dir / "pyproject.toml").exists():
                project_indicators.append("Python (pyproject.toml)")
            if (current_dir / "package.json").exists():
                project_indicators.append("Node.js (package.json)")
            if (current_dir / "Cargo.toml").exists():
                project_indicators.append("Rust (Cargo.toml)")
            if (current_dir / "pom.xml").exists():
                project_indicators.append("Java Maven (pom.xml)")
            if (current_dir / ".git").exists():
                project_indicators.append("Git repository")
            
            project_type = ", ".join(project_indicators) if project_indicators else "Unknown type"
            
            response = f"**Current Project Information:**\n\n"
            response += f"📁 **Project Name:** {project_name}\n"
            response += f"📍 **Location:** {current_dir}\n"
            response += f"🛠️ **Project Type:** {project_type}\n\n"
            
            # Add some basic stats
            try:
                file_count = len([f for f in current_dir.rglob("*") if f.is_file() and not any(part.startswith('.') for part in f.parts)])
                response += f"📊 **Files:** ~{file_count} files\n\n"
            except:
                pass
                
            response += ("**What I can help with:**\n"
                        "- Analyze project structure and suggest improvements\n"
                        "- Implement new features or fix bugs\n"
                        "- Review code quality and security\n"
                        "- Run tests and generate documentation\n"
                        "- Coordinate specialized agents for your project needs\n\n"
                        "Just describe what you'd like to work on!")
            
            return response
            
        except Exception as e:
            return ("**Current Project:** Unable to fully detect project details\n\n"
                   f"📍 **Working Directory:** {os.getcwd()}\n\n"
                   "I can still help you with:\n"
                   "- Code analysis and development\n"
                   "- Project improvements\n" 
                   "- Testing and quality assurance\n"
                   "- Documentation and architecture\n\n"
                   "What would you like to work on?")

    def _generate_system_info_response(self) -> str:
        """Generate response about system configuration and status."""
        try:
            # Get configuration info
            config_info = []
            
            # Check provider configuration
            try:
                from ..conversation.llm_client import LLMClient
                client = LLMClient()
                config_info.append(f"**LLM Provider:** {client.provider}")
                config_info.append(f"**Model:** {client.model}")
            except Exception:
                config_info.append("**LLM Provider:** Configuration not available")
            
            # System stats
            stats = self.get_orchestrator_stats()
            
            response = "**System Information:**\n\n"
            response += "\n".join(config_info) + "\n\n"
            
            response += "**Orchestrator Stats:**\n"
            response += f"- Mode: {stats['mode']}\n"
            response += f"- Total requests: {stats['total_requests']}\n"
            response += f"- Simple responses: {stats['simple_responses']}\n" 
            response += f"- Agent delegations: {stats['agent_delegations']}\n"
            response += f"- Multi-agent tasks: {stats['multi_agent_tasks']}\n"
            response += f"- Active tasks: {stats['active_tasks']}\n\n"
            
            response += "**Available Features:**\n"
            response += "✅ Multi-agent orchestration\n"
            response += "✅ Intelligent task classification\n" 
            response += "✅ Response synthesis\n"
            response += "✅ Communication isolation\n"
            response += "✅ Role-based agent selection\n\n"
            
            response += "System is running normally. How can I help you today?"
            
            return response
            
        except Exception as e:
            return (f"**System Status:** Running (limited info available)\n\n"
                   f"**Orchestrator Mode:** {self.config.mode.value}\n"
                   f"**Requests Handled:** {self.total_requests}\n\n"
                   "System is operational. What would you like me to help with?")

    async def _generate_fallback_response(self, user_input: str, error_context: str) -> OrchestratorResponse:
        """Generate a fallback response when agent delegation fails."""
        content = ("I understand what you're asking for, and I want to help you with that. "
                  "While I work on getting the best approach for your request, let me provide "
                  "some initial guidance. Could you tell me more specifically what you'd like to "
                  "accomplish? This will help me give you more targeted assistance.")
        
        return OrchestratorResponse(
            content=content,
            response_type="fallback",
            metadata={"error_context": error_context, "fallback_reason": "agent_unavailable"}
        )
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics."""
        return {
            "total_requests": self.total_requests,
            "simple_responses": self.simple_responses,
            "agent_delegations": self.agent_delegations, 
            "multi_agent_tasks": self.multi_agent_tasks,
            "mode": self.config.mode.value,
            "active_tasks": len(self.active_tasks),
            "cache_size": len(self.response_cache)
        }
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator."""
        logger.info("Orchestrator shutting down...")
        
        # Cancel any active tasks
        for task_id, task_data in self.active_tasks.items():
            if "task" in task_data and not task_data["task"].done():
                task_data["task"].cancel()
        
        self.active_tasks.clear()
        self.response_cache.clear()
        
        logger.info("Orchestrator shutdown complete")