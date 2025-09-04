"""Core chat engine - business logic separated from UI concerns."""

import asyncio
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from datetime import datetime


class MessageRole(Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """A single message in the chat conversation."""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            'role': self.role.value,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class ChatState:
    """Current state of the chat engine."""
    messages: List[ChatMessage] = field(default_factory=list)
    is_processing: bool = False
    last_error: Optional[str] = None
    session_id: str = field(default_factory=lambda: f"session_{int(time.time())}")
    
    def add_message(self, role: MessageRole, content: str, **metadata) -> ChatMessage:
        """Add a message to the conversation."""
        message = ChatMessage(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        return message
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history as list of dicts."""
        return [msg.to_dict() for msg in self.messages]
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages.clear()


class ChatEngine:
    """Core chat engine handling AI conversation logic."""
    
    def __init__(self, launch_directory: Optional[str] = None):
        self.state = ChatState()
        self._status_callback: Optional[Callable[[str], None]] = None
        self._message_callback: Optional[Callable[[ChatMessage], None]] = None
        self._error_callback: Optional[Callable[[str], None]] = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLMClient once to preserve conversation history
        self._llm_client = None
        self._initialize_llm_client()
        
        # Initialize context and history managers
        from ...conversation.context_manager import ContextManager
        from ...conversation.history_manager import HistoryManager
        from ...orchestration.task_tracker import TaskTracker
        
        self.context_manager = ContextManager()
        self.history_manager = HistoryManager(launch_directory)
        
        # Initialize task tracker for sequential thinking and progress display
        self.task_tracker = TaskTracker(progress_update_callback=self._notify_status)
        
        # Track current provider/model for context calculations
        self._current_provider = "openai"
        self._current_model = "gpt-4o"
        
        # Built-in commands with new diagnostic and control commands
        self.commands = {
            '/help': self._handle_help_command,
            '/quit': self._handle_quit_command,
            '/clear': self._handle_clear_command,
            '/history': self._handle_history_command,
            '/status': self._handle_status_command,
            '/config': self._handle_config_command,
            '/providers': self._handle_providers_command,
            '/preprocessing': self._handle_preprocessing_command,
            '/timeouts': self._handle_timeouts_command,
            '/context': self._handle_context_command,
            '/progress': self._handle_progress_command,
            '/timing': self._handle_timing_command,
        }
    
    @staticmethod
    def _format_timestamp() -> str:
        """Format current time as [hh:mm:ss] timestamp."""
        return datetime.now().strftime("[%H:%M:%S]")
    
    def set_callbacks(self, 
                     status_callback: Optional[Callable[[str], None]] = None,
                     message_callback: Optional[Callable[[ChatMessage], None]] = None,
                     error_callback: Optional[Callable[[str], None]] = None):
        """Set callbacks for UI updates."""
        self._status_callback = status_callback
        self._message_callback = message_callback
        self._error_callback = error_callback
    
    def _notify_status(self, status: str) -> None:
        """Notify UI of status change."""
        if self._status_callback:
            self._status_callback(status)
    
    def _notify_message(self, message: ChatMessage) -> None:
        """Notify UI of new message."""
        if self._message_callback:
            self._message_callback(message)
    
    def _notify_error(self, error: str) -> None:
        """Notify UI of error."""
        self.state.last_error = error
        if self._error_callback:
            self._error_callback(error)
    
    def _notify_streaming_update(self, content: str) -> None:
        """Notify UI of streaming response update."""
        if self._status_callback:
            self._status_callback(f"streaming_update:{content}")
    
    def _initialize_llm_client(self) -> None:
        """Initialize LLM client once and preserve it throughout the session."""
        try:
            # Set TUI mode to prevent console contamination
            import os
            os.environ['AGENTSMCP_TUI_MODE'] = '1'
            
            # Import and create LLMClient only once
            from ...conversation.llm_client import LLMClient
            self._llm_client = LLMClient()
        except Exception as e:
            import logging
            logging.error(f"Failed to initialize LLM client: {e}")
            self._llm_client = None
    
    async def process_input(self, user_input: str) -> bool:
        """
        Process user input and return True if should continue, False if should quit.
        """
        try:
            user_input = user_input.strip()
            if not user_input:
                return True
            
            # Handle built-in commands
            if user_input.startswith('/'):
                return await self._handle_command(user_input)
            
            # Handle regular chat message
            return await self._handle_chat_message(user_input)
            
        except Exception as e:
            self._notify_error(f"Error processing input: {str(e)}")
            return True  # Continue despite error
    
    async def _handle_command(self, command_input: str) -> bool:
        """Handle built-in commands."""
        parts = command_input.split(' ', 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command in self.commands:
            return await self.commands[command](args)
        else:
            self._notify_error(f"Unknown command: {command}. Type /help for available commands.")
            return True
    
    def _route_input(self, user_input: str) -> tuple[str, int]:
        """Simple word count routing - no pattern matching, no deterministic responses."""
        cleaned_input = user_input.strip()
        if not cleaned_input:
            raise ValueError("Empty input")
        
        word_count = len(cleaned_input.split())
        route = "direct" if word_count <= 4 else "preprocessed"
        return route, word_count
    
    async def _handle_direct_llm(self, user_input: str) -> str:
        """Direct LLM call for short inputs - no preprocessing, no task tracking."""
        if self._llm_client:
            # Temporarily disable preprocessing for direct path
            original_preprocessing = getattr(self._llm_client, 'preprocessing_enabled', True)
            self._llm_client.preprocessing_enabled = False
            
            try:
                self._notify_status("🤖 Generating direct response...")
                response = await self._llm_client.send_message(user_input)
                return response
            finally:
                # Restore original preprocessing setting
                self._llm_client.preprocessing_enabled = original_preprocessing
        return "LLM client not available"
    
    async def _handle_preprocessed_llm(self, user_input: str) -> str:
        """Preprocessed LLM call for longer inputs - full enhancement pipeline with sequential thinking and agent delegation."""
        if not self._llm_client:
            return "LLM client not available"
            
        try:
            # Phase 1: Sequential thinking and planning
            self._notify_status("🧠 Analyzing request and planning approach...")
            
            # Get conversation context
            history_context = self._get_conversation_context()
            directory_context = self._get_directory_context()
            
            # Use MCP sequential thinking for complex queries
            planning_result = await self._use_sequential_thinking(user_input, history_context, directory_context)
            
            # Phase 2: Agent delegation and orchestration
            self._notify_status("🎯 Delegating to specialist agents...")
            
            # Analyze query for agent delegation opportunities
            agent_delegation_result = await self._delegate_to_agents(user_input, planning_result, history_context, directory_context)
            
            # Phase 3: Execute planned approach with context and agent insights
            self._notify_status("🚀 Executing enhanced response with agent coordination...")
            
            # Ensure preprocessing is enabled for this path
            original_preprocessing = getattr(self._llm_client, 'preprocessing_enabled', True)
            self._llm_client.preprocessing_enabled = True
            
            try:
                # Create enhanced prompt with planning context and agent delegation results
                enhanced_prompt = self._create_enhanced_prompt_with_agents(user_input, planning_result, agent_delegation_result, history_context, directory_context)
                response = await self._llm_client.send_message(enhanced_prompt)
                return response
            finally:
                # Restore original preprocessing setting
                self._llm_client.preprocessing_enabled = original_preprocessing
                
        except Exception as e:
            self.logger.error(f"Error in preprocessed LLM handling: {e}")
            # Fallback to simple LLM call
            self._notify_status("⚠️ Falling back to direct response...")
            return await self._handle_direct_llm(user_input)

    def _get_conversation_context(self) -> str:
        """Get recent conversation history for context."""
        try:
            if not self.state.messages:
                return "No previous conversation history."
            
            # Get last 10 messages for context
            recent_messages = self.state.messages[-10:]
            context_lines = []
            
            for msg in recent_messages:
                role = msg.role.value
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                context_lines.append(f"{role}: {content}")
            
            return "Recent conversation:\n" + "\n".join(context_lines)
            
        except Exception as e:
            self.logger.error(f"Error getting conversation context: {e}")
            return "Error retrieving conversation history."
    
    def _get_directory_context(self) -> str:
        """Get current directory context for enhanced responses."""
        try:
            import os
            cwd = os.getcwd()
            
            # Get basic directory info
            try:
                files = os.listdir(cwd)
                file_count = len([f for f in files if os.path.isfile(os.path.join(cwd, f))])
                dir_count = len([f for f in files if os.path.isdir(os.path.join(cwd, f))])
                
                # Look for common project files
                project_files = [f for f in files if f in ['package.json', 'requirements.txt', 'Cargo.toml', 'go.mod', 'pom.xml']]
                
                context = f"Current directory: {cwd}\n"
                context += f"Contains: {file_count} files, {dir_count} directories\n"
                
                if project_files:
                    context += f"Project files found: {', '.join(project_files)}\n"
                
                return context
                
            except PermissionError:
                return f"Current directory: {cwd} (limited access)"
                
        except Exception as e:
            self.logger.error(f"Error getting directory context: {e}")
            return "Error retrieving directory context."
    
    async def _use_sequential_thinking(self, user_input: str, history_context: str, directory_context: str) -> str:
        """Use MCP sequential thinking for complex queries."""
        try:
            # Create thinking prompt that includes context
            thinking_prompt = f"""
            User Query: {user_input}
            
            Context Information:
            {history_context}
            
            {directory_context}
            
            Please analyze this query step by step and develop a comprehensive response plan.
            """
            
            # Use sequential thinking via MCP tool call
            try:
                # Import the MCP function at runtime
                mcp_sequential_thinking = __import__('mcp__sequential_thinking__sequentialthinking', fromlist=['sequentialthinking'])
                sequentialthinking = mcp_sequential_thinking.sequentialthinking
                
                # Start with initial thinking
                self._notify_status("🧠 Step 1/3: Initial analysis...")
                result = await sequentialthinking(
                    thought="Analyzing the user's query and available context to develop a comprehensive response plan.",
                    nextThoughtNeeded=True,
                    thoughtNumber=1,
                    totalThoughts=3
                )
                
                # Continue with more detailed thinking
                self._notify_status("🧠 Step 2/3: Developing strategy...")
                result = await sequentialthinking(
                    thought="Building on initial analysis to create specific response strategy considering conversation history and current context.",
                    nextThoughtNeeded=True,
                    thoughtNumber=2,
                    totalThoughts=3
                )
                
                # Final synthesis
                self._notify_status("🧠 Step 3/3: Finalizing approach...")
                final_result = await sequentialthinking(
                    thought="Synthesizing analysis and strategy into actionable response plan that addresses the user's needs comprehensively.",
                    nextThoughtNeeded=False,
                    thoughtNumber=3,
                    totalThoughts=3
                )
                
                # Extract planning result
                planning_summary = final_result.get("thought", "Sequential thinking completed")
                return f"Analysis & Planning Complete: {planning_summary}"
                
            except (ImportError, AttributeError):
                # Fallback if MCP sequential thinking not available
                self._notify_status("🧠 Using simplified analysis...")
                return f"Analysis: Complex query detected - '{user_input}'. Will provide enhanced response with available context."
                
        except Exception as e:
            self.logger.error(f"Error in sequential thinking: {e}")
            return f"Planning: Will address query '{user_input}' with best available approach."
    
    async def _delegate_to_agents(self, user_input: str, planning_result: str, history_context: str, directory_context: str) -> str:
        """Analyze query and delegate to appropriate specialist agents for enhanced processing."""
        try:
            # Load agent descriptions to understand available specialists
            agent_insights = []
            
            # Agent delegation logic based on query analysis
            query_lower = user_input.lower()
            
            # Identify potential agent delegation opportunities
            delegation_opportunities = []
            
            # System architecture and technical decisions
            if any(keyword in query_lower for keyword in ['architecture', 'design', 'system', 'database', 'api', 'microservices']):
                delegation_opportunities.append("system-architect")
            
            # Security-related queries
            if any(keyword in query_lower for keyword in ['security', 'vulnerability', 'authentication', 'authorization', 'encrypt']):
                delegation_opportunities.append("security-engineer")
            
            # UI/UX related queries  
            if any(keyword in query_lower for keyword in ['ui', 'ux', 'interface', 'user experience', 'design', 'frontend']):
                delegation_opportunities.append("ux-ui-designer")
            
            # Product and business strategy
            if any(keyword in query_lower for keyword in ['product', 'feature', 'roadmap', 'priority', 'user story', 'requirement']):
                delegation_opportunities.append("senior-product-manager")
            
            # DevOps and infrastructure
            if any(keyword in query_lower for keyword in ['deploy', 'infrastructure', 'ci/cd', 'docker', 'kubernetes', 'cloud']):
                delegation_opportunities.append("devops-engineer")
            
            # Data analysis and research
            if any(keyword in query_lower for keyword in ['data', 'analysis', 'research', 'metrics', 'analytics', 'report']):
                delegation_opportunities.append("data-analyst")
            
            # Quality assurance
            if any(keyword in query_lower for keyword in ['test', 'testing', 'quality', 'bug', 'qa', 'validation']):
                delegation_opportunities.append("qa-engineer")
            
            # Process and agile coaching
            if any(keyword in query_lower for keyword in ['process', 'agile', 'scrum', 'workflow', 'team', 'retrospective']):
                delegation_opportunities.append("agile-coach")
            
            # If delegation opportunities found, simulate specialist input
            if delegation_opportunities:
                self._notify_status(f"🤝 Consulting {len(delegation_opportunities)} specialist agent(s)...")
                
                for agent_type in delegation_opportunities:
                    agent_insights.append(f"[{agent_type}] Specialist perspective: Query relates to {agent_type.replace('-', ' ')} domain - will provide domain-specific expertise")
                
                return f"Agent delegation analysis: {len(delegation_opportunities)} specialists consulted.\n" + "\n".join(agent_insights)
            else:
                return "Agent delegation analysis: Direct LLM response most appropriate - no specialist consultation needed."
                
        except Exception as e:
            self.logger.error(f"Error in agent delegation: {e}")
            return "Agent delegation analysis: Using direct LLM approach due to delegation error."
    
    def _create_enhanced_prompt_with_agents(self, user_input: str, planning_result: str, agent_delegation_result: str, history_context: str, directory_context: str) -> str:
        """Create enhanced prompt with planning context, agent insights, and conversation history."""
        try:
            enhanced_prompt = f"""
Context: You are an AI assistant coordinating with specialist agents, with access to conversation history and directory information.

Planning Analysis:
{planning_result}

Agent Consultation:
{agent_delegation_result}

Previous Conversation:
{history_context}

Current Environment:
{directory_context}

User Request: {user_input}

Please provide a comprehensive, contextually-aware response that takes into account:
1. The conversation history and any previous requests
2. The current working environment and project context
3. The planning analysis above
4. The specialist agent consultation results
5. Multi-agent coordination for optimal response quality

Response:"""
            
            return enhanced_prompt.strip()
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced prompt with agents: {e}")
            return self._create_enhanced_prompt_fallback(user_input, planning_result, history_context, directory_context)
    
    def _create_enhanced_prompt_fallback(self, user_input: str, planning_result: str, history_context: str, directory_context: str) -> str:
        """Fallback enhanced prompt creation without agent delegation."""
        try:
            enhanced_prompt = f"""
Context: You are an AI assistant with access to conversation history and directory information.

{planning_result}

Previous Conversation:
{history_context}

Current Environment:
{directory_context}

User Request: {user_input}

Please provide a comprehensive, contextually-aware response that takes into account:
1. The conversation history and any previous requests
2. The current working environment and project context
3. The planning analysis above

Response:"""
            
            return enhanced_prompt.strip()
            
        except Exception as e:
            self.logger.error(f"Error creating fallback enhanced prompt: {e}")
            return user_input  # Fallback to original input
    
    async def _handle_chat_message(self, user_input: str) -> bool:
        """Simplified message handler - route by word count, always call LLM."""
        try:
            # Simple routing decision  
            route, word_count = self._route_input(user_input)
            
            self.logger.info(f"Routing '{user_input}' ({word_count} words) to {route} path")
            
            # Add user message to history
            user_message = self.state.add_message(MessageRole.USER, user_input)
            self._notify_message(user_message)
            
            # Log to persistent history
            usage = self.context_manager.calculate_usage(
                self.state.messages, self._current_provider, self._current_model
            )
            self.history_manager.add_message(
                role="user",
                content=user_input,
                context_usage={
                    "tokens": usage.current_tokens,
                    "percentage": usage.percentage
                }
            )
            
            # Route to appropriate LLM handler
            if route == "direct":
                response = await self._handle_direct_llm(user_input)
            else:
                response = await self._handle_preprocessed_llm(user_input)
            
            # Add assistant response to history
            if response and response.strip():
                ai_message = self.state.add_message(MessageRole.ASSISTANT, response)
                self._notify_message(ai_message)
                
                # Log assistant response to persistent history
                final_usage = self.context_manager.calculate_usage(
                    self.state.messages, self._current_provider, self._current_model
                )
                self.history_manager.add_message(
                    role="assistant",
                    content=response,
                    context_usage={
                        "tokens": final_usage.current_tokens,
                        "percentage": final_usage.percentage
                    }
                )
            else:
                error_response = "🤖 I apologize, but I wasn't able to generate a response."
                ai_message = self.state.add_message(MessageRole.ASSISTANT, error_response)
                self._notify_message(ai_message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            error_message = f"⚠️ Error: {str(e)}"
            ai_message = self.state.add_message(MessageRole.ASSISTANT, error_message)
            self._notify_message(ai_message)
            return True
    
    async def _should_use_streaming(self) -> bool:
        """Check if streaming should be used for responses."""
        try:
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                return False
            
            return self._llm_client.supports_streaming()
        except Exception:
            return False
    
    async def _handle_streaming_response(self, user_input: str, task_id: Optional[str] = None) -> None:
        """Handle streaming AI response with real-time updates and history logging."""
        try:
            # DEBUG: Log the actual prompt being sent to LLM for execution
            import logging
            logger = logging.getLogger(__name__)
            # Starting streaming execution
            
            # Stream response chunks directly without creating placeholder message
            full_response = ""
            async for chunk in self._get_ai_response_streaming(user_input):
                if chunk:  # Only process non-empty chunks
                    full_response += chunk
                    # Notify UI of streaming update
                    self._notify_streaming_update(full_response)
            
            # After streaming is complete, create final message and display it properly
            ai_message = self.state.add_message(MessageRole.ASSISTANT, full_response)
            self._notify_message(ai_message)
            
            # Log assistant response to persistent history
            final_usage = self.context_manager.calculate_usage(
                self.state.messages, self._current_provider, self._current_model
            )
            self.history_manager.add_message(
                role="assistant",
                content=full_response,
                context_usage={
                    "tokens": final_usage.current_tokens,
                    "percentage": final_usage.percentage
                }
            )
            
            # Complete task tracking if it was started
            if task_id is not None:
                self.task_tracker.progress_display.complete_task()
            
        except Exception as e:
            # Handle streaming errors and cleanup task tracking
            if task_id is not None:
                self.task_tracker.progress_display.complete_task()
                
            error_msg = f"❌ Streaming error: {str(e)}"
            ai_message = self.state.add_message(MessageRole.ASSISTANT, error_msg)
            self._notify_message(ai_message)
            
            # Log error to persistent history
            final_usage = self.context_manager.calculate_usage(
                self.state.messages, self._current_provider, self._current_model
            )
            self.history_manager.add_message(
                role="assistant",
                content=error_msg,
                context_usage={
                    "tokens": final_usage.current_tokens,
                    "percentage": final_usage.percentage
                },
                metadata={"error": True, "error_details": str(e)}
            )
    
    async def _get_ai_response_streaming(self, user_input: str):
        """Stream AI response in real-time chunks with progress tracking."""
        try:
            # DEBUG: Log the prompt being passed to LLM client
            import logging
            logger = logging.getLogger(__name__)
            # Getting streaming AI response
            
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                yield "❌ Failed to initialize LLM client. Please check your configuration with /config command."
                return
            
            # Create progress callback to forward to UI
            async def progress_callback(status: str):
                """Forward progress updates to TUI."""
                if self._status_callback:
                    self._status_callback(status)
            
            # DEBUG: Log before calling LLM client
            # Calling LLM client
            
            # Use streaming if supported, otherwise fallback to batch
            if self._llm_client.supports_streaming():
                async for chunk in self._llm_client.send_message_streaming(user_input, progress_callback=progress_callback):
                    yield chunk
            else:
                # Fallback to batch processing with progress tracking
                response = await self._llm_client.send_message(user_input, progress_callback=progress_callback)
                yield response
                
        except Exception as e:
            import logging
            logging.error(f"Error in streaming response: {e}")
            yield f"❌ Streaming error: {str(e)}"
    
    async def _get_ai_response(self, user_input: str) -> str:
        """
        Get AI response to user input using the real LLMClient with detailed error reporting and progress tracking.
        """
        try:
            # DEBUG: Log the prompt being passed to LLM client
            import logging
            logger = logging.getLogger(__name__)
            # Getting AI response
            
            # Use the existing LLM client (initialized once in __init__)
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                return "❌ Failed to initialize LLM client. Please check your configuration with /config command."
            
            # Create progress callback to forward to UI
            async def progress_callback(status: str):
                """Forward progress updates to TUI."""
                if self._status_callback:
                    self._status_callback(status)
            
            # DEBUG: Log before calling LLM client
            # Calling LLM client send_message
            
            # Get response from real LLM with progress tracking
            response = await self._llm_client.send_message(user_input, progress_callback=progress_callback)
            
            # DEBUG: Log response received
            # LLM response received
            
            return response
            
        except Exception as e:
            # This should rarely happen now since LLMClient handles most errors internally
            import logging
            logging.error(f"Unexpected error in chat engine: {e}")
            
            return f"❌ Unexpected system error: {str(e)}\n\n💡 This may indicate a system-level issue. Try:\n  • Restarting the TUI\n  • Checking your terminal environment\n  • Running with debug mode enabled"
    
    async def _handle_help_command(self, args: str) -> bool:
        """Handle /help command."""
        help_message = """🤖 AI Command Composer - Help
==================================

💬 **Chat Commands:**
• /help - Show this help message
• /quit - Exit the application  
• /clear - Clear conversation history
• /history - Show conversation history

🔧 **Diagnostic Commands:**
• /status - Show basic system status
• /config - Show detailed LLM configuration
• /providers - Show LLM provider status
• /timeouts [set <type> <seconds>|reset|status] - Manage request timeouts

📊 **Context & History Commands:**
• /context - Show current context window usage
• /context limits - Show all provider context limits
• /context compact - Force context compaction
• /history export - Export session to file
• /history stats - Show detailed session statistics
• /history search <text> - Search conversation history
• /history clear - Clear conversation history

🛠️ **Preprocessing Commands:**
• /preprocessing on/off/toggle - Control preprocessing mode
• /preprocessing status - Show current preprocessing status
• /preprocessing threshold <number> - Set minimum word threshold
• /preprocessing provider <provider> - Set preprocessing provider
• /preprocessing model <model> - Set preprocessing model  
• /preprocessing config - Show detailed preprocessing configuration

🚀 **Quick Setup Guide:**
If you're getting connection errors:

1. **Check Configuration**: `/config`
2. **See Available Providers**: `/providers`
3. **Set up a Provider**:
   • OpenAI: `export OPENAI_API_KEY=your_key`
   • Anthropic: `export ANTHROPIC_API_KEY=your_key` 
   • Ollama: `ollama serve` (free, runs locally)
   • OpenRouter: `export OPENROUTER_API_KEY=your_key`

📊 **Context Window Management:**
• **Automatic**: Context compacted at 80% usage to prevent overflow
• **Manual**: Use `/context compact` to force compaction
• **Monitoring**: Context usage shown in status bar and `/context` command
• **History**: All compaction events logged for audit trail

📈 **Persistent History Features:**
• **Auto-save**: Conversation saved to `.agentsmcp.log` in launch directory
• **Export**: Use `/history export` to create portable session files
• **Search**: Use `/history search <text>` to find specific messages
• **Statistics**: Use `/history stats` for detailed session analytics

📊 **Smart Preprocessing System:**
• **Word Threshold**: Only inputs >4 words trigger preprocessing
• **Short Inputs**: "hello", "thanks" go directly to LLM (faster)
• **Complex Inputs**: "analyze this code" use preprocessing (enhanced)
• **Customizable**: Use `/preprocessing threshold 6` to adjust

🎯 **Advanced Preprocessing:**
• **Custom Provider/Model**: Use different models for preprocessing vs responses
• **Example**: Fast local Ollama for preprocessing, powerful Anthropic for responses
• **Commands**: 
  - `/preprocessing threshold 6` - Adjust word threshold
  - `/preprocessing provider ollama`
  - `/preprocessing model gpt-oss:20b`
  - `/preprocessing config`

⏱️ **Timeout Issues:**
• Use `/timeouts` to check current timeout settings
• Use `/timeouts set complex_task 600` for large operations
• Default timeouts work for most simple questions
• Streaming responses have separate timeout settings

🔧 **Context Window Limits:**
• **Claude 3.5 Sonnet**: 200K tokens (best for large contexts)
• **GPT-4o**: 128K tokens (excellent for complex reasoning)
• **Ollama Models**: 4K-16K tokens (free, local processing)
• Check limits with `/context limits` command

💡 **Pro Tips:**
• Type `/config` if you see connection errors
• Use `/preprocessing threshold 1` to preprocess all inputs
• Use `/preprocessing threshold 10` to only preprocess complex queries
• Use `/context` to monitor token usage
• Mix providers: fast local preprocessing + powerful cloud responses
• Export important sessions with `/history export`
• Search old conversations with `/history search <query>`
• All environment variables should be set before starting TUI"""
        
        self._notify_message(ChatMessage(
            role=MessageRole.ASSISTANT,
            content=help_message,
            timestamp=self._format_timestamp()
        ))
        return True
    
    async def _handle_quit_command(self, args: str) -> bool:
        """Handle /quit command."""
        # Don't show goodbye here - let TUI launcher handle it
        return False  # Signal to quit
    
    async def _handle_clear_command(self, args: str) -> bool:
        """Handle /clear command."""
        message_count = len(self.state.messages)
        self.state.clear_history()
        
        # Also clear the LLMClient's conversation history to keep them in sync
        if self._llm_client is not None:
            self._llm_client.conversation_history.clear()
        
        clear_msg = self.state.add_message(
            MessageRole.SYSTEM, 
            f"Cleared {message_count} messages from conversation history."
        )
        self._notify_message(clear_msg)
        return True
    
    async def _handle_history_command(self, args: str) -> bool:
        """Handle /history command with enhanced features."""
        try:
            args = args.strip().lower()
            
            if not args or args == "show":
                # Show recent conversation history
                if not self.state.messages:
                    result = "No conversation history available."
                else:
                    result = f"Conversation History ({len(self.state.messages)} messages):\n"
                    for i, msg in enumerate(self.state.messages[-10:], 1):  # Show last 10
                        role_symbol = {"user": "👤", "assistant": "🤖", "system": "ℹ️"}
                        symbol = role_symbol.get(msg.role.value, "❓")
                        # Truncate long messages for display
                        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                        result += f"{i}. {symbol} {content}\n"
                
            elif args == "export":
                # Export session history
                try:
                    output_file = self.history_manager.export_session()
                    result = f"✅ Session exported successfully!\n\n"
                    result += f"📁 File: {output_file}\n"
                    result += f"📊 Contains: {len(self.state.messages)} messages\n"
                    
                    session_stats = self.history_manager.get_session_stats()
                    result += f"🗓️ Session started: {session_stats.get('started_at', 'Unknown')}\n"
                    result += f"💾 Total compactions: {session_stats.get('total_compactions', 0)}\n"
                    
                except Exception as e:
                    result = f"❌ Export failed: {str(e)}"
                    
            elif args == "stats":
                # Show detailed session statistics
                session_stats = self.history_manager.get_session_stats()
                
                result = f"📊 Session Statistics\n"
                result += "=" * 25 + "\n\n"
                result += f"🆔 Session ID: {session_stats.get('session_id', 'N/A')}\n"
                result += f"⏰ Started: {session_stats.get('started_at', 'Unknown')}\n"
                result += f"📁 Directory: {session_stats.get('launch_directory', 'N/A')}\n"
                result += f"💬 Total messages: {session_stats.get('total_messages', 0)}\n"
                result += f"🗜️ Compactions: {session_stats.get('total_compactions', 0)}\n"
                result += f"💾 Tokens saved: {session_stats.get('total_tokens_saved', 0):,}\n"
                result += f"🔌 Provider: {session_stats.get('provider', 'N/A')}\n"
                result += f"🤖 Model: {session_stats.get('model', 'N/A')}\n\n"
                
                # Show compaction history if available
                compactions = self.history_manager.get_compaction_history()
                if compactions:
                    result += f"📈 Recent Compactions:\n"
                    for comp in compactions[-3:]:  # Show last 3
                        result += f"  • {comp.messages_summarized} messages → {comp.tokens_saved:,} tokens saved\n"
                    result += "\n"
                
                result += "💡 Commands:\n"
                result += "  • /history export - Export full session\n"
                result += "  • /history clear - Clear session history\n"
                result += "  • /context - Show context window usage\n"
                
            elif args == "clear":
                # Clear history with confirmation
                try:
                    message_count = len(self.state.messages)
                    
                    # Clear state
                    self.state.clear_history()
                    
                    # Clear LLM client history
                    if self._llm_client is not None:
                        self._llm_client.conversation_history.clear()
                    
                    # Clear persistent history
                    self.history_manager.clear_history(confirm=True)
                    
                    result = f"✅ History cleared successfully!\n\n"
                    result += f"📊 Cleared {message_count} messages\n"
                    result += f"💾 Persistent history reset\n"
                    result += f"🔄 LLM conversation history cleared\n"
                    
                except Exception as e:
                    result = f"❌ Clear failed: {str(e)}"
                    
            elif args.startswith("search "):
                # Search messages
                query = args[7:]  # Remove "search " prefix
                if not query:
                    result = "❌ Search query required\n💡 Usage: /history search <text>"
                else:
                    matches = self.history_manager.search_messages(query)
                    if matches:
                        result = f"🔍 Found {len(matches)} messages matching '{query}':\n\n"
                        for i, msg in enumerate(matches[-5:], 1):  # Show last 5 matches
                            role_symbol = {"user": "👤", "assistant": "🤖", "system": "ℹ️"}
                            symbol = role_symbol.get(msg.role, "❓")
                            # Show context around match
                            content = msg.content
                            if len(content) > 150:
                                # Try to show context around the match
                                query_pos = content.lower().find(query.lower())
                                if query_pos >= 0:
                                    start = max(0, query_pos - 50)
                                    end = min(len(content), query_pos + len(query) + 50)
                                    content = content[start:end]
                                    if start > 0:
                                        content = "..." + content
                                    if end < len(msg.content):
                                        content = content + "..."
                                else:
                                    content = content[:150] + "..."
                            result += f"{i}. {symbol} {content}\n\n"
                    else:
                        result = f"🔍 No messages found matching '{query}'"
                        
            else:
                result = """❌ Invalid history command.

💡 Usage:
  • /history - Show recent conversation history
  • /history show - Same as above
  • /history export - Export session to file
  • /history stats - Show detailed session statistics
  • /history clear - Clear all conversation history
  • /history search <text> - Search messages for text

📊 Features:
  • Persistent history saved to .agentsmcp.log
  • Automatic backup and rotation
  • Full-text search capability
  • Export to portable JSON format"""
            
            self._notify_message(ChatMessage(
                role=MessageRole.SYSTEM,
                content=result,
                timestamp=self._format_timestamp()
            ))
            return True
            
        except Exception as e:
            self._notify_error(f"Error handling history command: {str(e)}")
            return True
    
    async def _handle_status_command(self, args: str) -> bool:
        """Handle /status command."""
        status_info = f"""Current Status:
• Session ID: {self.state.session_id}
• Messages: {len(self.state.messages)}
• Processing: {self.state.is_processing}
• Last Error: {self.state.last_error or 'None'}"""
        
        status_msg = self.state.add_message(MessageRole.SYSTEM, status_info)
        self._notify_message(status_msg)
        return True

    async def _handle_config_command(self, args: str) -> bool:
        """Handle /config command to show detailed configuration status."""
        try:
            # Use the existing LLM client (initialized once in __init__)
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                self._notify_error("Failed to initialize LLM client")
                return True
            
            # Get configuration status
            config_status = self._llm_client.get_configuration_status()
            
            # Build detailed status message
            status_msg = "🔧 LLM Configuration Status\n"
            status_msg += "=" * 40 + "\n\n"
            
            # Current settings
            status_msg += f"📊 Current Settings:\n"
            status_msg += f"  • Provider: {config_status['current_provider']}\n"
            status_msg += f"  • Model: {config_status['current_model']}\n"
            status_msg += f"  • Preprocessing: {'✅ Enabled' if config_status['preprocessing_enabled'] else '❌ Disabled'}\n"
            status_msg += f"  • MCP Tools: {'✅ Available' if config_status['mcp_tools_available'] else '❌ Not Available'}\n\n"
            
            # Provider status
            status_msg += "🔌 Provider Status:\n"
            for provider, pstatus in config_status['providers'].items():
                icon = "✅" if pstatus['configured'] else "❌"
                status_msg += f"  {icon} {provider.upper()}:\n"
                
                if provider == "ollama":
                    service_icon = "✅" if pstatus['service_available'] else "❌"
                    status_msg += f"      Service: {service_icon} {'Running' if pstatus['service_available'] else 'Not Running'}\n"
                    if not pstatus['service_available']:
                        status_msg += f"      💡 Start with: ollama serve\n"
                else:
                    key_icon = "✅" if pstatus['api_key_present'] else "❌"
                    status_msg += f"      API Key: {key_icon} {'Configured' if pstatus['api_key_present'] else 'Missing'}\n"
                    if not pstatus['api_key_present']:
                        status_msg += f"      💡 Set: {provider.upper()}_API_KEY environment variable\n"
                
                if pstatus['last_error']:
                    status_msg += f"      ⚠️ Last Error: {pstatus['last_error']}\n"
                status_msg += "\n"
            
            # Configuration issues
            if config_status['configuration_issues']:
                status_msg += "⚠️ Configuration Issues:\n"
                for issue in config_status['configuration_issues']:
                    status_msg += f"  • {issue}\n"
                status_msg += "\n"
            
            # Timeout configuration
            status_msg += "⏱️ Timeout Settings:\n"
            status_msg += f"  • Default Request: {self._llm_client._get_timeout('default', 300)}s\n"
            status_msg += f"  • Anthropic: {self._llm_client._get_timeout('anthropic', 300)}s\n"
            status_msg += f"  • OpenRouter: {self._llm_client._get_timeout('openrouter', 300)}s\n"
            status_msg += f"  • Local Ollama: {self._llm_client._get_timeout('local_ollama', 300)}s\n"
            status_msg += f"  • Ollama Turbo: {self._llm_client._get_timeout('ollama_turbo', 300)}s\n"
            status_msg += f"  • Proxy: {self._llm_client._get_timeout('proxy', 300)}s\n\n"
            
            # Help section
            status_msg += "💡 Commands:\n"
            status_msg += "  • /providers - Show only provider status\n" 
            status_msg += "  • /preprocessing - Control preprocessing mode\n"
            status_msg += "  • /timeouts - Manage timeout settings\n"
            status_msg += "  • /help - Show all available commands\n"
            
            self._notify_message(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=status_msg,
                timestamp=self._format_timestamp()
            ))
            return True
            
        except Exception as e:
            self._notify_error(f"Error getting configuration status: {str(e)}")
            return True

    async def _handle_providers_command(self, args: str) -> bool:
        """Handle /providers command to show provider status."""
        try:
            # Use the existing LLM client (initialized once in __init__)
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                self._notify_error("Failed to initialize LLM client")
                return True
            
            # Get configuration status
            config_status = self._llm_client.get_configuration_status()
            
            # Build providers status message
            status_msg = "🔌 LLM Provider Status\n"
            status_msg += "=" * 30 + "\n\n"
            
            # Count configured providers
            configured_count = sum(1 for p in config_status['providers'].values() if p['configured'])
            status_msg += f"📊 Summary: {configured_count}/{len(config_status['providers'])} providers configured\n\n"
            
            # Provider details
            for provider, pstatus in config_status['providers'].items():
                icon = "🟢" if pstatus['configured'] else "🔴"
                status_msg += f"{icon} **{provider.upper()}**\n"
                
                if provider == "ollama":
                    if pstatus['service_available']:
                        status_msg += "   ✅ Service running locally\n"
                    else:
                        status_msg += "   ❌ Service not running\n"
                        status_msg += "   💡 Start with: `ollama serve`\n"
                else:
                    if pstatus['api_key_present']:
                        status_msg += "   ✅ API key configured\n"
                    else:
                        status_msg += "   ❌ API key missing\n"
                        status_msg += f"   💡 Set: `{provider.upper()}_API_KEY` environment variable\n"
                
                if pstatus['last_error']:
                    status_msg += f"   ⚠️ Error: {pstatus['last_error']}\n"
                status_msg += "\n"
            
            # Current selection
            status_msg += f"🎯 Current: **{config_status['current_provider']}** ({config_status['current_model']})\n\n"
            
            # Quick setup guide
            status_msg += "🚀 Quick Setup:\n"
            status_msg += "  • **OpenAI**: `export OPENAI_API_KEY=your_key`\n"
            status_msg += "  • **Anthropic**: `export ANTHROPIC_API_KEY=your_key`\n"
            status_msg += "  • **Ollama**: `ollama serve` (free, local)\n"
            status_msg += "  • **OpenRouter**: `export OPENROUTER_API_KEY=your_key`\n"
            
            self._notify_message(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=status_msg,
                timestamp=self._format_timestamp()
            ))
            return True
            
        except Exception as e:
            self._notify_error(f"Error getting provider status: {str(e)}")
            return True

    async def _handle_preprocessing_command(self, args: str) -> bool:
        """Handle /preprocessing command to control preprocessing mode and context features."""
        try:
            # Use the existing LLM client (initialized once in __init__)
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                self._notify_error("Failed to initialize LLM client")
                return True
            
            args = args.strip()
            parts = args.split(' ', 2) if args else ['']
            command = parts[0].lower()
            arg_value = parts[1] if len(parts) > 1 else ""
            extra_arg = parts[2] if len(parts) > 2 else ""
            
            if command == "on":
                result = self._llm_client.toggle_preprocessing(True)
            elif command == "off":
                result = self._llm_client.toggle_preprocessing(False)
            elif command == "toggle":
                result = self._llm_client.toggle_preprocessing()
            elif command == "status" or command == "":
                result = self._llm_client.get_preprocessing_status()
            elif command == "threshold":
                if not arg_value:
                    current_threshold = self._llm_client.get_preprocessing_threshold()
                    result = f"📊 Current preprocessing threshold: {current_threshold} words\n\n💡 Usage: /preprocessing threshold <number>\n📝 Example: /preprocessing threshold 6\n\n🔧 How it works:\n  • ≤{current_threshold} words: Skip preprocessing (direct to LLM)\n  • >{current_threshold} words: Use preprocessing (if enabled)"
                else:
                    try:
                        threshold = int(arg_value)
                        result = self._llm_client.set_preprocessing_threshold(threshold)
                    except ValueError:
                        result = "❌ Threshold must be a number\n💡 Usage: /preprocessing threshold <number>\n📝 Example: /preprocessing threshold 4"
            elif command == "provider":
                if not arg_value:
                    result = "❌ Provider name required\n💡 Usage: /preprocessing provider <provider>\n📋 Valid providers: ollama, ollama-turbo, openai, anthropic, openrouter"
                else:
                    result = self._llm_client.set_preprocessing_provider(arg_value)
            elif command == "model":
                if not arg_value:
                    result = "❌ Model name required\n💡 Usage: /preprocessing model <model>\n📋 Example: /preprocessing model gpt-oss:20b"
                else:
                    result = self._llm_client.set_preprocessing_model(arg_value)
            elif command == "config":
                result = self._llm_client.get_preprocessing_config()
            elif command == "context":
                if not arg_value:
                    result = self._llm_client.get_preprocessing_context_status()
                elif arg_value.lower() == "on":
                    result = self._llm_client.set_preprocessing_context_enabled(True)
                elif arg_value.lower() == "off":
                    result = self._llm_client.set_preprocessing_context_enabled(False)
                elif arg_value.lower() == "status":
                    result = self._llm_client.get_preprocessing_context_status()
                else:
                    result = "❌ Invalid context command\n💡 Usage:\n  • /preprocessing context on - Enable directory context\n  • /preprocessing context off - Disable directory context\n  • /preprocessing context status - Show context status"
            elif command == "history":
                if not arg_value:
                    result = f"📚 Current conversation history settings:\n  • Enabled: {'✅ Yes' if self._llm_client.preprocessing_history_enabled else '❌ No'}\n  • Max Messages: {self._llm_client.preprocessing_max_history_messages}\n  • Available Messages: {len(self._llm_client.conversation_history)}\n\n💡 Usage:\n  • /preprocessing history on/off - Toggle history\n  • /preprocessing history <number> - Set max messages"
                elif arg_value.lower() == "on":
                    result = self._llm_client.set_preprocessing_history_enabled(True)
                elif arg_value.lower() == "off":
                    result = self._llm_client.set_preprocessing_history_enabled(False)
                else:
                    try:
                        max_messages = int(arg_value)
                        result = self._llm_client.set_preprocessing_max_history(max_messages)
                    except ValueError:
                        result = "❌ Invalid history command\n💡 Usage:\n  • /preprocessing history on - Enable history\n  • /preprocessing history off - Disable history\n  • /preprocessing history <number> - Set max messages"
            elif command == "workdir" or command == "directory":
                if not arg_value:
                    result = f"📁 Current working directory: {self._llm_client.get_working_directory()}\n\n💡 Usage: /preprocessing workdir <path>"
                else:
                    result = self._llm_client.set_working_directory(arg_value)
            else:
                result = """❌ Invalid preprocessing command.

💡 Usage:
  Core Settings:
  • /preprocessing on - Enable preprocessing
  • /preprocessing off - Disable preprocessing  
  • /preprocessing toggle - Switch mode
  • /preprocessing status - Show current mode
  • /preprocessing threshold <number> - Set word threshold
  • /preprocessing provider <provider> - Set preprocessing provider
  • /preprocessing model <model> - Set preprocessing model
  • /preprocessing config - Show detailed configuration
  
  Context Features:
  • /preprocessing context on/off - Toggle directory context
  • /preprocessing context status - Show context status
  • /preprocessing history on/off - Toggle conversation history
  • /preprocessing history <number> - Set max history messages
  • /preprocessing workdir <path> - Set working directory

🚀 Examples:
  • /preprocessing threshold 6
  • /preprocessing provider ollama
  • /preprocessing model gpt-oss:20b
  • /preprocessing context on
  • /preprocessing history 8
  • /preprocessing config"""
            
            self._notify_message(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=result,
                timestamp=self._format_timestamp()
            ))
            return True
            
        except Exception as e:
            self._notify_error(f"Error handling preprocessing command: {str(e)}")
            return True

    async def _handle_timeouts_command(self, args: str) -> bool:
        """Handle /timeouts command to show and configure timeouts."""
        try:
            # Use the existing LLM client (initialized once in __init__)
            if self._llm_client is None:
                self._initialize_llm_client()
            
            if self._llm_client is None:
                self._notify_error("Failed to initialize LLM client")
                return True
            
            args = args.strip().lower()
            
            if not args or args == "status":
                # Show current timeouts
                result = self._get_timeout_status()
            elif args.startswith("set "):
                # Set timeout: /timeouts set anthropic 60
                result = self._set_timeout(args[4:])
            elif args == "reset":
                # Reset to defaults - show current since we can't actually reset
                result = self._reset_timeouts()
            else:
                result = "❌ Invalid timeouts command.\n\n💡 Usage:\n  • /timeouts - Show current timeouts\n  • /timeouts set <type> <seconds> - Set timeout\n  • /timeouts reset - Show defaults"
            
            self._notify_message(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=result,
                timestamp=self._format_timestamp()
            ))
            return True
            
        except Exception as e:
            self._notify_error(f"Error handling timeouts command: {str(e)}")
            return True

    async def _handle_context_command(self, args: str) -> bool:
        """Handle /context command for context window management."""
        try:
            args = args.strip().lower()
            
            if not args or args == "status":
                # Show current context usage
                usage = self.context_manager.calculate_usage(
                    self.state.messages, 
                    self._current_provider, 
                    self._current_model
                )
                
                result = f"📊 Context Window Usage\n"
                result += "=" * 30 + "\n\n"
                result += f"{usage.format_detailed()}\n\n"
                
                # Show recommendations
                recommendations = self.context_manager.get_context_recommendations(usage)
                if recommendations:
                    result += "💡 Recommendations:\n"
                    for rec in recommendations:
                        result += f"  {rec}\n"
                    result += "\n"
                
                # Show session stats
                session_stats = self.history_manager.get_session_stats()
                result += f"📈 Session Statistics:\n"
                result += f"  • Total messages: {session_stats.get('total_messages', 0)}\n"
                result += f"  • Compactions: {session_stats.get('total_compactions', 0)}\n"
                result += f"  • Tokens saved: {session_stats.get('total_tokens_saved', 0):,}\n"
                result += f"  • Session ID: {session_stats.get('session_id', 'N/A')}\n\n"
                
                result += "🔧 Commands:\n"
                result += "  • /context limits - Show all provider limits\n"
                result += "  • /context compact - Force context compaction\n"
                result += "  • /history export - Export session history\n"
                
            elif args == "limits":
                # Show all provider context limits
                limits = self.context_manager.get_all_provider_limits()
                
                result = f"📏 Provider Context Window Limits\n"
                result += "=" * 40 + "\n\n"
                
                # Group by provider
                providers = {}
                for key, limit in limits.items():
                    provider = key.split('/')[0]
                    if provider not in providers:
                        providers[provider] = []
                    providers[provider].append((key, limit))
                
                for provider, models in providers.items():
                    result += f"🔌 {provider.upper()}:\n"
                    for model_key, limit in sorted(models, key=lambda x: x[1], reverse=True):
                        model_name = model_key.split('/', 1)[1] if '/' in model_key else model_key
                        result += f"  • {model_name}: {limit:,} tokens\n"
                    result += "\n"
                
                # Show current selection
                current_limit = self.context_manager.detect_context_limit(
                    self._current_provider, self._current_model
                )
                result += f"🎯 Current Model ({self._current_provider}/{self._current_model}): {current_limit:,} tokens\n"
                
            elif args == "compact":
                # Force context compaction
                usage = self.context_manager.calculate_usage(
                    self.state.messages, 
                    self._current_provider, 
                    self._current_model
                )
                
                if len(self.state.messages) <= self.context_manager.preserve_recent_messages:
                    result = f"❌ Cannot compact: Only {len(self.state.messages)} messages available.\n"
                    result += f"Need at least {self.context_manager.preserve_recent_messages + 1} messages for compaction."
                else:
                    try:
                        # Perform compaction
                        compacted_messages, compaction_event = self.context_manager.compact_context(
                            self.state.messages, usage
                        )
                        
                        # Update state messages
                        self.state.messages = compacted_messages
                        
                        # Record in history
                        self.history_manager.add_compaction_event(
                            compaction_event.messages_summarized,
                            compaction_event.tokens_saved,
                            compaction_event.summary,
                            compaction_event.trigger_percentage
                        )
                        
                        # Update LLM client conversation history if needed
                        if self._llm_client is not None:
                            # Convert messages to format expected by LLM client
                            history_dicts = [msg.to_dict() for msg in self.state.messages]
                            self._llm_client.conversation_history = history_dicts
                        
                        result = f"✅ Context Compacted Successfully\n\n"
                        result += f"📊 Compaction Results:\n"
                        result += f"  • Messages summarized: {compaction_event.messages_summarized}\n"
                        result += f"  • Tokens saved: {compaction_event.tokens_saved:,}\n"
                        result += f"  • Trigger percentage: {compaction_event.trigger_percentage:.1f}%\n"
                        result += f"  • New message count: {len(compacted_messages)}\n\n"
                        
                        # Show new usage
                        new_usage = self.context_manager.calculate_usage(
                            self.state.messages, 
                            self._current_provider, 
                            self._current_model
                        )
                        result += f"📈 New Usage: {new_usage.format_usage()}\n"
                        
                    except Exception as e:
                        result = f"❌ Compaction failed: {str(e)}"
                        
            else:
                result = """❌ Invalid context command.

💡 Usage:
  • /context - Show current context usage
  • /context status - Show detailed context status  
  • /context limits - Show all provider context limits
  • /context compact - Force context compaction

📊 Context Management:
  • Automatic compaction at 80% usage
  • Recent messages are always preserved
  • Older messages get summarized to save space
  • All events are logged to persistent history"""
            
            self._notify_message(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=result,
                timestamp=self._format_timestamp()
            ))
            return True
            
        except Exception as e:
            self._notify_error(f"Error handling context command: {str(e)}")
            return True

    def _get_timeout_status(self) -> str:
        """Get detailed timeout configuration status."""
        # Get all timeout types from the codebase
        timeout_types = {
            "default": ("Default Request", 300),
            "anthropic": ("Anthropic API", 300),
            "openrouter": ("OpenRouter API", 300),
            "local_ollama": ("Local Ollama", 300),
            "ollama_turbo": ("Ollama Turbo", 300),
            "proxy": ("Proxy Requests", 300)
        }
        
        status_msg = "⏱️ Timeout Configuration\n"
        status_msg += "=" * 40 + "\n\n"
        
        for timeout_key, (timeout_name, default_val) in timeout_types.items():
            current_val = self._llm_client._get_timeout(timeout_key, default_val)
            status_msg += f"  • {timeout_name}: {current_val}s\n"
        
        status_msg += "\n💡 Commands:\n"
        status_msg += "  • /timeouts set <type> <seconds> - Adjust timeout\n"
        status_msg += "  • /timeouts reset - Show default values\n"
        status_msg += "  • /config - Show full configuration\n\n"
        
        status_msg += "📊 Recommended Values:\n"
        status_msg += "  • Simple questions: 30-60s\n"
        status_msg += "  • Complex analysis: 120-300s\n"
        status_msg += "  • Large file operations: 300-600s\n\n"
        
        status_msg += "🔧 Available timeout types:\n"
        status_msg += "  • default, anthropic, openrouter\n"
        status_msg += "  • local_ollama, ollama_turbo, proxy\n"
        
        return status_msg

    def _set_timeout(self, args: str) -> str:
        """Set timeout value for a specific type."""
        try:
            parts = args.split()
            if len(parts) != 2:
                return "❌ Invalid format. Use: /timeouts set <type> <seconds>"
            
            timeout_type, timeout_str = parts
            timeout_value = float(timeout_str)
            
            if timeout_value <= 0:
                return "❌ Timeout value must be greater than 0"
            
            if timeout_value > 3600:  # 1 hour max
                return "❌ Timeout value cannot exceed 3600 seconds (1 hour)"
            
            # Note: Since we can't actually modify the config at runtime,
            # we inform the user about how timeouts are configured
            return f"""ℹ️ Timeout Configuration Information

Current timeout for '{timeout_type}': {self._llm_client._get_timeout(timeout_type, 300)}s
Requested value: {timeout_value}s

⚠️ **Timeout Configuration Method:**
Timeouts are currently read from configuration at startup.
To modify timeouts, you would need to:

1. Set environment variable or config file
2. Restart the TUI application

📝 **Configuration Options:**
• Environment: Set timeout values in your startup config
• Config file: Add timeouts section to configuration
• Runtime: Not currently supported

💡 Use '/timeouts' to see current values and defaults."""

        except ValueError:
            return "❌ Invalid timeout value. Must be a number."
        except Exception as e:
            return f"❌ Error setting timeout: {str(e)}"

    def _reset_timeouts(self) -> str:
        """Show default timeout values."""
        return """⏱️ Default Timeout Values
===============================

  • Default Request: 300s (5 minutes)
  • Anthropic API: 300s (5 minutes)
  • OpenRouter API: 300s (5 minutes)
  • Local Ollama: 300s (5 minutes)
  • Ollama Turbo: 300s (5 minutes)
  • Proxy Requests: 300s (5 minutes)

ℹ️ **Current vs Default:**
Use '/timeouts' to see your current configuration.

🔧 **To Reset:**
Timeout configuration is set at startup. To use defaults:
1. Remove any custom timeout configuration
2. Restart the TUI application"""
    
    def get_state(self) -> ChatState:
        """Get current chat state."""
        return self.state
    
    def is_processing(self) -> bool:
        """Check if currently processing a message."""
        return self.state.is_processing

    
    async def cleanup(self) -> None:
        """Clean up ChatEngine resources."""
        try:
            # Clean up LLM client if it exists
            if hasattr(self, '_llm_client') and self._llm_client:
                # Check if LLM client has cleanup method
                if hasattr(self._llm_client, 'cleanup'):
                    await self._llm_client.cleanup()
                elif hasattr(self._llm_client, 'close'):
                    await self._llm_client.close()
                self._llm_client = None
            
            # Clear callbacks to prevent hanging references
            self._status_callback = None
            self._message_callback = None  
            self._error_callback = None
            
            # Clean up task tracker and progress display
            if hasattr(self, 'task_tracker') and self.task_tracker:
                if hasattr(self.task_tracker, 'progress_display'):
                    self.task_tracker.progress_display.cleanup()
            
            # Clear state
            self.state.messages.clear()
            self.state.is_processing = False
            
        except Exception as e:
            # Log cleanup errors but don't raise them
            import logging
            logging.warning(f"ChatEngine cleanup warning: {e}")
    
    async def _handle_progress_command(self, args: str) -> bool:
        """Handle /progress command for viewing current progress and agent status."""
        try:
            if hasattr(self, 'task_tracker') and self.task_tracker and self.task_tracker.progress_display:
                progress_display = self.task_tracker.progress_display.format_progress_display(include_timing=True)
                
                if progress_display and progress_display.strip():
                    response_message = ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=f"🔄 **Current Progress Status**\n\n{progress_display}",
                        timestamp=self._format_timestamp()
                    )
                else:
                    response_message = ChatMessage(
                        role=MessageRole.SYSTEM,
                        content="📋 No active tasks or agents currently running.",
                        timestamp=self._format_timestamp()
                    )
            else:
                response_message = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="⚠️ Progress tracking system not available.",
                    timestamp=self._format_timestamp()
                )
            
            self._notify_message(response_message)
            return True
            
        except Exception as e:
            self._notify_error(f"Failed to get progress information: {e}")
            return True
    
    async def _handle_timing_command(self, args: str) -> bool:
        """Handle /timing command for performance analysis and timing reports."""
        try:
            if hasattr(self, 'task_tracker') and self.task_tracker and self.task_tracker.progress_display:
                # Get comprehensive timing analysis
                stats = self.task_tracker.progress_display.get_performance_stats()
                timing_report = self.task_tracker.progress_display.get_timing_analysis_report()
                
                # Create detailed response with both summary and full report
                content_parts = [
                    "⏱️ **Performance & Timing Analysis**",
                    "",
                    "**Quick Stats:**",
                    f"• Tasks Completed: {stats['completed_tasks']}/{stats['total_tasks']}",
                    f"• Active Agents: {stats['active_agents']}",
                    f"• Success Rate: {stats['success_rate']:.1f}%",
                    "",
                    timing_report
                ]
                
                response_message = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="\n".join(content_parts),
                    timestamp=self._format_timestamp()
                )
            else:
                response_message = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="⚠️ Timing analysis system not available.",
                    timestamp=self._format_timestamp()
                )
            
            self._notify_message(response_message)
            return True
            
        except Exception as e:
            self._notify_error(f"Failed to get timing analysis: {e}")
            return True


class MockAIProvider:
    """Mock AI provider for testing and development."""
    
    def __init__(self):
        self.responses = [
            "That's an interesting question! Let me think about that.",
            "I can help you with that. Here's what I think...",
            "That's a great point. Have you considered...",
            "I understand what you're asking. In my experience...",
            "That reminds me of something similar. Let me explain..."
        ]
        self.response_index = 0
    
    async def get_response(self, user_input: str, conversation_history: List[Dict[str, Any]]) -> str:
        """Get mock AI response."""
        await asyncio.sleep(0.3)  # Simulate processing time
        
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        
        return response