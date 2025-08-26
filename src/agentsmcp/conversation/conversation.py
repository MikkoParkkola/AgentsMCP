"""
Core conversation management for AgentsMCP conversational interface.
Handles natural language input parsing and command execution.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from .llm_client import LLMClient, ConversationMessage
from .structured_processor import StructuredProcessor

logger = logging.getLogger(__name__)


@dataclass
class CommandIntent:
    """Represents a parsed command intent from natural language."""
    command: str
    parameters: Dict[str, Any]
    confidence: float
    raw_text: str


class ConversationManager:
    """Manages conversational interface for AgentsMCP."""
    
    def __init__(self, command_interface=None, theme_manager=None, agent_manager=None):
        self.command_interface = command_interface
        self.theme_manager = theme_manager
        self.agent_manager = agent_manager
        self.llm_client = LLMClient()
        self.command_patterns = self._build_command_patterns()
        
        # Agent orchestration support
        self.agent_orchestration_enabled = True
        
        # Conversation context management for better UX
        self.conversation_history = []
        self.last_analysis_result = None
        self.context_cache = {}
        
        # Initialize structured processor for enhanced task handling
        self.structured_processor = StructuredProcessor(
            llm_client=self.llm_client,
            command_interface=command_interface,
            agent_manager=agent_manager
        )
        
        # Add status callback to show real-time updates
        self.structured_processor.add_status_callback(self._handle_status_update)
        self.use_structured_processing = True  # Toggle for enhanced processing
        
    def _build_command_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for recognizing command intents."""
        return {
            "status": [
                r"show.*status", r"check.*status", r"system.*status",
                r"what.*running", r"agents.*running", r"current.*state"
            ],
            "settings": [
                r"open.*settings", r"configure", r"change.*settings",
                r"modify.*config", r"preferences", r"settings"
            ],
            "dashboard": [
                r"start.*dashboard", r"open.*dashboard", r"monitor",
                r"dashboard", r"show.*dashboard", r"monitoring"
            ],
            "web": [
                r"web.*api", r"api.*endpoints", r"web.*interface",
                r"what.*endpoints", r"api.*info", r"web.*info"
            ],
            "help": [
                r"help", r"how.*use", r"commands.*available",
                r"what.*can.*do", r"usage", r"guide"
            ],
            "theme": [
                r"change.*theme", r"theme.*to", r"switch.*theme",
                r"set.*theme", r"theme.*dark", r"theme.*light"
            ],
            "agent": [
                r"switch.*agent", r"agent.*to", r"use.*agent",
                r"change.*agent", r"agent.*codex", r"agent.*claude", r"agent.*ollama"
            ],
            "model": [
                r"change.*model", r"set.*model", r"model.*to",
                r"use.*model", r"switch.*model"
            ],
            "provider": [
                r"change.*provider", r"set.*provider", r"provider.*to",
                r"use.*provider", r"switch.*provider"
            ],
            "orchestrate": [
                r"create.*task", r"spawn.*agent", r"delegate.*to",
                r"orchestrate", r"run.*task", r"execute.*task",
                r"create.*agent", r"new.*task"
            ],
            "new": [
                r"new.*session", r"start.*new", r"clear.*history",
                r"fresh.*start", r"reset.*conversation"
            ],
            "save": [
                r"save.*config", r"save.*settings", r"persist.*config",
                r"write.*config", r"store.*config"
            ],
            "models": [
                r"list.*models", r"show.*models", r"available.*models",
                r"what.*models", r"models.*for"
            ],
            "analyze_repository": [
                r"analyze.*repository", r"analyze.*project", r"investigate.*project",
                r"scan.*project", r"examine.*project", r"repository.*analysis"
            ],
            "exit": [
                r"exit", r"quit", r"bye", r"goodbye",
                r"stop", r"close", r"end"
            ]
        }
    
    async def process_input(self, user_input: str) -> str:
        """Process user input with enhanced structured processing."""
        try:
            # Check if we should use structured processing for this input
            if self.use_structured_processing and self._should_use_structured_processing(user_input):
                print("🔄 Initializing structured task analysis...")
                return await self.structured_processor.process_task(user_input)
            
            # Build enhanced context for better conversation continuity
            context = {
                "available_commands": list(self.command_patterns.keys()),
                "current_theme": getattr(self.theme_manager, 'current_theme', 'auto') if self.theme_manager else 'auto',
                "orchestration_enabled": self.agent_orchestration_enabled
            }
            
            # Add recent analysis context if available
            if self.last_analysis_result:
                context["last_analysis"] = {
                    "type": self.last_analysis_result.get('type'),
                    "summary": f"Recently analyzed project with {len(self.last_analysis_result.get('suggestions', []))} improvement suggestions",
                    "available": True
                }
            
            # Add conversation history for context  
            if len(self.conversation_history) > 0:
                context["recent_conversation"] = self.conversation_history[-3:]  # Last 3 exchanges
            
            # Primary workflow: LLM client with intelligent orchestration guidance
            response = await self.llm_client.send_message(user_input, context)
            
            # Store in conversation history
            self.conversation_history.append({
                "input": user_input,
                "response": response[:200] + "..." if len(response) > 200 else response,
                "timestamp": datetime.now()
            })
            
            # Check if LLM decided to orchestrate agents
            orchestration_request = self._extract_orchestration_from_response(response)
            if orchestration_request:
                # Try to delegate to agents
                agent_result = await self._handle_orchestration_request(orchestration_request)
                
                # If delegation failed, provide clear feedback without showing confusing orchestration syntax
                if "❌" in agent_result and ("Not Available" in agent_result or "failed" in agent_result.lower()):
                    # Remove orchestration patterns from the original response to avoid confusion
                    clean_response = self._remove_orchestration_patterns_from_response(response)
                    return f"{clean_response}\n\n{agent_result}"
                else:
                    return f"{response}\n\n{agent_result}"
            
            # Check for direct command execution requests from LLM
            command_request = self._extract_command_from_llm_response(response)
            if command_request:
                command_result = await self._execute_command(command_request)
                if command_result:
                    # Clean the response and return only the command result for better UX
                    clean_intro = self._remove_command_markers_from_response(response)
                    if clean_intro.strip() and len(clean_intro.strip()) > 5:
                        return f"{clean_intro}\n\n{command_result}"
                    else:
                        return command_result
            
            # Handle follow-up questions about analysis results
            if self.last_analysis_result and any(phrase in user_input.lower() for phrase in [
                'improvements', 'suggestions', 'what improvements', 'what can be improved',
                'project structure', 'tell me about', 'structure', 'about the project'
            ]):
                return self._handle_analysis_followup(user_input)
            
            # Fallback: Check for simple direct commands (backward compatibility)
            direct_command = self._extract_command_intent(user_input)
            if direct_command and direct_command.confidence > 0.8:
                result = await self._execute_command(direct_command)
                if result:
                    # Return only command result for direct commands to avoid dual responses
                    return result
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return f"Sorry, I encountered an error processing your request: {str(e)}"
    
    def _extract_command_intent(self, text: str) -> Optional[CommandIntent]:
        """Extract command intent from natural language text."""
        text_lower = text.lower().strip()
        
        best_match = None
        best_confidence = 0.0
        
        for command, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    confidence = len(re.findall(pattern, text_lower)) * 0.3
                    confidence += 1.0 if text_lower.startswith(pattern.split('.*')[0]) else 0.0
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        parameters = self._extract_parameters(command, text_lower)
                        best_match = CommandIntent(
                            command=command,
                            parameters=parameters,
                            confidence=confidence,
                            raw_text=text
                        )
        
        return best_match
    
    def _extract_parameters(self, command: str, text: str) -> Dict[str, Any]:
        """Extract parameters for specific commands."""
        parameters = {}
        
        if command == "theme":
            if "dark" in text:
                parameters["theme"] = "dark"
            elif "light" in text:
                parameters["theme"] = "light"
            elif "auto" in text:
                parameters["theme"] = "auto"
        
        return parameters
    
    def _extract_orchestration_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract agent orchestration requests from LLM responses."""
        # Look for single-agent orchestration markers
        single_agent_patterns = [
            r"→→\s*DELEGATE-TO-(\w+):\s*(.+?)(?:\n|$)",
            r"DELEGATE-TO-(\w+):\s*(.+?)(?:\n|$)",
            r"→→\s*DELEGATING\s*TO\s*(\w+):\s*(.+?)(?:\n|$)",
            r"→\s*Delegate\s+to\s+(\w+):\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in single_agent_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                agent_type, task = matches[0]
                return {
                    "agent_type": agent_type.lower().strip(),
                    "task": task.strip(),
                    "orchestration_type": "single_agent"
                }
        
        # Check for multi-agent orchestration
        multi_agent_patterns = [
            r"→→\s*MULTI-DELEGATE:\s*(.+?)(?:\n|$)",
            r"MULTI-DELEGATE:\s*(.+?)(?:\n|$)",
            r"→→\s*TEAM\s*ORCHESTRATION:\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in multi_agent_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                return {
                    "task": matches[0].strip(),
                    "orchestration_type": "multi_agent"
                }
        
        return None
    
    def _remove_orchestration_patterns_from_response(self, response: str) -> str:
        """Remove orchestration patterns from response to avoid user confusion when delegation fails."""
        # Remove single-agent orchestration markers
        patterns_to_remove = [
            r"→→\s*DELEGATE-TO-\w+:\s*[^\n]*",
            r"DELEGATE-TO-\w+:\s*[^\n]*", 
            r"→→\s*DELEGATING\s*TO\s*\w+:\s*[^\n]*",
            r"→\s*Delegate\s+to\s+\w+:\s*[^\n]*",
            r"→→\s*MULTI-DELEGATE:\s*[^\n]*",
            r"MULTI-DELEGATE:\s*[^\n]*",
            r"→→\s*TEAM\s*ORCHESTRATION:\s*[^\n]*"
        ]
        
        cleaned_response = response
        for pattern in patterns_to_remove:
            cleaned_response = re.sub(pattern, "", cleaned_response, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up extra whitespace and newlines
        cleaned_response = re.sub(r'\n\s*\n', '\n\n', cleaned_response)
        cleaned_response = cleaned_response.strip()
        
        return cleaned_response
    
    def _remove_command_markers_from_response(self, response: str) -> str:
        """Remove command execution markers from response to clean up user display."""
        command_markers = [
            r"\[execute:\s*\w+(?:\s+\w+)?\]",
            r"\[EXECUTE:\s*\w+(?:\s+\w+)?\]"
        ]
        
        cleaned_response = response
        for pattern in command_markers:
            cleaned_response = re.sub(pattern, "", cleaned_response, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned_response = re.sub(r'\s+', ' ', cleaned_response).strip()
        cleaned_response = re.sub(r'\.\s*$', '.', cleaned_response)  # Clean trailing periods
        
        return cleaned_response
    
    def _extract_command_from_llm_response(self, response: str) -> Optional[CommandIntent]:
        """Extract command execution requests from LLM responses."""
        # Look for command execution markers in LLM response
        command_markers = [
            r"execute\s+(\w+)(?:\s+(\w+))?",
            r"run\s+(\w+)(?:\s+(\w+))?",
            r"→\s*execute\s+(\w+)(?:\s+(\w+))?",
            r"\[execute:\s*(\w+)(?:\s+(\w+))?\]",
            r"\[EXECUTE:\s*(\w+)(?:\s+(\w+))?\]"  # Support uppercase format
        ]
        
        for pattern in command_markers:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                command = match.group(1)
                param = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                
                if command in self.command_patterns:
                    parameters = {}
                    if param and command == "theme":
                        parameters["theme"] = param
                    
                    return CommandIntent(
                        command=command,
                        parameters=parameters,
                        confidence=1.0,
                        raw_text=response
                    )
        
        return None
    
    async def _execute_command(self, command_intent: CommandIntent) -> Optional[str]:
        """Execute a command based on intent."""
        try:
            command = command_intent.command
            params = command_intent.parameters
            
            # Handle commands that don't require command interface first
            if command == "analyze_repository":
                return self._analyze_repository_directly()
            
            # For other commands, check if command interface is available
            if not self.command_interface:
                return f"Command '{command_intent.command}' recognized but command interface not available."
            
            if command == "status":
                return await self.command_interface.handle_command("status")
            elif command == "settings":
                return await self.command_interface.handle_command("settings")
            elif command == "dashboard":
                return await self.command_interface.handle_command("dashboard")
            elif command == "web":
                return await self.command_interface.handle_command("web")
            elif command == "help":
                return await self.command_interface.handle_command("help")
            elif command == "theme":
                if "theme" in params:
                    return await self.command_interface.handle_command(f"theme {params['theme']}")
                else:
                    return await self.command_interface.handle_command("theme")
            elif command == "agent":
                # Extract agent type from parameters or user input
                agent_type = self._extract_agent_type(command_intent.raw_text)
                if agent_type:
                    return await self.command_interface.handle_command(f"agent {agent_type}")
                else:
                    return "Please specify which agent to use: codex, claude, or ollama"
            elif command == "model":
                model_name = self._extract_model_name(command_intent.raw_text)
                if model_name:
                    return await self.command_interface.handle_command(f"model {model_name}")
                else:
                    return "Please specify a model name"
            elif command == "provider":
                provider_name = self._extract_provider_name(command_intent.raw_text)
                if provider_name:
                    return await self.command_interface.handle_command(f"provider {provider_name}")
                else:
                    return await self.command_interface.handle_command("provider")
            elif command == "new":
                return await self.command_interface.handle_command("new")
            elif command == "save":
                return await self.command_interface.handle_command("save")
            elif command == "models":
                provider = self._extract_provider_name(command_intent.raw_text)
                if provider:
                    return await self.command_interface.handle_command(f"models {provider}")
                else:
                    return await self.command_interface.handle_command("models")
            elif command == "orchestrate":
                return await self._handle_orchestration_request(command_intent.raw_text)
            elif command == "exit":
                return await self.command_interface.handle_command("exit")
            else:
                return f"Command '{command}' not yet implemented in conversational interface."
                
        except Exception as e:
            logger.error(f"Error executing command {command_intent.command}: {e}")
            return f"Sorry, I couldn't execute the '{command_intent.command}' command: {str(e)}"
    
    def _extract_agent_type(self, text: str) -> Optional[str]:
        """Extract agent type from text."""
        text_lower = text.lower()
        if "codex" in text_lower:
            return "codex"
        elif "claude" in text_lower:
            return "claude"  
        elif "ollama" in text_lower:
            return "ollama"
        return None
    
    def _extract_model_name(self, text: str) -> Optional[str]:
        """Extract model name from text."""
        # Look for common model name patterns
        model_patterns = [
            r"model\s+([a-zA-Z0-9\-\.]+)",
            r"to\s+([a-zA-Z0-9\-\.]+)",
            r"use\s+([a-zA-Z0-9\-\.]+)",
            r"gpt-[0-9]",
            r"claude-[0-9\-]+",
            r"llama[0-9\-]*"
        ]
        
        for pattern in model_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        return None
    
    def _extract_provider_name(self, text: str) -> Optional[str]:
        """Extract provider name from text."""
        text_lower = text.lower()
        providers = ["openai", "anthropic", "ollama", "openrouter"]
        for provider in providers:
            if provider in text_lower:
                return provider
        return None
    
    async def _handle_orchestrated_task(self, command_intent) -> str:
        """Handle orchestrated tasks that require agent delegation."""
        if not self.agent_orchestration_enabled:
            return "❌ Agent orchestration is not enabled"
        
        if not hasattr(self.command_interface, 'agent_manager') or not self.command_interface.agent_manager:
            # Fallback: use MCP agents directly
            return await self._delegate_to_mcp_agent(command_intent)
        
        try:
            task_type = command_intent.parameters.get("task_type", "generic")
            description = command_intent.parameters.get("description", command_intent.raw_text)
            agent_type = command_intent.agent_type or "codex"
            
            logger.info(f"Orchestrating {task_type} task using {agent_type} agent: {description}")
            
            # Special handling for test_and_fix task
            if task_type == "test_and_fix":
                return await self._handle_test_and_fix_task(command_intent)
            
            # General task delegation
            return await self._delegate_to_agent(agent_type, description, command_intent.parameters)
            
        except Exception as e:
            logger.error(f"Error in orchestrated task: {e}")
            return f"❌ Failed to orchestrate task: {str(e)}"
    
    async def _handle_test_and_fix_task(self, command_intent) -> str:
        """Handle comprehensive test and fix tasks."""
        description = command_intent.raw_text
        
        # Create a detailed prompt for the coding agent
        agent_prompt = f"""
I need you to comprehensively test and fix all issues in this project. Here's what I need:

Original request: {description}

Please perform these steps systematically:

1. **Project Analysis**: First, analyze the project structure and identify the main components
2. **Test Discovery**: Find and run all existing tests (pytest, npm test, etc.)
3. **Issue Identification**: List all failing tests, linting issues, type errors, and other problems
4. **Systematic Fixes**: Fix issues one by one, verifying each fix
5. **Final Validation**: Run all tests again to ensure everything is working

Please be thorough and systematic. Report your progress as you work through each step.
"""
        
        # Use the most capable agent for this complex task
        agent_type = command_intent.agent_type or "codex"
        return await self._delegate_to_mcp_agent_with_prompt(agent_type, agent_prompt)
    
    async def _delegate_to_agent(self, agent_type: str, description: str, parameters: Dict[str, Any]) -> str:
        """Delegate a task to a specific agent type."""
        try:
            # Enhanced prompt with parameters
            agent_prompt = f"Task: {description}\n\nParameters: {parameters}\n\nPlease execute this task systematically and report your progress."
            
            return await self._delegate_to_mcp_agent_with_prompt(agent_type, agent_prompt)
            
        except Exception as e:
            logger.error(f"Error delegating to {agent_type}: {e}")
            return f"❌ Failed to delegate to {agent_type} agent: {str(e)}"
    
    async def _delegate_to_mcp_agent(self, command_intent) -> str:
        """Delegate task directly to MCP agent."""
        agent_type = command_intent.agent_type or "codex"
        description = command_intent.parameters.get("description", command_intent.raw_text)
        
        return await self._delegate_to_mcp_agent_with_prompt(agent_type, description)
    
    async def _delegate_to_mcp_agent_with_prompt(self, agent_type: str, prompt: str) -> str:
        """Delegate task to MCP agent with specific prompt."""
        try:
            # Dynamic import to avoid circular dependencies
            import asyncio
            
            # Add timeout protection to prevent hanging
            timeout_seconds = 10  # 10 second timeout for agent operations
            
            if agent_type == "codex":
                # Use the LLM client with Codex provider for orchestration
                try:
                    return await asyncio.wait_for(
                        self._call_mcp_codex_via_llm_client(prompt), 
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    return f"⏰ **Codex Agent Timeout**\n\nThe Codex agent didn't respond within {timeout_seconds} seconds.\n\nThis usually means MCP connections are not working properly.\n\nTry using basic commands like 'help', 'status', or 'analyze the project' instead."
                    
            elif agent_type == "claude":
                # Claude MCP integration not implemented - be honest about it
                return f"❌ **Claude Agent Not Available**\n\nI'm unable to delegate this task to Claude as the MCP integration is not implemented.\n\nTask requested: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n\nTry using 'codex' or 'ollama' agents instead, or ask for help with basic commands."
                
            elif agent_type == "ollama":
                try:
                    return await asyncio.wait_for(
                        self._call_mcp_ollama(prompt), 
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    return f"⏰ **Ollama Agent Timeout**\n\nThe Ollama agent didn't respond within {timeout_seconds} seconds.\n\nThis usually means Ollama is not running or MCP connections are not working.\n\nTry using basic commands like 'help', 'status', or 'analyze the project' instead."
                    
            else:
                return f"❌ Unknown agent type: {agent_type}"
                
        except Exception as e:
            logger.error(f"Error calling MCP {agent_type} agent: {e}")
            return f"❌ Failed to delegate to {agent_type}: {str(e)}\n\nTry using basic commands like 'help', 'status', or 'analyze the project' instead."
    
    async def _call_mcp_codex(self, prompt: str) -> str:
        """Call MCP Codex agent."""
        try:
            # Import here to avoid circular dependencies
            import sys
            from pathlib import Path
            
            # Try to import MCP codex function with proper error handling
            try:
                import importlib
                codex_module = importlib.import_module('mcp__codex__codex')
                mcp_codex_fn = getattr(codex_module, 'mcp__codex__codex')
                
                # Delegate to Codex with a comprehensive configuration
                result = mcp_codex_fn(
                    prompt=prompt,
                    sandbox="workspace-write",  # Allow file modifications
                    cwd=str(Path.cwd())
                )
                
                if result:
                    content = ""
                    if hasattr(result, 'response'):
                        content = result.response
                    elif isinstance(result, str):
                        content = result
                    elif isinstance(result, dict):
                        content = result.get('response', str(result))
                    else:
                        content = str(result)
                    
                    return f"🤖 **Codex Agent Response**\n\n{content}"
                else:
                    raise Exception("No response from Codex agent")
                
            except Exception as codex_error:
                logger.warning(f"MCP Codex failed: {codex_error}, falling back to ollama")
                
                # Fallback to ollama
                try:
                    ollama_module = importlib.import_module('mcp__ollama__run')
                    mcp_ollama_fn = getattr(ollama_module, 'mcp__ollama__run')
                    
                    result = mcp_ollama_fn(
                        name="gpt-oss:20b",
                        prompt=prompt
                    )
                    
                    if result:
                        content = str(result)
                        return f"🤖 **Agent Response** (via Ollama)\n\n{content}"
                    else:
                        raise Exception("No response from Ollama agent")
                        
                except Exception as ollama_error:
                    logger.error(f"Both Codex and Ollama failed: {ollama_error}")
                    
                    # Final fallback: Try to execute the task directly ourselves
                    return self._execute_task_directly(prompt)
                
        except Exception as e:
            logger.error(f"Error calling MCP Codex: {e}")
            return f"❌ Codex delegation failed: {str(e)}"
    
    def _execute_task_directly(self, prompt: str) -> str:
        """Execute common tasks directly when MCP agents fail"""
        prompt_lower = prompt.lower()
        
        # Repository analysis task
        if any(keyword in prompt_lower for keyword in [
            "scan", "read", "documentation", "repository", "root directory", 
            "readme", "docs", "architecture", "project purpose", "investigate", 
            "project", "folder", "directory", "analyze", "explore", "examine",
            "check the", "look at", "what kind", "issues", "problems"
        ]):
            return self._analyze_repository_directly()
        
        # Status/system info tasks  
        elif any(keyword in prompt_lower for keyword in [
            "status", "system", "health", "info", "overview"
        ]):
            return self._get_system_status_directly()
        
        # Code review/analysis tasks
        elif any(keyword in prompt_lower for keyword in [
            "review", "analyze", "code", "codebase", "files"
        ]):
            return self._perform_code_analysis_directly()
        
        # Generic fallback
        else:
            return f"🤖 **Direct Execution**\n\nI understand you want me to: {prompt}\n\n⚠️ MCP agents are currently unavailable, but I can help you with this task directly. Could you provide more specific details about what you'd like me to do? I can:\n\n• Analyze files and documentation\n• Check system status\n• Review code structure\n• Provide guidance and recommendations\n\nWhat would you like to focus on?"
    
    def _analyze_repository_directly(self) -> str:
        """Directly analyze the repository when MCP agents fail"""
        try:
            from pathlib import Path
            import os
            
            repo_root = Path.cwd()
            analysis = []
            
            analysis.append("🤖 **Repository Analysis & Improvement Suggestions**\n")
            analysis.append(f"📁 **Project**: {repo_root.name}\n")
            
            # Find and read key documentation files
            doc_files = [
                "README.md", "README.rst", "README.txt",
                "CONTRIBUTING.md", "CONTRIBUTING.rst", 
                "CHANGELOG.md", "CHANGES.md",
                "LICENSE", "LICENSE.md", "LICENSE.txt",
                "ARCHITECTURE.md", "DESIGN.md"
            ]
            
            found_docs = []
            readme_content = ""
            for doc_file in doc_files:
                doc_path = repo_root / doc_file
                if doc_path.exists():
                    found_docs.append(doc_file)
                    try:
                        content = doc_path.read_text(encoding='utf-8')
                        if doc_file.upper().startswith("README"):
                            readme_content = content[:2000]  # Store for analysis
                            analysis.append(f"📄 **{doc_file}** (First 1000 chars):")
                            analysis.append(f"```\n{content[:1000]}{'...' if len(content) > 1000 else ''}\n```\n")
                        else:
                            analysis.append(f"📄 **{doc_file}**: Found ({len(content)} characters)")
                    except Exception:
                        analysis.append(f"📄 **{doc_file}**: Found but couldn't read\n")
            
            # Analyze project structure
            analysis.append("📂 **Project Structure**:")
            config_files = []
            source_dirs = []
            test_dirs = []
            
            try:
                for item in sorted(repo_root.iterdir()):
                    if item.name.startswith('.') and item.name not in ['.github', '.vscode']:
                        continue
                    
                    if item.is_dir():
                        dir_name = item.name.lower()
                        if dir_name in ['src', 'lib', 'app', 'agentsmcp']:
                            source_dirs.append(f"📁 {item.name}/ (source)")
                        elif dir_name in ['test', 'tests', '__tests__', 'spec']:
                            test_dirs.append(f"📁 {item.name}/ (tests)")
                        elif dir_name in ['.github']:
                            analysis.append(f"  📁 {item.name}/ (CI/CD)")
                        else:
                            analysis.append(f"  📁 {item.name}/")
                    else:
                        file_name = item.name.lower()
                        if file_name in ['pyproject.toml', 'setup.py', 'requirements.txt', 'package.json', 'cargo.toml']:
                            config_files.append(f"📄 {item.name} (config)")
                        else:
                            analysis.append(f"  📄 {item.name}")
                
                # Add categorized items
                for item in source_dirs + test_dirs + config_files:
                    analysis.append(f"  {item}")
                    
            except Exception as e:
                analysis.append(f"  ❌ Error reading directory: {e}")
            
            # Generate improvement suggestions
            analysis.append("\n🚀 **Improvement Suggestions**:")
            
            suggestions = []
            
            # Documentation suggestions
            if not found_docs:
                suggestions.append("📝 **Add documentation**: Create README.md, CONTRIBUTING.md, and CHANGELOG.md files")
            elif not any("readme" in doc.lower() for doc in found_docs):
                suggestions.append("📝 **Add README**: Create a comprehensive README.md file")
            
            # Configuration suggestions  
            if not config_files:
                suggestions.append("⚙️ **Add configuration**: Add setup.py/pyproject.toml for Python or package.json for Node.js")
            
            # Testing suggestions
            if not test_dirs:
                suggestions.append("🧪 **Add tests**: Create a test directory and implement unit tests")
            
            # CI/CD suggestions
            has_github_dir = (repo_root / '.github').exists()
            if not has_github_dir:
                suggestions.append("🔄 **Add CI/CD**: Create .github/workflows/ for automated testing and deployment")
            
            # Code quality suggestions
            suggestions.append("🔍 **Code quality**: Implement linting, formatting, and type checking")
            suggestions.append("📊 **Testing coverage**: Add test coverage reporting and aim for >80%")
            suggestions.append("🔒 **Security**: Add dependency scanning and security checks")
            suggestions.append("📈 **Performance**: Add performance monitoring and benchmarks")
            suggestions.append("🎨 **UX/UI improvements**: Enhance user interface and experience")
            
            for i, suggestion in enumerate(suggestions, 1):
                analysis.append(f"{i}. {suggestion}")
            
            # Save analysis result for context in follow-up questions
            result = "\n".join(analysis)
            self.last_analysis_result = {
                'type': 'repository_analysis',
                'content': result,
                'timestamp': datetime.now(),
                'project_structure': analysis,
                'suggestions': suggestions
            }
            
            return result
            
        except Exception as e:
            return f"🤖 **Repository Analysis Failed**\n\n❌ Error: {e}\n\nPlease check the current directory and permissions."
    
    def _handle_analysis_followup(self, user_input: str) -> str:
        """Handle follow-up questions about the last analysis."""
        if not self.last_analysis_result:
            return "I don't have any recent analysis results to refer to. Please run 'analyze the project' first."
        
        analysis = self.last_analysis_result
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['improvements', 'suggestions', 'what improvements']):
            suggestions = analysis.get('suggestions', [])
            if suggestions:
                result = ["🚀 **Based on my analysis, here are the key improvements I recommend:**\n"]
                for i, suggestion in enumerate(suggestions, 1):
                    result.append(f"{i}. {suggestion}")
                return "\n".join(result)
            else:
                return "I don't have specific improvement suggestions from the last analysis."
                
        elif any(word in user_lower for word in ['structure', 'project structure', 'tell me about']):
            structure = analysis.get('project_structure', [])
            if structure:
                return f"📁 **Project Structure from Analysis:**\n\n" + "\n".join(structure[:15])  # First 15 lines
            else:
                return "I don't have project structure details from the last analysis."
                
        else:
            # General follow-up - provide summary
            return f"📊 **Analysis Summary:**\n\n{analysis.get('content', 'No details available')[:500]}{'...' if len(analysis.get('content', '')) > 500 else ''}"
    
    def _get_system_status_directly(self) -> str:
        """Get system status directly when MCP agents fail"""
        try:
            import psutil
            import sys
            from datetime import datetime
            
            status = []
            status.append("🤖 **System Status** (Direct Execution)\n")
            
            # Python info
            status.append(f"🐍 Python: {sys.version.split()[0]}")
            status.append(f"📊 Memory Usage: {psutil.virtual_memory().percent:.1f}%")
            status.append(f"💽 Disk Usage: {psutil.disk_usage('/').percent:.1f}%")
            status.append(f"⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            status.append(f"🖥️ Platform: {sys.platform}")
            
            return "\n".join(status)
            
        except ImportError:
            return "🤖 **System Status** (Limited)\n\nBasic system info available (psutil not installed for detailed metrics)"
        except Exception as e:
            return f"🤖 **System Status Failed**\n\n❌ Error: {e}"
    
    def _perform_code_analysis_directly(self) -> str:
        """Perform basic code analysis when MCP agents fail"""
        try:
            from pathlib import Path
            import os
            
            repo_root = Path.cwd()
            analysis = []
            
            analysis.append("🤖 **Code Analysis** (Direct Execution)\n")
            
            # Find source code files
            code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.rb'}
            code_files = []
            
            for ext in code_extensions:
                code_files.extend(list(repo_root.rglob(f"*{ext}")))
            
            if code_files:
                analysis.append(f"📊 Found {len(code_files)} source files")
                
                # Group by extension
                by_ext = {}
                for file in code_files:
                    ext = file.suffix
                    by_ext[ext] = by_ext.get(ext, 0) + 1
                
                analysis.append("📋 **File Types**:")
                for ext, count in sorted(by_ext.items()):
                    analysis.append(f"  {ext}: {count} files")
                
            else:
                analysis.append("⚠️ No source code files found")
            
            return "\n".join(analysis)
            
        except Exception as e:
            return f"🤖 **Code Analysis Failed**\n\n❌ Error: {e}"
    
    async def _call_mcp_codex_via_llm_client(self, prompt: str) -> str:
        """Call MCP Codex agent via LLM client."""
        try:
            from .llm_client import LLMClient
            
            # Create a temporary LLM client configured for Codex
            codex_client = LLMClient()
            codex_client.provider = "codex"
            codex_client.model = "o1-mini"
            
            # Send the task to Codex
            response = await codex_client.send_message(prompt)
            
            return f"🤖 **Codex Agent Response**\n\n{response}"
            
        except Exception as e:
            logger.error(f"Error calling MCP Codex via LLM client: {e}")
            # Fallback to direct MCP call
            return await self._call_mcp_codex(prompt)
    
    async def _call_mcp_ollama(self, prompt: str) -> str:
        """Call MCP Ollama agent."""
        try:
            from mcp__ollama_turbo__run import mcp__ollama_turbo__run
            
            result = mcp__ollama_turbo__run(
                name="gpt-oss:20b",
                prompt=prompt,
                temperature=0.7
            )
            
            return f"🤖 **Ollama Agent Response**\n\n{result}"
            
        except ImportError:
            logger.warning("MCP Ollama not available")
            return f"❌ **Ollama Agent Not Available**\n\nMCP Ollama integration is not installed or not working.\n\nTask requested: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n\nTry using basic commands like 'status', 'help', or 'settings' instead."
        except Exception as e:
            logger.error(f"Error calling MCP Ollama: {e}")
            return f"❌ Ollama delegation failed: {str(e)}"

    async def _handle_orchestration_request(self, orchestration_request: Dict[str, Any]) -> str:
        """Handle agent orchestration requests from LLM."""
        if not self.agent_orchestration_enabled:
            return "❌ Agent orchestration is not enabled"
        
        orchestration_type = orchestration_request.get("orchestration_type", "single_agent")
        task = orchestration_request.get("task", "")
        
        if orchestration_type == "single_agent":
            agent_type = orchestration_request.get("agent_type", "codex")
            logger.info(f"LLM requested orchestration: {agent_type} for task: {task}")
            
            # Delegate to specific agent
            return await self._delegate_to_mcp_agent_with_prompt(agent_type, task)
            
        elif orchestration_type == "multi_agent":
            logger.info(f"LLM requested multi-agent orchestration for task: {task}")
            
            # For multi-agent, create a plan and delegate to primary agent
            enhanced_task = f"""
Multi-Agent Orchestration Request:
{task}

Please create a plan to execute this task, breaking it down into steps that could be handled by different specialized agents if needed. Consider:
1. Task analysis and decomposition
2. Parallel execution opportunities  
3. Dependencies between subtasks
4. Final integration and validation

Execute the plan systematically.
"""
            return await self._delegate_to_mcp_agent_with_prompt("codex", enhanced_task)
        
        else:
            return "❌ Unknown orchestration type requested"
    
    async def handle_settings_modification(self, modification_request: str) -> str:
        """Handle settings modification requests from conversation."""
        try:
            # Parse the modification request
            if "theme" in modification_request.lower():
                if "dark" in modification_request.lower():
                    if self.theme_manager:
                        self.theme_manager.current_theme = "dark"
                        return "✅ Theme changed to dark mode."
                elif "light" in modification_request.lower():
                    if self.theme_manager:
                        self.theme_manager.current_theme = "light"
                        return "✅ Theme changed to light mode."
                elif "auto" in modification_request.lower():
                    if self.theme_manager:
                        self.theme_manager.current_theme = "auto"
                        return "✅ Theme changed to auto mode."
            
            return "I understand you want to modify settings, but I need more specific instructions. You can say things like 'change theme to dark' or 'set theme to light'."
            
        except Exception as e:
            logger.error(f"Error modifying settings: {e}")
            return f"Sorry, I couldn't modify the settings: {str(e)}"
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get current conversation context."""
        return {
            "conversation_length": len(self.llm_client.conversation_history),
            "last_command": getattr(self, '_last_command', None),
            "theme": getattr(self.theme_manager, 'current_theme', 'auto') if self.theme_manager else 'auto',
            "commands_available": list(self.command_patterns.keys())
        }
    
    def _should_use_structured_processing(self, user_input: str) -> bool:
        """Determine if input should use structured processing."""
        # Keywords that indicate complex tasks
        structured_keywords = [
            "create", "build", "implement", "develop", "generate", "write",
            "design", "refactor", "optimize", "test", "debug", "fix",
            "analyze", "review", "document", "deploy", "setup", "configure"
        ]
        
        # Complexity indicators
        complexity_indicators = [
            "class", "function", "api", "database", "website", "application",
            "feature", "module", "component", "system", "framework", "library",
            "algorithm", "data structure", "unit test", "integration", "deployment"
        ]
        
        # Multi-step indicators
        multistep_indicators = [
            "first", "then", "after", "next", "finally", "step by step",
            "and also", "also need", "plus", "in addition", "furthermore"
        ]
        
        user_input_lower = user_input.lower()
        
        # Check for structured keywords
        has_action_keyword = any(keyword in user_input_lower for keyword in structured_keywords)
        has_complexity = any(indicator in user_input_lower for indicator in complexity_indicators)
        has_multistep = any(indicator in user_input_lower for indicator in multistep_indicators)
        
        # Use structured processing if:
        # 1. Has action keyword AND complexity indicator
        # 2. Has multistep indicators
        # 3. Input is longer than 50 characters and has action keyword
        return (
            (has_action_keyword and has_complexity) or
            has_multistep or
            (len(user_input) > 50 and has_action_keyword)
        )
    
    async def _handle_status_update(self, update: Dict[str, Any]):
        """Handle status updates from structured processor."""
        task_id = update.get("task_id", "unknown")
        status = update.get("status", "unknown")
        details = update.get("details", "")
        
        # Format and display status update
        if details:
            print(f"🔄 [{task_id[:8]}] {status}: {details}")
        else:
            print(f"🔄 [{task_id[:8]}] {status}")
    
    def toggle_structured_processing(self, enabled: bool = None) -> bool:
        """Toggle structured processing on/off."""
        if enabled is not None:
            self.use_structured_processing = enabled
        else:
            self.use_structured_processing = not self.use_structured_processing
        
        return self.use_structured_processing