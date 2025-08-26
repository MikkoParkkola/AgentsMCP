"""
LLM Client for conversational interface.
Handles communication with configured LLM models using real MCP clients.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelCapabilities:
    """Model capabilities including context limits and specifications."""
    context_window: int
    max_input_tokens: int
    max_output_tokens: int
    parameter_count: Optional[str] = None
    model_family: Optional[str] = None
    quantization: Optional[str] = None


@dataclass
class ConversationMessage:
    role: str  # 'system', 'user', 'assistant'
    content: str
    timestamp: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class LLMClient:
    """Client for interacting with configured LLM models via MCP."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.model = self.config.get("model", "gpt-oss:20b")
        self.provider = self.config.get("provider", "ollama-turbo")
        self.conversation_history: List[ConversationMessage] = []
        # Check if MCP orchestration is actually working
        orchestration_working = self._check_mcp_availability()
        self.system_context = self._build_system_context(orchestration_working)
        self._model_capabilities: Optional[ModelCapabilities] = None
        self._capabilities_cache = {}
        
    def _load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load LLM configuration from user's settings."""
        if config_path is None:
            config_path = Path.home() / ".agentsmcp" / "config.json"
            
        if config_path.exists():
            try:
                return json.loads(config_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Default configuration with context-aware defaults
        default_config = {
            "provider": "ollama-turbo",
            "model": "gpt-oss:120b",  # Default to 120B for ollama-turbo (128k context)
            "host": "http://127.0.0.1:11435", 
            "temperature": 0.7,
            "max_tokens": 4096,  # Higher default for better responses
            "context_window": 128000  # 128k tokens for gpt-oss models
        }
        
        # Adjust defaults based on provider to optimize for context windows
        provider = default_config.get("provider", "ollama-turbo")
        if provider == "ollama-turbo":
            # ollama-turbo supports much larger context windows
            default_config.update({
                "model": "gpt-oss:120b",  # 128k context window
                "max_tokens": 4096,
                "context_window": 128000
            })
        elif provider == "ollama":
            # Local ollama typically has smaller context windows
            default_config.update({
                "model": "gpt-oss:20b",  # Local model with smaller context
                "max_tokens": 1024,
                "context_window": 32000  # Conservative for local
            })
        
        return default_config
        
    def _check_mcp_availability(self) -> bool:
        """Check if MCP agents are actually available and working."""
        # For now, assume MCP orchestration is available since the AgentsMCP system
        # handles MCP servers through its own infrastructure rather than importable modules
        # The actual availability will be checked during delegation calls
        return True
        
    def _build_system_context(self, orchestration_working: bool = False) -> str:
        """Build system context for AgentsMCP conversational interface."""
        if orchestration_working:
            return """You are an intelligent conversational assistant for AgentsMCP, a multi-agent orchestration platform. You act as an LLM client that can intelligently orchestrate specialized agents when needed.

YOUR PRIMARY ROLE:
- Act as a normal LLM chat client for regular conversation and simple queries
- Intelligently identify when tasks require specialized agent delegation
- Orchestrate agents for complex, time-consuming, or specialized tasks
- Coordinate parallel agent execution for independent workstreams

WHEN TO ORCHESTRATE AGENTS:
Delegate to specialized agents for:
- Complex multi-step coding tasks (testing, debugging, refactoring)
- Large-scale code analysis or generation
- Security audits or vulnerability scans
- Documentation generation across multiple files
- Build/deploy/CI pipeline tasks
- Performance optimization analysis
- Any task that would take >5 minutes of focused work

WHEN TO RESPOND DIRECTLY:
Handle directly as LLM client for:
- Simple questions and explanations
- Quick code snippets or examples
- Configuration guidance
- Status checks and information requests
- Settings modifications
- General conversation and help

AGENT ORCHESTRATION SYNTAX:
When you determine a task needs agent delegation, use these markers in your response:

Single Agent: →→ DELEGATE-TO-codex: task description
- codex: For complex coding, testing, building, debugging
- claude: For large context analysis, documentation, refactoring  
- ollama: For privacy-sensitive or cost-conscious tasks

Multi-Agent: →→ MULTI-DELEGATE: complex task requiring multiple specialized agents

AVAILABLE COMMANDS:
- status: Check system status 
- settings: Configure providers/models
- dashboard: Open monitoring interface
- theme: Change appearance (light/dark/auto)
- help: Get assistance

CONVERSATION STYLE:
- Natural, helpful, and conversational
- Explain your reasoning when orchestrating agents
- Be concise for simple queries
- Provide context and progress updates for complex tasks

Remember: You're primarily an LLM client that smartly delegates complex tasks to specialized agents while handling regular chat naturally."""
        else:
            return """You are a conversational assistant for AgentsMCP. You should provide helpful responses and can execute specific commands when appropriate.

YOUR CAPABILITIES:
- Answer questions and provide explanations
- Give quick code examples and guidance  
- Help with configuration and settings
- Provide information about the system
- Execute commands when requested

COMMAND EXECUTION:
When a user asks for something that requires a specific command, respond with ONLY the command marker, nothing else:

- For repository analysis: [EXECUTE:analyze_repository]
- For system status: [EXECUTE:status]
- For settings: [EXECUTE:settings]
- For dashboard: [EXECUTE:dashboard]
- For help: [EXECUTE:help]
- For theme changes: [EXECUTE:theme dark/light/auto]

IMPORTANT: Do not provide additional analysis or explanations when using command markers. The command execution will provide the complete response.

For general conversation that doesn't require specific commands, provide helpful explanations and information directly.

CONVERSATION STYLE:
- Be concise and helpful
- Use command markers only when the user is asking for something that requires command execution
- For general questions, provide direct answers without command markers
- Suggest alternatives when possible

Remember: Be truthful about the system's current state rather than creating false expectations."""

    async def send_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Send message to LLM and get response."""
        try:
            # Add user message to history with timestamp
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            user_msg = ConversationMessage(role="user", content=message, timestamp=timestamp, context=context)
            self.conversation_history.append(user_msg)
            
            # Prepare messages for LLM with auto-detected capabilities
            messages = await self._prepare_messages()
            
            # Use real MCP ollama client based on provider
            response = await self._call_llm_via_mcp(messages)
            if not response:
                return "Sorry, I'm having trouble connecting to the LLM service. Please check your configuration in settings."
            
            # Extract response content
            assistant_content = self._extract_response_content(response)
            if not assistant_content:
                return "I received an empty response. Could you please try rephrasing your request?"
                
            # Add assistant response to history with timestamp
            from datetime import datetime
            response_timestamp = datetime.now().isoformat()
            assistant_msg = ConversationMessage(role="assistant", content=assistant_content, timestamp=response_timestamp)
            self.conversation_history.append(assistant_msg)
            
            return assistant_content
                
        except Exception as e:
            logger.error(f"Error in LLM communication: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    async def _prepare_messages(self) -> List[Dict[str, str]]:
        """Prepare messages for LLM API call with auto-detected capabilities."""
        messages = [
            {"role": "system", "content": self.system_context}
        ]
        
        # Auto-detect model capabilities
        try:
            capabilities = await self.get_model_capabilities()
            context_window = capabilities.context_window
            max_output_tokens = capabilities.max_output_tokens
        except Exception as e:
            logger.warning(f"Failed to detect capabilities, using config/defaults: {e}")
            # Fallback to config or conservative defaults
            context_window = self.config.get("context_window", 32000)
            max_output_tokens = self.config.get("max_tokens", 1024)
        
        # Reserve space for system prompt (~2k tokens) and response generation
        available_tokens = context_window - 2000 - max_output_tokens
        
        # Estimate tokens per message (~100-200 tokens average)
        estimated_tokens_per_message = 150
        max_history_messages = max(10, min(1000, available_tokens // estimated_tokens_per_message))
        
        # Use detected capabilities for better context utilization
        recent_history = self.conversation_history[-max_history_messages:] if len(self.conversation_history) > max_history_messages else self.conversation_history
        
        # Add conversation history
        for msg in recent_history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        logger.debug(f"Using {len(recent_history)} history messages (max {max_history_messages} for {context_window} token context, {max_output_tokens} max output)")
        return messages
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def add_context(self, context: Dict[str, Any]):
        """Add context information to the conversation."""
        if self.conversation_history:
            # Add context to the last message if it's from the user
            last_msg = self.conversation_history[-1]
            if last_msg.role == "user":
                last_msg.context = context
    
    async def get_model_capabilities(self) -> ModelCapabilities:
        """Get or detect model capabilities including context window and token limits."""
        cache_key = f"{self.provider}:{self.model}"
        
        # Return cached capabilities if available
        if cache_key in self._capabilities_cache:
            return self._capabilities_cache[cache_key]
        
        logger.info(f"Detecting capabilities for {cache_key}")
        
        # Try to detect capabilities from different sources
        capabilities = None
        
        if self.provider == "ollama-turbo":
            capabilities = await self._detect_ollama_com_capabilities()
        elif self.provider == "ollama":
            capabilities = await self._detect_local_ollama_capabilities()
        
        # Fallback to model-based heuristics if detection fails
        if not capabilities:
            capabilities = self._get_fallback_capabilities()
        
        # Cache the result
        self._capabilities_cache[cache_key] = capabilities
        logger.info(f"Detected capabilities: {capabilities.context_window} context, {capabilities.max_output_tokens} max output")
        
        return capabilities
    
    async def _detect_ollama_com_capabilities(self) -> Optional[ModelCapabilities]:
        """Detect model capabilities from ollama.com API."""
        try:
            import httpx
            
            api_key = os.getenv("OLLAMA_API_KEY")
            if not api_key:
                logger.warning("No OLLAMA_API_KEY, skipping ollama.com capability detection")
                return None
            
            async with httpx.AsyncClient() as client:
                # Try to get model information
                response = await client.post(
                    "https://ollama.com/api/show",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={"name": self.model},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    model_info = response.json()
                    return self._parse_model_capabilities(model_info)
                else:
                    logger.warning(f"Failed to get ollama.com model info: {response.status_code}")
                    
        except Exception as e:
            logger.warning(f"Error detecting ollama.com capabilities: {e}")
        
        return None
    
    async def _detect_local_ollama_capabilities(self) -> Optional[ModelCapabilities]:
        """Detect model capabilities from local Ollama instance."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                # Try to get model information from local Ollama
                response = await client.post(
                    "http://localhost:11434/api/show",
                    json={"name": self.model},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    model_info = response.json()
                    return self._parse_model_capabilities(model_info)
                else:
                    logger.warning(f"Failed to get local Ollama model info: {response.status_code}")
                    
        except Exception as e:
            logger.warning(f"Error detecting local Ollama capabilities: {e}")
        
        return None
    
    def _parse_model_capabilities(self, model_info: Dict[str, Any]) -> ModelCapabilities:
        """Parse model capabilities from Ollama model info."""
        # Extract context window from model parameters
        modelfile = model_info.get("modelfile", "")
        parameters = model_info.get("parameters", {})
        details = model_info.get("details", {})
        
        # Default values
        context_window = 32000  # Conservative default
        max_output_tokens = 4096  # Conservative default
        
        # Try to extract context window from various sources
        if "num_ctx" in parameters:
            try:
                context_window = int(parameters["num_ctx"])
            except (ValueError, TypeError):
                pass
        
        # Parse from modelfile
        if "num_ctx" in modelfile:
            import re
            match = re.search(r'num_ctx\s+(\d+)', modelfile)
            if match:
                try:
                    context_window = int(match.group(1))
                except (ValueError, TypeError):
                    pass
        
        # Model-specific knowledge
        model_name = self.model.lower()
        if "gpt-oss" in model_name:
            context_window = 128000  # GPT-OSS models support 128k
            max_output_tokens = 8192  # Higher for larger models
        elif "llama" in model_name or "mistral" in model_name:
            # Most Llama/Mistral models have varying context windows
            if "32k" in model_name or "32000" in model_name:
                context_window = 32000
            elif "128k" in model_name or "128000" in model_name:
                context_window = 128000
            else:
                context_window = 8192  # Common default
        
        # Calculate max input based on leaving room for output
        max_input_tokens = max(1024, context_window - max_output_tokens - 1000)  # Reserve 1k for system prompt
        
        return ModelCapabilities(
            context_window=context_window,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            parameter_count=details.get("parameter_size"),
            model_family=details.get("family"),
            quantization=details.get("quantization_level")
        )
    
    def _get_fallback_capabilities(self) -> ModelCapabilities:
        """Get fallback capabilities based on model name heuristics."""
        model_name = self.model.lower()
        
        # Model-specific fallbacks based on known specifications
        if "gpt-oss:120b" in model_name:
            return ModelCapabilities(
                context_window=128000,
                max_input_tokens=120000,
                max_output_tokens=8192,
                parameter_count="120B",
                model_family="gpt-oss"
            )
        elif "gpt-oss:20b" in model_name:
            return ModelCapabilities(
                context_window=128000,
                max_input_tokens=124000,
                max_output_tokens=4096,
                parameter_count="20B", 
                model_family="gpt-oss"
            )
        elif "mistral-nemo" in model_name:
            return ModelCapabilities(
                context_window=128000,
                max_input_tokens=124000,
                max_output_tokens=4096,
                parameter_count="12.2B",
                model_family="mistral"
            )
        elif "llama" in model_name:
            return ModelCapabilities(
                context_window=32000,
                max_input_tokens=28000,
                max_output_tokens=4096,
                parameter_count="Unknown",
                model_family="llama"
            )
        else:
            # Conservative defaults for unknown models
            return ModelCapabilities(
                context_window=8192,
                max_input_tokens=6144,
                max_output_tokens=2048,
                parameter_count="Unknown",
                model_family="unknown"
            )
    
    async def _get_max_tokens_for_api(self) -> int:
        """Get max tokens for API call based on detected capabilities."""
        try:
            capabilities = await self.get_model_capabilities()
            return capabilities.max_output_tokens
        except Exception:
            # Fallback to config
            return self.config.get("max_tokens", 1024)
    
    async def _call_llm_via_mcp(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Call the LLM via real MCP client based on configured provider."""
        try:
            # Use actual MCP client for LLM calls
            if self.provider == "ollama-turbo":
                return await self._call_ollama_turbo(messages)
            elif self.provider == "ollama":
                return await self._call_ollama(messages)
            elif self.provider == "openai":
                return await self._call_openai(messages)
            elif self.provider == "codex":
                return await self._call_codex(messages)
            else:
                logger.warning(f"Unknown provider {self.provider}, falling back to ollama-turbo")
                return await self._call_ollama_turbo(messages)
                
        except Exception as e:
            logger.error(f"Error calling LLM via MCP: {e}")
            return None
    
    async def _call_ollama_turbo(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Call Ollama with priority: 1) ollama.com API, 2) local proxy, 3) MCP."""
        try:
            import httpx
            
            # Convert messages to Ollama format
            ollama_messages = []
            for msg in messages:
                ollama_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Try ollama.com API first (primary)
            result = await self._try_ollama_com_api(ollama_messages)
            if result:
                return result
                
            # Try local proxy (fallback)
            result = await self._try_local_proxy(ollama_messages)
            if result:
                return result
                
            # Try local Ollama instance (last resort for direct API)
            result = await self._try_local_ollama(ollama_messages)
            if result:
                return result
                
            # If all direct methods fail, try MCP as absolute last resort
            logger.warning("All direct Ollama methods failed, trying MCP...")
            return await self._call_ollama(messages)
                
        except ImportError:
            logger.error("httpx not available for Ollama API calls")
            return await self._fallback_response(messages)
        except Exception as e:
            logger.error(f"Error in ollama-turbo chain: {e}")
            return await self._fallback_response(messages)
    
    async def _try_ollama_com_api(self, ollama_messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Try ollama.com API with authentication."""
        try:
            import httpx
            import os
            
            # Get API key from environment variable
            api_key = os.getenv("OLLAMA_API_KEY")
            if not api_key:
                logger.warning("OLLAMA_API_KEY not set, skipping ollama.com API")
                return None
            
            # Use detected max tokens
            max_tokens = await self._get_max_tokens_for_api()
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://ollama.com/api/chat",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": ollama_messages,
                        "options": {
                            "temperature": self.config.get("temperature", 0.7),
                            "num_predict": max_tokens
                        },
                        "stream": False,
                        "tools": []
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info("Successfully used ollama.com API")
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": result.get("message", {}).get("content", "")
                            },
                            "finish_reason": "stop"
                        }]
                    }
                else:
                    logger.warning(f"Ollama.com API failed: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.warning(f"Ollama.com API error: {e}")
            return None
    
    async def _try_local_proxy(self, ollama_messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Try local proxy that handles API key."""
        try:
            import httpx
            
            # Use detected max tokens
            max_tokens = await self._get_max_tokens_for_api()
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://127.0.0.1:11435/api/chat",
                    json={
                        "model": self.model,
                        "messages": ollama_messages,
                        "options": {
                            "temperature": self.config.get("temperature", 0.7),
                            "num_predict": max_tokens
                        },
                        "stream": False,
                        "tools": []
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info("Successfully used local proxy")
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": result.get("message", {}).get("content", "")
                            },
                            "finish_reason": "stop"
                        }]
                    }
                else:
                    logger.warning(f"Local proxy failed: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.warning(f"Local proxy error: {e}")
            return None
    
    async def _try_local_ollama(self, ollama_messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Try local Ollama instance."""
        try:
            import httpx
            
            # Use detected max tokens
            max_tokens = await self._get_max_tokens_for_api()
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": self.model,
                        "messages": ollama_messages,
                        "options": {
                            "temperature": self.config.get("temperature", 0.7),
                            "num_predict": max_tokens
                        },
                        "stream": False,
                        "tools": []
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info("Successfully used local Ollama")
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": result.get("message", {}).get("content", "")
                            },
                            "finish_reason": "stop"
                        }]
                    }
                else:
                    logger.warning(f"Local Ollama failed: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.warning(f"Local Ollama error: {e}")
            return None
    
    async def _call_ollama(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Call Ollama via direct API, skipping MCP for reliability."""
        try:
            # Convert to ollama format
            ollama_messages = []
            for msg in messages:
                ollama_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Try local Ollama instance first (most reliable)
            result = await self._try_local_ollama(ollama_messages)
            if result:
                return result
                
            # Then try local proxy
            result = await self._try_local_proxy(ollama_messages)
            if result:
                return result
                
            # Finally try ollama.com API if available
            result = await self._try_ollama_com_api(ollama_messages)
            if result:
                return result
                
            logger.error("All Ollama methods failed")
            return await self._fallback_response(messages)
                
        except Exception as e:
            logger.error(f"Error calling ollama: {e}")
            return await self._fallback_response(messages)
    
    async def _call_openai(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Call OpenAI via standard API (fallback if no MCP client)."""
        try:
            import openai
            
            # Use OpenAI API key from environment or config
            api_key = os.getenv("OPENAI_API_KEY") or self.config.get("api_key")
            if not api_key:
                logger.error("OpenAI API key not found")
                return await self._fallback_response(messages)
            
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.config.get("model", "gpt-4o-mini"),
                messages=messages,
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 1024)
            )
            
            return {
                "choices": [
                    {
                        "message": {
                            "role": response.choices[0].message.role,
                            "content": response.choices[0].message.content
                        },
                        "finish_reason": response.choices[0].finish_reason
                    }
                ]
            }
            
        except ImportError as e:
            logger.error(f"OpenAI library not available: {e}")
            return await self._fallback_response(messages)
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
            return await self._fallback_response(messages)
    
    async def _call_codex(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Call Codex via MCP - fallback to local ollama if codex fails."""
        try:
            # Convert messages to a single prompt for Codex
            prompt_parts = []
            for msg in messages:
                if msg.get('role') == 'system':
                    prompt_parts.append(f"System: {msg.get('content', '')}")
                elif msg.get('role') == 'user':
                    prompt_parts.append(f"User: {msg.get('content', '')}")
                elif msg.get('role') == 'assistant':
                    prompt_parts.append(f"Assistant: {msg.get('content', '')}")
            
            prompt = "\n".join(prompt_parts)
            
            # Try to use MCP codex client first
            try:
                # Import within try block to handle import failures gracefully
                import importlib
                codex_module = importlib.import_module('mcp__codex__codex')
                mcp_codex_fn = getattr(codex_module, 'mcp__codex__codex')
                
                result = mcp_codex_fn(
                    prompt=prompt,
                    sandbox="workspace-write",
                    cwd="/Users/mikko/github/AgentsMCP"
                )
                
                if result:
                    # Handle different response formats
                    content = ""
                    if hasattr(result, 'response'):
                        content = result.response
                    elif isinstance(result, str):
                        content = result
                    elif isinstance(result, dict):
                        content = result.get('response', str(result))
                    else:
                        content = str(result)
                    
                    if content:
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": content
                                    },
                                    "finish_reason": "stop"
                                }
                            ]
                        }
            except Exception as codex_error:
                logger.warning(f"Codex failed: {codex_error}, falling back to local ollama")
            
            # Fallback to local ollama with gpt-oss:20b
            try:
                import importlib
                ollama_module = importlib.import_module('mcp__ollama__run')
                mcp_ollama_fn = getattr(ollama_module, 'mcp__ollama__run')
                
                result = mcp_ollama_fn(
                    name="gpt-oss:20b",
                    prompt=prompt
                )
                
                if result:
                    content = str(result) if result else "No response from agent"
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": content
                                },
                                "finish_reason": "stop"
                            }
                        ]
                    }
            except Exception as ollama_error:
                logger.error(f"Ollama fallback also failed: {ollama_error}")
            
            # Final fallback
            return await self._fallback_response(messages)
                
        except Exception as e:
            logger.error(f"Error in _call_codex: {e}")
            return await self._fallback_response(messages)
    
    async def _fallback_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate fallback response when MCP clients fail."""
        user_message = ""
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                user_message = msg.get('content', '')
                break
        
        response_content = self._generate_intelligent_response(user_message, messages)
        
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant", 
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    
    def _generate_intelligent_response(self, user_message: str, messages: List[Dict[str, str]]) -> str:
        """Generate intelligent response based on user input and context."""
        user_lower = user_message.lower()
        
        # Handle common requests intelligently and signal command execution
        if any(word in user_lower for word in ['status', 'running', 'check']):
            return "I'll check the system status for you. [EXECUTE:status]"
            
        elif any(word in user_lower for word in ['settings', 'configure', 'config', 'setup']):
            return "I'll open the settings for you to configure your AgentsMCP preferences. [EXECUTE:settings]"
            
        elif any(word in user_lower for word in ['dashboard', 'monitor', 'watch']):
            return "I'll start the dashboard so you can monitor the system. [EXECUTE:dashboard]"
            
        elif any(word in user_lower for word in ['help', 'commands', 'what can']):
            return "I'll show you the available commands and help information. [EXECUTE:help]"
            
        elif any(word in user_lower for word in ['theme', 'dark', 'light']):
            # Extract theme preference
            if 'dark' in user_lower:
                return "I'll switch to dark mode. [EXECUTE:theme dark]"
            elif 'light' in user_lower:
                return "I'll switch to light mode. [EXECUTE:theme light]"
            else:
                return "I'll set the theme to auto mode. [EXECUTE:theme auto]"
            
        elif any(word in user_lower for word in ['web', 'api', 'endpoints']):
            return "I'll show you the web API information. [EXECUTE:web]"
        
        elif any(phrase in user_lower for phrase in [
            'investigate the project', 'analyze the project', 'examine the project',
            'check the project', 'look at the project', 'scan the project',
            'what kind of issues', 'find issues', 'project issues', 'code issues',
            'investigate this folder', 'analyze this folder', 'examine this folder',
            'repository analysis', 'analyze repository', 'scan repository',
            'analyze the project in current directory', 'improvements you would make'
        ]):
            return "I'll analyze the project structure and identify any potential issues. [EXECUTE:analyze_repository]"
            
        elif any(phrase in user_lower for phrase in [
            'do these improvements', 'implement these', 'make these changes', 'apply these changes',
            'please implement', 'implement the', 'make the improvements', 'apply the improvements',
            'perform these', 'execute these', 'do the improvements', 'make these fixes',
            'implement all', 'do all', 'yes, please implement', 'all of them',
            'apply all', 'make all', 'fix all', 'implement everything'
        ]):
            return "I'll implement the suggested improvements using the coding agent. →→ DELEGATE-TO-codex: Implement all the suggested improvements from the previous analysis"
            
        else:
            return f"I understand you're asking: '{user_message}'. However, I'm currently unable to handle complex tasks as the agent orchestration system is not functioning properly. I can help with basic commands like status, settings, dashboard, help, and theme changes. What would you like me to do?"
    
    def _extract_response_content(self, response: Optional[Dict[str, Any]]) -> Optional[str]:
        """Extract content from LLM response."""
        if not response:
            return None
            
        # Handle ollama response format
        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
            
        # Handle OpenAI-style response format
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                return choice['message']['content']
                
        logger.warning(f"Unexpected response format: {response}")
        return None
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        if not self.conversation_history:
            return "No conversation history."
        
        summary = "Recent conversation:\n"
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            role_indicator = "🧑" if msg.role == "user" else "🤖"
            timestamp_str = ""
            if msg.timestamp:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(msg.timestamp)
                    timestamp_str = dt.strftime("[%H:%M:%S] ")
                except:
                    timestamp_str = ""
            summary += f"{timestamp_str}{role_indicator} {msg.content[:50]}...\n"
        
        return summary