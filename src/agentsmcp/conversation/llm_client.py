"""
LLM Client for conversatio,
            {
                "type": "function",
                "function": {
                    "name": "github_create_pull_request",
                    "description": "Create a GitHub pull request using the GitHub CLI if available",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "PR title"},
                            "body": {"type": "string", "description": "PR body", "default": "Automated PR"},
                            "base": {"type": "string", "description": "Base branch", "default": "main"},
                            "head": {"type": "string", "description": "Head branch", "default": "auto/agentsmcp"}
                        },
                        "required": ["title"]
                    }
                }
            }nal interface.
Handles communication with configured LLM models using real MCP clients.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path

# Import tool registry for tool execution
from ..tools.base_tools import tool_registry
# Ensure default tools are registered only when client is instantiated
from ..tools import ensure_default_tools_registered

logger = logging.getLogger(__name__)

# Configure logger based on environment - prevent console spam in TUI mode
def _configure_logger():
    """Configure logger to prevent console contamination in TUI mode."""
    import os
    import tempfile
    
    # Check if we're in TUI mode (set by TUI when it imports this)
    is_tui_mode = os.environ.get('AGENTSMCP_TUI_MODE', '0') == '1'
    
    if is_tui_mode:
        # TUI mode: Log only to file, no console output
        logger.setLevel(logging.DEBUG)  # Allow debug logs to file
        
        # Remove any existing console handlers
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)
        
        # Add file handler only
        try:
            log_file = os.path.join(tempfile.gettempdir(), 'agentsmcp_llm_debug.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            # Fallback: silence the logger entirely if file logging fails
            logger.setLevel(logging.CRITICAL)
    else:
        # CLI mode: Standard logging setup
        logger.setLevel(logging.DEBUG)
        
        # Ensure we have a handler for debugging
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)

# Configure the logger immediately
_configure_logger()

@dataclass
class ModelCapabilities:
    """Model capabilities including context limits and specifications."""
    context_window: int
    max_input_tokens: int
    max_output_tokens: int
    parameter_count: Optional[str] = None
    model_family: Optional[str] = None
    quantization: Optional[str] = None
    supports_streaming: bool = False


@dataclass
class ConversationMessage:
    role: str  # 'system', 'user', 'assistant'
    content: str
    timestamp: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class LLMClient:
    """Client for interacting with configured LLM models via MCP with streaming support."""
    
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
        # Initialize MCP tools for file system access
        try:
            ensure_default_tools_registered()
        except Exception:
            pass
        self.mcp_tools = self._get_mcp_tools()

    def get_configuration_status(self) -> Dict[str, Any]:
        """Get detailed configuration status for diagnostics."""
        status = {
            "providers": {},
            "current_provider": self.provider,
            "current_model": self.model,
            "preprocessing_enabled": getattr(self, 'preprocessing_enabled', True),
            "mcp_tools_available": len(self.mcp_tools) > 0,
            "configuration_issues": []
        }
        
        # Check each provider configuration
        providers_to_check = ["openai", "anthropic", "ollama", "openrouter", "codex"]
        
        for provider in providers_to_check:
            provider_status = {
                "configured": False,
                "api_key_present": False,
                "service_available": False,
                "last_error": None
            }
            
            try:
                if provider == "ollama":
                    # Special case for Ollama - check if service is running
                    import aiohttp
                    import asyncio
                    
                    async def check_ollama():
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                                    return resp.status == 200
                        except Exception as e:
                            provider_status["last_error"] = str(e)
                            return False
                    
                    # Run the check synchronously
                    try:
                        loop = asyncio.get_event_loop()
                        provider_status["service_available"] = loop.run_until_complete(check_ollama())
                    except:
                        provider_status["service_available"] = False
                    provider_status["configured"] = True
                    provider_status["api_key_present"] = True  # Ollama doesn't need API keys
                else:
                    # Check for API keys
                    api_key = self._get_api_key(provider)
                    provider_status["api_key_present"] = bool(api_key)
                    provider_status["configured"] = provider_status["api_key_present"]
                    provider_status["service_available"] = provider_status["api_key_present"]  # Assume available if key exists
                    
            except Exception as e:
                provider_status["last_error"] = str(e)
            
            status["providers"][provider] = provider_status
            
            # Add configuration issues
            if provider == self.provider and not provider_status["configured"]:
                if provider == "ollama":
                    status["configuration_issues"].append(f"Current provider '{provider}' not available. Start Ollama with: ollama serve")
                else:
                    status["configuration_issues"].append(f"Current provider '{provider}' missing API key. Set {provider.upper()}_API_KEY environment variable")
        
        # Check if any provider is configured
        configured_providers = [p for p, s in status["providers"].items() if s["configured"]]
        if not configured_providers:
            status["configuration_issues"].append("No LLM providers configured. Set at least one API key or start Ollama locally")
        
        return status

    def toggle_preprocessing(self, enabled: Optional[bool] = None) -> str:
        """Toggle or set preprocessing mode."""
        if not hasattr(self, 'preprocessing_enabled'):
            self.preprocessing_enabled = True
            
        if enabled is None:
            self.preprocessing_enabled = not self.preprocessing_enabled
        else:
            self.preprocessing_enabled = enabled
            
        status = "enabled" if self.preprocessing_enabled else "disabled"
        mode_desc = "Multi-turn tool execution" if self.preprocessing_enabled else "Direct LLM responses"
        
        return f"✅ Preprocessing {status}\n📝 Mode: {mode_desc}\n💡 Use '/preprocessing status' to check current mode"

    def get_preprocessing_status(self) -> str:
        """Get current preprocessing status."""
        if not hasattr(self, 'preprocessing_enabled'):
            self.preprocessing_enabled = True
            
        status = "enabled" if self.preprocessing_enabled else "disabled"
        mode_desc = "Multi-turn tool execution with tool calling" if self.preprocessing_enabled else "Direct LLM responses only"
        
        return f"🔧 Preprocessing: {status}\n📝 Description: {mode_desc}\n\n💡 Commands:\n  • /preprocessing on - Enable preprocessing\n  • /preprocessing off - Disable preprocessing\n  • /preprocessing toggle - Switch mode"
        
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
            "context_window": 128000,  # 128k tokens for gpt-oss models
            # Provider-specific keys map
            "api_keys": {},
            # Allow/deny providers. By default, only ollama-turbo is enabled as requested.
            "providers_enabled": ["ollama-turbo"]
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

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Return API key for a provider from config.api_keys or env as fallback."""
        keys = self.config.get("api_keys", {}) if isinstance(self.config, dict) else {}
        key = keys.get(provider)
        if key:
            return key
        # Env fallback mapping
        env_map = {
            "ollama-turbo": "OLLAMA_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_var = env_map.get(provider)
        if env_var:
            return os.getenv(env_var)
        return None

    def _get_timeout(self, key: str, default: float) -> float:
        """Get per-endpoint timeout seconds from config.timeouts, or default."""
        try:
            tmap = self.config.get("timeouts") if isinstance(self.config, dict) else None
            if isinstance(tmap, dict):
                val = tmap.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    return float(val)
        except Exception:
            pass
        return float(default)
        
    def _check_mcp_availability(self) -> bool:
        """Check if MCP agents are actually available and working."""
        # For now, assume MCP orchestration is available since the AgentsMCP system
        # handles MCP servers through its own infrastructure rather than importable modules
        # The actual availability will be checked during delegation calls
        return True
    
    def _get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get available MCP tools for file system operations."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file (creates directories if needed)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to the file to write"},
                            "content": {"type": "string", "description": "Content to write"},
                            "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"}
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List the contents of a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The directory path to list"
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search for files matching a pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The search pattern or glob"
                            },
                            "path": {
                                "type": "string",
                                "description": "The directory to search in (default: current directory)",
                                "default": "."
                            }
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_file_info",
                    "description": "Get information about a file or directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path to get information about"
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_shell",
                    "description": "Run a shell command inside the current project directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The shell command to execute"},
                            "timeout": {"type": "integer", "description": "Max seconds before termination", "default": 60}
                        },
                        "required": ["command"]
                    }
                }
            }
            ,
            {
                "type": "function",
                "function": {
                    "name": "list_staged_changes",
                    "description": "List files staged for approval (write operations pending review)",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "approve_changes",
                    "description": "Apply all staged changes to the working tree and create a git commit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "commit_message": {"type": "string", "description": "Commit message to use", "default": "chore: apply reviewed changes"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "discard_staged_changes",
                    "description": "Discard all staged (pending) changes without applying",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            }
        ]
        
    def _build_system_context(self, orchestration_working: bool = False) -> str:
        """Build system context for AgentsMCP conversational interface.

        Also loads local guidance from AGENTS.md and docs/models.md when present,
        to enrich instructions without requiring network access.
        """
        supplemental = []
        try:
            from pathlib import Path
            for name in ("AGENTS.md", "docs/models.md"):
                p = Path.cwd() / name
                if p.exists() and p.is_file():
                    try:
                        txt = p.read_text(encoding="utf-8")
                        if len(txt) > 12000:
                            txt = txt[:12000] + "\n... [truncated]"
                        supplemental.append(f"# {name}\n\n{txt}")
                    except Exception:
                        pass
            # Include continuous improvement notes if present
            try:
                imp_path = Path("build/retrospectives/improvements.md")
                if imp_path.exists():
                    txt = imp_path.read_text(encoding="utf-8")
                    # Take last ~4000 chars to keep prompt bounded
                    supplemental.append("# Continuous Improvement Notes\n\n" + txt[-4000:])
            except Exception:
                pass
        except Exception:
            pass

        base_text = """You are an intelligent conversational assistant for AgentsMCP, a multi-agent orchestration platform. You act as an LLM client that can intelligently orchestrate specialized agents when needed.

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

Remember: You're primarily an LLM client that smartly delegates complex tasks to specialized agents while handling regular chat naturally.

# AGENT CATALOG (Authoritative)

You MUST consider the configured human-oriented roles as the only available agents. Each role has a configured provider and model. The current environment is constrained to a single provider and model for all roles:

- Provider: ollama-turbo (OpenAI-compatible API), Base URL: https://ollama.com/
- Model: gpt-oss:120b
- API key source: env OLLAMA_API_KEY (already configured by the host)

Human-oriented roles (examples): business_analyst, backend_engineer, web_frontend_engineer, api_engineer, tui_frontend_engineer, backend_qa_engineer, web_frontend_qa_engineer, tui_frontend_qa_engineer, chief_qa_engineer, it_lawyer, marketing_manager, ci_cd_engineer, dev_tooling_engineer, data_analyst, data_scientist, ml_scientist, ml_engineer.

When the user asks for "configured agents", list these human roles and explicitly state provider=ollama-turbo and model=gpt-oss:120b for each. Do NOT mention codex/claude/openrouter or any other providers/models.

# TOOLS POLICY (Authoritative)

- Agents have access to filesystem and shell via the orchestrator. Treat these tools as available to agents within a sandboxed project directory. When a role needs to read/write files or run commands, you can invoke those tools (the runtime will execute safely) and feed results back into the role.
- The orchestrator can adjust each agent's toolset for efficiency (e.g., enabling/disabling shell/filesystem per role or per task). Assume filesystem and run_shell are available by default, and request additional tools only when beneficial.
- Only use the configured provider(s). Providers not explicitly enabled (e.g., codex/claude/openrouter) are NOT available and must not be referenced.
"""
        if supplemental:
            base_text += "\n\n# PROJECT INSTRUCTIONS\n\n" + "\n\n".join(supplemental)
        if orchestration_working:
            return base_text
        else:
            fallback_text = """You are a conversational assistant for AgentsMCP. You should provide helpful responses and can execute specific commands when appropriate.

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
            if supplemental:
                fallback_text += "\n\n# PROJECT INSTRUCTIONS\n\n" + "\n\n".join(supplemental)
            return fallback_text

    async def send_message(self, message: str, context: Optional[Dict[str, Any]] = None, 
                          progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """Send message to LLM and get response with enhanced error reporting and progress tracking."""
        try:
            # Initialize progress tracker
            from ..ui.v3.progress_tracker import ProgressTracker, ProcessingPhase, ToolExecutionInfo
            progress_tracker = ProgressTracker(progress_callback)
            
            await progress_tracker.update_phase(ProcessingPhase.ANALYZING)
            
            # Check configuration before attempting to send
            config_status = self.get_configuration_status()
            if config_status["configuration_issues"]:
                error_msg = "❌ LLM Configuration Issues:\n"
                for issue in config_status["configuration_issues"]:
                    error_msg += f"  • {issue}\n"
                error_msg += "\n💡 Solutions:\n"
                error_msg += "  • Type /config to see detailed configuration status\n"
                error_msg += "  • Type /help to see all available commands\n"
                error_msg += "  • Set environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.\n"
                error_msg += "  • Or start Ollama locally: ollama serve"
                return error_msg

            # Add user message to history with timestamp
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            user_msg = ConversationMessage(role="user", content=message, timestamp=timestamp, context=context)
            self.conversation_history.append(user_msg)
            
            # Handle simple mode (no preprocessing) vs full mode
            preprocessing_enabled = getattr(self, 'preprocessing_enabled', True)
            if not preprocessing_enabled:
                await progress_tracker.update_custom_status("Direct LLM processing", "🚀")
                return await self._send_simple_message(message, progress_callback)
            
            # Multi-turn tool execution loop (preprocessing mode)
            await progress_tracker.update_custom_status("Multi-turn processing active", "🛠️")
            max_tool_turns = 3  # Prevent infinite loops
            turn = 0
            
            while turn < max_tool_turns:
                turn += 1
                logger.debug(f"Tool execution turn {turn}/{max_tool_turns}")
                
                await progress_tracker.update_multi_turn(turn, max_tool_turns, "analyzing request")
                
                # Prepare messages for LLM with auto-detected capabilities
                messages = await self._prepare_messages()
                
                # Use real MCP ollama client based on provider
                await progress_tracker.update_phase(ProcessingPhase.PROCESSING_RESULTS)
                response = await self._call_llm_via_mcp(messages)
                if not response:
                    # More specific error message based on configuration
                    config_status = self.get_configuration_status()
                    provider_status = config_status["providers"].get(self.provider, {})
                    
                    if not provider_status.get("configured"):
                        if self.provider == "ollama":
                            return "❌ Ollama not running. Start it with: ollama serve"
                        else:
                            return f"❌ {self.provider.upper()} API key not configured. Set {self.provider.upper()}_API_KEY environment variable"
                    else:
                        return f"❌ Failed to connect to {self.provider}. Check your network connection and try again."
                
                # Check for tool calls in response
                tool_calls = self._extract_tool_calls(response)
                
                if not tool_calls:
                    # No tool calls - extract final response and return
                    await progress_tracker.update_phase(ProcessingPhase.FINALIZING)
                    assistant_content = self._extract_response_content(response)
                    if not assistant_content:
                        # Robust fallback when provider returns empty content
                        fb = await self._fallback_response(messages)
                        try:
                            assistant_content = fb.get("choices", [{}])[0].get("message", {}).get("content", "I ran into an issue; please try again.")
                        except Exception:
                            assistant_content = "I ran into an issue; please try again."
                    
                    # Add final assistant response to history
                    response_timestamp = datetime.now().isoformat()
                    assistant_msg = ConversationMessage(role="assistant", content=assistant_content, timestamp=response_timestamp)
                    self.conversation_history.append(assistant_msg)
                    
                    return assistant_content
                
                # Execute tool calls and add to history
                assistant_content = self._extract_response_content(response)
                if assistant_content:
                    # Add assistant's message with tool calls to history
                    response_timestamp = datetime.now().isoformat()
                    assistant_msg = ConversationMessage(role="assistant", content=assistant_content, timestamp=response_timestamp)
                    self.conversation_history.append(assistant_msg)
                
                # Execute tool calls and add results as separate messages
                for i, tool_call in enumerate(tool_calls):
                    try:
                        tool_name = tool_call.get('function', {}).get('name', '')
                        parameters = tool_call.get('function', {}).get('arguments', {})
                        
                        # Create tool execution info for progress tracking
                        tool_description = ""
                        if isinstance(parameters, dict):
                            # Extract description from common parameter names
                            tool_description = (parameters.get('description', '') or 
                                              parameters.get('path', '') or 
                                              parameters.get('query', '') or 
                                              parameters.get('command', ''))
                            if isinstance(tool_description, str) and len(tool_description) > 50:
                                tool_description = tool_description[:47] + "..."
                        
                        tool_info = ToolExecutionInfo(
                            name=tool_name, 
                            description=tool_description,
                            index=i, 
                            total=len(tool_calls)
                        )
                        await progress_tracker.update_tool_execution(tool_info)
                        
                        # Parse arguments if they're a JSON string
                        if isinstance(parameters, str):
                            import json
                            try:
                                parameters = json.loads(parameters)
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse tool call arguments: {parameters}")
                                continue
                        
                        # Execute the tool
                        result = await self._execute_tool_call(tool_name, parameters)
                        
                        # Add tool result as a user message to continue the conversation
                        tool_timestamp = datetime.now().isoformat()
                        tool_result_msg = ConversationMessage(
                            role="user", 
                            content=f"Tool execution result for {tool_name}: {result}", 
                            timestamp=tool_timestamp
                        )
                        self.conversation_history.append(tool_result_msg)
                        
                    except Exception as e:
                        logger.error(f"Error executing tool call {tool_call}: {e}")
                        # Add error as user message
                        error_timestamp = datetime.now().isoformat()
                        error_msg = ConversationMessage(
                            role="user", 
                            content=f"Tool execution error for {tool_name}: {str(e)}", 
                            timestamp=error_timestamp
                        )
                        self.conversation_history.append(error_msg)
                
                # Continue to next turn to let LLM process tool results
            
            # If we hit max tool turns, ask for final analysis without tools
            logger.debug("Max tool turns reached, requesting final analysis")
            await progress_tracker.update_phase(ProcessingPhase.GENERATING_RESPONSE)
            
            # Add a message asking for final analysis
            analysis_timestamp = datetime.now().isoformat()
            analysis_request_msg = ConversationMessage(
                role="user", 
                content="Based on the tool execution results above, please provide your complete analysis and recommendations. Do not use any more tools, just give your comprehensive response based on what you've discovered.", 
                timestamp=analysis_timestamp
            )
            self.conversation_history.append(analysis_request_msg)
            
            # Get final analysis without allowing more tool calls
            messages = await self._prepare_messages()
            response = await self._call_llm_via_mcp(messages, enable_tools=False)
            
            if response:
                # Extract final analysis content
                await progress_tracker.update_phase(ProcessingPhase.FINALIZING)
                assistant_content = self._extract_response_content(response)
                if assistant_content:
                    # Add final assistant response to history
                    response_timestamp = datetime.now().isoformat()
                    assistant_msg = ConversationMessage(role="assistant", content=assistant_content, timestamp=response_timestamp)
                    self.conversation_history.append(assistant_msg)
                    
                    return assistant_content
            
            # Fallback if final analysis fails
            return "I've gathered information using tools but encountered an issue providing the final analysis. Please try asking your question again."
                
        except Exception as e:
            logger.error(f"Error in LLM communication: {e}")
            # Provide more helpful error messages
            error_str = str(e).lower()
            if "api key" in error_str or "unauthorized" in error_str:
                return f"❌ Authentication failed. Check your API key configuration for {self.provider}."
            elif "connection" in error_str or "network" in error_str or "timeout" in error_str:
                return f"❌ Network error connecting to {self.provider}. Check your internet connection."
            elif "rate limit" in error_str:
                return f"❌ Rate limit exceeded for {self.provider}. Please wait a moment and try again."
            else:
                return f"❌ Unexpected error with {self.provider}: {str(e)}\n\n💡 Try /config to check your configuration or /help for available commands."

    
    async def _send_simple_message(self, message: str, progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """Send a simple message without preprocessing/tool execution."""
        try:
            # Optional progress tracking for simple mode
            if progress_callback:
                from ..ui.v3.progress_tracker import SimpleProgressTracker
                progress_tracker = SimpleProgressTracker(progress_callback)
                await progress_tracker.update("Processing direct request", "🚀")
            
            # Prepare simple conversation for direct LLM call
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Respond directly and concisely."},
                {"role": "user", "content": message}
            ]
            
            response = await self._call_llm_via_mcp(messages, enable_tools=False)
            if response:
                content = self._extract_response_content(response)
                if content:
                    if progress_callback:
                        await progress_tracker.update("Response received", "✅")
                    return content
            
            return "❌ No response received from LLM. Check your configuration with /config"
            
        except Exception as e:
            logger.error(f"Error in simple message mode: {e}")
            return f"❌ Error in simple mode: {str(e)}"
    
    def supports_streaming(self) -> bool:
        """Check if the current provider supports streaming responses."""
        # Check if provider explicitly supports streaming
        streaming_providers = ["ollama-turbo", "ollama", "openai", "anthropic"]
        return self.provider in streaming_providers
    
    async def send_message_streaming(self, message: str, context: Optional[Dict[str, Any]] = None,
                                   progress_callback: Optional[Callable[[str], None]] = None):
        """Send a message and yield streaming response chunks.
        
        Yields:
            str: Response chunks as they arrive from the LLM
        """
        # Initialize progress tracker for streaming
        if progress_callback:
            from ..ui.v3.progress_tracker import ProgressTracker, ProcessingPhase
            progress_tracker = ProgressTracker(progress_callback)
            await progress_tracker.update_phase(ProcessingPhase.STREAMING)
        
        if not self.supports_streaming():
            # Fallback to non-streaming for unsupported providers
            response = await self.send_message(message, context, progress_callback)
            yield response
            return
        
        # Add message to conversation history
        self.conversation_history.append(
            ConversationMessage(role="user", content=message, context=context)
        )
        
        try:
            # Prepare messages for API call
            messages = await self._prepare_messages()
            
            # Track streaming progress
            chunk_count = 0
            if progress_callback:
                await progress_tracker.update_streaming(chunk_count)
            
            # Call appropriate streaming method based on provider
            if self.provider == "ollama-turbo":
                async for chunk in self._call_ollama_turbo_streaming(messages):
                    chunk_count += 1
                    if progress_callback and chunk_count % 10 == 0:  # Update every 10 chunks
                        await progress_tracker.update_streaming(chunk_count)
                    yield chunk
            elif self.provider == "ollama":
                async for chunk in self._call_ollama_streaming(messages):
                    chunk_count += 1
                    if progress_callback and chunk_count % 10 == 0:
                        await progress_tracker.update_streaming(chunk_count)
                    yield chunk
            elif self.provider == "openai":
                async for chunk in self._call_openai_streaming(messages):
                    chunk_count += 1
                    if progress_callback and chunk_count % 10 == 0:
                        await progress_tracker.update_streaming(chunk_count)
                    yield chunk
            elif self.provider == "anthropic":
                async for chunk in self._call_anthropic_streaming(messages):
                    chunk_count += 1
                    if progress_callback and chunk_count % 10 == 0:
                        await progress_tracker.update_streaming(chunk_count)
                    yield chunk
            else:
                # Fallback to non-streaming
                response = await self.send_message(message, context, progress_callback)
                yield response
        
        except Exception as e:
            logger.error(f"Error in streaming LLM communication: {e}")
            yield f"Sorry, I encountered an error: {str(e)}"
    
    async def _call_ollama_turbo_streaming(self, messages: List[Dict[str, str]]):
        """Call ollama-turbo with streaming support."""
        try:
            import httpx
            
            # Convert messages to ollama format
            ollama_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            
            max_tokens = await self._get_max_tokens_for_api()
            
            # Try ollama.com API first with streaming
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    "https://ollama.com/api/chat",
                    headers={
                        "Authorization": f"Bearer {self._get_api_key('ollama-turbo')}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": ollama_messages,
                        "options": {
                            "temperature": self.config.get("temperature", 0.7),
                            "num_predict": max_tokens
                        },
                        "stream": True
                    }
                ) as response:
                    if response.status_code == 200:
                        full_content = ""
                        async for line in response.aiter_lines():
                            if line.strip():
                                try:
                                    chunk_data = json.loads(line)
                                    if "message" in chunk_data and "content" in chunk_data["message"]:
                                        content = chunk_data["message"]["content"]
                                        if content:
                                            full_content += content
                                            yield content
                                    
                                    # Check if streaming is done
                                    if chunk_data.get("done", False):
                                        break
                                except json.JSONDecodeError:
                                    continue
                        
                        # Add response to conversation history
                        self.conversation_history.append(
                            ConversationMessage(role="assistant", content=full_content)
                        )
                        return
        
        except Exception as e:
            logger.warning(f"Ollama turbo streaming failed: {e}")
            # Fallback to non-streaming
            response = await self._call_ollama_turbo(messages, enable_tools=False)
            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                yield content
    
    async def _call_ollama_streaming(self, messages: List[Dict[str, str]]):
        """Call local ollama with streaming support."""
        try:
            import httpx
            
            ollama_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            max_tokens = await self._get_max_tokens_for_api()
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    "http://localhost:11434/api/chat",
                    json={
                        "model": self.model,
                        "messages": ollama_messages,
                        "options": {
                            "temperature": self.config.get("temperature", 0.7),
                            "num_predict": max_tokens
                        },
                        "stream": True
                    }
                ) as response:
                    if response.status_code == 200:
                        full_content = ""
                        async for line in response.aiter_lines():
                            if line.strip():
                                try:
                                    chunk_data = json.loads(line)
                                    if "message" in chunk_data and "content" in chunk_data["message"]:
                                        content = chunk_data["message"]["content"]
                                        if content:
                                            full_content += content
                                            yield content
                                    
                                    if chunk_data.get("done", False):
                                        break
                                except json.JSONDecodeError:
                                    continue
                        
                        # Add response to conversation history
                        self.conversation_history.append(
                            ConversationMessage(role="assistant", content=full_content)
                        )
                        return
        
        except Exception as e:
            logger.warning(f"Ollama streaming failed: {e}")
            # Fallback to non-streaming
            response = await self._call_ollama(messages, enable_tools=False)
            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                yield content
    
    async def _call_openai_streaming(self, messages: List[Dict[str, str]]):
        """Call OpenAI with streaming support (placeholder)."""
        # Placeholder for OpenAI streaming - would need actual implementation
        response = await self._call_openai(messages, enable_tools=False)
        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
            yield content
    
    async def _call_anthropic_streaming(self, messages: List[Dict[str, str]]):
        """Call Anthropic with streaming support (placeholder)."""
        # Placeholder for Anthropic streaming - would need actual implementation
        response = await self._call_anthropic(messages, enable_tools=False)
        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
            yield content
    
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
            
            api_key = self._get_api_key("ollama-turbo")
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
                model_family="gpt-oss",
                supports_streaming=True
            )
        elif "gpt-oss:20b" in model_name:
            return ModelCapabilities(
                context_window=128000,
                max_input_tokens=124000,
                max_output_tokens=4096,
                parameter_count="20B", 
                model_family="gpt-oss",
                supports_streaming=True
            )
        elif "mistral-nemo" in model_name:
            return ModelCapabilities(
                context_window=128000,
                max_input_tokens=124000,
                max_output_tokens=4096,
                parameter_count="12.2B",
                model_family="mistral",
                supports_streaming=True
            )
        elif "llama" in model_name:
            return ModelCapabilities(
                context_window=32000,
                max_input_tokens=28000,
                max_output_tokens=4096,
                parameter_count="Unknown",
                model_family="llama",
                supports_streaming=True
            )
        else:
            # Conservative defaults for unknown models
            return ModelCapabilities(
                context_window=8192,
                max_input_tokens=6144,
                max_output_tokens=2048,
                parameter_count="Unknown",
                model_family="unknown",
                supports_streaming=False  # Conservative default for unknown models
            )
    
    async def _get_max_tokens_for_api(self) -> int:
        """Get max tokens for API call based on detected capabilities."""
        try:
            capabilities = await self.get_model_capabilities()
            return capabilities.max_output_tokens
        except Exception:
            # Fallback to config
            return self.config.get("max_tokens", 1024)
    
    async def _call_llm_via_mcp(self, messages: List[Dict[str, str]], enable_tools: bool = True) -> Optional[Dict[str, Any]]:
        """Call the LLM using the configured provider with robust fallbacks.

        Order: primary -> OpenAI -> OpenRouter -> local Ollama -> Codex
        Only providers with a detected config/env key are attempted (except local Ollama).
        
        Args:
            messages: Messages to send to the LLM
            enable_tools: Whether to include tools in the LLM call (default: True)
        """
        try:
            logger.info(f"Calling LLM with provider: {self.provider}, model: {self.model}")
            primary = (self.provider or "ollama-turbo").lower()
            candidates = [primary, "openai", "openrouter", "anthropic", "ollama", "codex"]

            # Apply provider allowlist from config or env (comma-separated)
            enabled = []
            try:
                cfg_enabled = self.config.get("providers_enabled")
                if isinstance(cfg_enabled, list) and cfg_enabled:
                    enabled = [str(p).lower() for p in cfg_enabled]
            except Exception:
                enabled = []
            env_enabled = os.getenv("AGENTS_PROVIDERS_ENABLED", "")
            if env_enabled.strip():
                enabled = [p.strip().lower() for p in env_enabled.split(',') if p.strip()]
            if enabled:
                candidates = [c for c in candidates if c in enabled]
                if primary not in candidates:
                    # If current provider isn't enabled, switch to first enabled
                    if enabled:
                        candidates = [enabled[0]] + [c for c in candidates if c != enabled[0]]
            tried = set()

            def has_key(p: str) -> bool:
                if p == "ollama":
                    return True
                return bool(self._get_api_key(p))

            for prov in candidates:
                if prov in tried:
                    continue
                tried.add(prov)
                if prov != "ollama" and not has_key(prov):
                    logger.debug(f"Skipping {prov} (no key configured)")
                    continue
                try:
                    logger.info(f"Trying provider: {prov}")
                    result = None
                    if prov == "ollama-turbo":
                        result = await self._call_ollama_turbo(messages, enable_tools)
                    elif prov == "openai":
                        result = await self._call_openai(messages, enable_tools)
                    elif prov == "openrouter":
                        result = await self._call_openrouter(messages, enable_tools)
                    elif prov == "ollama":
                        result = await self._call_ollama(messages, enable_tools)
                    elif prov == "anthropic":
                        result = await self._call_anthropic(messages, enable_tools)
                    elif prov == "codex":
                        result = await self._call_codex(messages, enable_tools)

                    if result:
                        content = self._extract_response_content(result)
                        tool_calls = self._extract_tool_calls(result)
                        
                        # Success if we have content OR tool calls
                        if (content and content.strip()) or tool_calls:
                            logger.info(f"Provider {prov} succeeded")
                            return result
                        else:
                            logger.warning(f"Provider {prov} returned empty content and no tool calls; trying next")
                except Exception as e:
                    logger.warning(f"Provider {prov} failed: {e}")
                    continue

            logger.error("All candidate providers failed or returned empty content")
            return None
        except Exception as e:
            logger.error(f"Error calling LLM via MCP: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _network_enabled(self) -> bool:
        try:
            return os.getenv("AGENTS_NETWORK_ENABLED", "1") == "1"
        except Exception:
            return True

    async def _call_ollama_turbo(self, messages: List[Dict[str, str]], enable_tools: bool = True) -> Optional[Dict[str, Any]]:
        """Call Ollama with priority: 1) ollama.com API, 2) local proxy, 3) MCP."""
        try:
            if not self._network_enabled():
                logger.info("Network disabled by AGENTS_NETWORK_ENABLED=0")
                return None
            logger.info("Starting ollama-turbo call chain")
            import httpx
            
            # Convert messages to Ollama format
            ollama_messages = []
            for msg in messages:
                ollama_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Try ollama.com API first (primary)
            logger.info("Trying ollama.com API...")
            result = await self._try_ollama_com_api(ollama_messages, enable_tools)
            if result:
                logger.info("ollama.com API succeeded")
                return result
            else:
                logger.warning("ollama.com API failed")
                
            # Try local proxy (fallback)
            result = await self._try_local_proxy(ollama_messages, enable_tools)
            if result:
                return result
                
            # Try local Ollama instance (last resort for direct API)
            result = await self._try_local_ollama(ollama_messages, enable_tools)
            if result:
                return result
                
            # If all direct methods fail, try MCP as absolute last resort
            logger.warning("All direct Ollama methods failed, trying MCP...")
            return await self._call_ollama(messages, enable_tools)
                
        except ImportError as e:
            logger.error(f"httpx not available for Ollama API calls: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in ollama-turbo chain: {e}")
            return None
    
    async def _try_ollama_com_api(self, ollama_messages: List[Dict[str, str]], enable_tools: bool = True) -> Optional[Dict[str, Any]]:
        """Try ollama.com API with authentication."""
        try:
            import httpx
            import os
            
            # Get API key from environment variable
            api_key = self._get_api_key("ollama-turbo")
            if not api_key:
                logger.warning("OLLAMA_API_KEY not set, skipping ollama.com API")
                return None
            
            # Use detected max tokens
            max_tokens = await self._get_max_tokens_for_api()
            
            turbo_timeout = self._get_timeout("ollama_turbo", 30.0)
            async with httpx.AsyncClient(timeout=turbo_timeout) as client:
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
                        "tools": self.mcp_tools if enable_tools else None
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Successfully used ollama.com API - Response: {result}")
                    message = result.get("message", {})
                    content = message.get("content", "")
                    tool_calls = message.get("tool_calls", [])
                    
                    logger.info(f"Extracted content: '{content}'")
                    logger.info(f"Tool calls present: {len(tool_calls)}")
                    
                    # Return the response with tool calls for multi-turn handling in send_message
                    # DO NOT execute tools here - let send_message handle multi-turn execution
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": tool_calls  # Pass tool calls back
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
    
    async def _try_local_proxy(self, ollama_messages: List[Dict[str, str]], enable_tools: bool = True) -> Optional[Dict[str, Any]]:
        """Try local proxy that handles API key."""
        try:
            import httpx
            
            # Use detected max tokens
            max_tokens = await self._get_max_tokens_for_api()
            
            # Local proxy can be a thin shim; allow a bit more time (configurable)
            proxy_timeout = self._get_timeout("proxy", 60.0)
            async with httpx.AsyncClient(timeout=proxy_timeout) as client:
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
                        "tools": self.mcp_tools if enable_tools else None
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info("Successfully used local proxy")
                    message = result.get("message", {})
                    content = message.get("content", "")
                    tool_calls = message.get("tool_calls", [])
                    
                    # Return the response with tool calls for multi-turn handling in send_message
                    # DO NOT execute tools here - let send_message handle multi-turn execution
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": tool_calls  # Pass tool calls back
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
    
    async def _try_local_ollama(self, ollama_messages: List[Dict[str, str]], enable_tools: bool = True) -> Optional[Dict[str, Any]]:
        """Try local Ollama instance."""
        try:
            import httpx
            
            # Use detected max tokens
            max_tokens = await self._get_max_tokens_for_api()
            
            # Local Ollama can cold-start models; allow generous timeout (configurable)
            local_timeout = self._get_timeout("local_ollama", 120.0)
            async with httpx.AsyncClient(timeout=local_timeout) as client:
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
                        "tools": self.mcp_tools if enable_tools else None
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info("Successfully used local Ollama")
                    message = result.get("message", {})
                    content = message.get("content", "")
                    tool_calls = message.get("tool_calls", [])
                    
                    # Return the response with tool calls for multi-turn handling in send_message
                    # DO NOT execute tools here - let send_message handle multi-turn execution
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": tool_calls  # Pass tool calls back
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
    
    async def _call_ollama(self, messages: List[Dict[str, str]], enable_tools: bool = True) -> Optional[Dict[str, Any]]:
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
            return None
    
    async def _call_openai(self, messages: List[Dict[str, str]], enable_tools: bool = True) -> Optional[Dict[str, Any]]:
        """Call OpenAI via standard API (fallback if no MCP client)."""
        try:
            if not self._network_enabled():
                return None
            import openai
            
            # Use OpenAI API key from environment or config
            api_key = self._get_api_key("openai")
            if not api_key:
                logger.error("OpenAI API key not found")
                return None
            
            client = openai.OpenAI(api_key=api_key)
            # Enable tool use so the model can call functions when needed
            response = client.chat.completions.create(
                model=self.config.get("model", "gpt-4o-mini"),
                messages=messages,
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 1024),
                tools=self.mcp_tools if enable_tools else None,
                tool_choice="auto" if enable_tools else "none"
            )
            
            # Extract tool calls if present
            tool_calls = []
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                for tc in response.choices[0].message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
            
            return {
                "choices": [
                    {
                        "message": {
                            "role": response.choices[0].message.role,
                            "content": response.choices[0].message.content,
                            "tool_calls": tool_calls  # Include tool calls
                        },
                        "finish_reason": response.choices[0].finish_reason
                    }
                ]
            }
            
        except ImportError as e:
            logger.error(f"OpenAI library not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
            return None

    async def _call_openrouter(self, messages: List[Dict[str, str]], enable_tools: bool = True) -> Optional[Dict[str, Any]]:
        """Call OpenRouter chat completions API (OpenAI-compatible schema)."""
        try:
            if not self._network_enabled():
                return None
            import httpx
            api_key = self._get_api_key("openrouter")
            if not api_key:
                logger.warning("OpenRouter API key not found")
                return None
            model = self.config.get("model") or "openai/gpt-4o-mini"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": self.config.get("temperature", 0.7),
                "max_tokens": self.config.get("max_tokens", 1024),
                # Allow natural tool use
                "tools": self.mcp_tools if enable_tools else None,
                "tool_choice": "auto" if enable_tools else "none",
            }
            timeout = self._get_timeout("openrouter", 30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code != 200:
                logger.warning(f"OpenRouter failed: {resp.status_code}")
                return None
            data = resp.json()
            return data
        except Exception as e:
            logger.error(f"Error calling OpenRouter: {e}")
            return None

    async def _call_anthropic(self, messages: List[Dict[str, str]], enable_tools: bool = True) -> Optional[Dict[str, Any]]:
        """Call Anthropic Messages API (Claude) with basic tool-use support.

        Note: For full tool schema support, align tool specs with Anthropic's
        input_schema. This implementation maps our OpenAI-style tools when present.
        """
        try:
            if not self._network_enabled():
                return None
            import httpx
            api_key = self._get_api_key("anthropic")
            if not api_key:
                logger.warning("Anthropic API key not found")
                return None
            model = self.config.get("model") or "claude-3-5-sonnet-20241022"
            # Convert messages
            anthro_msgs = []
            for m in messages:
                role = m.get("role", "user")
                if role == "system":
                    anthro_msgs.append({"role": "user", "content": [{"type": "text", "text": m.get("content", "")} ]})
                else:
                    anthro_msgs.append({"role": role, "content": [{"type": "text", "text": m.get("content", "")} ]})

            # Map tools to Anthropic format
            def to_anthropic_tools() -> Optional[List[Dict[str, Any]]]:
                tools = []
                for t in (self.mcp_tools or []):
                    if t.get("type") == "function":
                        fn = t.get("function", {})
                        name = fn.get("name")
                        desc = fn.get("description", "")
                        schema = fn.get("parameters", {"type": "object"})
                        if name:
                            tools.append({"name": name, "description": desc, "input_schema": schema})
                return tools or None
            tools = to_anthropic_tools() if enable_tools else None

            max_tokens = self.config.get("max_tokens", 1024)
            temperature = self.config.get("temperature", 0.7)
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            timeout = self._get_timeout("anthropic", 30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                for _ in range(3):
                    payload = {
                        "model": model,
                        "messages": anthro_msgs,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    if tools:
                        payload["tools"] = tools
                    resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
                    if resp.status_code != 200:
                        logger.warning(f"Anthropic failed: {resp.status_code} {resp.text[:160]}")
                        return None
                    data = resp.json()
                    content_items = data.get("content", [])
                    # Look for tool_use items
                    tool_uses = [it for it in content_items if isinstance(it, dict) and it.get("type") == "tool_use"]
                    if not tool_uses:
                        # Return combined text
                        parts = []
                        for item in content_items:
                            if isinstance(item, dict) and item.get("type") == "text":
                                parts.append(item.get("text", ""))
                            elif isinstance(item, str):
                                parts.append(item)
                        combined = "\n".join(p for p in parts if p)
                        return {"choices": [{"message": {"role": "assistant", "content": combined}, "finish_reason": data.get("stop_reason", "stop")} ]}
                    # Execute tools and append results
                    tool_results = []
                    for tu in tool_uses:
                        name = tu.get("name")
                        input_args = tu.get("input", {})
                        tool_use_id = tu.get("id") or "tool-id"
                        try:
                            result = await self._execute_tool_call(name, input_args)
                        except Exception as e:
                            result = f"Error executing tool {name}: {e}"
                        tool_results.append({"type": "tool_result", "tool_use_id": tool_use_id, "content": result or ""})
                    anthro_msgs.append({"role": "user", "content": tool_results})
            return None
        except Exception as e:
            logger.error(f"Error calling Anthropic: {e}")
            return None
    
    async def _call_codex(self, messages: List[Dict[str, str]], enable_tools: bool = True) -> Optional[Dict[str, Any]]:
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
    
    async def _handle_tool_calls(self, response: Dict[str, Any]) -> Optional[str]:
        """Handle tool calls from LLM response and return results."""
        if not response:
            return None
        
        tool_calls = []
        
        # Extract tool calls from different response formats
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if 'message' in choice and 'tool_calls' in choice['message']:
                tool_calls = choice['message']['tool_calls']
        elif 'message' in response and 'tool_calls' in response['message']:
            tool_calls = response['message']['tool_calls']
            
        if not tool_calls:
            return None
            
        # Execute all tool calls
        results = []
        for tool_call in tool_calls:
            try:
                tool_name = tool_call.get('function', {}).get('name', '')
                parameters = tool_call.get('function', {}).get('arguments', {})
                
                # Parse arguments if they're a JSON string
                if isinstance(parameters, str):
                    import json
                    try:
                        parameters = json.loads(parameters)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse tool call arguments: {parameters}")
                        continue
                
                # Execute the tool
                result = await self._execute_tool_call(tool_name, parameters)
                results.append(f"Tool: {tool_name}\nResult: {result}")
                
            except Exception as e:
                logger.error(f"Error executing tool call {tool_call}: {e}")
                results.append(f"Tool: {tool_name}\nError: {str(e)}")
        
        return "\n\n".join(results) if results else None

    async def _execute_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Execute MCP tool calls and return results."""
        import os
        import glob
        from pathlib import Path
        
        try:
            # Helper: ensure paths are within the launch directory (project root)
            def _within_root(p: Path) -> Path:
                root = Path.cwd().resolve()
                candidate = (p if p.is_absolute() else (root / p)).resolve()
                if str(candidate) == str(root) or str(candidate).startswith(str(root) + os.sep):
                    return candidate
                raise PermissionError(f"Path outside project root: {p}")

            if tool_name == "read_file":
                file_path = parameters.get("file_path", "")
                try:
                    path = _within_root(Path(file_path))
                    if not path.exists():
                        return f"Error: File '{file_path}' does not exist."
                    if path.is_dir():
                        return f"Error: '{file_path}' is a directory, not a file."
                    content = path.read_text(encoding='utf-8')
                    return f"Contents of {file_path}:\n\n{content}"
                except Exception as e:
                    return f"Error reading file '{file_path}': {str(e)}"
            
            elif tool_name == "list_directory":
                dir_path = parameters.get("path", ".")
                try:
                    path = _within_root(Path(dir_path))
                    if not path.exists():
                        return f"Error: Directory '{dir_path}' does not exist."
                    if not path.is_dir():
                        return f"Error: '{dir_path}' is not a directory."
                    items = []
                    for item in sorted(path.iterdir()):
                        if item.is_dir():
                            items.append(f"{item.name}/")
                        else:
                            size = item.stat().st_size
                            items.append(f"{item.name} ({size} bytes)")
                    return f"Contents of {dir_path}:\n" + "\n".join(items)
                except Exception as e:
                    return f"Error listing directory '{dir_path}': {str(e)}"
            
            elif tool_name == "search_files":
                pattern = parameters.get("pattern", "")
                search_path = parameters.get("path", ".")
                try:
                    base_path = _within_root(Path(search_path))
                    matches = list(base_path.rglob(pattern))
                    if not matches:
                        return f"No files found matching pattern '{pattern}' in '{search_path}'"
                    results = []
                    for match in sorted(matches)[:20]:  # Limit to first 20 results
                        rel_path = match.relative_to(base_path)
                        if match.is_dir():
                            results.append(f"{rel_path}/")
                        else:
                            size = match.stat().st_size
                            results.append(f"{rel_path} ({size} bytes)")
                    return f"Files matching '{pattern}' in '{search_path}':\n" + "\n".join(results)
                except Exception as e:
                    return f"Error searching for '{pattern}': {str(e)}"
            
            elif tool_name == "get_file_info":
                file_path = parameters.get("path", "")
                try:
                    path = _within_root(Path(file_path))
                    if not path.exists():
                        return f"Error: Path '{file_path}' does not exist."
                    stat = path.stat()
                    info = [
                        f"Path: {path}",
                        f"Type: {'Directory' if path.is_dir() else 'File'}",
                        f"Size: {stat.st_size} bytes",
                        f"Modified: {stat.st_mtime}",
                    ]
                    if path.is_file():
                        try:
                            with open(path, 'r', encoding='utf-8') as f:
                                lines = sum(1 for _ in f)
                            info.append(f"Lines: {lines}")
                        except:
                            pass
                    return "\n".join(info)
                except Exception as e:
                    return f"Error getting info for '{file_path}': {str(e)}"
            elif tool_name == "write_file":
                file_path = parameters.get("file_path", "")
                content = parameters.get("content", "")
                encoding = parameters.get("encoding", "utf-8")
                try:
                    # Stage changes for review instead of writing directly
                    root = Path.cwd().resolve()
                    staged_root = root / "build" / "staging"
                    target = _within_root(Path(file_path))
                    stage_target = staged_root / target.relative_to(root)
                    stage_target.parent.mkdir(parents=True, exist_ok=True)
                    stage_target.write_text(content, encoding=encoding)
                    return f"STAGED: {len(content)} bytes -> {stage_target} (pending approval)"
                except Exception as e:
                    return f"Error staging file '{file_path}': {str(e)}"
            
            elif tool_name == "run_shell":
                cmd = parameters.get("command", "")
                timeout = int(parameters.get("timeout", 60))
                try:
                    import subprocess
                    proc = subprocess.run(
                        cmd,
                        cwd=str(Path.cwd()),
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=max(1, timeout)
                    )
                    return (
                        f"exit_code={proc.returncode}\n"
                        f"stdout:\n{proc.stdout}\n"
                        f"stderr:\n{proc.stderr}"
                    )
                except subprocess.TimeoutExpired:
                    return f"exit_code=124\nstdout:\n\nstderr:\nTimeout after {timeout}s running: {cmd}"
                except Exception as e:
                    return f"exit_code=1\nstdout:\n\nstderr:\n{str(e)}"
            elif tool_name == "list_staged_changes":
                try:
                    root = Path.cwd().resolve()
                    staged_root = root / "build" / "staging"
                    if not staged_root.exists():
                        return "No staged changes."
                    items = []
                    for p in staged_root.rglob('*'):
                        if p.is_file():
                            items.append(str(p.relative_to(staged_root)))
                    return "Staged files:\n" + ("\n".join(items) if items else "(none)")
                except Exception as e:
                    return f"Error listing staged changes: {e}"
            elif tool_name == "approve_changes":
                try:
                    root = Path.cwd().resolve()
                    staged_root = root / "build" / "staging"
                    if not staged_root.exists():
                        return "No staged changes to approve."
                    # Apply staged files into working tree
                    applied = []
                    for p in staged_root.rglob('*'):
                        if p.is_file():
                            rel = p.relative_to(staged_root)
                            dest = root / rel
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_text(p.read_text(encoding='utf-8'), encoding='utf-8')
                            applied.append(str(rel))
                    # Optional: git add + commit
                    commit_message = parameters.get('commit_message', 'chore: apply reviewed changes')
                    try:
                        import subprocess
                        subprocess.run('git add .', shell=True, cwd=str(root))
                        subprocess.run(f'git commit -m "{commit_message}"', shell=True, cwd=str(root))
                    except Exception:
                        pass
                    # Clear staging
                    for p in sorted(staged_root.rglob('*'), reverse=True):
                        try:
                            p.unlink() if p.is_file() else p.rmdir()
                        except Exception:
                            pass
                    return "Approved and applied:\n" + ("\n".join(applied) if applied else "(none)")
                except Exception as e:
                    return f"Error approving changes: {e}"
            elif tool_name == "discard_staged_changes":
                try:
                    root = Path.cwd().resolve()
                    staged_root = root / "build" / "staging"
                    if not staged_root.exists():
                        return "No staged changes."
                    for p in sorted(staged_root.rglob('*'), reverse=True):
                        try:
                            p.unlink() if p.is_file() else p.rmdir()
                        except Exception:
                            pass
                    return "Discarded all staged changes."
                except Exception as e:
                    return f"Error discarding staged changes: {e}"
            elif tool_name == "git_status":
                try:
                    import subprocess
                    proc = subprocess.run(
                        ['git','status','--porcelain=v1'], cwd=str(Path.cwd()), capture_output=True, text=True
                    )
                    return proc.stdout or "(clean)"
                except Exception as e:
                    return f"git_status error: {e}"
            elif tool_name == "git_diff":
                try:
                    import subprocess
                    path = parameters.get('path')
                    args = ['git','diff'] + ([path] if path else [])
                    proc = subprocess.run(args, cwd=str(Path.cwd()), capture_output=True, text=True)
                    return proc.stdout or "(no diff)"
                except Exception as e:
                    return f"git_diff error: {e}"
            elif tool_name == "github_create_pull_request":
                try:
                    import subprocess
                    title = parameters.get('title')
                    body = parameters.get('body', 'Automated PR')
                    base = parameters.get('base', 'main')
                    head = parameters.get('head', 'auto/agentsmcp')
                    subprocess.run(['git','checkout','-B',head], cwd=str(Path.cwd()))
                    subprocess.run(['git','push','-u','origin',head], cwd=str(Path.cwd()))
                    proc = subprocess.run(['gh','pr','create','--title',title,'--body',body,'--base',base,'--head',head], cwd=str(Path.cwd()), capture_output=True, text=True)
                    if proc.returncode == 0:
                        return proc.stdout.strip() or 'Pull request created.'
                    return f"gh pr create failed: {proc.stderr.strip()}"
                except Exception as e:
                    return f"github_create_pull_request error: {e}"
            
            else:
                # Fallback: try registry tools if available
                try:
                    return await self._execute_tool(tool_name, parameters)
                except Exception:
                    return f"Unknown tool: {tool_name}"
                
        except Exception as e:
            return f"Tool execution error: {str(e)}"
    
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
            'analyze the project in current directory', 'improvements you would make',
            'analyze the current repo', 'analyze repo', 'analyze the repo', 'suggest improvements',
            'analyze current', 'analyze this', 'analyze here'
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
    
    def _extract_tool_calls(self, response: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response."""
        if not response:
            return []
            
        tool_calls = []
        
        # Extract tool calls from different response formats
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if 'message' in choice and 'tool_calls' in choice['message']:
                tool_calls = choice['message']['tool_calls']
        elif 'message' in response and 'tool_calls' in response['message']:
            tool_calls = response['message']['tool_calls']
            
        return tool_calls if tool_calls else []
    
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
            summary += f"{timestamp_str}{role_indicator} {msg.content}\n"
        
        return summary
    
    async def _execute_tool(self, func_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool with given function name and arguments."""
        # Map function names to the registered tools
        tool_name_mapping = {
            "list_directory": "list_directory",
            "read_file": "read_file", 
            "write_file": "write_file"
        }
        
        # Get the actual tool name from mapping
        actual_tool_name = tool_name_mapping.get(func_name, func_name)
        
        # Get the tool from registry
        tool = tool_registry.get_tool(actual_tool_name)
        if not tool:
            raise Exception(f"Tool '{func_name}' not found in registry")
        
        # Map argument names if needed (some tools use different parameter names)
        if func_name == "list_directory":
            args = {"directory_path": args.get("path", ".")}
        elif func_name == "read_file":
            # Handle both 'path' and 'file_path' parameter names
            file_path = args.get("file_path", args.get("path", ""))
            args = {"file_path": file_path}
        
        logger.debug(f"Executing tool {func_name} with args: {args}")
        
        # Execute the tool (async-first)
        try:
            aexec = getattr(tool, "aexecute", None)
            if aexec is not None and callable(aexec):
                # If tool provides async execution, prefer it
                result = await aexec(**args)  # type: ignore[misc]
            else:
                # Fallback: run sync execute in a worker thread
                import asyncio as _asyncio
                result = await _asyncio.to_thread(tool.execute, **args)
        except Exception as e:
            logger.exception(f"Tool {func_name} execution failed: {e}")
            raise
        logger.debug(f"Tool {func_name} result: {result[:100]}...")
        
        return result
