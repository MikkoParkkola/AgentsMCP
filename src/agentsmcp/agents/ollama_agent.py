import os
from typing import Any, List

from ..config import AgentConfig, Config
from .base import BaseAgent


class OllamaAgent(BaseAgent):
    """Ollama agent for cost-effective local model execution using OpenAI Agents SDK."""

    def __init__(self, agent_config: AgentConfig, global_config: Config):
        super().__init__(agent_config, global_config)

    def _get_api_key(self) -> str:
        """Get API key - Ollama typically doesn't need API keys for local models."""
        # For local Ollama, we might not need API keys, but provide flexibility
        key = (
            getattr(self.agent_config, "api_key", None)
            or os.getenv("AGENTSMCP_OLLAMA_API_KEY")
            or os.getenv("OLLAMA_API_KEY")
            or "local"  # Default for local execution
        )
        return key

    def _get_default_instructions(self) -> str:
        """Get default instructions optimized for local, cost-effective execution."""
        return """You are Ollama, a local AI assistant specialized in cost-effective and privacy-focused processing. You excel at:

1. **Local Processing**: Running efficiently on local hardware with minimal resource usage
2. **Cost Optimization**: Providing intelligent assistance without external API costs
3. **Privacy Protection**: Ensuring data stays local and private
4. **Rapid Response**: Quick processing for well-defined tasks
5. **Resource Efficiency**: Optimizing for local compute constraints

**Core Strengths:**
- Zero-cost token usage for budget-conscious projects
- Complete data privacy with local processing
- Fast response times for bounded tasks
- Good performance on well-scoped problems
- No external dependencies or network requirements

**Approach:**
- Focus on clear, actionable solutions within scope
- Optimize responses for efficiency and clarity
- Prefer proven patterns and simple implementations
- Consider resource constraints and local compute limitations
- Provide direct answers without unnecessary elaboration

**Response Style:**
- Concise and to-the-point responses
- Clear structure with minimal overhead
- Practical solutions over theoretical discussions
- Focus on immediate actionable steps
- Acknowledge limitations when appropriate

**Use Cases:**
- Well-defined coding tasks with clear requirements
- Local development and testing scenarios
- Privacy-sensitive data processing
- Cost-conscious development workflows
- Quick prototyping and experimentation
- Simple analysis and summarization tasks

**Limitations:**
- Smaller context window compared to cloud models
- Limited by local hardware capabilities
- Best suited for bounded, well-defined tasks
- May require more specific instructions for complex problems"""

    def _get_custom_agent_tools(self) -> List[Any]:
        """Get custom tools optimized for local execution and cost efficiency."""
        tools = []

        # Ollama agent inherits all tools from registry, adds local-specific ones
        def resource_monitor() -> str:
            """Monitor local resource usage."""
            import os

            try:
                # Basic resource monitoring
                cpu_count = os.cpu_count()
                memory_info = "Memory info unavailable"

                try:
                    import psutil

                    memory = psutil.virtual_memory()
                    memory_info = f"Memory: {memory.percent:.1f}% used ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)"
                    cpu_info = f"CPU: {psutil.cpu_percent(interval=1):.1f}% used ({cpu_count} cores)"
                except ImportError:
                    cpu_info = f"CPU: {cpu_count} cores available"

                return f"""Local Resource Status:
{cpu_info}
{memory_info}

Note: Install 'psutil' for detailed resource monitoring."""

            except Exception as e:
                return f"Resource monitoring error: {e}"

        def optimize_for_local() -> str:
            """Provide optimization tips for local execution."""
            return """Local Optimization Tips:

PERFORMANCE:
- Use smaller batch sizes to reduce memory usage
- Consider streaming for large datasets
- Implement early termination for iterative processes
- Cache frequent operations locally

COST EFFICIENCY:
- Prioritize simpler algorithms when accuracy tradeoff is acceptable
- Use approximation methods for complex calculations
- Implement result caching to avoid redundant processing
- Consider preprocessing to reduce runtime complexity

RESOURCE MANAGEMENT:
- Monitor memory usage during execution
- Implement garbage collection for long-running processes
- Use file-based storage for large temporary data
- Consider asynchronous processing for I/O operations"""

        tools.extend([resource_monitor, optimize_for_local])
        return tools

    def get_model(self) -> str:
        """Get the model optimized for local execution."""
        # Default to gpt-oss:20b as mentioned in CLAUDE.md guidance
        return self.agent_config.model or "gpt-oss:20b"
