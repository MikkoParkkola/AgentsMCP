import asyncio
from typing import Dict, Any

from .base import BaseAgent
from ..config import AgentConfig, Config


class OllamaAgent(BaseAgent):
    """Ollama agent for cost-effective local model execution."""
    
    def __init__(self, agent_config: AgentConfig, global_config: Config):
        super().__init__(agent_config, global_config)
    
    async def execute_task(self, task: str) -> str:
        """Execute a task using Ollama local models."""
        try:
            # Import MCP Ollama module
            from mcp import ollama
            
            # Prepare the request
            model = self.get_model() or "llama2"
            
            messages = [
                {
                    "role": "system", 
                    "content": self.get_system_prompt()
                },
                {
                    "role": "user", 
                    "content": task
                }
            ]
            
            # Make the chat completion request
            result = await ollama.chat_completion({
                "model": model,
                "messages": messages,
                "temperature": self.get_temperature()
            })
            
            # Extract response from Ollama format
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return result.get("response", "Task completed")
            
        except ImportError:
            # Fallback if MCP Ollama is not available
            return await self._simulate_ollama_response(task)
        except Exception as e:
            raise Exception(f"Ollama agent error: {str(e)}")
    
    async def _simulate_ollama_response(self, task: str) -> str:
        """Simulate an Ollama response for development/testing."""
        # Simulate processing time
        await asyncio.sleep(2)
        
        model = self.get_model() or "llama2"
        
        response = f"""[Ollama Agent - {model}]

Task: {task}

I'll help you with this task using local model processing. This provides:
- Cost-effective execution
- Privacy-focused processing
- Fast response times

Available tools: {[tool.name for tool in self.tools]}
Temperature: {self.get_temperature()}

This is a simulation response. In production, this would use the actual Ollama API to run local models like Llama2, CodeLlama, or other open-source models."""
        
        return response