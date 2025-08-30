"""Local LLM integration for natural language command parsing.

This module provides secure integration with Ollama local models for
natural language processing without sending sensitive data externally.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

try:
    import httpx
except ImportError:
    httpx = None

from ..models.nlp_models import (
    LLMConfig, 
    LLMResponse, 
    TokenUsage, 
    ParsedCommand,
    ParsingMethod,
    LLMUnavailableError,
    ContextTooLargeError,
    ParsingFailedError
)


logger = logging.getLogger(__name__)


class LocalLLMIntegration:
    """Integration with local LLM models via Ollama for command parsing."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.base_urls = [
            "http://127.0.0.1:11435",  # Local proxy
            "http://localhost:11434"   # Direct Ollama
        ]
        self.system_prompt = self._build_system_prompt()
        
        # Validate dependencies
        if httpx is None:
            logger.warning("httpx not available - LLM integration will be disabled")
        
        logger.info(f"Initialized LocalLLMIntegration with model: {self.config.model_name}")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for command parsing."""
        return """You are a natural language command parser for AgentsMCP CLI. Your job is to convert natural language inputs into structured commands.

IMPORTANT INSTRUCTIONS:
1. Parse the user's natural language input into a structured command
2. Identify the primary action/command and extract relevant parameters  
3. Return a JSON response with this exact structure:
{
    "action": "command_name",
    "parameters": {"key": "value", ...},
    "confidence": 0.85,
    "explanation": "I understood this as..."
}

SUPPORTED ACTIONS:
- analyze: Analyze code, projects, or files
- help: Show help information  
- status: Check system status
- tui: Start text user interface
- init: Initialize or setup projects
- optimize: Optimize costs or performance
- run: Execute commands or tools
- file: File operations (read, write, list)
- settings: Configuration and preferences
- dashboard: Monitoring and dashboards

PARAMETER EXTRACTION:
- Extract relevant details like file paths, targets, options
- Use descriptive parameter names (e.g., "target", "type", "interactive")
- Include boolean flags when mentioned (e.g., "interactive": true)

CONFIDENCE SCORING:
- 0.9-1.0: Very clear, unambiguous commands
- 0.7-0.9: Clear commands with minor ambiguity
- 0.5-0.7: Somewhat unclear but parseable
- 0.3-0.5: Highly ambiguous, multiple interpretations
- 0.0-0.3: Cannot parse reliably

EXAMPLES:
Input: "analyze my code"
Output: {"action": "analyze", "parameters": {"target": "code"}, "confidence": 0.9, "explanation": "I understood this as a request to analyze your code."}

Input: "help me set up the project"  
Output: {"action": "init", "parameters": {"interactive": true}, "confidence": 0.85, "explanation": "I understood this as a request to initialize/setup the project with interactive guidance."}

Input: "start the TUI"
Output: {"action": "tui", "parameters": {}, "confidence": 0.95, "explanation": "I understood this as a request to start the text user interface."}

RULES:
- Always respond with valid JSON only
- Include explanation of your understanding
- Use appropriate confidence scores
- Extract meaningful parameters when present
- Default to lower confidence if uncertain"""

    async def parse_command(
        self, 
        natural_input: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[ParsedCommand], str]:
        """Parse natural language input into structured command using LLM.
        
        Returns:
            Tuple of (parsed_command, explanation)
        """
        if httpx is None:
            raise LLMUnavailableError("httpx library not available for LLM calls")
        
        start_time = time.time()
        
        try:
            # Prepare the prompt with context
            prompt = self._prepare_prompt(natural_input, context)
            
            # Check context size
            estimated_tokens = len(prompt.split()) * 1.3  # Rough estimation
            if estimated_tokens > self.config.context_window * 0.8:  # 80% safety margin
                raise ContextTooLargeError(f"Context too large: ~{estimated_tokens} tokens")
            
            # Call LLM
            response = await self._call_llm(prompt)
            if not response:
                return None, "LLM call failed - no response received"
            
            # Parse LLM response
            parsed_command, explanation = await self._parse_llm_response(response)
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"LLM parsing completed in {processing_time:.1f}ms")
            
            return parsed_command, explanation
            
        except ContextTooLargeError:
            raise
        except LLMUnavailableError:
            raise  
        except Exception as e:
            logger.error(f"Error in LLM parsing: {e}")
            raise ParsingFailedError(f"LLM parsing failed: {str(e)}")
    
    def _prepare_prompt(self, natural_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Prepare the complete prompt with context."""
        context_str = ""
        
        if context:
            # Add relevant context information
            if context.get("command_history"):
                recent_commands = context["command_history"][-5:]  # Last 5 commands
                context_str += f"\nRecent commands: {', '.join(recent_commands)}"
            
            if context.get("current_directory"):
                context_str += f"\nCurrent directory: {context['current_directory']}"
                
            if context.get("recent_files"):
                recent_files = context["recent_files"][-3:]  # Last 3 files
                context_str += f"\nRecent files: {', '.join(recent_files)}"
        
        # Build complete prompt
        prompt = f"""Parse this natural language command into structured JSON:

INPUT: "{natural_input}"
{context_str}

Respond with JSON only:"""
        
        return prompt
    
    async def _call_llm(self, prompt: str) -> Optional[LLMResponse]:
        """Call local LLM with the prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Try each base URL until one works
        for base_url in self.base_urls:
            try:
                response = await self._make_ollama_request(base_url, messages)
                if response:
                    return response
            except Exception as e:
                logger.debug(f"Failed to call LLM at {base_url}: {e}")
                continue
        
        logger.error("All LLM endpoints failed")
        return None
    
    async def _make_ollama_request(self, base_url: str, messages: List[Dict[str, str]]) -> Optional[LLMResponse]:
        """Make request to Ollama API endpoint."""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": self.config.model_name,
                        "messages": messages,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_tokens,
                            "top_p": 0.9,
                            "top_k": 40
                        },
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    message = data.get("message", {})
                    content = message.get("content", "")
                    
                    if not content:
                        logger.warning("Received empty content from LLM")
                        return None
                    
                    processing_time = int((time.time() - start_time) * 1000)
                    
                    # Estimate token usage (Ollama doesn't provide exact counts)
                    input_tokens = sum(len(msg["content"].split()) for msg in messages)
                    output_tokens = len(content.split())
                    
                    token_usage = TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens
                    )
                    
                    return LLMResponse(
                        content=content,
                        finish_reason="stop",
                        token_usage=token_usage,
                        model_name=self.config.model_name,
                        processing_time_ms=processing_time,
                        temperature=self.config.temperature
                    )
                else:
                    logger.warning(f"LLM request failed with status {response.status_code}: {response.text}")
                    return None
                    
        except httpx.TimeoutException:
            logger.error(f"LLM request timed out after {self.config.timeout_seconds}s")
            return None
        except Exception as e:
            logger.error(f"Error making LLM request to {base_url}: {e}")
            return None
    
    async def _parse_llm_response(self, llm_response: LLMResponse) -> Tuple[Optional[ParsedCommand], str]:
        """Parse LLM response into structured command."""
        try:
            # Clean up the response content to extract JSON
            content = llm_response.content.strip()
            
            # Try to find JSON in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to clean up common JSON issues
                    json_str = self._clean_json_string(json_str)
                    parsed = json.loads(json_str)
            else:
                # Try parsing the entire content as JSON
                parsed = json.loads(content)
            
            # Validate required fields
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a JSON object")
            
            action = parsed.get("action")
            if not action or not isinstance(action, str):
                raise ValueError("Missing or invalid 'action' field")
            
            parameters = parsed.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {}
            
            confidence = parsed.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                confidence = 0.5
            
            explanation = parsed.get("explanation", "Parsed by LLM")
            if not isinstance(explanation, str):
                explanation = "Parsed by LLM"
            
            command = ParsedCommand(
                action=action,
                parameters=parameters,
                confidence=float(confidence),
                method=ParsingMethod.LLM
            )
            
            return command, explanation
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {llm_response.content}")
            return None, f"Failed to parse LLM response: {str(e)}"
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean common JSON formatting issues."""
        # Remove any markdown code block indicators
        json_str = json_str.replace('```json', '').replace('```', '')
        
        # Remove leading/trailing whitespace
        json_str = json_str.strip()
        
        # Fix common escaping issues
        json_str = json_str.replace('\\"', '"').replace("'", '"')
        
        return json_str
    
    async def check_availability(self) -> bool:
        """Check if LLM service is available."""
        if httpx is None:
            return False
        
        for base_url in self.base_urls:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{base_url}/api/version")
                    if response.status_code == 200:
                        logger.info(f"LLM service available at {base_url}")
                        return True
            except Exception:
                continue
        
        logger.warning("No LLM services available")
        return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured model."""
        if httpx is None:
            return {"error": "httpx not available"}
        
        for base_url in self.base_urls:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        f"{base_url}/api/show",
                        json={"name": self.config.model_name}
                    )
                    
                    if response.status_code == 200:
                        return response.json()
            except Exception as e:
                logger.debug(f"Failed to get model info from {base_url}: {e}")
                continue
        
        return {"error": "Could not retrieve model information"}
    
    def update_config(self, new_config: LLMConfig) -> None:
        """Update LLM configuration."""
        self.config = new_config
        logger.info(f"Updated LLM config - model: {self.config.model_name}")
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimation."""
        # Simple estimation: ~1.3 tokens per word on average
        return max(1, int(len(text.split()) * 1.3))
    
    def is_context_too_large(self, text: str) -> bool:
        """Check if text would exceed context window."""
        estimated_tokens = self.estimate_tokens(text)
        return estimated_tokens > (self.config.context_window * 0.8)  # 80% safety margin
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test LLM connection and return status."""
        if httpx is None:
            return {
                "available": False,
                "error": "httpx library not installed",
                "endpoints": []
            }
        
        test_results = []
        
        for base_url in self.base_urls:
            start_time = time.time()
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    # Test basic connectivity
                    response = await client.get(f"{base_url}/api/version")
                    latency = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        test_results.append({
                            "url": base_url,
                            "available": True,
                            "latency_ms": round(latency, 1),
                            "version": response.json()
                        })
                    else:
                        test_results.append({
                            "url": base_url,
                            "available": False,
                            "error": f"HTTP {response.status_code}"
                        })
            except Exception as e:
                test_results.append({
                    "url": base_url,
                    "available": False,
                    "error": str(e)
                })
        
        # Check if any endpoints are available
        available = any(result.get("available", False) for result in test_results)
        
        return {
            "available": available,
            "model": self.config.model_name,
            "endpoints": test_results,
            "config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "timeout_seconds": self.config.timeout_seconds
            }
        }