"""
Ollama Turbo Integration for Enhanced AgentsMCP

Provides seamless integration between local Ollama and Ollama Turbo cloud service,
enabling automatic model selection and performance optimization for agent workloads.
"""

import asyncio
import aiohttp
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
import time

logger = logging.getLogger(__name__)


class OllamaMode(Enum):
    """Ollama deployment modes"""
    LOCAL = "local"
    TURBO = "turbo"
    HYBRID = "hybrid"  # Automatic selection based on performance/availability


class ModelTier(Enum):
    """Model performance tiers"""
    ULTRA = "ultra"      # gpt-oss:120b (Turbo only)
    PERFORMANCE = "performance"  # gpt-oss:20b (Local + Turbo)
    EFFICIENCY = "efficiency"    # Local models optimized for speed
    

@dataclass
class ModelCapability:
    """Model capability profile"""
    model_name: str
    tier: ModelTier
    context_window: int
    parameter_count: str
    specializations: List[str] = field(default_factory=list)
    cost_per_token: float = 0.0  # Turbo models may have costs
    performance_score: float = 1.0  # Relative performance benchmark
    available_locally: bool = False
    available_turbo: bool = False


@dataclass
class OllamaRequest:
    """Standardized request format for Ollama services"""
    model: str
    messages: List[Dict[str, str]]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class OllamaResponse:
    """Standardized response from Ollama services"""
    content: str
    model: str
    usage: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_time: float = 0.0
    source: OllamaMode = OllamaMode.LOCAL


class OllamaProvider(ABC):
    """Abstract base for Ollama providers"""
    
    @abstractmethod
    async def chat_completion(self, request: OllamaRequest) -> OllamaResponse:
        """Execute chat completion request"""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[ModelCapability]:
        """List available models"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider health"""
        pass


class OllamaLocalProvider(OllamaProvider):
    """Local Ollama provider using MCP ollama tools"""
    
    def __init__(self):
        self.available_models: Dict[str, ModelCapability] = {}
        self._last_model_refresh = datetime.min
        
    async def chat_completion(self, request: OllamaRequest) -> OllamaResponse:
        """Execute chat completion via local Ollama"""
        start_time = time.time()
        
        try:
            # Import MCP ollama functions dynamically to avoid circular imports
            from ..tools.mcp_ollama import chat_completion
            
            result = await chat_completion(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature
            )
            
            response_time = time.time() - start_time
            
            return OllamaResponse(
                content=result.get('message', {}).get('content', ''),
                model=request.model,
                usage={
                    'total_duration': result.get('total_duration', 0),
                    'load_duration': result.get('load_duration', 0),
                    'prompt_eval_count': result.get('prompt_eval_count', 0),
                    'eval_count': result.get('eval_count', 0)
                },
                response_time=response_time,
                source=OllamaMode.LOCAL,
                metadata={'provider': 'local_ollama', 'mcp': True}
            )
            
        except Exception as e:
            logger.error(f"Local Ollama request failed: {e}")
            raise
    
    async def list_models(self) -> List[ModelCapability]:
        """List locally available models"""
        try:
            from ..tools.mcp_ollama import list_models
            
            models_data = await list_models()
            capabilities = []
            
            for model_info in models_data.get('models', []):
                model_name = model_info['name']
                param_size = model_info.get('details', {}).get('parameter_size', 'Unknown')
                
                # Determine tier based on model name and size
                tier = ModelTier.EFFICIENCY
                if 'gpt-oss:20b' in model_name:
                    tier = ModelTier.PERFORMANCE
                elif any(x in model_name.lower() for x in ['70b', '120b', 'large']):
                    tier = ModelTier.ULTRA
                
                capability = ModelCapability(
                    model_name=model_name,
                    tier=tier,
                    context_window=self._estimate_context_window(model_name),
                    parameter_count=param_size,
                    available_locally=True,
                    available_turbo=False,
                    performance_score=self._calculate_performance_score(model_name, param_size)
                )
                capabilities.append(capability)
            
            self.available_models = {cap.model_name: cap for cap in capabilities}
            self._last_model_refresh = datetime.utcnow()
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Failed to list local models: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check local Ollama health"""
        try:
            models = await self.list_models()
            return len(models) > 0
        except Exception:
            return False
    
    def _estimate_context_window(self, model_name: str) -> int:
        """Estimate context window based on model name"""
        if 'gpt-oss' in model_name:
            return 32768  # GPT-OSS models typically have 32k context
        elif any(x in model_name.lower() for x in ['gemma', 'llama']):
            return 8192   # Most other models default to 8k
        return 4096       # Conservative default
    
    def _calculate_performance_score(self, model_name: str, param_size: str) -> float:
        """Calculate relative performance score"""
        base_score = 1.0
        
        # Boost for known high-performance models
        if 'gpt-oss:20b' in model_name:
            base_score = 1.5
        elif 'mistral' in model_name.lower():
            base_score = 1.2
        elif 'gemma' in model_name.lower():
            base_score = 1.1
        
        # Adjust for parameter count
        try:
            if 'B' in param_size:
                param_num = float(param_size.replace('B', '').replace('M', '').strip())
                if param_num >= 20:
                    base_score *= 1.3
                elif param_num >= 10:
                    base_score *= 1.1
        except ValueError:
            pass
        
        return base_score


class OllamaTurboProvider(OllamaProvider):
    """Ollama Turbo cloud provider"""
    
    def __init__(self, api_key: str, base_url: str = "https://ollama.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.available_models: Dict[str, ModelCapability] = {}
        
        # Initialize known Turbo models
        self._initialize_turbo_models()
    
    def _initialize_turbo_models(self):
        """Initialize known Ollama Turbo model capabilities"""
        turbo_models = [
            ModelCapability(
                model_name="gpt-oss:120b",
                tier=ModelTier.ULTRA,
                context_window=32768,
                parameter_count="120B",
                specializations=["reasoning", "coding", "analysis", "creative"],
                performance_score=2.0,
                available_locally=False,
                available_turbo=True
            ),
            ModelCapability(
                model_name="gpt-oss:20b",
                tier=ModelTier.PERFORMANCE,
                context_window=32768,
                parameter_count="20B",
                specializations=["coding", "analysis", "general"],
                performance_score=1.5,
                available_locally=False,
                available_turbo=True
            )
        ]
        
        self.available_models = {model.model_name: model for model in turbo_models}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session
    
    async def chat_completion(self, request: OllamaRequest) -> OllamaResponse:
        """Execute chat completion via Ollama Turbo"""
        start_time = time.time()
        session = await self._get_session()
        
        # Prepare Ollama API request
        payload = {
            "model": request.model,
            "messages": request.messages,
            "stream": request.stream
        }
        
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        
        try:
            async with session.post(f"{self.base_url}/api/chat", json=payload) as response:
                if response.status == 401:
                    raise Exception("Ollama Turbo authentication failed - check API key")
                elif response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama Turbo request failed: {response.status} - {error_text}")
                
                result = await response.json()
                response_time = time.time() - start_time
                
                return OllamaResponse(
                    content=result.get('message', {}).get('content', ''),
                    model=request.model,
                    usage={
                        'total_duration': result.get('total_duration', 0),
                        'load_duration': result.get('load_duration', 0),
                        'prompt_eval_count': result.get('prompt_eval_count', 0),
                        'eval_count': result.get('eval_count', 0)
                    },
                    response_time=response_time,
                    source=OllamaMode.TURBO,
                    metadata={'provider': 'ollama_turbo', 'api_version': 'v1'}
                )
        
        except aiohttp.ClientError as e:
            logger.error(f"Ollama Turbo network error: {e}")
            raise Exception(f"Network error connecting to Ollama Turbo: {e}")
        except Exception as e:
            logger.error(f"Ollama Turbo request error: {e}")
            raise
    
    async def list_models(self) -> List[ModelCapability]:
        """List available Turbo models"""
        return list(self.available_models.values())
    
    async def health_check(self) -> bool:
        """Check Ollama Turbo health"""
        try:
            # Simple health check with minimal request
            test_request = OllamaRequest(
                model="gpt-oss:20b",
                messages=[{"role": "user", "content": "test"}],
                stream=False
            )
            
            # Don't actually send the test - just check if we can create a session
            session = await self._get_session()
            return not session.closed
            
        except Exception as e:
            logger.warning(f"Ollama Turbo health check failed: {e}")
            return False
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()


class OllamaHybridOrchestrator:
    """
    Intelligent orchestrator for Ollama Local + Turbo hybrid deployment
    
    Features:
    - Automatic provider selection based on performance, availability, and cost
    - Failover between local and turbo providers
    - Model capability matching and routing
    - Performance monitoring and optimization
    """
    
    def __init__(self, turbo_api_key: Optional[str] = None, prefer_turbo: bool = False):
        self.local_provider = OllamaLocalProvider()
        self.turbo_provider = OllamaTurboProvider(turbo_api_key) if turbo_api_key else None
        self.prefer_turbo = prefer_turbo
        
        self.performance_history: Dict[str, List[float]] = {}
        self.provider_availability: Dict[OllamaMode, bool] = {
            OllamaMode.LOCAL: True,
            OllamaMode.TURBO: bool(turbo_api_key)
        }
        
        # Model routing preferences
        self.model_routing: Dict[str, OllamaMode] = {}
        self._initialize_routing_preferences()
    
    def _initialize_routing_preferences(self):
        """Initialize intelligent model routing preferences"""
        # Route ultra-tier models to Turbo (if available)
        self.model_routing["gpt-oss:120b"] = OllamaMode.TURBO
        
        # Performance models can use either (prefer based on settings)
        preferred_mode = OllamaMode.TURBO if self.prefer_turbo else OllamaMode.LOCAL
        self.model_routing["gpt-oss:20b"] = preferred_mode
    
    async def chat_completion(self, request: OllamaRequest) -> OllamaResponse:
        """Execute chat completion with intelligent routing"""
        provider = await self._select_provider(request)
        
        try:
            response = await provider.chat_completion(request)
            self._record_performance(request.model, response.response_time)
            return response
            
        except Exception as primary_error:
            logger.warning(f"Primary provider failed for {request.model}: {primary_error}")
            
            # Try fallback provider
            fallback_provider = await self._get_fallback_provider(provider, request)
            if fallback_provider:
                try:
                    logger.info(f"Attempting fallback for {request.model}")
                    response = await fallback_provider.chat_completion(request)
                    response.metadata = response.metadata.copy()  # Ensure we can modify it
                    response.metadata["fallback"] = True
                    return response
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
            
            raise primary_error
    
    async def _select_provider(self, request: OllamaRequest) -> OllamaProvider:
        """Select optimal provider for the request"""
        model = request.model
        
        # Check explicit routing preferences
        if model in self.model_routing:
            preferred_mode = self.model_routing[model]
            if preferred_mode == OllamaMode.TURBO and self.turbo_provider:
                if await self._check_provider_availability(OllamaMode.TURBO):
                    return self.turbo_provider
            elif preferred_mode == OllamaMode.LOCAL:
                if await self._check_provider_availability(OllamaMode.LOCAL):
                    return self.local_provider
        
        # Model-specific logic
        if model == "gpt-oss:120b":
            # 120b model only available on Turbo
            if self.turbo_provider and await self._check_provider_availability(OllamaMode.TURBO):
                return self.turbo_provider
            else:
                raise Exception("gpt-oss:120b model requires Ollama Turbo access")
        
        # Performance-based selection for other models
        if self.prefer_turbo and self.turbo_provider:
            if await self._check_provider_availability(OllamaMode.TURBO):
                return self.turbo_provider
        
        # Default to local provider
        if await self._check_provider_availability(OllamaMode.LOCAL):
            return self.local_provider
        
        # Last resort: try Turbo even if not preferred
        if self.turbo_provider and await self._check_provider_availability(OllamaMode.TURBO):
            return self.turbo_provider
        
        raise Exception("No available Ollama providers")
    
    async def _get_fallback_provider(self, failed_provider: OllamaProvider, request: OllamaRequest) -> Optional[OllamaProvider]:
        """Get fallback provider when primary fails"""
        if isinstance(failed_provider, OllamaTurboProvider):
            # Turbo failed, try local if model is available
            if await self._model_available_locally(request.model):
                return self.local_provider
        elif isinstance(failed_provider, OllamaLocalProvider):
            # Local failed, try Turbo if available
            if self.turbo_provider and await self._model_available_turbo(request.model):
                return self.turbo_provider
        
        return None
    
    async def _check_provider_availability(self, mode: OllamaMode) -> bool:
        """Check if provider is available and healthy"""
        if not self.provider_availability[mode]:
            return False
        
        try:
            if mode == OllamaMode.LOCAL:
                return await self.local_provider.health_check()
            elif mode == OllamaMode.TURBO and self.turbo_provider:
                return await self.turbo_provider.health_check()
        except Exception:
            self.provider_availability[mode] = False
        
        return False
    
    async def _model_available_locally(self, model: str) -> bool:
        """Check if model is available locally"""
        try:
            models = await self.local_provider.list_models()
            return any(m.model_name == model for m in models)
        except Exception:
            return False
    
    async def _model_available_turbo(self, model: str) -> bool:
        """Check if model is available on Turbo"""
        if not self.turbo_provider:
            return False
        
        try:
            models = await self.turbo_provider.list_models()
            return any(m.model_name == model for m in models)
        except Exception:
            return False
    
    def _record_performance(self, model: str, response_time: float):
        """Record performance metrics for optimization"""
        if model not in self.performance_history:
            self.performance_history[model] = []
        
        self.performance_history[model].append(response_time)
        
        # Keep only recent history (last 100 requests)
        if len(self.performance_history[model]) > 100:
            self.performance_history[model] = self.performance_history[model][-100:]
    
    async def get_best_model_for_task(self, task_type: str, performance_priority: str = "balanced") -> str:
        """
        Recommend best model for specific task type
        
        Args:
            task_type: Type of task (coding, analysis, creative, etc.)
            performance_priority: "speed", "quality", "balanced"
        """
        available_models = []
        
        # Collect models from all providers
        try:
            local_models = await self.local_provider.list_models()
            available_models.extend(local_models)
        except Exception:
            pass
        
        if self.turbo_provider:
            try:
                turbo_models = await self.turbo_provider.list_models()
                available_models.extend(turbo_models)
            except Exception:
                pass
        
        if not available_models:
            raise Exception("No models available")
        
        # Filter by task specialization
        suitable_models = []
        for model in available_models:
            if not model.specializations or task_type in model.specializations:
                suitable_models.append(model)
        
        if not suitable_models:
            suitable_models = available_models  # Fallback to all models
        
        # Select based on performance priority
        if performance_priority == "speed":
            # Prefer local models with good performance scores
            suitable_models.sort(key=lambda m: (-m.performance_score if m.available_locally else m.performance_score))
        elif performance_priority == "quality":
            # Prefer highest tier models regardless of location
            suitable_models.sort(key=lambda m: (m.tier.value, -m.performance_score))
        else:  # balanced
            # Consider both performance and availability
            suitable_models.sort(key=lambda m: (-m.performance_score, m.tier.value))
        
        return suitable_models[0].model_name
    
    async def get_orchestrator_analytics(self) -> Dict[str, Any]:
        """Get analytics about orchestrator performance"""
        local_health = await self._check_provider_availability(OllamaMode.LOCAL)
        turbo_health = await self._check_provider_availability(OllamaMode.TURBO)
        
        # Calculate average response times
        avg_response_times = {}
        for model, times in self.performance_history.items():
            if times:
                avg_response_times[model] = sum(times) / len(times)
        
        return {
            "provider_health": {
                "local": local_health,
                "turbo": turbo_health
            },
            "model_routing": self.model_routing,
            "performance_history": {
                "tracked_models": list(self.performance_history.keys()),
                "average_response_times": avg_response_times,
                "total_requests": sum(len(times) for times in self.performance_history.values())
            },
            "configuration": {
                "prefer_turbo": self.prefer_turbo,
                "turbo_enabled": bool(self.turbo_provider)
            }
        }
    
    async def close(self):
        """Clean shutdown of all providers"""
        if self.turbo_provider:
            await self.turbo_provider.close()


# Configuration and Factory Functions

def create_ollama_orchestrator(
    mode: OllamaMode = OllamaMode.HYBRID,
    turbo_api_key: Optional[str] = None,
    prefer_turbo: bool = False
) -> Union[OllamaLocalProvider, OllamaTurboProvider, OllamaHybridOrchestrator]:
    """Factory function to create appropriate Ollama orchestrator"""
    
    if mode == OllamaMode.LOCAL:
        return OllamaLocalProvider()
    elif mode == OllamaMode.TURBO:
        if not turbo_api_key:
            raise ValueError("Turbo API key required for TURBO mode")
        return OllamaTurboProvider(turbo_api_key)
    elif mode == OllamaMode.HYBRID:
        return OllamaHybridOrchestrator(turbo_api_key, prefer_turbo)
    else:
        raise ValueError(f"Unknown Ollama mode: {mode}")


def get_ollama_config_from_env() -> Dict[str, Any]:
    """Load Ollama configuration from environment variables"""
    return {
        "turbo_api_key": os.getenv("OLLAMA_API_KEY"),
        "prefer_turbo": os.getenv("OLLAMA_PREFER_TURBO", "false").lower() == "true",
        "turbo_base_url": os.getenv("OLLAMA_TURBO_BASE_URL", "https://ollama.com"),
        "mode": OllamaMode(os.getenv("OLLAMA_MODE", "hybrid"))
    }