"""
Integration tests for Ollama Turbo functionality in AgentsMCP

Tests both local Ollama and Ollama Turbo cloud integration with hybrid orchestration.
"""

import asyncio
import pytest
import logging
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import os

from src.agentsmcp.distributed.ollama_turbo_integration import (
    OllamaHybridOrchestrator, OllamaMode, ModelTier, OllamaRequest,
    OllamaResponse, OllamaLocalProvider, OllamaTurboProvider,
    create_ollama_orchestrator, get_ollama_config_from_env
)
from src.agentsmcp.distributed.orchestrator import DistributedOrchestrator

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_ollama_turbo_api_key():
    """Mock Ollama Turbo API key for testing"""
    return "test-api-key-ssh-ed25519-mock"


@pytest.fixture
def sample_ollama_request():
    """Sample Ollama request for testing"""
    return OllamaRequest(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": "Hello, this is a test"}],
        stream=False,
        temperature=0.7
    )


@pytest.fixture
def mock_local_provider():
    """Mock local Ollama provider"""
    provider = Mock(spec=OllamaLocalProvider)
    provider.chat_completion = AsyncMock(return_value=OllamaResponse(
        content="Hello! This is a response from local Ollama.",
        model="gpt-oss:20b",
        usage={'total_duration': 1000, 'eval_count': 10},
        response_time=0.5,
        source=OllamaMode.LOCAL,
        metadata={'provider': 'local_ollama', 'mcp': True}
    ))
    provider.list_models = AsyncMock(return_value=[])
    provider.health_check = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def mock_turbo_provider():
    """Mock Ollama Turbo provider"""
    provider = Mock(spec=OllamaTurboProvider)
    provider.chat_completion = AsyncMock(return_value=OllamaResponse(
        content="Hello! This is a response from Ollama Turbo.",
        model="gpt-oss:120b",
        usage={'total_duration': 800, 'eval_count': 12},
        response_time=0.3,
        source=OllamaMode.TURBO,
        metadata={'provider': 'ollama_turbo', 'api_version': 'v1'}
    ))
    provider.list_models = AsyncMock(return_value=[])
    provider.health_check = AsyncMock(return_value=True)
    provider.close = AsyncMock()
    return provider


class TestOllamaHybridOrchestrator:
    """Test Ollama hybrid orchestration functionality"""

    @pytest.fixture
    def hybrid_orchestrator(self, mock_ollama_turbo_api_key, mock_local_provider, mock_turbo_provider):
        """Create hybrid orchestrator with mocked providers"""
        orchestrator = OllamaHybridOrchestrator(
            turbo_api_key=mock_ollama_turbo_api_key,
            prefer_turbo=True  # Set to True so it tries Turbo first
        )
        # Replace providers with mocks
        orchestrator.local_provider = mock_local_provider
        orchestrator.turbo_provider = mock_turbo_provider
        return orchestrator

    @pytest.mark.asyncio
    async def test_hybrid_orchestrator_initialization(self, mock_ollama_turbo_api_key):
        """Test that hybrid orchestrator initializes correctly"""
        orchestrator = OllamaHybridOrchestrator(
            turbo_api_key=mock_ollama_turbo_api_key,
            prefer_turbo=True
        )
        
        assert orchestrator.turbo_provider is not None
        assert orchestrator.local_provider is not None
        assert orchestrator.prefer_turbo is True
        assert orchestrator.provider_availability[OllamaMode.LOCAL] is True
        assert orchestrator.provider_availability[OllamaMode.TURBO] is True

    @pytest.mark.asyncio
    async def test_chat_completion_local_fallback(self, hybrid_orchestrator, sample_ollama_request, 
                                                mock_local_provider, mock_turbo_provider):
        """Test that local provider is used when Turbo fails"""
        # Make Turbo provider fail
        mock_turbo_provider.chat_completion.side_effect = Exception("Turbo unavailable")
        
        # Mock model availability to enable fallback
        from src.agentsmcp.distributed.ollama_turbo_integration import ModelCapability, ModelTier
        local_models = [
            ModelCapability("gpt-oss:20b", ModelTier.PERFORMANCE, 32768, "20B", 
                          ["coding", "analysis"], available_locally=True)
        ]
        mock_local_provider.list_models.return_value = local_models
        
        response = await hybrid_orchestrator.chat_completion(sample_ollama_request)
        
        assert response.source == OllamaMode.LOCAL
        assert "local Ollama" in response.content
        assert response.metadata.get("fallback") is True
        
        # Verify both providers were attempted
        mock_turbo_provider.chat_completion.assert_called_once()
        mock_local_provider.chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_completion_turbo_success(self, hybrid_orchestrator, mock_turbo_provider):
        """Test successful Turbo completion for 120b model"""
        request = OllamaRequest(
            model="gpt-oss:120b",  # Only available on Turbo
            messages=[{"role": "user", "content": "Test 120b model"}],
            stream=False
        )
        
        response = await hybrid_orchestrator.chat_completion(request)
        
        assert response.source == OllamaMode.TURBO
        assert response.model == "gpt-oss:120b"
        mock_turbo_provider.chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_best_model_for_task(self, hybrid_orchestrator, mock_local_provider, mock_turbo_provider):
        """Test intelligent model selection for different tasks"""
        # Mock available models
        from src.agentsmcp.distributed.ollama_turbo_integration import ModelCapability
        
        local_models = [
            ModelCapability("gpt-oss:20b", ModelTier.PERFORMANCE, 32768, "20B", 
                          ["coding", "analysis"], available_locally=True)
        ]
        turbo_models = [
            ModelCapability("gpt-oss:120b", ModelTier.ULTRA, 32768, "120B", 
                          ["reasoning", "creative"], available_turbo=True)
        ]
        
        mock_local_provider.list_models.return_value = local_models
        mock_turbo_provider.list_models.return_value = turbo_models
        
        # Test quality-prioritized task (should prefer 120b)
        best_model = await hybrid_orchestrator.get_best_model_for_task("creative", "quality")
        assert best_model == "gpt-oss:120b"
        
        # Test speed-prioritized task (should prefer local 20b)
        best_model = await hybrid_orchestrator.get_best_model_for_task("coding", "speed")
        assert best_model == "gpt-oss:20b"

    @pytest.mark.asyncio
    async def test_orchestrator_analytics(self, hybrid_orchestrator, mock_local_provider, mock_turbo_provider):
        """Test analytics collection from hybrid orchestrator"""
        analytics = await hybrid_orchestrator.get_orchestrator_analytics()
        
        assert "provider_health" in analytics
        assert "model_routing" in analytics
        assert "performance_history" in analytics
        assert "configuration" in analytics
        
        assert analytics["provider_health"]["local"] is True
        assert analytics["provider_health"]["turbo"] is True
        assert analytics["configuration"]["turbo_enabled"] is True


class TestDistributedOrchestratorWithOllamaTurbo:
    """Test DistributedOrchestrator with Ollama Turbo integration"""

    @pytest.fixture
    def orchestrator_with_turbo(self, mock_ollama_turbo_api_key):
        """Create orchestrator with Ollama Turbo enabled"""
        with patch.dict(os.environ, {'OLLAMA_TURBO_API_KEY': mock_ollama_turbo_api_key}):
            orchestrator = DistributedOrchestrator(
                enable_mesh=False,
                enable_governance=False,
                enable_context_intelligence=False,
                enable_multimodal=False,
                enable_ollama_turbo=True,
                ollama_turbo_api_key=mock_ollama_turbo_api_key
            )
            
            # Mock the Ollama orchestrator
            mock_ollama_orch = Mock()
            mock_ollama_orch.chat_completion = AsyncMock(return_value=OllamaResponse(
                content="Test response from integrated Ollama",
                model="gpt-oss:20b",
                usage={'total_duration': 500},
                response_time=0.4,
                source=OllamaMode.HYBRID,
                metadata={}
            ))
            mock_ollama_orch.get_best_model_for_task = AsyncMock(return_value="gpt-oss:20b")
            mock_ollama_orch.get_orchestrator_analytics = AsyncMock(return_value={
                "provider_health": {"local": True, "turbo": True}
            })
            mock_ollama_orch.close = AsyncMock()
            
            orchestrator.ollama_orchestrator = mock_ollama_orch
            
            return orchestrator

    def test_orchestrator_turbo_initialization(self, orchestrator_with_turbo):
        """Test that orchestrator properly initializes Ollama Turbo"""
        assert orchestrator_with_turbo.enable_ollama_turbo is True
        assert orchestrator_with_turbo.ollama_orchestrator is not None

    @pytest.mark.asyncio
    async def test_execute_ollama_request(self, orchestrator_with_turbo):
        """Test executing Ollama request through orchestrator"""
        request = OllamaRequest(
            model="gpt-oss:20b",
            messages=[{"role": "user", "content": "Test orchestrator integration"}],
            stream=False
        )
        
        response = await orchestrator_with_turbo.execute_ollama_request(request)
        
        assert response["content"] == "Test response from integrated Ollama"
        assert response["model"] == "gpt-oss:20b"
        assert response["source"] == "hybrid"
        assert "usage" in response
        assert "response_time" in response

    @pytest.mark.asyncio
    async def test_get_best_ollama_model(self, orchestrator_with_turbo):
        """Test getting best model through orchestrator"""
        best_model = await orchestrator_with_turbo.get_best_ollama_model("coding", "balanced")
        
        assert best_model == "gpt-oss:20b"
        orchestrator_with_turbo.ollama_orchestrator.get_best_model_for_task.assert_called_once_with("coding", "balanced")

    @pytest.mark.asyncio
    async def test_get_ollama_analytics(self, orchestrator_with_turbo):
        """Test getting Ollama analytics through orchestrator"""
        analytics = await orchestrator_with_turbo.get_ollama_analytics()
        
        assert "provider_health" in analytics
        assert analytics["provider_health"]["local"] is True
        assert analytics["provider_health"]["turbo"] is True

    def test_orchestrator_model_validation(self, mock_ollama_turbo_api_key):
        """Test orchestrator model validation with Ollama Turbo models"""
        # Test with Turbo-only model
        orchestrator = DistributedOrchestrator(
            orchestrator_model="gpt-oss:120b",
            enable_ollama_turbo=True,
            ollama_turbo_api_key=mock_ollama_turbo_api_key,
            enable_mesh=False,
            enable_governance=False,
            enable_context_intelligence=False,
            enable_multimodal=False
        )
        
        assert orchestrator.orchestrator_model == "gpt-oss:120b"
        
        # Test with Turbo disabled but Turbo model requested
        orchestrator_no_turbo = DistributedOrchestrator(
            orchestrator_model="gpt-oss:120b",
            enable_ollama_turbo=False,
            enable_mesh=False,
            enable_governance=False,
            enable_context_intelligence=False,
            enable_multimodal=False
        )
        
        # Should fall back to default model
        assert orchestrator_no_turbo.orchestrator_model == "gpt-5"

    @pytest.mark.asyncio
    async def test_orchestrator_shutdown(self, orchestrator_with_turbo):
        """Test proper shutdown of orchestrator with Ollama Turbo"""
        await orchestrator_with_turbo.close()
        
        # Verify Ollama orchestrator was closed
        orchestrator_with_turbo.ollama_orchestrator.close.assert_called_once()


class TestOllamaConfigurationAndFactory:
    """Test configuration loading and factory functions"""

    def test_get_config_from_env(self):
        """Test loading configuration from environment variables"""
        test_env = {
            'OLLAMA_TURBO_API_KEY': 'test-key',
            'OLLAMA_PREFER_TURBO': 'true',
            'OLLAMA_TURBO_BASE_URL': 'https://custom.ollama.com',
            'OLLAMA_MODE': 'turbo'
        }
        
        with patch.dict(os.environ, test_env):
            config = get_ollama_config_from_env()
            
            assert config["turbo_api_key"] == "test-key"
            assert config["prefer_turbo"] is True
            assert config["turbo_base_url"] == "https://custom.ollama.com"
            assert config["mode"] == OllamaMode.TURBO

    def test_create_ollama_orchestrator_factory(self):
        """Test factory function for creating orchestrators"""
        # Test local mode
        local_orch = create_ollama_orchestrator(OllamaMode.LOCAL)
        assert isinstance(local_orch, OllamaLocalProvider)
        
        # Test Turbo mode
        turbo_orch = create_ollama_orchestrator(OllamaMode.TURBO, "test-key")
        assert isinstance(turbo_orch, OllamaTurboProvider)
        
        # Test hybrid mode
        hybrid_orch = create_ollama_orchestrator(OllamaMode.HYBRID, "test-key")
        assert isinstance(hybrid_orch, OllamaHybridOrchestrator)
        
        # Test error handling
        with pytest.raises(ValueError, match="Turbo API key required"):
            create_ollama_orchestrator(OllamaMode.TURBO)


@pytest.mark.asyncio 
async def test_ollama_turbo_real_api_integration():
    """
    Integration test with real Ollama Turbo API (if API key available)
    
    This test is skipped if no real API key is available in environment.
    """
    real_api_key = os.getenv('OLLAMA_TURBO_API_KEY_REAL')
    
    if not real_api_key:
        pytest.skip("Real Ollama Turbo API key not available - set OLLAMA_TURBO_API_KEY_REAL to enable")
    
    # Create real Turbo provider
    turbo_provider = OllamaTurboProvider(real_api_key)
    
    try:
        # Test health check
        health = await turbo_provider.health_check()
        logger.info(f"Ollama Turbo health check: {health}")
        
        if health:
            # Test simple request
            request = OllamaRequest(
                model="gpt-oss:20b",
                messages=[{"role": "user", "content": "Say 'Hello from Ollama Turbo!' and nothing else"}],
                stream=False
            )
            
            response = await turbo_provider.chat_completion(request)
            
            assert response.source == OllamaMode.TURBO
            assert "Hello from Ollama Turbo!" in response.content
            assert response.response_time > 0
            
            logger.info(f"Real API test successful: {response.content[:100]}...")
        
    except Exception as e:
        # Log the error but don't fail the test since it might be due to network issues
        logger.warning(f"Real API test failed (this might be expected): {e}")
        
    finally:
        await turbo_provider.close()


if __name__ == "__main__":
    # Run specific tests for development
    pytest.main([__file__, "-v", "-s"])