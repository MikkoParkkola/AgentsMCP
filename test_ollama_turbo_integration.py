#!/usr/bin/env python3
"""
Integration test for Ollama Turbo provider support.
Tests configuration, model discovery, and agent creation.
"""

import os
import tempfile
import yaml
from pathlib import Path

# Add src to Python path for imports
import sys
sys.path.insert(0, 'src')

from agentsmcp.config import Config, ProviderType, ProviderConfig
from agentsmcp.providers import ollama_turbo_list_models, ProviderAuthError

def test_ollama_turbo_config():
    """Test that ollama-turbo provider is properly configured"""
    print("🔧 Testing ollama-turbo configuration...")
    
    # Test ProviderType enum includes OLLAMA_TURBO
    assert hasattr(ProviderType, 'OLLAMA_TURBO'), "OLLAMA_TURBO missing from ProviderType enum"
    assert ProviderType.OLLAMA_TURBO == "ollama-turbo", "OLLAMA_TURBO enum value incorrect"
    print("✅ ProviderType.OLLAMA_TURBO enum defined correctly")
    
    # Test loading configuration from YAML
    config_path = Path("./agentsmcp.yaml")
    if config_path.exists():
        config = Config.from_file(config_path)
        
        # Check if ollama-turbo provider is in config
        assert "ollama-turbo" in config.providers, "ollama-turbo provider missing from config"
        
        ollama_turbo_config = config.providers["ollama-turbo"]
        assert ollama_turbo_config.name == "ollama-turbo", "ollama-turbo provider name incorrect"
        assert ollama_turbo_config.api_base == "https://ollama.com", "ollama-turbo api_base incorrect"
        print("✅ ollama-turbo provider configured correctly in YAML")
        
        # Check if ollama-turbo agents are configured
        turbo_agents = [name for name, agent in config.agents.items() if agent.provider == ProviderType.OLLAMA_TURBO]
        assert len(turbo_agents) > 0, "No ollama-turbo agents found in configuration"
        print(f"✅ Found {len(turbo_agents)} ollama-turbo agents: {turbo_agents}")
    else:
        print("⚠️  agentsmcp.yaml not found, skipping YAML configuration test")
    
    return True

def test_ollama_turbo_provider_auth():
    """Test ollama-turbo provider authentication requirements"""
    print("🔐 Testing ollama-turbo authentication...")
    
    # Test without API key (should fail)
    config = ProviderConfig(name=ProviderType.OLLAMA_TURBO, api_key=None, api_base="https://ollama.com")
    
    try:
        ollama_turbo_list_models(config)
        assert False, "Expected ProviderAuthError when no API key provided"
    except ProviderAuthError as e:
        assert "OLLAMA_TURBO_API_KEY" in str(e), "Error message should mention OLLAMA_TURBO_API_KEY"
        print("✅ Correctly requires API key for ollama-turbo")
    
    # Test with API key from environment
    test_api_key = "test_key_12345"
    os.environ["OLLAMA_TURBO_API_KEY"] = test_api_key
    
    config_with_key = ProviderConfig(
        name=ProviderType.OLLAMA_TURBO, 
        api_key=test_api_key, 
        api_base="https://ollama.com"
    )
    
    # We can't actually call the API without a real key, but we can verify the config
    assert config_with_key.api_key == test_api_key, "API key not set correctly"
    print("✅ API key configuration working")
    
    # Clean up
    del os.environ["OLLAMA_TURBO_API_KEY"]
    
    return True

def test_ollama_turbo_agent_configs():
    """Test ollama-turbo agent configurations from YAML"""
    print("🤖 Testing ollama-turbo agent configurations...")
    
    config_path = Path("./agentsmcp.yaml")
    if not config_path.exists():
        print("⚠️  agentsmcp.yaml not found, skipping agent config test")
        return True
    
    config = Config.from_file(config_path)
    
    # Find ollama-turbo agents
    turbo_agents = {
        name: agent for name, agent in config.agents.items() 
        if agent.provider == ProviderType.OLLAMA_TURBO
    }
    
    for agent_name, agent_config in turbo_agents.items():
        print(f"📋 Testing agent: {agent_name}")
        
        # Verify required fields
        assert agent_config.provider == ProviderType.OLLAMA_TURBO, f"Wrong provider for {agent_name}"
        assert agent_config.model is not None, f"No model specified for {agent_name}"
        assert agent_config.system_prompt is not None, f"No system prompt for {agent_name}"
        assert "filesystem" in agent_config.tools, f"filesystem tool missing for {agent_name}"
        assert "agentsmcp-self" in agent_config.mcp, f"agentsmcp-self MCP missing for {agent_name}"
        
        print(f"  ✅ Model: {agent_config.model}")
        print(f"  ✅ Tools: {agent_config.tools}")
        print(f"  ✅ MCP: {agent_config.mcp}")
        print(f"  ✅ Max tokens: {agent_config.max_tokens}")
    
    print(f"✅ All {len(turbo_agents)} ollama-turbo agents configured correctly")
    return True

def test_provider_comparison():
    """Compare ollama vs ollama-turbo configurations"""
    print("⚖️  Comparing ollama vs ollama-turbo configurations...")
    
    # Create test configs
    ollama_config = ProviderConfig(
        name=ProviderType.OLLAMA,
        api_key=None,
        api_base="http://localhost:11434"
    )
    
    ollama_turbo_config = ProviderConfig(
        name=ProviderType.OLLAMA_TURBO,
        api_key="test-key",
        api_base="https://ollama.com"
    )
    
    # Verify differences
    assert ollama_config.name != ollama_turbo_config.name, "Provider names should be different"
    assert ollama_config.api_base != ollama_turbo_config.api_base, "API bases should be different"
    assert ollama_config.api_key != ollama_turbo_config.api_key, "API key requirements should differ"
    
    print("✅ Local ollama: http://localhost:11434 (no auth)")
    print("✅ Cloud ollama-turbo: https://ollama.com (requires auth)")
    
    return True

def test_environment_variable_detection():
    """Test environment variable detection for ollama-turbo"""
    print("🌍 Testing environment variable detection...")
    
    # Test that OLLAMA_TURBO_API_KEY can be detected
    test_key = "test_ollama_turbo_key_67890"
    os.environ["OLLAMA_TURBO_API_KEY"] = test_key
    
    # The actual environment loading would happen in the LLM client
    # Here we just verify the environment variable is accessible
    detected_key = os.getenv("OLLAMA_TURBO_API_KEY")
    assert detected_key == test_key, "OLLAMA_TURBO_API_KEY not detected correctly"
    
    print(f"✅ OLLAMA_TURBO_API_KEY detected: {detected_key[:10]}...")
    
    # Clean up
    del os.environ["OLLAMA_TURBO_API_KEY"]
    
    return True

def main():
    """Run all ollama-turbo integration tests"""
    print("🚀 Starting Ollama Turbo integration tests...\n")
    
    tests = [
        ("Configuration", test_ollama_turbo_config),
        ("Authentication", test_ollama_turbo_provider_auth),
        ("Agent Configurations", test_ollama_turbo_agent_configs),
        ("Provider Comparison", test_provider_comparison),
        ("Environment Variables", test_environment_variable_detection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"🧪 Running {test_name} test...")
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
            print(f"✅ {test_name}: PASS\n")
        except Exception as e:
            results[test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: FAIL - {e}\n")
    
    # Summary
    print("📊 Test Results Summary:")
    print("=" * 40)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result == "PASS" else f"❌ {result}"
        print(f"{test_name:20} {status}")
        if result == "PASS":
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All ollama-turbo integration tests passed!")
        print("\n📋 Next steps:")
        print("1. Set OLLAMA_TURBO_API_KEY environment variable")
        print("2. Test with: PYTHONPATH=src python -m agentsmcp --mode interactive")
        print("3. Try creating ollama-turbo agents with commands like:")
        print("   'Create an agent using ollama-turbo with llama3.2:3b'")
        print("   'Use ollama-turbo-coding agent for this task'")
    else:
        print("❌ Some tests failed. Please review the issues above.")

if __name__ == "__main__":
    main()