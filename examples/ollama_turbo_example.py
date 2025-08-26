#!/usr/bin/env python3
"""
Example: Using Ollama Turbo provider with AgentsMCP

This example demonstrates how to:
1. Configure ollama-turbo provider with API key
2. Create agents that use cloud-hosted Ollama models
3. Use both general-purpose and coding-specialized agents
"""

import os
import sys
sys.path.insert(0, '../src')

from agentsmcp.config import Config, ProviderType, AgentConfig

def setup_ollama_turbo():
    """Setup example for ollama-turbo provider"""
    print("🔧 Setting up Ollama Turbo provider...")
    
    # 1. Set up environment variable (normally done in shell)
    api_key = os.getenv("OLLAMA_TURBO_API_KEY")
    if not api_key:
        print("⚠️  OLLAMA_TURBO_API_KEY environment variable not set")
        print("   To use Ollama Turbo, set: export OLLAMA_TURBO_API_KEY=your_api_key")
        return False
    
    print(f"✅ Found OLLAMA_TURBO_API_KEY: {api_key[:10]}...")
    return True

def create_ollama_turbo_agents():
    """Example of creating ollama-turbo agents programmatically"""
    print("🤖 Creating Ollama Turbo agents...")
    
    # General-purpose cloud agent using gpt-oss:120b
    general_agent = AgentConfig(
        type="ollama-turbo-assistant",
        provider=ProviderType.OLLAMA_TURBO,
        model="gpt-oss:120b",
        model_priority=["gpt-oss:120b", "gpt-oss:20b"],
        system_prompt="You are a powerful cloud-powered AI assistant using Ollama Turbo's gpt-oss:120b model for comprehensive task handling.",
        tools=["filesystem", "git", "bash", "web_search"],
        mcp=["git-mcp", "agentsmcp-self"],
        max_tokens=4000,
        temperature=0.2
    )
    
    # Specialized coding agent using gpt-oss:120b (excellent for coding)
    coding_agent = AgentConfig(
        type="ollama-turbo-coder",
        provider=ProviderType.OLLAMA_TURBO,
        model="gpt-oss:120b",
        model_priority=["gpt-oss:120b", "gpt-oss:20b"],
        system_prompt="You are a specialized coding assistant using Ollama Turbo's powerful gpt-oss:120b model - excellent for advanced code generation, debugging, and analysis.",
        tools=["filesystem", "git", "bash"],
        mcp=["git-mcp", "agentsmcp-self"],
        max_tokens=6000,
        temperature=0.1
    )
    
    print("✅ General agent configuration:")
    print(f"   Provider: {general_agent.provider}")
    print(f"   Model: {general_agent.model}")
    print(f"   Tools: {general_agent.tools}")
    
    print("✅ Coding agent configuration:")
    print(f"   Provider: {coding_agent.provider}")
    print(f"   Model: {coding_agent.model}")
    print(f"   Temperature: {coding_agent.temperature}")
    
    return general_agent, coding_agent

def compare_providers():
    """Compare different provider options"""
    print("⚖️  Comparing provider options...")
    
    providers = {
        "Local Ollama": {
            "provider": ProviderType.OLLAMA,
            "api_base": "http://localhost:11434",
            "auth_required": False,
            "cost": "Free (local compute)",
            "latency": "Low (local)",
            "availability": "Requires local setup",
            "default_model": "gpt-oss:20b"
        },
        "Ollama Turbo": {
            "provider": ProviderType.OLLAMA_TURBO,
            "api_base": "https://ollama.com",
            "auth_required": True,
            "cost": "Fixed subscription (unlimited)",
            "latency": "Medium (cloud)",
            "availability": "Always available (rate limits)",
            "default_model": "gpt-oss:120b"
        },
        "OpenAI": {
            "provider": ProviderType.OPENAI,
            "api_base": "https://api.openai.com/v1",
            "auth_required": True,
            "cost": "Premium (high)",
            "latency": "Low (optimized)",
            "availability": "Always available"
        }
    }
    
    print("\n📊 Provider Comparison:")
    print("-" * 60)
    for name, details in providers.items():
        print(f"{name:15} | {details['cost']:20} | {details['latency']:15} | {'Auth: ' + str(details['auth_required'])}")
    print("-" * 60)

def usage_examples():
    """Show usage examples for ollama-turbo"""
    print("💡 Usage Examples:")
    print("\n1. Interactive CLI usage:")
    print("   OLLAMA_TURBO_API_KEY=your_key PYTHONPATH=src python -m agentsmcp --mode interactive")
    
    print("\n2. Create ollama-turbo agent:")
    print("   User: 'Create a coding agent using ollama-turbo with deepseek-coder'")
    print("   AgentsMCP: Will create specialized agent with cloud Ollama models")
    
    print("\n3. Use ollama-turbo for unlimited cloud tasks:")
    print("   User: 'Use ollama-turbo to analyze this file (unlimited usage)'")
    print("   AgentsMCP: Will delegate to cloud Ollama for efficient processing")
    
    print("\n4. Environment setup:")
    print("   export OLLAMA_TURBO_API_KEY=your_api_key_here")
    print("   # Now all ollama-turbo agents will have access to cloud models")

def main():
    """Main example runner"""
    print("🚀 Ollama Turbo Provider Example\n")
    
    # Check environment setup
    if not setup_ollama_turbo():
        print("\n⚠️  Skipping live tests due to missing API key")
        print("    Configuration and examples will still be shown.\n")
    
    # Show agent creation
    general_agent, coding_agent = create_ollama_turbo_agents()
    print()
    
    # Compare providers
    compare_providers()
    print()
    
    # Show usage examples
    usage_examples()
    
    print("\n🎯 Key Benefits of Ollama Turbo:")
    print("✅ Same familiar Ollama models, but in the cloud")
    print("✅ No local setup required - always available")
    print("✅ Scales automatically with demand")
    print("✅ Fixed subscription cost - unlimited usage")
    print("✅ Same API spec as local Ollama - easy migration")
    
    print("\n🔧 Configuration Summary:")
    print("- Provider: ollama-turbo")
    print("- API Base: https://ollama.com")
    print("- Auth: OLLAMA_TURBO_API_KEY environment variable")
    print("- Models: gpt-oss:120b, gpt-oss:20b (dynamically discovered)")
    print("- Usage: Identical to local Ollama, but requires API key and may have rate limits")

if __name__ == "__main__":
    main()