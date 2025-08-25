"""
Example usage of Ollama Turbo integration in AgentsMCP

This example demonstrates how to use both local Ollama and Ollama Turbo
models through the enhanced AgentsMCP architecture.
"""

import asyncio
import os
import logging
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agentsmcp.distributed import (
    DistributedOrchestrator, OllamaHybridOrchestrator, OllamaRequest,
    OllamaMode, create_ollama_orchestrator, get_ollama_config_from_env
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_ollama_turbo_integration():
    """Demonstrate Ollama Turbo integration capabilities"""
    
    print("🦙 AgentsMCP Ollama Turbo Integration Demo")
    print("=" * 50)
    
    # Real API key for testing - use environment variable in production
    api_key = os.getenv("OLLAMA_TURBO_API_KEY", "5e226b755fb34e9c8ffa6937a034745b.YpPRc_wr8NcTJ6JKpMPvsVW9")
    
    use_mock = False  # Now we have a working API key
    
    # 1. Direct Ollama Hybrid Orchestrator Usage
    print("\n1. 🔄 Direct Ollama Hybrid Orchestrator")
    print("-" * 40)
    
    if not use_mock:
        try:
            hybrid_orchestrator = OllamaHybridOrchestrator(
                turbo_api_key=api_key,
                prefer_turbo=True
            )
            
            # Test with 20b model (available locally and on Turbo)
            request_20b = OllamaRequest(
                model="gpt-oss:20b",
                messages=[{
                    "role": "user", 
                    "content": "Explain the benefits of hybrid AI model deployment in 2 sentences."
                }],
                stream=False,
                temperature=0.7
            )
            
            response = await hybrid_orchestrator.chat_completion(request_20b)
            print(f"Model: {response.model}")
            print(f"Source: {response.source.value}")
            print(f"Response: {response.content}")
            print(f"Response Time: {response.response_time:.2f}s")
            
            # Test model selection for different tasks
            best_coding_model = await hybrid_orchestrator.get_best_model_for_task("coding", "speed")
            best_creative_model = await hybrid_orchestrator.get_best_model_for_task("creative", "quality")
            
            print(f"\nBest model for coding (speed): {best_coding_model}")
            print(f"Best model for creative (quality): {best_creative_model}")
            
            # Get analytics
            analytics = await hybrid_orchestrator.get_orchestrator_analytics()
            print(f"\nProvider Health: {analytics['provider_health']}")
            
            await hybrid_orchestrator.close()
            
        except Exception as e:
            print(f"❌ Hybrid orchestrator demo failed: {e}")
            use_mock = True
    
    # 2. Integration with DistributedOrchestrator
    print("\n2. 🎯 DistributedOrchestrator with Ollama Turbo")
    print("-" * 45)
    
    try:
        if not use_mock:
            # Real integration
            orchestrator = DistributedOrchestrator(
                orchestrator_model="gpt-oss:20b",  # Use Ollama model as orchestrator
                enable_mesh=False,
                enable_governance=False, 
                enable_context_intelligence=False,
                enable_multimodal=False,
                enable_ollama_turbo=True,
                ollama_turbo_api_key=api_key
            )
        else:
            # Mock integration for demo
            orchestrator = DistributedOrchestrator(
                enable_mesh=False,
                enable_governance=False,
                enable_context_intelligence=False, 
                enable_multimodal=False,
                enable_ollama_turbo=False  # Disabled for demo
            )
        
        if orchestrator.enable_ollama_turbo and orchestrator.ollama_orchestrator:
            # Test orchestrator-integrated Ollama requests
            request = OllamaRequest(
                model="gpt-oss:20b",
                messages=[{
                    "role": "user",
                    "content": "List 3 advantages of distributed AI agent architectures."
                }],
                stream=False
            )
            
            response = await orchestrator.execute_ollama_request(request)
            
            print(f"Orchestrator Ollama Response:")
            print(f"Model: {response['model']}")
            print(f"Source: {response['source']}")
            print(f"Content: {response['content']}")
            
            # Test analytics
            ollama_analytics = await orchestrator.get_ollama_analytics()
            print(f"\nOllama Analytics: {ollama_analytics}")
            
            await orchestrator.close()
        else:
            print("Ollama Turbo integration not enabled in orchestrator")
    
    except Exception as e:
        print(f"❌ DistributedOrchestrator demo failed: {e}")
    
    # 3. Configuration Examples
    print("\n3. ⚙️ Configuration Examples")
    print("-" * 30)
    
    # Show environment configuration
    config = get_ollama_config_from_env()
    print("Environment Configuration:")
    for key, value in config.items():
        if 'api_key' in key and value:
            print(f"  {key}: {'*' * 20}...")
        else:
            print(f"  {key}: {value}")
    
    # Show different orchestrator creation modes
    print("\nFactory Pattern Examples:")
    
    modes_to_demo = [
        (OllamaMode.LOCAL, "Local-only deployment"),
        (OllamaMode.HYBRID, "Hybrid local + Turbo deployment")
    ]
    
    for mode, description in modes_to_demo:
        try:
            if mode == OllamaMode.TURBO and use_mock:
                continue  # Skip Turbo-only in mock mode
                
            orch = create_ollama_orchestrator(
                mode=mode,
                turbo_api_key=api_key if mode != OllamaMode.LOCAL else None,
                prefer_turbo=False
            )
            print(f"  ✅ {mode.value}: {description} - Created successfully")
            
            # Cleanup if needed
            if hasattr(orch, 'close'):
                await orch.close()
                
        except Exception as e:
            print(f"  ❌ {mode.value}: {description} - Failed: {e}")
    
    print("\n4. 💡 Usage Recommendations")
    print("-" * 32)
    
    recommendations = [
        "Use gpt-oss:120b for complex reasoning and creative tasks (Turbo only)",
        "Use gpt-oss:20b for balanced performance (available locally + Turbo)",
        "Enable hybrid mode for automatic failover and cost optimization",
        "Set OLLAMA_PREFER_TURBO=true for cloud-first deployment",
        "Use local mode for privacy-sensitive workloads",
        "Monitor analytics to optimize model routing decisions"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    print(f"\n{'='*50}")
    print("🎉 Demo completed! Check the logs for detailed execution info.")


async def run_performance_comparison():
    """Compare performance between local and Turbo models"""
    
    print("\n🏁 Performance Comparison (Mock Data)")
    print("-" * 40)
    
    # Mock performance data since we can't guarantee real API access
    performance_data = {
        "gpt-oss:20b (local)": {
            "avg_response_time": "1.2s",
            "throughput": "15 requests/min", 
            "cost": "$0.00",
            "availability": "100% (local)"
        },
        "gpt-oss:20b (turbo)": {
            "avg_response_time": "0.8s",
            "throughput": "25 requests/min",
            "cost": "$0.00 (subscription)",
            "availability": "99.9% (cloud)"
        },
        "gpt-oss:120b (turbo)": {
            "avg_response_time": "1.5s", 
            "throughput": "12 requests/min",
            "cost": "$0.00 (subscription)",
            "availability": "99.9% (cloud)"
        }
    }
    
    for model, metrics in performance_data.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    print("\n💡 Key Insights:")
    print("  • Turbo models generally have lower latency")
    print("  • 120b model offers superior quality for complex tasks")
    print("  • Local models provide privacy and offline capability") 
    print("  • Hybrid mode combines benefits of both approaches")


async def main():
    """Main demo function"""
    await demonstrate_ollama_turbo_integration()
    await run_performance_comparison()


if __name__ == "__main__":
    asyncio.run(main())