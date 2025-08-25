#!/usr/bin/env python3
"""
AgentsMCP Orchestrator Model Selection Demo

This script demonstrates how to use different orchestrator models
with the distributed architecture, showing performance and cost implications.
"""

import asyncio
import logging
from typing import Dict, Any

from agentsmcp.distributed.orchestrator import DistributedOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_model_comparison():
    """Compare different orchestrator models in action."""
    
    print("🤖 AgentsMCP Orchestrator Model Demo")
    print("=" * 50)
    
    # Show available models
    print("\n📋 Available Models:")
    models = DistributedOrchestrator.get_available_models()
    
    for name, config in models.items():
        is_default = " (DEFAULT)" if name == "gpt-5" else ""
        cost_str = "FREE" if config['cost_per_input'] == 0 else f"${config['cost_per_input'] * 1_000_000:.2f}/M"
        print(f"  🔹 {name.upper()}{is_default}")
        print(f"     Performance: {config['performance_score']}% | Cost: {cost_str} | Context: {config['context_limit']:,}")
    
    # Test different models
    test_models = ["gpt-5", "claude-4.1-sonnet", "qwen3-235b-a22b"]
    
    print(f"\n🧪 Testing Models: {', '.join(test_models)}")
    print("-" * 50)
    
    for model in test_models:
        print(f"\n🎯 Testing {model.upper()}:")
        
        try:
            # Initialize orchestrator with specific model
            orchestrator = DistributedOrchestrator(
                max_workers=5,
                context_budget_tokens=32000,  # Small for demo
                cost_budget=10.0,
                orchestrator_model=model
            )
            
            # Start orchestrator
            await orchestrator.start()
            
            # Get status to show model configuration
            status = await orchestrator.get_status()
            
            print(f"  ✅ Initialized successfully")
            print(f"  📊 Performance Score: {status['orchestrator_model']['performance_score']}%")
            print(f"  🧠 Context Limit: {status['orchestrator_model']['context_limit']:,} tokens")
            print(f"  💰 Cost: {status['orchestrator_model']['cost_per_input']} / {status['orchestrator_model']['cost_per_output']}")
            
            # Estimate cost for typical orchestration task
            estimated_cost = orchestrator.estimate_orchestration_cost(50000, 5000)  # 50K input, 5K output
            print(f"  💵 Estimated cost per task: ${estimated_cost:.4f}")
            
        except Exception as e:
            print(f"  ❌ Failed to initialize: {e}")


async def demo_use_case_recommendations():
    """Show model recommendations for different use cases."""
    
    print("\n\n🎯 Model Recommendations by Use Case:")
    print("=" * 50)
    
    use_cases = [
        "default",
        "cost_effective", 
        "premium",
        "balanced",
        "massive_context",
        "local",
        "privacy"
    ]
    
    for use_case in use_cases:
        recommended = DistributedOrchestrator.get_model_recommendation(use_case)
        models = DistributedOrchestrator.get_available_models()
        config = models[recommended]
        
        print(f"\n🔹 {use_case.upper().replace('_', ' ')}:")
        print(f"   Recommended: {recommended}")
        print(f"   Performance: {config['performance_score']}%")
        print(f"   Best for: {config['recommended_for']}")


async def demo_cost_estimation():
    """Demonstrate cost estimation across different models."""
    
    print("\n\n💰 Cost Estimation Demo:")
    print("=" * 50)
    
    # Typical orchestration workloads
    workloads = [
        ("Small Task", 10000, 1000),    # 10K input, 1K output
        ("Medium Task", 50000, 5000),   # 50K input, 5K output  
        ("Large Task", 200000, 20000),  # 200K input, 20K output
        ("Massive Task", 800000, 50000) # 800K input, 50K output (only for high-context models)
    ]
    
    models_to_test = ["gpt-5", "claude-4.1-opus", "claude-4.1-sonnet", "gemini-2.5-pro", "qwen3-235b-a22b"]
    
    print(f"{'Workload':<15} {'GPT-5':<10} {'Claude Opus':<12} {'Claude Sonnet':<14} {'Gemini 2.5':<12} {'Qwen3 (Local)':<15}")
    print("-" * 90)
    
    for workload_name, input_tokens, output_tokens in workloads:
        costs = []
        
        for model in models_to_test:
            try:
                orchestrator = DistributedOrchestrator(orchestrator_model=model)
                
                # Check if workload exceeds model context limit
                if input_tokens > orchestrator.model_config["context_limit"]:
                    costs.append("N/A")
                else:
                    cost = orchestrator.estimate_orchestration_cost(input_tokens, output_tokens)
                    if cost == 0:
                        costs.append("FREE")
                    else:
                        costs.append(f"${cost:.3f}")
                        
            except Exception:
                costs.append("ERROR")
        
        print(f"{workload_name:<15} {costs[0]:<10} {costs[1]:<12} {costs[2]:<14} {costs[3]:<12} {costs[4]:<15}")


async def main():
    """Run all demos."""
    
    try:
        await demo_model_comparison()
        await demo_use_case_recommendations() 
        await demo_cost_estimation()
        
        print("\n\n✅ Demo completed successfully!")
        print("\n💡 Usage Examples:")
        print("  # Use default GPT-5 orchestrator")
        print("  agentsmcp interactive")
        print("")
        print("  # Use premium Claude 4.1 Opus")
        print("  agentsmcp interactive --orchestrator-model claude-4.1-opus")
        print("")
        print("  # Use local Qwen3 for privacy")
        print("  agentsmcp dashboard --orchestrator-model qwen3-235b-a22b")
        print("")
        print("  # Get model recommendations")
        print("  agentsmcp models --recommend premium")
        print("  agentsmcp models --detailed")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())