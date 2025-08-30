#!/usr/bin/env python3
"""Example demonstrating NLP integration with the AgentsMCP CLI v3 architecture."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentsmcp.cli.v3.nlp.processor import NaturalLanguageProcessor
from agentsmcp.cli.v3.models.nlp_models import (
    LLMConfig,
    ConversationContext,
    ParsingMethod
)


async def demonstrate_nlp_integration():
    """Demonstrate NLP processor integration with various natural language inputs."""
    
    print("🧠 AgentsMCP CLI v3 - Natural Language Processing Integration Demo")
    print("=" * 70)
    
    # Initialize NLP processor with local configuration
    config = LLMConfig(
        model_name="gpt-oss:20b",  # Local Ollama model
        max_tokens=1024,
        temperature=0.1,
        timeout_seconds=30.0,
        context_window=8000
    )
    
    processor = NaturalLanguageProcessor(config)
    
    # Check LLM availability
    print("\n📡 Checking LLM availability...")
    llm_available = await processor.check_llm_availability()
    print(f"   Local LLM (Ollama): {'✅ Available' if llm_available else '❌ Unavailable'}")
    
    if not llm_available:
        print("   💡 Note: Will fallback to rule-based pattern matching")
    
    # Initialize conversation context
    context = ConversationContext(
        current_directory="/Users/developer/projects/agentsmcp",
        recent_files=["main.py", "config.json", "README.md"]
    )
    
    # Test cases covering various command types
    test_cases = [
        {
            "category": "Help & Information",
            "inputs": [
                "help",
                "what can you do",
                "show me available commands",
                "how do I use this tool"
            ]
        },
        {
            "category": "Code Analysis",
            "inputs": [
                "analyze my code",
                "check the code for issues", 
                "review my code for security problems",
                "audit the codebase"
            ]
        },
        {
            "category": "System Operations",
            "inputs": [
                "check status",
                "what's the system status",
                "show me the dashboard",
                "start the TUI"
            ]
        },
        {
            "category": "Project Setup",
            "inputs": [
                "help me set up the project",
                "initialize a new project",
                "get started with setup"
            ]
        },
        {
            "category": "File Operations",
            "inputs": [
                "read the config file",
                "list files in the current directory",
                "edit the main file"
            ]
        },
        {
            "category": "Cost Optimization",
            "inputs": [
                "optimize my costs",
                "reduce expenses",
                "save money on operations"
            ]
        }
    ]
    
    # Process each test case
    for test_category in test_cases:
        print(f"\n🎯 {test_category['category']}")
        print("-" * 50)
        
        for natural_input in test_category['inputs']:
            print(f"\n💬 Input: \"{natural_input}\"")
            
            try:
                # Parse natural language input
                start_time = datetime.now()
                result = await processor.parse(natural_input, context)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                if result.success:
                    cmd = result.structured_command
                    method_emoji = {
                        ParsingMethod.LLM: "🤖",
                        ParsingMethod.RULE_BASED: "📋", 
                        ParsingMethod.HYBRID: "🔄"
                    }.get(cmd.method, "❓")
                    
                    print(f"   ✅ {method_emoji} Parsed successfully ({processing_time:.1f}ms)")
                    print(f"   📝 Action: {cmd.action}")
                    print(f"   ⚙️  Parameters: {cmd.parameters}")
                    print(f"   🎯 Confidence: {cmd.confidence:.2f}")
                    print(f"   💭 Explanation: {result.explanation}")
                    
                    # Show alternative interpretations if available
                    if result.alternative_interpretations:
                        print(f"   🔀 Alternatives: {len(result.alternative_interpretations)} other interpretations")
                        for i, alt in enumerate(result.alternative_interpretations[:2], 1):
                            print(f"      {i}. {alt.command.action} (confidence: {alt.confidence:.2f})")
                    
                    # Add to conversation context
                    context.add_command(natural_input)
                    
                else:
                    print(f"   ❌ Parsing failed ({processing_time:.1f}ms)")
                    if result.errors:
                        for error in result.errors[:2]:  # Show first 2 errors
                            print(f"      🚨 {error.error_type}: {error.message}")
                    
            except Exception as e:
                print(f"   💥 Unexpected error: {e}")
    
    # Show final metrics
    print(f"\n📊 Processing Metrics")
    print("-" * 50)
    metrics = processor.get_metrics()
    print(f"   Total requests: {metrics.total_requests}")
    print(f"   Successful parses: {metrics.successful_parses}")
    print(f"   Failed parses: {metrics.failed_parses}")
    print(f"   Success rate: {metrics.success_rate:.1f}%")
    print(f"   LLM calls: {metrics.llm_calls}")
    print(f"   Rule-based matches: {metrics.rule_based_matches}")
    print(f"   Average confidence: {metrics.average_confidence:.2f}")
    print(f"   Average processing time: {metrics.average_processing_time_ms:.1f}ms")
    
    # Show conversation context state
    print(f"\n🗨️  Conversation Context")
    print("-" * 50)
    print(f"   Session ID: {context.session_id[:8]}...")
    print(f"   Commands in history: {len(context.command_history)}")
    print(f"   Recent files tracked: {len(context.recent_files)}")
    print(f"   Current directory: {context.current_directory}")
    
    if context.command_history:
        print("   Recent commands:")
        for cmd in context.command_history[-3:]:
            print(f"     • {cmd}")


async def demonstrate_error_handling():
    """Demonstrate error handling scenarios."""
    
    print("\n🛡️  Error Handling Demonstration")
    print("=" * 70)
    
    processor = NaturalLanguageProcessor()
    
    error_scenarios = [
        ("", "Empty input"),
        ("xyz random gibberish 123", "Completely unclear input"),
        ("analyze " + "x" * 1000, "Very long input (context size)")
    ]
    
    for input_text, description in error_scenarios:
        print(f"\n🧪 Testing: {description}")
        print(f"   Input: \"{input_text[:50]}{'...' if len(input_text) > 50 else ''}\"")
        
        try:
            result = await processor.parse(input_text)
            
            if result.success:
                print("   ✅ Unexpectedly succeeded")
            else:
                print("   ❌ Failed as expected")
                for error in result.errors[:1]:
                    print(f"      🚨 {error.error_type}: {error.message}")
                    if hasattr(error, 'recovery_suggestions') and error.recovery_suggestions:
                        print(f"      💡 Suggestion: {error.recovery_suggestions[0]}")
        
        except Exception as e:
            print(f"   💥 Exception: {e}")


async def demonstrate_real_world_usage():
    """Demonstrate realistic usage patterns."""
    
    print("\n🌍 Real-World Usage Patterns")
    print("=" * 70)
    
    processor = NaturalLanguageProcessor()
    context = ConversationContext()
    
    # Simulate a realistic conversation flow
    conversation_flow = [
        ("help", "User starts by asking for help"),
        ("what's my current status", "User wants to check status"),
        ("analyze the code in my project", "User wants code analysis"),
        ("show me any issues", "Follow-up question with context"),
        ("start the interactive mode", "User switches to TUI"),
        ("optimize my API costs", "User wants cost optimization")
    ]
    
    print("\n💬 Simulating conversation flow:")
    
    for i, (input_text, description) in enumerate(conversation_flow, 1):
        print(f"\n{i}. {description}")
        print(f"   User: \"{input_text}\"")
        
        result = await processor.parse(input_text, context)
        
        if result.success:
            cmd = result.structured_command
            print(f"   System: Understood as '{cmd.action}' command")
            if cmd.parameters:
                print(f"   Parameters: {cmd.parameters}")
            
            # Add to context for next iteration
            context.add_command(input_text)
        else:
            print("   System: Sorry, I didn't understand that")


async def main():
    """Main demonstration function."""
    try:
        await demonstrate_nlp_integration()
        await demonstrate_error_handling()
        await demonstrate_real_world_usage()
        
        print("\n🎉 NLP Integration Demo Complete!")
        print("   The natural language processor is ready for integration")
        print("   into the AgentsMCP CLI v3 command pipeline.")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())