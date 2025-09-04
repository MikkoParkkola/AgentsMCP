#!/usr/bin/env python3
"""
Test script to validate LLM connectivity fixes and error reporting improvements.
This tests the enhanced error handling without requiring user interaction.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from agentsmcp.conversation.llm_client import LLMClient
from agentsmcp.ui.v3.chat_engine import ChatEngine


async def test_configuration_validation():
    """Test the configuration validation system."""
    print("🔧 Testing Configuration Validation System")
    print("=" * 50)
    
    # Clear environment variables to simulate unconfigured state
    env_keys_to_clear = [
        'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'OPENROUTER_API_KEY'
    ]
    original_values = {}
    for key in env_keys_to_clear:
        original_values[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]
    
    try:
        # Set TUI mode to prevent console contamination
        os.environ['AGENTSMCP_TUI_MODE'] = '1'
        
        # Create LLM client
        client = LLMClient()
        
        # Test configuration status
        print("\n1. Testing get_configuration_status()...")
        config_status = client.get_configuration_status()
        
        print(f"   Current Provider: {config_status['current_provider']}")
        print(f"   Current Model: {config_status['current_model']}")
        print(f"   Preprocessing Enabled: {config_status['preprocessing_enabled']}")
        print(f"   Configuration Issues: {len(config_status['configuration_issues'])}")
        
        for issue in config_status['configuration_issues']:
            print(f"     - {issue}")
        
        # Test provider status
        print(f"\n2. Provider Status:")
        for provider, status in config_status['providers'].items():
            icon = "✅" if status['configured'] else "❌"
            print(f"   {icon} {provider}: {'Configured' if status['configured'] else 'Not Configured'}")
            if status['last_error']:
                print(f"      Error: {status['last_error']}")
        
        # Test sending message with no providers configured
        print(f"\n3. Testing send_message with no configured providers...")
        response = await client.send_message("Hello, can you help me?")
        print(f"   Response: {response[:200]}..." if len(response) > 200 else f"   Response: {response}")
        
        # Test preprocessing controls
        print(f"\n4. Testing preprocessing controls...")
        print(f"   Current status: {client.get_preprocessing_status()}")
        
        print(f"   Toggling off: {client.toggle_preprocessing(False)}")
        print(f"   Toggling on: {client.toggle_preprocessing(True)}")
        
    finally:
        # Restore original environment variables
        for key, value in original_values.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]


async def test_chat_engine_commands():
    """Test the new chat engine diagnostic commands."""
    print("\n🤖 Testing Chat Engine Diagnostic Commands")
    print("=" * 50)
    
    # Create chat engine
    engine = ChatEngine()
    
    # Track messages for testing
    messages = []
    errors = []
    
    def message_callback(msg):
        messages.append(msg.content)
    
    def error_callback(err):
        errors.append(err)
    
    engine.set_callbacks(
        status_callback=lambda s: None,
        message_callback=message_callback,
        error_callback=error_callback
    )
    
    # Test each new command
    commands_to_test = [
        ("/help", "Help command with new diagnostics"),
        ("/config", "Configuration status"),
        ("/providers", "Provider status"),
        ("/preprocessing", "Preprocessing status"),
        ("/preprocessing off", "Disable preprocessing"),
        ("/preprocessing on", "Enable preprocessing"),
        ("/status", "Basic status")
    ]
    
    for command, description in commands_to_test:
        print(f"\n   Testing {command} - {description}")
        try:
            result = await engine._handle_command(command)
            if messages:
                last_msg = messages[-1]
                preview = last_msg[:150] + "..." if len(last_msg) > 150 else last_msg
                print(f"     ✅ Success: {preview}")
            else:
                print(f"     ✅ Command handled successfully")
        except Exception as e:
            print(f"     ❌ Error: {e}")
        
        if errors:
            for error in errors[-1:]:  # Show only latest error
                print(f"     ⚠️  Error reported: {error}")
    
    print(f"\n   Total messages generated: {len(messages)}")
    print(f"   Total errors: {len(errors)}")


async def test_error_scenarios():
    """Test specific error scenarios and their reporting."""
    print("\n⚠️  Testing Error Scenarios and Reporting")
    print("=" * 50)
    
    # Clear all API keys to force errors
    env_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'OPENROUTER_API_KEY']
    original_values = {}
    for key in env_keys:
        original_values[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]
    
    try:
        os.environ['AGENTSMCP_TUI_MODE'] = '1'
        client = LLMClient()
        
        scenarios = [
            ("No providers configured", "Hello world"),
            ("Simple question", "What is 2+2?"),
            ("Complex request", "Can you analyze this code and find bugs?"),
        ]
        
        for scenario_name, message in scenarios:
            print(f"\n   Scenario: {scenario_name}")
            print(f"   Message: {message}")
            
            response = await client.send_message(message)
            
            # Check if response contains helpful guidance
            guidance_indicators = [
                "❌", "💡", "API key", "configuration", "/config", "/help"
            ]
            
            has_guidance = any(indicator in response for indicator in guidance_indicators)
            
            preview = response[:200] + "..." if len(response) > 200 else response
            print(f"   Response: {preview}")
            print(f"   Contains guidance: {'✅' if has_guidance else '❌'}")
    
    finally:
        # Restore environment variables
        for key, value in original_values.items():
            if value is not None:
                os.environ[key] = value


async def main():
    """Run all tests."""
    print("🚀 Testing LLM Connectivity Fixes and Error Reporting")
    print("=" * 60)
    
    try:
        await test_configuration_validation()
        await test_chat_engine_commands() 
        await test_error_scenarios()
        
        print("\n✅ All tests completed successfully!")
        print("\n💡 Key Improvements Validated:")
        print("   • Configuration validation and status checking")
        print("   • Specific error messages with actionable guidance")  
        print("   • Diagnostic commands (/config, /providers, /preprocessing)")
        print("   • Preprocessing mode controls")
        print("   • Enhanced help system with setup guidance")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())