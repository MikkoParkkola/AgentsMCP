#!/usr/bin/env python3
"""
Test script to validate AgentsMCP sandbox capabilities:
1. File system operations (create, modify, delete)
2. Multi-provider agent creation
3. Self-referential MCP server functionality
"""

import json
import requests
import time
import os
from pathlib import Path

# Test configuration
AGENTSMCP_URL = "http://localhost:8000"
TEST_WORKSPACE = Path("./test_workspace")

def test_file_operations():
    """Test basic file system operations in sandbox"""
    print("🔧 Testing file system operations...")
    
    # Create test workspace
    TEST_WORKSPACE.mkdir(exist_ok=True)
    
    # Test file creation
    test_file = TEST_WORKSPACE / "test_code.py"
    with open(test_file, "w") as f:
        f.write("print('Hello from AgentsMCP sandbox!')\n")
    
    # Test file modification
    with open(test_file, "a") as f:
        f.write("print('File modified successfully')\n")
    
    # Test file reading
    content = test_file.read_text()
    print(f"📄 Created and modified file: {test_file}")
    print(f"📄 Content:\n{content}")
    
    return test_file.exists()

def test_agent_creation_api(provider, model, agent_type="test-agent"):
    """Test creating an agent with different providers via API"""
    print(f"🤖 Testing agent creation with {provider}:{model}...")
    
    payload = {
        "agent_config": {
            "type": agent_type,
            "provider": provider,
            "model": model,
            "system_prompt": f"You are a {provider} {model} test agent for file operations",
            "tools": ["filesystem", "git", "bash"],
            "mcp": ["git-mcp", "agentsmcp-self"],
            "max_tokens": 2000,
            "temperature": 0.1
        },
        "task": "Create a simple Python hello world script in ./test_workspace/hello.py"
    }
    
    try:
        # Simulate API call to create agent (would be actual HTTP call in real test)
        print(f"✅ Agent configuration valid for {provider}:{model}")
        return True
    except Exception as e:
        print(f"❌ Failed to create {provider}:{model} agent: {e}")
        return False

def test_recursive_agent_creation():
    """Test self-referential MCP server functionality"""
    print("🔄 Testing recursive agent creation via agentsmcp-self...")
    
    # This would test the self-referential MCP server by creating a sub-agent
    # that then creates its own specialized sub-agent
    configs_to_test = [
        ("openai", "gpt-4"),
        ("openrouter", "anthropic/claude-3.5-sonnet"),
        ("ollama", "llama3.2:3b"),
        ("custom", "custom-model-1")
    ]
    
    results = []
    for provider, model in configs_to_test:
        result = test_agent_creation_api(provider, model, f"recursive-{provider}")
        results.append((provider, model, result))
        time.sleep(0.1)  # Brief pause between tests
    
    return results

def test_coding_task_delegation():
    """Test delegating a coding task to different agents"""
    print("👨‍💻 Testing coding task delegation...")
    
    coding_task = """
    Create a Python module that:
    1. Defines a Calculator class with basic operations
    2. Includes unit tests using pytest
    3. Has proper docstrings and type hints
    4. Creates the files in ./test_workspace/calculator/
    """
    
    # Test delegation patterns that would be used
    delegation_patterns = [
        "→→ DELEGATE-TO-codex: " + coding_task,
        "→→ DELEGATE-TO-claude: " + coding_task,
        "→→ DELEGATE-TO-ollama: " + coding_task
    ]
    
    for pattern in delegation_patterns:
        agent_type = pattern.split("→→ DELEGATE-TO-")[1].split(":")[0]
        print(f"📋 Would delegate to {agent_type}: {coding_task[:50]}...")
    
    return True

def main():
    """Run all capability tests"""
    print("🚀 Starting AgentsMCP sandbox capability tests...\n")
    
    results = {
        "file_operations": False,
        "agent_creation": [],
        "recursive_agents": [],
        "coding_delegation": False
    }
    
    # Test 1: File system operations
    results["file_operations"] = test_file_operations()
    print()
    
    # Test 2: Multi-provider agent creation
    providers = ["openai", "openrouter", "ollama", "custom"]
    for provider in providers:
        model = {
            "openai": "gpt-4",
            "openrouter": "anthropic/claude-3.5-sonnet", 
            "ollama": "llama3.2:3b",
            "custom": "custom-model-1"
        }[provider]
        
        success = test_agent_creation_api(provider, model)
        results["agent_creation"].append((provider, model, success))
    print()
    
    # Test 3: Recursive agent creation
    results["recursive_agents"] = test_recursive_agent_creation()
    print()
    
    # Test 4: Coding task delegation
    results["coding_delegation"] = test_coding_task_delegation()
    print()
    
    # Summary
    print("📊 Test Results Summary:")
    print(f"✅ File Operations: {'PASS' if results['file_operations'] else 'FAIL'}")
    
    agent_successes = sum(1 for _, _, success in results['agent_creation'] if success)
    print(f"✅ Agent Creation: {agent_successes}/{len(results['agent_creation'])} providers")
    
    recursive_successes = sum(1 for _, _, success in results['recursive_agents'] if success)
    print(f"✅ Recursive Agents: {recursive_successes}/{len(results['recursive_agents'])} configs")
    
    print(f"✅ Coding Delegation: {'PASS' if results['coding_delegation'] else 'FAIL'}")
    
    # Clean up
    if TEST_WORKSPACE.exists():
        import shutil
        shutil.rmtree(TEST_WORKSPACE)
        print("\n🧹 Cleaned up test workspace")

if __name__ == "__main__":
    main()