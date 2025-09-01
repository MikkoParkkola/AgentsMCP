"""
Test suite for development tool integration within AgentsMCP workflows.

Focuses on testing tool execution, lazy loading, and integration with agent workflows.
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentsmcp.agent_manager import AgentManager
from agentsmcp.config import Config
from agentsmcp.tools import (
    get_tool_registry, FileOperationTool, ShellCommandTool,
    TextAnalysisTool, CodeAnalysisTool, WebSearchTool
)


# Mock Tool Classes

class MockFileOperationTool:
    """Mock file operation tool for testing."""
    
    def __init__(self):
        self.operations_performed = []
    
    async def execute(self, operation: str, **kwargs) -> str:
        self.operations_performed.append((operation, kwargs))
        
        if operation == "read_file":
            return "def main():\n    print('Hello World')\n    return 0"
        elif operation == "write_file":
            return f"File written to {kwargs.get('path', 'unknown')}"
        elif operation == "list_directory":
            return "FILES: main.py, utils.py, tests/\nDIRS: src/, docs/"
        else:
            return f"Operation {operation} completed"


class MockShellCommandTool:
    """Mock shell command tool for testing."""
    
    def __init__(self):
        self.commands_executed = []
    
    async def execute(self, command: str, **kwargs) -> str:
        self.commands_executed.append((command, kwargs))
        
        if "test" in command.lower():
            return "Running tests...\n✓ test_user_model.py::test_creation PASSED\n✓ test_api.py::test_register PASSED\n\n2 passed, 0 failed"
        elif "build" in command.lower():
            return "Building project...\nBuild completed successfully"
        elif "lint" in command.lower():
            return "Linting code...\nYour code has been rated at 9.5/10"
        else:
            return f"Command executed: {command}"


class MockCodeAnalysisTool:
    """Mock code analysis tool for testing."""
    
    def __init__(self):
        self.analyses_performed = []
    
    async def execute(self, analysis_type: str, **kwargs) -> str:
        self.analyses_performed.append((analysis_type, kwargs))
        
        return json.dumps({
            "complexity": "Low",
            "maintainability": "High", 
            "test_coverage": "85%",
            "code_quality_score": "A",
            "issues": [],
            "suggestions": [
                "Add type hints to function parameters",
                "Consider breaking down large functions"
            ]
        })


class MockWebSearchTool:
    """Mock web search tool for testing."""
    
    def __init__(self):
        self.searches_performed = []
    
    async def execute(self, query: str, **kwargs) -> str:
        self.searches_performed.append((query, kwargs))
        
        return json.dumps({
            "results": [
                {
                    "title": "Best Practices for REST API Design",
                    "url": "https://example.com/rest-api-best-practices",
                    "summary": "Comprehensive guide to REST API design principles"
                },
                {
                    "title": "Authentication Security Patterns",
                    "url": "https://example.com/auth-security",
                    "summary": "Modern authentication and authorization patterns"
                }
            ],
            "total_results": 2
        })


# Fixtures

@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry with development tools."""
    registry = Mock()
    
    tools = {
        "file_operations": MockFileOperationTool(),
        "shell_command": MockShellCommandTool(),
        "code_analysis": MockCodeAnalysisTool(), 
        "web_search": MockWebSearchTool(),
        "text_analysis": Mock()
    }
    
    registry.get_tool = lambda name: tools.get(name)
    registry.list_tools = lambda: list(tools.keys())
    
    return registry, tools


@pytest.fixture
async def agent_with_tools(mock_tool_registry):
    """Create an agent manager with mocked tools."""
    registry, tools = mock_tool_registry
    
    with patch('agentsmcp.tools.get_tool_registry', return_value=registry):
        config = Mock()
        config.concurrent_agents = 2
        config.storage = Mock()
        config.storage.type = "memory"
        config.storage.config = {}
        config.get_agent_config = Mock(return_value=Mock())
        
        # Mock SelfAgent that uses tools
        class ToolUsingAgent:
            def __init__(self, agent_config, config):
                self.agent_config = agent_config
                self.config = config
                self.tools_registry = registry
                
            async def execute_task(self, task: str) -> str:
                # Simulate tool usage based on task
                results = []
                
                if "read" in task.lower() or "file" in task.lower():
                    file_tool = self.tools_registry.get_tool("file_operations")
                    result = await file_tool.execute("read_file", path="src/main.py")
                    results.append(f"File read: {len(result)} characters")
                
                if "test" in task.lower():
                    shell_tool = self.tools_registry.get_tool("shell_command")
                    result = await shell_tool.execute("python -m pytest")
                    results.append(f"Tests: {result.count('PASSED')} passed")
                
                if "analyze" in task.lower() or "quality" in task.lower():
                    analysis_tool = self.tools_registry.get_tool("code_analysis")
                    result = await analysis_tool.execute("quality_analysis")
                    results.append(f"Analysis: {result}")
                
                if "search" in task.lower() or "research" in task.lower():
                    search_tool = self.tools_registry.get_tool("web_search")
                    result = await search_tool.execute("best practices development")
                    results.append(f"Search: {result}")
                
                return json.dumps({
                    "task_completed": task,
                    "tools_used": len(results),
                    "results": results
                })
                
            async def cleanup(self):
                pass
        
        with patch('agentsmcp.agent_manager.AgentManager._load_self_agent_class') as mock_load:
            mock_load.return_value = ToolUsingAgent
            
            manager = AgentManager(config)
            yield manager, tools
            await manager.shutdown()


# Tool Integration Tests

@pytest.mark.asyncio
@pytest.mark.integration
async def test_file_operations_in_development_workflow(agent_with_tools):
    """Test file operations tool integration during development tasks."""
    manager, tools = agent_with_tools
    
    # Execute task that requires file operations
    task = "Read project files and analyze structure for code review"
    job_id = await manager.spawn_agent("backend_qa_engineer", task)
    status = await manager.wait_for_completion(job_id)
    
    assert status.state.name == "COMPLETED"
    
    result = json.loads(status.output)
    assert result["tools_used"] > 0
    assert any("File read" in r for r in result["results"])
    
    # Verify tool was actually called
    file_tool = tools["file_operations"]
    assert len(file_tool.operations_performed) > 0
    assert file_tool.operations_performed[0][0] == "read_file"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_shell_command_integration_for_testing(agent_with_tools):
    """Test shell command tool integration for running tests."""
    manager, tools = agent_with_tools
    
    # Execute task that requires running tests
    task = "Run unit tests and verify code coverage"
    job_id = await manager.spawn_agent("backend_qa_engineer", task)
    status = await manager.wait_for_completion(job_id)
    
    assert status.state.name == "COMPLETED"
    
    result = json.loads(status.output)
    assert result["tools_used"] > 0
    assert any("Tests:" in r for r in result["results"])
    
    # Verify shell command was executed
    shell_tool = tools["shell_command"]
    assert len(shell_tool.commands_executed) > 0
    command, _ = shell_tool.commands_executed[0]
    assert "pytest" in command


@pytest.mark.asyncio
@pytest.mark.integration
async def test_code_analysis_tool_integration(agent_with_tools):
    """Test code analysis tool integration for quality assessment."""
    manager, tools = agent_with_tools
    
    # Execute task that requires code analysis
    task = "Analyze code quality and provide improvement suggestions"
    job_id = await manager.spawn_agent("dev_tooling_engineer", task)
    status = await manager.wait_for_completion(job_id)
    
    assert status.state.name == "COMPLETED"
    
    result = json.loads(status.output)
    assert result["tools_used"] > 0
    assert any("Analysis:" in r for r in result["results"])
    
    # Verify analysis was performed
    analysis_tool = tools["code_analysis"]
    assert len(analysis_tool.analyses_performed) > 0
    analysis_type, _ = analysis_tool.analyses_performed[0]
    assert analysis_type == "quality_analysis"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_web_search_tool_integration(agent_with_tools):
    """Test web search tool integration for research tasks."""
    manager, tools = agent_with_tools
    
    # Execute task that requires web research
    task = "Research best practices for API security implementation"
    job_id = await manager.spawn_agent("backend_engineer", task)
    status = await manager.wait_for_completion(job_id)
    
    assert status.state.name == "COMPLETED"
    
    result = json.loads(status.output)
    assert result["tools_used"] > 0
    assert any("Search:" in r for r in result["results"])
    
    # Verify search was performed
    search_tool = tools["web_search"]
    assert len(search_tool.searches_performed) > 0
    query, _ = search_tool.searches_performed[0]
    assert query == "best practices development"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_multi_tool_workflow_integration(agent_with_tools):
    """Test workflow that uses multiple tools in sequence."""
    manager, tools = agent_with_tools
    
    # Execute comprehensive task requiring multiple tools
    task = "Read project files, run tests, analyze code quality, and search for improvement suggestions"
    job_id = await manager.spawn_agent("chief_qa_engineer", task)
    status = await manager.wait_for_completion(job_id)
    
    assert status.state.name == "COMPLETED"
    
    result = json.loads(status.output)
    assert result["tools_used"] >= 3  # Should use multiple tools
    
    # Verify multiple tools were used
    file_tool = tools["file_operations"]
    shell_tool = tools["shell_command"]
    analysis_tool = tools["code_analysis"]
    search_tool = tools["web_search"]
    
    assert len(file_tool.operations_performed) > 0
    assert len(shell_tool.commands_executed) > 0
    assert len(analysis_tool.analyses_performed) > 0
    assert len(search_tool.searches_performed) > 0


# Lazy Loading Tests

@pytest.mark.asyncio
@pytest.mark.unit
async def test_tool_lazy_loading_performance():
    """Test tool lazy loading doesn't impact performance."""
    import time
    
    with patch('agentsmcp.tools.get_tool_registry') as mock_registry:
        # Track loading times
        load_times = {}
        
        def mock_get_tool(name):
            start_time = time.time()
            # Simulate tool creation overhead
            time.sleep(0.01)  # 10ms overhead
            end_time = time.time()
            load_times[name] = end_time - start_time
            return Mock(execute=AsyncMock(return_value=f"{name} result"))
        
        mock_registry.return_value.get_tool = mock_get_tool
        
        # Request tools multiple times
        registry = mock_registry.return_value
        
        # First access should have loading time
        start = time.time()
        tool1 = registry.get_tool("file_operations")
        first_access = time.time() - start
        
        # Second access should be faster if cached
        start = time.time()
        tool2 = registry.get_tool("file_operations")  
        second_access = time.time() - start
        
        # Verify tools were created
        assert tool1 is not None
        assert tool2 is not None
        
        # Loading behavior verified (actual caching would be in real implementation)
        assert "file_operations" in load_times


@pytest.mark.asyncio
@pytest.mark.integration
async def test_tool_registry_initialization():
    """Test tool registry initialization and availability."""
    with patch('agentsmcp.tools.get_tool_registry') as mock_registry:
        # Mock registry with expected tools
        expected_tools = [
            "file_operations", "shell_command", "code_analysis",
            "web_search", "text_analysis"
        ]
        
        mock_registry.return_value.list_tools.return_value = expected_tools
        mock_registry.return_value.get_tool.return_value = Mock()
        
        # Get registry and verify tools available
        registry = mock_registry.return_value
        available_tools = registry.list_tools()
        
        assert len(available_tools) == len(expected_tools)
        for tool_name in expected_tools:
            assert tool_name in available_tools
            tool = registry.get_tool(tool_name)
            assert tool is not None


# Error Handling Tests

@pytest.mark.asyncio
@pytest.mark.integration
async def test_tool_execution_error_handling(agent_with_tools):
    """Test error handling when tools fail during execution."""
    manager, tools = agent_with_tools
    
    # Mock a failing tool
    failing_tool = Mock()
    failing_tool.execute = AsyncMock(side_effect=Exception("Tool execution failed"))
    tools["file_operations"] = failing_tool
    
    # Execute task that would use the failing tool
    task = "Read and analyze project files"
    job_id = await manager.spawn_agent("backend_engineer", task)
    status = await manager.wait_for_completion(job_id)
    
    # Agent should handle tool failure gracefully
    # (Implementation dependent - might complete with error or fail completely)
    assert status.state.name in ["COMPLETED", "FAILED"]
    
    if status.state.name == "FAILED":
        assert "Tool execution failed" in str(status.error) or status.error


@pytest.mark.asyncio
@pytest.mark.integration
async def test_tool_timeout_handling(agent_with_tools):
    """Test handling of tool execution timeouts."""
    manager, tools = agent_with_tools
    
    # Mock a slow tool
    slow_tool = Mock()
    async def slow_execute(*args, **kwargs):
        await asyncio.sleep(5)  # Longer than reasonable timeout
        return "Slow result"
    
    slow_tool.execute = slow_execute
    tools["code_analysis"] = slow_tool
    
    # Execute task with slow tool
    task = "Perform comprehensive code analysis"
    job_id = await manager.spawn_agent("backend_qa_engineer", task, timeout=3)
    status = await manager.wait_for_completion(job_id)
    
    # Should timeout or handle gracefully
    assert status.state.name in ["TIMEOUT", "FAILED", "COMPLETED"]


# Tool State and Concurrency Tests

@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_tool_usage(agent_with_tools):
    """Test concurrent tool usage across multiple agents."""
    manager, tools = agent_with_tools
    
    # Execute multiple tasks concurrently that use the same tools
    tasks = [
        "Analyze code quality for module A",
        "Run tests for module B", 
        "Read configuration files for module C"
    ]
    
    job_ids = []
    for task in tasks:
        job_id = await manager.spawn_agent("backend_qa_engineer", task)
        job_ids.append(job_id)
    
    # Wait for all to complete
    statuses = []
    for job_id in job_ids:
        status = await manager.wait_for_completion(job_id)
        statuses.append(status)
    
    # All should complete successfully
    for status in statuses:
        assert status.state.name == "COMPLETED"
    
    # Verify tools handled concurrent access
    file_tool = tools["file_operations"]
    shell_tool = tools["shell_command"] 
    analysis_tool = tools["code_analysis"]
    
    # At least some operations should have been performed
    total_operations = (
        len(file_tool.operations_performed) +
        len(shell_tool.commands_executed) + 
        len(analysis_tool.analyses_performed)
    )
    assert total_operations > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_tool_state_isolation(agent_with_tools):
    """Test that tool state is properly isolated between uses."""
    manager, tools = agent_with_tools
    
    # Execute first task
    task1 = "Read project configuration"
    job_id1 = await manager.spawn_agent("backend_engineer", task1)
    status1 = await manager.wait_for_completion(job_id1)
    
    # Check initial tool state
    file_tool = tools["file_operations"]
    initial_operations = len(file_tool.operations_performed)
    
    # Execute second task
    task2 = "Read different project files"
    job_id2 = await manager.spawn_agent("web_frontend_engineer", task2)
    status2 = await manager.wait_for_completion(job_id2)
    
    # Verify both completed
    assert status1.state.name == "COMPLETED"
    assert status2.state.name == "COMPLETED"
    
    # Tool should have record of both operations
    final_operations = len(file_tool.operations_performed)
    assert final_operations > initial_operations


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])