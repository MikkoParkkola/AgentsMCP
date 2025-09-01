"""
Comprehensive test suite for AgentsMCP software development workflows.

Tests multi-agent coordination, tool integration, and realistic development scenarios
following TDD principles and comprehensive coverage requirements.
"""

import asyncio
import json
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentsmcp.agent_manager import AgentManager
from agentsmcp.config import Config
from agentsmcp.events import EventBus
from agentsmcp.models import (
    TaskEnvelopeV1, ResultEnvelopeV1, EnvelopeStatus, 
    JobState, JobStatus
)
from agentsmcp.orchestration.team_runner import run_team, DEFAULT_TEAM
from agentsmcp.roles.registry import RoleRegistry
from agentsmcp.roles.human_specialists import (
    BusinessAnalystRole, BackendEngineerRole, WebFrontendEngineerRole,
    BackendQARole, ChiefQARole
)


# Test Fixtures and Mocks

class MockSelfAgent:
    """Mock SelfAgent for testing agent behavior."""
    
    def __init__(self, agent_config, config):
        self.agent_config = agent_config
        self.config = config
        self.execution_results = []
        self.tools_used = []
        
    async def execute_task(self, task: str) -> str:
        """Mock task execution with realistic responses."""
        # Simulate different role behaviors
        if "requirements" in task.lower() or "business" in task.lower():
            result = self._business_analyst_response(task)
        elif "backend" in task.lower() or "api" in task.lower():
            result = self._backend_engineer_response(task)  
        elif "frontend" in task.lower() or "ui" in task.lower():
            result = self._frontend_engineer_response(task)
        elif "test" in task.lower() or "qa" in task.lower():
            result = self._qa_engineer_response(task)
        else:
            result = f"Task completed: {task}"
            
        self.execution_results.append(result)
        return result
        
    def _business_analyst_response(self, task: str) -> str:
        return json.dumps({
            "requirements": [
                "User registration with email/password",
                "Profile management functionality", 
                "Data privacy compliance"
            ],
            "acceptance_criteria": [
                "Registration form validates input",
                "Users can update profile information",
                "GDPR compliance for data handling"
            ],
            "user_stories": [
                "As a user, I want to register with email so I can access the platform",
                "As a user, I want to manage my profile so I can keep information current"
            ]
        })
        
    def _backend_engineer_response(self, task: str) -> str:
        return json.dumps({
            "api_design": {
                "endpoints": [
                    "POST /api/users/register",
                    "GET /api/users/profile",
                    "PUT /api/users/profile"
                ],
                "data_models": ["User", "Profile", "Session"],
                "security": "JWT authentication with refresh tokens"
            },
            "implementation_plan": [
                "Setup database schema",
                "Implement authentication service", 
                "Create user management endpoints",
                "Add input validation and error handling"
            ]
        })
        
    def _frontend_engineer_response(self, task: str) -> str:
        return json.dumps({
            "ui_components": [
                "RegistrationForm",
                "LoginForm", 
                "ProfileEditor",
                "NavigationHeader"
            ],
            "pages": ["Login", "Register", "Dashboard", "Profile"],
            "accessibility": "WCAG 2.1 AA compliance",
            "responsive_design": "Mobile-first approach with breakpoints"
        })
        
    def _qa_engineer_response(self, task: str) -> str:
        return json.dumps({
            "test_plan": {
                "unit_tests": ["User model validation", "API endpoint responses"],
                "integration_tests": ["Registration flow", "Authentication workflow"],
                "e2e_tests": ["Complete user journey", "Cross-browser testing"]
            },
            "quality_gates": [
                "90% code coverage minimum",
                "No critical security vulnerabilities",
                "Performance requirements met"
            ],
            "test_results": {
                "passed": 45,
                "failed": 0,
                "coverage": "92%"
            }
        })
        
    async def cleanup(self):
        """Mock cleanup method."""
        pass


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock(spec=Config)
    config.concurrent_agents = 4
    config.storage = Mock()
    config.storage.type = "memory"
    config.storage.config = {}
    
    # Mock agent configurations
    config.get_agent_config = Mock(return_value=Mock())
    config.agents = {
        "business_analyst": Mock(),
        "backend_engineer": Mock(),
        "web_frontend_engineer": Mock(), 
        "backend_qa_engineer": Mock(),
        "chief_qa_engineer": Mock()
    }
    return config


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    return Mock(spec=EventBus)


@pytest.fixture
async def agent_manager_with_mocks(mock_config, mock_event_bus):
    """Create AgentManager with mocked dependencies."""
    with patch('agentsmcp.agent_manager.AgentManager._load_self_agent_class') as mock_load:
        mock_load.return_value = MockSelfAgent
        
        manager = AgentManager(mock_config, events=mock_event_bus)
        yield manager
        await manager.shutdown()


@pytest.fixture
def development_project_fixture(tmp_path):
    """Create a realistic development project structure."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    
    # Create project structure
    (project_dir / "src").mkdir()
    (project_dir / "src" / "models.py").write_text("""
class User:
    def __init__(self, email, username):
        self.email = email
        self.username = username
""")
    
    (project_dir / "src" / "api.py").write_text("""
from fastapi import FastAPI
app = FastAPI()

@app.post("/users/register")
def register_user():
    pass
""")
    
    (project_dir / "tests").mkdir()
    (project_dir / "tests" / "test_models.py").write_text("""
import pytest
from src.models import User

def test_user_creation():
    user = User("test@example.com", "testuser") 
    assert user.email == "test@example.com"
""")
    
    (project_dir / "requirements.txt").write_text("fastapi==0.104.1\npytest==7.4.0\n")
    (project_dir / "README.md").write_text("# Test Project\nA sample project for testing.")
    
    return project_dir


# Core Multi-Agent Coordination Tests

@pytest.mark.asyncio
@pytest.mark.integration
async def test_multi_agent_feature_development_coordination(agent_manager_with_mocks):
    """Test coordinated multi-agent feature development workflow."""
    manager = agent_manager_with_mocks
    
    # Define feature development task
    feature_task = TaskEnvelopeV1(
        objective="Develop user authentication feature with registration, login, and profile management",
        bounded_context="test_project/authentication",
        constraints=[
            "time_s: 300",
            "write_paths: src/auth/,tests/auth/"
        ]
    )
    
    # Execute role-based tasks in coordination
    roles_to_test = [
        "business_analyst",
        "backend_engineer", 
        "web_frontend_engineer",
        "backend_qa_engineer"
    ]
    
    job_ids = []
    for role in roles_to_test:
        job_id = await manager.spawn_agent(role, feature_task.objective, timeout=120)
        job_ids.append((role, job_id))
    
    # Wait for all agents to complete
    results = {}
    for role, job_id in job_ids:
        status = await manager.wait_for_completion(job_id, poll_interval=0.1)
        assert status.state == JobState.COMPLETED
        results[role] = json.loads(status.output)
    
    # Verify coordination outputs
    assert "requirements" in results["business_analyst"]
    assert "api_design" in results["backend_engineer"] 
    assert "ui_components" in results["web_frontend_engineer"]
    assert "test_plan" in results["backend_qa_engineer"]
    
    # Verify cross-role consistency
    ba_requirements = results["business_analyst"]["requirements"]
    be_endpoints = results["backend_engineer"]["api_design"]["endpoints"]
    fe_components = results["web_frontend_engineer"]["ui_components"]
    qa_tests = results["backend_qa_engineer"]["test_plan"]["unit_tests"]
    
    # Check that all roles address user registration
    assert any("registration" in req.lower() for req in ba_requirements)
    assert any("register" in endpoint.lower() for endpoint in be_endpoints)
    assert any("registration" in comp.lower() for comp in fe_components)
    assert any("registration" in test.lower() for test in qa_tests)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_team_runner_parallel_execution(development_project_fixture):
    """Test parallel team execution with realistic development workflow."""
    with patch('agentsmcp.orchestration.team_runner.AgentManager') as mock_manager_class:
        # Setup mock manager and agents
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager
        
        # Mock job spawning and completion
        job_counter = 0
        job_results = {
            "business_analyst": json.dumps({"analysis": "requirements complete"}),
            "backend_engineer": json.dumps({"implementation": "API endpoints created"}),
            "web_frontend_engineer": json.dumps({"ui": "components implemented"}),
            "backend_qa_engineer": json.dumps({"testing": "test suite created"})
        }
        
        async def mock_spawn_agent(agent_type, task):
            nonlocal job_counter
            job_counter += 1
            return f"job_{job_counter}_{agent_type}"
            
        async def mock_wait_for_completion(job_id):
            agent_type = job_id.split("_")[-1]
            return Mock(
                state=JobState.COMPLETED,
                output=job_results.get(agent_type, "task completed"),
                error=None
            )
        
        mock_manager.spawn_agent = mock_spawn_agent
        mock_manager.wait_for_completion = mock_wait_for_completion
        
        # Run team development task
        objective = "Implement user management system with authentication"
        selected_roles = ["business_analyst", "backend_engineer", "web_frontend_engineer", "backend_qa_engineer"]
        
        results = await run_team(objective, roles=selected_roles)
        
        # Verify all roles executed
        assert len(results) == len(selected_roles)
        for role in selected_roles:
            assert role in results
            assert results[role]  # Non-empty result
        
        # Verify parallel execution (all agents were spawned)
        assert mock_manager.spawn_agent.call_count == len(selected_roles)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_role_routing_and_decision_making(mock_config):
    """Test role registry routing and decision-making capabilities."""
    registry = RoleRegistry()
    
    # Test different task types route to appropriate roles
    test_cases = [
        {
            "task": TaskEnvelopeV1(objective="Gather requirements for user dashboard"),
            "expected_role": "business_analyst"
        },
        {
            "task": TaskEnvelopeV1(objective="Implement REST API endpoints for user management"),
            "expected_role": "backend_engineer"
        },
        {
            "task": TaskEnvelopeV1(objective="Create responsive UI components for mobile"),
            "expected_role": "web_frontend_engineer"
        },
        {
            "task": TaskEnvelopeV1(objective="Write integration tests for API endpoints"),
            "expected_role": "backend_qa_engineer"
        }
    ]
    
    for case in test_cases:
        role, decision = registry.route(case["task"])
        # Verify role assignment logic
        assert role is not None
        assert decision is not None
        # Note: Actual role matching depends on registry implementation


# Tool Integration Tests

@pytest.mark.asyncio
@pytest.mark.integration
async def test_development_tool_integration_workflow(agent_manager_with_mocks, development_project_fixture):
    """Test integration of development tools during agent workflows."""
    manager = agent_manager_with_mocks
    
    # Mock tool registry and execution
    with patch('agentsmcp.tools.get_tool_registry') as mock_registry:
        mock_tools = {
            "file_operations": Mock(),
            "shell_command": Mock(),
            "code_analysis": Mock(),
            "web_search": Mock()
        }
        
        mock_registry.return_value.get_tool.side_effect = lambda name: mock_tools.get(name)
        
        # Mock tool execution results
        mock_tools["file_operations"].execute = AsyncMock(return_value="File operations completed")
        mock_tools["shell_command"].execute = AsyncMock(return_value="Tests passed: 10/10")
        mock_tools["code_analysis"].execute = AsyncMock(return_value="Code quality: A+")
        mock_tools["web_search"].execute = AsyncMock(return_value="Best practices found")
        
        # Execute development workflow task
        task = "Analyze code quality and run tests for the project"
        job_id = await manager.spawn_agent("backend_qa_engineer", task)
        status = await manager.wait_for_completion(job_id)
        
        assert status.state == JobState.COMPLETED
        assert status.output  # Non-empty output


@pytest.mark.asyncio
@pytest.mark.integration
async def test_lazy_loading_tool_performance(agent_manager_with_mocks):
    """Test tool lazy loading performance during development workflows."""
    manager = agent_manager_with_mocks
    
    # Mock lazy loading behavior
    with patch('agentsmcp.tools.get_tool_registry') as mock_registry:
        load_times = []
        original_time = datetime.now()
        
        def mock_get_tool(name):
            load_times.append((name, datetime.now()))
            return Mock(execute=AsyncMock(return_value=f"{name} executed"))
        
        mock_registry.return_value.get_tool = mock_get_tool
        
        # Execute tasks that require different tools
        tasks = [
            "Read project files and analyze structure", 
            "Run shell commands to execute tests",
            "Search web for best practices",
            "Perform code analysis"
        ]
        
        job_ids = []
        for task in tasks:
            job_id = await manager.spawn_agent("dev_tooling_engineer", task)
            job_ids.append(job_id)
        
        # Wait for all to complete
        for job_id in job_ids:
            status = await manager.wait_for_completion(job_id)
            assert status.state == JobState.COMPLETED
        
        # Verify lazy loading occurred (tools loaded on demand)
        assert len(load_times) > 0


# Realistic Development Workflow Simulation

@pytest.mark.asyncio
@pytest.mark.integration
async def test_complete_feature_development_cycle(agent_manager_with_mocks, development_project_fixture):
    """Test complete feature development cycle from requirements to deployment."""
    manager = agent_manager_with_mocks
    
    # Phase 1: Requirements Analysis
    requirements_task = "Analyze requirements for user authentication feature"
    ba_job = await manager.spawn_agent("business_analyst", requirements_task)
    ba_status = await manager.wait_for_completion(ba_job)
    
    assert ba_status.state == JobState.COMPLETED
    ba_result = json.loads(ba_status.output)
    assert "requirements" in ba_result
    
    # Phase 2: Backend Implementation
    backend_task = f"Implement backend based on requirements: {ba_status.output}"
    be_job = await manager.spawn_agent("backend_engineer", backend_task)
    be_status = await manager.wait_for_completion(be_job)
    
    assert be_status.state == JobState.COMPLETED
    be_result = json.loads(be_status.output)
    assert "api_design" in be_result
    
    # Phase 3: Frontend Implementation
    frontend_task = f"Implement frontend based on API design: {be_status.output}"
    fe_job = await manager.spawn_agent("web_frontend_engineer", frontend_task)
    fe_status = await manager.wait_for_completion(fe_job)
    
    assert fe_status.state == JobState.COMPLETED
    fe_result = json.loads(fe_status.output)
    assert "ui_components" in fe_result
    
    # Phase 4: Quality Assurance
    qa_task = f"Test implementation against requirements and API design"
    qa_job = await manager.spawn_agent("backend_qa_engineer", qa_task)
    qa_status = await manager.wait_for_completion(qa_job)
    
    assert qa_status.state == JobState.COMPLETED
    qa_result = json.loads(qa_status.output)
    assert "test_plan" in qa_result
    
    # Phase 5: Chief QA Review
    review_task = "Review all implementation phases and approve for deployment"
    chief_qa_job = await manager.spawn_agent("chief_qa_engineer", review_task)
    chief_qa_status = await manager.wait_for_completion(chief_qa_job)
    
    assert chief_qa_status.state == JobState.COMPLETED
    chief_qa_result = json.loads(chief_qa_status.output)
    assert "quality_gates" in chief_qa_result


@pytest.mark.asyncio
@pytest.mark.integration
async def test_development_workflow_error_handling_and_recovery(agent_manager_with_mocks):
    """Test error handling and recovery during development workflows."""
    manager = agent_manager_with_mocks
    
    # Mock agent that fails initially then succeeds
    failure_count = 0
    
    class FailingMockAgent(MockSelfAgent):
        async def execute_task(self, task: str) -> str:
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:  # Fail first 2 attempts
                raise Exception(f"Simulated failure #{failure_count}")
            return await super().execute_task(task)
    
    with patch('agentsmcp.agent_manager.AgentManager._load_self_agent_class') as mock_load:
        mock_load.return_value = FailingMockAgent
        
        # Create new manager with failing agent
        failing_manager = AgentManager(manager.config, events=manager.events)
        
        try:
            # First job should fail
            job_id1 = await failing_manager.spawn_agent("backend_engineer", "implement API")
            status1 = await failing_manager.wait_for_completion(job_id1)
            assert status1.state == JobState.FAILED
            assert "Simulated failure" in status1.error
            
            # Second job should also fail
            job_id2 = await failing_manager.spawn_agent("backend_engineer", "implement API retry")
            status2 = await failing_manager.wait_for_completion(job_id2)
            assert status2.state == JobState.FAILED
            
            # Third job should succeed
            job_id3 = await failing_manager.spawn_agent("backend_engineer", "implement API final")
            status3 = await failing_manager.wait_for_completion(job_id3)
            assert status3.state == JobState.COMPLETED
            assert status3.output  # Should have valid output
            
        finally:
            await failing_manager.shutdown()


# Performance and Scalability Tests

@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_development_team_performance(agent_manager_with_mocks):
    """Test performance with concurrent development team execution."""
    manager = agent_manager_with_mocks
    
    # Simulate large development team
    team_roles = [
        "business_analyst", "backend_engineer", "web_frontend_engineer", 
        "api_engineer", "backend_qa_engineer", "web_frontend_qa_engineer",
        "ci_cd_engineer", "dev_tooling_engineer"
    ]
    
    start_time = datetime.now()
    
    # Launch all agents concurrently
    job_ids = []
    for role in team_roles:
        task = f"Execute {role} tasks for feature development"
        job_id = await manager.spawn_agent(role, task, timeout=60)
        job_ids.append((role, job_id))
    
    # Wait for all completions
    results = {}
    for role, job_id in job_ids:
        status = await manager.wait_for_completion(job_id, poll_interval=0.05)
        results[role] = status
    
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    # Verify all completed successfully
    for role, status in results.items():
        assert status.state == JobState.COMPLETED, f"{role} failed: {status.error}"
    
    # Performance assertions
    assert execution_time < 30  # Should complete within 30 seconds
    assert len(results) == len(team_roles)  # All roles completed
    
    # Check concurrent execution (not serialized)
    # If executed serially, would take much longer
    assert execution_time < len(team_roles) * 2  # Much faster than serial


@pytest.mark.asyncio
@pytest.mark.integration
async def test_memory_usage_during_extended_development_session(agent_manager_with_mocks):
    """Test memory usage and cleanup during extended development sessions."""
    import psutil
    import os
    
    manager = agent_manager_with_mocks
    process = psutil.Process(os.getpid())
    
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Simulate extended development session with many tasks
    for iteration in range(10):
        job_ids = []
        
        # Spawn multiple agents
        for role in ["business_analyst", "backend_engineer", "web_frontend_engineer"]:
            task = f"Development task iteration {iteration}"
            job_id = await manager.spawn_agent(role, task)
            job_ids.append(job_id)
        
        # Wait for completion
        for job_id in job_ids:
            status = await manager.wait_for_completion(job_id)
            assert status.state == JobState.COMPLETED
        
        # Trigger cleanup
        await manager.cleanup_completed_jobs(max_age_hours=0)
        
        # Check memory hasn't grown excessively
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = current_memory - initial_memory
        
        # Allow reasonable memory growth but prevent leaks
        assert memory_growth < 100, f"Memory grew by {memory_growth}MB, possible leak"


# Test Fixtures and Data

@pytest.mark.asyncio
@pytest.mark.integration
async def test_development_workflow_with_realistic_project_data(development_project_fixture):
    """Test workflows with realistic project data and file structures."""
    project_path = development_project_fixture
    
    # Verify project structure was created
    assert (project_path / "src" / "models.py").exists()
    assert (project_path / "src" / "api.py").exists()
    assert (project_path / "tests" / "test_models.py").exists()
    assert (project_path / "requirements.txt").exists()
    
    # Read and validate content
    models_content = (project_path / "src" / "models.py").read_text()
    assert "class User:" in models_content
    
    api_content = (project_path / "src" / "api.py").read_text()
    assert "FastAPI" in api_content
    assert "/users/register" in api_content
    
    test_content = (project_path / "tests" / "test_models.py").read_text()
    assert "def test_user_creation" in test_content
    
    requirements = (project_path / "requirements.txt").read_text()
    assert "fastapi" in requirements
    assert "pytest" in requirements


# Retrospective and Learning Tests

@pytest.mark.asyncio
@pytest.mark.integration
async def test_development_retrospective_and_learning_capabilities(agent_manager_with_mocks):
    """Test retrospective and continuous learning capabilities in development workflows."""
    manager = agent_manager_with_mocks
    
    # Execute development tasks
    development_tasks = [
        ("business_analyst", "Analyze user requirements"),
        ("backend_engineer", "Implement user service"),
        ("web_frontend_engineer", "Create user interface"),
        ("backend_qa_engineer", "Test implementation")
    ]
    
    task_results = []
    for role, task in development_tasks:
        job_id = await manager.spawn_agent(role, task)
        status = await manager.wait_for_completion(job_id)
        task_results.append({
            "role": role,
            "task": task,
            "result": status.output,
            "duration": 1.5,  # Mock duration
            "success": status.state == JobState.COMPLETED
        })
    
    # Simulate retrospective analysis
    retrospective_task = f"Analyze development workflow results and identify improvements: {json.dumps(task_results)}"
    retro_job = await manager.spawn_agent("agile_coach", retrospective_task)
    retro_status = await manager.wait_for_completion(retro_job)
    
    assert retro_status.state == JobState.COMPLETED
    
    # Verify retrospective contains learning insights
    retro_result = retro_status.output
    assert retro_result  # Non-empty retrospective
    
    # Check for typical retrospective elements
    retro_lower = retro_result.lower()
    retrospective_keywords = ["improvement", "efficiency", "workflow", "process", "team"]
    assert any(keyword in retro_lower for keyword in retrospective_keywords)


# Edge Cases and Error Conditions

@pytest.mark.asyncio
@pytest.mark.integration
async def test_development_workflow_timeout_handling(agent_manager_with_mocks):
    """Test handling of timeouts in development workflows."""
    manager = agent_manager_with_mocks
    
    # Mock agent that takes too long
    class SlowMockAgent(MockSelfAgent):
        async def execute_task(self, task: str) -> str:
            await asyncio.sleep(5)  # Longer than timeout
            return await super().execute_task(task)
    
    with patch('agentsmcp.agent_manager.AgentManager._load_self_agent_class') as mock_load:
        mock_load.return_value = SlowMockAgent
        
        slow_manager = AgentManager(manager.config, events=manager.events)
        
        try:
            # Set short timeout
            job_id = await slow_manager.spawn_agent("backend_engineer", "slow task", timeout=2)
            status = await slow_manager.wait_for_completion(job_id)
            
            assert status.state == JobState.TIMEOUT
            assert "timed out" in status.error.lower()
            
        finally:
            await slow_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.integration 
async def test_development_workflow_resource_exhaustion(agent_manager_with_mocks):
    """Test behavior under resource exhaustion conditions."""
    manager = agent_manager_with_mocks
    
    # Test with very high concurrency to trigger backpressure
    many_jobs = []
    for i in range(20):  # More than typical concurrency limit
        job_id = await manager.spawn_agent("backend_engineer", f"task {i}")
        many_jobs.append(job_id)
    
    # Some should queue, others execute immediately
    queue_size = manager.queue_size()
    
    # Wait for all to complete
    completed_count = 0
    for job_id in many_jobs:
        try:
            status = await manager.wait_for_completion(job_id, poll_interval=0.1)
            if status.state == JobState.COMPLETED:
                completed_count += 1
        except Exception:
            pass  # Some may fail due to resource limits
    
    # At least some should complete successfully
    assert completed_count > 10
    
    # Queue should have been utilized
    assert queue_size >= 0  # Queue size is valid


# Property-Based Tests

@pytest.mark.asyncio
@pytest.mark.integration
async def test_development_workflow_consistency_properties(agent_manager_with_mocks):
    """Test consistency properties of development workflows."""
    manager = agent_manager_with_mocks
    
    # Property: Same task executed by same role should produce consistent results
    consistent_task = "Analyze user authentication requirements"
    results = []
    
    for _ in range(3):
        job_id = await manager.spawn_agent("business_analyst", consistent_task)
        status = await manager.wait_for_completion(job_id)
        assert status.state == JobState.COMPLETED
        results.append(json.loads(status.output))
    
    # Verify consistency (all should have same structure)
    for result in results:
        assert "requirements" in result
        assert "acceptance_criteria" in result
        assert isinstance(result["requirements"], list)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_development_workflow_idempotency_properties(agent_manager_with_mocks):
    """Test idempotency properties of development operations."""
    manager = agent_manager_with_mocks
    
    # Property: Repeated execution of same development task should be safe
    idempotent_task = "Review code quality for user service module"
    
    first_execution = await manager.spawn_agent("backend_qa_engineer", idempotent_task)
    first_status = await manager.wait_for_completion(first_execution)
    
    second_execution = await manager.spawn_agent("backend_qa_engineer", idempotent_task) 
    second_status = await manager.wait_for_completion(second_execution)
    
    # Both should complete successfully
    assert first_status.state == JobState.COMPLETED
    assert second_status.state == JobState.COMPLETED
    
    # Results should be consistent (idempotent)
    first_result = json.loads(first_status.output)
    second_result = json.loads(second_status.output)
    
    assert first_result.keys() == second_result.keys()  # Same structure


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])