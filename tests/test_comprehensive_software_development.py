"""
Comprehensive integration test for AgentsMCP software development workflows.

This test orchestrates multiple agents through a complete development lifecycle
to validate real-world software development scenarios.
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentsmcp.agent_manager import AgentManager
from agentsmcp.orchestration.team_runner import run_team
from agentsmcp.models import JobState


# Integration Test Scenarios

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_complete_microservice_development_lifecycle(
    microservice_project, 
    performance_config,
    realistic_agent_responses
):
    """Test complete microservice development from requirements to deployment."""
    
    # Mock sophisticated agents with domain knowledge
    class DomainExpertAgent:
        def __init__(self, agent_config, config):
            self.agent_config = agent_config
            self.config = config
            self.role = getattr(agent_config, 'type', 'unknown')
            
        async def execute_task(self, task: str) -> str:
            """Domain-specific responses based on role and task."""
            responses = realistic_agent_responses
            
            if self.role == "business_analyst":
                if "authentication" in task.lower():
                    return json.dumps(responses["business_analyst"]["user_authentication"])
                else:
                    return json.dumps({
                        "requirements": ["Generic requirement 1", "Generic requirement 2"],
                        "user_stories": ["As a user, I want to use the system"],
                        "acceptance_criteria": ["System should work correctly"]
                    })
                    
            elif self.role == "backend_engineer":
                if "authentication" in task.lower():
                    return json.dumps(responses["backend_engineer"]["user_authentication"])
                else:
                    return json.dumps({
                        "architecture": {"services": ["GenericService"], "database": "PostgreSQL"},
                        "api_endpoints": ["/api/generic"],
                        "implementation_tasks": ["Implement generic functionality"]
                    })
                    
            elif self.role == "web_frontend_engineer":
                if "authentication" in task.lower():
                    return json.dumps(responses["web_frontend_engineer"]["user_authentication"])
                else:
                    return json.dumps({
                        "components": ["GenericComponent"],
                        "pages": ["GenericPage"],
                        "state_management": {"store": "Redux"}
                    })
                    
            elif self.role == "backend_qa_engineer":
                if "authentication" in task.lower():
                    return json.dumps(responses["backend_qa_engineer"]["user_authentication"])
                else:
                    return json.dumps({
                        "test_strategy": {"unit_tests": "Jest", "integration_tests": "Supertest"},
                        "test_cases": ["Test case 1", "Test case 2"],
                        "coverage_target": "90%"
                    })
                    
            elif self.role == "chief_qa_engineer":
                return json.dumps({
                    "quality_assessment": "All tests passing",
                    "release_approval": True,
                    "recommendations": ["Continue with deployment"],
                    "quality_score": 9.2
                })
                
            else:
                return json.dumps({"task_completed": task, "role": self.role})
                
        async def cleanup(self):
            pass
    
    with patch('agentsmcp.agent_manager.AgentManager._load_self_agent_class') as mock_load:
        mock_load.return_value = DomainExpertAgent
        
        manager = AgentManager(performance_config)
        
        try:
            # Phase 1: Requirements Analysis
            ba_task = "Analyze requirements for user authentication microservice including security, scalability, and user experience considerations"
            ba_job = await manager.spawn_agent("business_analyst", ba_task)
            ba_status = await manager.wait_for_completion(ba_job)
            
            assert ba_status.state == JobState.COMPLETED
            ba_result = json.loads(ba_status.output)
            assert "requirements" in ba_result
            assert len(ba_result["requirements"]) > 0
            
            # Phase 2: Backend Architecture and Implementation
            be_task = f"Design and implement backend architecture for user authentication based on requirements: {ba_status.output}"
            be_job = await manager.spawn_agent("backend_engineer", be_task)
            be_status = await manager.wait_for_completion(be_job)
            
            assert be_status.state == JobState.COMPLETED
            be_result = json.loads(be_status.output)
            assert "architecture" in be_result
            assert "api_endpoints" in be_result
            
            # Phase 3: Frontend Implementation
            fe_task = f"Implement frontend components for authentication based on backend API: {be_status.output}"
            fe_job = await manager.spawn_agent("web_frontend_engineer", fe_task)
            fe_status = await manager.wait_for_completion(fe_job)
            
            assert fe_status.state == JobState.COMPLETED
            fe_result = json.loads(fe_status.output)
            assert "components" in fe_result
            assert "pages" in fe_result
            
            # Phase 4: Quality Assurance
            qa_task = f"Develop comprehensive test suite for authentication system covering backend and frontend: Backend: {be_status.output}, Frontend: {fe_status.output}"
            qa_job = await manager.spawn_agent("backend_qa_engineer", qa_task)
            qa_status = await manager.wait_for_completion(qa_job)
            
            assert qa_status.state == JobState.COMPLETED
            qa_result = json.loads(qa_status.output)
            assert "test_strategy" in qa_result
            assert "test_cases" in qa_result
            
            # Phase 5: Chief QA Review and Release Approval
            review_task = f"Review complete authentication implementation for release approval. BA: {ba_status.output}, BE: {be_status.output}, FE: {fe_status.output}, QA: {qa_status.output}"
            chief_qa_job = await manager.spawn_agent("chief_qa_engineer", review_task)
            chief_qa_status = await manager.wait_for_completion(chief_qa_job)
            
            assert chief_qa_status.state == JobState.COMPLETED
            chief_qa_result = json.loads(chief_qa_status.output)
            assert "quality_assessment" in chief_qa_result
            assert chief_qa_result.get("release_approval") is True
            
            # Verify end-to-end consistency
            # Requirements should align with implementation
            ba_requirements = ba_result["requirements"]
            be_endpoints = be_result["api_endpoints"]
            fe_components = fe_result["components"]
            
            # Check that authentication is addressed throughout
            assert any("auth" in req.lower() or "login" in req.lower() for req in ba_requirements)
            assert any("auth" in endpoint.lower() for endpoint in be_endpoints)
            assert any("login" in comp.lower() or "auth" in comp.lower() for comp in fe_components)
            
            print(f"✓ Complete microservice development lifecycle completed successfully")
            print(f"✓ Requirements: {len(ba_requirements)} items")
            print(f"✓ API Endpoints: {len(be_endpoints)} endpoints")
            print(f"✓ Frontend Components: {len(fe_components)} components")
            print(f"✓ Quality Score: {chief_qa_result.get('quality_score', 'N/A')}")
            
        finally:
            await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_parallel_feature_development_coordination(
    development_scenarios,
    performance_config
):
    """Test parallel development of multiple features with proper coordination."""
    
    class CoordinatedAgent:
        def __init__(self, agent_config, config):
            self.agent_config = agent_config
            self.config = config
            self.role = getattr(agent_config, 'type', 'unknown')
            
        async def execute_task(self, task: str) -> str:
            # Simulate realistic work time
            await asyncio.sleep(0.2)
            
            return json.dumps({
                "feature": self._extract_feature_name(task),
                "role": self.role,
                "deliverables": self._generate_deliverables(task),
                "dependencies": self._identify_dependencies(task),
                "estimated_effort": "2-3 days",
                "status": "completed"
            })
            
        def _extract_feature_name(self, task: str) -> str:
            if "authentication" in task.lower():
                return "user_authentication"
            elif "profile" in task.lower():
                return "user_profile"
            elif "notification" in task.lower():
                return "notifications"
            else:
                return "generic_feature"
                
        def _generate_deliverables(self, task: str) -> List[str]:
            feature = self._extract_feature_name(task)
            role_deliverables = {
                "business_analyst": [f"{feature}_requirements", f"{feature}_user_stories"],
                "backend_engineer": [f"{feature}_service", f"{feature}_api"],
                "web_frontend_engineer": [f"{feature}_components", f"{feature}_pages"],
                "backend_qa_engineer": [f"{feature}_tests", f"{feature}_coverage_report"]
            }
            return role_deliverables.get(self.role, [f"{feature}_deliverable"])
            
        def _identify_dependencies(self, task: str) -> List[str]:
            if self.role == "backend_engineer":
                return ["requirements_approved"]
            elif self.role == "web_frontend_engineer":
                return ["api_ready", "requirements_approved"]
            elif self.role == "backend_qa_engineer":
                return ["backend_complete", "frontend_complete"]
            else:
                return []
                
        async def cleanup(self):
            pass
    
    with patch('agentsmcp.agent_manager.AgentManager._load_self_agent_class') as mock_load:
        mock_load.return_value = CoordinatedAgent
        
        manager = AgentManager(performance_config)
        
        try:
            # Simulate development of multiple features in parallel
            features = [
                "user_authentication",
                "user_profile_management", 
                "real_time_notifications"
            ]
            
            # Track all jobs across features and roles
            feature_jobs = {}
            
            # Launch all feature development in parallel
            for feature in features:
                feature_jobs[feature] = {}
                
                # Each feature needs different roles
                roles = ["business_analyst", "backend_engineer", "web_frontend_engineer", "backend_qa_engineer"]
                
                for role in roles:
                    task = f"Implement {feature} - {role} responsibilities"
                    job_id = await manager.spawn_agent(role, task, timeout=120)
                    feature_jobs[feature][role] = job_id
            
            # Wait for all features to complete
            feature_results = {}
            total_jobs = 0
            completed_jobs = 0
            
            for feature, role_jobs in feature_jobs.items():
                feature_results[feature] = {}
                
                for role, job_id in role_jobs.items():
                    total_jobs += 1
                    status = await manager.wait_for_completion(job_id)
                    
                    if status.state == JobState.COMPLETED:
                        completed_jobs += 1
                        feature_results[feature][role] = json.loads(status.output)
                    else:
                        feature_results[feature][role] = {"error": status.error}
            
            # Validate parallel development results
            success_rate = completed_jobs / total_jobs if total_jobs > 0 else 0
            assert success_rate >= 0.8, f"Success rate too low: {success_rate:.1%}"
            
            # Verify each feature has results from all roles
            for feature, role_results in feature_results.items():
                assert len(role_results) == 4, f"Missing role results for {feature}"
                
                # Check that dependencies are properly handled
                for role, result in role_results.items():
                    if "error" not in result:
                        assert result["feature"] == feature
                        assert result["role"] == role
                        assert "deliverables" in result
                        assert len(result["deliverables"]) > 0
            
            print(f"✓ Parallel feature development: {completed_jobs}/{total_jobs} jobs completed")
            print(f"✓ Features developed: {len(features)}")
            print(f"✓ Success rate: {success_rate:.1%}")
            
        finally:
            await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_development_workflow_with_realistic_constraints(
    frontend_project,
    task_envelope_factory,
    performance_config
):
    """Test development workflow with realistic time and resource constraints."""
    
    class ConstraintAwareAgent:
        def __init__(self, agent_config, config):
            self.agent_config = agent_config
            self.config = config
            self.role = getattr(agent_config, 'type', 'unknown')
            
        async def execute_task(self, task: str) -> str:
            # Simulate constraint-aware execution
            if "urgent" in task.lower():
                work_time = 0.1  # Fast execution for urgent tasks
                quality_score = 7.5  # Lower quality due to time pressure
            elif "thorough" in task.lower():
                work_time = 0.5  # More time for thorough work
                quality_score = 9.2  # Higher quality
            else:
                work_time = 0.3  # Standard execution time
                quality_score = 8.5  # Standard quality
                
            await asyncio.sleep(work_time)
            
            return json.dumps({
                "task_type": "constraint_aware",
                "role": self.role,
                "execution_time": work_time,
                "quality_score": quality_score,
                "trade_offs": self._analyze_trade_offs(task),
                "recommendations": self._provide_recommendations(task)
            })
            
        def _analyze_trade_offs(self, task: str) -> Dict[str, str]:
            if "urgent" in task.lower():
                return {
                    "speed": "high",
                    "quality": "moderate", 
                    "completeness": "basic",
                    "risk": "elevated"
                }
            elif "thorough" in task.lower():
                return {
                    "speed": "moderate",
                    "quality": "high",
                    "completeness": "comprehensive", 
                    "risk": "low"
                }
            else:
                return {
                    "speed": "moderate",
                    "quality": "good",
                    "completeness": "adequate",
                    "risk": "acceptable"
                }
                
        def _provide_recommendations(self, task: str) -> List[str]:
            if self.role == "business_analyst":
                return ["Validate requirements with stakeholders", "Consider edge cases"]
            elif self.role == "backend_engineer":
                return ["Add comprehensive error handling", "Implement proper logging"]
            elif self.role == "web_frontend_engineer":
                return ["Ensure mobile responsiveness", "Add accessibility features"]
            elif self.role == "backend_qa_engineer":
                return ["Include performance tests", "Verify security requirements"]
            else:
                return ["Follow best practices", "Document decisions"]
                
        async def cleanup(self):
            pass
    
    with patch('agentsmcp.agent_manager.AgentManager._load_self_agent_class') as mock_load:
        mock_load.return_value = ConstraintAwareAgent
        
        manager = AgentManager(performance_config)
        
        try:
            # Test different constraint scenarios
            constraint_scenarios = [
                {
                    "name": "urgent_bug_fix",
                    "task": "Urgent: Fix critical authentication bug blocking production deployment",
                    "roles": ["backend_engineer", "backend_qa_engineer"],
                    "expected_speed": "high",
                    "time_limit": 60
                },
                {
                    "name": "thorough_security_review",
                    "task": "Thorough: Conduct comprehensive security review of authentication system",
                    "roles": ["backend_qa_engineer", "chief_qa_engineer"],
                    "expected_speed": "moderate",
                    "time_limit": 300
                },
                {
                    "name": "standard_feature_development",
                    "task": "Implement user profile editing functionality with standard requirements",
                    "roles": ["business_analyst", "backend_engineer", "web_frontend_engineer"],
                    "expected_speed": "moderate", 
                    "time_limit": 180
                }
            ]
            
            scenario_results = {}
            
            for scenario in constraint_scenarios:
                scenario_results[scenario["name"]] = {}
                
                # Execute scenario with multiple roles
                for role in scenario["roles"]:
                    job_id = await manager.spawn_agent(
                        role, 
                        scenario["task"], 
                        timeout=scenario["time_limit"]
                    )
                    status = await manager.wait_for_completion(job_id)
                    
                    assert status.state == JobState.COMPLETED, f"Scenario {scenario['name']} failed for {role}"
                    
                    result = json.loads(status.output)
                    scenario_results[scenario["name"]][role] = result
                    
                    # Verify constraint-aware behavior
                    trade_offs = result["trade_offs"]
                    assert trade_offs["speed"] == scenario["expected_speed"]
                    
                    # Quality should reflect constraint trade-offs
                    quality_score = result["quality_score"]
                    if scenario["expected_speed"] == "high":
                        assert quality_score < 9.0  # Quality trade-off for speed
                    else:
                        assert quality_score >= 8.0  # Higher quality when not rushed
            
            # Verify all scenarios completed successfully
            for scenario_name, role_results in scenario_results.items():
                assert len(role_results) > 0, f"No results for scenario: {scenario_name}"
                
                for role, result in role_results.items():
                    assert "quality_score" in result
                    assert "trade_offs" in result
                    assert "recommendations" in result
                    assert len(result["recommendations"]) > 0
            
            print(f"✓ Constraint-aware development: {len(constraint_scenarios)} scenarios completed")
            print(f"✓ Total role executions: {sum(len(r) for r in scenario_results.values())}")
            
        finally:
            await manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])