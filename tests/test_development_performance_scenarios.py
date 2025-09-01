"""
Performance and scalability test suite for AgentsMCP development workflows.

Tests system behavior under load, resource constraints, and realistic development workloads.
"""

import asyncio
import json
import time
import pytest
import psutil
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentsmcp.agent_manager import AgentManager
from agentsmcp.config import Config
from agentsmcp.events import EventBus
from agentsmcp.models import JobState, JobStatus


# Performance Test Fixtures

@pytest.fixture
def performance_config():
    """Configuration optimized for performance testing."""
    config = Mock(spec=Config)
    config.concurrent_agents = 8  # Higher concurrency for perf tests
    config.storage = Mock()
    config.storage.type = "memory"  # Fastest storage for perf tests
    config.storage.config = {}
    config.get_agent_config = Mock(return_value=Mock())
    return config


class PerformanceMockAgent:
    """Mock agent optimized for performance testing."""
    
    def __init__(self, agent_config, config):
        self.agent_config = agent_config
        self.config = config
        self.execution_count = 0
        
    async def execute_task(self, task: str) -> str:
        """Fast mock execution for performance testing."""
        self.execution_count += 1
        
        # Simulate realistic processing time
        await asyncio.sleep(0.1)  # 100ms task
        
        return json.dumps({
            "task_id": self.execution_count,
            "task": task[:50],  # Truncate for performance
            "timestamp": time.time(),
            "agent": getattr(self.agent_config, 'type', 'unknown')
        })
        
    async def cleanup(self):
        pass


@pytest.fixture
async def performance_manager(performance_config):
    """Create agent manager optimized for performance testing."""
    with patch('agentsmcp.agent_manager.AgentManager._load_self_agent_class') as mock_load:
        mock_load.return_value = PerformanceMockAgent
        
        manager = AgentManager(performance_config)
        yield manager
        await manager.shutdown()


# Throughput and Latency Tests

@pytest.mark.asyncio
@pytest.mark.integration 
@pytest.mark.slow
async def test_high_throughput_development_workflow(performance_manager):
    """Test system throughput with high-volume development tasks."""
    manager = performance_manager
    
    # Generate large number of development tasks
    development_roles = [
        "business_analyst", "backend_engineer", "web_frontend_engineer",
        "api_engineer", "backend_qa_engineer", "web_frontend_qa_engineer",
        "ci_cd_engineer", "dev_tooling_engineer"
    ]
    
    num_tasks_per_role = 10
    total_tasks = len(development_roles) * num_tasks_per_role
    
    start_time = time.time()
    
    # Launch all tasks
    job_ids = []
    for i in range(num_tasks_per_role):
        for role in development_roles:
            task = f"Development task {i} for {role}"
            job_id = await manager.spawn_agent(role, task, timeout=30)
            job_ids.append(job_id)
    
    launch_time = time.time() - start_time
    
    # Wait for all to complete
    completed = 0
    failed = 0
    
    for job_id in job_ids:
        try:
            status = await manager.wait_for_completion(job_id, poll_interval=0.01)
            if status.state == JobState.COMPLETED:
                completed += 1
            else:
                failed += 1
        except Exception:
            failed += 1
    
    total_time = time.time() - start_time
    throughput = completed / total_time  # tasks per second
    
    # Performance assertions
    assert completed >= total_tasks * 0.8  # At least 80% success rate
    assert throughput > 5  # At least 5 tasks per second
    assert launch_time < 10  # Launch should be fast
    
    print(f"Throughput: {throughput:.2f} tasks/sec, "
          f"Success rate: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%)")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_latency_under_load(performance_manager):
    """Test task latency under increasing load."""
    manager = performance_manager
    
    latencies = []
    load_levels = [1, 5, 10, 20]  # Concurrent tasks
    
    for load in load_levels:
        load_latencies = []
        
        # Launch concurrent tasks
        start_time = time.time()
        job_ids = []
        
        for i in range(load):
            job_id = await manager.spawn_agent("backend_engineer", f"Load test task {i}")
            job_ids.append((job_id, time.time()))
        
        # Measure completion latencies
        for job_id, spawn_time in job_ids:
            status = await manager.wait_for_completion(job_id)
            if status.state == JobState.COMPLETED:
                latency = time.time() - spawn_time
                load_latencies.append(latency)
        
        if load_latencies:
            avg_latency = sum(load_latencies) / len(load_latencies)
            latencies.append((load, avg_latency))
    
    # Latency should scale reasonably with load
    for i in range(1, len(latencies)):
        prev_load, prev_latency = latencies[i-1]
        curr_load, curr_latency = latencies[i]
        
        # Latency shouldn't increase dramatically
        latency_ratio = curr_latency / prev_latency
        load_ratio = curr_load / prev_load
        
        # Latency growth should be sub-linear to load growth
        assert latency_ratio < load_ratio * 2
    
    print(f"Latency profile: {latencies}")


# Memory and Resource Tests

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_memory_efficiency_under_load(performance_manager):
    """Test memory efficiency during high-load development scenarios."""
    manager = performance_manager
    process = psutil.Process(os.getpid())
    
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_samples = [initial_memory]
    
    # Execute waves of tasks
    num_waves = 5
    tasks_per_wave = 20
    
    for wave in range(num_waves):
        # Launch task wave
        job_ids = []
        for i in range(tasks_per_wave):
            role = ["backend_engineer", "web_frontend_engineer", "backend_qa_engineer"][i % 3]
            job_id = await manager.spawn_agent(role, f"Memory test wave {wave} task {i}")
            job_ids.append(job_id)
        
        # Wait for wave completion
        for job_id in job_ids:
            await manager.wait_for_completion(job_id)
        
        # Sample memory
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_samples.append(current_memory)
        
        # Force cleanup
        await manager.cleanup_completed_jobs(max_age_hours=0)
        
        # Brief pause between waves
        await asyncio.sleep(0.5)
    
    final_memory = process.memory_info().rss / 1024 / 1024
    max_memory = max(memory_samples)
    memory_growth = final_memory - initial_memory
    
    # Memory should not grow excessively
    assert memory_growth < 50  # Less than 50MB growth
    assert max_memory < initial_memory + 100  # Peak under 100MB growth
    
    print(f"Memory profile: initial={initial_memory:.1f}MB, "
          f"final={final_memory:.1f}MB, peak={max_memory:.1f}MB")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_resource_cleanup_efficiency(performance_manager):
    """Test efficiency of resource cleanup mechanisms."""
    manager = performance_manager
    
    # Track resource metrics
    initial_jobs = len(manager.jobs)
    
    # Create and complete many jobs
    job_ids = []
    for i in range(50):
        role = ["business_analyst", "backend_engineer", "web_frontend_engineer"][i % 3]
        job_id = await manager.spawn_agent(role, f"Cleanup test task {i}")
        job_ids.append(job_id)
    
    # Wait for completion
    for job_id in job_ids:
        await manager.wait_for_completion(job_id)
    
    jobs_before_cleanup = len(manager.jobs)
    
    # Trigger cleanup
    cleanup_start = time.time()
    await manager.cleanup_completed_jobs(max_age_hours=0)
    cleanup_time = time.time() - cleanup_start
    
    jobs_after_cleanup = len(manager.jobs)
    
    # Cleanup should be efficient
    assert cleanup_time < 1.0  # Cleanup should be fast
    assert jobs_after_cleanup < jobs_before_cleanup  # Should remove completed jobs
    assert jobs_after_cleanup <= initial_jobs + 10  # Reasonable cleanup
    
    print(f"Cleanup: {jobs_before_cleanup} -> {jobs_after_cleanup} jobs in {cleanup_time:.3f}s")


# Concurrent Load Tests

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_concurrent_development_teams(performance_manager):
    """Test multiple development teams working concurrently."""
    manager = performance_manager
    
    # Define multiple development teams
    teams = [
        {
            "name": "auth_team",
            "roles": ["business_analyst", "backend_engineer", "backend_qa_engineer"],
            "task_prefix": "Authentication feature"
        },
        {
            "name": "ui_team", 
            "roles": ["web_frontend_engineer", "tui_frontend_engineer", "web_frontend_qa_engineer"],
            "task_prefix": "User interface components"
        },
        {
            "name": "api_team",
            "roles": ["api_engineer", "backend_engineer", "backend_qa_engineer"],
            "task_prefix": "API development"
        },
        {
            "name": "infra_team",
            "roles": ["ci_cd_engineer", "dev_tooling_engineer"],
            "task_prefix": "Infrastructure setup"
        }
    ]
    
    start_time = time.time()
    
    # Launch all teams concurrently
    team_jobs = {}
    for team in teams:
        team_jobs[team["name"]] = []
        
        for role in team["roles"]:
            task = f"{team['task_prefix']} - {role} implementation"
            job_id = await manager.spawn_agent(role, task, timeout=60)
            team_jobs[team["name"]].append((role, job_id))
    
    # Wait for all teams to complete
    team_results = {}
    for team_name, jobs in team_jobs.items():
        team_results[team_name] = {"completed": 0, "failed": 0, "roles": []}
        
        for role, job_id in jobs:
            status = await manager.wait_for_completion(job_id)
            if status.state == JobState.COMPLETED:
                team_results[team_name]["completed"] += 1
            else:
                team_results[team_name]["failed"] += 1
            team_results[team_name]["roles"].append(role)
    
    total_time = time.time() - start_time
    
    # Verify all teams completed successfully
    total_completed = sum(r["completed"] for r in team_results.values())
    total_expected = sum(len(team["roles"]) for team in teams)
    
    assert total_completed >= total_expected * 0.8  # 80% success rate
    assert total_time < 120  # Complete within 2 minutes
    
    print(f"Concurrent teams: {total_completed}/{total_expected} tasks completed in {total_time:.1f}s")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_queue_management_under_pressure(performance_manager):
    """Test queue management under high pressure scenarios."""
    manager = performance_manager
    
    # Overwhelm the system with tasks
    burst_size = 100
    job_ids = []
    
    # Launch burst of tasks
    burst_start = time.time()
    for i in range(burst_size):
        role = ["backend_engineer", "web_frontend_engineer"][i % 2]
        job_id = await manager.spawn_agent(role, f"Burst task {i}")
        job_ids.append(job_id)
    
    burst_time = time.time() - burst_start
    
    # Monitor queue behavior
    max_queue_size = 0
    queue_samples = []
    
    # Sample queue size periodically
    for _ in range(20):
        queue_size = manager.queue_size()
        max_queue_size = max(max_queue_size, queue_size)
        queue_samples.append(queue_size)
        await asyncio.sleep(0.1)
    
    # Wait for completion
    completed = 0
    for job_id in job_ids:
        try:
            status = await manager.wait_for_completion(job_id, poll_interval=0.05)
            if status.state == JobState.COMPLETED:
                completed += 1
        except asyncio.TimeoutError:
            pass
    
    # Queue should handle burst gracefully
    assert burst_time < 5.0  # Burst submission should be fast
    assert completed >= burst_size * 0.7  # Most should complete
    assert max_queue_size > 0  # Queue should have been utilized
    
    print(f"Queue burst: {completed}/{burst_size} completed, "
          f"max queue size: {max_queue_size}")


# Stress Tests

@pytest.mark.asyncio
@pytest.mark.integration 
@pytest.mark.slow
async def test_extended_development_session_stress(performance_manager):
    """Test system stability during extended development sessions."""
    manager = performance_manager
    
    session_duration = 30  # 30 seconds of continuous activity
    session_start = time.time()
    
    task_counter = 0
    completed_count = 0
    failed_count = 0
    
    # Continuous task submission
    while time.time() - session_start < session_duration:
        # Submit batch of tasks
        batch_jobs = []
        for _ in range(5):
            role = ["business_analyst", "backend_engineer", "web_frontend_engineer"][task_counter % 3]
            job_id = await manager.spawn_agent(role, f"Stress test task {task_counter}")
            batch_jobs.append(job_id)
            task_counter += 1
        
        # Check some completions
        for job_id in batch_jobs:
            try:
                status = await manager.wait_for_completion(job_id, poll_interval=0.01)
                if status.state == JobState.COMPLETED:
                    completed_count += 1
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1
        
        # Brief pause
        await asyncio.sleep(0.1)
    
    # Wait for remaining tasks
    remaining_time = 10
    start_wait = time.time()
    
    while time.time() - start_wait < remaining_time:
        remaining_jobs = [j for j in manager.jobs.values() 
                         if j.status.state not in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]]
        if not remaining_jobs:
            break
        await asyncio.sleep(0.1)
    
    total_session_time = time.time() - session_start
    task_rate = task_counter / total_session_time
    completion_rate = completed_count / (completed_count + failed_count) if (completed_count + failed_count) > 0 else 0
    
    # System should remain stable
    assert completion_rate > 0.6  # At least 60% completion rate
    assert task_rate > 1  # At least 1 task per second
    assert completed_count > 20  # Significant number completed
    
    print(f"Stress test: {completed_count} completed, {failed_count} failed, "
          f"{task_rate:.1f} tasks/sec, {completion_rate:.1%} success rate")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_error_recovery_under_load(performance_manager):
    """Test error recovery mechanisms under high load."""
    manager = performance_manager
    
    # Create mix of successful and failing agents
    success_count = 0
    failure_count = 0
    recovery_count = 0
    
    class MixedReliabilityAgent:
        def __init__(self, agent_config, config):
            self.agent_config = agent_config
            self.config = config
            self.execution_count = 0
            
        async def execute_task(self, task: str) -> str:
            self.execution_count += 1
            
            # 30% failure rate initially
            if self.execution_count <= 3 and hash(task) % 10 < 3:
                raise Exception(f"Simulated failure for task: {task[:20]}")
            
            return json.dumps({"task": task, "attempt": self.execution_count})
            
        async def cleanup(self):
            pass
    
    with patch('agentsmcp.agent_manager.AgentManager._load_self_agent_class') as mock_load:
        mock_load.return_value = MixedReliabilityAgent
        
        # Create manager with failing agents
        failing_manager = AgentManager(manager.config, events=manager.events)
        
        try:
            # Launch many tasks expecting some failures
            job_ids = []
            for i in range(50):
                role = ["backend_engineer", "web_frontend_engineer"][i % 2]
                job_id = await failing_manager.spawn_agent(role, f"Recovery test task {i}")
                job_ids.append(job_id)
            
            # Check results
            for job_id in job_ids:
                status = await failing_manager.wait_for_completion(job_id)
                if status.state == JobState.COMPLETED:
                    success_count += 1
                elif status.state == JobState.FAILED:
                    failure_count += 1
                    # Could implement retry logic here
                    recovery_count += 1
            
            # System should handle mixed success/failure gracefully
            total_tasks = success_count + failure_count
            success_rate = success_count / total_tasks if total_tasks > 0 else 0
            
            assert total_tasks > 0
            assert success_rate >= 0.3  # At least some should succeed despite failures
            assert failure_count > 0  # Should have some failures as expected
            
        finally:
            await failing_manager.shutdown()
    
    print(f"Error recovery: {success_count} success, {failure_count} failures, "
          f"{success_rate:.1%} success rate")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])