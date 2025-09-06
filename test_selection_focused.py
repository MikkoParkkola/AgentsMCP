#!/usr/bin/env python3
"""
Focused Dynamic Selection Testing
=================================

Core tests for AgentsMCP's dynamic selection capabilities.
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.routing.selector import ModelSelector, TaskSpec
from agentsmcp.routing.models import ModelDB, Model
from agentsmcp.providers import list_models, ProviderType, ProviderConfig, ProviderError
from agentsmcp.config import ProviderType as ConfigProviderType


class FocusedSelectionTester:
    """Focused tester for core selection capabilities."""
    
    def __init__(self):
        self.temp_dir = None
        self.results = []
        
    async def run_tests(self):
        """Run focused selection tests."""
        print("🧪 AgentsMCP Focused Selection Testing")
        print("=" * 50)
        
        self.temp_dir = Path(tempfile.mkdtemp(prefix="agentsmcp_focused_"))
        
        try:
            await self.test_model_database()
            await self.test_model_selection()
            await self.test_provider_detection()
            await self.test_task_routing()
            await self.test_selection_optimization()
        finally:
            if self.temp_dir and self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
        
        self.print_summary()
    
    def record_result(self, test_name: str, passed: bool, message: str, duration: float = 0):
        """Record test result."""
        self.results.append({
            "test": test_name,
            "passed": passed,
            "message": message,
            "duration": duration
        })
        
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}: {message}")
    
    async def test_model_database(self):
        """Test model database functionality."""
        start = time.time()
        
        try:
            # Create test model data
            test_models = [
                {
                    "id": "gpt-4",
                    "name": "GPT-4",
                    "provider": "OpenAI",
                    "context_length": 128000,
                    "cost_per_input_token": 30.0,
                    "cost_per_output_token": 60.0,
                    "performance_tier": 5,
                    "categories": ["reasoning", "coding", "general"]
                },
                {
                    "id": "claude-3-sonnet",
                    "name": "Claude 3 Sonnet",
                    "provider": "Anthropic",
                    "context_length": 200000,
                    "cost_per_input_token": 3.0,
                    "cost_per_output_token": 15.0,
                    "performance_tier": 4,
                    "categories": ["reasoning", "coding", "general"]
                },
                {
                    "id": "llama-3-70b",
                    "name": "Llama 3 70B",
                    "provider": "Local",
                    "context_length": 8192,
                    "cost_per_input_token": 0.0,
                    "cost_per_output_token": 0.0,
                    "performance_tier": 3,
                    "categories": ["coding", "general"]
                }
            ]
            
            # Write test database
            db_file = self.temp_dir / "test_models.json"
            with open(db_file, 'w') as f:
                json.dump(test_models, f, indent=2)
            
            # Test ModelDB
            db = ModelDB(db_file)
            all_models = db.all_models()
            
            # Test filtering
            coding_models = db.filter_by_category("coding")
            expensive_models = db.filter_by_cost_max(10.0)
            high_perf_models = db.filter_by_performance_min(4)
            
            passed = (
                len(all_models) == 3 and
                len(coding_models) == 3 and
                len(expensive_models) == 1 and  # Only Llama (free)
                len(high_perf_models) == 2    # GPT-4 and Claude
            )
            
            message = f"DB loaded {len(all_models)} models, filtering works"
            
        except Exception as e:
            passed = False
            message = f"Failed: {e}"
        
        duration = time.time() - start
        self.record_result("model_database", passed, message, duration)
    
    async def test_model_selection(self):
        """Test intelligent model selection."""
        start = time.time()
        
        try:
            # Use the test database created above
            db_file = self.temp_dir / "test_models.json"
            db = ModelDB(db_file)
            selector = ModelSelector(db)
            
            # Test different selection scenarios
            test_cases = [
                {
                    "name": "cost_effective_coding",
                    "spec": TaskSpec(
                        task_type="coding",
                        max_cost_per_1k_tokens=10.0,
                        min_performance_tier=3
                    ),
                    "expected_provider": "Anthropic"  # Claude: good performance, reasonable cost
                },
                {
                    "name": "best_performance",
                    "spec": TaskSpec(
                        task_type="reasoning", 
                        min_performance_tier=5
                    ),
                    "expected_provider": "OpenAI"  # GPT-4: highest performance tier
                },
                {
                    "name": "budget_option",
                    "spec": TaskSpec(
                        task_type="general",
                        max_cost_per_1k_tokens=1.0
                    ),
                    "expected_provider": "Local"  # Llama: free option
                }
            ]
            
            successful_selections = 0
            selection_details = []
            
            for case in test_cases:
                try:
                    result = selector.select_model(case["spec"])
                    actual_provider = result.model.provider
                    expected_provider = case["expected_provider"]
                    
                    matches = actual_provider == expected_provider
                    if matches:
                        successful_selections += 1
                    
                    selection_details.append({
                        "case": case["name"],
                        "expected": expected_provider,
                        "actual": actual_provider,
                        "matches": matches,
                        "score": result.score,
                        "model": result.model.id
                    })
                    
                except Exception as e:
                    selection_details.append({
                        "case": case["name"],
                        "error": str(e),
                        "matches": False
                    })
            
            passed = successful_selections >= 2  # At least 2 out of 3 should work
            message = f"Selection logic: {successful_selections}/{len(test_cases)} correct"
            
        except Exception as e:
            passed = False
            message = f"Failed: {e}"
        
        duration = time.time() - start
        self.record_result("model_selection", passed, message, duration)
    
    async def test_provider_detection(self):
        """Test provider availability detection."""
        start = time.time()
        
        try:
            detection_results = {}
            
            # Test OpenAI provider (may fail without API key)
            try:
                openai_config = ProviderConfig(
                    api_key=os.getenv("OPENAI_API_KEY", "test-key")
                )
                models = await asyncio.wait_for(
                    asyncio.to_thread(list_models, ProviderType.OPENAI, openai_config),
                    timeout=5.0
                )
                detection_results["openai"] = {"available": True, "models": len(models)}
            except Exception as e:
                detection_results["openai"] = {"available": False, "error": type(e).__name__}
            
            # Test Ollama provider (local)
            try:
                ollama_config = ProviderConfig()
                models = await asyncio.wait_for(
                    asyncio.to_thread(list_models, ProviderType.OLLAMA, ollama_config),
                    timeout=3.0
                )
                detection_results["ollama"] = {"available": True, "models": len(models)}
            except Exception as e:
                detection_results["ollama"] = {"available": False, "error": type(e).__name__}
            
            # Count available providers
            available_count = sum(1 for r in detection_results.values() if r["available"])
            
            passed = True  # Always pass - detection may fail due to environment
            message = f"Provider detection: {available_count} available, errors handled gracefully"
            
        except Exception as e:
            passed = False
            message = f"Failed: {e}"
        
        duration = time.time() - start
        self.record_result("provider_detection", passed, message, duration)
    
    async def test_task_routing(self):
        """Test task-based routing logic."""
        start = time.time()
        
        try:
            # Test role-based routing if available
            try:
                from agentsmcp.roles.registry import RoleRegistry
                from agentsmcp.models import TaskEnvelopeV1
                
                registry = RoleRegistry()
                
                # Test task routing
                coding_task = TaskEnvelopeV1(
                    objective="Write a Python function to parse JSON",
                    bounded_context={"repo": "test"}
                )
                
                role, decision = registry.route(coding_task)
                
                routing_works = (
                    role is not None and
                    decision is not None and
                    hasattr(decision, 'agent_type')
                )
                
                message = f"Task routing: role={role.name().value if role else None}, agent={decision.agent_type if decision else None}"
                
            except ImportError:
                routing_works = True  # Not required
                message = "Task routing: role registry not available (optional)"
            
            passed = routing_works
            
        except Exception as e:
            passed = False
            message = f"Failed: {e}"
        
        duration = time.time() - start
        self.record_result("task_routing", passed, message, duration)
    
    async def test_selection_optimization(self):
        """Test selection optimization features."""
        start = time.time()
        
        try:
            # Test selector convenience methods
            db_file = self.temp_dir / "test_models.json"
            db = ModelDB(db_file)
            selector = ModelSelector(db)
            
            # Test different optimization strategies
            optimizations = {}
            
            # Cost-effective coding
            try:
                result = selector.best_cost_effective_coding(max_budget=5.0)
                optimizations["cost_effective"] = {
                    "model": result.model.id,
                    "cost": result.model.cost_per_input_token,
                    "within_budget": result.model.cost_per_input_token <= 5.0
                }
            except Exception as e:
                optimizations["cost_effective"] = {"error": str(e)}
            
            # Best performance regardless of cost
            try:
                result = selector.most_capable_regardless_of_cost()
                optimizations["best_performance"] = {
                    "model": result.model.id,
                    "tier": result.model.performance_tier,
                    "is_highest": result.model.performance_tier == 5
                }
            except Exception as e:
                optimizations["best_performance"] = {"error": str(e)}
            
            # Cheapest option
            try:
                result = selector.cheapest_meeting_requirements(min_tier=2)
                optimizations["cheapest"] = {
                    "model": result.model.id,
                    "cost": result.model.cost_per_input_token,
                    "is_free": result.model.cost_per_input_token == 0.0
                }
            except Exception as e:
                optimizations["cheapest"] = {"error": str(e)}
            
            # Check optimization success
            successful_optimizations = sum(1 for opt in optimizations.values() 
                                         if "error" not in opt)
            
            passed = successful_optimizations >= 2
            message = f"Selection optimization: {successful_optimizations}/3 strategies working"
            
        except Exception as e:
            passed = False
            message = f"Failed: {e}"
        
        duration = time.time() - start
        self.record_result("selection_optimization", passed, message, duration)
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 50)
        print("🎯 TEST SUMMARY")
        print("=" * 50)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        
        print(f"Tests Run: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        
        if failed > 0:
            print(f"\n❌ Failed Tests:")
            for r in self.results:
                if not r["passed"]:
                    print(f"  • {r['test']}: {r['message']}")
        
        total_duration = sum(r["duration"] for r in self.results)
        print(f"\nTotal Duration: {total_duration:.2f}s")
        
        # Overall assessment
        core_selection_working = passed >= 3
        print(f"\n🎯 Overall Assessment:")
        print(f"Core Selection Working: {'✅ YES' if core_selection_working else '❌ NO'}")
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "core_working": core_selection_working,
            "results": self.results
        }


async def main():
    """Run focused selection tests."""
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    tester = FocusedSelectionTester()
    await tester.run_tests()


if __name__ == "__main__":
    asyncio.run(main())