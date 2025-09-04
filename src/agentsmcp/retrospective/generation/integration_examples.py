"""
Integration Examples - Complete workflow demonstrations for the improvement generation system.

This module provides practical examples showing how to use the improvement generation
system components together in realistic scenarios.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Mock analysis result class for examples
from dataclasses import dataclass, field

from .improvement_engine import (
    ImprovementEngine,
    ImprovementFilter,
    ImprovementGenerationConfig
)
from .improvement_prioritizer import (
    ImprovementPrioritizer,
    PrioritizationStrategy,
    ResourceConstraints,
    StrategicGoals
)
from .improvement_implementer import (
    ImprovementImplementer,
    ImplementationStatus
)
from .improvement_generator import (
    ImprovementType,
    ImplementationEffort,
    RiskLevel
)


@dataclass
class MockAnalysisResult:
    """Mock analysis result for demonstrations."""
    session_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data_completeness: float = 0.85
    bottlenecks_identified: List[Dict] = field(default_factory=list)
    
    # Performance insights
    performance_insights: Dict[str, Any] = field(default_factory=lambda: {
        'response_times': {
            'avg_ms': 2500,
            'p95_ms': 5200,
            'p99_ms': 8100
        },
        'throughput': {
            'requests_per_second': 15.2,
            'peak_rps': 42.8
        },
        'resource_usage': {
            'cpu_percent': 78,
            'memory_mb': 1024,
            'peak_memory_mb': 1456
        }
    })
    
    # Quality insights  
    quality_insights: Dict[str, Any] = field(default_factory=lambda: {
        'success_rate': 0.92,
        'error_rate': 0.08,
        'retry_rate': 0.15,
        'timeout_rate': 0.03
    })
    
    # User experience insights
    user_experience_insights: Dict[str, Any] = field(default_factory=lambda: {
        'error_rate': 0.06,
        'user_satisfaction_score': 7.2,
        'task_completion_rate': 0.88,
        'average_session_duration': 425
    })


class ImprovementWorkflowExamples:
    """
    Complete workflow examples for the improvement generation system.
    
    Demonstrates real-world usage patterns and integration scenarios.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.engine = ImprovementEngine()
        self.prioritizer = ImprovementPrioritizer()
        self.implementer = ImprovementImplementer()
    
    async def example_1_basic_workflow(self) -> Dict[str, Any]:
        """
        Example 1: Basic improvement generation workflow.
        
        Demonstrates the simplest path from analysis to improvements.
        """
        self.logger.info("Running Example 1: Basic improvement generation workflow")
        
        # 1. Create mock analysis result
        analysis_result = MockAnalysisResult(
            session_id="example_1_session",
            data_completeness=0.90
        )
        
        # 2. Generate improvements using the engine
        improvements = await self.engine.generate_improvements(
            analysis_result=analysis_result,
            improvement_filter=None,  # No filtering
            config_override=ImprovementGenerationConfig(
                max_improvements=5,
                sort_by_priority=True
            )
        )
        
        # 3. Display results
        result_summary = {
            "workflow": "basic_generation",
            "total_improvements": len(improvements),
            "improvements": [
                {
                    "id": imp.opportunity_id,
                    "title": imp.title,
                    "type": imp.improvement_type.value,
                    "effort": imp.effort.value,
                    "risk": imp.risk.value,
                    "priority_score": round(imp.priority_score, 2),
                    "confidence": round(imp.confidence, 2),
                    "expected_benefits": imp.expected_benefits
                }
                for imp in improvements
            ],
            "generation_stats": self.engine.get_generation_stats()
        }
        
        self.logger.info(f"Generated {len(improvements)} improvements with basic workflow")
        return result_summary
    
    async def example_2_filtered_quick_wins(self) -> Dict[str, Any]:
        """
        Example 2: Generate quick wins with filtering.
        
        Shows how to filter for low-effort, high-impact improvements.
        """
        self.logger.info("Running Example 2: Filtered quick wins generation")
        
        # Create analysis result with performance issues
        analysis_result = MockAnalysisResult(
            session_id="example_2_session",
            performance_insights={
                'response_times': {
                    'avg_ms': 3200,  # Slower than ideal
                    'p95_ms': 7500,
                    'p99_ms': 12000
                }
            }
        )
        
        # Configure for quick wins
        quick_wins_filter = ImprovementFilter(
            max_effort=ImplementationEffort.LOW,
            max_risk=RiskLevel.LOW,
            min_priority_score=7.0,
            min_confidence=0.75
        )
        
        # Generate quick wins
        quick_wins = await self.engine.get_quick_wins(
            analysis_result=analysis_result,
            max_improvements=3
        )
        
        result_summary = {
            "workflow": "quick_wins_filtered",
            "filter_criteria": {
                "max_effort": "LOW",
                "max_risk": "LOW", 
                "min_priority_score": 7.0,
                "min_confidence": 0.75
            },
            "quick_wins": [
                {
                    "title": qw.title,
                    "description": qw.description[:100] + "...",
                    "effort": qw.effort.value,
                    "risk": qw.risk.value,
                    "expected_benefits": list(qw.expected_benefits.keys()),
                    "implementation_steps": len(qw.implementation_steps)
                }
                for qw in quick_wins
            ]
        }
        
        self.logger.info(f"Identified {len(quick_wins)} quick wins")
        return result_summary
    
    async def example_3_advanced_prioritization(self) -> Dict[str, Any]:
        """
        Example 3: Advanced prioritization with resource constraints.
        
        Demonstrates sophisticated prioritization considering team capacity and strategic goals.
        """
        self.logger.info("Running Example 3: Advanced prioritization workflow")
        
        # Create analysis result
        analysis_result = MockAnalysisResult(session_id="example_3_session")
        
        # Generate initial improvements
        improvements = await self.engine.generate_improvements(
            analysis_result=analysis_result,
            config_override=ImprovementGenerationConfig(
                max_improvements=15,  # Generate more for prioritization
                enable_cross_improvement_analysis=True,
                enable_dependency_analysis=True
            )
        )
        
        # Define resource constraints
        resource_constraints = ResourceConstraints(
            total_time_budget=timedelta(weeks=4),  # 4-week sprint
            available_developers=2,
            developer_skill_levels={
                "backend": 0.8,
                "frontend": 0.6,
                "devops": 0.7
            }
        )
        
        # Define strategic goals
        strategic_goals = StrategicGoals(
            target_response_time_ms=2000,  # Under 2 seconds
            target_error_rate=0.02,        # Under 2%
            priority_components={"response_processing", "error_handling"},
            strategic_initiatives={
                "performance_optimization": 0.8,
                "reliability_improvement": 0.6,
                "user_experience": 0.4
            }
        )
        
        # Prioritize with multiple strategies
        strategies_to_test = [
            PrioritizationStrategy.EXPECTED_VALUE,
            PrioritizationStrategy.RESOURCE_CONSTRAINED,
            PrioritizationStrategy.STRATEGIC_ALIGNMENT,
            PrioritizationStrategy.BALANCED_PORTFOLIO
        ]
        
        prioritization_results = {}
        
        for strategy in strategies_to_test:
            result = await self.prioritizer.prioritize_improvements(
                improvements=improvements.copy(),
                strategy=strategy,
                resource_constraints=resource_constraints,
                strategic_goals=strategic_goals,
                max_selections=8
            )
            
            prioritization_results[strategy.value] = {
                "total_selected": len(result.prioritized_improvements),
                "total_expected_value": round(result.total_expected_value, 2),
                "strategic_alignment": round(result.strategic_alignment_score, 2),
                "resource_utilization": result.resource_utilization,
                "risk_distribution": {k.value: v for k, v in result.risk_distribution.items()},
                "top_improvements": [
                    {
                        "title": imp.title,
                        "priority_score": round(imp.priority_score, 2),
                        "type": imp.improvement_type.value,
                        "effort": imp.effort.value
                    }
                    for imp in result.prioritized_improvements[:3]
                ]
            }
        
        result_summary = {
            "workflow": "advanced_prioritization",
            "initial_improvements": len(improvements),
            "resource_constraints": {
                "time_budget_weeks": 4,
                "available_developers": 2,
                "avg_skill_level": 0.7
            },
            "strategic_goals": {
                "target_response_time_ms": 2000,
                "target_error_rate": 0.02,
                "priority_components": list(strategic_goals.priority_components)
            },
            "prioritization_strategies": prioritization_results
        }
        
        self.logger.info("Completed advanced prioritization comparison")
        return result_summary
    
    async def example_4_end_to_end_implementation(self) -> Dict[str, Any]:
        """
        Example 4: Complete end-to-end workflow including implementation.
        
        Shows the full pipeline from generation to safe implementation.
        """
        self.logger.info("Running Example 4: End-to-end implementation workflow")
        
        # Generate and prioritize improvements
        analysis_result = MockAnalysisResult(session_id="example_4_session")
        
        improvements = await self.engine.get_quick_wins(
            analysis_result=analysis_result,
            max_improvements=2  # Keep small for demo
        )
        
        if not improvements:
            return {"workflow": "end_to_end", "status": "no_improvements_generated"}
        
        # Select the highest priority improvement for implementation
        selected_improvement = improvements[0]
        
        # Dry run implementation first
        self.logger.info("Performing dry run implementation")
        dry_run_result = await self.implementer.implement_improvement(
            improvement=selected_improvement,
            dry_run=True
        )
        
        # If dry run successful, do actual implementation
        actual_result = None
        if dry_run_result.status == ImplementationStatus.COMPLETED:
            self.logger.info("Dry run successful, proceeding with actual implementation")
            actual_result = await self.implementer.implement_improvement(
                improvement=selected_improvement,
                dry_run=False
            )
        
        # Gather implementation statistics
        impl_stats = self.implementer.get_implementation_stats()
        
        result_summary = {
            "workflow": "end_to_end_implementation",
            "selected_improvement": {
                "id": selected_improvement.opportunity_id,
                "title": selected_improvement.title,
                "effort": selected_improvement.effort.value,
                "risk": selected_improvement.risk.value,
                "implementation_steps": len(selected_improvement.implementation_steps)
            },
            "dry_run_result": {
                "status": dry_run_result.status.value,
                "steps_completed": len(dry_run_result.steps_completed),
                "steps_failed": len(dry_run_result.steps_failed),
                "duration_seconds": (
                    dry_run_result.total_duration.total_seconds() 
                    if dry_run_result.total_duration else 0
                )
            },
            "actual_implementation": {
                "status": actual_result.status.value if actual_result else "not_attempted",
                "steps_completed": len(actual_result.steps_completed) if actual_result else 0,
                "rollback_available": bool(selected_improvement.rollback_plan)
            } if actual_result else None,
            "implementation_stats": impl_stats
        }
        
        self.logger.info("Completed end-to-end implementation workflow")
        return result_summary
    
    async def example_5_monitoring_and_rollback(self) -> Dict[str, Any]:
        """
        Example 5: Implementation monitoring and rollback scenario.
        
        Demonstrates monitoring implementation success and performing rollbacks.
        """
        self.logger.info("Running Example 5: Monitoring and rollback workflow")
        
        # Generate and implement an improvement
        analysis_result = MockAnalysisResult(session_id="example_5_session")
        improvements = await self.engine.generate_improvements(analysis_result)
        
        if not improvements:
            return {"workflow": "monitoring_rollback", "status": "no_improvements"}
        
        selected_improvement = improvements[0]
        
        # Implement the improvement
        impl_result = await self.implementer.implement_improvement(
            improvement=selected_improvement,
            dry_run=False
        )
        
        # Monitor implementation status
        active_implementations = self.implementer.get_active_implementations()
        
        # Simulate monitoring period
        await asyncio.sleep(0.1)  # Brief delay to simulate monitoring
        
        # Simulate rollback scenario (for demonstration)
        rollback_success = False
        rollback_reason = "Simulated performance degradation detected"
        
        if impl_result.status == ImplementationStatus.COMPLETED:
            self.logger.info("Simulating rollback scenario")
            rollback_success = await self.implementer.rollback_improvement(
                improvement_id=selected_improvement.opportunity_id,
                reason=rollback_reason
            )
        
        # Get updated statistics
        impl_history = self.implementer.get_implementation_history(limit=5)
        
        result_summary = {
            "workflow": "monitoring_and_rollback",
            "improvement_implemented": {
                "id": selected_improvement.opportunity_id,
                "title": selected_improvement.title,
                "initial_status": impl_result.status.value
            },
            "monitoring": {
                "active_implementations": len(active_implementations),
                "monitoring_duration_seconds": 0.1,
                "issues_detected": ["simulated_performance_degradation"]
            },
            "rollback": {
                "performed": True,
                "reason": rollback_reason,
                "success": rollback_success,
                "rollback_plan_available": bool(selected_improvement.rollback_plan)
            },
            "implementation_history": [
                {
                    "id": result.improvement_id,
                    "status": result.status.value,
                    "rollback_performed": result.rollback_performed
                }
                for result in impl_history
            ]
        }
        
        self.logger.info("Completed monitoring and rollback workflow")
        return result_summary
    
    async def run_all_examples(self) -> Dict[str, Any]:
        """
        Run all workflow examples and return comprehensive results.
        """
        self.logger.info("Running all improvement workflow examples")
        
        examples = [
            ("basic_workflow", self.example_1_basic_workflow),
            ("filtered_quick_wins", self.example_2_filtered_quick_wins),
            ("advanced_prioritization", self.example_3_advanced_prioritization),
            ("end_to_end_implementation", self.example_4_end_to_end_implementation),
            ("monitoring_rollback", self.example_5_monitoring_and_rollback)
        ]
        
        results = {}
        total_start_time = datetime.utcnow()
        
        for example_name, example_func in examples:
            try:
                self.logger.info(f"Running example: {example_name}")
                example_start = datetime.utcnow()
                
                result = await example_func()
                
                example_duration = (datetime.utcnow() - example_start).total_seconds()
                result["execution_time_seconds"] = round(example_duration, 3)
                result["execution_status"] = "completed"
                
                results[example_name] = result
                
            except Exception as e:
                self.logger.error(f"Example {example_name} failed: {e}")
                results[example_name] = {
                    "execution_status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
        
        total_duration = (datetime.utcnow() - total_start_time).total_seconds()
        
        # Summary statistics
        successful_examples = sum(
            1 for result in results.values() 
            if result.get("execution_status") == "completed"
        )
        
        summary = {
            "workflow_examples_summary": {
                "total_examples": len(examples),
                "successful_examples": successful_examples,
                "failed_examples": len(examples) - successful_examples,
                "total_execution_time_seconds": round(total_duration, 3),
                "success_rate": successful_examples / len(examples)
            },
            "individual_examples": results
        }
        
        self.logger.info(
            f"Completed all examples: {successful_examples}/{len(examples)} successful"
        )
        
        return summary


# Convenience functions for easy integration

async def demo_basic_improvement_generation():
    """Quick demo of basic improvement generation."""
    examples = ImprovementWorkflowExamples()
    return await examples.example_1_basic_workflow()


async def demo_quick_wins_filtering():
    """Quick demo of filtered quick wins."""
    examples = ImprovementWorkflowExamples()
    return await examples.example_2_filtered_quick_wins()


async def demo_advanced_prioritization():
    """Quick demo of advanced prioritization."""
    examples = ImprovementWorkflowExamples()
    return await examples.example_3_advanced_prioritization()


async def demo_complete_workflow():
    """Run all examples and return summary."""
    examples = ImprovementWorkflowExamples()
    return await examples.run_all_examples()


# Example usage and testing
if __name__ == "__main__":
    async def main():
        """Run example workflows."""
        logging.basicConfig(level=logging.INFO)
        
        print("🚀 Starting Improvement Generation System Workflow Examples")
        print("=" * 70)
        
        examples = ImprovementWorkflowExamples()
        results = await examples.run_all_examples()
        
        print(f"\n✅ Completed all examples!")
        print(f"Success rate: {results['workflow_examples_summary']['success_rate']:.1%}")
        print(f"Total time: {results['workflow_examples_summary']['total_execution_time_seconds']:.2f}s")
        
        # Show brief summary of each example
        for example_name, result in results['individual_examples'].items():
            status = result.get('execution_status', 'unknown')
            time_taken = result.get('execution_time_seconds', 0)
            
            status_emoji = "✅" if status == "completed" else "❌"
            print(f"{status_emoji} {example_name}: {status} ({time_taken:.2f}s)")
            
            if status == "completed" and "total_improvements" in result:
                print(f"   Generated {result['total_improvements']} improvements")
    
    # Run the examples
    asyncio.run(main())