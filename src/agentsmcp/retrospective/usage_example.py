"""Usage example for the enhanced retrospective system.

This example demonstrates the complete retrospective flow:
1. Individual agent retrospectives
2. Agile coach comprehensive analysis
3. Action point enforcement
4. System readiness assessment
"""

import asyncio
from datetime import datetime, timezone

from ..models import TaskEnvelopeV1, ResultEnvelopeV1, EnvelopeStatus
from ..roles.base import RoleName
from ..orchestration.models import TeamComposition, TeamPerformanceMetrics, AgentSpec

from .individual_framework import IndividualRetrospectiveFramework, IndividualRetrospectiveConfig
from .coach_analyzer import AgileCoachAnalyzer
from .enforcement import OrchestratorEnforcementSystem, SystemState
from .integration_layer import EnhancedRetrospectiveIntegration, OrchestratorConfig


async def demonstrate_enhanced_retrospective_flow():
    """Demonstrate the complete enhanced retrospective flow."""
    
    print("🚀 Enhanced Retrospective System Demonstration")
    print("=" * 60)
    
    # Step 1: Initialize the integrated retrospective system
    print("\n📋 Step 1: Initializing Enhanced Retrospective System")
    print("-" * 50)
    
    config = OrchestratorConfig()
    config.enable_individual_retrospectives = True
    config.enable_comprehensive_analysis = True 
    config.enable_enforcement_system = True
    config.retrospective_timeout = 30
    
    integration = EnhancedRetrospectiveIntegration(orchestrator_config=config)
    
    # Initialize components
    await integration.initialize_integration(validate_compatibility=False)
    
    print(f"✅ Integration Status: {integration.integration_status.integrated}")
    print(f"📊 Individual Framework: {integration.integration_status.individual_framework_active}")
    print(f"🧠 Coach Analyzer: {integration.integration_status.coach_analyzer_active}")
    print(f"⚡ Enforcement System: {integration.integration_status.enforcement_system_active}")
    
    # Step 2: Simulate task execution and individual retrospectives
    print("\n🔍 Step 2: Conducting Individual Agent Retrospectives")
    print("-" * 50)
    
    # Create sample task context
    task_context = TaskEnvelopeV1(
        objective="Implement user authentication system with security best practices",
        inputs={
            "task_id": "auth_system_v2_001",
            "requirements": [
                "JWT-based authentication",
                "Password hashing with bcrypt", 
                "Rate limiting for login attempts",
                "Session management",
                "Multi-factor authentication support",
            ],
            "constraints": [
                "Must integrate with existing user database",
                "Follow OWASP security guidelines",
                "Support OAuth2 providers",
            ],
        },
        constraints=["Production ready", "Full test coverage", "Security audit ready"],
    )
    
    # Simulate successful execution results
    coder_results = ResultEnvelopeV1(
        status=EnvelopeStatus.SUCCESS,
        artifacts={
            "authentication_service": "auth_service.py",
            "user_models": "user_models.py", 
            "jwt_utils": "jwt_utils.py",
            "tests": "test_auth_suite.py",
            "documentation": "auth_api_docs.md",
        },
        metrics={
            "lines_of_code": 420,
            "test_coverage": 0.92,
            "completion_time_hours": 6.5,
            "security_scan_score": 9.2,
        },
        confidence=0.88,
        notes="Authentication system implemented with comprehensive security measures and testing",
    )
    
    # QA execution results
    qa_results = ResultEnvelopeV1(
        status=EnvelopeStatus.SUCCESS,
        artifacts={
            "security_test_suite": "security_tests.py",
            "performance_tests": "perf_tests.py",
            "vulnerability_report": "security_audit.md",
        },
        metrics={
            "tests_created": 35,
            "vulnerabilities_found": 2,
            "performance_tests": 8,
            "coverage_validation": 0.94,
        },
        confidence=0.85,
        notes="Comprehensive security and performance testing completed with minor vulnerabilities addressed",
    )
    
    # Architect results with some coordination challenges
    architect_results = ResultEnvelopeV1(
        status=EnvelopeStatus.SUCCESS,
        artifacts={
            "system_design": "auth_architecture.md",
            "security_analysis": "security_design.md",
            "api_specification": "auth_api_spec.yaml",
        },
        metrics={
            "design_documents": 3,
            "stakeholder_reviews": 4,
            "design_iterations": 2,
        },
        confidence=0.75,
        notes="Architecture design completed but required multiple iterations due to coordination issues",
    )
    
    # Conduct individual retrospectives
    individual_retrospectives = []
    
    # Coder retrospective
    print("  🛠️  Conducting Coder retrospective...")
    coder_retro, _ = await integration.conduct_enhanced_retrospective(
        agent_role=RoleName.CODER,
        task_context=task_context,
        execution_results=coder_results,
    )
    individual_retrospectives.append(coder_retro)
    print(f"     Performance Score: {coder_retro.performance_assessment.overall_score:.2f}")
    print(f"     Key Learnings: {len(coder_retro.key_learnings)}")
    print(f"     Improvement Actions: {len(coder_retro.self_improvement_actions)}")
    
    # QA retrospective 
    print("  🧪 Conducting QA retrospective...")
    qa_retro, _ = await integration.conduct_enhanced_retrospective(
        agent_role=RoleName.QA,
        task_context=task_context,
        execution_results=qa_results,
    )
    individual_retrospectives.append(qa_retro)
    print(f"     Performance Score: {qa_retro.performance_assessment.overall_score:.2f}")
    print(f"     Key Learnings: {len(qa_retro.key_learnings)}")
    print(f"     Challenges: {len(qa_retro.challenges_encountered)}")
    
    # Architect retrospective
    print("  🏗️  Conducting Architect retrospective...")
    architect_retro, _ = await integration.conduct_enhanced_retrospective(
        agent_role=RoleName.ARCHITECT,
        task_context=task_context,
        execution_results=architect_results,
    )
    individual_retrospectives.append(architect_retro)
    print(f"     Performance Score: {architect_retro.performance_assessment.overall_score:.2f}")
    print(f"     Coordination Score: {architect_retro.communication_effectiveness:.2f}")
    print(f"     Improvement Actions: {len(architect_retro.self_improvement_actions)}")
    
    # Step 3: Comprehensive Analysis by Agile Coach
    print("\n🧠 Step 3: Agile Coach Comprehensive Analysis")
    print("-" * 50)
    
    # Prepare team context for comprehensive analysis
    team_context = {
        "individual_retrospectives": individual_retrospectives,
        "team_composition": TeamComposition(
            primary_team=[
                AgentSpec(role="coder", capabilities=["python", "security", "testing"]),
                AgentSpec(role="qa", capabilities=["security_testing", "performance", "automation"]),
                AgentSpec(role="architect", capabilities=["system_design", "security_architecture", "api_design"]),
            ],
            coordination_strategy="collaborative",
        ),
        "execution_metrics": TeamPerformanceMetrics(
            success_rate=0.85,
            average_duration=7200.0,  # 2 hours average
            average_cost=25.0,
        ),
        "historical_reports": [],  # No historical data for this example
    }
    
    # Trigger comprehensive analysis with the last retrospective
    print("  🔍 Analyzing patterns across all agent retrospectives...")
    _, comprehensive_report = await integration.conduct_enhanced_retrospective(
        agent_role=RoleName.ARCHITECT,  # This will trigger comprehensive analysis
        task_context=task_context,
        execution_results=architect_results,
        team_context=team_context,
    )
    
    print(f"  📊 Analysis Results:")
    print(f"     Team Performance Score: {comprehensive_report.overall_team_performance:.2f}")
    print(f"     Patterns Identified: {len(comprehensive_report.pattern_analysis)}")
    print(f"     Systemic Issues: {len(comprehensive_report.systemic_issues)}")
    print(f"     Action Points Generated: {len(comprehensive_report.action_points)}")
    print(f"     Learning Outcomes: {len(comprehensive_report.learning_outcomes)}")
    
    # Show key insights
    if comprehensive_report.pattern_analysis:
        print(f"  🔎 Key Pattern: {comprehensive_report.pattern_analysis[0].pattern_description}")
    
    if comprehensive_report.systemic_issues:
        print(f"  ⚠️  Systemic Issue: {comprehensive_report.systemic_issues[0].title}")
    
    # Step 4: Action Point Enforcement
    print("\n⚡ Step 4: Orchestrator Action Point Enforcement")
    print("-" * 50)
    
    print("  📋 Creating enforcement plan...")
    enforcement_success = await integration.enforce_action_points(comprehensive_report)
    
    if enforcement_success:
        print("  ✅ All critical action points enforced successfully")
    else:
        print("  ⚠️  Some action points could not be enforced - manual intervention required")
    
    # Show priority matrix
    matrix = comprehensive_report.priority_matrix
    print(f"  📈 Priority Matrix:")
    print(f"     High Impact, Low Effort: {len(matrix.high_impact_low_effort)} actions (Quick Wins)")
    print(f"     High Impact, High Effort: {len(matrix.high_impact_high_effort)} actions (Strategic)")
    print(f"     Low Impact, Low Effort: {len(matrix.low_impact_low_effort)} actions (Easy Improvements)")
    print(f"     Low Impact, High Effort: {len(matrix.low_impact_high_effort)} actions (Future Consideration)")
    
    # Step 5: System Readiness Assessment
    print("\n🎯 Step 5: System Readiness Assessment")
    print("-" * 50)
    
    print("  🔍 Assessing system readiness for next task...")
    ready_for_next = await integration.assess_readiness_for_next_task()
    
    if ready_for_next:
        print("  ✅ System is ready for next task execution")
        print("     All critical action points have been addressed")
        print("     No blocking issues identified")
    else:
        print("  ⚠️  System not ready for next task")
        print("     Action points must be completed before proceeding")
    
    # Step 6: Summary and Recommendations
    print("\n📈 Step 6: Summary and Key Recommendations")
    print("-" * 50)
    
    avg_individual_score = sum(r.performance_assessment.overall_score for r in individual_retrospectives) / len(individual_retrospectives)
    
    print(f"  📊 Overall Results:")
    print(f"     Average Individual Performance: {avg_individual_score:.2f}")
    print(f"     Team Performance: {comprehensive_report.overall_team_performance:.2f}")
    print(f"     Collaboration Effectiveness: {comprehensive_report.collaboration_effectiveness:.2f}")
    
    print(f"\n  🎯 Key Success Factors:")
    for i, success in enumerate(comprehensive_report.success_factors[:3], 1):
        print(f"     {i}. {success}")
    
    print(f"\n  🔧 Top Improvement Opportunities:")
    for i, opportunity in enumerate(comprehensive_report.improvement_opportunities[:3], 1):
        print(f"     {i}. {opportunity}")
    
    print(f"\n  🚀 Next Task Recommendations:")
    for i, rec in enumerate(comprehensive_report.next_task_recommendations[:3], 1):
        print(f"     {i}. {rec}")
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced Retrospective Flow Completed Successfully!")
    print("💡 Continuous improvement loop established:")
    print("   Individual Retrospectives → Coach Analysis → Action Enforcement → Readiness Assessment")
    print("=" * 60)


async def demonstrate_individual_components():
    """Demonstrate individual components of the retrospective system."""
    
    print("\n🔧 Individual Component Demonstrations")
    print("=" * 40)
    
    # Individual Retrospective Framework
    print("\n1. Individual Retrospective Framework")
    config = IndividualRetrospectiveConfig(timeout_seconds=10)
    individual_framework = IndividualRetrospectiveFramework(config=config)
    
    task = TaskEnvelopeV1(objective="Simple test", inputs={"task_id": "demo_001"})
    result = ResultEnvelopeV1(status=EnvelopeStatus.SUCCESS, confidence=0.9)
    
    retro = await individual_framework.conduct_retrospective(
        agent_role=RoleName.CODER,
        task_context=task,
        execution_results=result,
    )
    
    print(f"   Individual retrospective created: {retro.retrospective_id[:8]}...")
    print(f"   Performance score: {retro.performance_assessment.overall_score:.2f}")
    
    # Agile Coach Analyzer
    print("\n2. Agile Coach Analyzer")
    coach = AgileCoachAnalyzer(analysis_timeout=10)
    
    # Create minimal team data
    team_comp = TeamComposition(
        primary_team=[AgentSpec(role="coder", capabilities=["python"])],
        coordination_strategy="sequential",
    )
    
    team_metrics = TeamPerformanceMetrics(success_rate=0.8, average_duration=1800.0, average_cost=5.0)
    
    report = await coach.analyze_retrospectives(
        individual_retrospectives=[retro],
        team_composition=team_comp,
        execution_metrics=team_metrics,
    )
    
    print(f"   Comprehensive report created: {report.report_id[:8]}...")
    print(f"   Action points generated: {len(report.action_points)}")
    
    # Enforcement System
    print("\n3. Orchestrator Enforcement System")
    enforcer = OrchestratorEnforcementSystem(validation_timeout=5)
    
    system_state = SystemState()
    system_state.health_status = "healthy"
    
    plan = await enforcer.create_enforcement_plan(report, system_state)
    print(f"   Enforcement plan created: {plan.plan_id[:8]}...")
    print(f"   Implementation sequence: {len(plan.implementation_sequence)} steps")
    
    # Readiness assessment
    readiness = await enforcer.assess_system_readiness(system_state=system_state)
    print(f"   System readiness score: {readiness.overall_readiness_score:.2f}")
    print(f"   Ready for next task: {readiness.next_task_clearance}")


if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_retrospective_flow())
    asyncio.run(demonstrate_individual_components())