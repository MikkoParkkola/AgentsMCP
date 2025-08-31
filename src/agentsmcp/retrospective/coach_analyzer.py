"""Agile coach comprehensive analyzer for synthesizing individual retrospectives.

This module provides the agile coach analysis engine that processes multiple individual
agent retrospectives to identify patterns, systemic issues, and generate comprehensive
insights for system-wide improvement.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from ..orchestration.models import TeamComposition, TeamPerformanceMetrics
from ..roles.base import RoleName
from .data_models import (
    IndividualRetrospective,
    ComprehensiveRetrospectiveReport,
    PatternAnalysis,
    SystemicIssue,
    CrossAgentInsights,
    SystemicImprovement,
    ActionPoint,
    PriorityMatrix,
    ImplementationRoadmap,
    ImprovementCategory,
    PriorityLevel,
    RetrospectiveType,
)


class InsufficientDataError(Exception):
    """Raised when insufficient data is available for analysis."""
    pass


class AnalysisTimeoutError(Exception):
    """Raised when analysis process times out."""
    pass


class PatternRecognitionFailure(Exception):
    """Raised when pattern recognition fails."""
    pass


class AgileCoachAnalyzer:
    """Comprehensive analyzer for synthesizing multiple individual retrospectives."""
    
    def __init__(
        self,
        analysis_timeout: int = 30,
        min_retrospectives_for_patterns: int = 2,
        logger: Optional[logging.Logger] = None,
    ):
        self.analysis_timeout = analysis_timeout
        self.min_retrospectives_for_patterns = min_retrospectives_for_patterns
        self.log = logger or logging.getLogger(__name__)
        
        # Pattern recognition thresholds
        self.pattern_confidence_threshold = 0.6
        self.systemic_issue_occurrence_threshold = 2
        
        self.log.info("AgileCoachAnalyzer initialized")
    
    async def analyze_retrospectives(
        self,
        individual_retrospectives: List[IndividualRetrospective],
        team_composition: TeamComposition,
        execution_metrics: TeamPerformanceMetrics,
        historical_data: Optional[List[ComprehensiveRetrospectiveReport]] = None,
    ) -> ComprehensiveRetrospectiveReport:
        """Analyze multiple individual retrospectives to generate comprehensive insights.
        
        Args:
            individual_retrospectives: List of individual agent retrospectives
            team_composition: Team structure and roles
            execution_metrics: Overall team performance data
            historical_data: Optional historical comprehensive reports
            
        Returns:
            ComprehensiveRetrospectiveReport: Complete analysis and recommendations
            
        Raises:
            InsufficientDataError: If insufficient data for analysis
            AnalysisTimeoutError: If analysis times out
            PatternRecognitionFailure: If pattern recognition fails
        """
        start_time = datetime.now(timezone.utc)
        
        # Validate input data
        if not individual_retrospectives:
            raise InsufficientDataError("No individual retrospectives provided")
        
        if len(individual_retrospectives) < self.min_retrospectives_for_patterns:
            self.log.warning(
                "Limited retrospective data (%d items) may reduce analysis quality",
                len(individual_retrospectives)
            )
        
        self.log.info(
            "Starting comprehensive analysis of %d retrospectives",
            len(individual_retrospectives)
        )
        
        try:
            # Execute analysis with timeout
            report = await asyncio.wait_for(
                self._execute_comprehensive_analysis(
                    individual_retrospectives,
                    team_composition,
                    execution_metrics,
                    historical_data,
                ),
                timeout=self.analysis_timeout
            )
            
            # Finalize report metadata
            end_time = datetime.now(timezone.utc)
            report.analysis_duration_seconds = (end_time - start_time).total_seconds()
            report.created_at = end_time
            
            self.log.info(
                "Comprehensive analysis completed in %.2fs with %d patterns and %d systemic issues",
                report.analysis_duration_seconds,
                len(report.pattern_analysis),
                len(report.systemic_issues)
            )
            
            return report
            
        except asyncio.TimeoutError:
            self.log.error("Analysis timed out after %d seconds", self.analysis_timeout)
            raise AnalysisTimeoutError(f"Analysis timed out after {self.analysis_timeout}s")
        
        except Exception as e:
            self.log.error("Comprehensive analysis failed: %s", e)
            raise
    
    async def _execute_comprehensive_analysis(
        self,
        individual_retrospectives: List[IndividualRetrospective],
        team_composition: TeamComposition,
        execution_metrics: TeamPerformanceMetrics,
        historical_data: Optional[List[ComprehensiveRetrospectiveReport]],
    ) -> ComprehensiveRetrospectiveReport:
        """Execute the complete comprehensive analysis process."""
        
        # Initialize report
        task_id = individual_retrospectives[0].task_id if individual_retrospectives else "unknown"
        report = ComprehensiveRetrospectiveReport(
            task_id=task_id,
            retrospective_type=RetrospectiveType.COMPREHENSIVE,
            individual_retrospectives_count=len(individual_retrospectives),
            participating_agents=[r.agent_role.value for r in individual_retrospectives],
        )
        
        # Phase 1: Pattern Analysis
        self.log.debug("Starting pattern analysis phase")
        report.pattern_analysis = await self._analyze_patterns(individual_retrospectives)
        
        # Phase 2: Systemic Issues Identification
        self.log.debug("Starting systemic issues identification")
        report.systemic_issues = await self._identify_systemic_issues(individual_retrospectives)
        
        # Phase 3: Cross-Agent Insights
        self.log.debug("Starting cross-agent analysis")
        report.cross_agent_insights = await self._analyze_cross_agent_interactions(
            individual_retrospectives, team_composition
        )
        
        # Phase 4: Performance Synthesis
        self.log.debug("Starting performance synthesis")
        await self._synthesize_team_performance(report, individual_retrospectives, execution_metrics)
        
        # Phase 5: Learning Outcomes Extraction
        self.log.debug("Extracting learning outcomes")
        report.learning_outcomes = await self._extract_learning_outcomes(individual_retrospectives)
        
        # Phase 6: Success Factors and Improvement Opportunities
        self.log.debug("Identifying success factors and improvements")
        report.success_factors = await self._identify_success_factors(individual_retrospectives)
        report.improvement_opportunities = await self._identify_improvement_opportunities(
            individual_retrospectives, report.systemic_issues
        )
        
        # Phase 7: Generate Systemic Improvements
        self.log.debug("Generating systemic improvements")
        report.systemic_improvements = await self._generate_systemic_improvements(
            report.pattern_analysis, report.systemic_issues, report.cross_agent_insights
        )
        
        # Phase 8: Create Action Points
        self.log.debug("Creating action points")
        report.action_points = await self._create_action_points(
            report.systemic_improvements, report.improvement_opportunities
        )
        
        # Phase 9: Generate Priority Matrix and Roadmap
        self.log.debug("Creating priority matrix and implementation roadmap")
        report.priority_matrix = await self._create_priority_matrix(report.action_points)
        report.implementation_roadmap = await self._create_implementation_roadmap(
            report.action_points, report.priority_matrix
        )
        
        # Phase 10: Future Recommendations
        self.log.debug("Generating future recommendations")
        await self._generate_future_recommendations(report, historical_data)
        
        return report
    
    async def _analyze_patterns(
        self, individual_retrospectives: List[IndividualRetrospective]
    ) -> List[PatternAnalysis]:
        """Analyze patterns across individual retrospectives."""
        
        patterns = []
        
        try:
            # Success patterns
            success_patterns = await self._identify_success_patterns(individual_retrospectives)
            patterns.extend(success_patterns)
            
            # Failure patterns
            failure_patterns = await self._identify_failure_patterns(individual_retrospectives)
            patterns.extend(failure_patterns)
            
            # Collaboration patterns
            collaboration_patterns = await self._identify_collaboration_patterns(individual_retrospectives)
            patterns.extend(collaboration_patterns)
            
            # Learning patterns
            learning_patterns = await self._identify_learning_patterns(individual_retrospectives)
            patterns.extend(learning_patterns)
            
            # Filter patterns by confidence threshold
            high_confidence_patterns = [
                p for p in patterns if p.confidence_score >= self.pattern_confidence_threshold
            ]
            
            self.log.debug(
                "Identified %d patterns (%d high-confidence)",
                len(patterns), len(high_confidence_patterns)
            )
            
            return high_confidence_patterns
            
        except Exception as e:
            self.log.error("Pattern analysis failed: %s", e)
            raise PatternRecognitionFailure(f"Pattern analysis failed: {str(e)}")
    
    async def _identify_success_patterns(
        self, retrospectives: List[IndividualRetrospective]
    ) -> List[PatternAnalysis]:
        """Identify patterns in successful outcomes."""
        
        patterns = []
        
        # Group high-performing agents
        high_performers = [r for r in retrospectives if r.performance_assessment.overall_score > 0.8]
        
        if len(high_performers) >= self.min_retrospectives_for_patterns:
            # Analyze common success factors
            success_themes = Counter()
            for retro in high_performers:
                success_themes.update(retro.what_went_well)
            
            if success_themes:
                most_common = success_themes.most_common(3)
                pattern = PatternAnalysis(
                    pattern_type="success",
                    pattern_description=f"High-performing agents consistently report: {', '.join([item[0] for item in most_common])}",
                    confidence_score=min(0.9, len(high_performers) / len(retrospectives) + 0.3),
                    supporting_evidence=[f"{item[0]} (mentioned {item[1]} times)" for item in most_common],
                    implications=["Successful patterns should be shared across team", "Scale effective approaches"],
                    recommended_actions=["Document successful approaches", "Train other agents on effective patterns"],
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _identify_failure_patterns(
        self, retrospectives: List[IndividualRetrospective]
    ) -> List[PatternAnalysis]:
        """Identify patterns in failures and challenges."""
        
        patterns = []
        
        # Analyze common improvement areas
        improvement_themes = Counter()
        for retro in retrospectives:
            improvement_themes.update(retro.what_could_improve)
        
        # Identify recurring challenges
        challenge_themes = Counter()
        for retro in retrospectives:
            for challenge in retro.challenges_encountered:
                challenge_themes[challenge.description] += 1
        
        # Create patterns for recurring issues
        for issue, count in improvement_themes.most_common(3):
            if count >= self.systemic_issue_occurrence_threshold:
                pattern = PatternAnalysis(
                    pattern_type="failure",
                    pattern_description=f"Recurring improvement area: {issue}",
                    confidence_score=min(0.8, (count / len(retrospectives)) + 0.2),
                    supporting_evidence=[f"Mentioned by {count} agents"],
                    implications=["Systemic issue requiring attention", "May indicate training or process gap"],
                    recommended_actions=["Address root cause", "Provide targeted training or support"],
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _identify_collaboration_patterns(
        self, retrospectives: List[IndividualRetrospective]
    ) -> List[PatternAnalysis]:
        """Identify patterns in team collaboration."""
        
        patterns = []
        
        # Analyze collaboration effectiveness scores
        collab_scores = [r.communication_effectiveness for r in retrospectives if r.communication_effectiveness > 0]
        
        if collab_scores:
            avg_collab_score = sum(collab_scores) / len(collab_scores)
            
            if avg_collab_score > 0.8:
                pattern = PatternAnalysis(
                    pattern_type="collaboration",
                    pattern_description="Strong team collaboration and communication effectiveness",
                    confidence_score=0.8,
                    supporting_evidence=[f"Average collaboration score: {avg_collab_score:.2f}"],
                    implications=["Team collaboration is a strength", "Current communication patterns are effective"],
                    recommended_actions=["Maintain current collaboration practices", "Document effective communication patterns"],
                )
                patterns.append(pattern)
            elif avg_collab_score < 0.6:
                pattern = PatternAnalysis(
                    pattern_type="collaboration",
                    pattern_description="Team collaboration needs improvement",
                    confidence_score=0.7,
                    supporting_evidence=[f"Average collaboration score: {avg_collab_score:.2f}"],
                    implications=["Communication barriers affecting performance", "Coordination challenges present"],
                    recommended_actions=["Improve communication protocols", "Enhance coordination mechanisms"],
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _identify_learning_patterns(
        self, retrospectives: List[IndividualRetrospective]
    ) -> List[PatternAnalysis]:
        """Identify patterns in learning and knowledge development."""
        
        patterns = []
        
        # Analyze learning themes
        learning_themes = Counter()
        knowledge_gaps = Counter()
        
        for retro in retrospectives:
            learning_themes.update(retro.key_learnings)
            knowledge_gaps.update(retro.knowledge_gaps_identified)
        
        # Identify common learning areas
        if learning_themes:
            top_learning = learning_themes.most_common(1)[0]
            if top_learning[1] >= self.min_retrospectives_for_patterns:
                pattern = PatternAnalysis(
                    pattern_type="learning",
                    pattern_description=f"Common learning theme: {top_learning[0]}",
                    confidence_score=min(0.8, (top_learning[1] / len(retrospectives)) + 0.3),
                    supporting_evidence=[f"Identified by {top_learning[1]} agents"],
                    implications=["Valuable learning opportunity for team", "Knowledge sharing potential"],
                    recommended_actions=["Share learnings across team", "Create knowledge base entry"],
                )
                patterns.append(pattern)
        
        # Identify common knowledge gaps
        if knowledge_gaps:
            top_gap = knowledge_gaps.most_common(1)[0]
            if top_gap[1] >= self.min_retrospectives_for_patterns:
                pattern = PatternAnalysis(
                    pattern_type="learning",
                    pattern_description=f"Common knowledge gap: {top_gap[0]}",
                    confidence_score=min(0.8, (top_gap[1] / len(retrospectives)) + 0.3),
                    supporting_evidence=[f"Identified by {top_gap[1]} agents"],
                    implications=["Training opportunity identified", "Potential performance blocker"],
                    recommended_actions=["Provide targeted training", "Create learning resources"],
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _identify_systemic_issues(
        self, retrospectives: List[IndividualRetrospective]
    ) -> List[SystemicIssue]:
        """Identify systemic issues affecting multiple agents."""
        
        issues = []
        
        # Analyze improvement areas for systemic patterns
        improvement_counter = Counter()
        affected_agents = defaultdict(set)
        
        for retro in retrospectives:
            agent_role = retro.agent_role.value
            for improvement in retro.what_could_improve:
                improvement_counter[improvement] += 1
                affected_agents[improvement].add(agent_role)
        
        # Create systemic issues for recurring problems
        for improvement, count in improvement_counter.items():
            if count >= self.systemic_issue_occurrence_threshold:
                severity = self._assess_issue_severity(count, len(retrospectives))
                
                issue = SystemicIssue(
                    title=f"Recurring improvement need: {improvement[:50]}...",
                    description=improvement,
                    category=self._categorize_systemic_issue(improvement),
                    severity=severity,
                    affected_agents=list(affected_agents[improvement]),
                    occurrence_count=count,
                    root_cause_analysis=await self._analyze_root_cause(improvement, retrospectives),
                    potential_solutions=await self._suggest_solutions(improvement, severity),
                )
                issues.append(issue)
        
        # Analyze challenges for additional systemic issues
        challenge_counter = Counter()
        challenge_agents = defaultdict(set)
        
        for retro in retrospectives:
            agent_role = retro.agent_role.value
            for challenge in retro.challenges_encountered:
                challenge_key = challenge.description
                challenge_counter[challenge_key] += 1
                challenge_agents[challenge_key].add(agent_role)
        
        for challenge_desc, count in challenge_counter.items():
            if count >= self.systemic_issue_occurrence_threshold:
                severity = self._assess_issue_severity(count, len(retrospectives))
                
                issue = SystemicIssue(
                    title=f"Recurring challenge: {challenge_desc[:50]}...",
                    description=challenge_desc,
                    category="operational_challenge",
                    severity=severity,
                    affected_agents=list(challenge_agents[challenge_desc]),
                    occurrence_count=count,
                    root_cause_analysis=await self._analyze_challenge_root_cause(challenge_desc),
                    potential_solutions=await self._suggest_challenge_solutions(challenge_desc),
                )
                issues.append(issue)
        
        return issues
    
    async def _analyze_cross_agent_interactions(
        self,
        retrospectives: List[IndividualRetrospective],
        team_composition: TeamComposition,
    ) -> CrossAgentInsights:
        """Analyze interactions and collaboration between agents."""
        
        insights = CrossAgentInsights()
        
        # Calculate collaboration effectiveness
        collab_scores = [r.communication_effectiveness for r in retrospectives if r.communication_effectiveness > 0]
        if collab_scores:
            insights.collaboration_effectiveness = sum(collab_scores) / len(collab_scores)
        else:
            insights.collaboration_effectiveness = 0.7  # Default neutral
        
        # Identify communication patterns
        communication_patterns = []
        team_dynamics = []
        
        for retro in retrospectives:
            if retro.team_dynamics_observations:
                team_dynamics.extend(retro.team_dynamics_observations)
            
            # Analyze collaboration feedback for patterns
            if retro.collaboration_feedback and len(retro.collaboration_feedback) > 10:
                communication_patterns.append(retro.collaboration_feedback)
        
        # Extract common patterns
        if communication_patterns:
            insights.communication_patterns = list(set(communication_patterns))[:5]
        else:
            insights.communication_patterns = ["Standard communication protocols followed"]
        
        # Identify coordination challenges
        coordination_issues = []
        for retro in retrospectives:
            for challenge in retro.challenges_encountered:
                if "coordination" in challenge.description.lower() or "communication" in challenge.description.lower():
                    coordination_issues.append(challenge.description)
        
        insights.coordination_challenges = list(set(coordination_issues))[:5]
        
        # Suggest synergy opportunities
        high_performers = [r for r in retrospectives if r.performance_assessment.overall_score > 0.8]
        if len(high_performers) > 1:
            insights.synergy_opportunities = [
                "Pair high-performing agents for knowledge transfer",
                "Create cross-functional collaboration opportunities",
            ]
        
        # Identify role boundary issues
        role_issues = []
        role_performance = defaultdict(list)
        
        for retro in retrospectives:
            role_performance[retro.agent_role].append(retro.performance_assessment.overall_score)
        
        for role, scores in role_performance.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 0.6:
                role_issues.append(f"{role.value} role showing performance challenges")
        
        insights.role_boundary_issues = role_issues
        
        return insights
    
    async def _synthesize_team_performance(
        self,
        report: ComprehensiveRetrospectiveReport,
        retrospectives: List[IndividualRetrospective],
        execution_metrics: TeamPerformanceMetrics,
    ) -> None:
        """Synthesize overall team performance metrics."""
        
        # Calculate overall team performance from individual scores
        individual_scores = [r.performance_assessment.overall_score for r in retrospectives]
        if individual_scores:
            report.overall_team_performance = sum(individual_scores) / len(individual_scores)
        else:
            report.overall_team_performance = 0.7  # Default neutral
        
        # Use cross-agent insights collaboration score
        report.collaboration_effectiveness = report.cross_agent_insights.collaboration_effectiveness
    
    async def _extract_learning_outcomes(
        self, retrospectives: List[IndividualRetrospective]
    ) -> List[str]:
        """Extract synthesized learning outcomes from all retrospectives."""
        
        # Collect all learnings
        all_learnings = []
        for retro in retrospectives:
            all_learnings.extend(retro.key_learnings)
        
        # Count and prioritize learnings
        learning_counter = Counter(all_learnings)
        
        # Return top learnings with occurrence counts
        top_learnings = []
        for learning, count in learning_counter.most_common(10):
            if count > 1:
                top_learnings.append(f"{learning} (identified by {count} agents)")
            else:
                top_learnings.append(learning)
        
        return top_learnings
    
    async def _identify_success_factors(
        self, retrospectives: List[IndividualRetrospective]
    ) -> List[str]:
        """Identify factors that contributed to team success."""
        
        success_factors = []
        
        # Collect what went well across all agents
        all_successes = []
        for retro in retrospectives:
            all_successes.extend(retro.what_went_well)
        
        # Identify common success themes
        success_counter = Counter(all_successes)
        
        for success, count in success_counter.most_common(5):
            if count >= self.min_retrospectives_for_patterns:
                success_factors.append(f"{success} (noted by {count} agents)")
            else:
                success_factors.append(success)
        
        return success_factors
    
    async def _identify_improvement_opportunities(
        self,
        retrospectives: List[IndividualRetrospective],
        systemic_issues: List[SystemicIssue],
    ) -> List[str]:
        """Identify improvement opportunities from analysis."""
        
        opportunities = []
        
        # Extract opportunities from systemic issues
        for issue in systemic_issues:
            if issue.severity in ["high", "critical"]:
                opportunities.append(f"Address {issue.category} issues affecting {len(issue.affected_agents)} agents")
        
        # Look for performance improvement opportunities
        low_performers = [r for r in retrospectives if r.performance_assessment.overall_score < 0.6]
        if low_performers:
            opportunities.append(f"Support {len(low_performers)} agents with performance improvement")
        
        # Identify learning opportunities
        knowledge_gaps = Counter()
        for retro in retrospectives:
            knowledge_gaps.update(retro.knowledge_gaps_identified)
        
        for gap, count in knowledge_gaps.most_common(3):
            if count >= self.min_retrospectives_for_patterns:
                opportunities.append(f"Team training opportunity: {gap}")
        
        return opportunities[:10]  # Limit to top 10
    
    async def _generate_systemic_improvements(
        self,
        patterns: List[PatternAnalysis],
        systemic_issues: List[SystemicIssue],
        cross_agent_insights: CrossAgentInsights,
    ) -> List[SystemicImprovement]:
        """Generate comprehensive system-wide improvement recommendations."""
        
        improvements = []
        
        # Generate improvements from patterns
        for pattern in patterns:
            if pattern.pattern_type == "failure" and pattern.confidence_score > 0.7:
                improvement = SystemicImprovement(
                    title=f"Address pattern: {pattern.pattern_description[:50]}...",
                    description=pattern.pattern_description,
                    category=ImprovementCategory.PROCESS,
                    priority=PriorityLevel.HIGH,
                    impact_assessment="High - affects multiple agents",
                    effort_assessment="Medium - requires process changes",
                    implementation_approach="; ".join(pattern.recommended_actions),
                    success_metrics=["Reduced occurrence of pattern", "Improved agent performance scores"],
                )
                improvements.append(improvement)
        
        # Generate improvements from systemic issues
        for issue in systemic_issues:
            priority = PriorityLevel.HIGH if issue.severity in ["high", "critical"] else PriorityLevel.MEDIUM
            
            improvement = SystemicImprovement(
                title=f"Resolve systemic issue: {issue.title}",
                description=issue.description,
                category=ImprovementCategory.PROCESS,
                priority=priority,
                impact_assessment=f"Affects {len(issue.affected_agents)} agents",
                effort_assessment=self._estimate_effort(issue.severity),
                implementation_approach="; ".join(issue.potential_solutions),
                success_metrics=["Reduced issue occurrence", "Improved affected agent performance"],
            )
            improvements.append(improvement)
        
        # Generate improvements from cross-agent insights
        if cross_agent_insights.collaboration_effectiveness < 0.7:
            improvement = SystemicImprovement(
                title="Improve team collaboration and communication",
                description="Team collaboration effectiveness below optimal threshold",
                category=ImprovementCategory.COORDINATION,
                priority=PriorityLevel.MEDIUM,
                impact_assessment="Affects entire team coordination",
                effort_assessment="Medium - requires communication protocol updates",
                implementation_approach="Enhance communication protocols; provide collaboration training",
                success_metrics=["Improved collaboration scores", "Faster task completion"],
            )
            improvements.append(improvement)
        
        return improvements
    
    async def _create_action_points(
        self,
        systemic_improvements: List[SystemicImprovement],
        improvement_opportunities: List[str],
    ) -> List[ActionPoint]:
        """Create specific action points for orchestrator enforcement."""
        
        action_points = []
        
        # Convert systemic improvements to action points
        for improvement in systemic_improvements:
            action = ActionPoint(
                title=improvement.title,
                description=improvement.description,
                category=improvement.category.value,
                priority=improvement.priority,
                implementation_type=self._determine_implementation_type(improvement),
                implementation_steps=self._break_down_implementation(improvement),
                validation_criteria=improvement.success_metrics,
                estimated_effort_hours=self._estimate_effort_hours(improvement.effort_assessment),
                expected_impact=improvement.impact_assessment,
                success_metrics=improvement.success_metrics,
            )
            action_points.append(action)
        
        # Create action points from improvement opportunities
        for opportunity in improvement_opportunities[:5]:  # Limit to top 5
            action = ActionPoint(
                title=f"Capitalize on opportunity: {opportunity[:50]}...",
                description=opportunity,
                category="opportunity",
                priority=PriorityLevel.MEDIUM,
                implementation_type="manual",
                implementation_steps=[
                    "Analyze opportunity in detail",
                    "Create implementation plan",
                    "Execute improvement",
                    "Measure results",
                ],
                validation_criteria=["Opportunity successfully addressed"],
                estimated_effort_hours=4.0,
                expected_impact="Medium - team performance improvement",
                success_metrics=["Measurable performance improvement"],
            )
            action_points.append(action)
        
        return action_points
    
    async def _create_priority_matrix(self, action_points: List[ActionPoint]) -> PriorityMatrix:
        """Create impact/effort priority matrix for action points."""
        
        matrix = PriorityMatrix()
        
        for action in action_points:
            # Categorize by priority and effort
            high_impact = action.priority in [PriorityLevel.HIGH, PriorityLevel.CRITICAL]
            low_effort = action.estimated_effort_hours <= 4.0
            
            if high_impact and low_effort:
                matrix.high_impact_low_effort.append(action)
            elif high_impact and not low_effort:
                matrix.high_impact_high_effort.append(action)
            elif not high_impact and low_effort:
                matrix.low_impact_low_effort.append(action)
            else:
                matrix.low_impact_high_effort.append(action)
        
        return matrix
    
    async def _create_implementation_roadmap(
        self, action_points: List[ActionPoint], priority_matrix: PriorityMatrix
    ) -> ImplementationRoadmap:
        """Create sequenced implementation roadmap."""
        
        roadmap = ImplementationRoadmap()
        
        # Define implementation phases
        phase_1_actions = [a.action_id for a in priority_matrix.high_impact_low_effort]
        phase_2_actions = [a.action_id for a in priority_matrix.low_impact_low_effort]
        phase_3_actions = [a.action_id for a in priority_matrix.high_impact_high_effort]
        phase_4_actions = [a.action_id for a in priority_matrix.low_impact_high_effort]
        
        roadmap.phases = [
            {
                "name": "Phase 1: Quick Wins",
                "action_ids": phase_1_actions,
                "timeline": "0-2 weeks",
                "description": "High-impact, low-effort improvements",
            },
            {
                "name": "Phase 2: Easy Improvements",
                "action_ids": phase_2_actions,
                "timeline": "2-4 weeks",
                "description": "Low-risk improvements and optimizations",
            },
            {
                "name": "Phase 3: Strategic Initiatives",
                "action_ids": phase_3_actions,
                "timeline": "1-3 months",
                "description": "High-impact improvements requiring significant effort",
            },
            {
                "name": "Phase 4: Future Considerations",
                "action_ids": phase_4_actions,
                "timeline": "3+ months",
                "description": "Lower priority improvements for future consideration",
            },
        ]
        
        # Set critical path (Phase 1 + high-priority Phase 3)
        critical_actions = phase_1_actions + [
            a for a in phase_3_actions 
            if any(action.priority == PriorityLevel.CRITICAL for action in action_points if action.action_id == a)
        ]
        roadmap.critical_path = critical_actions
        
        # Estimate total duration
        total_effort = sum(a.estimated_effort_hours for a in action_points)
        roadmap.estimated_total_duration = f"{total_effort:.0f} hours ({total_effort/40:.1f} weeks)"
        
        return roadmap
    
    async def _generate_future_recommendations(
        self,
        report: ComprehensiveRetrospectiveReport,
        historical_data: Optional[List[ComprehensiveRetrospectiveReport]],
    ) -> None:
        """Generate recommendations for future tasks and improvements."""
        
        # Next task recommendations
        next_task_recs = []
        
        if report.overall_team_performance > 0.8:
            next_task_recs.append("Team is performing well - consider taking on more complex tasks")
            next_task_recs.append("Maintain current successful practices")
        elif report.overall_team_performance < 0.6:
            next_task_recs.append("Focus on simpler tasks while addressing performance issues")
            next_task_recs.append("Implement improvement actions before taking on complex work")
        
        if len(report.systemic_issues) > 3:
            next_task_recs.append("Address systemic issues before starting new major initiatives")
        
        report.next_task_recommendations = next_task_recs
        
        # Team optimization suggestions
        team_opts = []
        
        if report.cross_agent_insights.collaboration_effectiveness < 0.7:
            team_opts.append("Improve team communication and coordination protocols")
        
        if len(report.participating_agents) > 8:
            team_opts.append("Consider smaller team sizes for better coordination")
        
        # Analyze role performance
        role_performance = defaultdict(list)
        for retro_id in range(report.individual_retrospectives_count):
            # This would need access to individual retros for detailed analysis
            pass
        
        report.team_optimization_suggestions = team_opts
        
        # Process improvements
        process_improvements = []
        
        for improvement in report.systemic_improvements:
            if improvement.category == ImprovementCategory.PROCESS:
                process_improvements.append(f"Implement: {improvement.title}")
        
        report.process_improvements = process_improvements[:5]  # Top 5
    
    # Helper methods
    
    def _assess_issue_severity(self, occurrence_count: int, total_agents: int) -> str:
        """Assess severity of a systemic issue based on occurrence."""
        
        occurrence_rate = occurrence_count / total_agents
        
        if occurrence_rate >= 0.8:
            return "critical"
        elif occurrence_rate >= 0.6:
            return "high"
        elif occurrence_rate >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _categorize_systemic_issue(self, issue_description: str) -> str:
        """Categorize a systemic issue based on its description."""
        
        desc_lower = issue_description.lower()
        
        if any(word in desc_lower for word in ["performance", "speed", "efficiency"]):
            return "performance"
        elif any(word in desc_lower for word in ["communication", "coordination"]):
            return "coordination"
        elif any(word in desc_lower for word in ["tool", "technique", "method"]):
            return "tooling"
        elif any(word in desc_lower for word in ["process", "workflow"]):
            return "process"
        elif any(word in desc_lower for word in ["quality", "error", "validation"]):
            return "quality"
        else:
            return "general"
    
    async def _analyze_root_cause(
        self, improvement_area: str, retrospectives: List[IndividualRetrospective]
    ) -> str:
        """Analyze root cause of a recurring improvement area."""
        
        # Simple heuristic root cause analysis
        if "communication" in improvement_area.lower():
            return "Communication protocols may be unclear or insufficient"
        elif "efficiency" in improvement_area.lower():
            return "Current processes or tools may not be optimized for the task types"
        elif "quality" in improvement_area.lower():
            return "Quality assurance processes may need enhancement"
        else:
            return "Underlying process or training gap likely contributing to this issue"
    
    async def _suggest_solutions(self, improvement_area: str, severity: str) -> List[str]:
        """Suggest potential solutions for an improvement area."""
        
        solutions = []
        
        if "communication" in improvement_area.lower():
            solutions.extend([
                "Improve communication protocols",
                "Provide communication training",
                "Implement better coordination tools",
            ])
        elif "efficiency" in improvement_area.lower():
            solutions.extend([
                "Optimize current processes",
                "Provide efficiency training",
                "Upgrade tools and techniques",
            ])
        elif "quality" in improvement_area.lower():
            solutions.extend([
                "Enhance quality assurance processes",
                "Provide quality-focused training",
                "Implement better validation tools",
            ])
        else:
            solutions.extend([
                "Provide targeted training",
                "Update relevant processes",
                "Consider tool or technique improvements",
            ])
        
        # Add severity-specific solutions
        if severity in ["high", "critical"]:
            solutions.insert(0, "Immediate intervention required")
        
        return solutions[:3]  # Limit to top 3 solutions
    
    async def _analyze_challenge_root_cause(self, challenge_description: str) -> str:
        """Analyze root cause of a recurring challenge."""
        
        if "timeout" in challenge_description.lower():
            return "Performance or resource constraints causing timeouts"
        elif "error" in challenge_description.lower():
            return "Technical or process issues leading to errors"
        elif "coordination" in challenge_description.lower():
            return "Team coordination or communication breakdowns"
        else:
            return "Operational or systematic issue requiring investigation"
    
    async def _suggest_challenge_solutions(self, challenge_description: str) -> List[str]:
        """Suggest solutions for a recurring challenge."""
        
        solutions = []
        
        if "timeout" in challenge_description.lower():
            solutions.extend([
                "Increase timeout limits",
                "Optimize performance",
                "Implement better resource management",
            ])
        elif "error" in challenge_description.lower():
            solutions.extend([
                "Improve error handling",
                "Add validation checks",
                "Enhance debugging capabilities",
            ])
        else:
            solutions.extend([
                "Investigate root cause",
                "Implement preventive measures",
                "Provide additional training",
            ])
        
        return solutions
    
    def _estimate_effort(self, severity: str) -> str:
        """Estimate effort required based on issue severity."""
        
        effort_map = {
            "critical": "High - requires immediate significant resources",
            "high": "Medium-High - substantial effort required", 
            "medium": "Medium - moderate effort required",
            "low": "Low - minimal effort required",
        }
        
        return effort_map.get(severity, "Medium - moderate effort required")
    
    def _determine_implementation_type(self, improvement: SystemicImprovement) -> str:
        """Determine implementation type based on improvement characteristics."""
        
        if improvement.category in [ImprovementCategory.PERFORMANCE, ImprovementCategory.TOOLING]:
            return "automatic"
        elif improvement.category in [ImprovementCategory.PROCESS, ImprovementCategory.COORDINATION]:
            return "configuration"
        else:
            return "manual"
    
    def _break_down_implementation(self, improvement: SystemicImprovement) -> List[str]:
        """Break down implementation into specific steps."""
        
        steps = improvement.implementation_approach.split("; ") if improvement.implementation_approach else []
        
        if not steps:
            steps = [
                "Analyze current state",
                "Design improvement approach",
                "Implement changes",
                "Validate results",
                "Document changes",
            ]
        
        return steps
    
    def _estimate_effort_hours(self, effort_assessment: str) -> float:
        """Estimate effort hours from text assessment."""
        
        if "high" in effort_assessment.lower():
            return 16.0
        elif "medium" in effort_assessment.lower():
            return 8.0
        elif "low" in effort_assessment.lower():
            return 2.0
        else:
            return 4.0  # Default