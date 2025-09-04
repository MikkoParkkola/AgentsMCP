"""
Analysis Reporting System

Comprehensive reporting system for retrospective analysis results,
providing structured insights and actionable recommendations.
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64

from .retrospective_analyzer import AnalysisResult, AnalysisScope
from .pattern_detection import DetectedPattern
from .bottleneck_identification import Bottleneck
from ..generation.improvement_generator import ImprovementOpportunity


class ReportFormat(Enum):
    """Available report formats."""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    CSV = "csv"


class ReportSection(Enum):
    """Report sections."""
    EXECUTIVE_SUMMARY = "executive_summary"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    WORKFLOW_ANALYSIS = "workflow_analysis"
    QUALITY_ANALYSIS = "quality_analysis"
    USER_EXPERIENCE_ANALYSIS = "user_experience_analysis"
    PATTERNS_IDENTIFIED = "patterns_identified"
    BOTTLENECKS_IDENTIFIED = "bottlenecks_identified"
    IMPROVEMENT_OPPORTUNITIES = "improvement_opportunities"
    RECOMMENDATIONS = "recommendations"
    APPENDIX = "appendix"


class AnalysisReporter:
    """
    Comprehensive reporting system for retrospective analysis results.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def generate_comprehensive_report(
        self,
        analysis_result: AnalysisResult,
        patterns: List[DetectedPattern] = None,
        bottlenecks: List[Bottleneck] = None,
        improvements: List[ImprovementOpportunity] = None,
        format: ReportFormat = ReportFormat.JSON,
        sections: List[ReportSection] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.
        
        Args:
            analysis_result: Results from retrospective analysis
            patterns: Detected patterns
            bottlenecks: Identified bottlenecks
            improvements: Generated improvement opportunities
            format: Output format for the report
            sections: Specific sections to include (all if None)
            
        Returns:
            Comprehensive report in specified format
        """
        try:
            self.logger.info(f"Generating comprehensive report for session {analysis_result.session_id}")
            
            if sections is None:
                sections = list(ReportSection)
            
            report_data = {}
            
            # Generate each requested section
            for section in sections:
                if section == ReportSection.EXECUTIVE_SUMMARY:
                    report_data['executive_summary'] = await self._generate_executive_summary(
                        analysis_result, patterns, bottlenecks, improvements
                    )
                
                elif section == ReportSection.PERFORMANCE_ANALYSIS:
                    report_data['performance_analysis'] = await self._generate_performance_analysis(
                        analysis_result
                    )
                
                elif section == ReportSection.WORKFLOW_ANALYSIS:
                    report_data['workflow_analysis'] = await self._generate_workflow_analysis(
                        analysis_result
                    )
                
                elif section == ReportSection.QUALITY_ANALYSIS:
                    report_data['quality_analysis'] = await self._generate_quality_analysis(
                        analysis_result
                    )
                
                elif section == ReportSection.USER_EXPERIENCE_ANALYSIS:
                    report_data['user_experience_analysis'] = await self._generate_ux_analysis(
                        analysis_result
                    )
                
                elif section == ReportSection.PATTERNS_IDENTIFIED:
                    report_data['patterns_identified'] = await self._generate_patterns_section(
                        patterns or []
                    )
                
                elif section == ReportSection.BOTTLENECKS_IDENTIFIED:
                    report_data['bottlenecks_identified'] = await self._generate_bottlenecks_section(
                        bottlenecks or []
                    )
                
                elif section == ReportSection.IMPROVEMENT_OPPORTUNITIES:
                    report_data['improvement_opportunities'] = await self._generate_improvements_section(
                        improvements or []
                    )
                
                elif section == ReportSection.RECOMMENDATIONS:
                    report_data['recommendations'] = await self._generate_recommendations_section(
                        analysis_result, patterns, bottlenecks, improvements
                    )
                
                elif section == ReportSection.APPENDIX:
                    report_data['appendix'] = await self._generate_appendix_section(
                        analysis_result
                    )
            
            # Add metadata
            report_data['metadata'] = {
                'generated_at': datetime.utcnow().isoformat(),
                'session_id': analysis_result.session_id,
                'analysis_timestamp': analysis_result.analysis_timestamp.isoformat(),
                'report_format': format.value,
                'sections_included': [s.value for s in sections]
            }
            
            # Format the report
            formatted_report = await self._format_report(report_data, format)
            
            self.logger.info("Comprehensive report generation completed")
            return formatted_report
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {'error': str(e)}
    
    async def _generate_executive_summary(
        self,
        analysis_result: AnalysisResult,
        patterns: List[DetectedPattern] = None,
        bottlenecks: List[Bottleneck] = None,
        improvements: List[ImprovementOpportunity] = None
    ) -> Dict[str, Any]:
        """Generate executive summary section."""
        patterns = patterns or []
        bottlenecks = bottlenecks or []
        improvements = improvements or []
        
        # Calculate key metrics
        total_events = analysis_result.events_processed
        pattern_accuracy = analysis_result.pattern_accuracy
        confidence_score = analysis_result.confidence_score
        
        # Categorize findings by severity/priority
        critical_issues = []
        high_priority_improvements = []
        
        for pattern in patterns:
            if pattern.severity.value in ['critical', 'high']:
                critical_issues.append(f"Pattern: {pattern.description}")
        
        for bottleneck in bottlenecks:
            if bottleneck.severity.value in ['critical', 'high']:
                critical_issues.append(f"Bottleneck: {bottleneck.description}")
        
        for improvement in improvements:
            if improvement.priority_score > 8.0:
                high_priority_improvements.append(improvement.title)
        
        # Calculate potential impact
        total_potential_benefits = {}
        for improvement in improvements:
            for benefit_type, value in improvement.expected_benefits.items():
                if benefit_type not in total_potential_benefits:
                    total_potential_benefits[benefit_type] = []
                total_potential_benefits[benefit_type].append(value)
        
        # Average the benefits (simplified aggregation)
        aggregated_benefits = {}
        for benefit_type, values in total_potential_benefits.items():
            if values:
                aggregated_benefits[benefit_type] = sum(values) / len(values)
        
        return {
            'analysis_overview': {
                'events_analyzed': total_events,
                'analysis_confidence': confidence_score,
                'pattern_detection_accuracy': pattern_accuracy,
                'data_completeness': analysis_result.data_completeness,
                'processing_time_ms': analysis_result.processing_time_ms
            },
            'key_findings': {
                'patterns_detected': len(patterns),
                'bottlenecks_identified': len(bottlenecks),
                'improvement_opportunities': len(improvements),
                'critical_issues_count': len(critical_issues),
                'high_priority_improvements_count': len(high_priority_improvements)
            },
            'critical_issues': critical_issues[:5],  # Top 5 critical issues
            'top_improvement_opportunities': high_priority_improvements[:5],  # Top 5 improvements
            'potential_impact_summary': aggregated_benefits,
            'overall_assessment': self._generate_overall_assessment(
                analysis_result, len(critical_issues), len(improvements)
            )
        }
    
    async def _generate_performance_analysis(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Generate performance analysis section."""
        perf_insights = analysis_result.performance_insights
        
        if not perf_insights or perf_insights.get('status') == 'no_performance_data':
            return {
                'status': 'insufficient_data',
                'message': 'Insufficient performance data for analysis'
            }
        
        response_times = perf_insights.get('response_times', {})
        token_usage = perf_insights.get('token_usage', {})
        memory_usage = perf_insights.get('memory_usage', {})
        
        analysis = {
            'response_time_analysis': {
                'metrics': response_times,
                'assessment': self._assess_response_times(response_times),
                'trends': self._analyze_response_time_trends(perf_insights)
            },
            'resource_utilization': {
                'token_usage': token_usage,
                'memory_usage': memory_usage,
                'efficiency_score': self._calculate_efficiency_score(token_usage, memory_usage)
            },
            'performance_bottlenecks': self._identify_performance_bottlenecks(perf_insights),
            'optimization_opportunities': self._identify_performance_optimizations(perf_insights)
        }
        
        return analysis
    
    async def _generate_workflow_analysis(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Generate workflow analysis section."""
        workflow_insights = analysis_result.workflow_insights
        
        if not workflow_insights or workflow_insights.get('status') == 'no_workflow_data':
            return {
                'status': 'insufficient_data',
                'message': 'Insufficient workflow data for analysis'
            }
        
        return {
            'delegation_patterns': {
                'total_delegations': workflow_insights.get('total_delegations', 0),
                'agent_distribution': workflow_insights.get('agent_distribution', {}),
                'success_rates': workflow_insights.get('agent_success_rates', {}),
                'efficiency_assessment': self._assess_delegation_efficiency(workflow_insights)
            },
            'task_completion_analysis': {
                'completion_rate': workflow_insights.get('completion_rate', 0),
                'task_starts': workflow_insights.get('task_starts', 0),
                'task_completions': workflow_insights.get('task_completions', 0),
                'completion_assessment': self._assess_completion_rates(workflow_insights)
            },
            'workflow_optimization_opportunities': self._identify_workflow_optimizations(workflow_insights)
        }
    
    async def _generate_quality_analysis(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Generate quality analysis section."""
        quality_insights = analysis_result.quality_insights
        
        if not quality_insights or quality_insights.get('status') == 'no_quality_data':
            return {
                'status': 'insufficient_data',
                'message': 'Insufficient quality data for analysis'
            }
        
        success_rate = quality_insights.get('success_rate', 0)
        failure_rate = quality_insights.get('failure_rate', 0)
        quality_gate_pass_rate = quality_insights.get('quality_gate_pass_rate', 0)
        
        return {
            'success_metrics': {
                'success_rate': success_rate,
                'failure_rate': failure_rate,
                'quality_gate_pass_rate': quality_gate_pass_rate,
                'quality_score': self._calculate_quality_score(quality_insights)
            },
            'quality_assessment': self._assess_quality_levels(quality_insights),
            'quality_trends': self._analyze_quality_trends(quality_insights),
            'improvement_recommendations': self._recommend_quality_improvements(quality_insights)
        }
    
    async def _generate_ux_analysis(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Generate user experience analysis section."""
        ux_insights = analysis_result.user_experience_insights
        
        if not ux_insights or ux_insights.get('status') == 'no_ux_data':
            return {
                'status': 'insufficient_data',
                'message': 'Insufficient user experience data for analysis'
            }
        
        return {
            'interaction_analysis': {
                'total_interactions': ux_insights.get('total_interactions', 0),
                'interaction_distribution': ux_insights.get('interaction_distribution', {}),
                'interaction_patterns': self._analyze_interaction_patterns(ux_insights)
            },
            'error_analysis': {
                'error_rate': ux_insights.get('error_rate', 0),
                'error_distribution': ux_insights.get('error_distribution', {}),
                'error_impact_assessment': self._assess_error_impact(ux_insights)
            },
            'user_satisfaction_indicators': self._calculate_satisfaction_indicators(ux_insights),
            'ux_improvement_opportunities': self._identify_ux_improvements(ux_insights)
        }
    
    async def _generate_patterns_section(self, patterns: List[DetectedPattern]) -> Dict[str, Any]:
        """Generate patterns identified section."""
        if not patterns:
            return {
                'status': 'no_patterns',
                'message': 'No significant patterns detected'
            }
        
        # Group patterns by type and severity
        by_type = defaultdict(list)
        by_severity = defaultdict(list)
        
        for pattern in patterns:
            by_type[pattern.pattern_type.value].append(pattern)
            by_severity[pattern.severity.value].append(pattern)
        
        # Convert patterns to serializable format
        patterns_data = []
        for pattern in patterns:
            pattern_data = {
                'pattern_id': pattern.pattern_id,
                'type': pattern.pattern_type.value,
                'severity': pattern.severity.value,
                'description': pattern.description,
                'confidence': pattern.confidence,
                'frequency': pattern.frequency,
                'evidence': pattern.evidence,
                'suggested_action': pattern.suggested_action,
                'impact_estimate': pattern.impact_estimate
            }
            patterns_data.append(pattern_data)
        
        return {
            'total_patterns': len(patterns),
            'patterns_by_type': {ptype: len(plist) for ptype, plist in by_type.items()},
            'patterns_by_severity': {sev: len(plist) for sev, plist in by_severity.items()},
            'detailed_patterns': patterns_data,
            'pattern_summary': self._summarize_patterns(patterns)
        }
    
    async def _generate_bottlenecks_section(self, bottlenecks: List[Bottleneck]) -> Dict[str, Any]:
        """Generate bottlenecks identified section."""
        if not bottlenecks:
            return {
                'status': 'no_bottlenecks',
                'message': 'No significant bottlenecks identified'
            }
        
        # Group bottlenecks by type and severity
        by_type = defaultdict(list)
        by_severity = defaultdict(list)
        
        for bottleneck in bottlenecks:
            by_type[bottleneck.bottleneck_type.value].append(bottleneck)
            by_severity[bottleneck.severity.value].append(bottleneck)
        
        # Convert bottlenecks to serializable format
        bottlenecks_data = []
        for bottleneck in bottlenecks:
            bottleneck_data = {
                'bottleneck_id': bottleneck.bottleneck_id,
                'type': bottleneck.bottleneck_type.value,
                'severity': bottleneck.severity.value,
                'description': bottleneck.description,
                'location': bottleneck.location,
                'impact_metrics': bottleneck.impact_metrics,
                'root_cause': bottleneck.root_cause,
                'suggested_remediation': bottleneck.suggested_remediation,
                'estimated_improvement': bottleneck.estimated_improvement,
                'confidence': bottleneck.confidence
            }
            bottlenecks_data.append(bottleneck_data)
        
        return {
            'total_bottlenecks': len(bottlenecks),
            'bottlenecks_by_type': {btype: len(blist) for btype, blist in by_type.items()},
            'bottlenecks_by_severity': {sev: len(blist) for sev, blist in by_severity.items()},
            'detailed_bottlenecks': bottlenecks_data,
            'bottleneck_summary': self._summarize_bottlenecks(bottlenecks)
        }
    
    async def _generate_improvements_section(self, improvements: List[ImprovementOpportunity]) -> Dict[str, Any]:
        """Generate improvement opportunities section."""
        if not improvements:
            return {
                'status': 'no_improvements',
                'message': 'No improvement opportunities identified'
            }
        
        # Convert improvements to serializable format
        improvements_data = []
        for improvement in improvements:
            improvement_data = {
                'opportunity_id': improvement.opportunity_id,
                'title': improvement.title,
                'description': improvement.description,
                'type': improvement.improvement_type.value,
                'effort': improvement.effort.value,
                'risk': improvement.risk.value,
                'expected_benefits': improvement.expected_benefits,
                'confidence': improvement.confidence,
                'priority_score': improvement.priority_score,
                'implementation_steps': improvement.implementation_steps,
                'success_metrics': improvement.success_metrics,
                'affected_components': improvement.affected_components
            }
            improvements_data.append(improvement_data)
        
        # Sort by priority
        improvements_data.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return {
            'total_opportunities': len(improvements),
            'high_priority_count': len([i for i in improvements if i.priority_score > 8.0]),
            'medium_priority_count': len([i for i in improvements if 6.0 <= i.priority_score <= 8.0]),
            'low_priority_count': len([i for i in improvements if i.priority_score < 6.0]),
            'detailed_opportunities': improvements_data,
            'implementation_roadmap': self._create_implementation_roadmap(improvements)
        }
    
    async def _generate_recommendations_section(
        self,
        analysis_result: AnalysisResult,
        patterns: List[DetectedPattern] = None,
        bottlenecks: List[Bottleneck] = None,
        improvements: List[ImprovementOpportunity] = None
    ) -> Dict[str, Any]:
        """Generate recommendations section."""
        patterns = patterns or []
        bottlenecks = bottlenecks or []
        improvements = improvements or []
        
        # Immediate actions (high priority, low effort)
        immediate_actions = [
            imp for imp in improvements
            if imp.priority_score > 8.0 and imp.effort.value in ['automatic', 'minimal', 'low']
        ]
        
        # Short-term goals (medium priority/effort)
        short_term_goals = [
            imp for imp in improvements
            if 6.0 <= imp.priority_score <= 8.0 and imp.effort.value in ['low', 'medium']
        ]
        
        # Long-term strategic improvements
        long_term_strategic = [
            imp for imp in improvements
            if imp.effort.value in ['high', 'complex']
        ]
        
        return {
            'immediate_actions': {
                'count': len(immediate_actions),
                'items': [{'title': imp.title, 'effort': imp.effort.value} for imp in immediate_actions[:5]]
            },
            'short_term_goals': {
                'count': len(short_term_goals),
                'items': [{'title': imp.title, 'effort': imp.effort.value} for imp in short_term_goals[:5]]
            },
            'long_term_strategic': {
                'count': len(long_term_strategic),
                'items': [{'title': imp.title, 'effort': imp.effort.value} for imp in long_term_strategic[:3]]
            },
            'prioritization_guidance': self._create_prioritization_guidance(analysis_result, improvements),
            'success_measurement': self._recommend_success_metrics(improvements)
        }
    
    async def _generate_appendix_section(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Generate appendix section with technical details."""
        return {
            'analysis_configuration': {
                'depth': analysis_result.configuration.depth.value,
                'scopes': [scope.value for scope in analysis_result.configuration.scopes],
                'time_window_hours': analysis_result.configuration.time_window.total_seconds() / 3600,
                'accuracy_threshold': analysis_result.configuration.pattern_accuracy_threshold
            },
            'data_sources': {
                'events_processed': analysis_result.events_processed,
                'data_completeness': analysis_result.data_completeness,
                'processing_time_ms': analysis_result.processing_time_ms
            },
            'methodology': {
                'pattern_detection': 'Statistical analysis with clustering and correlation',
                'bottleneck_identification': 'Workflow analysis and queuing theory',
                'impact_estimation': 'Historical data and statistical modeling'
            },
            'limitations': [
                'Analysis limited to captured event data',
                'Some patterns may require longer observation periods',
                'Impact estimates based on historical averages'
            ]
        }
    
    async def _format_report(self, report_data: Dict[str, Any], format: ReportFormat) -> Dict[str, Any]:
        """Format report according to specified format."""
        if format == ReportFormat.JSON:
            return report_data
        
        elif format == ReportFormat.MARKDOWN:
            return {'markdown': self._convert_to_markdown(report_data)}
        
        elif format == ReportFormat.HTML:
            return {'html': self._convert_to_html(report_data)}
        
        else:
            # For other formats, return JSON for now
            return report_data
    
    def _generate_overall_assessment(
        self,
        analysis_result: AnalysisResult,
        critical_issues_count: int,
        improvements_count: int
    ) -> str:
        """Generate overall system assessment."""
        confidence = analysis_result.confidence_score
        
        if critical_issues_count > 5:
            return "CRITICAL: Multiple high-severity issues detected requiring immediate attention"
        elif critical_issues_count > 2:
            return "HIGH PRIORITY: Several important issues identified with good improvement potential"
        elif confidence > 0.8 and improvements_count > 5:
            return "GOOD: System shows healthy operation with clear optimization opportunities"
        elif confidence < 0.5:
            return "LIMITED ANALYSIS: Insufficient data for comprehensive assessment"
        else:
            return "STABLE: System operating well with minor optimization opportunities"
    
    def _assess_response_times(self, response_times: Dict[str, Any]) -> str:
        """Assess response time performance."""
        avg_ms = response_times.get('avg_ms', 0)
        p95_ms = response_times.get('p95_ms', 0)
        
        if avg_ms > 5000 or p95_ms > 10000:
            return "POOR: Response times significantly above acceptable thresholds"
        elif avg_ms > 2000 or p95_ms > 5000:
            return "NEEDS IMPROVEMENT: Response times higher than optimal"
        elif avg_ms < 1000 and p95_ms < 2000:
            return "EXCELLENT: Response times well within acceptable ranges"
        else:
            return "GOOD: Response times acceptable with room for optimization"
    
    def _calculate_efficiency_score(
        self,
        token_usage: Dict[str, Any],
        memory_usage: Dict[str, Any]
    ) -> float:
        """Calculate overall efficiency score."""
        # Simplified efficiency calculation
        score = 5.0  # Base score
        
        # Token efficiency (lower usage per task is better)
        avg_tokens = token_usage.get('avg_tokens', 1000)
        if avg_tokens < 500:
            score += 2.0
        elif avg_tokens > 2000:
            score -= 2.0
        
        # Memory efficiency
        avg_memory = memory_usage.get('avg_mb', 100)
        if avg_memory < 50:
            score += 1.0
        elif avg_memory > 200:
            score -= 1.0
        
        return max(0.0, min(10.0, score))
    
    def _create_implementation_roadmap(self, improvements: List[ImprovementOpportunity]) -> Dict[str, List[str]]:
        """Create implementation roadmap for improvements."""
        roadmap = {
            'phase_1_immediate': [],
            'phase_2_short_term': [],
            'phase_3_long_term': []
        }
        
        for improvement in improvements:
            if improvement.effort.value in ['automatic', 'minimal']:
                roadmap['phase_1_immediate'].append(improvement.title)
            elif improvement.effort.value in ['low', 'medium']:
                roadmap['phase_2_short_term'].append(improvement.title)
            else:
                roadmap['phase_3_long_term'].append(improvement.title)
        
        return roadmap
    
    def _convert_to_markdown(self, report_data: Dict[str, Any]) -> str:
        """Convert report data to Markdown format."""
        markdown = "# Retrospective Analysis Report\n\n"
        
        if 'executive_summary' in report_data:
            markdown += "## Executive Summary\n\n"
            summary = report_data['executive_summary']
            
            if 'analysis_overview' in summary:
                overview = summary['analysis_overview']
                markdown += f"- **Events Analyzed**: {overview.get('events_analyzed', 0)}\n"
                markdown += f"- **Analysis Confidence**: {overview.get('analysis_confidence', 0):.2f}\n"
                markdown += f"- **Pattern Accuracy**: {overview.get('pattern_detection_accuracy', 0):.2f}\n\n"
            
            if 'key_findings' in summary:
                findings = summary['key_findings']
                markdown += "### Key Findings\n"
                markdown += f"- Patterns Detected: {findings.get('patterns_detected', 0)}\n"
                markdown += f"- Bottlenecks Identified: {findings.get('bottlenecks_identified', 0)}\n"
                markdown += f"- Improvement Opportunities: {findings.get('improvement_opportunities', 0)}\n\n"
        
        # Add other sections as needed
        return markdown
    
    def _convert_to_html(self, report_data: Dict[str, Any]) -> str:
        """Convert report data to HTML format."""
        html = "<html><head><title>Retrospective Analysis Report</title></head><body>"
        html += "<h1>Retrospective Analysis Report</h1>"
        
        # Add HTML content based on report_data
        # This is a simplified version
        html += "<p>Report generated successfully</p>"
        
        html += "</body></html>"
        return html
    
    # Helper methods for analysis assessments
    def _analyze_response_time_trends(self, perf_insights: Dict[str, Any]) -> str:
        if perf_insights.get('performance_degradation', {}).get('detected'):
            return "DEGRADING: Performance degradation detected over time"
        return "STABLE: No significant performance trends detected"
    
    def _identify_performance_bottlenecks(self, perf_insights: Dict[str, Any]) -> List[str]:
        bottlenecks = []
        if perf_insights.get('response_times', {}).get('p95_ms', 0) > 5000:
            bottlenecks.append("High response time variance")
        if perf_insights.get('memory_usage', {}).get('peak_mb', 0) > 500:
            bottlenecks.append("High memory usage")
        return bottlenecks
    
    def _identify_performance_optimizations(self, perf_insights: Dict[str, Any]) -> List[str]:
        optimizations = []
        if perf_insights.get('response_times', {}).get('avg_ms', 0) > 2000:
            optimizations.append("Response time optimization")
        if perf_insights.get('token_usage', {}).get('avg_tokens', 0) > 1500:
            optimizations.append("Token usage optimization")
        return optimizations
    
    def _assess_delegation_efficiency(self, workflow_insights: Dict[str, Any]) -> str:
        success_rates = workflow_insights.get('agent_success_rates', {})
        if not success_rates:
            return "UNKNOWN: No delegation data available"
        
        avg_success = sum(success_rates.values()) / len(success_rates)
        if avg_success > 0.9:
            return "EXCELLENT: High delegation success rates"
        elif avg_success > 0.7:
            return "GOOD: Acceptable delegation performance"
        else:
            return "NEEDS IMPROVEMENT: Low delegation success rates"
    
    def _assess_completion_rates(self, workflow_insights: Dict[str, Any]) -> str:
        completion_rate = workflow_insights.get('completion_rate', 0)
        if completion_rate > 0.95:
            return "EXCELLENT: Very high task completion rate"
        elif completion_rate > 0.85:
            return "GOOD: High task completion rate"
        elif completion_rate > 0.70:
            return "ACCEPTABLE: Moderate task completion rate"
        else:
            return "POOR: Low task completion rate needs attention"
    
    def _identify_workflow_optimizations(self, workflow_insights: Dict[str, Any]) -> List[str]:
        optimizations = []
        if workflow_insights.get('completion_rate', 0) < 0.9:
            optimizations.append("Improve task completion rates")
        
        success_rates = workflow_insights.get('agent_success_rates', {})
        if success_rates and min(success_rates.values()) < 0.8:
            optimizations.append("Optimize agent delegation patterns")
            
        return optimizations
    
    def _calculate_quality_score(self, quality_insights: Dict[str, Any]) -> float:
        success_rate = quality_insights.get('success_rate', 0)
        quality_gate_pass_rate = quality_insights.get('quality_gate_pass_rate', 0)
        
        # Weighted average of quality metrics
        score = (success_rate * 0.7 + quality_gate_pass_rate * 0.3) * 10
        return min(10.0, max(0.0, score))
    
    def _assess_quality_levels(self, quality_insights: Dict[str, Any]) -> str:
        quality_score = self._calculate_quality_score(quality_insights)
        if quality_score > 9:
            return "EXCELLENT: Very high quality metrics"
        elif quality_score > 7:
            return "GOOD: High quality with minor issues"
        elif quality_score > 5:
            return "ACCEPTABLE: Moderate quality levels"
        else:
            return "POOR: Quality issues need immediate attention"
    
    def _analyze_quality_trends(self, quality_insights: Dict[str, Any]) -> str:
        # Simplified - would need historical data for real trend analysis
        return "STABLE: Quality metrics appear stable"
    
    def _recommend_quality_improvements(self, quality_insights: Dict[str, Any]) -> List[str]:
        recommendations = []
        success_rate = quality_insights.get('success_rate', 1.0)
        
        if success_rate < 0.95:
            recommendations.append("Implement comprehensive error handling")
        if quality_insights.get('quality_gate_pass_rate', 1.0) < 0.9:
            recommendations.append("Strengthen quality gate validation")
            
        return recommendations
    
    def _analyze_interaction_patterns(self, ux_insights: Dict[str, Any]) -> str:
        total_interactions = ux_insights.get('total_interactions', 0)
        if total_interactions > 1000:
            return "HIGH ACTIVITY: High user engagement levels"
        elif total_interactions > 100:
            return "MODERATE ACTIVITY: Regular user engagement"
        else:
            return "LOW ACTIVITY: Limited user interaction data"
    
    def _assess_error_impact(self, ux_insights: Dict[str, Any]) -> str:
        error_rate = ux_insights.get('error_rate', 0)
        if error_rate > 0.1:
            return "HIGH IMPACT: Error rate significantly affecting users"
        elif error_rate > 0.05:
            return "MODERATE IMPACT: Error rate above optimal levels"
        else:
            return "LOW IMPACT: Error rate within acceptable range"
    
    def _calculate_satisfaction_indicators(self, ux_insights: Dict[str, Any]) -> Dict[str, Any]:
        error_rate = ux_insights.get('error_rate', 0)
        total_interactions = ux_insights.get('total_interactions', 0)
        
        # Simplified satisfaction calculation
        satisfaction_score = max(0, min(10, 10 * (1 - error_rate * 10)))
        
        return {
            'estimated_satisfaction_score': satisfaction_score,
            'user_friction_indicators': {
                'error_rate': error_rate,
                'total_interactions': total_interactions
            }
        }
    
    def _identify_ux_improvements(self, ux_insights: Dict[str, Any]) -> List[str]:
        improvements = []
        if ux_insights.get('error_rate', 0) > 0.05:
            improvements.append("Improve input validation and error handling")
        if ux_insights.get('total_interactions', 0) < 100:
            improvements.append("Enhance user engagement and interaction flows")
        return improvements
    
    def _summarize_patterns(self, patterns: List[DetectedPattern]) -> str:
        if not patterns:
            return "No patterns detected"
        
        critical_count = len([p for p in patterns if p.severity.value == 'critical'])
        high_count = len([p for p in patterns if p.severity.value == 'high'])
        
        if critical_count > 0:
            return f"CRITICAL: {critical_count} critical patterns require immediate attention"
        elif high_count > 0:
            return f"HIGH PRIORITY: {high_count} high-severity patterns identified"
        else:
            return f"MONITORING: {len(patterns)} patterns detected for optimization"
    
    def _summarize_bottlenecks(self, bottlenecks: List[Bottleneck]) -> str:
        if not bottlenecks:
            return "No bottlenecks detected"
        
        critical_count = len([b for b in bottlenecks if b.severity.value == 'critical'])
        high_count = len([b for b in bottlenecks if b.severity.value == 'high'])
        
        if critical_count > 0:
            return f"CRITICAL: {critical_count} critical bottlenecks impacting performance"
        elif high_count > 0:
            return f"HIGH IMPACT: {high_count} significant bottlenecks identified"
        else:
            return f"OPTIMIZATION: {len(bottlenecks)} bottlenecks found for improvement"
    
    def _create_prioritization_guidance(
        self,
        analysis_result: AnalysisResult,
        improvements: List[ImprovementOpportunity]
    ) -> Dict[str, Any]:
        high_priority = [i for i in improvements if i.priority_score > 8.0]
        low_effort = [i for i in improvements if i.effort.value in ['automatic', 'minimal', 'low']]
        
        return {
            'quick_wins': len([i for i in improvements if i.priority_score > 7.0 and i.effort.value in ['automatic', 'minimal']]),
            'high_impact_projects': len(high_priority),
            'resource_requirements': {
                'immediate_capacity_needed': len([i for i in improvements if i.effort.value == 'automatic']),
                'short_term_planning_needed': len([i for i in improvements if i.effort.value in ['minimal', 'low']]),
                'long_term_investment_needed': len([i for i in improvements if i.effort.value in ['high', 'complex']])
            }
        }
    
    def _recommend_success_metrics(self, improvements: List[ImprovementOpportunity]) -> List[str]:
        all_metrics = set()
        for improvement in improvements:
            all_metrics.update(improvement.success_metrics)
        
        return list(all_metrics)[:10]  # Top 10 most common metrics