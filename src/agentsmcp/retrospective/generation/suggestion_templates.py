"""
Suggestion Templates

Template-based suggestions for common performance issues and improvement patterns.
Provides standardized improvement suggestions with proven effectiveness.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any


@dataclass
class SuggestionTemplate:
    """Template for generating improvement suggestions."""
    template_id: str
    category: str
    title_template: str
    description_template: str
    implementation_steps: List[str]
    expected_benefits: Dict[str, float]
    effort_level: str
    risk_level: str
    success_metrics: List[str]
    prerequisites: List[str] = None
    common_pitfalls: List[str] = None


class SuggestionCategory(Enum):
    """Categories of improvement suggestions."""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"
    USER_EXPERIENCE = "user_experience"
    COST_OPTIMIZATION = "cost_optimization"
    MONITORING = "monitoring"


class SuggestionTemplates:
    """
    Repository of proven suggestion templates for common improvement scenarios.
    """
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, SuggestionTemplate]:
        """Initialize the library of suggestion templates."""
        templates = {}
        
        # Performance optimization templates
        templates.update(self._create_performance_templates())
        
        # Reliability improvement templates
        templates.update(self._create_reliability_templates())
        
        # Scalability templates
        templates.update(self._create_scalability_templates())
        
        # User experience templates
        templates.update(self._create_ux_templates())
        
        # Cost optimization templates
        templates.update(self._create_cost_templates())
        
        # Monitoring templates
        templates.update(self._create_monitoring_templates())
        
        return templates
    
    def _create_performance_templates(self) -> Dict[str, SuggestionTemplate]:
        """Create performance optimization suggestion templates."""
        return {
            "response_time_optimization": SuggestionTemplate(
                template_id="response_time_optimization",
                category="performance",
                title_template="Optimize Response Times (Currently {avg_response_time}ms)",
                description_template="Response times averaging {avg_response_time}ms with p95 at {p95_response_time}ms. Target reduction of {target_improvement}% through caching and algorithm optimization.",
                implementation_steps=[
                    "Profile slow operations to identify specific bottlenecks",
                    "Implement Redis caching for frequently accessed data",
                    "Optimize database queries with proper indexing",
                    "Add response time monitoring with alerts",
                    "Implement request batching where applicable"
                ],
                expected_benefits={
                    "response_time_improvement_percent": 25,
                    "user_satisfaction_increase": 0.15,
                    "throughput_increase_percent": 20
                },
                effort_level="medium",
                risk_level="low",
                success_metrics=[
                    "Average response time < {target_avg_ms}ms",
                    "95th percentile response time < {target_p95_ms}ms",
                    "Cache hit rate > 80%"
                ],
                prerequisites=[
                    "Response time monitoring in place",
                    "Database performance baseline established"
                ],
                common_pitfalls=[
                    "Over-caching leading to stale data",
                    "Not invalidating cache properly",
                    "Optimizing wrong bottlenecks"
                ]
            ),
            
            "context_window_optimization": SuggestionTemplate(
                template_id="context_window_optimization",
                category="performance",
                title_template="Optimize Context Window Usage ({current_utilization}% utilization)",
                description_template="Context window utilization at {current_utilization}% with {token_waste}% wasted tokens. Smart context management can reduce costs and improve response times.",
                implementation_steps=[
                    "Implement smart context compaction at {threshold}% utilization",
                    "Add context relevance scoring to prioritize important information",
                    "Implement sliding window for long conversations",
                    "Add context usage monitoring and alerts"
                ],
                expected_benefits={
                    "token_usage_reduction_percent": 15,
                    "cost_reduction_percent": 12,
                    "response_time_improvement_percent": 8
                },
                effort_level="low",
                risk_level="minimal",
                success_metrics=[
                    "Average token usage < {target_tokens}",
                    "Context utilization 70-80%",
                    "Response quality maintained"
                ]
            ),
            
            "memory_optimization": SuggestionTemplate(
                template_id="memory_optimization",
                category="performance",
                title_template="Optimize Memory Usage (Peak: {peak_memory}MB)",
                description_template="Memory usage peaking at {peak_memory}MB with average of {avg_memory}MB. Memory optimization can improve performance and reduce infrastructure costs.",
                implementation_steps=[
                    "Implement object pooling for frequently used objects",
                    "Optimize data structures for memory efficiency",
                    "Add memory profiling and monitoring",
                    "Implement lazy loading for large datasets",
                    "Configure garbage collection parameters"
                ],
                expected_benefits={
                    "memory_reduction_percent": 30,
                    "performance_improvement_percent": 15,
                    "cost_reduction_percent": 20
                },
                effort_level="medium",
                risk_level="medium",
                success_metrics=[
                    "Peak memory usage < {target_peak}MB",
                    "Average memory usage < {target_avg}MB",
                    "Memory allocation rate reduced by 25%"
                ]
            )
        }
    
    def _create_reliability_templates(self) -> Dict[str, SuggestionTemplate]:
        """Create reliability improvement suggestion templates."""
        return {
            "error_handling_improvement": SuggestionTemplate(
                template_id="error_handling_improvement",
                category="reliability",
                title_template="Improve Error Handling (Current success rate: {success_rate}%)",
                description_template="Task success rate at {success_rate}% with {error_count} errors in analysis period. Comprehensive error handling can improve reliability significantly.",
                implementation_steps=[
                    "Analyze error patterns and categorize failure modes",
                    "Implement retry mechanisms with exponential backoff",
                    "Add circuit breakers for external service calls",
                    "Improve input validation with clear error messages",
                    "Add comprehensive error logging and monitoring"
                ],
                expected_benefits={
                    "success_rate_improvement_percent": 15,
                    "error_reduction_percent": 60,
                    "user_satisfaction_increase": 0.20
                },
                effort_level="medium",
                risk_level="low",
                success_metrics=[
                    "Task success rate > 98%",
                    "Unhandled errors < 0.1%",
                    "Mean time to recovery < 2 minutes"
                ]
            ),
            
            "timeout_optimization": SuggestionTemplate(
                template_id="timeout_optimization",
                category="reliability",
                title_template="Optimize Timeout Handling ({timeout_errors} timeout errors)",
                description_template="Detected {timeout_errors} timeout errors. Optimizing timeout strategies can improve reliability and user experience.",
                implementation_steps=[
                    "Analyze timeout patterns and adjust timeout values",
                    "Implement progressive timeouts for different operation types",
                    "Add timeout monitoring and alerting",
                    "Implement partial result handling for long operations",
                    "Add user feedback during long-running operations"
                ],
                expected_benefits={
                    "timeout_error_reduction_percent": 80,
                    "user_experience_improvement": 0.25,
                    "completion_rate_improvement_percent": 10
                },
                effort_level="low",
                risk_level="low",
                success_metrics=[
                    "Timeout errors < 1%",
                    "Average operation completion rate > 95%",
                    "User abandonment rate < 5%"
                ]
            )
        }
    
    def _create_scalability_templates(self) -> Dict[str, SuggestionTemplate]:
        """Create scalability improvement suggestion templates."""
        return {
            "load_balancing_optimization": SuggestionTemplate(
                template_id="load_balancing_optimization",
                category="scalability",
                title_template="Optimize Load Distribution (Agent {overloaded_agent} handling {load_ratio}x average)",
                description_template="Agent {overloaded_agent} handling {load_percentage}% of requests. Better load distribution can improve performance and prevent bottlenecks.",
                implementation_steps=[
                    "Implement intelligent routing based on agent capabilities",
                    "Add dynamic load balancing with health checks",
                    "Implement request queuing with priority levels",
                    "Add agent performance monitoring",
                    "Configure auto-scaling based on load metrics"
                ],
                expected_benefits={
                    "throughput_improvement_percent": 25,
                    "response_time_improvement_percent": 20,
                    "reliability_improvement": 0.15
                },
                effort_level="high",
                risk_level="medium",
                success_metrics=[
                    "Load distribution variance < 20%",
                    "No single agent > 40% of total load",
                    "System throughput increased by 25%"
                ]
            ),
            
            "queue_management": SuggestionTemplate(
                template_id="queue_management",
                category="scalability",
                title_template="Implement Smart Queue Management ({queue_buildup_count} queue buildups detected)",
                description_template="Detected {queue_buildup_count} instances of queue buildup with max wait time of {max_wait_time}s. Smart queue management can prevent bottlenecks.",
                implementation_steps=[
                    "Implement priority queuing based on request urgency",
                    "Add queue monitoring with depth and wait time metrics",
                    "Implement back-pressure mechanisms",
                    "Add dynamic scaling based on queue depth",
                    "Implement request shedding for overload protection"
                ],
                expected_benefits={
                    "wait_time_reduction_percent": 40,
                    "throughput_improvement_percent": 30,
                    "system_stability_improvement": 0.25
                },
                effort_level="high",
                risk_level="medium",
                success_metrics=[
                    "Average queue wait time < 5s",
                    "Queue depth never exceeds 100 items",
                    "Request drop rate < 1%"
                ]
            )
        }
    
    def _create_ux_templates(self) -> Dict[str, SuggestionTemplate]:
        """Create user experience improvement suggestion templates."""
        return {
            "input_validation_improvement": SuggestionTemplate(
                template_id="input_validation_improvement",
                category="user_experience",
                title_template="Improve Input Validation (Error rate: {error_rate}%)",
                description_template="User input error rate at {error_rate}%. Better validation and user guidance can significantly improve experience.",
                implementation_steps=[
                    "Add real-time input validation with helpful error messages",
                    "Implement input suggestions and auto-completion",
                    "Add examples and format hints for complex inputs",
                    "Implement progressive disclosure for advanced features",
                    "Add input history and favorites for common inputs"
                ],
                expected_benefits={
                    "error_rate_reduction_percent": 70,
                    "user_satisfaction_increase": 0.30,
                    "task_completion_rate_improvement_percent": 15
                },
                effort_level="low",
                risk_level="minimal",
                success_metrics=[
                    "Input error rate < 2%",
                    "User satisfaction score > 4.5/5",
                    "First-attempt success rate > 90%"
                ]
            ),
            
            "feedback_system_enhancement": SuggestionTemplate(
                template_id="feedback_system_enhancement",
                category="user_experience",
                title_template="Enhance User Feedback System",
                description_template="Limited user feedback visibility detected. Enhanced feedback systems improve user confidence and task success rates.",
                implementation_steps=[
                    "Add progress indicators for long-running operations",
                    "Implement real-time status updates",
                    "Add success/failure notifications with details",
                    "Implement undo/redo functionality where applicable",
                    "Add contextual help and tips"
                ],
                expected_benefits={
                    "user_satisfaction_increase": 0.25,
                    "task_abandonment_reduction_percent": 30,
                    "support_request_reduction_percent": 20
                },
                effort_level="medium",
                risk_level="low",
                success_metrics=[
                    "User satisfaction score > 4.5/5",
                    "Task abandonment rate < 10%",
                    "Support tickets reduced by 20%"
                ]
            )
        }
    
    def _create_cost_templates(self) -> Dict[str, SuggestionTemplate]:
        """Create cost optimization suggestion templates."""
        return {
            "token_usage_optimization": SuggestionTemplate(
                template_id="token_usage_optimization",
                category="cost_optimization",
                title_template="Optimize Token Usage (Current: {avg_tokens} tokens/request)",
                description_template="Average token usage at {avg_tokens} tokens per request. Smart optimization can reduce costs by {estimated_savings}% while maintaining quality.",
                implementation_steps=[
                    "Implement smart prompt compression",
                    "Add response length optimization",
                    "Implement local processing for simple queries",
                    "Add token usage monitoring and budgets",
                    "Optimize context window management"
                ],
                expected_benefits={
                    "cost_reduction_percent": 25,
                    "token_efficiency_improvement_percent": 30,
                    "response_time_improvement_percent": 10
                },
                effort_level="medium",
                risk_level="low",
                success_metrics=[
                    "Average tokens per request < {target_tokens}",
                    "Monthly token costs reduced by 25%",
                    "Response quality maintained (>4/5 rating)"
                ]
            ),
            
            "resource_right_sizing": SuggestionTemplate(
                template_id="resource_right_sizing",
                category="cost_optimization",
                title_template="Right-size Resource Allocation (Utilization: {utilization}%)",
                description_template="Resource utilization at {utilization}%. Right-sizing can optimize costs while maintaining performance.",
                implementation_steps=[
                    "Analyze resource usage patterns over time",
                    "Implement dynamic scaling based on demand",
                    "Add resource utilization monitoring",
                    "Optimize resource allocation algorithms",
                    "Implement cost tracking and budgets"
                ],
                expected_benefits={
                    "cost_reduction_percent": 20,
                    "resource_efficiency_improvement_percent": 35,
                    "operational_overhead_reduction_percent": 15
                },
                effort_level="high",
                risk_level="medium",
                success_metrics=[
                    "Resource utilization 75-85%",
                    "Infrastructure costs reduced by 20%",
                    "Performance targets maintained"
                ]
            )
        }
    
    def _create_monitoring_templates(self) -> Dict[str, SuggestionTemplate]:
        """Create monitoring improvement suggestion templates."""
        return {
            "performance_monitoring_enhancement": SuggestionTemplate(
                template_id="performance_monitoring_enhancement",
                category="monitoring",
                title_template="Enhance Performance Monitoring",
                description_template="Limited performance visibility detected. Enhanced monitoring enables proactive optimization and faster issue resolution.",
                implementation_steps=[
                    "Add comprehensive performance metrics collection",
                    "Implement real-time performance dashboards",
                    "Add automated alerting for performance degradation",
                    "Implement distributed tracing for complex workflows",
                    "Add performance trend analysis and reporting"
                ],
                expected_benefits={
                    "issue_detection_time_reduction_percent": 60,
                    "mttr_improvement_percent": 40,
                    "proactive_optimization_opportunities": 5
                },
                effort_level="medium",
                risk_level="low",
                success_metrics=[
                    "Performance issue detection < 5 minutes",
                    "Mean time to resolution < 30 minutes",
                    "Proactive issue prevention > 80%"
                ]
            ),
            
            "user_analytics_enhancement": SuggestionTemplate(
                template_id="user_analytics_enhancement",
                category="monitoring",
                title_template="Enhance User Analytics and Insights",
                description_template="Limited user behavior visibility. Enhanced analytics enable data-driven UX improvements and better personalization.",
                implementation_steps=[
                    "Implement comprehensive user interaction tracking",
                    "Add user journey analysis and funnel metrics",
                    "Implement A/B testing infrastructure",
                    "Add user satisfaction measurement",
                    "Create user behavior analysis dashboards"
                ],
                expected_benefits={
                    "user_insight_quality_improvement": 0.40,
                    "feature_adoption_rate_improvement_percent": 25,
                    "user_satisfaction_increase": 0.15
                },
                effort_level="medium",
                risk_level="low",
                success_metrics=[
                    "User journey completion rate > 85%",
                    "Feature adoption rate increased by 25%",
                    "User satisfaction score > 4.2/5"
                ]
            )
        }
    
    def get_template(self, template_id: str) -> Optional[SuggestionTemplate]:
        """Get a specific template by ID."""
        return self.templates.get(template_id)
    
    def get_templates_by_category(self, category: str) -> List[SuggestionTemplate]:
        """Get all templates in a specific category."""
        return [
            template for template in self.templates.values()
            if template.category == category
        ]
    
    def find_applicable_templates(
        self,
        context: Dict[str, Any]
    ) -> List[SuggestionTemplate]:
        """Find templates applicable to the given context."""
        applicable_templates = []
        
        # Example context-based matching logic
        if context.get('high_response_time'):
            applicable_templates.append(self.get_template('response_time_optimization'))
        
        if context.get('high_error_rate'):
            applicable_templates.append(self.get_template('error_handling_improvement'))
        
        if context.get('memory_pressure'):
            applicable_templates.append(self.get_template('memory_optimization'))
        
        if context.get('load_imbalance'):
            applicable_templates.append(self.get_template('load_balancing_optimization'))
        
        if context.get('high_token_usage'):
            applicable_templates.append(self.get_template('token_usage_optimization'))
        
        return [t for t in applicable_templates if t is not None]