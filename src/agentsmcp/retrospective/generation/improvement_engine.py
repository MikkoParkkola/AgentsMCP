"""
ImprovementEngine - Main coordinator for the improvement generation system.

This component orchestrates templates, impact estimation, and prioritization
to provide a high-level interface for generating improvement opportunities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

from .improvement_generator import (
    ImprovementGenerator,
    ImprovementOpportunity,
    ImprovementType,
    ImplementationEffort,
    RiskLevel
)
from .suggestion_templates import SuggestionTemplates
from .impact_estimation import ImpactEstimator


@dataclass
class ImprovementFilter:
    """Configuration for filtering improvement opportunities."""
    
    # Type filters
    included_types: Optional[Set[ImprovementType]] = None
    excluded_types: Optional[Set[ImprovementType]] = None
    
    # Effort filters
    max_effort: Optional[ImplementationEffort] = None
    min_effort: Optional[ImplementationEffort] = None
    
    # Risk filters
    max_risk: Optional[RiskLevel] = None
    min_risk: Optional[RiskLevel] = None
    
    # Score thresholds
    min_priority_score: float = 0.0
    min_confidence: float = 0.0
    
    # Impact thresholds
    min_expected_benefits: Optional[Dict[str, float]] = None
    
    # Component filters
    component_allowlist: Optional[Set[str]] = None
    component_blocklist: Optional[Set[str]] = None
    
    # Time constraints
    max_completion_time: Optional[timedelta] = None


@dataclass
class ImprovementGenerationConfig:
    """Configuration for improvement generation process."""
    
    max_improvements: int = 10
    enable_parallel_generation: bool = True
    include_low_confidence: bool = False
    deduplicate_similar: bool = True
    sort_by_priority: bool = True
    
    # Generation sources
    generate_from_patterns: bool = True
    generate_from_bottlenecks: bool = True
    generate_from_metrics: bool = True
    generate_from_quality: bool = True
    generate_from_ux: bool = True
    
    # Advanced options
    enable_cross_improvement_analysis: bool = False
    enable_resource_conflict_detection: bool = False
    enable_dependency_analysis: bool = False


class ImprovementEngine:
    """
    Main coordinator for the improvement generation system.
    
    This class provides a high-level interface for generating, filtering,
    and prioritizing improvement opportunities using the underlying
    generator, templates, and impact estimation components.
    """
    
    def __init__(self, config: Optional[ImprovementGenerationConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or ImprovementGenerationConfig()
        
        # Initialize components
        self.generator = ImprovementGenerator()
        self.templates = SuggestionTemplates()
        self.impact_estimator = ImpactEstimator()
        
        # State management
        self._generation_history: List[Dict] = []
        self._cached_improvements: Dict[str, List[ImprovementOpportunity]] = {}
        self._last_generation_time: Optional[datetime] = None
    
    async def generate_improvements(
        self,
        analysis_result,
        improvement_filter: Optional[ImprovementFilter] = None,
        config_override: Optional[ImprovementGenerationConfig] = None
    ) -> List[ImprovementOpportunity]:
        """
        Generate filtered and prioritized improvement opportunities.
        
        Args:
            analysis_result: Results from retrospective analysis
            improvement_filter: Optional filter configuration
            config_override: Optional configuration override
            
        Returns:
            List of improvement opportunities matching filter criteria
        """
        try:
            start_time = datetime.utcnow()
            self.logger.info("Starting improvement generation with ImprovementEngine")
            
            # Use config override if provided
            config = config_override or self.config
            
            # Check cache first
            cache_key = self._generate_cache_key(analysis_result, improvement_filter, config)
            if cache_key in self._cached_improvements:
                self.logger.info("Returning cached improvements")
                return self._cached_improvements[cache_key]
            
            # Generate raw improvements
            raw_improvements = await self.generator.generate_improvements(
                analysis_result=analysis_result,
                max_suggestions=config.max_improvements * 2  # Generate more to filter
            )
            
            # Apply filtering
            filtered_improvements = await self._apply_filters(
                raw_improvements, 
                improvement_filter
            )
            
            # Apply advanced analysis if enabled
            if config.enable_cross_improvement_analysis:
                filtered_improvements = await self._analyze_improvement_interactions(
                    filtered_improvements
                )
            
            if config.enable_resource_conflict_detection:
                filtered_improvements = await self._detect_resource_conflicts(
                    filtered_improvements
                )
            
            if config.enable_dependency_analysis:
                filtered_improvements = await self._analyze_dependencies(
                    filtered_improvements
                )
            
            # Final sorting and limiting
            if config.sort_by_priority:
                filtered_improvements.sort(
                    key=lambda x: (x.priority_score, x.confidence), 
                    reverse=True
                )
            
            final_improvements = filtered_improvements[:config.max_improvements]
            
            # Update cache and history
            self._cached_improvements[cache_key] = final_improvements
            self._record_generation(analysis_result, final_improvements, start_time)
            self._last_generation_time = datetime.utcnow()
            
            self.logger.info(
                f"Generated {len(final_improvements)} improvements "
                f"(filtered from {len(raw_improvements)})"
            )
            
            return final_improvements
            
        except Exception as e:
            self.logger.error(f"Improvement generation failed: {e}")
            return []
    
    async def generate_targeted_improvements(
        self,
        analysis_result,
        target_components: Set[str],
        target_types: Set[ImprovementType],
        max_improvements: int = 5
    ) -> List[ImprovementOpportunity]:
        """
        Generate improvements targeted at specific components and types.
        
        Args:
            analysis_result: Results from retrospective analysis
            target_components: Set of component names to target
            target_types: Set of improvement types to focus on
            max_improvements: Maximum number of improvements to return
            
        Returns:
            List of targeted improvement opportunities
        """
        improvement_filter = ImprovementFilter(
            included_types=target_types,
            component_allowlist=target_components,
            min_priority_score=6.0,  # Higher threshold for targeted improvements
            min_confidence=0.7
        )
        
        config_override = ImprovementGenerationConfig(
            max_improvements=max_improvements,
            enable_cross_improvement_analysis=True,
            enable_dependency_analysis=True
        )
        
        return await self.generate_improvements(
            analysis_result,
            improvement_filter,
            config_override
        )
    
    async def get_quick_wins(
        self,
        analysis_result,
        max_improvements: int = 3
    ) -> List[ImprovementOpportunity]:
        """
        Generate quick-win improvements (low effort, high impact, low risk).
        
        Args:
            analysis_result: Results from retrospective analysis
            max_improvements: Maximum number of improvements to return
            
        Returns:
            List of quick-win improvement opportunities
        """
        improvement_filter = ImprovementFilter(
            max_effort=ImplementationEffort.LOW,
            max_risk=RiskLevel.LOW,
            min_priority_score=7.0,
            min_confidence=0.8
        )
        
        config_override = ImprovementGenerationConfig(
            max_improvements=max_improvements,
            sort_by_priority=True
        )
        
        return await self.generate_improvements(
            analysis_result,
            improvement_filter,
            config_override
        )
    
    async def get_high_impact_improvements(
        self,
        analysis_result,
        max_improvements: int = 5
    ) -> List[ImprovementOpportunity]:
        """
        Generate high-impact improvements regardless of effort.
        
        Args:
            analysis_result: Results from retrospective analysis
            max_improvements: Maximum number of improvements to return
            
        Returns:
            List of high-impact improvement opportunities
        """
        improvement_filter = ImprovementFilter(
            min_priority_score=8.0,
            min_confidence=0.6,
            min_expected_benefits={
                'performance_improvement_percent': 20.0,
                'user_satisfaction_improvement': 0.15
            }
        )
        
        config_override = ImprovementGenerationConfig(
            max_improvements=max_improvements,
            enable_cross_improvement_analysis=True,
            enable_dependency_analysis=True
        )
        
        return await self.generate_improvements(
            analysis_result,
            improvement_filter,
            config_override
        )
    
    async def _apply_filters(
        self,
        improvements: List[ImprovementOpportunity],
        improvement_filter: Optional[ImprovementFilter]
    ) -> List[ImprovementOpportunity]:
        """Apply filtering criteria to improvements."""
        if not improvement_filter:
            return improvements
        
        filtered = []
        
        for improvement in improvements:
            # Type filters
            if improvement_filter.included_types:
                if improvement.improvement_type not in improvement_filter.included_types:
                    continue
            
            if improvement_filter.excluded_types:
                if improvement.improvement_type in improvement_filter.excluded_types:
                    continue
            
            # Effort filters
            if improvement_filter.max_effort:
                if self._effort_level(improvement.effort) > self._effort_level(improvement_filter.max_effort):
                    continue
            
            if improvement_filter.min_effort:
                if self._effort_level(improvement.effort) < self._effort_level(improvement_filter.min_effort):
                    continue
            
            # Risk filters
            if improvement_filter.max_risk:
                if self._risk_level(improvement.risk) > self._risk_level(improvement_filter.max_risk):
                    continue
            
            if improvement_filter.min_risk:
                if self._risk_level(improvement.risk) < self._risk_level(improvement_filter.min_risk):
                    continue
            
            # Score thresholds
            if improvement.priority_score < improvement_filter.min_priority_score:
                continue
            
            if improvement.confidence < improvement_filter.min_confidence:
                continue
            
            # Benefit thresholds
            if improvement_filter.min_expected_benefits:
                meets_benefits = False
                for benefit_key, min_value in improvement_filter.min_expected_benefits.items():
                    if improvement.expected_benefits.get(benefit_key, 0) >= min_value:
                        meets_benefits = True
                        break
                if not meets_benefits:
                    continue
            
            # Component filters
            if improvement_filter.component_allowlist:
                if not any(comp in improvement_filter.component_allowlist 
                          for comp in improvement.affected_components):
                    continue
            
            if improvement_filter.component_blocklist:
                if any(comp in improvement_filter.component_blocklist 
                      for comp in improvement.affected_components):
                    continue
            
            # Time constraints
            if improvement_filter.max_completion_time and improvement.estimated_completion_time:
                if improvement.estimated_completion_time > improvement_filter.max_completion_time:
                    continue
            
            filtered.append(improvement)
        
        return filtered
    
    async def _analyze_improvement_interactions(
        self,
        improvements: List[ImprovementOpportunity]
    ) -> List[ImprovementOpportunity]:
        """Analyze interactions between improvements."""
        # Group improvements by affected components
        component_groups = {}
        for improvement in improvements:
            for component in improvement.affected_components:
                if component not in component_groups:
                    component_groups[component] = []
                component_groups[component].append(improvement)
        
        # Identify synergistic improvements
        for component, related_improvements in component_groups.items():
            if len(related_improvements) > 1:
                # Boost priority for complementary improvements
                for improvement in related_improvements:
                    improvement.priority_score *= 1.1
                    improvement.supporting_evidence.append(
                        f"Synergistic with {len(related_improvements)-1} other improvements on {component}"
                    )
        
        return improvements
    
    async def _detect_resource_conflicts(
        self,
        improvements: List[ImprovementOpportunity]
    ) -> List[ImprovementOpportunity]:
        """Detect and resolve resource conflicts between improvements."""
        # Simple conflict detection based on affected components
        conflicts = {}
        
        for i, improvement1 in enumerate(improvements):
            for j, improvement2 in enumerate(improvements[i+1:], i+1):
                shared_components = set(improvement1.affected_components) & set(improvement2.affected_components)
                if shared_components:
                    conflict_key = frozenset([improvement1.opportunity_id, improvement2.opportunity_id])
                    conflicts[conflict_key] = {
                        'improvements': [improvement1, improvement2],
                        'shared_components': shared_components,
                        'conflict_severity': len(shared_components) / max(
                            len(improvement1.affected_components),
                            len(improvement2.affected_components)
                        )
                    }
        
        # Resolve conflicts by keeping higher priority improvement
        conflicted_ids = set()
        for conflict_data in conflicts.values():
            if conflict_data['conflict_severity'] > 0.5:  # Significant overlap
                improvements_in_conflict = conflict_data['improvements']
                improvements_in_conflict.sort(key=lambda x: x.priority_score, reverse=True)
                
                # Keep the highest priority, mark others as conflicted
                for improvement in improvements_in_conflict[1:]:
                    conflicted_ids.add(improvement.opportunity_id)
        
        # Filter out conflicted improvements
        return [imp for imp in improvements if imp.opportunity_id not in conflicted_ids]
    
    async def _analyze_dependencies(
        self,
        improvements: List[ImprovementOpportunity]
    ) -> List[ImprovementOpportunity]:
        """Analyze and order improvements by dependencies."""
        # Simple dependency analysis based on improvement types
        type_dependencies = {
            ImprovementType.INFRASTRUCTURE: 0,  # Foundation
            ImprovementType.ALGORITHM_OPTIMIZATION: 1,
            ImprovementType.RESOURCE_SCALING: 1,
            ImprovementType.ERROR_HANDLING: 2,
            ImprovementType.USER_INTERFACE: 3,  # Build on top
            ImprovementType.INTEGRATION: 3
        }
        
        # Adjust priority scores based on dependency order
        for improvement in improvements:
            dependency_level = type_dependencies.get(improvement.improvement_type, 2)
            
            # Earlier dependencies get slight priority boost
            if dependency_level == 0:
                improvement.priority_score *= 1.05
            elif dependency_level == 3:
                improvement.priority_score *= 0.95
        
        return improvements
    
    def _effort_level(self, effort: ImplementationEffort) -> int:
        """Convert effort enum to numeric level."""
        levels = {
            ImplementationEffort.AUTOMATIC: 0,
            ImplementationEffort.MINIMAL: 1,
            ImplementationEffort.LOW: 2,
            ImplementationEffort.MEDIUM: 3,
            ImplementationEffort.HIGH: 4,
            ImplementationEffort.COMPLEX: 5
        }
        return levels.get(effort, 3)
    
    def _risk_level(self, risk: RiskLevel) -> int:
        """Convert risk enum to numeric level."""
        levels = {
            RiskLevel.MINIMAL: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }
        return levels.get(risk, 2)
    
    def _generate_cache_key(
        self,
        analysis_result,
        improvement_filter: Optional[ImprovementFilter],
        config: ImprovementGenerationConfig
    ) -> str:
        """Generate cache key for improvements."""
        # Simple hash based on session ID and filter/config
        base_key = f"{analysis_result.session_id}_{analysis_result.timestamp.isoformat()}"
        
        if improvement_filter:
            filter_key = f"_{hash(str(improvement_filter))}"
        else:
            filter_key = "_nofilter"
        
        config_key = f"_{hash(str(config))}"
        
        return base_key + filter_key + config_key
    
    def _record_generation(
        self,
        analysis_result,
        improvements: List[ImprovementOpportunity],
        start_time: datetime
    ):
        """Record generation metrics for analysis."""
        generation_record = {
            'timestamp': start_time,
            'session_id': analysis_result.session_id,
            'improvements_generated': len(improvements),
            'generation_time_ms': (datetime.utcnow() - start_time).total_seconds() * 1000,
            'improvement_types': [imp.improvement_type for imp in improvements],
            'average_priority_score': sum(imp.priority_score for imp in improvements) / len(improvements) if improvements else 0,
            'average_confidence': sum(imp.confidence for imp in improvements) / len(improvements) if improvements else 0
        }
        
        self._generation_history.append(generation_record)
        
        # Keep only last 100 records
        if len(self._generation_history) > 100:
            self._generation_history = self._generation_history[-100:]
    
    def get_generation_stats(self) -> Dict:
        """Get statistics about improvement generation performance."""
        if not self._generation_history:
            return {}
        
        recent_generations = self._generation_history[-10:]  # Last 10 generations
        
        return {
            'total_generations': len(self._generation_history),
            'average_generation_time_ms': sum(
                gen['generation_time_ms'] for gen in recent_generations
            ) / len(recent_generations),
            'average_improvements_per_generation': sum(
                gen['improvements_generated'] for gen in recent_generations
            ) / len(recent_generations),
            'last_generation_time': self._last_generation_time.isoformat() if self._last_generation_time else None,
            'cache_hit_rate': len(self._cached_improvements) / max(len(self._generation_history), 1)
        }
    
    def clear_cache(self):
        """Clear the improvement cache."""
        self._cached_improvements.clear()
        self.logger.info("Improvement cache cleared")