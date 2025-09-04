"""
Extensible criteria and filtering system for approval decisions.

This module provides a flexible framework for defining and applying approval criteria,
enabling sophisticated filtering and automated decision-making based on various factors.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
import logging

from .approval_decision import ApprovalStatus, ApprovalMode, RejectionReason
from ..data_models import SelfImprovementAction, ImprovementCategory, PriorityLevel

logger = logging.getLogger(__name__)


class CriterionType(Enum):
    """Types of approval criteria."""
    CATEGORY_BASED = "category_based"
    PRIORITY_BASED = "priority_based"
    CONTENT_BASED = "content_based"
    IMPACT_BASED = "impact_based"
    RISK_BASED = "risk_based"
    RESOURCE_BASED = "resource_based"
    TIME_BASED = "time_based"
    HISTORY_BASED = "history_based"
    CUSTOM = "custom"


class FilterAction(Enum):
    """Actions that can be taken based on filter criteria."""
    AUTO_APPROVE = "auto_approve"
    AUTO_REJECT = "auto_reject"
    REQUIRE_MANUAL = "require_manual"
    FLAG_FOR_REVIEW = "flag_for_review"
    ESCALATE = "escalate"
    BATCH_APPROVE = "batch_approve"
    BATCH_REJECT = "batch_reject"


@dataclass
class CriterionResult:
    """Result of applying a single criterion."""
    criterion_name: str
    matched: bool
    confidence: float = 1.0
    action: Optional[FilterAction] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.matched


class ApprovalCriterion(ABC):
    """Abstract base class for approval criteria."""
    
    def __init__(self, name: str, priority: int = 50, enabled: bool = True):
        self.name = name
        self.priority = priority  # Higher number = higher priority
        self.enabled = enabled
        self.application_count = 0
        self.match_count = 0
    
    @abstractmethod
    def evaluate(self, improvement: SelfImprovementAction, 
                context: Optional[Dict[str, Any]] = None) -> CriterionResult:
        """Evaluate if the criterion matches the improvement."""
        pass
    
    def get_match_rate(self) -> float:
        """Get the match rate for this criterion."""
        if self.application_count == 0:
            return 0.0
        return self.match_count / self.application_count
    
    def apply(self, improvement: SelfImprovementAction,
             context: Optional[Dict[str, Any]] = None) -> CriterionResult:
        """Apply the criterion and track statistics."""
        self.application_count += 1
        result = self.evaluate(improvement, context)
        
        if result.matched:
            self.match_count += 1
        
        return result


class CategoryCriterion(ApprovalCriterion):
    """Criterion based on improvement category."""
    
    def __init__(self, 
                 name: str,
                 target_categories: Set[ImprovementCategory],
                 action: FilterAction,
                 priority: int = 50,
                 enabled: bool = True):
        super().__init__(name, priority, enabled)
        self.target_categories = target_categories
        self.action = action
    
    def evaluate(self, improvement: SelfImprovementAction,
                context: Optional[Dict[str, Any]] = None) -> CriterionResult:
        """Evaluate based on improvement category."""
        matched = improvement.category in self.target_categories
        
        return CriterionResult(
            criterion_name=self.name,
            matched=matched,
            action=self.action if matched else None,
            reason=f"Category {improvement.category.value} {'in' if matched else 'not in'} target set",
            metadata={"category": improvement.category.value}
        )


class PriorityCriterion(ApprovalCriterion):
    """Criterion based on improvement priority."""
    
    def __init__(self,
                 name: str,
                 min_priority: Optional[PriorityLevel] = None,
                 max_priority: Optional[PriorityLevel] = None,
                 action: FilterAction = FilterAction.REQUIRE_MANUAL,
                 priority: int = 50,
                 enabled: bool = True):
        super().__init__(name, priority, enabled)
        self.min_priority = min_priority
        self.max_priority = max_priority
        self.action = action
        
        # Priority value mapping for comparisons
        self._priority_values = {
            PriorityLevel.LOW: 1,
            PriorityLevel.MEDIUM: 2,
            PriorityLevel.HIGH: 3,
            PriorityLevel.CRITICAL: 4
        }
    
    def evaluate(self, improvement: SelfImprovementAction,
                context: Optional[Dict[str, Any]] = None) -> CriterionResult:
        """Evaluate based on improvement priority."""
        improvement_value = self._priority_values.get(improvement.priority, 0)
        
        matches_min = (self.min_priority is None or 
                      improvement_value >= self._priority_values.get(self.min_priority, 0))
        matches_max = (self.max_priority is None or
                      improvement_value <= self._priority_values.get(self.max_priority, 5))
        
        matched = matches_min and matches_max
        
        reason_parts = []
        if self.min_priority:
            reason_parts.append(f"priority >= {self.min_priority.value}")
        if self.max_priority:
            reason_parts.append(f"priority <= {self.max_priority.value}")
        
        reason = f"Priority {improvement.priority.value} {'meets' if matched else 'does not meet'} criteria: {' and '.join(reason_parts)}"
        
        return CriterionResult(
            criterion_name=self.name,
            matched=matched,
            action=self.action if matched else None,
            reason=reason,
            metadata={"priority": improvement.priority.value, "priority_value": improvement_value}
        )


class ContentCriterion(ApprovalCriterion):
    """Criterion based on improvement content (title, description, etc.)."""
    
    def __init__(self,
                 name: str,
                 patterns: List[str],
                 fields: List[str] = None,
                 action: FilterAction = FilterAction.FLAG_FOR_REVIEW,
                 case_sensitive: bool = False,
                 require_all_patterns: bool = False,
                 priority: int = 50,
                 enabled: bool = True):
        super().__init__(name, priority, enabled)
        self.patterns = patterns
        self.fields = fields or ["title", "description", "implementation_notes"]
        self.action = action
        self.case_sensitive = case_sensitive
        self.require_all_patterns = require_all_patterns
        
        # Compile regex patterns
        flags = 0 if case_sensitive else re.IGNORECASE
        self.compiled_patterns = [re.compile(pattern, flags) for pattern in patterns]
    
    def evaluate(self, improvement: SelfImprovementAction,
                context: Optional[Dict[str, Any]] = None) -> CriterionResult:
        """Evaluate based on content patterns."""
        # Get text content from specified fields
        content_parts = []
        for field in self.fields:
            value = getattr(improvement, field, "")
            if value:
                content_parts.append(str(value))
        
        full_content = " ".join(content_parts)
        
        # Check patterns
        pattern_matches = []
        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.findall(full_content)
            pattern_matches.append({
                "pattern": self.patterns[i],
                "matches": len(matches),
                "found": len(matches) > 0
            })
        
        # Determine if criterion is matched
        if self.require_all_patterns:
            matched = all(pm["found"] for pm in pattern_matches)
        else:
            matched = any(pm["found"] for pm in pattern_matches)
        
        # Build detailed reason
        found_patterns = [pm["pattern"] for pm in pattern_matches if pm["found"]]
        reason = f"Content {'matches' if matched else 'does not match'} patterns. Found: {found_patterns}"
        
        return CriterionResult(
            criterion_name=self.name,
            matched=matched,
            action=self.action if matched else None,
            reason=reason,
            metadata={
                "pattern_matches": pattern_matches,
                "found_patterns": found_patterns,
                "content_length": len(full_content)
            }
        )


class ImpactCriterion(ApprovalCriterion):
    """Criterion based on estimated impact."""
    
    def __init__(self,
                 name: str,
                 min_impact_score: Optional[float] = None,
                 max_impact_score: Optional[float] = None,
                 impact_keywords: List[str] = None,
                 action: FilterAction = FilterAction.REQUIRE_MANUAL,
                 priority: int = 50,
                 enabled: bool = True):
        super().__init__(name, priority, enabled)
        self.min_impact_score = min_impact_score
        self.max_impact_score = max_impact_score
        self.impact_keywords = impact_keywords or [
            "significant", "major", "critical", "substantial", "dramatic",
            "high impact", "game changing", "transformative"
        ]
        self.action = action
    
    def evaluate(self, improvement: SelfImprovementAction,
                context: Optional[Dict[str, Any]] = None) -> CriterionResult:
        """Evaluate based on estimated impact."""
        # Try to get impact score from context
        impact_score = None
        if context and "impact_score" in context:
            impact_score = context["impact_score"]
        
        # If no explicit score, estimate from content
        if impact_score is None:
            impact_score = self._estimate_impact_from_content(improvement)
        
        # Check score thresholds
        score_matched = True
        if self.min_impact_score is not None:
            score_matched = score_matched and impact_score >= self.min_impact_score
        if self.max_impact_score is not None:
            score_matched = score_matched and impact_score <= self.max_impact_score
        
        # Check for impact keywords in content
        content = f"{improvement.title} {improvement.description} {improvement.expected_benefit}"
        keyword_matches = [kw for kw in self.impact_keywords 
                          if kw.lower() in content.lower()]
        keyword_matched = len(keyword_matches) > 0
        
        # Combine criteria
        matched = score_matched or keyword_matched
        
        confidence = 0.8 if impact_score is not None else 0.6  # Lower confidence for estimated scores
        
        return CriterionResult(
            criterion_name=self.name,
            matched=matched,
            confidence=confidence,
            action=self.action if matched else None,
            reason=f"Impact score: {impact_score:.2f}, Keywords found: {keyword_matches}",
            metadata={
                "impact_score": impact_score,
                "score_matched": score_matched,
                "keyword_matched": keyword_matched,
                "keyword_matches": keyword_matches
            }
        )
    
    def _estimate_impact_from_content(self, improvement: SelfImprovementAction) -> float:
        """Estimate impact score from content."""
        score = 0.5  # Base score
        
        # Adjust based on priority
        priority_boost = {
            PriorityLevel.LOW: 0.0,
            PriorityLevel.MEDIUM: 0.1, 
            PriorityLevel.HIGH: 0.2,
            PriorityLevel.CRITICAL: 0.3
        }
        score += priority_boost.get(improvement.priority, 0.0)
        
        # Adjust based on category
        category_boost = {
            ImprovementCategory.PERFORMANCE: 0.1,
            ImprovementCategory.RELIABILITY: 0.15,
            ImprovementCategory.SECURITY: 0.2,
            ImprovementCategory.PROCESS: 0.05
        }
        score += category_boost.get(improvement.category, 0.0)
        
        # Look for impact keywords
        content = f"{improvement.title} {improvement.description} {improvement.expected_benefit}".lower()
        for keyword in self.impact_keywords:
            if keyword.lower() in content:
                score += 0.1
                break  # Only count once
        
        return min(score, 1.0)  # Cap at 1.0


class RiskCriterion(ApprovalCriterion):
    """Criterion based on risk assessment."""
    
    def __init__(self,
                 name: str,
                 max_risk_score: float = 0.7,
                 risk_keywords: List[str] = None,
                 action: FilterAction = FilterAction.REQUIRE_MANUAL,
                 priority: int = 60,  # Higher priority for risk
                 enabled: bool = True):
        super().__init__(name, priority, enabled)
        self.max_risk_score = max_risk_score
        self.risk_keywords = risk_keywords or [
            "breaking", "dangerous", "risky", "unstable", "experimental",
            "major change", "system-wide", "critical path", "irreversible"
        ]
        self.action = action
    
    def evaluate(self, improvement: SelfImprovementAction,
                context: Optional[Dict[str, Any]] = None) -> CriterionResult:
        """Evaluate based on risk assessment."""
        # Try to get risk score from context
        risk_score = None
        if context and "risk_score" in context:
            risk_score = context["risk_score"]
        
        # If no explicit score, estimate from content
        if risk_score is None:
            risk_score = self._estimate_risk_from_content(improvement)
        
        # Check if risk exceeds threshold
        risk_exceeded = risk_score > self.max_risk_score
        
        # Check for risk keywords
        content = f"{improvement.title} {improvement.description} {improvement.implementation_notes}"
        keyword_matches = [kw for kw in self.risk_keywords
                          if kw.lower() in content.lower()]
        keyword_risk = len(keyword_matches) > 0
        
        # High risk if either condition is met
        matched = risk_exceeded or keyword_risk
        
        confidence = 0.8 if risk_score is not None else 0.6
        
        return CriterionResult(
            criterion_name=self.name,
            matched=matched,
            confidence=confidence,
            action=self.action if matched else None,
            reason=f"Risk score: {risk_score:.2f} (threshold: {self.max_risk_score}), Risk keywords: {keyword_matches}",
            metadata={
                "risk_score": risk_score,
                "risk_exceeded": risk_exceeded,
                "keyword_risk": keyword_risk,
                "keyword_matches": keyword_matches
            }
        )
    
    def _estimate_risk_from_content(self, improvement: SelfImprovementAction) -> float:
        """Estimate risk score from content."""
        score = 0.2  # Base risk score
        
        # Higher risk for certain categories
        category_risk = {
            ImprovementCategory.SECURITY: 0.3,
            ImprovementCategory.RELIABILITY: 0.2,
            ImprovementCategory.PERFORMANCE: 0.1,
            ImprovementCategory.PROCESS: 0.05
        }
        score += category_risk.get(improvement.category, 0.0)
        
        # Higher risk for critical priorities
        if improvement.priority == PriorityLevel.CRITICAL:
            score += 0.2
        elif improvement.priority == PriorityLevel.HIGH:
            score += 0.1
        
        # Check for risk keywords in content
        content = f"{improvement.title} {improvement.description} {improvement.implementation_notes}".lower()
        for keyword in self.risk_keywords:
            if keyword.lower() in content:
                score += 0.15
                break  # Only count once
        
        return min(score, 1.0)


class TimeCriterion(ApprovalCriterion):
    """Criterion based on timing constraints."""
    
    def __init__(self,
                 name: str,
                 business_hours_only: bool = False,
                 excluded_days: List[str] = None,
                 max_age_hours: Optional[int] = None,
                 action: FilterAction = FilterAction.REQUIRE_MANUAL,
                 priority: int = 40,
                 enabled: bool = True):
        super().__init__(name, priority, enabled)
        self.business_hours_only = business_hours_only
        self.excluded_days = excluded_days or []  # e.g., ["saturday", "sunday"]
        self.max_age_hours = max_age_hours
        self.action = action
    
    def evaluate(self, improvement: SelfImprovementAction,
                context: Optional[Dict[str, Any]] = None) -> CriterionResult:
        """Evaluate based on timing constraints."""
        now = datetime.now(timezone.utc)
        reasons = []
        matched = True
        
        # Check business hours
        if self.business_hours_only:
            # Assume business hours are 9 AM to 5 PM UTC (this could be configurable)
            hour = now.hour
            if not (9 <= hour <= 17):
                matched = False
                reasons.append(f"Outside business hours ({hour}:00)")
        
        # Check excluded days
        if self.excluded_days:
            day_name = now.strftime("%A").lower()
            if day_name in [day.lower() for day in self.excluded_days]:
                matched = False
                reasons.append(f"Excluded day ({day_name})")
        
        # Check maximum age
        if self.max_age_hours:
            age = (now - improvement.created_at).total_seconds() / 3600  # hours
            if age > self.max_age_hours:
                matched = False
                reasons.append(f"Too old ({age:.1f} hours > {self.max_age_hours})")
        
        reason = f"Time constraints: {'; '.join(reasons) if reasons else 'All constraints met'}"
        
        return CriterionResult(
            criterion_name=self.name,
            matched=matched,
            action=self.action if matched else None,
            reason=reason,
            metadata={
                "current_time": now.isoformat(),
                "current_hour": now.hour,
                "current_day": now.strftime("%A"),
                "improvement_age_hours": (now - improvement.created_at).total_seconds() / 3600
            }
        )


class CustomCriterion(ApprovalCriterion):
    """Custom criterion with user-defined evaluation function."""
    
    def __init__(self,
                 name: str,
                 evaluation_func: Callable[[SelfImprovementAction, Optional[Dict[str, Any]]], CriterionResult],
                 priority: int = 50,
                 enabled: bool = True):
        super().__init__(name, priority, enabled)
        self.evaluation_func = evaluation_func
    
    def evaluate(self, improvement: SelfImprovementAction,
                context: Optional[Dict[str, Any]] = None) -> CriterionResult:
        """Evaluate using custom function."""
        try:
            result = self.evaluation_func(improvement, context)
            # Ensure the criterion name is set correctly
            result.criterion_name = self.name
            return result
        except Exception as e:
            logger.error(f"Error in custom criterion {self.name}: {e}")
            return CriterionResult(
                criterion_name=self.name,
                matched=False,
                reason=f"Evaluation error: {e}",
                confidence=0.0
            )


@dataclass
class ApprovalFilter:
    """Collection of criteria for filtering improvements."""
    
    name: str
    criteria: List[ApprovalCriterion] = field(default_factory=list)
    require_all: bool = False  # If True, all criteria must match
    enabled: bool = True
    
    def add_criterion(self, criterion: ApprovalCriterion) -> None:
        """Add a criterion to this filter."""
        self.criteria.append(criterion)
        # Sort by priority (highest first)
        self.criteria.sort(key=lambda c: c.priority, reverse=True)
    
    def evaluate(self, improvement: SelfImprovementAction,
                context: Optional[Dict[str, Any]] = None) -> List[CriterionResult]:
        """Evaluate all criteria against an improvement."""
        if not self.enabled:
            return []
        
        results = []
        for criterion in self.criteria:
            if criterion.enabled:
                result = criterion.apply(improvement, context)
                results.append(result)
        
        return results
    
    def matches(self, improvement: SelfImprovementAction,
               context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if the improvement matches this filter."""
        results = self.evaluate(improvement, context)
        
        if not results:
            return False
        
        if self.require_all:
            return all(result.matched for result in results)
        else:
            return any(result.matched for result in results)
    
    def get_recommended_action(self, improvement: SelfImprovementAction,
                             context: Optional[Dict[str, Any]] = None) -> Optional[FilterAction]:
        """Get the recommended action based on matching criteria."""
        results = self.evaluate(improvement, context)
        matching_results = [r for r in results if r.matched and r.action]
        
        if not matching_results:
            return None
        
        # Return action from highest priority criterion
        matching_results.sort(key=lambda r: next(c.priority for c in self.criteria 
                                                if c.name == r.criterion_name), reverse=True)
        return matching_results[0].action


class ApprovalCriteriaEngine:
    """Main engine for managing and applying approval criteria."""
    
    def __init__(self):
        self.filters: List[ApprovalFilter] = []
        self.global_criteria: List[ApprovalCriterion] = []
        self._statistics = {
            "evaluations": 0,
            "auto_approvals": 0,
            "auto_rejections": 0,
            "manual_reviews": 0
        }
    
    def add_filter(self, filter_obj: ApprovalFilter) -> None:
        """Add a filter to the engine."""
        self.filters.append(filter_obj)
    
    def add_global_criterion(self, criterion: ApprovalCriterion) -> None:
        """Add a global criterion that applies to all evaluations."""
        self.global_criteria.append(criterion)
        # Sort by priority
        self.global_criteria.sort(key=lambda c: c.priority, reverse=True)
    
    def evaluate_improvement(self, improvement: SelfImprovementAction,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate an improvement against all criteria and filters."""
        self._statistics["evaluations"] += 1
        
        evaluation_result = {
            "improvement_id": improvement.action_id,
            "filter_results": {},
            "global_results": [],
            "recommended_action": None,
            "confidence": 1.0,
            "reasoning": []
        }
        
        # Evaluate global criteria
        for criterion in self.global_criteria:
            if criterion.enabled:
                result = criterion.apply(improvement, context)
                evaluation_result["global_results"].append(result)
        
        # Evaluate filters
        for filter_obj in self.filters:
            if filter_obj.enabled:
                results = filter_obj.evaluate(improvement, context)
                evaluation_result["filter_results"][filter_obj.name] = {
                    "results": results,
                    "matches": filter_obj.matches(improvement, context),
                    "recommended_action": filter_obj.get_recommended_action(improvement, context)
                }
        
        # Determine overall recommendation
        recommended_action, confidence, reasoning = self._determine_recommendation(evaluation_result)
        evaluation_result["recommended_action"] = recommended_action
        evaluation_result["confidence"] = confidence
        evaluation_result["reasoning"] = reasoning
        
        # Update statistics
        if recommended_action == FilterAction.AUTO_APPROVE:
            self._statistics["auto_approvals"] += 1
        elif recommended_action == FilterAction.AUTO_REJECT:
            self._statistics["auto_rejections"] += 1
        else:
            self._statistics["manual_reviews"] += 1
        
        return evaluation_result
    
    def _determine_recommendation(self, evaluation_result: Dict[str, Any]) -> tuple[Optional[FilterAction], float, List[str]]:
        """Determine overall recommendation from evaluation results."""
        actions = []
        reasoning = []
        
        # Collect actions from global criteria
        for result in evaluation_result["global_results"]:
            if result.matched and result.action:
                actions.append((result.action, result.confidence, result.reason))
        
        # Collect actions from filters
        for filter_name, filter_data in evaluation_result["filter_results"].items():
            if filter_data["recommended_action"]:
                # Find the highest confidence from matching results
                matching_results = [r for r in filter_data["results"] if r.matched]
                if matching_results:
                    max_confidence = max(r.confidence for r in matching_results)
                    reasons = [r.reason for r in matching_results if r.matched]
                    actions.append((filter_data["recommended_action"], max_confidence, f"Filter '{filter_name}': {'; '.join(reasons)}"))
        
        if not actions:
            return None, 1.0, ["No matching criteria"]
        
        # Sort by confidence and choose the highest confidence action
        actions.sort(key=lambda x: x[1], reverse=True)
        best_action, best_confidence, best_reason = actions[0]
        
        reasoning = [best_reason]
        # Add other significant reasons
        for action, conf, reason in actions[1:3]:  # Top 3
            if conf > 0.5:
                reasoning.append(f"Also: {reason}")
        
        return best_action, best_confidence, reasoning
    
    def create_default_filters(self) -> None:
        """Create a set of default filters for common scenarios."""
        # High-risk filter
        risk_filter = ApprovalFilter("high_risk", require_all=False)
        risk_filter.add_criterion(RiskCriterion(
            "high_risk_check",
            max_risk_score=0.6,
            action=FilterAction.REQUIRE_MANUAL
        ))
        risk_filter.add_criterion(CategoryCriterion(
            "security_check",
            target_categories={ImprovementCategory.SECURITY},
            action=FilterAction.REQUIRE_MANUAL
        ))
        self.add_filter(risk_filter)
        
        # Auto-approval filter for low-risk improvements
        auto_approve_filter = ApprovalFilter("auto_approve_safe", require_all=True)
        auto_approve_filter.add_criterion(CategoryCriterion(
            "safe_categories",
            target_categories={ImprovementCategory.PERFORMANCE, ImprovementCategory.PROCESS},
            action=FilterAction.AUTO_APPROVE
        ))
        auto_approve_filter.add_criterion(PriorityCriterion(
            "medium_plus_priority",
            min_priority=PriorityLevel.MEDIUM,
            action=FilterAction.AUTO_APPROVE
        ))
        self.add_filter(auto_approve_filter)
        
        # Business hours filter
        timing_filter = ApprovalFilter("business_hours")
        timing_filter.add_criterion(TimeCriterion(
            "business_hours_only",
            business_hours_only=True,
            excluded_days=["saturday", "sunday"],
            action=FilterAction.REQUIRE_MANUAL
        ))
        self.add_filter(timing_filter)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        total = self._statistics["evaluations"]
        return {
            **self._statistics,
            "auto_approval_rate": self._statistics["auto_approvals"] / max(total, 1),
            "auto_rejection_rate": self._statistics["auto_rejections"] / max(total, 1),
            "manual_review_rate": self._statistics["manual_reviews"] / max(total, 1),
            "criteria_stats": {
                criterion.name: {
                    "applications": criterion.application_count,
                    "matches": criterion.match_count,
                    "match_rate": criterion.get_match_rate()
                }
                for criterion in self.global_criteria
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._statistics = {
            "evaluations": 0,
            "auto_approvals": 0,
            "auto_rejections": 0,
            "manual_reviews": 0
        }
        
        # Reset criterion statistics
        for criterion in self.global_criteria:
            criterion.application_count = 0
            criterion.match_count = 0
        
        for filter_obj in self.filters:
            for criterion in filter_obj.criteria:
                criterion.application_count = 0
                criterion.match_count = 0