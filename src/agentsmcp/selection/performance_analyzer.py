"""
Performance Analyzer with Statistical Significance Testing

Provides rigorous statistical analysis for selection performance comparisons,
including significance testing, effect size calculation, and confidence intervals.
"""

import logging
import math
import statistics
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
import numpy as np
from datetime import datetime, timedelta

from .selection_history import SelectionHistory, SelectionRecord
from .benchmark_tracker import BenchmarkTracker, SelectionMetrics


logger = logging.getLogger(__name__)


class StatisticalTest(Enum):
    """Available statistical tests."""
    CHI_SQUARE = "chi_square"
    T_TEST = "t_test"
    WELCH_T_TEST = "welch_t_test"
    MANN_WHITNEY = "mann_whitney"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    BOOTSTRAP = "bootstrap"


@dataclass
class ComparisonResult:
    """Result of comparing two options."""
    
    option_a: str
    option_b: str
    
    # Sample sizes
    n_a: int
    n_b: int
    
    # Metrics
    metric_a: float
    metric_b: float
    difference: float  # metric_b - metric_a
    relative_difference: float  # (metric_b - metric_a) / metric_a
    
    # Statistical tests
    test_statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    confidence_interval: Tuple[float, float]
    
    # Effect size
    effect_size: float
    effect_magnitude: str  # "negligible", "small", "medium", "large"
    
    # Test metadata
    test_method: StatisticalTest
    assumptions_met: bool
    warnings: List[str]
    
    # Recommendation
    recommended_winner: Optional[str] = None
    recommendation_confidence: float = 0.0


class SignificanceResult(NamedTuple):
    """Result of significance testing."""
    is_significant: bool
    p_value: float
    test_statistic: float
    critical_value: float
    degrees_freedom: Optional[int] = None


@dataclass 
class MultipleComparisonResult:
    """Result of comparing multiple options simultaneously."""
    
    options: List[str]
    pairwise_comparisons: Dict[Tuple[str, str], ComparisonResult]
    overall_significant: bool
    best_option: Optional[str]
    rankings: List[Tuple[str, float, float]]  # (option, metric, confidence)
    
    # Multiple testing correction
    correction_method: str = "bonferroni"
    corrected_alpha: float = 0.05
    
    # Overall metrics
    f_statistic: Optional[float] = None
    anova_p_value: Optional[float] = None


class PerformanceAnalyzer:
    """
    Statistical analyzer for selection performance comparisons.
    
    Provides rigorous statistical testing with proper handling of multiple
    comparisons, effect sizes, and confidence intervals.
    """
    
    def __init__(self, 
                 selection_history: SelectionHistory,
                 benchmark_tracker: BenchmarkTracker,
                 default_confidence_level: float = 0.95,
                 min_sample_size: int = 30):
        """
        Initialize performance analyzer.
        
        Args:
            selection_history: Historical selection data
            benchmark_tracker: Real-time performance metrics  
            default_confidence_level: Default confidence level for tests
            min_sample_size: Minimum sample size for valid comparisons
        """
        self.selection_history = selection_history
        self.benchmark_tracker = benchmark_tracker
        self.default_confidence_level = default_confidence_level
        self.min_sample_size = min_sample_size
        
        # Analysis cache
        self._analysis_cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        
        logger.info("PerformanceAnalyzer initialized")
    
    def compare_options(self,
                       selection_type: str,
                       option_a: str,
                       option_b: str,
                       metric: str = "success_rate",
                       days: int = 30,
                       confidence_level: float = None,
                       test_method: StatisticalTest = None) -> Optional[ComparisonResult]:
        """
        Compare two options with statistical significance testing.
        
        Args:
            selection_type: Type of selection to analyze
            option_a: First option to compare
            option_b: Second option to compare  
            metric: Metric to compare ("success_rate", "completion_time", "quality_score")
            days: Number of days of data to include
            confidence_level: Confidence level for test (default: 0.95)
            test_method: Statistical test to use
            
        Returns:
            Comparison result with statistical analysis
        """
        confidence_level = confidence_level or self.default_confidence_level
        
        try:
            # Get data for both options
            since = datetime.now() - timedelta(days=days)
            
            records_a = self.selection_history.get_records(
                selection_type=selection_type,
                selected_option=option_a,
                since=since,
                only_completed=True,
                limit=10000
            )
            
            records_b = self.selection_history.get_records(
                selection_type=selection_type,
                selected_option=option_b,
                since=since,
                only_completed=True,
                limit=10000
            )
            
            if len(records_a) < self.min_sample_size or len(records_b) < self.min_sample_size:
                logger.warning(f"Insufficient data: {option_a}={len(records_a)}, {option_b}={len(records_b)}")
                return None
            
            # Extract metric values
            values_a = self._extract_metric_values(records_a, metric)
            values_b = self._extract_metric_values(records_b, metric)
            
            if not values_a or not values_b:
                logger.warning(f"No metric values found for {metric}")
                return None
            
            # Calculate summary statistics
            metric_a = statistics.mean(values_a)
            metric_b = statistics.mean(values_b)
            difference = metric_b - metric_a
            relative_difference = difference / max(0.001, abs(metric_a))  # Avoid division by zero
            
            # Choose appropriate statistical test
            if test_method is None:
                test_method = self._choose_statistical_test(values_a, values_b, metric)
            
            # Perform statistical test
            test_result = self._perform_statistical_test(
                values_a, values_b, test_method, confidence_level
            )
            
            # Calculate effect size
            effect_size = self._calculate_effect_size(values_a, values_b)
            effect_magnitude = self._interpret_effect_size(effect_size)
            
            # Calculate confidence interval for difference
            confidence_interval = self._calculate_confidence_interval(
                values_a, values_b, confidence_level
            )
            
            # Check assumptions
            assumptions_met, warnings = self._check_test_assumptions(
                values_a, values_b, test_method
            )
            
            # Make recommendation
            recommended_winner, recommendation_confidence = self._make_recommendation(
                test_result.is_significant, effect_size, difference, option_a, option_b
            )
            
            return ComparisonResult(
                option_a=option_a,
                option_b=option_b,
                n_a=len(values_a),
                n_b=len(values_b),
                metric_a=metric_a,
                metric_b=metric_b,
                difference=difference,
                relative_difference=relative_difference,
                test_statistic=test_result.test_statistic,
                p_value=test_result.p_value,
                is_significant=test_result.is_significant,
                confidence_level=confidence_level,
                confidence_interval=confidence_interval,
                effect_size=effect_size,
                effect_magnitude=effect_magnitude,
                test_method=test_method,
                assumptions_met=assumptions_met,
                warnings=warnings,
                recommended_winner=recommended_winner,
                recommendation_confidence=recommendation_confidence
            )
            
        except Exception as e:
            logger.error(f"Error comparing {option_a} vs {option_b}: {e}")
            return None
    
    def compare_multiple_options(self,
                               selection_type: str,
                               options: List[str],
                               metric: str = "success_rate",
                               days: int = 30,
                               confidence_level: float = None,
                               correction_method: str = "bonferroni") -> Optional[MultipleComparisonResult]:
        """
        Compare multiple options with proper multiple testing correction.
        
        Args:
            selection_type: Type of selection to analyze
            options: List of options to compare
            metric: Metric to compare
            days: Number of days of data to include
            confidence_level: Confidence level for tests
            correction_method: Multiple testing correction method
            
        Returns:
            Multiple comparison result
        """
        if len(options) < 2:
            return None
        
        confidence_level = confidence_level or self.default_confidence_level
        
        try:
            # Perform pairwise comparisons
            pairwise_comparisons = {}
            p_values = []
            
            for i, option_a in enumerate(options):
                for j, option_b in enumerate(options[i+1:], i+1):
                    comparison = self.compare_options(
                        selection_type=selection_type,
                        option_a=option_a,
                        option_b=option_b,
                        metric=metric,
                        days=days,
                        confidence_level=confidence_level
                    )
                    
                    if comparison:
                        pairwise_comparisons[(option_a, option_b)] = comparison
                        p_values.append(comparison.p_value)
            
            if not p_values:
                return None
            
            # Apply multiple testing correction
            corrected_alpha = self._apply_multiple_testing_correction(
                p_values, confidence_level, correction_method
            )
            
            # Determine overall significance
            overall_significant = any(p < corrected_alpha for p in p_values)
            
            # Rank options by performance
            option_metrics = {}
            option_confidence = {}
            
            for option in options:
                # Get metrics from current data
                metrics = self.benchmark_tracker.get_metrics(selection_type, option)
                key = (selection_type, option)
                
                if key in metrics:
                    if metric == "success_rate":
                        option_metrics[option] = metrics[key].success_rate
                        option_confidence[option] = metrics[key].sample_confidence
                    elif metric == "completion_time":
                        option_metrics[option] = -metrics[key].avg_completion_time_ms  # Negative for ranking
                        option_confidence[option] = metrics[key].sample_confidence
                    elif metric == "quality_score":
                        option_metrics[option] = metrics[key].avg_quality_score
                        option_confidence[option] = metrics[key].sample_confidence
                    else:
                        option_metrics[option] = 0.0
                        option_confidence[option] = 0.0
                else:
                    option_metrics[option] = 0.0
                    option_confidence[option] = 0.0
            
            # Sort by metric value
            rankings = sorted(
                [(option, option_metrics[option], option_confidence[option]) for option in options],
                key=lambda x: x[1],
                reverse=True
            )
            
            best_option = rankings[0][0] if rankings else None
            
            # Perform ANOVA if applicable
            f_statistic, anova_p_value = self._perform_anova(selection_type, options, metric, days)
            
            return MultipleComparisonResult(
                options=options,
                pairwise_comparisons=pairwise_comparisons,
                overall_significant=overall_significant,
                best_option=best_option,
                rankings=rankings,
                correction_method=correction_method,
                corrected_alpha=corrected_alpha,
                f_statistic=f_statistic,
                anova_p_value=anova_p_value
            )
            
        except Exception as e:
            logger.error(f"Error in multiple comparison: {e}")
            return None
    
    def detect_performance_regression(self,
                                    selection_type: str,
                                    option: str,
                                    baseline_days: int = 30,
                                    recent_days: int = 7,
                                    metric: str = "success_rate",
                                    significance_threshold: float = 0.05) -> Optional[ComparisonResult]:
        """
        Detect if an option has regressed in performance.
        
        Args:
            selection_type: Type of selection to analyze
            option: Option to check for regression
            baseline_days: Days to use for baseline performance
            recent_days: Days to use for recent performance
            metric: Metric to analyze
            significance_threshold: P-value threshold for significance
            
        Returns:
            Comparison result between baseline and recent performance
        """
        try:
            now = datetime.now()
            
            # Get baseline data (excluding recent period)
            baseline_start = now - timedelta(days=baseline_days)
            baseline_end = now - timedelta(days=recent_days)
            
            baseline_records = self.selection_history.get_records(
                selection_type=selection_type,
                selected_option=option,
                since=baseline_start,
                only_completed=True,
                limit=10000
            )
            
            # Filter to baseline period only
            baseline_records = [
                r for r in baseline_records 
                if r.timestamp <= baseline_end
            ]
            
            # Get recent data
            recent_start = now - timedelta(days=recent_days)
            recent_records = self.selection_history.get_records(
                selection_type=selection_type,
                selected_option=option,
                since=recent_start,
                only_completed=True,
                limit=10000
            )
            
            if len(baseline_records) < 20 or len(recent_records) < 10:
                logger.info(f"Insufficient data for regression analysis: baseline={len(baseline_records)}, recent={len(recent_records)}")
                return None
            
            # Extract metric values
            baseline_values = self._extract_metric_values(baseline_records, metric)
            recent_values = self._extract_metric_values(recent_records, metric)
            
            if not baseline_values or not recent_values:
                return None
            
            # Compare baseline vs recent (treat baseline as option_a, recent as option_b)
            comparison = self._perform_statistical_test(
                baseline_values, recent_values, 
                StatisticalTest.WELCH_T_TEST, 0.95
            )
            
            baseline_mean = statistics.mean(baseline_values)
            recent_mean = statistics.mean(recent_values)
            
            # For success rate and quality, lower recent values indicate regression
            # For completion time, higher recent values indicate regression
            is_regression = False
            if metric in ["success_rate", "quality_score"]:
                is_regression = recent_mean < baseline_mean and comparison.is_significant
            elif metric == "completion_time":
                is_regression = recent_mean > baseline_mean and comparison.is_significant
            
            if is_regression:
                logger.warning(f"Performance regression detected for {option}: {metric} changed from {baseline_mean:.3f} to {recent_mean:.3f}")
            
            return ComparisonResult(
                option_a=f"{option}_baseline",
                option_b=f"{option}_recent", 
                n_a=len(baseline_values),
                n_b=len(recent_values),
                metric_a=baseline_mean,
                metric_b=recent_mean,
                difference=recent_mean - baseline_mean,
                relative_difference=(recent_mean - baseline_mean) / max(0.001, abs(baseline_mean)),
                test_statistic=comparison.test_statistic,
                p_value=comparison.p_value,
                is_significant=comparison.is_significant,
                confidence_level=0.95,
                confidence_interval=self._calculate_confidence_interval(baseline_values, recent_values, 0.95),
                effect_size=self._calculate_effect_size(baseline_values, recent_values),
                effect_magnitude=self._interpret_effect_size(self._calculate_effect_size(baseline_values, recent_values)),
                test_method=StatisticalTest.WELCH_T_TEST,
                assumptions_met=True,
                warnings=["Regression analysis"] if is_regression else [],
                recommended_winner=f"{option}_baseline" if is_regression else f"{option}_recent"
            )
            
        except Exception as e:
            logger.error(f"Error detecting regression for {option}: {e}")
            return None
    
    def _extract_metric_values(self, records: List[SelectionRecord], metric: str) -> List[float]:
        """Extract metric values from selection records."""
        values = []
        
        for record in records:
            if metric == "success_rate":
                values.append(1.0 if record.success else 0.0)
            elif metric == "completion_time":
                if record.completion_time_ms is not None:
                    values.append(record.completion_time_ms)
            elif metric == "quality_score":
                if record.quality_score is not None:
                    values.append(record.quality_score)
            elif metric == "cost":
                if record.cost is not None:
                    values.append(record.cost)
            elif metric in record.custom_metrics:
                values.append(record.custom_metrics[metric])
        
        return values
    
    def _choose_statistical_test(self, 
                               values_a: List[float], 
                               values_b: List[float], 
                               metric: str) -> StatisticalTest:
        """Choose appropriate statistical test based on data characteristics."""
        # For binary metrics (success_rate), use chi-square or exact tests
        if metric == "success_rate":
            return StatisticalTest.CHI_SQUARE
        
        # For continuous metrics, check normality and equal variance
        n_a, n_b = len(values_a), len(values_b)
        
        if n_a < 30 or n_b < 30:
            # Small samples - use non-parametric test
            return StatisticalTest.MANN_WHITNEY
        
        # Check normality (simplified)
        try:
            _, p_norm_a = stats.normaltest(values_a)
            _, p_norm_b = stats.normaltest(values_b)
            
            if p_norm_a < 0.05 or p_norm_b < 0.05:
                # Non-normal data - use non-parametric test
                return StatisticalTest.MANN_WHITNEY
            
            # Check equal variances
            _, p_var = stats.levene(values_a, values_b)
            
            if p_var < 0.05:
                # Unequal variances - use Welch's t-test
                return StatisticalTest.WELCH_T_TEST
            else:
                # Equal variances - use standard t-test
                return StatisticalTest.T_TEST
                
        except:
            # Default to robust non-parametric test
            return StatisticalTest.MANN_WHITNEY
    
    def _perform_statistical_test(self,
                                values_a: List[float],
                                values_b: List[float], 
                                test_method: StatisticalTest,
                                confidence_level: float) -> SignificanceResult:
        """Perform the specified statistical test."""
        alpha = 1.0 - confidence_level
        
        try:
            if test_method == StatisticalTest.CHI_SQUARE:
                # For binary data (success/failure)
                successes_a = sum(values_a)
                failures_a = len(values_a) - successes_a
                successes_b = sum(values_b)
                failures_b = len(values_b) - failures_b
                
                # Contingency table
                observed = [[successes_a, failures_a], [successes_b, failures_b]]
                chi2, p_value, dof, expected = stats.chi2_contingency(observed)
                
                critical_value = stats.chi2.ppf(1 - alpha, dof)
                is_significant = p_value < alpha
                
                return SignificanceResult(
                    is_significant=is_significant,
                    p_value=p_value,
                    test_statistic=chi2,
                    critical_value=critical_value,
                    degrees_freedom=dof
                )
            
            elif test_method == StatisticalTest.T_TEST:
                t_stat, p_value = stats.ttest_ind(values_a, values_b)
                dof = len(values_a) + len(values_b) - 2
                critical_value = stats.t.ppf(1 - alpha/2, dof)
                
                return SignificanceResult(
                    is_significant=p_value < alpha,
                    p_value=p_value,
                    test_statistic=abs(t_stat),
                    critical_value=critical_value,
                    degrees_freedom=dof
                )
            
            elif test_method == StatisticalTest.WELCH_T_TEST:
                t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
                # Welch-Satterthwaite degrees of freedom (approximation)
                s_a = np.var(values_a, ddof=1)
                s_b = np.var(values_b, ddof=1)
                n_a, n_b = len(values_a), len(values_b)
                dof = ((s_a/n_a + s_b/n_b)**2) / ((s_a/n_a)**2/(n_a-1) + (s_b/n_b)**2/(n_b-1))
                critical_value = stats.t.ppf(1 - alpha/2, dof)
                
                return SignificanceResult(
                    is_significant=p_value < alpha,
                    p_value=p_value,
                    test_statistic=abs(t_stat),
                    critical_value=critical_value,
                    degrees_freedom=int(dof)
                )
            
            elif test_method == StatisticalTest.MANN_WHITNEY:
                u_stat, p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')
                
                return SignificanceResult(
                    is_significant=p_value < alpha,
                    p_value=p_value,
                    test_statistic=u_stat,
                    critical_value=0.0  # No simple critical value for Mann-Whitney
                )
            
            else:
                # Default to t-test
                t_stat, p_value = stats.ttest_ind(values_a, values_b)
                dof = len(values_a) + len(values_b) - 2
                critical_value = stats.t.ppf(1 - alpha/2, dof)
                
                return SignificanceResult(
                    is_significant=p_value < alpha,
                    p_value=p_value,
                    test_statistic=abs(t_stat),
                    critical_value=critical_value,
                    degrees_freedom=dof
                )
        
        except Exception as e:
            logger.error(f"Error performing {test_method.value}: {e}")
            # Return non-significant result as fallback
            return SignificanceResult(
                is_significant=False,
                p_value=1.0,
                test_statistic=0.0,
                critical_value=0.0
            )
    
    def _calculate_effect_size(self, values_a: List[float], values_b: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        try:
            mean_a = statistics.mean(values_a)
            mean_b = statistics.mean(values_b)
            
            # Pooled standard deviation
            var_a = statistics.variance(values_a)
            var_b = statistics.variance(values_b)
            n_a, n_b = len(values_a), len(values_b)
            
            pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
            
            if pooled_std == 0:
                return 0.0
            
            return abs(mean_b - mean_a) / pooled_std
            
        except:
            return 0.0
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def _calculate_confidence_interval(self,
                                     values_a: List[float],
                                     values_b: List[float], 
                                     confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means."""
        try:
            mean_a = statistics.mean(values_a)
            mean_b = statistics.mean(values_b)
            diff = mean_b - mean_a
            
            se_a = statistics.stdev(values_a) / math.sqrt(len(values_a))
            se_b = statistics.stdev(values_b) / math.sqrt(len(values_b))
            se_diff = math.sqrt(se_a**2 + se_b**2)
            
            # Use t-distribution
            dof = len(values_a) + len(values_b) - 2
            alpha = 1.0 - confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, dof)
            
            margin = t_critical * se_diff
            
            return (diff - margin, diff + margin)
            
        except:
            return (0.0, 0.0)
    
    def _check_test_assumptions(self,
                              values_a: List[float],
                              values_b: List[float],
                              test_method: StatisticalTest) -> Tuple[bool, List[str]]:
        """Check if statistical test assumptions are met."""
        warnings = []
        assumptions_met = True
        
        try:
            if test_method in [StatisticalTest.T_TEST, StatisticalTest.WELCH_T_TEST]:
                # Check normality
                if len(values_a) >= 8:  # Minimum for normality test
                    _, p_norm_a = stats.normaltest(values_a)
                    if p_norm_a < 0.05:
                        warnings.append("Group A may not be normally distributed")
                        assumptions_met = False
                
                if len(values_b) >= 8:
                    _, p_norm_b = stats.normaltest(values_b)
                    if p_norm_b < 0.05:
                        warnings.append("Group B may not be normally distributed")
                        assumptions_met = False
                
                # Check for outliers (simple IQR method)
                def has_outliers(values):
                    if len(values) < 4:
                        return False
                    q1 = np.percentile(values, 25)
                    q3 = np.percentile(values, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    return any(v < lower_bound or v > upper_bound for v in values)
                
                if has_outliers(values_a):
                    warnings.append("Group A contains potential outliers")
                
                if has_outliers(values_b):
                    warnings.append("Group B contains potential outliers")
                
                # Check equal variance assumption for standard t-test
                if test_method == StatisticalTest.T_TEST:
                    _, p_var = stats.levene(values_a, values_b)
                    if p_var < 0.05:
                        warnings.append("Groups have unequal variances - consider Welch's t-test")
                        assumptions_met = False
            
            elif test_method == StatisticalTest.CHI_SQUARE:
                # Check minimum expected frequencies
                successes_a = sum(values_a)
                failures_a = len(values_a) - successes_a
                successes_b = sum(values_b) 
                failures_b = len(values_b) - failures_b
                
                total = successes_a + failures_a + successes_b + failures_b
                expected_min = min(
                    (successes_a + successes_b) * (successes_a + failures_a) / total,
                    (successes_a + successes_b) * (successes_b + failures_b) / total,
                    (failures_a + failures_b) * (successes_a + failures_a) / total,
                    (failures_a + failures_b) * (successes_b + failures_b) / total
                )
                
                if expected_min < 5:
                    warnings.append("Expected frequencies too low for chi-square test")
                    assumptions_met = False
        
        except Exception as e:
            warnings.append(f"Could not check assumptions: {e}")
            assumptions_met = False
        
        return assumptions_met, warnings
    
    def _make_recommendation(self,
                           is_significant: bool,
                           effect_size: float,
                           difference: float,
                           option_a: str,
                           option_b: str) -> Tuple[Optional[str], float]:
        """Make recommendation based on statistical analysis."""
        if not is_significant:
            return None, 0.0
        
        # Require meaningful effect size
        if effect_size < 0.2:  # Negligible effect
            return None, 0.1
        
        # Determine winner based on difference direction and magnitude
        confidence = min(0.95, 0.5 + effect_size * 0.3)  # Scale confidence with effect size
        
        if difference > 0:
            winner = option_b
        else:
            winner = option_a
        
        return winner, confidence
    
    def _apply_multiple_testing_correction(self,
                                         p_values: List[float],
                                         confidence_level: float,
                                         method: str) -> float:
        """Apply multiple testing correction."""
        alpha = 1.0 - confidence_level
        
        if method == "bonferroni":
            return alpha / len(p_values)
        elif method == "sidak":
            return 1.0 - (1.0 - alpha)**(1.0/len(p_values))
        elif method == "benjamini_hochberg":
            # Simplified BH procedure
            sorted_p = sorted(p_values)
            for i, p in enumerate(sorted_p):
                if p <= (i + 1) / len(p_values) * alpha:
                    continue
                else:
                    return sorted_p[i-1] if i > 0 else alpha / len(p_values)
            return alpha  # All tests significant
        else:
            # Default to Bonferroni
            return alpha / len(p_values)
    
    def _perform_anova(self, 
                      selection_type: str,
                      options: List[str],
                      metric: str,
                      days: int) -> Tuple[Optional[float], Optional[float]]:
        """Perform one-way ANOVA to test overall differences."""
        try:
            since = datetime.now() - timedelta(days=days)
            all_groups = []
            
            for option in options:
                records = self.selection_history.get_records(
                    selection_type=selection_type,
                    selected_option=option,
                    since=since,
                    only_completed=True,
                    limit=5000
                )
                
                values = self._extract_metric_values(records, metric)
                if len(values) >= 10:  # Minimum group size
                    all_groups.append(values)
            
            if len(all_groups) >= 2:
                f_stat, p_value = stats.f_oneway(*all_groups)
                return f_stat, p_value
            
        except Exception as e:
            logger.error(f"Error performing ANOVA: {e}")
        
        return None, None