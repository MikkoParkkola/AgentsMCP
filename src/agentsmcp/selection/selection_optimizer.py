"""
Multi-Armed Bandit Selection Optimizer

Implements various multi-armed bandit algorithms for intelligent selection
optimization including Thompson Sampling, UCB1, and LinUCB.
"""

import logging
import random
import math
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading

from .selection_history import SelectionHistory, SelectionRecord
from .benchmark_tracker import BenchmarkTracker, SelectionMetrics


logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    EPSILON_GREEDY = "epsilon_greedy"
    THOMPSON_SAMPLING = "thompson_sampling" 
    UCB1 = "ucb1"
    LINUCB = "linucb"
    CONTEXTUAL_THOMPSON = "contextual_thompson"


@dataclass
class BanditArm:
    """Represents a single arm in the multi-armed bandit."""
    
    arm_id: str
    selection_type: str
    
    # Beta distribution parameters for Thompson Sampling
    alpha: float = 1.0  # Success count + 1
    beta: float = 1.0   # Failure count + 1
    
    # Statistics for UCB1
    total_pulls: int = 0
    total_reward: float = 0.0
    
    # LinUCB parameters
    A: np.ndarray = None  # Context covariance matrix
    b: np.ndarray = None  # Context-reward vector
    
    # Performance tracking
    last_reward: float = 0.0
    last_updated: Optional[datetime] = None
    
    # Confidence metrics
    confidence_bound: float = 0.0
    estimated_value: float = 0.5
    
    def __post_init__(self):
        """Initialize numpy arrays if not provided."""
        if self.A is None:
            # Default to 10-dimensional context
            self.A = np.eye(10)
        if self.b is None:
            self.b = np.zeros(10)
    
    def get_thompson_sample(self) -> float:
        """Sample from Beta distribution for Thompson Sampling."""
        return np.random.beta(self.alpha, self.beta)
    
    def get_ucb1_value(self, total_pulls: int) -> float:
        """Calculate UCB1 upper confidence bound."""
        if self.total_pulls == 0:
            return float('inf')  # Unexplored arms have infinite value
        
        exploitation = self.total_reward / self.total_pulls
        exploration = math.sqrt(2 * math.log(total_pulls) / self.total_pulls)
        
        return exploitation + exploration
    
    def get_linucb_value(self, context: np.ndarray, alpha: float = 0.1) -> float:
        """Calculate LinUCB upper confidence bound."""
        try:
            A_inv = np.linalg.inv(self.A)
            theta = A_inv @ self.b
            
            confidence_width = alpha * math.sqrt(context.T @ A_inv @ context)
            
            return theta.T @ context + confidence_width
            
        except np.linalg.LinAlgError:
            # Handle singular matrix
            return 0.5
    
    def update_thompson(self, reward: float):
        """Update Beta parameters for Thompson Sampling."""
        if reward > 0.5:  # Success
            self.alpha += reward
        else:  # Failure
            self.beta += (1.0 - reward)
        
        self.last_reward = reward
        self.last_updated = datetime.now()
        self.estimated_value = self.alpha / (self.alpha + self.beta)
    
    def update_ucb1(self, reward: float):
        """Update statistics for UCB1."""
        self.total_pulls += 1
        self.total_reward += reward
        self.last_reward = reward
        self.last_updated = datetime.now()
        self.estimated_value = self.total_reward / self.total_pulls
    
    def update_linucb(self, context: np.ndarray, reward: float):
        """Update LinUCB parameters."""
        self.A += np.outer(context, context)
        self.b += reward * context
        self.total_pulls += 1
        self.last_reward = reward
        self.last_updated = datetime.now()
        
        # Update estimated value
        try:
            A_inv = np.linalg.inv(self.A)
            theta = A_inv @ self.b
            self.estimated_value = theta.T @ context
        except np.linalg.LinAlgError:
            self.estimated_value = 0.5


class SelectionOptimizer:
    """
    Multi-armed bandit optimizer for intelligent selection decisions.
    
    Balances exploration of new options with exploitation of known good options
    using various bandit algorithms.
    """
    
    def __init__(self,
                 selection_history: SelectionHistory,
                 benchmark_tracker: BenchmarkTracker,
                 strategy: OptimizationStrategy = OptimizationStrategy.THOMPSON_SAMPLING,
                 exploration_rate: float = 0.1,
                 context_dim: int = 10):
        """
        Initialize selection optimizer.
        
        Args:
            selection_history: Historical selection data
            benchmark_tracker: Real-time performance metrics
            strategy: Optimization strategy to use
            exploration_rate: Initial exploration rate (for epsilon-greedy)
            context_dim: Dimensionality for contextual bandits
        """
        self.selection_history = selection_history
        self.benchmark_tracker = benchmark_tracker
        self.strategy = strategy
        self.exploration_rate = exploration_rate
        self.context_dim = context_dim
        
        # Bandit arms by (selection_type, option_name)
        self.arms: Dict[Tuple[str, str], BanditArm] = {}
        self.arms_lock = threading.RLock()
        
        # Strategy-specific parameters
        self.epsilon_decay_rate = 0.995  # Decay exploration over time
        self.linucb_alpha = 0.1  # LinUCB confidence parameter
        self.min_samples_for_optimization = 10
        
        # Performance tracking
        self.total_selections_made = 0
        self.exploration_selections = 0
        self.exploitation_selections = 0
        
        logger.info(f"SelectionOptimizer initialized with strategy: {strategy.value}")
    
    def select_option(self,
                     selection_type: str,
                     available_options: List[str],
                     task_context: Dict[str, Any] = None,
                     force_exploration: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Select the optimal option using the configured strategy.
        
        Args:
            selection_type: Type of selection needed
            available_options: List of available options
            task_context: Context for contextual bandits
            force_exploration: Force exploration regardless of strategy
            
        Returns:
            Tuple of (selected_option, selection_metadata)
        """
        if not available_options:
            raise ValueError("No available options provided")
        
        # Initialize arms for new options
        self._ensure_arms_exist(selection_type, available_options)
        
        # Get relevant arms
        arms = {
            option: self.arms[(selection_type, option)]
            for option in available_options
            if (selection_type, option) in self.arms
        }
        
        if not arms:
            # Fallback to random selection
            selected = random.choice(available_options)
            return selected, {'method': 'random_fallback', 'reason': 'no_arms_found'}
        
        # Apply selection strategy
        if force_exploration or self.strategy == OptimizationStrategy.EPSILON_GREEDY:
            selected, metadata = self._epsilon_greedy_selection(arms, available_options)
        elif self.strategy == OptimizationStrategy.THOMPSON_SAMPLING:
            selected, metadata = self._thompson_sampling_selection(arms)
        elif self.strategy == OptimizationStrategy.UCB1:
            selected, metadata = self._ucb1_selection(arms)
        elif self.strategy == OptimizationStrategy.LINUCB:
            context = self._extract_context(task_context)
            selected, metadata = self._linucb_selection(arms, context)
        else:
            # Default to Thompson Sampling
            selected, metadata = self._thompson_sampling_selection(arms)
        
        # Update counters
        self.total_selections_made += 1
        if metadata.get('exploration'):
            self.exploration_selections += 1
        else:
            self.exploitation_selections += 1
        
        # Decay exploration rate
        self.exploration_rate *= self.epsilon_decay_rate
        self.exploration_rate = max(0.01, self.exploration_rate)  # Minimum 1%
        
        logger.debug(f"Selected {selected} using {self.strategy.value}: {metadata}")
        return selected, metadata
    
    def update_outcome(self,
                      selection_type: str,
                      selected_option: str,
                      outcome: SelectionRecord):
        """
        Update the optimizer with the outcome of a selection.
        
        Args:
            selection_type: Type of selection that was made
            selected_option: Option that was selected
            outcome: Complete outcome record
        """
        arm_key = (selection_type, selected_option)
        
        if arm_key not in self.arms:
            logger.warning(f"No arm found for {arm_key}")
            return
        
        # Calculate reward from outcome
        reward = self._calculate_reward(outcome)
        
        with self.arms_lock:
            arm = self.arms[arm_key]
            
            # Update based on strategy
            if self.strategy == OptimizationStrategy.THOMPSON_SAMPLING:
                arm.update_thompson(reward)
            elif self.strategy == OptimizationStrategy.UCB1:
                arm.update_ucb1(reward)
            elif self.strategy == OptimizationStrategy.LINUCB:
                context = self._extract_context(outcome.task_context)
                arm.update_linucb(context, reward)
            else:
                # Default update
                arm.update_thompson(reward)
        
        logger.debug(f"Updated arm {arm_key} with reward {reward:.3f}")
    
    def get_arm_rankings(self, 
                        selection_type: str,
                        available_options: List[str] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Get current rankings of arms for a selection type.
        
        Args:
            selection_type: Type of selection to rank
            available_options: Filter to these options only
            
        Returns:
            List of (option, estimated_value, metadata) tuples, sorted by value
        """
        rankings = []
        
        with self.arms_lock:
            for (sel_type, option), arm in self.arms.items():
                if sel_type != selection_type:
                    continue
                
                if available_options and option not in available_options:
                    continue
                
                metadata = {
                    'total_pulls': arm.total_pulls,
                    'last_reward': arm.last_reward,
                    'confidence_bound': arm.confidence_bound,
                    'last_updated': arm.last_updated.isoformat() if arm.last_updated else None
                }
                
                # Add strategy-specific information
                if self.strategy == OptimizationStrategy.THOMPSON_SAMPLING:
                    metadata.update({
                        'alpha': arm.alpha,
                        'beta': arm.beta,
                        'thompson_sample': arm.get_thompson_sample()
                    })
                
                rankings.append((option, arm.estimated_value, metadata))
        
        # Sort by estimated value descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get exploration vs exploitation statistics."""
        total = max(1, self.total_selections_made)
        
        return {
            'total_selections': self.total_selections_made,
            'exploration_selections': self.exploration_selections,
            'exploitation_selections': self.exploitation_selections,
            'exploration_rate_current': self.exploration_rate,
            'exploration_percentage': (self.exploration_selections / total) * 100,
            'exploitation_percentage': (self.exploitation_selections / total) * 100,
            'strategy': self.strategy.value
        }
    
    def reset_arm(self, selection_type: str, option_name: str):
        """Reset an arm to initial state (useful for relearning)."""
        arm_key = (selection_type, option_name)
        
        if arm_key in self.arms:
            with self.arms_lock:
                self.arms[arm_key] = BanditArm(
                    arm_id=option_name,
                    selection_type=selection_type
                )
            
            logger.info(f"Reset arm {arm_key}")
    
    def _ensure_arms_exist(self, selection_type: str, options: List[str]):
        """Ensure bandit arms exist for all options."""
        with self.arms_lock:
            for option in options:
                arm_key = (selection_type, option)
                if arm_key not in self.arms:
                    self.arms[arm_key] = BanditArm(
                        arm_id=option,
                        selection_type=selection_type
                    )
                    
                    # Initialize with historical data if available
                    self._initialize_arm_from_history(selection_type, option)
    
    def _initialize_arm_from_history(self, selection_type: str, option: str):
        """Initialize arm with historical performance data."""
        try:
            # Get recent performance metrics
            metrics = self.benchmark_tracker.get_metrics(selection_type, option)
            arm_key = (selection_type, option)
            
            if arm_key in metrics:
                metric = metrics[arm_key]
                
                if metric.total_selections >= self.min_samples_for_optimization:
                    arm = self.arms[arm_key]
                    
                    # Initialize based on historical success rate
                    successes = metric.successful_selections
                    failures = metric.total_selections - successes
                    
                    if self.strategy == OptimizationStrategy.THOMPSON_SAMPLING:
                        arm.alpha = successes + 1
                        arm.beta = failures + 1
                        arm.estimated_value = arm.alpha / (arm.alpha + arm.beta)
                    
                    elif self.strategy == OptimizationStrategy.UCB1:
                        arm.total_pulls = metric.total_selections
                        arm.total_reward = metric.success_rate * metric.total_selections
                        arm.estimated_value = metric.success_rate
                    
                    logger.debug(f"Initialized arm {arm_key} with {metric.total_selections} historical samples")
        
        except Exception as e:
            logger.warning(f"Failed to initialize arm from history: {e}")
    
    def _epsilon_greedy_selection(self, 
                                arms: Dict[str, BanditArm],
                                available_options: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Epsilon-greedy selection strategy."""
        if random.random() < self.exploration_rate:
            # Explore: choose randomly
            selected = random.choice(available_options)
            return selected, {
                'method': 'epsilon_greedy',
                'exploration': True,
                'epsilon': self.exploration_rate
            }
        else:
            # Exploit: choose best known option
            best_option = max(arms.items(), key=lambda x: x[1].estimated_value)[0]
            return best_option, {
                'method': 'epsilon_greedy',
                'exploration': False,
                'epsilon': self.exploration_rate
            }
    
    def _thompson_sampling_selection(self, arms: Dict[str, BanditArm]) -> Tuple[str, Dict[str, Any]]:
        """Thompson Sampling selection strategy."""
        samples = {}
        for option, arm in arms.items():
            samples[option] = arm.get_thompson_sample()
        
        selected = max(samples.items(), key=lambda x: x[1])[0]
        
        return selected, {
            'method': 'thompson_sampling',
            'samples': samples,
            'exploration': samples[selected] < arms[selected].estimated_value
        }
    
    def _ucb1_selection(self, arms: Dict[str, BanditArm]) -> Tuple[str, Dict[str, Any]]:
        """UCB1 selection strategy."""
        total_pulls = sum(arm.total_pulls for arm in arms.values())
        
        if total_pulls == 0:
            # First selection, choose randomly
            selected = random.choice(list(arms.keys()))
            return selected, {'method': 'ucb1', 'reason': 'first_selection'}
        
        ucb_values = {}
        for option, arm in arms.items():
            ucb_values[option] = arm.get_ucb1_value(total_pulls)
        
        selected = max(ucb_values.items(), key=lambda x: x[1])[0]
        
        return selected, {
            'method': 'ucb1',
            'ucb_values': ucb_values,
            'exploration': arms[selected].total_pulls < math.sqrt(total_pulls)
        }
    
    def _linucb_selection(self, 
                         arms: Dict[str, BanditArm],
                         context: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """LinUCB contextual bandit selection."""
        linucb_values = {}
        for option, arm in arms.items():
            linucb_values[option] = arm.get_linucb_value(context, self.linucb_alpha)
        
        selected = max(linucb_values.items(), key=lambda x: x[1])[0]
        
        return selected, {
            'method': 'linucb',
            'linucb_values': linucb_values,
            'context_norm': float(np.linalg.norm(context)),
            'exploration': True  # LinUCB inherently explores
        }
    
    def _extract_context(self, task_context: Dict[str, Any]) -> np.ndarray:
        """Extract numerical context vector from task context."""
        if not task_context:
            return np.ones(self.context_dim)  # Default context
        
        # Simple context extraction (can be made more sophisticated)
        context_features = []
        
        # Time-based features
        now = datetime.now()
        context_features.extend([
            now.hour / 24.0,  # Hour of day
            now.weekday() / 7.0,  # Day of week
            (now.timestamp() % 3600) / 3600.0  # Position within hour
        ])
        
        # Task complexity
        complexity = task_context.get('complexity', 'medium')
        complexity_map = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        context_features.append(complexity_map.get(complexity, 0.6))
        
        # Task type features
        task_type = task_context.get('task_type', 'general')
        task_types = ['coding', 'reasoning', 'general', 'multimodal']
        for t in task_types:
            context_features.append(1.0 if task_type == t else 0.0)
        
        # Pad or truncate to desired dimension
        while len(context_features) < self.context_dim:
            context_features.append(0.0)
        
        context_features = context_features[:self.context_dim]
        
        return np.array(context_features)
    
    def _calculate_reward(self, outcome: SelectionRecord) -> float:
        """Calculate reward from selection outcome (0.0 - 1.0)."""
        if outcome.success is None:
            return 0.5  # Neutral for incomplete outcomes
        
        # Base reward from success/failure
        base_reward = 1.0 if outcome.success else 0.0
        
        # Adjust based on quality score
        if outcome.quality_score is not None:
            quality_weight = 0.3
            base_reward = (1.0 - quality_weight) * base_reward + quality_weight * outcome.quality_score
        
        # Adjust based on completion time (faster is better)
        if outcome.completion_time_ms is not None:
            # Assume 30 seconds is baseline, scale accordingly
            baseline_ms = 30000
            time_factor = min(1.0, baseline_ms / max(1000, outcome.completion_time_ms))
            time_weight = 0.2
            base_reward = (1.0 - time_weight) * base_reward + time_weight * time_factor
        
        # Adjust based on user feedback
        if outcome.user_feedback is not None:
            feedback_score = (outcome.user_feedback + 1) / 2.0  # Map -1,0,1 to 0,0.5,1
            feedback_weight = 0.1
            base_reward = (1.0 - feedback_weight) * base_reward + feedback_weight * feedback_score
        
        return max(0.0, min(1.0, base_reward))
    
    def export_arms_state(self) -> Dict[str, Any]:
        """Export current state of all arms for persistence."""
        arms_data = {}
        
        with self.arms_lock:
            for (sel_type, option), arm in self.arms.items():
                key = f"{sel_type}:{option}"
                arms_data[key] = {
                    'selection_type': sel_type,
                    'arm_id': option,
                    'alpha': float(arm.alpha),
                    'beta': float(arm.beta),
                    'total_pulls': arm.total_pulls,
                    'total_reward': arm.total_reward,
                    'estimated_value': arm.estimated_value,
                    'last_reward': arm.last_reward,
                    'last_updated': arm.last_updated.isoformat() if arm.last_updated else None
                }
        
        return {
            'arms': arms_data,
            'strategy': self.strategy.value,
            'exploration_rate': self.exploration_rate,
            'total_selections': self.total_selections_made,
            'exported_at': datetime.now().isoformat()
        }
    
    def import_arms_state(self, state_data: Dict[str, Any]) -> bool:
        """Import arms state from exported data."""
        try:
            with self.arms_lock:
                self.arms.clear()
                
                arms_data = state_data.get('arms', {})
                for key, arm_data in arms_data.items():
                    sel_type = arm_data['selection_type']
                    option = arm_data['arm_id']
                    arm_key = (sel_type, option)
                    
                    arm = BanditArm(
                        arm_id=option,
                        selection_type=sel_type,
                        alpha=arm_data['alpha'],
                        beta=arm_data['beta'],
                        total_pulls=arm_data['total_pulls'],
                        total_reward=arm_data['total_reward']
                    )
                    
                    arm.estimated_value = arm_data['estimated_value']
                    arm.last_reward = arm_data['last_reward']
                    if arm_data['last_updated']:
                        arm.last_updated = datetime.fromisoformat(arm_data['last_updated'])
                    
                    self.arms[arm_key] = arm
                
                # Update optimizer state
                self.exploration_rate = state_data.get('exploration_rate', self.exploration_rate)
                self.total_selections_made = state_data.get('total_selections', 0)
            
            logger.info(f"Imported state for {len(self.arms)} arms")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import arms state: {e}")
            return False