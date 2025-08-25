"""Model selection optimization for cost-effectiveness."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .tracker import CostTracker


@dataclass
class ModelRecommendation:
    """Recommendation for model selection."""
    
    model: str
    provider: str
    confidence: float
    reasoning: str
    estimated_cost: float
    estimated_quality: float


class ModelOptimizer:
    """Optimizes model selection for cost-effectiveness."""
    
    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker
        
        # Mock model data - in real implementation would be dynamic
        self.model_costs = {
            "gpt-4": 0.00003,
            "gpt-3.5-turbo": 0.0000015, 
            "claude-3-sonnet": 0.000003,
            "gpt-oss:20b": 0.0,  # Free local model
        }
        
        self.model_quality = {
            "gpt-4": 0.95,
            "gpt-3.5-turbo": 0.85,
            "claude-3-sonnet": 0.92,
            "gpt-oss:20b": 0.80,
        }
    
    def recommend_model(self, 
                       task_type: str = "general",
                       quality_priority: float = 0.5,
                       cost_priority: float = 0.5) -> ModelRecommendation:
        """Recommend optimal model based on task and priorities."""
        
        best_score = -1
        best_model = None
        
        for model, cost_per_token in self.model_costs.items():
            quality = self.model_quality.get(model, 0.5)
            
            # Normalize scores (lower cost is better, higher quality is better)
            cost_score = 1.0 - (cost_per_token / max(self.model_costs.values())) if max(self.model_costs.values()) > 0 else 1.0
            quality_score = quality
            
            # Weighted combination
            combined_score = (cost_priority * cost_score) + (quality_priority * quality_score)
            
            if combined_score > best_score:
                best_score = combined_score
                best_model = model
        
        if best_model:
            provider = "ollama" if "ollama" in best_model or "gpt-oss" in best_model else "openai"
            return ModelRecommendation(
                model=best_model,
                provider=provider,
                confidence=best_score,
                reasoning=f"Best balance of cost ({cost_priority:.1f}) and quality ({quality_priority:.1f})",
                estimated_cost=self.model_costs[best_model],
                estimated_quality=self.model_quality[best_model]
            )
        
        # Fallback
        return ModelRecommendation(
            model="gpt-3.5-turbo",
            provider="openai",
            confidence=0.7,
            reasoning="Fallback recommendation",
            estimated_cost=self.model_costs["gpt-3.5-turbo"],
            estimated_quality=self.model_quality["gpt-3.5-turbo"]
        )