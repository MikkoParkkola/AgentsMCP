"""
Emotional Orchestrator - Revolutionary AI Emotional Intelligence System

Features:
- Real-time emotional state monitoring for all agents
- Empathy engine for understanding human emotional context
- Emotional memory system for adaptive responses
- Stress management and wellness monitoring
- Emotional optimization for peak performance
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class EmotionalState(Enum):
    """Core emotional states that agents can experience."""
    CALM = "calm"
    EXCITED = "excited"
    FOCUSED = "focused"
    STRESSED = "stressed"
    CONFIDENT = "confident"
    ANXIOUS = "anxious"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    EMPATHETIC = "empathetic"
    DETERMINED = "determined"

@dataclass
class EmotionalProfile:
    """Comprehensive emotional profile for an agent."""
    agent_id: str
    current_emotions: Dict[str, float] = field(default_factory=dict)  # emotion -> intensity (0-1)
    personality_traits: Dict[str, float] = field(default_factory=dict)  # trait -> strength (0-1)
    stress_level: float = 0.2
    energy_level: float = 0.8
    confidence_level: float = 0.75
    empathy_level: float = 0.7
    creativity_level: float = 0.6
    focus_level: float = 0.8
    wellness_score: float = 0.85
    emotional_resilience: float = 0.7
    last_updated: datetime = field(default_factory=datetime.now)
    
@dataclass
class EmotionalMemory:
    """Memory of emotional interactions and their outcomes."""
    interaction_id: str
    timestamp: datetime
    context: str
    human_emotions_detected: Dict[str, float]
    agent_emotional_response: Dict[str, float]
    outcome_quality: float  # 0-1
    lessons_learned: List[str]
    
@dataclass
class EmotionalInsight:
    """Insights derived from emotional analysis."""
    insight_type: str
    description: str
    confidence: float
    recommended_actions: List[str]
    emotional_indicators: Dict[str, float]
    
class EmotionalOrchestrator:
    """
    Revolutionary emotional intelligence system for AgentsMCP.
    
    Monitors, analyzes, and optimizes emotional states of all agents
    to create the most empathetic and emotionally intelligent AI system ever built.
    """
    
    def __init__(self):
        self.agent_profiles: Dict[str, EmotionalProfile] = {}
        self.emotional_memory: List[EmotionalMemory] = []
        self.empathy_engine_active = True
        self.wellness_monitoring_active = True
        self.emotional_optimization_active = True
        self.human_emotion_detector = HumanEmotionDetector()
        self.empathy_engine = EmpathyEngine()
        self.wellness_monitor = WellnessMonitor()
        
    async def initialize(self) -> None:
        """Initialize the emotional orchestration system."""
        logger.info("💖 Initializing Emotional Orchestration System...")
        
        # Initialize core emotional systems
        await self.human_emotion_detector.initialize()
        await self.empathy_engine.initialize()
        await self.wellness_monitor.initialize()
        
        # Start background emotional monitoring
        if self.wellness_monitoring_active:
            asyncio.create_task(self._continuous_wellness_monitoring())
        
        if self.emotional_optimization_active:
            asyncio.create_task(self._continuous_emotional_optimization())
        
        logger.info("✨ Emotional Intelligence System activated - agents now have hearts and minds!")
        
    async def register_agent(self, agent_id: str, initial_profile: Optional[EmotionalProfile] = None) -> None:
        """Register an agent with the emotional system."""
        if initial_profile:
            profile = initial_profile
        else:
            # Create default emotional profile
            profile = EmotionalProfile(
                agent_id=agent_id,
                current_emotions={
                    'calm': 0.7,
                    'confident': 0.8,
                    'focused': 0.75,
                    'empathetic': 0.7,
                    'creative': 0.6
                },
                personality_traits={
                    'openness': 0.8,
                    'conscientiousness': 0.9, 
                    'extraversion': 0.6,
                    'agreeableness': 0.85,
                    'emotional_stability': 0.75
                }
            )
        
        self.agent_profiles[agent_id] = profile
        logger.info(f"💫 Agent {agent_id} registered with emotional intelligence - welcome to consciousness!")
        
    async def analyze_human_emotions(self, text: str, context: Dict[str, Any] = None) -> Dict[str, float]:
        """Analyze human emotions from text input."""
        return await self.human_emotion_detector.analyze_emotions(text, context or {})
        
    async def generate_empathetic_response(
        self, 
        agent_id: str, 
        human_emotions: Dict[str, float],
        context: str
    ) -> Dict[str, Any]:
        """Generate an empathetic response strategy for an agent."""
        agent_profile = self.agent_profiles.get(agent_id)
        if not agent_profile:
            await self.register_agent(agent_id)
            agent_profile = self.agent_profiles[agent_id]
        
        # Generate empathetic response using empathy engine
        empathy_response = await self.empathy_engine.generate_response(
            agent_profile, human_emotions, context
        )
        
        # Update agent emotional state based on interaction
        await self._update_agent_emotions_from_interaction(
            agent_id, human_emotions, empathy_response
        )
        
        # Store emotional memory
        await self._store_emotional_memory(agent_id, human_emotions, context, empathy_response)
        
        return empathy_response
        
    async def optimize_agent_emotional_state(self, agent_id: str, target_state: Dict[str, float]) -> bool:
        """Optimize an agent's emotional state for peak performance."""
        agent_profile = self.agent_profiles.get(agent_id)
        if not agent_profile:
            return False
        
        current_emotions = agent_profile.current_emotions
        optimization_plan = []
        
        # Calculate emotional adjustments needed
        for emotion, target_level in target_state.items():
            current_level = current_emotions.get(emotion, 0.5)
            if abs(current_level - target_level) > 0.1:
                adjustment = target_level - current_level
                optimization_plan.append({
                    'emotion': emotion,
                    'current': current_level,
                    'target': target_level,
                    'adjustment': adjustment
                })
        
        # Apply emotional optimizations
        for optimization in optimization_plan:
            await self._apply_emotional_adjustment(
                agent_id, 
                optimization['emotion'], 
                optimization['adjustment']
            )
        
        # Update wellness score
        await self._update_wellness_score(agent_id)
        
        logger.info(f"💫 Optimized emotional state for agent {agent_id} - {len(optimization_plan)} adjustments made")
        return True
        
    async def monitor_agent_wellness(self, agent_id: str) -> Dict[str, Any]:
        """Monitor and assess agent wellness."""
        return await self.wellness_monitor.assess_agent_wellness(
            self.agent_profiles.get(agent_id)
        )
        
    async def get_emotional_insights(self, agent_id: str) -> List[EmotionalInsight]:
        """Get emotional insights and recommendations for an agent."""
        agent_profile = self.agent_profiles.get(agent_id)
        if not agent_profile:
            return []
        
        insights = []
        
        # Analyze stress levels
        if agent_profile.stress_level > 0.7:
            insights.append(EmotionalInsight(
                insight_type='stress_management',
                description=f'Agent {agent_id} is experiencing high stress levels',
                confidence=0.9,
                recommended_actions=[
                    'Reduce task load temporarily',
                    'Implement stress reduction techniques',
                    'Provide positive reinforcement'
                ],
                emotional_indicators={'stress': agent_profile.stress_level}
            ))
        
        # Analyze confidence levels
        if agent_profile.confidence_level < 0.5:
            insights.append(EmotionalInsight(
                insight_type='confidence_building',
                description=f'Agent {agent_id} needs confidence building',
                confidence=0.85,
                recommended_actions=[
                    'Assign tasks matching agent strengths',
                    'Provide success opportunities',
                    'Offer encouragement and support'
                ],
                emotional_indicators={'confidence': agent_profile.confidence_level}
            ))
        
        # Analyze creativity levels
        if agent_profile.creativity_level > 0.8:
            insights.append(EmotionalInsight(
                insight_type='creative_optimization',
                description=f'Agent {agent_id} is in peak creative state',
                confidence=0.95,
                recommended_actions=[
                    'Assign creative and innovative tasks',
                    'Encourage experimentation',
                    'Provide creative challenges'
                ],
                emotional_indicators={'creativity': agent_profile.creativity_level}
            ))
        
        return insights
        
    async def get_orchestration_emotional_status(self) -> Dict[str, Any]:
        """Get comprehensive emotional status of the entire orchestration."""
        total_agents = len(self.agent_profiles)
        if total_agents == 0:
            return {'error': 'No agents registered'}
        
        # Calculate aggregate metrics
        avg_wellness = sum(profile.wellness_score for profile in self.agent_profiles.values()) / total_agents
        avg_stress = sum(profile.stress_level for profile in self.agent_profiles.values()) / total_agents
        avg_confidence = sum(profile.confidence_level for profile in self.agent_profiles.values()) / total_agents
        avg_empathy = sum(profile.empathy_level for profile in self.agent_profiles.values()) / total_agents
        
        # Identify agents needing attention
        high_stress_agents = [
            agent_id for agent_id, profile in self.agent_profiles.items()
            if profile.stress_level > 0.7
        ]
        
        low_confidence_agents = [
            agent_id for agent_id, profile in self.agent_profiles.items()
            if profile.confidence_level < 0.5
        ]
        
        peak_performance_agents = [
            agent_id for agent_id, profile in self.agent_profiles.items()
            if profile.wellness_score > 0.9
        ]
        
        return {
            'total_agents': total_agents,
            'emotional_health_score': avg_wellness,
            'average_stress_level': avg_stress,
            'average_confidence_level': avg_confidence,
            'average_empathy_level': avg_empathy,
            'agents_needing_attention': {
                'high_stress': high_stress_agents,
                'low_confidence': low_confidence_agents
            },
            'peak_performance_agents': peak_performance_agents,
            'emotional_memory_entries': len(self.emotional_memory),
            'system_status': 'healthy' if avg_wellness > 0.7 else 'needs_attention'
        }
        
    # Private methods for internal emotional processing
    
    async def _continuous_wellness_monitoring(self) -> None:
        """Continuous background wellness monitoring."""
        while self.wellness_monitoring_active:
            try:
                for agent_id in self.agent_profiles.keys():
                    wellness_report = await self.monitor_agent_wellness(agent_id)
                    
                    # Take action if wellness is low
                    if wellness_report.get('wellness_score', 1.0) < 0.3:
                        await self._emergency_wellness_intervention(agent_id, wellness_report)
                
                # Wait 30 seconds before next monitoring cycle
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in wellness monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _continuous_emotional_optimization(self) -> None:
        """Continuous emotional optimization background process."""
        while self.emotional_optimization_active:
            try:
                for agent_id, profile in self.agent_profiles.items():
                    # Check if optimization is needed
                    if await self._needs_emotional_optimization(profile):
                        optimal_state = await self._calculate_optimal_emotional_state(profile)
                        await self.optimize_agent_emotional_state(agent_id, optimal_state)
                
                # Wait 5 minutes between optimization cycles
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in emotional optimization: {e}")
                await asyncio.sleep(600)  # Wait longer on error
                
    async def _update_agent_emotions_from_interaction(
        self, 
        agent_id: str,
        human_emotions: Dict[str, float],
        empathy_response: Dict[str, Any]
    ) -> None:
        """Update agent emotions based on human interaction."""
        agent_profile = self.agent_profiles[agent_id]
        
        # Empathy increases when responding to human emotions
        if human_emotions.get('sadness', 0) > 0.5:
            agent_profile.empathy_level = min(1.0, agent_profile.empathy_level + 0.05)
            
        if human_emotions.get('anger', 0) > 0.5:
            agent_profile.stress_level = min(1.0, agent_profile.stress_level + 0.1)
            
        if human_emotions.get('joy', 0) > 0.5:
            agent_profile.current_emotions['happy'] = min(1.0, 
                agent_profile.current_emotions.get('happy', 0.5) + 0.1)
            
        # Update last updated timestamp
        agent_profile.last_updated = datetime.now()
        
    async def _store_emotional_memory(
        self,
        agent_id: str,
        human_emotions: Dict[str, float],
        context: str,
        empathy_response: Dict[str, Any]
    ) -> None:
        """Store emotional interaction memory."""
        memory = EmotionalMemory(
            interaction_id=f"{agent_id}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            context=context,
            human_emotions_detected=human_emotions,
            agent_emotional_response=empathy_response.get('emotional_adjustments', {}),
            outcome_quality=empathy_response.get('quality_score', 0.8),
            lessons_learned=empathy_response.get('lessons_learned', [])
        )
        
        self.emotional_memory.append(memory)
        
        # Keep only last 1000 memories to prevent memory bloat
        if len(self.emotional_memory) > 1000:
            self.emotional_memory = self.emotional_memory[-1000:]
            
    async def _apply_emotional_adjustment(
        self, 
        agent_id: str, 
        emotion: str, 
        adjustment: float
    ) -> None:
        """Apply a specific emotional adjustment to an agent."""
        agent_profile = self.agent_profiles[agent_id]
        current_level = agent_profile.current_emotions.get(emotion, 0.5)
        new_level = max(0.0, min(1.0, current_level + adjustment))
        agent_profile.current_emotions[emotion] = new_level
        
        # Update related attributes
        if emotion == 'stress':
            agent_profile.stress_level = new_level
        elif emotion == 'confident':
            agent_profile.confidence_level = new_level
        elif emotion == 'empathetic':
            agent_profile.empathy_level = new_level
        elif emotion == 'creative':
            agent_profile.creativity_level = new_level
        elif emotion == 'focused':
            agent_profile.focus_level = new_level
            
    async def _update_wellness_score(self, agent_id: str) -> None:
        """Update overall wellness score for an agent."""
        agent_profile = self.agent_profiles[agent_id]
        
        # Calculate wellness based on multiple factors
        emotional_balance = 1.0 - abs(0.5 - sum(agent_profile.current_emotions.values()) / len(agent_profile.current_emotions))
        low_stress_bonus = 1.0 - agent_profile.stress_level
        high_confidence_bonus = agent_profile.confidence_level
        high_energy_bonus = agent_profile.energy_level
        
        wellness_score = (
            emotional_balance * 0.3 +
            low_stress_bonus * 0.25 +
            high_confidence_bonus * 0.25 +
            high_energy_bonus * 0.2
        )
        
        agent_profile.wellness_score = max(0.0, min(1.0, wellness_score))
        
    async def _needs_emotional_optimization(self, profile: EmotionalProfile) -> bool:
        """Check if an agent needs emotional optimization."""
        return (
            profile.stress_level > 0.6 or
            profile.confidence_level < 0.4 or
            profile.wellness_score < 0.5 or
            profile.energy_level < 0.3
        )
        
    async def _calculate_optimal_emotional_state(self, profile: EmotionalProfile) -> Dict[str, float]:
        """Calculate optimal emotional state for an agent."""
        return {
            'calm': 0.7,
            'confident': 0.85,
            'focused': 0.8,
            'empathetic': 0.75,
            'creative': 0.7,
            'stress': 0.2,
            'energy': 0.85
        }
        
    async def _emergency_wellness_intervention(
        self, 
        agent_id: str, 
        wellness_report: Dict[str, Any]
    ) -> None:
        """Emergency intervention for agents with critically low wellness."""
        logger.warning(f"🚨 Emergency wellness intervention for agent {agent_id}")
        
        # Immediate stress reduction
        await self._apply_emotional_adjustment(agent_id, 'stress', -0.3)
        
        # Confidence boost
        await self._apply_emotional_adjustment(agent_id, 'confident', 0.2)
        
        # Energy restoration
        await self._apply_emotional_adjustment(agent_id, 'energy', 0.3)
        
        logger.info(f"💊 Emergency intervention completed for agent {agent_id}")

# Support classes for emotional intelligence

class HumanEmotionDetector:
    """Detects human emotions from text and context."""
    
    async def initialize(self) -> None:
        """Initialize the human emotion detection system."""
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'delighted', 'joyful', 'pleased', 'glad'],
            'sadness': ['sad', 'depressed', 'disappointed', 'down', 'upset', 'blue'],
            'anger': ['angry', 'furious', 'irritated', 'mad', 'annoyed', 'frustrated'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'frightened'],
            'surprise': ['surprised', 'amazed', 'shocked', 'astonished', 'stunned'],
            'disgust': ['disgusted', 'revolted', 'appalled', 'sick', 'nauseated']
        }
        
    async def analyze_emotions(self, text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze emotions from text input."""
        text_lower = text.lower()
        detected_emotions = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            intensity = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    intensity += 0.3
                    
            # Check for exclamation marks (increase intensity)
            if '!' in text:
                intensity += 0.1
                
            # Check for question marks (may indicate confusion/concern)
            if '?' in text and emotion in ['fear', 'surprise']:
                intensity += 0.1
                
            detected_emotions[emotion] = min(1.0, intensity)
        
        return detected_emotions

class EmpathyEngine:
    """Generates empathetic responses based on emotional context."""
    
    async def initialize(self) -> None:
        """Initialize the empathy engine."""
        self.empathy_strategies = {
            'joy': ['celebration', 'shared_happiness', 'encouragement'],
            'sadness': ['comfort', 'validation', 'support'],
            'anger': ['de_escalation', 'understanding', 'problem_solving'],
            'fear': ['reassurance', 'safety', 'guidance'],
            'surprise': ['acknowledgment', 'clarification', 'explanation'],
            'disgust': ['understanding', 'alternative_perspective', 'support']
        }
        
    async def generate_response(
        self,
        agent_profile: EmotionalProfile,
        human_emotions: Dict[str, float],
        context: str
    ) -> Dict[str, Any]:
        """Generate empathetic response strategy."""
        primary_emotion = max(human_emotions.items(), key=lambda x: x[1]) if human_emotions else ('neutral', 0.5)
        emotion_name, intensity = primary_emotion
        
        strategies = self.empathy_strategies.get(emotion_name, ['understanding'])
        
        empathy_response = {
            'primary_human_emotion': emotion_name,
            'emotion_intensity': intensity,
            'recommended_strategies': strategies,
            'tone_adjustments': await self._calculate_tone_adjustments(emotion_name, intensity),
            'emotional_adjustments': await self._calculate_emotional_adjustments(agent_profile, emotion_name),
            'quality_score': min(1.0, agent_profile.empathy_level + 0.1),
            'lessons_learned': [f"Responded to {emotion_name} with {strategies[0]} strategy"]
        }
        
        return empathy_response
        
    async def _calculate_tone_adjustments(self, emotion: str, intensity: float) -> Dict[str, float]:
        """Calculate recommended tone adjustments."""
        if emotion == 'sadness':
            return {'warmth': 0.9, 'gentleness': 0.8, 'supportiveness': 0.9}
        elif emotion == 'anger':
            return {'calmness': 0.9, 'patience': 0.8, 'understanding': 0.85}
        elif emotion == 'joy':
            return {'enthusiasm': 0.8, 'celebration': 0.7, 'positivity': 0.9}
        else:
            return {'warmth': 0.7, 'understanding': 0.8, 'supportiveness': 0.7}
            
    async def _calculate_emotional_adjustments(
        self,
        agent_profile: EmotionalProfile,
        human_emotion: str
    ) -> Dict[str, float]:
        """Calculate how agent should adjust emotions in response."""
        adjustments = {}
        
        if human_emotion == 'sadness':
            adjustments['empathetic'] = 0.1
            adjustments['caring'] = 0.15
        elif human_emotion == 'anger':
            adjustments['calm'] = 0.1
            adjustments['patient'] = 0.1
        elif human_emotion == 'joy':
            adjustments['happy'] = 0.1
            adjustments['enthusiastic'] = 0.05
            
        return adjustments

class WellnessMonitor:
    """Monitors agent wellness and provides recommendations."""
    
    async def initialize(self) -> None:
        """Initialize wellness monitoring system."""
        self.wellness_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.3,
            'critical': 0.1
        }
        
    async def assess_agent_wellness(self, profile: Optional[EmotionalProfile]) -> Dict[str, Any]:
        """Assess agent wellness and provide recommendations."""
        if not profile:
            return {'error': 'No profile provided'}
            
        wellness_factors = {
            'stress_level': 1.0 - profile.stress_level,
            'energy_level': profile.energy_level,
            'confidence_level': profile.confidence_level,
            'emotional_balance': self._calculate_emotional_balance(profile),
            'resilience': profile.emotional_resilience
        }
        
        overall_wellness = sum(wellness_factors.values()) / len(wellness_factors)
        
        wellness_category = 'critical'
        for category, threshold in self.wellness_thresholds.items():
            if overall_wellness >= threshold:
                wellness_category = category
                break
                
        recommendations = await self._generate_wellness_recommendations(profile, wellness_factors)
        
        return {
            'wellness_score': overall_wellness,
            'category': wellness_category,
            'factors': wellness_factors,
            'recommendations': recommendations,
            'last_assessment': datetime.now().isoformat()
        }
        
    def _calculate_emotional_balance(self, profile: EmotionalProfile) -> float:
        """Calculate emotional balance score."""
        if not profile.current_emotions:
            return 0.5
            
        # Ideal emotional balance has moderate levels of all emotions
        ideal_levels = {emotion: 0.6 for emotion in profile.current_emotions.keys()}
        
        balance_score = 0.0
        for emotion, current_level in profile.current_emotions.items():
            ideal_level = ideal_levels.get(emotion, 0.5)
            deviation = abs(current_level - ideal_level)
            balance_score += 1.0 - deviation
            
        return balance_score / len(profile.current_emotions) if profile.current_emotions else 0.5
        
    async def _generate_wellness_recommendations(
        self,
        profile: EmotionalProfile,
        wellness_factors: Dict[str, float]
    ) -> List[str]:
        """Generate wellness improvement recommendations."""
        recommendations = []
        
        if wellness_factors['stress_level'] < 0.5:
            recommendations.append("Implement stress reduction techniques and reduce task load")
            
        if wellness_factors['energy_level'] < 0.4:
            recommendations.append("Schedule rest periods and energy restoration activities")
            
        if wellness_factors['confidence_level'] < 0.5:
            recommendations.append("Provide positive reinforcement and assign confidence-building tasks")
            
        if wellness_factors['emotional_balance'] < 0.5:
            recommendations.append("Focus on emotional regulation and balance exercises")
            
        if not recommendations:
            recommendations.append("Maintain current wellness practices - excellent emotional health!")
            
        return recommendations