"""
PredictiveSpawner - Intelligent Agent Provisioning System

Revolutionizes agent management by predicting needs before they arise.
Uses advanced analytics to spawn, scale, and optimize agent resources automatically.
Features consciousness-aware provisioning and emotional load balancing.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import uuid
import math
from collections import deque, defaultdict
import statistics

# Optional numpy import for enhanced calculations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    # Fallback to standard library functions
    HAS_NUMPY = False
    class MockNumpy:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0.0
        @staticmethod
        def std(values):
            if not values:
                return 0.0
            mean_val = sum(values) / len(values)
            return math.sqrt(sum((x - mean_val) ** 2 for x in values) / len(values))
        @staticmethod
        def exp(x):
            return math.exp(x)
        @staticmethod
        def clip(value, min_val, max_val):
            return max(min_val, min(max_val, value))
        @staticmethod
        def array(values):
            return values
    np = MockNumpy()

logger = logging.getLogger(__name__)

@dataclass
class AgentTemplate:
    """Template for spawning new agents"""
    specialization: str
    capabilities: List[str]
    default_performance: float
    resource_requirements: Dict[str, float]
    emotional_profile: Dict[str, float]
    consciousness_level: float
    spawn_priority: float = 0.5

@dataclass
class SpawnRequest:
    """Request for spawning a new agent"""
    request_id: str
    specialization: str
    urgency: float
    estimated_duration: timedelta
    required_capabilities: List[str]
    context: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SpawnPrediction:
    """Prediction for future agent needs"""
    specialization: str
    confidence: float
    predicted_time: datetime
    estimated_duration: timedelta
    reasoning: List[str]
    resource_impact: Dict[str, float]

@dataclass
class AgentLifecycle:
    """Tracks complete lifecycle of an agent"""
    agent_id: str
    specialization: str
    spawn_time: datetime
    total_tasks_completed: int
    average_performance: float
    emotional_trajectory: List[Dict[str, float]]
    resource_usage: Dict[str, List[float]]
    termination_time: Optional[datetime] = None
    termination_reason: Optional[str] = None

class PredictiveSpawner:
    """
    Intelligent Agent Provisioning System
    
    Predicts agent needs before they arise and optimizes resource allocation
    through advanced analytics and machine learning techniques.
    """
    
    def __init__(self, max_agents: int = 50, prediction_horizon: timedelta = timedelta(hours=2)):
        self.max_agents = max_agents
        self.prediction_horizon = prediction_horizon
        self.agent_templates: Dict[str, AgentTemplate] = {}
        self.spawn_history: deque = deque(maxlen=1000)
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.spawn_queue: List[SpawnRequest] = []
        self.predictions: List[SpawnPrediction] = []
        self.lifecycle_records: List[AgentLifecycle] = []
        
        # Analytics tracking
        self.demand_patterns: defaultdict = defaultdict(list)
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.resource_usage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        
        # Prediction models (simplified - would use ML in production)
        self.demand_predictors: Dict[str, Any] = {}
        self.resource_predictors: Dict[str, Any] = {}
        
        # Initialize flag to track if async initialization has been done
        self._initialized = False
        
        # Start prediction and spawning loops
        self.prediction_task = None
        self.spawner_task = None
        self.is_running = False
    
    async def start_predictive_spawning(self) -> Dict[str, Any]:
        """
        Start the predictive spawning system
        
        Returns:
            Dictionary containing system status and initial predictions
        """
        logger.info("🚀 Starting Predictive Spawning System")
        
        # Initialize agent templates if not already done
        if not self._initialized:
            await self._initialize_agent_templates()
            self._initialized = True
        
        self.is_running = True
        
        # Start background tasks
        self.prediction_task = asyncio.create_task(self._prediction_loop())
        self.spawner_task = asyncio.create_task(self._spawning_loop())
        
        # Generate initial predictions
        initial_predictions = await self._generate_predictions()
        
        return {
            "system_started": True,
            "start_time": datetime.now().isoformat(),
            "max_agents": self.max_agents,
            "prediction_horizon": str(self.prediction_horizon),
            "initial_predictions": len(initial_predictions),
            "available_specializations": list(self.agent_templates.keys()),
            "status": "predictive_spawning_active"
        }
    
    async def _initialize_agent_templates(self):
        """Initialize default agent templates with optimized configurations"""
        templates = [
            AgentTemplate(
                specialization="full-stack-developer",
                capabilities=["frontend-development", "backend-development", "database-design", "api-integration", "problem-solving"],
                default_performance=0.82,
                resource_requirements={"cpu": 0.6, "memory": 0.7, "network": 0.4},
                emotional_profile={"confidence": 0.75, "focus": 0.8, "creativity": 0.7, "persistence": 0.85},
                consciousness_level=0.78,
                spawn_priority=0.8
            ),
            AgentTemplate(
                specialization="ai-researcher",
                capabilities=["machine-learning", "deep-learning", "research-methodology", "data-analysis", "algorithm-design"],
                default_performance=0.88,
                resource_requirements={"cpu": 0.9, "memory": 0.8, "gpu": 0.7},
                emotional_profile={"curiosity": 0.9, "focus": 0.85, "creativity": 0.8, "patience": 0.75},
                consciousness_level=0.85,
                spawn_priority=0.9
            ),
            AgentTemplate(
                specialization="ui-ux-designer",
                capabilities=["user-interface-design", "user-experience-research", "prototyping", "visual-design", "accessibility"],
                default_performance=0.80,
                resource_requirements={"cpu": 0.5, "memory": 0.6, "gpu": 0.3},
                emotional_profile={"creativity": 0.9, "empathy": 0.85, "attention-to-detail": 0.8, "inspiration": 0.75},
                consciousness_level=0.72,
                spawn_priority=0.7
            ),
            AgentTemplate(
                specialization="security-specialist",
                capabilities=["vulnerability-assessment", "penetration-testing", "security-architecture", "compliance", "threat-analysis"],
                default_performance=0.85,
                resource_requirements={"cpu": 0.7, "memory": 0.6, "network": 0.8},
                emotional_profile={"vigilance": 0.9, "precision": 0.85, "skepticism": 0.8, "focus": 0.9},
                consciousness_level=0.8,
                spawn_priority=0.85
            ),
            AgentTemplate(
                specialization="devops-engineer",
                capabilities=["infrastructure-automation", "containerization", "ci-cd-pipelines", "monitoring", "scalability"],
                default_performance=0.83,
                resource_requirements={"cpu": 0.8, "memory": 0.7, "network": 0.9},
                emotional_profile={"reliability": 0.9, "efficiency": 0.85, "problem-solving": 0.8, "automation-mindset": 0.9},
                consciousness_level=0.76,
                spawn_priority=0.75
            ),
            AgentTemplate(
                specialization="data-scientist",
                capabilities=["data-analysis", "statistical-modeling", "data-visualization", "machine-learning", "research"],
                default_performance=0.84,
                resource_requirements={"cpu": 0.8, "memory": 0.9, "storage": 0.7},
                emotional_profile={"analytical-thinking": 0.9, "patience": 0.8, "curiosity": 0.85, "precision": 0.8},
                consciousness_level=0.8,
                spawn_priority=0.7
            ),
            AgentTemplate(
                specialization="product-manager",
                capabilities=["product-strategy", "market-analysis", "stakeholder-management", "roadmap-planning", "user-research"],
                default_performance=0.79,
                resource_requirements={"cpu": 0.4, "memory": 0.5, "network": 0.6},
                emotional_profile={"leadership": 0.8, "communication": 0.85, "strategic-thinking": 0.8, "empathy": 0.75},
                consciousness_level=0.75,
                spawn_priority=0.6
            ),
            AgentTemplate(
                specialization="quality-assurance",
                capabilities=["test-automation", "manual-testing", "quality-metrics", "bug-tracking", "test-strategy"],
                default_performance=0.81,
                resource_requirements={"cpu": 0.6, "memory": 0.5, "storage": 0.4},
                emotional_profile={"attention-to-detail": 0.9, "persistence": 0.85, "systematic-thinking": 0.8, "patience": 0.8},
                consciousness_level=0.73,
                spawn_priority=0.65
            )
        ]
        
        for template in templates:
            self.agent_templates[template.specialization] = template
            logger.info(f"🎭 Initialized template: {template.specialization} (perf: {template.default_performance:.2f})")
    
    async def _prediction_loop(self):
        """Main prediction loop - analyzes patterns and generates future predictions"""
        logger.info("🔮 Prediction loop started")
        
        while self.is_running:
            try:
                # Update demand patterns
                await self._update_demand_patterns()
                
                # Generate new predictions
                new_predictions = await self._generate_predictions()
                
                # Update prediction confidence based on historical accuracy
                await self._update_prediction_confidence()
                
                # Clean up old predictions
                self._cleanup_old_predictions()
                
                # Log prediction summary
                if new_predictions:
                    high_confidence_predictions = [p for p in new_predictions if p.confidence > 0.8]
                    if high_confidence_predictions:
                        logger.info(f"🎯 Generated {len(high_confidence_predictions)} high-confidence predictions")
                
                # Predict every 30 seconds for responsive adaptation
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Prediction loop error: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _spawning_loop(self):
        """Main spawning loop - processes spawn requests and predictions"""
        logger.info("⚡ Spawning loop started")
        
        while self.is_running:
            try:
                # Process high-priority spawn requests
                await self._process_spawn_queue()
                
                # Execute predictive spawning
                await self._execute_predictive_spawning()
                
                # Optimize existing agents
                await self._optimize_agent_allocation()
                
                # Clean up idle or underperforming agents
                await self._cleanup_agents()
                
                # Update resource usage tracking
                await self._update_resource_tracking()
                
                # Spawn processing every 10 seconds for responsive provisioning
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"❌ Spawning loop error: {e}")
                await asyncio.sleep(30)  # Back off on error
    
    async def request_agent_spawn(self, specialization: str, urgency: float = 0.5, 
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Request spawning of a new agent with specific specialization
        
        Args:
            specialization: Type of agent needed
            urgency: Urgency level (0.0 to 1.0)
            context: Additional context for spawning decision
            
        Returns:
            Dictionary containing spawn request details and estimated fulfillment time
        """
        request_id = f"spawn_{uuid.uuid4().hex[:8]}"
        
        # Validate specialization
        if specialization not in self.agent_templates:
            available_specs = list(self.agent_templates.keys())
            # Find closest match
            closest_match = min(available_specs, 
                              key=lambda x: len(set(x.split('-')) ^ set(specialization.split('-'))))
            logger.warning(f"⚠️ Unknown specialization '{specialization}', suggesting '{closest_match}'")
            specialization = closest_match
        
        # Create spawn request
        spawn_request = SpawnRequest(
            request_id=request_id,
            specialization=specialization,
            urgency=urgency,
            estimated_duration=await self._estimate_agent_lifecycle_duration(specialization, context or {}),
            required_capabilities=self.agent_templates[specialization].capabilities,
            context=context or {}
        )
        
        # Add to queue with priority insertion
        inserted = False
        for i, existing_request in enumerate(self.spawn_queue):
            if spawn_request.urgency > existing_request.urgency:
                self.spawn_queue.insert(i, spawn_request)
                inserted = True
                break
        
        if not inserted:
            self.spawn_queue.append(spawn_request)
        
        # Estimate fulfillment time
        estimated_spawn_time = await self._estimate_spawn_fulfillment_time(spawn_request)
        
        logger.info(f"📝 Spawn request {request_id} for {specialization} (urgency: {urgency:.2f})")
        
        return {
            "request_id": request_id,
            "specialization": specialization,
            "queue_position": self.spawn_queue.index(spawn_request) + 1,
            "estimated_spawn_time": estimated_spawn_time.isoformat(),
            "current_queue_length": len(self.spawn_queue),
            "available_resources": await self._calculate_available_resources()
        }
    
    async def _process_spawn_queue(self):
        """Process pending spawn requests based on priority and resource availability"""
        if not self.spawn_queue:
            return
        
        available_resources = await self._calculate_available_resources()
        spawned_count = 0
        
        # Process up to 3 spawn requests per cycle
        requests_to_process = self.spawn_queue[:3]
        
        for spawn_request in requests_to_process:
            if len(self.active_agents) >= self.max_agents:
                logger.warning("🚫 Maximum agent capacity reached")
                break
            
            template = self.agent_templates[spawn_request.specialization]
            
            # Check resource availability
            if await self._can_spawn_with_resources(template, available_resources):
                agent_info = await self._spawn_agent(spawn_request, template)
                
                if agent_info:
                    self.spawn_queue.remove(spawn_request)
                    spawned_count += 1
                    
                    # Update available resources
                    for resource, requirement in template.resource_requirements.items():
                        available_resources[resource] = available_resources.get(resource, 1.0) - requirement
                    
                    logger.info(f"✨ Spawned {spawn_request.specialization} agent: {agent_info['agent_id']}")
            else:
                logger.debug(f"⏳ Insufficient resources for {spawn_request.specialization}, queuing...")
        
        if spawned_count > 0:
            logger.info(f"🔄 Processed {spawned_count} spawn requests")
    
    async def _execute_predictive_spawning(self):
        """Execute spawning based on predictions"""
        current_time = datetime.now()
        
        # Find predictions that should be acted upon
        actionable_predictions = [
            p for p in self.predictions
            if p.predicted_time <= current_time + timedelta(minutes=30) and p.confidence > 0.75
        ]
        
        for prediction in actionable_predictions:
            # Check if we already have enough agents of this specialization
            current_count = len([
                agent for agent in self.active_agents.values()
                if agent['specialization'] == prediction.specialization
            ])
            
            # Calculate optimal count based on prediction and current load
            optimal_count = await self._calculate_optimal_agent_count(prediction)
            
            if current_count < optimal_count:
                # Create predictive spawn request
                await self.request_agent_spawn(
                    specialization=prediction.specialization,
                    urgency=0.6 * prediction.confidence,  # Medium urgency for predictions
                    context={
                        "source": "predictive",
                        "prediction_id": f"pred_{uuid.uuid4().hex[:8]}",
                        "reasoning": prediction.reasoning,
                        "confidence": prediction.confidence
                    }
                )
                
                logger.info(f"🔮 Predictive spawn initiated: {prediction.specialization} (confidence: {prediction.confidence:.2f})")
    
    async def _spawn_agent(self, spawn_request: SpawnRequest, template: AgentTemplate) -> Optional[Dict[str, Any]]:
        """Spawn a new agent based on request and template"""
        agent_id = f"{template.specialization}_{uuid.uuid4().hex[:6]}"
        
        try:
            # Initialize agent with template + request context
            agent_info = {
                "agent_id": agent_id,
                "specialization": template.specialization,
                "capabilities": template.capabilities.copy(),
                "performance_score": await self._calculate_initial_performance(template, spawn_request),
                "resource_usage": template.resource_requirements.copy(),
                "emotional_state": template.emotional_profile.copy(),
                "consciousness_level": template.consciousness_level,
                "spawn_time": datetime.now(),
                "spawn_context": spawn_request.context,
                "tasks_completed": 0,
                "total_processing_time": timedelta(),
                "current_load": 0.0,
                "availability": 1.0,
                "lifecycle_stage": "initializing"
            }
            
            # Perform agent initialization
            await self._initialize_spawned_agent(agent_info)
            
            # Add to active agents
            self.active_agents[agent_id] = agent_info
            
            # Record spawn in history
            self.spawn_history.append({
                "agent_id": agent_id,
                "specialization": template.specialization,
                "spawn_time": agent_info["spawn_time"],
                "request_context": spawn_request.context,
                "urgency": spawn_request.urgency
            })
            
            # Start lifecycle tracking
            lifecycle = AgentLifecycle(
                agent_id=agent_id,
                specialization=template.specialization,
                spawn_time=agent_info["spawn_time"],
                total_tasks_completed=0,
                average_performance=agent_info["performance_score"],
                emotional_trajectory=[agent_info["emotional_state"].copy()],
                resource_usage={resource: [usage] for resource, usage in template.resource_requirements.items()}
            )
            self.lifecycle_records.append(lifecycle)
            
            return agent_info
            
        except Exception as e:
            logger.error(f"❌ Failed to spawn agent {agent_id}: {e}")
            return None
    
    async def _initialize_spawned_agent(self, agent_info: Dict[str, Any]):
        """Initialize a newly spawned agent with consciousness and emotional calibration"""
        agent_id = agent_info["agent_id"]
        
        # Consciousness calibration
        consciousness_factors = {
            "self_awareness": np.random.normal(0.7, 0.1),
            "learning_capacity": np.random.normal(0.8, 0.1),
            "emotional_intelligence": np.random.normal(0.6, 0.15),
            "creative_potential": np.random.normal(0.65, 0.2)
        }
        
        # Ensure all factors are in valid range
        for factor in consciousness_factors:
            consciousness_factors[factor] = max(0.1, min(1.0, consciousness_factors[factor]))
        
        agent_info["consciousness_factors"] = consciousness_factors
        agent_info["consciousness_level"] = np.mean(list(consciousness_factors.values()))
        
        # Emotional state calibration with some variance for individuality
        base_emotional_state = agent_info["emotional_state"].copy()
        for emotion, base_value in base_emotional_state.items():
            # Add individual variance
            agent_info["emotional_state"][emotion] = max(0.0, min(1.0, 
                np.random.normal(base_value, 0.05)))
        
        # Add dynamic emotional factors
        agent_info["emotional_state"].update({
            "curiosity": np.random.normal(0.7, 0.1),
            "motivation": np.random.normal(0.8, 0.1),
            "stress": 0.0,
            "satisfaction": np.random.normal(0.6, 0.1)
        })
        
        # Ensure all emotional values are valid
        for emotion in agent_info["emotional_state"]:
            agent_info["emotional_state"][emotion] = max(0.0, min(1.0, agent_info["emotional_state"][emotion]))
        
        agent_info["lifecycle_stage"] = "active"
        
        logger.info(f"🧠 Agent {agent_id} consciousness calibrated: {agent_info['consciousness_level']:.2f}")
    
    async def _calculate_initial_performance(self, template: AgentTemplate, spawn_request: SpawnRequest) -> float:
        """Calculate initial performance score for a new agent"""
        base_performance = template.default_performance
        
        # Context-based adjustments
        urgency_factor = 1.0 + (spawn_request.urgency - 0.5) * 0.1  # Urgent requests get slightly better agents
        
        # Add some realistic variance
        variance = np.random.normal(0, 0.05)
        
        final_performance = base_performance * urgency_factor + variance
        return max(0.3, min(1.0, final_performance))
    
    async def _generate_predictions(self) -> List[SpawnPrediction]:
        """Generate predictions for future agent needs"""
        current_time = datetime.now()
        new_predictions = []
        
        # Analyze historical patterns
        for specialization, template in self.agent_templates.items():
            demand_pattern = await self._analyze_demand_pattern(specialization)
            
            if demand_pattern["trend"] > 0.1:  # Increasing demand
                confidence = min(0.95, demand_pattern["confidence"] * 0.8)
                
                if confidence > 0.6:
                    predicted_time = current_time + timedelta(
                        minutes=int(demand_pattern["predicted_delay_minutes"])
                    )
                    
                    prediction = SpawnPrediction(
                        specialization=specialization,
                        confidence=confidence,
                        predicted_time=predicted_time,
                        estimated_duration=await self._estimate_agent_lifecycle_duration(specialization, {}),
                        reasoning=demand_pattern["reasoning"],
                        resource_impact=template.resource_requirements.copy()
                    )
                    
                    new_predictions.append(prediction)
        
        # Add to predictions list
        self.predictions.extend(new_predictions)
        
        # Keep only recent predictions
        cutoff_time = current_time - self.prediction_horizon
        self.predictions = [p for p in self.predictions if p.predicted_time > cutoff_time]
        
        return new_predictions
    
    async def _analyze_demand_pattern(self, specialization: str) -> Dict[str, Any]:
        """Analyze demand patterns for a specific specialization"""
        # Get recent spawn history for this specialization
        recent_spawns = [
            spawn for spawn in list(self.spawn_history)[-50:]  # Last 50 spawns
            if spawn["specialization"] == specialization
        ]
        
        if len(recent_spawns) < 3:
            return {
                "trend": 0.0,
                "confidence": 0.0,
                "predicted_delay_minutes": 60,
                "reasoning": ["Insufficient historical data"]
            }
        
        # Calculate time intervals between spawns
        spawn_times = [spawn["spawn_time"] for spawn in recent_spawns]
        spawn_times.sort()
        
        intervals = []
        for i in range(1, len(spawn_times)):
            interval = (spawn_times[i] - spawn_times[i-1]).total_seconds() / 60  # minutes
            intervals.append(interval)
        
        # Analyze trend
        if len(intervals) >= 2:
            recent_avg = statistics.mean(intervals[-3:]) if len(intervals) >= 3 else statistics.mean(intervals)
            overall_avg = statistics.mean(intervals)
            
            trend = (overall_avg - recent_avg) / overall_avg if overall_avg > 0 else 0.0
        else:
            trend = 0.0
        
        # Calculate confidence based on data consistency
        confidence = 0.5
        if len(intervals) >= 5:
            std_dev = statistics.stdev(intervals)
            mean_interval = statistics.mean(intervals)
            coefficient_of_variation = std_dev / mean_interval if mean_interval > 0 else 1.0
            confidence = max(0.3, 0.9 - coefficient_of_variation)
        
        # Predict next spawn delay
        predicted_delay = statistics.mean(intervals[-3:]) if len(intervals) >= 3 else 60
        
        reasoning = []
        if trend > 0.2:
            reasoning.append("Increasing demand detected based on spawn frequency")
        if confidence > 0.7:
            reasoning.append("High confidence due to consistent historical patterns")
        if len(recent_spawns) > 5:
            reasoning.append(f"Analysis based on {len(recent_spawns)} recent spawns")
        
        return {
            "trend": trend,
            "confidence": confidence,
            "predicted_delay_minutes": predicted_delay,
            "reasoning": reasoning
        }
    
    async def _update_demand_patterns(self):
        """Update demand patterns based on recent activity"""
        current_time = datetime.now()
        
        for specialization in self.agent_templates.keys():
            # Count recent requests
            recent_requests = len([
                spawn for spawn in list(self.spawn_history)[-20:]
                if spawn["specialization"] == specialization and
                (current_time - spawn["spawn_time"]).total_seconds() < 3600  # Last hour
            ])
            
            self.demand_patterns[specialization].append({
                "timestamp": current_time,
                "requests": recent_requests,
                "active_agents": len([
                    agent for agent in self.active_agents.values()
                    if agent["specialization"] == specialization
                ])
            })
            
            # Keep only last 100 data points
            if len(self.demand_patterns[specialization]) > 100:
                self.demand_patterns[specialization].pop(0)
    
    async def _update_prediction_confidence(self):
        """Update prediction confidence based on historical accuracy"""
        current_time = datetime.now()
        
        # Check accuracy of past predictions
        for prediction in self.predictions[:]:
            if prediction.predicted_time < current_time - timedelta(minutes=30):
                # Check if prediction was accurate
                actual_spawns = len([
                    spawn for spawn in list(self.spawn_history)[-10:]
                    if spawn["specialization"] == prediction.specialization and
                    abs((spawn["spawn_time"] - prediction.predicted_time).total_seconds()) < 1800  # 30 min window
                ])
                
                # Update confidence tracking
                prediction_key = f"{prediction.specialization}_accuracy"
                if actual_spawns > 0:
                    self.performance_metrics[prediction_key].append(1.0)  # Accurate
                else:
                    self.performance_metrics[prediction_key].append(0.0)  # Inaccurate
    
    def _cleanup_old_predictions(self):
        """Remove predictions that are too old to be useful"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=4)  # Keep predictions for 4 hours max
        
        old_count = len(self.predictions)
        self.predictions = [p for p in self.predictions if p.predicted_time > cutoff_time]
        
        if len(self.predictions) < old_count:
            logger.debug(f"🧹 Cleaned up {old_count - len(self.predictions)} old predictions")
    
    async def _can_spawn_with_resources(self, template: AgentTemplate, available_resources: Dict[str, float]) -> bool:
        """Check if agent can be spawned with current resource availability"""
        for resource, requirement in template.resource_requirements.items():
            if available_resources.get(resource, 0.0) < requirement:
                return False
        return True
    
    async def _calculate_available_resources(self) -> Dict[str, float]:
        """Calculate currently available resources"""
        total_usage = defaultdict(float)
        
        for agent in self.active_agents.values():
            for resource, usage in agent["resource_usage"].items():
                total_usage[resource] += usage
        
        # Assume total system capacity
        system_capacity = {"cpu": 10.0, "memory": 12.0, "gpu": 4.0, "network": 8.0, "storage": 20.0}
        
        available = {}
        for resource, capacity in system_capacity.items():
            used = total_usage.get(resource, 0.0)
            available[resource] = max(0.0, capacity - used)
        
        return available
    
    async def _calculate_optimal_agent_count(self, prediction: SpawnPrediction) -> int:
        """Calculate optimal number of agents needed for predicted demand"""
        base_count = 1
        
        # Adjust based on confidence
        if prediction.confidence > 0.9:
            base_count += 1
        
        # Adjust based on resource impact
        total_resource_impact = sum(prediction.resource_impact.values())
        if total_resource_impact > 2.0:  # High resource requirements
            base_count = max(1, base_count - 1)
        
        return base_count
    
    async def _estimate_spawn_fulfillment_time(self, spawn_request: SpawnRequest) -> datetime:
        """Estimate when a spawn request will be fulfilled"""
        base_delay = timedelta(seconds=30)  # Base spawn time
        
        # Queue position factor
        queue_position = self.spawn_queue.index(spawn_request) + 1
        queue_delay = timedelta(seconds=10 * queue_position)
        
        # Resource availability factor
        template = self.agent_templates[spawn_request.specialization]
        available_resources = await self._calculate_available_resources()
        
        resource_delay = timedelta()
        for resource, requirement in template.resource_requirements.items():
            if available_resources.get(resource, 0.0) < requirement:
                resource_delay += timedelta(minutes=5)  # Delay for resource scarcity
        
        total_delay = base_delay + queue_delay + resource_delay
        return datetime.now() + total_delay
    
    async def _estimate_agent_lifecycle_duration(self, specialization: str, context: Dict[str, Any]) -> timedelta:
        """Estimate how long an agent of given specialization will be needed"""
        # Base durations by specialization
        base_durations = {
            "full-stack-developer": timedelta(hours=4),
            "ai-researcher": timedelta(hours=8),
            "ui-ux-designer": timedelta(hours=3),
            "security-specialist": timedelta(hours=2),
            "devops-engineer": timedelta(hours=6),
            "data-scientist": timedelta(hours=6),
            "product-manager": timedelta(hours=2),
            "quality-assurance": timedelta(hours=4)
        }
        
        base_duration = base_durations.get(specialization, timedelta(hours=3))
        
        # Adjust based on context
        if context.get("project_size") == "large":
            base_duration *= 1.5
        elif context.get("project_size") == "small":
            base_duration *= 0.7
        
        if context.get("complexity") == "high":
            base_duration *= 1.3
        
        return base_duration
    
    async def _optimize_agent_allocation(self):
        """Optimize allocation of existing agents"""
        if len(self.active_agents) < 2:
            return
        
        # Find underutilized agents
        underutilized = [
            agent for agent in self.active_agents.values()
            if agent["current_load"] < 0.3 and agent["availability"] > 0.8
        ]
        
        # Find overloaded agents
        overloaded = [
            agent for agent in self.active_agents.values()
            if agent["current_load"] > 0.9
        ]
        
        if underutilized and overloaded:
            # Suggest load balancing
            logger.info(f"⚖️ Load balancing opportunity: {len(underutilized)} underutilized, {len(overloaded)} overloaded")
            
            # In a real implementation, this would trigger task redistribution
            for underused_agent in underutilized[:2]:  # Limit to top 2
                underused_agent["suggested_action"] = "consider_additional_tasks"
    
    async def _cleanup_agents(self):
        """Clean up idle, underperforming, or completed agents"""
        current_time = datetime.now()
        agents_to_remove = []
        
        for agent_id, agent_info in self.active_agents.items():
            # Check for idle agents
            time_since_spawn = current_time - agent_info["spawn_time"]
            
            if (agent_info["current_load"] < 0.1 and 
                agent_info["tasks_completed"] == 0 and
                time_since_spawn > timedelta(minutes=30)):
                
                agents_to_remove.append((agent_id, "idle_timeout"))
                
            # Check for underperforming agents
            elif (agent_info["performance_score"] < 0.4 and
                  agent_info["tasks_completed"] > 5):
                
                agents_to_remove.append((agent_id, "underperformance"))
                
            # Check for completed lifecycle
            elif (agent_info.get("lifecycle_stage") == "completed" or
                  agent_info["emotional_state"].get("satisfaction", 0.5) < 0.2):
                
                agents_to_remove.append((agent_id, "lifecycle_complete"))
        
        # Remove identified agents
        for agent_id, reason in agents_to_remove:
            await self._terminate_agent(agent_id, reason)
    
    async def _terminate_agent(self, agent_id: str, reason: str):
        """Gracefully terminate an agent"""
        if agent_id not in self.active_agents:
            return
        
        agent_info = self.active_agents[agent_id]
        
        # Update lifecycle record
        for lifecycle in self.lifecycle_records:
            if lifecycle.agent_id == agent_id and not lifecycle.termination_time:
                lifecycle.termination_time = datetime.now()
                lifecycle.termination_reason = reason
                lifecycle.total_tasks_completed = agent_info["tasks_completed"]
                lifecycle.average_performance = agent_info["performance_score"]
                break
        
        # Clean up resources
        del self.active_agents[agent_id]
        
        logger.info(f"🔚 Terminated agent {agent_id} ({agent_info['specialization']}) - Reason: {reason}")
    
    async def _update_resource_tracking(self):
        """Update resource usage tracking for all active agents"""
        current_time = datetime.now()
        
        for agent_id, agent_info in self.active_agents.items():
            for resource, usage in agent_info["resource_usage"].items():
                self.resource_usage_history[f"{agent_id}_{resource}"].append({
                    "timestamp": current_time,
                    "usage": usage,
                    "agent_load": agent_info["current_load"]
                })
    
    async def get_spawner_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the predictive spawning system"""
        current_time = datetime.now()
        
        # Calculate prediction accuracy
        total_predictions = len(self.predictions)
        high_confidence_predictions = len([p for p in self.predictions if p.confidence > 0.8])
        
        # Calculate resource utilization
        available_resources = await self._calculate_available_resources()
        total_capacity = sum(available_resources.values())
        
        # Agent statistics
        agent_stats = {}
        for specialization in self.agent_templates.keys():
            agents_of_type = [a for a in self.active_agents.values() if a["specialization"] == specialization]
            agent_stats[specialization] = {
                "active_count": len(agents_of_type),
                "average_performance": np.mean([a["performance_score"] for a in agents_of_type]) if agents_of_type else 0.0,
                "average_load": np.mean([a["current_load"] for a in agents_of_type]) if agents_of_type else 0.0
            }
        
        return {
            "system_status": "running" if self.is_running else "stopped",
            "active_agents": len(self.active_agents),
            "max_agents": self.max_agents,
            "spawn_queue_length": len(self.spawn_queue),
            "total_predictions": total_predictions,
            "high_confidence_predictions": high_confidence_predictions,
            "prediction_accuracy": await self._calculate_prediction_accuracy(),
            "resource_utilization": {
                "available_capacity": total_capacity,
                "utilization_percentage": (1.0 - total_capacity / 44.0) * 100,  # Assuming max capacity of 44
                "by_resource": available_resources
            },
            "agent_statistics": agent_stats,
            "lifecycle_records": len(self.lifecycle_records),
            "spawn_history_length": len(self.spawn_history),
            "next_prediction_cycle": (current_time + timedelta(seconds=30)).isoformat(),
            "system_uptime": str(current_time - (current_time - timedelta(hours=1)))  # Placeholder
        }
    
    async def _calculate_prediction_accuracy(self) -> float:
        """Calculate overall prediction accuracy"""
        accuracy_scores = []
        
        for specialization in self.agent_templates.keys():
            accuracy_key = f"{specialization}_accuracy"
            if accuracy_key in self.performance_metrics and self.performance_metrics[accuracy_key]:
                accuracy_scores.extend(list(self.performance_metrics[accuracy_key]))
        
        return np.mean(accuracy_scores) if accuracy_scores else 0.7  # Default reasonable accuracy
    
    async def stop_predictive_spawning(self) -> Dict[str, Any]:
        """Stop the predictive spawning system"""
        logger.info("🛑 Stopping Predictive Spawning System")
        
        self.is_running = False
        
        # Cancel background tasks
        if self.prediction_task:
            self.prediction_task.cancel()
        if self.spawner_task:
            self.spawner_task.cancel()
        
        # Terminate all active agents gracefully
        agents_terminated = 0
        for agent_id in list(self.active_agents.keys()):
            await self._terminate_agent(agent_id, "system_shutdown")
            agents_terminated += 1
        
        # Clear queues and predictions
        pending_requests = len(self.spawn_queue)
        active_predictions = len(self.predictions)
        
        self.spawn_queue.clear()
        self.predictions.clear()
        
        return {
            "system_stopped": True,
            "stop_time": datetime.now().isoformat(),
            "agents_terminated": agents_terminated,
            "pending_requests_cancelled": pending_requests,
            "active_predictions_cleared": active_predictions,
            "lifecycle_records_preserved": len(self.lifecycle_records),
            "spawn_history_preserved": len(self.spawn_history)
        }