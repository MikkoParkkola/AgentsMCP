"""
Task Classifier for Smart Agent Orchestration

Intelligently classifies user tasks to determine whether:
1. Simple response (no agents needed) - handle directly with orchestrator
2. Single agent needed - delegate to one specialized agent  
3. Multi-agent needed - coordinate multiple agents for complex tasks

This prevents unnecessary agent spawning for simple interactions like greetings,
while ensuring complex tasks get appropriate agent resources.
"""

import re
import logging
import asyncio
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskClassification(Enum):
    """Types of task classifications."""
    SIMPLE_RESPONSE = "simple_response"
    SINGLE_AGENT_NEEDED = "single_agent_needed"  
    MULTI_AGENT_NEEDED = "multi_agent_needed"


@dataclass
class ClassificationResult:
    """Result of task classification."""
    classification: TaskClassification
    confidence: float  # 0.0 - 1.0
    required_agents: List[str]
    reasoning: str
    task_complexity: str  # "trivial", "simple", "moderate", "complex"
    estimated_response_time: str  # "immediate", "quick", "moderate", "extended"


class TaskClassifier:
    """
    Intelligent task classifier that determines appropriate response strategy.
    
    Uses pattern matching, keyword analysis, and heuristics to classify tasks
    into categories that determine whether agents are needed and which ones.
    """
    
    def __init__(self):
        """Initialize the task classifier."""
        self.simple_patterns = self._build_simple_patterns()
        self.agent_mapping = self._build_agent_mapping()
        self.complexity_indicators = self._build_complexity_indicators()
        self.multi_agent_indicators = self._build_multi_agent_indicators()
        
        # Statistics
        self.classifications_made = 0
        self.classification_stats = {
            TaskClassification.SIMPLE_RESPONSE: 0,
            TaskClassification.SINGLE_AGENT_NEEDED: 0,
            TaskClassification.MULTI_AGENT_NEEDED: 0
        }
    
    def _build_simple_patterns(self) -> List[Dict]:
        """Build patterns for simple tasks that don't need agents."""
        return [
            # Greetings and pleasantries
            {
                "patterns": [r'\bhello\b', r'\bhi\b', r'\bhey\b', r'\bgood (morning|afternoon|evening)\b'],
                "confidence": 0.95,
                "reasoning": "Simple greeting"
            },
            
            # Status and health checks
            {
                "patterns": [r'\bstatus\b', r'\bhow are you\b', r'\bworking\b', r'\bup\b', r'\bonline\b'],
                "confidence": 0.90,
                "reasoning": "Status inquiry"
            },
            
            # Basic help requests
            {
                "patterns": [r'\bhelp\b(?!\s+\w+\s+\w+)', r'\bwhat can you do\b', r'\bcapabilities\b'],
                "confidence": 0.85,
                "reasoning": "General help request"
            },
            
            # Simple questions without complexity
            {
                "patterns": [r'^\w{1,20}\?$', r'^(what|who|when|where|why|how)\s+\w{1,30}\?$'],
                "confidence": 0.70,
                "reasoning": "Simple question"
            },
            
            # Acknowledgments and confirmations
            {
                "patterns": [r'\b(ok|okay|yes|no|thanks|thank you)\b$'],
                "confidence": 0.80,
                "reasoning": "Simple acknowledgment"
            }
        ]
    
    def _build_agent_mapping(self) -> Dict[str, Dict]:
        """Build mapping of task types to appropriate roles."""
        return {
            "code": {
                "roles": ["coder"],  # Coding tasks route to coder role
                "patterns": [
                    r'\b(write|create|implement|code|program|script|function|class|method)\b',
                    r'\b(debug|fix|error|bug|issue)\b',
                    r'\b(refactor|optimize|improve|enhance)\b',
                    r'\b(python|javascript|java|c\+\+|rust|go|typescript)\b'
                ],
                "confidence": 0.85
            },
            
            "analysis": {
                "roles": ["analyst"],  # Analysis tasks route to analyst role
                "patterns": [
                    r'\b(analyz|review|examin|assess|evaluat)\w*\b',  # Matches analyze/analysis, examine/examination, assess/assessment, evaluate/evaluation
                    r'\b(explain|describe|documentation|docs)\b',
                    r'\b(performance|optimization|efficiency)\b',
                    r'\b(suggest.*improvement|recommend)\b',
                    r'\bquality\b.*(check|review|assessment|evaluation|audit)',
                    r'\bcomprehensive\b.*(review|analysis|assessment|evaluation|audit|quality)'
                ],
                "confidence": 0.85
            },
            
            "local_tasks": {
                "roles": ["general"],  # General role for simple local tasks
                "patterns": [
                    r'\b(local|offline|private|secure)\b',
                    r'\b(simple|quick|basic|straightforward)\b',
                    r'\btest\b(?!\s+(suite|framework|automation))'
                ],
                "confidence": 0.75
            },
            
            "architecture": {
                "roles": ["architect"],  # Architect role for system design
                "patterns": [
                    r'\b(design|architect)\b.*(system|architecture|solution|framework)\b',
                    r'\b(system|architecture|microservices|design pattern)\b',
                    r'\b(technical.*design|system.*design|architecture.*design)\b',
                    r'\b(scalability|distributed|enterprise)\b'
                ],
                "confidence": 0.85
            },
            
            "complex_projects": {
                "roles": ["architect"],  # Architect role for complex projects  
                "patterns": [
                    r'\b(create|build).*\b(project|application|system|platform|website|app)\b',
                    r'\b(web application|mobile app|desktop app|full stack)\b',
                    r'\b(with.*authentication|with.*database|with.*api)\b',
                    r'\b(migrate|integrate|deploy|configure)\b',
                    r'\b(production|scale)\b'
                ],
                "confidence": 0.80
            },
            
            "project_management": {
                "roles": ["project_manager"],  # Project manager role for PM tasks
                "patterns": [
                    r'\b(backlog|priorities|p0|p1|p2|roadmap)\b',
                    r'\b(project.*roadmap|create.*roadmap|prioritize.*features)\b',
                    r'\b(what.*next|next.*task|current.*status)\b',
                    r'\b(product.*priorities|product.*backlog|development.*plan)\b',
                    r'\b(planning|plan.*project|manage.*project)\b',
                    r'\b(current.*priorities|important.*task)\b',
                    r'\b(what.*should.*do|recommendations|suggest)\b'
                ],
                "confidence": 0.85
            },
            
            "qa_review": {
                "roles": ["qa_reviewer"],  # QA reviewer role for code review and testing
                "patterns": [
                    r'\b(review.*code|code.*review|quality.*review)\b',
                    r'\b(security.*issues|security.*review|security.*check)\b', 
                    r'\b(test|testing|quality.*assurance|qa)\b',
                    r'\b(bug|issues|problems|defects)\b.*\b(review|check|find)\b',
                    r'\b(review.*quality|quality.*check)\b'
                ],
                "confidence": 0.85
            }
        }
    
    def _build_complexity_indicators(self) -> Dict[str, List[str]]:
        """Build indicators of task complexity."""
        return {
            "high": [
                "create application", "build system", "implement framework",
                "design architecture", "multi-step", "enterprise", "production",
                "integrate with", "migrate from", "full stack", "end-to-end"
            ],
            
            "moderate": [
                "write function", "create script", "implement feature", 
                "analyze code", "debug issue", "optimize performance",
                "configure setup", "write tests", "documentation"
            ],
            
            "low": [
                "explain", "show example", "quick fix", "simple question",
                "what is", "how to", "basic", "tutorial"
            ]
        }
    
    def _build_multi_agent_indicators(self) -> List[str]:
        """Build indicators that suggest multiple agents may be needed."""
        return [
            # Multiple action words
            r'\b(and|also|plus|additionally|furthermore|moreover)\b',
            
            # Complex project indicators
            r'\b(frontend and backend|full stack|end-to-end)\b',
            r'\b(design and implement|create and deploy|build and test)\b',
            
            # Architecture complexity indicators
            r'\b(microservices|distributed system|multi-tier|architecture)\b',
            r'\b(scalable|enterprise|production-ready)\b',
            
            # Multiple technology mentions
            r'\b(react|vue|angular).*(node|express|django|flask)\b',
            r'\b(database|api|ui|frontend|backend|deployment)\b.*\b(database|api|ui|frontend|backend|deployment)\b',
            
            # Process indicators
            r'\b(first.*then|step by step|phase|stage|workflow)\b',
            
            # Coordination words
            r'\b(coordinate|orchestrate|manage|oversee)\b'
        ]
    
    def classify(self, user_input: str, context: Dict = None) -> ClassificationResult:
        """Classify a user task and determine orchestration strategy (synchronous version)."""
        if context is None:
            context = {}
        
        # Try to run in existing event loop, or create new one if needed
        try:
            loop = asyncio.get_running_loop()
            # If we're in an event loop, use create_task and wait
            task = loop.create_task(self.classify_task(user_input, context))
            # This is a bit tricky - we need to run the task synchronously
            # For now, let's just run the classification logic directly
            return self._classify_task_sync(user_input, context)
        except RuntimeError:
            # No running event loop, we can use asyncio.run
            return asyncio.run(self.classify_task(user_input, context))
    
    def _classify_task_sync(self, user_input: str, context: Dict) -> ClassificationResult:
        """Synchronous version of the classification logic."""
        self.classifications_made += 1
        
        user_input_clean = user_input.strip().lower()
        
        # Step 1: Check for simple patterns first
        simple_result = self._check_simple_patterns(user_input_clean)
        if simple_result:
            self.classification_stats[TaskClassification.SIMPLE_RESPONSE] += 1
            return simple_result
        
        # Step 2: Check for multi-agent indicators
        multi_agent_score = self._calculate_multi_agent_score(user_input_clean)
        if multi_agent_score > 0.6:
            agents = self._determine_multi_agent_team(user_input_clean)
            result = ClassificationResult(
                classification=TaskClassification.MULTI_AGENT_NEEDED,
                confidence=multi_agent_score,
                required_agents=agents,
                reasoning=f"Multi-agent coordination needed (score: {multi_agent_score:.2f})",
                task_complexity=self._assess_complexity(user_input_clean),
                estimated_response_time="extended"
            )
            self.classification_stats[TaskClassification.MULTI_AGENT_NEEDED] += 1
            return result
        
        # Step 3: Determine single agent need
        agent_match = self._find_best_agent_match(user_input_clean)
        if agent_match:
            result = ClassificationResult(
                classification=TaskClassification.SINGLE_AGENT_NEEDED,
                confidence=agent_match["confidence"],
                required_agents=[agent_match["agent"]],
                reasoning=f"Single agent task: {agent_match['reason']}",
                task_complexity=self._assess_complexity(user_input_clean),
                estimated_response_time="moderate"
            )
            self.classification_stats[TaskClassification.SINGLE_AGENT_NEEDED] += 1
            return result
        
        # Step 4: Fallback to simple response with low confidence
        result = ClassificationResult(
            classification=TaskClassification.SIMPLE_RESPONSE,
            confidence=0.3,
            required_agents=[],
            reasoning="No clear agent match, defaulting to simple response",
            task_complexity="simple",
            estimated_response_time="immediate"
        )
        self.classification_stats[TaskClassification.SIMPLE_RESPONSE] += 1
        return result
    
    async def classify_task(self, user_input: str, context: Dict) -> ClassificationResult:
        """Classify a user task and determine orchestration strategy."""
        self.classifications_made += 1
        
        user_input_clean = user_input.strip().lower()
        
        # Step 1: Check for simple patterns first
        simple_result = self._check_simple_patterns(user_input_clean)
        if simple_result:
            self.classification_stats[TaskClassification.SIMPLE_RESPONSE] += 1
            return simple_result
        
        # Step 2: Check for multi-agent indicators
        multi_agent_score = self._calculate_multi_agent_score(user_input_clean)
        if multi_agent_score > 0.6:
            agents = self._determine_multi_agent_team(user_input_clean)
            result = ClassificationResult(
                classification=TaskClassification.MULTI_AGENT_NEEDED,
                confidence=multi_agent_score,
                required_agents=agents,
                reasoning=f"Multi-agent coordination needed (score: {multi_agent_score:.2f})",
                task_complexity=self._assess_complexity(user_input_clean),
                estimated_response_time="extended"
            )
            self.classification_stats[TaskClassification.MULTI_AGENT_NEEDED] += 1
            return result
        
        # Step 3: Determine single agent need
        agent_match = self._find_best_agent_match(user_input_clean)
        if agent_match:
            result = ClassificationResult(
                classification=TaskClassification.SINGLE_AGENT_NEEDED,
                confidence=agent_match["confidence"],
                required_agents=[agent_match["agent"]],
                reasoning=f"Single agent task: {agent_match['reason']}",
                task_complexity=self._assess_complexity(user_input_clean),
                estimated_response_time="moderate"
            )
            self.classification_stats[TaskClassification.SINGLE_AGENT_NEEDED] += 1
            return result
        
        # Step 4: Fallback to simple response with low confidence
        result = ClassificationResult(
            classification=TaskClassification.SIMPLE_RESPONSE,
            confidence=0.3,
            required_agents=[],
            reasoning="No clear agent match, defaulting to simple response",
            task_complexity="simple",
            estimated_response_time="immediate"
        )
        self.classification_stats[TaskClassification.SIMPLE_RESPONSE] += 1
        return result
    
    def _check_simple_patterns(self, user_input: str) -> Optional[ClassificationResult]:
        """Check if input matches simple task patterns."""
        
        # First check if this looks like a complex task that should override simple patterns
        complexity_overrides = [
            r'\b(write|create|implement|code|program|script|function|class|method)\b',
            r'\b(analyze|review|examine|assess|evaluate|audit)\b.*(project|codebase|code|performance|system|data|algorithm|structure|quality|architecture)',
            r'\bcomprehensive\b.*(assessment|review|analysis|evaluation|audit)',
            r'\bquality\b.*(assessment|review|analysis|evaluation|audit)',
            r'\b(debug|fix|error|bug|issue|problem)\b',
            r'\b(optimize|improve|enhance|refactor)\b',
            r'\b(build|deploy|configure|setup|install)\b',
            r'\b(performance|efficiency|speed|memory|cpu)\b',
            r'\b(suggest.*improvement|recommend.*change)\b'
        ]
        
        for override_pattern in complexity_overrides:
            if re.search(override_pattern, user_input, re.IGNORECASE):
                return None  # Don't classify as simple if it matches complexity override
        
        for pattern_group in self.simple_patterns:
            for pattern in pattern_group["patterns"]:
                if re.search(pattern, user_input, re.IGNORECASE):
                    return ClassificationResult(
                        classification=TaskClassification.SIMPLE_RESPONSE,
                        confidence=pattern_group["confidence"],
                        required_agents=[],
                        reasoning=pattern_group["reasoning"],
                        task_complexity="trivial",
                        estimated_response_time="immediate"
                    )
        return None
    
    def _calculate_multi_agent_score(self, user_input: str) -> float:
        """Calculate score indicating likelihood of multi-agent need."""
        score = 0.0
        
        # Check multi-agent indicators
        for indicator in self.multi_agent_indicators:
            if re.search(indicator, user_input, re.IGNORECASE):
                score += 0.2
        
        # Check for multiple technology/domain mentions
        domains = ["frontend", "backend", "database", "api", "ui", "deployment", "testing"]
        domain_mentions = sum(1 for domain in domains if domain in user_input)
        if domain_mentions >= 2:
            score += 0.3
        
        # Check for complexity indicators
        if any(indicator in user_input for indicator in self.complexity_indicators["high"]):
            score += 0.3
        
        # Length heuristic (longer requests often need more coordination)
        if len(user_input) > 200:
            score += 0.1
        
        return min(score, 1.0)
    
    def _find_best_agent_match(self, user_input: str) -> Optional[Dict]:
        """Find the best role match for a single-agent task."""
        best_match = None
        best_score = 0.0
        
        for task_type, config in self.agent_mapping.items():
            score = 0.0
            
            # Check pattern matches
            pattern_matches = 0
            for pattern in config["patterns"]:
                if re.search(pattern, user_input, re.IGNORECASE):
                    pattern_matches += 1
                    score += 0.3  # Increased from 0.2 to make it easier to reach threshold
            
            if pattern_matches > 0:
                score = min(score * config["confidence"], 0.95)  # Apply confidence but cap it
                
                if score > best_score:
                    best_score = score
                    # Use roles instead of agents
                    role_key = "roles" if "roles" in config else "agents"  # Backward compatibility
                    best_match = {
                        "agent": config[role_key][0],  # Preferred role
                        "confidence": score,
                        "reason": f"Best match for {task_type} task"
                    }
        
        return best_match if best_score > 0.25 else None
    
    def _determine_multi_agent_team(self, user_input: str) -> List[str]:
        """Determine which agents to include in multi-agent team."""
        agents = set()
        
        # Add agents based on detected patterns
        for task_type, config in self.agent_mapping.items():
            for pattern in config["patterns"]:
                if re.search(pattern, user_input, re.IGNORECASE):
                    agents.update(config["agents"])
        
        # Ensure we have at least a basic agent
        if not agents:
            agents.add("codex")  # Default fallback
        
        # Limit team size
        return list(agents)[:4]  # Max 4 agents to avoid chaos
    
    def _assess_complexity(self, user_input: str) -> str:
        """Assess the complexity of the task."""
        for complexity, indicators in self.complexity_indicators.items():
            if any(indicator in user_input for indicator in indicators):
                return complexity
        return "simple"
    
    def get_classification_stats(self) -> Dict:
        """Get classification statistics."""
        return {
            "total_classifications": self.classifications_made,
            "classification_breakdown": {
                k.value: v for k, v in self.classification_stats.items()
            },
            "simple_response_rate": (
                self.classification_stats[TaskClassification.SIMPLE_RESPONSE] / 
                max(self.classifications_made, 1)
            )
        }