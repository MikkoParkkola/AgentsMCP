"""
Intent Analyzer - NLP-based intent extraction from user prompts

Analyzes user input to extract:
- Core intent and goals
- Action verbs and subjects 
- Technical domain and complexity
- Urgency and priority indicators
- Context requirements
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intents."""
    INFORMATION_SEEKING = "information_seeking"
    TASK_EXECUTION = "task_execution" 
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE_GENERATION = "creative_generation"
    ANALYSIS_REVIEW = "analysis_review"
    PLANNING_STRATEGY = "planning_strategy"
    LEARNING_TUTORIAL = "learning_tutorial"
    SOCIAL_CONVERSATIONAL = "social_conversational"


class TechnicalDomain(Enum):
    """Technical domains for context."""
    SOFTWARE_DEVELOPMENT = "software_development"
    DATA_SCIENCE = "data_science"
    SYSTEM_ADMINISTRATION = "system_administration"
    WEB_DEVELOPMENT = "web_development"
    MOBILE_DEVELOPMENT = "mobile_development"
    DEVOPS_INFRASTRUCTURE = "devops_infrastructure"
    SECURITY_PRIVACY = "security_privacy"
    BUSINESS_PROCESS = "business_process"
    GENERAL_TECHNICAL = "general_technical"
    NON_TECHNICAL = "non_technical"


class UrgencyLevel(Enum):
    """Urgency/priority levels."""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class ExtractedEntity:
    """Entity extracted from user input."""
    text: str
    entity_type: str  # person, place, technology, concept, etc.
    confidence: float
    context: Optional[str] = None


@dataclass 
class IntentAnalysis:
    """Complete intent analysis result."""
    # Core intent
    primary_intent: IntentType
    secondary_intents: List[IntentType] = field(default_factory=list)
    confidence: float = 0.0
    
    # Action analysis
    action_verbs: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    
    # Context and domain
    technical_domain: TechnicalDomain = TechnicalDomain.GENERAL_TECHNICAL
    complexity_level: str = "medium"  # low, medium, high, very_high
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    
    # Entities and keywords  
    entities: List[ExtractedEntity] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    
    # Requirements
    has_constraints: bool = False
    constraint_types: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    
    # Ambiguity indicators
    ambiguous_terms: List[str] = field(default_factory=list)
    missing_context: List[str] = field(default_factory=list)
    assumptions_needed: List[str] = field(default_factory=list)
    
    # Processing metadata
    processing_time_ms: int = 0
    language: str = "en"
    raw_input_length: int = 0


class IntentAnalyzer:
    """
    Advanced NLP-based intent analyzer for user prompts.
    
    Extracts comprehensive intent information from user input including
    goals, context, complexity, and ambiguity indicators.
    """
    
    def __init__(self):
        """Initialize the intent analyzer."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Build pattern libraries
        self.intent_patterns = self._build_intent_patterns()
        self.action_patterns = self._build_action_patterns()
        self.domain_patterns = self._build_domain_patterns()
        self.complexity_patterns = self._build_complexity_patterns()
        self.urgency_patterns = self._build_urgency_patterns()
        self.constraint_patterns = self._build_constraint_patterns()
        self.ambiguity_patterns = self._build_ambiguity_patterns()
        self.technology_patterns = self._build_technology_patterns()
        
        # Statistics tracking
        self.analyses_performed = 0
        self.intent_distribution = {intent: 0 for intent in IntentType}
        
        self.logger.info("IntentAnalyzer initialized with comprehensive pattern libraries")
    
    async def analyze_intent(self, user_input: str, context: Optional[Dict] = None) -> IntentAnalysis:
        """Analyze user input and extract comprehensive intent information."""
        start_time = asyncio.get_event_loop().time()
        self.analyses_performed += 1
        
        if context is None:
            context = {}
            
        # Normalize input
        normalized_input = self._normalize_input(user_input)
        
        # Extract core intent
        primary_intent, secondary_intents, intent_confidence = self._analyze_primary_intent(normalized_input)
        self.intent_distribution[primary_intent] += 1
        
        # Extract actions and subjects
        action_verbs = self._extract_action_verbs(normalized_input)
        subjects, objects = self._extract_subjects_objects(normalized_input)
        
        # Analyze technical domain and complexity
        domain = self._identify_technical_domain(normalized_input)
        complexity = self._assess_complexity(normalized_input)
        urgency = self._assess_urgency(normalized_input)
        
        # Extract entities and keywords
        entities = self._extract_entities(normalized_input)
        keywords = self._extract_keywords(normalized_input)
        technologies = self._extract_technologies(normalized_input)
        
        # Analyze constraints and requirements
        has_constraints, constraint_types = self._analyze_constraints(normalized_input)
        success_criteria = self._extract_success_criteria(normalized_input)
        
        # Identify ambiguity indicators
        ambiguous_terms = self._identify_ambiguous_terms(normalized_input)
        missing_context = self._identify_missing_context(normalized_input, context)
        assumptions_needed = self._identify_assumptions_needed(normalized_input)
        
        processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        analysis = IntentAnalysis(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=intent_confidence,
            action_verbs=action_verbs,
            subjects=subjects,
            objects=objects,
            technical_domain=domain,
            complexity_level=complexity,
            urgency=urgency,
            entities=entities,
            keywords=keywords,
            technologies=technologies,
            has_constraints=has_constraints,
            constraint_types=constraint_types,
            success_criteria=success_criteria,
            ambiguous_terms=ambiguous_terms,
            missing_context=missing_context,
            assumptions_needed=assumptions_needed,
            processing_time_ms=processing_time,
            language=self._detect_language(user_input),
            raw_input_length=len(user_input)
        )
        
        self.logger.debug(f"Intent analysis completed in {processing_time}ms: {primary_intent.value}")
        return analysis
    
    def _normalize_input(self, text: str) -> str:
        """Normalize user input for analysis."""
        # Basic normalization
        text = text.strip().lower()
        # Preserve punctuation for sentence boundary detection
        return text
    
    def _analyze_primary_intent(self, text: str) -> Tuple[IntentType, List[IntentType], float]:
        """Analyze primary and secondary intents."""
        intent_scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0.0
            pattern_matches = 0
            
            for pattern_info in patterns:
                if re.search(pattern_info["pattern"], text, re.IGNORECASE):
                    score += pattern_info["weight"]
                    pattern_matches += 1
            
            if pattern_matches > 0:
                # Normalize by pattern count to avoid bias toward intents with more patterns
                intent_scores[intent_type] = score / max(pattern_matches, 1)
        
        if not intent_scores:
            return IntentType.INFORMATION_SEEKING, [], 0.3
        
        # Find primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        confidence = min(primary_intent[1], 0.95)  # Cap confidence
        
        # Find secondary intents (scores within 70% of primary)
        threshold = primary_intent[1] * 0.7
        secondary_intents = [
            intent for intent, score in intent_scores.items() 
            if intent != primary_intent[0] and score >= threshold
        ]
        
        return primary_intent[0], secondary_intents, confidence
    
    def _extract_action_verbs(self, text: str) -> List[str]:
        """Extract action verbs from text."""
        actions = []
        
        for category, patterns in self.action_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    actions.extend([match.lower() for match in matches if isinstance(match, str)])
        
        return list(set(actions))  # Remove duplicates
    
    def _extract_subjects_objects(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract subjects and objects from text using simple patterns."""
        # This is a simplified implementation - in production you'd use proper NLP
        subjects = []
        objects = []
        
        # Look for patterns like "create a [object]", "analyze the [subject]"
        subject_patterns = [
            r'\b(?:the|this|my|our|your)\s+(\w+(?:\s+\w+)?)\b',
            r'\banalyze\s+(?:the\s+)?(\w+(?:\s+\w+)?)\b',
            r'\breview\s+(?:the\s+)?(\w+(?:\s+\w+)?)\b'
        ]
        
        object_patterns = [
            r'\bcreate\s+(?:a\s+)?(\w+(?:\s+\w+)?)\b',
            r'\bbuild\s+(?:a\s+)?(\w+(?:\s+\w+)?)\b',
            r'\bimplement\s+(?:a\s+)?(\w+(?:\s+\w+)?)\b'
        ]
        
        for pattern in subject_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            subjects.extend(matches)
        
        for pattern in object_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            objects.extend(matches)
        
        return subjects[:5], objects[:5]  # Limit results
    
    def _identify_technical_domain(self, text: str) -> TechnicalDomain:
        """Identify the primary technical domain."""
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        if not domain_scores:
            return TechnicalDomain.GENERAL_TECHNICAL
        
        return max(domain_scores.items(), key=lambda x: x[1])[0]
    
    def _assess_complexity(self, text: str) -> str:
        """Assess task complexity level."""
        complexity_scores = {"low": 0, "medium": 0, "high": 0, "very_high": 0}
        
        for level, patterns in self.complexity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    complexity_scores[level] += 1
        
        # Default to medium if no clear indicators
        if sum(complexity_scores.values()) == 0:
            return "medium"
        
        return max(complexity_scores.items(), key=lambda x: x[1])[0]
    
    def _assess_urgency(self, text: str) -> UrgencyLevel:
        """Assess urgency/priority level."""
        urgency_scores = {urgency: 0 for urgency in UrgencyLevel}
        
        for urgency, patterns in self.urgency_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    urgency_scores[urgency] += 1
        
        if sum(urgency_scores.values()) == 0:
            return UrgencyLevel.MEDIUM
        
        return max(urgency_scores.items(), key=lambda x: x[1])[0]
    
    def _extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract named entities from text."""
        entities = []
        
        # Simple entity extraction patterns
        entity_patterns = {
            "technology": r'\b(python|javascript|react|django|flask|kubernetes|docker|aws|azure|gcp)\b',
            "file_type": r'\b\w+\.(?:py|js|html|css|json|yaml|yml|md|txt|sql|sh)\b',
            "concept": r'\b(?:api|database|frontend|backend|microservice|algorithm|architecture)\b'
        }
        
        for entity_type, pattern in entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    confidence=0.8,
                    context=text[max(0, match.start()-20):match.end()+20]
                ))
        
        return entities[:10]  # Limit results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms and concepts."""
        # Filter out common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'i', 'you', 'we', 'they', 'he', 'she', 'it', 'this', 'that', 'these', 
            'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
            'should', 'may', 'might', 'must', 'can'
        }
        
        # Extract words of 3+ characters that aren't stop words
        words = re.findall(r'\b\w{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Count frequency and return most common
        word_counts = {}
        for word in keywords:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return top keywords
        sorted_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_keywords[:15]]
    
    def _extract_technologies(self, text: str) -> List[str]:
        """Extract technology mentions."""
        technologies = []
        
        for tech_category, patterns in self.technology_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    technologies.extend([match.lower() for match in matches if isinstance(match, str)])
        
        return list(set(technologies))[:10]  # Remove duplicates and limit
    
    def _analyze_constraints(self, text: str) -> Tuple[bool, List[str]]:
        """Analyze constraints and requirements."""
        constraint_types = []
        
        for constraint_type, patterns in self.constraint_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    constraint_types.append(constraint_type)
                    break  # Only count each constraint type once
        
        return len(constraint_types) > 0, constraint_types
    
    def _extract_success_criteria(self, text: str) -> List[str]:
        """Extract success criteria and goals."""
        criteria_patterns = [
            r'\bso that\s+(.+?)(?:\.|$)',
            r'\bshould\s+(.+?)(?:\.|$)', 
            r'\bmust\s+(.+?)(?:\.|$)',
            r'\bneed(?:s)?\s+to\s+(.+?)(?:\.|$)',
            r'\bgoal\s+is\s+to\s+(.+?)(?:\.|$)'
        ]
        
        criteria = []
        for pattern in criteria_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            criteria.extend([match.strip() for match in matches])
        
        return criteria[:5]  # Limit results
    
    def _identify_ambiguous_terms(self, text: str) -> List[str]:
        """Identify potentially ambiguous terms."""
        ambiguous = []
        
        for ambiguity_type, patterns in self.ambiguity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    ambiguous.extend([match.lower() for match in matches if isinstance(match, str)])
        
        return list(set(ambiguous))[:10]  # Remove duplicates and limit
    
    def _identify_missing_context(self, text: str, context: Dict) -> List[str]:
        """Identify potentially missing context."""
        missing = []
        
        # Check for pronouns without clear antecedents
        pronouns = re.findall(r'\b(it|this|that|they|them)\b', text, re.IGNORECASE)
        if pronouns and not context.get('previous_subjects'):
            missing.append("unclear_references")
        
        # Check for incomplete specifications
        incomplete_patterns = [
            r'\b(?:the|my|our)\s+(\w+)\b(?!\s+(?:is|was|will|should|can|must))',
            r'\bfile\b(?!\s+\w+\.\w+)',  # "file" without extension
            r'\bproject\b(?!\s+\w+)',    # "project" without name
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                missing.append("incomplete_specification")
                break
        
        return missing[:5]
    
    def _identify_assumptions_needed(self, text: str) -> List[str]:
        """Identify assumptions that might need to be made."""
        assumptions = []
        
        # Check for vague quantifiers
        if re.search(r'\b(some|many|few|several|multiple)\b', text, re.IGNORECASE):
            assumptions.append("vague_quantities")
        
        # Check for unspecified technologies
        if re.search(r'\b(language|framework|database|system)\b(?!\s+\w)', text, re.IGNORECASE):
            assumptions.append("unspecified_technologies")
        
        # Check for unspecified scope
        if re.search(r'\b(improve|optimize|fix)\b(?!\s+\w)', text, re.IGNORECASE):
            assumptions.append("unspecified_scope")
        
        return assumptions
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # This is a placeholder - in production you'd use a proper language detection library
        return "en"  # Assume English for now
    
    # Pattern building methods
    def _build_intent_patterns(self) -> Dict[IntentType, List[Dict]]:
        """Build intent recognition patterns."""
        return {
            IntentType.INFORMATION_SEEKING: [
                {"pattern": r'\b(what|how|why|when|where|who)\b', "weight": 0.8},
                {"pattern": r'\b(explain|describe|tell me|show me)\b', "weight": 0.9},
                {"pattern": r'\b(learn|understand|know)\b', "weight": 0.7},
            ],
            IntentType.TASK_EXECUTION: [
                {"pattern": r'\b(create|build|make|develop|implement|generate)\b', "weight": 0.9},
                {"pattern": r'\b(do|execute|run|perform|complete)\b', "weight": 0.8},
                {"pattern": r'\b(write|code|program|script)\b', "weight": 0.9},
            ],
            IntentType.PROBLEM_SOLVING: [
                {"pattern": r'\b(fix|solve|debug|troubleshoot|resolve)\b', "weight": 0.9},
                {"pattern": r'\b(error|bug|issue|problem|broken)\b', "weight": 0.8},
                {"pattern": r'\b(not working|doesn\'t work|failing)\b', "weight": 0.8},
            ],
            IntentType.CREATIVE_GENERATION: [
                {"pattern": r'\b(design|creative|innovative|novel)\b', "weight": 0.8},
                {"pattern": r'\b(brainstorm|ideate|come up with)\b', "weight": 0.9},
                {"pattern": r'\b(artistic|creative|unique)\b', "weight": 0.7},
            ],
            IntentType.ANALYSIS_REVIEW: [
                {"pattern": r'\b(analyze|review|examine|assess|evaluate)\b', "weight": 0.9},
                {"pattern": r'\b(audit|inspect|check|validate)\b', "weight": 0.8},
                {"pattern": r'\b(quality|performance|security)\b.*\b(review|check|analysis)\b', "weight": 0.9},
            ],
            IntentType.PLANNING_STRATEGY: [
                {"pattern": r'\b(plan|strategy|roadmap|approach)\b', "weight": 0.9},
                {"pattern": r'\b(organize|structure|prioritize)\b', "weight": 0.8},
                {"pattern": r'\b(next steps|what should|how to proceed)\b', "weight": 0.8},
            ],
            IntentType.LEARNING_TUTORIAL: [
                {"pattern": r'\b(tutorial|guide|walkthrough|step.by.step)\b', "weight": 0.9},
                {"pattern": r'\b(teach|learn|study|practice)\b', "weight": 0.8},
                {"pattern": r'\b(beginner|getting started|introduction)\b', "weight": 0.7},
            ],
            IntentType.SOCIAL_CONVERSATIONAL: [
                {"pattern": r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b', "weight": 0.9},
                {"pattern": r'\b(how are you|thank you|thanks|please)\b', "weight": 0.8},
                {"pattern": r'\b(chat|talk|conversation)\b', "weight": 0.7},
            ]
        }
    
    def _build_action_patterns(self) -> Dict[str, List[str]]:
        """Build action verb patterns."""
        return {
            "creation": [r'\b(create|build|make|generate|produce|construct|develop)\b'],
            "modification": [r'\b(edit|modify|update|change|alter|adjust|improve|enhance)\b'],
            "analysis": [r'\b(analyze|examine|review|assess|evaluate|study|investigate)\b'],
            "execution": [r'\b(run|execute|perform|do|implement|deploy|launch)\b'],
            "communication": [r'\b(send|share|communicate|notify|inform|report)\b'],
            "management": [r'\b(manage|organize|coordinate|supervise|control|maintain)\b']
        }
    
    def _build_domain_patterns(self) -> Dict[TechnicalDomain, List[str]]:
        """Build technical domain patterns."""
        return {
            TechnicalDomain.SOFTWARE_DEVELOPMENT: [
                r'\b(code|programming|software|application|development)\b',
                r'\b(function|class|method|variable|algorithm)\b',
                r'\b(debug|refactor|optimize|testing|deployment)\b'
            ],
            TechnicalDomain.WEB_DEVELOPMENT: [
                r'\b(website|web|frontend|backend|html|css|javascript)\b',
                r'\b(react|vue|angular|node|express|django|flask)\b',
                r'\b(api|rest|graphql|http|server|client)\b'
            ],
            TechnicalDomain.DATA_SCIENCE: [
                r'\b(data|analysis|machine learning|ai|statistics)\b',
                r'\b(python|pandas|numpy|sklearn|tensorflow|pytorch)\b',
                r'\b(model|algorithm|dataset|visualization|prediction)\b'
            ],
            TechnicalDomain.DEVOPS_INFRASTRUCTURE: [
                r'\b(deploy|infrastructure|server|cloud|docker|kubernetes)\b',
                r'\b(aws|azure|gcp|ci/cd|automation|monitoring)\b',
                r'\b(configuration|scaling|load balancing|networking)\b'
            ],
            TechnicalDomain.SECURITY_PRIVACY: [
                r'\b(security|privacy|encryption|authentication|authorization)\b',
                r'\b(vulnerability|threat|compliance|audit|pentest)\b',
                r'\b(ssl|https|certificate|firewall|access control)\b'
            ]
        }
    
    def _build_complexity_patterns(self) -> Dict[str, List[str]]:
        """Build complexity assessment patterns."""
        return {
            "low": [
                r'\b(simple|easy|quick|basic|straightforward)\b',
                r'\b(small|minor|trivial)\b'
            ],
            "medium": [
                r'\b(moderate|standard|normal|typical)\b',
                r'\b(implement|create|build)\b(?!\s+\b(system|architecture|framework)\b)'
            ],
            "high": [
                r'\b(complex|advanced|sophisticated|comprehensive)\b',
                r'\b(system|architecture|framework|platform)\b',
                r'\b(enterprise|production|scalable)\b'
            ],
            "very_high": [
                r'\b(distributed|microservices|multi-tenant|enterprise-grade)\b',
                r'\b(migrate|integration|end-to-end|full-scale)\b',
                r'\b(performance-critical|mission-critical|high-availability)\b'
            ]
        }
    
    def _build_urgency_patterns(self) -> Dict[UrgencyLevel, List[str]]:
        """Build urgency assessment patterns."""
        return {
            UrgencyLevel.CRITICAL: [
                r'\b(urgent|critical|emergency|asap|immediately)\b',
                r'\b(broken|down|failing|crashed)\b'
            ],
            UrgencyLevel.HIGH: [
                r'\b(important|priority|soon|quickly|fast)\b',
                r'\b(deadline|due|time-sensitive)\b'
            ],
            UrgencyLevel.MEDIUM: [
                r'\b(normal|standard|regular|typical)\b'
            ],
            UrgencyLevel.LOW: [
                r'\b(when you can|whenever|no rush|low priority)\b',
                r'\b(future|later|eventually)\b'
            ],
            UrgencyLevel.NONE: [
                r'\b(explore|research|investigate|consider)\b'
            ]
        }
    
    def _build_constraint_patterns(self) -> Dict[str, List[str]]:
        """Build constraint recognition patterns."""
        return {
            "time": [r'\b(deadline|due|by \w+|within \d+|time limit|schedule)\b'],
            "budget": [r'\b(budget|cost|expensive|cheap|affordable|price)\b'],
            "technology": [r'\b(using|with|in \w+|specific to|must use|only)\b'],
            "quality": [r'\b(performance|security|quality|standards|requirements)\b'],
            "scope": [r'\b(only|just|simple|minimal|basic|without)\b'],
            "compliance": [r'\b(compliance|regulation|standard|policy|guideline)\b']
        }
    
    def _build_ambiguity_patterns(self) -> Dict[str, List[str]]:
        """Build ambiguity detection patterns.""" 
        return {
            "vague_quantifiers": [r'\b(some|many|few|several|multiple|various)\b'],
            "unclear_references": [r'\b(it|this|that|they|them)(?!\s+\w+)\b'],
            "general_terms": [r'\b(thing|stuff|item|element|component)(?!\s+\w+)\b'],
            "subjective_terms": [r'\b(better|good|bad|nice|pretty|ugly|fast|slow)\b'],
            "incomplete_specs": [r'\b(file|project|system|application)(?!\s+\w+)\b']
        }
    
    def _build_technology_patterns(self) -> Dict[str, List[str]]:
        """Build technology detection patterns."""
        return {
            "programming_languages": [
                r'\b(python|javascript|java|c\+\+|c#|rust|go|typescript|php|ruby|kotlin|swift)\b'
            ],
            "web_frameworks": [
                r'\b(react|vue|angular|django|flask|express|node\.js|spring|laravel)\b'
            ],
            "databases": [
                r'\b(mysql|postgresql|mongodb|redis|sqlite|oracle|sql server|dynamodb)\b'
            ],
            "cloud_platforms": [
                r'\b(aws|azure|gcp|google cloud|amazon web services|microsoft azure)\b'
            ],
            "tools": [
                r'\b(docker|kubernetes|jenkins|git|github|gitlab|terraform|ansible)\b'
            ]
        }
    
    def get_analysis_stats(self) -> Dict:
        """Get intent analysis statistics."""
        return {
            "total_analyses": self.analyses_performed,
            "intent_distribution": {
                intent.value: count for intent, count in self.intent_distribution.items()
            },
            "most_common_intent": max(self.intent_distribution.items(), key=lambda x: x[1])[0].value if self.analyses_performed > 0 else None
        }