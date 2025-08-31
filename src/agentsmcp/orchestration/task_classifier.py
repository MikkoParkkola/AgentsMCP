"""Task classification system for dynamic agent loading.

This module implements the TaskClassifier that analyzes objectives and context
to determine appropriate task classifications, complexity levels, required roles,
and technology stacks.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone

from .models import (
    TaskClassification,
    TaskType,
    ComplexityLevel,
    RiskLevel,
    TechnologyStack,
    ClassificationCache,
    InvalidObjective,
    InsufficientContext,
    UnsupportedTaskType,
)


class TaskClassifier:
    """Classifier for analyzing tasks and determining appropriate agent assignments."""
    
    # Keyword mappings for task type detection with weighted priority
    TASK_TYPE_KEYWORDS = {
        TaskType.DESIGN: {
            "high": ["design", "architecture", "blueprint", "wireframe", "plan system", "architect"],
            "medium": ["plan", "diagram", "schema", "structure", "layout", "mockup", "prototype", "concept"]
        },
        TaskType.IMPLEMENTATION: {
            "high": ["implement", "build", "create", "develop", "code", "write code", "develop system"],
            "medium": ["construct", "generate", "produce", "make", "establish", "write"]
        },
        TaskType.REVIEW: {
            "high": ["review", "audit", "inspect", "evaluate", "audit security", "check quality", "check code"],
            "medium": ["check", "examine", "assess", "validate", "verify", "analyze quality"]
        },
        TaskType.ANALYSIS: {
            "high": ["analyze", "analysis", "study", "investigate"],
            "medium": ["research", "explore", "understand", "breakdown", "dissect", "examine"]
        },
        TaskType.TESTING: {
            "high": ["test", "testing", "qa", "quality assurance", "unit test", "integration test", "e2e"],
            "medium": ["verify", "validate", "check"]
        },
        TaskType.DOCUMENTATION: {
            "high": ["document", "documentation", "docs", "readme", "write guide", "write manual"],
            "medium": ["guide", "manual", "tutorial", "help", "explain", "describe"]
        },
        TaskType.REFACTORING: {
            "high": ["refactor", "refactoring", "restructure", "modernize"],
            "medium": ["cleanup", "optimize", "improve", "reorganize", "simplify"]
        },
        TaskType.BUG_FIX: {
            "high": ["fix bug", "fix error", "debug", "troubleshoot", "fix memory leak"],
            "medium": ["fix", "bug", "error", "issue", "problem", "resolve", "correct", "patch"]
        },
        TaskType.MAINTENANCE: {
            "high": ["maintenance", "migrate", "upgrade", "housekeeping"],
            "medium": ["maintain", "update", "patch", "cleanup", "dependency"]
        },
        TaskType.RESEARCH: {
            "high": ["research", "investigate", "explore", "proof of concept"],
            "medium": ["discover", "learn", "study", "experiment", "prototype"]
        }
    }
    
    # Role keyword mappings
    ROLE_KEYWORDS = {
        "architect": [
            "design", "architecture", "system", "plan", "structure",
            "blueprint", "strategy", "framework", "pattern"
        ],
        "coder": [
            "implement", "code", "develop", "build", "create",
            "function", "class", "module", "script", "program"
        ],
        "backend_engineer": [
            "backend", "server", "api", "database", "service",
            "microservice", "rest", "graphql", "endpoint"
        ],
        "web_frontend_engineer": [
            "frontend", "web", "react", "javascript", "typescript",
            "html", "css", "ui", "interface", "component"
        ],
        "tui_frontend_engineer": [
            "tui", "terminal", "cli", "console", "command line",
            "textual", "curses", "ncurses", "terminal ui"
        ],
        "api_engineer": [
            "api", "rest", "graphql", "endpoint", "route",
            "openapi", "swagger", "postman", "http"
        ],
        "qa": [
            "test", "qa", "quality", "verify", "validate",
            "check", "review", "audit", "inspect"
        ],
        "backend_qa_engineer": [
            "backend test", "api test", "integration test",
            "database test", "server test", "service test"
        ],
        "web_frontend_qa_engineer": [
            "frontend test", "ui test", "web test", "e2e test",
            "component test", "browser test", "selenium"
        ],
        "tui_frontend_qa_engineer": [
            "tui test", "terminal test", "cli test",
            "console test", "command test"
        ],
        "chief_qa_engineer": [
            "comprehensive test", "test strategy", "quality assurance",
            "test plan", "qa lead", "testing framework"
        ],
        "business_analyst": [
            "requirement", "analysis", "business", "stakeholder",
            "specification", "user story", "acceptance criteria"
        ],
        "docs": [
            "documentation", "readme", "guide", "manual",
            "tutorial", "help", "explain", "document"
        ],
        "ci_cd_engineer": [
            "ci", "cd", "pipeline", "deploy", "deployment",
            "build", "automation", "github actions", "docker"
        ],
        "dev_tooling_engineer": [
            "tooling", "devtools", "developer tools", "automation",
            "scripts", "utilities", "cli tools", "workflow"
        ],
        "data_analyst": [
            "data", "analytics", "sql", "query", "report",
            "dashboard", "metrics", "statistics", "visualization"
        ],
        "data_scientist": [
            "data science", "machine learning", "statistics",
            "analysis", "modeling", "prediction", "algorithm"
        ],
        "ml_scientist": [
            "machine learning research", "ml research", "deep learning",
            "neural network", "ai research", "model research"
        ],
        "ml_engineer": [
            "ml engineering", "model deployment", "training",
            "inference", "mlops", "feature engineering"
        ],
        "it_lawyer": [
            "legal", "compliance", "gdpr", "privacy", "license",
            "copyright", "terms", "policy", "regulation"
        ],
        "marketing_manager": [
            "marketing", "seo", "content", "promotion",
            "branding", "outreach", "communication"
        ]
    }
    
    # Technology stack detection
    TECHNOLOGY_KEYWORDS = {
        TechnologyStack.PYTHON: [
            "python", "py", "pip", "conda", "django", "flask",
            "fastapi", "pydantic", "pandas", "numpy", "pytest"
        ],
        TechnologyStack.JAVASCRIPT: [
            "javascript", "js", "npm", "yarn", "node", "v8",
            "es6", "es2015", "babel", "webpack"
        ],
        TechnologyStack.TYPESCRIPT: [
            "typescript", "ts", "tsc", "type", "interface",
            "generic", "decorator", "ambient"
        ],
        TechnologyStack.REACT: [
            "react", "jsx", "tsx", "component", "hook",
            "state", "props", "redux", "context"
        ],
        TechnologyStack.NODEJS: [
            "node", "nodejs", "npm", "express", "koa",
            "hapi", "nestjs", "socket.io"
        ],
        TechnologyStack.API: [
            "api", "rest", "restful", "graphql", "grpc",
            "openapi", "swagger", "endpoint", "route"
        ],
        TechnologyStack.DATABASE: [
            "database", "db", "sql", "nosql", "postgres",
            "mysql", "mongodb", "redis", "elasticsearch"
        ],
        TechnologyStack.DEVOPS: [
            "devops", "docker", "kubernetes", "aws", "gcp",
            "azure", "ci", "cd", "pipeline", "terraform"
        ],
        TechnologyStack.TESTING: [
            "test", "testing", "unittest", "pytest", "jest",
            "mocha", "cypress", "selenium", "testcafe"
        ],
        TechnologyStack.DOCUMENTATION: [
            "docs", "documentation", "markdown", "sphinx",
            "gitbook", "readme", "wiki", "guide"
        ],
        TechnologyStack.TUI: [
            "tui", "terminal", "console", "cli", "textual",
            "curses", "ncurses", "command line"
        ],
        TechnologyStack.CLI: [
            "cli", "command line", "terminal", "shell",
            "bash", "zsh", "script", "command"
        ],
        TechnologyStack.MACHINE_LEARNING: [
            "ml", "machine learning", "ai", "neural network",
            "deep learning", "tensorflow", "pytorch", "scikit"
        ],
        TechnologyStack.DATA_ANALYSIS: [
            "data analysis", "analytics", "pandas", "numpy",
            "matplotlib", "seaborn", "plotly", "jupyter"
        ]
    }
    
    # Complexity indicators
    COMPLEXITY_INDICATORS = {
        ComplexityLevel.TRIVIAL: {
            "keywords": ["simple", "basic", "small", "quick", "minor"],
            "max_length": 50,
            "max_effort": 10
        },
        ComplexityLevel.LOW: {
            "keywords": ["easy", "straightforward", "standard", "routine"],
            "max_length": 150,
            "max_effort": 25
        },
        ComplexityLevel.MEDIUM: {
            "keywords": ["moderate", "regular", "typical", "standard"],
            "max_length": 300,
            "max_effort": 50
        },
        ComplexityLevel.HIGH: {
            "keywords": ["complex", "advanced", "sophisticated", "comprehensive"],
            "max_length": 600,
            "max_effort": 80
        },
        ComplexityLevel.CRITICAL: {
            "keywords": ["critical", "enterprise", "large-scale", "mission-critical"],
            "max_length": float('inf'),
            "max_effort": 100
        }
    }
    
    def __init__(self):
        """Initialize the task classifier with caching support."""
        self._classification_cache: Dict[str, ClassificationCache] = {}
        self._cache_ttl = 3600  # 1 hour cache TTL
    
    def classify(
        self,
        objective: str,
        context: Optional[Dict] = None,
        constraints: Optional[Dict] = None
    ) -> TaskClassification:
        """Classify a task and determine required roles and technologies.
        
        Args:
            objective: The task objective/description
            context: Additional context including repository info, file paths, etc.
            constraints: Resource and other constraints
            
        Returns:
            TaskClassification with detected roles, technologies, and complexity
            
        Raises:
            InvalidObjective: If objective is empty or invalid
            InsufficientContext: If context is insufficient for classification
            UnsupportedTaskType: If task type is not supported
        """
        start_time = time.time()
        
        # Validate inputs
        if not objective or not objective.strip():
            raise InvalidObjective("Objective cannot be empty")
        
        objective = objective.strip()
        context = context or {}
        constraints = constraints or {}
        
        # Check cache
        cache_key = self._generate_cache_key(objective, context)
        cached = self._get_cached_classification(cache_key)
        if cached:
            cached.hit_count += 1
            return cached.classification
        
        try:
            # Extract keywords and normalize text
            text_to_analyze = self._prepare_text_for_analysis(objective, context)
            keywords = self._extract_keywords(text_to_analyze)
            
            # Classify task type
            task_type = self._detect_task_type(keywords, objective)
            
            # Assess complexity and effort
            complexity = self._assess_complexity(objective, keywords, context)
            effort = self._estimate_effort(complexity, keywords, len(objective))
            
            # Determine risk level
            risk_level = self._assess_risk(complexity, task_type, keywords)
            
            # Identify required and optional roles
            required_roles, optional_roles = self._determine_roles(
                task_type, keywords, complexity, context
            )
            
            # Detect technology stacks
            technologies = self._detect_technologies(keywords, context)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                keywords, task_type, required_roles, technologies
            )
            
            # Create classification
            classification = TaskClassification(
                task_type=task_type,
                complexity=complexity,
                required_roles=required_roles,
                optional_roles=optional_roles,
                technologies=technologies,
                estimated_effort=effort,
                risk_level=risk_level,
                keywords=keywords,
                confidence=confidence
            )
            
            # Cache the result
            self._cache_classification(cache_key, classification)
            
            # Performance check
            duration = time.time() - start_time
            if duration > 0.2:  # 200ms threshold
                print(f"Warning: Classification took {duration:.3f}s, exceeding 200ms threshold")
            
            return classification
            
        except Exception as e:
            if isinstance(e, (InvalidObjective, InsufficientContext, UnsupportedTaskType)):
                raise
            raise UnsupportedTaskType(f"Classification failed: {str(e)}") from e
    
    def _prepare_text_for_analysis(self, objective: str, context: Dict) -> str:
        """Prepare and combine text for analysis."""
        text_parts = [objective.lower()]
        
        # Add context information
        if context.get('repo'):
            text_parts.append(str(context['repo']).lower())
        
        if context.get('module'):
            text_parts.append(str(context['module']).lower())
        
        if context.get('file_paths'):
            paths = context['file_paths']
            if isinstance(paths, list):
                text_parts.extend(path.lower() for path in paths if isinstance(path, str))
        
        if context.get('technologies'):
            techs = context['technologies']
            if isinstance(techs, list):
                text_parts.extend(tech.lower() for tech in techs if isinstance(tech, str))
        
        return ' '.join(text_parts)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Extract words and clean them
        words = re.findall(r'\b[a-z]+\b', text.lower())
        keywords = []
        
        for word in words:
            if len(word) > 2 and word not in stop_words:
                keywords.append(word)
        
        # Add multi-word phrases
        phrases = self._extract_phrases(text)
        keywords.extend(phrases)
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful multi-word phrases."""
        phrases = []
        
        # Common technical phrases
        phrase_patterns = [
            r'\b(?:machine learning|data science|deep learning|neural network)\b',
            r'\b(?:api design|rest api|graphql api)\b',
            r'\b(?:frontend development|backend development)\b',
            r'\b(?:unit test|integration test|e2e test)\b',
            r'\b(?:code review|security audit)\b',
            r'\b(?:ci/cd|continuous integration|continuous deployment)\b',
            r'\b(?:user interface|user experience|ui/ux)\b',
            r'\b(?:database design|schema design)\b',
            r'\b(?:terminal ui|command line interface)\b',
            r'\b(?:quality assurance|test automation)\b'
        ]
        
        for pattern in phrase_patterns:
            matches = re.findall(pattern, text.lower())
            phrases.extend(matches)
        
        return phrases
    
    def _detect_task_type(self, keywords: List[str], objective: str) -> TaskType:
        """Detect the primary task type based on weighted keywords."""
        scores = {}
        
        # Check keywords with weighted scoring
        for task_type, priority_keywords in self.TASK_TYPE_KEYWORDS.items():
            score = 0
            
            # High priority keywords get more weight
            for keyword in keywords:
                for high_keyword in priority_keywords["high"]:
                    if high_keyword in keyword or keyword in high_keyword:
                        score += 3  # High priority weight
                
                for medium_keyword in priority_keywords["medium"]:
                    if medium_keyword in keyword or keyword in medium_keyword:
                        score += 1  # Medium priority weight
            
            scores[task_type] = score
        
        # If no clear winner from keywords, analyze objective directly
        if max(scores.values()) == 0:
            objective_lower = objective.lower()
            for task_type, priority_keywords in self.TASK_TYPE_KEYWORDS.items():
                # Check high priority keywords in objective
                for high_keyword in priority_keywords["high"]:
                    if high_keyword in objective_lower:
                        scores[task_type] += 5  # Even higher weight for direct objective matches
                
                # Check medium priority keywords in objective
                for medium_keyword in priority_keywords["medium"]:
                    if medium_keyword in objective_lower:
                        scores[task_type] += 2
        
        # Special case handling for ambiguous cases
        objective_lower = objective.lower()
        
        # If both design and implement are present, determine primary intent FIRST
        if ("design" in objective_lower and "implement" in objective_lower and 
            len(objective_lower) > 80):  # Long objectives with both design and implement
            # Check which comes first or has more context
            design_pos = objective_lower.find("design")
            implement_pos = objective_lower.find("implement") 
            if design_pos < implement_pos and design_pos != -1:
                return TaskType.DESIGN
            elif implement_pos < design_pos and implement_pos != -1:
                # If implement comes first, it's likely an implementation task
                return TaskType.IMPLEMENTATION
        
        # Strong indicators that override scoring - but after design/implement check
        if "comprehensive testing" in objective_lower or ("test" in objective_lower and "comprehensive" in objective_lower and 
                                                         not ("design" in objective_lower or "implement" in objective_lower)):
            return TaskType.TESTING
        
        # Audit + security should be review, not implementation 
        if "audit" in objective_lower and "security" in objective_lower:
            return TaskType.REVIEW
        
        if max(scores.values()) == 0:
            return TaskType.IMPLEMENTATION  # Default fallback
        
        return max(scores, key=scores.get)
    
    def _assess_complexity(
        self,
        objective: str,
        keywords: List[str],
        context: Dict
    ) -> ComplexityLevel:
        """Assess task complexity based on multiple factors."""
        obj_length = len(objective)
        
        # Check for explicit complexity indicators first
        for level, indicators in self.COMPLEXITY_INDICATORS.items():
            for indicator in indicators["keywords"]:
                if indicator in objective.lower():
                    return level
        
        # Adjust based on keyword complexity first (higher priority than length)
        complexity_boosters = [
            "enterprise", "scalable", "distributed", "microservice",
            "architecture", "framework", "comprehensive", "advanced",
            "optimization", "performance", "security", "integration",
            "authentication", "authorization", "e-commerce", "platform",
            "memory leak", "session management", "database", "production",
            "critical", "system", "concurrent", "threading"
        ]
        
        boost_count = sum(1 for keyword in keywords if any(
            booster in keyword for booster in complexity_boosters
        ))
        
        # Also check the objective text directly for complex patterns
        objective_lower = objective.lower()
        if ("memory" in keywords and "leak" in keywords) or "memory leak" in objective_lower:
            boost_count += 1
        if ("session" in keywords and "management" in keywords) or "session management" in objective_lower:
            boost_count += 1
        
        # Strong boost for complex keywords
        if boost_count >= 4:
            return ComplexityLevel.CRITICAL
        elif boost_count >= 3:
            return ComplexityLevel.HIGH
        elif boost_count >= 2:
            return ComplexityLevel.MEDIUM
        elif boost_count >= 1:
            # If we have complexity boosters but short text, still boost from trivial
            if obj_length <= 50:
                return ComplexityLevel.LOW
            else:
                return ComplexityLevel.MEDIUM
        
        # Length-based assessment (fallback when no complexity boosters)
        if obj_length <= 20:  # Very short objectives like "Fix typo"
            return ComplexityLevel.TRIVIAL
        elif obj_length <= 50:
            return ComplexityLevel.TRIVIAL
        elif obj_length <= 150:
            return ComplexityLevel.LOW
        elif obj_length <= 300:
            return ComplexityLevel.MEDIUM
        elif obj_length <= 600:
            return ComplexityLevel.HIGH
        else:
            return ComplexityLevel.CRITICAL
    
    def _estimate_effort(
        self,
        complexity: ComplexityLevel,
        keywords: List[str],
        objective_length: int
    ) -> int:
        """Estimate effort on a 1-100 scale."""
        base_effort = {
            ComplexityLevel.TRIVIAL: 5,
            ComplexityLevel.LOW: 15,
            ComplexityLevel.MEDIUM: 35,
            ComplexityLevel.HIGH: 65,
            ComplexityLevel.CRITICAL: 85
        }[complexity]
        
        # Adjust based on keyword density and objective length
        keyword_factor = min(len(keywords) / 10, 0.3)  # Max 30% adjustment
        length_factor = min(objective_length / 500, 0.2)  # Max 20% adjustment
        
        adjusted_effort = base_effort * (1 + keyword_factor + length_factor)
        
        return min(int(adjusted_effort), 100)
    
    def _assess_risk(
        self,
        complexity: ComplexityLevel,
        task_type: TaskType,
        keywords: List[str]
    ) -> RiskLevel:
        """Assess risk level based on complexity and task characteristics."""
        base_risk = {
            ComplexityLevel.TRIVIAL: RiskLevel.LOW,
            ComplexityLevel.LOW: RiskLevel.LOW,
            ComplexityLevel.MEDIUM: RiskLevel.MEDIUM,
            ComplexityLevel.HIGH: RiskLevel.HIGH,
            ComplexityLevel.CRITICAL: RiskLevel.CRITICAL
        }[complexity]
        
        # High-risk task types
        high_risk_types = {TaskType.REFACTORING, TaskType.MAINTENANCE}
        if task_type in high_risk_types:
            risk_levels = list(RiskLevel)
            current_idx = risk_levels.index(base_risk)
            if current_idx < len(risk_levels) - 1:
                base_risk = risk_levels[current_idx + 1]
        
        # High-risk keywords
        high_risk_keywords = [
            "security", "production", "database", "migration",
            "breaking", "legacy", "critical", "enterprise"
        ]
        
        risk_count = sum(1 for keyword in keywords if any(
            risk_keyword in keyword for risk_keyword in high_risk_keywords
        ))
        
        if risk_count >= 2:
            risk_levels = list(RiskLevel)
            current_idx = risk_levels.index(base_risk)
            if current_idx < len(risk_levels) - 1:
                base_risk = risk_levels[current_idx + 1]
        
        return base_risk
    
    def _determine_roles(
        self,
        task_type: TaskType,
        keywords: List[str],
        complexity: ComplexityLevel,
        context: Dict
    ) -> Tuple[List[str], List[str]]:
        """Determine required and optional roles for the task."""
        role_scores = {}
        
        # Score roles based on keywords
        for role, role_keywords in self.ROLE_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                for role_keyword in role_keywords:
                    if role_keyword in keyword or keyword in role_keyword:
                        score += 1
            role_scores[role] = score
        
        # Add base roles based on task type
        base_roles = {
            TaskType.DESIGN: ["architect"],
            TaskType.IMPLEMENTATION: ["coder"],
            TaskType.REVIEW: ["qa"],
            TaskType.ANALYSIS: ["business_analyst"],
            TaskType.TESTING: ["qa"],
            TaskType.DOCUMENTATION: ["docs"],
            TaskType.REFACTORING: ["coder", "architect"],
            TaskType.BUG_FIX: ["coder"],
            TaskType.MAINTENANCE: ["coder"],
            TaskType.RESEARCH: ["business_analyst"]
        }
        
        required_roles = base_roles.get(task_type, ["coder"])
        
        # Add high-scoring roles to required
        threshold = max(2, len(keywords) // 5)  # Dynamic threshold
        for role, score in role_scores.items():
            if score >= threshold and role not in required_roles:
                required_roles.append(role)
        
        # Optional roles (lower scoring but still relevant)
        optional_threshold = max(1, threshold // 2)
        optional_roles = []
        for role, score in role_scores.items():
            if optional_threshold <= score < threshold and role not in required_roles:
                optional_roles.append(role)
        
        # Add complexity-based roles
        if complexity in {ComplexityLevel.HIGH, ComplexityLevel.CRITICAL}:
            if "architect" not in required_roles and "architect" not in optional_roles:
                optional_roles.append("architect")
            if "chief_qa_engineer" not in required_roles and "chief_qa_engineer" not in optional_roles:
                optional_roles.append("chief_qa_engineer")
        
        return required_roles, optional_roles
    
    def _detect_technologies(self, keywords: List[str], context: Dict) -> List[TechnologyStack]:
        """Detect relevant technology stacks."""
        tech_scores = {}
        
        for tech, tech_keywords in self.TECHNOLOGY_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                for tech_keyword in tech_keywords:
                    if tech_keyword in keyword or keyword in tech_keyword:
                        score += 1
            tech_scores[tech] = score
        
        # Add technologies from context
        if context.get('technologies'):
            context_techs = context['technologies']
            if isinstance(context_techs, list):
                for tech_str in context_techs:
                    try:
                        tech = TechnologyStack(tech_str.lower())
                        tech_scores[tech] = tech_scores.get(tech, 0) + 2
                    except ValueError:
                        pass  # Unknown technology
        
        # Return technologies with scores > 0
        return [tech for tech, score in tech_scores.items() if score > 0]
    
    def _calculate_confidence(
        self,
        keywords: List[str],
        task_type: TaskType,
        required_roles: List[str],
        technologies: List[TechnologyStack]
    ) -> float:
        """Calculate classification confidence score."""
        confidence = 0.5  # Base confidence
        
        # Keyword quality (more specific keywords = higher confidence)
        if len(keywords) >= 5:
            confidence += 0.2
        elif len(keywords) >= 3:
            confidence += 0.1
        
        # Role determination confidence
        if len(required_roles) > 0:
            confidence += 0.2
        
        # Technology detection confidence
        if len(technologies) > 0:
            confidence += 0.1
        
        # Task type confidence (some types are easier to detect)
        high_confidence_types = {
            TaskType.TESTING, TaskType.DOCUMENTATION,
            TaskType.BUG_FIX, TaskType.REVIEW
        }
        if task_type in high_confidence_types:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_cache_key(self, objective: str, context: Dict) -> str:
        """Generate a cache key for the classification."""
        content = f"{objective}|{sorted(context.items())}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cached_classification(self, cache_key: str) -> Optional[ClassificationCache]:
        """Retrieve cached classification if available and not expired."""
        if cache_key not in self._classification_cache:
            return None
        
        cached = self._classification_cache[cache_key]
        if cached.is_expired(self._cache_ttl):
            del self._classification_cache[cache_key]
            return None
        
        return cached
    
    def _cache_classification(self, cache_key: str, classification: TaskClassification):
        """Cache a classification result."""
        cache_entry = ClassificationCache(
            objective_hash=cache_key[:16],  # Shortened hash for display
            context_hash=cache_key[16:32],
            classification=classification,
            created_at=datetime.now(timezone.utc),
            hit_count=1
        )
        
        self._classification_cache[cache_key] = cache_entry
        
        # Cleanup old entries if cache gets too large
        if len(self._classification_cache) > 1000:
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Remove expired and least used cache entries."""
        now = datetime.now(timezone.utc)
        
        # Remove expired entries
        expired_keys = [
            key for key, entry in self._classification_cache.items()
            if entry.is_expired(self._cache_ttl)
        ]
        
        for key in expired_keys:
            del self._classification_cache[key]
        
        # If still too large, remove least used entries
        if len(self._classification_cache) > 800:
            sorted_entries = sorted(
                self._classification_cache.items(),
                key=lambda x: (x[1].hit_count, x[1].created_at)
            )
            
            # Remove bottom 20%
            remove_count = len(sorted_entries) // 5
            for key, _ in sorted_entries[:remove_count]:
                del self._classification_cache[key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_entries = len(self._classification_cache)
        total_hits = sum(entry.hit_count for entry in self._classification_cache.values())
        
        if total_entries == 0:
            return {
                "total_entries": 0,
                "total_hits": 0,
                "average_hits": 0,
                "cache_size_mb": 0
            }
        
        average_hits = total_hits / total_entries
        
        # Rough cache size estimation
        cache_size_mb = total_entries * 2  # Rough estimate: 2KB per entry
        
        return {
            "total_entries": total_entries,
            "total_hits": total_hits,
            "average_hits": round(average_hits, 2),
            "cache_size_mb": cache_size_mb / 1024  # Convert to MB
        }