"""Advanced command discovery engine for CLI v3.

Provides intelligent command discovery with fuzzy matching, skill-level filtering,
context-aware suggestions, and relevance scoring for optimal user experience.
"""

import asyncio
import difflib
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from ..models.command_models import ExecutionMode, SkillLevel
from ..models.registry_models import (
    CommandMetadata,
    CommandCategory,
    DiscoveryRequest,
    DiscoveryResult,
)
from .command_registry import CommandRegistry

logger = logging.getLogger(__name__)


@dataclass
class SearchIndex:
    """Optimized search index for fast text matching."""
    
    # Word-based indexes
    exact_words: Dict[str, Set[str]]  # word -> command names
    prefix_words: Dict[str, Set[str]]  # prefix -> command names
    fuzzy_variants: Dict[str, Set[str]]  # variant -> command names
    
    # N-gram indexes for fuzzy matching
    bigrams: Dict[str, Set[str]]  # bigram -> command names
    trigrams: Dict[str, Set[str]]  # trigram -> command names
    
    # Category and context indexes
    categories: Dict[str, Set[str]]  # category term -> command names
    contexts: Dict[str, Set[str]]  # context term -> command names
    
    def __init__(self):
        self.exact_words = defaultdict(set)
        self.prefix_words = defaultdict(set)
        self.fuzzy_variants = defaultdict(set)
        self.bigrams = defaultdict(set)
        self.trigrams = defaultdict(set)
        self.categories = defaultdict(set)
        self.contexts = defaultdict(set)


class FuzzyMatcher:
    """Advanced fuzzy string matching with context awareness."""
    
    @staticmethod
    def calculate_similarity(query: str, target: str) -> float:
        """Calculate similarity score between query and target strings.
        
        Uses multiple algorithms and combines scores for best results:
        - Exact match bonus
        - Prefix match bonus  
        - Levenshtein distance
        - Subsequence matching
        - Common n-grams
        
        Returns:
            Similarity score from 0.0 to 1.0
        """
        query_lower = query.lower().strip()
        target_lower = target.lower().strip()
        
        if not query_lower or not target_lower:
            return 0.0
        
        # Exact match
        if query_lower == target_lower:
            return 1.0
        
        # Exact substring match
        if query_lower in target_lower:
            return 0.9 + (len(query_lower) / len(target_lower)) * 0.1
        
        # Prefix match
        if target_lower.startswith(query_lower):
            return 0.8 + (len(query_lower) / len(target_lower)) * 0.2
        
        # Calculate multiple similarity metrics
        scores = []
        
        # Levenshtein-based similarity using difflib
        seq_matcher = difflib.SequenceMatcher(None, query_lower, target_lower)
        scores.append(seq_matcher.ratio())
        
        # Word-level matching for multi-word targets
        query_words = query_lower.split()
        target_words = target_lower.split()
        
        if len(query_words) > 1 or len(target_words) > 1:
            word_matches = 0
            total_words = len(query_words)
            
            for q_word in query_words:
                best_match = 0
                for t_word in target_words:
                    word_sim = difflib.SequenceMatcher(None, q_word, t_word).ratio()
                    best_match = max(best_match, word_sim)
                word_matches += best_match
            
            if total_words > 0:
                scores.append(word_matches / total_words)
        
        # Character n-gram similarity
        bigram_score = FuzzyMatcher._ngram_similarity(query_lower, target_lower, 2)
        trigram_score = FuzzyMatcher._ngram_similarity(query_lower, target_lower, 3)
        scores.extend([bigram_score, trigram_score])
        
        # Subsequence matching (character order preservation)
        subseq_score = FuzzyMatcher._subsequence_similarity(query_lower, target_lower)
        scores.append(subseq_score)
        
        # Return weighted average of all scores
        if scores:
            return sum(scores) / len(scores)
        
        return 0.0
    
    @staticmethod
    def _ngram_similarity(s1: str, s2: str, n: int) -> float:
        """Calculate n-gram based similarity."""
        if len(s1) < n or len(s2) < n:
            return 0.0
        
        ngrams1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
        ngrams2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _subsequence_similarity(s1: str, s2: str) -> float:
        """Calculate similarity based on common subsequence."""
        def lcs_length(x, y):
            m, n = len(x), len(y)
            if m == 0 or n == 0:
                return 0
            
            # Use dynamic programming with space optimization
            prev = [0] * (n + 1)
            
            for i in range(1, m + 1):
                curr = [0] * (n + 1)
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        curr[j] = prev[j-1] + 1
                    else:
                        curr[j] = max(prev[j], curr[j-1])
                prev = curr
            
            return prev[n]
        
        lcs_len = lcs_length(s1, s2)
        max_len = max(len(s1), len(s2))
        
        return lcs_len / max_len if max_len > 0 else 0.0
    
    @staticmethod
    def generate_variants(word: str) -> List[str]:
        """Generate common misspelling variants for a word."""
        variants = []
        word_lower = word.lower()
        
        # Common typing errors
        # Adjacent key substitutions (QWERTY layout)
        keyboard_map = {
            'q': 'wa', 'w': 'qeas', 'e': 'wrds', 'r': 'etdf', 't': 'ryfg',
            'y': 'tugh', 'u': 'yihj', 'i': 'uojk', 'o': 'ipkl', 'p': 'ol',
            'a': 'qwsz', 's': 'awedrxz', 'd': 'serfcx', 'f': 'drtgvc',
            'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm',
            'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb',
            'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
        }
        
        # Single character substitutions
        for i, char in enumerate(word_lower):
            if char in keyboard_map:
                for replacement in keyboard_map[char]:
                    variant = word_lower[:i] + replacement + word_lower[i+1:]
                    variants.append(variant)
        
        # Single character deletions
        for i in range(len(word_lower)):
            variant = word_lower[:i] + word_lower[i+1:]
            if variant and variant != word_lower:
                variants.append(variant)
        
        # Single character insertions (vowels)
        vowels = 'aeiou'
        for i in range(len(word_lower) + 1):
            for vowel in vowels:
                variant = word_lower[:i] + vowel + word_lower[i:]
                if variant != word_lower:
                    variants.append(variant)
        
        # Character transpositions (adjacent swaps)
        for i in range(len(word_lower) - 1):
            variant = (word_lower[:i] + word_lower[i+1] + 
                      word_lower[i] + word_lower[i+2:])
            variants.append(variant)
        
        return variants


class ContextAnalyzer:
    """Analyzes user context to improve discovery relevance."""
    
    def __init__(self):
        self.project_patterns = {
            'python': ['setup.py', 'requirements.txt', 'pyproject.toml', '.py'],
            'javascript': ['package.json', 'node_modules', '.js', '.ts'],
            'docker': ['Dockerfile', 'docker-compose.yml', '.dockerignore'],
            'git': ['.git', '.gitignore', '.gitmodules'],
            'web': ['index.html', 'css', 'js', 'html'],
            'data': ['.csv', '.json', '.xml', '.sql', '.parquet']
        }
    
    def analyze_context(self, request: DiscoveryRequest) -> Dict[str, float]:
        """Analyze discovery request context and return relevance boosts.
        
        Args:
            request: Discovery request with context information
            
        Returns:
            Dictionary mapping context types to boost scores (0.0-1.0)
        """
        context_scores = defaultdict(float)
        
        # Project type context
        if request.current_project_type:
            context_scores[request.current_project_type] = 1.0
            
            # Add related contexts
            if request.current_project_type == 'python':
                context_scores['data'] = 0.6
                context_scores['testing'] = 0.7
            elif request.current_project_type == 'web':
                context_scores['javascript'] = 0.8
                context_scores['css'] = 0.6
        
        # Recent commands context
        if request.recent_commands:
            command_categories = self._categorize_commands(request.recent_commands)
            for category, weight in command_categories.items():
                context_scores[category] = min(1.0, context_scores[category] + weight * 0.5)
        
        # User preferences context
        if request.user_preferences:
            for pref_key, pref_value in request.user_preferences.items():
                if pref_key == 'preferred_tools':
                    for tool in pref_value if isinstance(pref_value, list) else [pref_value]:
                        context_scores[tool] = min(1.0, context_scores[tool] + 0.3)
        
        return dict(context_scores)
    
    def _categorize_commands(self, commands: List[str]) -> Dict[str, float]:
        """Categorize recent commands and calculate category weights."""
        categories = defaultdict(float)
        total_commands = len(commands)
        
        if total_commands == 0:
            return {}
        
        # Define command category patterns
        category_patterns = {
            'git': r'^git\.|clone|commit|push|pull|branch|merge',
            'docker': r'^docker\.|build|run|exec|compose',
            'python': r'^python\.|pip|pytest|conda|venv',
            'file': r'^file\.|ls|cat|cp|mv|rm|find|grep',
            'system': r'^system\.|ps|top|kill|df|mount',
            'network': r'^net\.|curl|wget|ping|ssh|scp',
            'database': r'^db\.|mysql|postgres|sqlite|mongo',
            'testing': r'^test\.|pytest|jest|mocha|unittest',
        }
        
        for command in commands:
            command_lower = command.lower()
            for category, pattern in category_patterns.items():
                if re.search(pattern, command_lower):
                    categories[category] += 1.0 / total_commands
        
        return dict(categories)


class RelevanceScorer:
    """Scores command relevance based on multiple factors."""
    
    def __init__(self):
        self.base_weights = {
            'exact_match': 1.0,
            'fuzzy_match': 0.4,
            'usage_frequency': 0.3,
            'skill_match': 0.2,
            'context_match': 0.5,
            'recency': 0.15,
            'category_match': 0.25
        }
    
    def score_command(
        self,
        command_meta: CommandMetadata,
        query: str,
        context: Dict[str, Any],
        skill_level: SkillLevel,
        execution_mode: ExecutionMode
    ) -> Tuple[float, List[str]]:
        """Calculate relevance score for a command.
        
        Args:
            command_meta: Command metadata
            query: Search query
            context: Context information for scoring
            skill_level: User skill level
            execution_mode: Target execution mode
            
        Returns:
            Tuple of (relevance_score, match_reasons)
        """
        score = 0.0
        reasons = []
        
        definition = command_meta.definition
        
        # Text matching scores
        name_similarity = FuzzyMatcher.calculate_similarity(query, definition.name)
        desc_similarity = FuzzyMatcher.calculate_similarity(query, definition.description)
        
        # Check alias matches
        alias_similarity = 0.0
        for alias in definition.aliases:
            alias_sim = FuzzyMatcher.calculate_similarity(query, alias)
            alias_similarity = max(alias_similarity, alias_sim)
        
        # Best text match score
        best_text_match = max(name_similarity, desc_similarity, alias_similarity)
        
        if best_text_match >= 0.9:
            score += self.base_weights['exact_match'] * best_text_match
            reasons.append("Exact match")
        elif best_text_match >= 0.4:
            score += self.base_weights['fuzzy_match'] * best_text_match
            reasons.append("Fuzzy match")
        
        # Usage frequency score
        if command_meta.usage_count > 0:
            # Normalize usage count (cap at 100 for scoring)
            normalized_usage = min(command_meta.usage_count, 100) / 100.0
            usage_score = self.base_weights['usage_frequency'] * normalized_usage
            score += usage_score
            reasons.append(f"Used {command_meta.usage_count} times")
        
        # Skill level matching
        if self._skill_level_matches(definition.min_skill_level, skill_level):
            score += self.base_weights['skill_match']
            reasons.append("Skill level match")
        else:
            # Penalty for skill mismatch
            score -= 0.1
        
        # Execution mode compatibility
        if execution_mode in definition.supported_modes:
            score += 0.1
            reasons.append("Mode compatible")
        else:
            score -= 0.2
        
        # Context matching
        context_boost = context.get('context_scores', {})
        for ctx_type, boost in context_boost.items():
            # Check if command is relevant to this context
            if self._command_matches_context(definition, ctx_type):
                context_score = self.base_weights['context_match'] * boost
                score += context_score
                reasons.append(f"Context: {ctx_type}")
        
        # Recency bonus
        if command_meta.last_used:
            days_since_use = (datetime.now(timezone.utc) - command_meta.last_used).days
            if days_since_use < 7:
                recency_boost = self.base_weights['recency'] * (1.0 - days_since_use / 7.0)
                score += recency_boost
                reasons.append("Recently used")
        
        # Category relevance
        category_terms = {
            'git': ['version', 'commit', 'branch', 'repository'],
            'docker': ['container', 'image', 'build', 'deploy'],
            'file': ['file', 'directory', 'path', 'search'],
            'system': ['system', 'process', 'monitor', 'resource'],
        }
        
        query_lower = query.lower()
        for category, terms in category_terms.items():
            if any(term in query_lower for term in terms):
                if any(term in definition.description.lower() for term in terms):
                    score += self.base_weights['category_match']
                    reasons.append(f"Category: {category}")
                    break
        
        # Deprecated command penalty
        if definition.deprecated:
            score *= 0.3
            reasons.append("Deprecated")
        
        # Experimental command handling
        if definition.category == CommandCategory.EXPERIMENTAL:
            if skill_level == SkillLevel.EXPERT:
                score += 0.05  # Small bonus for experts
            else:
                score -= 0.1   # Small penalty for non-experts
        
        return max(0.0, min(1.0, score)), reasons
    
    def _skill_level_matches(self, required: SkillLevel, user: SkillLevel) -> bool:
        """Check if user skill level meets command requirements."""
        skill_order = [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, SkillLevel.EXPERT]
        required_idx = skill_order.index(required)
        user_idx = skill_order.index(user)
        return user_idx >= required_idx
    
    def _command_matches_context(self, definition: CommandDefinition, context_type: str) -> bool:
        """Check if command is relevant to given context type."""
        # Check command name, description, and tags for context relevance
        text_to_check = f"{definition.name} {definition.description} {' '.join(definition.tags)}".lower()
        
        context_keywords = {
            'python': ['python', 'py', 'pip', 'conda', 'venv', 'django', 'flask'],
            'javascript': ['js', 'javascript', 'node', 'npm', 'yarn', 'react', 'vue'],
            'docker': ['docker', 'container', 'image', 'dockerfile', 'compose'],
            'git': ['git', 'version', 'commit', 'branch', 'clone', 'repository'],
            'web': ['web', 'http', 'html', 'css', 'url', 'server', 'browser'],
            'data': ['data', 'csv', 'json', 'xml', 'database', 'query', 'analysis'],
            'testing': ['test', 'spec', 'unittest', 'pytest', 'jest', 'assertion'],
            'system': ['system', 'process', 'memory', 'cpu', 'disk', 'monitor']
        }
        
        keywords = context_keywords.get(context_type, [])
        return any(keyword in text_to_check for keyword in keywords)


class DiscoveryEngine:
    """Advanced command discovery engine with intelligent search capabilities.
    
    Features:
    - Fuzzy matching with multiple algorithms
    - Skill-level-based progressive disclosure
    - Context-aware suggestions
    - Usage-based ranking
    - Real-time search optimization
    - Category and tag filtering
    """
    
    def __init__(self, registry: CommandRegistry):
        """Initialize discovery engine with command registry.
        
        Args:
            registry: CommandRegistry instance for command data
        """
        self.registry = registry
        self.search_index = SearchIndex()
        self.fuzzy_matcher = FuzzyMatcher()
        self.context_analyzer = ContextAnalyzer()
        self.relevance_scorer = RelevanceScorer()
        
        # Build initial search index
        self._build_search_index()
        
        # Performance tracking
        self._search_stats = {
            'total_searches': 0,
            'avg_response_time_ms': 0,
            'cache_hits': 0,
            'last_index_update': datetime.now(timezone.utc)
        }
        
        # Simple LRU cache for recent searches
        self._search_cache = {}
        self._max_cache_size = 100
        
        logger.info(f"DiscoveryEngine initialized with {len(registry)} commands")
    
    async def discover_commands(self, request: DiscoveryRequest) -> List[DiscoveryResult]:
        """Discover commands matching the request criteria.
        
        Args:
            request: Discovery request with search pattern and filters
            
        Returns:
            List of DiscoveryResult objects sorted by relevance
        """
        start_time = time.perf_counter()
        self._search_stats['total_searches'] += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self._search_cache:
            self._search_stats['cache_hits'] += 1
            return self._search_cache[cache_key]
        
        try:
            # Get base command set
            candidates = await self._get_candidate_commands(request)
            
            # Analyze context for relevance boosting
            context = self.context_analyzer.analyze_context(request)
            context_data = {
                'context_scores': context,
                'project_type': request.current_project_type,
                'recent_commands': request.recent_commands
            }
            
            # Score and rank candidates
            results = []
            for cmd_name in candidates:
                metadata = self.registry.get_command(cmd_name)
                if not metadata:
                    continue
                
                # Skip filtered commands
                if not self._passes_filters(metadata, request):
                    continue
                
                # Calculate relevance score
                relevance_score, match_reasons = self.relevance_scorer.score_command(
                    metadata, request.pattern, context_data, 
                    request.skill_level, request.mode
                )
                
                if relevance_score > 0.05:  # Minimum relevance threshold
                    result = DiscoveryResult(
                        metadata=metadata,
                        relevance_score=relevance_score,
                        match_reasons=match_reasons,
                        usage_rank=metadata.usage_count,
                        fuzzy_match=self._is_fuzzy_match(request.pattern, metadata)
                    )
                    results.append(result)
            
            # Sort by relevance score (descending), then usage count (descending)
            results.sort(key=lambda x: (-x.relevance_score, -x.usage_rank))
            
            # Limit results
            results = results[:request.max_results]
            
            # Cache results
            self._cache_results(cache_key, results)
            
            # Update performance stats
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._update_search_stats(duration_ms)
            
            logger.debug(
                f"Discovery found {len(results)} results for '{request.pattern}' "
                f"in {duration_ms:.2f}ms"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Discovery failed for pattern '{request.pattern}': {e}")
            return []
    
    async def suggest_commands(
        self,
        context: Dict[str, Any],
        skill_level: SkillLevel = SkillLevel.INTERMEDIATE,
        limit: int = 5
    ) -> List[DiscoveryResult]:
        """Generate contextual command suggestions.
        
        Args:
            context: Current user context
            skill_level: User skill level for filtering
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested commands
        """
        suggestions = []
        
        # Get commands based on context
        context_scores = context.get('context_scores', {})
        
        for context_type, boost in context_scores.items():
            if boost < 0.3:  # Skip low-relevance contexts
                continue
            
            # Find commands relevant to this context
            relevant_commands = self._get_context_commands(context_type, skill_level)
            
            for cmd_name in relevant_commands[:3]:  # Top 3 per context
                metadata = self.registry.get_command(cmd_name)
                if metadata and not any(s.metadata.definition.name == cmd_name for s in suggestions):
                    
                    result = DiscoveryResult(
                        metadata=metadata,
                        relevance_score=boost * 0.8,  # Context-based score
                        match_reasons=[f"Contextually relevant for {context_type}"],
                        usage_rank=metadata.usage_count,
                        fuzzy_match=False
                    )
                    suggestions.append(result)
        
        # Add popular commands if we need more suggestions
        if len(suggestions) < limit:
            popular_commands = self.registry.list_commands(
                skill_level=skill_level,
                include_deprecated=False
            )
            
            for metadata in popular_commands[:limit - len(suggestions)]:
                if not any(s.metadata.definition.name == metadata.definition.name for s in suggestions):
                    result = DiscoveryResult(
                        metadata=metadata,
                        relevance_score=0.3,
                        match_reasons=["Popular command"],
                        usage_rank=metadata.usage_count,
                        fuzzy_match=False
                    )
                    suggestions.append(result)
        
        # Sort by relevance and limit
        suggestions.sort(key=lambda x: -x.relevance_score)
        return suggestions[:limit]
    
    def update_search_index(self) -> None:
        """Update the search index with current registry state."""
        logger.info("Updating discovery search index...")
        self._build_search_index()
        self._search_cache.clear()  # Clear cache after index update
        self._search_stats['last_index_update'] = datetime.now(timezone.utc)
        logger.info("Search index updated successfully")
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery engine performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            **self._search_stats,
            'cache_size': len(self._search_cache),
            'index_size': {
                'exact_words': len(self.search_index.exact_words),
                'fuzzy_variants': len(self.search_index.fuzzy_variants),
                'bigrams': len(self.search_index.bigrams),
                'trigrams': len(self.search_index.trigrams)
            }
        }
    
    # Internal methods
    
    async def _get_candidate_commands(self, request: DiscoveryRequest) -> Set[str]:
        """Get initial set of candidate commands for filtering."""
        candidates = set()
        pattern_lower = request.pattern.lower().strip()
        
        if not pattern_lower:
            # Return all commands if no pattern
            return set(self.registry._commands.keys())
        
        # Exact word matches
        if pattern_lower in self.search_index.exact_words:
            candidates.update(self.search_index.exact_words[pattern_lower])
        
        # Prefix matches
        for word, cmd_set in self.search_index.prefix_words.items():
            if word.startswith(pattern_lower):
                candidates.update(cmd_set)
        
        # Fuzzy matching if enabled
        if request.fuzzy_matching:
            # Check fuzzy variants
            if pattern_lower in self.search_index.fuzzy_variants:
                candidates.update(self.search_index.fuzzy_variants[pattern_lower])
            
            # N-gram based fuzzy matching
            pattern_bigrams = set(pattern_lower[i:i+2] for i in range(len(pattern_lower)-1))
            pattern_trigrams = set(pattern_lower[i:i+3] for i in range(len(pattern_lower)-2))
            
            # Find commands with similar n-grams
            fuzzy_candidates = set()
            
            for bigram in pattern_bigrams:
                if bigram in self.search_index.bigrams:
                    fuzzy_candidates.update(self.search_index.bigrams[bigram])
            
            for trigram in pattern_trigrams:
                if trigram in self.search_index.trigrams:
                    fuzzy_candidates.update(self.search_index.trigrams[trigram])
            
            # Only include fuzzy matches that meet similarity threshold
            for cmd_name in fuzzy_candidates:
                metadata = self.registry.get_command(cmd_name)
                if metadata:
                    similarity = FuzzyMatcher.calculate_similarity(
                        pattern_lower, metadata.definition.name.lower()
                    )
                    if similarity >= 0.4:  # Fuzzy similarity threshold
                        candidates.add(cmd_name)
        
        # If no direct matches, do broad similarity search
        if not candidates and request.fuzzy_matching:
            all_commands = list(self.registry._commands.keys())
            for cmd_name in all_commands:
                similarity = FuzzyMatcher.calculate_similarity(pattern_lower, cmd_name.lower())
                if similarity >= 0.3:  # Broader threshold for empty results
                    candidates.add(cmd_name)
        
        return candidates
    
    def _passes_filters(self, metadata: CommandMetadata, request: DiscoveryRequest) -> bool:
        """Check if command passes all discovery filters."""
        definition = metadata.definition
        
        # Category filter
        if request.categories and definition.category not in request.categories:
            return False
        
        # Skill level filter
        skill_order = [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, SkillLevel.EXPERT]
        required_idx = skill_order.index(definition.min_skill_level)
        user_idx = skill_order.index(request.skill_level)
        if user_idx < required_idx:
            return False
        
        # Execution mode filter
        if request.mode not in definition.supported_modes:
            return False
        
        # Deprecated filter
        if definition.deprecated and not request.include_deprecated:
            return False
        
        # Experimental filter
        if (definition.category == CommandCategory.EXPERIMENTAL and 
            not request.include_experimental):
            return False
        
        return True
    
    def _is_fuzzy_match(self, query: str, metadata: CommandMetadata) -> bool:
        """Determine if this is a fuzzy match result."""
        query_lower = query.lower()
        name_lower = metadata.definition.name.lower()
        
        # Not fuzzy if exact match or starts with query
        if query_lower == name_lower or name_lower.startswith(query_lower):
            return False
        
        # Check if it's in description (also not fuzzy)
        if query_lower in metadata.definition.description.lower():
            return False
        
        return True
    
    def _get_context_commands(self, context_type: str, skill_level: SkillLevel) -> List[str]:
        """Get commands relevant to a specific context type."""
        if context_type not in self.search_index.contexts:
            return []
        
        context_commands = list(self.search_index.contexts[context_type])
        
        # Filter by skill level and sort by usage
        filtered_commands = []
        for cmd_name in context_commands:
            metadata = self.registry.get_command(cmd_name)
            if metadata and self._passes_skill_level(metadata, skill_level):
                filtered_commands.append((cmd_name, metadata.usage_count))
        
        # Sort by usage count (descending)
        filtered_commands.sort(key=lambda x: -x[1])
        return [cmd_name for cmd_name, _ in filtered_commands]
    
    def _passes_skill_level(self, metadata: CommandMetadata, user_skill: SkillLevel) -> bool:
        """Check if command passes skill level requirements."""
        skill_order = [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, SkillLevel.EXPERT]
        required_idx = skill_order.index(metadata.definition.min_skill_level)
        user_idx = skill_order.index(user_skill)
        return user_idx >= required_idx
    
    def _build_search_index(self) -> None:
        """Build optimized search index from registry commands."""
        self.search_index = SearchIndex()
        
        for cmd_name, metadata in self.registry._commands.items():
            definition = metadata.definition
            
            # Index command name
            self._index_text(cmd_name, cmd_name, self.search_index)
            
            # Index aliases
            for alias in definition.aliases:
                self._index_text(alias, cmd_name, self.search_index)
            
            # Index description words
            desc_words = definition.description.lower().split()
            for word in desc_words:
                clean_word = word.strip('.,!?()[]{}":;')
                if len(clean_word) > 1:
                    self._index_text(clean_word, cmd_name, self.search_index)
            
            # Index tags
            for tag in definition.tags:
                self._index_text(tag, cmd_name, self.search_index)
            
            # Index category
            self.search_index.categories[definition.category.value].add(cmd_name)
            
            # Build context associations
            self._build_context_associations(metadata, cmd_name)
    
    def _index_text(self, text: str, command_name: str, index: SearchIndex) -> None:
        """Index a text string for search optimization."""
        text_lower = text.lower().strip()
        if not text_lower:
            return
        
        # Exact word index
        index.exact_words[text_lower].add(command_name)
        
        # Prefix index
        for i in range(1, len(text_lower) + 1):
            prefix = text_lower[:i]
            index.prefix_words[prefix].add(command_name)
        
        # Generate and index fuzzy variants
        variants = self.fuzzy_matcher.generate_variants(text_lower)
        for variant in variants:
            index.fuzzy_variants[variant].add(command_name)
        
        # N-gram index
        if len(text_lower) >= 2:
            for i in range(len(text_lower) - 1):
                bigram = text_lower[i:i+2]
                index.bigrams[bigram].add(command_name)
        
        if len(text_lower) >= 3:
            for i in range(len(text_lower) - 2):
                trigram = text_lower[i:i+3]
                index.trigrams[trigram].add(command_name)
    
    def _build_context_associations(self, metadata: CommandMetadata, cmd_name: str) -> None:
        """Build context associations for a command."""
        definition = metadata.definition
        text_content = f"{definition.name} {definition.description} {' '.join(definition.tags)}".lower()
        
        # Context keyword mapping
        context_patterns = {
            'python': ['python', 'py', 'pip', 'conda', 'venv', 'django', 'flask', 'pytest'],
            'javascript': ['javascript', 'js', 'node', 'npm', 'yarn', 'react', 'vue', 'angular'],
            'docker': ['docker', 'container', 'image', 'dockerfile', 'compose', 'kubernetes'],
            'git': ['git', 'version', 'commit', 'branch', 'clone', 'repository', 'merge'],
            'web': ['web', 'http', 'html', 'css', 'url', 'server', 'browser', 'api'],
            'data': ['data', 'csv', 'json', 'xml', 'database', 'query', 'analysis', 'pandas'],
            'testing': ['test', 'spec', 'unittest', 'pytest', 'jest', 'mocha', 'assertion'],
            'system': ['system', 'process', 'memory', 'cpu', 'disk', 'monitor', 'performance'],
            'file': ['file', 'directory', 'path', 'folder', 'search', 'find', 'grep'],
            'network': ['network', 'http', 'tcp', 'udp', 'ssh', 'ftp', 'curl', 'wget']
        }
        
        for context_type, keywords in context_patterns.items():
            if any(keyword in text_content for keyword in keywords):
                self.search_index.contexts[context_type].add(cmd_name)
    
    def _generate_cache_key(self, request: DiscoveryRequest) -> str:
        """Generate cache key for discovery request."""
        # Create a string representation of key request attributes
        key_parts = [
            request.pattern.lower().strip(),
            str(request.skill_level.value),
            str(request.mode.value),
            str(sorted([cat.value for cat in request.categories])),
            str(request.include_deprecated),
            str(request.include_experimental),
            str(request.fuzzy_matching),
            str(request.max_results)
        ]
        
        return "|".join(key_parts)
    
    def _cache_results(self, cache_key: str, results: List[DiscoveryResult]) -> None:
        """Cache discovery results with LRU eviction."""
        # Simple LRU: remove oldest if at capacity
        if len(self._search_cache) >= self._max_cache_size:
            # Remove oldest entry (first in dict)
            oldest_key = next(iter(self._search_cache))
            del self._search_cache[oldest_key]
        
        self._search_cache[cache_key] = results
    
    def _update_search_stats(self, duration_ms: float) -> None:
        """Update search performance statistics."""
        current_avg = self._search_stats['avg_response_time_ms']
        total_searches = self._search_stats['total_searches']
        
        # Calculate running average
        if total_searches == 1:
            self._search_stats['avg_response_time_ms'] = duration_ms
        else:
            self._search_stats['avg_response_time_ms'] = (
                (current_avg * (total_searches - 1) + duration_ms) / total_searches
            )