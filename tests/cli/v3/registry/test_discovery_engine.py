"""Test suite for CLI v3 DiscoveryEngine."""

import asyncio
import pytest
import time
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from agentsmcp.cli.v3.models.registry_models import (
    CommandDefinition,
    CommandMetadata,
    CommandParameter,
    ParameterType,
    DiscoveryRequest,
    SkillLevel,
    CommandCategory,
)
from agentsmcp.cli.v3.registry.discovery_engine import (
    DiscoveryEngine,
    FuzzyMatcher,
    ContextAnalyzer,
    RelevanceScorer,
    SearchIndex,
)


class TestFuzzyMatcher:
    """Test FuzzyMatcher functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.matcher = FuzzyMatcher()
    
    def test_levenshtein_distance_exact_match(self):
        """Test exact matches return distance 0."""
        assert self.matcher._levenshtein_distance("hello", "hello") == 0
        assert self.matcher._levenshtein_distance("command", "command") == 0
    
    def test_levenshtein_distance_single_character(self):
        """Test single character differences."""
        assert self.matcher._levenshtein_distance("hello", "hallo") == 1
        assert self.matcher._levenshtein_distance("cat", "bat") == 1
        assert self.matcher._levenshtein_distance("test", "best") == 1
    
    def test_levenshtein_distance_insertions_deletions(self):
        """Test insertions and deletions."""
        assert self.matcher._levenshtein_distance("hello", "hell") == 1
        assert self.matcher._levenshtein_distance("hello", "helloo") == 1
        assert self.matcher._levenshtein_distance("abc", "abcd") == 1
    
    def test_bigram_similarity_exact(self):
        """Test bigram similarity for exact matches."""
        similarity = self.matcher._bigram_similarity("hello", "hello")
        assert similarity == 1.0
    
    def test_bigram_similarity_partial(self):
        """Test bigram similarity for partial matches."""
        similarity = self.matcher._bigram_similarity("hello", "hallo")
        assert 0.5 < similarity < 1.0
        
        similarity = self.matcher._bigram_similarity("test", "best")
        assert 0.3 < similarity < 0.7
    
    def test_bigram_similarity_no_match(self):
        """Test bigram similarity for no matches."""
        similarity = self.matcher._bigram_similarity("abc", "xyz")
        assert similarity == 0.0
    
    def test_subsequence_match_exact(self):
        """Test subsequence matching for exact matches."""
        ratio = self.matcher._subsequence_match("hello", "hello")
        assert ratio == 1.0
    
    def test_subsequence_match_partial(self):
        """Test subsequence matching for partial matches."""
        ratio = self.matcher._subsequence_match("hello", "hlo")
        assert ratio > 0.5
        
        ratio = self.matcher._subsequence_match("command", "cmd")
        assert ratio > 0.3
    
    def test_subsequence_match_no_match(self):
        """Test subsequence matching for no matches."""
        ratio = self.matcher._subsequence_match("hello", "xyz")
        assert ratio == 0.0
    
    def test_calculate_similarity_combined(self):
        """Test combined similarity calculation."""
        # Test exact match
        similarity = self.matcher.calculate_similarity("hello", "hello")
        assert similarity == 1.0
        
        # Test close match
        similarity = self.matcher.calculate_similarity("hello", "hallo")
        assert 0.7 < similarity < 1.0
        
        # Test distant match
        similarity = self.matcher.calculate_similarity("hello", "world")
        assert 0.0 <= similarity < 0.3
    
    def test_calculate_similarity_empty_strings(self):
        """Test similarity calculation with empty strings."""
        assert self.matcher.calculate_similarity("", "") == 1.0
        assert self.matcher.calculate_similarity("hello", "") == 0.0
        assert self.matcher.calculate_similarity("", "hello") == 0.0
    
    def test_calculate_similarity_case_insensitive(self):
        """Test case insensitive similarity calculation."""
        similarity = self.matcher.calculate_similarity("Hello", "HELLO")
        assert similarity == 1.0
        
        similarity = self.matcher.calculate_similarity("Command", "command")
        assert similarity == 1.0


class TestContextAnalyzer:
    """Test ContextAnalyzer functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = ContextAnalyzer()
    
    def test_analyze_project_context_git_repo(self):
        """Test project context analysis for git repositories."""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.side_effect = lambda: str(self).endswith('.git')
            
            context = self.analyzer.analyze_project_context(Path("/fake/project"))
            
            assert context["project_type"] == "git"
            assert context["has_git"] is True
    
    def test_analyze_project_context_python_project(self):
        """Test project context analysis for Python projects."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.glob') as mock_glob:
            
            def mock_exists_impl(self):
                path_str = str(self)
                return (path_str.endswith('requirements.txt') or 
                       path_str.endswith('setup.py') or
                       path_str.endswith('pyproject.toml'))
            
            mock_exists.side_effect = mock_exists_impl
            mock_glob.return_value = [Path("test.py")]
            
            context = self.analyzer.analyze_project_context(Path("/fake/project"))
            
            assert "python" in context["languages"]
            assert context["has_requirements"] is True
    
    def test_analyze_project_context_node_project(self):
        """Test project context analysis for Node.js projects."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.glob') as mock_glob:
            
            def mock_exists_impl(self):
                return str(self).endswith('package.json')
            
            mock_exists.side_effect = mock_exists_impl
            mock_glob.return_value = [Path("script.js")]
            
            context = self.analyzer.analyze_project_context(Path("/fake/project"))
            
            assert "javascript" in context["languages"]
            assert context["has_package_json"] is True
    
    def test_get_contextual_suggestions_git_commands(self):
        """Test contextual suggestions for git-related commands."""
        commands = [
            CommandDefinition(
                name="git-status",
                category=CommandCategory.GIT,
                description="Show git status",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="test",
                    skill_level=SkillLevel.BEGINNER,
                    tags=["git", "status"]
                ),
                parameters=[]
            ),
            CommandDefinition(
                name="file-list",
                category=CommandCategory.FILE,
                description="List files",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="test",
                    skill_level=SkillLevel.BEGINNER,
                    tags=["files"]
                ),
                parameters=[]
            )
        ]
        
        context = {"project_type": "git", "has_git": True}
        
        suggestions = self.analyzer.get_contextual_suggestions(commands, context)
        
        # Git commands should be ranked higher
        git_commands = [cmd for cmd in suggestions if cmd.category == CommandCategory.GIT]
        assert len(git_commands) > 0
    
    def test_get_contextual_suggestions_python_commands(self):
        """Test contextual suggestions for Python-related commands."""
        commands = [
            CommandDefinition(
                name="python-run",
                category=CommandCategory.DEVELOPMENT,
                description="Run Python script",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="test",
                    skill_level=SkillLevel.BEGINNER,
                    tags=["python", "run"]
                ),
                parameters=[]
            ),
            CommandDefinition(
                name="generic-command",
                category=CommandCategory.GENERAL,
                description="Generic command",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="test",
                    skill_level=SkillLevel.BEGINNER,
                    tags=["general"]
                ),
                parameters=[]
            )
        ]
        
        context = {"languages": ["python"], "has_requirements": True}
        
        suggestions = self.analyzer.get_contextual_suggestions(commands, context)
        
        # Python commands should be present and prioritized
        python_commands = [cmd for cmd in suggestions 
                         if any(tag == "python" for tag in cmd.metadata.tags)]
        assert len(python_commands) > 0


class TestRelevanceScorer:
    """Test RelevanceScorer functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.scorer = RelevanceScorer()
    
    def test_calculate_relevance_score_exact_match(self):
        """Test relevance scoring for exact name matches."""
        command = CommandDefinition(
            name="test-command",
            category=CommandCategory.GENERAL,
            description="Test command",
            metadata=CommandMetadata(
                version="1.0.0",
                author="test",
                skill_level=SkillLevel.BEGINNER,
                tags=["test"],
                usage_count=100,
                last_used="2023-01-01T00:00:00Z"
            ),
            parameters=[]
        )
        
        score = self.scorer.calculate_relevance_score(
            command, "test-command", [], {}
        )
        
        # Exact match should have high score
        assert score >= 0.8
    
    def test_calculate_relevance_score_fuzzy_match(self):
        """Test relevance scoring for fuzzy matches."""
        command = CommandDefinition(
            name="test-command",
            category=CommandCategory.GENERAL,
            description="Test command for testing",
            metadata=CommandMetadata(
                version="1.0.0",
                author="test",
                skill_level=SkillLevel.BEGINNER,
                tags=["test"],
                usage_count=50,
                last_used="2023-01-01T00:00:00Z"
            ),
            parameters=[]
        )
        
        score = self.scorer.calculate_relevance_score(
            command, "test-cmd", [], {}
        )
        
        # Fuzzy match should have moderate score
        assert 0.3 <= score < 0.8
    
    def test_calculate_relevance_score_usage_boost(self):
        """Test relevance scoring with usage count boost."""
        high_usage_command = CommandDefinition(
            name="popular-command",
            category=CommandCategory.GENERAL,
            description="Popular command",
            metadata=CommandMetadata(
                version="1.0.0",
                author="test",
                skill_level=SkillLevel.BEGINNER,
                tags=["popular"],
                usage_count=1000,
                last_used="2023-01-01T00:00:00Z"
            ),
            parameters=[]
        )
        
        low_usage_command = CommandDefinition(
            name="unpopular-command",
            category=CommandCategory.GENERAL,
            description="Unpopular command",
            metadata=CommandMetadata(
                version="1.0.0",
                author="test",
                skill_level=SkillLevel.BEGINNER,
                tags=["unpopular"],
                usage_count=1,
                last_used="2023-01-01T00:00:00Z"
            ),
            parameters=[]
        )
        
        high_score = self.scorer.calculate_relevance_score(
            high_usage_command, "command", [], {}
        )
        low_score = self.scorer.calculate_relevance_score(
            low_usage_command, "command", [], {}
        )
        
        # High usage should result in higher score
        assert high_score > low_score
    
    def test_calculate_relevance_score_recency_boost(self):
        """Test relevance scoring with recency boost."""
        recent_command = CommandDefinition(
            name="recent-command",
            category=CommandCategory.GENERAL,
            description="Recently used command",
            metadata=CommandMetadata(
                version="1.0.0",
                author="test",
                skill_level=SkillLevel.BEGINNER,
                tags=["recent"],
                usage_count=10,
                last_used="2023-12-31T00:00:00Z"
            ),
            parameters=[]
        )
        
        old_command = CommandDefinition(
            name="old-command",
            category=CommandCategory.GENERAL,
            description="Old command",
            metadata=CommandMetadata(
                version="1.0.0",
                author="test",
                skill_level=SkillLevel.BEGINNER,
                tags=["old"],
                usage_count=10,
                last_used="2020-01-01T00:00:00Z"
            ),
            parameters=[]
        )
        
        recent_score = self.scorer.calculate_relevance_score(
            recent_command, "command", [], {}
        )
        old_score = self.scorer.calculate_relevance_score(
            old_command, "command", [], {}
        )
        
        # Recent usage should result in higher score
        assert recent_score > old_score


class TestSearchIndex:
    """Test SearchIndex functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.commands = [
            CommandDefinition(
                name="git-status",
                category=CommandCategory.GIT,
                description="Show git repository status",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="test",
                    skill_level=SkillLevel.BEGINNER,
                    tags=["git", "status", "repository"]
                ),
                parameters=[]
            ),
            CommandDefinition(
                name="file-list",
                category=CommandCategory.FILE,
                description="List files in directory",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="test",
                    skill_level=SkillLevel.BEGINNER,
                    tags=["files", "directory", "list"]
                ),
                parameters=[]
            ),
            CommandDefinition(
                name="python-run",
                category=CommandCategory.DEVELOPMENT,
                description="Run Python script",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="test",
                    skill_level=SkillLevel.INTERMEDIATE,
                    tags=["python", "script", "run"]
                ),
                parameters=[]
            )
        ]
        self.index = SearchIndex()
        self.index.build(self.commands)
    
    def test_build_index(self):
        """Test search index building."""
        # Index should contain bigrams and trigrams
        assert len(self.index.bigrams) > 0
        assert len(self.index.trigrams) > 0
        assert len(self.index.word_index) > 0
    
    def test_search_exact_name_match(self):
        """Test search for exact command name."""
        results = self.index.search("git-status", limit=5)
        
        assert len(results) > 0
        assert results[0].name == "git-status"
    
    def test_search_partial_name_match(self):
        """Test search for partial command name."""
        results = self.index.search("git", limit=5)
        
        assert len(results) > 0
        git_commands = [cmd for cmd in results if "git" in cmd.name.lower()]
        assert len(git_commands) > 0
    
    def test_search_description_match(self):
        """Test search in command descriptions."""
        results = self.index.search("repository", limit=5)
        
        assert len(results) > 0
        # Should find git-status command which has "repository" in description
        repo_commands = [cmd for cmd in results 
                        if "repository" in cmd.description.lower()]
        assert len(repo_commands) > 0
    
    def test_search_tag_match(self):
        """Test search in command tags."""
        results = self.index.search("python", limit=5)
        
        assert len(results) > 0
        # Should find python-run command
        python_commands = [cmd for cmd in results
                          if "python" in [tag.lower() for tag in cmd.metadata.tags]]
        assert len(python_commands) > 0
    
    def test_search_fuzzy_match(self):
        """Test fuzzy search capabilities."""
        results = self.index.search("fil", limit=5)  # Should match "file-list"
        
        assert len(results) > 0
        # Should find commands with similar names/descriptions
        file_related = [cmd for cmd in results 
                       if "file" in cmd.name.lower() or "file" in cmd.description.lower()]
        assert len(file_related) > 0
    
    def test_search_limit(self):
        """Test search result limiting."""
        results = self.index.search("", limit=2)  # Empty query should return all
        
        assert len(results) <= 2
    
    def test_search_empty_query(self):
        """Test search with empty query."""
        results = self.index.search("", limit=10)
        
        # Should return all commands when query is empty
        assert len(results) == len(self.commands)


class TestDiscoveryEngine:
    """Test DiscoveryEngine main functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.commands = [
            CommandDefinition(
                name="git-status",
                category=CommandCategory.GIT,
                description="Show git repository status",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="test",
                    skill_level=SkillLevel.BEGINNER,
                    tags=["git", "status"],
                    usage_count=100,
                    last_used="2023-01-01T00:00:00Z"
                ),
                parameters=[]
            ),
            CommandDefinition(
                name="file-list",
                category=CommandCategory.FILE,
                description="List files in directory",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="test",
                    skill_level=SkillLevel.BEGINNER,
                    tags=["files", "directory"],
                    usage_count=50,
                    last_used="2023-01-01T00:00:00Z"
                ),
                parameters=[]
            ),
            CommandDefinition(
                name="advanced-tool",
                category=CommandCategory.ADVANCED,
                description="Advanced development tool",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="test",
                    skill_level=SkillLevel.EXPERT,
                    tags=["advanced", "development"],
                    usage_count=10,
                    last_used="2023-01-01T00:00:00Z"
                ),
                parameters=[]
            )
        ]
        self.engine = DiscoveryEngine()
        self.engine.initialize(self.commands)
    
    @pytest.mark.asyncio
    async def test_discover_commands_basic(self):
        """Test basic command discovery."""
        request = DiscoveryRequest(
            query="git",
            limit=5,
            skill_level=SkillLevel.BEGINNER,
            categories=None,
            include_deprecated=False
        )
        
        results = await self.engine.discover_commands(request)
        
        assert len(results) > 0
        # Should include git-status command
        git_commands = [cmd for cmd in results if "git" in cmd.name.lower()]
        assert len(git_commands) > 0
    
    @pytest.mark.asyncio
    async def test_discover_commands_skill_level_filtering(self):
        """Test discovery with skill level filtering."""
        request = DiscoveryRequest(
            query="",
            limit=10,
            skill_level=SkillLevel.BEGINNER,
            categories=None,
            include_deprecated=False
        )
        
        results = await self.engine.discover_commands(request)
        
        # Should not include expert-level commands for beginner users
        expert_commands = [cmd for cmd in results 
                          if cmd.metadata.skill_level == SkillLevel.EXPERT]
        assert len(expert_commands) == 0
    
    @pytest.mark.asyncio
    async def test_discover_commands_category_filtering(self):
        """Test discovery with category filtering."""
        request = DiscoveryRequest(
            query="",
            limit=10,
            skill_level=SkillLevel.EXPERT,
            categories=[CommandCategory.GIT],
            include_deprecated=False
        )
        
        results = await self.engine.discover_commands(request)
        
        # Should only include git commands
        for cmd in results:
            assert cmd.category == CommandCategory.GIT
    
    @pytest.mark.asyncio
    async def test_discover_commands_limit(self):
        """Test discovery result limiting."""
        request = DiscoveryRequest(
            query="",
            limit=2,
            skill_level=SkillLevel.EXPERT,
            categories=None,
            include_deprecated=False
        )
        
        results = await self.engine.discover_commands(request)
        
        assert len(results) <= 2
    
    @pytest.mark.asyncio
    async def test_discover_commands_performance(self):
        """Test discovery performance requirement (<10ms)."""
        # Create a larger dataset to test performance
        large_command_set = []
        for i in range(100):
            large_command_set.append(CommandDefinition(
                name=f"command-{i}",
                category=CommandCategory.GENERAL,
                description=f"Test command {i}",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="test",
                    skill_level=SkillLevel.BEGINNER,
                    tags=[f"tag{i}"],
                    usage_count=i,
                    last_used="2023-01-01T00:00:00Z"
                ),
                parameters=[]
            ))
        
        engine = DiscoveryEngine()
        engine.initialize(large_command_set)
        
        request = DiscoveryRequest(
            query="command",
            limit=10,
            skill_level=SkillLevel.BEGINNER,
            categories=None,
            include_deprecated=False
        )
        
        # Measure discovery time
        start_time = time.time()
        results = await engine.discover_commands(request)
        end_time = time.time()
        
        discovery_time_ms = (end_time - start_time) * 1000
        
        # Should complete within 10ms requirement
        assert discovery_time_ms < 10.0, f"Discovery took {discovery_time_ms}ms, expected <10ms"
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_discover_commands_with_context(self):
        """Test discovery with project context."""
        with patch.object(self.engine.context_analyzer, 'analyze_project_context') as mock_analyze:
            mock_analyze.return_value = {"project_type": "git", "has_git": True}
            
            request = DiscoveryRequest(
                query="status",
                limit=5,
                skill_level=SkillLevel.BEGINNER,
                categories=None,
                include_deprecated=False,
                context_path="/fake/project"
            )
            
            results = await self.engine.discover_commands(request)
            
            assert len(results) > 0
            # Git commands should be prioritized in git context
            git_commands = [cmd for cmd in results if cmd.category == CommandCategory.GIT]
            assert len(git_commands) > 0
    
    @pytest.mark.asyncio
    async def test_discover_commands_empty_query(self):
        """Test discovery with empty query."""
        request = DiscoveryRequest(
            query="",
            limit=5,
            skill_level=SkillLevel.EXPERT,
            categories=None,
            include_deprecated=False
        )
        
        results = await self.engine.discover_commands(request)
        
        # Should return commands based on relevance/popularity
        assert len(results) > 0
        assert len(results) <= 5
    
    def test_get_command_suggestions_empty_list(self):
        """Test getting suggestions with empty command list."""
        empty_engine = DiscoveryEngine()
        empty_engine.initialize([])
        
        suggestions = empty_engine.get_command_suggestions("test", limit=5)
        assert len(suggestions) == 0
    
    def test_get_command_suggestions_basic(self):
        """Test basic command suggestions."""
        suggestions = self.engine.get_command_suggestions("git", limit=3)
        
        assert len(suggestions) > 0
        # Should include git-related commands
        git_suggestions = [cmd for cmd in suggestions if "git" in cmd.name.lower()]
        assert len(git_suggestions) > 0
    
    def test_initialize_empty_commands(self):
        """Test initialization with empty command list."""
        empty_engine = DiscoveryEngine()
        empty_engine.initialize([])
        
        # Should not raise an error
        assert empty_engine.search_index is not None
    
    def test_initialize_duplicate_commands(self):
        """Test initialization with duplicate commands."""
        duplicate_commands = self.commands + self.commands  # Duplicate the list
        
        engine = DiscoveryEngine()
        engine.initialize(duplicate_commands)
        
        # Should handle duplicates gracefully
        assert engine.search_index is not None


@pytest.mark.integration
class TestDiscoveryEngineIntegration:
    """Integration tests for DiscoveryEngine with real-world scenarios."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        # Create a realistic command set
        self.realistic_commands = [
            # Git commands
            CommandDefinition(
                name="git-status",
                category=CommandCategory.GIT,
                description="Show the working tree status",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="git-team",
                    skill_level=SkillLevel.BEGINNER,
                    tags=["git", "status", "working-tree"],
                    usage_count=500,
                    last_used="2023-12-01T00:00:00Z"
                ),
                parameters=[]
            ),
            CommandDefinition(
                name="git-commit",
                category=CommandCategory.GIT,
                description="Record changes to the repository",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="git-team",
                    skill_level=SkillLevel.INTERMEDIATE,
                    tags=["git", "commit", "changes"],
                    usage_count=300,
                    last_used="2023-11-30T00:00:00Z"
                ),
                parameters=[
                    CommandParameter(
                        name="message",
                        param_type=ParameterType.STRING,
                        description="Commit message",
                        required=True
                    )
                ]
            ),
            # File operations
            CommandDefinition(
                name="file-copy",
                category=CommandCategory.FILE,
                description="Copy files and directories",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="file-team",
                    skill_level=SkillLevel.BEGINNER,
                    tags=["file", "copy", "directory"],
                    usage_count=200,
                    last_used="2023-11-29T00:00:00Z"
                ),
                parameters=[
                    CommandParameter(
                        name="source",
                        param_type=ParameterType.PATH,
                        description="Source path",
                        required=True
                    ),
                    CommandParameter(
                        name="destination", 
                        param_type=ParameterType.PATH,
                        description="Destination path",
                        required=True
                    )
                ]
            ),
            # Development tools
            CommandDefinition(
                name="python-test",
                category=CommandCategory.DEVELOPMENT,
                description="Run Python unit tests",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="dev-team",
                    skill_level=SkillLevel.INTERMEDIATE,
                    tags=["python", "test", "unittest"],
                    usage_count=150,
                    last_used="2023-11-28T00:00:00Z"
                ),
                parameters=[
                    CommandParameter(
                        name="test_file",
                        param_type=ParameterType.PATH,
                        description="Test file path",
                        required=False
                    )
                ]
            ),
            # Advanced tools
            CommandDefinition(
                name="docker-compose",
                category=CommandCategory.ADVANCED,
                description="Define and run multi-container Docker applications",
                metadata=CommandMetadata(
                    version="1.0.0",
                    author="docker-team",
                    skill_level=SkillLevel.EXPERT,
                    tags=["docker", "compose", "container"],
                    usage_count=75,
                    last_used="2023-11-27T00:00:00Z"
                ),
                parameters=[
                    CommandParameter(
                        name="action",
                        param_type=ParameterType.STRING,
                        description="Action to perform",
                        required=True
                    )
                ]
            )
        ]
        
        self.engine = DiscoveryEngine()
        self.engine.initialize(self.realistic_commands)
    
    @pytest.mark.asyncio
    async def test_realistic_search_scenarios(self):
        """Test realistic search scenarios."""
        
        # Scenario 1: User looking for git commands
        request = DiscoveryRequest(
            query="git",
            limit=5,
            skill_level=SkillLevel.INTERMEDIATE,
            categories=None,
            include_deprecated=False
        )
        
        results = await self.engine.discover_commands(request)
        git_commands = [cmd for cmd in results if cmd.category == CommandCategory.GIT]
        assert len(git_commands) >= 2  # Should find both git-status and git-commit
        
        # Scenario 2: Beginner user looking for file operations
        request = DiscoveryRequest(
            query="copy file",
            limit=5,
            skill_level=SkillLevel.BEGINNER,
            categories=[CommandCategory.FILE],
            include_deprecated=False
        )
        
        results = await self.engine.discover_commands(request)
        assert len(results) > 0
        file_commands = [cmd for cmd in results if cmd.category == CommandCategory.FILE]
        assert len(file_commands) > 0
        
        # Scenario 3: Expert user looking for advanced tools
        request = DiscoveryRequest(
            query="docker",
            limit=5,
            skill_level=SkillLevel.EXPERT,
            categories=None,
            include_deprecated=False
        )
        
        results = await self.engine.discover_commands(request)
        docker_commands = [cmd for cmd in results 
                          if "docker" in cmd.name.lower() or 
                             "docker" in [tag.lower() for tag in cmd.metadata.tags]]
        assert len(docker_commands) > 0
    
    @pytest.mark.asyncio
    async def test_typo_tolerance(self):
        """Test discovery with common typos."""
        
        # Test common typos
        typo_queries = [
            ("git-stauts", "git-status"),  # Missing 't'
            ("file-coyp", "file-copy"),    # Transposed letters
            ("pythn-test", "python-test"), # Missing letter
            ("doker-compose", "docker-compose")  # Wrong letter
        ]
        
        for typo_query, expected_command in typo_queries:
            request = DiscoveryRequest(
                query=typo_query,
                limit=3,
                skill_level=SkillLevel.EXPERT,
                categories=None,
                include_deprecated=False
            )
            
            results = await self.engine.discover_commands(request)
            
            # Should still find the intended command despite typo
            found_expected = any(cmd.name == expected_command for cmd in results)
            assert found_expected, f"Failed to find '{expected_command}' with typo query '{typo_query}'"
    
    @pytest.mark.asyncio
    async def test_performance_with_realistic_dataset(self):
        """Test performance with realistic dataset size."""
        
        # Create a dataset similar to what might exist in production
        large_dataset = []
        categories = list(CommandCategory)
        skill_levels = list(SkillLevel)
        
        for i in range(500):  # 500 commands
            large_dataset.append(CommandDefinition(
                name=f"command-{i}-{''.join([chr(97 + (i + j) % 26) for j in range(5)])}",
                category=categories[i % len(categories)],
                description=f"Command {i} for performing various operations and tasks",
                metadata=CommandMetadata(
                    version=f"{(i % 3) + 1}.{(i % 5)}.0",
                    author=f"team-{i % 10}",
                    skill_level=skill_levels[i % len(skill_levels)],
                    tags=[f"tag{i}", f"category{i % 10}", f"feature{i % 20}"],
                    usage_count=i % 1000,
                    last_used="2023-11-01T00:00:00Z"
                ),
                parameters=[]
            ))
        
        large_engine = DiscoveryEngine()
        large_engine.initialize(large_dataset)
        
        request = DiscoveryRequest(
            query="command",
            limit=20,
            skill_level=SkillLevel.INTERMEDIATE,
            categories=None,
            include_deprecated=False
        )
        
        # Test discovery performance
        start_time = time.time()
        results = await large_engine.discover_commands(request)
        end_time = time.time()
        
        discovery_time_ms = (end_time - start_time) * 1000
        
        # Should meet performance requirement even with large dataset
        assert discovery_time_ms < 10.0, f"Discovery took {discovery_time_ms}ms with large dataset"
        assert len(results) <= 20
        assert len(results) > 0