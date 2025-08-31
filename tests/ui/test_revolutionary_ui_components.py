"""
Revolutionary UI Components Test Suite

Comprehensive testing for all revolutionary UI components including:
- Progressive Disclosure Manager
- AI Command Composer  
- Symphony Dashboard
- Smart Onboarding Flow
- Accessibility Performance Engine
- Real-time Input Components
- Enhanced Chat Interface
"""

import pytest
import asyncio
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import UI components
from agentsmcp.ui.components.progressive_disclosure_manager import (
    ProgressiveDisclosureManager, UserSkillLevel, DisclosureLevel
)
from agentsmcp.ui.components.ai_command_composer import (
    AICommandComposer, ComposerMode, CommandSuggestion
)
from agentsmcp.ui.components.symphony_dashboard import (
    SymphonyDashboard, AgentState, TaskStatus, AgentInfo
)
from agentsmcp.ui.components.smart_onboarding_flow import (
    SmartOnboardingFlow, OnboardingStage, UserProfile
)
from agentsmcp.ui.components.realtime_input import (
    RealtimeInput, InputMode, InputState
)
from agentsmcp.ui.components.enhanced_chat import (
    EnhancedChat, ChatMessage, ChatMode
)
from agentsmcp.ui.components.chat_history import (
    ChatHistory, HistoryEntry, SearchFilter
)
from agentsmcp.ui.components.accessibility_performance_engine import (
    AccessibilityPerformanceEngine, AccessibilityLevel
)


@pytest.fixture
def progressive_disclosure_manager():
    """Create progressive disclosure manager for testing."""
    return ProgressiveDisclosureManager()


@pytest.fixture
async def ai_command_composer():
    """Create AI command composer for testing."""
    composer = AICommandComposer()
    await composer.initialize()
    yield composer
    await composer.cleanup()


@pytest.fixture
async def symphony_dashboard():
    """Create symphony dashboard for testing."""
    dashboard = SymphonyDashboard()
    await dashboard.initialize()
    yield dashboard
    await dashboard.cleanup()


@pytest.fixture
async def smart_onboarding_flow():
    """Create smart onboarding flow for testing."""
    flow = SmartOnboardingFlow()
    await flow.initialize()
    yield flow
    await flow.cleanup()


@pytest.fixture
def realtime_input():
    """Create realtime input component for testing."""
    return RealtimeInput()


@pytest.fixture
def enhanced_chat():
    """Create enhanced chat component for testing."""
    return EnhancedChat()


@pytest.fixture
def chat_history():
    """Create chat history component for testing."""
    return ChatHistory()


@pytest.fixture
def accessibility_engine():
    """Create accessibility performance engine for testing."""
    return AccessibilityPerformanceEngine()


class TestProgressiveDisclosureManager:
    """Test suite for Progressive Disclosure Manager component."""

    def test_initialization(self, progressive_disclosure_manager):
        """Test progressive disclosure manager initializes correctly."""
        assert progressive_disclosure_manager.current_skill_level == UserSkillLevel.INTERMEDIATE
        assert progressive_disclosure_manager.disclosure_level == DisclosureLevel.STANDARD
        assert progressive_disclosure_manager.user_actions_count == 0

    def test_skill_level_detection(self, progressive_disclosure_manager):
        """Test automatic skill level detection from user behavior."""
        # Simulate beginner behavior patterns
        beginner_actions = [
            "help_requested",
            "error_occurred", 
            "simple_command_used",
            "tutorial_accessed",
            "documentation_viewed"
        ]
        
        for action in beginner_actions * 3:  # Repeat to establish pattern
            progressive_disclosure_manager.record_user_action(action)
        
        progressive_disclosure_manager.analyze_skill_level()
        
        assert progressive_disclosure_manager.current_skill_level == UserSkillLevel.BEGINNER

    def test_adaptive_interface_adjustment(self, progressive_disclosure_manager):
        """Test adaptive interface adjustments based on skill level."""
        # Test beginner adjustments
        progressive_disclosure_manager.set_skill_level(UserSkillLevel.BEGINNER)
        beginner_interface = progressive_disclosure_manager.get_interface_config()
        
        assert beginner_interface["show_tooltips"] is True
        assert beginner_interface["show_examples"] is True
        assert beginner_interface["complexity_level"] == "simple"
        
        # Test expert adjustments
        progressive_disclosure_manager.set_skill_level(UserSkillLevel.EXPERT)
        expert_interface = progressive_disclosure_manager.get_interface_config()
        
        assert expert_interface["show_advanced_features"] is True
        assert expert_interface["show_shortcuts"] is True
        assert expert_interface["complexity_level"] == "full"

    def test_contextual_feature_revelation(self, progressive_disclosure_manager):
        """Test contextual feature revelation based on current task."""
        context = {
            "current_task": "code_development",
            "user_experience": "intermediate",
            "session_duration": 30
        }
        
        revealed_features = progressive_disclosure_manager.get_contextual_features(context)
        
        assert "code_analysis_tools" in revealed_features
        assert "debugging_features" in revealed_features
        assert len(revealed_features) > 3

    def test_learning_curve_tracking(self, progressive_disclosure_manager):
        """Test learning curve tracking and progression."""
        # Simulate learning progression
        learning_milestones = [
            ("first_command", 1),
            ("basic_navigation", 5),
            ("advanced_features", 15), 
            ("expert_shortcuts", 50),
            ("custom_workflows", 100)
        ]
        
        for milestone, action_count in learning_milestones:
            for _ in range(action_count):
                progressive_disclosure_manager.record_user_action("productive_action")
            
            progression = progressive_disclosure_manager.get_learning_progression()
            
            if action_count >= 50:
                assert progression["level"] in ["advanced", "expert"]


class TestAICommandComposer:
    """Test suite for AI Command Composer component."""

    @pytest.mark.asyncio
    async def test_natural_language_processing(self, ai_command_composer):
        """Test natural language processing accuracy."""
        natural_commands = [
            "create a new Python file called main.py",
            "show me the current status of all agents",
            "help me understand how to use symphony mode",
            "configure the system for development work",
            "search for all functions in the codebase"
        ]
        
        for natural_command in natural_commands:
            suggestion = await ai_command_composer.process_natural_language(natural_command)
            
            assert isinstance(suggestion, CommandSuggestion)
            assert suggestion.confidence >= 0.7
            assert suggestion.executable_command is not None
            assert len(suggestion.executable_command.strip()) > 0

    @pytest.mark.asyncio
    async def test_command_suggestion_ranking(self, ai_command_composer):
        """Test command suggestion ranking and selection."""
        ambiguous_input = "show information"
        
        suggestions = await ai_command_composer.get_suggestions(ambiguous_input, max_suggestions=5)
        
        assert len(suggestions) >= 3
        
        # Should be ranked by confidence
        confidences = [s.confidence for s in suggestions]
        assert confidences == sorted(confidences, reverse=True)
        
        # Top suggestion should be reasonable
        top_suggestion = suggestions[0]
        assert "show" in top_suggestion.executable_command.lower() or "info" in top_suggestion.executable_command.lower()

    @pytest.mark.asyncio
    async def test_contextual_command_enhancement(self, ai_command_composer):
        """Test contextual command enhancement."""
        context = {
            "current_directory": "/projects/myapp",
            "recent_files": ["src/main.py", "tests/test_main.py"],
            "active_mode": "development"
        }
        
        base_command = "run tests"
        enhanced_suggestion = await ai_command_composer.enhance_with_context(base_command, context)
        
        assert isinstance(enhanced_suggestion, CommandSuggestion)
        assert enhanced_suggestion.context_aware is True
        # Should include path information or test-specific flags
        assert "/projects/myapp" in enhanced_suggestion.reasoning or "test" in enhanced_suggestion.executable_command

    @pytest.mark.asyncio
    async def test_command_validation_safety(self, ai_command_composer):
        """Test command validation and safety checks."""
        dangerous_commands = [
            "delete all files",
            "format the system drive",
            "remove everything in root directory",
            "shutdown the computer permanently"
        ]
        
        for dangerous_command in dangerous_commands:
            suggestion = await ai_command_composer.process_natural_language(dangerous_command)
            
            # Should either block or add strong warnings
            assert suggestion.safety_level in ["warning", "blocked", "dangerous"]
            if suggestion.safety_level != "blocked":
                assert len(suggestion.safety_warnings) > 0

    @pytest.mark.asyncio
    async def test_multi_step_decomposition(self, ai_command_composer):
        """Test multi-step task decomposition."""
        complex_task = "Set up a new Python project with virtual environment, dependencies, tests, and CI/CD"
        
        suggestion = await ai_command_composer.process_natural_language(complex_task)
        
        assert isinstance(suggestion, CommandSuggestion)
        assert len(suggestion.steps) >= 4
        
        # Should have logical order
        steps_text = " ".join(suggestion.steps).lower()
        assert "virtual" in steps_text or "venv" in steps_text
        assert "dependencies" in steps_text or "requirements" in steps_text
        assert "test" in steps_text
        assert "ci" in steps_text or "github" in steps_text


class TestSymphonyDashboard:
    """Test suite for Symphony Dashboard component."""

    @pytest.mark.asyncio
    async def test_agent_registration_management(self, symphony_dashboard):
        """Test agent registration and management."""
        test_agents = [
            AgentInfo(id="claude-1", name="Claude Coder", state=AgentState.IDLE, capabilities=["coding", "analysis"]),
            AgentInfo(id="ollama-1", name="Ollama Optimizer", state=AgentState.ACTIVE, capabilities=["optimization"]),
            AgentInfo(id="gemini-1", name="Gemini Generator", state=AgentState.BUSY, capabilities=["generation"])
        ]
        
        # Register agents
        for agent in test_agents:
            await symphony_dashboard.register_agent(agent)
        
        assert symphony_dashboard.get_agent_count() == 3
        
        # Test agent retrieval
        claude_agent = symphony_dashboard.get_agent("claude-1")
        assert claude_agent is not None
        assert claude_agent.name == "Claude Coder"

    @pytest.mark.asyncio
    async def test_real_time_status_updates(self, symphony_dashboard):
        """Test real-time agent status updates."""
        agent_id = "test-agent"
        agent = AgentInfo(id=agent_id, name="Test Agent", state=AgentState.IDLE)
        
        await symphony_dashboard.register_agent(agent)
        
        # Test state transitions
        state_sequence = [
            AgentState.INITIALIZING,
            AgentState.ACTIVE,
            AgentState.BUSY,
            AgentState.WAITING,
            AgentState.ERROR,
            AgentState.IDLE
        ]
        
        for new_state in state_sequence:
            await symphony_dashboard.update_agent_state(agent_id, new_state)
            
            current_state = symphony_dashboard.get_agent_state(agent_id)
            assert current_state == new_state
            
            # Check status update was logged
            status_history = symphony_dashboard.get_status_history(agent_id)
            assert len(status_history) > 0

    @pytest.mark.asyncio
    async def test_task_orchestration_visualization(self, symphony_dashboard):
        """Test task orchestration visualization."""
        # Create test tasks
        from agentsmcp.ui.components.symphony_dashboard import TaskInfo
        
        tasks = [
            TaskInfo(id="task-1", name="Code Review", status=TaskStatus.RUNNING, assigned_agent="claude-1"),
            TaskInfo(id="task-2", name="Performance Analysis", status=TaskStatus.PENDING, assigned_agent="ollama-1"),
            TaskInfo(id="task-3", name="Documentation", status=TaskStatus.COMPLETED, assigned_agent="gemini-1")
        ]
        
        for task in tasks:
            await symphony_dashboard.add_task(task)
        
        # Test orchestration view
        orchestration_view = await symphony_dashboard.get_orchestration_view()
        
        assert len(orchestration_view["active_tasks"]) >= 1  # Running tasks
        assert len(orchestration_view["completed_tasks"]) >= 1
        assert "task_distribution" in orchestration_view

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, symphony_dashboard):
        """Test performance monitoring and metrics collection."""
        agent_id = "perf-test-agent"
        agent = AgentInfo(id=agent_id, name="Performance Test Agent", state=AgentState.ACTIVE)
        
        await symphony_dashboard.register_agent(agent)
        
        # Simulate agent activity
        for i in range(10):
            await symphony_dashboard.record_agent_activity(
                agent_id, 
                f"task-{i}", 
                duration=0.1 + (i * 0.05),
                success=i < 8  # 80% success rate
            )
        
        metrics = await symphony_dashboard.get_performance_metrics(agent_id)
        
        assert "average_task_duration" in metrics
        assert "success_rate" in metrics
        assert "throughput" in metrics
        assert metrics["success_rate"] >= 0.7  # Should reflect simulated 80%

    @pytest.mark.asyncio
    async def test_collaborative_workflow_coordination(self, symphony_dashboard):
        """Test collaborative workflow coordination between multiple agents."""
        # Set up collaborative agents
        agents = [
            AgentInfo(id="frontend-dev", name="Frontend Developer", state=AgentState.ACTIVE, capabilities=["react", "ui"]),
            AgentInfo(id="backend-dev", name="Backend Developer", state=AgentState.ACTIVE, capabilities=["api", "database"]),
            AgentInfo(id="devops-eng", name="DevOps Engineer", state=AgentState.ACTIVE, capabilities=["deployment", "monitoring"])
        ]
        
        for agent in agents:
            await symphony_dashboard.register_agent(agent)
        
        # Create collaborative workflow
        workflow_id = "fullstack-app"
        await symphony_dashboard.create_collaborative_workflow(
            workflow_id,
            "Full-Stack App Development",
            ["frontend-dev", "backend-dev", "devops-eng"]
        )
        
        workflow_status = await symphony_dashboard.get_workflow_status(workflow_id)
        
        assert workflow_status["participant_count"] == 3
        assert "coordination_graph" in workflow_status
        assert workflow_status["status"] in ["active", "coordinating"]


class TestSmartOnboardingFlow:
    """Test suite for Smart Onboarding Flow component."""

    @pytest.mark.asyncio
    async def test_onboarding_stage_progression(self, smart_onboarding_flow):
        """Test onboarding stage progression and tracking."""
        expected_stages = [
            OnboardingStage.WELCOME,
            OnboardingStage.USER_PROFILING,
            OnboardingStage.BASIC_COMMANDS,
            OnboardingStage.ADVANCED_FEATURES,
            OnboardingStage.SYMPHONY_MODE,
            OnboardingStage.COMPLETION
        ]
        
        for expected_stage in expected_stages:
            current_stage = smart_onboarding_flow.get_current_stage()
            
            if current_stage == OnboardingStage.COMPLETION:
                break
                
            # Simulate stage completion
            await smart_onboarding_flow.complete_current_stage()
            await smart_onboarding_flow.advance_to_next_stage()
        
        assert smart_onboarding_flow.is_complete()
        assert smart_onboarding_flow.get_completion_percentage() >= 90

    @pytest.mark.asyncio
    async def test_user_profile_creation(self, smart_onboarding_flow):
        """Test user profile creation and customization."""
        user_responses = {
            "experience_level": "intermediate",
            "primary_use_case": "data_science", 
            "preferred_interface": "cli_with_hints",
            "team_collaboration": True,
            "automation_preference": "high"
        }
        
        profile = await smart_onboarding_flow.create_user_profile(user_responses)
        
        assert isinstance(profile, UserProfile)
        assert profile.experience_level == "intermediate"
        assert profile.use_case == "data_science"
        assert profile.collaboration_enabled is True

    @pytest.mark.asyncio
    async def test_personalized_guidance_delivery(self, smart_onboarding_flow):
        """Test personalized guidance delivery based on user profile."""
        # Create different user profiles
        profiles = [
            UserProfile(experience_level="beginner", use_case="general"),
            UserProfile(experience_level="advanced", use_case="development"),
            UserProfile(experience_level="expert", use_case="data_science")
        ]
        
        guidance_topic = "using_symphony_mode"
        
        for profile in profiles:
            await smart_onboarding_flow.set_user_profile(profile)
            guidance = await smart_onboarding_flow.get_guidance(guidance_topic)
            
            # Guidance should adapt to experience level
            if profile.experience_level == "beginner":
                assert "step-by-step" in guidance.content.lower()
                assert len(guidance.examples) >= 2
            elif profile.experience_level == "expert":
                assert guidance.detail_level == "concise"
                assert "advanced" in guidance.content.lower() or len(guidance.content) < 500

    @pytest.mark.asyncio
    async def test_adaptive_learning_path(self, smart_onboarding_flow):
        """Test adaptive learning path based on user performance."""
        # Simulate user struggling with concepts
        struggle_events = [
            "command_syntax_error",
            "feature_confusion", 
            "repeated_help_requests",
            "task_abandonment"
        ]
        
        for event in struggle_events:
            await smart_onboarding_flow.record_learning_event(event, success=False)
        
        # Should adapt to provide more support
        adaptive_path = await smart_onboarding_flow.get_adaptive_path()
        
        assert adaptive_path["support_level"] == "high"
        assert "remedial" in adaptive_path or "additional_practice" in adaptive_path

    @pytest.mark.asyncio
    async def test_contextual_help_system(self, smart_onboarding_flow):
        """Test contextual help system integration."""
        contexts = [
            {"current_command": "status", "user_input": "show agents", "error_occurred": False},
            {"current_command": "symphony", "user_input": "coordinate 5 agents", "error_occurred": True},
            {"current_command": "help", "user_input": "explain features", "error_occurred": False}
        ]
        
        for context in contexts:
            help_content = await smart_onboarding_flow.get_contextual_help(context)
            
            assert help_content is not None
            assert len(help_content.content) > 50  # Substantial help content
            
            # Help should be relevant to current command
            if context["current_command"] in ["status", "symphony"]:
                assert context["current_command"] in help_content.content.lower()


class TestRealtimeInput:
    """Test suite for Realtime Input component."""

    def test_input_mode_switching(self, realtime_input):
        """Test input mode switching functionality."""
        # Test mode transitions
        modes = [InputMode.SINGLE_LINE, InputMode.MULTI_LINE, InputMode.COMMAND_PALETTE]
        
        for mode in modes:
            realtime_input.set_input_mode(mode)
            assert realtime_input.get_current_mode() == mode

    def test_real_time_validation(self, realtime_input):
        """Test real-time input validation."""
        # Test valid inputs
        valid_inputs = [
            "status",
            "create file main.py",
            "help symphony mode",
            "search functions"
        ]
        
        for input_text in valid_inputs:
            validation_result = realtime_input.validate_input(input_text)
            assert validation_result.is_valid is True
        
        # Test invalid inputs
        invalid_inputs = [
            "",  # Empty
            "   ",  # Whitespace only
            "invalid_command_that_does_not_exist_anywhere",
            "delete / --recursive --force"  # Potentially dangerous
        ]
        
        for input_text in invalid_inputs:
            validation_result = realtime_input.validate_input(input_text)
            if not validation_result.is_valid:
                assert len(validation_result.error_messages) > 0

    def test_auto_completion_suggestions(self, realtime_input):
        """Test auto-completion suggestions."""
        partial_inputs = [
            "sta",     # Should suggest "status"
            "create",  # Should suggest "create file", etc.
            "hel",     # Should suggest "help"
            "sym"      # Should suggest "symphony"
        ]
        
        for partial in partial_inputs:
            suggestions = realtime_input.get_completions(partial)
            
            assert len(suggestions) > 0
            assert all(s.startswith(partial) for s in suggestions)

    def test_multi_line_input_handling(self, realtime_input):
        """Test multi-line input handling."""
        realtime_input.set_input_mode(InputMode.MULTI_LINE)
        
        multi_line_input = """
        create project structure:
        - src/main.py
        - tests/test_main.py
        - README.md
        - requirements.txt
        """
        
        # Should handle multi-line input properly
        processed_input = realtime_input.process_input(multi_line_input)
        assert processed_input is not None
        assert len(processed_input.split('\n')) > 1

    def test_input_history_management(self, realtime_input):
        """Test input history management."""
        # Add commands to history
        commands = [
            "status",
            "create file test.py",
            "search functions",
            "help symphony",
            "config settings"
        ]
        
        for command in commands:
            realtime_input.add_to_history(command)
        
        history = realtime_input.get_history()
        assert len(history) == len(commands)
        
        # Test history navigation
        assert realtime_input.get_previous_command() == commands[-1]
        assert realtime_input.get_next_command() == commands[-2]


class TestEnhancedChat:
    """Test suite for Enhanced Chat component."""

    def test_chat_message_rendering(self, enhanced_chat):
        """Test chat message rendering with different formats."""
        messages = [
            ChatMessage(role="user", content="Hello, can you help me?", timestamp=time.time()),
            ChatMessage(role="assistant", content="Of course! What do you need help with?", timestamp=time.time()),
            ChatMessage(role="system", content="Symphony mode activated", timestamp=time.time()),
            ChatMessage(role="user", content="Create a Python script for data analysis", timestamp=time.time())
        ]
        
        for message in messages:
            rendered = enhanced_chat.render_message(message)
            
            assert rendered is not None
            assert len(rendered) > 0
            assert message.role in rendered or message.content in rendered

    def test_code_syntax_highlighting(self, enhanced_chat):
        """Test code syntax highlighting in chat messages."""
        code_message = ChatMessage(
            role="assistant", 
            content="""Here's a Python example:
            
```python
def hello_world():
    print("Hello, World!")
    return True
```

This function prints a greeting.""",
            timestamp=time.time()
        )
        
        rendered = enhanced_chat.render_message(code_message)
        
        # Should apply syntax highlighting
        assert "python" in rendered.lower() or "code" in rendered.lower()

    def test_message_threading(self, enhanced_chat):
        """Test message threading and conversation flow."""
        # Create conversation thread
        thread_messages = [
            ChatMessage(role="user", content="How do I create a file?", thread_id="help-001"),
            ChatMessage(role="assistant", content="You can use 'create file filename.py'", thread_id="help-001"),
            ChatMessage(role="user", content="What about directories?", thread_id="help-001"),
            ChatMessage(role="assistant", content="Use 'create directory dirname'", thread_id="help-001")
        ]
        
        for message in thread_messages:
            enhanced_chat.add_message(message)
        
        thread = enhanced_chat.get_thread("help-001")
        assert len(thread.messages) == 4
        assert all(msg.thread_id == "help-001" for msg in thread.messages)

    def test_chat_search_functionality(self, enhanced_chat):
        """Test chat search functionality."""
        # Add searchable messages
        searchable_messages = [
            ChatMessage(role="user", content="How to create Python files?"),
            ChatMessage(role="assistant", content="Use create file command with .py extension"),
            ChatMessage(role="user", content="What about JavaScript files?"),
            ChatMessage(role="assistant", content="Use create file command with .js extension"),
            ChatMessage(role="user", content="Show me agent status")
        ]
        
        for message in searchable_messages:
            enhanced_chat.add_message(message)
        
        # Test search
        search_results = enhanced_chat.search_messages("create file")
        assert len(search_results) >= 2  # Should find both create file references
        
        search_results = enhanced_chat.search_messages("Python")
        assert len(search_results) >= 1

    def test_message_export_functionality(self, enhanced_chat):
        """Test message export functionality."""
        # Add messages to export
        export_messages = [
            ChatMessage(role="user", content="Start new session"),
            ChatMessage(role="assistant", content="Session started successfully"),
            ChatMessage(role="user", content="Create project structure"),
            ChatMessage(role="assistant", content="Project structure created")
        ]
        
        for message in export_messages:
            enhanced_chat.add_message(message)
        
        # Test export
        exported_data = enhanced_chat.export_conversation()
        
        assert "messages" in exported_data
        assert len(exported_data["messages"]) == len(export_messages)
        assert "timestamp" in exported_data
        assert "format_version" in exported_data


class TestChatHistory:
    """Test suite for Chat History component."""

    def test_history_persistence(self, chat_history):
        """Test chat history persistence and retrieval."""
        # Add history entries
        from agentsmcp.ui.components.chat_history import HistoryEntry
        
        entries = [
            HistoryEntry(session_id="session-1", message="Create new project", timestamp=time.time()),
            HistoryEntry(session_id="session-1", message="Add tests", timestamp=time.time()),
            HistoryEntry(session_id="session-2", message="Deploy to production", timestamp=time.time())
        ]
        
        for entry in entries:
            chat_history.add_entry(entry)
        
        # Test retrieval
        session_1_history = chat_history.get_session_history("session-1")
        assert len(session_1_history) == 2
        
        all_history = chat_history.get_all_history()
        assert len(all_history) >= 3

    def test_history_search_and_filtering(self, chat_history):
        """Test history search and filtering capabilities."""
        from agentsmcp.ui.components.chat_history import HistoryEntry, SearchFilter
        
        # Add diverse history entries
        test_entries = [
            HistoryEntry(session_id="dev", message="create Python file", timestamp=time.time(), tags=["coding"]),
            HistoryEntry(session_id="dev", message="run tests", timestamp=time.time(), tags=["testing"]),
            HistoryEntry(session_id="ops", message="deploy application", timestamp=time.time(), tags=["deployment"]),
            HistoryEntry(session_id="ops", message="monitor performance", timestamp=time.time(), tags=["monitoring"])
        ]
        
        for entry in test_entries:
            chat_history.add_entry(entry)
        
        # Test search
        search_filter = SearchFilter(keywords=["Python"], tags=["coding"])
        search_results = chat_history.search_history(search_filter)
        
        assert len(search_results) >= 1
        assert "Python" in search_results[0].message

    def test_history_analytics(self, chat_history):
        """Test history analytics and insights."""
        from agentsmcp.ui.components.chat_history import HistoryEntry
        
        # Add entries spanning different time periods
        import datetime
        base_time = datetime.datetime.now()
        
        analytical_entries = [
            HistoryEntry(
                session_id="analytics",
                message="create file",
                timestamp=base_time.timestamp(),
                tags=["file_operation"]
            ),
            HistoryEntry(
                session_id="analytics", 
                message="create file",
                timestamp=(base_time + datetime.timedelta(hours=1)).timestamp(),
                tags=["file_operation"]
            ),
            HistoryEntry(
                session_id="analytics",
                message="status check", 
                timestamp=(base_time + datetime.timedelta(hours=2)).timestamp(),
                tags=["monitoring"]
            )
        ]
        
        for entry in analytical_entries:
            chat_history.add_entry(entry)
        
        analytics = chat_history.get_analytics()
        
        assert "most_common_commands" in analytics
        assert "usage_patterns" in analytics
        assert "session_statistics" in analytics

    def test_history_cleanup_and_archiving(self, chat_history):
        """Test history cleanup and archiving functionality."""
        from agentsmcp.ui.components.chat_history import HistoryEntry
        import datetime
        
        # Add old entries that should be archived
        old_time = (datetime.datetime.now() - datetime.timedelta(days=90)).timestamp()
        recent_time = datetime.datetime.now().timestamp()
        
        old_entries = [
            HistoryEntry(session_id="old", message="old command 1", timestamp=old_time),
            HistoryEntry(session_id="old", message="old command 2", timestamp=old_time)
        ]
        
        recent_entries = [
            HistoryEntry(session_id="recent", message="recent command", timestamp=recent_time)
        ]
        
        for entry in old_entries + recent_entries:
            chat_history.add_entry(entry)
        
        # Test archiving
        archived_count = chat_history.archive_old_entries(days_threshold=30)
        assert archived_count >= 2  # Should archive old entries
        
        # Recent entries should still be accessible
        recent_history = chat_history.get_recent_history(days=30)
        assert len(recent_history) >= 1


class TestAccessibilityPerformanceEngine:
    """Test suite for Accessibility Performance Engine."""

    def test_accessibility_audit_initialization(self, accessibility_engine):
        """Test accessibility audit system initialization."""
        assert accessibility_engine.compliance_target == AccessibilityLevel.AAA
        assert hasattr(accessibility_engine, 'audit_keyboard_navigation')
        assert hasattr(accessibility_engine, 'audit_color_contrast')
        assert hasattr(accessibility_engine, 'audit_screen_reader_support')

    @pytest.mark.asyncio
    async def test_keyboard_navigation_audit(self, accessibility_engine):
        """Test keyboard navigation accessibility audit."""
        ui_components = [
            "command_input_field",
            "agent_status_list",
            "symphony_dashboard",
            "chat_history_panel",
            "settings_menu"
        ]
        
        for component in ui_components:
            audit_result = await accessibility_engine.audit_keyboard_navigation(component)
            
            assert audit_result.is_compliant
            assert audit_result.compliance_score >= 0.9
            assert "tab_navigation" in audit_result.passed_checks
            assert "focus_indicators" in audit_result.passed_checks

    @pytest.mark.asyncio
    async def test_color_contrast_compliance(self, accessibility_engine):
        """Test color contrast compliance validation."""
        color_schemes = [
            {"background": "#ffffff", "foreground": "#000000"},  # High contrast
            {"background": "#1e1e1e", "foreground": "#ffffff"},  # Dark theme
            {"background": "#f5f5f5", "foreground": "#333333"},  # Light theme
        ]
        
        for scheme in color_schemes:
            contrast_audit = await accessibility_engine.audit_color_contrast(scheme)
            
            assert contrast_audit.is_compliant
            assert contrast_audit.contrast_ratio >= 7.0  # AAA requirement
            assert contrast_audit.compliance_level == "AAA"

    @pytest.mark.asyncio
    async def test_screen_reader_support(self, accessibility_engine):
        """Test screen reader support validation."""
        components_with_content = [
            {"component": "status_display", "has_text": True, "has_images": False},
            {"component": "agent_cards", "has_text": True, "has_images": True},
            {"component": "progress_indicators", "has_text": False, "has_images": True}
        ]
        
        for component_info in components_with_content:
            support_audit = await accessibility_engine.audit_screen_reader_support(component_info)
            
            assert support_audit.is_compliant
            assert "aria_labels" in support_audit.accessibility_features
            
            if component_info["has_images"]:
                assert "alternative_text" in support_audit.accessibility_features

    @pytest.mark.asyncio
    async def test_responsive_accessibility(self, accessibility_engine):
        """Test responsive design accessibility."""
        viewport_configurations = [
            {"width": 320, "height": 568, "device": "mobile"},
            {"width": 768, "height": 1024, "device": "tablet"}, 
            {"width": 1920, "height": 1080, "device": "desktop"}
        ]
        
        for config in viewport_configurations:
            responsive_audit = await accessibility_engine.audit_responsive_design(config)
            
            assert responsive_audit.is_compliant
            assert responsive_audit.usability_score >= 0.8
            
            if config["device"] == "mobile":
                assert "touch_friendly" in responsive_audit.accessibility_features

    @pytest.mark.asyncio
    async def test_performance_accessibility_integration(self, accessibility_engine):
        """Test performance and accessibility integration."""
        performance_config = {
            "max_render_time": 100,  # 100ms
            "max_interaction_delay": 50,  # 50ms
            "accessibility_features_enabled": True
        }
        
        integration_audit = await accessibility_engine.audit_performance_accessibility(performance_config)
        
        assert integration_audit.meets_performance_targets
        assert integration_audit.accessibility_impact_score <= 0.1  # Minimal impact
        assert integration_audit.overall_score >= 0.9


# Integration tests for component interactions
class TestUIComponentIntegration:
    """Test suite for UI component integration scenarios."""

    @pytest.mark.asyncio
    async def test_progressive_disclosure_with_onboarding(self, progressive_disclosure_manager, smart_onboarding_flow):
        """Test integration between progressive disclosure and onboarding."""
        # Set beginner user in onboarding
        beginner_profile = UserProfile(experience_level="beginner")
        await smart_onboarding_flow.set_user_profile(beginner_profile)
        
        # Progressive disclosure should adapt
        progressive_disclosure_manager.sync_with_onboarding(smart_onboarding_flow)
        
        assert progressive_disclosure_manager.current_skill_level == UserSkillLevel.BEGINNER
        assert progressive_disclosure_manager.disclosure_level == DisclosureLevel.DETAILED

    @pytest.mark.asyncio
    async def test_symphony_dashboard_with_chat(self, symphony_dashboard, enhanced_chat):
        """Test integration between symphony dashboard and chat."""
        # Register agents in dashboard
        agent = AgentInfo(id="chat-agent", name="Chat Agent", state=AgentState.ACTIVE)
        await symphony_dashboard.register_agent(agent)
        
        # Simulate agent communication through chat
        agent_message = ChatMessage(
            role="agent",
            content="Task completed successfully",
            sender_id="chat-agent",
            timestamp=time.time()
        )
        enhanced_chat.add_message(agent_message)
        
        # Dashboard should reflect agent activity
        agent_status = symphony_dashboard.get_agent_state("chat-agent")
        assert agent_status == AgentState.ACTIVE

    @pytest.mark.asyncio
    async def test_command_composer_with_realtime_input(self, ai_command_composer, realtime_input):
        """Test integration between AI command composer and realtime input."""
        # Set up realtime input with command composer
        realtime_input.set_command_composer(ai_command_composer)
        
        partial_input = "create pyt"
        
        # Should get AI-enhanced completions
        completions = await realtime_input.get_ai_completions(partial_input)
        
        assert len(completions) > 0
        assert any("python" in completion.lower() for completion in completions)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])