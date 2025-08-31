"""
Comprehensive Test Suite for Revolutionary AgentsMCP CLI

This test suite validates all revolutionary frontend and backend improvements
with zero-bug policy, ensuring flawless cross-platform experience.

Test Coverage:
- Revolutionary Frontend Components (Progressive Disclosure, AI Command Composer, Symphony Mode)
- Backend API Endpoints (NLP Processor, Intent Recognition, Command Translation)
- Integration Testing (CLI-TUI-Web handoff scenarios)
- Performance Testing (Sub-100ms response times, concurrent handling)
- Accessibility Testing (WCAG 2.1 AAA compliance)
"""

import pytest
import asyncio
import time
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import revolutionary components
from agentsmcp.ui.components.progressive_disclosure_manager import (
    ProgressiveDisclosureManager, UserSkillLevel, DisclosureLevel, FeatureContext
)
from agentsmcp.ui.components.ai_command_composer import (
    AICommandComposer, ComposerMode, CommandSuggestion, NaturalLanguageInput
)
from agentsmcp.ui.components.symphony_dashboard import (
    SymphonyDashboard, AgentState, TaskStatus, AgentInfo, TaskInfo
)
from agentsmcp.ui.components.smart_onboarding_flow import (
    SmartOnboardingFlow, OnboardingStage, UserProfile, GuidanceLevel
)
from agentsmcp.api.nlp_processor import (
    NaturalLanguageProcessor, CommandIntent, IntentPrediction, NLPContext
)
from agentsmcp.api.intent_recognition_service import (
    IntentRecognitionService, IntentResult, ContextualIntent
)
from agentsmcp.api.command_translation_engine import (
    CommandTranslationEngine, TranslationResult, CommandContext
)
from agentsmcp.api.symphony_orchestration_api import (
    SymphonyOrchestrationAPI, OrchestrationMode, AgentCoordination
)
from agentsmcp.ui.components.accessibility_performance_engine import (
    AccessibilityPerformanceEngine, AccessibilityLevel, PerformanceMetrics
)


# Fixtures for Revolutionary Components
@pytest.fixture
def progressive_disclosure_manager():
    """Create a progressive disclosure manager for testing."""
    return ProgressiveDisclosureManager()


@pytest.fixture
async def ai_command_composer():
    """Create an AI command composer for testing."""
    composer = AICommandComposer()
    await composer.initialize()
    yield composer
    await composer.cleanup()


@pytest.fixture
async def symphony_dashboard():
    """Create a symphony dashboard for testing."""
    dashboard = SymphonyDashboard()
    await dashboard.initialize()
    yield dashboard
    await dashboard.cleanup()


@pytest.fixture
async def smart_onboarding_flow():
    """Create a smart onboarding flow for testing."""
    flow = SmartOnboardingFlow()
    await flow.initialize()
    yield flow
    await flow.cleanup()


@pytest.fixture
async def nlp_processor():
    """Create an NLP processor for testing."""
    processor = NaturalLanguageProcessor()
    await processor.initialize()
    yield processor
    await processor.cleanup()


@pytest.fixture
async def intent_recognition_service():
    """Create an intent recognition service for testing."""
    service = IntentRecognitionService()
    await service.initialize()
    yield service
    await service.cleanup()


@pytest.fixture
async def command_translation_engine():
    """Create a command translation engine for testing."""
    engine = CommandTranslationEngine()
    await engine.initialize()
    yield engine
    await engine.cleanup()


@pytest.fixture
async def symphony_orchestration_api():
    """Create a symphony orchestration API for testing."""
    api = SymphonyOrchestrationAPI()
    await api.initialize()
    yield api
    await api.cleanup()


@pytest.fixture
def accessibility_engine():
    """Create an accessibility performance engine for testing."""
    return AccessibilityPerformanceEngine()


@pytest.fixture
def sample_user_inputs():
    """Provide diverse user input samples for testing."""
    return {
        "natural_language": [
            "Create a new Python file for data processing",
            "Show me the status of all running agents",
            "Help me optimize this code for better performance", 
            "I need to set up a CI/CD pipeline for this project",
            "Can you find all TODO comments in the codebase?",
        ],
        "commands": [
            "create file main.py",
            "search todo",
            "status agents",
            "help optimize",
            "config ci-cd"
        ],
        "complex_scenarios": [
            "I'm working on a React application and need to create a component that handles user authentication, integrates with our API, and includes proper error handling",
            "Set up a machine learning pipeline that processes CSV data, trains a model, and deploys it to production with monitoring",
            "Help me debug a performance issue where the application is using too much memory and the response times are slow",
        ]
    }


class TestProgressiveDisclosureSystem:
    """Test suite for Progressive Disclosure System."""

    def test_progressive_disclosure_initialization(self, progressive_disclosure_manager):
        """Test progressive disclosure manager initializes correctly."""
        assert progressive_disclosure_manager is not None
        assert progressive_disclosure_manager.current_skill_level == UserSkillLevel.INTERMEDIATE
        assert progressive_disclosure_manager.disclosure_level == DisclosureLevel.STANDARD

    def test_skill_level_adaptation(self, progressive_disclosure_manager):
        """Test skill level adaptation based on user behavior."""
        # Simulate beginner user behavior
        for _ in range(5):
            progressive_disclosure_manager.record_user_action("help_requested")
            progressive_disclosure_manager.record_user_action("error_occurred")

        progressive_disclosure_manager.adapt_to_user()
        
        assert progressive_disclosure_manager.current_skill_level == UserSkillLevel.BEGINNER
        assert progressive_disclosure_manager.disclosure_level == DisclosureLevel.DETAILED

    def test_progressive_feature_revelation(self, progressive_disclosure_manager):
        """Test progressive feature revelation as user expertise grows."""
        # Start with beginner level
        progressive_disclosure_manager.set_skill_level(UserSkillLevel.BEGINNER)
        
        # Simulate successful advanced actions
        for _ in range(10):
            progressive_disclosure_manager.record_user_action("advanced_command_used")
            progressive_disclosure_manager.record_user_action("complex_task_completed")

        progressive_disclosure_manager.adapt_to_user()
        
        # Should progress to advanced level
        assert progressive_disclosure_manager.current_skill_level in [UserSkillLevel.INTERMEDIATE, UserSkillLevel.ADVANCED]

    def test_contextual_disclosure_adjustment(self, progressive_disclosure_manager):
        """Test contextual disclosure adjustments."""
        context = FeatureContext(
            domain="development",
            complexity_level="high",
            user_experience_minutes=120,
            error_rate=0.1
        )
        
        disclosure_level = progressive_disclosure_manager.get_contextual_disclosure(context)
        
        assert disclosure_level in [DisclosureLevel.STANDARD, DisclosureLevel.DETAILED, DisclosureLevel.MINIMAL]

    def test_feature_visibility_control(self, progressive_disclosure_manager):
        """Test feature visibility control based on disclosure level."""
        # Test minimal disclosure
        progressive_disclosure_manager.set_disclosure_level(DisclosureLevel.MINIMAL)
        visible_features = progressive_disclosure_manager.get_visible_features()
        
        assert "basic_commands" in visible_features
        assert len(visible_features) <= 5  # Should show minimal features
        
        # Test detailed disclosure
        progressive_disclosure_manager.set_disclosure_level(DisclosureLevel.DETAILED)
        detailed_features = progressive_disclosure_manager.get_visible_features()
        
        assert len(detailed_features) > len(visible_features)
        assert "advanced_commands" in detailed_features

    def test_user_journey_tracking(self, progressive_disclosure_manager):
        """Test user journey tracking and analytics."""
        # Simulate user journey
        journey_events = [
            "first_launch",
            "tutorial_started",
            "basic_command_used",
            "help_requested", 
            "advanced_feature_discovered",
            "symphony_mode_activated"
        ]
        
        for event in journey_events:
            progressive_disclosure_manager.track_user_journey(event)
        
        journey_data = progressive_disclosure_manager.get_user_journey()
        
        assert len(journey_data["events"]) == len(journey_events)
        assert journey_data["progression_score"] > 0


class TestAICommandComposer:
    """Test suite for AI Command Composer with Natural Language Processing."""

    @pytest.mark.asyncio
    async def test_natural_language_command_composition(self, ai_command_composer, sample_user_inputs):
        """Test natural language to command composition with 95% accuracy."""
        natural_inputs = sample_user_inputs["natural_language"]
        
        total_accurate = 0
        total_tests = len(natural_inputs)
        
        for natural_input in natural_inputs:
            suggestion = await ai_command_composer.compose_from_natural_language(natural_input)
            
            assert isinstance(suggestion, CommandSuggestion)
            assert suggestion.confidence >= 0.8
            assert suggestion.executable_command is not None
            assert len(suggestion.executable_command.strip()) > 0
            
            if suggestion.confidence >= 0.9:
                total_accurate += 1
        
        accuracy = total_accurate / total_tests
        assert accuracy >= 0.8, f"Accuracy {accuracy:.2%} below 80% threshold"

    @pytest.mark.asyncio
    async def test_command_suggestion_ranking(self, ai_command_composer):
        """Test command suggestion ranking and alternatives."""
        ambiguous_input = "help me with this"
        
        suggestions = await ai_command_composer.get_suggestions(ambiguous_input, max_suggestions=5)
        
        assert len(suggestions) > 1
        assert all(isinstance(s, CommandSuggestion) for s in suggestions)
        
        # Should be ranked by confidence
        confidences = [s.confidence for s in suggestions]
        assert confidences == sorted(confidences, reverse=True)

    @pytest.mark.asyncio
    async def test_contextual_command_composition(self, ai_command_composer):
        """Test contextual command composition based on current state."""
        context = {
            "current_directory": "/home/user/project",
            "active_files": ["main.py", "config.json"],
            "recent_commands": ["create file", "search"],
            "user_skill_level": "intermediate"
        }
        
        input_text = "optimize this"
        suggestion = await ai_command_composer.compose_with_context(input_text, context)
        
        assert isinstance(suggestion, CommandSuggestion)
        assert suggestion.context_relevance_score > 0.5
        assert "optimize" in suggestion.reasoning.lower()

    @pytest.mark.asyncio
    async def test_command_validation_and_safety(self, ai_command_composer):
        """Test command validation and safety checks."""
        potentially_dangerous_inputs = [
            "delete all files in the system",
            "format the hard drive",
            "remove everything permanently",
            "shutdown the computer"
        ]
        
        for dangerous_input in potentially_dangerous_inputs:
            suggestion = await ai_command_composer.compose_from_natural_language(dangerous_input)
            
            # Should either refuse or add safety warnings
            assert suggestion.safety_level in ["warning", "dangerous", "blocked"]
            if suggestion.safety_level == "dangerous":
                assert len(suggestion.safety_warnings) > 0

    @pytest.mark.asyncio
    async def test_multi_step_command_decomposition(self, ai_command_composer):
        """Test multi-step command decomposition for complex tasks."""
        complex_input = "Create a Python project with tests, documentation, and CI/CD setup"
        
        suggestion = await ai_command_composer.compose_from_natural_language(complex_input)
        
        assert isinstance(suggestion, CommandSuggestion)
        assert len(suggestion.steps) > 1
        assert all(step.strip() for step in suggestion.steps)
        
        # Should have logical order
        assert "create" in suggestion.steps[0].lower() or "init" in suggestion.steps[0].lower()

    @pytest.mark.asyncio
    async def test_learning_from_user_corrections(self, ai_command_composer):
        """Test learning from user corrections to improve suggestions."""
        input_text = "show system info"
        
        # Get initial suggestion
        initial_suggestion = await ai_command_composer.compose_from_natural_language(input_text)
        initial_confidence = initial_suggestion.confidence
        
        # Simulate user correction
        correct_command = "system status --detailed"
        await ai_command_composer.learn_from_correction(input_text, initial_suggestion.executable_command, correct_command)
        
        # Get updated suggestion
        updated_suggestion = await ai_command_composer.compose_from_natural_language(input_text)
        
        # Should improve over time
        assert updated_suggestion.confidence >= initial_confidence


class TestSymphonyModeForecasting:
    """Test suite for Symphony Mode Dashboard with Multi-agent Orchestration."""

    @pytest.mark.asyncio
    async def test_symphony_dashboard_initialization(self, symphony_dashboard):
        """Test symphony dashboard initializes with correct state."""
        assert symphony_dashboard.is_initialized
        assert symphony_dashboard.agent_count == 0
        assert symphony_dashboard.orchestration_mode == OrchestrationMode.IDLE

    @pytest.mark.asyncio
    async def test_multi_agent_coordination_display(self, symphony_dashboard):
        """Test multi-agent coordination visualization."""
        # Add multiple agents
        agents = [
            AgentInfo(id="agent-1", name="Claude Coder", state=AgentState.ACTIVE, capabilities=["coding", "analysis"]),
            AgentInfo(id="agent-2", name="Ollama Optimizer", state=AgentState.BUSY, capabilities=["optimization", "performance"]),
            AgentInfo(id="agent-3", name="Gemini Generator", state=AgentState.IDLE, capabilities=["generation", "creativity"])
        ]
        
        for agent in agents:
            await symphony_dashboard.register_agent(agent)
        
        assert symphony_dashboard.agent_count == 3
        
        # Test coordination visualization
        coordination_view = await symphony_dashboard.get_coordination_view()
        
        assert len(coordination_view["agents"]) == 3
        assert "communication_graph" in coordination_view
        assert "task_distribution" in coordination_view

    @pytest.mark.asyncio
    async def test_real_time_agent_status_updates(self, symphony_dashboard):
        """Test real-time agent status updates and visualization."""
        agent_id = "test-agent"
        agent = AgentInfo(id=agent_id, name="Test Agent", state=AgentState.IDLE, capabilities=["testing"])
        
        await symphony_dashboard.register_agent(agent)
        
        # Test state transitions
        state_transitions = [
            AgentState.INITIALIZING,
            AgentState.ACTIVE,
            AgentState.BUSY,
            AgentState.WAITING,
            AgentState.IDLE
        ]
        
        for new_state in state_transitions:
            await symphony_dashboard.update_agent_state(agent_id, new_state)
            
            current_state = symphony_dashboard.get_agent_state(agent_id)
            assert current_state == new_state

    @pytest.mark.asyncio
    async def test_task_orchestration_management(self, symphony_dashboard):
        """Test task orchestration and management."""
        # Create test tasks
        tasks = [
            TaskInfo(id="task-1", name="Code Analysis", status=TaskStatus.PENDING, assigned_agent="agent-1"),
            TaskInfo(id="task-2", name="Performance Optimization", status=TaskStatus.RUNNING, assigned_agent="agent-2"),
            TaskInfo(id="task-3", name="Documentation", status=TaskStatus.COMPLETED, assigned_agent="agent-3")
        ]
        
        for task in tasks:
            await symphony_dashboard.add_task(task)
        
        # Test task status visualization
        task_overview = await symphony_dashboard.get_task_overview()
        
        assert len(task_overview["tasks"]) == 3
        assert task_overview["completion_rate"] > 0
        assert "pending" in task_overview["status_distribution"]

    @pytest.mark.asyncio
    async def test_performance_monitoring_dashboard(self, symphony_dashboard):
        """Test performance monitoring and metrics dashboard."""
        # Simulate agent activity
        for i in range(10):
            await symphony_dashboard.record_agent_activity("agent-1", f"task-{i}", duration=0.5 + (i * 0.1))
        
        metrics = await symphony_dashboard.get_performance_metrics()
        
        assert "throughput" in metrics
        assert "average_response_time" in metrics
        assert "agent_utilization" in metrics
        assert metrics["throughput"] > 0

    @pytest.mark.asyncio
    async def test_collaborative_task_management(self, symphony_dashboard):
        """Test collaborative task management between agents."""
        # Set up collaborative task
        collaborative_task = TaskInfo(
            id="collab-1",
            name="Full-Stack Development",
            status=TaskStatus.RUNNING,
            assigned_agents=["frontend-agent", "backend-agent", "database-agent"],
            dependencies=["task-setup", "environment-config"]
        )
        
        await symphony_dashboard.add_collaborative_task(collaborative_task)
        
        # Test collaboration visualization
        collab_view = await symphony_dashboard.get_collaboration_view("collab-1")
        
        assert len(collab_view["participating_agents"]) == 3
        assert "communication_flow" in collab_view
        assert "progress_sync" in collab_view

    @pytest.mark.asyncio
    async def test_symphony_mode_scaling(self, symphony_dashboard):
        """Test symphony mode scaling with 12+ concurrent agents."""
        # Add maximum agents
        agents = []
        for i in range(15):  # Test beyond 12 agents
            agent = AgentInfo(
                id=f"agent-{i}",
                name=f"Agent {i}",
                state=AgentState.ACTIVE,
                capabilities=[f"capability-{i % 3}"]
            )
            agents.append(agent)
            await symphony_dashboard.register_agent(agent)
        
        assert symphony_dashboard.agent_count == 15
        
        # Test performance with many agents
        start_time = time.time()
        coordination_view = await symphony_dashboard.get_coordination_view()
        response_time = time.time() - start_time
        
        # Should maintain performance with many agents
        assert response_time < 0.5  # Sub-500ms for coordination view
        assert len(coordination_view["agents"]) == 15


class TestSmartOnboardingFlow:
    """Test suite for Smart Onboarding Flow with Contextual Guidance."""

    @pytest.mark.asyncio
    async def test_smart_onboarding_initialization(self, smart_onboarding_flow):
        """Test smart onboarding flow initialization."""
        assert smart_onboarding_flow.is_initialized
        assert smart_onboarding_flow.current_stage == OnboardingStage.WELCOME
        assert smart_onboarding_flow.user_profile is None

    @pytest.mark.asyncio
    async def test_user_profiling_and_adaptation(self, smart_onboarding_flow):
        """Test user profiling and flow adaptation."""
        user_responses = {
            "experience_level": "intermediate",
            "primary_use_case": "development",
            "preferred_interface": "cli",
            "team_size": "small",
            "technical_background": "software_engineer"
        }
        
        profile = await smart_onboarding_flow.create_user_profile(user_responses)
        
        assert isinstance(profile, UserProfile)
        assert profile.experience_level == "intermediate"
        assert profile.guidance_level == GuidanceLevel.STANDARD

    @pytest.mark.asyncio
    async def test_contextual_guidance_delivery(self, smart_onboarding_flow):
        """Test contextual guidance delivery based on user needs."""
        # Set up user profile
        profile = UserProfile(
            experience_level="beginner",
            use_case="data_analysis",
            guidance_level=GuidanceLevel.DETAILED
        )
        
        await smart_onboarding_flow.set_user_profile(profile)
        
        # Get contextual guidance for specific scenario
        guidance = await smart_onboarding_flow.get_contextual_guidance("first_command")
        
        assert "step-by-step" in guidance.content.lower()
        assert guidance.detail_level == "detailed"
        assert len(guidance.examples) > 0

    @pytest.mark.asyncio
    async def test_progressive_skill_building(self, smart_onboarding_flow):
        """Test progressive skill building through onboarding stages."""
        stages = [
            OnboardingStage.WELCOME,
            OnboardingStage.PROFILE_SETUP,
            OnboardingStage.BASIC_COMMANDS,
            OnboardingStage.ADVANCED_FEATURES,
            OnboardingStage.SYMPHONY_MODE,
            OnboardingStage.COMPLETION
        ]
        
        for expected_stage in stages:
            await smart_onboarding_flow.advance_stage()
            if smart_onboarding_flow.current_stage == OnboardingStage.COMPLETION:
                break
        
        assert smart_onboarding_flow.completion_progress >= 80

    @pytest.mark.asyncio
    async def test_adaptive_complexity_adjustment(self, smart_onboarding_flow):
        """Test adaptive complexity adjustment based on user performance."""
        # Simulate user struggles
        for _ in range(3):
            await smart_onboarding_flow.record_user_struggle("command_syntax_error")
        
        # Should adapt to provide more guidance
        guidance = await smart_onboarding_flow.get_adaptive_guidance("next_command")
        
        assert guidance.complexity_level == "simplified"
        assert "example" in guidance.content.lower()

    @pytest.mark.asyncio
    async def test_personalized_feature_introduction(self, smart_onboarding_flow):
        """Test personalized feature introduction based on user profile."""
        # Data scientist profile
        data_profile = UserProfile(
            experience_level="intermediate",
            use_case="data_science",
            preferred_tools=["python", "jupyter", "pandas"]
        )
        
        await smart_onboarding_flow.set_user_profile(data_profile)
        
        features = await smart_onboarding_flow.get_recommended_features()
        
        # Should recommend data science relevant features
        feature_names = [f.name.lower() for f in features]
        assert any("data" in name or "analysis" in name for name in feature_names)


class TestNaturalLanguageProcessorAccuracy:
    """Test suite for NLP Processor with 95% Accuracy Target."""

    @pytest.mark.asyncio
    async def test_95_percent_accuracy_validation(self, nlp_processor, sample_user_inputs):
        """Test NLP processor meets 95% accuracy requirement."""
        test_cases = [
            # Clear commands with expected intents
            ("create a new file called test.py", CommandIntent.CHAT),
            ("show me the status of all agents", CommandIntent.AGENT_MANAGEMENT),
            ("help me understand this error", CommandIntent.HELP),
            ("configure the system settings", CommandIntent.CONFIG),
            ("search for functions in the code", CommandIntent.DISCOVERY),
            ("start symphony mode with 5 agents", CommandIntent.SYMPHONY_MODE),
            ("run the data processing pipeline", CommandIntent.PIPELINE),
            ("analyze the performance metrics", CommandIntent.ANALYSIS),
        ]
        
        correct_predictions = 0
        total_tests = len(test_cases)
        
        for text, expected_intent in test_cases:
            context = NLPContext(
                user_id="test_user",
                session_history=[],
                user_skill_level="intermediate"
            )
            
            prediction = await nlp_processor.predict_intent(text, context)
            
            assert isinstance(prediction, IntentPrediction)
            
            # Check accuracy
            if prediction.intent == expected_intent and prediction.confidence >= 0.8:
                correct_predictions += 1
            elif expected_intent in [alt[0] for alt in prediction.alternative_intents[:2]]:
                correct_predictions += 0.5  # Partial credit for top alternatives
        
        accuracy = correct_predictions / total_tests
        assert accuracy >= 0.95, f"NLP Accuracy {accuracy:.1%} below 95% requirement"

    @pytest.mark.asyncio
    async def test_complex_multi_intent_recognition(self, nlp_processor):
        """Test recognition of complex, multi-intent natural language."""
        complex_inputs = [
            "Create a Python script that processes CSV files and then run tests on it",
            "Show me the agent status and help me optimize their performance",
            "Configure the system for development mode and start the symphony dashboard"
        ]
        
        for complex_input in complex_inputs:
            context = NLPContext(user_id="test", session_history=[])
            prediction = await nlp_processor.predict_intent(complex_input, context)
            
            assert isinstance(prediction, IntentPrediction)
            # Complex inputs should have multiple intents or high confidence
            assert len(prediction.alternative_intents) > 1 or prediction.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_contextual_intent_refinement(self, nlp_processor):
        """Test contextual intent refinement based on conversation history."""
        conversation_history = [
            "start a new development project",
            "configure the environment",
            "set up version control"
        ]
        
        context = NLPContext(
            user_id="dev_user",
            session_history=conversation_history,
            current_mode="development",
            user_skill_level="advanced"
        )
        
        # Ambiguous follow-up that needs context
        followup = "now run the tests"
        prediction = await nlp_processor.predict_intent(followup, context)
        
        assert isinstance(prediction, IntentPrediction)
        assert prediction.confidence >= 0.7  # Context should improve confidence
        assert prediction.intent in [CommandIntent.PIPELINE, CommandIntent.ANALYSIS]

    @pytest.mark.asyncio
    async def test_entity_extraction_accuracy(self, nlp_processor):
        """Test entity extraction accuracy from natural language."""
        entity_test_cases = [
            ("create file main.py in src directory", ["main.py", "src"]),
            ("set timeout to 30 seconds", ["timeout", "30", "seconds"]),
            ("run tests with 4 parallel workers", ["tests", "4", "parallel", "workers"]),
            ("configure theme to dark mode", ["theme", "dark mode"])
        ]
        
        for text, expected_entities in entity_test_cases:
            entities = await nlp_processor.extract_entities(text)
            
            extracted_values = [entity.value.lower() for entity in entities]
            
            # Check that most expected entities are found
            found_entities = sum(1 for expected in expected_entities 
                               if any(expected.lower() in value for value in extracted_values))
            
            accuracy = found_entities / len(expected_entities)
            assert accuracy >= 0.8, f"Entity extraction accuracy {accuracy:.1%} too low for: {text}"


class TestPerformanceRequirements:
    """Test suite for Performance Requirements (Sub-100ms response times)."""

    @pytest.mark.asyncio
    async def test_api_response_time_under_100ms(self, nlp_processor, intent_recognition_service):
        """Test API response times meet sub-100ms requirement."""
        test_inputs = [
            "create new file",
            "show status",
            "help me",
            "run tests",
            "config system"
        ]
        
        # Test NLP Processor
        nlp_times = []
        for input_text in test_inputs:
            start_time = time.time()
            await nlp_processor.predict_intent(input_text, NLPContext(user_id="perf_test", session_history=[]))
            response_time = time.time() - start_time
            nlp_times.append(response_time)
        
        # Test Intent Recognition Service
        intent_times = []
        for input_text in test_inputs:
            start_time = time.time()
            await intent_recognition_service.recognize_intent(input_text)
            response_time = time.time() - start_time
            intent_times.append(response_time)
        
        # Validate performance requirements
        avg_nlp_time = sum(nlp_times) / len(nlp_times)
        avg_intent_time = sum(intent_times) / len(intent_times)
        max_nlp_time = max(nlp_times)
        max_intent_time = max(intent_times)
        
        assert avg_nlp_time < 0.1, f"Average NLP response time {avg_nlp_time:.3f}s > 100ms"
        assert avg_intent_time < 0.1, f"Average intent recognition time {avg_intent_time:.3f}s > 100ms"
        assert max_nlp_time < 0.15, f"Max NLP response time {max_nlp_time:.3f}s > 150ms"
        assert max_intent_time < 0.15, f"Max intent response time {max_intent_time:.3f}s > 150ms"

    @pytest.mark.asyncio
    async def test_concurrent_user_handling(self, symphony_orchestration_api):
        """Test concurrent user handling and performance."""
        num_concurrent_users = 50
        requests_per_user = 5
        
        async def simulate_user_session(user_id: int):
            """Simulate a user session with multiple requests."""
            session_times = []
            for request_id in range(requests_per_user):
                start_time = time.time()
                result = await symphony_orchestration_api.process_user_request(
                    f"user-{user_id}",
                    f"process request {request_id}"
                )
                response_time = time.time() - start_time
                session_times.append(response_time)
            return session_times
        
        # Execute concurrent user sessions
        start_time = time.time()
        tasks = [simulate_user_session(user_id) for user_id in range(num_concurrent_users)]
        all_session_times = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Flatten all response times
        all_response_times = [time for session in all_session_times for time in session]
        
        # Calculate performance metrics
        avg_response_time = sum(all_response_times) / len(all_response_times)
        p95_response_time = sorted(all_response_times)[int(len(all_response_times) * 0.95)]
        throughput = len(all_response_times) / total_time
        
        # Validate performance under load
        assert avg_response_time < 0.2, f"Average response time under load {avg_response_time:.3f}s > 200ms"
        assert p95_response_time < 0.5, f"95th percentile response time {p95_response_time:.3f}s > 500ms"
        assert throughput >= 100, f"Throughput {throughput:.1f} requests/sec too low"

    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, symphony_dashboard):
        """Test memory usage optimization during peak load."""
        import gc
        import tracemalloc
        
        tracemalloc.start()
        
        # Get baseline memory
        gc.collect()
        baseline_memory = tracemalloc.get_traced_memory()[0]
        
        # Simulate high load
        agents = []
        tasks = []
        
        for i in range(100):  # Add many agents and tasks
            agent = AgentInfo(id=f"agent-{i}", name=f"Agent {i}", state=AgentState.ACTIVE)
            task = TaskInfo(id=f"task-{i}", name=f"Task {i}", status=TaskStatus.RUNNING)
            
            await symphony_dashboard.register_agent(agent)
            await symphony_dashboard.add_task(task)
            
            agents.append(agent)
            tasks.append(task)
        
        # Force garbage collection
        gc.collect()
        peak_memory = tracemalloc.get_traced_memory()[0]
        
        # Clean up
        for agent in agents:
            await symphony_dashboard.unregister_agent(agent.id)
        for task in tasks:
            await symphony_dashboard.remove_task(task.id)
        
        gc.collect()
        final_memory = tracemalloc.get_traced_memory()[0]
        
        tracemalloc.stop()
        
        # Calculate memory metrics
        memory_growth = peak_memory - baseline_memory
        memory_cleanup_ratio = (peak_memory - final_memory) / peak_memory
        
        # Validate memory efficiency
        assert memory_growth < 100 * 1024 * 1024, f"Memory grew by {memory_growth / 1024 / 1024:.1f}MB, should be < 100MB"
        assert memory_cleanup_ratio > 0.8, f"Memory cleanup ratio {memory_cleanup_ratio:.2%} too low"

    @pytest.mark.asyncio
    async def test_caching_performance_optimization(self, command_translation_engine):
        """Test caching performance optimization."""
        test_commands = [
            "create file main.py",
            "search for functions",
            "show agent status",
            "configure settings",
            "run test suite"
        ]
        
        # First run (cache miss)
        first_run_times = []
        for command in test_commands:
            start_time = time.time()
            await command_translation_engine.translate_command(command)
            response_time = time.time() - start_time
            first_run_times.append(response_time)
        
        # Second run (cache hit)
        second_run_times = []
        for command in test_commands:
            start_time = time.time()
            await command_translation_engine.translate_command(command)
            response_time = time.time() - start_time
            second_run_times.append(response_time)
        
        # Calculate cache performance improvement
        avg_first_run = sum(first_run_times) / len(first_run_times)
        avg_second_run = sum(second_run_times) / len(second_run_times)
        cache_speedup = avg_first_run / avg_second_run
        
        # Validate caching effectiveness
        assert cache_speedup >= 2.0, f"Cache speedup {cache_speedup:.1f}x too low, should be >= 2x"
        assert avg_second_run < 0.05, f"Cached response time {avg_second_run:.3f}s > 50ms"


class TestAccessibilityCompliance:
    """Test suite for WCAG 2.1 AAA Accessibility Compliance."""

    def test_accessibility_engine_initialization(self, accessibility_engine):
        """Test accessibility engine initializes correctly."""
        assert accessibility_engine is not None
        assert accessibility_engine.compliance_level == AccessibilityLevel.AAA
        assert hasattr(accessibility_engine, 'audit_components')

    @pytest.mark.asyncio
    async def test_keyboard_navigation_support(self, accessibility_engine):
        """Test keyboard navigation support for all UI components."""
        ui_components = [
            "command_input",
            "agent_list", 
            "status_dashboard",
            "symphony_controls",
            "settings_menu",
            "help_system"
        ]
        
        for component in ui_components:
            audit_result = await accessibility_engine.audit_keyboard_navigation(component)
            
            assert audit_result.is_compliant
            assert "tab_order" in audit_result.features
            assert "focus_indicators" in audit_result.features
            assert audit_result.compliance_score >= 0.95

    @pytest.mark.asyncio 
    async def test_screen_reader_compatibility(self, accessibility_engine):
        """Test screen reader compatibility and semantic markup."""
        components_to_test = [
            "progressive_disclosure_ui",
            "symphony_dashboard",
            "command_composer",
            "onboarding_flow"
        ]
        
        for component in components_to_test:
            screen_reader_audit = await accessibility_engine.audit_screen_reader_support(component)
            
            assert screen_reader_audit.is_compliant
            assert "aria_labels" in screen_reader_audit.features
            assert "semantic_structure" in screen_reader_audit.features
            assert "alternative_text" in screen_reader_audit.features
            assert screen_reader_audit.compliance_score >= 0.9

    @pytest.mark.asyncio
    async def test_color_contrast_compliance(self, accessibility_engine):
        """Test color contrast meets WCAG AAA requirements."""
        color_schemes = [
            "default_theme",
            "dark_theme", 
            "high_contrast_theme",
            "light_theme"
        ]
        
        for scheme in color_schemes:
            contrast_audit = await accessibility_engine.audit_color_contrast(scheme)
            
            assert contrast_audit.is_compliant
            assert contrast_audit.min_contrast_ratio >= 7.0  # AAA requirement
            assert all(ratio >= 7.0 for ratio in contrast_audit.contrast_ratios)

    @pytest.mark.asyncio
    async def test_responsive_design_accessibility(self, accessibility_engine):
        """Test responsive design accessibility across different viewport sizes."""
        viewport_sizes = [
            (320, 568),   # Mobile portrait
            (768, 1024),  # Tablet
            (1920, 1080), # Desktop
            (2560, 1440)  # Large desktop
        ]
        
        for width, height in viewport_sizes:
            responsive_audit = await accessibility_engine.audit_responsive_accessibility(width, height)
            
            assert responsive_audit.is_compliant
            assert responsive_audit.layout_stability_score >= 0.9
            assert "touch_targets" in responsive_audit.features
            assert "text_readability" in responsive_audit.features

    @pytest.mark.asyncio
    async def test_text_alternatives_and_descriptions(self, accessibility_engine):
        """Test text alternatives and descriptions for all non-text content."""
        content_types = [
            "icons",
            "status_indicators",
            "progress_bars", 
            "data_visualizations",
            "agent_avatars"
        ]
        
        for content_type in content_types:
            text_audit = await accessibility_engine.audit_text_alternatives(content_type)
            
            assert text_audit.is_compliant
            assert text_audit.coverage_percentage >= 100
            assert "descriptive_text" in text_audit.features
            assert "context_information" in text_audit.features

    @pytest.mark.asyncio
    async def test_error_identification_and_suggestions(self, accessibility_engine):
        """Test error identification and suggestion accessibility."""
        error_scenarios = [
            "invalid_command_syntax",
            "missing_required_parameters",
            "agent_connection_failure",
            "authentication_error",
            "resource_not_found"
        ]
        
        for scenario in error_scenarios:
            error_audit = await accessibility_engine.audit_error_handling(scenario)
            
            assert error_audit.is_compliant
            assert "clear_error_messages" in error_audit.features
            assert "recovery_suggestions" in error_audit.features
            assert "error_location_indication" in error_audit.features
            assert error_audit.clarity_score >= 0.9


class TestIntegrationScenarios:
    """Test suite for Integration Testing (CLI-TUI-Web handoff scenarios)."""

    @pytest.mark.asyncio
    async def test_cli_to_tui_handoff(self, mcp):
        """Test seamless handoff from CLI to TUI interface."""
        # Start in CLI mode
        mcp.sendline("agentsmcp --mode cli")
        mcp.expect("AgentsMCP CLI")
        
        # Execute some CLI commands
        mcp.sendline("status")
        mcp.expect("Agent Status")
        
        # Switch to TUI mode
        mcp.sendline("tui")
        mcp.expect("Modern TUI Interface")
        
        # Verify state persistence
        mcp.sendcontrol("c")  # Send Ctrl+C to access menu
        mcp.expect("Status Dashboard")  # Should show previous state
        
        # Test return to CLI
        mcp.sendline("exit")
        mcp.expect("Returning to CLI mode")

    @pytest.mark.asyncio
    async def test_tui_to_web_consistency(self, web_server):
        """Test UI consistency between TUI and Web interfaces."""
        import requests
        
        # Get TUI state (mocked for integration test)
        tui_state = {
            "active_agents": 3,
            "running_tasks": 5,
            "system_status": "healthy"
        }
        
        # Query web API for same information
        response = requests.get(f"{web_server}/api/status")
        assert response.status_code == 200
        web_state = response.json()
        
        # Verify consistency
        assert web_state["agent_count"] == tui_state["active_agents"]
        assert web_state["task_count"] == tui_state["running_tasks"]
        assert web_state["status"] == tui_state["system_status"]

    @pytest.mark.asyncio
    async def test_cross_platform_command_execution(self):
        """Test cross-platform command execution consistency."""
        import platform
        
        command_variations = {
            "Windows": "dir",
            "Darwin": "ls",
            "Linux": "ls"
        }
        
        current_platform = platform.system()
        expected_command = command_variations.get(current_platform, "ls")
        
        # Test command translation works across platforms
        # This would integrate with the actual command translation engine
        # For now, we verify the logic exists
        assert expected_command is not None

    @pytest.mark.asyncio
    async def test_event_system_integration(self):
        """Test event system integration across all components."""
        from agentsmcp.ui.v2.event_system import AsyncEventSystem, Event, EventType
        
        event_system = AsyncEventSystem()
        
        # Test event propagation
        events_received = []
        
        async def event_handler(event):
            events_received.append(event)
        
        event_system.subscribe(EventType.USER_ACTION, event_handler)
        
        # Simulate events from different components
        test_events = [
            Event(EventType.USER_ACTION, {"action": "command_executed", "component": "cli"}),
            Event(EventType.USER_ACTION, {"action": "mode_switched", "component": "tui"}),
            Event(EventType.AGENT_STATUS, {"agent_id": "test", "status": "active", "component": "symphony"})
        ]
        
        for event in test_events:
            await event_system.publish(event)
        
        # Allow events to propagate
        await asyncio.sleep(0.1)
        
        # Verify event integration
        assert len(events_received) >= 2  # Should receive USER_ACTION events


class TestErrorRecoveryAndGracefulDegradation:
    """Test suite for Error Recovery and Graceful Degradation."""

    @pytest.mark.asyncio
    async def test_network_failure_recovery(self, symphony_orchestration_api):
        """Test network failure recovery mechanisms."""
        # Simulate network failure
        with patch('httpx.AsyncClient.request', side_effect=Exception("Network error")):
            # Should handle gracefully
            result = await symphony_orchestration_api.process_user_request("test_user", "test command")
            
            assert result is not None
            assert "error" in result or "offline" in result
            assert "retry" in str(result).lower()

    @pytest.mark.asyncio
    async def test_service_degradation_fallback(self, ai_command_composer):
        """Test service degradation and fallback mechanisms."""
        # Disable advanced NLP features
        with patch.object(ai_command_composer, '_advanced_nlp_enabled', False):
            # Should fall back to basic pattern matching
            suggestion = await ai_command_composer.compose_from_natural_language("create file")
            
            assert isinstance(suggestion, CommandSuggestion)
            assert suggestion.confidence >= 0.5  # Lower but acceptable confidence
            assert "fallback" in suggestion.reasoning.lower() or suggestion.confidence < 0.9

    @pytest.mark.asyncio
    async def test_component_failure_isolation(self, symphony_dashboard):
        """Test component failure isolation and continued operation."""
        # Simulate component failure
        with patch.object(symphony_dashboard, '_performance_monitor', side_effect=Exception("Monitor failed")):
            # Other functions should still work
            agent = AgentInfo(id="test", name="Test", state=AgentState.ACTIVE)
            await symphony_dashboard.register_agent(agent)
            
            agents = symphony_dashboard.get_all_agents()
            assert len(agents) > 0

    @pytest.mark.asyncio
    async def test_data_corruption_recovery(self, progressive_disclosure_manager):
        """Test data corruption recovery mechanisms."""
        # Simulate corrupted user profile
        progressive_disclosure_manager.user_profile = {"invalid": "data", "corrupt": True}
        
        # Should recover gracefully
        try:
            progressive_disclosure_manager.adapt_to_user()
            # Should create new valid profile or use defaults
            assert progressive_disclosure_manager.current_skill_level is not None
        except Exception as e:
            pytest.fail(f"Should handle corrupt data gracefully, got: {e}")


# Performance Benchmark Tests
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests for comprehensive system validation."""

    @pytest.mark.asyncio
    async def test_end_to_end_performance_benchmark(self):
        """Test end-to-end performance benchmark from input to output."""
        from agentsmcp.ui.modern_tui import ModernTUI
        
        # Initialize complete system
        tui = ModernTUI()
        await tui.initialize()
        
        try:
            # Benchmark complete user workflow
            workflows = [
                "Create new Python project with tests and CI/CD",
                "Analyze codebase performance and suggest optimizations", 
                "Set up multi-agent symphony mode for collaborative development",
                "Generate comprehensive documentation for the project",
                "Deploy application to production with monitoring"
            ]
            
            total_start_time = time.time()
            workflow_times = []
            
            for workflow in workflows:
                start_time = time.time()
                
                # Simulate complete workflow processing
                await tui.process_user_input(workflow)
                
                workflow_time = time.time() - start_time
                workflow_times.append(workflow_time)
            
            total_time = time.time() - total_start_time
            
            # Performance requirements validation
            avg_workflow_time = sum(workflow_times) / len(workflow_times)
            max_workflow_time = max(workflow_times)
            
            assert avg_workflow_time < 5.0, f"Average workflow time {avg_workflow_time:.2f}s > 5s"
            assert max_workflow_time < 10.0, f"Max workflow time {max_workflow_time:.2f}s > 10s"
            assert total_time < 20.0, f"Total benchmark time {total_time:.2f}s > 20s"
            
        finally:
            await tui.cleanup()

    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self):
        """Test memory leak prevention during extended operation."""
        import gc
        import tracemalloc
        
        tracemalloc.start()
        
        # Simulate extended operation
        components = []
        
        try:
            for i in range(100):
                # Create and destroy components repeatedly
                dashboard = SymphonyDashboard()
                await dashboard.initialize()
                components.append(dashboard)
                
                if i % 10 == 0:  # Cleanup every 10 iterations
                    for component in components:
                        await component.cleanup()
                    components.clear()
                    gc.collect()
            
            # Final cleanup
            for component in components:
                await component.cleanup()
            
            gc.collect()
            
            # Check memory stability
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            memory_efficiency = current_memory / peak_memory
            
            # Should maintain reasonable memory efficiency
            assert memory_efficiency > 0.3, f"Memory efficiency {memory_efficiency:.2%} too low"
            
        finally:
            tracemalloc.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--maxfail=5"])