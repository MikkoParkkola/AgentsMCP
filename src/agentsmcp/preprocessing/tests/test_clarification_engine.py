"""
Tests for Clarification Engine component.

Tests the clarification question generation and management including:
- Clarification need assessment
- Question generation quality
- User interaction handling
- Session management
- Confidence improvement tracking
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from ..clarification_engine import (
    ClarificationEngine, ClarificationSession, ClarificationQuestion,
    QuestionType, QuestionPriority
)
from ..intent_analyzer import IntentAnalysis, IntentType, TechnicalDomain


class TestClarificationEngine:
    """Test suite for ClarificationEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create clarification engine instance."""
        return ClarificationEngine(confidence_threshold=0.9)
    
    @pytest.fixture
    def sample_intent_analysis(self):
        """Create sample intent analysis for testing."""
        return IntentAnalysis(
            primary_intent=IntentType.TASK_EXECUTION,
            confidence=0.6,
            technical_domain=TechnicalDomain.WEB_DEVELOPMENT,
            ambiguous_terms=["it", "system"],
            missing_context=["unclear_references"],
            assumptions_needed=["unspecified_technologies"],
            keywords=["create", "app", "user", "authentication"]
        )
    
    @pytest.mark.asyncio
    async def test_clarification_need_assessment(self, engine, sample_intent_analysis):
        """Test assessment of whether clarification is needed.""" 
        needs_clarification, confidence_gap, reasons = await engine.assess_clarification_need(
            sample_intent_analysis
        )
        
        assert needs_clarification is True
        assert confidence_gap > 0.2  # Should identify significant gap
        assert len(reasons) > 0
        assert any("confidence" in reason.lower() for reason in reasons)
    
    @pytest.mark.asyncio
    async def test_high_confidence_no_clarification(self, engine):
        """Test that high confidence requests don't need clarification."""
        high_confidence_analysis = IntentAnalysis(
            primary_intent=IntentType.TASK_EXECUTION,
            confidence=0.95,
            technical_domain=TechnicalDomain.WEB_DEVELOPMENT,
            success_criteria=["working authentication system"]
        )
        
        needs_clarification, confidence_gap, reasons = await engine.assess_clarification_need(
            high_confidence_analysis
        )
        
        assert needs_clarification is False
        assert confidence_gap < 0.2
    
    @pytest.mark.asyncio
    async def test_clarification_session_creation(self, engine, sample_intent_analysis):
        """Test creation of clarification sessions."""
        session = await engine.create_clarification_session(
            user_input="Create an app with user authentication",
            intent_analysis=sample_intent_analysis
        )
        
        assert isinstance(session, ClarificationSession)
        assert session.session_id in engine.active_sessions
        assert len(session.questions) > 0
        assert session.current_confidence == sample_intent_analysis.confidence
        assert session.target_confidence == engine.confidence_threshold
        assert not session.session_complete
    
    @pytest.mark.asyncio
    async def test_question_generation_quality(self, engine, sample_intent_analysis):
        """Test quality of generated clarification questions."""
        session = await engine.create_clarification_session(
            user_input="Build something",
            intent_analysis=sample_intent_analysis
        )
        
        # Should generate questions for detected issues
        question_types = [q.question_type for q in session.questions]
        
        # Should have questions for ambiguity resolution
        assert QuestionType.RESOLVE_AMBIGUITY in question_types
        
        # Should prioritize critical questions
        critical_questions = [q for q in session.questions if q.priority == QuestionPriority.CRITICAL]
        high_questions = [q for q in session.questions if q.priority == QuestionPriority.HIGH]
        
        assert len(critical_questions) + len(high_questions) > 0
    
    @pytest.mark.asyncio
    async def test_answer_processing(self, engine, sample_intent_analysis):
        """Test processing of user answers to questions."""
        session = await engine.create_clarification_session(
            user_input="Create an app",
            intent_analysis=sample_intent_analysis
        )
        
        initial_confidence = session.current_confidence
        
        # Answer first question
        updated_session, is_complete = await engine.process_answer(
            session_id=session.session_id,
            question_index=0,
            answer="I want to build a React web application for managing tasks"
        )
        
        assert updated_session.current_confidence >= initial_confidence
        assert len(updated_session.answers_received) == 1
        
        # Session should not be complete after one answer
        assert not is_complete
    
    @pytest.mark.asyncio
    async def test_high_quality_answer_processing(self, engine, sample_intent_analysis):
        """Test processing of high-quality detailed answers."""
        session = await engine.create_clarification_session(
            user_input="Fix the issue",
            intent_analysis=sample_intent_analysis
        )
        
        # Provide detailed, specific answer
        detailed_answer = "I need to fix a React component authentication error where users can't log in using JWT tokens in my Node.js backend"
        
        updated_session, is_complete = await engine.process_answer(
            session_id=session.session_id,
            question_index=0,
            answer=detailed_answer
        )
        
        # Should improve confidence more with detailed answer
        improvement = updated_session.current_confidence - session.current_confidence
        assert improvement > 0.1  # Significant improvement
    
    @pytest.mark.asyncio
    async def test_session_completion(self, engine, sample_intent_analysis):
        """Test session completion logic."""
        # Create session with lower threshold for faster completion
        engine.confidence_threshold = 0.8
        
        session = await engine.create_clarification_session(
            user_input="Create an app",
            intent_analysis=sample_intent_analysis
        )
        
        # Answer questions until completion
        question_index = 0
        max_iterations = 5
        
        while not session.session_complete and question_index < max_iterations:
            available_questions = await engine.get_next_questions(session.session_id)
            if not available_questions:
                break
            
            # Provide good answers to improve confidence
            good_answer = f"Detailed answer {question_index}: React web application with user authentication using JWT"
            
            session, is_complete = await engine.process_answer(
                session_id=session.session_id,
                question_index=0,  # Always answer first available question
                answer=good_answer
            )
            
            question_index += 1
            
            if is_complete:
                break
        
        assert session.session_complete or session.current_confidence >= 0.8
        if session.session_complete:
            assert session.refined_prompt is not None
    
    @pytest.mark.asyncio
    async def test_follow_up_question_generation(self, engine, sample_intent_analysis):
        """Test generation of follow-up questions."""
        session = await engine.create_clarification_session(
            user_input="Build an API",
            intent_analysis=sample_intent_analysis
        )
        
        # Answer with information that could trigger follow-ups
        session, is_complete = await engine.process_answer(
            session_id=session.session_id,
            question_index=0,
            answer="I want to use technology for my project"  # Vague answer about technology
        )
        
        # Should generate follow-up questions
        next_questions = await engine.get_next_questions(session.session_id)
        
        # Should have follow-up questions about specific technologies
        tech_related_questions = [
            q for q in next_questions 
            if "technology" in q.question.lower() or "specific" in q.question.lower()
        ]
        
        assert len(tech_related_questions) > 0
    
    @pytest.mark.asyncio
    async def test_intent_specific_questions(self, engine):
        """Test that questions are tailored to specific intents."""
        # Task execution intent
        task_analysis = IntentAnalysis(
            primary_intent=IntentType.TASK_EXECUTION,
            confidence=0.5,
            technical_domain=TechnicalDomain.SOFTWARE_DEVELOPMENT,
            success_criteria=[]  # Missing success criteria
        )
        
        task_session = await engine.create_clarification_session(
            user_input="Build something",
            intent_analysis=task_analysis
        )
        
        # Should ask about success criteria for task execution
        success_questions = [
            q for q in task_session.questions
            if "success" in q.question.lower() or "completed" in q.question.lower()
        ]
        assert len(success_questions) > 0
        
        # Analysis/review intent
        analysis_analysis = IntentAnalysis(
            primary_intent=IntentType.ANALYSIS_REVIEW,
            confidence=0.5,
            technical_domain=TechnicalDomain.SOFTWARE_DEVELOPMENT
        )
        
        analysis_session = await engine.create_clarification_session(
            user_input="Review my code",
            intent_analysis=analysis_analysis
        )
        
        # Should ask different types of questions for analysis
        analysis_questions = [
            q for q in analysis_session.questions
            if any(word in q.question.lower() for word in ["review", "analyze", "specific", "focus"])
        ]
        assert len(analysis_questions) > 0
    
    @pytest.mark.asyncio
    async def test_domain_specific_questions(self, engine):
        """Test domain-specific question generation."""
        web_dev_analysis = IntentAnalysis(
            primary_intent=IntentType.TASK_EXECUTION,
            confidence=0.6,
            technical_domain=TechnicalDomain.WEB_DEVELOPMENT,
            keywords=["web", "app", "frontend"]
        )
        
        session = await engine.create_clarification_session(
            user_input="Create a web app",
            intent_analysis=web_dev_analysis
        )
        
        # Should have web development specific questions
        web_questions = [
            q for q in session.questions
            if any(word in q.question.lower() for word in ["frontend", "backend", "full-stack"])
        ]
        
        assert len(web_questions) > 0
    
    @pytest.mark.asyncio
    async def test_question_prioritization(self, engine, sample_intent_analysis):
        """Test that questions are properly prioritized."""
        session = await engine.create_clarification_session(
            user_input="Fix the critical system error",
            intent_analysis=sample_intent_analysis
        )
        
        # Get prioritized questions
        next_questions = await engine.get_next_questions(session.session_id, limit=3)
        
        # Should return highest priority questions first
        priorities = [q.priority for q in next_questions]
        
        # Check that critical/high priority questions come first
        if len(priorities) > 1:
            assert priorities[0].value >= priorities[1].value
    
    @pytest.mark.asyncio
    async def test_refined_prompt_generation(self, engine, sample_intent_analysis):
        """Test generation of refined prompts after clarification."""
        session = await engine.create_clarification_session(
            user_input="Create an app",
            intent_analysis=sample_intent_analysis
        )
        
        # Simulate completion with good answers
        session.answers_received = {
            "What type of application?": "React web application for task management",
            "What technologies?": "React, Node.js, PostgreSQL",
            "Success criteria?": "Users can create, edit, and delete tasks with authentication"
        }
        session.current_confidence = 0.95
        session.session_complete = True
        
        refined_prompt = await engine._generate_refined_prompt(session)
        
        assert refined_prompt != session.original_input
        assert len(refined_prompt) > len(session.original_input)
        assert "React" in refined_prompt
        assert "authentication" in refined_prompt
    
    @pytest.mark.asyncio
    async def test_session_timeout_handling(self, engine, sample_intent_analysis):
        """Test handling of session timeouts and cleanup."""
        session = await engine.create_clarification_session(
            user_input="Test input",
            intent_analysis=sample_intent_analysis
        )
        
        session_id = session.session_id
        
        # Simulate max iterations reached
        session.iteration_count = session.max_iterations
        
        # Process answer should complete session due to timeout
        updated_session, is_complete = await engine.process_answer(
            session_id=session_id,
            question_index=0,
            answer="Test answer"
        )
        
        assert is_complete or updated_session.iteration_count >= session.max_iterations
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, engine, sample_intent_analysis):
        """Test handling of multiple concurrent clarification sessions.""" 
        sessions = []
        
        # Create multiple sessions
        for i in range(3):
            session = await engine.create_clarification_session(
                user_input=f"Test input {i}",
                intent_analysis=sample_intent_analysis
            )
            sessions.append(session)
        
        assert len(engine.active_sessions) == 3
        
        # Process answers for different sessions
        for i, session in enumerate(sessions):
            updated_session, _ = await engine.process_answer(
                session_id=session.session_id,
                question_index=0,
                answer=f"Answer for session {i}"
            )
            assert len(updated_session.answers_received) == 1
        
        # Clean up
        for session in sessions:
            engine.close_session(session.session_id)
        
        assert len(engine.active_sessions) == 0
    
    @pytest.mark.asyncio
    async def test_ambiguity_resolution_questions(self, engine):
        """Test specific ambiguity resolution questions."""
        ambiguous_analysis = IntentAnalysis(
            primary_intent=IntentType.TASK_EXECUTION,
            confidence=0.5,
            ambiguous_terms=["it", "this", "system"],
            missing_context=["unclear_references"]
        )
        
        session = await engine.create_clarification_session(
            user_input="Fix it and update this system",
            intent_analysis=ambiguous_analysis
        )
        
        # Should generate questions to resolve ambiguity
        ambiguity_questions = [
            q for q in session.questions
            if q.question_type == QuestionType.RESOLVE_AMBIGUITY
        ]
        
        assert len(ambiguity_questions) > 0
        
        # Questions should reference specific ambiguous terms
        ambiguity_text = " ".join([q.question for q in ambiguity_questions])
        assert any(term in ambiguity_text.lower() for term in ["it", "this", "system"])
    
    @pytest.mark.asyncio
    async def test_low_quality_answer_handling(self, engine, sample_intent_analysis):
        """Test handling of low-quality user answers."""
        session = await engine.create_clarification_session(
            user_input="Do something",
            intent_analysis=sample_intent_analysis
        )
        
        initial_confidence = session.current_confidence
        
        # Provide low-quality answer
        updated_session, _ = await engine.process_answer(
            session_id=session.session_id,
            question_index=0,
            answer="yes"  # Very brief, low-quality answer
        )
        
        # Should have minimal confidence improvement
        confidence_improvement = updated_session.current_confidence - initial_confidence
        assert confidence_improvement <= 0.1  # Limited improvement for poor answer
    
    def test_statistics_tracking(self, engine):
        """Test that engine statistics are properly tracked."""
        initial_stats = engine.get_engine_stats()
        
        assert isinstance(initial_stats, dict)
        assert "sessions_created" in initial_stats
        assert "questions_generated" in initial_stats
        assert "successful_clarifications" in initial_stats
        assert "success_rate" in initial_stats
    
    @pytest.mark.asyncio
    async def test_error_handling(self, engine):
        """Test error handling in clarification engine."""
        # Test with invalid session ID
        with pytest.raises(ValueError):
            await engine.process_answer("invalid_session", 0, "test")
        
        # Test with invalid question index
        session = await engine.create_clarification_session(
            "test", IntentAnalysis(IntentType.TASK_EXECUTION, confidence=0.5)
        )
        
        with pytest.raises(ValueError):
            await engine.process_answer(session.session_id, 999, "test")
    
    @pytest.mark.asyncio
    async def test_question_limit_enforcement(self, engine, sample_intent_analysis):
        """Test that question limits are enforced."""
        session = await engine.create_clarification_session(
            user_input="Very ambiguous request with many issues to clarify",
            intent_analysis=sample_intent_analysis
        )
        
        # Should limit number of questions to avoid overwhelming user
        assert len(session.questions) <= 5
        
        # Get next questions with limit
        next_questions = await engine.get_next_questions(session.session_id, limit=2)
        assert len(next_questions) <= 2


@pytest.mark.integration
class TestClarificationEngineIntegration:
    """Integration tests for ClarificationEngine."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_clarification_flow(self):
        """Test complete clarification flow from ambiguous input to refined prompt."""
        engine = ClarificationEngine(confidence_threshold=0.85)
        
        # Start with very ambiguous input
        ambiguous_analysis = IntentAnalysis(
            primary_intent=IntentType.TASK_EXECUTION,
            confidence=0.4,
            ambiguous_terms=["it", "system", "better"],
            missing_context=["unclear_references", "incomplete_specification"],
            assumptions_needed=["unspecified_technologies", "unspecified_scope"]
        )
        
        # Create session
        session = await engine.create_clarification_session(
            user_input="Make the system better and fix it",
            intent_analysis=ambiguous_analysis
        )
        
        assert not session.session_complete
        assert session.current_confidence < 0.85
        
        # Simulate user interaction with good answers
        answers = [
            "I want to improve the user authentication system in my React web application",
            "React frontend with Node.js backend and PostgreSQL database",
            "Users should be able to log in securely without errors and have better UX"
        ]
        
        for i, answer in enumerate(answers):
            if session.session_complete:
                break
                
            available_questions = await engine.get_next_questions(session.session_id)
            if not available_questions:
                break
            
            session, is_complete = await engine.process_answer(
                session_id=session.session_id,
                question_index=0,
                answer=answer
            )
            
            # Confidence should improve with each good answer
            assert session.current_confidence > ambiguous_analysis.confidence
            
            if is_complete:
                break
        
        # Should eventually complete or reach high confidence
        assert session.session_complete or session.current_confidence >= 0.8
        
        # Refined prompt should be much more specific
        if session.refined_prompt:
            assert len(session.refined_prompt) > len(session.original_input)
            assert "React" in session.refined_prompt
            assert "authentication" in session.refined_prompt