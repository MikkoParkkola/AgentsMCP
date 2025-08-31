"""
Revolutionary Backend API Improvements for AgentsMCP

This module provides advanced backend APIs to support the revolutionary CLI experience:
- Natural language command processing with ML-powered intent recognition  
- Symphony mode orchestration with real-time agent coordination
- Progressive disclosure services with adaptive complexity management
- High-performance caching, monitoring, and security layers

Architecture:
- Sub-100ms API response times for optimal user experience
- Comprehensive error handling with graceful degradation
- Real-time streaming capabilities for live updates
- Advanced security with audit trails and rate limiting
- ML-powered user profiling and adaptive interfaces
"""

# Core API components
from .nlp_processor import NLPProcessor
from .intent_recognition_service import IntentRecognitionService
from .command_translation_engine import CommandTranslationEngine
from .symphony_orchestration_api import SymphonyOrchestrationAPI
from .progressive_disclosure_service import ProgressiveDisclosureService
from .realtime_status_streaming import RealtimeStatusStreaming
from .smart_caching_layer import SmartCachingLayer
from .comprehensive_monitoring import ComprehensiveMonitoring

# API utilities and shared components
from .base import APIBase, APIResponse, APIError
from .security import SecurityAPI
from .performance import PerformanceMonitoring

__all__ = [
    # Core API Services
    "NLPProcessor",
    "IntentRecognitionService", 
    "CommandTranslationEngine",
    "SymphonyOrchestrationAPI",
    "ProgressiveDisclosureService",
    "RealtimeStatusStreaming",
    "SmartCachingLayer", 
    "ComprehensiveMonitoring",
    
    # Base Components
    "APIBase",
    "APIResponse",
    "APIError",
    
    # Security & Performance
    "SecurityAPI",
    "PerformanceMonitoring"
]

__version__ = "1.0.0"